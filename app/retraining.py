import json
import math
import os
import re
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


LABELS = ("positive", "neutral", "negative")
TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ0-9_]+")


class AdaptiveSentimentTrainer:
    """
    Archivia feedback, rileva label drift, ed addestra il modello adattivo.
    """

    def __init__(self, data_dir: Optional[str] = None, min_samples: Optional[int] = None):
        default_dir = Path(__file__).resolve().parent / "data"
        root = Path(data_dir or os.getenv("RETRAINING_DATA_DIR", str(default_dir)))
        root.mkdir(parents=True, exist_ok=True)
        self.feedback_path = root / "feedback.jsonl"
        self.predictions_path = root / "predictions.jsonl"
        self.model_path = root / "adaptive_model.json"
        self.min_samples = min_samples or int(os.getenv("RETRAINING_MIN_SAMPLES", "20"))
        self.drift_threshold = float(os.getenv("RETRAINING_DRIFT_THRESHOLD", "0.25"))
        self.window = deque(maxlen=int(os.getenv("RETRAINING_WINDOW_SIZE", "100")))
        self.reference_counts = Counter()
        self._model = self._load_model()
        self._lock = threading.RLock()
        self.last_retrain_at = None

    @staticmethod
    def _tokens(text):
        # Tokenizza il testo in base al pattern definito, convertendo tutto in minuscolo.
        return TOKEN_PATTERN.findall(text.lower())

    def _load_model(self):
        # Carica il modello adattivo dal file JSON se esiste, altrimenti restituisce None.
        if not self.model_path.exists():
            return None
        try:
            return json.loads(self.model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _append(path, row):
        # Apre il file in modalità append e scrive una riga JSON.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_prediction(self, text, label, score):
        # Registra una predizione nel file delle predizioni e aggiorna la finestra dei label recenti.
        with self._lock:
            self.window.append(label)
            if not self.reference_counts:
                self.reference_counts.update([label])
            self._append(self.predictions_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "label": label,
                "score": score,
            })

    def add_feedback(self, text, label, predicted_label=None):
        # Aggiunge un feedback al file di feedback e restituisce il numero totale di feedback registrati.
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        with self._lock:
            self._append(self.feedback_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "label": label,
                "predicted_label": predicted_label,
            })
            return self.feedback_count()

    def feedback_count(self):
        #  Restituisce il numero di righe di feedback registrate nel file di feedback.
        if not self.feedback_path.exists():
            return 0
        with self.feedback_path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def drift_score(self):
        # Calcola il punteggio di drift basato sulla distribuzione dei label recenti rispetto ai label di riferimento.
        with self._lock:
            if len(self.window) < 10 or not self.reference_counts:
                return 0.0
            reference_total = sum(self.reference_counts.values())
            current = Counter(self.window)
            current_total = len(self.window)
            return sum(
                abs(current[label] / current_total - self.reference_counts[label] / reference_total)
                for label in LABELS
            ) / 2

    def _feedback_rows(self):
        # Restituisce tutte le righe di feedback come una lista di dizionari, caricando il file JSONL.
        if not self.feedback_path.exists():
            return []
        with self.feedback_path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def retrain(self, force=False):
        # Addestra un nuovo modello adattivo basato sui feedback registrati, se ci sono abbastanza campioni e almeno due label diverse.
        with self._lock:
            rows = self._feedback_rows()
            if not force and len(rows) < self.min_samples:
                return False
            if len({row["label"] for row in rows}) < 2:
                return False

            document_counts = Counter()
            token_counts = {label: Counter() for label in LABELS}
            for row in rows:
                label = row["label"]
                document_counts[label] += 1
                token_counts[label].update(self._tokens(row["text"]))
            vocabulary = sorted(set().union(*(counts.keys() for counts in token_counts.values())))
            self._model = {
                "version": (self._model or {}).get("version", 0) + 1,
                "documents": dict(document_counts),
                "tokens": {label: dict(counts) for label, counts in token_counts.items()},
                "vocabulary_size": len(vocabulary),
            }
            self.model_path.write_text(json.dumps(self._model), encoding="utf-8")
            self.last_retrain_at = datetime.now(timezone.utc).isoformat()
            return True

    def predict(self, text):
        # Restituisce la label predetta e la probabilità associata per un dato testo, utilizzando il modello adattivo addestrato.
        with self._lock:
            if not self._model:
                return None
            documents = self._model["documents"]
            tokens = self._model["tokens"]
            total_documents = sum(documents.values())
            vocabulary_size = max(self._model["vocabulary_size"], 1)
            scores = {}
            for label in LABELS:
                prior = (documents.get(label, 0) + 1) / (total_documents + len(LABELS))
                total_tokens = sum(tokens.get(label, {}).values())
                log_score = math.log(prior)
                for token in self._tokens(text):
                    likelihood = (tokens.get(label, {}).get(token, 0) + 1) / (total_tokens + vocabulary_size)
                    log_score += math.log(likelihood)
                scores[label] = log_score
            best_label = max(scores, key=scores.get)
            probabilities = {label: math.exp(value - max(scores.values())) for label, value in scores.items()}
            return best_label, probabilities[best_label] / sum(probabilities.values())

    def status(self):
        # Restituisce lo stato attuale del sistema di addestramento adattivo, inclusi il numero di feedback, il punteggio di drift e se è necessario un nuovo addestramento.
        with self._lock:
            return {
                "feedback_samples": self.feedback_count(),
                "min_samples": self.min_samples,
                "drift_score": round(self.drift_score(), 4),
                "drift_threshold": self.drift_threshold,
                "model_version": (self._model or {}).get("version", 0),
                "retraining_required": self.feedback_count() >= self.min_samples and (
                    self.drift_score() >= self.drift_threshold or not self._model
                ),
                "last_retrain_at": self.last_retrain_at,
            }