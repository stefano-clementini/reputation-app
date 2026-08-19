import logging
import json
from datetime import datetime
from pathlib import Path

# Configurazione logger strutturato per salvare le metriche in formato JSON
logger = logging.getLogger("reputation_monitor")
logger.setLevel(logging.INFO)

# Scrive il log nella directory dell'app (portabile e senza percorsi assoluti hard-coded)
log_path = Path(__file__).resolve().parent / "monitoring.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

handler = logging.FileHandler(str(log_path))
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_prediction(text: str, label: str, score: float, elapsed_time: float):
    """
    Registra i dati di inferenza per valutare costantemente il drift del sentiment
    e le performance di latenza del modello.
    """
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "text_length": len(text),
        "predicted_sentiment": label,
        "confidence_score": score,
        "latency_seconds": elapsed_time
    }
    # Logga in formato JSON per una facile ingestione da parte di strumenti BI o script ELT
    logger.info(json.dumps(log_data))