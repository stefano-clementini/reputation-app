from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import pipeline
import time
import os

# Strumentazione Prometheus
from prometheus_client import make_asgi_app, Counter, Histogram
from app.retraining import AdaptiveSentimentTrainer, LABELS

# Creazione dell'app FastAPI
app = FastAPI(title="Sentiment Analysis API")

# Endpoint /metrics nativo di Prometheus Python Client integrato in FastAPI
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Definizione delle metriche Prometheus
SENTIMENT_COUNT = Counter(
    "sentiment_predictions_total",
    "Numero totale di predizioni eseguite",
    ["sentiment_label"]  # Label per filtrare (positive, neutral, negative)
)

LATENCY_HISTOGRAM = Histogram(
    "sentiment_prediction_latency_seconds",
    "Latenza del processo di inferenza del sentiment in secondi",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Inizializzazione modello
classifier = None
# Inizializzazione AdaptiveSentimentTrainer
adaptive_trainer = AdaptiveSentimentTrainer()

def get_classifier():
    global classifier
    # Durante i test evitiamo di scaricare/caricare un modello grande
    # si verifica che sia presente la proprietà TESTING=1 nelle variabili d'ambiente per ritornare una predizione fittizia
    if os.getenv("TESTING") == "1":
        return lambda text: [{"label": "positive", "score": 0.99}]

    if classifier is None:
        try:
            classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            raise RuntimeError(f"Errore nel caricamento del modello: {e}")
    return classifier

class TextInput(BaseModel):
    # Definizione del modello di input per la richiesta POST
    text: str
    
class SentimentOutput(BaseModel):
    # Definizione del modello di output per la risposta
    label: str
    score: float
    elapsed_time: float

class FeedbackInput(BaseModel):
    # Definizione del modello di input per il feedback
    text: str
    label: str
    predicted_label: str | None = None

def run_retraining():
    adaptive_trainer.retrain()

@app.post("/predict", response_model=SentimentOutput)
async def predict_sentiment(input_data: TextInput):
    # Validazione dell'input
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto.")
    
    start_time = time.time()

    # Predizione con il modello adattivo
    adaptive_prediction = adaptive_trainer.predict(input_data.text)
    if adaptive_prediction:
        # Utilizzo del modello adattivo se disponibile
        label, score = adaptive_prediction
    else:
        # In alternativa, utilizzo del modello pre-addestrato
        predictions = get_classifier()(input_data.text)
        result = predictions[0]
        label, score = result["label"], result["score"]
    

    # Calcolo del tempo di esecuzione
    elapsed_time = time.time() - start_time
    
    # Registrazione metriche su Prometheus
    SENTIMENT_COUNT.labels(sentiment_label=label).inc()
    LATENCY_HISTOGRAM.observe(elapsed_time)
    # Registrazione della predizione per il retraining
    adaptive_trainer.record_prediction(input_data.text, label, score)
    
    return SentimentOutput(
        label=label,
        score=score,
        elapsed_time=elapsed_time
    )

@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    # Validazione dell'input del feedback
    if not feedback.text.strip():
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto.")
    # Se il predicted_label è fornito, deve essere una delle etichette valide
    if feedback.label not in LABELS:
        raise HTTPException(status_code=400, detail=f"label must be one of {LABELS}")
    # Aggiunta del feedback al sistema di retraining adattivo
    count = adaptive_trainer.add_feedback(
        feedback.text, feedback.label, feedback.predicted_label
    )
    if count >= adaptive_trainer.min_samples:
        # Avvio del retraining in background se il numero di feedback supera la soglia minima
        background_tasks.add_task(run_retraining)
    return {"accepted": True, "feedback_samples": count}


@app.get("/retraining/status")
def retraining_status():
    # ERestituisce lo stato del processo di retraining
    return adaptive_trainer.status()


@app.post("/retraining/trigger")
def trigger_retraining():
    # Forza il retraining del modello adattivo
    if not adaptive_trainer.retrain(force=True):
        raise HTTPException(status_code=400, detail="Servono almeno due classi etichettate per il retraining.")
    return adaptive_trainer.status()


@app.get("/health")
def health_check():
    # Endpoint di health check per verificare lo stato dell'applicazione
    return {"status": "healthy"}
