from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import time
import os

# Strumentazione Prometheus
from prometheus_client import make_asgi_app, Counter, Histogram

app = FastAPI(title="Sentiment Analysis API")

# 1. Endpoint /metrics nativo integrato in FastAPI come applicazione ASGI sussidiaria
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 2. Definizione delle metriche Prometheus
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

# Inizializzazione modello (lazy)
classifier = None

def get_classifier():
    global classifier
    # During tests we may want to avoid downloading/loading a large model
    if os.getenv("TESTING") == "1":
        # simple stub classifier for tests
        return lambda text: [{"label": "positive", "score": 0.99}]

    if classifier is None:
        try:
            classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            raise RuntimeError(f"Errore nel caricamento del modello: {e}")
    return classifier

class TextInput(BaseModel):
    text: str
    
class SentimentOutput(BaseModel):
    label: str
    score: float
    elapsed_time: float

@app.post("/predict", response_model=SentimentOutput)
async def predict_sentiment(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto.")
    
    start_time = time.time()
    
    # Inferenza del modello (usa loader lazy)
    clf = get_classifier()
    predictions = clf(input_data.text)
    result = predictions[0]
    
    elapsed_time = time.time() - start_time
    
    # Registrazione metriche su Prometheus
    SENTIMENT_COUNT.labels(sentiment_label=result['label']).inc()
    LATENCY_HISTOGRAM.observe(elapsed_time)
    
    return SentimentOutput(
        label=result['label'],
        score=result['score'],
        elapsed_time=elapsed_time
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}
