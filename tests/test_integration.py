import os
# Imposta la variabile d'ambiente TESTING a "1" per indicare che stiamo eseguendo i test
os.environ["TESTING"] = "1"
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_sentiment_prediction():
    # invia una richiesta POST all'endpoint /predict con un testo di esempio
    response = client.post("/predict", json={"text": "I love MachineInnovators Inc!"})
    # verifica che la risposta abbia lo status code 200
    assert response.status_code == 200
    data = response.json()
    # verifica che la risposta contenga le chiavi "label" e "score"
    assert "label" in data
    assert "score" in data
    # verifica che il valore della chiave "label" sia uno dei valori attesi
    assert data["label"] in ["positive", "neutral", "negative"]
    