import os
os.environ["TESTING"] = "1"
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_sentiment_prediction():
    response = client.post("/predict", json={"text": "I love MachineInnovators Inc!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["label"] in ["positive", "neutral", "negative"]
    