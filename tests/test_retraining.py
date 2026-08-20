from app.retraining import AdaptiveSentimentTrainer


def test_retraining_builds_adaptive_classifier(tmp_path):
    # Addestra l'AdaptiveSentimentTrainer con nuovi dati
    trainer = AdaptiveSentimentTrainer(data_dir=str(tmp_path), min_samples=2)
    trainer.add_feedback("the product is brilliant", "positive")
    trainer.add_feedback("the service is awful", "negative")

    # Verifica che il modello sia stato addestrato correttamente e che le predizioni siano accurate
    assert trainer.retrain() is True
    assert trainer.status()["model_version"] == 1
    assert trainer.predict("brilliant product")[0] == "positive"
    assert trainer.predict("awful service")[0] == "negative"


def test_retraining_waits_for_two_labels(tmp_path):
    # Addestra l'AdaptiveSentimentTrainer con nuovi dati
    trainer = AdaptiveSentimentTrainer(data_dir=str(tmp_path), min_samples=1)
    trainer.add_feedback("great", "positive")
    # Verifica che il modello non sia addestrato se non ci sono abbastanza etichette
    assert trainer.retrain() is False