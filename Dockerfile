FROM python:3.12-slim
# Crea una directory di lavoro per l'applicazione
WORKDIR /code
# Installa le librerie di sistema necessarie
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# Copia il file requirements.txt nella directory di lavoro
COPY ./app/requirements.txt /code/requirements.txt
# Installa le dipendenze Python specificate nel file requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN mkdir -p /code/.cache && chmod -R 777 /code/.cache
ENV HF_HOME=/code/.cache
# Installa il modello di sentiment analysis di HuggingFace
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"
# Copia il codice dell'applicazione nella directory di lavoro
COPY ./app /code/app
EXPOSE 7860
# Avvia l'applicazione FastAPI utilizzando Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
