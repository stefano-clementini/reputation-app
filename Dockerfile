FROM python:3.11-slim
WORKDIR /code
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*
COPY ./app/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN mkdir -p /code/.cache && chmod -R 777 /code/.cache
ENV HF_HOME=/code/.cache
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"
COPY ./app /code/app
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
