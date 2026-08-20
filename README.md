# reputation-app

Il progetto prevede l'implementazione di un applicativo che:

1. Automazione dell'Analisi del sentiment: Implementando un modello di analisi del sentiment, MLOps Innovators Inc. automatizzerà l'elaborazione dei dati dai social media per identificare sentiment positivi, neutrali e negativi. Ciò permetterà una risposta rapida e mirata ai feedback degli utenti.

2. Monitoraggio Continuo della Reputazione: Utilizzando metodologie MLOps, l'azienda implementerà un sistema di monitoraggio continuo per valutare l'andamento del sentiment degli utenti nel tempo. Questo consentirà di rilevare rapidamente cambiamenti nella percezione dell'azienda e di intervenire prontamente se necessario.

3. Retraining del Modello: Introdurre un sistema di retraining automatico per il modello di analisi del sentiment assicurerà che l'algoritmo si adatti dinamicamente ai nuovi dati e alle variazioni nel linguaggio e nei comportamenti degli utenti sui social media. Mantenere alta l'accuratezza predittiva del modello è essenziale per una valutazione corretta del sentiment.

Le fasi di sviluppo sono state:
Fase 1: Implementazione di un'applicazione FastAPI che consentisse la consultazione di un Modello di Analisi del sentiment. In particolare il modello utilizzato è stato "cardiffnlp/twitter-roberta-base-sentiment-latest"

Dataset: Utilizzare dataset pubblici contenenti testi e le rispettive etichette di sentiment.

Fase 2: Creazione della Pipeline CI/CD per automatizzare il training del modello, i test di integrazione e il deploy dell'applicazione su HuggingFace.

Per implementare il retraining automatico, l'API raccoglie feedback supervisionati tramite `POST /feedback`. Al raggiungimento di `RETRAINING_MIN_SAMPLES` il retraining viene avviato in background e il modello adattivo viene salvato nel volume Docker `adaptive_data`.
Il drift della distribuzione delle predizioni è esposto in `/retraining/status`. Il retraining manuale, utile per operazioni controllate, è disponibile su `POST /retraining/trigger`.

Fase 3: Deploy e Monitoraggio Continuo Deploy su HuggingFace.
Il Sistema di Monitoraggio è stato configurato utilizzando Prometheus e Grafana.