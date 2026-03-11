
# Gender Prediction API — MLOps Deployment

This project implements a **machine learning API for gender prediction from first names**, deployed using **Docker, Google Cloud Run, and CI/CD with GitHub Actions**.

The application supports **two prediction models**:

* a **classical machine learning model** (scikit-learn pipeline)
* a **Large Language Model (LLM)** served through **Ollama**

Predictions are stored in a **PostgreSQL database**, allowing retrieval of prediction history.



# Project Architecture

```
Client
   ↓
FastAPI API
   ↓
Prediction Models
   ├─ scikit-learn model (CountVectorizer + classifier)
   └─ LLM (Ollama / llama3)
   ↓
PostgreSQL database (prediction history)
```

Deployment pipeline:

```
GitHub push
   ↓
GitHub Actions CI/CD
   ↓
Docker build
   ↓
Artifact Registry
   ↓
Cloud Run deployment
```



# Project Structure

```
gender-prediction-api/
│
├── app.py
├── model.joblib
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
│
└── .github/
    └── workflows/
        └── deploy.yml
```



# Features

### API endpoints

| Endpoint   | Description                     |
| ---------- | ------------------------------- |
| `/predict` | Predict gender from a name      |
| `/history` | Returns the last 10 predictions |
| `/health`  | API health check                |

Example request:

```
GET /predict?name=Marie&model=classic
```

Response:

```
{
  "name": "Marie",
  "model": "classic",
  "prediction": "female"
}
```



# Models

### Classical model

The classical model uses:

* **CountVectorizer (character features)**
* **Logistic Regression classifier**

Example features extracted from names:

* last letter
* character frequencies
* character n-grams

This model is stored as:

```
model.joblib
```


### LLM model

An alternative prediction mode uses **Ollama with Llama3**.

Example prompt sent to the model:

```
Is the French first name "Marie" typically Male or Female?
Answer with only one word: Male or Female.
```


# Running the project locally

## Requirements

* Docker
* Docker Compose
* Python 3.11



## Start the application

```
docker compose up --build
```

This starts:

* the FastAPI API
* a PostgreSQL database

The API becomes available at:

```
http://localhost:8001/docs
```



# Environment variables

Configuration is stored in `.env`.

Example:

```
DB_HOST=db
DB_NAME=genderdb
DB_USER=genderuser
DB_PASSWORD=genderpass

OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3
```


# Database

Predictions are stored in a PostgreSQL table created automatically at startup.

Schema:

```
predictions(
  id SERIAL PRIMARY KEY,
  name TEXT,
  model TEXT,
  prediction TEXT,
  created_at TIMESTAMP
)
```



# CI/CD Pipeline

Deployment is automated using **GitHub Actions**.

Pipeline steps:

1. Checkout repository
2. Authenticate to Google Cloud
3. Build Docker image
4. Push image to **Artifact Registry**
5. Deploy to **Cloud Run**

Pipeline file:

```
.github/workflows/deploy.yml
```



# Deployment

The application is deployed on **Google Cloud Run**, a serverless container platform.

Advantages:

* automatic scaling
* HTTPS endpoint
* no infrastructure management



# Technologies Used

* FastAPI
* scikit-learn
* Docker
* PostgreSQL
* Ollama (LLM)
* Google Cloud Run
* Artifact Registry
* GitHub Actions (CI/CD)



# Future Improvements

Possible improvements include:

* authentication and API security
* request logging and monitoring
* model confidence scoring
* automatic retraining pipeline
* infrastructure management with Terraform





