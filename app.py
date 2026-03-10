from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
import requests
import os
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI(title="Gender Prediction API")

# =========================
# Configuration
# =========================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
REQUEST_TIMEOUT = 20

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "genderdb")
DB_USER = os.getenv("DB_USER", "genderuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "genderpass")

# =========================
# Load Classical Model
# =========================

try:
    sklearn_model = joblib.load("model.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load model.joblib: {e}")

# =========================
# Database Utilities
# =========================

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

@app.on_event("startup")
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            model TEXT NOT NULL,
            prediction TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def save_prediction(name: str, model: str, prediction: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (name, model, prediction) VALUES (%s, %s, %s)",
        (name, model, prediction)
    )
    conn.commit()
    cursor.close()
    conn.close()

# =========================
# Response Model
# =========================

class PredictionResponse(BaseModel):
    name: str
    model: str
    prediction: str

# =========================
# Prediction Functions
# =========================

def predict_classic(name: str) -> str:
    prediction = sklearn_model.predict([name])[0]
    return str(prediction)

def predict_llm(name: str) -> str:
    prompt = (
        f"Is the French first name '{name}' typically Male or Female? "
        f"Answer with only one word: Male or Female."
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="LLM request timed out")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

# =========================
# API Endpoints
# =========================

@app.get("/predict", response_model=PredictionResponse)
def predict(
    name: str,
    model: Literal["classic", "llm"] = Query("classic")
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name must not be empty")

    name = name.strip()

    if model == "classic":
        prediction = predict_classic(name)

    elif model == "llm":
        prediction = predict_llm(name)

    else:
        raise HTTPException(status_code=400, detail="Invalid model parameter")

    # Save prediction to database
    save_prediction(name, model, prediction)

    return PredictionResponse(
        name=name,
        model=model,
        prediction=prediction
    )

@app.get("/history")
def history():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT id, name, model, prediction, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 10
    """)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

@app.get("/health")
def health():
    return {"status": "ok"}