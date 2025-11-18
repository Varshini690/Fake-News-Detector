import os
import re
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "model.pkl")
DATA_PATH_TRUE = os.path.join(PROJECT_ROOT, "dataset", "True.csv")
DATA_PATH_FAKE = os.path.join(PROJECT_ROOT, "dataset", "Fake.csv")

app = FastAPI(title="Fake News Detector API")

# CORS — allow your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fake-news-detector-flame-one.vercel.app",
        "http://localhost:5173",
        "*"  # keep for now, can remove later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request + Response models
class PredictRequest(BaseModel):
    title: str = ""
    text: str = ""

class PredictResponse(BaseModel):
    prediction: str
    label: int
    confidence: float

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Train model & save
def train_and_save():
    print("[INFO] Loading dataset...")

    true_df = pd.read_csv(DATA_PATH_TRUE)
    fake_df = pd.read_csv(DATA_PATH_FAKE)

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)

    df["title"] = df["title"].fillna("").apply(clean_text)
    df["text"] = df["text"].fillna("").apply(clean_text)
    df["content"] = df["title"] + " " + df["text"]

    X = df["content"]
    y = df["label"]

    print("[INFO] Training MultinomialNB model...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
        ("clf", MultinomialNB()),
    ])

    pipeline.fit(X, y)

    print("[INFO] Saving model...")
    joblib.dump(pipeline, MODEL_PATH)
    print("[INFO] Model saved at:", MODEL_PATH)

# Load model on startup
@app.on_event("startup")
def load_model():
    global pipeline

    if not os.path.exists(MODEL_PATH):
        print("[WARNING] model.pkl not found. Training...")
        train_and_save()

    print("[INFO] Loading model...")
    pipeline = joblib.load(MODEL_PATH)
    print("[INFO] Model loaded.")

# Predict endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    content = clean_text(req.title + " " + req.text)

    proba = pipeline.predict_proba([content])[0]
    label = int(pipeline.predict([content])[0])

    return {
        "prediction": "True" if label == 1 else "Fake",
        "label": label,
        "confidence": float(proba[label])
    }

# Root
@app.get("/")
def root():
    return {"message": "API running!"}
