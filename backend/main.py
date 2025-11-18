import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from fastapi.middleware.cors import CORSMiddleware

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH_TRUE = os.path.join(PROJECT_ROOT, "dataset", "True.csv")
DATA_PATH_FAKE = os.path.join(PROJECT_ROOT, "dataset", "Fake.csv")

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    title: str = ""
    text: str = ""

class PredictResponse(BaseModel):
    prediction: str
    label: int
    confidence: float

pipeline = None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_train():
    global pipeline

    print("=== TRAINING MODEL ===")

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

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
        ("clf", MultinomialNB())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[MODEL] Training complete. Accuracy: {acc:.4f}")


@app.on_event("startup")
def startup():
    load_and_train()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global pipeline

    content = clean_text(req.title + " " + req.text)
    proba = pipeline.predict_proba([content])[0]
    label = int(pipeline.predict([content])[0])
    confidence = float(proba[label])

    return {
        "prediction": "True" if label == 1 else "Fake",
        "label": label,
        "confidence": confidence
    }


@app.get("/")
def root():
    return {"message": "API running"}
