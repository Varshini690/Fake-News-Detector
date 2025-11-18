import os
import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

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

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.on_event("startup")
def load_model():
    global pipeline
    print("[INFO] Loading pre-trained model...")
    pipeline = joblib.load(MODEL_PATH)
    print("[INFO] Model loaded.")

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

@app.get("/")
def root():
    return {"message": "API running"}
