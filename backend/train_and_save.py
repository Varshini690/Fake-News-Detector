import os
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH_TRUE = os.path.join(PROJECT_ROOT, "dataset", "True.csv")
DATA_PATH_FAKE = os.path.join(PROJECT_ROOT, "dataset", "Fake.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "model.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_save():
    print("Loading dataset...")
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

    print("Training MultinomialNB model...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
        ("clf", MultinomialNB()),
    ])

    pipeline.fit(X, y)

    print("Saving model...")
    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train_and_save()
