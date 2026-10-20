"""Shared data preparation, model fitting, and artifact persistence."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_training_data(training_file, target_column):
    """Load and clean descriptions and labels for the requested target."""
    training_file = Path(training_file)
    if not training_file.exists():
        raise FileNotFoundError(
            f"{training_file} not found. Please create it with Description and "
            f"{target_column} columns."
        )

    df = pd.read_csv(training_file)
    if "Description" not in df.columns or target_column not in df.columns:
        raise ValueError(f"Your CSV must have columns: Description and {target_column}")

    df["Description"] = df["Description"].astype(str).str.lower().str.strip()
    df[target_column] = df[target_column].astype(str).str.strip()
    return df["Description"], df[target_column]


def fit_model(descriptions, labels):
    """Fit TF-IDF and logistic regression using only the supplied examples."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(descriptions)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    return model, vectorizer


def train_and_save_model(training_file, target_column, model_file, vectorizer_file):
    """Fit on the full dataset and save artifacts for statement processing."""
    descriptions, labels = load_training_data(training_file, target_column)
    model, vectorizer = fit_model(descriptions, labels)

    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    Path(vectorizer_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

    print(f"Model trained on {len(descriptions)} examples and saved to '{model_file}'")
    print(f"Vectorizer saved to '{vectorizer_file}'")
    return model, vectorizer
