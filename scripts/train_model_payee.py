import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

TRAINING_FILE_PAYEE = os.path.join(DATA_DIR, "training_data_payee.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "transaction_model_payee.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer_payee.pkl")


def train_model(
    training_file=TRAINING_FILE_PAYEE,
    model_file=MODEL_FILE,
    vectorizer_file=VECTORIZER_FILE,
):
    """Train and save the payee model using the supplied file paths."""
    # === 1. Load and Prepare Training Data ===
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"{training_file} not found. Please create it with Description and Payee columns.")

    df = pd.read_csv(training_file)

    # Basic checks
    if 'Description' not in df.columns or 'Payee' not in df.columns:
        raise ValueError("Your CSV must have columns: Description and Payee")

    # Clean text
    df['Description'] = df['Description'].astype(str).str.lower().str.strip()
    df['Payee'] = df['Payee'].astype(str).str.strip()

    # === 2. Vectorize Text ===
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['Description'])
    y = df['Payee']

    # === 3. Train the Model ===
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # === 4. Save Model and Vectorizer ===
    os.makedirs(os.path.dirname(os.path.abspath(model_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(vectorizer_file)), exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

    print(f"Model trained on {len(df)} examples and saved to '{model_file}'")
    print(f"Vectorizer saved to '{vectorizer_file}'")
    return model, vectorizer


def main():
    """Train using the default project files."""
    train_model()


if __name__ == "__main__":
    main()
