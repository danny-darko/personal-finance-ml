import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to /src
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAINING_FILE_PAYEE = os.path.join(DATA_DIR, "training_data_payee.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "transaction_model_payee.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer_payee.pkl")

# === 1. Load and Prepare Training Data ===
if not os.path.exists(TRAINING_FILE_PAYEE):
    raise FileNotFoundError(f"{TRAINING_FILE_PAYEE} not found. Please create it with Description and Payee columns.")

df = pd.read_csv(TRAINING_FILE_PAYEE)

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
joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)

print(f"✅ Model trained on {len(df)} examples and saved to '{MODEL_FILE}'")
print(f"✅ Vectorizer saved to '{VECTORIZER_FILE}'")
