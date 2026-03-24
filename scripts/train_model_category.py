import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # points to /src
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAINING_FILE_CATEGORY = os.path.join(DATA_DIR, "training_data_category.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "transaction_model_category.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer_category.pkl")

# === 1. Load and Prepare Training Data ===
if not os.path.exists(TRAINING_FILE_CATEGORY):
    raise FileNotFoundError(f"{TRAINING_FILE_CATEGORY} not found. Please create it with Description and Category columns.")

df = pd.read_csv(TRAINING_FILE_CATEGORY)

# Basic checks
if 'Description' not in df.columns or 'Category' not in df.columns:
    raise ValueError("Your CSV must have columns: Description and Category")

# Clean text
df['Description'] = df['Description'].astype(str).str.lower().str.strip()
df['Category'] = df['Category'].astype(str).str.strip()

# === 2. Vectorize Text ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigrams + bigrams
X = vectorizer.fit_transform(df['Description'])
y = df['Category']

# === 3. Train the Model ===
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# === 4. Save Model and Vectorizer ===
joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)

print(f"✅ Model trained on {len(df)} examples and saved to '{MODEL_FILE}'")
print(f"✅ Vectorizer saved to '{VECTORIZER_FILE}'")
