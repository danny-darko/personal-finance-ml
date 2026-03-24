import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === 1. Load training data ===
df = pd.read_csv('.venv/src/Data/training_data.csv')
df['Description'] = df['Description'].astype(str).str.lower().str.strip()
df['Category'] = df['Category'].astype(str).str.strip()

# === 2. Split into train/test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    df['Description'], df['Category'], test_size=0.2, random_state=42
)

# === 3. Vectorize ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 4. Train model ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# === 5. Predict and evaluate ===
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
