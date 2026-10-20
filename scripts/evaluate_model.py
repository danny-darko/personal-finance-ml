import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_FILE = os.path.join(BASE_DIR, "..", "data", "training_data_category.csv")


def evaluate_model(training_file=TRAINING_FILE):
    """Evaluate category classification and return accuracy and a text report."""
    # === 1. Load training data ===
    df = pd.read_csv(training_file)
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
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)


def main():
    """Print evaluation results for the default category training data."""
    accuracy, report = evaluate_model()
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
