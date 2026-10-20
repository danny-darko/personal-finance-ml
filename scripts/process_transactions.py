import pandas as pd
import joblib
import os

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

BANK_XLSX = os.path.join(DATA_DIR, "bank_statement.xlsx")
MODEL_FILE_CATEGORY = os.path.join(MODEL_DIR, "transaction_model_category.pkl")
VECTORIZER_FILE_CATEGORY = os.path.join(MODEL_DIR, "vectorizer_category.pkl")
MODEL_FILE_PAYEE = os.path.join(MODEL_DIR, "transaction_model_payee.pkl")
VECTORIZER_FILE_PAYEE = os.path.join(MODEL_DIR, "vectorizer_payee.pkl")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_transactions.csv")


def process_transactions(
    bank_xlsx=BANK_XLSX,
    model_file_category=MODEL_FILE_CATEGORY,
    vectorizer_file_category=VECTORIZER_FILE_CATEGORY,
    model_file_payee=MODEL_FILE_PAYEE,
    vectorizer_file_payee=VECTORIZER_FILE_PAYEE,
    output_file=OUTPUT_FILE,
):
    """Classify a statement and save and return the tracker-formatted output."""
    # === 1. Check model files exist ===
    if not os.path.exists(model_file_category) or not os.path.exists(vectorizer_file_category):
        raise FileNotFoundError("Category model or vectorizer not found. Run train_model_category.py first.")
    if not os.path.exists(model_file_payee) or not os.path.exists(vectorizer_file_payee):
        raise FileNotFoundError("Payee model or vectorizer not found. Run train_model_payee.py first.")

    # === 2. Load models and vectorizers ===
    model_category = joblib.load(model_file_category)
    vectorizer_category = joblib.load(vectorizer_file_category)

    model_payee = joblib.load(model_file_payee)
    vectorizer_payee = joblib.load(vectorizer_file_payee)

    # === 3. Load and clean bank data ===
    df = pd.read_excel(bank_xlsx, skiprows=4, usecols="B,D,F,G", engine='openpyxl')

    # Rename columns to consistent names
    df.columns = ['Date', 'Description', 'MoneyIn', 'MoneyOut']

    # Clean description text
    df['Description'] = df['Description'].astype(str).str.lower().str.strip()

    # === 4. Predict categories ===
    X_category = vectorizer_category.transform(df['Description'])
    df['PredictedCategory'] = model_category.predict(X_category)

    # === 5. Predict payees ===
    X_payee = vectorizer_payee.transform(df['Description'])
    df['PredictedPayee'] = model_payee.predict(X_payee)

    # === 6. Format output ===
    output_df = pd.DataFrame({
        'Date': df['Date'],
        'Outflow': df['MoneyOut'],
        'Inflow': df['MoneyIn'],
        'Category': df['PredictedCategory'],
        'Account': 'Main',
        'Payee': df['PredictedPayee'],
        'Memo': "'" + df['PredictedPayee'] + "' " + df['Description']
    })

    # === 7. Sort by Date (ascending) ===
    output_df = output_df.sort_values(by='Date', ascending=True)

    # === 8. Save to CSV ===
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    print(f"Categorized transactions saved to '{output_file}'")
    return output_df


def main():
    """Process the default project statement."""
    process_transactions()


if __name__ == "__main__":
    main()
