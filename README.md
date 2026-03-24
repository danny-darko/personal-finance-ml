# Personal Finance ML

A Python ML tool that classifies Santander bank statement transactions using Logistic Regression. Automatically predicts payee and spending category from transaction descriptions, then outputs a formatted CSV ready to paste into a personal finance Excel tracker (Aspire Budgeting v4).

## Setup

1. Clone the repo
2. Create a virtual environment and activate it
```bash
    python -m venv .venv
    .venv\Scripts\activate
```
3. Install dependencies
```bash
    pip install -r requirements.txt
```

## Usage

Run the scripts in this order:

1. `train_model_payee.py` — trains the payee classification model
2. `train_model_category.py` — trains the category classification model
3. `process_transactions.py` — processes your bank statement and outputs a formatted CSV

## Project Structure
```
PersonalFinanceHelper/
├── data/          # Training data and bank statement
├── models/        # Trained model artifacts
├── output/        # Processed transaction output
└── src/           # Scripts
```

## Notes
- Add your bank statement as `data/bank_statement.xlsx`
- Personal data files should be named with `_personal` suffix to keep them gitignored
- Output is displayed from dummy data as an example