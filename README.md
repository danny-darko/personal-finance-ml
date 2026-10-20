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

Run these commands from the repository root:

```bash
python scripts/train_model_payee.py
python scripts/train_model_category.py
python scripts/process_transactions.py
```

Each script has a `main()` entry point guarded by `if __name__ == "__main__"`.
Importing a script only defines its configuration and functions; it does not
read data, train models, create directories, or write output. Call `train_model()`,
`process_transactions()`, or `evaluate_model()` explicitly to run that work.
These functions accept file paths so callers can use separate inputs and outputs.

### Evaluation

Evaluate both prediction targets with separate held-out reports:

```bash
python scripts/evaluate_model.py
```

Select a target, optionally using a different labelled dataset:

```bash
python scripts/evaluate_model.py --target payee
python scripts/evaluate_model.py --target category --training-file data/training_data_category.csv
```

Category data needs `Description` and `Category` columns; payee data needs
`Description` and `Payee`. Default paths are resolved relative to the project,
so they do not depend on the terminal's working directory.

Training and evaluation share data preparation and TF-IDF/logistic-regression
fitting in `scripts/model_training.py`. Evaluation splits descriptions and labels
before fitting TF-IDF, uses an 80/20 split with seed 42 by default, and does not
save or overwrite models. Training scripts still fit on their full datasets
and save the model and vectorizer files used by transaction processing.

Reports include all classes, including classes with zero examples in the test
split; undefined precision/recall values are reported as zero. The checked-in
datasets contain only 26 examples each, so these reports demonstrate the workflow
and are not reliable estimates of real-world accuracy. The split is currently
unstratified: six test rows cannot represent all seven categories or thirteen
payees. More labelled data is needed for representative evaluation.

## Tests

Run the import-safety and workflow checks with the standard-library test runner:

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data and temporary directories for models and exports.

## Project Structure
```
PersonalFinanceHelper/
├── data/          # Training data and bank statement
├── models/        # Trained model artifacts
├── output/        # Processed transaction output
├── scripts/       # Entry points and shared model training
└── tests/         # Synthetic-data checks
```

## Notes
- Add your bank statement as `data/bank_statement.xlsx`
- Personal data files should be named with `_personal` suffix to keep them gitignored
- Output is displayed from dummy data as an example
