"""Evaluate category and payee classification on held-out examples."""

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

if __package__:
    from .model_training import fit_model, load_training_data
else:
    from model_training import fit_model, load_training_data


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_FILES = {
    "Category": DATA_DIR / "training_data_category.csv",
    "Payee": DATA_DIR / "training_data_payee.csv",
}


def evaluate_model(
    training_file=None, target_column="Category", test_size=0.2, random_state=42
):
    """Fit on a training split and return held-out accuracy and a text report.

    Evaluation does not save or overwrite models. When no path is supplied,
    select the project's training CSV for the requested target.
    """
    if target_column not in TRAINING_FILES:
        raise ValueError("target_column must be 'Category' or 'Payee'")
    if training_file is None:
        training_file = TRAINING_FILES[target_column]
    descriptions, labels = load_training_data(training_file, target_column)

    descriptions_train, descriptions_test, y_train, y_test = train_test_split(
        descriptions, labels, test_size=test_size, random_state=random_state
    )
    model, vectorizer = fit_model(descriptions_train, y_train)
    X_test = vectorizer.transform(descriptions_test)
    y_pred = model.predict(X_test)

    # Include every dataset class, even those absent from this small test split.
    report = classification_report(
        y_test, y_pred, labels=sorted(labels.unique()), zero_division=0
    )
    return accuracy_score(y_test, y_pred), report


def main(argv=None):
    """Print separate reports for both targets, or a selected target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("category", "payee", "both"), default="both",
        help="Which prediction target to evaluate (default: both).",
    )
    parser.add_argument(
        "--training-file", type=Path,
        help="Custom CSV; requires --target category or --target payee.",
    )
    args = parser.parse_args(argv)
    if args.training_file is not None and args.target == "both":
        parser.error("--training-file requires --target category or --target payee")

    targets = ("Category", "Payee") if args.target == "both" else (args.target.title(),)
    for target in targets:
        accuracy, report = evaluate_model(args.training_file, target_column=target)
        print(f"{target} evaluation")
        print("Accuracy:", round(accuracy * 100, 2), "%")
        print("Classification Report:")
        print(report)


if __name__ == "__main__":
    main()
