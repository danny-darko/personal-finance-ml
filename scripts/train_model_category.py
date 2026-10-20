import os

if __package__:
    from .model_training import train_and_save_model
else:
    from model_training import train_and_save_model

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

TRAINING_FILE_CATEGORY = os.path.join(DATA_DIR, "training_data_category.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "transaction_model_category.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer_category.pkl")


def train_model(
    training_file=TRAINING_FILE_CATEGORY,
    model_file=MODEL_FILE,
    vectorizer_file=VECTORIZER_FILE,
):
    """Train and save the category model using the supplied file paths."""
    return train_and_save_model(
        training_file, "Category", model_file, vectorizer_file
    )


def main():
    """Train using the default project files."""
    train_model()


if __name__ == "__main__":
    main()
