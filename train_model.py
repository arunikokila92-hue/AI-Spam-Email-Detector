import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
BASE_DATASET = ROOT / "spam.csv"
EXTRA_DATASET = ROOT / "dataset" / "spamassassin_emails.csv"
MODEL_PATH = ROOT / "model.pkl"


def load_csv(path: Path) -> pd.DataFrame:
    """Load a dataset and keep the project's common label/text columns."""
    try:
        data = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    except UnicodeDecodeError:
        data = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)

    required = {"label", "text"}
    if not required.issubset(data.columns):
        raise ValueError(
            f"{path} must contain columns named 'label' and 'text'. "
            f"Found: {list(data.columns)}"
        )

    return data[["label", "text"]].copy()


def normalize_message(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def main() -> None:
    if not BASE_DATASET.exists():
        raise FileNotFoundError(f"Could not find {BASE_DATASET}")

    datasets = [load_csv(BASE_DATASET)]
    print(f"Loaded existing dataset: {BASE_DATASET}")

    if EXTRA_DATASET.exists():
        datasets.append(load_csv(EXTRA_DATASET))
        print(f"Loaded additional email dataset: {EXTRA_DATASET}")
    else:
        print(
            "Additional SpamAssassin dataset not found. "
            "Run: python dataset/download_spamassassin.py"
        )

    data = pd.concat(datasets, ignore_index=True)
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["label"].isin(["spam", "ham"])]
    data = data[data["text"].str.len() > 0]

    # Remove duplicates across the original and additional datasets.
    data["_dedupe_key"] = data["text"].map(normalize_message)
    before = len(data)
    data = data.drop_duplicates(subset="_dedupe_key").drop(columns="_dedupe_key")
    print(f"Removed duplicate/empty/invalid rows: {before - len(data)}")

    counts = data["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError(f"Need at least two examples of each class. Counts: {counts.to_dict()}")

    X = data["text"]
    y = data["label"]

    print("\nDataset summary:")
    print(f"Total messages: {len(data)}")
    print(counts.to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel evaluation (held-out test set):")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix [ham, spam]:")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"\nModel saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()
