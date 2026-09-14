"""
Train a small classifier on the arm-pose features recorded by
collect_gesture_data.py, and save it for drive_with_gestures.py to use.

    python3 train_gesture_classifier.py

Prints a held-out accuracy and per-class precision/recall so you can see how
well it actually generalizes, then saves the fitted model to
models/gesture_classifier.joblib.
"""

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pose_features import FEATURE_NAMES

CSV_PATH = Path(__file__).parent / "gesture_data.csv"
MODEL_PATH = Path(__file__).parent / "models" / "gesture_classifier.joblib"


def load_data():
    X, y = [], []
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == [*FEATURE_NAMES, "label"], (
            "gesture_data.csv doesn't match the current feature set — "
            "delete it and re-record with collect_gesture_data.py."
        )
        for row in reader:
            *features, label = row
            X.append([float(v) for v in features])
            y.append(label)
    return np.array(X), np.array(y)


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found — run collect_gesture_data.py first."
        )

    X, y = load_data()
    print(f"Loaded {len(X)} examples across classes: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nHeld-out test accuracy:", model.score(X_test, y_test))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=predicted):")
    labels = sorted(set(y))
    print(labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
