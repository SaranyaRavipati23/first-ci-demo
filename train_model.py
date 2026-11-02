"""
Educational demonstration script for student result classification.
Note: This is a synthetic dataset and demonstration model for MLOps CI/CD learning purposes
and is not intended for real-world student assessment.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "student_results.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_result_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

RANDOM_SEED = 42

# Feature columns used for training and prediction (must match app.py)
FEATURE_COLS = ["internal_marks", "attendance", "assignment_marks", "previous_score"]


def generate_dataset(n_samples: int = 300) -> pd.DataFrame:
    """Generate a deterministic synthetic dataset for demonstration purposes."""
    np.random.seed(RANDOM_SEED)

    internal_marks = np.random.randint(15, 101, size=n_samples)
    attendance = np.random.randint(30, 101, size=n_samples)
    assignment_marks = np.random.randint(15, 101, size=n_samples)
    previous_score = np.random.randint(20, 101, size=n_samples)

    # Weighted composite score to determine binary pass (1) or fail (0)
    score = (
        0.35 * internal_marks
        + 0.25 * attendance
        + 0.20 * assignment_marks
        + 0.20 * previous_score
    )
    passed = (score >= 50).astype(int)

    df = pd.DataFrame(
        {
            "internal_marks": internal_marks,
            "attendance": attendance,
            "assignment_marks": assignment_marks,
            "previous_score": previous_score,
            "passed": passed,
        }
    )
    return df


def train_and_evaluate():
    """Train classification model, evaluate performance, and save artifacts."""
    print("=== Student Result Prediction ML Training ===")

    # 1. Generate demonstration dataset (300 records)
    df = generate_dataset(n_samples=300)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated dataset saved to: {DATASET_PATH}")
    print(f"Total dataset size: {len(df)} records")

    # 2. Features and target split
    X = df[FEATURE_COLS]
    y = df["passed"]

    # 3. Train-test split (240 training, 60 testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=60, train_size=240, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # 4. Train classification model
    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(np.array(cm))

    # 6. Save model artifact
    joblib.dump(model, MODEL_PATH)
    print(f"Model artifact saved to: {MODEL_PATH}")

    # 7. Save evaluation metrics
    metrics = {
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
