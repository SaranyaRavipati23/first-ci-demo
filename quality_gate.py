"""
Practical 3: ML Model Quality Gate
Reads model evaluation metrics from metrics.json and enforces a
minimum accuracy threshold. Exits with code 1 if the gate fails so
that the GitHub Actions workflow step is marked as failed.
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# Configurable minimum accuracy threshold (80%)
ACCURACY_THRESHOLD = 0.80


def load_metrics(metrics_path: str = METRICS_PATH) -> dict:
    """Load and return the metrics dictionary from a JSON file.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
        ValueError: If the file is not valid JSON or is not a JSON object.
    """
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metrics.json contains invalid JSON: {exc}") from exc

    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain a JSON object (dict).")

    return metrics


def check_accuracy(metrics: dict, threshold: float = ACCURACY_THRESHOLD) -> bool:
    """Return True if accuracy meets or exceeds the threshold, False otherwise.

    Raises:
        ValueError: If the 'accuracy' key is missing or is not a number.
    """
    if "accuracy" not in metrics:
        raise ValueError("'accuracy' key is missing from metrics.")

    accuracy = metrics["accuracy"]
    if not isinstance(accuracy, (int, float)):
        raise ValueError(
            f"'accuracy' must be a numeric value, got {type(accuracy).__name__}."
        )

    return float(accuracy) >= threshold


def run_quality_gate(
    metrics_path: str = METRICS_PATH,
    threshold: float = ACCURACY_THRESHOLD,
) -> None:
    """Run the quality gate end-to-end and exit with an appropriate code.

    Prints a PASS or FAIL banner to stdout. Exits with code 1 on failure
    so that the CI workflow step is marked as failed.
    """
    print("=" * 50)
    print("  ML Model Quality Gate")
    print(f"  Minimum Accuracy Threshold: {threshold:.2f}")
    print("=" * 50)

    try:
        metrics = load_metrics(metrics_path)
        accuracy = metrics["accuracy"]
        passed = check_accuracy(metrics, threshold)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] Quality gate failed due to an exception:\n  {exc}")
        print("\n[RESULT] FAIL — quality gate could not be evaluated.\n")
        sys.exit(1)

    print(f"\n  Model Accuracy : {accuracy:.4f}")
    print(f"  Threshold      : {threshold:.2f}")

    if passed:
        print("\n[RESULT] PASS — model accuracy meets the quality threshold.\n")
    else:
        print(
            f"\n[RESULT] FAIL — model accuracy ({accuracy:.4f}) is below "
            f"the threshold ({threshold:.2f}).\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    run_quality_gate()
