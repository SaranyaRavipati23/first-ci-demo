import json
import os
import unittest

try:
    import joblib
    import numpy as np
    import pandas as pd
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "student_results.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_result_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# All four feature columns used by the model (must match train_model.py and app.py)
FEATURE_COLS = ["internal_marks", "attendance", "assignment_marks", "previous_score"]


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Required ML dependencies are not installed.")
class TestMLPipeline(unittest.TestCase):
    def test_dataset_created(self):
        """Verify student_results.csv exists and has 300 records with required columns."""
        self.assertTrue(os.path.isfile(DATASET_PATH), f"Dataset file not found at {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        self.assertEqual(len(df), 300, f"Expected 300 rows, found {len(df)}")
        expected_columns = set(FEATURE_COLS) | {"passed"}
        self.assertTrue(expected_columns.issubset(df.columns), f"Missing columns in {df.columns}")

    def test_model_created(self):
        """Verify student_result_model.pkl exists and can be loaded via joblib."""
        self.assertTrue(os.path.isfile(MODEL_PATH), f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        self.assertIsNotNone(model, "Loaded model is None")
        self.assertTrue(hasattr(model, "predict"), "Loaded artifact is not a predictive model")

    def test_metrics_created(self):
        """Verify metrics.json exists, is valid JSON, and contains an accuracy field."""
        self.assertTrue(os.path.isfile(METRICS_PATH), f"Metrics file not found at {METRICS_PATH}")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertIsInstance(metrics, dict, "metrics.json does not contain a JSON object")
        self.assertIn("accuracy", metrics, "metrics.json is missing 'accuracy' field")

    def test_accuracy_is_valid(self):
        """Verify that accuracy is within a valid probability range [0.0, 1.0]."""
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        accuracy = metrics.get("accuracy")
        self.assertIsInstance(accuracy, (float, int), "Accuracy must be a numerical value")
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy cannot be negative")
        self.assertLessEqual(accuracy, 1.0, "Accuracy cannot exceed 1.0")
        self.assertGreaterEqual(accuracy, 0.70, "Accuracy is unexpectedly low for synthetic data")

    def test_model_prediction(self):
        """Verify model prediction outputs expected shape and valid binary class values."""
        model = joblib.load(MODEL_PATH)
        sample = pd.DataFrame(
            {
                "internal_marks": [60, 40],
                "attendance": [80, 50],
                "assignment_marks": [70, 45],
                "previous_score": [75, 40],
            }
        )
        preds = model.predict(sample)
        self.assertEqual(len(preds), 2, "Model did not return predictions for all input rows")
        for pred in preds:
            self.assertIn(int(pred), [0, 1], f"Unexpected prediction value: {pred}")

    def test_high_performance_student(self):
        """Verify that a high-performing student passes (prediction == 1)."""
        model = joblib.load(MODEL_PATH)
        high_perf_student = pd.DataFrame(
            [
                {
                    "internal_marks": 90,
                    "attendance": 95,
                    "assignment_marks": 90,
                    "previous_score": 92,
                }
            ]
        )
        prediction = model.predict(high_perf_student)
        self.assertEqual(int(prediction[0]), 1, "High performance student should PASS (1)")

    def test_low_performance_student(self):
        """Verify that a low-performing student fails (prediction == 0)."""
        model = joblib.load(MODEL_PATH)
        low_perf_student = pd.DataFrame(
            [
                {
                    "internal_marks": 20,
                    "attendance": 30,
                    "assignment_marks": 20,
                    "previous_score": 22,
                }
            ]
        )
        prediction = model.predict(low_perf_student)
        self.assertEqual(int(prediction[0]), 0, "Low performance student should FAIL (0)")


if __name__ == "__main__":
    unittest.main()
