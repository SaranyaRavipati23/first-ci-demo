"""
Practical 5: Unit tests for the Student Result Prediction Flask API (app.py).
Uses Flask's built-in test client — no running server required.
"""

import json
import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):
    def setUp(self):
        """Configure the Flask test client before each test."""
        app.config["TESTING"] = True
        self.client = app.test_client()

    # ------------------------------------------------------------------
    # Test 1: Health endpoint
    # ------------------------------------------------------------------
    def test_health_endpoint(self):
        """GET / should return HTTP 200 and status == 'ok'."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body.get("status"), "ok")

    # ------------------------------------------------------------------
    # Test 2: High-performance student → PASS
    # ------------------------------------------------------------------
    def test_high_performance_prediction(self):
        """POST /predict with high marks should return prediction == 'PASS'."""
        payload = {
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80,
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body.get("prediction"), "PASS")
        self.assertEqual(body.get("prediction_code"), 1)

    # ------------------------------------------------------------------
    # Test 3: Low-performance student → FAIL
    # ------------------------------------------------------------------
    def test_low_performance_prediction(self):
        """POST /predict with low marks should return prediction == 'FAIL'."""
        payload = {
            "attendance": 55,
            "internal_marks": 30,
            "assignment_marks": 40,
            "previous_score": 35,
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body.get("prediction"), "FAIL")
        self.assertEqual(body.get("prediction_code"), 0)

    # ------------------------------------------------------------------
    # Test 4: Missing field → HTTP 400
    # ------------------------------------------------------------------
    def test_missing_field_validation(self):
        """POST /predict with a missing required field should return HTTP 400."""
        payload = {
            "attendance": 90,
            "internal_marks": 85,
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertIn("missing_fields", body)


if __name__ == "__main__":
    unittest.main()
