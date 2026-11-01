"""
Practical 3: Unit tests for the ML Model Quality Gate.
Tests the quality_gate module in isolation using mock metrics dicts —
no trained model or metrics.json file is required for these tests.
"""

import json
import os
import tempfile
import unittest

from quality_gate import (
    ACCURACY_THRESHOLD,
    check_accuracy,
    load_metrics,
    run_quality_gate,
)


class TestCheckAccuracy(unittest.TestCase):
    """Tests for the check_accuracy() function logic."""

    def test_accuracy_above_threshold(self):
        """Accuracy above 0.80 must return True (gate PASSES)."""
        metrics = {"accuracy": 0.95}
        self.assertTrue(
            check_accuracy(metrics, threshold=0.80),
            "Accuracy 0.95 >= 0.80 should PASS the quality gate.",
        )

    def test_accuracy_exactly_at_threshold(self):
        """Accuracy exactly equal to 0.80 must return True (gate PASSES)."""
        metrics = {"accuracy": 0.80}
        self.assertTrue(
            check_accuracy(metrics, threshold=0.80),
            "Accuracy 0.80 == threshold should PASS the quality gate.",
        )

    def test_accuracy_below_threshold(self):
        """Accuracy below 0.80 must return False (gate FAILS)."""
        metrics = {"accuracy": 0.75}
        self.assertFalse(
            check_accuracy(metrics, threshold=0.80),
            "Accuracy 0.75 < 0.80 should FAIL the quality gate.",
        )

    def test_accuracy_zero(self):
        """Accuracy of 0.0 must return False (gate FAILS)."""
        metrics = {"accuracy": 0.0}
        self.assertFalse(check_accuracy(metrics, threshold=0.80))

    def test_accuracy_one(self):
        """Accuracy of 1.0 must return True (gate PASSES)."""
        metrics = {"accuracy": 1.0}
        self.assertTrue(check_accuracy(metrics, threshold=0.80))


class TestLoadMetrics(unittest.TestCase):
    """Tests for the load_metrics() function."""

    def _write_tmp(self, content: str) -> str:
        """Helper: write content to a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_missing_metrics_file(self):
        """load_metrics() raises FileNotFoundError for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            load_metrics("/nonexistent/path/metrics.json")

    def test_invalid_json_content(self):
        """load_metrics() raises ValueError when the file contains invalid JSON."""
        tmp_path = self._write_tmp("this is not json {{{")
        try:
            with self.assertRaises(ValueError):
                load_metrics(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_valid_metrics_loaded(self):
        """load_metrics() correctly returns a dict from a valid JSON file."""
        data = {"accuracy": 0.92, "dataset_size": 300}
        tmp_path = self._write_tmp(json.dumps(data))
        try:
            metrics = load_metrics(tmp_path)
            self.assertIsInstance(metrics, dict)
            self.assertAlmostEqual(metrics["accuracy"], 0.92)
        finally:
            os.unlink(tmp_path)

    def test_missing_accuracy_key_raises(self):
        """check_accuracy() raises ValueError when 'accuracy' key is absent."""
        metrics = {"dataset_size": 300}
        with self.assertRaises(ValueError):
            check_accuracy(metrics)

    def test_non_numeric_accuracy_raises(self):
        """check_accuracy() raises ValueError when accuracy is not a number."""
        metrics = {"accuracy": "high"}
        with self.assertRaises(ValueError):
            check_accuracy(metrics)


class TestRunQualityGate(unittest.TestCase):
    """Integration tests for run_quality_gate() using temporary metrics files."""

    def _write_metrics(self, accuracy: float) -> str:
        """Helper: write a valid metrics.json to a temp file."""
        data = {"accuracy": accuracy, "dataset_size": 300}
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, tmp)
        tmp.close()
        return tmp.name

    def test_gate_passes_when_accuracy_above_threshold(self):
        """run_quality_gate() must NOT raise SystemExit when accuracy >= threshold."""
        tmp_path = self._write_metrics(0.92)
        try:
            # Should complete without calling sys.exit(1)
            run_quality_gate(metrics_path=tmp_path, threshold=0.80)
        except SystemExit as exc:
            self.fail(
                f"Quality gate raised SystemExit({exc.code}) "
                "for accuracy 0.92 >= threshold 0.80"
            )
        finally:
            os.unlink(tmp_path)

    def test_gate_fails_when_accuracy_below_threshold(self):
        """run_quality_gate() must raise SystemExit(1) when accuracy < threshold."""
        tmp_path = self._write_metrics(0.60)
        try:
            with self.assertRaises(SystemExit) as ctx:
                run_quality_gate(metrics_path=tmp_path, threshold=0.80)
            self.assertEqual(
                ctx.exception.code,
                1,
                "Expected SystemExit code 1 when accuracy 0.60 < threshold 0.80",
            )
        finally:
            os.unlink(tmp_path)

    def test_gate_fails_when_metrics_file_missing(self):
        """run_quality_gate() must raise SystemExit(1) when metrics.json is absent."""
        with self.assertRaises(SystemExit) as ctx:
            run_quality_gate(metrics_path="/nonexistent/metrics.json", threshold=0.80)
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
