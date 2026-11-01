# MLOps Practicals Repository

This repository hosts practical implementations for MLOps CI pipelines using GitHub Actions.

---

## Practical 1: Implementing the First CI Pipeline Using GitHub Actions

### Project Overview
Demonstrates a minimal Continuous Integration (CI) pipeline for a rule-based Python application.

### Practical 1 Files
- **`result_logic.py`**: Business logic function `predict_result(internal_marks, attendance)`.
- **`test_result_logic.py`**: Unit tests covering pass, marks failure, and attendance failure cases.
- **`.github/workflows/ci.yml`**: GitHub Actions workflow running `unittest discover -v`.

---

## Practical 2: Student Result Prediction ML CI

### Project Overview
Demonstrates an end-to-end Machine Learning CI pipeline. It generates a reproducible synthetic dataset, trains a scikit-learn classification model, exports model artifacts and evaluation metrics, and verifies model behavior via automated unit tests.

### Practical 2 Files
- **`requirements.txt`**: Project dependencies (`numpy`, `pandas`, `scikit-learn`, `joblib`).
- **`train_model.py`**: Generates 300 synthetic student records, splits into 240 train / 60 test, trains a `LogisticRegression` classifier, saves `student_results.csv`, `student_result_model.pkl`, and `metrics.json`.
- **`test_ml_pipeline.py`**: Unit test suite verifying:
  - `test_dataset_created`: Checks presence and validity of 300-row CSV.
  - `test_model_created`: Verifies model artifact loads with joblib.
  - `test_metrics_created`: Verifies metrics JSON creation and structure.
  - `test_accuracy_is_valid`: Confirms accuracy is a valid score between 0.0 and 1.0.
  - `test_model_prediction`: Checks model inference output shape and classes.
  - `test_high_performance_student`: Confirms high-performance inputs predict PASS (1).
  - `test_low_performance_student`: Confirms low-performance inputs predict FAIL (0).
- **`.github/workflows/ml-ci.yml`**: GitHub Actions workflow named **"Student Result ML CI"** that installs dependencies, executes `train_model.py`, and runs the full test suite.

---

## Project Structure

```text
first-ci-demo/
├── README.md
├── requirements.txt
├── result_logic.py
├── test_result_logic.py
├── train_model.py
├── test_ml_pipeline.py
├── student_results.csv
├── student_result_model.pkl
├── metrics.json
└── .github/
    └── workflows/
        ├── ci.yml
        └── ml-ci.yml
```

---

## Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Practical 1: Rule-Based Logic
```bash
python result_logic.py
```

### 3. Practical 2: Train Model and Generate Artifacts
```bash
python train_model.py
```

### 4. Run All Unit Tests
```bash
python -m unittest discover -v
```
