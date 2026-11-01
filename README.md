# MLOps Practical 1: Implementing the First CI Pipeline Using GitHub Actions

## Project Overview

This repository demonstrates a minimal Continuous Integration (CI) pipeline using GitHub Actions for a Python application. It includes a simple rule-based result prediction module and automated unit tests.

## Project Structure

```text
first-ci-demo/
├── README.md
├── result_logic.py
├── test_result_logic.py
└── .github/
    └── workflows/
        └── ci.yml
```

### Files Description

- **`result_logic.py`**: Contains the core business logic `predict_result(internal_marks, attendance)`.
  - Returns `"PASS"` if `internal_marks >= 40` and `attendance >= 75`.
  - Otherwise returns `"FAIL"`.
- **`test_result_logic.py`**: Built-in `unittest` test suite covering:
  - `test_pass_case`: Verifies passing conditions (`70`, `85`).
  - `test_fail_due_to_marks`: Verifies failure due to low marks (`30`, `85`).
  - `test_fail_due_to_attendance`: Verifies failure due to low attendance (`70`, `60`).
- **`.github/workflows/ci.yml`**: GitHub Actions workflow that executes the test suite automatically.

---

## How the CI Pipeline Works

The workflow **"First Python CI"** runs automated checks whenever changes are made:

1. **Triggers (`on`)**:
   - Pushes to the `main` branch.
   - Pull Requests targeting the `main` branch.
   - Manual trigger (`workflow_dispatch`) from the GitHub Actions tab.

2. **Runner Environment**:
   - Runs on a GitHub-hosted Ubuntu runner (`ubuntu-latest`).

3. **Steps**:
   - **Check out repository**: Uses `actions/checkout@v4` to download the code to the runner.
   - **Set up Python**: Uses `actions/setup-python@v5` to configure Python 3.11.
   - **Run unit tests**: Executes `python -m unittest discover -v` to automatically discover and run all tests.

If all tests pass, the CI pipeline status turns green (success). If any test fails, the job fails, preventing faulty code from passing unnoticed.

---

## Running Locally

### 1. Run the Main Script
```bash
python result_logic.py
```

Output:
```text
Predicted Result: PASS
```

### 2. Run Unit Tests
```bash
python -m unittest discover -v
```
