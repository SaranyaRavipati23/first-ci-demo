def predict_result(internal_marks, attendance):
    if internal_marks >= 40 and attendance >= 75:
        return "PASS"
    return "FAIL"


if __name__ == "__main__":
    result = predict_result(70, 85)
    print(f"Predicted Result: {result}")
