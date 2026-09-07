import joblib
import pandas as pd

MODEL_PATH = "models/student_performance_model.joblib"

model = joblib.load(MODEL_PATH)

# Change these values to test another student.
student = pd.DataFrame([{
    "study_hours": 5.0,
    "attendance": 85,
    "previous_marks": 72,
    "assignment_score": 80
}])

prediction = model.predict(student)[0]
print("Prediction:", "PASS" if prediction == 1 else "FAIL")
