import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/student_data.csv"
MODEL_PATH = "models/student_performance_model.joblib"

df = pd.read_csv(DATA_PATH)

features = ["study_hours", "attendance", "previous_marks", "assignment_score"]
X = df[features]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ])
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = accuracy_score(y_test, predictions)
    print(f"{name}: {results[name]:.2%}")

best_name = max(results, key=results.get)
best_model = models[best_name]
joblib.dump(best_model, MODEL_PATH)

print(f"\nBest model: {best_name}")
print(f"Saved to: {MODEL_PATH}")
print("\nClassification report:")
print(classification_report(y_test, best_model.predict(X_test)))
