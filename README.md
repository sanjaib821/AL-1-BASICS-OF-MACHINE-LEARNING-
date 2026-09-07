# Student Performance Prediction using Machine Learning

## Project overview
This beginner-friendly machine learning project predicts whether a student is likely to **Pass or Fail** using:

- Study hours
- Attendance
- Previous marks
- Assignment score

The project compares four classification algorithms and saves the best-performing model.

## Algorithms used
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)

`train_test_split` is used to create separate training and testing data, and the best model is selected using test-set accuracy. Scikit-learn documents `train_test_split` as a utility for splitting data into random training and testing subsets.

## Project structure

```text
student-performance-ml/
├── data/
│   └── student_data.csv
├── models/
│   └── student_performance_model.joblib
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python src/train.py
```

### 3. Make a prediction

```bash
python src/predict.py
```

## Dataset note
The included CSV is a **synthetic educational dataset created for this project**. It is intended for learning and demonstration, not for making real academic decisions.

## Model-building description for the college form

> A classification-based machine learning model is developed to predict student performance from study hours, attendance, previous marks and assignment scores. The dataset is divided into training and testing sets. Logistic Regression, Decision Tree, Random Forest and KNN algorithms are trained and compared using classification accuracy. The best-performing model is saved and used for prediction.

## Algorithms used for the college form

> Logistic Regression, Decision Tree, Random Forest Classifier, K-Nearest Neighbors (KNN)

## Technologies

- Python
- Pandas
- Scikit-learn
- Joblib
