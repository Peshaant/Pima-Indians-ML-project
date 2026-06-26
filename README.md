# Diabetes Classification — ML Portfolio Project

A supervised machine learning project that predicts the onset of diabetes in patients
based on clinical diagnostic measurements.

Built using the **Pima Indians Diabetes Database** (UCI / Kaggle) as a portfolio project
to demonstrate end-to-end ML workflow skills.

---

## Problem Statement

Given 8 clinical features (glucose level, BMI, age, etc), predict whether a patient
will develop diabetes within 5 years. This is a **binary classification** problem.

---

## Dataset

- **Source:** [Pima Indians Diabetes Database — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Rows:** 768 patients (all female, age ≥ 21)
- **Features:** 8 clinical measurements
- **Target:** `Outcome` — 1 = Diabetic, 0 = Non-Diabetic

> Dataset not included in this repo. Download the CSV from the link above and place it
> in the root folder as `diabetes.csv` before running the notebook.

---

## Project Workflow

### 1. Exploratory Data Analysis
- Inspected shape, data types, and summary statistics
- Identified class imbalance (~65% non-diabetic, ~35% diabetic)
- Detected biologically impossible zero values in columns like Glucose and BMI

### 2. Data Preprocessing
- **Outlier capping** using the IQR method (Winsorizing) — preserves all 768 rows
- **Zero imputation** with column medians for impossible values
- **Train/test split** — 80/20 with stratification to preserve class ratio
- **Feature scaling** with StandardScaler (fit on train only to prevent data leakage)

### 3. Model Training & Comparison
Trained and evaluated four classifiers:

| Model | ROC-AUC |
|---|---|
| Logistic Regression | *run notebook to see* |
| Random Forest | *run notebook to see* |
| Gradient Boosting | *run notebook to see* |
| SVM | *run notebook to see* |

### 4. Hyperparameter Tuning
- Applied `GridSearchCV` with 5-fold cross-validation on Random Forest
- Optimised for **ROC-AUC** rather than accuracy due to class imbalance

### 5. Evaluation
- ROC Curve comparison across all models
- Confusion matrix with focus on **False Negatives** (missed diagnoses are the costlier error in healthcare)
- Feature importance analysis

---

## Key Findings

- **Glucose** was the strongest predictor of diabetes onset
- **BMI** and **Age** were the next most significant features
- Tuned Random Forest achieved the best ROC-AUC score

---



## 🚀 How to Run

1. Clone the repo
```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
```

2. Install dependencies
```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
   and place `diabetes.csv` in the root folder

4. Launch the notebook
```bash
   jupyter notebook
```

---

## Author

**Peshaant** — Computer Science Student, Monash University Malaysia  
NVIDIA Deep Learning Institute — Certified in Fundamentals of Deep Learning  
[GitHub](https://github.com/Peshaant)
