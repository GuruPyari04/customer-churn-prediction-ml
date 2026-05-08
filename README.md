# Customer Churn Prediction ML Project

## Project Overview
This is an end-to-end Machine Learning project that predicts whether a customer will leave a telecom company or not.

The project uses:
- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression

---

## Problem Statement
Customer churn is a major problem for telecom companies.  
This project helps predict customer churn based on customer details and service usage.

---

## Steps Performed

### 1. Data Cleaning
- Removed null values
- Converted TotalCharges column to numeric datatype

### 2. Data Preprocessing
- Converted categorical columns using `pd.get_dummies()`
- Converted target column (`Churn`) into numerical values

### 3. Feature & Target Separation
- X = input features
- y = target column

### 4. Train Test Split
- Training data = 80%
- Testing data = 20%

### 5. Model Building
Used Logistic Regression model from Scikit-learn.

### 6. Model Evaluation
Performed:
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Model Accuracy
Accuracy achieved: **78%**

---

## Learning Outcome
Through this project I learned:
- Data preprocessing
- Feature engineering
- Logistic Regression
- Model evaluation
- End-to-end ML workflow
- GitHub project deployment

---

## Author
GuruPyari04# customer-churn-prediction-ml
End-to-end Machine Learning project to predict customer churn using Logistic Regression, Pandas, and Scikit-learn.
