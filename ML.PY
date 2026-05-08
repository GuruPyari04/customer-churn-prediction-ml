import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# print("All libraries installed successfully")

#step 1 
import pandas as pd 
df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())
print(df.info())

#step 2 

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(df["TotalCharges"].dtype)

print(df.isnull().sum())

# step to remove null and na 
df = df.dropna()

print(df.isnull().sum())

print(df.shape)

#next step about map(),Means:Replace values using dictionary.

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print(df["Churn"].head())

# next step 

df = df.drop("customerID", axis=1)

print(df.head())

#next step This performs:ONE-HOT ENCODING,
# eaning:Convert categories into separate binary columns.

df = pd.get_dummies(df, drop_first=True)

print(df.head())

print(df.shape)

# step:# X contains all input features used for prediction.
# y contains the target/output column (Churn) that the model will learn to predict.

X = df.drop("Churn", axis=1)

y = df["Churn"]

print(X.head())

print(y.head())

#next step: # Split dataset into training and testing data so the model can learn on one part and be tested on unseen data.
# test_size=0.2 means 80% data is used for training and 20% for testing model performance.


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

# next step:# LogisticRegression creates a classification model used for predicting Yes/No type outputs.
# model.fit(X_train, y_train) trains the model by learning patterns from training data.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully")

# next step is the prediction step now :
# model.predict(X_test) uses the trained model to make predictions on unseen testing data.
# y_pred stores predicted outputs where 0 = stay and 1 = churn.

y_pred = model.predict(X_test)

print(y_pred[:5])

# step: # confusion_matrix() shows correct and incorrect predictions made by the model.
# It helps understand where the model is making mistakes in classification.

from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

#step to avoid confusion of the model and give accurate result 

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)


# step : # classification_report() shows Precision, Recall, and F1-score of the model.
# It helps evaluate how well the model performs for each class separately.

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))