import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('E:/kavucika/codsoft/IRIS.csv')

print("Columns in dataset:", data.columns)
print("")
print("Initial data sample:")
print(data.head())
print("")
print("Missing values in each column:")
print(data.isnull().sum())

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Features
y = data['species'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("")

def make_prediction():
    print("\nEnter the features of the Iris flower for prediction:")

    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    user_input_scaled = scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)
    print(f"\nPrediction type: {type(prediction)}")
    print(f"Prediction value: {prediction}")
    
    print(f"\nPredicted species: {prediction[0]}")

make_prediction()
