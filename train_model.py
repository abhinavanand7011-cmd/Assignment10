"""
train_model.py

Heart Disease Prediction - Model Training Script
==================================================
Covers Task 1 (Data Understanding & Preprocessing) and Task 2 (Model Development).

Run this script to reproduce model.pkl:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Task 1: Data Understanding and Preprocessing
# ============================================================

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())
print()

# 3. Identify numerical features and target variable
target_variable = "target"
numerical_features = [col for col in df.columns if col != target_variable]

print("Numerical features:", numerical_features)
print("Target variable:", target_variable)
print()

# 4. Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print()

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")
print()

# ============================================================
# Task 2: Model Development
# ============================================================

# Standardize features (helps Logistic Regression converge and perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate using Accuracy Score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model AND the scaler AND the feature column order together,
# since the Flask API needs all three to make consistent predictions on new input.
joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "feature_names": numerical_features,
        "accuracy": accuracy
    },
    "model.pkl"
)

print("\nModel, scaler, and feature metadata saved to model.pkl")
