#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train script for customer churn prediction models.
Creates two logistic regression models with different random states.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Generate synthetic data for customer churn prediction
# Binary classification: 1 = churn, 0 = no churn
# Using 5 features as specified
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42,
    shuffle=True
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model_v1 with random_state=42
print("Training model_v1 with random_state=42...")
model_v1 = LogisticRegression(random_state=42, max_iter=1000)
model_v1.fit(X_train, y_train)

# Evaluate model_v1
accuracy_v1 = model_v1.score(X_test, y_test)
print(f"Model v1 accuracy: {accuracy_v1:.4f}")

# Save model_v1
joblib.dump(model_v1, '..\\models\\model_v1.joblib')
print("Model v1 saved as ..\\models\\model_v1.joblib")

# Generate new data with different random state for model_v2
X_v2, y_v2 = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=99,
    shuffle=True
)

# Split data for model_v2
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
    X_v2, y_v2, test_size=0.2, random_state=99
)

# Train model_v2 with random_state=99
print("Training model_v2 with random_state=99...")
model_v2 = LogisticRegression(random_state=99, max_iter=1000)
model_v2.fit(X_train_v2, y_train_v2)

# Evaluate model_v2
accuracy_v2 = model_v2.score(X_test_v2, y_test_v2)
print(f"Model v2 accuracy: {accuracy_v2:.4f}")

# Save model_v2
joblib.dump(model_v2, '..\\models\\model_v2.joblib')
print("Model v2 saved as ..\\models\\model_v2.joblib")

print("Training complete. Both models have been saved.")