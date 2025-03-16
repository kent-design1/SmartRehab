import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# --------------------------
# Load Synthetic Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_dataset.csv")

# Use "Patient Progress" as target; drop non-informative columns.
features = df.drop(columns=["Patient ID", "Patient Progress"])
target = df["Patient Progress"]

# Convert target labels (strings) to numeric values
le = LabelEncoder()
target_numeric = le.fit_transform(target)
# Now, le.classes_ holds the order, e.g., ['Deteriorating', 'Improving', 'Stable']

# One-hot encode categorical features.
features = pd.get_dummies(features)

# Split dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(features, target_numeric, test_size=0.2, random_state=42)

# --------------------------
# Define Repeated K-Fold for robust CV
# --------------------------
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# --------------------------
# Expanded Hyperparameter Tuning for Extra Trees
# --------------------------
etc = ExtraTreesClassifier(random_state=42)
etc_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
etc_grid = GridSearchCV(etc, etc_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
etc_grid.fit(X_train, y_train)
print("Best Extra Trees Parameters:")
print(etc_grid.best_params_)
print("Extra Trees CV Accuracy: {:.4f}".format(etc_grid.best_score_))

# --------------------------
# Expanded Hyperparameter Tuning for Random Forest
# --------------------------
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)
print("RF CV Accuracy: {:.4f}".format(rf_grid.best_score_))

# --------------------------
# Expanded Hyperparameter Tuning for MLP (Deep Learning Model)
# --------------------------
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50,50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
mlp_grid = GridSearchCV(mlp, mlp_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
mlp_grid.fit(X_train, y_train)
print("\nBest MLP Parameters:")
print(mlp_grid.best_params_)
print("MLP CV Accuracy: {:.4f}".format(mlp_grid.best_score_))

# --------------------------
# Expanded Hyperparameter Tuning for XGBoost
# --------------------------
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print("\nBest XGBoost Parameters:")
print(xgb_grid.best_params_)
print("XGBoost CV Accuracy: {:.4f}".format(xgb_grid.best_score_))

# --------------------------
# Create a Voting Ensemble to blend predictions (weighted voting)
# --------------------------
# Define base models using the best estimators from tuning.
estimators = [
    ('rf', rf_grid.best_estimator_),
    ('etc', etc_grid.best_estimator_),
    ('mlp', mlp_grid.best_estimator_),
    ('xgb', xgb_grid.best_estimator_)
]

# VotingClassifier with soft voting averages the predicted probabilities.
voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
voting_clf.fit(X_train, y_train)

# Evaluate the Voting Ensemble on the test set.
y_pred_voting = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print("\nVoting Ensemble Test Accuracy: {:.4f}".format(voting_accuracy))
print("\nClassification Report (Voting Ensemble):")
print(classification_report(y_test, y_pred_voting, target_names=le.classes_))
cm_voting = confusion_matrix(y_test, y_pred_voting)
print("\nConfusion Matrix (Voting Ensemble):")
print(cm_voting)

# --------------------------
# Visualizations
# --------------------------
# Confusion Matrix Heatmap for Voting Ensemble
plt.figure(figsize=(8, 6))
sns.heatmap(cm_voting, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Voting Ensemble")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Example: Plot the distribution of BMI values in the dataset
plt.figure(figsize=(8, 6))
sns.histplot(df["BMI"].dropna(), bins=30, kde=True)
plt.title("Distribution of BMI in Synthetic Dataset")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()