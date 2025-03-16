import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# --------------------------
# Load Synthetic Dataset
# --------------------------
file_path = "synthetic_rehab_dataset.csv"
df = pd.read_csv(file_path)

# Display basic info
print("Dataset shape:", df.shape)
print(df.head())

# Use "Patient Progress" as target, drop non-informative columns (e.g., Patient ID)
features = df.drop(columns=["Patient ID", "Patient Progress"])
target = df["Patient Progress"]

# Convert categorical variables using one-hot encoding
features = pd.get_dummies(features)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# --------------------------
# Define and Train Multiple ML Models
# --------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc

    print(f"--- {model_name} ---")
    print("Accuracy: {:.4f}".format(acc))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

# Select best model based on accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print("Best model based on accuracy:", best_model_name, "with accuracy {:.4f}".format(results[best_model_name]))

# --------------------------
# Visualization for the Best Model
# --------------------------
# Confusion Matrix Heatmap
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Example: Visualize distribution of one numeric feature (e.g., BMI)
plt.figure(figsize=(8, 6))
sns.histplot(df["BMI"].dropna(), bins=30, kde=True)
plt.title("Distribution of BMI in Synthetic Dataset")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()




















# # # Encode categorical variables
# # categorical_cols = ["spinal_injury_type", "injury_cause", "rehabilitation_type", "use_of_assistive_devices", "recommended_therapy_adjustment"]
# # label_encoders = {}
# #
# # for col in categorical_cols:
# #     le = LabelEncoder()
# #     df[col] = le.fit_transform(df[col])
# #     label_encoders[col] = le
#
# # Define features and target
# X = df.drop(columns=["recommended_therapy_adjustment"], errors='ignore')
# y = df["recommended_therapy_adjustment"]
#
# # Train/Test Split with stratification
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#
# # Initialize Gradient Boosting and Extra Trees Classifier
# gb_model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=8, random_state=42)
# et_model = ExtraTreesClassifier(n_estimators=500, max_depth=12, n_jobs=-1, random_state=42)
#
# # Train both models
# gb_model.fit(X_train, y_train)
# et_model.fit(X_train, y_train)
#
# # Make predictions
# y_pred_gb = gb_model.predict(X_test)
# y_pred_et = et_model.predict(X_test)
#
# # Compute accuracy
# gb_accuracy = accuracy_score(y_test, y_pred_gb)
# et_accuracy = accuracy_score(y_test, y_pred_et)
#
# # Store results
# model_results = [
#     {"Model": "Gradient Boosting", "Accuracy": gb_accuracy},
#     {"Model": "Extra Trees Classifier", "Accuracy": et_accuracy}
# ]
#
# # Convert results to DataFrame and display
# model_results_df = pd.DataFrame(model_results)
# print(model_results_df)
