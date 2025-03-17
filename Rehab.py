import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
# --------------------------
# Load Synthetic Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_dataset.csv")

# For prediction, we'll use "SCIM Total" (a continuous measure of patient independence) as our target.
# Drop non-informative columns (e.g., "Patient ID").
features = df.drop(columns=["Patient ID", "SCIM Total"])
target = df["SCIM Total"]

# One-hot encode categorical features (e.g., Gender, Prior Health Conditions, Health Condition, Therapy Plan, Therapy Adjustment)
features = pd.get_dummies(features)

# Split data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# --------------------------
# Train a Random Forest Regressor
# --------------------------
reg_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
reg_model.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = reg_model.predict(X_test)

# Evaluate the model.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nPrediction Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# -------------------------
# Visualization: Actual vs. Predicted SCIM Total
# -------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual SCIM Total")
plt.ylabel("Predicted SCIM Total")
plt.title("Actual vs Predicted Final SCIM Scores")
plt.show()

# Bar plot of top feature importances.
importances = reg_model.feature_importances_
feat_imp_df = pd.DataFrame({"Feature": X_train.columns, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(10))
plt.title("Top 10 Feature Importances for SCIM Prediction")
plt.show()

# --------------------------
# Therapy Recommendation Function
# --------------------------
def recommend_therapy(new_patient_data, model=reg_model):
    """
    Given a new patient's data, predict the final SCIM Total and recommend a therapy adjustment.

    Parameters:
        new_patient_data (list or array): A list of feature values in the same order as used for training.
        model: The trained regression model.

    Returns:
        str: Recommendation for therapy adjustment.
    """
    # Convert input data into the correct shape for prediction.
    new_patient_array = np.array(new_patient_data).reshape(1, -1)

    # Note: new_patient_data should be preprocessed in the same way as training data.
    # For simplicity, here we assume it is already one-hot encoded and has the same columns as X_train.
    pred_scim = model.predict(new_patient_array)[0]

    # Define threshold-based recommendations:
    if pred_scim < 60:
        recommendation = "Increase Frequency"
    elif pred_scim < 70:
        recommendation = "Change Therapy"
    else:
        recommendation = "Maintain Plan"

    return f"Predicted SCIM Total: {pred_scim:.1f} → Suggested Therapy Adjustment: {recommendation}"

# --------------------------
# Example: Perform Prediction for a New Patient
# --------------------------
# For a new patient, you need to supply the same features as used in training.
# Here is an example with a made-up feature vector.
# IMPORTANT: The order of features must match X_train columns.
# In practice, you would prepare the new patient data similarly (including one-hot encoding).

# For demonstration, we'll create a new patient DataFrame from scratch.
new_patient_dict = {
    "Age": 50,
    "Gender": "Male",
    "Prior Health Conditions": "None",
    "Health Condition": "Spinal Cord Injury",
    "Therapy Plan": "SCI Physiotherapy",
    "Therapy Adjustment": "Maintain Plan",
    "Sessions per Week": 3,
    "Duration of Treatment (weeks)": 30,
    "Patient Progress": "Improving",
    "Follow-up Status": "Completed",
    "Height (cm)": 170,
    "Weight (kg)": 75.0,
    "BMI": 25.9,
    "Injury Severity Score": 6,
    "Time Since Injury (months)": 12,
    "Session Duration (minutes)": 60,
    "Exercise Intensity": 5,
    "Patient Adherence Rate": 0.85,
    "Mobility Score": 80,
    "Grip Strength (kg)": 30,
    "Muscle Strength Score": 6,
    "Pain Level": 4,
    "Balance Test Score": 75,
    "Quality of Life Score": 80,
    "Mental Health Status": 4,
    "Sleep Quality Score": 7,
    "Fatigue Level": 3,
    "Daily Activity Score": 80,
    "Patient Engagement Score": 90
}
new_patient_df = pd.DataFrame([new_patient_dict])
# One-hot encode the new patient data to match training set
new_patient_processed = pd.get_dummies(new_patient_df)

# Ensure the new patient data has the same columns as X_train.
for col in X_train.columns:
    if col not in new_patient_processed.columns:
        new_patient_processed[col] = 0
new_patient_processed = new_patient_processed[X_train.columns]

# Get Therapy Recommendation for the new patient
therapy_suggestion = recommend_therapy(new_patient_processed.values[0])
print("\nTherapy Recommendation for New Patient:")
print(therapy_suggestion)















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
