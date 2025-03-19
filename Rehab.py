import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------
# Step 1: Load the Enhanced Synthetic Dataset
# --------------------------
# Our dataset 'synthetic_rehab_dataset.csv' contains 5000 patients with various features
df = pd.read_csv("synthetic_rehab_dataset.csv")

# --------------------------
# Step 2: Define the Target and Features
# --------------------------
# We will predict "SCIM Final", which is our final measure of patient independence.
# Drop non-informative columns like "Patient ID".
features = df.drop(columns=["Patient ID", "SCIM Final"])
target = df["SCIM Final"]

# --------------------------
# Step 3: Preprocess the Data
# --------------------------
# Convert categorical variables to numerical values using one-hot encoding.
# This includes columns such as Gender, Prior Health Conditions, Health Condition,
# Therapy Plan, Therapy Adjustment, Patient Progress, and Follow-up Status.
features = pd.get_dummies(features)

# --------------------------
# Step 4: Split the Data into Training and Testing Sets
# --------------------------
# We use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# --------------------------
# Step 5: Train a Random Forest Regressor Model
# --------------------------
# RandomForestRegressor is robust for tabular data and can capture nonlinear relationships.
reg_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
reg_model.fit(X_train, y_train)

# --------------------------
# Step 6: Make Predictions on the Test Set and Evaluate the Model
# --------------------------
y_pred = reg_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nPrediction Model Performance:")
print(f"RMSE: {rmse:.2f}")   # Lower RMSE means better predictions
print(f"R^2 Score: {r2:.2f}")  # R^2 closer to 1 indicates an excellent fit

# --------------------------
# Step 7: Visualize the Model's Predictions
# --------------------------
# Scatter plot: Actual vs. Predicted SCIM Final scores
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual SCIM Final")
plt.ylabel("Predicted SCIM Final")
plt.title("Actual vs. Predicted Final SCIM Scores")
plt.show()

# Bar plot of top 10 feature importances
importances = reg_model.feature_importances_
feat_imp_df = pd.DataFrame({"Feature": X_train.columns, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(10))
plt.title("Top 10 Feature Importances for SCIM Prediction")
plt.show()

# --------------------------
# Step 8: Define a Therapy Recommendation Function
# --------------------------
def recommend_therapy(new_patient_data, model=reg_model):
    """
    Predict the final SCIM score for a new patient and recommend a therapy adjustment.
    
    Parameters:
        new_patient_data (list or array): Feature values in the same order as X_train columns.
        model: The trained regression model.
        
    Returns:
        str: A string with the predicted SCIM score and recommended therapy adjustment.
    """
    # Reshape the new patient data for prediction
    new_patient_data = np.array(new_patient_data).reshape(1, -1)
    pred_scim = model.predict(new_patient_data)[0]

    # Simple rule-based recommendations:
    # If predicted SCIM is below 60, it suggests the patient is still quite dependent → Increase Frequency.
    # If between 60 and 70, it suggests moderate progress → Change Therapy.
    # If 70 or above, it indicates good progress → Maintain Plan.
    if pred_scim < 60:
        recommendation = "Increase Frequency"
    elif pred_scim < 70:
        recommendation = "Change Therapy"
    else:
        recommendation = "Maintain Plan"

    return f"Predicted SCIM Final: {pred_scim:.1f} → Recommended Adjustment: {recommendation}"

# --------------------------
# Step 9: Perform a Prediction for a New Patient
# --------------------------
# For a new patient, the features must be prepared in the same order as in X_train.
# For demonstration, we'll take one sample from the test set.
example_new_patient = X_test.iloc[0].values

therapy_suggestion = recommend_therapy(example_new_patient)
print("\nTherapy Recommendation for New Patient:")
print(therapy_suggestion)