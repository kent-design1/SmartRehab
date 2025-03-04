# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import shap
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
file_path = "Data/Large_Synthetic_Rehabilitation_Data"  # Ensure the correct file path
df = pd.read_csv(file_path)

# Define feature name mappings
feature_name_mapping = {
    "Patient_ID": "patient_id",
    "Age": "age",
    "Gender": "gender",
    "Height_cm": "height_cm",
    "Weight_kg": "weight_kg",
    "BMI": "bmi",
    "Spinal_Injury_Type": "spinal_injury_type",
    "Injury_Cause": "injury_cause",
    "Injury_Severity_Score": "injury_severity_score",
    "Time_Since_Injury_months": "time_since_injury_months",
    "Previous_Surgeries": "previous_surgeries",
    "Rehabilitation_Type": "rehabilitation_type",
    "Session_Frequency_per_week": "session_frequency_per_week",
    "Session_Duration_minutes": "session_duration_minutes",
    "Exercise_Intensity": "exercise_intensity",
    "Use_of_Assistive_Devices": "use_of_assistive_devices",
    "Patient_Adherence_Rate_%": "patient_adherence_rate",
    "Mobility_Score": "mobility_score",
    "Grip_Strength_kg": "grip_strength_kg",
    "Muscle_Strength_Score": "muscle_strength_score",
    "Pain_Level": "pain_level",
    "Spasticity_Level": "spasticity_level",
    "Balance_Test_Score": "balance_test_score",
    "Quality_of_Life_Score": "quality_of_life_score",
    "Mental_Health_Status": "mental_health_status",
    "Sleep_Quality_Score": "sleep_quality_score",
    "Fatigue_Level": "fatigue_level",
    "Daily_Activity_Score": "daily_activity_score",
    "Gait_Speed_mps": "gait_speed_mps",
    "Step_Count_per_Day": "step_count_per_day",
    "Heart_Rate_Variability": "heart_rate_variability",
    "Joint_Range_of_Motion_degrees": "joint_range_of_motion_degrees",
    "Predicted_Recovery_Score": "predicted_recovery_score",
    "Recommended_Therapy_Adjustment": "recommended_therapy_adjustment",
    "Anomalies_Detected_by_AI": "anomalies_detected_by_ai",
    "Patient_Engagement_Score": "patient_engagement_score"
}

# Apply feature name mapping to the dataset
df_large = df_large.rename(columns=feature_name_mapping)






















# Step 1: Handle Missing Values
df = df.dropna(subset=['SCIM_admission'])  # Drop rows with missing SCIM_admission

# Step 2: Convert Categorical Variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Convert 'sex' to numerical
df['etiology'] = df['etiology'].map({'accident': 0, 'sickness': 1})  # Convert 'etiology'

# Step 3: Check for Duplicate Records
duplicates = df.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")  # Should be 0

# Step 4: Data Summary
print("\nSummary Statistics:\n", df.describe())

# Step 5: Visualizing Data Distributions
plt.figure(figsize=(12, 5))

# Age Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')

# SCIM Admission Score Distribution
plt.subplot(1, 2, 2)
sns.histplot(df['SCIM_admission'], bins=20, kde=True, color='green')
plt.title('SCIM Admission Score Distribution')

plt.show()

# Step 6: Correlation Heatmap
plt.figure(figsize=(15, 10))
corr_matrix = df.corr(numeric_only=True)  # Ensure numeric-only columns
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Scatterplot - Therapy Time vs Recovery Score
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['mean_physt'], y=df['SCIM_admission'], alpha=0.6)
plt.xlabel('Mean Physiotherapy Minutes')
plt.ylabel('SCIM Admission Score')
plt.title('Physiotherapy Time vs Recovery Score')
plt.show()


# Step 8: Feature Engineering
df['therapy_minutes_total'] = (df['mean_logo'] + df['mean_psyc'] + df['mean_imag'] + df['mean_lab'] +
                               df['mean_nurs'] + df['mean_anaest'] + df['mean_int_care'] + df['mean_ergo'] +
                               df['mean_physt'] + df['mean_non_med'])

df['therapy_efficiency'] = df['SCIM_admission'] / (df['therapy_minutes_total'] + 1)  # +1 to avoid division by zero

df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, np.inf], labels=['<30', '30-50', '50+'])

df['total_cost'] = (df['op_room'] + df['anaest'] + df['int_care'] + df['imag'] + df['lab'] +
                    df['physt'] + df['ergo'] + df['logo'] + df['non_med_ter'] + df['nurs'] + df['psyc'])

df['cost_efficiency'] = df['SCIM_admission'] / (df['total_cost'] + 1)

df['non_med_ratio'] = df['mean_non_med'] / (df['therapy_minutes_total'] + 1)

df['high_therapy_patient'] = df['therapy_minutes_total'] > df['therapy_minutes_total'].median()

df['therapy_diversity'] = (df[['mean_logo', 'mean_psyc', 'mean_imag', 'mean_lab', 'mean_nurs',
                               'mean_anaest', 'mean_int_care', 'mean_ergo', 'mean_physt', 'mean_non_med']] > 0).sum(axis=1)




# Step 9: Feature Selection - Identify top correlated features
corr_matrix = df.corr(numeric_only=True)
top_corr_features = corr_matrix['SCIM_admission'].abs().sort_values(ascending=False).index[1:10]
X = df[top_corr_features]  # Selecting top 10 correlated features
y = df['SCIM_admission']   # Target variable

# # Rename columns using the mapping dictionary
# X = X.rename(columns=feature_name_mapping)


# Step 10: Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Train a Predictive Model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 12: Make Predictions
# 1 Random Forest
y_pred_rf_best = model.predict(X_test)

# 2 Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# 3 Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# # Step 13: Evaluate Model Performance
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# # Step 14: Display Performance Metrics
# model_performance = {
#     "Mean Absolute Error (MAE)": mae,
#     "Mean Squared Error (MSE)": mse,
#     "Root Mean Squared Error (RMSE)": rmse,
#     "R-Squared (R²)": r2
# }


# Function to evaluate model performance
def evaluate_model(y_test, y_pred, model_name):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    }

# Evaluate all models
model_performance_results = [
    evaluate_model(y_test, y_pred_linear, "Linear Regression"),
    evaluate_model(y_test, y_pred_gb, "Gradient Boosting"),
    evaluate_model(y_test, y_pred_rf_best, "Tuned Random Forest")
]

#Convert Data to float64 Format
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

#Verify That X_train is Numeric Before SHAP
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

# Create SHAP Explainer
explainer = shap.Explainer(gb_model, X_train)

# Generate SHAP values for test data
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)



# Create LIME Explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    mode="regression"
)

# Pick a random patient
random_index = np.random.randint(0, X_test.shape[0])
patient_data = X_test.iloc[random_index].values

# Generate LIME explanation for this patient
lime_exp = lime_explainer.explain_instance(patient_data, gb_model.predict, num_features=5)

import matplotlib.pyplot as plt

lime_exp.as_pyplot_figure()
plt.show()


from pprint import pprint
pprint(model_performance_results)

# Step 15: Feature Importance Analysis
feature_importances = pd.Series(model.feature_importances_, index=top_corr_features).sort_values(ascending=False)

# # Display Results
# print("\n✅ Model Performance Metrics:")
# for metric, value in model_performance.items():
#     print(f"{metric}: {value:.4f}")

print("\n🔥 Feature Importance:")
print(feature_importances)


def recommend_therapy(new_patient_data, model=gb_model):
    """
    Suggests the most important therapy type based on the model's prediction.

    Parameters:
    new_patient_data (list): A list of feature values in the same order as X_train columns.
    model (sklearn model): The trained Gradient Boosting model.

    Returns:
    str: Suggested therapy based on most important factor.
    """
    # Reshape input for prediction
    new_patient_data = np.array(new_patient_data).reshape(1, -1)

    # Get SHAP values for this new patient
    shap_explanation = explainer(new_patient_data)

    # Find the most important therapy factor
    most_important_feature = X_train.columns[np.argmax(np.abs(shap_explanation.values))]

    return f"Suggested therapy: Focus on {most_important_feature} for best recovery."

# New patient data with a low therapy_efficiency value.
# Order: therapy_efficiency, therapy_minutes_total, ergo, non_med_ter, mean_nurs, nurs, mean_los, total_cost, high_therapy_patient
new_patient = [0.05, 2000, 2, 5, 10, 50, 3, 10000, 0]

# Get Therapy Recommendation
therapy_suggestion = recommend_therapy(new_patient)
print(therapy_suggestion)


# Save the model
joblib.dump(gb_model, "SmartRehabModel.pkl")

# Save the feature names
joblib.dump(top_corr_features, "SmartRehabFeatures.pkl")

print("✅ Model and features saved successfully!")

# # Save cleaned dataset
# df.to_csv("cleaned_data_spz.csv", index=False)
#

# print("\n✅ Data Cleaning & Visualization Completed. Cleaned dataset saved as 'cleaned_data_spz.csv'.")