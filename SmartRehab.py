# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import shap
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # Step 1: Handle Missing Values
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
file_path = "Data/Synthetic_Rehabilitation_Data.csv"
df_large = pd.read_csv(file_path)

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

# Define numerical and categorical columns
numerical_cols = df_large.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_large.select_dtypes(include=['object']).columns.tolist()

# Impute missing numerical values with median
num_imputer = SimpleImputer(strategy='median')
df_large[numerical_cols] = num_imputer.fit_transform(df_large[numerical_cols])

# Impute missing categorical values with most frequent category
cat_imputer = SimpleImputer(strategy='most_frequent')
df_large[categorical_cols] = cat_imputer.fit_transform(df_large[categorical_cols])

# Display the first few rows of the cleaned dataset
print(df_large.head())

# Save the dataset to a CSV file for review
df_large.to_csv("Mapped_and_Cleaned_Dataset.csv", index=False)










