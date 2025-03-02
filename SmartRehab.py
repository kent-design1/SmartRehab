# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import shap
import lime.lime_tabular
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Step 2: Convert Categorical Values
from sklearn.impute import SimpleImputer # Step 1: Handle Missing Values
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# Drop Patient ID as it is not useful for ML
df_large = df_large.drop(columns=["patient_id", "anomalies_detected_by_ai"], errors='ignore')

# Define numerical and categorical columns
numerical_cols = df_large.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_large.select_dtypes(include=['object']).columns.tolist()

# Impute missing numerical values with median
num_imputer = SimpleImputer(strategy='median')
df_large[numerical_cols] = num_imputer.fit_transform(df_large[numerical_cols])

# Impute missing categorical values with most frequent category
cat_imputer = SimpleImputer(strategy='most_frequent')
df_large[categorical_cols] = cat_imputer.fit_transform(df_large[categorical_cols])

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_large[col] = le.fit_transform(df_large[col])
    label_encoders[col] = le  # Store encoder for inverse transformation if needed

# Step 3: Check for Duplicate Records
duplicate_records = df_large.duplicated().sum()
if duplicate_records > 0:
    df_large = df_large.drop_duplicates()
    print(f"Removed {duplicate_records} duplicate records.")
else:
    print("No duplicate records found.")


# # Step 4: Identify and Print Outliers
# import numpy as np
# outliers_dict = {}
#
# for col in numerical_cols:
#     Q1 = np.percentile(df_large[col], 25)
#     Q3 = np.percentile(df_large[col], 75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#
#     outliers = df_large[(df_large[col] < lower_bound) | (df_large[col] > upper_bound)]
#     outliers_dict[col] = len(outliers)
#
#     print(f"{col}: {len(outliers)} outliers detected")

# # Display the number of outliers per feature
# import pandas as pd
# outliers_df = pd.DataFrame(list(outliers_dict.items()), columns=['Feature', 'Number of Outliers'])



# Step 4: Data Summary
summary_stats = df_large.describe(include='all')
print("Dataset Summary:")
print(summary_stats)

# Plot histograms for numerical features
df_large[numerical_cols].hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_large[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 5: Feature Engineering

# Create new feature: BMI category
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal Weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df_large["bmi_category"] = df_large["bmi"].apply(categorize_bmi)

# Convert new categorical feature to numerical
bmi_le = LabelEncoder()
df_large["bmi_category"] = bmi_le.fit_transform(df_large["bmi_category"])

# Create new feature: Recovery Potential Score
df_large["recovery_potential"] = df_large["predicted_recovery_score"] * df_large["patient_engagement_score"]

# Normalize recovery potential
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_large["recovery_potential"] = scaler.fit_transform(df_large[["recovery_potential"]])


# Step 9: Feature Selection - Identify Top Correlated Features
correlation_matrix = df_large.corr()
top_features = correlation_matrix["recommended_therapy_adjustment"].abs().sort_values(ascending=False)
print("Top Correlated Features with Target Variable:")
print(top_features.head(10))

# Visualize Feature Correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Select top features for ML
selected_features = top_features.index[1:11].tolist()  # Exclude target variable
df_selected = df_large[selected_features + ["recommended_therapy_adjustment"]]

# # Ensure the dataset is ready for ML
# df_selected.to_csv("selected_features_dataset.csv", index=False)
# print("Selected features dataset saved as 'selected_features_dataset.csv'.")


# Feature Selection
# Define features and target variable (Using more features)
X = df_large.drop(columns=["recommended_therapy_adjustment"], errors='ignore')  # Keep all possible features
y = df_large["recommended_therapy_adjustment"]

# # Train a simple Random Forest to determine feature importance
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X, y)

# # Get feature importance scores
# feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
# feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# # Select top 10 most important features
# selected_features = feature_importance.head(10)['Feature'].tolist()
# X = X[selected_features]
#
# # Print selected features
# print("Selected Features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Hyperparameter Tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5, n_jobs=-1)
grid_gb.fit(X_train, y_train)

# Step 4: Train XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate models
models = {
    "Random Forest (Tuned)": grid_rf.best_estimator_,
    "Gradient Boosting (Tuned)": grid_gb.best_estimator_,
    "XGBoost": xgb_model
}

results = []

# models = {
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42)
# }

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": accuracy})

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)

# Corrected print statement
print("Model Performance Comparison:")
print(results_df)

print("Feature selection and model training completed.")





df_large.to_csv("processed_rehab_data.csv", index=False)






