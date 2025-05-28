import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("synthetic_rehab_refined.csv")

# Define target and features
target_col = "TotalSCIM_12"
exclude = ['PatientID', 'TotalSCIM_0', 'TotalSCIM_6', 'TotalSCIM_18', 'TotalSCIM_24']
df = df.drop(columns=[col for col in exclude if col in df.columns], errors='ignore')
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# Identify column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Full pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor())
])

# Fit model
model_pipeline.fit(X, y)

# Get transformed feature names
X_transformed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

# SHAP explanation
explainer = shap.Explainer(model_pipeline.named_steps['model'], X_transformed)
shap_values = explainer(X_transformed[[0]])

# Format SHAP values
shap_series = pd.Series(shap_values.values[0], index=feature_names)
top_feats = shap_series.abs().sort_values(ascending=False).head(10)
colors = ['green' if shap_series[f] > 0 else 'red' for f in top_feats.index]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top_feats.index[::-1], shap_series[top_feats.index][::-1], color=colors[::-1])
plt.xlabel("SHAP Value")
plt.title("Top 10 SHAP Feature Impacts (Week 12, Patient Example)")
plt.tight_layout()
plt.savefig("shap_bar_patient_week12.png", dpi=300)
plt.show()







import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("synthetic_rehab_refined.csv")

# Calculate SCIM gain and efficiency
df['SCIM_Gain_24'] = df['TotalSCIM_24'] - df['TotalSCIM_0']
df['Efficiency'] = df['SCIM_Gain_24'] / df['TotalCost']

# Drop invalid values
df = df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['Efficiency'])

# Plot
plt.figure(figsize=(8, 5))
plt.hist(df['Efficiency'], bins=40, color='skyblue', edgecolor='black')
plt.axvline(df['Efficiency'].mean(), color='red', linestyle='--', label=f"Mean = {df['Efficiency'].mean():.4f}")
plt.xlabel("SCIM points per CHF (Efficiency)")
plt.ylabel("Number of Patients")
plt.title("Distribution of Efficiency Scores Across Synthetic Cohort")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency_distribution.png", dpi=300)
plt.show()