# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
file_path = "Data/synthetic_data_spz.csv"  # Ensure the correct file path
df = pd.read_csv(file_path)

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

# Show LIME explanation
lime_exp.show_in_notebook()



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


# # Save cleaned dataset
# df.to_csv("cleaned_data_spz.csv", index=False)
#

# print("\n✅ Data Cleaning & Visualization Completed. Cleaned dataset saved as 'cleaned_data_spz.csv'.")