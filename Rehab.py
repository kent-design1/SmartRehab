import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------
# Load Enhanced Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_dataset.csv")
# Convert weekly scores from string to list (if stored as a string)
df["Weekly SCIM Scores"] = df["Weekly SCIM Scores"].apply(ast.literal_eval)

# For columns that are arrays, convert them to a string so that one-hot encoding can handle them.
for col in ["Therapy Adjustment", "Adjustment Week"]:
    df[col] = df[col].apply(lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x))

# (Optional) Fill missing values if needed.
df['Therapy Adjustment'] = df['Therapy Adjustment'].fillna('None')
df['Adjustment Week'] = df['Adjustment Week'].fillna('0')

print("Missing values in key columns:")
print(df[['Therapy Adjustment', 'Adjustment Week']].isna().sum())
print(df.isna().sum())

# --------------------------
# Feature Engineering: Weekly Progress Trends
# --------------------------
def compute_weekly_trends(scim_list):
    """
    Compute summary statistics from weekly SCIM scores.
    Returns average weekly improvement and slope (change from week 0 to week 8 divided by 8).
    """
    if len(scim_list) < 9:
        return np.nan, np.nan
    improvements = [scim_list[i] - scim_list[i-1] for i in range(1, 9)]
    avg_improvement = np.mean(improvements)
    slope = (scim_list[8] - scim_list[0]) / 8
    return avg_improvement, slope

df[['Avg_Improvement', 'Slope']] = df.apply(lambda row: pd.Series(compute_weekly_trends(row["Weekly SCIM Scores"])), axis=1)

def compute_improvement_variance(scim_list):
    if len(scim_list) < 9:
        return np.nan
    improvements = [scim_list[i] - scim_list[i-1] for i in range(1, 9)]
    return np.var(improvements)

df['Improvement_Var'] = df["Weekly SCIM Scores"].apply(compute_improvement_variance)

# --------------------------
# Prepare Data for the Static Model
# --------------------------
# Use "SCIM Final" as target; drop non-informative columns and the weekly scores lists.
cols_to_drop = ["Patient ID", "Weekly SCIM Scores", "Weekly Self-Care", "Weekly Respiration", "Weekly Mobility"]
baseline_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
target_static = baseline_df["SCIM Final"]
features_static = baseline_df.drop(columns=["SCIM Final"])

# One-hot encode all categorical variables.
features_static = pd.get_dummies(features_static)

# --------------------------
# Split Data into Training and Testing Sets
# --------------------------
X_train_static, X_test_static, y_train_static, y_test_static = train_test_split(
    features_static, target_static, test_size=0.2, random_state=42
)

# --------------------------
# Hyperparameter Tuning with GridSearchCV
# --------------------------

# # 1. Random Forest Regressor
# rf = RandomForestRegressor(random_state=42)
# rf_param_grid = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [None, 10, 15, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# rf_grid.fit(X_train_static, y_train_static)
# best_rf = rf_grid.best_estimator_
# print("Best Random Forest Parameters:")
# print(rf_grid.best_params_)
# print("Random Forest CV RMSE: {:.2f}".format(np.sqrt(-rf_grid.best_score_)))

# 2. Extra Trees Regressor
et = ExtraTreesRegressor(random_state=42)
et_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
et_grid = GridSearchCV(et, et_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
et_grid.fit(X_train_static, y_train_static)
best_et = et_grid.best_estimator_
print("\nBest Extra Trees Parameters:")
print(et_grid.best_params_)
print("Extra Trees CV RMSE: {:.2f}".format(np.sqrt(-et_grid.best_score_)))

# 3. Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_samples_split': [2, 5, 10]
}
gb_grid = GridSearchCV(gb, gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gb_grid.fit(X_train_static, y_train_static)
best_gb = gb_grid.best_estimator_
print("\nBest Gradient Boosting Parameters:")
print(gb_grid.best_params_)
print("Gradient Boosting CV RMSE: {:.2f}".format(np.sqrt(-gb_grid.best_score_)))

# --------------------------
# Create a Voting Regressor Ensemble
# --------------------------
voting_reg = VotingRegressor(estimators=[
    # ('rf', best_rf),
    ('et', best_et),
    ('gb', best_gb)
])
voting_reg.fit(X_train_static, y_train_static)

# --------------------------
# Evaluate Ensemble Model on Test Set
# --------------------------
y_pred_voting = voting_reg.predict(X_test_static)
rmse_voting = np.sqrt(mean_squared_error(y_test_static, y_pred_voting))
r2_voting = r2_score(y_test_static, y_pred_voting)
print("\nVoting Ensemble Test Performance:")
print(f"RMSE: {rmse_voting:.2f}")
print(f"R^2: {r2_voting:.2f}")

# --------------------------
# Visualization: Actual vs. Predicted SCIM Final Scores
# --------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test_static, y_pred_voting, alpha=0.6, color='blue')
plt.plot([y_test_static.min(), y_test_static.max()], [y_test_static.min(), y_test_static.max()], 'r--', linewidth=2)
plt.xlabel("Actual SCIM Final")
plt.ylabel("Predicted SCIM Final")
plt.title("Voting Ensemble: Actual vs Predicted Final SCIM Scores")
plt.show()

# # --------------------------
# # Feature Importance Visualization (using Random Forest as an example)
# # --------------------------
# importances = best_rf.feature_importances_
# feat_imp_df = pd.DataFrame({"Feature": X_train_static.columns, "Importance": importances})
# feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(10))
# plt.title("Top 10 Feature Importances (Random Forest)")
# plt.show()














# --------------------------
# Time Series Model (LSTM): Predict Final SCIM from Early Weekly Scores
# --------------------------
# We use the first 9 weeks (week 0 to week 8) as input to predict the final SCIM (week 24).
def extract_sequence_features(row, input_length=9, target_week=24):
    seq = row["Weekly SCIM Scores"]
    if len(seq) >= target_week + 1:
        return seq[:input_length], seq[target_week]
    else:
        return None, None

inputs = []
targets = []
for _, row in df.iterrows():
    inp, tar = extract_sequence_features(row, input_length=9, target_week=24)
    if inp is not None:
        inputs.append(inp)
        targets.append(tar)

inputs = np.array(inputs)
targets = np.array(targets)

# Normalize SCIM scores (assuming range 0-100)
inputs_norm = inputs / 100.0
targets_norm = targets / 100.0

# Reshape for LSTM: (samples, timesteps, features)
X_ts = inputs_norm.reshape(-1, 9, 1)
y_ts = targets_norm

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(9, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

# Train LSTM model
history = lstm_model.fit(X_train_ts, y_train_ts, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate LSTM model
y_pred_ts = lstm_model.predict(X_test_ts).flatten()
rmse_ts = np.sqrt(mean_squared_error(y_test_ts, y_pred_ts))
r2_ts = r2_score(y_test_ts, y_pred_ts)
print("\nTime Series Model Performance:")
print(f"RMSE: {rmse_ts:.3f}")
print(f"R^2: {r2_ts:.3f}")


import numpy as np

# --- Define your therapy cost dictionary (should be already defined in your script) ---
therapy_costs = {
    "Physiotherapy": 100,
    "Occupational Therapy": 120,
    "Medication": 80,
    "Lifestyle Changes": 50,
    "Respiratory Management": 150,
    "Spasticity Management": 130,
    "Mobility & Upper Limb Training": 110,
    "Strength & FES Training": 140,
    "Comprehensive SCI Rehab": 200
}

# --- Hybrid Prediction Function for 4-Week Forecast and Therapy Recommendation ---
def predict_week8_and_recommend(new_patient_baseline, new_patient_weekly_4w=None, alpha=0.5):
    """
    Predict the SCIM score at week 8, recommend therapy adjustments, and estimate cost efficiency.

    Parameters:
        new_patient_baseline (array-like): Preprocessed baseline features (matching X_train_static).
                                          (Assumes the first element is Baseline SCIM and the second is Total Therapy Cost)
        new_patient_weekly_4w (list, optional): List of weekly SCIM scores for weeks 0-4.
        alpha (float): Blending weight (0 means only static model; 1 means only LSTM model).

    Returns:
        dict: Contains predicted Week8 SCIM score, recommended therapy plan, predicted total cost, and cost efficiency.
    """
    # Use best_et as the static model (make sure best_et is defined from your previous training)
    static_model = best_et
    pred_static = static_model.predict(new_patient_baseline.reshape(1, -1))[0]

    # Use LSTM model if weekly data for weeks 0-4 is provided (make sure lstm_model is defined)
    if new_patient_weekly_4w is not None and len(new_patient_weekly_4w) >= 5:
        new_patient_weekly_4w = np.array(new_patient_weekly_4w[:5]) / 100.0  # normalize assuming scores are 0-100
        new_patient_weekly_4w = new_patient_weekly_4w.reshape(1, 5, 1)
        pred_lstm = lstm_model.predict(new_patient_weekly_4w)[0][0] * 100  # rescale to 0-100
    else:
        pred_lstm = pred_static  # fallback if no weekly data

    # Blend the predictions using weight alpha
    week8_scim = alpha * pred_lstm + (1 - alpha) * pred_static

    # Extract baseline SCIM and Total Therapy Cost from the baseline features.
    baseline_scim = new_patient_baseline[0]
    total_cost = new_patient_baseline[1]

    # Calculate improvement and determine therapy recommendation.
    improvement = week8_scim - baseline_scim
    if improvement < 5:
        therapy_rec = "Change Therapy"
    elif improvement < 10:
        therapy_rec = "Increase Frequency"
    else:
        therapy_rec = "Maintain Plan"

    # Compute cost efficiency: (Week8 SCIM) divided by (Total Therapy Cost + 1)
    cost_efficiency = week8_scim / (total_cost + 1)

    return {
        "Predicted Week8 SCIM": week8_scim,
        "Therapy Recommendation": therapy_rec,
        "Predicted Total Cost": total_cost,
        "Cost Efficiency": cost_efficiency
    }

# --------------------------
# Accepting User Inputs
# --------------------------
# Prompt the user for baseline data.
print("Please enter the following baseline data:")
baseline_scim = float(input("Baseline SCIM score (0-100): "))
therapy_plan = input("Therapy Plan (choose from: Physiotherapy, Occupational Therapy, Medication, Lifestyle Changes, Respiratory Management, Spasticity Management, Mobility & Upper Limb Training, Strength & FES Training, Comprehensive SCI Rehab): ")
sessions_per_week = int(input("Number of sessions per week: "))
duration_weeks = int(input("Treatment duration in weeks: "))

# Calculate total therapy cost based on the selected therapy plan.
if therapy_plan in therapy_costs:
    per_session_cost = therapy_costs[therapy_plan]
else:
    per_session_cost = 100  # Default cost
total_cost = per_session_cost * sessions_per_week * duration_weeks

# Create baseline feature array.
# IMPORTANT: new_patient_baseline should have the same number of features as used in your static model.
# Here, we assume the first two features are Baseline SCIM and Total Therapy Cost.
# If your model expects more features, you may need to prompt for them or pad with default values.
num_features_required = X_train_static.shape[1]
user_baseline = [baseline_scim, total_cost]
while len(user_baseline) < num_features_required:
    user_baseline.append(0.0)  # Pad with zeros if necessary
user_baseline = np.array(user_baseline)

# Prompt the user for weekly SCIM scores for weeks 0-4.
weekly_input = input("Enter weekly SCIM scores for weeks 0-4 separated by commas (or leave blank if not available): ")
if weekly_input.strip():
    new_patient_weekly_4w = [float(x.strip()) for x in weekly_input.split(",")]
else:
    new_patient_weekly_4w = None

# --------------------------
# Perform Prediction
# --------------------------
predictions = predict_week8_and_recommend(user_baseline, new_patient_weekly_4w=new_patient_weekly_4w, alpha=0.5)
print("\nPredictions for New Patient:")
for key, value in predictions.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

#
# # --------------------------
# # Hybrid Prediction Function for 4-Week Forecast and Therapy Recommendation
# # --------------------------
# def predict_week8_and_recommend(new_patient_baseline, new_patient_weekly_4w=None, alpha=0.5):
#     """
#     Predict the SCIM score at week 8, recommend therapy adjustments, and estimate cost efficiency.
#
#     Parameters:
#         new_patient_baseline (array-like): Preprocessed baseline features (matching X_train_static).
#         new_patient_weekly_4w (list, optional): List of weekly SCIM scores for weeks 0-4.
#         alpha (float): Blending weight (0: only static model; 1: only LSTM model).
#
#     Returns:
#         dict: Contains predicted week8 SCIM score, recommended therapy plan, predicted cost, and cost efficiency.
#     """
#     # Predict SCIM at week 8 using the static model
#     static_model = best_et
#     pred_static = static_model.predict(new_patient_baseline.reshape(1, -1))[0]
#
#     # Use LSTM if weekly data for weeks 0-4 is provided
#     if new_patient_weekly_4w is not None and len(new_patient_weekly_4w) >= 5:
#         new_patient_weekly_4w = np.array(new_patient_weekly_4w[:5]) / 100.0  # normalize
#         new_patient_weekly_4w = new_patient_weekly_4w.reshape(1, 5, 1)
#         pred_lstm = lstm_model.predict(new_patient_weekly_4w)[0][0] * 100  # rescale
#     else:
#         pred_lstm = pred_static  # fallback
#
#     # Blend predictions
#     week8_scim = alpha * pred_lstm + (1 - alpha) * pred_static
#
#     # Assume the baseline includes a "Baseline SCIM" and "Total Therapy Cost" column
#     # Here, we extract them from new_patient_baseline using the appropriate column indices.
#     # (Adjust these indices as needed; here we assume "Baseline SCIM" is the first column and "Total Therapy Cost" is the second.)
#     baseline_scim = new_patient_baseline[0]
#     total_cost = new_patient_baseline[1]
#
#     # Determine improvement
#     improvement = week8_scim - baseline_scim
#     # Recommend therapy adjustment based on improvement thresholds
#     if improvement < 5:
#         therapy_rec = "Change Therapy"
#     elif improvement < 10:
#         therapy_rec = "Increase Frequency"
#     else:
#         therapy_rec = "Maintain Plan"
#
#     # Predict cost efficiency: defined as week8_scim / (Total Therapy Cost + 1)
#     cost_efficiency = week8_scim / (total_cost + 1)
#
#     return {
#         "Predicted Week8 SCIM": week8_scim,
#         "Therapy Recommendation": therapy_rec,
#         "Predicted Total Cost": total_cost,
#         "Cost Efficiency": cost_efficiency
#     }
#
# # --------------------------
# # Example Usage:
# # --------------------------
# # Assume new_patient_baseline is provided as an array with at least two features:
# # [Baseline SCIM, Total Therapy Cost, ...other baseline features...]
# # For demonstration, we take the first row from the static test set and assume that it includes these two columns at positions 0 and 1.
# example_baseline = X_test_static.iloc[0].values
# # And assume we have 4 weeks of SCIM scores for this patient (from the dataset)
# example_weekly_4w = df.iloc[0]["Weekly SCIM Scores"][:5]
#
# predictions = predict_week8_and_recommend(example_baseline, new_patient_weekly_4w=example_weekly_4w, alpha=0.5)
# print("\nPredictions for New Patient:")
# for key, value in predictions.items():
#     print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
#

# # --------------------------
# # Hybrid Prediction Functionl
# # --------------------------
# def hybrid_prediction(new_patient_baseline, new_patient_weekly=None, alpha=0.5):
#     """
#     Predict the final SCIM score for a new patient using both the static model and the LSTM model.
#
#     Parameters:
#         new_patient_baseline (array-like): Baseline features for the patient, already preprocessed (matching X_train_static).
#         new_patient_weekly (list, optional): List of weekly SCIM scores for weeks 0-8. If provided, will use LSTM.
#         alpha (float): Weight for blending predictions (0 means only static, 1 means only LSTM).
#
#     Returns:
#         float: Final predicted SCIM score.
#     """
#
#
#     static_model = best_et
#     pred_static = static_model.predict(new_patient_baseline.reshape(1, -1))[0]
#
#     if new_patient_weekly is not None and len(new_patient_weekly) >= 9:
#         new_patient_weekly = np.array(new_patient_weekly[:9]) / 100.0  # normalize
#         new_patient_weekly = new_patient_weekly.reshape(1, 9, 1)
#         pred_ts = lstm_model.predict(new_patient_weekly)[0][0] * 100  # rescale
#     else:
#         pred_ts = pred_static  # fallback
#
#     # Blend the predictions
#     final_pred = alpha * pred_ts + (1 - alpha) * pred_static
#     return final_pred
#
# # Example usage:
# # For a new patient with only baseline data (static features)
# # new_patient_baseline should be preprocessed like X_train_static; here we use the first row of X_test_static as an example.
# example_baseline = X_test_static.iloc[0].values
# # And if available, provide weekly SCIM scores (first 9 weeks). Here we use a sample from our dataset.
# example_weekly = df.iloc[0]["Weekly SCIM Scores"][:9]
#
# final_prediction = hybrid_prediction(example_baseline, new_patient_weekly=example_weekly, alpha=0.5)
# print("\nHybrid Prediction for New Patient (Final SCIM Score): {:.1f}".format(final_prediction))