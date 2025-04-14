from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt  # (optional; used only for debugging/plots)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

###############################################
# Global Model Training & Loading (Executed Once)
###############################################

# --------------------------
# Load and Preprocess Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_dataset.csv")
df["Weekly SCIM Scores"] = df["Weekly SCIM Scores"].apply(ast.literal_eval)

# Convert array-type columns to a string for one-hot encoding.
for col in ["Therapy Adjustments", "Adjustment Weeks"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x))

# Fill missing values if needed.
if "Therapy Adjustments" in df.columns:
    df["Therapy Adjustments"] = df["Therapy Adjustments"].fillna('None')
if "Adjustment Weeks" in df.columns:
    df["Adjustment Weeks"] = df["Adjustment Weeks"].fillna('0')

print("Missing values in key columns:")
print(df[['Therapy Adjustments', 'Adjustment Weeks']].isna().sum())

# --------------------------
# Feature Engineering Functions
# --------------------------
def compute_weekly_trends(scim_list):
    """
    Returns average weekly improvement and slope (week 0 to week 8) for a list of weekly SCIM scores.
    """
    if len(scim_list) < 9:
        return np.nan, np.nan
    improvements = [scim_list[i] - scim_list[i-1] for i in range(1, 9)]
    return np.mean(improvements), (scim_list[8] - scim_list[0]) / 8

def compute_improvement_variance(scim_list):
    if len(scim_list) < 9:
        return np.nan
    improvements = [scim_list[i] - scim_list[i-1] for i in range(1, 9)]
    return np.var(improvements)

def compute_plateau(scim_list, threshold=1.0, consecutive_weeks=3):
    """
    Returns the maximum number of consecutive weeks in which the change is less than the threshold.
    """
    if len(scim_list) < 2:
        return 0
    differences = [scim_list[i] - scim_list[i-1] for i in range(1, len(scim_list))]
    max_plateau = 0
    current_plateau = 0
    for diff in differences:
        if abs(diff) < threshold:
            current_plateau += 1
            max_plateau = max(max_plateau, current_plateau)
        else:
            current_plateau = 0
    return max_plateau if max_plateau >= consecutive_weeks else 0

# Compute engineered features
df[['Avg_Improvement', 'Slope']] = df.apply(lambda row: pd.Series(compute_weekly_trends(row["Weekly SCIM Scores"])), axis=1)
df['Improvement_Var'] = df["Weekly SCIM Scores"].apply(compute_improvement_variance)
df['Plateau_Length'] = df["Weekly SCIM Scores"].apply(lambda x: compute_plateau(x, threshold=1.0, consecutive_weeks=3))

# --------------------------
# Prepare Data for the Static Model
# --------------------------
cols_to_drop = ["Patient ID", "Weekly SCIM Scores", "Weekly Self-Care", "Weekly Respiration", "Weekly Mobility"]
baseline_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
target_static = baseline_df["SCIM Final"]
features_static = baseline_df.drop(columns=["SCIM Final"])
features_static = pd.get_dummies(features_static)

X_train_static, X_test_static, y_train_static, y_test_static = train_test_split(
    features_static, target_static, test_size=0.2, random_state=42
)

# --------------------------
# Train Static Models (Voting Ensemble)
# --------------------------
# Extra Trees Regressor
et = ExtraTreesRegressor(random_state=42)
et_param_grid = {'n_estimators': [100, 300],
                 'max_depth': [None, 10, 15],
                 'min_samples_split': [2, 5],
                 'min_samples_leaf': [1, 2]}
et_rand_search = RandomizedSearchCV(et, et_param_grid, cv=3, scoring='neg_mean_squared_error',
                                    n_iter=20, random_state=42, n_jobs=-1)
et_rand_search.fit(X_train_static, y_train_static)
best_et = et_rand_search.best_estimator_

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb_param_grid = {'n_estimators': [100, 200],
                 'max_depth': [3, 5, 7],
                 'learning_rate': [0.05, 0.1],
                 'min_samples_split': [2, 5]}
gb_rand_search = RandomizedSearchCV(gb, gb_param_grid, cv=3, scoring='neg_mean_squared_error',
                                    n_iter=20, random_state=42, n_jobs=-1)
gb_rand_search.fit(X_train_static, y_train_static)
best_gb = gb_rand_search.best_estimator_

# Voting Regressor Ensemble using the two models
voting_reg = VotingRegressor(estimators=[('et', best_et), ('gb', best_gb)])
voting_reg.fit(X_train_static, y_train_static)
print("\nVoting Ensemble Test Performance:")
y_pred_voting = voting_reg.predict(X_test_static)
rmse_voting = np.sqrt(mean_squared_error(y_test_static, y_pred_voting))
r2_voting = r2_score(y_test_static, y_pred_voting)
print(f"RMSE: {rmse_voting:.2f} | R^2: {r2_voting:.2f}")

# --------------------------
# Train the LSTM Time-Series Model (Using 18 Weeks of Data)
# --------------------------
def extract_sequence_features(row, input_length=18, target_week=24):
    seq = row["Weekly SCIM Scores"]
    if len(seq) >= target_week + 1:
        return seq[:input_length], seq[target_week]
    else:
        return None, None

lstm_inputs, lstm_targets = [], []
for _, row in df.iterrows():
    inp, tar = extract_sequence_features(row, input_length=18, target_week=24)
    if inp is not None:
        lstm_inputs.append(inp)
        lstm_targets.append(tar)
lstm_inputs = np.array(lstm_inputs)
lstm_targets = np.array(lstm_targets)

inputs_norm = lstm_inputs / 100.0
targets_norm = lstm_targets / 100.0

X_ts = inputs_norm.reshape(-1, 18, 1)
y_ts = targets_norm

X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)

lstm_model = Sequential()
lstm_model.add(LSTM(100, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=(18, 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lstm_model.fit(X_train_ts, y_train_ts, epochs=50, batch_size=32, validation_split=0.2,
               callbacks=[early_stop], verbose=1)

y_pred_ts = lstm_model.predict(X_test_ts).flatten()
rmse_ts = np.sqrt(mean_squared_error(y_test_ts, y_pred_ts))
r2_ts = r2_score(y_test_ts, y_pred_ts)
print("\nLSTM Time Series Model Performance:")
print(f"RMSE: {rmse_ts:.3f} | R^2: {r2_ts:.3f}")

# --------------------------
# Define Hybrid Prediction Function
# --------------------------
def predict_week8_and_recommend(new_patient_baseline, new_patient_weekly_data=None, alpha=0.5):
    """
    Predict the SCIM score at week 8 (or a designated target) by blending the static model and LSTM prediction.

    Parameters:
        new_patient_baseline (array-like): Baseline features (first element: Baseline SCIM, second: Total Therapy Cost).
        new_patient_weekly_data (list, optional): List of weekly SCIM scores for weeks 0 to 17.
        alpha (float): Weight for blending (0: only static; 1: only LSTM).

    Returns:
        dict: Contains predicted SCIM score, therapy recommendation, predicted total cost, and cost efficiency.
    """
    # Static model prediction
    pred_static = best_et.predict(new_patient_baseline.reshape(1, -1))[0]

    # LSTM prediction if enough weekly data is provided (18 weeks required)
    if new_patient_weekly_data is None or len(new_patient_weekly_data) < 18:
        week8_scim = pred_static
    else:
        new_patient_weekly = np.array(new_patient_weekly_data[:18]) / 100.0
        new_patient_weekly = new_patient_weekly.reshape(1, 18, 1)
        pred_lstm = lstm_model.predict(new_patient_weekly)[0][0] * 100
        week8_scim = alpha * pred_lstm + (1 - alpha) * pred_static

    baseline_scim = new_patient_baseline[0]
    total_cost = new_patient_baseline[1]

    improvement = week8_scim - baseline_scim
    if improvement < 5:
        therapy_rec = "Change Therapy"
    elif improvement < 10:
        therapy_rec = "Increase Frequency"
    else:
        therapy_rec = "Maintain Plan"

    cost_efficiency = week8_scim / (total_cost + 1)

    return {
        "Predicted Week8 SCIM": week8_scim,
        "Therapy Recommendation": therapy_rec,
        "Predicted Total Cost": total_cost,
        "Cost Efficiency": cost_efficiency
    }

print("\nGlobal models have been trained and are loaded into memory.")

###############################################
# Flask API Section
###############################################

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Root route for information
@app.route("/", methods=["GET"])
def index():
    return "Flask API for Smart Rehab is running. Use /api/calculate for predictions."

# API Endpoint for Hybrid Prediction
@app.route("/api/calculate", methods=["POST"])
def calculate():
    """
    Expects a JSON payload:
    {
      "baseline": [<Baseline SCIM>, <Total Therapy Cost>, ... other static features if needed],
      "weekly": [list of weekly SCIM scores for weeks 0 to 17],
      "alpha": <blending weight, optional>
    }
    Returns a JSON with the predicted Week8 SCIM, therapy recommendation, total cost and cost efficiency.
    """
    data = request.json
    baseline = np.array(data.get("baseline"))
    weekly = data.get("weekly", None)
    alpha = data.get("alpha", 0.5)

    result = predict_week8_and_recommend(baseline, new_patient_weekly_data=weekly, alpha=alpha)
    return jsonify(result)

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)