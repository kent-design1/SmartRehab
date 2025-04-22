import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # using RandomizedSearchCV for tuning
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------
# Load Enhanced Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_dataset.csv")

# Convert weekly scores from string to list (if stored as a string)
df["Weekly SCIM Scores"] = df["Weekly SCIM Scores"].apply(ast.literal_eval)

# For columns that are arrays, convert them to a string so that one-hot encoding can handle them.
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

def compute_plateau(scim_list, threshold=1.0, consecutive_weeks=3):
    """
    Computes a plateau measure based on weekly SCIM scores.
    Returns the maximum consecutive weeks where improvement is less than the threshold.
    """
    if len(scim_list) < 2:
        return 0
    differences = [scim_list[i] - scim_list[i-1] for i in range(1, len(scim_list))]
    max_plateau = 0
    current_plateau = 0
    for diff in differences:
        if abs(diff) < threshold:  # check if change is minor
            current_plateau += 1
            max_plateau = max(max_plateau, current_plateau)
        else:
            current_plateau = 0
    return max_plateau if max_plateau >= consecutive_weeks else 0

# Compute average improvement and slope
df[['Avg_Improvement', 'Slope']] = df.apply(lambda row: pd.Series(compute_weekly_trends(row["Weekly SCIM Scores"])), axis=1)

# Compute the variance of improvements
df['Improvement_Var'] = df["Weekly SCIM Scores"].apply(compute_improvement_variance)

# Compute Plateau
df['Plateau_Length'] = df["Weekly SCIM Scores"].apply(lambda x: compute_plateau(x, threshold=1.0, consecutive_weeks=3))


# --------------------------
# Prepare Data for the Static Model
# --------------------------
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
# Hyperparameter Tuning with RandomizedSearchCV
# --------------------------
# Extra Trees Regressor
et = ExtraTreesRegressor(random_state=42)
et_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
et_rand_search = RandomizedSearchCV(et, et_param_grid, cv=3, scoring='neg_mean_squared_error',
                                    n_iter=20, random_state=42, n_jobs=-1)  # CHANGE: using RandomizedSearchCV with n_iter=20 and cv=3
et_rand_search.fit(X_train_static, y_train_static)
best_et = et_rand_search.best_estimator_
print("\nBest Extra Trees Parameters:")
print(et_rand_search.best_params_)
print("Extra Trees CV RMSE: {:.2f}".format(np.sqrt(-et_rand_search.best_score_)))

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'min_samples_split': [2, 5]
}
gb_rand_search = RandomizedSearchCV(gb, gb_param_grid, cv=3, scoring='neg_mean_squared_error',
                                    n_iter=20, random_state=42, n_jobs=-1)
gb_rand_search.fit(X_train_static, y_train_static)
best_gb = gb_rand_search.best_estimator_
print("\nBest Gradient Boosting Parameters:")
print(gb_rand_search.best_params_)
print("Gradient Boosting CV RMSE: {:.2f}".format(np.sqrt(-gb_rand_search.best_score_)))

# --------------------------
# Create a Voting Regressor Ensemble
# --------------------------
voting_reg = VotingRegressor(estimators=[
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
plt.tight_layout()
plt.savefig("voting_ensemble.png")  # CHANGE: Save plot instead of plt.show()
print("\nA plot of Actual vs Predicted SCIM Final Scores has been saved as 'voting_ensemble.png'.")



# --------------------------
# Time Series Model (LSTM): Predict Final SCIM from Early Weekly Scores
# --------------------------
def extract_sequence_features(row, input_length=18, target_week=24):
    seq = row["Weekly SCIM Scores"]
    if len(seq) >= target_week + 1:
        return seq[:input_length], seq[target_week]
    else:
        return None, None

# Create lists for inputs and targets
inputs = []
targets = []
for _, row in df.iterrows():
    inp, tar = extract_sequence_features(row, input_length=18, target_week=24)
    if inp is not None:
        inputs.append(inp)
        targets.append(tar)

inputs = np.array(inputs)
targets = np.array(targets)

# Normalize the data (assuming scores are in the range 0-100)
inputs_norm = inputs / 100.0
targets_norm = targets / 100.0

# Reshape inputs to match LSTM expectations: (samples, timesteps, features)
X_ts = inputs_norm.reshape(-1, 18, 1)
y_ts = targets_norm

# Split the dataset into training and testing sets
X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)

# --------------------------
# Build an Improved LSTM Model
# --------------------------
lstm_model = Sequential()
# First LSTM layer with 100 units; using 'tanh' for cell activation and 'sigmoid' for recurrent activation.
lstm_model.add(LSTM(100, activation='tanh', recurrent_activation='sigmoid',
                    return_sequences=True, input_shape=(18, 1)))
lstm_model.add(Dropout(0.2))
# Second (stacked) LSTM layer with 50 units
lstm_model.add(LSTM(50, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
lstm_model.add(Dropout(0.2))
# Output layer to produce the final score prediction
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

# --------------------------
# Train the LSTM Model with Early Stopping
# --------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(X_train_ts, y_train_ts,
                         epochs=50,           # Increase epochs to 100 for deeper training
                         batch_size=32,
                         validation_split=0.2,
                         callbacks=[early_stop],
                         verbose=1)

# --------------------------
# Evaluate the LSTM Model
# --------------------------
y_pred_ts = lstm_model.predict(X_test_ts).flatten()
rmse_ts = np.sqrt(mean_squared_error(y_test_ts, y_pred_ts))
r2_ts = r2_score(y_test_ts, y_pred_ts)
print("\nTime Series Model Performance:")
print(f"RMSE: {rmse_ts:.3f}")
print(f"R^2: {r2_ts:.3f}")

# --------------------------
# (Optional) Plotting Actual vs. Predicted Scores
# --------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test_ts * 100, y_pred_ts * 100, alpha=0.6, color='blue')
plt.plot([min(y_test_ts * 100), max(y_test_ts * 100)],
         [min(y_test_ts * 100), max(y_test_ts * 100)], 'r--', linewidth=2)
plt.xlabel("Actual SCIM Final Score")
plt.ylabel("Predicted SCIM Final Score")
plt.title("LSTM: Actual vs Predicted SCIM Final Scores")
plt.tight_layout()
plt.show()

# --------------------------
# Hybrid Prediction Function for 4-Week Forecast and Therapy Recommendation
# --------------------------
def predict_week8_and_recommend(new_patient_baseline, new_patient_weekly_data=None, alpha=0.5):
    """
    Predict the SCIM score at week 8 (or any target) and recommend therapy adjustments.

    If weekly SCIM scores for the required number of weeks (here 18) are provided,
    the prediction is a blend of the static model prediction (best_et) and the LSTM model prediction.
    Otherwise, it uses the static model's prediction.

    Parameters:
        new_patient_baseline (array-like): Baseline features (the first element is Baseline SCIM,
                                           the second is Total Therapy Cost).
        new_patient_weekly_data (list, optional): List of weekly SCIM scores for weeks 0 to 17.
        alpha (float): Blending weight (0: only static; 1: only LSTM).

    Returns:
        dict: Contains the predicted SCIM score, therapy recommendation, total cost, and cost efficiency.
    """
    # Static model prediction:
    pred_static = best_et.predict(new_patient_baseline.reshape(1, -1))[0]

    # Use LSTM prediction if at least 18 weeks are provided:
    if new_patient_weekly_data is None or len(new_patient_weekly_data) < 18:
        week8_scim = pred_static
    else:
        # Use the first 18 weeks
        new_patient_weekly = np.array(new_patient_weekly_data[:18]) / 100.0  # normalize
        new_patient_weekly = new_patient_weekly.reshape(1, 18, 1)
        pred_lstm = lstm_model.predict(new_patient_weekly)[0][0] * 100  # scale back
        week8_scim = alpha * pred_lstm + (1 - alpha) * pred_static

    # Extract baseline SCIM and total therapy cost
    baseline_scim = new_patient_baseline[0]
    total_cost = new_patient_baseline[1]

    # Calculate improvement and determine a therapy recommendation:
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

