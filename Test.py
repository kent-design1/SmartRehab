import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import ast

# --------------------------
# Load Swiss Synthetic Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_600_swiss.csv")

# Parse Therapy Plan History from string to list
df["Therapy Plan History"] = df["Therapy Plan History"].apply(ast.literal_eval)

# --------------------------
# Feature Engineering
# --------------------------
# Trend slope from admission to discharge
df["Trend_Slope"] = (df["Total_SCIM_24"] - df["Total_SCIM_0"]) / 24.0

# Improvements over first and second half
df["Imp_0_12"] = (df["Total_SCIM_12"] - df["Total_SCIM_0"]) / 12.0
df["Imp_12_24"] = (df["Total_SCIM_24"] - df["Total_SCIM_12"]) / 12.0

# Count therapy plan changes
df["Num_Therapy_Changes"] = df["Therapy Plan History"].apply(lambda lst: max(len(lst) - 1, 0))

# --------------------------
# Prepare Data for Static Model
# --------------------------
# Target: discharge SCIM
y = df["Total_SCIM_24"]

# Features: drop identifiers, list‑columns, instruments, and target
drop_cols = [
    "Patient ID",
    "Therapy Plan History",
    "Instrument_Self-Care", "Instrument_Respiration",
    "Instrument_Mobility", "Instrument_Total_SCIM",
    "Total_SCIM_24"
]
X = df.drop(columns=drop_cols)

# One‑hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split to train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Static Model Tuning & Training
# --------------------------
# 1) Extra Trees Regressor
et = ExtraTreesRegressor(random_state=42)
et_params = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
et_search = RandomizedSearchCV(
    et, et_params, n_iter=10, cv=3,
    scoring="neg_mean_squared_error", random_state=42, n_jobs=-1
)
et_search.fit(X_train, y_train)
best_et = et_search.best_estimator_

# 2) Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}
gb_search = RandomizedSearchCV(
    gb, gb_params, n_iter=10, cv=3,
    scoring="neg_mean_squared_error", random_state=42, n_jobs=-1
)
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_

# 3) Voting Ensemble
voting = VotingRegressor([("et", best_et), ("gb", best_gb)])
voting.fit(X_train, y_train)

# --------------------------
# Evaluation on Test Set
# --------------------------
y_pred = voting.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²:   {r2:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.xlabel("Actual SCIM_24")
plt.ylabel("Predicted SCIM_24")
plt.title("Static Model: Actual vs Predicted")
plt.tight_layout()
plt.savefig("static_model_actual_vs_pred_swiss.png")


importances = pd.Series(best_et.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False).head(15))
