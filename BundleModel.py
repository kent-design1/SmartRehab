import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
    cross_val_score
)
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --------------------------
# 1) Load & Parse Realistic Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_600_swiss_realistic.csv")
df["Therapy Plan History"] = df["Therapy Plan History"].apply(ast.literal_eval)

# --------------------------
# 2) Handle Missingness & Dropouts
# --------------------------
df = df.dropna(subset=["Total_SCIM_24"])  # drop target‐missing

# --------------------------
# 3) Feature Engineering
# --------------------------
# target
y = df["Total_SCIM_24"]

# slopes and improvements
df["Trend_Slope"] = (df["Total_SCIM_24"] - df["Total_SCIM_0"]) / 24.0
df["Imp_0_12"]  = (df["Total_SCIM_12"] - df["Total_SCIM_0"]) / 12.0
df["Imp_12_24"] = (df["Total_SCIM_24"] - df["Total_SCIM_12"]) / 12.0
df["Imp_0_6"]   = (df["Total_SCIM_6"] - df["Total_SCIM_0"]) / 6.0

# plateau length
def plateau_length(vals, thr=1.0):
    diffs = np.abs(np.diff(vals))
    current = max_plateau = 0
    for d in diffs:
        if d < thr:
            current += 1
            max_plateau = max(max_plateau, current)
        else:
            current = 0
    return max_plateau

scim_cols = ["Total_SCIM_0","Total_SCIM_6","Total_SCIM_12","Total_SCIM_18","Total_SCIM_24"]
df["Plateau"] = df[scim_cols].apply(
    lambda row: plateau_length(row.dropna().values),
    axis=1
)

# therapy changes count
df["Num_Therapy_Changes"] = df["Therapy Plan History"] \
    .apply(lambda L: max(len([p for p in L if p]) - 1, 0))

# --------------------------
# 4) Prepare Features
# --------------------------
# drop unused
drop_cols = [
    "Patient ID", "Therapy Plan History",
    "Instrument_Self-Care","Instrument_Respiration",
    "Instrument_Mobility","Instrument_Total_SCIM",
    "Total_SCIM_24"
]
X = df.drop(columns=drop_cols)

# split off numeric vs. categorical for ColumnTransformer
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols     = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", OneHotEncoder(drop='first', handle_unknown="ignore"), cat_cols)
])

# --------------------------
# 5) Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 6) Tune & Train Base Models
# --------------------------
et_search = RandomizedSearchCV(
    ExtraTreesRegressor(random_state=42),
    {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    n_iter=10, cv=3, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1
)
et_search.fit(preprocessor.fit_transform(X_train), y_train)
best_et = et_search.best_estimator_

gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    },
    n_iter=10, cv=3, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1
)
gb_search.fit(preprocessor.transform(X_train), y_train)
best_gb = gb_search.best_estimator_

# --------------------------
# 7) Full Pipeline + Voting
# --------------------------
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("voting", VotingRegressor([
        ("et", best_et),
        ("gb", best_gb)
    ]))
])
pipeline.fit(X_train, y_train)

# eval
y_pred = pipeline.predict(X_test)
print("Full Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Full Model Test R²:  ", r2_score(y_test, y_pred))

cv_err = -cross_val_score(
    pipeline, X, y,
    cv=KFold(5, shuffle=True, random_state=42),
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
print("5-Fold CV RMSE:", np.sqrt(cv_err).mean())

# --------------------------
# 8) Diagnostics Plots
# --------------------------
resid = y_test - y_pred
plt.figure()
plt.hist(resid, bins=30, edgecolor="k")
plt.title("Residuals Distribution")
plt.savefig("resid_hist.png")

plt.figure()
plt.scatter(y_pred, resid, alpha=0.5)
plt.axhline(0, ls="--")
plt.title("Residuals vs Predicted")
plt.savefig("resid_vs_pred.png")

importances = pd.Series(
    best_et.feature_importances_,
    index=numeric_cols + pipeline.named_steps["preproc"]
    .named_transformers_["cat"]
    .get_feature_names_out(cat_cols).tolist()
)
top3 = importances.nlargest(3).index.tolist()

fig, ax = plt.subplots(figsize=(6,4))
PartialDependenceDisplay.from_estimator(
    pipeline, X_train, features=top3, ax=ax
)
plt.tight_layout()
plt.savefig("partial_dep.png")

# --------------------------
# 9) Serialize Pipeline
# --------------------------
with open("ml_api/model_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("✅ Saved model_pipeline.pkl")