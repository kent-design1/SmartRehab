import pandas as pd
import numpy as np
import ast
import pickle

from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------------
# 1) Load & Parse Dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_600_swiss_realistic.csv")
df["Therapy Plan History"] = df["Therapy Plan History"].apply(ast.literal_eval)

# --------------------------
# 2) Define Feature Sets
# --------------------------
drop_cols = [
    "Patient ID", "Therapy Plan History",
    "Instrument_Self-Care", "Instrument_Respiration",
    "Instrument_Mobility", "Instrument_Total_SCIM"
]
all_features = [c for c in df.columns if c not in drop_cols]

def is_scim_column(col):
    prefixes = ["Total_SCIM_", "Self-Care_", "Respiration_", "Mobility_"]
    return any(col.startswith(p) for p in prefixes)

base_feats = [c for c in all_features if not is_scim_column(c)]

previous_scim = {
    6:  [0],
    12: [0, 6],
    18: [0, 6, 12],
    24: [0, 6, 12, 18]
}

target_weeks = [6, 12, 18, 24]
results = {}

# --------------------------
# 3) Per-Week Modeling
# --------------------------
for wk in target_weeks:
    target_col = f"Total_SCIM_{wk}"
    data = df.dropna(subset=[target_col]).copy()

    scim_feats = [f"Total_SCIM_{i}" for i in previous_scim[wk]]
    feat_list = base_feats + scim_feats

    X = data[feat_list]
    y = data[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    models = {
        'ExtraTrees': (
            ExtraTreesRegressor(random_state=42),
            {
                'model__n_estimators': [100, 300, 500],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }
        ),
        'GradientBoosting': (
            GradientBoostingRegressor(random_state=42),
            {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.05, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        )
    }

    best_models = {}
    for name, (estimator, params) in models.items():
        pipe = Pipeline([
            ('preproc', preprocessor),
            ('model', estimator)
        ])
        search = RandomizedSearchCV(
            pipe, params,
            n_iter=10,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        search.fit(X, y)
        best = search.best_estimator_

        rmse_cv = np.sqrt(-cross_val_score(best, X, y, cv=5, scoring='neg_mean_squared_error').mean())
        mae_cv = -cross_val_score(best, X, y, cv=5, scoring='neg_mean_absolute_error').mean()

        best_models[name] = {
            'estimator': best,
            'RMSE_CV': rmse_cv,
            'MAE_CV': mae_cv
        }

    voting = Pipeline([
        ('preproc', preprocessor),
        ('model', VotingRegressor([
            ('et', best_models['ExtraTrees']['estimator'].named_steps['model']),
            ('gb', best_models['GradientBoosting']['estimator'].named_steps['model'])
        ]))
    ])
    voting.fit(X, y)

    # Compute RMSE as sqrt of MSE
    rmse_v = np.sqrt(mean_squared_error(y, voting.predict(X)))
    mae_v = mean_absolute_error(y, voting.predict(X))

    with open(f"model_week{wk}.pkl", "wb") as f:
        pickle.dump({
            'et': best_models['ExtraTrees']['estimator'],
            'gb': best_models['GradientBoosting']['estimator'],
            'voting': voting
        }, f)

    results[wk] = {
        'ExtraTrees': (best_models['ExtraTrees']['RMSE_CV'], best_models['ExtraTrees']['MAE_CV']),
        'GradientBoosting': (best_models['GradientBoosting']['RMSE_CV'], best_models['GradientBoosting']['MAE_CV']),
        'Voting': (rmse_v, mae_v)
    }

# --------------------------
# 4) Summary of Results
# --------------------------
for wk, res in results.items():
    print(f"Week {wk} prediction performance:")
    for model_name, (rmse, mae) in res.items():
        print(f"  {model_name:15} RMSE={rmse:.2f}, MAE={mae:.2f}")
    print()


# --------------------------
# 5) Visualization: Voting Ensemble vs Actual
# --------------------------
import matplotlib.pyplot as plt

for wk in target_weeks:
    # Load the voting ensemble for this week
    with open(f"model_week{wk}.pkl", "rb") as f:
        models = pickle.load(f)
    voting = models['voting']

    # Prepare data
    data_wk = df.dropna(subset=[f"Total_SCIM_{wk}"])
    feat_list = base_feats + [f"Total_SCIM_{i}" for i in previous_scim[wk]]
    X_wk = data_wk[feat_list]
    y_wk = data_wk[f"Total_SCIM_{wk}"]

    # Predict and plot
    y_pred_wk = voting.predict(X_wk)
    plt.figure(figsize=(6,4))
    plt.scatter(y_wk, y_pred_wk, alpha=0.5)
    lims = [min(y_wk.min(), y_pred_wk.min()), max(y_wk.max(), y_pred_wk.max())]
    plt.plot(lims, lims, ls='--')
    plt.xlabel(f"Actual SCIM Week {wk}")
    plt.ylabel(f"Predicted SCIM Week {wk}")
    plt.title(f"Week {wk}: Actual vs Predicted (Voting Ensemble)")
    plt.tight_layout()
    plt.savefig(f"scatter_week{wk}.png")
    print(f"Saved scatter_week{wk}.png")

