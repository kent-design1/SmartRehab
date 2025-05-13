
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------------
# 1) Load dataset
# --------------------------
df = pd.read_csv("synthetic_rehab_refined.csv")

# therapy plan list columns:
plan_cols = [f"TherapyPlans_{wk}" for wk in [0, 6, 12, 18, 24]]
# parse the Python-repr lists back to real lists
for col in plan_cols:
    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else [])

# multi-label binarizer over all plans
mlb = MultiLabelBinarizer()
all_plans = sorted({p for col in plan_cols for p in df[col].explode().dropna()})
mlb.fit([all_plans])

# create one-hot for each week×plan
for col in plan_cols:
    bin_df = pd.DataFrame(
        mlb.transform(df[col]),
        columns=[f"{col}_{p}" for p in mlb.classes_],
        index=df.index
    )
    df = pd.concat([df, bin_df], axis=1)

# drop the original list columns
df.drop(columns=plan_cols, inplace=True)

# --------------------------
# 1b) Cost-efficiency features
# --------------------------
# assumes 'BaselineCost','OveruseCost','TotalCost' are present
df['SCIM_Gain_24']    = df['TotalSCIM_24'] - df['TotalSCIM_0']
df['CostPerPoint']    = df['TotalCost'] / df['SCIM_Gain_24'].replace(0, np.nan)

# --------------------------
# 2) Settings
# --------------------------
MEAS_WEEKS     = [6, 12, 18, 24]
PREV_WEEKS     = {6:[0],12:[0,6],18:[0,6,12],24:[0,6,12,18]}
SCIM_COLS      = [f"TotalSCIM_{wk}" for wk in [0,6,12,18,24]]
COST_FEATS     = ['BaselineCost','OveruseCost','TotalCost','CostPerPoint']
PATIENT_ID     = 'PatientID'

# build predictor list (everything except SCIM & PatientID, then append cost feats)
all_feats = [c for c in df.columns if c not in SCIM_COLS + [PATIENT_ID]]
predictors = [c for c in all_feats if c not in COST_FEATS] + COST_FEATS

results = {}

for week in MEAS_WEEKS:
    print(f"\n=== Week {week} ===")
    target = f"TotalSCIM_{week}"
    sub = df.dropna(subset=[target]).copy()

    # feature matrix
    prior = [f"TotalSCIM_{w}" for w in PREV_WEEKS[week]]
    feat_cols = predictors + prior
    X = sub[feat_cols]
    y = sub[target]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # numeric vs categorical
    num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X_train.select_dtypes(include=['object']).columns.tolist()

    # preprocessing pipelines
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ], remainder='drop')

    # models + hyperparam spaces
    configs = {
        'et': (ExtraTreesRegressor(random_state=42), {
            'model__n_estimators': [100, 300],
            'model__max_depth':    [None, 10]
        }),
        'gb': (GradientBoostingRegressor(random_state=42), {
            'model__n_estimators':  [100, 200],
            'model__learning_rate': [0.05, 0.1]
        })
    }

    best = {}
    for name, (estimator, params) in configs.items():
        pipe = Pipeline([('pre', preprocessor), ('model', estimator)])
        search = RandomizedSearchCV(
            pipe, params, n_iter=5,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error',
            random_state=42, n_jobs=-1
        )
        print(f"Tuning {name}...")
        search.fit(X_train, y_train)
        best[name] = search.best_estimator_
        print(f" Best {name}: {search.best_params_}")

    # ensemble
    ensemble = Pipeline([
        ('pre', preprocessor),
        ('model', VotingRegressor([
            ('et', best['et'].named_steps['model']),
            ('gb', best['gb'].named_steps['model'])
        ]))
    ])
    ensemble.fit(X_train, y_train)

    # save model
    with open(f"model_week{week}.pkl", 'wb') as f:
        pickle.dump(ensemble, f)

    # evaluate
    y_pred = ensemble.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    results[week] = {'RMSE': rmse, 'MAE': mae}
    print(f"RMSE={rmse:.2f}, MAE={mae:.2f}")

    # scatter plot
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], ls='--', color='gray')
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title(f'Week {week} Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f"scatter_week{week}.png")
    plt.close()

    # SHAP explanations
    Xtr_p = preprocessor.transform(X_train)
    Xte_p = preprocessor.transform(X_test)
    masker = shap.maskers.Independent(Xtr_p, max_samples=100)
    expl   = shap.Explainer(
        ensemble.named_steps['model'].predict,
        masker,
        feature_names=preprocessor.get_feature_names_out()
    )
    sv = expl(Xte_p)

    # save SHAP objects
    with open(f"shap_explainer_week{week}.pkl", 'wb') as f:
        pickle.dump(expl, f)
    np.save(f"shap_values_week{week}.npy", sv.values)

    # SHAP summary plot
    plt.figure(figsize=(8,6))
    shap.summary_plot(sv, Xte_p, feature_names=preprocessor.get_feature_names_out(), show=False)
    plt.title(f'Week {week} SHAP Summary')
    plt.tight_layout()
    plt.savefig(f"shap_summary_week{week}.png")
    plt.close()

# final summary
print("\n=== Summary Metrics ===")
for wk, m in results.items():
    print(f"Week {wk}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}")