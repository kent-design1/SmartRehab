# backend/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

app = FastAPI(
    title="Rehab SCIM Prediction API",
    version="1.0",
    description="Predict Total_SCIM at weeks 6,12,18,24—send only prior data, never the target."
)

# 1) Load the four voting ensembles
MODELS: Dict[int, Pipeline] = {}
for wk in (6, 12, 18, 24):
    with open(f"model_week{wk}.pkl", "rb") as f:
        MODELS[wk] = pickle.load(f)["voting"]

def _prepare_df(wk: int, features: Dict[str, Any]) -> pd.DataFrame:
    """
    Given week and a dict of feature→value (no target), returns a one-row
    DataFrame with exactly the columns the pipeline expects.
    """
    pipe = MODELS[wk]
    preproc = pipe.named_steps["preproc"]

    num_cols = preproc.transformers_[0][2]
    cat_cols = preproc.transformers_[1][2]

    data = features.copy()
    # fill missing numerics with NaN
    for c in num_cols:
        data.setdefault(c, np.nan)
    # fill missing categoricals with a placeholder
    for c in cat_cols:
        data.setdefault(c, "missing")

    # ordering matters
    cols = list(num_cols) + list(cat_cols)
    return pd.DataFrame([data], columns=cols)

class FeaturePayload(BaseModel):
    features: Dict[str, Any]

@app.post("/predict/week6")
async def predict_week6(payload: FeaturePayload):
    # forbid sending the target itself
    if "Total_SCIM_6" in payload.features:
        raise HTTPException(400, "Do not include Total_SCIM_6; that’s what you’re predicting.")
    try:
        df = _prepare_df(6, payload.features)
        pred = MODELS[6].predict(df)[0]
        return {"week": 6, "prediction": float(pred)}
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

@app.post("/predict/week12")
async def predict_week12(payload: FeaturePayload):
    if "Total_SCIM_12" in payload.features:
        raise HTTPException(400, "Do not include Total_SCIM_12; that’s what you’re predicting.")
    try:
        df = _prepare_df(12, payload.features)
        pred = MODELS[12].predict(df)[0]
        return {"week": 12, "prediction": float(pred)}
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

@app.post("/predict/week18")
async def predict_week18(payload: FeaturePayload):
    if "Total_SCIM_18" in payload.features:
        raise HTTPException(400, "Do not include Total_SCIM_18; that’s what you’re predicting.")
    try:
        df = _prepare_df(18, payload.features)
        pred = MODELS[18].predict(df)[0]
        return {"week": 18, "prediction": float(pred)}
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

@app.post("/predict/week24")
async def predict_week24(payload: FeaturePayload):
    if "Total_SCIM_24" in payload.features:
        raise HTTPException(400, "Do not include Total_SCIM_24; that’s what you’re predicting.")
    try:
        df = _prepare_df(24, payload.features)
        pred = MODELS[24].predict(df)[0]
        return {"week": 24, "prediction": float(pred)}
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")