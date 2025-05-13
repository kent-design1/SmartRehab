from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import Dict, Any, List
import pickle
import pandas as pd
import numpy as np
import shap
import random

app = FastAPI(
    title="Rehab SCIM Prediction API",
    version="1.7",
    description="Predict TotalSCIM at weeks 6,12,18,24 with detailed, layman‑friendly recommendations and cost‑efficiency breakdown."
)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
PREVIOUS_WEEKS         = {6:[0],12:[0,6],18:[0,6,12],24:[0,6,12,18]}
SESSION_BUMP_THRESHOLD = 5    # If gain < 5 points, suggest bumping sessions
TOP_K_SHAP             = 3    # Show top 3 SHAP drivers
MAX_THERAPIES          = 3    # Max extra therapy suggestions
DURATION_WEEKS         = 24   # For cost calculation

# Therapy costs (CHF/session)
therapy_costs = {
    "Respiratory Management": 500,
    "Spasticity Management": 450,
    "Mobility & Upper Limb Training": 400,
    "Strength & FES Training": 480,
    "Comprehensive SCI Rehab": 650,
    "Robotic Gait Training": 900,
    "Virtual Reality Therapy": 800,
    "Aquatic Therapy": 550,
    "Constraint-Induced Movement Therapy": 700,
    "Mirror Therapy": 350,
    "Physiotherapy": 380,
    "Occupational Therapy": 420,
    "Manual Therapy": 460,
    "Hydrotherapy": 500,
    "Proprioceptive Neuromuscular Facilitation": 550,
    "Functional Strength Training": 400,
    "Balance & Proprioception Drills": 320,
    "Medication": 200,
    "Lifestyle Changes": 150,
    "Cognitive Behavioral Therapy": 600,
    "Mindfulness-Based Stress Reduction": 520,
    "Pain Neuroscience Education": 480,
    "Yoga/Tai-Chi": 350,
    "Graded Activity/Exposure": 450,
    "Nutritional Counseling": 480,
    "Aerobic Exercise Program": 360,
    "Resistance Training": 400,
    "Foot Care Education": 300,
    "Glucose-Guided Activity": 330,
    "Structured Aerobic Training": 380,
    "Resistance Exercise": 420,
    "Stress Management": 460,
    "Dietary Sodium Reduction": 280,
    "Tele-rehab Monitoring": 600
}

# Insurance multipliers
insurance_factors = {
    "Basic Mandatory": 0.9,
    "Supplementary Private": 1.1,
    "Employer-Sponsored": 1.0,
    "Uninsured": 1.2
}

# Static rule thresholds & messages
THRESHOLDS = {
    'SessionsPerWeek': 3,
    'CharlsonIndex':   2,
    'TUG_sec':        14.0,
    'CRP_mg_L':        5.0,
    'Albumin_g_dL':    3.5,
    'PainLevel':       5
}
RECOMMENDATIONS = {
    'SessionsPerWeek': {
        'low': {
            'text': 'Raise your therapy frequency by 1–2 sessions per week',
            'rationale': (
                "Clinical trials have shown that moving from two to three sessions weekly "
                "can produce an extra 3–5 SCIM points over six weeks. Since you are below "
                "the three‑sessions threshold, adding one more session could meaningfully boost your recovery."
            )
        },
        'high': {
            'text': 'Maintain your current session schedule and adherence',
            'rationale': (
                "You are already meeting or exceeding the recommended minimum of three "
                "sessions per week. Consistency at this level supports ongoing gains, so "
                "continuing your routine helps lock in your progress."
            )
        }
    },
    'CharlsonIndex': {
        'low': {
            'text': 'Proceed with the standard rehabilitation plan',
            'rationale': (
                "A low comorbidity burden indicates you can tolerate the usual mix of exercises "
                "without special modifications, so the standard plan is appropriate."
            )
        },
        'high': {
            'text': 'Incorporate low‑impact modalities (e.g., Aquatic Therapy)',
            'rationale': (
                "Higher comorbidities often benefit from gentle, water‑based exercises "
                "to reduce joint stress while still promoting strength and mobility."
            )
        }
    },
    'TUG_sec': {
        'low': {
            'text': 'Continue your current balance and gait exercises',
            'rationale': (
                "Your TUG time is relatively fast, indicating good balance and mobility—"
                "maintaining current drills will help fine‑tune these skills."
            )
        },
        'high': {
            'text': 'Focus additional sessions on gait and balance training',
            'rationale': (
                "A slower TUG suggests deficits in balance or walking speed—"
                "targeted gait drills can improve functional mobility."
            )
        }
    },
    'CRP_mg_L': {
        'low': {
            'text': 'Monitor inflammation but proceed with strength training',
            'rationale': (
                "Normal CRP levels suggest inflammation is under control, so you can safely "
                "maintain or increase strength‑building activities."
            )
        },
        'high': {
            'text': 'Add anti‑inflammatory adjuncts (e.g., medical management)',
            'rationale': (
                "Elevated CRP is linked to slower recovery—incorporating anti‑inflammatory "
                "strategies may create a better environment for rehabilitation gains."
            )
        }
    },
    'Albumin_g_dL': {
        'low': {
            'text': 'Arrange a nutritional consultation',
            'rationale': (
                "Low albumin signals potential malnutrition, which impairs muscle repair. "
                "A dietitian can optimize your protein intake for better recovery."
            )
        },
        'high': {
            'text': 'Maintain your current nutritional regimen',
            'rationale': (
                "Healthy albumin levels indicate good nutritional status—continuing your diet "
                "supports ongoing healing and strength gains."
            )
        }
    },
    'PainLevel': {
        'low': {
            'text': 'Proceed with your standard exercise plan',
            'rationale': (
                "Low pain levels allow full participation in rehabilitation without additional "
                "pain interventions."
            )
        },
        'high': {
            'text': 'Integrate pain‑focused therapies (e.g., Pain CBT, Mindfulness)',
            'rationale': (
                "Higher pain can limit engagement—adding cognitive or mindfulness‑based pain "
                "management can improve your ability to participate in physical therapy."
            )
        }
    }
}

class FeaturePayload(BaseModel):
    features: Dict[str, Any]


# ─── UTILITIES ──────────────────────────────────────────────────────────────────
def _prepare_df(week: int, feats: Dict[str, Any]) -> pd.DataFrame:
    pipe = MODELS[week]
    pre = pipe.named_steps['pre']
    cols = list(pre.feature_names_in_)
    nums = set(pre.transformers_[0][2])
    row = {
        c: (feats.get(c, np.nan) if c in nums else feats.get(c, 'missing'))
        for c in cols
    }
    return pd.DataFrame([row], columns=cols)


def _get_static_recs(feats: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = []
    for feat, rules in RECOMMENDATIONS.items():
        if feat not in feats:
            continue
        level = 'high' if feats[feat] >= THRESHOLDS[feat] else 'low'
        entry = rules[level]
        recs.append({
            'feature': feat,
            'recommendation': entry['text'],
            'rationale': entry['rationale']
        })
    return recs


def _get_shap_recs(week: int, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identify the most influential features for this prediction according to SHAP,
    then translate each into a clear suggestion plus an explanatory sentence
    that a non‑expert can understand.
    """
    pipe = MODELS[week]
    pre = pipe.named_steps['pre']
    explainer = EXPLAINERS[week]
    Xp = pre.transform(df)
    sv = explainer(Xp).values[0]
    names = explainer.feature_names
    idxs = np.argsort(-np.abs(sv))[:TOP_K_SHAP]
    recs = []

    for i in idxs:
        feat, val = names[i], float(sv[i])
        if val > 0:
            impact_desc = f"increased your predicted SCIM by {val:.2f} points"
        else:
            impact_desc = f"decreased your predicted SCIM by {abs(val):.2f} points"

        if feat.startswith('num__SelfCare_'):
            wk = feat.split('_')[-1]
            if val > 0:
                text = f"Keep up your self‑care exercises from week {wk}."
                rationale = (
                    f"Your self‑care activities at week {wk} {impact_desc}, "
                    "so continuing them supports your functional independence."
                )
            else:
                text = f"Increase your self‑care practice, especially at week {wk}."
                rationale = (
                    f"Your self‑care at week {wk} {impact_desc}, "
                    "so adding more dressing and grooming drills could boost your recovery."
                )

        elif feat.startswith('num__Mobility_'):
            wk = feat.split('_')[-1]
            if val > 0:
                text = f"Continue your mobility drills for week {wk}."
                rationale = (
                    f"Mobility training at week {wk} {impact_desc}, "
                    "demonstrating its importance in improving transfers and walking."
                )
            else:
                text = f"Add extra mobility training for week {wk}."
                rationale = (
                    f"Mobility at week {wk} {impact_desc}, "
                    "so dedicating more time to walking or balance tasks may help."
                )

        elif feat.startswith('num__Respiration_'):
            wk = feat.split('_')[-1]
            if val > 0:
                text = f"Keep up breathing and cough exercises in week {wk}."
                rationale = (
                    f"Respiratory exercises at week {wk} {impact_desc}, "
                    "supporting airway clearance and overall endurance."
                )
            else:
                text = f"Add extra respiratory work in week {wk}."
                rationale = (
                    f"Respiration routines at week {wk} {impact_desc}, "
                    "so increasing breathing exercises may improve lung function and activity tolerance."
                )

        elif feat.startswith('cat__TherapyPlan_'):
            plan = feat.split('cat__TherapyPlan_', 1)[1]
            if val > 0:
                text = f"Schedule more sessions of “{plan}.”"
                rationale = (
                    f"The therapy “{plan}” {impact_desc}, "
                    "so adding extra sessions could further enhance your recovery."
                )
            else:
                text = f"Consider adding a new therapy alongside “{plan}.”"
                rationale = (
                    f"The current use of “{plan}” {impact_desc}, "
                    "indicating you might benefit from supplementing it with another complementary modality."
                )

        elif feat.startswith('num__SessionsPerWeek'):
            key = 'high' if val > 0 else 'low'
            text = RECOMMENDATIONS['SessionsPerWeek'][key]['text']
            rationale = (
                f"SHAP analysis shows your number of sessions per week {impact_desc}. "
                "Adjusting this frequency can significantly affect your predicted outcome."
            )

        else:
            label = feat.split('__', 1)[-1]
            text = f"Review the factor “{label}.”"
            rationale = (
                f"The feature “{label}” {impact_desc}, "
                "but we do not yet have a tailored suggestion for it."
            )

        recs.append({
            'feature': feat,
            'shap_value': val,
            'recommendation': text,
            'rationale': rationale
        })

    return recs


# ─── LOAD MODELS & EXPLAINERS ──────────────────────────────────────────────────
MODELS: Dict[int, Any] = {}
EXPLAINERS: Dict[int, Any] = {}
for wk in (6, 12, 18, 24):
    MODELS[wk] = pickle.load(open(f"model_week{wk}.pkl", "rb"))
    EXPLAINERS[wk] = pickle.load(open(f"shap_explainer_week{wk}.pkl", "rb"))


@app.post("/predict/{week}")
async def predict(
        week: int = Path(..., ge=6, le=24),
        payload: FeaturePayload = None
):
    if not payload or 'features' not in payload.__dict__:
        raise HTTPException(400, "Missing JSON body with 'features'")
    feats = payload.features
    tgt = f"TotalSCIM_{week}"
    if tgt in feats:
        raise HTTPException(400, f"Do not include '{tgt}' in features")

    # 1) Prediction
    df_in = _prepare_df(week, feats)
    pred = float(MODELS[week].predict(df_in)[0])

    # 2) Cost & efficiency
    spw = int(feats.get("SessionsPerWeek", 0))
    total_sess = spw * DURATION_WEEKS
    base_plans = feats.get("TherapyPlan", []) or []
    per_plan = total_sess / max(1, len(base_plans))
    raw_base = sum(therapy_costs[p] * per_plan for p in base_plans)
    over_n = np.random.poisson(0.1 * spw)
    raw_over = sum(therapy_costs[random.choice(base_plans)] for _ in range(over_n))
    factor = insurance_factors.get(feats.get("Insurance", ""), 1.0)
    baseline_cost = raw_base * factor
    overuse_cost = raw_over * factor
    total_cost = (raw_base + raw_over) * factor
    efficiency = pred / total_cost if total_cost > 0 else None

    cost_breakdown = {
        'baseline_cost': round(baseline_cost, 2),
        'overuse_cost': round(overuse_cost, 2),
        'total_cost': round(total_cost, 2),
        'efficiency': round(efficiency, 6) if efficiency else None,
        'explanations': [
            {
                'feature': 'BaselineCost',
                'recommendation': f"{spw} sess/wk × {DURATION_WEEKS} wk = {total_sess} sess → CHF{raw_base:.2f} ×{factor:.2f}"
            },
            {
                'feature': 'OveruseCost',
                'recommendation': f"{over_n} extra sess → CHF{raw_over:.2f} ×{factor:.2f}"
            },
            {
                'feature': 'TotalCost',
                'recommendation': f"Total billed = CHF{total_cost:.2f}"
            },
            {
                'feature': 'Efficiency',
                'recommendation': f"{pred:.1f} pts ÷ CHF{total_cost:.2f} ≃ {efficiency:.4f} pts/CHF"
            }
        ]
    }

    # 3) Gather recommendations
    static_recs = _get_static_recs(feats)
    shap_recs = _get_shap_recs(week, df_in)

    return {
        'week': week,
        'prediction': round(pred, 2),
        'cost': cost_breakdown,
        'static_recommendations': static_recs,
        'shap_recommendations': shap_recs
    }