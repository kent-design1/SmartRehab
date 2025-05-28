#!/usr/bin/env python3
# synthetic_rehab_refined_generator.py
# Generates a refined synthetic rehab dataset with realistic total rehab costs
# and breakdown of raw costs vs. insured vs. patient payments.

import random
import pandas as pd
import numpy as np

# ==========================
# Configuration
# ==========================
NUM_PATIENTS = 600
MEAS_WEEKS    = [0, 6, 12, 18, 24]
MISSING_PROB  = 0.05  # Missing SCIM measurement probability
MAX_SELF, MAX_RESP, MAX_MOB = 40, 20, 40

# Health conditions & therapy options
health_conditions = [
    "Stroke Recovery", "Orthopedic Injury", "Chronic Pain", "Diabetes", "Hypertension"
]
condition_probs = [0.2] * len(health_conditions)

therapy_plan_options = {
    "Stroke Recovery": [
        "Respiratory Management", "Spasticity Management",
        "Mobility & Upper Limb Training", "Strength & FES Training",
        "Comprehensive SCI Rehab", "Robotic Gait Training",
        "Virtual Reality Therapy", "Aquatic Therapy",
        "Constraint-Induced Movement Therapy", "Mirror Therapy"
    ],
    "Orthopedic Injury": [
        "Physiotherapy", "Occupational Therapy", "Manual Therapy",
        "Hydrotherapy", "Proprioceptive Neuromuscular Facilitation",
        "Functional Strength Training", "Balance & Proprioception Drills"
    ],
    "Chronic Pain": [
        "Physiotherapy", "Medication", "Lifestyle Changes",
        "Cognitive Behavioral Therapy", "Mindfulness-Based Stress Reduction",
        "Pain Neuroscience Education", "Yoga/Tai-Chi", "Graded Activity/Exposure"
    ],
    "Diabetes": [
        "Medication", "Lifestyle Changes", "Nutritional Counseling",
        "Aerobic Exercise Program", "Resistance Training",
        "Foot Care Education", "Glucose-Guided Activity"
    ],
    "Hypertension": [
        "Medication", "Lifestyle Changes", "Structured Aerobic Training",
        "Resistance Exercise", "Stress Management",
        "Dietary Sodium Reduction", "Tele-rehab Monitoring"
    ]
}

# Per-session base costs (CHF)
therapy_costs = {
    "Respiratory Management": 500, "Spasticity Management": 450,
    "Mobility & Upper Limb Training": 400, "Strength & FES Training": 480,
    "Comprehensive SCI Rehab": 650, "Robotic Gait Training": 900,
    "Virtual Reality Therapy": 800, "Aquatic Therapy": 550,
    "Constraint-Induced Movement Therapy": 700, "Mirror Therapy": 350,
    "Physiotherapy": 380, "Occupational Therapy": 420, "Manual Therapy": 460,
    "Hydrotherapy": 500, "Proprioceptive Neuromuscular Facilitation": 550,
    "Functional Strength Training": 400, "Balance & Proprioception Drills": 320,
    "Medication": 200, "Lifestyle Changes": 150, "Cognitive Behavioral Therapy": 600,
    "Mindfulness-Based Stress Reduction": 520, "Pain Neuroscience Education": 480,
    "Yoga/Tai-Chi": 350, "Graded Activity/Exposure": 450,
    "Nutritional Counseling": 480, "Aerobic Exercise Program": 360,
    "Resistance Training": 400, "Foot Care Education": 300,
    "Glucose-Guided Activity": 330, "Structured Aerobic Training": 380,
    "Resistance Exercise": 420, "Stress Management": 460,
    "Dietary Sodium Reduction": 280, "Tele-rehab Monitoring": 600
}

# Insurance factors (to apply to raw costs) and coverage percentages
insurance_factors = {
    "Basic Mandatory": 0.9,
    "Supplementary Private": 1.1,
    "Employer-Sponsored": 1.0,
    "Uninsured": 1.2
}
coverage_pct = {
    "Basic Mandatory":    0.90,
    "Supplementary Private": 1.00,
    "Employer-Sponsored": 0.95,
    "Uninsured":          0.00
}

# Demographics & labs
ethnicities      = ["Swiss","German","French","Italian","Other"]
education_levels = ["Compulsory","Apprenticeship","Vocational","Bachelor","Master/Doctorate"]
insurance_types  = list(insurance_factors.keys())
regions          = ["Zurich","Bern","Geneva","Vaud","Other"]

# Pre-generate lab values
np.random.seed(42)
crp_vals   = np.clip(np.random.normal(5,3,NUM_PATIENTS),0.1,50)
hb_vals    = np.clip(np.random.normal(13.5,1,NUM_PATIENTS),8,18)
gluc_vals  = np.clip(np.random.normal(100,20,NUM_PATIENTS),50,300)
phq9_vals  = np.random.randint(0,28,NUM_PATIENTS)
gad7_vals  = np.random.randint(0,22,NUM_PATIENTS)

def simulate_scim_path(base, rates, weeks):
    """
    Logistic‐style growth that asymptotically approaches max subscale values.
    base: [self, resp, mob] initial scores at week 0
    rates: growth rate parameters per subscale
    weeks: list of measurement weeks
    """
    max_vals = [MAX_SELF, MAX_RESP, MAX_MOB]
    self_c, resp_c, mob_c, total_sc = [], [], [], []

    for idx, wk in enumerate(weeks):
        if idx == 0:
            vals = base
        else:
            vals = []
            frac = wk / weeks[-1]  # normalize time 0→1
            for i, init in enumerate(base):
                mv   = max_vals[i]
                k    = rates[i]  # larger → faster approach
                # logistic‐style approach via exponential decay
                val = mv - (mv - init) * np.exp(-k * frac)
                # add small noise
                noise = np.random.normal(0, 0.5 + 0.02 * wk)
                val  = np.clip(val + noise, 0, mv)
                vals.append(round(val,1))
        self_c.append(vals[0])
        resp_c.append(vals[1])
        mob_c.append(vals[2])
        total_sc.append(round(vals[0] + vals[1] + vals[2],1))

    return self_c, resp_c, mob_c, total_sc

records = []
for pid in range(1, NUM_PATIENTS+1):
    # Demographics & labs
    age       = int(np.clip(np.random.normal(60,15),18,90))
    gender    = random.choice(["Male","Female","Other"])
    ethnicity = random.choice(ethnicities)
    education = random.choice(education_levels)
    insurance = random.choice(insurance_types)
    region    = random.choice(regions)

    labs = {
        'CRP_mg_L': round(crp_vals[pid-1],1),
        'Hemoglobin_g_dL': round(hb_vals[pid-1],1),
        'Glucose_mg_dL': round(gluc_vals[pid-1],1),
        'PHQ9': int(phq9_vals[pid-1]),
        'GAD7': int(gad7_vals[pid-1])
    }

    # Condition & SCIM
    cond        = random.choices(health_conditions, weights=condition_probs)[0]
    sessions_pw = random.choice([2,3,4,5])
    base_scores = [
        random.randint(int(0.2*MAX_SELF), int(0.4*MAX_SELF)),
        random.randint(int(0.2*MAX_RESP), int(0.4*MAX_RESP)),
        random.randint(int(0.2*MAX_MOB),  int(0.4*MAX_MOB))
    ]
    # tie growth rates to therapy intensity
    rates = [sessions_pw * random.uniform(0.1,0.6) for _ in base_scores]
    sc_self, sc_resp, sc_mob, sc_total = simulate_scim_path(base_scores, rates, MEAS_WEEKS)

    rec = {
        'PatientID': pid, 'Age': age, 'Gender': gender,
        'Ethnicity': ethnicity, 'Education': education,
        'Insurance': insurance, 'Region': region,
        'HealthCondition': cond,
        **labs,
        'SessionsPerWeek': sessions_pw
    }

    # ——— Raw cost calculation (before insurance) —————————————————
    duration = MEAS_WEEKS[-1]
    total_sessions = sessions_pw * duration
    plans0 = random.sample(therapy_plan_options[cond], k=random.randint(1,3))
    per_plan = total_sessions / len(plans0)
    raw_base_cost = sum(therapy_costs[p] * per_plan for p in plans0)
    raw_overuse_sessions = np.random.poisson(0.1 * sessions_pw)
    raw_overuse_cost = sum(
        therapy_costs[random.choice(plans0)] for _ in range(raw_overuse_sessions)
    )
    raw_total_cost = raw_base_cost + raw_overuse_cost

    # ——— Apply insurance factor to get billed amounts —————————————
    factor = insurance_factors[insurance]
    billed_base    = raw_base_cost * factor
    billed_overuse = raw_overuse_cost * factor
    billed_total   = raw_total_cost * factor

    # ——— Split raw_total between insurer & patient ——————————————
    cov = coverage_pct[insurance]
    insurance_pays = raw_total_cost * cov
    patient_pays   = raw_total_cost - insurance_pays

    rec.update({
        'RawBaselineCost': round(raw_base_cost,2),
        'RawOveruseCost':  round(raw_overuse_cost,2),
        'RawTotalCost':    round(raw_total_cost,2),
        'BaselineCost':    round(billed_base,2),
        'OveruseCost':     round(billed_overuse,2),
        'TotalCost':       round(billed_total,2),
        'InsurancePays':   round(insurance_pays,2),
        'PatientPays':     round(patient_pays,2),
    })

    # ——— Therapy plans & SCIM per week ———————————————————————————
    for idx, wk in enumerate(MEAS_WEEKS):
        opts   = therapy_plan_options[cond]
        chosen = random.sample(opts, k=random.randint(1,3))
        rec[f'TherapyPlans_{wk}']  = chosen
        rec[f'NumPlans_{wk}']      = len(chosen)
        rec[f'SelfCare_{wk}']      = sc_self[idx]
        rec[f'Respiration_{wk}']   = sc_resp[idx]
        rec[f'Mobility_{wk}']       = sc_mob[idx]
        rec[f'TotalSCIM_{wk}']      = sc_total[idx]

    records.append(rec)

# ——— Save to CSV ————————————————————————————————————————————————
df_out = pd.DataFrame(records)
df_out.to_csv('synthetic_rehab_refined.csv', index=False)

# quick summary
print(df_out[
          ['RawBaselineCost','RawOveruseCost','RawTotalCost',
           'BaselineCost','OveruseCost','TotalCost',
           'InsurancePays','PatientPays']
      ].describe())