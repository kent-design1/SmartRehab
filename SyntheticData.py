import random
import pandas as pd
import numpy as np

# ==========================
# Configuration Parameters
# ==========================
NUM_PATIENTS = 5000         # Number of patient records to generate
NUM_WEEKS = 24              # Number of weeks for rehabilitation

# Domain maximum scores for SCIM calculation
MAX_SELF_CARE = 40          # Maximum score for Self-Care
MAX_RESPIRATION = 20        # Maximum score for Respiration & Sphincter Management
MAX_MOBILITY = 40           # Maximum score for Mobility

# Health conditions and their probabilities
health_conditions = ["Stroke Recovery", "Orthopedic Injury", "Chronic Pain", "Diabetes", "Hypertension"]
condition_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal distribution

# Therapy plan options for non-stroke conditions (for Stroke Recovery, we use a custom classifier)
therapy_plan_options = {
    "Orthopedic Injury": (["Physiotherapy", "Occupational Therapy"], [0.8, 0.2]),
    "Chronic Pain": (["Physiotherapy", "Medication", "Lifestyle Changes"], [0.5, 0.4, 0.1]),
    "Diabetes": (["Medication", "Lifestyle Changes"], [0.8, 0.2]),
    "Hypertension": (["Medication", "Lifestyle Changes"], [0.7, 0.3])
}

# Define per-session costs for each therapy plan
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

# Patient progress probabilities
progress_probs = {"Improving": 0.45, "Stable": 0.40, "Deteriorating": 0.15}

# Therapy adjustment probabilities (used when a plateau is detected)
adjustment_probs_given_progress = {
    "Improving": {"Maintain Plan": 0.6, "Reduce Intensity": 0.4},
    "Stable": {"Maintain Plan": 0.5, "Increase Frequency": 0.3, "Change Therapy": 0.2},
    "Deteriorating": {"Change Therapy": 0.7, "Increase Frequency": 0.3}
}

# Follow-up status probabilities (not used in weekly simulation)
followup_probs_given_progress = {
    "Improving": {"Scheduled": 0.40, "Completed": 0.55, "Missed": 0.05},
    "Stable": {"Scheduled": 0.50, "Completed": 0.40, "Missed": 0.10},
    "Deteriorating": {"Scheduled": 0.50, "Completed": 0.20, "Missed": 0.30}
}

# Gender distribution
genders = ["Male", "Female"]
gender_probs = [0.7, 0.3]

# Age distribution by condition (mean and std)
age_distribution_by_condition = {
    "Stroke Recovery": {"mean": 65, "std": 12},
    "Orthopedic Injury": {"mean": 45, "std": 15},
    "Chronic Pain": {"mean": 50, "std": 12},
    "Diabetes": {"mean": 55, "std": 10},
    "Hypertension": {"mean": 60, "std": 10}
}

# ==========================
# Custom Therapy Plan Classifier Functions
# ==========================
def classify_therapy_plan(condition):
    """
    For patients with Stroke Recovery, assign a tailored therapy plan.
    """
    if condition == "Stroke Recovery":
        therapy_options = [
            "Respiratory Management",
            "Spasticity Management",
            "Mobility & Upper Limb Training",
            "Strength & FES Training",
            "Comprehensive SCI Rehab"
        ]
        return random.choice(therapy_options)
    else:
        plans, _ = therapy_plan_options[condition]
        return random.choice(plans)

def pick_alternate_therapy(current_therapy, condition):
    """
    Choose an alternative therapy plan from the available options (excluding the current one).
    """
    if condition == "Stroke Recovery":
        therapy_options = [
            "Respiratory Management",
            "Spasticity Management",
            "Mobility & Upper Limb Training",
            "Strength & FES Training",
            "Comprehensive SCI Rehab"
        ]
    else:
        therapy_options, _ = therapy_plan_options[condition]
    alternatives = [plan for plan in therapy_options if plan != current_therapy]
    return random.choice(alternatives) if alternatives else current_therapy

# ==========================
# Helper Functions for Weekly SCIM Simulation
# ==========================
def initial_domain_scores():
    """Generate baseline domain scores for SCIM (assume 20-40% of maximum)."""
    self_care = random.randint(0, int(0.4 * MAX_SELF_CARE))
    respiration = random.randint(0, int(0.4 * MAX_RESPIRATION))
    mobility = random.randint(0, int(0.4 * MAX_MOBILITY))
    return self_care, respiration, mobility

def update_weekly_score(current_score, improvement_factor):
    """Update a domain score for one week by adding improvement plus noise."""
    noise = np.random.normal(0, 0.3)
    return max(0, current_score + improvement_factor + noise)

def calculate_total_scim(self_care, respiration, mobility):
    """Calculate total SCIM from the three domains."""
    return self_care + respiration + mobility

# ==========================
# Generate Patient Record with Weekly SCIM Scores and Therapy Adjustments
# ==========================
def generate_patient_record(patient_id):
    """Generate a synthetic record for one patient over 24 weeks with realistic SCIM scores, cost, and therapy transitions."""
    # Choose health condition and initial therapy plan
    condition = random.choices(health_conditions, weights=condition_probs, k=1)[0]
    current_therapy = classify_therapy_plan(condition)

    # Record therapy stages and weeks (T1, T2, ...)
    therapy_stages = [current_therapy]
    therapy_stage_weeks = [0]  # Starting at week 0

    # Baseline demographics
    age = int(random.gauss(age_distribution_by_condition.get(condition, {"mean":50, "std":15})["mean"],
                           age_distribution_by_condition.get(condition, {"mean":50, "std":15})["std"]))
    age = max(18, min(age, 90))
    gender = random.choices(genders, weights=gender_probs, k=1)[0]

    # Physical measurements
    height_cm = random.randint(150, 200)
    height_m = height_cm / 100.0
    bmi_target = random.uniform(18, 35)
    weight_kg = round(bmi_target * (height_m ** 2), 1)
    bmi = round(weight_kg / (height_m ** 2), 1)

    # Therapy session info
    sessions_per_week = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2], k=1)[0]
    duration_weeks = random.randint(12, 52)

    # Calculate therapy cost based on current therapy plan
    per_session_cost = therapy_costs.get(current_therapy, 100)
    total_therapy_cost = per_session_cost * sessions_per_week * duration_weeks

    # Generate baseline SCIM scores (week 0)
    base_self, base_resp, base_mob = initial_domain_scores()
    base_scim = calculate_total_scim(base_self, base_resp, base_mob)

    # Set improvement factors for each domain
    factor_self = sessions_per_week * random.uniform(0.2, 0.8)
    factor_resp = sessions_per_week * random.uniform(0.1, 0.5)
    factor_mob  = sessions_per_week * random.uniform(0.2, 0.8)

    # Initialize weekly score lists
    weekly_self = [base_self]
    weekly_resp = [base_resp]
    weekly_mob  = [base_mob]
    weekly_scim = [base_scim]

    # Initialize variables for adjustments
    adjustment_types = []      # This will record suggestions like "Increase Frequency", "Reduce Frequency", or the new therapy plan.
    adjustment_weeks = []      # Weeks when adjustments occur.
    plateau_counter = 0
    improvement_threshold = 0.5  # Minimal improvement threshold per week

    for week in range(1, NUM_WEEKS + 1):
        new_self = min(MAX_SELF_CARE, update_weekly_score(weekly_self[-1], factor_self))
        new_resp = min(MAX_RESPIRATION, update_weekly_score(weekly_resp[-1], factor_resp))
        new_mob  = min(MAX_MOBILITY, update_weekly_score(weekly_mob[-1], factor_mob))
        new_scim = calculate_total_scim(new_self, new_resp, new_mob)

        weekly_self.append(new_self)
        weekly_resp.append(new_resp)
        weekly_mob.append(new_mob)
        weekly_scim.append(new_scim)

        # Check weekly improvement
        if new_scim - weekly_scim[-2] < improvement_threshold:
            plateau_counter += 1
        else:
            plateau_counter = 0

        # If plateau persists for 4 consecutive weeks, perform an adjustment
        if plateau_counter >= 4:
            r = random.random()
            # If sessions per week are low, suggest increasing frequency
            if r < 0.33:
                if sessions_per_week < 6:
                    new_adjustment = "Increase Frequency"
                    sessions_per_week = min(sessions_per_week + random.choice([1, 2]), 6)
                else:
                    new_adjustment = "Maintain Frequency"
            # If sessions per week are high, suggest reducing frequency
            elif r < 0.66:
                if sessions_per_week > 2:
                    new_adjustment = "Reduce Frequency"
                    sessions_per_week = max(sessions_per_week - random.choice([1, 2]), 2)
                else:
                    new_adjustment = "Maintain Frequency"
            else:
                # Change Therapy: pick an alternate therapy plan
                new_adjustment = pick_alternate_therapy(current_therapy, condition)
                if new_adjustment != current_therapy:
                    current_therapy = new_adjustment
                    therapy_stages.append(current_therapy)
                    therapy_stage_weeks.append(week)
                    per_session_cost = therapy_costs.get(current_therapy, 100)
            # Recalculate total therapy cost with updated sessions and/or therapy
            total_therapy_cost = per_session_cost * sessions_per_week * duration_weeks
            adjustment_types.append(new_adjustment)
            adjustment_weeks.append(week)
            plateau_counter = 0  # Reset counter after adjustment

        # Optional: Taper improvement factors as scores approach maximum
        if new_scim > 90:
            factor_self *= 0.95
            factor_resp *= 0.95
            factor_mob  *= 0.95

    # Final SCIM score: add a small random boost, capped at 100.
    scim_final = min(100, int(weekly_scim[-1] + random.uniform(0, 5)))
    cost_efficiency = scim_final / (total_therapy_cost + 1)

    record = {
        "Patient ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Health Condition": condition,
        "Therapy Plan": therapy_stages[0],
        "Therapy Adjustments": adjustment_types,     # Array of adjustments (suggested changes)
        "Adjustment Weeks": adjustment_weeks,          # Array of weeks when adjustments occurred
        "Therapy Stages": therapy_stages,              # The actual therapy plan history (T1, T2, ...)
        "Initial Sessions per Week": sessions_per_week,
        "Duration (weeks)": duration_weeks,
        "Baseline SCIM": base_scim,
        "SCIM Final": scim_final,
        "Weekly SCIM Scores": weekly_scim,             # List from week 0 to week 24
        "Weekly Self-Care": weekly_self,
        "Weekly Respiration": weekly_resp,
        "Weekly Mobility": weekly_mob,
        "Total Therapy Cost": total_therapy_cost,
        "Cost Efficiency": cost_efficiency,
        # Additional measures
        "Height (cm)": random.randint(150, 200),
        "Weight (kg)": round(random.uniform(50, 120), 1),
        "BMI": round(random.uniform(18, 35), 1),
        "Patient Engagement Score": random.randint(1, 100),
        "Mental Health Scale": random.randint(1, 6),
        "Pain Level": random.randint(1, 11),
        "Fatigue Level": random.randint(1, 11),
        "Muscle Strength Upper": random.randint(1, 10),
        "Muscle Strength Lower": random.randint(1, 10),
        "Balance Test Score": random.randint(1, 101),
        "Mobility Score": random.randint(1, 101)
    }
    return record

# ==========================
# Generate Dataset for 5000 Patients and Save as CSV
# ==========================
data = [generate_patient_record(pid) for pid in range(1, NUM_PATIENTS + 1)]
df = pd.DataFrame(data)
df.to_csv("synthetic_rehab_dataset.csv", index=False)

print("Dataset generated and saved as 'synthetic_rehab_dataset.csv'.")
print(df.head(5))