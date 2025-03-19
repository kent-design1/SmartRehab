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

# Patient progress probabilities (for initial random assignment, not used in weekly simulation)
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
# Custom Therapy Plan Classifier Function
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
        plans, plan_weights = therapy_plan_options[condition]
        return random.choices(plans, weights=plan_weights, k=1)[0]

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
    """Update a domain score for one week by adding a small improvement plus noise."""
    noise = np.random.normal(0, 0.3)  # Reduced noise for more realistic progression
    return max(0, current_score + improvement_factor + noise)

def calculate_total_scim(self_care, respiration, mobility):
    """Calculate total SCIM from the three domains."""
    return self_care + respiration + mobility

# ==========================
# Generate Patient Record with Weekly SCIM Scores
# ==========================
def generate_patient_record(patient_id):
    """Generate a synthetic record for one patient over 24 weeks with realistic SCIM scores."""
    # Choose a health condition
    condition = random.choices(health_conditions, weights=condition_probs, k=1)[0]

    # Determine therapy plan: use custom classifier for Stroke Recovery; else use predefined options.
    if condition == "Stroke Recovery":
        therapy_plan = classify_therapy_plan(condition)
    else:
        plans, plan_weights = therapy_plan_options[condition]
        therapy_plan = random.choices(plans, weights=plan_weights, k=1)[0]

    # Baseline demographics
    age = int(random.gauss(age_distribution_by_condition.get(condition, {"mean":50, "std":15})["mean"],
                           age_distribution_by_condition.get(condition, {"mean":50, "std":15})["std"]))
    age = max(18, min(age, 90))
    gender = random.choices(genders, weights=gender_probs, k=1)[0]

    # Physical measurements: height, weight, BMI
    height_cm = random.randint(150, 200)
    height_m = height_cm / 100.0
    bmi_target = random.uniform(18, 35)
    weight_kg = round(bmi_target * (height_m ** 2), 1)
    bmi = round(weight_kg / (height_m ** 2), 1)

    # Therapy session info
    sessions_per_week = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2], k=1)[0]
    duration_weeks = random.randint(12, 52)  # treatment duration in weeks

    # Generate baseline SCIM domain scores (at week 0)
    base_self, base_resp, base_mob = initial_domain_scores()
    base_scim = calculate_total_scim(base_self, base_resp, base_mob)

    # Set initial improvement factors (reduced ranges for more gradual improvement)
    factor_self = sessions_per_week * random.uniform(0.2, 0.8)
    factor_resp = sessions_per_week * random.uniform(0.1, 0.5)
    factor_mob  = sessions_per_week * random.uniform(0.2, 0.8)

    # Simulate weekly scores lists for each domain
    weekly_self = [base_self]
    weekly_resp = [base_resp]
    weekly_mob  = [base_mob]
    weekly_scim = [base_scim]

    # Flag and record therapy adjustments (if plateau is detected)
    therapy_adjusted = 0
    adjustment_week = None
    adjustment_type = "None"
    plateau_counter = 0  # count consecutive weeks with minimal improvement
    improvement_threshold = 0.5  # minimal SCIM improvement threshold per week

    for week in range(1, NUM_WEEKS + 1):
        # Update domain scores with a small weekly gain
        new_self = min(MAX_SELF_CARE, update_weekly_score(weekly_self[-1], factor_self))
        new_resp = min(MAX_RESPIRATION, update_weekly_score(weekly_resp[-1], factor_resp))
        new_mob  = min(MAX_MOBILITY, update_weekly_score(weekly_mob[-1], factor_mob))
        new_scim = calculate_total_scim(new_self, new_resp, new_mob)

        weekly_self.append(new_self)
        weekly_resp.append(new_resp)
        weekly_mob.append(new_mob)
        weekly_scim.append(new_scim)

        # Check for plateau: if weekly improvement is below threshold
        if new_scim - weekly_scim[-2] < improvement_threshold:
            plateau_counter += 1
        else:
            plateau_counter = 0

        # If plateau persists for 4 consecutive weeks, perform one therapy adjustment
        if plateau_counter >= 4 and therapy_adjusted == 0:
            therapy_adjusted = 1
            adjustment_week = week
            if random.random() < 0.5:
                adjustment_type = "Increase Frequency"
                sessions_per_week = min(sessions_per_week + random.choice([1, 2]), 6)
            else:
                adjustment_type = "Change Therapy"
                # Simulate a modest boost in improvement factors when switching therapy
                factor_self *= random.uniform(1.1, 1.3)
                factor_resp *= random.uniform(1.1, 1.3)
                factor_mob  *= random.uniform(1.1, 1.3)
            plateau_counter = 0  # reset counter after adjustment

        # Optional: If scores approach the maximum, improvement naturally tapers off.
        if new_scim > 90:
            factor_self *= 0.95
            factor_resp *= 0.95
            factor_mob  *= 0.95

    # To make the final SCIM score more realistic, we add a small random boost from week 24 but do not let it hit 100 for everyone.
    scim_final = min(100, int(weekly_scim[-1] + random.uniform(0, 5)))

    record = {
        "Patient ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Health Condition": condition,
        "Therapy Plan": therapy_plan,
        "Therapy Adjustment": adjustment_type,
        "Initial Sessions per Week": sessions_per_week,
        "Duration (weeks)": duration_weeks,
        "Baseline SCIM": base_scim,
        "SCIM Final": scim_final,
        "Therapy Adjusted": therapy_adjusted,
        "Adjustment Week": adjustment_week,
        "Weekly SCIM Scores": weekly_scim,  # List from week 0 to week 24
        "Weekly Self-Care": weekly_self,
        "Weekly Respiration": weekly_resp,
        "Weekly Mobility": weekly_mob,
        # Additional measures:
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
        "Mobility Score": random.randint(1, 101),
        # Cost features
        "Total Therapy Cost": random.randint(5000, 20000),
        "Cost Efficiency": round(random.uniform(0.001, 0.01), 4)
    }
    return record

# ==========================
# Generate Dataset for 5000 Patients and Create DataFrame
# ==========================
data = [generate_patient_record(pid) for pid in range(1, NUM_PATIENTS + 1)]
df = pd.DataFrame(data)

# Save the DataFrame to CSV (weekly scores are stored as a list, here converted to string)
df.to_csv("synthetic_rehab_dataset.csv", index=False)

print("Dataset generated and saved as 'synthetic_rehab_dataset.csv'.")
print(df.head(5))