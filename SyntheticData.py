import random
import pandas as pd
import numpy as np

# ==========================
# Configuration Parameters
# ==========================
NUM_PATIENTS = 5000  # Number of patient records to generate

# Health conditions and their probabilities
health_conditions = ["Stroke Recovery", "Orthopedic Injury", "Chronic Pain", "Diabetes", "Hypertension"]
condition_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal distribution for simplicity

# Therapy plan options for each condition (in our restructured dataset, these represent relevant SCI rehab approaches)
therapy_plan_options = {
    "Stroke Recovery": (["SCI Physiotherapy", "Gait & Balance Training", "Strength & FES Training"], [0.5, 0.3, 0.2]),
    "Orthopedic Injury": (["SCI Physiotherapy", "Gait & Balance Training", "Strength & FES Training"], [0.5, 0.3, 0.2]),
    "Chronic Pain": (["SCI Physiotherapy", "Gait & Balance Training", "Strength & FES Training"], [0.5, 0.3, 0.2]),
    "Diabetes": (["SCI Physiotherapy", "Gait & Balance Training", "Strength & FES Training"], [0.5, 0.3, 0.2]),
    "Hypertension": (["SCI Physiotherapy", "Gait & Balance Training", "Strength & FES Training"], [0.5, 0.3, 0.2])
}

# Patient progress probabilities (for classification purposes)
progress_probs = {"Improving": 0.45, "Stable": 0.40, "Deteriorating": 0.15}

# Therapy adjustment probabilities conditional on progress
adjustment_probs_given_progress = {
    "Improving": {"Maintain Plan": 0.6, "Reduce Intensity": 0.4},
    "Stable": {"Maintain Plan": 0.5, "Increase Frequency": 0.3, "Change Therapy": 0.2},
    "Deteriorating": {"Change Therapy": 0.7, "Increase Frequency": 0.3}
}

# Follow-up status probabilities given progress
followup_probs_given_progress = {
    "Improving": {"Scheduled": 0.40, "Completed": 0.55, "Missed": 0.05},
    "Stable": {"Scheduled": 0.50, "Completed": 0.40, "Missed": 0.10},
    "Deteriorating": {"Scheduled": 0.50, "Completed": 0.20, "Missed": 0.30}
}

# Gender distribution
genders = ["Male", "Female"]
gender_probs = [0.5, 0.5]

# Age distribution by condition (mean and std for normal distribution)
age_distribution_by_condition = {
    "Stroke Recovery": {"mean": 65, "std": 12},
    "Orthopedic Injury": {"mean": 45, "std": 15},
    "Chronic Pain": {"mean": 50, "std": 12},
    "Diabetes": {"mean": 55, "std": 10},
    "Hypertension": {"mean": 60, "std": 10}
}

# ==========================
# Data Generation Functions
# ==========================

def get_age_for_condition(condition):
    """Generate a realistic age for a given condition using a normal distribution."""
    dist = age_distribution_by_condition.get(condition, {"mean": 50, "std": 15})
    age = int(random.gauss(dist["mean"], dist["std"]))
    return max(18, min(age, 90))

def get_progress_and_adjustment():
    """Randomly determine patient progress and therapy adjustment based on defined probabilities."""
    progress = random.choices(list(progress_probs.keys()), weights=list(progress_probs.values()), k=1)[0]
    adjust_options = adjustment_probs_given_progress.get(progress)
    adjustment = random.choices(list(adjust_options.keys()), weights=list(adjust_options.values()), k=1)[0]
    return progress, adjustment

def get_followup_status(progress):
    """Randomly determine follow-up status based on patient progress."""
    status_options = followup_probs_given_progress.get(progress)
    follow_status = random.choices(list(status_options.keys()), weights=list(status_options.values()), k=1)[0]
    return follow_status

def get_sessions_per_week(plan):
    """Generate a realistic number of sessions per week based on the therapy plan."""
    # Here, all therapy plans are treated similarly.
    return random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.15, 0.05], k=1)[0]

def get_duration_weeks(condition):
    """Generate a treatment duration (in weeks) based on the condition type."""
    if condition in ["Diabetes", "Hypertension", "Chronic Pain"]:
        return random.randint(12, 52)
    else:
        return random.randint(4, 40)

def generate_scim_scores():
    """
    Generate SCIM subscale scores:
    - Self-Care (0-40)
    - Respiration & Sphincter Management (0-20)
    - Mobility (0-40)
    Total SCIM is the sum (0-100).
    """
    self_care = random.randint(0, 40)
    resp_sphincter = random.randint(0, 20)
    mobility = random.randint(0, 40)
    total_scim = self_care + resp_sphincter + mobility
    return self_care, resp_sphincter, mobility, total_scim

def generate_patient_record(patient_id):
    """Generate a synthetic record for one patient, including SCIM and additional features."""
    # Basic attributes: choose a health condition based on the defined probabilities.
    condition = random.choices(health_conditions, weights=condition_probs, k=1)[0]
    # Choose therapy plan from the options specific to the condition.
    plans, plan_weights = therapy_plan_options[condition]
    therapy_plan = random.choices(plans, weights=plan_weights, k=1)[0]
    # Determine patient progress and the corresponding therapy adjustment.
    progress, adjustment = get_progress_and_adjustment()
    follow_up = get_followup_status(progress)
    sessions_per_week = get_sessions_per_week(therapy_plan)
    duration_weeks = get_duration_weeks(condition)
    age = get_age_for_condition(condition)
    gender = random.choices(genders, weights=gender_probs, k=1)[0]

    # Additional patient attributes: Physical measurements and computed BMI.
    height_cm = random.randint(150, 200)
    height_m = height_cm / 100.0
    bmi_target = random.uniform(18, 35)
    weight_kg = round(bmi_target * (height_m ** 2), 1)
    bmi = round(weight_kg / (height_m ** 2), 1)

    # Other clinical and functional measures.
    injury_severity_score = random.randint(1, 10)
    time_since_injury_months = random.randint(1, 120)
    session_duration_minutes = random.randint(30, 120)
    exercise_intensity = random.randint(1, 10)
    patient_adherence_rate = round(random.uniform(0.5, 1.0), 2)
    mobility_score = random.randint(1, 101)
    grip_strength_kg = random.randint(5, 50)
    muscle_strength_score = random.randint(1, 10)
    pain_level = random.randint(1, 10)
    balance_test_score = random.randint(1, 101)
    quality_of_life_score = random.randint(1, 101)
    mental_health_status = random.randint(1, 6)
    sleep_quality_score = random.randint(1, 11)
    fatigue_level = random.randint(1, 11)
    daily_activity_score = random.randint(1, 101)
    patient_engagement_score = random.randint(1, 101)

    # Generate SCIM scores
    scim_self_care, scim_resp_sphincter, scim_mobility, scim_total = generate_scim_scores()

    return {
        "Patient ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Prior Health Conditions": random.choice(["None", "Diabetes", "Hypertension", "Obesity", "Osteoporosis", "Multiple"]),
        "Health Condition": condition,
        "Therapy Plan": therapy_plan,
        "Therapy Adjustment": adjustment,
        "Sessions per Week": sessions_per_week,
        "Duration of Treatment (weeks)": duration_weeks,
        "Patient Progress": progress,
        "Follow-up Status": follow_up,
        "Height (cm)": height_cm,
        "Weight (kg)": weight_kg,
        "BMI": bmi,
        "Injury Severity Score": injury_severity_score,
        "Time Since Injury (months)": time_since_injury_months,
        "Session Duration (minutes)": session_duration_minutes,
        "Exercise Intensity": exercise_intensity,
        "Patient Adherence Rate": patient_adherence_rate,
        "Mobility Score": mobility_score,
        "Grip Strength (kg)": grip_strength_kg,
        "Muscle Strength Score": muscle_strength_score,
        "Pain Level": pain_level,
        "Balance Test Score": balance_test_score,
        "Quality of Life Score": quality_of_life_score,
        "Mental Health Status": mental_health_status,
        "Sleep Quality Score": sleep_quality_score,
        "Fatigue Level": fatigue_level,
        "Daily Activity Score": daily_activity_score,
        "Patient Engagement Score": patient_engagement_score,
        "SCIM Self-Care": scim_self_care,
        "SCIM Respiration & Sphincter": scim_resp_sphincter,
        "SCIM Mobility": scim_mobility,
        "SCIM Total": scim_total
    }

# ==========================
# Generate Dataset for 5000 Patients and Create DataFrame
# ==========================
data = [generate_patient_record(pid) for pid in range(1, NUM_PATIENTS + 1)]
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv("synthetic_rehab_dataset.csv", index=False)

print("Dataset generated and saved as 'synthetic_rehab_dataset_5000.csv'.")
print(df.head(10))

















# import random
# import pandas as pd
# import numpy as np
#
# # ==========================
# # Configuration Parameters
# # ==========================
# NUM_PATIENTS = 5000  # Number of patient records to generate
#
# # Health conditions and their probabilities
# health_conditions = ["Stroke Recovery", "Orthopedic Injury", "Chronic Pain", "Diabetes", "Hypertension"]
# condition_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal distribution for simplicity
#
# # Therapy plan options for each condition
# therapy_plan_options = {
#     "Stroke Recovery": (["Physiotherapy", "Occupational Therapy"], [0.7, 0.3]),
#     "Orthopedic Injury": (["Physiotherapy", "Occupational Therapy"], [0.8, 0.2]),
#     "Chronic Pain": (["Physiotherapy", "Medication", "Lifestyle Changes"], [0.5, 0.4, 0.1]),
#     "Diabetes": (["Medication", "Lifestyle Changes"], [0.8, 0.2]),
#     "Hypertension": (["Medication", "Lifestyle Changes"], [0.7, 0.3])
# }
#
# # Patient progress probabilities
# progress_probs = {"Improving": 0.45, "Stable": 0.40, "Deteriorating": 0.15}
#
# # Therapy adjustment probabilities conditional on progress
# adjustment_probs_given_progress = {
#     "Improving": {"Maintain Plan": 0.6, "Reduce Intensity": 0.4},
#     "Stable": {"Maintain Plan": 0.5, "Increase Frequency": 0.3, "Change Therapy": 0.2},
#     "Deteriorating": {"Change Therapy": 0.7, "Increase Frequency": 0.3}
# }
#
# # Follow-up status probabilities given progress
# followup_probs_given_progress = {
#     "Improving": {"Scheduled": 0.40, "Completed": 0.55, "Missed": 0.05},
#     "Stable": {"Scheduled": 0.50, "Completed": 0.40, "Missed": 0.10},
#     "Deteriorating": {"Scheduled": 0.50, "Completed": 0.20, "Missed": 0.30}
# }
#
# # Gender distribution
# genders = ["Male", "Female"]
# gender_probs = [0.5, 0.5]
#
# # Age distribution by condition (mean and std for normal distribution)
# age_distribution_by_condition = {
#     "Stroke Recovery": {"mean": 65, "std": 12},
#     "Orthopedic Injury": {"mean": 45, "std": 15},
#     "Chronic Pain": {"mean": 50, "std": 12},
#     "Diabetes": {"mean": 55, "std": 10},
#     "Hypertension": {"mean": 60, "std": 10}
# }
#
# # ==========================
# # Data Generation Functions
# # ==========================
#
# def get_age_for_condition(condition):
#     """Generate a realistic age for a given condition using a normal distribution."""
#     dist = age_distribution_by_condition.get(condition, {"mean": 50, "std": 15})
#     age = int(random.gauss(dist["mean"], dist["std"]))
#     return max(18, min(age, 90))
#
# def get_progress_and_adjustment():
#     """Randomly determine patient progress and therapy adjustment based on defined probabilities."""
#     progress = random.choices(list(progress_probs.keys()), weights=list(progress_probs.values()), k=1)[0]
#     adjust_options = adjustment_probs_given_progress.get(progress)
#     adjustment = random.choices(list(adjust_options.keys()), weights=list(adjust_options.values()), k=1)[0]
#     return progress, adjustment
#
# def get_followup_status(progress):
#     """Randomly determine follow-up status based on patient progress."""
#     status_options = followup_probs_given_progress.get(progress)
#     follow_status = random.choices(list(status_options.keys()), weights=list(status_options.values()), k=1)[0]
#     return follow_status
#
# def get_sessions_per_week(plan):
#     """Generate a realistic number of sessions per week based on the therapy plan."""
#     if plan in ["Physiotherapy", "Occupational Therapy"]:
#         return random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.15, 0.05], k=1)[0]
#     else:
#         return random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1], k=1)[0]
#
# def get_duration_weeks(condition):
#     """Generate a treatment duration (in weeks) based on the condition type."""
#     if condition in ["Diabetes", "Hypertension", "Chronic Pain"]:
#         return random.randint(12, 52)
#     else:
#         return random.randint(4, 40)
#
# def generate_patient_record(patient_id):
#     """Generate a synthetic record for one patient, including additional features."""
#     # Basic attributes: health condition, therapy plan, progress, adjustments, etc.
#     condition = random.choices(health_conditions, weights=condition_probs, k=1)[0]
#     plans, plan_weights = therapy_plan_options[condition]
#     therapy_plan = random.choices(plans, weights=plan_weights, k=1)[0]
#     progress, adjustment = get_progress_and_adjustment()
#     follow_up = get_followup_status(progress)
#     sessions_per_week = get_sessions_per_week(therapy_plan)
#     duration_weeks = get_duration_weeks(condition)
#     age = get_age_for_condition(condition)
#     gender = random.choices(genders, weights=gender_probs, k=1)[0]
#
#     # Additional patient attributes
#     height_cm = random.randint(150, 200)
#     height_m = height_cm / 100.0
#     # Generate weight such that the derived BMI is plausible
#     # We'll simulate weight by choosing a BMI in a realistic range (18 to 35) and then computing weight.
#     bmi_target = random.uniform(18, 35)
#     weight_kg = round(bmi_target * (height_m ** 2), 1)
#     # Recalculate BMI precisely from generated height and weight
#     bmi = round(weight_kg / (height_m ** 2), 1)
#
#     injury_severity_score = random.randint(1, 10)
#     time_since_injury_months = random.randint(1, 120)
#
#     # The following fields can be linked to therapy parameters or patient status:
#     # For simplicity, we'll generate them independently using plausible ranges.
#     session_duration_minutes = random.randint(30, 120)
#     exercise_intensity = random.randint(1, 10)
#     patient_adherence_rate = round(random.uniform(0.5, 1.0), 2)
#     mobility_score = random.randint(1, 100)
#     grip_strength_kg = random.randint(5, 50)
#     muscle_strength_score = random.randint(1, 10)
#     pain_level = random.randint(1, 10)
#     balance_test_score = random.randint(1, 100)
#     quality_of_life_score = random.randint(1, 100)
#     mental_health_status = random.randint(1, 5)
#     sleep_quality_score = random.randint(1, 10)
#     fatigue_level = random.randint(1, 10)
#     daily_activity_score = random.randint(1, 100)
#     patient_engagement_score = random.randint(1, 100)
#
#     # Return the record as a dictionary
#     return {
#         "Patient ID": patient_id,
#         "Age": age,
#         "Gender": gender,
#         "Health Condition": condition,
#         "Therapy Plan": therapy_plan,
#         "Therapy Adjustment": adjustment,
#         "Number of Sessions per Week": sessions_per_week,
#         "Duration of Treatment (weeks)": duration_weeks,
#         "Patient Progress": progress,
#         "Follow-up Status": follow_up,
#         "Height (cm)": height_cm,
#         "Weight (kg)": weight_kg,
#         "BMI": bmi,
#         "Injury Severity Score": injury_severity_score,
#         "Time Since Injury (months)": time_since_injury_months,
#         "Session Duration (minutes)": session_duration_minutes,
#         "Exercise Intensity": exercise_intensity,
#         "Patient Adherence Rate": patient_adherence_rate,
#         "Mobility Score": mobility_score,
#         "Grip Strength (kg)": grip_strength_kg,
#         "Muscle Strength Score": muscle_strength_score,
#         "Pain Level": pain_level,
#         "Balance Test Score": balance_test_score,
#         "Quality of Life Score": quality_of_life_score,
#         "Mental Health Status": mental_health_status,
#         "Sleep Quality Score": sleep_quality_score,
#         "Fatigue Level": fatigue_level,
#         "Daily Activity Score": daily_activity_score,
#         "Patient Engagement Score": patient_engagement_score
#     }
#
# # ==========================
# # Generate Dataset and Create DataFrame
# # ==========================
# data = [generate_patient_record(patient_id) for patient_id in range(1, NUM_PATIENTS + 1)]
# df = pd.DataFrame(data)
#
# # Save DataFrame to CSV
# df.to_csv("synthetic_rehab_dataset.csv", index=False)
#
# print("Dataset has been generated and saved as 'synthetic_rehab_dataset.csv'.")
# print(df.head())