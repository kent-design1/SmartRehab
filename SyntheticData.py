import pandas as pd
import numpy as np

# Define the number of synthetic records
num_synthetic_patients = 70000

# Generate synthetic numerical features
synthetic_data = {
    "age": np.random.randint(18, 80, num_synthetic_patients),
    "height_cm": np.random.randint(150, 200, num_synthetic_patients),
    "weight_kg": np.random.randint(50, 120, num_synthetic_patients),
    "bmi": np.round(np.random.uniform(18, 35, num_synthetic_patients), 1),
    "injury_severity_score": np.random.randint(1, 10, num_synthetic_patients),
    "time_since_injury_months": np.random.randint(1, 120, num_synthetic_patients),
    "session_frequency_per_week": np.random.randint(1, 5, num_synthetic_patients),
    "session_duration_minutes": np.random.randint(30, 120, num_synthetic_patients),
    "exercise_intensity": np.random.randint(1, 10, num_synthetic_patients),
    "patient_adherence_rate": np.round(np.random.uniform(0.5, 1.0, num_synthetic_patients), 2),
    "mobility_score": np.random.randint(1, 100, num_synthetic_patients),
    "grip_strength_kg": np.random.randint(5, 50, num_synthetic_patients),
    "muscle_strength_score": np.random.randint(1, 10, num_synthetic_patients),
    "pain_level": np.random.randint(1, 10, num_synthetic_patients),
    "balance_test_score": np.random.randint(1, 100, num_synthetic_patients),
    "quality_of_life_score": np.random.randint(1, 100, num_synthetic_patients),
    "mental_health_status": np.random.randint(1, 5, num_synthetic_patients),
    "sleep_quality_score": np.random.randint(1, 10, num_synthetic_patients),
    "fatigue_level": np.random.randint(1, 10, num_synthetic_patients),
    "daily_activity_score": np.random.randint(1, 100, num_synthetic_patients),
    "patient_engagement_score": np.random.randint(1, 100, num_synthetic_patients)
}

# Generate categorical features
synthetic_data["spinal_injury_type"] = np.random.choice(["Cervical", "Thoracic", "Lumbar", "Sacral"], num_synthetic_patients)
synthetic_data["injury_cause"] = np.random.choice(["Car Accident", "Fall", "Sports Injury", "Work Accident", "Other"], num_synthetic_patients)
synthetic_data["rehabilitation_type"] = np.random.choice(["Physical Therapy", "Occupational Therapy", "Hydrotherapy", "Electrotherapy"], num_synthetic_patients)
synthetic_data["use_of_assistive_devices"] = np.random.choice(["Yes", "No"], num_synthetic_patients)

# Generate the target variable
synthetic_data["recommended_therapy_adjustment"] = np.random.choice(
    ["Maintain Plan", "Change Therapy", "Reduce Intensity", "Increase Frequency"],
    num_synthetic_patients,
    p=[0.4, 0.2, 0.2, 0.2]  # Assuming "Maintain Plan" is more common
)

# Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Save dataset to CSV
synthetic_df.to_csv("synthetic_rehab_data_70000.csv", index=False)

print("✅ Synthetic dataset with 70,000 records has been generated and saved as 'synthetic_rehab_data_70000.csv'")