import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "Data/synthetic_rehab_data_70000.csv"
df = pd.read_csv(file_path)

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(df["recommended_therapy_adjustment"])

# Encode categorical features
categorical_cols = ["spinal_injury_type", "injury_cause", "rehabilitation_type", "use_of_assistive_devices"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for inverse transformation if needed

# Extract features
X = df.drop(columns=["recommended_therapy_adjustment"], errors="ignore")

# Scale numerical features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train/Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Optimized Gradient Boosting (LightGBM)
gb = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42)
gb.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred)

print(f"Optimized Gradient Boosting Accuracy: {gb_accuracy:.4f}")