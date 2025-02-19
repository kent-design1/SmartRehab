# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = "synthetic_data_spz.csv"  # Ensure the correct file path
df = pd.read_csv(file_path)

# Step 1: Handle Missing Values
df = df.dropna(subset=['SCIM_admission'])  # Drop rows with missing SCIM_admission

# Step 2: Convert Categorical Variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Convert 'sex' to numerical
df['etiology'] = df['etiology'].map({'accident': 0, 'sickness': 1})  # Convert 'etiology'

# Step 3: Check for Duplicate Records
duplicates = df.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")  # Should be 0

# Step 4: Data Summary
print("\nSummary Statistics:\n", df.describe())

# Step 5: Visualizing Data Distributions
plt.figure(figsize=(12, 5))

# Age Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')

# SCIM Admission Score Distribution
plt.subplot(1, 2, 2)
sns.histplot(df['SCIM_admission'], bins=20, kde=True, color='green')
plt.title('SCIM Admission Score Distribution')

plt.show()

# Step 6: Correlation Heatmap
plt.figure(figsize=(15, 10))
corr_matrix = df.corr(numeric_only=True)  # Ensure numeric-only columns
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Scatterplot - Therapy Time vs Recovery Score
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['mean_physt'], y=df['SCIM_admission'], alpha=0.6)
plt.xlabel('Mean Physiotherapy Minutes')
plt.ylabel('SCIM Admission Score')
plt.title('Physiotherapy Time vs Recovery Score')
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_data_spz.csv", index=False)

print("\n✅ Data Cleaning & Visualization Completed. Cleaned dataset saved as 'cleaned_data_spz.csv'.")