import matplotlib.pyplot as plt
import pandas as pd

# Load the synthetic dataset
df = pd.read_csv("synthetic_rehab_dataset.csv")

# Use an appropriate column (adjust if your dataset names differ)
baseline_column = "Baseline SCIM" if "Baseline SCIM" in df.columns else "SCIM Final"

plt.figure(figsize=(8, 6))
plt.hist(df[baseline_column], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Baseline SCIM Score")
plt.ylabel("Frequency")
plt.title("Distribution of Baseline SCIM Scores")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save the histogram as an image
plt.savefig("baseline_scim_distribution.png")
plt.show()