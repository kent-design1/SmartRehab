import matplotlib.pyplot as plt

# Week and SCIM data
weeks = [0, 6, 12, 18, 24]
total_scim = [42.0, 53.1, 67.2, 77.1, 85.4]
selfcare = [13.0, 20.4, 25.7, 29.6, 34.1]
respiration = [5.0, 9.4, 14.0, 17.1, 18.2]
mobility = [16.0, 23.3, 27.5, 30.4, 33.1]

# Calculate SCIM gains between 6-week intervals
gains = [total_scim[i] - total_scim[i-1] for i in range(1, len(total_scim))]

# Plot
plt.figure(figsize=(10, 6))

# Line styles (grayscale-friendly)
plt.plot(weeks, total_scim, marker='o', color='black', label="Total SCIM", linewidth=2)
plt.plot(weeks, selfcare, marker='s', linestyle='--', color='dimgray', label="Self-Care")
plt.plot(weeks, respiration, marker='^', linestyle='-.', color='gray', label="Respiration")
plt.plot(weeks, mobility, marker='d', linestyle=':', color='darkgray', label="Mobility")

# Highlight low-gain intervals using hatching
for i, (wk1, wk2, gain) in enumerate(zip(weeks[:-1], weeks[1:], gains)):
    if gain < 5:
        plt.axvspan(wk1, wk2, facecolor='none', hatch='////', edgecolor='gray', linewidth=0)
        plt.text((wk1 + wk2) / 2, total_scim[i] + 1,
                 f"Low gain: {gain:.1f}", ha='center', va='bottom', fontsize=9, color='black')

# Labels and layout
plt.title("Patient SCIM Trajectory with Heuristic Triggers", fontsize=14)
plt.xlabel("Week", fontsize=12)
plt.ylabel("SCIM Score", fontsize=12)
plt.xticks(weeks)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig("scim_trajectory_with_bumps.png", dpi=300)
plt.show()