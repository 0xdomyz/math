# Auto-extracted from markdown file
# Source: benefit_percentage_replacement_ratio.md

# --- Code Block 1 ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Empirical observation: Claim duration increases with benefit percentage
# Model: Duration = base_duration * exp(elasticity * (replacement% - baseline%))

# Historical data from actuarial studies
replacement_ratios = np.array([50, 55, 60, 65, 66.67, 70])
avg_claim_duration = np.array([18, 20, 23, 25, 26, 28])  # days (illustrative)

# Fit exponential model
def duration_model(pct, base_duration, elasticity):
    """Duration increases exponentially with replacement percentage"""
    baseline_pct = 60  # Reference point
    return base_duration * np.exp(elasticity * (pct - baseline_pct) / 100)

# Fit to data
popt, pcov = curve_fit(duration_model, replacement_ratios, avg_claim_duration, 
                       p0=[23, 0.1], maxfev=5000)
base_duration, elasticity = popt

print(f"Model Fit:")
print(f"  Base duration at 60%: {base_duration:.2f} days")
print(f"  Elasticity: {elasticity:.4f} (% change per 1% increase in benefit%)")

# Predict across range
pct_range = np.linspace(40, 75, 50)
duration_predicted = duration_model(pct_range, base_duration, elasticity)

# Cost projection: assume daily cost = average benefit amount
# Higher % → higher daily cost AND longer duration (compounding effect)
daily_benefit = 100  # Illustrative
total_cost = (replacement_ratios / 100) * daily_benefit * avg_claim_duration

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Duration vs Replacement Ratio
axes[0].scatter(replacement_ratios, avg_claim_duration, s=100, 
                alpha=0.7, color='steelblue', label='Observed')
axes[0].plot(pct_range, duration_predicted, 'r-', linewidth=2, label='Fitted Model')
axes[0].axvline(60, color='gray', linestyle='--', alpha=0.5, label='Standard 60%')
axes[0].set_xlabel('Replacement Ratio (%)')
axes[0].set_ylabel('Average Claim Duration (days)')
axes[0].set_title('Claim Duration vs Benefit Percentage')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Total Expected Cost (Duration × Benefit)
daily_benefit_range = (pct_range / 100) * 200  # Daily benefit increases with %
total_cost_range = daily_benefit_range * duration_model(pct_range, base_duration, elasticity)

axes[1].plot(pct_range, total_cost_range, 'o-', linewidth=2, 
             markersize=6, color='darkred')
axes[1].set_xlabel('Replacement Ratio (%)')
axes[1].set_ylabel('Total Expected Cost per Claim ($)')
axes[1].set_title('Total Claim Cost: Duration × Benefit')
axes[1].grid(alpha=0.3)

# Plot 3: Cost difference vs 60% baseline
baseline_cost = total_cost_range[np.argmin(np.abs(pct_range - 60))]
cost_increase = ((total_cost_range - baseline_cost) / baseline_cost) * 100

axes[2].fill_between(pct_range, 0, cost_increase, alpha=0.5, color='orange')
axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[2].axvline(60, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Replacement Ratio (%)')
axes[2].set_ylabel('% Increase in Total Cost vs 60%')
axes[2].set_title('Cost Sensitivity to Replacement Ratio')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity table
print("\n\nCost Sensitivity Analysis:")
print("Replacement % | Avg Duration (days) | Daily Benefit | Total Cost | % vs 60%")
print("-" * 75)
for pct in [50, 55, 60, 65, 66.67, 70, 75]:
    dur = duration_model(pct, base_duration, elasticity)
    daily = (pct / 100) * 200
    total = daily * dur
    pct_diff = ((total - baseline_cost) / baseline_cost) * 100
    print(f"{pct:6.1f}% | {dur:18.1f} | ${daily:11.2f} | ${total:9.2f} | {pct_diff:+6.1f}%")

