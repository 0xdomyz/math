# Auto-extracted from markdown file
# Source: elimination_period_waiting_period.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulated population: 1000 employees, 1 year
np.random.seed(42)
n_employees = 1000
claims_data = []

# Generate claim events: rate ~0.5 per employee per year
for emp_id in range(n_employees):
    # Poisson process: average 0.5 claims/year
    n_claims = np.random.poisson(0.5)
    
    for claim_idx in range(n_claims):
        # Random claim start date
        day_of_year = np.random.randint(1, 366)
        
        # Duration (log-normal: most short, some long)
        duration = np.random.lognormal(mean=2, sigma=1.2)  # ~7 days median
        duration = max(1, int(duration))
        
        claims_data.append({
            'emp_id': emp_id,
            'start_day': day_of_year,
            'duration_days': duration
        })

df = pd.DataFrame(claims_data)

# Calculate claims paid under different elimination periods
elimination_periods = [0, 7, 14, 21]
results = []

for elim_days in elimination_periods:
    # Days benefit paid = max(0, duration - elimination)
    df['days_paid'] = df['duration_days'].apply(
        lambda d: max(0, d - elim_days)
    )
    
    # Count non-zero claims (those that trigger any payment)
    claims_paid = (df['days_paid'] > 0).sum()
    
    # Total benefit days (proxy for cost)
    total_benefit_days = df['days_paid'].sum()
    
    results.append({
        'Elimination Days': elim_days,
        'Claims Paid': claims_paid,
        'Total Benefit Days': total_benefit_days,
        'Avg Days per Claim': total_benefit_days / max(1, claims_paid)
    })

results_df = pd.DataFrame(results)
print("Elimination Period Impact on Claims:\n")
print(results_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Claims paid vs elimination period
axes[0].bar(results_df['Elimination Days'], results_df['Claims Paid'], 
            color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Elimination Period (days)')
axes[0].set_ylabel('Number of Claims Paid')
axes[0].set_title('Claims Frequency vs Elimination Period')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Total benefit days vs elimination period
axes[1].plot(results_df['Elimination Days'], results_df['Total Benefit Days'], 
             'o-', linewidth=2, markersize=8, color='darkred')
axes[1].set_xlabel('Elimination Period (days)')
axes[1].set_ylabel('Total Benefit Days Paid')
axes[1].set_title('Total Claims Cost vs Elimination Period')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity: show claim duration distribution
print("\n\nClaim Duration Distribution:")
print(df['duration_days'].describe())
print(f"Median duration: {df['duration_days'].median():.1f} days")
print(f"% claims < 7 days: {(df['duration_days'] < 7).sum() / len(df) * 100:.1f}%")
print(f"% claims < 14 days: {(df['duration_days'] < 14).sum() / len(df) * 100:.1f}%")

