# Auto-extracted from markdown file
# Source: underwriting_eligibility_waiting_periods_guaranteed_issue.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate employee cohorts with different health profiles
np.random.seed(42)

# Health distribution in hiring population
# Assume new hires have mix of healthy, moderately ill, severely ill
health_categories = ['Healthy', 'Moderate Condition', 'Severe Condition']
health_probs = [0.70, 0.20, 0.10]  # 70% healthy, 20% moderate, 10% severe
health_mortality = {
    'Healthy': 0.001,  # 0.1% annual mortality
    'Moderate Condition': 0.005,  # 0.5% annual
    'Severe Condition': 0.020,  # 2% annual
}

# Waiting period scenarios
waiting_periods = [0, 30, 60, 90]  # days
scenarios = {}

for waiting_days in waiting_periods:
    # Cohort size
    n_hires = 1000
    
    # Generate health status for new hires
    health_status = np.random.choice(health_categories, size=n_hires, p=health_probs)
    
    # Adverse selection: sick employees are more likely to enroll immediately
    # Model: If waiting = 0, sick are all enrolled; if waiting = 90, some drop out
    # Assumption: 80% of severe drop out after 90-day waiting; 20% of moderate drop out
    
    if waiting_days == 0:
        # Everyone enrolls (or all who would enroll, do so)
        enrollee_mask = np.ones(n_hires, dtype=bool)
    else:
        enrollee_mask = np.ones(n_hires, dtype=bool)
        # Attrition during waiting:
        for i, health in enumerate(health_status):
            if health == 'Severe Condition' and np.random.random() < 0.80:
                enrollee_mask[i] = False  # Employee drops out during waiting (quits, etc.)
            elif health == 'Moderate Condition' and np.random.random() < 0.20:
                enrollee_mask[i] = False
    
    enrolled_health = health_status[enrollee_mask]
    n_enrolled = enrollee_mask.sum()
    
    # Calculate expected annual claims (using average mortality * benefit)
    average_benefit = 100000
    expected_mortality = np.mean([health_mortality[h] for h in enrolled_health])
    annual_expected_claims = n_enrolled * expected_mortality * average_benefit
    
    # Premium calculation: claims + admin + profit margin
    admin_load = 0.15  # 15%
    profit_margin = 0.10  # 10%
    annual_premium = annual_expected_claims * (1 + admin_load + profit_margin)
    
    # Per-employee monthly cost
    monthly_per_employee = annual_premium / n_enrolled / 12
    
    # Enrollment rate
    enrollment_rate = n_enrolled / n_hires * 100
    
    scenarios[waiting_days] = {
        'n_enrolled': n_enrolled,
        'enrollment_rate': enrollment_rate,
        'expected_mortality': expected_mortality,
        'annual_expected_claims': annual_expected_claims,
        'annual_premium': annual_premium,
        'monthly_per_employee': monthly_per_employee,
        'enrolled_health': enrolled_health,
    }

# Create summary dataframe
summary_data = []
for waiting_days, scenario in scenarios.items():
    summary_data.append({
        'Waiting Period (days)': waiting_days,
        'Enrollment Rate (%)': scenario['enrollment_rate'],
        'Expected Mortality Rate': scenario['expected_mortality'],
        'Monthly Premium per Employee': scenario['monthly_per_employee'],
        'Annual Expected Claims': scenario['annual_expected_claims'],
    })

summary_df = pd.DataFrame(summary_data)
print("Adverse Selection Impact by Waiting Period:\n")
print(summary_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Enrollment rate vs waiting period
ax = axes[0, 0]
waiting = summary_df['Waiting Period (days)']
enrollment = summary_df['Enrollment Rate (%)']
ax.plot(waiting, enrollment, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('Enrollment Rate (%)')
ax.set_title('Enrollment Rate vs Waiting Period')
ax.grid(alpha=0.3)

# Plot 2: Expected mortality rate vs waiting period
ax = axes[0, 1]
mortality = summary_df['Expected Mortality Rate']
ax.plot(waiting, mortality, 'o-', linewidth=2, markersize=8, color='darkred')
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('Expected Mortality Rate')
ax.set_title('Population Mortality Risk vs Waiting Period')
ax.grid(alpha=0.3)

# Plot 3: Monthly premium vs waiting period
ax = axes[1, 0]
premium = summary_df['Monthly Premium per Employee']
ax.plot(waiting, premium, 'o-', linewidth=2, markersize=8, color='darkgreen')
ax.fill_between(waiting, premium, alpha=0.3, color='darkgreen')
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('Monthly Premium per Employee ($)')
ax.set_title('Premium Cost vs Waiting Period')
ax.grid(alpha=0.3)

# Plot 4: Health status distribution by waiting period
ax = axes[1, 1]
health_dist = []
for waiting_days in waiting_periods:
    enrolled = scenarios[waiting_days]['enrolled_health']
    healthy_pct = (enrolled == 'Healthy').mean() * 100
    moderate_pct = (enrolled == 'Moderate Condition').mean() * 100
    severe_pct = (enrolled == 'Severe Condition').mean() * 100
    health_dist.append({'Healthy': healthy_pct, 'Moderate': moderate_pct, 'Severe': severe_pct})

health_dist_df = pd.DataFrame(health_dist, index=waiting_periods)
health_dist_df.plot(kind='bar', stacked=True, ax=ax, color=['green', 'orange', 'red'], alpha=0.7)
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('% of Enrolled Employees')
ax.set_title('Health Status Mix of Enrolled Employees')
ax.legend(title='Health Status', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(waiting_periods, rotation=0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Key insights summary
print("\n\nKEY INSIGHTS:")
print(f"60-day waiting reduces annual premium by {((premium.iloc[0] - premium.iloc[2]) / premium.iloc[0] * 100):.1f}% vs 0-day")
print(f"90-day waiting reduces annual premium by {((premium.iloc[0] - premium.iloc[3]) / premium.iloc[0] * 100):.1f}% vs 0-day")
print(f"Enrollment drops from {enrollment.iloc[0]:.1f}% (0-day) to {enrollment.iloc[3]:.1f}% (90-day)")
print(f"Enrolled health improves: Severe condition risk drops from {(scenarios[0]['enrolled_health'] == 'Severe Condition').mean():.1%} to {(scenarios[90]['enrolled_health'] == 'Severe Condition').mean():.1%}")

