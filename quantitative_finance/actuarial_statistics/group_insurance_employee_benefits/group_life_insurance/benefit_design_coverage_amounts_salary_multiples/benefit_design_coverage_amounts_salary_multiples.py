# Auto-extracted from markdown file
# Source: benefit_design_coverage_amounts_salary_multiples.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Company profiles
companies = {
    'Startup (Tech)': {'n': 150, 'salary_mean': 95000, 'salary_std': 35000},
    'Fortune 500': {'n': 5000, 'salary_mean': 65000, 'salary_std': 25000},
    'Manufacturing': {'n': 800, 'salary_mean': 45000, 'salary_std': 12000},
    'Professional Services': {'n': 250, 'salary_mean': 85000, 'salary_std': 40000},
}

designs = {
    '1x Salary': lambda s: s * 1,
    '2x Salary': lambda s: s * 2,
    '3x Salary': lambda s: s * 3,
    'Fixed $100K': lambda s: 100000,
}

# Generate salary distributions (lognormal is realistic)
np.random.seed(42)
salary_data = {}

for company, params in companies.items():
    n = params['n']
    mean = params['salary_mean']
    std = params['salary_std']
    
    # Fit lognormal (more realistic than normal for salary)
    sigma = np.sqrt(np.log(1 + (std/mean)**2))
    mu = np.log(mean) - sigma**2 / 2
    
    salaries = np.random.lognormal(mu, sigma, n)
    salary_data[company] = salaries

# Calculate benefit costs and statistics
results = []

for company, salaries in salary_data.items():
    for design_name, design_func in designs.items():
        benefits = np.array([design_func(s) for s in salaries])
        
        # Mortality assumption: simplified (age-averaged)
        mortality_rate = 0.003  # 0.3% annual mortality (group average)
        annual_claim_cost = (benefits * mortality_rate).sum()
        
        # Admin load
        admin_load = 0.12  # 12% administrative cost
        total_premium = annual_claim_cost * (1 + admin_load)
        
        monthly_per_employee = total_premium / len(salaries) / 12
        
        results.append({
            'Company': company,
            'Design': design_name,
            'Avg Benefit': benefits.mean(),
            'Max Benefit': benefits.max(),
            'Min Benefit': benefits.min(),
            'Std Dev Benefits': benefits.std(),
            'Annual Premium': total_premium,
            'Monthly per Employee': monthly_per_employee,
            'Premium as % of Payroll': (total_premium / salaries.sum()) * 100,
        })

results_df = pd.DataFrame(results)

# Display
print("BENEFIT DESIGN COMPARISON BY COMPANY\n")
for company in companies.keys():
    print(f"\n{company}:")
    subset = results_df[results_df['Company'] == company]
    print(subset[['Design', 'Avg Benefit', 'Monthly per Employee', 'Premium as % of Payroll']].to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Average benefit by design and company
pivot_avg = results_df.pivot_table(values='Avg Benefit', index='Company', columns='Design')
pivot_avg.plot(kind='bar', ax=axes[0, 0], color=['steelblue', 'orange', 'green', 'red'], alpha=0.7)
axes[0, 0].set_title('Average Benefit by Company & Design')
axes[0, 0].set_ylabel('Benefit ($)')
axes[0, 0].legend(title='Design', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Monthly cost per employee
pivot_cost = results_df.pivot_table(values='Monthly per Employee', index='Company', columns='Design')
pivot_cost.plot(kind='bar', ax=axes[0, 1], color=['steelblue', 'orange', 'green', 'red'], alpha=0.7)
axes[0, 1].set_title('Monthly Cost per Employee')
axes[0, 1].set_ylabel('Cost ($)')
axes[0, 1].legend(title='Design', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Salary distribution for Startup
ax = axes[1, 0]
salaries_startup = salary_data['Startup (Tech)']
ax.hist(salaries_startup, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(salaries_startup.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${salaries_startup.mean():.0f}')
ax.set_xlabel('Salary ($)')
ax.set_ylabel('Number of Employees')
ax.set_title('Salary Distribution: Startup (Tech)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Benefit vs salary scatter (Startup, all designs)
ax = axes[1, 1]
colors_scatter = {'1x Salary': 'steelblue', '2x Salary': 'orange', '3x Salary': 'green', 'Fixed $100K': 'red'}
for design in designs.keys():
    benefits = np.array([designs[design](s) for s in salaries_startup])
    ax.scatter(salaries_startup, benefits, alpha=0.5, s=30, label=design, color=colors_scatter[design])

ax.set_xlabel('Salary ($)')
ax.set_ylabel('Death Benefit ($)')
ax.set_title('Benefit vs Salary: Startup (Tech)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Adequacy analysis: % of employees with benefit < 3× salary (80% replacement)
print("\n\nADEQUACY ANALYSIS (% with benefit < 3× salary):")
for company, salaries in salary_data.items():
    for design_name, design_func in designs.items():
        benefits = np.array([design_func(s) for s in salaries])
        target = salaries * 3
        pct_adequate = (benefits >= target).mean() * 100
        results_df.loc[(results_df['Company'] == company) & (results_df['Design'] == design_name), 'Adequacy %'] = pct_adequate

# Summary table
adequacy_pivot = results_df.pivot_table(values='Adequacy %', index='Company', columns='Design')
print(adequacy_pivot.to_string())

