import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar

def calculate_breakeven_mortality(net_prem_given, pv_annuity_given, benefit_amount, 
                                  term_given, annual_rate_given):
    """Find mortality rate that yields given premium"""
    # This is complex; we'll estimate by testing mortality multiples
    
    mortality_test = {}
    for age in range(30, 70):
        if age in mortality_data:
            mortality_test[age] = mortality_data[age]
        else:
            mortality_test[age] = 0.001
    
    # Solve numerically
    def objective(mult):
        test_mort = {age: qx * mult for age, qx in mortality_test.items()}
        
        apv_test = 0
        kpx_test = 1.0
        
        for k in range(1, term_given + 1):
            age_test = 35 + k - 1
            qx_test = test_mort.get(age_test, 0.001)
            vk_test = (1 / (1 + annual_rate_given)) ** k
            apv_test += kpx_test * qx_test * vk_test * benefit_amount
            
            px_test = 1 - qx_test
            kpx_test *= px_test
        
        implied_premium = apv_test / pv_annuity_given
        return (implied_premium - net_prem_given) ** 2
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded')
    return result.x

print(f"\nGross Premium: ${results_standard['gross_annual']:,.2f}/year")
print(f"Net Premium (actuarial cost): ${results_standard['net_annual']:,.2f}/year")
print(f"Expense & Profit Component: ${results_standard['gross_annual'] - results_standard['net_annual']:,.2f}/year")
print()

# Margin analysis
expense_margin = results_standard['pv_expense']
profit_margin = results_standard['pv_profit']
total_margin = expense_margin + profit_margin

print(f"Expense margin (PV): ${expense_margin:,.2f}")
print(f"Profit margin (PV): ${profit_margin:,.2f}")
print(f"Total margin (PV): ${total_margin:,.2f}")
print(f"Margin as % of benefit: {total_margin / results_standard['apv_benefits'] * 100:.1f}%")
print()

# What if mortality worse than expected?
print("Sensitivity to Adverse Experience:")
print()

for mortality_increase in [0.10, 0.25, 0.50]:  # 10%, 25%, 50% worse
    mortality_stressed = {age: qx * (1 + mortality_increase) for age, qx in mortality_data.items()}
    
    results_stressed = calculate_term_premiums(
        35, benefit, 20, mortality_stressed, annual_rate,
        expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
        profit_pct=0  # No profit to absorb
    )
    
    inadequacy = results_stressed['net_annual'] - results_standard['net_annual']
    margin_cover_pct = results_standard['gross_annual'] / results_stressed['net_annual'] if results_stressed['net_annual'] > 0 else 0
    
    print(f"  Mortality {mortality_increase*100:>3.0f}% worse: Net premium required ${results_stressed['net_annual']:>10,.2f}")
    print(f"    Gross premium covers {margin_cover_pct*100:>5.1f}% of cost (margin adequacy)")

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Premium by age and term
ax = axes[0, 0]
ages_for_plot = list(range(25, 66, 5))
terms_for_plot = [10, 20, 30]

for term in terms_for_plot:
    premiums_by_age = []
    
    for age in ages_for_plot:
        mort_expanded = mortality_data.copy()
        for test_age in range(age, age + term):
            if test_age not in mort_expanded:
                if test_age > 65:
                    mu = 0.0001 * (1.075 ** test_age)
                    mort_expanded[test_age] = 1 - np.exp(-mu)
                else:
                    mort_expanded[test_age] = 0.001
        
        res = calculate_term_premiums(
            age, benefit, term, mort_expanded, annual_rate,
            expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
            profit_pct=profit_margin_pct
        )
        premiums_by_age.append(res['gross_monthly'])
    
    ax.plot(ages_for_plot, premiums_by_age, linewidth=2.5, marker='o', markersize=6, label=f'{term}-Year')

ax.set_xlabel('Starting Age', fontsize=11)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title(f'Term Insurance Premiums (${benefit/1000:.0f}K Benefit)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Smoker vs Standard vs Preferred
ax = axes[0, 1]
categories = ['Smoker', 'Standard', 'Preferred']
multipliers = [2.0, 1.0, 0.8]
colors_cat = ['red', 'blue', 'green']

premiums_cat = []
for mult in multipliers:
    res_cat = calculate_term_premiums(
        35, benefit, 20, mortality_data, annual_rate,
        smoker_mult=1.0, preferred_mult=mult,
        expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
        profit_pct=profit_margin_pct
    )
    premiums_cat.append(res_cat['gross_monthly'])

bars = ax.bar(categories, premiums_cat, color=colors_cat, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title('Underwriting Class Impact (Age 35, 20-Year Term)', fontsize=12, fontweight='bold')

for bar, prem in zip(bars, premiums_cat):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${prem:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Cash flow and profitability
ax = axes[1, 0]
years_cf_plot = np.arange(1, 11)
premiums_cf = []
expected_claims_cf = []
expenses_cf = []

for k in years_cf_plot:
    age_cf_plot = 35 + k - 1
    qx_cf_plot = mortality_data.get(age_cf_plot, 0.001)
    
    premiums_cf.append(annual_premium)
    expected_claims_cf.append(qx_cf_plot * benefit)
    
    exp_y1 = acquisition_expense_pct * annual_premium if k == 1 else 0
    expenses_cf.append(exp_y1 + maintenance_per_year)

ax.bar(years_cf_plot - 0.25, premiums_cf, width=0.5, label='Premium', alpha=0.6, color='green')
ax.bar(years_cf_plot + 0.25, expected_claims_cf, width=0.5, label='Expected Claims (per policy)', alpha=0.6, color='red')
ax.plot(years_cf_plot, expenses_cf, linewidth=2.5, marker='o', color='orange', label='Expenses', markersize=6)

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Annual Cash Flows (10-Year Term)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Benefit amount sensitivity
ax = axes[1, 1]
benefits_for_sens = np.array([100, 250, 500, 750, 1000]) * 1000
premiums_by_benefit = []

for ben in benefits_for_sens:
    res_ben = calculate_term_premiums(
        35, ben, 20, mortality_data, annual_rate,
        expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
        profit_pct=profit_margin_pct
    )
    premiums_by_benefit.append(res_ben['gross_monthly'])

ax.plot(benefits_for_sens / 1000, premiums_by_benefit, linewidth=2.5, marker='s', markersize=7, color='purple')
ax.fill_between(benefits_for_sens / 1000, 0, premiums_by_benefit, alpha=0.2, color='purple')

ax.set_xlabel('Benefit Amount ($000s)', fontsize=11)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title('Premium vs Benefit Amount (Age 35, 20-Year)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('term_insurance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

