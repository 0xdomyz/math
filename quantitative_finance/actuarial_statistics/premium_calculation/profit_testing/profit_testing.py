# Auto-extracted from markdown file
# Source: profit_testing.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. SETUP
print("=" * 80)
print("PROFIT TESTING: COMPREHENSIVE FINANCIAL PROJECTION")
print("=" * 80)

# Mortality table
mortality_data = {
    35: 0.00103, 40: 0.00131, 45: 0.00172, 50: 0.00233, 55: 0.00325,
    60: 0.00459, 65: 0.00653, 70: 0.00933, 75: 0.01330, 80: 0.01934
}

# Assumptions
start_age = 40
contract_term = 20
gross_premium = 450
net_premium = 340
benefit_amount = 250000
initial_policies = 1000

# Expenses
acq_expense_y1 = gross_premium * 0.20  # 20% of Y1 premium
maintenance_annual = 30
renewal_commission_pct = 0.05  # 5% of annual premium
overhead_pct = 0.02  # 2% of premium

# Assumption rates
interest_rate = 0.04
lapse_rate = 0.05  # 5% annual lapse
cost_of_capital = 0.10  # 10% target ROE

print(f"\nProduct: 20-Year Term Insurance")
print(f"Assumptions:")
print(f"  Age at issue: {start_age}")
print(f"  Gross Premium: ${gross_premium:,.0f}/year")
print(f"  Net Premium: ${net_premium:,.0f}/year")
print(f"  Benefit: ${benefit_amount:,.0f}")
print(f"  Initial cohort: {initial_policies:,} policies")
print(f"  Interest: {interest_rate*100:.1f}%")
print(f"  Annual lapse: {lapse_rate*100:.1f}%\n")

# 2. PROFIT TEST TABLE
print("=" * 80)
print("ANNUAL PROFIT TEST TABLE")
print("=" * 80)

v = 1 / (1 + interest_rate)

profit_data = []
policies_inforce_beg = initial_policies
reserve_balance = 0
cumulative_profit = 0

print(f"\n{'Yr':<4} {'Age':<5} {'PIF':<8} {'Deaths':<8} {'Lapses':<8} {'Premium':<12} {'Claims':<12} {'Exp':<12} {'Profit':<12} {'PV Factor':<10} {'PV Profit':<12}")
print("-" * 125)

for year in range(1, contract_term + 1):
    age_now = start_age + year - 1
    qx_rate = mortality_data.get(age_now, 0.001)
    
    # Policy count during year
    policies_deaths = int(policies_inforce_beg * qx_rate)
    policies_lapses = int(policies_inforce_beg * lapse_rate * (1 - qx_rate))
    policies_end = policies_inforce_beg - policies_deaths - policies_lapses
    
    # Revenue
    premium_income = policies_inforce_beg * gross_premium
    investment_income = reserve_balance * interest_rate if reserve_balance > 0 else 0
    
    # Expenses
    if year == 1:
        acquisition = policies_inforce_beg * acq_expense_y1
    else:
        acquisition = 0
    
    maintenance = policies_inforce_beg * maintenance_annual
    commissions = policies_inforce_beg * gross_premium * renewal_commission_pct
    overhead = policies_inforce_beg * gross_premium * overhead_pct
    
    total_expenses = acquisition + maintenance + commissions + overhead
    
    # Claims
    claims_paid = policies_deaths * benefit_amount
    
    # Reserve changes
    reserve_contribution = (premium_income - claims_paid - total_expenses) * (1 + interest_rate)
    reserve_new = reserve_balance * (1 + interest_rate) + reserve_contribution - reserve_balance
    
    # Simplified reserve (approximate)
    reserve_new = max(0, reserve_balance + (policies_inforce_beg - policies_deaths - policies_lapses) * 15)
    
    # Profit calculation
    statutory_profit = premium_income + investment_income - claims_paid - total_expenses
    
    # Adjust for reserve changes (important for accounting)
    reserve_change = reserve_new - reserve_balance
    profit_after_reserve = statutory_profit - reserve_change
    
    # PV of profit
    pv_factor = v ** year
    pv_profit = profit_after_reserve * pv_factor
    
    cumulative_profit += pv_profit
    
    profit_data.append({
        'year': year,
        'age': age_now,
        'pif_beg': policies_inforce_beg,
        'deaths': policies_deaths,
        'lapses': policies_lapses,
        'pif_end': policies_end,
        'premium': premium_income,
        'claims': claims_paid,
        'expenses': total_expenses,
        'profit': profit_after_reserve,
        'pv_factor': pv_factor,
        'pv_profit': pv_profit,
        'cumulative_pv': cumulative_profit
    })
    
    if year in [1, 5, 10, 15, 20]:
        print(f"{year:<4} {age_now:<5} {policies_inforce_beg:<8} {policies_deaths:<8} {policies_lapses:<8} "
              f"${premium_income:<11,.0f} ${claims_paid:<11,.0f} ${total_expenses:<11,.0f} "
              f"${profit_after_reserve:<11,.0f} {pv_factor:<10.6f} ${pv_profit:<11,.0f}")
    
    policies_inforce_beg = policies_end
    reserve_balance = reserve_new
    
    if policies_inforce_beg == 0:
        break

print()

# 3. PROFITABILITY SUMMARY
print("=" * 80)
print("PROFITABILITY SUMMARY")
print("=" * 80)

total_premiums_pv = sum([d['premium'] * d['pv_factor'] for d in profit_data])
total_claims_pv = sum([d['claims'] * d['pv_factor'] for d in profit_data])
total_expenses_pv = sum([d['expenses'] * d['pv_factor'] for d in profit_data])
total_profit_pv = cumulative_profit

profit_margin = (total_profit_pv / total_premiums_pv) * 100 if total_premiums_pv > 0 else 0
roi_per_policy = total_profit_pv / initial_policies

print(f"\n{'Metric':<40} {'Amount':<20} {'% of Premium':<15}")
print("-" * 75)
print(f"{'PV of Premium Income':<40} ${total_premiums_pv:>18,.0f} {'100.0%':>14}")
print(f"{'PV of Claims':<40} ${total_claims_pv:>18,.0f} {(total_claims_pv/total_premiums_pv)*100:>13.1f}%")
print(f"{'PV of Expenses':<40} ${total_expenses_pv:>18,.0f} {(total_expenses_pv/total_premiums_pv)*100:>13.1f}%")
print(f"{'PV of Net Profit (NPV)':<40} ${total_profit_pv:>18,.0f} {profit_margin:>13.1f}%")
print()
print(f"Profit per policy (over {contract_term} years): ${roi_per_policy:,.2f}")
print()

# 4. IRR CALCULATION
print("=" * 80)
print("INTERNAL RATE OF RETURN (IRR) CALCULATION")
print("=" * 80)

def npv_function(rate):
    """NPV as function of discount rate"""
    npv_val = 0
    for year, d in enumerate(profit_data, 1):
        discount_factor = 1 / ((1 + rate) ** year)
        npv_val += d['profit'] * discount_factor
    return npv_val

# Solve for IRR
irr_estimate = fsolve(npv_function, 0.15)[0]

print(f"\nInternal Rate of Return (IRR): {irr_estimate*100:.2f}%")
print(f"Cost of Capital (Target): {cost_of_capital*100:.1f}%")
print(f"Spread: {(irr_estimate - cost_of_capital)*100:+.2f} percentage points")
print()

if irr_estimate > cost_of_capital:
    print("✓ Product exceeds profitability threshold")
else:
    print("✗ Product below profitability threshold")

print()

# 5. SENSITIVITY ANALYSIS
print("=" * 80)
print("SENSITIVITY ANALYSIS: IMPACT ON NPV & IRR")
print("=" * 80)

def calculate_npv_irr(mort_mult=1.0, interest_adj=0, lapse_adj=0):
    """Calculate NPV and IRR for scenario"""
    
    pif = initial_policies
    reserve = 0
    npv = 0
    
    for year in range(1, contract_term + 1):
        age = start_age + year - 1
        qx = mortality_data.get(age, 0.001) * mort_mult
        lapse = lapse_rate + lapse_adj
        
        deaths = int(pif * qx)
        lapses = int(pif * lapse * (1 - qx))
        pif_end = pif - deaths - lapses
        
        premium = pif * gross_premium
        claims = deaths * benefit_amount
        
        if year == 1:
            acq = pif * acq_expense_y1
        else:
            acq = 0
        
        maint = pif * maintenance_annual
        comm = pif * gross_premium * renewal_commission_pct
        overhead = pif * gross_premium * overhead_pct
        
        expenses = acq + maint + comm + overhead
        
        profit = premium - claims - expenses
        
        discount_factor = 1 / ((1 + interest_rate + interest_adj) ** year)
        npv += profit * discount_factor
        
        pif = pif_end
    
    # Calculate IRR
    def npv_func(rate):
        pif_irr = initial_policies
        npv_irr = 0
        
        for year in range(1, contract_term + 1):
            age = start_age + year - 1
            qx_irr = mortality_data.get(age, 0.001) * mort_mult
            lapse_irr = lapse_rate + lapse_adj
            
            deaths_irr = int(pif_irr * qx_irr)
            lapses_irr = int(pif_irr * lapse_irr * (1 - qx_irr))
            pif_irr_end = pif_irr - deaths_irr - lapses_irr
            
            premium_irr = pif_irr * gross_premium
            claims_irr = deaths_irr * benefit_amount
            
            if year == 1:
                acq_irr = pif_irr * acq_expense_y1
            else:
                acq_irr = 0
            
            maint_irr = pif_irr * maintenance_annual
            comm_irr = pif_irr * gross_premium * renewal_commission_pct
            overhead_irr = pif_irr * gross_premium * overhead_pct
            
            expenses_irr = acq_irr + maint_irr + comm_irr + overhead_irr
            profit_irr = premium_irr - claims_irr - expenses_irr
            
            discount_irr = 1 / ((1 + rate) ** year)
            npv_irr += profit_irr * discount_irr
            
            pif_irr = pif_irr_end
        
        return npv_irr
    
    irr_scenario = fsolve(npv_func, 0.15)[0]
    
    return npv, irr_scenario

print(f"\n{'Scenario':<35} {'NPV':<20} {'IRR':<15} {'vs Base':<15}")
print("-" * 85)

scenarios = {
    'Base Case': {'mort': 1.0, 'int': 0, 'lapse': 0},
    'Mortality +20%': {'mort': 1.20, 'int': 0, 'lapse': 0},
    'Mortality -20%': {'mort': 0.80, 'int': 0, 'lapse': 0},
    'Interest -1%': {'mort': 1.0, 'int': -0.01, 'lapse': 0},
    'Lapse +2%': {'mort': 1.0, 'int': 0, 'lapse': 0.02},
    'Lapse -2%': {'mort': 1.0, 'int': 0, 'lapse': -0.02},
    'Combined Stress': {'mort': 1.20, 'int': -0.01, 'lapse': 0.02},
}

base_npv, base_irr = calculate_npv_irr(1.0, 0, 0)

for scenario, params in scenarios.items():
    npv_s, irr_s = calculate_npv_irr(params['mort'], params['int'], params['lapse'])
    
    irr_spread = (irr_s - base_irr) * 100
    
    print(f"{scenario:<35} ${npv_s:>18,.0f} {irr_s*100:>13.2f}% {irr_spread:>+13.1f} bps")

print()

# 6. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Profit over time
ax = axes[0, 0]
years_plot = [d['year'] for d in profit_data]
profits_plot = [d['profit'] for d in profit_data]
pv_profits = [d['pv_profit'] for d in profit_data]

ax.bar(years_plot, profits_plot, alpha=0.5, label='Nominal Profit', color='steelblue', edgecolor='black')
ax.plot(years_plot, pv_profits, linewidth=2, marker='o', markersize=5, label='PV of Profit', color='darkblue')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Annual Profit Over Contract Term', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 2: Cumulative PV profit
ax = axes[0, 1]
cumulative_pv = [d['cumulative_pv'] for d in profit_data]

ax.plot(years_plot, cumulative_pv, linewidth=2.5, marker='o', markersize=6, color='green')
ax.fill_between(years_plot, 0, cumulative_pv, alpha=0.3, color='green')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Cumulative PV Profit ($)', fontsize=11)
ax.set_title('Cumulative Profitability Path', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3: Sensitivity tornado
ax = axes[1, 0]
scenario_names = list(scenarios.keys())
scenario_irrs = []

for scenario, params in scenarios.items():
    _, irr_s = calculate_npv_irr(params['mort'], params['int'], params['lapse'])
    scenario_irrs.append((irr_s - base_irr) * 100)

sorted_idx = np.argsort(scenario_irrs)
sorted_names = [scenario_names[i] for i in sorted_idx]
sorted_irrs = [scenario_irrs[i] for i in sorted_idx]

colors_tornado = ['red' if x < 0 else 'green' for x in sorted_irrs]

ax.barh(range(len(sorted_names)), sorted_irrs, color=colors_tornado, alpha=0.6, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=9)
ax.set_xlabel('IRR Change (bps)', fontsize=11)
ax.set_title('Sensitivity: IRR Impact', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Plot 4: Premium decomposition
ax = axes[1, 1]
premium_total_pv = total_premiums_pv
claims_pct = (total_claims_pv / premium_total_pv) * 100
expenses_pct = (total_expenses_pv / premium_total_pv) * 100
profit_pct = (total_profit_pv / premium_total_pv) * 100

components = ['Claims', 'Expenses', 'Profit']
amounts_pct = [claims_pct, expenses_pct, profit_pct]
colors_comp = ['red', 'orange', 'green']

bars = ax.bar(components, amounts_pct, color=colors_comp, alpha=0.6, edgecolor='black', linewidth=1.5)

for bar, pct in zip(bars, amounts_pct):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('% of Premium', fontsize=11)
ax.set_title('PV Premium Allocation', fontsize=12, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('profit_testing.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

