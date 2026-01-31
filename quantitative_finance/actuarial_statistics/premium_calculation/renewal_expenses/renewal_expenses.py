# Auto-extracted from markdown file
# Source: renewal_expenses.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP
print("=" * 80)
print("RENEWAL EXPENSES & COST OF PERSISTENCY")
print("=" * 80)

# Expense components (annual per policy)
expense_structure = {
    'direct': {
        'policy_admin': 5,
        'billing': 2,
        'customer_service': 3,
        'compliance': 1,
        'systems': 2,
        'total_fixed': 13,
        'renewal_commission_pct': 0.01,  # 1% of renewal premium
        'overhead_allocation_pct': 0.02   # 2% of premium
    },
    'agent': {
        'policy_admin': 8,
        'billing': 4,
        'customer_service': 8,
        'compliance': 2,
        'systems': 3,
        'total_fixed': 25,
        'renewal_commission_pct': 0.06,  # 6% of renewal premium
        'overhead_allocation_pct': 0.03   # 3% of premium
    },
    'bank': {
        'policy_admin': 6,
        'billing': 2,
        'customer_service': 5,
        'compliance': 1.5,
        'systems': 2.5,
        'total_fixed': 17,
        'renewal_commission_pct': 0.04,  # 4% of renewal premium
        'overhead_allocation_pct': 0.02   # 2% of premium
    }
}

interest_rate = 0.04
gross_premium = 400  # Annual
contract_term = 20
lapse_rate_annual = 0.05  # 5% annual lapse

print(f"\nAssumptions:")
print(f"  Gross Premium: ${gross_premium:,.2f}/year")
print(f"  Contract Term: {contract_term} years")
print(f"  Annual Lapse Rate: {lapse_rate_annual*100:.1f}%")
print(f"  Interest Rate: {interest_rate*100:.1f}%\n")

# 2. RENEWAL EXPENSE STRUCTURE
print("=" * 80)
print("RENEWAL EXPENSE COMPONENTS BY DISTRIBUTION CHANNEL")
print("=" * 80)

for channel, expenses in expense_structure.items():
    print(f"\n{channel.upper()} Distribution:\n")
    print(f"{'Component':<35} {'Amount/Rate':<20}")
    print("-" * 55)
    
    for component, amount in expenses.items():
        if 'pct' in component:
            print(f"{component:<35} {amount*100:>18.1f}%")
        else:
            print(f"{component:<35} ${amount:>18,.0f}")
    
    total_fixed = expenses['total_fixed']
    renewal_comm_pct = expenses['renewal_commission_pct']
    renewal_comm_annual = gross_premium * renewal_comm_pct
    overhead = gross_premium * expenses['overhead_allocation_pct']
    
    total_annual = total_fixed + renewal_comm_annual + overhead
    total_pct_premium = (total_annual / gross_premium) * 100
    
    print(f"\n  Renewal Commission ({renewal_comm_pct*100:.1f}%): ${renewal_comm_annual:,.2f}")
    print(f"  Overhead allocation ({expenses['overhead_allocation_pct']*100:.1f}%): ${overhead:,.2f}")
    print(f"  Total annual renewal expense: ${total_annual:,.2f}")
    print(f"  As % of premium: {total_pct_premium:.1f}%\n")

print()

# 3. RENEWAL EXPENSE PV ANALYSIS
print("=" * 80)
print("PRESENT VALUE OF RENEWAL EXPENSES (DIRECT CHANNEL)")
print("=" * 80)

v = 1 / (1 + interest_rate)
channel = 'direct'
expenses = expense_structure[channel]

total_fixed = expenses['total_fixed']
renewal_comm_pct = expenses['renewal_commission_pct']
overhead_pct = expenses['overhead_allocation_pct']

pv_expenses = 0
annual_lapse = 0  # Cumulative

print(f"\n{'Year':<8} {'Renewal Prob':<15} {'Fixed Cost':<15} {'Commission':<15} {'Overhead':<15} {'Total Cost':<15} {'PV Factor':<12} {'PV Cost':<15}")
print("-" * 120)

policy_survival = 1.0

for year in range(1, contract_term + 1):
    # Probability of policy surviving to this year
    policy_survival *= (1 - lapse_rate_annual)
    
    # Renewal expense components
    fixed_cost = total_fixed
    commission = gross_premium * renewal_comm_pct
    overhead = gross_premium * overhead_pct
    total_cost = fixed_cost + commission + overhead
    
    # Discount to present
    v_factor = v ** year
    pv_cost = total_cost * policy_survival * v_factor
    pv_expenses += pv_cost
    
    if year in [1, 5, 10, 15, 20]:
        print(f"{year:<8} {policy_survival:<15.4f} ${fixed_cost:<14,.0f} ${commission:<14,.0f} ${overhead:<14,.0f} ${total_cost:<14,.0f} {v_factor:<12.6f} ${pv_cost:<14,.0f}")

print(f"\nTotal PV of Renewal Expenses: ${pv_expenses:,.2f}\n")

# 4. EXPENSE RATIO OVER LIFETIME
print("=" * 80)
print("RENEWAL EXPENSE RATIO: % OF ANNUAL PREMIUM")
print("=" * 80)

print(f"\nDirect Channel: Annual Breakdown\n")
print(f"{'Year':<8} {'Premium':<15} {'Total Exp':<15} {'Exp Ratio':<15} {'Surviv Prob':<15}")
print("-" * 68)

policy_survival = 1.0

for year in range(1, contract_term + 1):
    policy_survival *= (1 - lapse_rate_annual)
    
    total_exp = total_fixed + (gross_premium * renewal_comm_pct) + (gross_premium * overhead_pct)
    exp_ratio = (total_exp / gross_premium) * 100
    
    if year in [1, 5, 10, 15, 20]:
        print(f"{year:<8} ${gross_premium:<14,.0f} ${total_exp:<14,.0f} {exp_ratio:<14.1f}% {policy_survival:<14.1f}%")

print()

# 5. CHANNEL COMPARISON
print("=" * 80)
print("CHANNEL COMPARISON: EXPENSE RATIOS & PV")
print("=" * 80)

print(f"\n{'Channel':<20} {'Annual Expense':<20} {'% of Premium':<18} {'PV Expenses':<20} {'Impact on Load':<15}")
print("-" * 93)

pv_results = {}

for channel_name, expenses in expense_structure.items():
    total_fixed = expenses['total_fixed']
    renewal_comm = gross_premium * expenses['renewal_commission_pct']
    overhead = gross_premium * expenses['overhead_allocation_pct']
    
    total_annual = total_fixed + renewal_comm + overhead
    exp_pct = (total_annual / gross_premium) * 100
    
    # Calculate PV
    pv_exp = 0
    survival = 1.0
    
    for year in range(1, contract_term + 1):
        survival *= (1 - lapse_rate_annual)
        v_factor = v ** year
        pv_exp += total_annual * survival * v_factor
    
    pv_results[channel_name] = pv_exp
    
    load_impact = (pv_exp / contract_term) / 1.0  # Approximate
    
    print(f"{channel_name:<20} ${total_annual:<19,.0f} {exp_pct:<17.1f}% ${pv_exp:<19,.0f} ${load_impact:<14,.0f}")

print()

# 6. LAPSE CORRELATION IMPACT
print("=" * 80)
print("LAPSE CORRELATION: IMPACT ON EXPENSE RECOVERY")
print("=" * 80)

print(f"\nDirect Channel - PV of Renewal Expenses by Lapse Rate:\n")

lapse_scenarios = [0.00, 0.03, 0.05, 0.07, 0.10, 0.15]
expenses_direct = expense_structure['direct']

print(f"{'Lapse Rate':<15} {'PV Expenses':<20} {'% of Base':<15} {'Impact':<20}")
print("-" * 70)

base_pv_exp = 0
for lapse_rate in [0.05]:
    pv_exp = 0
    survival = 1.0
    
    for year in range(1, contract_term + 1):
        survival *= (1 - lapse_rate)
        v_factor = v ** year
        total_exp = expenses_direct['total_fixed'] + (gross_premium * expenses_direct['renewal_commission_pct']) + (gross_premium * expenses_direct['overhead_allocation_pct'])
        pv_exp += total_exp * survival * v_factor
    
    base_pv_exp = pv_exp

for lapse_rate in lapse_scenarios:
    pv_exp = 0
    survival = 1.0
    
    for year in range(1, contract_term + 1):
        survival *= (1 - lapse_rate)
        v_factor = v ** year
        total_exp = expenses_direct['total_fixed'] + (gross_premium * expenses_direct['renewal_commission_pct']) + (gross_premium * expenses_direct['overhead_allocation_pct'])
        pv_exp += total_exp * survival * v_factor
    
    pct_of_base = (pv_exp / base_pv_exp) * 100 if base_pv_exp > 0 else 100
    impact = "Higher lapse" if lapse_rate > 0.05 else "Lower lapse"
    
    print(f"{lapse_rate*100:>13.1f}% ${pv_exp:>18,.0f} {pct_of_base:>13.1f}% {impact:>20}")

print()

# 7. EXPENSE INFLATION SCENARIO
print("=" * 80)
print("INFLATION IMPACT: EXPENSE GROWTH OVER TIME")
print("=" * 80)

inflation_rate = 0.03
print(f"\nAssuming {inflation_rate*100:.1f}% Annual Expense Inflation\n")

print(f"{'Year':<8} {'Fixed Cost':<15} {'Commission':<15} {'Total Expense':<15} {'Actual vs Assumed':<20}")
print("-" * 73)

assumed_annual_exp = total_fixed + (gross_premium * renewal_comm_pct) + (gross_premium * overhead_pct)
actual_annual_exp = assumed_annual_exp

for year in range(1, contract_term + 1):
    actual_annual_exp = assumed_annual_exp * ((1 + inflation_rate) ** year)
    
    difference = actual_annual_exp - assumed_annual_exp
    difference_pct = (difference / assumed_annual_exp) * 100
    
    if year in [1, 5, 10, 15, 20]:
        print(f"{year:<8} ${actual_annual_exp * 0.60:<14,.0f} ${actual_annual_exp * 0.25:<14,.0f} ${actual_annual_exp:<14,.0f} {difference_pct:>+18.1f}%")

print()

# 8. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Channel expense comparison
ax = axes[0, 0]
channels_plot = list(expense_structure.keys())
fixed_costs = []
commission_costs = []
overhead_costs = []

for channel_name in channels_plot:
    expenses = expense_structure[channel_name]
    fixed_costs.append(expenses['total_fixed'])
    commission_costs.append(gross_premium * expenses['renewal_commission_pct'])
    overhead_costs.append(gross_premium * expenses['overhead_allocation_pct'])

x_pos = np.arange(len(channels_plot))
width = 0.25

ax.bar(x_pos - width, fixed_costs, width, label='Fixed Cost', alpha=0.6, edgecolor='black')
ax.bar(x_pos, commission_costs, width, label='Renewal Commission', alpha=0.6, edgecolor='black')
ax.bar(x_pos + width, overhead_costs, width, label='Overhead Alloc', alpha=0.6, edgecolor='black')

ax.set_ylabel('Annual Expense ($)', fontsize=11)
ax.set_title('Renewal Expense Components by Channel', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([c.capitalize() for c in channels_plot], fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Plot 2: PV of expenses by lapse
ax = axes[0, 1]
lapse_rates_plot = np.linspace(0, 0.15, 16)
pv_expenses_plot = []

for lapse_rate in lapse_rates_plot:
    pv_exp = 0
    survival = 1.0
    
    for year in range(1, contract_term + 1):
        survival *= (1 - lapse_rate)
        v_factor = v ** year
        total_exp = expenses_direct['total_fixed'] + (gross_premium * expenses_direct['renewal_commission_pct']) + (gross_premium * expenses_direct['overhead_allocation_pct'])
        pv_exp += total_exp * survival * v_factor
    
    pv_expenses_plot.append(pv_exp)

ax.plot(lapse_rates_plot * 100, pv_expenses_plot, linewidth=2.5, marker='o', markersize=5, color='steelblue')
ax.axvline(x=5, color='red', linestyle='--', linewidth=1.5, label='Base Assumption (5%)', alpha=0.7)

ax.set_xlabel('Annual Lapse Rate (%)', fontsize=11)
ax.set_ylabel('PV of Renewal Expenses ($)', fontsize=11)
ax.set_title('Renewal Expense Sensitivity to Lapse', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Expense ratio over years
ax = axes[1, 0]
years_plot = np.arange(1, contract_term + 1)
survival_plot = []
survival = 1.0

for year in years_plot:
    survival *= (1 - lapse_rate_annual)
    survival_plot.append(survival * 100)

ax.plot(years_plot, survival_plot, linewidth=2.5, marker='o', markersize=5, color='darkgreen')
ax.fill_between(years_plot, 0, survival_plot, alpha=0.3, color='green')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Policy Survival Rate (%)', fontsize=11)
ax.set_title(f'Policy Persistency (Annual Lapse {lapse_rate_annual*100:.1f}%)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim([0, 105])

# Plot 4: Cumulative expense recovery
ax = axes[1, 1]
years_cumul = np.arange(1, contract_term + 1)
cumulative_recovery = []

cumul_load = 0
annual_load_amount = 30  # Assume $30/year load for expense recovery

for year in years_cumul:
    survival = (1 - lapse_rate_annual) ** year
    annual_recovery = annual_load_amount * survival
    cumul_load += annual_recovery
    cumulative_recovery.append(cumul_load)

total_expense_pv_plot = pv_results['direct']

ax.plot(years_cumul, cumulative_recovery, linewidth=2.5, marker='o', markersize=5, 
       label='Cumulative Recovery', color='green')
ax.axhline(y=total_expense_pv_plot, color='red', linestyle='--', linewidth=2, 
          label=f'Target (PV Expenses: ${total_expense_pv_plot:,.0f})', alpha=0.7)

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Cumulative Recovery ($)', fontsize=11)
ax.set_title('Renewal Expense Recovery Profile', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('renewal_expenses.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

