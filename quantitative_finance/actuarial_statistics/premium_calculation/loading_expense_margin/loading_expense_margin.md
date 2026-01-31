# Loading & Expense Margin

## 1. Concept Skeleton
**Definition:** Additional margin on net premium covering acquisition, maintenance, profit loadings; Gross Premium = Net + Load  
**Purpose:** Recovers non-mortality costs (commissions, underwriting, policyholder servicing); generates company profit; enables premium competitiveness  
**Prerequisites:** Net premium, expense types (fixed, variable, % of premium), load factor (typical 25-35%)

## 2. Comparative Framing
| Load Component | Type | Amount | Timing | Recovery |
|----------------|------|--------|--------|----------|
| **Acquisition** | % of premium | 10-15% | Year 1 | 1-3 years |
| **Maintenance** | Per policy | $25-50/yr | Annually | Throughout life |
| **Profit load** | % of net premium | 15-25% | Throughout | Annual release |
| **Risk margin** | % of APV | 5-10% | At issue | Statutory reserve |
| **Contingency** | % | 2-5% | Variable | Surplus drain |

## 3. Examples + Counterexamples

**Simple Example:**  
Net premium = $100/yr; Acquisition $15 (Y1), Maintenance $30/yr, Profit 20% ($20) → Gross ≈ $135/yr (rounded)

**Failure Case:**  
Set load too low ($110 total): After-tax maintenance costs $18; profit = $1 after expenses; cannot sustain operations

**Edge Case:**  
High-risk term insurance (age 75, smoker): Net = $8,000/yr; load = 25% = $2,000; load barely covers actual commission ($1,200)

## 4. Layer Breakdown
```
Expense Loading Structure:
├─ Load Components:
│   ├─ Acquisition Expense:
│   │   ├─ Agent commission: 40-110% of first-year premium
│   │   ├─ Medical exam: $100-300 per issue
│   │   ├─ Underwriting: $50-150 per issue
│   │   ├─ Marketing: Fixed per campaign
│   │   ├─ Sales support: % of sales
│   │   └─ First-year total: 15-25% of first-year gross premium
│   ├─ Maintenance Expense:
│   │   ├─ Policy administration: $5-15/year
│   │   ├─ Billing/collection: $3-8/year (if annual billing)
│   │   ├─ Customer service: $2-5/year
│   │   ├─ Compliance: $2-5/year
│   │   ├─ Systems: $3-8/year
│   │   └─ Annual total: $20-50/year per policy
│   ├─ Profit Load:
│   │   ├─ Target return on equity: 10-15%
│   │   ├─ Risk adjustment: Mortality variance margin
│   │   ├─ Competitive adjustment: Market-based load
│   │   ├─ Growth subsidy: Lower for high-volume products
│   │   └─ Load as % of net premium: 15-30%
│   └─ Regulatory Load:
│       ├─ Solvency II risk margin: 5-10%
│       ├─ Statutory minimum reserve buffer: ~5%
│       ├─ Policyholder protection fund: 0-2%
│       └─ Jurisdictional variation: Significant
├─ Calculation Methods:
│   ├─ Method 1: Add flat load to net premium
│   │   └─ Gross = Net × (1 + Load%)
│   ├─ Method 2: Expense-based allocation
│   │   └─ Gross = [Net + PV(Fixed) + % × Net + Profit] / PV(Annuity)
│   ├─ Method 3: Iterative (most accurate)
│   │   ├─ Start: Gross₀ = Net × 1.25
│   │   ├─ Calculate: Expense PV = Acq × P(0) + Maint × äₙ̄| + Profit × Net
│   │   ├─ Update: Gross₁ = [Net + Expense PV] / äₙ̄|
│   │   ├─ Repeat until convergence
│   │   └─ Converges quickly (2-3 iterations)
│   └─ Method 4: Segment-specific
│       ├─ High commissions (agent distribution): +30%
│       ├─ Medium commissions (bank channel): +20%
│       └─ Low commissions (direct-online): +15%
├─ Variations by Distribution:
│   ├─ Agent/captive: 15-25% load for high commission payouts
│   ├─ Bank distribution: 12-18% load (moderate commissions)
│   ├─ Direct mail: 10-15% load (minimal commission)
│   ├─ Online/portal: 8-12% load (tech-leveraged)
│   └─ Group/employment: 5-8% load (spread over many lives)
├─ Regulation & Solvency:
│   ├─ Minimum premium rule: Gross ≥ Net + (Expenses PV / Benefits)
│   ├─ Deficiency reserve: If Gross underestimates expenses, statutory reserve increases
│   ├─ Expense disclosure: Transparency requirements (EU, US states)
│   ├─ Loaded premium sustainability: Must exceed expenses over policy lifetime
│   └─ Profit testing: Compare projected vs actual expenses annually
├─ Sensitivity to Load Assumptions:
│   ├─ Acquisition +50% → Load increases 2-3 percentage points
│   ├─ Maintenance inflation 3%/yr → Reserve shortfall by year 10
│   ├─ Commission rate change: Direct impact on first-year cost
│   └─ Market share impact: Higher load may reduce sales volume → unit cost rises
└─ Practical Load Assignment:
    ├─ Simplified load: Flat 25-30% for standard term insurance
    ├─ Tiered load:
    │   ├─ Young/healthy age 30-40: 20% (lower risk, high demand)
    │   ├─ Middle age 40-55: 25% (standard)
    │   └─ Senior age 65+: 30-35% (higher expenses, lower volume)
    ├─ Product-specific:
    │   ├─ Term (simple): 20-25%
    │   ├─ Whole life (complex): 30-40%
    │   ├─ Annuity (investment): 15-20%
    │   └─ Disability income (risky): 35-45%
    └─ Distribution-specific:
        ├─ Agent: 25-35%
        ├─ Bank: 15-20%
        ├─ Online: 10-15%
        └─ Group: 5-10%
```

**Key Insight:** Load must balance profitability against market competitiveness; too low → losses, too high → uncompetitive

## 5. Mini-Project
Calculate expense-loaded premiums, test sensitivity to load factors, and compare distribution channels:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. SETUP
print("=" * 80)
print("LOADING & EXPENSE MARGIN ANALYSIS")
print("=" * 80)

# Mortality table
mortality_data = {
    25: 0.00067, 30: 0.00084, 35: 0.00103, 40: 0.00131, 45: 0.00172,
    50: 0.00233, 55: 0.00325, 60: 0.00459, 65: 0.00653, 70: 0.00933
}

interest_rate = 0.04
v = 1 / (1 + interest_rate)
benefit = 250000

print(f"\nAssumptions:")
print(f"  Benefit: ${benefit:,.0f}")
print(f"  Interest: {interest_rate*100:.1f}%")
print(f"  Product: 20-Year Term\n")

def calculate_net_premium(start_age, term_years, mortality_dict, interest_rate, benefit_amt):
    """Calculate net premium"""
    
    v_rate = 1 / (1 + interest_rate)
    apv_b = 0
    apv_p = 0
    kpx = 1.0
    
    for k in range(1, term_years + 1):
        age_k = start_age + k - 1
        qx_k = mortality_dict.get(age_k, 0.001)
        vk = v_rate ** k
        
        apv_b += kpx * qx_k * vk * benefit_amt
        px_k = 1 - qx_k
        kpx *= px_k
    
    kpx = 1.0
    for k in range(0, term_years):
        vk = v_rate ** k
        apv_p += kpx * vk
        
        if k < term_years - 1:
            age_k = start_age + k
            qx_k = mortality_dict.get(age_k, 0.001)
            px_k = 1 - qx_k
            kpx *= px_k
    
    net_prem = apv_b / apv_p
    return net_prem, apv_p

# Calculate net premium
start_age = 40
term_years = 20
net_prem, annuity_pv = calculate_net_premium(start_age, term_years, mortality_data, interest_rate, benefit)

print(f"Net Premium (40 years, {term_years}-year term): ${net_prem:,.2f}/year")
print(f"PV(Annuity-due): {annuity_pv:.6f}\n")

# 2. EXPENSE STRUCTURE
print("=" * 80)
print("EXPENSE STRUCTURE & COMPONENTS")
print("=" * 80)

# Year 1 acquisition expenses
acquisition_pct = 0.15  # 15% of Y1 gross premium
medical_exam = 150      # One-time
underwriting = 75       # One-time
setup_cost = 50         # One-time

# Annual maintenance expenses
policy_admin = 12
billing_collection = 4
customer_service = 3
compliance = 2
systems_cost = 5

maintenance_annual = policy_admin + billing_collection + customer_service + compliance + systems_cost

# Profit margin
profit_pct = 0.20  # 20% of net premium

print(f"\nYear 1 Acquisition Costs:")
print(f"  Commission (% of gross): {acquisition_pct*100:.0f}%")
print(f"  Medical exam: ${medical_exam}")
print(f"  Underwriting: ${underwriting}")
print(f"  Setup: ${setup_cost}")
print(f"  Total Y1 (fixed): ${medical_exam + underwriting + setup_cost}\n")

print(f"Annual Maintenance Costs:")
print(f"  Policy administration: ${policy_admin}")
print(f"  Billing/collection: ${billing_collection}")
print(f"  Customer service: ${customer_service}")
print(f"  Compliance: ${compliance}")
print(f"  Systems: ${systems_cost}")
print(f"  Total annual (fixed): ${maintenance_annual}\n")

print(f"Profit Target: {profit_pct*100:.0f}% of net premium\n")

# 3. EXPENSE-LOADED PREMIUM CALCULATION (ITERATIVE)
print("=" * 80)
print("EXPENSE-LOADED PREMIUM CALCULATION (ITERATIVE METHOD)")
print("=" * 80)

def calculate_gross_premium_iterative(net_prem, annuity_pv, acq_fixed, acq_pct, 
                                     maintenance, profit_pct):
    """Calculate gross premium iteratively"""
    
    gross = net_prem * 1.25  # Initial guess
    
    for iteration in range(5):
        # Commission is % of gross
        commission = gross * acq_pct
        
        # Total acquisition
        total_acq = acq_fixed + commission
        
        # PV of maintenance costs
        pv_maintenance = maintenance * annuity_pv
        
        # Profit load
        profit_load = net_prem * profit_pct
        
        # Total expense PV
        total_expense_pv = total_acq + pv_maintenance + profit_load
        
        # New gross premium
        gross_new = (net_prem + total_expense_pv / annuity_pv) / 1.0
        
        if abs(gross_new - gross) < 0.01:
            break
        
        gross = gross_new
    
    return gross, total_acq, pv_maintenance, profit_load, total_expense_pv

gross_prem, acq_total, maint_pv, profit_load, exp_pv = calculate_gross_premium_iterative(
    net_prem, annuity_pv, medical_exam + underwriting + setup_cost, acquisition_pct, 
    maintenance_annual, profit_pct
)

print(f"\nIteration Results:\n")
print(f"{'Component':<40} {'Amount':<20}")
print("-" * 60)
print(f"{'Net Premium (actuarial cost)':<40} ${net_prem:>18,.2f}")
print()
print(f"{'Year 1 Fixed Acquisition':<40} ${medical_exam + underwriting + setup_cost:>18,.2f}")
print(f"{'Commission (15% × Gross Premium)':<40} ${acq_total - (medical_exam + underwriting + setup_cost):>18,.2f}")
print(f"{'Total Acquisition Expense':<40} ${acq_total:>18,.2f}")
print()
print(f"{'PV of Annual Maintenance ($25/yr)':<40} ${maint_pv:>18,.2f}")
print(f"{'Profit Load (20% of net)':<40} ${profit_load:>18,.2f}")
print()
print(f"{'Total Expense PV':<40} ${exp_pv:>18,.2f}")
print()
print(f"{'GROSS ANNUAL PREMIUM':<40} ${gross_prem:>18,.2f}")
print()

load_pct = ((gross_prem - net_prem) / net_prem) * 100
print(f"Load as % of Net Premium: {load_pct:.1f}%\n")

# 4. EXPENSE RATIO ANALYSIS
print("=" * 80)
print("EXPENSE RATIO ANALYSIS")
print("=" * 80)

print(f"\nPer $1 of Premium Revenue:\n")

# Assuming we collect gross premium
premium_revenue = gross_prem
net_cost = net_prem
acq_ratio = acq_total / premium_revenue
maint_ratio = (maintenance_annual * annuity_pv) / premium_revenue
profit_ratio = profit_load / premium_revenue

print(f"{'Expense Type':<35} {'% of Premium':<20}")
print("-" * 55)
print(f"{'Net (benefit cost)':<35} {(net_cost/premium_revenue)*100:>18.1f}%")
print(f"{'Acquisition (Y1)':<35} {(acq_ratio)*100:>18.1f}%")
print(f"{'Maintenance (PV)':<35} {(maint_ratio)*100:>18.1f}%")
print(f"{'Profit margin':<35} {(profit_ratio)*100:>18.1f}%")
print()

# 5. COMPARISON: DIFFERENT LOAD SCENARIOS
print("=" * 80)
print("LOAD COMPARISON: DIFFERENT SCENARIOS")
print("=" * 80)

scenarios = {
    'Direct Online (Low)': {'acq_pct': 0.08, 'maint': 15, 'profit': 0.10},
    'Digital (Medium-Low)': {'acq_pct': 0.10, 'maint': 20, 'profit': 0.15},
    'Bank Distribution': {'acq_pct': 0.12, 'maint': 25, 'profit': 0.18},
    'Standard (Composite)': {'acq_pct': 0.15, 'maint': 25, 'profit': 0.20},
    'Agent Distribution': {'acq_pct': 0.25, 'maint': 30, 'profit': 0.25},
    'Premium Agent': {'acq_pct': 0.35, 'maint': 35, 'profit': 0.30},
}

print(f"\n20-Year Term, Age 40, $250K Benefit\n")
print(f"{'Channel':<25} {'Acq %':<10} {'Gross Prem':<15} {'Load %':<10} {'Y1 Expense':<15}")
print("-" * 75)

scenario_results = {}

for channel, params in scenarios.items():
    acq_fixed_scenario = medical_exam + underwriting + setup_cost  # Fixed portion
    
    gross_s, acq_s, maint_s, profit_s, exp_s = calculate_gross_premium_iterative(
        net_prem, annuity_pv, acq_fixed_scenario, params['acq_pct'], 
        params['maint'], params['profit']
    )
    
    load_pct_s = ((gross_s - net_prem) / net_prem) * 100
    y1_commission = gross_s * params['acq_pct']
    
    scenario_results[channel] = {
        'gross': gross_s,
        'load_pct': load_pct_s,
        'commission': y1_commission
    }
    
    print(f"{channel:<25} {params['acq_pct']*100:>8.0f}% ${gross_s:>13,.2f} {load_pct_s:>8.1f}% ${y1_commission:>13,.0f}")

print()

# 6. BREAK-EVEN ANALYSIS
print("=" * 80)
print("BREAK-EVEN ANALYSIS: EXPENSE RECOVERY")
print("=" * 80)

# How long to recover Y1 acquisition from load
y1_acquisition = medical_exam + underwriting + setup_cost + (gross_prem * acquisition_pct)
annual_load = (maintenance_annual * annuity_pv + profit_load) / term_years  # Spread over contract term

breakeven_years = y1_acquisition / (annual_load / annuity_pv)  # Approximate

print(f"\nYear 1 Acquisition Cost: ${y1_acquisition:,.2f}")
print(f"Annual Load Contribution: ${annual_load / annuity_pv:,.2f}")
print(f"Approximate Break-Even: Year {breakeven_years:.1f}")
print(f"  (Assumes load fully covers expenses; doesn't account for profit)\n")

# 7. SENSITIVITY TO LOAD PARAMETERS
print("=" * 80)
print("SENSITIVITY: GROSS PREMIUM TO LOAD CHANGES")
print("=" * 80)

print(f"\nBase Case (Standard Distribution): ${gross_prem:,.2f}\n")

load_changes = {
    'Commission rate -5%': {'acq_pct': acquisition_pct - 0.05, 'maint': maintenance_annual, 'profit': profit_pct},
    'Maintenance +$10/yr': {'acq_pct': acquisition_pct, 'maint': maintenance_annual + 10, 'profit': profit_pct},
    'Profit target +5%': {'acq_pct': acquisition_pct, 'maint': maintenance_annual, 'profit': profit_pct + 0.05},
    'Commission +5%': {'acq_pct': acquisition_pct + 0.05, 'maint': maintenance_annual, 'profit': profit_pct},
    'Maintenance +$20/yr': {'acq_pct': acquisition_pct, 'maint': maintenance_annual + 20, 'profit': profit_pct},
    'All three +20%': {'acq_pct': acquisition_pct * 1.20, 'maint': maintenance_annual * 1.20, 'profit': profit_pct * 1.20},
}

print(f"{'Scenario':<30} {'Gross Premium':<20} {'Change':<15} {'% Change':<12}")
print("-" * 77)

for scenario, params in load_changes.items():
    gross_s, _, _, _, _ = calculate_gross_premium_iterative(
        net_prem, annuity_pv, medical_exam + underwriting + setup_cost, 
        params['acq_pct'], params['maint'], params['profit']
    )
    
    change = gross_s - gross_prem
    pct_change = (change / gross_prem) * 100
    
    print(f"{scenario:<30} ${gross_s:>18,.2f} {change:>+13,.2f} {pct_change:>+10.1f}%")

print()

# 8. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Premium decomposition
ax = axes[0, 0]
components = ['Net\nPremium', 'Acquisition\nExpense', 'Maintenance\n(PV)', 'Profit\nMargin']
amounts = [net_prem, acq_total / annuity_pv, maintenance_annual, profit_load / annuity_pv]
cumulative = 0

for i, (comp, amt) in enumerate(zip(components, amounts)):
    ax.bar(i, amt, bottom=cumulative if i > 0 else 0, alpha=0.6, 
          edgecolor='black', linewidth=1.5)
    cumulative += amt

ax.bar(4, gross_prem, color='darkblue', alpha=0.6, edgecolor='black', linewidth=2)
ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Gross Premium Composition', fontsize=12, fontweight='bold')
ax.set_xticks(range(5))
ax.set_xticklabels(components + ['GROSS'])
ax.grid(alpha=0.3, axis='y')

# Plot 2: Distribution channel comparison
ax = axes[0, 1]
channels = list(scenario_results.keys())
gross_amounts = [scenario_results[c]['gross'] for c in channels]
loads = [scenario_results[c]['load_pct'] for c in channels]

ax.scatter(loads, gross_amounts, s=200, alpha=0.6, edgecolor='black', linewidth=1.5)

for i, channel in enumerate(channels):
    ax.annotate(channel, (loads[i], gross_amounts[i]), 
               xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('Load as % of Net Premium', fontsize=11)
ax.set_ylabel('Gross Annual Premium ($)', fontsize=11)
ax.set_title('Channel Strategy: Load vs Gross Premium', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3: Sensitivity tornado
ax = axes[1, 0]
scenarios_names = [k.replace(' 20%', '').replace('Commission +5%', 'Comm +5%').replace('Commission -5%', 'Comm -5%') for k in load_changes.keys()]
pct_changes = []

for scenario, params in load_changes.items():
    gross_s, _, _, _, _ = calculate_gross_premium_iterative(
        net_prem, annuity_pv, medical_exam + underwriting + setup_cost, 
        params['acq_pct'], params['maint'], params['profit']
    )
    
    pct_change = ((gross_s - gross_prem) / gross_prem) * 100
    pct_changes.append(pct_change)

# Sort and plot as tornado
sorted_indices = np.argsort(np.abs(pct_changes))
sorted_names = [scenarios_names[i] for i in sorted_indices]
sorted_changes = [pct_changes[i] for i in sorted_indices]

colors_tornado = ['red' if x > 0 else 'green' for x in sorted_changes]

ax.barh(range(len(sorted_changes)), sorted_changes, color=colors_tornado, alpha=0.6, edgecolor='black')
ax.set_yticks(range(len(sorted_changes)))
ax.set_yticklabels(sorted_names, fontsize=9)
ax.set_xlabel('% Change in Gross Premium', fontsize=11)
ax.set_title('Sensitivity: Load Parameter Impacts', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Plot 4: Expense ratio over contract lifetime
ax = axes[1, 1]
years = np.arange(1, term_years + 1)
net_cost_ratio = []
acq_ratio_annual = []
maint_ratio_annual = []

for year in years:
    # Y1 acquisition amortized
    acq_amort = acq_total / term_years if year > 1 else acq_total
    
    # Ratio to annual gross
    net_ratio = net_prem / gross_prem
    acq_ratio_yr = acq_amort / gross_prem
    maint_ratio_yr = maintenance_annual / gross_prem
    
    net_cost_ratio.append(net_ratio * 100)
    acq_ratio_annual.append(acq_ratio_yr * 100)
    maint_ratio_annual.append(maint_ratio_yr * 100)

ax.fill_between(years, 0, net_cost_ratio, alpha=0.5, label='Net (Benefit Cost)', color='green')
ax.fill_between(years, net_cost_ratio, 
               np.array(net_cost_ratio) + np.array(acq_ratio_annual), 
               alpha=0.5, label='Acquisition (Amortized)', color='orange')
ax.fill_between(years, 
               np.array(net_cost_ratio) + np.array(acq_ratio_annual),
               100,
               alpha=0.5, label='Maintenance + Profit', color='blue')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('% of Premium', fontsize=11)
ax.set_title('Expense Ratio Over Contract Lifetime', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='right')
ax.grid(alpha=0.3)
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('loading_expense_margin.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When expense loading breaks down:
- **Expense inflation**: Actual costs 5%/year; assume flat; margin erodes quickly
- **Lapse correlation**: Higher-lapse policies recover Y1 acquisition slower; losses if lapse > assumption
- **Competition squeezes load**: Market forces load down to 18-20%; insufficient to cover 15% commissions
- **Per-policy expenses**: $30/year assumption breaks if customer service demand increases (complaints, policy changes)
- **Commission claw-backs**: Early lapse triggers commission recapture; negative cash flow in year 2
- **Tax/regulatory changes**: Increase in compliance burden; expense load insufficient without rerating

## 7. Key References
- [Actuarial Standards of Practice (ASOP 23)](https://www.soa.org/standards/) - Expense assumptions for valuation
- [SOA Exam EM Expense Management](https://www.soa.org/education/exam-req/edu-exam-em-detail.aspx) - Practice problems
- [General Insurance Practice Regulation](https://www.giro.org.uk/) - Expense allocation methodologies

---
**Status:** Load component of gross premium | **Complements:** Net Premium, Gross Premium, Profit Testing
