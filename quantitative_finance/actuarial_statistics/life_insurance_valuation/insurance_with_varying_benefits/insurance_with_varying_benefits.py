# Auto-extracted from markdown file
# Source: insurance_with_varying_benefits.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. MORTALITY & ASSUMPTIONS
print("=" * 80)
print("INSURANCE WITH VARYING BENEFITS: PRICING & ANALYSIS")
print("=" * 80)

# Standard mortality
mortality_data = {
    25: 0.00067, 30: 0.00084, 35: 0.00103, 40: 0.00131, 45: 0.00172,
    50: 0.00233, 55: 0.00325, 60: 0.00459, 65: 0.00653, 70: 0.00933,
    75: 0.01330, 80: 0.01934, 85: 0.02873, 90: 0.04213
}

def gompertz_mortality(age, A=0.0001, B=1.075):
    return A * (B ** age)

annual_rate = 0.04
v = 1 / (1 + annual_rate)

print(f"\nAssumptions:")
print(f"  Interest Rate: {annual_rate*100:.1f}%")
print(f"  Mortality: US Standard")
print(f"  Benefit patterns: Level, Increasing (Linear), Decreasing (Linear)\n")

# 2. LEVEL INSURANCE (BASELINE)
print("=" * 80)
print("BASELINE: LEVEL 20-YEAR TERM")
print("=" * 80)

def calculate_level_apv(start_age, term_length, benefit_amt, mortality_dict, annual_rate_calc):
    """Calculate APV of level insurance"""
    
    apv = 0
    kpx = 1.0
    
    for k in range(1, term_length + 1):
        age_k = start_age + k - 1
        
        if age_k in mortality_dict:
            qx_k = mortality_dict[age_k]
        else:
            mu = gompertz_mortality(age_k)
            qx_k = 1 - np.exp(-mu)
        
        vk = v ** k
        apv += kpx * qx_k * vk * benefit_amt
        
        px_k = 1 - qx_k
        kpx *= px_k
    
    return apv

def calculate_annuity_due_limited(start_age, term_length, mortality_dict, annual_rate_calc):
    """Calculate PV of annuity due for limited term"""
    
    pv = 0
    kpx = 1.0
    
    for k in range(0, term_length):
        vk = v ** k
        pv += kpx * vk
        
        if k < term_length - 1:
            age_k = start_age + k
            
            if age_k in mortality_dict:
                qx_k = mortality_dict[age_k]
            else:
                mu = gompertz_mortality(age_k)
                qx_k = 1 - np.exp(-mu)
            
            px_k = 1 - qx_k
            kpx *= px_k
    
    return pv

# Parameters
start_age = 40
term_length = 20
level_benefit = 300000

# Calculate level insurance
apv_level = calculate_level_apv(start_age, term_length, level_benefit, mortality_data, annual_rate)
pv_ann = calculate_annuity_due_limited(start_age, term_length, mortality_data, annual_rate)

net_prem_level = apv_level / pv_ann
gross_prem_level = net_prem_level * 1.25  # Add 25% for expenses

print(f"\nAge {start_age}, {term_length}-Year Term, ${level_benefit:,.0f} Level Benefit\n")
print(f"APV: ${apv_level:,.2f}")
print(f"Net Premium: ${net_prem_level:,.2f}/year (${net_prem_level/12:.2f}/month)")
print(f"Gross Premium: ${gross_prem_level:,.2f}/year (${gross_prem_level/12:.2f}/month)")
print()

# 3. DECREASING BENEFIT (MORTGAGE PAYOFF)
print("=" * 80)
print("DECREASING BENEFIT: MORTGAGE PAYOFF STRUCTURE")
print("=" * 80)

# Mortgage payoff: $400K initially, declining linearly to $0 over 30 years
mortgage_initial = 400000
mortgage_term = 30

# APV of decreasing term
apv_decreasing = 0
kpx_dec = 1.0

print(f"\nMortgage Protection: ${mortgage_initial:,.0f}, {mortgage_term}-year decline\n")
print(f"{'Year':<8} {'Benefit':<15} {'qₓ':<12} {'PV Factor':<12} {'Annual PV':<15}")
print("-" * 70)

for k in range(1, mortgage_term + 1):
    age_k = start_age + k - 1
    
    # Benefit: Linear decline from mortgage_initial to 0
    benefit_k = max(0, mortgage_initial * (1 - k / mortgage_term))
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        mu = gompertz_mortality(age_k)
        qx_k = 1 - np.exp(-mu)
    
    vk = v ** k
    annual_pv = kpx_dec * qx_k * vk * benefit_k
    apv_decreasing += annual_pv
    
    if k <= 5 or k % 5 == 0 or k == mortgage_term:
        print(f"{k:<8} ${benefit_k:<14,.0f} {qx_k:<12.6f} {vk:<12.6f} ${annual_pv:<14,.2f}")
    
    px_k = 1 - qx_k
    kpx_dec *= px_k

pv_ann_dec = calculate_annuity_due_limited(start_age, mortgage_term, mortality_data, annual_rate)

net_prem_dec = apv_decreasing / pv_ann_dec
gross_prem_dec = net_prem_dec * 1.25

print(f"\nAPV (Decreasing): ${apv_decreasing:,.2f}")
print(f"Annual Net Premium: ${net_prem_dec:,.2f} (${net_prem_dec/12:.2f}/month)")
print(f"Annual Gross Premium: ${gross_prem_dec:,.2f} (${gross_prem_dec/12:.2f}/month)")
print()

# Compare to level term
apv_level_30yr = calculate_level_apv(start_age, mortgage_term, mortgage_initial, 
                                     mortality_data, annual_rate)
net_prem_level_30yr = apv_level_30yr / pv_ann_dec
gross_prem_level_30yr = net_prem_level_30yr * 1.25

print(f"Comparison to Level ${mortgage_initial:,.0f} (30 years):")
print(f"  Level annual premium: ${gross_prem_level_30yr:,.2f}")
print(f"  Decreasing annual premium: ${gross_prem_dec:,.2f}")
print(f"  Savings (decreasing): ${gross_prem_level_30yr - gross_prem_dec:,.2f} ({(1-gross_prem_dec/gross_prem_level_30yr)*100:.1f}%)")
print()

# 4. INCREASING BENEFIT (FAMILY PROTECTION)
print("=" * 80)
print("INCREASING BENEFIT: GROWING FAMILY NEEDS")
print("=" * 80)

# Increasing linearly: Start $100K, increase $20K/year for 20 years
init_benefit_inc = 100000
annual_increase = 20000
increase_term = 20

apv_increasing = 0
kpx_inc = 1.0

print(f"\nFamily Protection: ${init_benefit_inc:,.0f}, +${annual_increase:,.0f}/year, {increase_term}-year term\n")
print(f"{'Year':<8} {'Benefit':<15} {'qₓ':<12} {'PV Factor':<12} {'Annual PV':<15}")
print("-" * 70)

for k in range(1, increase_term + 1):
    age_k = start_age + k - 1
    
    # Benefit: Linear increase
    benefit_k = init_benefit_inc + (k - 1) * annual_increase
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        mu = gompertz_mortality(age_k)
        qx_k = 1 - np.exp(-mu)
    
    vk = v ** k
    annual_pv = kpx_inc * qx_k * vk * benefit_k
    apv_increasing += annual_pv
    
    if k <= 5 or k % 5 == 0 or k == increase_term:
        print(f"{k:<8} ${benefit_k:<14,.0f} {qx_k:<12.6f} {vk:<12.6f} ${annual_pv:<14,.2f}")
    
    px_k = 1 - qx_k
    kpx_inc *= px_k

pv_ann_inc = calculate_annuity_due_limited(start_age, increase_term, mortality_data, annual_rate)

net_prem_inc = apv_increasing / pv_ann_inc
gross_prem_inc = net_prem_inc * 1.25

avg_benefit_inc = init_benefit_inc + (increase_term - 1) * annual_increase / 2
apv_level_avg = calculate_level_apv(start_age, increase_term, avg_benefit_inc, 
                                    mortality_data, annual_rate)
net_prem_level_avg = apv_level_avg / pv_ann_inc
gross_prem_level_avg = net_prem_level_avg * 1.25

print(f"\nAPV (Increasing): ${apv_increasing:,.2f}")
print(f"Annual Net Premium: ${net_prem_inc:,.2f} (${net_prem_inc/12:.2f}/month)")
print(f"Annual Gross Premium: ${gross_prem_inc:,.2f} (${gross_prem_inc/12:.2f}/month)")
print()
print(f"Comparison to Level ${avg_benefit_inc:,.0f} (average benefit):")
print(f"  Level at average: ${gross_prem_level_avg:,.2f}")
print(f"  Increasing: ${gross_prem_inc:,.2f}")
print(f"  Difference: ${gross_prem_inc - gross_prem_level_avg:,.2f} ({((gross_prem_inc/gross_prem_level_avg)-1)*100:.1f}%)")
print()

# 5. INFLATION-PROTECTED INCREASING BENEFIT
print("=" * 80)
print("GEOMETRIC INCREASING: INFLATION PROTECTION")
print("=" * 80)

base_benefit_geo = 200000
annual_inflation_rate = 0.03  # 3% annual increase
geo_term = 20

apv_geometric = 0
kpx_geo = 1.0

print(f"\nInflation-Protected: Base ${base_benefit_geo:,.0f}, {annual_inflation_rate*100:.1f}%/year increase, {geo_term}-year\n")

for k in range(1, geo_term + 1):
    age_k = start_age + k - 1
    
    # Benefit: Geometric increase
    benefit_k = base_benefit_geo * ((1 + annual_inflation_rate) ** (k - 1))
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        mu = gompertz_mortality(age_k)
        qx_k = 1 - np.exp(-mu)
    
    vk = v ** k
    annual_pv = kpx_geo * qx_k * vk * benefit_k
    apv_geometric += annual_pv
    
    if k <= 5 or k % 5 == 0 or k == geo_term:
        print(f"Year {k}: Benefit = ${benefit_k:>10,.0f}, PV = ${annual_pv:>12,.2f}")
    
    px_k = 1 - qx_k
    kpx_geo *= px_k

pv_ann_geo = calculate_annuity_due_limited(start_age, geo_term, mortality_data, annual_rate)

net_prem_geo = apv_geometric / pv_ann_geo
gross_prem_geo = net_prem_geo * 1.25

print(f"\nAPV (Geometric): ${apv_geometric:,.2f}")
print(f"Annual Net Premium: ${net_prem_geo:,.2f} (${net_prem_geo/12:.2f}/month)")
print(f"Annual Gross Premium: ${gross_prem_geo:,.2f} (${gross_prem_geo/12:.2f}/month)")
print()

# 6. PREMIUM COMPARISON TABLE
print("=" * 80)
print("PREMIUM COMPARISON: LEVEL vs VARYING BENEFITS")
print("=" * 80)

print(f"\nAge {start_age}\n")
print(f"{'Product':<40} {'Annual Premium':<20} {'Monthly':<15} {'% of Level':<15}")
print("-" * 90)

products_comp = [
    ("Level $300K (20-year)", gross_prem_level),
    ("Decreasing Mortgage $400K→0 (30-year)", gross_prem_dec),
    ("Increasing Family $100K→$480K (20-year)", gross_prem_inc),
    ("Geometric Inflation-Protected $200K (20-year)", gross_prem_geo),
]

for prod_name, prem in products_comp:
    pct_level = (prem / gross_prem_level) * 100
    print(f"{prod_name:<40} ${prem:<19,.2f} ${prem/12:<14,.2f} {pct_level:<14.1f}%")

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Benefit schedules comparison
ax = axes[0, 0]
years_plot = np.arange(1, 21)

level_schedule = np.full(20, level_benefit)
decreasing_schedule = np.array([max(0, mortgage_initial * (1 - k / mortgage_term)) for k in range(1, 21)])
increasing_schedule = np.array([init_benefit_inc + (k - 1) * annual_increase for k in range(1, 21)])
geometric_schedule = np.array([base_benefit_geo * ((1 + annual_inflation_rate) ** (k - 1)) for k in range(1, 21)])

ax.plot(years_plot, level_schedule / 1000, linewidth=2.5, marker='o', label='Level $300K', color='steelblue')
ax.plot(years_plot, decreasing_schedule / 1000, linewidth=2.5, marker='s', label='Decreasing Mortgage', color='red')
ax.plot(years_plot, increasing_schedule / 1000, linewidth=2.5, marker='^', label='Increasing Family', color='green')
ax.plot(years_plot, geometric_schedule / 1000, linewidth=2.5, marker='d', label='Geometric 3%', color='orange')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Benefit ($000s)', fontsize=11)
ax.set_title('Benefit Schedules Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: APV contribution by year (decreasing vs increasing)
ax = axes[0, 1]
apv_contrib_dec = []
apv_contrib_inc = []
kpx_d = 1.0
kpx_i = 1.0

for k in range(1, mortgage_term + 1):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        mu = gompertz_mortality(age_k)
        qx_k = 1 - np.exp(-mu)
    
    vk = v ** k
    
    # Decreasing
    benefit_dec_k = max(0, mortgage_initial * (1 - k / mortgage_term))
    contrib_dec = kpx_d * qx_k * vk * benefit_dec_k
    apv_contrib_dec.append(contrib_dec)
    
    px_k = 1 - qx_k
    kpx_d *= px_k

years_30 = np.arange(1, mortgage_term + 1)
ax.bar(years_30 - 0.2, np.array(apv_contrib_dec) / 1000, width=0.4, label='Decreasing', alpha=0.6, color='red')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('APV Contribution ($000s)', fontsize=11)
ax.set_title('Year-by-Year APV Contribution (Decreasing Term)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Premium comparison
ax = axes[1, 0]
products_names = ['Level\n$300K', 'Decreasing\nMtg', 'Increasing\nFamily', 'Geometric\nInflation']
premium_values = [gross_prem_level, gross_prem_dec, gross_prem_inc, gross_prem_geo]
colors_prem = ['steelblue', 'red', 'green', 'orange']

bars = ax.bar(products_names, premium_values, color=colors_prem, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Annual Gross Premium ($)', fontsize=11)
ax.set_title('Premium Comparison by Benefit Pattern', fontsize=12, fontweight='bold')

for bar, prem in zip(bars, premium_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${prem:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Cumulative payout expectation
ax = axes[1, 1]
years_20 = np.arange(1, 21)

cumul_level = []
cumul_dec = []
cumul_inc = []
cumul_geo = []

kpx_l = 1.0
kpx_d = 1.0
kpx_i = 1.0
kpx_g = 1.0

cum_l = 0
cum_d = 0
cum_i = 0
cum_g = 0

for k in years_20:
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        mu = gompertz_mortality(age_k)
        qx_k = 1 - np.exp(-mu)
    
    vk = v ** k
    px_k = 1 - qx_k
    
    # Level
    cum_l += kpx_l * qx_k * vk * level_benefit
    cumul_level.append(cum_l)
    kpx_l *= px_k
    
    # Decreasing (only 30 years, but show 20)
    if k <= mortgage_term:
        benefit_d = max(0, mortgage_initial * (1 - k / mortgage_term))
        cum_d += kpx_d * qx_k * vk * benefit_d
    cumul_dec.append(cum_d)
    if k <= mortgage_term:
        kpx_d *= px_k
    
    # Increasing
    benefit_i = init_benefit_inc + (k - 1) * annual_increase
    cum_i += kpx_i * qx_k * vk * benefit_i
    cumul_inc.append(cum_i)
    kpx_i *= px_k
    
    # Geometric
    benefit_g = base_benefit_geo * ((1 + annual_inflation_rate) ** (k - 1))
    cum_g += kpx_g * qx_k * vk * benefit_g
    cumul_geo.append(cum_g)
    kpx_g *= px_k

ax.plot(years_20, np.array(cumul_level) / 1000, linewidth=2.5, marker='o', label='Level $300K', color='steelblue')
ax.plot(years_20, np.array(cumul_dec) / 1000, linewidth=2.5, marker='s', label='Decreasing Mtg', color='red')
ax.plot(years_20, np.array(cumul_inc) / 1000, linewidth=2.5, marker='^', label='Increasing Fam', color='green')
ax.plot(years_20, np.array(cumul_geo) / 1000, linewidth=2.5, marker='d', label='Geometric 3%', color='orange')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Cumulative APV ($000s)', fontsize=11)
ax.set_title('Cumulative Expected Payout by Year', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('varying_benefits_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

