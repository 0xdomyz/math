# Auto-extracted from markdown file
# Source: reserve_adequacy.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP
print("=" * 80)
print("RESERVE ADEQUACY TESTING: STATUTORY VS ECONOMIC")
print("=" * 80)

# Mortality and assumptions
mortality_standard = {
    40: 0.00131, 45: 0.00172, 50: 0.00233, 55: 0.00325,
    60: 0.00459, 65: 0.00653, 70: 0.00933
}

# Assumptions
issue_age = 40
contract_term = 20
benefit_amount = 200000
policies_inforce = 10000

# Interest rate assumptions
interest_statutory = 0.03   # Conservative 3%
interest_economic = 0.04    # Realistic 4%
interest_actual = 0.025     # Actual earned (stressed)

print(f"\nAssumptions:")
print(f"  Product: 20-Year Term")
print(f"  Issue Age: {issue_age}")
print(f"  Policies: {policies_inforce:,}")
print(f"  Benefit: ${benefit_amount:,.0f}\n")

def calculate_reserve_prospective(age, remaining_term, mortality_dict, benefit_amt, 
                                 annual_premium, interest_rate):
    """Calculate prospective reserve"""
    
    v_rate = 1 / (1 + interest_rate)
    reserve = 0
    kpx = 1.0
    
    # APV of benefits
    apv_benefits = 0
    for k in range(1, remaining_term + 1):
        age_k = age + k - 1
        qx_k = mortality_dict.get(age_k, 0.001)
        vk = v_rate ** k
        
        apv_benefits += kpx * qx_k * vk * benefit_amt
        px_k = 1 - qx_k
        kpx *= px_k
    
    # APV of premiums
    apv_premiums = 0
    kpx = 1.0
    for k in range(0, remaining_term):
        vk = v_rate ** k
        apv_premiums += kpx * vk
        
        if k < remaining_term - 1:
            age_k = age + k
            qx_k = mortality_dict.get(age_k, 0.001)
            px_k = 1 - qx_k
            kpx *= px_k
    
    reserve = apv_benefits - (annual_premium * apv_premiums)
    
    return max(0, reserve)

# Calculate net premium (for statutory reserve)
net_premium = 340  # Assume from earlier calculation

# 2. ADEQUACY TEST AT DIFFERENT DURATIONS
print("=" * 80)
print("RESERVE ADEQUACY BY POLICY DURATION")
print("=" * 80)

durations = [1, 5, 10, 15, 20]
gross_premium = 450

print(f"\n{'Year':<8} {'Age':<8} {'Stat. Res':<15} {'Econ. Res':<15} {'Held':<15} {'Shortfall':<15} {'% Gap':<12}")
print("-" * 95)

adequacy_data = []

for duration in durations:
    age_now = issue_age + duration - 1
    remaining_term = contract_term - duration + 1
    
    # Statutory reserve (conservative)
    stat_reserve = calculate_reserve_prospective(age_now, remaining_term, mortality_standard, 
                                                benefit_amount, net_premium, interest_statutory)
    
    # Economic reserve (realistic)
    econ_reserve = calculate_reserve_prospective(age_now, remaining_term, mortality_standard, 
                                                benefit_amount, gross_premium, interest_economic)
    
    # Held reserve (assume at statutory minimum)
    held_reserve = stat_reserve
    
    # Shortfall analysis
    shortfall = econ_reserve - held_reserve
    shortfall_pct = (shortfall / econ_reserve * 100) if econ_reserve > 0 else 0
    
    adequacy_data.append({
        'duration': duration,
        'age': age_now,
        'stat_res': stat_reserve,
        'econ_res': econ_reserve,
        'held': held_reserve,
        'shortfall': shortfall
    })
    
    print(f"{duration:<8} {age_now:<8} ${stat_reserve:<14,.0f} ${econ_reserve:<14,.0f} ${held_reserve:<14,.0f} ${shortfall:>+13,.0f} {shortfall_pct:>+10.1f}%")

print()

# 3. SENSITIVITY: INTEREST RATE SHOCK
print("=" * 80)
print("SENSITIVITY: INTEREST RATE IMPACT ON RESERVE ADEQUACY")
print("=" * 80)

year_test = 10
age_test = issue_age + year_test - 1
remaining_test = contract_term - year_test + 1

print(f"\nYear {year_test} Analysis (Age {age_test}):\n")

interest_rates = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

print(f"{'Interest Rate':<18} {'Economic Res':<18} {'vs 4% Base':<18} {'Shortfall':<18}")
print("-" * 72)

base_econ_res = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                             benefit_amount, gross_premium, 0.04)

for rate in interest_rates:
    econ_res = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                            benefit_amount, gross_premium, rate)
    
    pct_vs_base = ((econ_res / base_econ_res) - 1) * 100
    shortfall = econ_res - (stat_reserve)
    
    print(f"{rate*100:>16.2f}% ${econ_res:>16,.0f} {pct_vs_base:>+16.1f}% ${shortfall:>+16,.0f}")

print()

# 4. MORTALITY SHOCK SCENARIO
print("=" * 80)
print("MORTALITY SHOCK: +20% WORSE THAN ASSUMPTION")
print("=" * 80)

mortality_shocked = {age: qx * 1.20 for age, qx in mortality_standard.items()}

print(f"\nYear {year_test} Under Mortality Stress:\n")

econ_res_base = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                             benefit_amount, gross_premium, interest_economic)

econ_res_stressed = calculate_reserve_prospective(age_test, remaining_test, mortality_shocked,
                                                 benefit_amount, gross_premium, interest_economic)

print(f"{'Scenario':<30} {'Reserve':<18} {'vs Base':<18}")
print("-" * 66)
print(f"{'Base case (standard mort)':<30} ${econ_res_base:>16,.0f}")
print(f"{'Mortality +20%':<30} ${econ_res_stressed:>16,.0f} {((econ_res_stressed/econ_res_base)-1)*100:>+16.1f}%")

print()

# 5. COMPOUND STRESS TEST
print("=" * 80)
print("COMPOUND STRESS: INTEREST -1%, MORTALITY +20%")
print("=" * 80)

interest_stressed = interest_economic - 0.01

econ_res_compound = calculate_reserve_prospective(age_test, remaining_test, mortality_shocked,
                                                 benefit_amount, gross_premium, interest_stressed)

print(f"\n{'Scenario':<40} {'Reserve':<18}")
print("-" * 58)
print(f"{'Base (4% int, std mort)':<40} ${econ_res_base:>16,.0f}")
print(f"{'Stressed (3% int, +20% mort)':<40} ${econ_res_compound:>16,.0f}")
print()
print(f"Reserve increase needed: ${econ_res_compound - econ_res_base:,.0f}")
print(f"Per-policy: ${(econ_res_compound - econ_res_base):.0f}")
print(f"Total for cohort: ${(econ_res_compound - econ_res_base) * policies_inforce:,.0f}")

print()

# 6. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reserve by duration
ax = axes[0, 0]
durations_plot = [d['duration'] for d in adequacy_data]
stat_reserves = [d['stat_res'] for d in adequacy_data]
econ_reserves = [d['econ_res'] for d in adequacy_data]

ax.plot(durations_plot, stat_reserves, linewidth=2.5, marker='o', markersize=6, 
       label='Statutory Reserve', color='blue')
ax.plot(durations_plot, econ_reserves, linewidth=2.5, marker='s', markersize=6, 
       label='Economic Reserve', color='green')
ax.fill_between(durations_plot, stat_reserves, econ_reserves, alpha=0.2, color='red', label='Potential Shortfall')

ax.set_xlabel('Policy Year', fontsize=11)
ax.set_ylabel('Reserve per Policy ($)', fontsize=11)
ax.set_title('Statutory vs Economic Reserves Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Interest rate sensitivity
ax = axes[0, 1]
rates_plot = np.array([r * 100 for r in interest_rates])
reserves_by_rate = []

for rate in interest_rates:
    res = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                       benefit_amount, gross_premium, rate)
    reserves_by_rate.append(res)

ax.plot(rates_plot, reserves_by_rate, linewidth=2.5, marker='o', markersize=6, color='steelblue')
ax.axvline(x=4.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Base (4%)')
ax.axvline(x=2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Stressed (2.5%)')

ax.set_xlabel('Interest Rate (%)', fontsize=11)
ax.set_ylabel('Economic Reserve ($)', fontsize=11)
ax.set_title(f'Reserve Sensitivity to Interest Rates (Year {year_test})', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Shortfall by duration
ax = axes[1, 0]
shortfalls = [d['shortfall'] for d in adequacy_data]
shortfall_pcts = [(d['shortfall'] / d['econ_res'] * 100) if d['econ_res'] > 0 else 0 
                 for d in adequacy_data]

colors_sf = ['red' if sf > 0 else 'green' for sf in shortfalls]

ax.bar(durations_plot, shortfalls, color=colors_sf, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Policy Year', fontsize=11)
ax.set_ylabel('Reserve Shortfall ($)', fontsize=11)
ax.set_title('Reserve Adequacy Gap: Economic - Held', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Stress scenario comparison
ax = axes[1, 1]
scenarios_stress = ['Base\n(4%, Std)', 'Interest\nStress\n(3%)', 'Mortality\nStress\n(+20%)', 'Combined\n(-1%, +20%)']
reserves_stress = [
    econ_res_base,
    calculate_reserve_prospective(age_test, remaining_test, mortality_standard, benefit_amount, gross_premium, 0.03),
    econ_res_stressed,
    econ_res_compound
]

bars = ax.bar(scenarios_stress, reserves_stress, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.axhline(y=econ_res_base, color='green', linestyle='--', linewidth=1.5, label='Base')

for bar, res in zip(bars, reserves_stress):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${res:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Reserve per Policy ($)', fontsize=11)
ax.set_title(f'Stress Testing Results (Year {year_test}, Age {age_test})', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reserve_adequacy.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

