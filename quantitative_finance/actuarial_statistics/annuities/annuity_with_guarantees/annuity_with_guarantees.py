# Auto-extracted from markdown file
# Source: annuity_with_guarantees.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Annuity with Guarantee Period Analysis ===\n")

# Build mortality table
def build_mortality_table():
    ages = np.arange(0, 121)
    A, B, C = 0.0001, 1.08, 0.00035
    mu_x = A + C * (B ** ages)
    q_x = 1 - np.exp(-mu_x)
    
    l_x = np.zeros(len(ages))
    l_x[0] = 100000
    for i in range(1, len(ages)):
        l_x[i] = l_x[i-1] * (1 - q_x[i-1])
    
    return pd.DataFrame({'Age': ages, 'l_x': l_x, 'q_x': q_x})

mortality = build_mortality_table()

# Annuity-certain
def annuity_certain(n, i):
    v = 1 / (1 + i)
    d = i / (1 + i)
    return (1 - v**n) / d

# Pure life annuity
def life_annuity(x, i, mortality):
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 0
    for k in range(1, 121 - x):
        l_future = mortality.loc[mortality['Age'] == x + k, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        k_p_x = l_future[0] / l_current
        value += (v ** k) * k_p_x
    
    return value

# Annuity with guarantee period
def annuity_with_guarantee(x, n_guarantee, i, mortality):
    """
    aₓ[n-year guarantee] = aₙ̄| + v^n · ₙpₓ · aₓ₊ₙ
    = Certain for n years + Deferred life annuity
    """
    v = 1 / (1 + i)
    
    # Part 1: n-year certain annuity
    certain_component = annuity_certain(n_guarantee, i)
    
    # Part 2: Deferred life annuity starting after n years
    if x + n_guarantee > 120:
        deferred_component = 0
    else:
        l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
        l_future = mortality.loc[mortality['Age'] == x + n_guarantee, 'l_x'].values[0]
        n_p_x = l_future / l_current
        
        a_x_plus_n = life_annuity(x + n_guarantee, i, mortality)
        deferred_component = (v ** n_guarantee) * n_p_x * a_x_plus_n
    
    return certain_component + deferred_component

# Calculate for different guarantee periods
print("=== Annuity Values (Age 65, i = 5%) ===\n")
age_base = 65
i_rate = 0.05
guarantee_periods = [0, 5, 10, 15, 20, 25]

results = []
for n_guar in guarantee_periods:
    if n_guar == 0:
        # Pure life annuity
        value = life_annuity(age_base, i_rate, mortality)
        certain = 0
        label = "Pure life (no guarantee)"
    else:
        value = annuity_with_guarantee(age_base, n_guar, i_rate, mortality)
        certain = annuity_certain(n_guar, i_rate)
        label = f"{n_guar}-year guarantee"
    
    pure_life_value = life_annuity(age_base, i_rate, mortality)
    premium_pct = (value / pure_life_value - 1) * 100 if n_guar > 0 else 0
    
    results.append({
        'Guarantee': label,
        'Value': value,
        'Certain Component': certain if n_guar > 0 else 0,
        'Premium vs Pure Life (%)': premium_pct,
        'Annual Payment ($1k)': 1000 / value if value > 0 else 0
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format='%.2f'))

# Decomposition analysis
print("\n=== Value Decomposition (Age 65, 10-Year Guarantee) ===\n")
n_guar_test = 10
certain_comp = annuity_certain(n_guar_test, i_rate)

v = 1 / (1 + i_rate)
l_current = mortality.loc[mortality['Age'] == age_base, 'l_x'].values[0]
l_future = mortality.loc[mortality['Age'] == age_base + n_guar_test, 'l_x'].values[0]
n_p_x = l_future / l_current
a_x_plus_n = life_annuity(age_base + n_guar_test, i_rate, mortality)
deferred_comp = (v ** n_guar_test) * n_p_x * a_x_plus_n

total_value = certain_comp + deferred_comp

print(f"Components:")
print(f"  Certain annuity ({n_guar_test} years): {certain_comp:.4f}")
print(f"  Deferred life annuity:")
print(f"    v^{n_guar_test}: {v**n_guar_test:.4f}")
print(f"    {n_guar_test}p₆₅: {n_p_x:.4f}")
print(f"    a₇₅: {a_x_plus_n:.4f}")
print(f"    Product: {deferred_comp:.4f}")
print(f"  Total value: {total_value:.4f}")
print(f"\nVerification: {abs(total_value - annuity_with_guarantee(age_base, n_guar_test, i_rate, mortality)) < 0.01}")

# Early death scenario
print("\n=== Early Death Scenario ===\n")
age_purchase = 65
guarantee = 10
death_age = 70  # Dies 5 years into guarantee

years_received = death_age - age_purchase
years_remaining = guarantee - years_received

v = 1 / (1 + i_rate)

# Beneficiary receives remaining guaranteed payments
beneficiary_pv = sum([v**(k+years_received) for k in range(1, years_remaining + 1)])

print(f"Purchaser age: {age_purchase}")
print(f"Guarantee period: {guarantee} years")
print(f"Death at age: {death_age} (after {years_received} years)")
print(f"Beneficiary receives: {years_remaining} remaining payments")
print(f"Present value to beneficiary: {beneficiary_pv:.4f}")
print(f"(Discounted from time of purchase)")

# Cost-benefit by health status
print("\n=== Value by Health Status (Age 65, 10-Year Guarantee) ===\n")

# Adjust mortality for health
def adjust_mortality_health(mortality, age, health_factor):
    """
    health_factor < 1: Better health (lower mortality)
    health_factor > 1: Worse health (higher mortality)
    """
    mortality_adjusted = mortality.copy()
    mortality_adjusted.loc[mortality_adjusted['Age'] >= age, 'q_x'] *= health_factor
    mortality_adjusted.loc[mortality_adjusted['Age'] >= age, 'q_x'] = \
        mortality_adjusted.loc[mortality_adjusted['Age'] >= age, 'q_x'].clip(upper=1.0)
    
    # Rebuild l_x
    l_x_new = mortality_adjusted['l_x'].values.copy()
    for i in range(age, len(l_x_new) - 1):
        l_x_new[i+1] = l_x_new[i] * (1 - mortality_adjusted.iloc[i]['q_x'])
    
    mortality_adjusted['l_x'] = l_x_new
    return mortality_adjusted

health_scenarios = {
    'Excellent (0.7x mortality)': 0.7,
    'Average (1.0x mortality)': 1.0,
    'Poor (1.5x mortality)': 1.5
}

guarantee_test = 10

print(f"10-year guarantee annuity value:")
print("Health Status | Pure Life | With Guarantee | Premium | Better Deal")
print("-" * 75)

for health_name, health_factor in health_scenarios.items():
    mort_adj = adjust_mortality_health(mortality, age_base, health_factor)
    
    pure_life = life_annuity(age_base, i_rate, mort_adj)
    with_guar = annuity_with_guarantee(age_base, guarantee_test, i_rate, mort_adj)
    premium_pct = (with_guar / pure_life - 1) * 100
    
    # Better deal: Higher value = more attractive
    # Poor health: Guarantee more valuable (protection against early death)
    # Excellent health: Pure life more valuable (mortality credit)
    better_deal = "Guarantee" if health_factor > 1.2 else "Pure life" if health_factor < 0.8 else "Similar"
    
    print(f"{health_name:24s} | {pure_life:9.2f} | {with_guar:14.2f} | {premium_pct:6.1f}% | {better_deal}")

# Age sensitivity
print("\n=== Guarantee Value by Issue Age ===\n")
ages_test = [55, 60, 65, 70, 75, 80]
guarantee_fixed = 10

print(f"{guarantee_fixed}-year guarantee annuity:")
print("Age | Pure Life | With Guarantee | Premium (%)")
print("-" * 50)

for age in ages_test:
    pure = life_annuity(age, i_rate, mortality)
    guar = annuity_with_guarantee(age, guarantee_fixed, i_rate, mortality)
    prem = (guar / pure - 1) * 100
    
    print(f"{age:3d} | {pure:9.2f} | {guar:14.2f} | {prem:10.1f}")

# Interest rate sensitivity
print("\n=== Interest Rate Sensitivity (Age 65, 10-Year Guarantee) ===\n")
interest_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

print("Interest | Pure Life | With Guarantee | Premium (%)")
print("-" * 55)

for i_val in interest_rates:
    pure = life_annuity(age_base, i_val, mortality)
    guar = annuity_with_guarantee(age_base, 10, i_val, mortality)
    prem = (guar / pure - 1) * 100
    
    print(f"{i_val*100:7.0f}%  | {pure:9.2f} | {guar:14.2f} | {prem:10.1f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Value by guarantee period
ax1 = axes[0, 0]
guarantees_plot = [0, 5, 10, 15, 20, 25, 30]
values_by_guarantee = []
certain_components = []

for n_g in guarantees_plot:
    if n_g == 0:
        values_by_guarantee.append(life_annuity(65, 0.05, mortality))
        certain_components.append(0)
    else:
        values_by_guarantee.append(annuity_with_guarantee(65, n_g, 0.05, mortality))
        certain_components.append(annuity_certain(n_g, 0.05))

ax1.plot(guarantees_plot, values_by_guarantee, 'o-', linewidth=2, markersize=8, label='Total value')
ax1.plot(guarantees_plot, certain_components, 's-', linewidth=2, markersize=6, label='Certain component')
ax1.set_xlabel('Guarantee Period (years)')
ax1.set_ylabel('Annuity Value')
ax1.set_title('Value Increases with Guarantee\n(Age 65, i = 5%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Premium over pure life
ax2 = axes[0, 1]
pure_life_val = life_annuity(65, 0.05, mortality)
premiums = [(val / pure_life_val - 1) * 100 for val in values_by_guarantee[1:]]

ax2.bar(guarantees_plot[1:], premiums, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Guarantee Period (years)')
ax2.set_ylabel('Premium vs Pure Life (%)')
ax2.set_title('Cost of Guarantee Protection')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Decomposition stacked bar
ax3 = axes[0, 2]
guarantees_decomp = [5, 10, 15, 20, 25]
certain_vals = [annuity_certain(n, 0.05) for n in guarantees_decomp]
deferred_vals = []

for n_g in guarantees_decomp:
    total_val = annuity_with_guarantee(65, n_g, 0.05, mortality)
    cert = annuity_certain(n_g, 0.05)
    deferred_vals.append(total_val - cert)

x_pos = np.arange(len(guarantees_decomp))
ax3.bar(x_pos, certain_vals, label='Certain component', alpha=0.7, edgecolor='black')
ax3.bar(x_pos, deferred_vals, bottom=certain_vals, label='Deferred life', alpha=0.7, edgecolor='black')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(guarantees_decomp)
ax3.set_xlabel('Guarantee Period')
ax3.set_ylabel('Value')
ax3.set_title('Decomposition: Certain + Deferred')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Age impact
ax4 = axes[1, 0]
ages_impact = np.arange(55, 81, 2)
pure_by_age = [life_annuity(age, 0.05, mortality) for age in ages_impact]
guar_10_by_age = [annuity_with_guarantee(age, 10, 0.05, mortality) for age in ages_impact]

ax4.plot(ages_impact, pure_by_age, 'o-', linewidth=2, label='Pure life', markersize=5)
ax4.plot(ages_impact, guar_10_by_age, 's-', linewidth=2, label='10-year guarantee', markersize=5)
ax4.set_xlabel('Issue Age')
ax4.set_ylabel('Annuity Value')
ax4.set_title('Value Decreases with Age\n(Both types)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Premium percentage by age
ax5 = axes[1, 1]
premium_by_age = [(guar_10_by_age[i] / pure_by_age[i] - 1) * 100 
                  for i in range(len(ages_impact))]

ax5.plot(ages_impact, premium_by_age, 'o-', linewidth=2, color='purple', markersize=6)
ax5.fill_between(ages_impact, 0, premium_by_age, alpha=0.2, color='purple')
ax5.set_xlabel('Issue Age')
ax5.set_ylabel('Premium vs Pure Life (%)')
ax5.set_title('Guarantee Premium Increases with Age\n(10-year guarantee)')
ax5.grid(True, alpha=0.3)

# Plot 6: Interest rate impact on premium
ax6 = axes[1, 2]
i_range = np.linspace(0.02, 0.08, 20)
premiums_by_i = []

for i_val in i_range:
    pure = life_annuity(65, i_val, mortality)
    guar = annuity_with_guarantee(65, 10, i_val, mortality)
    prem_pct = (guar / pure - 1) * 100
    premiums_by_i.append(prem_pct)

ax6.plot(i_range * 100, premiums_by_i, linewidth=2)
ax6.fill_between(i_range * 100, 0, premiums_by_i, alpha=0.2)
ax6.set_xlabel('Interest Rate (%)')
ax6.set_ylabel('Premium vs Pure Life (%)')
ax6.set_title('Guarantee Premium vs Interest Rate\n(Age 65, 10-year guarantee)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print("Annuity with guarantee: Pays for n years minimum, then continues if alive")
print("Formula: aₓ[n-guar] = aₙ̄| + v^n·ₙpₓ·aₓ₊ₙ")
print("Trade-off: Higher cost for downside protection; valuable for poor health/beneficiary protection")

