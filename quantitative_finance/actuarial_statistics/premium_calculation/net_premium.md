# Net Premium

## 1. Concept Skeleton
**Definition:** Pₙ = actuarial cost of insurance coverage; E[Benefits] = E[Premiums], zero expected profit pricing basis  
**Purpose:** Foundation for pricing; separates actuarial cost from expense loadings and profit margins; regulatory minimum for reserve calculations  
**Prerequisites:** Present value of benefits (Aₓ), annuity-due factor (äₙ̄|), mortality probability (qₓ)

## 2. Comparative Framing
| Premium Type | Formula | Purpose | Profit | Target |
|--------------|---------|---------|--------|--------|
| **Net Premium** | Pₙ = APV(Benefits) / APV(Premiums) | Actuarial equivalence | 0% | Cost-only basis |
| **Gross Premium** | Pₘ = Pₙ + Load | Revenue for all costs | Yes | Full pricing |
| **Modified Premium** | P = [Pₙ + adjustment] | First-year smoothing | Partial | Reality-based |
| **Expense-loaded** | P = Pₙ + per-policy + % | Segment-specific | Yes | Product-level |
| **Market Premium** | P = competitor rates | Competitively driven | Variable | Market clearing |

## 3. Examples + Counterexamples

**Simple Example:**  
20-year term, age 35, $200K: APV = $6,200, Annuity factor = 18.1 → Net premium ≈ $342/year

**Failure Case:**  
Using net premium as price to customer: Company loses if actual mortality/lapses exceed assumptions; insufficient reserves to pay claims

**Edge Case:**  
Net premium at age 80 (high mortality): Approaches 20% of benefit annually; gross premium may be 50%+ (for expenses/profit)

## 4. Layer Breakdown
```
Net Premium Structure:
├─ Definition & Formula:
│   ├─ Discrete: Pₙ = Aₓ:n̄| / äₙ̄|  (benefit APV / premium APV)
│   ├─ Continuous: Pₙ = Āₓ:n̄| / ā̅ₓ:n̄|
│   ├─ Lifetime: Pₙ = Āₓ / ä̅ₓ  (whole life)
│   ├─ Principle: E[Premiums] = E[Benefits]  (actuarial equivalence)
│   └─ Notation: Pₓ:n̄| for net premium, Pₓ for lifetime
├─ Components of Calculation:
│   ├─ Numerator: APV(Benefits) = Aₓ:n̄| = ∑ v^k · ₖ₋₁pₓ · qₓ₊ₖ₋₁
│   ├─ Denominator: APV(Premiums) = äₙ̄| = ∑ v^k · ₖpₓ  (annuity due)
│   ├─ Mortality assumption: Life table (select/ultimate)
│   ├─ Interest assumption: Discount rate (typically 3-5%)
│   └─ Timing: Premiums start immediately; benefits at each possible death
├─ Variations by Product:
│   ├─ Term insurance Pₓ:n̄|: Simple; high probability of no payout
│   ├─ Whole life Pₓ: Complex; benefit certain eventually
│   ├─ Endowment: Hybrid; death + maturity benefit
│   ├─ Annuity: PV premiums > PV benefits (customer pays upfront)
│   ├─ Disability income: Multiple decrements (death, recovery, other)
│   └─ Long-term care: Stochastic duration (uncapped liability)
├─ Actuarial Assumptions:
│   ├─ Mortality basis: Standard, Preferred, Smoker, Substandard tables
│   ├─ Select period: First 5-15 years (higher mortality for impaired)
│   ├─ Ultimate rates: Rates after select period
│   ├─ Lapse rate assumptions: % of policies terminating annually
│   ├─ Interest rate: Risk-free + margin (4% typical for US)
│   └─ Expense assumption: Embedded in loading, not in net premium
├─ Regulatory Treatment:
│   ├─ Statutory net premium: Conservative (higher mortality, lower interest)
│   ├─ GAAP valuation: Best estimate assumptions
│   ├─ Solvency II: Risk margin added to best estimate
│   ├─ Minimum reserve: Based on net premium (conservative)
│   └─ Reserve requirement: V ≥ max(statutory net premium reserve, deficiency)
├─ Sensitivity Analysis:
│   ├─ Mortality sensitivity: 10% worse → Pₙ increases ~3-5% (term) to ~8-10% (whole life)
│   ├─ Interest sensitivity: 1% lower → Pₙ increases ~5-10% (term) to ~15-20% (whole life)
│   ├─ Lapse sensitivity: If assume 5% lapse → Pₙ decreases ~2-3% (benefits not paid)
│   └─ Combined shock: All three 20% worse → Pₙ could increase 15-40%
└─ Practical Considerations:
    ├─ Profit margin: Gross = Net × 1.25 to 1.35 (25-35% load for expenses/profit)
    ├─ First-year drain: Acquisition expenses 15-25% of premium; recovered over 5-10 years
    ├─ Persistency impact: Premium inadequate if lapse rate > assumption
    ├─ Medical underwriting: Smoker/preferred adjustments 1.5-2.5× standard
    └─ Antiselection: Higher benefit amounts attract riskier applicants
```

**Key Decision:** Conservative assumptions → Higher net premium → Comfortable surplus; Liberal assumptions → Lower premium → Competitive but riskier

## 5. Mini-Project
Calculate net premiums for various products, perform sensitivity analysis, and compare to market pricing:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize

# 1. SETUP & ASSUMPTIONS
print("=" * 80)
print("NET PREMIUM CALCULATION & ACTUARIAL EQUIVALENCE")
print("=" * 80)

# Mortality table (standard basis)
mortality_data = {
    25: 0.00067, 30: 0.00084, 35: 0.00103, 40: 0.00131, 45: 0.00172,
    50: 0.00233, 55: 0.00325, 60: 0.00459, 65: 0.00653, 70: 0.00933,
    75: 0.01330, 80: 0.01934, 85: 0.02873, 90: 0.04213
}

# Smoker mortality multiplier
smoker_multiplier = 2.0

# Preferred mortality discount
preferred_discount = 0.80

# Interest assumptions
standard_rate = 0.04  # 4% standard
conservative_rate = 0.03  # 3% conservative
optimistic_rate = 0.05  # 5% optimistic

v_standard = 1 / (1 + standard_rate)

print(f"\nAssumptions:")
print(f"  Mortality Table: US Standard 2020")
print(f"  Interest Rate (standard): {standard_rate*100:.1f}%")
print(f"  Smoker multiplier: {smoker_multiplier}×")
print(f"  Preferred discount: {preferred_discount}×\n")

def gompertz_mortality(age, A=0.0001, B=1.075):
    """Extrapolate mortality beyond table"""
    return A * (B ** age)

def calculate_apv_benefits(start_age, term_length, mortality_dict, interest_rate, 
                          benefit_amount=1.0):
    """Calculate APV of insurance benefits"""
    
    v_rate = 1 / (1 + interest_rate)
    apv = 0
    kpx = 1.0
    
    for k in range(1, term_length + 1):
        age_k = start_age + k - 1
        
        if age_k in mortality_dict:
            qx_k = mortality_dict[age_k]
        else:
            mu = gompertz_mortality(age_k)
            qx_k = 1 - np.exp(-mu)
        
        vk = v_rate ** k
        apv += kpx * qx_k * vk * benefit_amount
        
        px_k = 1 - qx_k
        kpx *= px_k
    
    return apv

def calculate_apv_premiums(start_age, term_length, mortality_dict, interest_rate):
    """Calculate APV of premium annuity-due"""
    
    v_rate = 1 / (1 + interest_rate)
    apv_prem = 0
    kpx = 1.0
    
    for k in range(0, term_length):
        vk = v_rate ** k
        apv_prem += kpx * vk
        
        if k < term_length - 1:
            age_k = start_age + k
            
            if age_k in mortality_dict:
                qx_k = mortality_dict[age_k]
            else:
                mu = gompertz_mortality(age_k)
                qx_k = 1 - np.exp(-mu)
            
            px_k = 1 - qx_k
            kpx *= px_k
    
    return apv_prem

# 2. NET PREMIUM CALCULATION (STANDARD BASIS)
print("=" * 80)
print("NET PREMIUM: STANDARD 4% ASSUMPTIONS")
print("=" * 80)

start_age = 40
benefit = 250000

# Term insurance (20 years)
term_years = 20
apv_benefits_term = calculate_apv_benefits(start_age, term_years, mortality_data, standard_rate, benefit)
apv_premiums_term = calculate_apv_premiums(start_age, term_years, mortality_data, standard_rate)

net_premium_term = apv_benefits_term / apv_premiums_term

print(f"\n20-Year Term, Age {start_age}, ${benefit:,.0f}\n")
print(f"{'Metric':<35} {'Amount':<20}")
print("-" * 55)
print(f"{'APV of Benefits':<35} ${apv_benefits_term:>18,.2f}")
print(f"{'APV of Premiums (annuity-due)':<35} {apv_premiums_term:>19.6f}")
print(f"{'Net Annual Premium':<35} ${net_premium_term:>18,.2f}")
print(f"{'Net Monthly Premium':<35} ${net_premium_term/12:>18,.2f}")
print()

# Whole Life
apv_benefits_wl = 0
kpx_wl = 1.0

for k in range(1, 121 - start_age):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        mu = gompertz_mortality(age_k)
        qx_k = 1 - np.exp(-mu)
    
    vk = v_standard ** k
    apv_benefits_wl += kpx_wl * qx_k * vk * benefit
    
    if apv_benefits_wl >= apv_benefits_wl + kpx_wl * qx_k * vk * benefit - 1e-6:
        if vk < 1e-10:
            break
    
    px_k = 1 - qx_k
    kpx_wl *= px_k

apv_premiums_wl = 0
kpx_ann = 1.0

for k in range(0, 121 - start_age):
    vk = v_standard ** k
    apv_premiums_wl += kpx_ann * vk
    
    if k < 120 - start_age:
        age_k = start_age + k
        
        if age_k in mortality_data:
            qx_k = mortality_data[age_k]
        else:
            mu = gompertz_mortality(age_k)
            qx_k = 1 - np.exp(-mu)
        
        px_k = 1 - qx_k
        kpx_ann *= px_k
    
    if kpx_ann < 1e-10:
        break

net_premium_wl = apv_benefits_wl / apv_premiums_wl

print(f"Whole Life, Age {start_age}, ${benefit:,.0f}\n")
print(f"{'Metric':<35} {'Amount':<20}")
print("-" * 55)
print(f"{'APV of Benefits':<35} ${apv_benefits_wl:>18,.2f}")
print(f"{'APV of Premiums (annuity-due)':<35} {apv_premiums_wl:>19.6f}")
print(f"{'Net Annual Premium':<35} ${net_premium_wl:>18,.2f}")
print(f"{'Net Monthly Premium':<35} ${net_premium_wl/12:>18,.2f}")
print()

# 3. UNDERWRITING CLASSES (SMOKER vs PREFERRED vs STANDARD)
print("=" * 80)
print("NET PREMIUM BY UNDERWRITING CLASS")
print("=" * 80)

classes = {
    'Standard': {'smoker': 1.0, 'preferred': 1.0},
    'Smoker': {'smoker': smoker_multiplier, 'preferred': 1.0},
    'Preferred': {'smoker': 1.0, 'preferred': preferred_discount}
}

print(f"\n20-Year Term, Age {start_age}, ${benefit:,.0f}\n")
print(f"{'Class':<20} {'Mortality':<15} {'Annual Prem':<18} {'Monthly Prem':<18} {'% vs Standard':<15}")
print("-" * 90)

standard_annual = 0

for class_name, adjustments in classes.items():
    mortality_adjusted = {age: qx * adjustments['smoker'] * adjustments['preferred'] 
                         for age, qx in mortality_data.items()}
    
    apv_b = calculate_apv_benefits(start_age, term_years, mortality_adjusted, standard_rate, benefit)
    apv_p = calculate_apv_premiums(start_age, term_years, mortality_adjusted, standard_rate)
    
    net_prem = apv_b / apv_p
    
    if class_name == 'Standard':
        standard_annual = net_prem
    
    pct_vs_standard = (net_prem / standard_annual) * 100 if standard_annual > 0 else 100
    
    mortality_adjust_pct = (adjustments['smoker'] * adjustments['preferred'] - 1) * 100
    
    print(f"{class_name:<20} {mortality_adjust_pct:>+13.0f}% ${net_prem:>16,.2f} ${net_prem/12:>16,.2f} {pct_vs_standard:>13.1f}%")

print()

# 4. SENSITIVITY ANALYSIS
print("=" * 80)
print("SENSITIVITY ANALYSIS: NET PREMIUM")
print("=" * 80)

print(f"\n20-Year Term, Age {start_age}, ${benefit:,.0f}\n")

# Mortality sensitivity
print("Mortality Impact:\n")
print(f"{'Mortality Scenario':<30} {'Annual Premium':<20} {'% vs Base':<15}")
print("-" * 65)

mortality_scenarios = {
    'Base (100%)': 1.0,
    'Optimistic (80%)': 0.80,
    'Pessimistic (120%)': 1.20,
    'Severe (150%)': 1.50
}

base_premium = standard_annual

for scenario, mult in mortality_scenarios.items():
    mort_test = {age: qx * mult for age, qx in mortality_data.items()}
    apv_b_test = calculate_apv_benefits(start_age, term_years, mort_test, standard_rate, benefit)
    apv_p_test = calculate_apv_premiums(start_age, term_years, mort_test, standard_rate)
    
    prem_test = apv_b_test / apv_p_test
    pct_vs_base = ((prem_test / base_premium) - 1) * 100
    
    print(f"{scenario:<30} ${prem_test:>18,.2f} {pct_vs_base:>+13.1f}%")

print()

# Interest rate sensitivity
print("Interest Rate Impact:\n")
print(f"{'Interest Rate':<30} {'Annual Premium':<20} {'% vs 4%':<15}")
print("-" * 65)

interest_scenarios = {
    '2%': 0.02,
    '3%': 0.03,
    '4% (Base)': 0.04,
    '5%': 0.05,
    '6%': 0.06
}

for rate_label, rate_val in interest_scenarios.items():
    apv_b_int = calculate_apv_benefits(start_age, term_years, mortality_data, rate_val, benefit)
    apv_p_int = calculate_apv_premiums(start_age, term_years, mortality_data, rate_val)
    
    prem_int = apv_b_int / apv_p_int
    pct_vs_4 = ((prem_int / base_premium) - 1) * 100
    
    print(f"{rate_label:<30} ${prem_int:>18,.2f} {pct_vs_4:>+13.1f}%")

print()

# Combined shock
print("Combined Shock (Mortality +20%, Interest -1%, Lapse -2%):\n")

mort_shock = {age: qx * 1.20 for age, qx in mortality_data.items()}
apv_b_shock = calculate_apv_benefits(start_age, term_years, mort_shock, 0.03, benefit)
apv_p_shock = calculate_apv_premiums(start_age, term_years, mort_shock, 0.03)

prem_shock = apv_b_shock / apv_p_shock
pct_shock = ((prem_shock / base_premium) - 1) * 100

print(f"Base premium: ${base_premium:,.2f}/year")
print(f"Shocked premium: ${prem_shock:,.2f}/year")
print(f"Increase: {pct_shock:+.1f}%")
print()

# 5. GROSS VS NET PREMIUM
print("=" * 80)
print("GROSS PREMIUM = NET PREMIUM + LOADINGS")
print("=" * 80)

# Typical loading structure
acquisition_rate = 0.15  # 15% of gross premium
maintenance_cost = 30    # $30/year fixed
profit_rate = 0.20       # 20% of net premium

print(f"\n20-Year Term, Age {start_age}, ${benefit:,.0f}\n")
print(f"{'Component':<35} {'Amount':<20}")
print("-" * 55)

# Approximate gross premium calculation
# Gross = (Net + Acquisition% + Maintenance + Profit) / (1 - Acq%)
# Simplified: Gross ≈ Net × (1 + Load Factor)

load_factor = 0.30  # Assume 30% total load
gross_premium_approx = base_premium * (1 + load_factor)

print(f"{'Net Premium (actuarial cost)':<35} ${base_premium:>18,.2f}")
print(f"{'Acquisition expense (15% Y1)':<35} ${base_premium * acquisition_rate:>18,.2f}")
print(f"{'Maintenance cost':<35} ${maintenance_cost:>18,.2f}")
print(f"{'Profit margin (20%)':<35} ${base_premium * profit_rate:>18,.2f}")
print()
print(f"{'Load factor (combined)':<35} {load_factor*100:>18.1f}%")
print(f"{'Gross Annual Premium':<35} ${gross_premium_approx:>18,.2f}")
print(f"{'Gross Monthly Premium':<35} ${gross_premium_approx/12:>18,.2f}")
print()

# 6. NET PREMIUM RESERVE
print("=" * 80)
print("NET PREMIUM RESERVE: RESERVE ADEQUACY")
print("=" * 80)

print(f"\n20-Year Term, Age {start_age}\n")
print(f"{'Year':<8} {'Reserve':<15} {'% of Benefit':<15}")
print("-" * 38)

reserve_year_1 = (apv_benefits_term / apv_premiums_term) - base_premium  # First year reserve

for year in [1, 5, 10, 15, 20]:
    # Simplified reserve calculation (prospective method)
    remaining_term = term_years - year + 1
    
    apv_b_remaining = calculate_apv_benefits(start_age + year - 1, remaining_term, 
                                            mortality_data, standard_rate, benefit)
    apv_p_remaining = calculate_apv_premiums(start_age + year - 1, remaining_term, 
                                            mortality_data, standard_rate)
    
    premium_in_force = base_premium
    pv_future_premiums = premium_in_force * apv_p_remaining
    
    reserve = apv_b_remaining - pv_future_premiums
    reserve_pct = (reserve / benefit) * 100
    
    print(f"{year:<8} ${reserve:>13,.0f} {reserve_pct:>13.1f}%")

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Net premium by age and term
ax = axes[0, 0]
ages_plot = list(range(25, 66, 5))
terms_plot = [10, 20, 30]

for term in terms_plot:
    premiums = []
    
    for age in ages_plot:
        apv_b = calculate_apv_benefits(age, term, mortality_data, standard_rate, benefit)
        apv_p = calculate_apv_premiums(age, term, mortality_data, standard_rate)
        prem = apv_b / apv_p
        premiums.append(prem / 12)  # Monthly
    
    ax.plot(ages_plot, premiums, linewidth=2.5, marker='o', markersize=6, label=f'{term}-Year')

ax.set_xlabel('Age at Issue', fontsize=11)
ax.set_ylabel('Monthly Net Premium ($)', fontsize=11)
ax.set_title(f'Net Premium by Age & Term (${benefit/1000:.0f}K Benefit)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Sensitivity - Mortality & Interest
ax = axes[0, 1]
mort_range = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
int_range = np.array([0.02, 0.03, 0.04, 0.05, 0.06])

# Heatmap-style
premiums_sensitivity = []

for rate in int_range:
    row = []
    for mult in mort_range:
        mort_adj = {age: qx * mult for age, qx in mortality_data.items()}
        apv_b = calculate_apv_benefits(start_age, term_years, mort_adj, rate, benefit)
        apv_p = calculate_apv_premiums(start_age, term_years, mort_adj, rate)
        prem = apv_b / apv_p
        row.append(prem)
    premiums_sensitivity.append(row)

im = ax.imshow(premiums_sensitivity, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(mort_range)))
ax.set_yticks(range(len(int_range)))
ax.set_xticklabels([f'{m:.0%}' for m in mort_range], fontsize=9)
ax.set_yticklabels([f'{i:.1%}' for i in int_range], fontsize=9)
ax.set_xlabel('Mortality Assumption', fontsize=11)
ax.set_ylabel('Interest Rate', fontsize=11)
ax.set_title('Net Premium Sensitivity Analysis', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(int_range)):
    for j in range(len(mort_range)):
        text = ax.text(j, i, f'${premiums_sensitivity[i][j]:.0f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax, label='Annual Premium ($)')

# Plot 3: Class comparisons
ax = axes[1, 0]
class_names = ['Standard', 'Smoker', 'Preferred']
class_prems = []

for class_name in class_names:
    adjustments = classes[class_name]
    mort_adj = {age: qx * adjustments['smoker'] * adjustments['preferred'] 
               for age, qx in mortality_data.items()}
    
    apv_b = calculate_apv_benefits(start_age, term_years, mort_adj, standard_rate, benefit)
    apv_p = calculate_apv_premiums(start_age, term_years, mort_adj, standard_rate)
    
    prem = apv_b / apv_p
    class_prems.append(prem / 12)  # Monthly

bars = ax.bar(class_names, class_prems, color=['steelblue', 'red', 'green'], 
             alpha=0.6, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Monthly Net Premium ($)', fontsize=11)
ax.set_title(f'Net Premium by Underwriting Class (20-Yr Term)', fontsize=12, fontweight='bold')

for bar, prem in zip(bars, class_prems):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${prem:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Net vs Gross premium breakdown
ax = axes[1, 1]
components = ['Net Premium', 'Acquisition\n(15%)', 'Maintenance\n($30/yr)', 'Profit\n(20%)', 'Gross\nPremium']
amounts = [
    base_premium,
    base_premium * acquisition_rate,
    maintenance_cost,
    base_premium * profit_rate,
    gross_premium_approx
]
colors_comp = ['green', 'orange', 'yellow', 'red', 'darkblue']

cumulative = 0
for i, (comp, amt) in enumerate(zip(components[:-1], amounts[:-1])):
    ax.bar(i, amt, bottom=cumulative, color=colors_comp[i], alpha=0.6, edgecolor='black', linewidth=1.5)
    cumulative += amt

ax.bar(4, gross_premium_approx, color=colors_comp[4], alpha=0.6, edgecolor='black', linewidth=2)

ax.set_ylabel('Annual Premium ($)', fontsize=11)
ax.set_title('Net Premium Build to Gross Premium', fontsize=12, fontweight='bold')
ax.set_xticks(range(5))
ax.set_xticklabels(components, fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('net_premium_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When net premium assumptions break down:
- **Mortality worse than table**: If actual deaths exceed assumption, reserve inadequate; deficit accumulates in force
- **Interest rates fall**: If 4% assumption but actual 2%, reserves insufficient; company must add capital
- **Lapse rate varies**: If assumed 5% but actual 2%, fewer policies offset deaths (or vice versa); reserve mismatch
- **Antiselection at issue**: Sicker applicants more likely to buy; actual mortality exceeds "standard" table
- **Underwriting correlation**: Smokers who quit may not be captured in smoker rate; dual-smoker classification error
- **Mass lapse event**: Economic downturn; policyholders drop en masse, breaking lapse assumption; loss of persistent premium

## 7. Key References
- [Bowers et al., Actuarial Mathematics (Chapter 2-3)](https://www.soa.org/) - Net premium formulations
- [SOA Exam FM Premium Calculation](https://www.soa.org/education/exam-req/edu-exam-fm-detail.aspx) - Practice problems
- [Actuarial Standards of Practice (ASOP 35)](https://www.soa.org/standards/) - Pricing practice guidance

---
**Status:** Foundational pricing concept | **Complements:** Gross Premium, Premium Reserves, Profit Testing
