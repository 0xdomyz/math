# Present Value of Benefits

## 1. Concept Skeleton
**Definition:** Aₓ = E[Z] = present value of death benefit random variable; actuarial present value discounting expected death payouts  
**Purpose:** Core valuation principle for life insurance; basis for premium calculation, reserve setting, and product pricing  
**Prerequisites:** Discount factor (v^n), survival probability (ₚₓ), force of mortality (μₓ)

## 2. Comparative Framing
| Benefit Structure | Formula | Discounting | Mortality Weight | Use Case |
|------------------|---------|-------------|------------------|----------|
| **Fixed benefit, fixed term** | Aₓ:n̄| = ∑ᵏ₌₁ⁿ v^k · qₓ₊ₖ₋₁ | v^k for k periods | Annual ₚₓ | Term insurance pricing |
| **Fixed benefit, lifetime** | Āₓ = ∑ₖ₌₁^∞ v^k · ₖpₓ · μₓ₊ₖ | Continuous v^t | Force μₓ₊ₜ (moments) | Whole life pricing |
| **Continuous payment, no term** | Āₓ = ∫₀^∞ e^{-δt} · ₜpₓ · μₓ₊ₜ dt | e^{-δt} | Continuous hazard | Theoretical perpetual |
| **Increasing benefit (linear)** | (IA)ₓ:n̄| = ∑ᵏ₌₁ⁿ k·v^k · ₖpₓ · μₓ₊ₖ | v^k escalating | Weighted by year | Insurance + savings hybrid |
| **Decreasing benefit** | (DA)ₓ:n̄| = ∑ᵏ₌₁ⁿ (n+1-k)·v^k · qₓ₊ₖ₋₁ | v^k declining | Early deaths heavier | Mortgage-tied life insurance |

## 3. Examples + Counterexamples

**Simple Example:**  
$100K 10-year term, age 40, i=5%, mortality realistic: APV ≈ $4,200 (4.2% of benefit because most survive decade)

**Failure Case:**  
Using PV of annuity-certain formula (ignoring mortality): 100K × a₁₀̄| ≈ $100K × 7.72 ≈ $772K (absurdly high; assumes 100% death probability each year)

**Edge Case:**  
Whole life (n→∞) at age 85: APV approaches B × v^1 (typically ≈ 0.95B) because remaining life expectancy ≈ 6 years; almost all value concentrated in years 1-2

## 4. Layer Breakdown
```
Present Value of Benefits Structure:
├─ Definition & Core Formula:
│   ├─ Discrete: Aₓ:n̄| = ∑ᵏ₌₁ⁿ v^k · qₓ₊ₖ₋₁  (year-end death)
│   ├─ Continuous: Āₓ = ∫₀^∞ v^t · ₜpₓ · μₓ₊ₜ dt  (force of mortality)
│   ├─ Relationship: Āₓ ≈ Aₓ for small forces (Taylor expansion)
│   └─ Single-payment: APV = Benefit × ∑ probability_discounted
├─ Components of Valuation:
│   ├─ Discount factor v^k = 1/(1+i)^k (time-value)
│   ├─ Survival to age k: ₖpₓ = ₚₓ · ₚₓ₊₁ · ... · ₚₓ₊ₖ₋₁
│   ├─ Death probability in year k: qₓ₊ₖ₋₁ = 1 - ₚₓ₊ₖ₋₁
│   └─ Force of mortality: μₓ = -d/dx ln(ₚₓ)  (instantaneous rate)
├─ Term Structures:
│   ├─ Term n-year: Aₓ:n̄| (death within n years only)
│   ├─ Whole life: Āₓ = lim{n→∞} Aₓ:n̄|  (perpetual)
│   ├─ Endowment: ₙEₓ (probability survive n years) + Aₓ:n̄| (death before)
│   ├─ Deferred n-years: ₙ|Āₓ = v^n · Aₓ₊ₙ  (coverage starts at n)
│   └─ Temporary: ₙ̄Āₓ = Aₓ:n̄|  (coverage for n years only)
├─ Approximations for Practitioners:
│   ├─ Annual calculation: Aₓ:n̄| ≈ ∑ᵏ₌₁ⁿ v^{k-0.5} · qₓ₊ₖ₋₁  (mid-year death assumption)
│   ├─ UDD (uniform distribution of deaths): ₚₓ₊ₜ ≈ ₚₓ · (1 - t·qₓ)  for 0≤t≤1
│   ├─ Constant force between int: μₓ₊ₜ = constant  (exponential survival)
│   └─ Mortality-interest commutation: Dₓ = v^x · lₓ  (actuarial commutation functions)
├─ Multi-Decrement Extension:
│   ├─ Multiple causes: Aₓ:n̄|^(j) = ∑ qₓ₊ₖ₋₁^(j) · v^k  (specific cause)
│   ├─ Competing risks: qₓ = qₓ^(death) + qₓ^(lapse) + qₓ^(other)
│   └─ Service table: Combined decrements (death, withdrawal, retirement)
└─ Regulatory Treatment:
    ├─ Statutory minimum: Conservative assumptions (select mortality tables)
    ├─ GAAP / IFRS 17: Best estimate + margin for adverse variation
    ├─ Solvency II: Economic value with risk margin
    └─ Longevity hedging: Reinsurance or financial instruments to offset risk
```

**Interaction:** Benefit structure chosen → Mortality table applied → Discount rate selected → Sum present values of payouts weighted by probabilities

## 5. Mini-Project
Calculate APV for term, whole life, endowment, and varying benefit structures:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# 1. MORTALITY TABLE & INTEREST SETUP
print("=" * 80)
print("PRESENT VALUE OF BENEFITS: LIFE INSURANCE VALUATION")
print("=" * 80)

# Standard US Mortality (simplified illustrative)
ages = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])
# qx values (death probability during year)
qx_standard = np.array([0.00084, 0.00103, 0.00131, 0.00172, 0.00233, 0.00325, 0.00459, 0.00653, 0.00933, 0.01330, 0.01934, 0.02873, 0.04213])

# Calculate survival probabilities
px_standard = 1 - qx_standard

# Parametric extension (Gompertz) for ages beyond table
def gompertz_force_mortality(x, A=0.0001, B=1.075):
    """Force of mortality μₓ = A·B^x"""
    return A * (B ** x)

# Interest rate
annual_rate = 0.04
v = 1 / (1 + annual_rate)  # Discount factor

print(f"\nAssumptions:")
print(f"  Mortality Table: US 2020 (illustrated)")
print(f"  Interest Rate: {annual_rate*100:.1f}%")
print(f"  Discount Factor v: {v:.6f}")
print(f"\nMortality by Age:")
print(f"{'Age':<8} {'qₓ':<12} {'ₚₓ':<12}")
print("-" * 32)

for i, age in enumerate(ages):
    print(f"{age:<8} {qx_standard[i]:<12.6f} {px_standard[i]:<12.6f}")

print()

# 2. TERM INSURANCE: 10-YEAR TERM AT AGE 40
print("=" * 80)
print("TERM INSURANCE: 10-YEAR TERM, AGE 40, $100,000 BENEFIT")
print("=" * 80)

benefit = 100000
start_age = 40
n_years = 10
v_discount = v

# Find index for age 40
idx_start = np.where(ages == start_age)[0][0]

apv_term = 0
print(f"\n{'Year':<8} {'Age':<8} {'ₖpₓ':<12} {'qₓ₊ₖ₋₁':<12} {'v^k':<12} {'Payment':<15} {'PV':<15}")
print("-" * 92)

kpx_running = 1.0  # k-year survival probability

for k in range(1, n_years + 1):
    age_at_death = start_age + k - 1
    
    # Find mortality rate for age_at_death
    if age_at_death < ages[-1]:
        idx = np.where(ages == age_at_death)[0]
        if len(idx) > 0:
            qx_k = qx_standard[idx[0]]
        else:
            # Linear interpolation for ages between table values
            qx_k = np.interp(age_at_death, ages, qx_standard)
    else:
        qx_k = gompertz_force_mortality(age_at_death)
    
    # Discount factor
    vk = v_discount ** k
    
    # PV of benefit paid at end of year k (given death in year k)
    pv_benefit = kpx_running * qx_k * vk * benefit
    apv_term += pv_benefit
    
    print(f"{k:<8} {age_at_death:<8} {kpx_running:<12.6f} {qx_k:<12.6f} {vk:<12.6f} ${benefit:<14,.0f} ${pv_benefit:<14,.2f}")
    
    # Update survival probability for next iteration
    px_k = 1 - qx_k
    kpx_running *= px_k

print(f"\nAPV of 10-Year Term Insurance: ${apv_term:,.2f}")
print(f"  As % of benefit: {apv_term/benefit*100:.2f}%")
print()

# 3. WHOLE LIFE INSURANCE: AGE 40, $100,000 BENEFIT
print("=" * 80)
print("WHOLE LIFE INSURANCE: AGE 40, $100,000 BENEFIT")
print("=" * 80)

# Extend mortality table to age 120 using Gompertz
ages_extended = np.arange(start_age, 121)
qx_extended = []

for age in ages_extended:
    if age in ages:
        idx = np.where(ages == age)[0][0]
        qx_extended.append(qx_standard[idx])
    else:
        # Use Gompertz for extrapolation
        mu = gompertz_force_mortality(age, A=0.0001, B=1.075)
        # Convert force to annual probability: qₓ ≈ 1 - exp(-μₓ)
        qx_extended.append(1 - np.exp(-mu))

qx_extended = np.array(qx_extended)

# Calculate APV for whole life
apv_whole_life = 0
kpx = 1.0

print(f"\n{'Year':<8} {'Age':<8} {'qₓ':<12} {'v^k':<12} {'Contribution':<15}")
print("-" * 67)

for k in range(1, min(len(ages_extended), 61)):  # Cap at 60 years (minimal contribution after)
    age_death = start_age + k - 1
    idx_ext = age_death - start_age
    
    qx_k = qx_extended[idx_ext]
    vk = v_discount ** k
    
    contribution = kpx * qx_k * vk * benefit
    apv_whole_life += contribution
    
    if k <= 10 or k % 5 == 0:  # Print first 10 years + every 5th
        print(f"{k:<8} {age_death:<8} {qx_k:<12.6f} {vk:<12.6f} ${contribution:<14,.2f}")
    
    # Update survival
    px_k = 1 - qx_k
    kpx *= px_k

print(f"... (continued beyond year 10)")
print(f"\nAPV of Whole Life Insurance: ${apv_whole_life:,.2f}")
print(f"  As % of benefit: {apv_whole_life/benefit*100:.2f}%")
print()

# 4. ENDOWMENT INSURANCE: 20-YEAR ENDOWMENT
print("=" * 80)
print("ENDOWMENT INSURANCE: 20-YEAR ENDOWMENT, AGE 40, $100,000")
print("=" * 80)

n_endow = 20
# APV = [benefit paid on death within n years] + [benefit paid if survive n years]

# Death benefit component
apv_death_endow = 0
kpx_e = 1.0

for k in range(1, n_endow + 1):
    age_at_death = start_age + k - 1
    idx_ext = age_at_death - start_age
    
    qx_k = qx_extended[idx_ext]
    vk = v_discount ** k
    
    pv_death_benefit = kpx_e * qx_k * vk * benefit
    apv_death_endow += pv_death_benefit
    
    px_k = 1 - qx_k
    kpx_e *= px_k

# Survival benefit component (maturity benefit)
survival_prob_n = kpx_e  # Probability of surviving n years
vn = v_discount ** n_endow
pv_survival_benefit = survival_prob_n * vn * benefit

apv_endowment = apv_death_endow + pv_survival_benefit

print(f"\nBenefit Structure:")
print(f"  Death benefit (if death within {n_endow} years): ${benefit:,.0f}")
print(f"  Maturity benefit (if survive {n_endow} years): ${benefit:,.0f}")
print(f"\nCalculation:")
print(f"  APV of death benefits: ${apv_death_endow:,.2f}")
print(f"  APV of survival benefit: ${pv_survival_benefit:,.2f}")
print(f"  Total APV: ${apv_endowment:,.2f}")
print(f"    As % of benefit: {apv_endowment/benefit*100:.2f}%")
print()

# 5. INCREASING DEATH BENEFIT (LINEAR)
print("=" * 80)
print("INCREASING DEATH BENEFIT: LINEAR GROWTH")
print("=" * 80)

# Benefit increases by $10,000 each year
initial_benefit = 50000
annual_increase = 10000
n_increasing = 15

apv_increasing = 0
kpx_i = 1.0

print(f"\nBenefit grows: ${initial_benefit:,.0f} (year 1) → ${initial_benefit + (n_increasing-1)*annual_increase:,.0f} (year {n_increasing})")
print(f"\n{'Year':<8} {'Benefit':<15} {'qₓ':<12} {'v^k':<12} {'PV of Benefit':<15}")
print("-" * 72)

for k in range(1, n_increasing + 1):
    age_at_death = start_age + k - 1
    idx_ext = age_at_death - start_age
    
    # Benefit in year k
    benefit_k = initial_benefit + (k - 1) * annual_increase
    
    qx_k = qx_extended[idx_ext]
    vk = v_discount ** k
    
    pv_benefit_k = kpx_i * qx_k * vk * benefit_k
    apv_increasing += pv_benefit_k
    
    print(f"{k:<8} ${benefit_k:<14,.0f} {qx_k:<12.6f} {vk:<12.6f} ${pv_benefit_k:<14,.2f}")
    
    px_k = 1 - qx_k
    kpx_i *= px_k

print(f"\nAPV of Increasing Benefit: ${apv_increasing:,.2f}")
print()

# 6. SENSITIVITY ANALYSIS: APV BY AGE AND INTEREST RATE
print("=" * 80)
print("SENSITIVITY ANALYSIS: 10-YEAR TERM, $100,000, BY AGE & RATE")
print("=" * 80)

start_ages_sens = np.array([30, 40, 50, 60])
rates_sens = np.array([0.02, 0.04, 0.06, 0.08])

print(f"\nAPV by Starting Age and Interest Rate:\n")
print(f"{'Age':<10}", end='')
for rate in rates_sens:
    print(f"i={rate*100:.0f}%", end='\t')
print()
print("-" * 65)

for start_age_s in start_ages_sens:
    print(f"{start_age_s:<10}", end='')
    
    for rate_s in rates_sens:
        v_s = 1 / (1 + rate_s)
        apv_s = 0
        kpx_s = 1.0
        
        for k in range(1, 11):  # 10-year term
            age_at_death_s = start_age_s + k - 1
            idx_ext_s = age_at_death_s - start_age
            
            if idx_ext_s < len(qx_extended):
                qx_k_s = qx_extended[idx_ext_s]
            else:
                qx_k_s = 0.99  # Assume high mortality at extreme ages
            
            vk_s = v_s ** k
            apv_s += kpx_s * qx_k_s * vk_s * benefit
            
            px_k_s = 1 - qx_k_s
            kpx_s *= px_k_s
        
        print(f"${apv_s:>8,.0f}", end='\t')
    
    print()

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: APV comparison across product types
ax = axes[0, 0]
products = ['10-Yr Term', 'Whole Life', '20-Yr Endow', 'Increasing*']
apv_values = [apv_term, apv_whole_life, apv_endowment, apv_increasing]
colors = ['steelblue', 'darkred', 'green', 'orange']

bars = ax.bar(products, apv_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
ax.set_ylabel('APV ($)', fontsize=11, fontweight='bold')
ax.set_title('APV Comparison: Different Product Types', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(apv_values) * 1.15)

# Add value labels on bars
for bar, val in zip(bars, apv_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:,.0f}\n({val/benefit*100:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.text(0.5, -0.15, '*Increasing: $50K→$190K over 15 years', 
        transform=ax.transAxes, ha='center', fontsize=8, style='italic')

# Plot 2: APV by term length (age 40)
ax = axes[0, 1]
n_range = np.arange(1, 41)
apv_by_term = []
kpx_term = 1.0

for n in n_range:
    apv_n = 0
    kpx_n = 1.0
    
    for k in range(1, n + 1):
        age_death = start_age + k - 1
        idx_ext = age_death - start_age
        
        if idx_ext < len(qx_extended):
            qx_k_n = qx_extended[idx_ext]
        else:
            qx_k_n = 0.95
        
        vk_n = v_discount ** k
        apv_n += kpx_n * qx_k_n * vk_n * benefit
        
        px_k_n = 1 - qx_k_n
        kpx_n *= px_k_n
    
    apv_by_term.append(apv_n)

ax.plot(n_range, apv_by_term, linewidth=2.5, color='darkblue', marker='o', markersize=5)
ax.fill_between(n_range, 0, apv_by_term, alpha=0.2, color='blue')
ax.set_xlabel('Term Length (years)', fontsize=11)
ax.set_ylabel('APV ($)', fontsize=11)
ax.set_title('APV vs Term Length (Age 40, $100K)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3: APV sensitivity by starting age (10-year term)
ax = axes[1, 0]
start_ages_plot = np.arange(25, 71, 5)
apv_by_age = []

for start_age_plot in start_ages_plot:
    apv_age = 0
    kpx_age = 1.0
    
    for k in range(1, 11):  # 10-year term
        age_death_plot = start_age_plot + k - 1
        idx_ext_plot = age_death_plot - start_age
        
        if idx_ext_plot >= 0 and idx_ext_plot < len(qx_extended):
            qx_k_age = qx_extended[idx_ext_plot]
        else:
            qx_k_age = 0.05
        
        vk_age = v_discount ** k
        apv_age += kpx_age * qx_k_age * vk_age * benefit
        
        px_k_age = 1 - qx_k_age
        kpx_age *= px_k_age
    
    apv_by_age.append(apv_age)

ax.plot(start_ages_plot, apv_by_age, linewidth=2.5, color='darkgreen', marker='s', markersize=6)
ax.fill_between(start_ages_plot, 0, apv_by_age, alpha=0.2, color='green')
ax.set_xlabel('Starting Age', fontsize=11)
ax.set_ylabel('APV ($)', fontsize=11)
ax.set_title('APV vs Starting Age (10-Year Term, $100K)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Mortality-interest relationship (contribution to APV)
ax = axes[1, 1]
years_contrib = np.arange(1, 21)
term_contrib = []
kpx_contrib = 1.0

for k in years_contrib:
    age_contrib = start_age + k - 1
    idx_ext_c = age_contrib - start_age
    
    qx_k_c = qx_extended[idx_ext_c]
    vk_c = v_discount ** k
    
    contrib = kpx_contrib * qx_k_c * vk_c * benefit
    term_contrib.append(contrib)
    
    px_k_c = 1 - qx_k_c
    kpx_contrib *= px_k_c

ax.bar(years_contrib, term_contrib, color='purple', alpha=0.6, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Year of Death', fontsize=11)
ax.set_ylabel('PV of Benefit ($)', fontsize=11)
ax.set_title('Contribution by Year: 20-Year Term APV', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('present_value_of_benefits_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When APV calculations break down:
- **Basis risk**: Assumed mortality (table) differs from actual; reserves inadequate if experience worse than expected
- **Assumption consistency**: Interest rate and mortality must be from same year/source; mismatches cause systematic bias
- **Long-tail uncertainty**: 50+ year horizons for whole life; small assumption errors compound dramatically
- **Embedded options**: Surrender/lapse not reflected in pure APV; dynamic modeling required
- **Decrement interactions**: Death, disability, lapse compete; single-decrement formulas overstate survival probabilities
- **Continuous vs discrete**: Switching formulas without adjustment (continuous Āₓ vs discrete Aₓ) causes 5-10% errors

## 7. Key References
- [Bowers et al., Actuarial Mathematics (Chapter 4-5)](https://www.soa.org/) - Present value, insurance notation
- [SOA Exam FM Study Manual](https://www.soa.org/education/exam-req/edu-exam-fm-detail.aspx) - Practice problems, real examples
- [Wikipedia - Actuarial Present Value](https://en.wikipedia.org/wiki/Actuarial_present_value) - Definitions, commutation functions

---
**Status:** Foundational valuation | **Complements:** Term Insurance, Whole Life, Endowment Insurance
