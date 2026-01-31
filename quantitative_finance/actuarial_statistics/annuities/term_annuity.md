# Term Annuity (aₓ:n̄|)

## 1. Concept Skeleton
**Definition:** Payments for n years OR until death, whichever occurs first; combines fixed term with mortality contingency; notation aₓ:n̄| (temporary annuity)  
**Purpose:** Value pension payments with known retirement duration, term-certain annuities with life contingency, structured settlements  
**Prerequisites:** Life annuities, annuity-certain, survival probabilities, temporary life insurance concepts

## 2. Comparative Framing
| Type | Duration | Contingency | Value Relation | Use Case |
|------|----------|-------------|----------------|----------|
| **Term Life (aₓ:n̄|)** | n years max | Death stops | aₓ:n̄| < aₙ̄| | Pension before guaranteed |
| **Whole Life (aₓ)** | Until death | Mortality only | aₓ > aₓ:n̄| | Lifetime income |
| **Certain (aₙ̄|)** | n years fixed | None | aₙ̄| > aₓ:n̄| | Fixed obligation |
| **Deferred (n|aₓ)** | After n years | Survival to n | Value delayed | Retirement income starts later |

## 3. Examples + Counterexamples

**Simple Example:**  
65-year-old receives $1000/year for 20 years OR until death; aₓ:₂₀̄| = $12,150 (vs whole life $13,000, certain $12,462)

**Failure Case:**  
Treating as certain annuity ignores mortality; overvalues by ~2-3% since death probability reduces expected payments

**Edge Case:**  
Very long term (n=50 at age 65): aₓ:₅₀̄| ≈ aₓ since survival past age 115 negligible; term becomes irrelevant

## 4. Layer Breakdown
```
Term Life Annuity Structure:
├─ Payment Schedule:
│   ├─ Payments: Years 1, 2, ..., min(n, time to death)
│   ├─ Stop conditions: (1) n years elapse, OR (2) Death occurs
│   ├─ Dual contingency: Both term limit AND mortality
│   └─ No payments after earlier of {n years, death}
├─ Present Value Formula:
│   ├─ aₓ:n̄| = Σ(k=1 to n) vᵏ · ₖpₓ
│   │   where vᵏ = (1+i)^(-k), ₖpₓ = P(survive k years from age x)
│   ├─ Decomposition: aₓ:n̄| = aₓ - n|aₓ
│   │   where n|aₓ = deferred annuity starting after n years
│   ├─ Alternative: aₓ:n̄| = aₙ̄| - v^n · ₙpₓ · aₓ₊ₙ
│   │   (certain annuity minus deferred whole life component)
│   └─ Using commutation: aₓ:n̄| = (Nₓ₊₁ - Nₓ₊ₙ₊₁) / Dₓ
├─ Relationship to Other Products:
│   ├─ To whole life: aₓ:n̄| + n|aₓ = aₓ
│   ├─ To certain: aₓ:n̄| ≤ aₙ̄| (mortality reduces value)
│   ├─ To insurance: Aₓ:n̄| + d·aₓ:n̄| = 1 - v^n·ₙpₓ (term life identity)
│   └─ Limiting cases: limₙ→∞ aₓ:n̄| = aₓ, limₙ→0 aₓ:n̄| = 0
├─ Computational Methods:
│   ├─ Direct summation: Sum k=1 to n of discounted survival weights
│   ├─ Commutation functions: Use precomputed Nₓ, Dₓ tables
│   ├─ Recursive: aₓ:n̄| = v·pₓ + v·pₓ·aₓ₊₁:n₋₁̄| (dynamic programming)
│   └─ Interpolation: For fractional terms, linear or uniform distribution
├─ Premium Applications:
│   ├─ Limited payment life insurance: Pay premiums for n years only
│   ├─ Formula: P̈·äₓ:n̄| = Aₓ (benefits for whole life, premiums for n years)
│   ├─ Higher premium: Compressed payment period increases annual cost
│   └─ Reserve calculation: Different patterns than level-premium whole life
└─ Practical Considerations:
    ├─ Retirement planning: Annuity until age 85 (bridge to government pension)
    ├─ Structured settlements: Payments for 15 years certain, life-contingent
    ├─ Disability income: Benefits for 2 years or until recovery
    └─ Pension buyouts: Lump sum vs 10-year annuity decision
```

## 5. Mini-Project
Calculate and analyze term annuities:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Term Life Annuity (aₓ:n̄|) Analysis ===\n")

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

# Term annuity calculation
def term_annuity(x, n, i, mortality):
    """
    aₓ:n̄| = Σ(k=1 to n) vᵏ · ₖpₓ
    """
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 0
    for k in range(1, n + 1):
        future_age = x + k
        if future_age > 120:
            break
        
        l_future = mortality.loc[mortality['Age'] == future_age, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        
        k_p_x = l_future[0] / l_current
        value += (v ** k) * k_p_x
    
    return value

# Whole life annuity (for comparison)
def whole_life_annuity(x, i, mortality):
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

# Annuity-certain
def annuity_certain(n, i):
    """aₙ̄| = (1 - v^n) / d"""
    v = 1 / (1 + i)
    d = i / (1 + i)
    return (1 - v**n) / d

# Calculate term annuities for different terms
print("=== Term Annuity Values (Age 65, i = 5%) ===\n")
age_base = 65
i_rate = 0.05
terms = [5, 10, 15, 20, 25, 30]

results = []
whole_life_val = whole_life_annuity(age_base, i_rate, mortality)

for n in terms:
    term_val = term_annuity(age_base, n, i_rate, mortality)
    certain_val = annuity_certain(n, i_rate)
    
    mortality_discount = (certain_val - term_val) / certain_val * 100
    pct_of_whole_life = term_val / whole_life_val * 100
    
    results.append({
        'Term (n)': n,
        'Term Life (aₓ:n̄|)': term_val,
        'Certain (aₙ̄|)': certain_val,
        'Mortality Discount': mortality_discount,
        '% of Whole Life': pct_of_whole_life
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format='%.3f'))
print(f"\nWhole life annuity (aₓ): {whole_life_val:.3f}")

# Decomposition: aₓ = aₓ:n̄| + n|aₓ
print("\n=== Decomposition: Whole Life = Term + Deferred ===\n")

def deferred_annuity(x, n, i, mortality):
    """
    n|aₓ = v^n · ₙpₓ · aₓ₊ₙ
    Value of whole life annuity starting after n years
    """
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    l_future = mortality.loc[mortality['Age'] == x + n, 'l_x'].values[0]
    
    n_p_x = l_future / l_current
    a_x_plus_n = whole_life_annuity(x + n, i, mortality)
    
    return (v ** n) * n_p_x * a_x_plus_n

n_test = 20
term_val = term_annuity(age_base, n_test, i_rate, mortality)
deferred_val = deferred_annuity(age_base, n_test, i_rate, mortality)
whole_life_check = term_val + deferred_val

print(f"Age {age_base}, n = {n_test} years, i = {i_rate*100:.0f}%:")
print(f"  Term annuity (aₓ:{n_test}̄|): {term_val:.4f}")
print(f"  Deferred annuity ({n_test}|aₓ): {deferred_val:.4f}")
print(f"  Sum: {whole_life_check:.4f}")
print(f"  Whole life (aₓ): {whole_life_val:.4f}")
print(f"  Identity holds: {abs(whole_life_check - whole_life_val) < 0.01}")

# Age sensitivity
print("\n=== Term Annuity by Issue Age (n = 20 years) ===\n")
ages_test = [35, 45, 55, 65, 75, 85]
term_fixed = 20

print(f"Age | a₆₅:{term_fixed}̄| | a{term_fixed}̄| (certain) | Ratio")
print("-" * 55)

for age in ages_test:
    if age + term_fixed > 100:  # Skip if term exceeds reasonable age
        continue
    
    term_life = term_annuity(age, term_fixed, i_rate, mortality)
    certain = annuity_certain(term_fixed, i_rate)
    ratio = term_life / certain
    
    print(f"{age:3d} | {term_life:13.4f} | {certain:17.4f} | {ratio:6.4f}")

# Premium calculation: Limited payment life insurance
print("\n=== Limited Payment Life Insurance ===\n")
# $100,000 whole life insurance, premiums paid for 20 years only

def whole_life_insurance(x, i, mortality):
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 0
    for k in range(1, 121 - x):
        age_start = x + k - 1
        age_end = x + k
        if age_end > 120:
            break
        
        l_start = mortality.loc[mortality['Age'] == age_start, 'l_x'].values
        l_end = mortality.loc[mortality['Age'] == age_end, 'l_x'].values
        if len(l_start) == 0 or len(l_end) == 0:
            break
        
        deaths = l_start[0] - l_end[0]
        k_minus_1_q_x = deaths / l_current
        value += (v ** k) * k_minus_1_q_x
    
    return value

def annuity_due_term(x, n, i, mortality):
    """äₓ:n̄| = (1+i) · aₓ:n̄|"""
    return (1 + i) * term_annuity(x, n, i, mortality)

benefit = 100000
age_insured = 40
payment_years = 20

A_x = whole_life_insurance(age_insured, i_rate, mortality)
a_due_term = annuity_due_term(age_insured, payment_years, i_rate, mortality)

# Premium: P̈·äₓ:n̄| = Aₓ·benefit
annual_premium = (A_x * benefit) / a_due_term

# Compare to level-premium whole life
a_due_whole_life = (1 + i_rate) * whole_life_annuity(age_insured, i_rate, mortality)
annual_premium_level = (A_x * benefit) / a_due_whole_life

print(f"${benefit:,} Whole Life Insurance, Age {age_insured}:")
print(f"  Aₓ = {A_x:.4f}")
print(f"\nLimited payment (20 years):")
print(f"  Annual premium: ${annual_premium:,.2f}")
print(f"  Total paid: ${annual_premium * payment_years:,.2f}")
print(f"\nLevel premium (lifetime):")
print(f"  Annual premium: ${annual_premium_level:,.2f}")
print(f"  Ratio: {annual_premium / annual_premium_level:.2f}x higher")

# Interest rate sensitivity
print("\n=== Interest Rate Sensitivity (Age 65, n = 20) ===\n")
interest_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

print("Interest | Term (aₓ:₂₀̄|) | Whole Life (aₓ) | Difference")
print("-" * 60)

for i_val in interest_rates:
    term_val = term_annuity(65, 20, i_val, mortality)
    whole_val = whole_life_annuity(65, i_val, mortality)
    diff = whole_val - term_val
    
    print(f"{i_val*100:7.0f}%  | {term_val:14.4f} | {whole_val:15.4f} | {diff:9.4f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Term annuity vs certain annuity
ax1 = axes[0, 0]
terms_plot = np.arange(5, 51, 2)
term_vals = [term_annuity(65, n, 0.05, mortality) for n in terms_plot]
certain_vals = [annuity_certain(n, 0.05) for n in terms_plot]

ax1.plot(terms_plot, term_vals, 'o-', linewidth=2, label='Term life (aₓ:n̄|)', markersize=5)
ax1.plot(terms_plot, certain_vals, 's-', linewidth=2, label='Certain (aₙ̄|)', markersize=5)
ax1.axhline(whole_life_annuity(65, 0.05, mortality), color='r', linestyle='--', 
           linewidth=2, label='Whole life limit')
ax1.set_xlabel('Term (n years)')
ax1.set_ylabel('Annuity Value')
ax1.set_title('Term Annuity Convergence\n(Age 65, i = 5%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Mortality discount
ax2 = axes[0, 1]
mortality_discounts = [(certain_vals[i] - term_vals[i]) / certain_vals[i] * 100 
                       for i in range(len(terms_plot))]

ax2.plot(terms_plot, mortality_discounts, 'o-', linewidth=2, color='red', markersize=5)
ax2.fill_between(terms_plot, 0, mortality_discounts, alpha=0.2, color='red')
ax2.set_xlabel('Term (n years)')
ax2.set_ylabel('Mortality Discount (%)')
ax2.set_title('Value Reduction from Mortality\n(aₙ̄| - aₓ:n̄|) / aₙ̄|')
ax2.grid(True, alpha=0.3)

# Plot 3: Term coverage as % of whole life
ax3 = axes[0, 2]
pct_whole_life = [term_vals[i] / whole_life_annuity(65, 0.05, mortality) * 100 
                  for i in range(len(terms_plot))]

ax3.plot(terms_plot, pct_whole_life, 'o-', linewidth=2, color='green', markersize=5)
ax3.axhline(100, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax3.fill_between(terms_plot, 0, pct_whole_life, alpha=0.2, color='green')
ax3.set_xlabel('Term (n years)')
ax3.set_ylabel('% of Whole Life Value')
ax3.set_title('Term Annuity as % of Whole Life\nConverges to 100%')
ax3.grid(True, alpha=0.3)

# Plot 4: Age impact on term annuity
ax4 = axes[1, 0]
ages_impact = np.arange(30, 81, 5)
term_20_by_age = [term_annuity(age, 20, 0.05, mortality) for age in ages_impact]
certain_20 = annuity_certain(20, 0.05)

ax4.plot(ages_impact, term_20_by_age, 'o-', linewidth=2, markersize=6)
ax4.axhline(certain_20, color='r', linestyle='--', linewidth=2, label=f'20-year certain = {certain_20:.2f}')
ax4.set_xlabel('Issue Age')
ax4.set_ylabel('20-Year Term Annuity Value')
ax4.set_title('Age Impact on aₓ:₂₀̄|\n(Higher age → Lower value)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Decomposition visualization
ax5 = axes[1, 1]
terms_decomp = [5, 10, 15, 20, 25, 30]
term_components = [term_annuity(65, n, 0.05, mortality) for n in terms_decomp]
deferred_components = [deferred_annuity(65, n, 0.05, mortality) for n in terms_decomp]

x_pos = np.arange(len(terms_decomp))
ax5.bar(x_pos, term_components, label='Term (aₓ:n̄|)', alpha=0.7, edgecolor='black')
ax5.bar(x_pos, deferred_components, bottom=term_components, label='Deferred (n|aₓ)', alpha=0.7, edgecolor='black')
ax5.axhline(whole_life_annuity(65, 0.05, mortality), color='r', linestyle='--', 
           linewidth=2, label='Total = Whole life')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(terms_decomp)
ax5.set_xlabel('Deferral Period (n years)')
ax5.set_ylabel('Annuity Value')
ax5.set_title('Decomposition: aₓ = aₓ:n̄| + n|aₓ')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Premium comparison (limited vs level payment)
ax6 = axes[1, 2]
payment_periods = [10, 15, 20, 25, 30]
ages_premium = [30, 40, 50, 60]

for age in ages_premium:
    premiums_limited = []
    for n in payment_periods:
        A_x = whole_life_insurance(age, 0.05, mortality)
        a_due_term = annuity_due_term(age, n, 0.05, mortality)
        prem = (A_x * 100000) / a_due_term
        premiums_limited.append(prem)
    
    ax6.plot(payment_periods, premiums_limited, 'o-', linewidth=2, label=f'Age {age}', markersize=6)

ax6.set_xlabel('Payment Period (years)')
ax6.set_ylabel('Annual Premium ($)')
ax6.set_title('Limited Payment Premium\n($100k Whole Life)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print("Term annuity (aₓ:n̄|): Payments for n years OR until death")
print("Identity: aₓ = aₓ:n̄| + n|aₓ (term + deferred = whole life)")
print("Application: Limited payment insurance, pension bridges, structured settlements")
```

## 6. Challenge Round
When is term annuity analysis problematic?
- **Term choice**: Selecting appropriate term requires mortality forecasts; too short leaves gap, too long wastes premium
- **Mortality improvement**: If longevity increases, term annuities become more valuable; pricing assumes fixed mortality
- **Interest rate changes**: Long-term annuities sensitive to rate changes; valuation volatile
- **Behavioral**: Policyholders may lapse before term end; actual payments < theoretical
- **Inflation**: Fixed payments for 20+ years erode purchasing power; need inflation indexing

## 7. Key References
- [Society of Actuaries - Temporary Annuities](https://www.soa.org/) - Standard actuarial formulas
- [Wiki - Term Annuity](https://en.wikipedia.org/wiki/Life_annuity#Term_certain_annuity) - Overview of term structures
- [Actuarial Mathematics](https://www.actuary.org/) - Decomposition identities and premium applications

---
**Status:** Common in retirement planning and structured products | **Complements:** Whole Life Annuity, Deferred Annuity, Limited Payment Insurance
