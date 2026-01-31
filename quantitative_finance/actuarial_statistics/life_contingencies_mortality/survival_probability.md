# Survival Probability

## 1. Concept Skeleton
**Definition:** Probability ₚₓ that person age x survives p additional years; cumulative survival measure  
**Purpose:** Quantify longevity risk, price life insurance and annuities, project life expectancy, assess population health  
**Prerequisites:** Force of mortality, life tables, conditional probability, survival functions

## 2. Comparative Framing
| Measure | Survival ₚₓ | Mortality qₓ | Force μₓ |
|---------|------------|-------------|---------|
| **Definition** | P(live x→x+p) | P(death in [x, x+p)) | Instantaneous death rate |
| **Relationship** | ₚₓ + ₚqₓ = 1 | ₚqₓ = 1 - ₚₓ | ₚₓ = exp(-∫₀ᵖ μ_{x+t} dt) |
| **Domain** | [0, 1]; increasing in p | [0, 1]; increasing in p | ℝ₊; usually increasing with age |
| **Typical Use** | Annuity pricing, life expectancy | Insurance premium, reserve | Mortality model fitting |

## 3. Examples + Counterexamples

**Simple Example:**  
Male age 30: ₁₀p₃₀ = 0.99 (99% chance of reaching 40); applies to life insurance underwriting

**Failure Case:**  
Assuming constant survival ₚₓ = 0.95 across all ages: Reality shows ₚₓ drops sharply after age 75; constant assumption causes mispricing

**Edge Case:**  
Truncated life table (data only to age 110): ₚₓ = 0 for x > 110; forces assumption of ultimate rate for projections beyond table

## 4. Layer Breakdown
```
Survival Probability Structure:
├─ Definition & Axioms:
│   ├─ ₚₓ = P(Tₓ > p) where Tₓ = remaining lifetime
│   ├─ ₀pₓ = 1 (always survive 0 years)
│   ├─ ₘ₊ₙpₓ = ₘpₓ · ₙp_{x+m} (chain rule)
│   └─ ₚₓ + ₚqₓ = 1 (exhaustive)
├─ Relationship to Other Functions:
│   ├─ From qₓ: ₚₓ = ∏ᵢ₌₀^{p-1} (1 - q_{x+i})  [discrete]
│   ├─ From μₓ: ₚₓ = exp(-∫₀ᵖ μ_{x+t} dt)  [continuous]
│   ├─ Life table: ₚₓ = l_{x+p} / lₓ
│   └─ Survival curve: Sₓ(p) = ₚₓ
├─ Empirical Calculation:
│   ├─ 1. Obtain life table: lₓ at each age
│   ├─ 2. Compute p-year survival: ₚₓ = l_{x+p} / lₓ
│   ├─ 3. Interpolate for fractional ages (Karup-King, Beers)
│   └─ 4. Aggregate for cohorts/groups (stratified survival)
├─ Adjustment Factors:
│   ├─ Health status: Non-smoker surcharge ↑ 5-10% vs smoker
│   ├─ Occupation: Hazardous jobs ↓ 2-5% mortality loading
│   ├─ Underwriting class: Preferred/standard/substandard
│   └─ Impairment ratings: Individual health conditions adjust table rates
└─ Parametric Models:
    ├─ Gompertz: ₚₓ = exp[-B/C · (e^{C(x+p)} - e^{Cx})]
    ├─ Weibull: ₚₓ = exp[-(λ(x+p))^k + (λx)^k]
    └─ Lee-Carter: ₚₓ(t) parametrized by stochastic κₜ
```

**Interaction:** Life table → Calculate ₚₓ → Apply adjustments → Price products

## 5. Mini-Project
Calculate survival probabilities, life expectancy, and cohort survival tables:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. CREATE LIFE TABLE from qx
# Using realistic mortality rates (US male-like)
ages = np.arange(0, 121)
qx_base = np.array([
    0.00714, 0.00055, 0.00040, 0.00032, 0.00029,  # Ages 0-4
    0.00028, 0.00027, 0.00027, 0.00028, 0.00030,  # Ages 5-9
    0.00033, 0.00038, 0.00048, 0.00065, 0.00090,  # Ages 10-14
    0.00124, 0.00166, 0.00214, 0.00261, 0.00297,  # Ages 15-19
    0.00311, 0.00311, 0.00307, 0.00301, 0.00295,  # Ages 20-24
] + list(np.linspace(0.00295, 0.05, 96)))  # Extend to age 120

# Pad to 121 ages
qx = np.zeros(121)
qx[:len(qx_base)] = qx_base
qx[len(qx_base):] = 0.99999  # Force death by ultimate age

radix = 100000  # Starting population

# Build life table
lx = np.zeros(121)
dx = np.zeros(121)
px = np.zeros(120)  # p-year survival (one-year for simplicity)
Lx = np.zeros(121)  # Person-years lived
Tx = np.zeros(121)  # Total remaining person-years
ex = np.zeros(121)  # Life expectancy

lx[0] = radix

for x in range(120):
    dx[x] = lx[x] * qx[x]
    lx[x + 1] = lx[x] - dx[x]
    px[x] = lx[x + 1] / lx[x]
    
    # Approximate person-years (assume deaths occur mid-year)
    Lx[x] = (lx[x] + lx[x + 1]) / 2

lx[120] = 0
Lx[120] = 0

# Calculate Tx (cumulative person-years)
Tx = np.zeros(121)
for x in range(119, -1, -1):
    Tx[x] = Lx[x] + Tx[x + 1]

# Life expectancy
ex = np.where(lx > 0, Tx / lx, 0)

# Create life table DataFrame
life_table = pd.DataFrame({
    'Age': ages,
    'lx': lx,
    'dx': dx,
    'qx': qx,
    'px': np.concatenate([px, [0]]),
    'Lx': Lx,
    'Tx': Tx,
    'ex': ex
})

print("LIFE TABLE (sample ages):")
print(life_table[life_table['Age'].isin([0, 20, 40, 60, 80, 100])].to_string(index=False))
print()

# 2. MULTI-YEAR SURVIVAL PROBABILITIES
def calculate_px(start_age, years, px_single):
    """Calculate p-year survival using chain rule"""
    prob = 1.0
    for year in range(years):
        if start_age + year < len(px_single):
            prob *= px_single[start_age + year]
        else:
            prob = 0
            break
    return prob

# Create matrix of p-year survival
max_p = 50
px_matrix = np.zeros((len(ages), max_p))

for x in range(len(ages) - max_p):
    for p in range(max_p):
        px_matrix[x, p] = calculate_px(x, p, px)

print("MULTI-YEAR SURVIVAL (selected ages):")
print("Age\t1-Year\t5-Year\t10-Year\t25-Year\t50-Year")
for age in [20, 40, 60, 80]:
    print(f"{age}\t{px_matrix[age, 1]:.4f}\t{px_matrix[age, 5]:.4f}\t" +
          f"{px_matrix[age, 10]:.4f}\t{px_matrix[age, 25]:.4f}\t{px_matrix[age, 50]:.4f}")
print()

# 3. ADJUSTMENT FOR HEALTH STATUS / MORTALITY CLASS
# Define mortality multipliers for different groups
adjustments = {
    'Preferred Non-Smoker': 0.85,   # 15% lower mortality
    'Standard Non-Smoker': 1.00,    # Baseline
    'Standard Smoker': 1.30,        # 30% higher mortality
    'Substandard': 1.50             # 50% higher mortality
}

# Apply to survival probabilities
px_adjusted = {}
for group, factor in adjustments.items():
    # Adjusted qx = min(1, factor * qx)
    qx_adj = np.minimum(1.0, factor * qx)
    px_group = np.zeros(120)
    for x in range(120):
        px_group[x] = 1 - qx_adj[x]
    px_adjusted[group] = px_group

# Calculate 10-year survival for each group at age 40
print("IMPACT OF MORTALITY CLASS (10-year survival from age 40):")
for group, px_group in px_adjusted.items():
    survival_10yr = calculate_px(40, 10, px_group)
    print(f"{group:30s}: {survival_10yr:.4f}")
print()

# 4. COHORT ANALYSIS
# Track specific birth cohort through time
birth_cohort_1950 = life_table[life_table['Age'] == 0]['ex'].values[0]
cohort_ages = ages[:101]
cohort_survival = lx[:101] / radix

print(f"COHORT ANALYSIS (Birth Cohort 1950):")
print(f"Life expectancy at birth: {birth_cohort_1950:.1f} years")
print(f"Probability of reaching:")
for target_age in [25, 50, 75, 100]:
    surv_prob = lx[target_age] / radix
    print(f"  Age {target_age}: {surv_prob:.4f} ({surv_prob*100:.2f}%)")
print()

# 5. INTERPOLATION FOR FRACTIONAL AGES
# Using Karup-King formula for smooth interpolation
def karup_king_interpolation(x_int, lx_table):
    """Karup-King formula for fractional age interpolation"""
    x_lower = int(x_int)
    frac = x_int - x_lower
    
    if x_lower < 0 or x_lower + 3 >= len(lx_table):
        return np.nan
    
    l0, l1, l2, l3 = lx_table[x_lower:x_lower+4]
    
    # Karup-King interpolation formula
    coeff = (frac * (frac - 1) * (frac - 2)) / 6
    lx_frac = l1 + frac * (l2 - l1) + coeff * (l3 - l0 + 3*l1 - 3*l2)
    return lx_frac

# Calculate survival for fractional ages
fractional_ages = np.array([40.0, 40.25, 40.5, 40.75, 41.0])
print("FRACTIONAL AGE SURVIVAL (Karup-King interpolation):")
print("Age\t\tlx\t\tpx")
for frac_age in fractional_ages:
    lx_frac = karup_king_interpolation(frac_age, lx)
    px_frac = karup_king_interpolation(frac_age + 1, lx) / lx_frac if lx_frac > 0 else 0
    print(f"{frac_age:.2f}\t\t{lx_frac:.0f}\t\t{px_frac:.6f}")
print()

# 6. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Life table curves
ax = axes[0, 0]
ax.plot(ages, lx/radix, linewidth=2.5, color='darkblue', label='lx / radix')
ax.fill_between(ages, 0, lx/radix, alpha=0.2, color='blue')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Proportion Surviving', fontsize=11)
ax.set_title('Life Table: Survivors by Age', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim([0, 1.05])

# Plot 2: Mortality and survival rates
ax = axes[0, 1]
ax_dual = ax.twinx()
ax.semilogy(ages[:-1], qx[:-1], 'r-', linewidth=2, label='qx (mortality)')
ax_dual.plot(ages[:-1], px, 'b-', linewidth=2, label='px (survival)')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Mortality qx (log scale)', fontsize=11, color='r')
ax_dual.set_ylabel('Survival px', fontsize=11, color='b')
ax.set_title('Mortality vs Survival Rates', fontsize=12, fontweight='bold')
ax.tick_params(axis='y', labelcolor='r')
ax_dual.tick_params(axis='y', labelcolor='b')
ax.grid(alpha=0.3)
ax.set_ylim([1e-4, 1])

# Plot 3: Multi-year survival heatmap
ax = axes[1, 0]
im = ax.imshow(px_matrix[::5, :30].T, aspect='auto', cmap='RdYlGn', origin='lower')
ax.set_xlabel('Age (every 5 years)', fontsize=11)
ax.set_ylabel('Years Ahead (p)', fontsize=11)
ax.set_title('Survival Probability Heatmap (pₓ)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Probability', fontsize=10)

# Plot 4: Mortality class comparison
ax = axes[1, 1]
base_ages = ages[40:101]
for group in ['Preferred Non-Smoker', 'Standard Non-Smoker', 'Standard Smoker', 'Substandard']:
    survival_curve = []
    for p in range(1, 61):
        surv_p = calculate_px(40, p, px_adjusted[group])
        survival_curve.append(surv_p)
    ax.plot(base_ages[:60], survival_curve, linewidth=2.5, marker='o', 
            markersize=3, alpha=0.7, label=group)

ax.set_xlabel('Age at evaluation: 40 + years ahead', fontsize=11)
ax.set_ylabel('Cumulative Survival Probability', fontsize=11)
ax.set_title('Impact of Mortality Class on Survival', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('survival_probability_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. LIFE EXPECTANCY DECOMPOSITION
print("LIFE EXPECTANCY BY AGE:")
print("Age\tLife Expectancy (years)")
for age in [0, 20, 40, 60, 80]:
    print(f"{age}\t{ex[age]:.1f}")
```

## 6. Challenge Round
When survival probabilities are unreliable:
- **Selection bias**: Insurance applicants healthier than general population; table rates too pessimistic
- **Mortality improvements**: Historical data shows 1-2% annual improvement; static table ages poorly
- **Pandemic/crisis**: 2020 COVID spikes broke all historical models; use dynamic scenario analysis
- **Small populations**: Few deaths in subgroup (smokers <30); high estimation error; credibility weighting needed
- **Competing risks**: Person-years lost to migration/lapse affects denominator; use multi-decrement tables

## 7. Key References
- [Life Table Construction (Human Mortality Database)](https://www.mortality.org/) - Empirical survival data
- [Survival Analysis Fundamentals (Kleinbaum & Klein)](https://www.springer.com/) - Statistical methods
- [Kaplan-Meier Estimator (Wikipedia)](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) - Non-parametric survival

---
**Status:** Foundational actuarial metric | **Complements:** Life Expectancy, Mortality Tables, Life Insurance Pricing
