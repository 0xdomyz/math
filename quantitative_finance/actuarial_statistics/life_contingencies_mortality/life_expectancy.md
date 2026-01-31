# Life Expectancy

## 1. Concept Skeleton
**Definition:** Expected remaining lifetime at age x; eₓ = E[Tₓ] where Tₓ is future years  
**Purpose:** Measure population health, project pension obligations, compare populations, assess mortality improvements  
**Prerequisites:** Survival probability, life tables, expected value, demographic rates

## 2. Comparative Framing
| Metric | Life Expectancy eₓ | Median Lifespan | Survival Probability |
|--------|-------------------|-----------------|----------------------|
| **Definition** | Mean remaining years | 50th percentile age | P(survive to age x+p) |
| **Calculation** | eₓ = ∑ ₚpₓ | From cumulative distribution | Direct from life table |
| **Interpretation** | Average person lives this long | Half population exceeds | Individual probability |
| **Application** | Pension, annuity valuation | Policy marketing | Underwriting decisions |

## 3. Examples + Counterexamples

**Simple Example:**  
US male at birth (2023): e₀ ≈ 74 years; reflects current age-specific mortality across all ages

**Failure Case:**  
Using e₀ = 75 for everyone age 60: Incorrect—e₆₀ ≈ 19 additional years, not original 75

**Edge Case:**  
War/pandemic year: e₀ drops sharply (WWII ≈5% decline), but survivors face normal hazards; short-term spike, not structural

## 4. Layer Breakdown
```
Life Expectancy Structure:
├─ Definition:
│   ├─ eₓ = E[Tₓ] = ∑ₚ₌₀^∞ ₚpₓ  [discrete]
│   ├─ eₓ = ∫₀^∞ ₚpₓ dp  [continuous]
│   ├─ eₓ = Tₓ / lₓ  [from life table]
│   └─ Conditional: e_{x|y} = remaining life given survive to y
├─ Calculation Methods:
│   ├─ Cohort (generation): Follow birth cohort to death
│   ├─ Period: Cross-section of current age-specific rates
│   ├─ Adjusted: Exclude specific causes (e.g., without cancer)
│   └─ Healthy life expectancy (HALE): Adjust for disability
├─ Decomposition:
│   ├─ By cause: Mortality contribution to lost years
│   ├─ By age: Which ages drive change (e.g., 60-80 accounts for 40%)
│   ├─ By gender/region: Equity analysis
│   └─ Gains: Compare periods (e.g., 1990 vs 2020 difference)
├─ Adjustments:
│   ├─ Smoking: Non-smoker +3-5 years vs smoker
│   ├─ Socioeconomic: Top 20% vs bottom 20% gap 10-15 years
│   ├─ Occupation: Hazardous -2-3 years
│   └─ Health status: Chronic disease -5-10 years
└─ Projection Models:
    ├─ Lee-Carter: Captures mortality improvement trend
    ├─ Age-period-cohort: Separates generation effects
    └─ Scenario: Policy-based (smoking reduction, healthcare)
```

**Interaction:** Life table → Calculate Tₓ → Divide by lₓ → Analyze trends

## 5. Mini-Project
Calculate life expectancy, decompose by age/cause, and project improvements:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. BUILD LIFE TABLE & CALCULATE LIFE EXPECTANCY
ages = np.arange(0, 121)

# Realistic mortality (based on developed country data)
qx_data = np.array([
    0.00714, 0.00055, 0.00040, 0.00032, 0.00029,  # Ages 0-4
    0.00028, 0.00027, 0.00027, 0.00028, 0.00030,  # Ages 5-9
] + list(np.linspace(0.00033, 0.40, 111)))  # Linear increase to age 110

qx = np.minimum(qx_data, 0.9999)  # Cap at 0.9999

radix = 100000
lx = np.zeros(121)
dx = np.zeros(121)
Lx = np.zeros(121)
Tx = np.zeros(121)
ex = np.zeros(121)

lx[0] = radix

# Build life table
for x in range(120):
    dx[x] = lx[x] * qx[x]
    lx[x + 1] = lx[x] - dx[x]
    
    # Person-years: assume deaths uniformly distributed within year
    # For final year, assume person-years = lx[x]/2
    if lx[x] > 0:
        Lx[x] = (lx[x] + lx[x + 1]) / 2
    else:
        Lx[x] = 0

# Calculate total remaining person-years (Tx)
Tx[120] = Lx[120]  # Last year
for x in range(119, -1, -1):
    Tx[x] = Lx[x] + Tx[x + 1]

# Life expectancy
ex = np.where(lx > 0, Tx / lx, 0)

# Create life table
life_table = pd.DataFrame({
    'Age': ages,
    'qx': qx,
    'lx': lx,
    'dx': dx,
    'Lx': Lx,
    'Tx': Tx,
    'ex': ex
})

print("LIFE TABLE WITH LIFE EXPECTANCY:")
print(life_table[life_table['Age'].isin([0, 20, 40, 60, 80, 100])][['Age', 'qx', 'lx', 'ex']].to_string(index=False))
print()

# 2. CONDITIONAL LIFE EXPECTANCY
print("CONDITIONAL LIFE EXPECTANCY (remaining years):")
print("Age\tLife Expectancy (years)")
for age in [0, 20, 40, 60, 80]:
    remaining_years = ex[age]
    print(f"{age}\t{remaining_years:.1f}")
print()

# 3. LIFE EXPECTANCY GAIN/LOSS BY AGE
# Calculate contribution of each age to total life expectancy
def calculate_age_contribution(lx, Lx, ex):
    """Contribution of mortality at each age to e0"""
    contributions = np.zeros(len(lx))
    for x in range(len(lx) - 1):
        if lx[0] > 0:
            # Change in Tx when mortality at x changes slightly
            contributions[x] = Lx[x] / lx[0]
    return contributions

age_contrib = calculate_age_contribution(lx, Lx, ex)

print("AGE-SPECIFIC CONTRIBUTION TO LIFE EXPECTANCY AT BIRTH:")
print("Age Range\tPerson-Years\t% of Total")
print("-" * 50)
for age_range in [(0, 5), (5, 15), (15, 30), (30, 60), (60, 80), (80, 120)]:
    start, end = age_range
    mask = (ages >= start) & (ages < end)
    contrib_range = Lx[mask].sum()
    pct = 100 * contrib_range / Lx.sum()
    print(f"{start:2d}-{end:2d}\t\t{contrib_range:.0f}\t\t{pct:.1f}%")
print()

# 4. MORTALITY IMPROVEMENTS & LIFE EXPECTANCY GAINS
# Simulate improvements over time
years = np.arange(2020, 2051)
improvement_rate = 0.015  # 1.5% annual improvement
ex_projection = np.zeros((len(ages), len(years)))

# Baseline (year 0)
ex_projection[:, 0] = ex

# Project forward with annual improvement
for year_idx in range(1, len(years)):
    # Apply improvement: reduce qx by factor
    qx_improved = qx * (1 - improvement_rate) ** year_idx
    qx_improved = np.minimum(qx_improved, 0.9999)
    
    # Rebuild life table
    lx_proj = np.zeros(121)
    lx_proj[0] = radix
    Lx_proj = np.zeros(121)
    Tx_proj = np.zeros(121)
    
    for x in range(120):
        dx_proj = lx_proj[x] * qx_improved[x]
        lx_proj[x + 1] = lx_proj[x] - dx_proj
        Lx_proj[x] = (lx_proj[x] + lx_proj[x + 1]) / 2
    
    Tx_proj[120] = Lx_proj[120]
    for x in range(119, -1, -1):
        Tx_proj[x] = Lx_proj[x] + Tx_proj[x + 1]
    
    ex_proj = np.where(lx_proj > 0, Tx_proj / lx_proj, 0)
    ex_projection[:, year_idx] = ex_proj

# Print projections
print("LIFE EXPECTANCY PROJECTIONS (1.5% annual improvement):")
print("Year\te₀ (at birth)\te₆₀ (at age 60)\te₈₀ (at age 80)")
for year_idx in [0, 10, 20, 30]:
    if year_idx < len(years):
        year = years[year_idx]
        print(f"{year}\t{ex_projection[0, year_idx]:.1f}\t\t" +
              f"{ex_projection[60, year_idx]:.1f}\t\t{ex_projection[80, year_idx]:.1f}")
print()

# 5. LIFE EXPECTANCY DECOMPOSITION BY CAUSE
# Simulate multiple causes of death
causes = {
    'Cardiovascular': 0.35,
    'Cancer': 0.25,
    'Respiratory': 0.10,
    'Accidents': 0.05,
    'Other': 0.25
}

ex_by_cause = {}
print("LIFE EXPECTANCY LOSS BY CAUSE (if cause eliminated):")

for cause, fraction in causes.items():
    # Reduce qx for this cause
    qx_reduced = qx * (1 - fraction)
    qx_reduced = np.minimum(qx_reduced, 0.9999)
    
    # Rebuild life table without this cause
    lx_no_cause = np.zeros(121)
    lx_no_cause[0] = radix
    Lx_no_cause = np.zeros(121)
    Tx_no_cause = np.zeros(121)
    
    for x in range(120):
        dx_no_cause = lx_no_cause[x] * qx_reduced[x]
        lx_no_cause[x + 1] = lx_no_cause[x] - dx_no_cause
        Lx_no_cause[x] = (lx_no_cause[x] + lx_no_cause[x + 1]) / 2
    
    Tx_no_cause[120] = Lx_no_cause[120]
    for x in range(119, -1, -1):
        Tx_no_cause[x] = Lx_no_cause[x] + Tx_no_cause[x + 1]
    
    ex_no_cause = np.where(lx_no_cause > 0, Tx_no_cause / lx_no_cause, 0)
    ex_by_cause[cause] = ex_no_cause[0]
    
    gain = ex_no_cause[0] - ex[0]
    print(f"{cause:20s}: {gain:5.2f} years gained")

print()

# 6. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Life Expectancy by age
ax = axes[0, 0]
ax.plot(ages[:-1], ex[:-1], linewidth=2.5, color='darkblue')
ax.fill_between(ages[:-1], ex[:-1], alpha=0.2, color='blue')
ax.scatter([0, 20, 40, 60, 80], ex[[0, 20, 40, 60, 80]], 
          color='red', s=100, zorder=5)
ax.set_xlabel('Age (x)', fontsize=11)
ax.set_ylabel('Remaining Life Expectancy (years)', fontsize=11)
ax.set_title('Life Expectancy by Age', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
for age in [0, 20, 40, 60, 80]:
    ax.annotate(f'{ex[age]:.1f}', xy=(age, ex[age]), 
               xytext=(5, 5), textcoords='offset points', fontsize=9)

# Plot 2: Projected improvements
ax = axes[0, 1]
for age in [0, 20, 60, 80]:
    ax.plot(years, ex_projection[age, :], linewidth=2.5, marker='o', 
           markersize=4, label=f'e{age}')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Life Expectancy (years)', fontsize=11)
ax.set_title('Projected Life Expectancy (1.5% annual improvement)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Person-years contribution by age
ax = axes[1, 0]
age_groups = ['0-5', '5-15', '15-30', '30-60', '60-80', '80+']
age_ranges = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 80), (80, 121)]
person_years = []
for start, end in age_ranges:
    mask = (ages >= start) & (ages < end)
    person_years.append(Lx[mask].sum())

colors = plt.cm.viridis(np.linspace(0, 1, len(age_groups)))
bars = ax.bar(age_groups, person_years, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Person-Years in Cohort', fontsize=11)
ax.set_title('Life Table: Person-Years by Age Group', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.0f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Life expectancy gains by cause
ax = axes[1, 1]
causes_sorted = sorted(ex_by_cause.items(), 
                      key=lambda x: x[1] - ex[0], reverse=True)
causes_names = [c[0] for c in causes_sorted]
gains = [c[1] - ex[0] for c in causes_sorted]

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(causes_names)))
bars = ax.barh(causes_names, gains, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Years of Life Expectancy Gained', fontsize=11)
ax.set_title('Impact of Eliminating Each Cause of Death', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

for i, (bar, gain) in enumerate(zip(bars, gains)):
    ax.text(gain, bar.get_y() + bar.get_height()/2.,
           f' {gain:.2f}y', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('life_expectancy_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 6. Challenge Round
When life expectancy misleads:
- **Averaging fallacy**: e₀ = 75 doesn't mean "most people die at 75"; bimodal distribution hides child/elder survival
- **Selection bias**: Life insurance applicant pool has e₀ ≈ 5-10 years higher than population
- **Aging cohort**: Population ages → e₀ stays flat despite mortality improvements (period effect masking)
- **Projection hubris**: Assuming 1.5% annual improvement forever; historical shocks unpredictable
- **Cause elimination unrealistic**: Eliminating cancer might shift deaths to another cause; not pure gain

## 7. Key References
- [Human Mortality Database](https://www.mortality.org/) - International life expectancy data
- [Life Table Construction (Actuarial Mathematics by Bowers et al.)](https://www.soa.org/) - Technical methods
- [WHO Global Health Observatory](https://www.who.int/data/gho) - Health life expectancy (HALE)

---
**Status:** Key demographic metric | **Complements:** Mortality Tables, Survival Probability, Pension Valuation
