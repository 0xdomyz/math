# Immediate Annuity (aₓ)

## 1. Concept Skeleton
**Definition:** Series of payments made at end of each period conditional on survival; present value computed via mortality-weighted discounting  
**Purpose:** Value retirement income streams, pension obligations, structured settlements where payments occur after period elapses  
**Prerequisites:** Mortality tables, discount factors, survival probabilities, present value concepts

## 2. Comparative Framing
| Type | Payment Timing | Notation | Value Relation | Use Case |
|------|----------------|----------|----------------|----------|
| **Immediate (Ordinary)** | End of period | aₓ | Lower than due | Standard pension |
| **Annuity Due** | Start of period | äₓ | äₓ = (1+i)·aₓ | Pre-paid rent |
| **Continuous** | Instantaneous | āₓ | āₓ ≈ aₓ + 0.5 | Theoretical models |
| **Certain** | Fixed term, no mortality | aₙ̄⏐ | Higher value | Guaranteed payments |

## 3. Examples + Counterexamples

**Simple Example:**  
65-year-old receives $1000/year for life (payments at year-end); immediate annuity value = $12,453 using 5% interest, mortality table

**Failure Case:**  
Ignoring mortality: Treating as annuity-certain overstates value since death probability reduces expected payments

**Edge Case:**  
Very high age (e.g., 105): Survival probabilities → 0 rapidly; annuity value approaches single payment discounted value

## 4. Layer Breakdown
```
Immediate Life Annuity Structure:
├─ Payment Schedule:
│   ├─ First payment: End of year 1 (individual survives 1 year)
│   ├─ Subsequent payments: End of years 2, 3, ..., until death
│   ├─ Conditionality: Payment k requires survival to age x+k
│   └─ No payment if death occurs before period end
├─ Present Value Formula:
│   ├─ aₓ = Σ(k=1 to ∞) vᵏ · ₖpₓ
│   │   where vᵏ = discount factor = 1/(1+i)ᵏ
│   │         ₖpₓ = P(survive from age x to x+k)
│   ├─ Alternative: aₓ = Σ(k=1 to ∞) vᵏ · lₓ₊ₖ / lₓ
│   │   where lₓ = number alive at age x from mortality table
│   └─ Recursive: aₓ = v·pₓ + v·pₓ·aₓ₊₁ (commutation identity)
├─ Computational Methods:
│   ├─ Direct summation: Sum discounted survival-weighted payments
│   ├─ Commutation functions: Use Dₓ, Nₓ tables → aₓ = Nₓ₊₁/Dₓ
│   ├─ Woolhouse approximation: aₓ ≈ āₓ - 1/2 (continuous → discrete)
│   └─ Life expectancy approximation: aₓ ≈ eₓ/(1+i) for low rates
├─ Relationship to Insurance:
│   ├─ Complementary products: Annuity pays while alive, insurance pays at death
│   ├─ Identity: aₓ = (1 - Aₓ)/d where Aₓ = whole life insurance, d = discount rate
│   ├─ Economic interpretation: Annuity provides mortality credit (gains from early deaths)
│   └─ Hedge: Insurance company matches annuity obligations with insurance premiums
└─ Practical Considerations:
    ├─ Mortality improvement: Longevity increases → higher annuity values
    ├─ Interest rate sensitivity: Low rates → high present values
    ├─ Adverse selection: Healthy individuals purchase annuities → use annuitant mortality
    └─ Inflation protection: Fixed nominal payments lose real value over time
```

## 5. Mini-Project
Calculate immediate annuity values across age and interest rates:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.random.seed(42)

print("=== Immediate Life Annuity (aₓ) Valuation ===\n")

# Create simplified mortality table (Gompertz-Makeham approximation)
def gompertz_mortality(age, A=0.0001, B=1.08, C=0.00035):
    """
    Force of mortality: μₓ = A + B^x · C
    """
    return A + C * (B ** age)

# Build mortality table
ages = np.arange(0, 121)
mu_x = gompertz_mortality(ages)
q_x = 1 - np.exp(-mu_x)  # Annual death probability

# Survival probabilities
l_x = np.zeros(len(ages))
l_x[0] = 100000  # Radix (starting population)
for i in range(1, len(ages)):
    l_x[i] = l_x[i-1] * (1 - q_x[i-1])

# Create mortality dataframe
mortality_table = pd.DataFrame({
    'Age': ages,
    'l_x': l_x,
    'q_x': q_x,
    'p_x': 1 - q_x
})

print("Sample Mortality Table:")
print(mortality_table.iloc[::10, :].to_string(index=False))

# Function to compute immediate annuity value
def immediate_annuity(x, i, mortality_table, max_age=120):
    """
    Calculate aₓ = Σ(k=1 to ω-x) vᵏ · ₖpₓ
    
    Parameters:
    - x: Current age
    - i: Annual interest rate
    - mortality_table: DataFrame with Age, l_x columns
    """
    v = 1 / (1 + i)  # Discount factor
    l_current = mortality_table.loc[mortality_table['Age'] == x, 'l_x'].values[0]
    
    # Sum over future ages
    annuity_value = 0
    for k in range(1, max_age - x + 1):
        future_age = x + k
        if future_age > max_age:
            break
        
        l_future = mortality_table.loc[mortality_table['Age'] == future_age, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        
        k_p_x = l_future[0] / l_current  # Survival probability
        annuity_value += (v ** k) * k_p_x
    
    return annuity_value

# Calculate annuity values for different ages
print("\n=== Immediate Annuity Values (i = 5%) ===\n")
sample_ages = [25, 35, 45, 55, 65, 75, 85]
interest_rate = 0.05

results = []
for age in sample_ages:
    a_x = immediate_annuity(age, interest_rate, mortality_table)
    # Life expectancy at age x
    l_curr = mortality_table.loc[mortality_table['Age'] == age, 'l_x'].values[0]
    e_x = sum(mortality_table.loc[mortality_table['Age'] > age, 'l_x'].values) / l_curr
    
    results.append({
        'Age': age,
        'aₓ (i=5%)': a_x,
        'Life Expectancy': e_x,
        '$1000 Annual Value': a_x * 1000
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Compare immediate vs annuity-certain
print("\n=== Immediate Annuity vs Annuity-Certain ===\n")
age_comparison = 65
a_x_life = immediate_annuity(age_comparison, interest_rate, mortality_table)

# Annuity-certain for same expected duration
l_curr = mortality_table.loc[mortality_table['Age'] == age_comparison, 'l_x'].values[0]
e_x = sum(mortality_table.loc[mortality_table['Age'] > age_comparison, 'l_x'].values) / l_curr
n_certain = int(e_x)

# a_n̄⏐ = (1 - v^n) / d, where d = i/(1+i)
v = 1 / (1 + interest_rate)
a_certain = (1 - v**n_certain) / (interest_rate / (1 + interest_rate))

print(f"Age {age_comparison}:")
print(f"  Immediate life annuity (aₓ): {a_x_life:.2f}")
print(f"  Life expectancy (years): {e_x:.1f}")
print(f"  {n_certain}-year certain annuity: {a_certain:.2f}")
print(f"  Mortality credit: {(a_certain / a_x_life - 1) * 100:.1f}% higher value for certain")

# Interest rate sensitivity
print("\n=== Interest Rate Sensitivity ===\n")
interest_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
age_sensitivity = 65

print(f"Age {age_sensitivity} Immediate Annuity Values:")
print("Interest Rate | aₓ Value | Duration (Macaulay)")
print("-" * 50)

for i_rate in interest_rates:
    a_val = immediate_annuity(age_sensitivity, i_rate, mortality_table)
    
    # Calculate Macaulay duration
    v = 1 / (1 + i_rate)
    l_current = mortality_table.loc[mortality_table['Age'] == age_sensitivity, 'l_x'].values[0]
    
    duration_numerator = 0
    for k in range(1, 121 - age_sensitivity):
        future_age = age_sensitivity + k
        l_future = mortality_table.loc[mortality_table['Age'] == future_age, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        k_p_x = l_future[0] / l_current
        duration_numerator += k * (v ** k) * k_p_x
    
    duration = duration_numerator / a_val if a_val > 0 else 0
    
    print(f"{i_rate*100:12.0f}%  | {a_val:8.2f} | {duration:8.2f} years")

# Relationship to whole life insurance: aₓ = (1 - Aₓ)/d
print("\n=== Relationship to Whole Life Insurance ===\n")

def whole_life_insurance(x, i, mortality_table, max_age=120):
    """
    Calculate Aₓ = Σ(k=1 to ω-x) vᵏ · ₖ₋₁|qₓ
    where ₖ₋₁|qₓ = probability of death in year k
    """
    v = 1 / (1 + i)
    l_current = mortality_table.loc[mortality_table['Age'] == x, 'l_x'].values[0]
    
    insurance_value = 0
    for k in range(1, max_age - x + 1):
        age_start = x + k - 1
        age_end = x + k
        
        if age_end > max_age:
            break
        
        l_start = mortality_table.loc[mortality_table['Age'] == age_start, 'l_x'].values
        l_end = mortality_table.loc[mortality_table['Age'] == age_end, 'l_x'].values
        
        if len(l_start) == 0 or len(l_end) == 0:
            break
        
        deaths = l_start[0] - l_end[0]
        k_minus_1_q_x = deaths / l_current
        
        insurance_value += (v ** k) * k_minus_1_q_x
    
    return insurance_value

age_test = 65
i_test = 0.05
d = i_test / (1 + i_test)  # Discount rate

a_x_direct = immediate_annuity(age_test, i_test, mortality_table)
A_x = whole_life_insurance(age_test, i_test, mortality_table)
a_x_from_insurance = (1 - A_x) / d

print(f"Age {age_test}, Interest {i_test*100:.0f}%:")
print(f"  aₓ (direct calculation): {a_x_direct:.4f}")
print(f"  Aₓ (whole life insurance): {A_x:.4f}")
print(f"  aₓ = (1 - Aₓ)/d: {a_x_from_insurance:.4f}")
print(f"  Identity verification: {abs(a_x_direct - a_x_from_insurance) < 0.01}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Mortality curve
ax1 = axes[0, 0]
ax1.plot(mortality_table['Age'], mortality_table['q_x'], linewidth=2)
ax1.set_xlabel('Age')
ax1.set_ylabel('Annual Death Probability (qₓ)')
ax1.set_title('Mortality Curve (Gompertz-Makeham)')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Annuity value by age
ax2 = axes[0, 1]
ages_plot = np.arange(20, 100, 5)
annuity_values = [immediate_annuity(age, 0.05, mortality_table) for age in ages_plot]

ax2.plot(ages_plot, annuity_values, 'o-', linewidth=2, markersize=6)
ax2.set_xlabel('Age (x)')
ax2.set_ylabel('Immediate Annuity Value (aₓ)')
ax2.set_title('Annuity Value Decreases with Age\n(i = 5%)')
ax2.grid(True, alpha=0.3)

# Plot 3: Interest rate sensitivity
ax3 = axes[0, 2]
i_rates = np.linspace(0.01, 0.10, 20)
annuity_by_rate = [immediate_annuity(65, i, mortality_table) for i in i_rates]

ax3.plot(i_rates * 100, annuity_by_rate, linewidth=2)
ax3.fill_between(i_rates * 100, 0, annuity_by_rate, alpha=0.2)
ax3.set_xlabel('Interest Rate (%)')
ax3.set_ylabel('aₓ (Age 65)')
ax3.set_title('Interest Rate Sensitivity\n(Higher rates → Lower values)')
ax3.grid(True, alpha=0.3)

# Plot 4: Immediate vs Certain annuity comparison
ax4 = axes[1, 0]
ages_comp = np.arange(50, 86, 5)
immediate_vals = []
certain_vals = []

for age in ages_comp:
    a_imm = immediate_annuity(age, 0.05, mortality_table)
    immediate_vals.append(a_imm)
    
    l_curr = mortality_table.loc[mortality_table['Age'] == age, 'l_x'].values[0]
    e_x = sum(mortality_table.loc[mortality_table['Age'] > age, 'l_x'].values) / l_curr
    n = int(e_x)
    
    v = 1 / 1.05
    d = 0.05 / 1.05
    a_cert = (1 - v**n) / d
    certain_vals.append(a_cert)

ax4.plot(ages_comp, immediate_vals, 'o-', linewidth=2, label='Immediate (mortality)', markersize=6)
ax4.plot(ages_comp, certain_vals, 's-', linewidth=2, label='Certain (no mortality)', markersize=6)
ax4.set_xlabel('Age')
ax4.set_ylabel('Annuity Value')
ax4.set_title('Life Annuity vs Annuity-Certain')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Payment stream visualization
ax5 = axes[1, 1]
age_stream = 70
years = np.arange(1, 31)
payments = []
cumulative_pv = []

v = 1 / 1.05
l_current = mortality_table.loc[mortality_table['Age'] == age_stream, 'l_x'].values[0]
cum_val = 0

for k in years:
    future_age = age_stream + k
    if future_age > 120:
        break
    
    l_future = mortality_table.loc[mortality_table['Age'] == future_age, 'l_x'].values
    if len(l_future) == 0 or l_future[0] <= 0:
        break
    
    k_p_x = l_future[0] / l_current
    pv_payment = (v ** k) * k_p_x
    payments.append(pv_payment)
    cum_val += pv_payment
    cumulative_pv.append(cum_val)

ax5.bar(years[:len(payments)], payments, alpha=0.7, edgecolor='black')
ax5.set_xlabel('Year')
ax5.set_ylabel('PV of Payment (discounted & mortality-adjusted)')
ax5.set_title(f'Payment Stream (Age {age_stream})\nEarlier payments worth more')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Cumulative present value
ax6 = axes[1, 2]
ax6.plot(years[:len(cumulative_pv)], cumulative_pv, linewidth=2)
ax6.axhline(immediate_annuity(age_stream, 0.05, mortality_table), 
           color='r', linestyle='--', linewidth=2, label=f'Total aₓ = {immediate_annuity(age_stream, 0.05, mortality_table):.2f}')
ax6.fill_between(years[:len(cumulative_pv)], 0, cumulative_pv, alpha=0.2)
ax6.set_xlabel('Years')
ax6.set_ylabel('Cumulative PV')
ax6.set_title(f'Cumulative Value Converges to a₇₀')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print(f"Immediate annuity (aₓ): Payments at period END conditional on survival")
print(f"Key relationships: aₓ = (1 - Aₓ)/d, äₓ = (1+i)·aₓ")
print(f"Mortality credit reduces value vs certain annuity; sensitive to interest rates")
```

## 6. Challenge Round
When is immediate annuity analysis problematic?
- **Mortality improvement**: Historical tables underestimate longevity; annuity underpriced if not updated
- **Adverse selection**: Healthier individuals purchase annuities; population mortality tables overstate death rates
- **Interest rate changes**: Long duration → high sensitivity; low rates inflate present values significantly
- **Inflation risk**: Fixed nominal payments erode purchasing power; need inflation-indexed variants
- **Liquidity**: Once purchased, annuities illiquid; cannot access lump sum in emergencies

## 7. Key References
- [Society of Actuaries - Annuity Mathematics](https://www.soa.org/) - Standard actuarial notation and formulas
- [Wiki - Life Annuity](https://en.wikipedia.org/wiki/Life_annuity) - Overview of annuity structures
- [Actuarial Mathematics (Bowers et al.)](https://www.actuary.org/) - Foundational textbook for life contingencies

---
**Status:** Core product for retirement income | **Complements:** Annuity Due, Life Insurance, Pension obligations
