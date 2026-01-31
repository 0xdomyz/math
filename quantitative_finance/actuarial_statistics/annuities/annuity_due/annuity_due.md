# Annuity Due (äₓ)

## 1. Concept Skeleton
**Definition:** Series of payments made at START of each period conditional on survival; valued one period higher than immediate annuity due to earlier payment timing  
**Purpose:** Value prepaid insurance premiums, rent, leases where payment precedes period consumption  
**Prerequisites:** Immediate annuity concepts, discount factors, mortality tables, present value timing differences

## 2. Comparative Framing
| Type | Payment Timing | First Payment | Value Relation | Common Use |
|------|----------------|---------------|----------------|------------|
| **Annuity Due** | Start of period | At t=0 (immediate) | äₓ = (1+i)·aₓ | Prepaid rent/premiums |
| **Immediate** | End of period | At t=1 (after 1 year) | Lower than due | Pension payments |
| **Continuous** | Instantaneous | Infinitesimal intervals | āₓ ≈ (äₓ + aₓ)/2 | Theoretical models |
| **Due m-thly** | Start of 1/m periods | m times per year | Higher frequency | Monthly insurance premiums |

## 3. Examples + Counterexamples

**Simple Example:**  
Rent $1000/month paid at month start vs month end; due value = immediate × (1 + monthly rate) = immediate × 1.004167

**Failure Case:**  
Confusing due with immediate in premium calculations; underprices insurance by one period's interest/mortality credit

**Edge Case:**  
First payment guaranteed (äₓ always ≥ 1 since payment at t=0); immediate annuity aₓ can be < 1 at very high ages

## 4. Layer Breakdown
```
Annuity Due Structure:
├─ Payment Schedule:
│   ├─ First payment: Time 0 (immediate, no waiting)
│   ├─ Subsequent payments: Start of years 1, 2, 3, ..., until death
│   ├─ Conditionality: Payment k requires survival to age x+k-1
│   └─ Guaranteed first payment (alive at contract start)
├─ Present Value Formula:
│   ├─ äₓ = Σ(k=0 to ∞) vᵏ · ₖpₓ
│   │   where k=0 term = 1 (guaranteed payment at t=0)
│   ├─ Relationship to immediate: äₓ = 1 + aₓ (first payment + remaining)
│   ├─ Alternative: äₓ = (1 + i) · aₓ (one period's interest advantage)
│   └─ Recursive: äₓ = 1 + v·pₓ·äₓ₊₁ (payment now + future due value)
├─ Computational Shortcut:
│   ├─ Given aₓ: äₓ = aₓ · (1 + i) = aₓ / v
│   ├─ From commutation functions: äₓ = Nₓ/Dₓ (vs aₓ = Nₓ₊₁/Dₓ)
│   ├─ Identity: äₓ - aₓ = 1 (difference is first payment)
│   └─ Continuous approximation: āₓ ≈ (äₓ + aₓ)/2 (Woolhouse)
├─ Premium Calculation:
│   ├─ Premiums typically paid at period start (annuity due structure)
│   ├─ Net premium: P̈ · äₓ = Aₓ (benefits discounted to present)
│   ├─ Advantage: Collects premium before period risk
│   └─ Timing matters: Due premiums reduce reserve strain vs immediate
├─ Frequency Considerations:
│   ├─ Annual due: äₓ = (1+i)·aₓ
│   ├─ Semi-annual: ä⁽²⁾ₓ ≈ äₓ - (1/4) (Woolhouse approximation)
│   ├─ Monthly: ä⁽¹²⁾ₓ ≈ äₓ - (11/24) (higher frequency → slightly lower PV)
│   └─ Continuous limit: āₓ = limₘ→∞ ä⁽ᵐ⁾ₓ
└─ Practical Applications:
    ├─ Life insurance premiums: Paid at start of coverage period
    ├─ Rent/lease contracts: Monthly rent due at month start
    ├─ Pension contributions: Employer contributions at period start
    └─ Annuity income: Retiree receives payment immediately upon period start
```

## 5. Mini-Project
Compare annuity due vs immediate annuity across scenarios:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Annuity Due (äₓ) vs Immediate Annuity (aₓ) ===\n")

# Simplified mortality table (Gompertz)
def build_mortality_table(max_age=120):
    ages = np.arange(0, max_age + 1)
    A, B, C = 0.0001, 1.08, 0.00035
    mu_x = A + C * (B ** ages)
    q_x = 1 - np.exp(-mu_x)
    
    l_x = np.zeros(len(ages))
    l_x[0] = 100000
    for i in range(1, len(ages)):
        l_x[i] = l_x[i-1] * (1 - q_x[i-1])
    
    return pd.DataFrame({'Age': ages, 'l_x': l_x, 'q_x': q_x})

mortality = build_mortality_table()

# Immediate annuity function
def immediate_annuity(x, i, mortality, max_age=120):
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 0
    for k in range(1, max_age - x + 1):
        future_age = x + k
        if future_age > max_age:
            break
        l_future = mortality.loc[mortality['Age'] == future_age, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        
        k_p_x = l_future[0] / l_current
        value += (v ** k) * k_p_x
    
    return value

# Annuity due: Two calculation methods
def annuity_due_from_immediate(x, i, mortality):
    """Method 1: äₓ = (1+i) · aₓ"""
    a_x = immediate_annuity(x, i, mortality)
    return (1 + i) * a_x

def annuity_due_direct(x, i, mortality, max_age=120):
    """Method 2: Direct calculation äₓ = Σ(k=0 to ∞) vᵏ · ₖpₓ"""
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 1.0  # k=0 term (payment at t=0)
    for k in range(1, max_age - x + 1):
        future_age = x + k
        if future_age > max_age:
            break
        l_future = mortality.loc[mortality['Age'] == future_age, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        
        k_p_x = l_future[0] / l_current
        value += (v ** k) * k_p_x
    
    return value

# Calculate values for various ages
print("=== Annuity Comparison (i = 5%) ===\n")
ages_test = [30, 45, 60, 75, 90]
i_rate = 0.05

results = []
for age in ages_test:
    a_x = immediate_annuity(age, i_rate, mortality)
    a_due_method1 = annuity_due_from_immediate(age, i_rate, mortality)
    a_due_method2 = annuity_due_direct(age, i_rate, mortality)
    
    difference = a_due_method1 - a_x
    ratio = a_due_method1 / a_x if a_x > 0 else 0
    
    results.append({
        'Age': age,
        'Immediate (aₓ)': a_x,
        'Due (äₓ)': a_due_method1,
        'Difference': difference,
        'Ratio (äₓ/aₓ)': ratio
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format='%.3f'))

# Verify relationship: äₓ = (1+i)·aₓ
print(f"\n=== Verification: äₓ = (1+i)·aₓ ===\n")
age_verify = 65
a_immediate = immediate_annuity(age_verify, i_rate, mortality)
a_due_formula = (1 + i_rate) * a_immediate
a_due_direct_calc = annuity_due_direct(age_verify, i_rate, mortality)

print(f"Age {age_verify}, i = {i_rate*100:.0f}%:")
print(f"  aₓ (immediate): {a_immediate:.4f}")
print(f"  äₓ = (1+i)·aₓ: {a_due_formula:.4f}")
print(f"  äₓ (direct): {a_due_direct_calc:.4f}")
print(f"  Match: {abs(a_due_formula - a_due_direct_calc) < 0.001}")

# Premium implications
print("\n=== Premium Calculation: Due vs Immediate ===\n")

# Whole life insurance benefit = $100,000
benefit = 100000
age_insured = 40

# Whole life insurance (single premium)
def whole_life_insurance(x, i, mortality, max_age=120):
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 0
    for k in range(1, max_age - x + 1):
        age_start = x + k - 1
        age_end = x + k
        if age_end > max_age:
            break
        
        l_start = mortality.loc[mortality['Age'] == age_start, 'l_x'].values
        l_end = mortality.loc[mortality['Age'] == age_end, 'l_x'].values
        if len(l_start) == 0 or len(l_end) == 0:
            break
        
        deaths = l_start[0] - l_end[0]
        k_minus_1_q_x = deaths / l_current
        value += (v ** k) * k_minus_1_q_x
    
    return value

A_x = whole_life_insurance(age_insured, i_rate, mortality)
single_premium = A_x * benefit

print(f"Whole life insurance (${benefit:,} benefit) at age {age_insured}:")
print(f"  Single premium: ${single_premium:,.2f}")
print(f"  Aₓ = {A_x:.4f}")

# Annual premiums
a_immediate = immediate_annuity(age_insured, i_rate, mortality)
a_due = annuity_due_from_immediate(age_insured, i_rate, mortality)

# Net annual premium: P·äₓ = Aₓ·benefit
premium_due = (A_x * benefit) / a_due
premium_immediate = (A_x * benefit) / a_immediate

print(f"\nNet annual premium:")
print(f"  Paid at start (due): ${premium_due:,.2f}")
print(f"  Paid at end (immediate): ${premium_immediate:,.2f}")
print(f"  Difference: ${premium_immediate - premium_due:,.2f} ({(premium_immediate/premium_due - 1)*100:.1f}% higher)")

# Interest rate sensitivity
print("\n=== Interest Rate Sensitivity ===\n")
interest_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
age_sens = 65

print(f"Age {age_sens} Annuity Values:")
print("Interest | Immediate (aₓ) | Due (äₓ) | Difference | Ratio")
print("-" * 65)

for i_val in interest_rates:
    a_imm = immediate_annuity(age_sens, i_val, mortality)
    a_d = annuity_due_from_immediate(age_sens, i_val, mortality)
    diff = a_d - a_imm
    ratio = a_d / a_imm if a_imm > 0 else 0
    
    print(f"{i_val*100:7.0f}%  | {a_imm:14.3f} | {a_d:11.3f} | {diff:9.3f} | {ratio:6.4f}")

# Monthly vs annual payments
print("\n=== Payment Frequency: Monthly vs Annual ===\n")

# Approximate monthly annuity due using Woolhouse
def annuity_due_monthly(x, i, mortality):
    """
    ä⁽¹²⁾ₓ ≈ äₓ - (11/24)
    Woolhouse approximation for monthly payments
    """
    a_due_annual = annuity_due_from_immediate(x, i, mortality)
    return a_due_annual - (11/24)

age_monthly = 65
a_annual_due = annuity_due_from_immediate(age_monthly, i_rate, mortality)
a_monthly_due = annuity_due_monthly(age_monthly, i_rate, mortality)

print(f"Age {age_monthly}, i = {i_rate*100:.0f}%:")
print(f"  Annual due (äₓ): {a_annual_due:.3f}")
print(f"  Monthly due (ä⁽¹²⁾ₓ): {a_monthly_due:.3f}")
print(f"  Difference: {a_annual_due - a_monthly_due:.3f}")
print(f"\nFor $1000/month annuity:")
print(f"  Annual value: ${a_annual_due * 12000:,.2f}")
print(f"  Monthly value: ${a_monthly_due * 12000:,.2f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Due vs Immediate by age
ax1 = axes[0, 0]
ages_plot = np.arange(25, 96, 5)
immediate_vals = [immediate_annuity(age, 0.05, mortality) for age in ages_plot]
due_vals = [annuity_due_from_immediate(age, 0.05, mortality) for age in ages_plot]

ax1.plot(ages_plot, immediate_vals, 'o-', linewidth=2, label='Immediate (aₓ)', markersize=6)
ax1.plot(ages_plot, due_vals, 's-', linewidth=2, label='Due (äₓ)', markersize=6)
ax1.set_xlabel('Age')
ax1.set_ylabel('Annuity Value')
ax1.set_title('Annuity Due vs Immediate\n(i = 5%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Difference (äₓ - aₓ)
ax2 = axes[0, 1]
differences = np.array(due_vals) - np.array(immediate_vals)

ax2.plot(ages_plot, differences, 'o-', linewidth=2, color='green', markersize=6)
ax2.fill_between(ages_plot, 0, differences, alpha=0.2, color='green')
ax2.set_xlabel('Age')
ax2.set_ylabel('Difference (äₓ - aₓ)')
ax2.set_title('Value Advantage of Annuity Due\n(Earlier payment timing)')
ax2.grid(True, alpha=0.3)

# Plot 3: Ratio (äₓ/aₓ)
ax3 = axes[0, 2]
ratios = np.array(due_vals) / np.array(immediate_vals)

ax3.plot(ages_plot, ratios, 'o-', linewidth=2, color='purple', markersize=6)
ax3.axhline(1.05, color='r', linestyle='--', linewidth=2, label='1+i = 1.05')
ax3.set_xlabel('Age')
ax3.set_ylabel('Ratio (äₓ/aₓ)')
ax3.set_title('Ratio Approaches 1+i\n(Perfect at all ages)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Interest rate sensitivity
ax4 = axes[1, 0]
i_range = np.linspace(0.01, 0.10, 30)
immediate_by_i = [immediate_annuity(65, i, mortality) for i in i_range]
due_by_i = [annuity_due_from_immediate(65, i, mortality) for i in i_range]

ax4.plot(i_range * 100, immediate_by_i, linewidth=2, label='Immediate')
ax4.plot(i_range * 100, due_by_i, linewidth=2, label='Due')
ax4.set_xlabel('Interest Rate (%)')
ax4.set_ylabel('Annuity Value (Age 65)')
ax4.set_title('Interest Rate Impact\n(Both decrease with higher rates)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Premium comparison
ax5 = axes[1, 1]
ages_premium = np.arange(30, 71, 5)
premium_immediate = []
premium_due = []

for age in ages_premium:
    A_x = whole_life_insurance(age, 0.05, mortality)
    a_imm = immediate_annuity(age, 0.05, mortality)
    a_d = annuity_due_from_immediate(age, 0.05, mortality)
    
    premium_immediate.append((A_x * 100000) / a_imm)
    premium_due.append((A_x * 100000) / a_d)

ax5.plot(ages_premium, premium_immediate, 'o-', linewidth=2, label='End of period', markersize=6)
ax5.plot(ages_premium, premium_due, 's-', linewidth=2, label='Start of period (due)', markersize=6)
ax5.set_xlabel('Issue Age')
ax5.set_ylabel('Annual Net Premium ($)')
ax5.set_title('Premium Timing Effect\n($100k Whole Life)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Payment stream visualization
ax6 = axes[1, 2]
years = np.arange(0, 21)
age_stream = 65

# Payment values for each year
immediate_payments = []
due_payments = []

v = 1 / 1.05
l_current = mortality.loc[mortality['Age'] == age_stream, 'l_x'].values[0]

for k in years:
    future_age = age_stream + k
    if future_age > 120:
        break
    
    l_future = mortality.loc[mortality['Age'] == future_age, 'l_x'].values
    if len(l_future) == 0 or l_future[0] <= 0:
        break
    
    k_p_x = l_future[0] / l_current
    
    # Due: payment at start of year k
    due_pv = (v ** k) * k_p_x
    due_payments.append(due_pv)
    
    # Immediate: payment at end of year k (one year later)
    if k > 0:
        imm_pv = (v ** k) * k_p_x
        immediate_payments.append(imm_pv)

width = 0.4
x_pos = np.arange(len(immediate_payments))

ax6.bar(x_pos - width/2, immediate_payments, width, label='Immediate (end)', alpha=0.7, edgecolor='black')
ax6.bar(x_pos + width/2, due_payments[1:len(immediate_payments)+1], width, label='Due (start)', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Payment Number')
ax6.set_ylabel('Present Value')
ax6.set_title('Payment Timing Comparison\n(Age 65, first 20 payments)')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print("Annuity due (äₓ): Payments at period START, äₓ = (1+i)·aₓ")
print("Key identity: äₓ - aₓ = 1 (first payment difference)")
print("Used for premiums, rent, prepaid contracts; always higher value than immediate")
```

## 6. Challenge Round
When is annuity due analysis problematic?
- **Timing confusion**: Mixed payment schedules (some due, some immediate) hard to value; need careful period-by-period analysis
- **Mid-period changes**: If payments switch from due to immediate (contract modification); requires hybrid valuation
- **Frequency mismatch**: Premium monthly but benefits annual; need consistent period conversion
- **First payment uncertainty**: If first payment delayed (underwriting period); due assumption overstates value
- **Tax timing**: Due payments may have different tax treatment depending on jurisdiction cash vs accrual rules

## 7. Key References
- [Society of Actuaries - Annuity Due](https://www.soa.org/) - Standard actuarial formulas and notation
- [Wiki - Annuity Due](https://en.wikipedia.org/wiki/Annuity_(finance)#Annuity_due) - Payment timing concepts
- [Actuarial Mathematics (Bowers)](https://www.actuary.org/) - Foundational relationships between annuity types

---
**Status:** Standard for prepaid contracts (rent, premiums) | **Complements:** Immediate Annuity, Premium Calculation, Life Insurance
