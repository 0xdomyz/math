# Deferred Annuity (n|aₓ)

## 1. Concept Skeleton
**Definition:** Life annuity where first payment delayed n years; requires survival to deferral period end; notation n|aₓ (n-year deferred whole life annuity)  
**Purpose:** Model retirement income starting at future age, accumulation-phase products, pension obligations with vesting  
**Prerequisites:** Life annuities, survival probabilities, discount factors, accumulation concepts, vesting periods

## 2. Comparative Framing
| Type | First Payment | Value | Survival Requirement | Use Case |
|------|---------------|-------|---------------------|----------|
| **Deferred (n\|aₓ)** | After n years | v^n·ₙpₓ·aₓ₊ₙ | Must reach age x+n | Retirement planning |
| **Immediate (aₓ)** | End of year 1 | Higher than deferred | Ongoing | Current income need |
| **Term (aₓ:n̄\|)** | Years 1 to n only | Limited duration | Standard | Bridge income |
| **Whole Life (aₓ)** | Year 1 until death | n\|aₓ + aₓ:n̄\| = aₓ | Standard | Lifetime income |

## 3. Examples + Counterexamples

**Simple Example:**  
40-year-old buys deferred annuity starting at 65 (n=25); value = discount^25 × P(survive 25 years) × annuity value at 65

**Failure Case:**  
Ignoring survival probability: Using only discounting undervalues deferral; die before age 65 → no payments received

**Edge Case:**  
Extremely long deferral (n=50 at age 40): ₅₀p₄₀ very low; annuity value approaches zero due to survival probability

## 4. Layer Breakdown
```
Deferred Annuity Structure:
├─ Deferral Period:
│   ├─ Years 1 to n: No payments, accumulation phase
│   ├─ Requirement: Must survive to age x+n
│   ├─ Death before n: No payments received (forfeiture)
│   └─ Accumulation: Premiums compound or lump-sum investment grows
├─ Payment Phase:
│   ├─ Starts: Year n+1 (age x+n)
│   ├─ Continues: Until death (whole life annuity)
│   ├─ First payment: At end of year n+1 (age x+n+1)
│   └─ Standard life annuity from age x+n onward
├─ Present Value Formula:
│   ├─ n|aₓ = v^n · ₙpₓ · aₓ₊ₙ
│   │   where v = 1/(1+i), ₙpₓ = P(survive n years)
│   │         aₓ₊ₙ = immediate annuity value at age x+n
│   ├─ Interpretation: Discount n years × Survival probability × Future annuity value
│   ├─ Commutation functions: n|aₓ = Nₓ₊ₙ₊₁ / Dₓ
│   └─ Recursive: n|aₓ = v·pₓ · (n-1|aₓ₊₁) (dynamic programming)
├─ Relationship to Term Annuity:
│   ├─ Identity: aₓ = aₓ:n̄| + n|aₓ
│   │   (Whole life = Term + Deferred)
│   ├─ Decomposition: Present payments + Future payments = Total
│   ├─ Verification: Sum of term and deferred equals whole life
│   └─ Application: Value retirement income in segments
├─ Accumulation Phase:
│   ├─ Single premium: Lump sum invested at contract start
│   ├─ Flexible premiums: Periodic contributions during deferral
│   ├─ Growth: At credited rate (may differ from valuation rate)
│   ├─ Surrender value: Available if annuitized early (with penalty)
│   └─ Death benefit: Return of premiums if die during deferral (optional)
├─ Pricing Considerations:
│   ├─ Lower cost: Deferral reduces present value significantly
│   ├─ Mortality selection: Buyer must be healthy at purchase
│   ├─ Accumulation benefit: Time allows investment growth
│   └─ Longevity insurance: Protects against outliving assets
├─ Due Version:
│   ├─ n|äₓ = v^n · ₙpₓ · äₓ₊ₙ (payments at period start)
│   ├─ Relationship: n|äₓ = (1+i) · n|aₓ
│   └─ Application: Premium-paying products
└─ Practical Applications:
    ├─ Qualified Longevity Annuity Contract (QLAC): Deferred to age 80-85
    ├─ Pension vesting: Payments start after service period
    ├─ Retirement planning: Purchase at 40, payments start at 65
    └─ Longevity insurance: Coverage for late-life expenses
```

## 5. Mini-Project
Value and analyze deferred annuities:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Deferred Annuity (n|aₓ) Analysis ===\n")

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

# Immediate annuity (helper function)
def immediate_annuity(x, i, mortality):
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

# Deferred annuity
def deferred_annuity(x, n, i, mortality):
    """
    n|aₓ = v^n · ₙpₓ · aₓ₊ₙ
    """
    v = 1 / (1 + i)
    
    # Survival probability to age x+n
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    l_future = mortality.loc[mortality['Age'] == x + n, 'l_x'].values[0]
    n_p_x = l_future / l_current
    
    # Annuity value at age x+n
    a_x_plus_n = immediate_annuity(x + n, i, mortality)
    
    # Deferred value
    return (v ** n) * n_p_x * a_x_plus_n

# Term annuity (for decomposition)
def term_annuity(x, n, i, mortality):
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 0
    for k in range(1, n + 1):
        l_future = mortality.loc[mortality['Age'] == x + k, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        k_p_x = l_future[0] / l_current
        value += (v ** k) * k_p_x
    
    return value

# Calculate deferred annuities for different deferral periods
print("=== Deferred Annuity Values (Age 40, i = 5%) ===\n")
age_purchase = 40
i_rate = 0.05
deferral_periods = [10, 15, 20, 25, 30]

results = []
immediate_value = immediate_annuity(age_purchase, i_rate, mortality)

for n in deferral_periods:
    deferred_val = deferred_annuity(age_purchase, n, i_rate, mortality)
    
    # Components
    v = 1 / (1 + i_rate)
    l_curr = mortality.loc[mortality['Age'] == age_purchase, 'l_x'].values[0]
    l_fut = mortality.loc[mortality['Age'] == age_purchase + n, 'l_x'].values[0]
    n_p_x = l_fut / l_curr
    a_x_plus_n = immediate_annuity(age_purchase + n, i_rate, mortality)
    
    discount_factor = v ** n
    
    results.append({
        'Deferral (n)': n,
        'Age at Start': age_purchase + n,
        'Discount (v^n)': discount_factor,
        'Survival (ₙpₓ)': n_p_x,
        'Future Annuity (aₓ₊ₙ)': a_x_plus_n,
        'Deferred Value (n|aₓ)': deferred_val,
        '% of Immediate': deferred_val / immediate_value * 100
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format='%.4f'))
print(f"\nImmediate annuity (aₓ) at age {age_purchase}: {immediate_value:.4f}")

# Decomposition verification: aₓ = aₓ:n̄| + n|aₓ
print("\n=== Decomposition Identity: aₓ = aₓ:n̄| + n|aₓ ===\n")
n_test = 25
age_test = 40

whole_life = immediate_annuity(age_test, i_rate, mortality)
term_component = term_annuity(age_test, n_test, i_rate, mortality)
deferred_component = deferred_annuity(age_test, n_test, i_rate, mortality)
sum_components = term_component + deferred_component

print(f"Age {age_test}, deferral period {n_test} years:")
print(f"  Whole life (aₓ): {whole_life:.4f}")
print(f"  Term (aₓ:{n_test}̄|): {term_component:.4f}")
print(f"  Deferred ({n_test}|aₓ): {deferred_component:.4f}")
print(f"  Sum: {sum_components:.4f}")
print(f"  Identity holds: {abs(whole_life - sum_components) < 0.01}")

# Retirement planning scenario
print("\n=== Retirement Planning: Purchase at 40, Start at 65 ===\n")
age_purchase_ret = 40
age_retirement = 65
deferral_ret = age_retirement - age_purchase_ret
annual_income_desired = 50000  # $50k/year

# Calculate lump sum needed
deferred_value = deferred_annuity(age_purchase_ret, deferral_ret, i_rate, mortality)
lump_sum_needed = annual_income_desired / (1000 / deferred_value)  # Scale to $50k

# Alternative: Accumulation approach
# How much to invest today to have enough at 65?
v = 1 / (1 + i_rate)
l_curr = mortality.loc[mortality['Age'] == age_purchase_ret, 'l_x'].values[0]
l_ret = mortality.loc[mortality['Age'] == age_retirement, 'l_x'].values[0]
survival_prob = l_ret / l_curr

a_at_retirement = immediate_annuity(age_retirement, i_rate, mortality)
cost_per_dollar_at_retirement = 1 / a_at_retirement

future_value_needed = annual_income_desired * cost_per_dollar_at_retirement
present_value_accumulation = future_value_needed * (v ** deferral_ret)

print(f"Desired retirement income: ${annual_income_desired:,}/year starting at {age_retirement}")
print(f"\nApproach 1: Deferred Annuity")
print(f"  Deferred annuity value (per $1): {deferred_value:.4f}")
print(f"  Lump sum needed today: ${lump_sum_needed:,.2f}")
print(f"\nApproach 2: Accumulation")
print(f"  Survival probability to {age_retirement}: {survival_prob:.4f}")
print(f"  Annuity value at {age_retirement}: {a_at_retirement:.4f}")
print(f"  Future value needed (at {age_retirement}): ${future_value_needed:,.2f}")
print(f"  Present value (discounted): ${present_value_accumulation:,.2f}")
print(f"\nDifference: ${abs(lump_sum_needed - present_value_accumulation):,.2f}")
print(f"(Approaches should match; small differences from rounding)")

# Longevity insurance (QLAC-style)
print("\n=== Longevity Insurance: Deferred to Age 80 ===\n")
age_purchase_qlac = 60
age_start_qlac = 80
deferral_qlac = age_start_qlac - age_purchase_qlac

deferred_qlac = deferred_annuity(age_purchase_qlac, deferral_qlac, i_rate, mortality)
immediate_at_60 = immediate_annuity(age_purchase_qlac, i_rate, mortality)

# $100k premium
premium = 100000
annual_payment_deferred = premium * (1000 / deferred_qlac) / 1000
annual_payment_immediate = premium * (1000 / immediate_at_60) / 1000

print(f"$100,000 premium at age {age_purchase_qlac}:")
print(f"\nImmediate annuity (starts at {age_purchase_qlac+1}):")
print(f"  Annual payment: ${annual_payment_immediate:,.2f}")
print(f"\nDeferred to {age_start_qlac} (QLAC-style):")
print(f"  Annual payment: ${annual_payment_deferred:,.2f}")
print(f"  Payment ratio: {annual_payment_deferred / annual_payment_immediate:.2f}x higher")
print(f"\nTrade-off:")
print(f"  Immediate: Lower annual payment, starts now")
print(f"  Deferred: Much higher payment, covers late-life longevity risk")

# Age sensitivity
print("\n=== Purchase Age Impact (25-Year Deferral) ===\n")
deferral_fixed = 25
ages_purchase = [30, 35, 40, 45, 50, 55]

print(f"Deferral: {deferral_fixed} years")
print("Purchase Age | Start Age | Deferred Value | Immediate Value | Ratio")
print("-" * 75)

for age_purch in ages_purchase:
    age_start = age_purch + deferral_fixed
    if age_start > 95:
        continue
    
    deferred = deferred_annuity(age_purch, deferral_fixed, i_rate, mortality)
    immediate = immediate_annuity(age_purch, i_rate, mortality)
    ratio = deferred / immediate
    
    print(f"{age_purch:12d} | {age_start:9d} | {deferred:14.4f} | {immediate:15.4f} | {ratio:6.4f}")

# Interest rate sensitivity
print("\n=== Interest Rate Sensitivity (Age 40, 25-Year Deferral) ===\n")
interest_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

print("Interest | Deferred (n|aₓ) | Immediate (aₓ) | Ratio")
print("-" * 60)

for i_val in interest_rates:
    deferred = deferred_annuity(40, 25, i_val, mortality)
    immediate = immediate_annuity(40, i_val, mortality)
    ratio = deferred / immediate
    
    print(f"{i_val*100:7.0f}%  | {deferred:15.4f} | {immediate:14.4f} | {ratio:6.4f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Deferred value by deferral period
ax1 = axes[0, 0]
deferrals_plot = np.arange(5, 46, 2)
deferred_vals = [deferred_annuity(40, n, 0.05, mortality) for n in deferrals_plot]
immediate_val = immediate_annuity(40, 0.05, mortality)

ax1.plot(deferrals_plot, deferred_vals, 'o-', linewidth=2, markersize=6)
ax1.axhline(immediate_val, color='r', linestyle='--', linewidth=2, 
           label=f'Immediate = {immediate_val:.2f}')
ax1.fill_between(deferrals_plot, 0, deferred_vals, alpha=0.2)
ax1.set_xlabel('Deferral Period (years)')
ax1.set_ylabel('Deferred Annuity Value')
ax1.set_title('Value Decreases with Deferral\n(Age 40, i = 5%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Component decomposition
ax2 = axes[0, 1]
deferrals_decomp = [10, 15, 20, 25, 30]
discount_components = []
survival_components = []
future_annuity_components = []

v = 1 / 1.05
for n in deferrals_decomp:
    discount_components.append(v ** n)
    
    l_curr = mortality.loc[mortality['Age'] == 40, 'l_x'].values[0]
    l_fut = mortality.loc[mortality['Age'] == 40 + n, 'l_x'].values[0]
    survival_components.append(l_fut / l_curr)
    
    future_annuity_components.append(immediate_annuity(40 + n, 0.05, mortality))

x_pos = np.arange(len(deferrals_decomp))
width = 0.25

ax2_1 = ax2
ax2_1.bar(x_pos - width, discount_components, width, label='v^n', alpha=0.7, edgecolor='black')
ax2_1.bar(x_pos, survival_components, width, label='ₙpₓ', alpha=0.7, edgecolor='black')
ax2_2 = ax2_1.twinx()
ax2_2.plot(x_pos, future_annuity_components, 'ro-', linewidth=2, markersize=8, label='aₓ₊ₙ')

ax2_1.set_ylabel('Discount & Survival')
ax2_2.set_ylabel('Future Annuity Value', color='r')
ax2_1.set_xticks(x_pos)
ax2_1.set_xticklabels(deferrals_decomp)
ax2_1.set_xlabel('Deferral Period')
ax2_1.set_title('Components: n|aₓ = v^n·ₙpₓ·aₓ₊ₙ')
ax2_1.legend(loc='upper left')
ax2_2.legend(loc='upper right')
ax2_1.grid(True, alpha=0.3, axis='y')

# Plot 3: Decomposition identity visualization
ax3 = axes[0, 2]
deferrals_identity = [5, 10, 15, 20, 25, 30]
term_vals = [term_annuity(40, n, 0.05, mortality) for n in deferrals_identity]
deferred_vals_identity = [deferred_annuity(40, n, 0.05, mortality) for n in deferrals_identity]

x_pos = np.arange(len(deferrals_identity))
ax3.bar(x_pos, term_vals, label='Term (aₓ:n̄|)', alpha=0.7, edgecolor='black')
ax3.bar(x_pos, deferred_vals_identity, bottom=term_vals, label='Deferred (n|aₓ)', alpha=0.7, edgecolor='black')
ax3.axhline(immediate_annuity(40, 0.05, mortality), color='r', linestyle='--', 
           linewidth=2, label='Whole life (aₓ)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(deferrals_identity)
ax3.set_xlabel('Split Point (n years)')
ax3.set_ylabel('Annuity Value')
ax3.set_title('Identity: aₓ = aₓ:n̄| + n|aₓ')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Purchase age impact
ax4 = axes[1, 0]
ages_purch_plot = np.arange(30, 66, 2)
deferred_25_by_age = [deferred_annuity(age, 25, 0.05, mortality) for age in ages_purch_plot]
immediate_by_age = [immediate_annuity(age, 0.05, mortality) for age in ages_purch_plot]

ax4.plot(ages_purch_plot, deferred_25_by_age, 'o-', linewidth=2, label='25-year deferred', markersize=5)
ax4.plot(ages_purch_plot, immediate_by_age, 's-', linewidth=2, label='Immediate', markersize=5)
ax4.set_xlabel('Purchase Age')
ax4.set_ylabel('Annuity Value')
ax4.set_title('Age Impact on Deferred Value\n(Both decrease with age)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Interest rate sensitivity
ax5 = axes[1, 1]
i_range = np.linspace(0.02, 0.08, 20)
deferred_by_i = [deferred_annuity(40, 25, i, mortality) for i in i_range]
immediate_by_i = [immediate_annuity(40, i, mortality) for i in i_range]

ax5.plot(i_range * 100, deferred_by_i, linewidth=2, label='25-year deferred')
ax5.plot(i_range * 100, immediate_by_i, linewidth=2, label='Immediate')
ax5.set_xlabel('Interest Rate (%)')
ax5.set_ylabel('Annuity Value (Age 40)')
ax5.set_title('Interest Rate Impact\n(Deferred more sensitive)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Ratio of deferred to immediate
ax6 = axes[1, 2]
deferrals_ratio = np.arange(5, 41, 2)
ratios = [deferred_annuity(40, n, 0.05, mortality) / immediate_annuity(40, 0.05, mortality) 
          for n in deferrals_ratio]

ax6.plot(deferrals_ratio, ratios, 'o-', linewidth=2, markersize=6, color='purple')
ax6.fill_between(deferrals_ratio, 0, ratios, alpha=0.2, color='purple')
ax6.set_xlabel('Deferral Period (years)')
ax6.set_ylabel('Ratio: n|aₓ / aₓ')
ax6.set_title('Deferred as % of Immediate\n(Exponential decay)')
ax6.grid(True, alpha=0.3)
ax6.set_ylim(bottom=0)

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print("Deferred annuity (n|aₓ): First payment after n years; n|aₓ = v^n·ₙpₓ·aₓ₊ₙ")
print("Identity: aₓ = aₓ:n̄| + n|aₓ (term + deferred = whole life)")
print("Applications: Retirement planning, longevity insurance (QLAC), pension vesting")
```

## 6. Challenge Round
When is deferred annuity analysis problematic?
- **Forfeiture risk**: Death before deferral end → total loss of premiums (unless death benefit included)
- **Interest rate risk**: Long deferral period sensitive to rate changes; valuation volatile
- **Mortality improvement**: If longevity increases, survival probability rises; deferred annuity more valuable
- **Inflation**: Long deferral erodes real value of fixed payments; need inflation indexing
- **Liquidity**: Cannot access funds during deferral without surrender penalty; inflexible

## 7. Key References
- [Society of Actuaries - Deferred Annuities](https://www.soa.org/) - Standard formulas and applications
- [Wiki - Deferred Annuity](https://en.wikipedia.org/wiki/Deferred_annuity) - Product overview
- [QLAC Regulations](https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-required-minimum-distributions-rmds) - Qualified Longevity Annuity Contract rules

---
**Status:** Key retirement planning tool, longevity insurance | **Complements:** Term Annuity, Whole Life Annuity, QLAC products
