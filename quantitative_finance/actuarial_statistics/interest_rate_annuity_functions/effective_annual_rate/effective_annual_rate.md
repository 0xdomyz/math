# Effective Annual Rate

## 1. Concept Skeleton
**Definition:** Annual interest rate i reflecting compounding effect; (1+i) = growth factor per year; standard actuarial discount rate  
**Purpose:** Standardize interest rates for comparison, price financial instruments, calculate present/future values, annuity valuation  
**Prerequisites:** Compound interest, time value of money, discount factor, present value concepts

## 2. Comparative Framing
| Rate Type | Effective Annual (i) | Nominal Rate (i^(m)) | Force of Interest (δ) |
|-----------|----------------------|---------------------|----------------------|
| **Compounding** | Once per year | m times per year | Continuous |
| **Formula** | (1+i) = principal growth | i^(m)/m per period | δ = ln(1+i) |
| **Use** | APY, annuity pricing | Mortgage, bonds | Theoretical, stochastic models |
| **Conversion** | i = (1 + i^(m)/m)^m - 1 | i^(m) = m[(1+i)^{1/m} - 1] | i = e^δ - 1 |

## 3. Examples + Counterexamples

**Simple Example:**  
Bank savings i = 0.03 (3% effective annual): $1,000 grows to $1,030 in one year; standard for long-term contracts

**Failure Case:**  
Confusing nominal 6% semi-annual with 6% effective: Actual effective = (1 + 0.06/2)² - 1 = 6.09%, not 6%

**Edge Case:**  
Very high inflation (i = 0.50 annually): Real return = (1 + 0.50)/(1 + inflation) - 1; nominal vs real distinction critical

## 4. Layer Breakdown
```
Effective Annual Rate Structure:
├─ Definition & Relationships:
│   ├─ (1+i) = growth multiplier after 1 year
│   ├─ i = (1 + i^(m)/m)^m - 1  [convert from nominal]
│   ├─ i = e^δ - 1  [convert from force of interest]
│   ├─ v = 1/(1+i)  [discount factor]
│   └─ v^n = 1/(1+i)^n  [present value of $1 in n years]
├─ Accumulation & Discounting:
│   ├─ Future Value: FV = PV · (1+i)^n
│   ├─ Present Value: PV = FV · v^n = FV/(1+i)^n
│   ├─ Annuity Present Value: PV = PMT · aₙ̄| = PMT · [1 - v^n]/i
│   └─ Annuity Future Value: FV = PMT · sₙ̄| = PMT · [(1+i)^n - 1]/i
├─ Period-to-Period Relationships:
│   ├─ 1-year ahead: (1+i)
│   ├─ 2-year ahead: (1+i)²
│   ├─ Mid-period: (1+i)^{1/m} for m periods per year
│   └─ Fractional: (1+i)^t for t ∈ [0,1] years
├─ Comparison Across Rates:
│   ├─ 5% effective = 4.88% nominal semi-annual
│   ├─ 5% effective = 4.88% nominal quarterly = 4.88% nominal monthly
│   ├─ 5% effective ≈ 4.879% force of interest
│   └─ Ranking: i^(∞) < δ < i (force continuous always between nominal and effective)
└─ Actuarial Applications:
    ├─ Annuity-certain: aₙ̄| = [1 - v^n]/i  (no mortality)
    ├─ Life annuity: aₓ = ∑ₖ₌₀^∞ v^k · ₖpₓ  (discounted survival)
    ├─ Bond pricing: Bond value = ∑ Coupon·v^t + Face·v^n
    └─ Pension liability: PV = ∑ Benefit·v^t
```

**Interaction:** Choose i → Calculate v, (1+i)^n → Discount all cash flows → Sum for present value

## 5. Mini-Project
Calculate effective rates, convert between formats, and price annuities:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. EFFECTIVE RATE FUNDAMENTALS
print("=" * 60)
print("EFFECTIVE ANNUAL RATE CALCULATIONS")
print("=" * 60)

i_effective = 0.05  # 5% effective annual rate

# Key derived values
v = 1 / (1 + i_effective)  # Discount factor
d = i_effective / (1 + i_effective)  # Discount rate (1 - v)
delta = np.log(1 + i_effective)  # Force of interest

print(f"\nGiven: Effective annual rate i = {i_effective:.4f} ({i_effective*100:.2f}%)")
print(f"Discount factor: v = 1/(1+i) = {v:.6f}")
print(f"Discount rate: d = i/(1+i) = {d:.6f}")
print(f"Force of interest: δ = ln(1+i) = {delta:.6f}")
print(f"Relationship check: d = 1 - v = {1 - v:.6f} ✓")
print()

# 2. ACCUMULATION & DISCOUNTING EXAMPLES
print("=" * 60)
print("ACCUMULATION AND DISCOUNTING")
print("=" * 60)

principal = 1000  # $1,000 initial investment
years = np.arange(0, 11)

# Future value: FV = PV * (1+i)^n
future_values = principal * (1 + i_effective) ** years

# Present value: PV = FV / (1+i)^n
future_amount = 1000
present_values = future_amount * v ** years

accumulation_df = pd.DataFrame({
    'Year': years,
    'FV of $1000': future_values.round(2),
    'PV of $1000': present_values.round(2),
    'Growth Factor': ((1 + i_effective) ** years).round(6),
    'Discount Factor': (v ** years).round(6)
})

print("\nAccumulation and Discounting:")
print(accumulation_df.to_string(index=False))
print()

# 3. CONVERT BETWEEN NOMINAL AND EFFECTIVE RATES
print("=" * 60)
print("CONVERSION: EFFECTIVE ↔ NOMINAL RATES")
print("=" * 60)

print(f"\nStarting with effective rate i = {i_effective:.4f}")

# Calculate nominal rates for different compounding frequencies
m_values = [1, 2, 4, 12, 365, np.inf]  # Annually, semi-annual, quarterly, monthly, daily, continuous

nominal_rates = []

for m in m_values:
    if m == np.inf:
        # Continuous: i = e^δ - 1, so δ = ln(1+i) = force of interest
        delta_val = np.log(1 + i_effective)
        nominal_rates.append(delta_val)
    else:
        # Nominal: i^(m) = m[(1+i)^(1/m) - 1]
        i_nom = m * ((1 + i_effective) ** (1/m) - 1)
        nominal_rates.append(i_nom)

conversion_df = pd.DataFrame({
    'Compounding': ['Annual (m=1)', 'Semi-Annual (m=2)', 'Quarterly (m=4)', 
                   'Monthly (m=12)', 'Daily (m=365)', 'Continuous'],
    'Nominal Rate': [f"{r*100:.4f}%" for r in nominal_rates],
    'Per-Period Rate': [f"{(r/m if m != np.inf else r)*100:.4f}%" if m != np.inf 
                       else f"δ={r*100:.4f}%" for m, r in zip(m_values, nominal_rates)],
    'Effective i': [f"{i_effective*100:.4f}%" for _ in m_values]
})

print(conversion_df.to_string(index=False))
print()

# Verify conversions back to effective
print("Verification: Convert nominal rates back to effective")
print("-" * 60)
for m, i_nom in zip(m_values[:-1], nominal_rates[:-1]):
    i_check = (1 + i_nom/m) ** m - 1
    print(f"m={m:3d}: i^({m}) = {i_nom:.6f} → i = {i_check:.6f} (original: {i_effective:.6f}) ✓")

# Continuous
delta_val = nominal_rates[-1]
i_check_continuous = np.exp(delta_val) - 1
print(f"Continuous: δ = {delta_val:.6f} → i = {i_check_continuous:.6f} (original: {i_effective:.6f}) ✓")
print()

# 4. ANNUITY-CERTAIN FUNCTIONS
print("=" * 60)
print("ANNUITY-CERTAIN FUNCTIONS (no mortality)")
print("=" * 60)

# aₙ̄| = [1 - v^n] / i (present value of annuity-immediate)
# sₙ̄| = [(1+i)^n - 1] / i (future value of annuity-immediate)

n_periods = np.array([1, 5, 10, 20, 30])
payment = 1000  # $1,000 per year

annuity_data = []

for n in n_periods:
    # Present value
    an_imm = (1 - v**n) / i_effective  # Annuity-immediate
    aend_pv = payment * an_imm
    
    # Annuity-due (payments at start of period): ä = a * (1+i)
    aDn = an_imm * (1 + i_effective)
    astart_pv = payment * aDn
    
    # Future value
    sn_imm = ((1 + i_effective)**n - 1) / i_effective
    aend_fv = payment * sn_imm
    
    # Annuity-due future value
    sDn = sn_imm * (1 + i_effective)
    astart_fv = payment * sDn
    
    annuity_data.append({
        'n': n,
        'aₙ̄| (imm)': an_imm,
        'äₙ̄| (due)': aDn,
        'PV (imm) $': aend_pv,
        'PV (due) $': astart_pv,
        'sₙ̄| (imm)': sn_imm,
        'FV (imm) $': aend_fv,
        'FV (due) $': astart_fv
    })

annuity_df = pd.DataFrame(annuity_data)

print(f"\nPayment = ${payment}/period, i = {i_effective:.4f}")
print("\nPresent Value of Annuities:")
print(annuity_df[['n', 'aₙ̄| (imm)', 'äₙ̄| (due)', 'PV (imm) $', 'PV (due) $']].to_string(index=False))
print("\nFuture Value of Annuities:")
print(annuity_df[['n', 'sₙ̄| (imm)', 'FV (imm) $', 'FV (due) $']].to_string(index=False))
print()

# 5. TERM STRUCTURE (multi-year rates)
print("=" * 60)
print("TERM STRUCTURE: RATES FOR VARIOUS TIME HORIZONS")
print("=" * 60)

# Assume flat yield curve at 5%
times = np.arange(0, 11)
spot_rates = np.full_like(times, i_effective, dtype=float)
discount_factors = (1 / (1 + i_effective)) ** times
present_values_unit = 1 / ((1 + i_effective) ** times)

term_structure = pd.DataFrame({
    'Year': times,
    'Spot Rate': [f"{r*100:.2f}%" for r in spot_rates],
    'Discount Factor': [f"{d:.6f}" for d in discount_factors],
    'PV of $1': [f"${v:.4f}" for v in present_values_unit]
})

print("\nFlat yield curve (all rates = 5%):")
print(term_structure.to_string(index=False))
print()

# 6. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accumulation growth
ax = axes[0, 0]
years_detailed = np.linspace(0, 20, 100)
principal_growth = 1000 * (1 + i_effective) ** years_detailed

ax.plot(years_detailed, principal_growth, linewidth=2.5, color='darkblue', label=f'i = {i_effective*100:.1f}%')
ax.fill_between(years_detailed, 1000, principal_growth, alpha=0.2, color='blue')

# Add comparison with other rates
for i_alt in [0.02, 0.04, 0.06]:
    if i_alt != i_effective:
        growth_alt = 1000 * (1 + i_alt) ** years_detailed
        ax.plot(years_detailed, growth_alt, linewidth=2, alpha=0.6, 
               label=f'i = {i_alt*100:.1f}%', linestyle='--')

ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Value ($)', fontsize=11)
ax.set_title('Accumulation: Growth of $1,000 at Different Rates', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Discount factors
ax = axes[0, 1]
for i_comp in [0.02, 0.05, 0.08]:
    v_comp = 1 / (1 + i_comp)
    discounts = v_comp ** years_detailed
    ax.plot(years_detailed, discounts, linewidth=2.5, 
           label=f'i = {i_comp*100:.1f}%', marker='o' if i_comp == i_effective else None,
           markersize=3, alpha=0.7)

ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Discount Factor v^n', fontsize=11)
ax.set_title('Discount Factors: Effect of Interest Rate', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Annuity present values
ax = axes[1, 0]
n_range = np.arange(1, 41)
an_values = (1 - (v ** n_range)) / i_effective
pv_annuity = 1000 * an_values

ax.plot(n_range, pv_annuity, linewidth=2.5, color='darkgreen', marker='o', 
       markersize=4, alpha=0.7, label='Annuity-immediate (aₙ̄|)')
ax.plot(n_range, pv_annuity * (1 + i_effective), linewidth=2.5, 
       color='coral', marker='s', markersize=4, alpha=0.7, linestyle='--',
       label='Annuity-due (äₙ̄|)')

ax.set_xlabel('Number of Periods (n)', fontsize=11)
ax.set_ylabel('Present Value of $1,000/period', fontsize=11)
ax.set_title('Annuity Present Values: Immediate vs Due', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 4: Conversion factors
ax = axes[1, 1]
compoundings = ['Annual', 'Semi-Anl', 'Quarterly', 'Monthly', 'Daily', 'Continuous']
nominal_pcts = [r * 100 for r in nominal_rates]

colors = plt.cm.viridis(np.linspace(0, 1, len(nominal_pcts)))
bars = ax.bar(compoundings, nominal_pcts, color=colors, edgecolor='black', linewidth=1.5)

# Add horizontal line for effective rate
ax.axhline(i_effective * 100, color='red', linestyle='--', linewidth=2, 
          label=f'Effective i = {i_effective*100:.2f}%')

ax.set_ylabel('Rate (%)', fontsize=11)
ax.set_title('Nominal Rates: Conversion from Effective i = 5%', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar, rate in zip(bars, nominal_pcts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{rate:.3f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('effective_annual_rate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print("Analysis complete. Chart saved as 'effective_annual_rate_analysis.png'")
print("=" * 60)
```

## 6. Challenge Round
When effective rates mislead:
- **Inflation illusion**: Nominal 3% effective with 4% inflation → real return = -0.96%
- **Compounding frequency switching**: Bond pays semi-annual but annuity priced with annual i; must convert to common base
- **Forecasting rates**: Current 2% rate may rise to 5% mid-contract; static i assumes stable environment
- **Negative rates**: Central bank policy (ECB, SNB) creates v > 1, breaking standard formulas; requires re-derivation
- **Redemption yield vs coupon**: Bond selling at discount requires iteration to find effective annual yield

## 7. Key References
- [Bowers et al., Actuarial Mathematics (Chapter 1)](https://www.soa.org/) - Fundamental rate relationships
- [Yield Curve Mathematics (Wikipedia)](https://en.wikipedia.org/wiki/Yield_curve) - Term structure concepts
- [Society of Actuaries Exam FM](https://www.soa.org/education/exam-req/edu-exam-fm-detail.aspx) - Practice problems

---
**Status:** Foundational rate concept | **Complements:** Nominal Rate, Force of Interest, Discount Factor, Annuity Functions
