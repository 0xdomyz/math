# Premium Reserves

## 1. Concept Skeleton
**Definition:** Liability reserve held at policy level to cover future obligations; V(t) = PV(future benefits) - PV(future premiums)  
**Purpose:** Statutory minimum solvency requirement; policyholder protection; measure of contract profitability over time  
**Prerequisites:** Net premium, benefit payable, remaining term, discount rate, current age/health status

## 2. Comparative Framing
| Reserve Type | Formula | Basis | Timing | Purpose |
|--------------|---------|-------|--------|---------|
| **Prospective** | V = PV(Benef) - PV(Prem) | Future obligations | End of year | Standard, forward-looking |
| **Retrospective** | V = Prem acc - Claims pd | Past accumulation | End of year | Audit check, backward-looking |
| **Net premium** | Based on net premium only | Statutory minimum | Regulatory | Floor for reserve adequacy |
| **Gross premium** | Based on gross premium collected | Realistic | GAAP/IFRS | Balance sheet liability |
| **Modified** | Reduced Y1 (FPT smoothing) | Blended | First-year issues | Avoid first-year drain |

## 3. Examples + Counterexamples

**Simple Example:**  
20-year term, age 40, end of year 3: Remaining 17 years; PV(benefits) = $8,000, PV(premiums) = $7,200 → Reserve = $800

**Failure Case:**  
Premium set too low initially; reserve = $0 at issue (good for first-year income but negative reserve year 2 if mortality worse than assumed)

**Edge Case:**  
Whole life at age 85: Reserve approaches face amount; PV(premiums) near zero; reserve ≈ benefit amount

## 4. Layer Breakdown
```
Premium Reserve Structure:
├─ Reserve Calculation Methods:
│   ├─ Prospective (standard):
│   │   ├─ Formula: ₜV = A̅ₓ₊ₜ:n̄₋ₜ| - P · ä̅ₓ₊ₜ:n̄₋ₜ|
│   │   ├─ Components:
│   │   │   ├─ A̅ₓ₊ₜ:n̄₋ₜ| = APV of remaining benefit (at age x+t)
│   │   │   ├─ P = Net or gross premium in force
│   │   │   └─ ä̅ₓ₊ₜ:n̄₋ₜ| = APV of remaining premiums
│   │   └─ Interpretation: Reserve = expected future obligations discounted
│   ├─ Retrospective (verification):
│   │   ├─ Formula: ₜV = [Premiums accd - Claims pd] × (1+i)ᵗ
│   │   ├─ Components:
│   │   │   ├─ Premiums with interest: Σ P(1+i)^(t-s)
│   │   │   ├─ Claims with interest: Σ Claims(1+i)^(t-s)
│   │   │   └─ Profit recognition: Difference between methods
│   │   └─ Should equal prospective for verification
│   ├─ Statutes of Practice (ASOP):
│   │   ├─ Net premium reserve: Regulatory minimum
│   │   ├─ Gross premium reserve: GAAP standard
│   │   └─ Deficiency reserve: If gross < net, add difference
│   └─ Modified Reserve (First-year):
│       ├─ Issue: Reserve = $0 (unrealistic if claims incurred)
│       ├─ Solution: Reduced first-year premium for reserve calc
│       ├─ FPT: Fuller Prospective Test blends net/gross
│       └─ Outcome: Positive reserve year 1 if justified by experience
├─ Reserve at Different Policy Stages:
│   ├─ At issue (t=0): V₀ = 0 (for term insurance), or negative for whole life
│   ├─ First year (t=1): V₁ = A̅ₓ₊₁:n̄₋₁| - P · ä̅ₓ₊₁:n̄₋₁| > 0 (usually)
│   ├─ Mid-term (t=n/2): V = increases, typically 5-20% of benefit
│   ├─ Final year (t=n-1): V ≈ benefit - one year premium
│   └─ Mature contract: Large reserve as benefit approaches certainty
├─ Impact of Assumptions:
│   ├─ Mortality assumption:
│   │   ├─ If actual > assumed: Reserve too low; shortfall emerges
│   │   ├─ If actual < assumed: Reserve excess; profit at end
│   │   └─ Sensitivity: Small % change → large reserve impact for whole life
│   ├─ Interest assumption:
│   │   ├─ If actual < assumed: Reserve too low; must add deficiency
│   │   ├─ If actual > assumed: Reserve excess; but reinvestment risk
│   │   └─ Sensitivity: 1% change in rate → 5-15% change in reserve
│   ├─ Lapse assumption:
│   │   ├─ If actual > assumed: Fewer policies in-force; average reserve lower
│   │   ├─ If actual < assumed: More policies persist; higher aggregate reserve
│   │   └─ Impact: Moderate; main effect on cash flows, not reserves
│   └─ Expenses:
│       ├─ Gross premium reserve > Net premium reserve (by design)
│       ├─ Difference = PV(future expense recovery)
│       └─ If expenses exceed projection, deficiency reserve required
├─ Regulation & Accounting:
│   ├─ Statutory (US):
│   │   ├─ Minimum reserve formula: Reserve ≥ Net premium reserve
│   │   ├─ Deficiency reserve: If gross premium reserve < net, add
│   │   └─ Supervisory approval: For alternative methodologies
│   ├─ GAAP (US GAAP, emerging standards):
│   │   ├─ Contract liability: Gross premium reserve + service margin
│   │   └─ Service margin release: Over contract lifetime as profit realized
│   ├─ Solvency II (EU):
│   │   ├─ Technical provision: Best estimate + risk margin
│   │   ├─ Best estimate: Realistic assumption liabilities
│   │   └─ Risk margin: 5-10% over best estimate
│   └─ IFRS 17 (Global):
│       ├─ Liability for remaining coverage: Future claims & expenses
│       ├─ Service margin: Unearned profit at issue
│       └─ Release profile: Proportional to coverage or benefits
└─ Cash Value & Surrender:
    ├─ Definition: CSV = Reserve - Surrender charge (if any)
    ├─ Typical structure: CSV = Reserve - charge (5-10% of reserve)
    ├─ Surrender charge purpose: Prevent lapse arbitrage
    ├─ Example: Reserve $5,000, charge 7% → CSV = $4,650
    └─ Regulatory: CSV must be non-forfeiture option
```

**Key Insight:** Reserve grows as time passes and benefit certainty increases; sudden drops if benefits paid (death, surrender)

## 5. Mini-Project
Calculate reserves prospectively and retrospectively, analyze growth patterns, and test sensitivity:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP
print("=" * 80)
print("PREMIUM RESERVE ANALYSIS: PROSPECTIVE & RETROSPECTIVE")
print("=" * 80)

# Mortality table
mortality_data = {
    25: 0.00067, 30: 0.00084, 35: 0.00103, 40: 0.00131, 45: 0.00172,
    50: 0.00233, 55: 0.00325, 60: 0.00459, 65: 0.00653, 70: 0.00933,
    75: 0.01330, 80: 0.01934, 85: 0.02873
}

interest_rate = 0.04
v = 1 / (1 + interest_rate)
benefit = 200000

print(f"\nAssumptions:")
print(f"  Benefit: ${benefit:,.0f}")
print(f"  Interest: {interest_rate*100:.1f}%\n")

def calculate_apv_benefits(start_age, remaining_term, mortality_dict, interest_rate, benefit_amt):
    """APV of benefits for remaining term"""
    
    v_rate = 1 / (1 + interest_rate)
    apv = 0
    kpx = 1.0
    
    for k in range(1, remaining_term + 1):
        age_k = start_age + k - 1
        qx_k = mortality_dict.get(age_k, 0.001)
        vk = v_rate ** k
        
        apv += kpx * qx_k * vk * benefit_amt
        px_k = 1 - qx_k
        kpx *= px_k
    
    return apv

def calculate_apv_annuity(start_age, remaining_term, mortality_dict, interest_rate):
    """APV of premium annuity (annuity-due) for remaining term"""
    
    v_rate = 1 / (1 + interest_rate)
    apv = 0
    kpx = 1.0
    
    for k in range(0, remaining_term):
        vk = v_rate ** k
        apv += kpx * vk
        
        if k < remaining_term - 1:
            age_k = start_age + k
            qx_k = mortality_dict.get(age_k, 0.001)
            px_k = 1 - qx_k
            kpx *= px_k
    
    return apv

# 2. TERM INSURANCE RESERVE ANALYSIS
print("=" * 80)
print("TERM INSURANCE: 20-YEAR TERM, AGE 40 AT ISSUE")
print("=" * 80)

start_age = 40
original_term = 20
net_premium = 340  # Assume calculated net premium

print(f"\nNet Premium: ${net_premium:,.2f}/year\n")

# Calculate reserves for each policy year
reserve_data = []

print(f"{'Year':<8} {'Age':<8} {'Rem.Term':<10} {'APV Benef':<15} {'APV Prem':<15} {'Reserve':<15} {'% Benefit':<12}")
print("-" * 95)

for year in range(1, original_term + 1):
    age_now = start_age + year - 1
    remaining_term = original_term - year + 1
    
    # Prospective reserve
    apv_b = calculate_apv_benefits(age_now, remaining_term, mortality_data, interest_rate, benefit)
    apv_p = calculate_apv_annuity(age_now, remaining_term, mortality_data, interest_rate)
    
    reserve_prosp = apv_b - (net_premium * apv_p)
    reserve_pct = (reserve_prosp / benefit) * 100
    
    reserve_data.append({
        'year': year,
        'age': age_now,
        'remaining_term': remaining_term,
        'apv_benefit': apv_b,
        'apv_annuity': apv_p,
        'reserve': reserve_prosp
    })
    
    print(f"{year:<8} {age_now:<8} {remaining_term:<10} ${apv_b:<14,.2f} {apv_p:<15.6f} ${reserve_prosp:<14,.2f} {reserve_pct:<11.1f}%")

print()

# 3. RETROSPECTIVE VERIFICATION
print("=" * 80)
print("RETROSPECTIVE RESERVE VERIFICATION (Sample Year)")
print("=" * 80)

year_check = 5
reserve_retro = 0
premiums_collected = 0

print(f"\nYear {year_check} Reserve Verification:\n")

for y in range(1, year_check + 1):
    age_y = start_age + y - 1
    
    # Assume no claims for this illustration
    premium_with_interest = net_premium * (1 + interest_rate) ** (year_check - y + 1)
    premiums_collected += premium_with_interest

reserve_retro = premiums_collected  # No claims for illustration

prospective_year = reserve_data[year_check - 1]['reserve']

print(f"Premiums collected (with interest): ${premiums_collected:,.2f}")
print(f"Claims paid (none): $0.00")
print(f"Retrospective reserve: ${reserve_retro:,.2f}")
print()
print(f"Prospective reserve (year {year_check}): ${prospective_year:,.2f}")
print(f"Difference: ${abs(reserve_retro - prospective_year):,.2f} (verification)")
print()

# 4. WHOLE LIFE RESERVE COMPARISON
print("=" * 80)
print("WHOLE LIFE vs TERM: RESERVE COMPARISON")
print("=" * 80)

# Calculate whole life net premium
wl_apv_b = 0
kpx_wl = 1.0

for k in range(1, 121 - start_age):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        qx_k = 0.001
    
    vk = v ** k
    wl_apv_b += kpx_wl * qx_k * vk * benefit
    
    if vk < 1e-10:
        break
    
    px_k = 1 - qx_k
    kpx_wl *= px_k

wl_apv_p = 0
kpx_ann = 1.0

for k in range(0, 121 - start_age):
    vk = v ** k
    wl_apv_p += kpx_ann * vk
    
    if k < 120 - start_age:
        age_k = start_age + k
        
        if age_k in mortality_data:
            qx_k = mortality_data[age_k]
        else:
            qx_k = 0.001
        
        px_k = 1 - qx_k
        kpx_ann *= px_k
    
    if kpx_ann < 1e-10:
        break

wl_net_prem = wl_apv_b / wl_apv_p

print(f"\nWhole Life Net Premium (Age 40): ${wl_net_prem:,.2f}/year")
print(f"20-Yr Term Net Premium (Age 40): ${net_premium:,.2f}/year\n")

print(f"{'Year':<8} {'Age':<8} {'Term Reserve':<20} {'WL Reserve':<20} {'WL % of Benefit':<20}")
print("-" * 76)

wl_reserves = []

for year in range(1, 40):
    age_now = start_age + year - 1
    
    # Term reserve (only if within term)
    if year <= original_term:
        remaining = original_term - year + 1
        apv_b_term = calculate_apv_benefits(age_now, remaining, mortality_data, interest_rate, benefit)
        apv_p_term = calculate_apv_annuity(age_now, remaining, mortality_data, interest_rate)
        reserve_term = apv_b_term - (net_premium * apv_p_term)
    else:
        reserve_term = 0
    
    # Whole life reserve
    remaining_wl = min(60 - year, 121 - age_now)
    apv_b_wl = calculate_apv_benefits(age_now, remaining_wl, mortality_data, interest_rate, benefit)
    apv_p_wl = calculate_apv_annuity(age_now, remaining_wl, mortality_data, interest_rate)
    reserve_wl = apv_b_wl - (wl_net_prem * apv_p_wl)
    
    wl_pct = (reserve_wl / benefit) * 100
    wl_reserves.append(reserve_wl)
    
    if year in [1, 5, 10, 15, 20, 25, 30]:
        print(f"{year:<8} {age_now:<8} ${reserve_term:<19,.2f} ${reserve_wl:<19,.2f} {wl_pct:<19.1f}%")

print()

# 5. CASH SURRENDER VALUE
print("=" * 80)
print("CASH SURRENDER VALUE: RESERVE - SURRENDER CHARGE")
print("=" * 80)

surrender_charge_rate = 0.07  # 7% of reserve

print(f"\nSurrender Charge: {surrender_charge_rate*100:.0f}% of Reserve\n")

print(f"{'Year':<8} {'Age':<8} {'Reserve':<15} {'CSV':<15} {'Charge':<15} {'CSV %':<12}")
print("-" * 70)

for year in range(1, min(21, original_term + 1)):
    reserve = reserve_data[year - 1]['reserve']
    charge = reserve * surrender_charge_rate
    csv = reserve - charge
    csv_pct = (csv / benefit) * 100
    age_now = start_age + year - 1
    
    print(f"{year:<8} {age_now:<8} ${reserve:<14,.2f} ${csv:<14,.2f} ${charge:<14,.2f} {csv_pct:<11.1f}%")

print()

# 6. SENSITIVITY: ASSUMPTION CHANGES
print("=" * 80)
print("SENSITIVITY: RESERVE TO ASSUMPTION CHANGES")
print("=" * 80)

year_test = 10
age_test = start_age + year_test - 1
remaining_test = original_term - year_test + 1

# Base case
apv_b_base = calculate_apv_benefits(age_test, remaining_test, mortality_data, interest_rate, benefit)
apv_p_base = calculate_apv_annuity(age_test, remaining_test, mortality_data, interest_rate)
reserve_base = apv_b_base - (net_premium * apv_p_base)

print(f"\nYear {year_test} Reserve Analysis (Age {age_test})\n")
print(f"Base Case Reserve: ${reserve_base:,.2f}\n")

# Mortality shock
mortality_shocked = {age: qx * 1.20 for age, qx in mortality_data.items()}
apv_b_shock = calculate_apv_benefits(age_test, remaining_test, mortality_shocked, interest_rate, benefit)
apv_p_shock = calculate_apv_annuity(age_test, remaining_test, mortality_shocked, interest_rate)
reserve_mort_shock = apv_b_shock - (net_premium * apv_p_shock)

# Interest rate change
v_lower = 1 / (1.03)
apv_b_int = 0
kpx_i = 1.0

for k in range(1, remaining_test + 1):
    age_k = age_test + k - 1
    qx_k = mortality_data.get(age_k, 0.001)
    vk = v_lower ** k
    
    apv_b_int += kpx_i * qx_k * vk * benefit
    px_k = 1 - qx_k
    kpx_i *= px_k

apv_p_int = 0
kpx_p = 1.0

for k in range(0, remaining_test):
    vk = v_lower ** k
    apv_p_int += kpx_p * vk
    
    if k < remaining_test - 1:
        age_k = age_test + k
        qx_k = mortality_data.get(age_k, 0.001)
        px_k = 1 - qx_k
        kpx_p *= px_k

reserve_int_shock = apv_b_int - (net_premium * apv_p_int)

print(f"{'Scenario':<35} {'Reserve':<20} {'Change':<15} {'% Change':<12}")
print("-" * 82)
print(f"{'Base (4%, Standard mortality)':<35} ${reserve_base:>18,.2f}")
print(f"{'Mortality +20%':<35} ${reserve_mort_shock:>18,.2f} {reserve_mort_shock - reserve_base:>+13,.2f} {((reserve_mort_shock / reserve_base) - 1)*100:>+10.1f}%")
print(f"{'Interest rate -1% (3%)':<35} ${reserve_int_shock:>18,.2f} {reserve_int_shock - reserve_base:>+13,.2f} {((reserve_int_shock / reserve_base) - 1)*100:>+10.1f}%")

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Term vs Whole Life reserves
ax = axes[0, 0]
years_plot = list(range(1, 31))
term_reserves = [reserve_data[min(y-1, len(reserve_data)-1)]['reserve'] if y <= original_term else 0 for y in years_plot]
wl_reserves_plot = wl_reserves[:len(years_plot)]

ax.plot(years_plot, term_reserves, linewidth=2.5, marker='o', markersize=5, 
       label='20-Yr Term', color='blue')
ax.plot(years_plot, wl_reserves_plot, linewidth=2.5, marker='s', markersize=5, 
       label='Whole Life', color='green')

ax.axvline(x=20, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Term End')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Reserve ($)', fontsize=11)
ax.set_title('Reserve Growth: Term vs Whole Life', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: CSV analysis
ax = axes[0, 1]
csv_years = []
csv_amounts = []
charges = []

for year in range(1, 21):
    reserve = reserve_data[year - 1]['reserve']
    charge = reserve * surrender_charge_rate
    csv = reserve - charge
    
    csv_years.append(year)
    csv_amounts.append(csv)
    charges.append(charge)

ax.fill_between(csv_years, 0, csv_amounts, alpha=0.5, label='Cash Surrender Value', color='green')
ax.fill_between(csv_years, csv_amounts, np.array(csv_amounts) + np.array(charges), 
               alpha=0.5, label='Surrender Charge', color='red')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Cash Surrender Value Buildup', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 3: Sensitivity tornado
ax = axes[1, 0]
scenarios = ['Base', 'Mortality +20%', 'Interest -1%']
reserves_sens = [reserve_base, reserve_mort_shock, reserve_int_shock]
pct_changes = [0, ((reserve_mort_shock / reserve_base) - 1)*100, ((reserve_int_shock / reserve_base) - 1)*100]

colors_sens = ['green' if x >= 0 else 'red' for x in pct_changes]

ax.barh(scenarios, pct_changes, color=colors_sens, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.set_xlabel('% Change in Reserve', fontsize=11)
ax.set_title(f'Sensitivity: Year {year_test} Reserve (Age {age_test})', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Plot 4: Reserve as % of benefit
ax = axes[1, 1]
years_pct = list(range(1, 21))
reserves_pct = [(reserve_data[y-1]['reserve'] / benefit) * 100 for y in years_pct]

ax.plot(years_pct, reserves_pct, linewidth=2.5, marker='o', markersize=6, 
       color='steelblue', markerfacecolor='lightblue', markeredgewidth=1.5)
ax.fill_between(years_pct, 0, reserves_pct, alpha=0.3, color='steelblue')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Reserve as % of Benefit', fontsize=11)
ax.set_title('Reserve Adequacy: % of Face Amount', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim([0, max(reserves_pct) * 1.2])

plt.tight_layout()
plt.savefig('premium_reserves.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When premium reserves break down:
- **Reserve inadequacy**: Actual mortality worse than assumption; reserve insufficient to cover claims in later years
- **Interest risk**: Rates fall; reserve must increase (deficiency reserve); company adds capital or reduces dividend
- **Lapse correlation**: Healthier policyholders lapse more; remaining policies sicker; average reserve incorrect
- **Assumption changes**: Regulators mandate new mortality table; all old policies get deficiency reserve charge
- **Surrender pattern**: Early surrender wipes out administrative recovery; loss on small policies
- **Investment mismatch**: Long-term contracts; interest rates volatile; liability duration mismatched to assets

## 7. Key References
- [Bowers et al., Actuarial Mathematics (Chapter 5-7)](https://www.soa.org/) - Reserve formulas and calculations
- [SOA Exam LTAM Reserve Analysis](https://www.soa.org/education/exam-req/edu-exam-ltam-detail.aspx) - Practice exam
- [ASOP 22: Statements of Actuarial Opinion](https://www.soa.org/standards/) - Adequacy of reserves

---
**Status:** Liability quantification | **Complements:** Net Premium, Gross Premium, Profit Testing, Renewal Expenses
