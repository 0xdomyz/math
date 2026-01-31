# Reserve Adequacy

## 1. Concept Skeleton
**Definition:** Testing whether reserves held are sufficient to cover future obligations under realistic assumptions; V(t) ≥ Future liabilities  
**Purpose:** Regulatory compliance (solvency); policyholder protection; management confidence in liability estimates  
**Prerequisites:** Reserve formula (prospective/retrospective), assumption basis (conservative vs realistic), cash flow projections

## 2. Comparative Framing
| Approach | Basis | Purpose | Assumption | Regulatory |
|----------|-------|---------|-----------|-----------|
| **Statutory** | Net premium reserve | Minimum safety floor | Conservative mortality, low interest | Required (US) |
| **Deficiency** | If gross < net | Floors reserve at net | Same as statutory | Required |
| **GAAP/IFRS** | Gross premium reserve | Balance sheet liability | Best estimate | Required (public companies) |
| **Solvency II** | Best estimate + margin | Risk capital requirement | Best estimate + 5-10% | EU requirement |
| **Economic** | Match-funding | Duration-aligned investing | Realistic returns | Internal management |

## 3. Examples + Counterexamples

**Simple Example:**  
20-year term, year 5: Net reserve = $1,200/policy, Gross reserve = $1,500/policy; Both adequate if claims = $1,200-1,500

**Failure Case:**  
Assumption: 4% interest; actual 2%; reserves insufficient; must add deficiency reserve at scale; company adds $5M capital

**Edge Case:**  
Whole life age 95: Reserve = ~$200K (nearly full benefit); tiny margin for error; one unexpected claim → loss on pool

## 4. Layer Breakdown
```
Reserve Adequacy Structure:
├─ Testing Framework:
│   ├─ Step 1: Calculate reserves under statutory basis
│   │   ├─ Mortality: Conservative (higher death rates)
│   │   ├─ Interest: Below current market (lower returns)
│   │   ├─ Expenses: Omitted (separate reserve if needed)
│   │   └─ Output: Statutory reserve minimum
│   ├─ Step 2: Calculate reserves under realistic basis
│   │   ├─ Mortality: Best estimate (actual experience)
│   │   ├─ Interest: Market consistent
│   │   ├─ Expenses: Included if relevant
│   │   └─ Output: Economic reserve estimate
│   ├─ Step 3: Calculate reserves under pessimistic basis
│   │   ├─ Mortality: Worse than actual (stress +20%)
│   │   ├─ Interest: Below market (stress -1%)
│   │   └─ Output: Worst-case reserve need
│   ├─ Step 4: Compare & validate
│   │   ├─ Held reserves ≥ Statutory minimum (always required)
│   │   ├─ Held reserves ≥ Economic estimate (best practice)
│   │   ├─ Held reserves ≥ Pessimistic scenario (stress test)
│   │   └─ Gap analysis: Identify shortfall source
│   └─ Step 5: Take corrective action if needed
│       ├─ Add capital (equity injection)
│       ├─ Reduce dividends/distributions
│       ├─ Improve risk assumptions
│       ├─ Reinsure portion of liability
│       └─ Close to new business
├─ Reserve Sufficiency Metrics:
│   ├─ Ratio tests:
│   │   ├─ Reserve ratio: Held reserves / Statutory minimum
│   │   │   ├─ Ratio < 1.0: Inadequate (red flag)
│   │   │   ├─ Ratio = 1.0-1.1: Statutory minimum (bare)
│   │   │   ├─ Ratio = 1.1-1.3: Comfortable margin
│   │   │   └─ Ratio > 1.3: Conservative/excess
│   │   ├─ Benefit adequacy: Reserve / Face amount
│   │   │   ├─ Term insurance: Typically 2-10% of benefit
│   │   │   ├─ Whole life: 10-80% of benefit (increases with age)
│   │   │   └─ Annuity: 90-100%+ of benefit PV
│   │   └─ Durational: Reserve by policy duration
│   │       ├─ Early years: Low reserve (high persistency hope)
│   │       ├─ Mid-term: Rising reserve (higher certainty)
│   │       └─ Late term: High reserve (certainty near 100%)
│   ├─ Sensitivity metrics:
│   │   ├─ Interest sensitivity: 1% rate change → % reserve change
│   │   ├─ Mortality sensitivity: 10% worse → % reserve change
│   │   ├─ Lapse sensitivity: 2% change → % reserve impact
│   │   └─ Combined: All assumptions simultaneously shocked
│   └─ Scenario testing:
│       ├─ Base case: Expected outcome
│       ├─ Upside: Better than expected (mortality good, interest high)
│       ├─ Downside: Worse than expected (mortality bad, interest low)
│       ├─ Tail risk: Extreme scenarios (mortality +50%, interest -2%)
│       └─ Reverse stress: At what assumption change does adequacy break?
├─ Sources of Reserve Inadequacy:
│   ├─ Assumption errors:
│   │   ├─ Mortality: Assumed 10% but actual 12% (or vice versa)
│   │   ├─ Interest: Assumed 4% but earned 2.5%
│   │   ├─ Lapses: Assumed 5% but actual 8% (or recovery if lower)
│   │   ├─ Expenses: Assumed $30/year but actual $50+
│   │   └─ Frequency: Small errors compound over 20-40 year duration
│   ├─ Experience variance (randomness):
│   │   ├─ Expected: 10,000 policies × 1% mortality = 100 deaths
│   │   ├─ Actual: 2 standard deviations = 85-115 deaths
│   │   ├─ Range: Could be 7-14% above/below expectation
│   │   └─ Impact: Bigger for smaller cohorts, less for large books
│   ├─ Model error:
│   │   ├─ Calculation mistakes in reserve formulas
│   │   ├─ System errors (implementation bugs)
│   │   ├─ Assumption mismatches (inconsistent across products)
│   │   └─ Data quality (mortality rates wrong, incomplete exposure)
│   ├─ Structural changes:
│   │   ├─ Market stress: Equity/bond market crash → asset underperformance
│   │   ├─ Interest rate regime shift: Deflation → earned rates low permanently
│   │   ├─ Regulatory change: New mortality table → overnight adequacy loss
│   │   └─ Demographic: Improved mortality (scientific advances) → insurer loses
│   └─ Anti-selection (adverse):
│       ├─ Sicker applicants disproportionately buy insurance
│       ├─ Impact: Actual mortality > population table, especially early years
│       ├─ Mitigation: Medical underwriting, risk rating
│       └─ Testing: Compare actual to underwritten expectation, not general population
├─ Regulatory Requirements (US):
│   ├─ Model Office Method (deprecated, historical):
│   │   ├─ Projected cash flows 30 years forward
│   │   ├─ Test under statutory basis
│   │   └─ Reserve = Maximum liability in any future year
│   ├─ Net Premium Reserve (current standard):
│   │   ├─ Reserve = PV(Future benefits) - PV(Future net premiums)
│   │   ├─ Basis: Conservative mortality + low interest
│   │   ├─ Formula simplicity: Easy to calculate, audit
│   │   └─ Limitation: May understate economic liability
│   ├─ Gross Premium Reserve (GAAP):
│   │   ├─ Reserve = PV(Future benefits + expenses) - PV(Future gross premiums)
│   │   ├─ Basis: Best estimate assumptions
│   │   ├─ More realistic than net premium
│   │   └─ Used for GAAP balance sheet reporting
│   ├─ Deficiency Reserve:
│   │   ├─ Triggers: Gross premium reserve < Statutory (net premium) reserve
│   │   ├─ Action: Increase statutory reserve to fill gap
│   │   ├─ Example: Net = $1,200, Gross = $800 → Must hold $1,200
│   │   └─ Frequency: ~10-20% of products may trigger deficiency
│   ├─ Adequacy Testing (ASOP 22):
│   │   ├─ Annual actuarial opinion on reserve adequacy
│   │   ├─ Compare held reserves to best estimate + margin
│   │   ├─ Adjust if shortfall identified
│   │   └─ Sign-off: Qualified actuary statement
│   └─ Capital requirement (risk-based):
│       ├─ C1: Mortality risk (death rate variance)
│       ├─ C2: Interest rate risk (reinvestment mismatch)
│       ├─ C3: Lapse risk (mass lapse exposure)
│       ├─ C4: Expense risk (inflation/efficiency)
│       ├─ Total: C1 + C2 + C3 + C4 = Risk-based capital needed
│       └─ Ratio: Risk-based capital / Required minimum ≥ 100% (healthy)
├─ Solvency II (EU):
│   ├─ Technical provisions:
│   │   ├─ Best estimate: Realistic probability-weighted cash flows
│   │   ├─ Risk margin: Additional provision for uncertainty
│   │   ├─ Total = Best estimate + Risk margin
│   │   └─ Risk margin typically 5-10% of best estimate
│   ├─ Match funding:
│   │   ├─ Asset duration ≈ Liability duration
│   │   ├─ Objective: Minimize reinvestment risk
│   │   ├─ Impact: Technical provisions decrease with better asset match
│   │   └─ Benefit: Reduces capital requirements if well-matched
│   └─ Own Risk & Solvency Assessment (ORSA):
│       ├─ Internal assessment of capital needs
│       ├─ Stress testing required
│       ├─ Scenario analysis (base, upside, downside, reverse)
│       └─ Board/Management reporting annually
└─ Management Actions if Inadequate:
    ├─ Capital management:
    │   ├─ Equity injection: Shareholders add capital
    │   ├─ Retain earnings: Suspend dividends
    │   ├─ Reinsurance: Cede portion of liabilities
    │   └─ Securitization: Convert risks to capital markets
    ├─ Business management:
    │   ├─ Close to new business: No new policy sales
    │   ├─ Reduce in-force exposure: Accelerated run-off
    │   ├─ Improve underwriting: Tighter risk selection
    │   └─ Rate increases: Premium hikes on renewals
    ├─ Financial management:
    │   ├─ Asset optimization: Improve portfolio returns
    │   ├─ Liability management: Modify benefits (e.g., reduce rider)
    │   └─ Expense reduction: Cut costs to improve margin
    └─ Regulatory/Governance:
        ├─ Regulatory filing: Disclose shortfall
        ├─ Corrective action plan: Timeline for remediation
        ├─ Board oversight: Regular monitoring
        └─ Audit confirmation: External actuary signs opinion
```

**Key Insight:** Adequacy not one-time test; continuous monitoring needed because assumptions change and randomness creates variance

## 5. Mini-Project
Test reserve adequacy across scenarios and identify shortfall drivers:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP
print("=" * 80)
print("RESERVE ADEQUACY TESTING: STATUTORY VS ECONOMIC")
print("=" * 80)

# Mortality and assumptions
mortality_standard = {
    40: 0.00131, 45: 0.00172, 50: 0.00233, 55: 0.00325,
    60: 0.00459, 65: 0.00653, 70: 0.00933
}

# Assumptions
issue_age = 40
contract_term = 20
benefit_amount = 200000
policies_inforce = 10000

# Interest rate assumptions
interest_statutory = 0.03   # Conservative 3%
interest_economic = 0.04    # Realistic 4%
interest_actual = 0.025     # Actual earned (stressed)

print(f"\nAssumptions:")
print(f"  Product: 20-Year Term")
print(f"  Issue Age: {issue_age}")
print(f"  Policies: {policies_inforce:,}")
print(f"  Benefit: ${benefit_amount:,.0f}\n")

def calculate_reserve_prospective(age, remaining_term, mortality_dict, benefit_amt, 
                                 annual_premium, interest_rate):
    """Calculate prospective reserve"""
    
    v_rate = 1 / (1 + interest_rate)
    reserve = 0
    kpx = 1.0
    
    # APV of benefits
    apv_benefits = 0
    for k in range(1, remaining_term + 1):
        age_k = age + k - 1
        qx_k = mortality_dict.get(age_k, 0.001)
        vk = v_rate ** k
        
        apv_benefits += kpx * qx_k * vk * benefit_amt
        px_k = 1 - qx_k
        kpx *= px_k
    
    # APV of premiums
    apv_premiums = 0
    kpx = 1.0
    for k in range(0, remaining_term):
        vk = v_rate ** k
        apv_premiums += kpx * vk
        
        if k < remaining_term - 1:
            age_k = age + k
            qx_k = mortality_dict.get(age_k, 0.001)
            px_k = 1 - qx_k
            kpx *= px_k
    
    reserve = apv_benefits - (annual_premium * apv_premiums)
    
    return max(0, reserve)

# Calculate net premium (for statutory reserve)
net_premium = 340  # Assume from earlier calculation

# 2. ADEQUACY TEST AT DIFFERENT DURATIONS
print("=" * 80)
print("RESERVE ADEQUACY BY POLICY DURATION")
print("=" * 80)

durations = [1, 5, 10, 15, 20]
gross_premium = 450

print(f"\n{'Year':<8} {'Age':<8} {'Stat. Res':<15} {'Econ. Res':<15} {'Held':<15} {'Shortfall':<15} {'% Gap':<12}")
print("-" * 95)

adequacy_data = []

for duration in durations:
    age_now = issue_age + duration - 1
    remaining_term = contract_term - duration + 1
    
    # Statutory reserve (conservative)
    stat_reserve = calculate_reserve_prospective(age_now, remaining_term, mortality_standard, 
                                                benefit_amount, net_premium, interest_statutory)
    
    # Economic reserve (realistic)
    econ_reserve = calculate_reserve_prospective(age_now, remaining_term, mortality_standard, 
                                                benefit_amount, gross_premium, interest_economic)
    
    # Held reserve (assume at statutory minimum)
    held_reserve = stat_reserve
    
    # Shortfall analysis
    shortfall = econ_reserve - held_reserve
    shortfall_pct = (shortfall / econ_reserve * 100) if econ_reserve > 0 else 0
    
    adequacy_data.append({
        'duration': duration,
        'age': age_now,
        'stat_res': stat_reserve,
        'econ_res': econ_reserve,
        'held': held_reserve,
        'shortfall': shortfall
    })
    
    print(f"{duration:<8} {age_now:<8} ${stat_reserve:<14,.0f} ${econ_reserve:<14,.0f} ${held_reserve:<14,.0f} ${shortfall:>+13,.0f} {shortfall_pct:>+10.1f}%")

print()

# 3. SENSITIVITY: INTEREST RATE SHOCK
print("=" * 80)
print("SENSITIVITY: INTEREST RATE IMPACT ON RESERVE ADEQUACY")
print("=" * 80)

year_test = 10
age_test = issue_age + year_test - 1
remaining_test = contract_term - year_test + 1

print(f"\nYear {year_test} Analysis (Age {age_test}):\n")

interest_rates = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

print(f"{'Interest Rate':<18} {'Economic Res':<18} {'vs 4% Base':<18} {'Shortfall':<18}")
print("-" * 72)

base_econ_res = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                             benefit_amount, gross_premium, 0.04)

for rate in interest_rates:
    econ_res = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                            benefit_amount, gross_premium, rate)
    
    pct_vs_base = ((econ_res / base_econ_res) - 1) * 100
    shortfall = econ_res - (stat_reserve)
    
    print(f"{rate*100:>16.2f}% ${econ_res:>16,.0f} {pct_vs_base:>+16.1f}% ${shortfall:>+16,.0f}")

print()

# 4. MORTALITY SHOCK SCENARIO
print("=" * 80)
print("MORTALITY SHOCK: +20% WORSE THAN ASSUMPTION")
print("=" * 80)

mortality_shocked = {age: qx * 1.20 for age, qx in mortality_standard.items()}

print(f"\nYear {year_test} Under Mortality Stress:\n")

econ_res_base = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                             benefit_amount, gross_premium, interest_economic)

econ_res_stressed = calculate_reserve_prospective(age_test, remaining_test, mortality_shocked,
                                                 benefit_amount, gross_premium, interest_economic)

print(f"{'Scenario':<30} {'Reserve':<18} {'vs Base':<18}")
print("-" * 66)
print(f"{'Base case (standard mort)':<30} ${econ_res_base:>16,.0f}")
print(f"{'Mortality +20%':<30} ${econ_res_stressed:>16,.0f} {((econ_res_stressed/econ_res_base)-1)*100:>+16.1f}%")

print()

# 5. COMPOUND STRESS TEST
print("=" * 80)
print("COMPOUND STRESS: INTEREST -1%, MORTALITY +20%")
print("=" * 80)

interest_stressed = interest_economic - 0.01

econ_res_compound = calculate_reserve_prospective(age_test, remaining_test, mortality_shocked,
                                                 benefit_amount, gross_premium, interest_stressed)

print(f"\n{'Scenario':<40} {'Reserve':<18}")
print("-" * 58)
print(f"{'Base (4% int, std mort)':<40} ${econ_res_base:>16,.0f}")
print(f"{'Stressed (3% int, +20% mort)':<40} ${econ_res_compound:>16,.0f}")
print()
print(f"Reserve increase needed: ${econ_res_compound - econ_res_base:,.0f}")
print(f"Per-policy: ${(econ_res_compound - econ_res_base):.0f}")
print(f"Total for cohort: ${(econ_res_compound - econ_res_base) * policies_inforce:,.0f}")

print()

# 6. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reserve by duration
ax = axes[0, 0]
durations_plot = [d['duration'] for d in adequacy_data]
stat_reserves = [d['stat_res'] for d in adequacy_data]
econ_reserves = [d['econ_res'] for d in adequacy_data]

ax.plot(durations_plot, stat_reserves, linewidth=2.5, marker='o', markersize=6, 
       label='Statutory Reserve', color='blue')
ax.plot(durations_plot, econ_reserves, linewidth=2.5, marker='s', markersize=6, 
       label='Economic Reserve', color='green')
ax.fill_between(durations_plot, stat_reserves, econ_reserves, alpha=0.2, color='red', label='Potential Shortfall')

ax.set_xlabel('Policy Year', fontsize=11)
ax.set_ylabel('Reserve per Policy ($)', fontsize=11)
ax.set_title('Statutory vs Economic Reserves Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Interest rate sensitivity
ax = axes[0, 1]
rates_plot = np.array([r * 100 for r in interest_rates])
reserves_by_rate = []

for rate in interest_rates:
    res = calculate_reserve_prospective(age_test, remaining_test, mortality_standard,
                                       benefit_amount, gross_premium, rate)
    reserves_by_rate.append(res)

ax.plot(rates_plot, reserves_by_rate, linewidth=2.5, marker='o', markersize=6, color='steelblue')
ax.axvline(x=4.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Base (4%)')
ax.axvline(x=2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Stressed (2.5%)')

ax.set_xlabel('Interest Rate (%)', fontsize=11)
ax.set_ylabel('Economic Reserve ($)', fontsize=11)
ax.set_title(f'Reserve Sensitivity to Interest Rates (Year {year_test})', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Shortfall by duration
ax = axes[1, 0]
shortfalls = [d['shortfall'] for d in adequacy_data]
shortfall_pcts = [(d['shortfall'] / d['econ_res'] * 100) if d['econ_res'] > 0 else 0 
                 for d in adequacy_data]

colors_sf = ['red' if sf > 0 else 'green' for sf in shortfalls]

ax.bar(durations_plot, shortfalls, color=colors_sf, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Policy Year', fontsize=11)
ax.set_ylabel('Reserve Shortfall ($)', fontsize=11)
ax.set_title('Reserve Adequacy Gap: Economic - Held', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Stress scenario comparison
ax = axes[1, 1]
scenarios_stress = ['Base\n(4%, Std)', 'Interest\nStress\n(3%)', 'Mortality\nStress\n(+20%)', 'Combined\n(-1%, +20%)']
reserves_stress = [
    econ_res_base,
    calculate_reserve_prospective(age_test, remaining_test, mortality_standard, benefit_amount, gross_premium, 0.03),
    econ_res_stressed,
    econ_res_compound
]

bars = ax.bar(scenarios_stress, reserves_stress, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.axhline(y=econ_res_base, color='green', linestyle='--', linewidth=1.5, label='Base')

for bar, res in zip(bars, reserves_stress):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${res:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Reserve per Policy ($)', fontsize=11)
ax.set_title(f'Stress Testing Results (Year {year_test}, Age {age_test})', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reserve_adequacy.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When reserve adequacy assumptions fail:
- **Persistent interest rate decline**: Model 4% returns; actual 2% for 5 years; cumulative shortfall 15-25% on reserves
- **Mortality table update**: Regulator mandates new mortality table (40% higher); overnight all reserves need increase; capital injection required
- **Anti-selection discovery**: Actual mortality 30% worse than assumed in first 5 years; immediate deficiency reserve
- **Economic capital stress**: Market downturn → asset values decline; even if liabilities unchanged, solvency ratio breached
- **Model recalibration**: New experience study shows assumptions systematically wrong; reserves retroactively inadequate; accounting adjustment
- **Regulatory change**: New risk-based capital formula; suddenly required capital doubles; forced business exit or raise equity

## 7. Key References
- [ASOP 22: Statements of Actuarial Opinion (Reserves)](https://www.soa.org/standards/) - Adequacy testing standards
- [Bowers et al., Actuarial Mathematics (Chapter 5-8)](https://www.soa.org/) - Reserve calculations
- [Solvency II Technical Provisions (EIOPA Guidelines)](https://www.eiopa.europa.eu/) - EU framework

---
**Status:** Solvency & risk management | **Complements:** Premium Reserves, Statutory Bases, Capital Requirements
