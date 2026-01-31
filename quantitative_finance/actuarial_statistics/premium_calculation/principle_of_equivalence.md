# Principle of Equivalence

## 1. Concept Skeleton
**Definition:** E[Benefits] = E[Premiums]; actuarial equivalence principle establishing zero-profit pricing; fundamental equation for net premium  
**Purpose:** Core pricing axiom; ensures fair risk exchange between insurer and insured; basis for regulatory reserve minimums  
**Prerequisites:** Present value of benefits (Aₓ), annuity-due factor (äₓ), probability weighting (ₚₓ, qₓ)

## 2. Comparative Framing
| Principle | Equation | Profit | Assumption | Use Case |
|-----------|----------|--------|-----------|----------|
| **Equivalence** | E[B] = E[P] | 0% | Perfect estimation | Theoretical fair price |
| **Equivalence + Margin** | E[B] + Margin = E[P] | Yes | Conservative | Regulatory minimum |
| **Conservative** | Higher E[B] assumption | Higher load | Pessimistic | Statutory reserve |
| **Optimistic** | Lower E[B] assumption | Lower load | Best estimate | GAAP/IFRS valuation |
| **Market** | E[P] = competitor rates | Variable | Competitive | Actual market prices |

## 3. Examples + Counterexamples

**Simple Example:**  
Term insurance: PV(expected claims) = $8,200, PV(expected premiums at rate $450/yr) = $8,200 → Equivalence achieved

**Failure Case:**  
Premium set at $400/yr: PV(premiums) = $7,300 < $8,200 PV(claims); Company loses money on every policy

**Edge Case:**  
Whole life at age 85: Equivalence calculation highly sensitive to mortality assumptions (small % change → large premium impact)

## 4. Layer Breakdown
```
Principle of Equivalence Structure:
├─ Mathematical Foundation:
│   ├─ Equation: ∑ P · v^k · ₖpₓ = ∑ B · v^k · ₖ₋₁pₓ · qₓ₊ₖ₋₁
│   ├─ Rearrangement: P = (APV of Benefits) / (APV of Premiums)
│   ├─ Annuity notation: äₓ:n̄| = ∑ v^k · ₖpₓ  (premium payment annuity)
│   ├─ Benefit notation: Aₓ:n̄| = ∑ v^k · ₖ₋₁pₓ · qₓ₊ₖ₋₁  (benefit APV)
│   └─ Net premium: Pₙ = Aₓ:n̄| / äₓ:n̄|
├─ Key Assumptions:
│   ├─ Mortality probability: ₖpₓ, qₓ (from life table)
│   ├─ Interest rate: Discount all future values at rate i
│   ├─ Timing: Benefits at end of year of death; premiums at start of year
│   ├─ Determinism: Probabilities treated as certain (ignores randomness)
│   └─ Stationarity: Assumptions constant throughout contract (no inflation/changes)
├─ Variations & Extensions:
│   ├─ Multiple benefits: B varies by time; principle applies separately to each
│   ├─ Multiple premiums: P varies; e.g., lower initially, higher after year n
│   ├─ Continuous payment: Principle extends to integrals (force of mortality)
│   ├─ Multiple decrements: Compete for each life (death, lapse, recovery)
│   └─ Expense-loaded: Equivalence P* = P + Expense/PV(Annuity)
├─ Practical Application:
│   ├─ Step 1: Calculate APV(Benefits) using assumed mortality & interest
│   ├─ Step 2: Calculate APV(Premiums - generic annuity)
│   ├─ Step 3: Solve P = APV(B) / APV(P)
│   ├─ Step 4: Verify equality: E[Premiums] = E[Benefits]
│   └─ Step 5: Iterate if assumptions change
├─ Regulatory Treatment:
│   ├─ Statutory minimum: Net premium reserve = statutory net premium × benefit count
│   ├─ Reserve adequacy: Gross premium reserve ≥ net premium reserve
│   ├─ Deficiency reserve: If gross < net premium, additional reserve required
│   ├─ Solvency II: Equivalence principle + risk margin
│   └─ IFRS 17: Equivalence basis for contract service margin
├─ Limitations & Caveats:
│   ├─ Ignores randomness: Assumes expected values realized; actual claims stochastic
│   ├─ Ignores expense uncertainty: Expense loading separate and uncertain
│   ├─ Ignores lapses: Assumes full persistency; actual lapses vary by policy/market
│   ├─ Assumption error risk: If mortality/interest assumptions wrong → reserve inadequate
│   └─ Antiselection: Premium based on average; sicker applicants subsidized by healthier
└─ Balance Sheet Impact:
    ├─ Asset side: Premium income flows in
    ├─ Liability side: Reserve grows (future obligations)
    ├─ Profit: Difference between actual and assumed (mortality variance, lapse gains)
    ├─ Surplus margin: Gross premium - Net premium accumulates as benefit reserve grows
    └─ Release: Reserve released at contract end (death or surrender)
```

**Key Insight:** Equivalence ensures actuarial fairness at issue; profit emerges from actual experience better than assumptions

## 5. Mini-Project
Demonstrate equivalence principle, test assumption sensitivity, and compare to market prices:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP
print("=" * 80)
print("PRINCIPLE OF EQUIVALENCE: ACTUARIAL FAIRNESS & NET PREMIUM")
print("=" * 80)

# Mortality table
mortality_data = {
    25: 0.00067, 30: 0.00084, 35: 0.00103, 40: 0.00131, 45: 0.00172,
    50: 0.00233, 55: 0.00325, 60: 0.00459, 65: 0.00653, 70: 0.00933,
    75: 0.01330, 80: 0.01934, 85: 0.02873, 90: 0.04213
}

interest_rate = 0.04
v = 1 / (1 + interest_rate)
benefit = 200000

print(f"\nAssumptions:")
print(f"  Benefit: ${benefit:,.0f}")
print(f"  Interest: {interest_rate*100:.1f}%")
print(f"  Principle: E[Premiums] = E[Benefits]\n")

# 2. EQUIVALENCE DEMONSTRATION
print("=" * 80)
print("EQUIVALENCE PRINCIPLE: 10-YEAR TERM, AGE 35")
print("=" * 80)

start_age = 35
term_years = 10

# Calculate APV(Benefits)
apv_benefits = 0
kpx = 1.0
print(f"\nCalculating APV of Benefits:\n")
print(f"{'Year':<8} {'Age':<8} {'ₖpₓ':<12} {'qₓ':<12} {'v^k':<12} {'Contribution':<15}")
print("-" * 75)

for k in range(1, term_years + 1):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        qx_k = 0.001  # Placeholder
    
    vk = v ** k
    contribution = kpx * qx_k * vk * benefit
    apv_benefits += contribution
    
    print(f"{k:<8} {age_k:<8} {kpx:<12.6f} {qx_k:<12.6f} {vk:<12.6f} ${contribution:<14,.2f}")
    
    px_k = 1 - qx_k
    kpx *= px_k

print(f"\nAPV(Benefits) = ${apv_benefits:,.2f}\n")

# Calculate APV(Premiums)
apv_premiums = 0
kpx = 1.0
print(f"Calculating APV of Premiums (Annuity Due):\n")
print(f"{'Year':<8} {'ₖpₓ':<12} {'v^k':<12} {'Contribution':<15}")
print("-" * 47)

for k in range(0, term_years):
    vk = v ** k
    contribution = kpx * vk
    apv_premiums += contribution
    
    print(f"{k+1:<8} {kpx:<12.6f} {vk:<12.6f} {contribution:<15.6f}")
    
    if k < term_years - 1:
        age_k = start_age + k
        
        if age_k in mortality_data:
            qx_k = mortality_data[age_k]
        else:
            qx_k = 0.001
        
        px_k = 1 - qx_k
        kpx *= px_k

print(f"\nAPV(Premiums) = {apv_premiums:.6f}\n")

# Net premium from equivalence
net_premium = apv_benefits / apv_premiums

print(f"{'Metric':<35} {'Value':<20}")
print("-" * 55)
print(f"{'APV of Benefits':<35} ${apv_benefits:>18,.2f}")
print(f"{'APV of Premium Annuity':<35} {apv_premiums:>19.6f}")
print(f"{'Net Premium (from equivalence)':<35} ${net_premium:>18,.2f}")
print()

# Verify equivalence
pv_premiums_received = net_premium * apv_premiums
print(f"Verification of Equivalence:\n")
print(f"  If premium = ${net_premium:,.2f}/year for {term_years} years:")
print(f"  PV(Premiums received) = ${net_premium:,.2f} × {apv_premiums:.6f}")
print(f"                        = ${pv_premiums_received:,.2f}")
print()
print(f"  PV(Benefits paid)     = ${apv_benefits:,.2f}")
print()
print(f"  Difference: ${abs(pv_premiums_received - apv_benefits):,.2f}")
print(f"  → Equivalence achieved (E[Premiums] = E[Benefits])")
print()

# 3. CASH FLOW TABLE (EQUIVALENCE PERSPECTIVE)
print("=" * 80)
print("CASH FLOW & RESERVE ANALYSIS: EQUIVALENCE VIEW")
print("=" * 80)

print(f"\nPer-Policy Cash Flows (Assuming {term_years}-year survival to end):\n")
print(f"{'Year':<8} {'Premium':<15} {'Expected Claim':<18} {'Net Cash Flow':<18} {'Acc Reserve':<15}")
print("-" * 80)

accumulated_reserve = 0

for year in range(1, term_years + 1):
    age_y = start_age + year - 1
    
    if age_y in mortality_data:
        qx_y = mortality_data[age_y]
    else:
        qx_y = 0.001
    
    # Premium received at start of year
    premium_in = net_premium
    
    # Expected claim paid at end of year
    expected_claim = qx_y * benefit
    
    # Net cash flow
    net_cf = premium_in - expected_claim
    
    # Accumulated reserve (interest-adjusted)
    accumulated_reserve = accumulated_reserve * (1 + interest_rate) + premium_in - expected_claim
    
    print(f"{year:<8} ${premium_in:<14,.2f} ${expected_claim:<17,.2f} ${net_cf:<17,.2f} ${accumulated_reserve:<14,.2f}")

print()

# 4. ASSUMPTION SENSITIVITY
print("=" * 80)
print("SENSITIVITY: HOW ASSUMPTIONS AFFECT EQUIVALENCE")
print("=" * 80)

# Base case
base_premium = net_premium

print(f"\nBase Assumptions (4% interest, Standard mortality): Premium = ${base_premium:,.2f}\n")

# Mortality shock
print("If Actual Mortality 20% Worse Than Assumed:\n")

apv_benefits_worse = 0
kpx_w = 1.0

for k in range(1, term_years + 1):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k] * 1.20  # 20% worse
    else:
        qx_k = 0.001 * 1.20
    
    vk = v ** k
    apv_benefits_worse += kpx_w * qx_k * vk * benefit
    
    px_k = 1 - qx_k
    kpx_w *= px_k

actual_cost_worse = apv_benefits_worse / apv_premiums
shortfall = actual_cost_worse - base_premium

print(f"  Actual APV(Benefits) = ${apv_benefits_worse:,.2f}")
print(f"  Actual cost per policy = ${actual_cost_worse:,.2f}")
print(f"  Charged premium = ${base_premium:,.2f}")
print(f"  Shortfall = ${shortfall:,.2f} (loss per policy)")
print()

# Interest rate shock
print("If Actual Interest 1% Lower (3% vs 4%):\n")

v_lower = 1 / (1.03)
apv_benefits_int = 0
kpx_i = 1.0

for k in range(1, term_years + 1):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        qx_k = 0.001
    
    vk = v_lower ** k
    apv_benefits_int += kpx_i * qx_k * vk * benefit
    
    px_k = 1 - qx_k
    kpx_i *= px_k

# Premiums also need revaluation at new rate
apv_premiums_lower = 0
kpx_p = 1.0

for k in range(0, term_years):
    vk = v_lower ** k
    apv_premiums_lower += kpx_p * vk
    
    if k < term_years - 1:
        age_k = start_age + k
        
        if age_k in mortality_data:
            qx_k = mortality_data[age_k]
        else:
            qx_k = 0.001
        
        px_k = 1 - qx_k
        kpx_p *= px_k

required_cost_lower = apv_benefits_int / apv_premiums_lower
reserve_increase = required_cost_lower - base_premium

print(f"  APV(Benefits) at 3%: ${apv_benefits_int:,.2f}")
print(f"  APV(Premiums) at 3%: {apv_premiums_lower:.6f}")
print(f"  Required premium at 3% = ${required_cost_lower:,.2f}")
print(f"  Reserve increase needed: ${reserve_increase:,.2f} per policy")
print()

# Combined shock
print("Combined Shock (Mortality +20%, Interest -1%):\n")

apv_benefits_combined = 0
kpx_c = 1.0

for k in range(1, term_years + 1):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k] * 1.20
    else:
        qx_k = 0.001 * 1.20
    
    vk = v_lower ** k
    apv_benefits_combined += kpx_c * qx_k * vk * benefit
    
    px_k = 1 - qx_k
    kpx_c *= px_k

required_cost_combined = apv_benefits_combined / apv_premiums_lower
total_shortfall = required_cost_combined - base_premium

print(f"  Required premium = ${required_cost_combined:,.2f}")
print(f"  Charged premium = ${base_premium:,.2f}")
print(f"  Total shortfall = ${total_shortfall:,.2f}")
print()

# 5. MARKET vs EQUIVALENCE PRICES
print("=" * 80)
print("MARKET PRICING vs EQUIVALENCE PRICING")
print("=" * 80)

# Estimate market prices (typical load factors)
gross_premium_no_load = base_premium  # Net premium = base
gross_premium_conservative = base_premium * 1.35  # +35% for expenses/profit
gross_premium_competitive = base_premium * 1.25   # +25% for expenses/profit
gross_premium_aggressive = base_premium * 1.20    # +20% for expenses/profit

print(f"\nMarket Pricing Strategy (10-Year Term, Age 35, $200K):\n")
print(f"{'Strategy':<30} {'Gross Premium':<20} {'Load %':<15} {'Rationale':<20}")
print("-" * 85)
print(f"{'Equivalence (Net only)':<30} ${base_premium:<19,.2f} {'0%':<15} {'Breaks even':<20}")
print(f"{'Conservative (Mutual)':<30} ${gross_premium_conservative:<19,.2f} {'+35%':<15} {'Surplus buildup':<20}")
print(f"{'Competitive (Stock)':<30} ${gross_premium_competitive:<19,.2f} {'+25%':<15} {'Market standard':<20}")
print(f"{'Aggressive (Direct)':<30} ${gross_premium_aggressive:<19,.2f} {'+20%':<15} {'Volume-based':<20}")
print()

# 6. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Equivalence principle illustration
ax = axes[0, 0]
years_plot = np.arange(1, term_years + 1)
expected_premiums = [net_premium] * term_years

# Calculate expected claims per year
expected_claims_annual = []
kpx_plot = 1.0

for k in range(1, term_years + 1):
    age_k = start_age + k - 1
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        qx_k = 0.001
    
    expected_claim_year = qx_k * benefit
    expected_claims_annual.append(expected_claim_year)
    
    px_k = 1 - qx_k
    kpx_plot *= px_k

ax.bar(years_plot - 0.2, expected_premiums, width=0.4, label='Premium Income', 
      alpha=0.6, color='green', edgecolor='black')
ax.bar(years_plot + 0.2, expected_claims_annual, width=0.4, label='Expected Claims', 
      alpha=0.6, color='red', edgecolor='black')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Equivalence: Expected Premium Income vs Expected Claims', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 2: Assumption sensitivity
ax = axes[0, 1]
scenarios_labels = ['Base\n(4%, Std)', 'Mort +20%\n(4%, +20%)', 'Interest\n-1% (3%)', 'Combined\n(+20%, 3%)']
scenario_costs = [base_premium, actual_cost_worse, required_cost_lower, required_cost_combined]
colors_scen = ['green', 'red', 'orange', 'darkred']

bars = ax.bar(scenarios_labels, scenario_costs, color=colors_scen, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.axhline(y=base_premium, color='green', linestyle='--', linewidth=2, label='Base Premium')

ax.set_ylabel('Actual Cost per Policy ($)', fontsize=11)
ax.set_title('Impact of Assumption Changes on Equivalence', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

for bar, cost in zip(bars, scenario_costs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${cost:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Gross premium strategies
ax = axes[1, 0]
strategies = ['Equivalence\n(Net)', 'Conservative\n(+35%)', 'Competitive\n(+25%)', 'Aggressive\n(+20%)']
gross_premiums = [base_premium, gross_premium_conservative, gross_premium_competitive, gross_premium_aggressive]
profit_margins = [0, 
                 gross_premium_conservative - base_premium,
                 gross_premium_competitive - base_premium,
                 gross_premium_aggressive - base_premium]

ax.bar(strategies, gross_premiums, label='Gross Premium', alpha=0.6, color='steelblue', edgecolor='black')
ax.axhline(y=base_premium, color='green', linestyle='--', linewidth=2, label='Equivalence (Net)')

ax.set_ylabel('Annual Premium ($)', fontsize=11)
ax.set_title('Market Pricing Strategies', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Cumulative PV comparison
ax = axes[1, 1]
years_cumul = np.arange(1, term_years + 1)
cumul_premiums = []
cumul_claims = []

cumul_prem = 0
cumul_claim = 0
kpx_c = 1.0

for k in years_cumul:
    v_factor = v ** k
    cumul_prem += net_premium * v_factor
    cumul_premiums.append(cumul_prem)
    
    age_k = start_age + k - 1
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        qx_k = 0.001
    
    cumul_claim += kpx_c * qx_k * v_factor * benefit
    cumul_claims.append(cumul_claim)
    
    px_k = 1 - qx_k
    kpx_c *= px_k

ax.plot(years_cumul, cumul_premiums, linewidth=2.5, marker='o', markersize=5, 
       label='Cumulative PV(Premiums)', color='green')
ax.plot(years_cumul, cumul_claims, linewidth=2.5, marker='s', markersize=5, 
       label='Cumulative PV(Claims)', color='red')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Cumulative PV ($)', fontsize=11)
ax.set_title('Cumulative Equivalence: Premiums vs Claims', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('principle_of_equivalence.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When equivalence principle breaks down:
- **Assumption errors**: If mortality/interest assumptions systematically wrong, policy becomes loss-making or loss-avoiding
- **Adverse selection**: Sicker applicants buy at prices priced for average; equivalence violated at issue
- **Persistent inflation**: Premium set in year 1; expense costs 5% annually; 10 years later, expenses consume margin
- **Lapse correlation**: Economic stress causes both lapse (good, avoids claims) and antiselection (bad, claims from remaining); net effect uncertain
- **Model error**: Continuous equivalence principle assumes discrete payments work analogously; timing mismatches cause errors
- **Regulatory conservatism**: Statutory minimum reserves ≠ market reserves; companies over-reserve even if equivalence holds

## 7. Key References
- [Bowers et al., Actuarial Mathematics (Chapter 2)](https://www.soa.org/) - Equivalence axiom and derivations
- [SOA Exam FM Principle (Chapter 3-4)](https://www.soa.org/education/exam-req/edu-exam-fm-detail.aspx) - Problems and solutions
- [ASOP 35: Pricing of Insurance Products](https://www.soa.org/standards/) - Professional practice guidance

---
**Status:** Foundational pricing principle | **Complements:** Net Premium, Gross Premium, Premium Reserves
