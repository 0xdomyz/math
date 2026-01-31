# Lifetime vs 12-Month ECL

## 1. Concept Skeleton
**Definition:** Stage 1 recognizes 12-month expected credit losses (losses from defaults in next year only); Stage 2/3 recognize lifetime ECL (all expected losses over remaining contractual life); distinction drives provisioning magnitude  
**Purpose:** Progressive loss recognition as credit risk increases; Stage 1 (low provisions) for performing loans; Stage 2/3 (high provisions) for deteriorated/impaired loans; incentivizes early SICR detection  
**Prerequisites:** PD term structure, survival analysis, maturity profiles, discounting (effective interest rate), SICR criteria, Stage classification

## 2. Comparative Framing
| Metric | Stage 1 (12-Month ECL) | Stage 2 (Lifetime ECL) | Stage 3 (Lifetime ECL) |
|--------|------------------------|------------------------|------------------------|
| **ECL Horizon** | Next 12 months only | Full remaining life | Full remaining life |
| **PD Used** | 12-month PD | Lifetime (cumulative) PD | 100% (defaulted) |
| **Typical Coverage** | 0.1%-1% of EAD | 3%-15% of EAD | 30%-90% of EAD |
| **Maturity Sensitivity** | Low (1-year cap) | High (longer = more ECL) | High (recovery period) |
| **Interest Revenue** | Gross carrying amount × EIR | Gross carrying amount × EIR | Net carrying amount × EIR |
| **Transfer Trigger** | None (initial classification) | SICR detected | Default occurred |

## 3. Examples + Counterexamples

**Short Maturity (1-Year Loan):**  
12-month ECL ≈ Lifetime ECL (both cover same period). $100k loan, PD = 2%, LGD = 40%. Stage 1 ECL = Stage 2 ECL = $100k × 2% × 40% = $800.

**Long Maturity (10-Year Corporate Loan):**  
Stage 1: 12-month PD = 1%, ECL = $100k × 1% × 40% = $400. Stage 2: Lifetime PD = 15% (cumulative over 10 years), ECL = $100k × 15% × 40% = $6,000 (15× higher).

**Stage 1→2 Transfer Impact:**  
$10M loan portfolio, average 5-year maturity. Stage 1: 12-month ECL = $50k. SICR triggered → Stage 2: Lifetime ECL = $400k. Provision increase = $350k (7× jump).

**Amortizing Loan (Mortgage):**  
$200k mortgage, 30-year amortization. Outstanding balance declines over time → EAD reduces → Lifetime ECL lower than non-amortizing. Year 1 EAD = $200k; Year 15 EAD = $100k; Year 30 EAD = $0.

**Edge Case (Revolving Credit):**  
Credit card $10k limit; current balance $2k. EAD = $2k + $8k × CCF (credit conversion factor). Lifetime ECL accounts for potential drawdown (CCF = 20-50% typically; stressed = 100%).

## 4. Layer Breakdown
```
Lifetime vs 12-Month ECL Framework:

├─ 12-Month ECL (Stage 1):
│   ├─ Definition:
│   │   └─ IFRS 9.5.5.5: "ECL from default events possible within 12 months after reporting date"
│   ├─ Rationale:
│   │   ├─ Low credit risk → Short-horizon focus
│   │   ├─ Avoids excessive provisioning for performing loans
│   │   └─ Aligns with Basel PD (12-month regulatory PD)
│   ├─ Calculation:
│   │   ├─ Formula: ECL = EAD × PD(12m) × LGD
│   │   ├─ PD(12m): Probability of default in next 12 months
│   │   │   ├─ Point-in-Time (PIT): Adjusted for current economic conditions
│   │   │   └─ Forward-looking: Incorporate macro scenarios for next year
│   │   ├─ LGD: Loss given default (typically 30-60% for secured; 70-90% unsecured)
│   │   └─ EAD: Current outstanding balance + accrued interest
│   ├─ Discounting:
│   │   ├─ Typically NOT discounted (materiality; 12-month horizon short)
│   │   └─ If discounted: Use effective interest rate (EIR)
│   ├─ Maturity Insensitivity:
│   │   ├─ 12-month ECL same for 1-year, 5-year, or 30-year loan (if same PD)
│   │   └─ Rationale: Horizon capped at 12 months regardless of maturity
│   ├─ Example:
│   │   ├─ Loan: $1M, 5-year maturity, 12-month PD = 0.5%, LGD = 40%
│   │   └─ ECL = $1M × 0.5% × 40% = $2,000
│   └─ Coverage Ratio:
│       ├─ Typically 0.1%-1% of exposure (investment-grade)
│       └─ Higher for sub-investment-grade (1%-2%)
│
├─ Lifetime ECL (Stage 2 & 3):
│   ├─ Definition:
│   │   └─ IFRS 9.5.5.3: "ECL from all possible default events over expected life of instrument"
│   ├─ Rationale:
│   │   ├─ SICR or default → Heightened risk → Full lifetime provisioning
│   │   └─ Timely loss recognition (avoid "too little too late")
│   ├─ Calculation (General):
│   │   ├─ Formula: ECL = ∑[t=1 to T] { EAD(t) × PD(t) × LGD(t) × DF(t) }
│   │   ├─ T: Remaining contractual maturity (years or months)
│   │   ├─ PD(t): Marginal probability of default at time t (conditional on survival)
│   │   ├─ LGD(t): Loss given default (may vary by scenario)
│   │   ├─ EAD(t): Exposure at time t (amortization, prepayments, drawdowns)
│   │   └─ DF(t): Discount factor = 1 / (1 + EIR)^t
│   │
│   ├─ PD Term Structure:
│   │   ├─ Marginal PD(t): Default probability in period t given survival to t-1
│   │   │   └─ Hazard rate λ(t): Instantaneous default intensity
│   │   ├─ Survival Probability:
│   │   │   └─ S(t) = ∏[τ=1 to t] (1 - PD(τ)) = exp(-∫λ(τ)dτ)
│   │   ├─ Cumulative PD:
│   │   │   └─ CPD(t) = 1 - S(t)
│   │   ├─ Term Structure Shapes:
│   │   │   ├─ Flat: PD constant over time (simplest assumption)
│   │   │   ├─ Increasing: PD rises with maturity (credit deterioration over time)
│   │   │   ├─ Hump-Shaped: PD peaks mid-term (default risk highest Year 2-3)
│   │   │   └─ Reversion: Explicit forecast Years 1-3; revert to TTC thereafter
│   │   └─ Example:
│   │       ├─ Flat: PD = 2% per year → Lifetime PD (5 years) = 1 - (0.98)^5 = 9.6%
│   │       └─ Increasing: PD = [1%, 1.5%, 2%, 2.5%, 3%] → CPD = 9.5%
│   │
│   ├─ EAD Term Structure (Amortization):
│   │   ├─ Amortizing Loans (Mortgages, Auto):
│   │   │   ├─ Outstanding principal declines over time
│   │   │   ├─ EAD(t) = Outstanding(t) from amortization schedule
│   │   │   └─ Example: $100k 10-year loan; Year 5 EAD = $50k
│   │   ├─ Bullet Loans (Corporate):
│   │   │   ├─ Principal repaid at maturity
│   │   │   └─ EAD(t) = Constant (no amortization)
│   │   ├─ Revolving Credit (Credit Cards, Lines of Credit):
│   │   │   ├─ EAD uncertain (future drawdowns)
│   │   │   ├─ EAD = Drawn + Undrawn × CCF
│   │   │   └─ CCF (Credit Conversion Factor): 20-50% (stressed = 100%)
│   │   └─ Prepayments:
│   │       ├─ Mortgages: Voluntary prepayments reduce EAD
│   │       └─ Model: Constant prepayment rate (CPR) or conditional (CPR varies with rates)
│   │
│   ├─ Discounting:
│   │   ├─ Mandatory for Lifetime ECL (material impact over long horizons)
│   │   ├─ Discount Rate: Effective Interest Rate (EIR)
│   │   │   ├─ Definition: Rate that discounts future cash flows to amortized cost
│   │   │   └─ Includes: Origination fees, transaction costs (not credit risk premium)
│   │   ├─ Example:
│   │   │   ├─ Loss $10k in Year 5, EIR = 5%
│   │   │   └─ PV = $10k / (1.05)^5 = $7,835 (22% discount)
│   │   └─ Stage 3 Debate:
│   │       ├─ IFRS 9 allows original EIR or credit-adjusted rate
│   │       └─ Original EIR common (simpler; consistent with Stage 2)
│   │
│   ├─ Example (5-Year Corporate Loan):
│   │   ├─ Loan: $1M, 5-year bullet, LGD = 40%, EIR = 5%
│   │   ├─ PD term structure (annual marginal): [1%, 1.5%, 2%, 2.5%, 3%]
│   │   ├─ Survival probabilities: [99%, 97.5%, 95.6%, 93.2%, 90.4%]
│   │   ├─ Year-by-year ECL:
│   │   │   ├─ Year 1: $1M × 1% × 40% × 0.9524 = $3,810
│   │   │   ├─ Year 2: $1M × 1.5% × 40% × 0.9070 = $5,442
│   │   │   ├─ Year 3: $1M × 2% × 40% × 0.8638 = $6,910
│   │   │   ├─ Year 4: $1M × 2.5% × 40% × 0.8227 = $8,227
│   │   │   └─ Year 5: $1M × 3% × 40% × 0.7835 = $9,402
│   │   └─ Total Lifetime ECL = $33,791 (vs 12-month ECL = $4,000; 8.4× higher)
│   │
│   └─ Coverage Ratio:
│       ├─ Stage 2: Typically 3%-15% of exposure
│       ├─ Stage 3: Typically 30%-90% (LGD-driven; PD ≈ 100%)
│       └─ Higher for longer maturities (more time for default; cumulative PD higher)
│
├─ Maturity Impact on Lifetime ECL:
│   ├─ Short Maturity (< 2 years):
│   │   ├─ Lifetime ECL ≈ 12-month ECL (horizons similar)
│   │   └─ Stage 1→2 transfer: Modest ECL increase (1.5-2×)
│   ├─ Medium Maturity (2-5 years):
│   │   ├─ Lifetime ECL 3-8× higher than 12-month ECL
│   │   └─ Stage 1→2 transfer: Significant provision impact
│   ├─ Long Maturity (> 10 years):
│   │   ├─ Lifetime ECL 10-20× higher than 12-month ECL (if flat PD)
│   │   ├─ Discounting reduces impact (long-dated losses heavily discounted)
│   │   └─ Reversion to TTC: Mitigates extreme long-term forecasts
│   └─ Example (Fixed $1M Loan, PD = 2%/year, LGD = 40%, EIR = 5%):
│       ├─ 12-month ECL: $8,000 (constant)
│       ├─ Lifetime ECL (2-year): $15,200 (1.9× 12m)
│       ├─ Lifetime ECL (5-year): $34,000 (4.3× 12m)
│       ├─ Lifetime ECL (10-year): $58,000 (7.3× 12m)
│       └─ Lifetime ECL (30-year): $110,000 (13.8× 12m; but discounting reduces)
│
├─ Stage 2 vs Stage 3 Lifetime ECL:
│   ├─ Stage 2 (Performing Lifetime ECL):
│   │   ├─ PD < 100%: Credit risk elevated but not defaulted
│   │   ├─ Full PD term structure: Account for survival probabilities
│   │   ├─ Interest revenue: Calculated on gross carrying amount
│   │   └─ Example: Lifetime PD = 10%, ECL = $40k on $1M loan
│   │
│   ├─ Stage 3 (Impaired Lifetime ECL):
│   │   ├─ PD = 100% (default already occurred)
│   │   ├─ ECL = EAD × LGD (no PD uncertainty; focus on recovery)
│   │   ├─ Recovery timing critical:
│   │   │   ├─ Discount expected recoveries to present value
│   │   │   ├─ Foreclosure: 2-3 years; Bankruptcy: 1-5 years
│   │   │   └─ Example: Recovery $60k in 2 years → PV = $54.4k @ 5% EIR
│   │   ├─ Interest revenue: Calculated on net carrying amount (after ECL)
│   │   └─ Example: $1M loan, LGD = 60%, ECL = $600k (recovery $400k)
│   │
│   └─ Key Difference:
│       ├─ Stage 2: Probabilistic (PD × LGD); uncertainty in timing/occurrence
│       └─ Stage 3: Deterministic (LGD only); uncertainty in recovery amount/timing
│
├─ Practical Simplifications:
│   ├─ Flat PD Approximation:
│   │   ├─ Assume constant annual PD (simplifies calculation)
│   │   ├─ Lifetime ECL ≈ EAD × [1 - (1 - PD)^T] × LGD × avg DF
│   │   └─ Acceptable for portfolios with stable credit risk
│   │
│   ├─ Vintage Analysis:
│   │   ├─ Group loans by origination cohort
│   │   ├─ Apply cohort-specific default curves (based on historical performance)
│   │   └─ Common for retail portfolios (auto, credit card)
│   │
│   ├─ Roll Rates (Consumer Credit):
│   │   ├─ Migration through delinquency buckets (current → 30 DPD → 60 DPD → default)
│   │   ├─ Lifetime ECL = Sum over buckets weighted by transition probabilities
│   │   └─ Avoids explicit PD term structure modeling
│   │
│   └─ Portfolio-Level Models:
│       ├─ Aggregate exposures by segment (product, rating, maturity)
│       ├─ Apply segment-level PD/LGD; allocate back to loans
│       └─ Efficiency: Reduces computation for large portfolios
│
├─ Stage 1→2 Transfer Impact:
│   ├─ Provisioning Cliff:
│   │   ├─ 12-month ECL → Lifetime ECL transition causes large P&L charge
│   │   ├─ Magnitude depends on maturity (longer = larger jump)
│   │   └─ Example: 10-year loan; 12m ECL = $5k → Lifetime ECL = $50k (+$45k charge)
│   │
│   ├─ Timeliness of SICR Detection:
│   │   ├─ Early SICR detection → Gradual provisioning increase
│   │   ├─ Late SICR detection → Sudden large charge (cliff effect)
│   │   └─ Regulatory expectation: Timely SICR triggers; avoid delayed recognition
│   │
│   └─ P&L Volatility:
│       ├─ Frequent Stage 1↔2 oscillations → Volatile provisions
│       ├─ Mitigation: Hysteresis (different thresholds for upgrade vs downgrade)
│       └─ Cure probation: Require sustained improvement before Stage 2→1 transfer
│
└─ Regulatory & Disclosure:
    ├─ IFRS 7 Disclosure Requirements:
    │   ├─ ECL breakdown: Stage 1 (12m) vs Stage 2/3 (lifetime)
    │   ├─ Stage migrations: Transfers between stages; opening/closing balances
    │   ├─ Maturity analysis: ECL by maturity bucket
    │   └─ Sensitivity: Impact of alternative scenarios/assumptions
    │
    ├─ Basel IRB Alignment:
    │   ├─ 12-month PD (Basel) ≈ Stage 1 PD (IFRS 9)
    │   ├─ Lifetime PD (downturn) ≈ Stage 2 PD (IFRS 9 adverse scenario)
    │   └─ Efficiency: Use Basel models for IFRS 9 ECL (with adjustments)
    │
    └─ Audit Focus:
        ├─ PD term structure: Calibration; reasonableness of long-term PDs
        ├─ Discounting: EIR calculation; consistency across stages
        ├─ Maturity assumptions: Revolving credit expected life; prepayment rates
        └─ SICR triggers: Timeliness; avoid delayed Stage 1→2 transfers
```

**Key Insight:** 12-month ECL (Stage 1) = short-horizon, low provisions (0.1-1%); Lifetime ECL (Stage 2/3) = full maturity, high provisions (3-90%); Lifetime ECL 3-20× higher depending on maturity; Stage 1→2 transfer causes provisioning cliff; discounting reduces long-dated ECL impact.

## 5. Mini-Project
Compare 12-month vs lifetime ECL for loans with varying maturities:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# Portfolio parameters
maturities = [1, 2, 3, 5, 7, 10, 15, 20, 30]  # Years
loan_amount = 1_000_000  # $1M per loan
annual_pd = 0.02  # 2% flat annual PD for simplicity
lgd = 0.40  # 40% LGD
eir = 0.05  # 5% effective interest rate

# Calculate 12-month ECL (constant across maturities)
ecl_12m = loan_amount * annual_pd * lgd
print("="*70)
print("12-Month vs Lifetime ECL: Maturity Sensitivity")
print("="*70)
print(f"Loan Amount: ${loan_amount:,}")
print(f"Annual PD: {annual_pd:.1%}")
print(f"LGD: {lgd:.1%}")
print(f"EIR (Discount Rate): {eir:.1%}")
print("")
print(f"12-Month ECL (Stage 1): ${ecl_12m:,.0f} (constant across maturities)")
print("")

# Calculate lifetime ECL for each maturity
results = []

for maturity in maturities:
    # Marginal PD each year (constant annual_pd for simplicity)
    pds = [annual_pd] * maturity
    
    # Calculate year-by-year ECL
    lifetime_ecl = 0
    survival_prob = 1.0
    
    for year in range(1, maturity + 1):
        pd_year = pds[year - 1]
        marginal_loss = loan_amount * pd_year * lgd * survival_prob
        discount_factor = 1 / (1 + eir) ** year
        ecl_year = marginal_loss * discount_factor
        lifetime_ecl += ecl_year
        survival_prob *= (1 - pd_year)
    
    # Cumulative PD over lifetime
    cumulative_pd = 1 - survival_prob
    
    # Store results
    results.append({
        'maturity': maturity,
        'ecl_12m': ecl_12m,
        'ecl_lifetime': lifetime_ecl,
        'ratio': lifetime_ecl / ecl_12m,
        'cumulative_pd': cumulative_pd,
        'coverage_ratio_12m': (ecl_12m / loan_amount) * 100,
        'coverage_ratio_lifetime': (lifetime_ecl / loan_amount) * 100
    })

df = pd.DataFrame(results)

# Display table
print("Lifetime ECL by Maturity:")
print("-"*70)
print(f"{'Maturity':<10} {'12m ECL':<15} {'Lifetime ECL':<15} {'Ratio':<10} {'Cumul PD':<12}")
print("-"*70)
for _, row in df.iterrows():
    print(f"{row['maturity']:<10} ${row['ecl_12m']:<14,.0f} ${row['ecl_lifetime']:<14,.0f} {row['ratio']:<10.1f}× {row['cumulative_pd']:<11.1%}")

print("")

# Key observations
print("Key Observations:")
print("-"*70)
short_term = df[df['maturity'] == 2].iloc[0]
medium_term = df[df['maturity'] == 5].iloc[0]
long_term = df[df['maturity'] == 30].iloc[0]

print(f"Short-term (2Y): Lifetime ECL {short_term['ratio']:.1f}× higher than 12m")
print(f"Medium-term (5Y): Lifetime ECL {medium_term['ratio']:.1f}× higher than 12m")
print(f"Long-term (30Y): Lifetime ECL {long_term['ratio']:.1f}× higher than 12m")
print("")
print(f"Stage 1→2 transfer impact (5Y loan): +${medium_term['ecl_lifetime'] - medium_term['ecl_12m']:,.0f} provision charge")

# Scenario: Portfolio of mixed maturities
print("\n" + "="*70)
print("Portfolio Example: Mixed Maturities")
print("="*70)

# Simulate portfolio with distribution across maturities
portfolio_distribution = {
    1: 50,   # 50 loans with 1-year maturity
    2: 100,
    3: 150,
    5: 200,
    7: 150,
    10: 100,
    15: 50,
    20: 30,
    30: 20
}

total_loans = sum(portfolio_distribution.values())
portfolio_ecl_12m = 0
portfolio_ecl_lifetime = 0

for maturity, count in portfolio_distribution.items():
    row = df[df['maturity'] == maturity].iloc[0]
    portfolio_ecl_12m += row['ecl_12m'] * count
    portfolio_ecl_lifetime += row['ecl_lifetime'] * count

print(f"Total Loans: {total_loans}")
print(f"Total Exposure: ${loan_amount * total_loans:,.0f}")
print("")
print(f"Portfolio 12-Month ECL (Stage 1): ${portfolio_ecl_12m:,.0f}")
print(f"Portfolio Lifetime ECL (Stage 2): ${portfolio_ecl_lifetime:,.0f}")
print(f"Average Ratio: {portfolio_ecl_lifetime / portfolio_ecl_12m:.1f}×")
print("")
print(f"If 20% of portfolio migrates Stage 1→2:")
print(f"  → Provision increase: ${0.20 * (portfolio_ecl_lifetime - portfolio_ecl_12m):,.0f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ECL by maturity
ax = axes[0, 0]
ax.plot(df['maturity'], df['ecl_12m'] / 1e3, 'g-', linewidth=2, marker='o', label='12-Month ECL')
ax.plot(df['maturity'], df['ecl_lifetime'] / 1e3, 'r-', linewidth=2, marker='s', label='Lifetime ECL')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('ECL ($1000s)')
ax.set_title('12-Month vs Lifetime ECL by Maturity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Ratio (Lifetime / 12m)
ax = axes[0, 1]
ax.plot(df['maturity'], df['ratio'], 'b-', linewidth=2, marker='o')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Lifetime ECL / 12-Month ECL (Ratio)')
ax.set_title('ECL Ratio: Maturity Sensitivity')
ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='1× (Equal)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Coverage ratios
ax = axes[1, 0]
width = 0.35
x = np.arange(len(df['maturity']))
bars1 = ax.bar(x - width/2, df['coverage_ratio_12m'], width, label='12m ECL', alpha=0.7, color='green')
bars2 = ax.bar(x + width/2, df['coverage_ratio_lifetime'], width, label='Lifetime ECL', alpha=0.7, color='red')

ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Coverage Ratio (%)')
ax.set_title('Coverage Ratio by Maturity')
ax.set_xticks(x)
ax.set_xticklabels(df['maturity'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Cumulative PD vs Maturity
ax = axes[1, 1]
ax.plot(df['maturity'], df['cumulative_pd'] * 100, 'purple', linewidth=2, marker='o')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Cumulative PD (%)')
ax.set_title('Cumulative Default Probability vs Maturity')
ax.grid(alpha=0.3)

# Add annotation for key points
for mat in [5, 10, 30]:
    row = df[df['maturity'] == mat].iloc[0]
    ax.annotate(f"{row['cumulative_pd']:.1%}", 
                xy=(mat, row['cumulative_pd'] * 100), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('lifetime_vs_12month_ecl.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Lifetime ECL increases with maturity (more time = more default risk)")
print("   → 5Y: 4× higher; 10Y: 7× higher; 30Y: 14× higher")
print("")
print("2. Discounting reduces long-dated ECL impact")
print("   → Without discounting, 30Y would be ~20× higher")
print("")
print("3. Stage 1→2 transfer causes provisioning cliff")
print("   → 5Y loan: +$25k; 10Y loan: +$50k; 30Y loan: +$100k")
print("")
print("4. Portfolio weighted average ratio depends on maturity mix")
print("   → Longer-dated portfolios (mortgages): Higher Stage 1→2 impact")
```

## 6. Challenge Round
When 12-month vs lifetime ECL frameworks fail or introduce complexity:
- **Revolving Credit (Uncertain Maturity)**: Credit card with no fixed maturity → Lifetime ECL horizon ambiguous; solution: Use expected behavioral life (e.g., 3-5 years based on historical usage); exclude contractual cancel-on-demand clauses if not exercised
- **Long-Dated Loans (30+ Years)**: Mortgage with 30-year term → PD term structure extremely uncertain; solution: Revert to TTC mean after Year 5; flatten PD curve; rely on discounting to reduce far-future impact
- **Short Maturity (< 1 Year)**: 6-month loan → Lifetime ECL ≈ 12-month ECL → Stage 1→2 transfer negligible impact; solution: Accept minimal Stage transfer effect; focus on Stage 3 (default) detection
- **Prepayments (Mortgages)**: Early repayment reduces EAD → Lifetime ECL lower; solution: Model conditional prepayment rates (CPR); sensitivity to interest rates; stress test low prepayment scenario
- **Stage 2 Cure**: Loan in Stage 2 for 6 months; cures → Revert to Stage 1 → Provision release → P&L volatility; solution: Require probation period (3-6 months current) before cure; avoid oscillation
- **Discounting Ambiguity (Stage 3)**: IFRS 9 allows original EIR or credit-adjusted rate → Choice impacts ECL by 20-40%; solution: Consistent policy; original EIR common (simpler); disclose choice

## 7. Key References
- [IFRS 9 Financial Instruments (Section 5.5.3-5.5.5)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official standard; 12-month vs lifetime ECL definitions; measurement requirements
- [PwC IFRS 9: Expected Credit Loss (2019)](https://www.pwc.com/gx/en/audit-services/ifrs/publications/ifrs-9/expected-credit-loss-ifrs-9-practical-guide.pdf) - Practical guide; ECL calculation examples; maturity considerations; discount rate application
- [EY IFRS 9 Impairment Banking Survey (2020)](https://www.ey.com/en_gl/ifrs-technical-resources/ifrs-9-impairment-banking-survey-2020) - Industry practices; Stage 1/2 coverage ratios; lifetime ECL methodologies; benchmarking data

---
**Status:** IFRS 9 Core Concept | **Complements:** Three-Stage Approach, Expected Credit Loss Models, SICR, Forward-Looking Information
