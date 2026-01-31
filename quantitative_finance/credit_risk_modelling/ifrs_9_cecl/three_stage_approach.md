# Three-Stage Approach (IFRS 9)

## 1. Concept Skeleton
**Definition:** Classification framework for financial instruments into Stage 1 (12-month ECL), Stage 2 (lifetime ECL performing), Stage 3 (credit-impaired lifetime ECL) based on credit deterioration since origination  
**Purpose:** Timely recognition of credit losses; forward-looking expected loss accounting; replace incurred loss model (IAS 39); escalate provisioning as credit risk increases  
**Prerequisites:** Expected credit loss (ECL) concepts, significant increase in credit risk (SICR) criteria, default definition, probability of default (PD), loss given default (LGD)

## 2. Comparative Framing
| Stage | Credit Quality | ECL Horizon | Loss Recognition | Interest Revenue | Typical Example |
|-------|----------------|-------------|------------------|------------------|-----------------|
| **Stage 1** | No significant deterioration | 12-month ECL | Low (expected losses in next year) | Gross carrying amount × EIR | New loan; current on payments |
| **Stage 2** | Significant increase in credit risk (SICR) | Lifetime ECL | Higher (all expected losses to maturity) | Gross carrying amount × EIR | 30+ days past due (presumption); rating downgrade |
| **Stage 3** | Credit-impaired (default) | Lifetime ECL | Highest (impaired loss) | Net carrying amount × EIR | 90+ days past due; bankruptcy; covenant breach |

## 3. Examples + Counterexamples

**Stage 1 Example:**  
Corporate loan $100M originated at BBB rating; current on payments; market conditions stable. 12-month PD = 0.5%, LGD = 40%. ECL = $100M × 0.5% × 40% = $200k provision.

**Stage 2 Trigger:**  
Same loan downgrades to BB after 1 year (3-notch downgrade = SICR). Lifetime PD = 5%, LGD = 40%, maturity 4 years remaining. Lifetime ECL = $100M × 5% × 40% = $2M provision (10× Stage 1).

**Stage 3 Impairment:**  
Borrower files bankruptcy; 90+ days past due. Default certain (PD ≈ 100%), LGD estimated 60% based on collateral. ECL = $100M × 100% × 60% = $60M provision.

**Edge Case (Cured Loan):**  
Stage 2 loan cures (payments resume; rating upgrade to BBB). SICR no longer present → Transfer back to Stage 1 → Reduce ECL to 12-month. Provision reversal through P&L.

**Failure Case (No SICR Detection):**  
Bank ignores rating downgrade (BB→B), keeps loan in Stage 1. Borrower defaults 6 months later. Insufficient provisioning → Sudden large loss; regulatory criticism.

## 4. Layer Breakdown
```
Three-Stage Model Framework:

├─ Stage 1: Performing (No SICR):
│   ├─ Definition: Credit risk has NOT increased significantly since initial recognition
│   ├─ ECL Recognition: 12-month expected credit losses
│   │   ├─ Horizon: Expected losses from defaults in next 12 months only
│   │   ├─ Formula: ECL = EAD × PD(12m) × LGD
│   │   └─ Rationale: Low deterioration risk; short-horizon focus
│   ├─ Interest Revenue: Calculated on gross carrying amount
│   │   └─ Gross amount = Amortized cost before ECL allowance
│   ├─ Criteria for Stage 1:
│   │   ├─ No 30+ days past due (rebuttable presumption)
│   │   ├─ No significant rating downgrade (e.g., < 2 notches)
│   │   ├─ No adverse qualitative indicators (restructuring, covenant breach)
│   │   └─ Low default probability (PD < threshold, e.g., 1%)
│   ├─ Examples:
│   │   ├─ Newly originated loans (Day 1)
│   │   ├─ Current on payments; stable credit metrics
│   │   ├─ Investment-grade securities (AAA to BBB-)
│   │   └─ No negative watchlist flags
│   └─ Provisioning: Typically 0.1%-1% of exposure (depends on PD/LGD)
│
├─ Stage 2: Performing (SICR):
│   ├─ Definition: Credit risk HAS increased significantly since origination
│   ├─ ECL Recognition: Lifetime expected credit losses
│   │   ├─ Horizon: All expected losses over remaining life of instrument
│   │   ├─ Formula: ECL = EAD × PD(lifetime) × LGD
│   │   └─ Integration: Sum over all future periods weighted by PD(t)
│   ├─ Interest Revenue: Still calculated on gross carrying amount
│   │   └─ Not yet credit-impaired; revenue recognition continues
│   ├─ Significant Increase in Credit Risk (SICR) Triggers:
│   │   ├─ Quantitative Indicators:
│   │   │   ├─ 30+ days past due (rebuttable presumption per IFRS 9)
│   │   │   ├─ Credit rating downgrade (e.g., 2+ notches)
│   │   │   ├─ Relative PD increase: PD(current) / PD(origination) > threshold (e.g., 2×)
│   │   │   ├─ Absolute PD increase: PD > 5% (institution-specific)
│   │   │   └─ LTV deterioration: Loan-to-value > 100% (negative equity)
│   │   ├─ Qualitative Indicators:
│   │   │   ├─ Borrower financial distress (covenant breach, restructuring request)
│   │   │   ├─ Industry/sector deterioration (oil price collapse for energy loans)
│   │   │   ├─ Economic downturn in borrower geography
│   │   │   ├─ Management changes, litigation, regulatory action
│   │   │   └─ Adverse news (earnings warnings, credit watch negative)
│   │   └─ Backstop: 30 days past due (mandatory unless rebutted)
│   ├─ Stage Transfer Logic:
│   │   ├─ Stage 1 → Stage 2: SICR criteria met
│   │   ├─ Stage 2 → Stage 1: SICR no longer present (cure)
│   │   │   └─ Typically requires 6-12 months of satisfactory performance
│   │   └─ Stage 2 → Stage 3: Default occurs
│   ├─ Provisioning: Typically 3%-15% of exposure (higher PD × longer horizon)
│   └─ Key Challenge: Defining SICR thresholds (avoid cliff effects; balance timeliness vs stability)
│
├─ Stage 3: Non-Performing (Credit-Impaired):
│   ├─ Definition: Objective evidence of impairment (default)
│   ├─ ECL Recognition: Lifetime expected credit losses (default-adjusted)
│   │   ├─ Horizon: Remaining life (but default already occurred)
│   │   ├─ Formula: ECL = EAD × PD(default = 100%) × LGD = EAD × LGD
│   │   └─ Focus shifts to recovery estimation (collateral value, workout process)
│   ├─ Interest Revenue: Calculated on net carrying amount (after ECL deduction)
│   │   └─ Net amount = Amortized cost - ECL allowance
│   │   └─ Lower interest revenue (reflects credit-impaired status)
│   ├─ Default Triggers (IFRS 9 aligns with Basel):
│   │   ├─ 90+ days past due (presumption of default)
│   │   ├─ Bankruptcy, insolvency, administration
│   │   ├─ Covenant breach leading to acceleration
│   │   ├─ Distressed debt restructuring (concessions due to financial difficulty)
│   │   ├─ Sale of financial asset at material credit-related loss
│   │   └─ Internal rating = Default grade (D)
│   ├─ LGD Estimation:
│   │   ├─ Collateral valuation: Market value - costs to sell
│   │   ├─ Discounted cash flow: Expected recoveries from workout
│   │   ├─ Historical recovery rates: Industry-specific LGD (e.g., secured = 30%, unsecured = 70%)
│   │   └─ Time to recovery: Discount recoveries to present value
│   ├─ Stage Transfer Logic:
│   │   ├─ Stage 2 → Stage 3: Default occurs
│   │   ├─ Stage 3 → Stage 2: Cure (default status removed)
│   │   │   ├─ Rare; requires full payment of arrears + probation period
│   │   │   └─ Typically 12+ months satisfactory performance
│   │   └─ Direct Stage 1 → Stage 3 possible (sudden default)
│   ├─ Provisioning: Typically 30%-90% of exposure (high LGD; low recovery)
│   └─ Write-Off: When no reasonable expectation of recovery
│       ├─ Remove from balance sheet
│       ├─ ECL allowance utilized
│       └─ Continue collection efforts (off-balance sheet)
│
├─ Stage Transfer Mechanics:
│   ├─ Monthly (or more frequent) assessment:
│   │   ├─ Evaluate SICR criteria for all Stage 1 exposures
│   │   ├─ Evaluate default criteria for Stage 2 exposures
│   │   ├─ Check cure criteria for Stage 2/3 exposures
│   │   └─ Update ECL allowances for transfers
│   ├─ Cliff Effects Mitigation:
│   │   ├─ Gradual PD increase approach (smooth transition)
│   │   ├─ Multiple SICR indicators (avoid single metric dominance)
│   │   ├─ Expert judgment overlay (qualitative factors)
│   │   └─ Backstop prevents delayed recognition (30 DPD mandatory)
│   └─ P&L Impact:
│       ├─ ECL increase: Provision expense (credit loss)
│       ├─ ECL decrease: Provision release (gain)
│       ├─ Stage 1→2 transfer: Lifetime ECL charge (significant impact)
│       └─ Stage 3 default: Large impairment loss (one-time)
│
├─ Practical Implementation:
│   ├─ Data Requirements:
│   │   ├─ Origination data: PD, rating, LTV at inception
│   │   ├─ Current data: Payment status, rating, collateral value
│   │   ├─ Macroeconomic scenarios: GDP, unemployment, interest rates
│   │   └─ Historical data: Default rates, recovery rates by segment
│   ├─ Model Infrastructure:
│   │   ├─ PD models: Credit scoring, transition matrices, survival analysis
│   │   ├─ LGD models: Recovery rate estimation, collateral valuation
│   │   ├─ EAD models: Credit conversion factors, drawdown at default
│   │   ├─ SICR framework: Quantitative + qualitative rules engine
│   │   └─ Scenario engine: Forward-looking macro scenarios weighted by probability
│   ├─ Governance:
│   │   ├─ Model validation: Annual review; backtesting PD/LGD
│   │   ├─ SICR thresholds: Documented rationale; Board approval
│   │   ├─ Stage migration reports: Monthly monitoring; trend analysis
│   │   └─ Management overlays: Expert adjustments for model limitations
│   └─ Systems:
│       ├─ Data warehouse: Centralized exposure, payment, rating data
│       ├─ ECL engine: Calculate 12-month and lifetime ECL by instrument
│       ├─ Stage classification module: Apply SICR/default rules
│       ├─ Reporting: Regulatory (EBA ITS 2018/1627); financial statements
│       └─ Audit trail: All stage transfers, model assumptions documented
│
└─ Comparison to IAS 39 (Incurred Loss Model):
    ├─ IAS 39: Recognized losses only when objective evidence of impairment (backward-looking)
    ├─ IFRS 9: Recognizes expected losses immediately (forward-looking)
    ├─ Impact: Earlier loss recognition; higher provisions in economic downturns
    ├─ Procyclicality: IFRS 9 more countercyclical (builds provisions in good times)
    └─ Complexity: IFRS 9 requires sophisticated PD/LGD models; IAS 39 simpler (historical loss rates)
```

**Key Insight:** Stage 1 (12m ECL) = low risk; Stage 2 (lifetime ECL) = elevated risk; Stage 3 (impaired lifetime ECL) = default. SICR triggers Stage 1→2 transfer (major provisioning impact); early SICR detection critical to avoid sudden losses.

## 5. Mini-Project
Simulate three-stage ECL provisioning for a loan portfolio:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# Portfolio parameters
n_loans = 1000
loan_amount = 100_000  # $100k per loan
lgd = 0.40  # 40% loss given default

# Origination: All loans start in Stage 1
df = pd.DataFrame({
    'loan_id': range(n_loans),
    'amount': loan_amount,
    'stage': 1,
    'pd_12m_orig': np.random.uniform(0.002, 0.01, n_loans),  # 0.2%-1% 12m PD at origination
    'pd_lifetime_orig': np.random.uniform(0.03, 0.08, n_loans),  # 3%-8% lifetime PD
    'rating_orig': np.random.choice(['AAA', 'AA', 'A', 'BBB'], n_loans, p=[0.1, 0.3, 0.4, 0.2])
})

# Simulate 1 year forward: Some loans migrate to Stage 2 or Stage 3
# Stage 2 triggers: PD increase > 2x origination, or rating downgrade 2+ notches
# Stage 3 triggers: Default (random draw based on PD)

# Current PD (after 1 year): Most stable, some deteriorate
pd_multiplier = np.random.lognormal(0, 0.5, n_loans)  # Log-normal shocks
df['pd_12m_current'] = df['pd_12m_orig'] * pd_multiplier
df['pd_lifetime_current'] = df['pd_lifetime_orig'] * pd_multiplier

# Clip PD to [0, 1]
df['pd_12m_current'] = df['pd_12m_current'].clip(0, 1)
df['pd_lifetime_current'] = df['pd_lifetime_current'].clip(0, 1)

# SICR detection: Relative PD increase > 2x
df['sicr_flag'] = (df['pd_12m_current'] / df['pd_12m_orig']) > 2.0

# Default simulation: Draw from Bernoulli(PD_12m)
df['default_flag'] = np.random.binomial(1, df['pd_12m_current'])

# Stage classification
def classify_stage(row):
    if row['default_flag'] == 1:
        return 3  # Default
    elif row['sicr_flag']:
        return 2  # SICR
    else:
        return 1  # No SICR

df['stage'] = df.apply(classify_stage, axis=1)

# ECL calculation
def calculate_ecl(row):
    if row['stage'] == 1:
        # 12-month ECL
        return row['amount'] * row['pd_12m_current'] * lgd
    elif row['stage'] == 2:
        # Lifetime ECL (performing)
        return row['amount'] * row['pd_lifetime_current'] * lgd
    else:
        # Stage 3: Default; ECL = LGD (PD = 100%)
        return row['amount'] * lgd

df['ecl'] = df.apply(calculate_ecl, axis=1)

# Summary statistics
print("="*70)
print("IFRS 9 Three-Stage Model: Loan Portfolio ECL")
print("="*70)
print(f"Total Loans: {n_loans}")
print(f"Total Exposure: ${df['amount'].sum():,.0f}")
print("")

stage_summary = df.groupby('stage').agg({
    'loan_id': 'count',
    'amount': 'sum',
    'ecl': 'sum'
}).rename(columns={'loan_id': 'count'})

stage_summary['coverage_ratio'] = (stage_summary['ecl'] / stage_summary['amount']) * 100

print("Stage Distribution:")
print("-"*70)
print(stage_summary)
print("")

total_ecl = df['ecl'].sum()
total_exposure = df['amount'].sum()
overall_coverage = (total_ecl / total_exposure) * 100

print(f"Total ECL Allowance: ${total_ecl:,.0f}")
print(f"Overall Coverage Ratio: {overall_coverage:.2f}%")
print("")

# Stage 2 breakdown
stage2_df = df[df['stage'] == 2]
if len(stage2_df) > 0:
    avg_pd_increase = (stage2_df['pd_12m_current'] / stage2_df['pd_12m_orig']).mean()
    print(f"Stage 2 Loans: Average PD increase {avg_pd_increase:.1f}× from origination")
    print("")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Stage distribution
ax = axes[0, 0]
stage_counts = df['stage'].value_counts().sort_index()
colors = ['green', 'orange', 'red']
ax.bar(stage_counts.index, stage_counts.values, color=colors, alpha=0.7)
ax.set_xlabel('Stage')
ax.set_ylabel('Number of Loans')
ax.set_title('Loan Distribution by Stage')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Stage 1\n(12m ECL)', 'Stage 2\n(Lifetime ECL)', 'Stage 3\n(Default)'])
ax.grid(axis='y', alpha=0.3)

# Plot 2: ECL by stage
ax = axes[0, 1]
stage_ecl = df.groupby('stage')['ecl'].sum()
ax.bar(stage_ecl.index, stage_ecl.values / 1e6, color=colors, alpha=0.7)
ax.set_xlabel('Stage')
ax.set_ylabel('ECL Allowance ($M)')
ax.set_title('Total ECL by Stage')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax.grid(axis='y', alpha=0.3)

# Plot 3: PD distribution by stage
ax = axes[1, 0]
for stage, color, label in zip([1, 2, 3], colors, ['Stage 1', 'Stage 2', 'Stage 3']):
    stage_data = df[df['stage'] == stage]['pd_12m_current']
    if len(stage_data) > 0:
        ax.hist(stage_data, bins=20, alpha=0.5, color=color, label=label)

ax.set_xlabel('Current 12-Month PD')
ax.set_ylabel('Frequency')
ax.set_title('PD Distribution by Stage')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Coverage ratio by stage
ax = axes[1, 1]
coverage = (df.groupby('stage')['ecl'].sum() / df.groupby('stage')['amount'].sum()) * 100
ax.bar(coverage.index, coverage.values, color=colors, alpha=0.7)
ax.set_xlabel('Stage')
ax.set_ylabel('Coverage Ratio (%)')
ax.set_title('ECL Coverage Ratio by Stage')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax.axhline(overall_coverage, color='black', linestyle='--', linewidth=2, label=f'Overall: {overall_coverage:.2f}%')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('three_stage_approach.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("Key Observations:")
print("="*70)
print("1. Stage 1 loans: Majority of portfolio; low ECL (0.1-1% coverage)")
print("2. Stage 2 loans: Elevated PD; significantly higher ECL (3-10% coverage)")
print("3. Stage 3 loans: Defaulted; highest ECL (40%+ coverage = LGD)")
print("")
print("4. SICR detection critical: Delayed Stage 2 migration → understated provisions")
print("5. Total ECL = sum across stages; weighted by exposure distribution")
```

## 6. Challenge Round
When three-stage classification fails or introduces complexity:
- **Cliff Effects**: Single loan crosses 30 DPD threshold → Immediate Stage 1→2 transfer → Large ECL jump; solution: Use probationary period (multiple months SICR before transfer); smooth PD increase approach
- **Curing Instability**: Loan oscillates between Stage 1 and Stage 2 (volatile PD) → Frequent ECL adjustments; P&L volatility; solution: Require sustained improvement (6-12 months) before cure; hysteresis in thresholds
- **SICR Threshold Sensitivity**: Small PD change near threshold → Large ECL impact; solution: Multiple SICR indicators (rating, PD, DPD, qualitative); weight of evidence approach; avoid single metric dominance
- **New Originations (No Baseline)**: Loan originated today → No "origination PD" for comparison; solution: Use underwriting PD as baseline; Stage 1 until SICR observed relative to underwriting
- **Purchased Credit-Impaired (PCI)**: Loan bought at discount (already impaired) → Day 1 Stage 3? Solution: IFRS 9 has special PCI rules (recognize gross-up method; different impairment calc)
- **Model Risk**: PD model overstates deterioration → Excessive Stage 2 migrations; solution: Backtesting PD models; management overlays; independent validation

## 7. Key References
- [IFRS 9 Financial Instruments (Full Standard)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official IFRS Foundation standard; classification and measurement; impairment
- [EBA Guidelines on Credit Institutions' Credit Risk Management (2017)](https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-credit-institutions-credit-risk-management-practices-and-accounting-for-expected-credit-losses) - European Banking Authority implementation guidance; SICR criteria; staging approaches
- [Deloitte IFRS 9 Impairment Guide (2019)](https://www2.deloitte.com/content/dam/Deloitte/global/Documents/Financial-Services/gx-fsi-ifrs9-guide-impairment.pdf) - Practical implementation; worked examples; model approaches; industry practices

---
**Status:** IFRS 9 Core Concept | **Complements:** Expected Credit Loss Models, SICR, Forward-Looking Information, Lifetime ECL
