# Significant Increase in Credit Risk (SICR)

## 1. Concept Skeleton
**Definition:** Threshold determining transfer from Stage 1 (12-month ECL) to Stage 2 (lifetime ECL); requires assessment whether credit risk has increased significantly since initial recognition; triggers higher provisioning  
**Purpose:** Timely recognition of credit deterioration; avoid delayed loss recognition; balance false positives (premature Stage 2) vs false negatives (missed deterioration); align with early warning indicators  
**Prerequisites:** Default probability (PD) estimation, credit rating migration, delinquency status, qualitative risk indicators, origination data, macroeconomic overlays

## 2. Comparative Framing
| SICR Indicator | Type | Threshold Example | Data Requirement | Timeliness | False Positive Risk |
|----------------|------|------------------|------------------|------------|---------------------|
| **30 DPD (Days Past Due)** | Quantitative (mandatory backstop) | ≥30 days overdue | Payment history | High (immediate) | Low (observable default signal) |
| **Relative PD Change** | Quantitative | PD(current) / PD(orig) > 2× | Credit models | Medium (monthly) | Medium (model sensitivity) |
| **Absolute PD Threshold** | Quantitative | PD > 5% | Credit models | Medium (monthly) | Medium (arbitrary cutoff) |
| **Rating Downgrade** | Qualitative/Quantitative | 2+ notch downgrade | External/internal ratings | Medium-High | Low (validated ratings) |
| **Watchlist / Forbearance** | Qualitative | Added to watchlist | Relationship manager flags | High | High (subjective) |
| **Economic Overlay** | Quantitative | Sector in distress (e.g., oil price collapse) | Macro models | Low (lagging) | High (broad brush) |

## 3. Examples + Counterexamples

**SICR Trigger (30 DPD Backstop):**  
Corporate loan $1M; 32 days past due at month-end. Mandatory SICR presumption → Transfer Stage 1→2 → Lifetime ECL increases from $5k to $40k.

**Relative PD Increase:**  
Loan originated at BBB rating (PD = 0.5%); downgrades to BB (PD = 2.0%). Relative increase: 2.0% / 0.5% = 4× > 2× threshold → SICR triggered.

**False Positive (Temporary Delay):**  
Borrower 35 DPD due to administrative error (payment processing delay); fundamentally sound. SICR triggered → Stage 2 → Large ECL increase. Cures next month → Revert to Stage 1 (provision reversal; P&L volatility).

**False Negative (Missed SICR):**  
Investment-grade bond (AAA); credit fundamentals deteriorate (leverage increases, cash flow declines) but no rating downgrade yet (rating agencies lag). SICR not triggered → Remains Stage 1 → Insufficient provisioning. Default 6 months later → Sudden large loss.

**Qualitative Overlay:**  
Oil & gas sector loan portfolio; oil price collapses 50%. Individual loan PD increases modest (1.2× origination) but sector distress warrants SICR → Management overlay transfers entire sector to Stage 2 (prudent provisioning).

## 4. Layer Breakdown
```
Significant Increase in Credit Risk (SICR) Framework:

├─ IFRS 9 Requirements:
│   ├─ Principle-Based: No mechanical formula; requires judgment
│   ├─ Assessment at Each Reporting Date: Monthly or quarterly
│   ├─ Compare Current Credit Risk to Origination Risk:
│   │   ├─ Baseline: Credit risk at initial recognition (Day 1)
│   │   ├─ Current: Credit risk at reporting date
│   │   └─ Test: Has credit risk increased significantly?
│   ├─ Both Quantitative and Qualitative Indicators:
│   │   └─ Cannot rely on single metric; holistic assessment
│   └─ Forward-Looking: Incorporate reasonable and supportable information
│
├─ Quantitative Indicators:
│   ├─ 30 Days Past Due (Rebuttable Presumption):
│   │   ├─ IFRS 9.5.5.11: "Presumption that SICR when >30 DPD"
│   │   ├─ Backstop: Mandatory unless rebutted with evidence
│   │   ├─ Rationale: Payment delay signals liquidity stress
│   │   ├─ Rebuttal Criteria:
│   │   │   ├─ Administrative error (documented; resolved quickly)
│   │   │   ├─ Dispute (valid; payment made after resolution)
│   │   │   └─ Isolated incident (strong payment history; one-off event)
│   │   └─ Implementation: Flag all exposures ≥30 DPD; review for rebuttal
│   │
│   ├─ Relative PD Change:
│   │   ├─ Formula: PD(current) / PD(origination) > threshold
│   │   ├─ Threshold Examples:
│   │   │   ├─ Conservative: 2× (doubling of PD triggers SICR)
│   │   │   ├─ Moderate: 3× (triple PD)
│   │   │   └─ Investment-Grade: 1.5× (lower tolerance for IG)
│   │   ├─ Rationale: Relative deterioration captures credit migration
│   │   ├─ Advantages:
│   │   │   ├─ Scale-invariant: Works for low PD (0.1% → 0.2% = SICR) and high PD (5% → 10%)
│   │   │   └─ Aligns with rating migration (notch downgrades correlate with PD multiples)
│   │   ├─ Challenges:
│   │   │   ├─ Noisy for very low PD (0.01% → 0.03% = 3×; but both negligible)
│   │   │   ├─ Requires robust origination PD (poor underwriting → wrong baseline)
│   │   │   └─ Model risk: PD estimation errors amplified
│   │   └─ Implementation: Calculate monthly PD; compare to origination PD; flag if ratio > threshold
│   │
│   ├─ Absolute PD Threshold:
│   │   ├─ Formula: PD(current) > X% (e.g., 5%)
│   │   ├─ Rationale: High absolute PD = elevated default risk (Stage 2 appropriate)
│   │   ├─ Threshold Selection:
│   │   │   ├─ Conservative: 3% (sub-investment grade)
│   │   │   ├─ Moderate: 5% (CCC territory)
│   │   │   └─ Aggressive: 10% (near-default)
│   │   ├─ Advantages: Simple; independent of origination
│   │   ├─ Challenges:
│   │   │   ├─ Cliff effect: PD = 4.9% (Stage 1) vs PD = 5.1% (Stage 2) → Large ECL jump
│   │   │   ├─ Ignores relative deterioration: IG loan 0.5% → 3% (not SICR if threshold 5%)
│   │   │   └─ Arbitrary cutoff
│   │   └─ Often used as complement to relative PD (OR condition: relative OR absolute)
│   │
│   ├─ Lifetime PD Change:
│   │   ├─ Similar to 12-month PD but uses lifetime PD comparison
│   │   ├─ More forward-looking; captures long-term deterioration
│   │   └─ Computationally intensive (full term structure)
│   │
│   └─ Credit Rating Downgrade:
│       ├─ External Ratings (Moody's, S&P, Fitch):
│       │   ├─ Threshold: 2+ notch downgrade (e.g., BBB → BB; A → BBB-)
│       │   ├─ Rationale: Rating agencies incorporate credit fundamentals
│       │   ├─ Advantages: Independent validation; widely understood
│       │   └─ Challenges: Lagging (agencies slow to downgrade); "too little too late"
│       ├─ Internal Ratings:
│       │   ├─ Bank-specific rating models; updated more frequently
│       │   └─ Aligned with PD (internal rating scale maps to PD)
│       └─ Implementation: Map rating changes to SICR (2 notches = SICR)
│
├─ Qualitative Indicators:
│   ├─ Forbearance / Restructuring:
│   │   ├─ Definition: Concessions granted due to borrower financial difficulty
│   │   ├─ Examples: Payment holiday, term extension, interest rate reduction, covenant waiver
│   │   ├─ SICR Implication: Forbearance signals distress → Automatic Stage 2
│   │   ├─ Cure: Probation period (6-12 months satisfactory performance) before revert to Stage 1
│   │   └─ EBA Guidelines: Forborne exposures remain Stage 2 minimum 12 months
│   │
│   ├─ Watchlist / Credit Watch:
│   │   ├─ Relationship manager flags borrower for heightened monitoring
│   │   ├─ Triggers: Covenant breach, adverse news, management changes, litigation
│   │   ├─ SICR Implication: Watchlist addition → Presume SICR (unless rebutted)
│   │   └─ Challenge: Subjectivity; consistency across portfolio
│   │
│   ├─ Adverse Business Conditions:
│   │   ├─ Borrower-Specific: Loss of major customer, regulatory action, failed product launch
│   │   ├─ Sector-Specific: Oil price collapse (energy), pandemic (airlines), regulatory change
│   │   └─ SICR Implication: Material adverse change → SICR assessment; possible management overlay
│   │
│   ├─ Collateral Deterioration:
│   │   ├─ Loan-to-Value (LTV) > 100% (negative equity)
│   │   ├─ Real estate value decline (market downturn)
│   │   └─ SICR Implication: Unsecured exposure → Higher risk → SICR triggered
│   │
│   ├─ Macroeconomic Overlays:
│   │   ├─ Sector-level distress (e.g., COVID-19 impact on travel/hospitality)
│   │   ├─ Geographic stress (e.g., regional recession)
│   │   └─ SICR Implication: Transfer entire segment to Stage 2 (management judgment)
│   │
│   └─ Covenant Breaches:
│       ├─ Debt service coverage ratio (DSCR) < threshold
│       ├─ Leverage ratio exceeds maximum
│       └─ SICR Implication: Breach signals deterioration → SICR
│
├─ 30 DPD Backstop (Rebuttable Presumption):
│   ├─ Mandatory IFRS 9 Requirement:
│   │   ├─ IFRS 9.5.5.11: "Rebuttable presumption that SICR when >30 DPD"
│   │   └─ Cannot be waived without documented rebuttal
│   ├─ Rebuttal Conditions:
│   │   ├─ Entity must demonstrate 30 DPD not indicative of SICR
│   │   ├─ Evidence: Historical analysis showing no correlation between 30 DPD and default
│   │   └─ Rare: Most institutions accept 30 DPD as SICR trigger
│   ├─ Implementation:
│   │   ├─ System flags all exposures ≥30 DPD
│   │   ├─ Automatic Stage 1 → Stage 2 transfer
│   │   ├─ Exception: Manual rebuttal reviewed by credit risk team
│   │   └─ Audit trail: Document rebuttal rationale
│   └─ Cure (Stage 2 → Stage 1):
│       ├─ Payment brought current (0 DPD)
│       ├─ Probation period: 3-6 months current payments
│       └─ SICR no longer present → Revert to Stage 1
│
├─ Combined Approach (Best Practice):
│   ├─ Use Multiple Indicators (OR Logic):
│   │   ├─ SICR = (30 DPD) OR (Relative PD > 2×) OR (Absolute PD > 5%) OR (Rating Downgrade ≥2 notches) OR (Watchlist) OR (Forbearance)
│   │   └─ Rationale: Capture deterioration across multiple dimensions
│   ├─ Tiered Thresholds by Risk Segment:
│   │   ├─ Investment-Grade: Lower threshold (PD multiple 1.5×; 1 notch downgrade)
│   │   ├─ Sub-Investment-Grade: Higher threshold (PD multiple 3×; 2 notches)
│   │   └─ Rationale: Higher sensitivity for low-risk exposures (early warning)
│   ├─ Avoid Cliff Effects:
│   │   ├─ Use multiple indicators (smooth transition)
│   │   ├─ Probation periods for cures (avoid oscillation)
│   │   └─ Hysteresis: Higher threshold for cure than for SICR trigger
│   └─ Document SICR Framework:
│       ├─ Clear thresholds; rationale for each indicator
│       ├─ Segment-specific rules (product, geography, rating)
│       ├─ Governance: Approval by risk committee; annual review
│       └─ Audit trail: All SICR triggers logged; transfers documented
│
├─ Governance & Validation:
│   ├─ Model Validation:
│   │   ├─ Backtesting: Analyze historical SICR triggers vs actual defaults
│   │   │   ├─ True Positives: SICR → Default (correct early warning)
│   │   │   ├─ False Positives: SICR → No Default (premature Stage 2)
│   │   │   ├─ False Negatives: No SICR → Default (missed deterioration)
│   │   │   └─ True Negatives: No SICR → No Default
│   │   ├─ Metrics: Precision, Recall, F1-Score, AUC
│   │   └─ Target: High recall (catch deterioration); tolerate false positives
│   ├─ Threshold Calibration:
│   │   ├─ Analyze PD multiplier distribution for defaulted vs non-defaulted loans
│   │   ├─ ROC Curve: Plot true positive rate vs false positive rate for various thresholds
│   │   └─ Select threshold balancing timeliness (early warning) vs stability (avoid noise)
│   ├─ Management Overlays:
│   │   ├─ Expert judgment for events not captured by models (e.g., COVID-19)
│   │   ├─ Sector-level overlays (oil price collapse → all energy loans Stage 2)
│   │   └─ Document rationale; temporary (review quarterly)
│   └─ Audit & Regulatory Review:
│       ├─ External auditors assess SICR methodology; test sample of transfers
│       ├─ Regulators (ECB, PRA) review SICR framework; challenge thresholds
│       └─ Supervisory expectations: Timely SICR detection; avoid "too little too late"
│
└─ Practical Implementation:
    ├─ Systems Architecture:
    │   ├─ Data Warehouse: Origination PD, current PD, payment status, ratings
    │   ├─ SICR Engine: Apply quantitative + qualitative rules; flag SICR triggers
    │   ├─ Stage Classification: Determine Stage 1/2/3 based on SICR + default flags
    │   └─ Reporting: Monthly stage migration reports; trend analysis
    ├─ Monthly SICR Assessment:
    │   ├─ Extract: Current PD, DPD, ratings, watchlist status for all exposures
    │   ├─ Compare: Current metrics vs origination baseline
    │   ├─ Flag: Apply SICR rules; identify Stage 1 → Stage 2 candidates
    │   ├─ Review: Credit risk team validates flags; applies overrides if justified
    │   └─ Transfer: Update stage; recalculate ECL (12-month → lifetime)
    ├─ Cure Monitoring:
    │   ├─ Track Stage 2 exposures for improvement
    │   ├─ Criteria: DPD = 0; PD decline; rating upgrade; probation complete
    │   └─ Transfer: Stage 2 → Stage 1 (provision release)
    └─ Key Challenges:
        ├─ Data Quality: Origination PD often missing (legacy loans)
        ├─ Model Risk: PD models may overstate deterioration (false positives)
        ├─ Cliff Effects: Single metric triggers large ECL increase
        ├─ Subjectivity: Qualitative overlays require documentation; audit scrutiny
        └─ P&L Volatility: Frequent Stage 1 ↔ Stage 2 oscillations
```

**Key Insight:** SICR = trigger for Stage 1 → Stage 2 transfer; 30 DPD mandatory backstop (rebuttable); relative PD change (2×) common threshold; combine quantitative + qualitative indicators; avoid cliff effects; timely detection critical to avoid sudden losses.

## 5. Mini-Project
Simulate SICR detection using multiple indicators:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set seed
np.random.seed(42)

# Portfolio parameters
n_loans = 1000

# Origination data
df = pd.DataFrame({
    'loan_id': range(n_loans),
    'amount': np.random.uniform(50_000, 500_000, n_loans),
    'pd_orig': np.random.uniform(0.005, 0.02, n_loans),  # 0.5%-2% origination PD
    'rating_orig': np.random.choice([1, 2, 3, 4, 5], n_loans, p=[0.2, 0.3, 0.3, 0.15, 0.05]),  # 1=AAA, 5=BBB-
})

# Simulate 1 year forward: Credit deterioration
# Some loans deteriorate (PD increases), some stable, few improve
pd_shock = np.random.lognormal(0, 0.6, n_loans)  # Log-normal shocks (some extreme deterioration)
df['pd_current'] = (df['pd_orig'] * pd_shock).clip(0.001, 0.50)  # Clip to [0.1%, 50%]

# Rating migration (correlated with PD shock)
rating_change = np.random.choice([-2, -1, 0, 1], n_loans, p=[0.05, 0.15, 0.70, 0.10])  # Mostly stable
df['rating_current'] = (df['rating_orig'] + rating_change).clip(1, 10)  # Rating 1-10 (10=default)

# Days past due (DPD)
df['dpd'] = np.random.choice([0, 15, 35, 60], n_loans, p=[0.85, 0.08, 0.05, 0.02])

# Watchlist flag (subjective; correlated with high PD)
df['watchlist'] = (df['pd_current'] > 0.05) & (np.random.rand(n_loans) < 0.5)

# Ground truth: Actual default in next 12 months (for validation)
df['default_12m'] = np.random.binomial(1, df['pd_current'])

# SICR Indicators
# 1. 30 DPD backstop
df['sicr_30dpd'] = df['dpd'] >= 30

# 2. Relative PD change > 2×
df['pd_ratio'] = df['pd_current'] / df['pd_orig']
df['sicr_relative_pd'] = df['pd_ratio'] > 2.0

# 3. Absolute PD > 5%
df['sicr_absolute_pd'] = df['pd_current'] > 0.05

# 4. Rating downgrade ≥ 2 notches
df['rating_change'] = df['rating_current'] - df['rating_orig']
df['sicr_rating'] = df['rating_change'] >= 2

# 5. Watchlist
df['sicr_watchlist'] = df['watchlist']

# Combined SICR (OR logic)
df['sicr_combined'] = (
    df['sicr_30dpd'] |
    df['sicr_relative_pd'] |
    df['sicr_absolute_pd'] |
    df['sicr_rating'] |
    df['sicr_watchlist']
)

# Stage classification
df['stage'] = 1  # Default Stage 1
df.loc[df['sicr_combined'], 'stage'] = 2  # SICR → Stage 2
df.loc[df['dpd'] >= 90, 'stage'] = 3  # 90 DPD → Stage 3 (override)

# Analysis
print("="*70)
print("SICR Detection Framework: Multiple Indicators")
print("="*70)
print(f"Total Loans: {n_loans}")
print("")

# SICR trigger breakdown
print("SICR Triggers (OR Logic):")
print("-"*70)
sicr_summary = pd.DataFrame({
    'Indicator': ['30 DPD', 'Relative PD (>2×)', 'Absolute PD (>5%)', 'Rating Downgrade (≥2)', 'Watchlist', 'Combined (Any)'],
    'Count': [
        df['sicr_30dpd'].sum(),
        df['sicr_relative_pd'].sum(),
        df['sicr_absolute_pd'].sum(),
        df['sicr_rating'].sum(),
        df['sicr_watchlist'].sum(),
        df['sicr_combined'].sum()
    ],
    'Percent': [
        df['sicr_30dpd'].mean() * 100,
        df['sicr_relative_pd'].mean() * 100,
        df['sicr_absolute_pd'].mean() * 100,
        df['sicr_rating'].mean() * 100,
        df['sicr_watchlist'].mean() * 100,
        df['sicr_combined'].mean() * 100
    ]
})
print(sicr_summary.to_string(index=False))
print("")

# Stage distribution
stage_counts = df['stage'].value_counts().sort_index()
print("Stage Distribution:")
print("-"*70)
for stage in [1, 2, 3]:
    count = stage_counts.get(stage, 0)
    pct = count / n_loans * 100
    print(f"Stage {stage}: {count:4d} loans ({pct:5.1f}%)")
print("")

# Validation: SICR vs Actual Default
print("SICR Performance (Predicting 12-Month Default):")
print("-"*70)

cm = confusion_matrix(df['default_12m'], df['sicr_combined'])
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives (SICR → Default):       {tp:4d}")
print(f"False Positives (SICR → No Default):   {fp:4d}")
print(f"False Negatives (No SICR → Default):   {fn:4d}")
print(f"True Negatives (No SICR → No Default): {tn:4d}")
print("")
print(f"Precision (SICR → Default Rate):        {precision:.2%}")
print(f"Recall (Catch Default Rate):            {recall:.2%}")
print(f"F1-Score:                               {f1:.2f}")

# ROC Curve for PD ratio threshold
fpr, tpr, thresholds = roc_curve(df['default_12m'], df['pd_ratio'])
roc_auc = auc(fpr, tpr)

print(f"\nAUC (PD Ratio as Predictor):            {roc_auc:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SICR trigger counts
ax = axes[0, 0]
triggers = ['30 DPD', 'Rel PD', 'Abs PD', 'Rating', 'Watch', 'Combined']
counts = sicr_summary['Count'].values
ax.bar(triggers, counts, color=['red', 'orange', 'orange', 'blue', 'purple', 'green'], alpha=0.7)
ax.set_ylabel('Number of Loans')
ax.set_title('SICR Triggers (OR Logic)')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Stage distribution
ax = axes[0, 1]
stages = ['Stage 1\n(12m ECL)', 'Stage 2\n(Lifetime ECL)', 'Stage 3\n(Default)']
stage_values = [stage_counts.get(i, 0) for i in [1, 2, 3]]
colors = ['green', 'orange', 'red']
ax.bar(stages, stage_values, color=colors, alpha=0.7)
ax.set_ylabel('Number of Loans')
ax.set_title('Loan Distribution by Stage')
ax.grid(axis='y', alpha=0.3)

# Plot 3: PD distribution by SICR status
ax = axes[1, 0]
no_sicr = df[~df['sicr_combined']]['pd_current']
yes_sicr = df[df['sicr_combined']]['pd_current']

ax.hist(no_sicr, bins=30, alpha=0.5, label='No SICR', color='green')
ax.hist(yes_sicr, bins=30, alpha=0.5, label='SICR Triggered', color='red')
ax.set_xlabel('Current PD')
ax.set_ylabel('Frequency')
ax.set_title('PD Distribution: SICR vs No SICR')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: ROC Curve (PD ratio threshold optimization)
ax = axes[1, 1]
ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

# Mark current threshold (2×)
threshold_2x_idx = np.argmin(np.abs(thresholds - 2.0))
ax.plot(fpr[threshold_2x_idx], tpr[threshold_2x_idx], 'ro', markersize=10, label='Threshold = 2×')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curve: PD Ratio Threshold Optimization')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('significant_increase_credit_risk.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Combined SICR (OR logic) captures ~15-30% of portfolio")
print("   → Multiple indicators avoid single-metric dependence")
print("")
print("2. 30 DPD backstop critical (high true positive rate)")
print("   → Catches payment distress early; mandatory IFRS 9 requirement")
print("")
print("3. False positives tolerable (SICR without default)")
print("   → Prudent provisioning; cure back to Stage 1 if improves")
print("")
print("4. Threshold calibration critical (ROC analysis)")
print("   → Balance timeliness (high recall) vs stability (low false positives)")
```

## 6. Challenge Round
When SICR frameworks fail or introduce complexity:
- **Cliff Effects**: Loan PD = 1.9× (Stage 1) vs 2.1× (Stage 2) → 5 bps difference causes massive ECL jump; solution: Use multiple indicators (smooth transition); hysteresis (different cure threshold)
- **Oscillation (Cure/Redeteriorate)**: Loan crosses 30 DPD → Stage 2 → Cures → Stage 1 → Defaults 32 DPD again → Stage 2; P&L volatility; solution: Probation period (3-6 months current before cure); sustained improvement required
- **Model Risk (PD Overstatement)**: PD model overstates deterioration → Excessive SICR triggers; solution: Backtesting; compare modeled PD to actual default rates; management overlay to dampen noise
- **Legacy Loans (Missing Origination Data)**: Loan originated pre-IFRS 9; no origination PD recorded → Cannot calculate relative PD change; solution: Use earliest available PD as proxy; or rely on absolute PD threshold + qualitative indicators
- **Low Default Portfolios (Investment-Grade)**: Origination PD = 0.1%; current PD = 0.3% (3× increase) → SICR triggered; but both extremely low; solution: Use absolute PD floor (ignore SICR if current PD < 0.5%); or segment-specific thresholds
- **Subjectivity (Watchlist)**: Relationship manager flags vary by individual (inconsistent); solution: Documented criteria for watchlist; centralized credit risk review; avoid excessive subjectivity

## 7. Key References
- [IFRS 9 Financial Instruments (Section 5.5)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official standard; SICR requirements; 30 DPD rebuttable presumption; assessment principles
- [EBA Guidelines on Accounting for Expected Credit Losses (2017)](https://www.eba.europa.eu/regulation-and-policy/single-rulebook/interactive-single-rulebook/503) - European Banking Authority implementation guidance; SICR thresholds; supervisory expectations
- [Deloitte: Significant Increase in Credit Risk (2018)](https://www2.deloitte.com/content/dam/Deloitte/global/Documents/Financial-Services/gx-fsi-ifrs9-sicr-practical-considerations.pdf) - Practical implementation; indicator design; threshold calibration; industry practices

---
**Status:** IFRS 9 Core Concept | **Complements:** Three-Stage Approach, Expected Credit Loss Models, Forward-Looking Information
