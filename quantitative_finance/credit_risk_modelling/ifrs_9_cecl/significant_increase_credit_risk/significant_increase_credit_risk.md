# Significant Increase in Credit Risk (SICR)

## 1. Concept Skeleton
**Definition:** Threshold determining transfer from Stage 1 (12-month ECL) to Stage 2 (lifetime ECL); requires assessment whether credit risk has increased significantly since initial recognition; triggers higher provisioning  
**Purpose:** Timely recognition of credit deterioration; avoid delayed loss recognition; balance false positives (premature Stage 2) vs false negatives (missed deterioration); align with early warning indicators  
**Prerequisites:** Default probability (PD) estimation, credit rating migration, delinquency status, qualitative risk indicators, origination data, macroeconomic overlays

## 2. Comparative Framing
| SICR Indicator | Type | Threshold Example | Data Requirement | Timeliness | False Positive Risk |
|----------------|------|------------------|------------------|------------|---------------------|
| **30 DPD (Days Past Due)** | Quantitative (mandatory backstop) | â‰¥30 days overdue | Payment history | High (immediate) | Low (observable default signal) |
| **Relative PD Change** | Quantitative | PD(current) / PD(orig) > 2Ã— | Credit models | Medium (monthly) | Medium (model sensitivity) |
| **Absolute PD Threshold** | Quantitative | PD > 5% | Credit models | Medium (monthly) | Medium (arbitrary cutoff) |
| **Rating Downgrade** | Qualitative/Quantitative | 2+ notch downgrade | External/internal ratings | Medium-High | Low (validated ratings) |
| **Watchlist / Forbearance** | Qualitative | Added to watchlist | Relationship manager flags | High | High (subjective) |
| **Economic Overlay** | Quantitative | Sector in distress (e.g., oil price collapse) | Macro models | Low (lagging) | High (broad brush) |

## 3. Examples + Counterexamples

**SICR Trigger (30 DPD Backstop):**  
Corporate loan $1M; 32 days past due at month-end. Mandatory SICR presumption â†’ Transfer Stage 1â†’2 â†’ Lifetime ECL increases from $5k to $40k.

**Relative PD Increase:**  
Loan originated at BBB rating (PD = 0.5%); downgrades to BB (PD = 2.0%). Relative increase: 2.0% / 0.5% = 4Ã— > 2Ã— threshold â†’ SICR triggered.

**False Positive (Temporary Delay):**  
Borrower 35 DPD due to administrative error (payment processing delay); fundamentally sound. SICR triggered â†’ Stage 2 â†’ Large ECL increase. Cures next month â†’ Revert to Stage 1 (provision reversal; P&L volatility).

**False Negative (Missed SICR):**  
Investment-grade bond (AAA); credit fundamentals deteriorate (leverage increases, cash flow declines) but no rating downgrade yet (rating agencies lag). SICR not triggered â†’ Remains Stage 1 â†’ Insufficient provisioning. Default 6 months later â†’ Sudden large loss.

**Qualitative Overlay:**  
Oil & gas sector loan portfolio; oil price collapses 50%. Individual loan PD increases modest (1.2Ã— origination) but sector distress warrants SICR â†’ Management overlay transfers entire sector to Stage 2 (prudent provisioning).

## 4. Layer Breakdown
```
Significant Increase in Credit Risk (SICR) Framework:

â”œâ”€ IFRS 9 Requirements:
â”‚   â”œâ”€ Principle-Based: No mechanical formula; requires judgment
â”‚   â”œâ”€ Assessment at Each Reporting Date: Monthly or quarterly
â”‚   â”œâ”€ Compare Current Credit Risk to Origination Risk:
â”‚   â”‚   â”œâ”€ Baseline: Credit risk at initial recognition (Day 1)
â”‚   â”‚   â”œâ”€ Current: Credit risk at reporting date
â”‚   â”‚   â””â”€ Test: Has credit risk increased significantly?
â”‚   â”œâ”€ Both Quantitative and Qualitative Indicators:
â”‚   â”‚   â””â”€ Cannot rely on single metric; holistic assessment
â”‚   â””â”€ Forward-Looking: Incorporate reasonable and supportable information
â”‚
â”œâ”€ Quantitative Indicators:
â”‚   â”œâ”€ 30 Days Past Due (Rebuttable Presumption):
â”‚   â”‚   â”œâ”€ IFRS 9.5.5.11: "Presumption that SICR when >30 DPD"
â”‚   â”‚   â”œâ”€ Backstop: Mandatory unless rebutted with evidence
â”‚   â”‚   â”œâ”€ Rationale: Payment delay signals liquidity stress
â”‚   â”‚   â”œâ”€ Rebuttal Criteria:
â”‚   â”‚   â”‚   â”œâ”€ Administrative error (documented; resolved quickly)
â”‚   â”‚   â”‚   â”œâ”€ Dispute (valid; payment made after resolution)
â”‚   â”‚   â”‚   â””â”€ Isolated incident (strong payment history; one-off event)
â”‚   â”‚   â””â”€ Implementation: Flag all exposures â‰¥30 DPD; review for rebuttal
â”‚   â”‚
â”‚   â”œâ”€ Relative PD Change:
â”‚   â”‚   â”œâ”€ Formula: PD(current) / PD(origination) > threshold
â”‚   â”‚   â”œâ”€ Threshold Examples:
â”‚   â”‚   â”‚   â”œâ”€ Conservative: 2Ã— (doubling of PD triggers SICR)
â”‚   â”‚   â”‚   â”œâ”€ Moderate: 3Ã— (triple PD)
â”‚   â”‚   â”‚   â””â”€ Investment-Grade: 1.5Ã— (lower tolerance for IG)
â”‚   â”‚   â”œâ”€ Rationale: Relative deterioration captures credit migration
â”‚   â”‚   â”œâ”€ Advantages:
â”‚   â”‚   â”‚   â”œâ”€ Scale-invariant: Works for low PD (0.1% â†’ 0.2% = SICR) and high PD (5% â†’ 10%)
â”‚   â”‚   â”‚   â””â”€ Aligns with rating migration (notch downgrades correlate with PD multiples)
â”‚   â”‚   â”œâ”€ Challenges:
â”‚   â”‚   â”‚   â”œâ”€ Noisy for very low PD (0.01% â†’ 0.03% = 3Ã—; but both negligible)
â”‚   â”‚   â”‚   â”œâ”€ Requires robust origination PD (poor underwriting â†’ wrong baseline)
â”‚   â”‚   â”‚   â””â”€ Model risk: PD estimation errors amplified
â”‚   â”‚   â””â”€ Implementation: Calculate monthly PD; compare to origination PD; flag if ratio > threshold
â”‚   â”‚
â”‚   â”œâ”€ Absolute PD Threshold:
â”‚   â”‚   â”œâ”€ Formula: PD(current) > X% (e.g., 5%)
â”‚   â”‚   â”œâ”€ Rationale: High absolute PD = elevated default risk (Stage 2 appropriate)
â”‚   â”‚   â”œâ”€ Threshold Selection:
â”‚   â”‚   â”‚   â”œâ”€ Conservative: 3% (sub-investment grade)
â”‚   â”‚   â”‚   â”œâ”€ Moderate: 5% (CCC territory)
â”‚   â”‚   â”‚   â””â”€ Aggressive: 10% (near-default)
â”‚   â”‚   â”œâ”€ Advantages: Simple; independent of origination
â”‚   â”‚   â”œâ”€ Challenges:
â”‚   â”‚   â”‚   â”œâ”€ Cliff effect: PD = 4.9% (Stage 1) vs PD = 5.1% (Stage 2) â†’ Large ECL jump
â”‚   â”‚   â”‚   â”œâ”€ Ignores relative deterioration: IG loan 0.5% â†’ 3% (not SICR if threshold 5%)
â”‚   â”‚   â”‚   â””â”€ Arbitrary cutoff
â”‚   â”‚   â””â”€ Often used as complement to relative PD (OR condition: relative OR absolute)
â”‚   â”‚
â”‚   â”œâ”€ Lifetime PD Change:
â”‚   â”‚   â”œâ”€ Similar to 12-month PD but uses lifetime PD comparison
â”‚   â”‚   â”œâ”€ More forward-looking; captures long-term deterioration
â”‚   â”‚   â””â”€ Computationally intensive (full term structure)
â”‚   â”‚
â”‚   â””â”€ Credit Rating Downgrade:
â”‚       â”œâ”€ External Ratings (Moody's, S&P, Fitch):
â”‚       â”‚   â”œâ”€ Threshold: 2+ notch downgrade (e.g., BBB â†’ BB; A â†’ BBB-)
â”‚       â”‚   â”œâ”€ Rationale: Rating agencies incorporate credit fundamentals
â”‚       â”‚   â”œâ”€ Advantages: Independent validation; widely understood
â”‚       â”‚   â””â”€ Challenges: Lagging (agencies slow to downgrade); "too little too late"
â”‚       â”œâ”€ Internal Ratings:
â”‚       â”‚   â”œâ”€ Bank-specific rating models; updated more frequently
â”‚       â”‚   â””â”€ Aligned with PD (internal rating scale maps to PD)
â”‚       â””â”€ Implementation: Map rating changes to SICR (2 notches = SICR)
â”‚
â”œâ”€ Qualitative Indicators:
â”‚   â”œâ”€ Forbearance / Restructuring:
â”‚   â”‚   â”œâ”€ Definition: Concessions granted due to borrower financial difficulty
â”‚   â”‚   â”œâ”€ Examples: Payment holiday, term extension, interest rate reduction, covenant waiver
â”‚   â”‚   â”œâ”€ SICR Implication: Forbearance signals distress â†’ Automatic Stage 2
â”‚   â”‚   â”œâ”€ Cure: Probation period (6-12 months satisfactory performance) before revert to Stage 1
â”‚   â”‚   â””â”€ EBA Guidelines: Forborne exposures remain Stage 2 minimum 12 months
â”‚   â”‚
â”‚   â”œâ”€ Watchlist / Credit Watch:
â”‚   â”‚   â”œâ”€ Relationship manager flags borrower for heightened monitoring
â”‚   â”‚   â”œâ”€ Triggers: Covenant breach, adverse news, management changes, litigation
â”‚   â”‚   â”œâ”€ SICR Implication: Watchlist addition â†’ Presume SICR (unless rebutted)
â”‚   â”‚   â””â”€ Challenge: Subjectivity; consistency across portfolio
â”‚   â”‚
â”‚   â”œâ”€ Adverse Business Conditions:
â”‚   â”‚   â”œâ”€ Borrower-Specific: Loss of major customer, regulatory action, failed product launch
â”‚   â”‚   â”œâ”€ Sector-Specific: Oil price collapse (energy), pandemic (airlines), regulatory change
â”‚   â”‚   â””â”€ SICR Implication: Material adverse change â†’ SICR assessment; possible management overlay
â”‚   â”‚
â”‚   â”œâ”€ Collateral Deterioration:
â”‚   â”‚   â”œâ”€ Loan-to-Value (LTV) > 100% (negative equity)
â”‚   â”‚   â”œâ”€ Real estate value decline (market downturn)
â”‚   â”‚   â””â”€ SICR Implication: Unsecured exposure â†’ Higher risk â†’ SICR triggered
â”‚   â”‚
â”‚   â”œâ”€ Macroeconomic Overlays:
â”‚   â”‚   â”œâ”€ Sector-level distress (e.g., COVID-19 impact on travel/hospitality)
â”‚   â”‚   â”œâ”€ Geographic stress (e.g., regional recession)
â”‚   â”‚   â””â”€ SICR Implication: Transfer entire segment to Stage 2 (management judgment)
â”‚   â”‚
â”‚   â””â”€ Covenant Breaches:
â”‚       â”œâ”€ Debt service coverage ratio (DSCR) < threshold
â”‚       â”œâ”€ Leverage ratio exceeds maximum
â”‚       â””â”€ SICR Implication: Breach signals deterioration â†’ SICR
â”‚
â”œâ”€ 30 DPD Backstop (Rebuttable Presumption):
â”‚   â”œâ”€ Mandatory IFRS 9 Requirement:
â”‚   â”‚   â”œâ”€ IFRS 9.5.5.11: "Rebuttable presumption that SICR when >30 DPD"
â”‚   â”‚   â””â”€ Cannot be waived without documented rebuttal
â”‚   â”œâ”€ Rebuttal Conditions:
â”‚   â”‚   â”œâ”€ Entity must demonstrate 30 DPD not indicative of SICR
â”‚   â”‚   â”œâ”€ Evidence: Historical analysis showing no correlation between 30 DPD and default
â”‚   â”‚   â””â”€ Rare: Most institutions accept 30 DPD as SICR trigger
â”‚   â”œâ”€ Implementation:
â”‚   â”‚   â”œâ”€ System flags all exposures â‰¥30 DPD
â”‚   â”‚   â”œâ”€ Automatic Stage 1 â†’ Stage 2 transfer
â”‚   â”‚   â”œâ”€ Exception: Manual rebuttal reviewed by credit risk team
â”‚   â”‚   â””â”€ Audit trail: Document rebuttal rationale
â”‚   â””â”€ Cure (Stage 2 â†’ Stage 1):
â”‚       â”œâ”€ Payment brought current (0 DPD)
â”‚       â”œâ”€ Probation period: 3-6 months current payments
â”‚       â””â”€ SICR no longer present â†’ Revert to Stage 1
â”‚
â”œâ”€ Combined Approach (Best Practice):
â”‚   â”œâ”€ Use Multiple Indicators (OR Logic):
â”‚   â”‚   â”œâ”€ SICR = (30 DPD) OR (Relative PD > 2Ã—) OR (Absolute PD > 5%) OR (Rating Downgrade â‰¥2 notches) OR (Watchlist) OR (Forbearance)
â”‚   â”‚   â””â”€ Rationale: Capture deterioration across multiple dimensions
â”‚   â”œâ”€ Tiered Thresholds by Risk Segment:
â”‚   â”‚   â”œâ”€ Investment-Grade: Lower threshold (PD multiple 1.5Ã—; 1 notch downgrade)
â”‚   â”‚   â”œâ”€ Sub-Investment-Grade: Higher threshold (PD multiple 3Ã—; 2 notches)
â”‚   â”‚   â””â”€ Rationale: Higher sensitivity for low-risk exposures (early warning)
â”‚   â”œâ”€ Avoid Cliff Effects:
â”‚   â”‚   â”œâ”€ Use multiple indicators (smooth transition)
â”‚   â”‚   â”œâ”€ Probation periods for cures (avoid oscillation)
â”‚   â”‚   â””â”€ Hysteresis: Higher threshold for cure than for SICR trigger
â”‚   â””â”€ Document SICR Framework:
â”‚       â”œâ”€ Clear thresholds; rationale for each indicator
â”‚       â”œâ”€ Segment-specific rules (product, geography, rating)
â”‚       â”œâ”€ Governance: Approval by risk committee; annual review
â”‚       â””â”€ Audit trail: All SICR triggers logged; transfers documented
â”‚
â”œâ”€ Governance & Validation:
â”‚   â”œâ”€ Model Validation:
â”‚   â”‚   â”œâ”€ Backtesting: Analyze historical SICR triggers vs actual defaults
â”‚   â”‚   â”‚   â”œâ”€ True Positives: SICR â†’ Default (correct early warning)
â”‚   â”‚   â”‚   â”œâ”€ False Positives: SICR â†’ No Default (premature Stage 2)
â”‚   â”‚   â”‚   â”œâ”€ False Negatives: No SICR â†’ Default (missed deterioration)
â”‚   â”‚   â”‚   â””â”€ True Negatives: No SICR â†’ No Default
â”‚   â”‚   â”œâ”€ Metrics: Precision, Recall, F1-Score, AUC
â”‚   â”‚   â””â”€ Target: High recall (catch deterioration); tolerate false positives
â”‚   â”œâ”€ Threshold Calibration:
â”‚   â”‚   â”œâ”€ Analyze PD multiplier distribution for defaulted vs non-defaulted loans
â”‚   â”‚   â”œâ”€ ROC Curve: Plot true positive rate vs false positive rate for various thresholds
â”‚   â”‚   â””â”€ Select threshold balancing timeliness (early warning) vs stability (avoid noise)
â”‚   â”œâ”€ Management Overlays:
â”‚   â”‚   â”œâ”€ Expert judgment for events not captured by models (e.g., COVID-19)
â”‚   â”‚   â”œâ”€ Sector-level overlays (oil price collapse â†’ all energy loans Stage 2)
â”‚   â”‚   â””â”€ Document rationale; temporary (review quarterly)
â”‚   â””â”€ Audit & Regulatory Review:
â”‚       â”œâ”€ External auditors assess SICR methodology; test sample of transfers
â”‚       â”œâ”€ Regulators (ECB, PRA) review SICR framework; challenge thresholds
â”‚       â””â”€ Supervisory expectations: Timely SICR detection; avoid "too little too late"
â”‚
â””â”€ Practical Implementation:
    â”œâ”€ Systems Architecture:
    â”‚   â”œâ”€ Data Warehouse: Origination PD, current PD, payment status, ratings
    â”‚   â”œâ”€ SICR Engine: Apply quantitative + qualitative rules; flag SICR triggers
    â”‚   â”œâ”€ Stage Classification: Determine Stage 1/2/3 based on SICR + default flags
    â”‚   â””â”€ Reporting: Monthly stage migration reports; trend analysis
    â”œâ”€ Monthly SICR Assessment:
    â”‚   â”œâ”€ Extract: Current PD, DPD, ratings, watchlist status for all exposures
    â”‚   â”œâ”€ Compare: Current metrics vs origination baseline
    â”‚   â”œâ”€ Flag: Apply SICR rules; identify Stage 1 â†’ Stage 2 candidates
    â”‚   â”œâ”€ Review: Credit risk team validates flags; applies overrides if justified
    â”‚   â””â”€ Transfer: Update stage; recalculate ECL (12-month â†’ lifetime)
    â”œâ”€ Cure Monitoring:
    â”‚   â”œâ”€ Track Stage 2 exposures for improvement
    â”‚   â”œâ”€ Criteria: DPD = 0; PD decline; rating upgrade; probation complete
    â”‚   â””â”€ Transfer: Stage 2 â†’ Stage 1 (provision release)
    â””â”€ Key Challenges:
        â”œâ”€ Data Quality: Origination PD often missing (legacy loans)
        â”œâ”€ Model Risk: PD models may overstate deterioration (false positives)
        â”œâ”€ Cliff Effects: Single metric triggers large ECL increase
        â”œâ”€ Subjectivity: Qualitative overlays require documentation; audit scrutiny
        â””â”€ P&L Volatility: Frequent Stage 1 â†” Stage 2 oscillations
```

**Key Insight:** SICR = trigger for Stage 1 â†’ Stage 2 transfer; 30 DPD mandatory backstop (rebuttable); relative PD change (2Ã—) common threshold; combine quantitative + qualitative indicators; avoid cliff effects; timely detection critical to avoid sudden losses.

## 5. Challenge Round
When SICR frameworks fail or introduce complexity:
- **Cliff Effects**: Loan PD = 1.9Ã— (Stage 1) vs 2.1Ã— (Stage 2) â†’ 5 bps difference causes massive ECL jump; solution: Use multiple indicators (smooth transition); hysteresis (different cure threshold)
- **Oscillation (Cure/Redeteriorate)**: Loan crosses 30 DPD â†’ Stage 2 â†’ Cures â†’ Stage 1 â†’ Defaults 32 DPD again â†’ Stage 2; P&L volatility; solution: Probation period (3-6 months current before cure); sustained improvement required
- **Model Risk (PD Overstatement)**: PD model overstates deterioration â†’ Excessive SICR triggers; solution: Backtesting; compare modeled PD to actual default rates; management overlay to dampen noise
- **Legacy Loans (Missing Origination Data)**: Loan originated pre-IFRS 9; no origination PD recorded â†’ Cannot calculate relative PD change; solution: Use earliest available PD as proxy; or rely on absolute PD threshold + qualitative indicators
- **Low Default Portfolios (Investment-Grade)**: Origination PD = 0.1%; current PD = 0.3% (3Ã— increase) â†’ SICR triggered; but both extremely low; solution: Use absolute PD floor (ignore SICR if current PD < 0.5%); or segment-specific thresholds
- **Subjectivity (Watchlist)**: Relationship manager flags vary by individual (inconsistent); solution: Documented criteria for watchlist; centralized credit risk review; avoid excessive subjectivity

## 6. Key References
- [IFRS 9 Financial Instruments (Section 5.5)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official standard; SICR requirements; 30 DPD rebuttable presumption; assessment principles
- [EBA Guidelines on Accounting for Expected Credit Losses (2017)](https://www.eba.europa.eu/regulation-and-policy/single-rulebook/interactive-single-rulebook/503) - European Banking Authority implementation guidance; SICR thresholds; supervisory expectations
- [Deloitte: Significant Increase in Credit Risk (2018)](https://www2.deloitte.com/content/dam/Deloitte/global/Documents/Financial-Services/gx-fsi-ifrs9-sicr-practical-considerations.pdf) - Practical implementation; indicator design; threshold calibration; industry practices

---
**Status:** IFRS 9 Core Concept | **Complements:** Three-Stage Approach, Expected Credit Loss Models, Forward-Looking Information
