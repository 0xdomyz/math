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
| **Interest Revenue** | Gross carrying amount Ã— EIR | Gross carrying amount Ã— EIR | Net carrying amount Ã— EIR |
| **Transfer Trigger** | None (initial classification) | SICR detected | Default occurred |

## 3. Examples + Counterexamples

**Short Maturity (1-Year Loan):**  
12-month ECL â‰ˆ Lifetime ECL (both cover same period). $100k loan, PD = 2%, LGD = 40%. Stage 1 ECL = Stage 2 ECL = $100k Ã— 2% Ã— 40% = $800.

**Long Maturity (10-Year Corporate Loan):**  
Stage 1: 12-month PD = 1%, ECL = $100k Ã— 1% Ã— 40% = $400. Stage 2: Lifetime PD = 15% (cumulative over 10 years), ECL = $100k Ã— 15% Ã— 40% = $6,000 (15Ã— higher).

**Stage 1â†’2 Transfer Impact:**  
$10M loan portfolio, average 5-year maturity. Stage 1: 12-month ECL = $50k. SICR triggered â†’ Stage 2: Lifetime ECL = $400k. Provision increase = $350k (7Ã— jump).

**Amortizing Loan (Mortgage):**  
$200k mortgage, 30-year amortization. Outstanding balance declines over time â†’ EAD reduces â†’ Lifetime ECL lower than non-amortizing. Year 1 EAD = $200k; Year 15 EAD = $100k; Year 30 EAD = $0.

**Edge Case (Revolving Credit):**  
Credit card $10k limit; current balance $2k. EAD = $2k + $8k Ã— CCF (credit conversion factor). Lifetime ECL accounts for potential drawdown (CCF = 20-50% typically; stressed = 100%).

## 4. Layer Breakdown
```
Lifetime vs 12-Month ECL Framework:

â”œâ”€ 12-Month ECL (Stage 1):
â”‚   â”œâ”€ Definition:
â”‚   â”‚   â””â”€ IFRS 9.5.5.5: "ECL from default events possible within 12 months after reporting date"
â”‚   â”œâ”€ Rationale:
â”‚   â”‚   â”œâ”€ Low credit risk â†’ Short-horizon focus
â”‚   â”‚   â”œâ”€ Avoids excessive provisioning for performing loans
â”‚   â”‚   â””â”€ Aligns with Basel PD (12-month regulatory PD)
â”‚   â”œâ”€ Calculation:
â”‚   â”‚   â”œâ”€ Formula: ECL = EAD Ã— PD(12m) Ã— LGD
â”‚   â”‚   â”œâ”€ PD(12m): Probability of default in next 12 months
â”‚   â”‚   â”‚   â”œâ”€ Point-in-Time (PIT): Adjusted for current economic conditions
â”‚   â”‚   â”‚   â””â”€ Forward-looking: Incorporate macro scenarios for next year
â”‚   â”‚   â”œâ”€ LGD: Loss given default (typically 30-60% for secured; 70-90% unsecured)
â”‚   â”‚   â””â”€ EAD: Current outstanding balance + accrued interest
â”‚   â”œâ”€ Discounting:
â”‚   â”‚   â”œâ”€ Typically NOT discounted (materiality; 12-month horizon short)
â”‚   â”‚   â””â”€ If discounted: Use effective interest rate (EIR)
â”‚   â”œâ”€ Maturity Insensitivity:
â”‚   â”‚   â”œâ”€ 12-month ECL same for 1-year, 5-year, or 30-year loan (if same PD)
â”‚   â”‚   â””â”€ Rationale: Horizon capped at 12 months regardless of maturity
â”‚   â”œâ”€ Example:
â”‚   â”‚   â”œâ”€ Loan: $1M, 5-year maturity, 12-month PD = 0.5%, LGD = 40%
â”‚   â”‚   â””â”€ ECL = $1M Ã— 0.5% Ã— 40% = $2,000
â”‚   â””â”€ Coverage Ratio:
â”‚       â”œâ”€ Typically 0.1%-1% of exposure (investment-grade)
â”‚       â””â”€ Higher for sub-investment-grade (1%-2%)
â”‚
â”œâ”€ Lifetime ECL (Stage 2 & 3):
â”‚   â”œâ”€ Definition:
â”‚   â”‚   â””â”€ IFRS 9.5.5.3: "ECL from all possible default events over expected life of instrument"
â”‚   â”œâ”€ Rationale:
â”‚   â”‚   â”œâ”€ SICR or default â†’ Heightened risk â†’ Full lifetime provisioning
â”‚   â”‚   â””â”€ Timely loss recognition (avoid "too little too late")
â”‚   â”œâ”€ Calculation (General):
â”‚   â”‚   â”œâ”€ Formula: ECL = âˆ‘[t=1 to T] { EAD(t) Ã— PD(t) Ã— LGD(t) Ã— DF(t) }
â”‚   â”‚   â”œâ”€ T: Remaining contractual maturity (years or months)
â”‚   â”‚   â”œâ”€ PD(t): Marginal probability of default at time t (conditional on survival)
â”‚   â”‚   â”œâ”€ LGD(t): Loss given default (may vary by scenario)
â”‚   â”‚   â”œâ”€ EAD(t): Exposure at time t (amortization, prepayments, drawdowns)
â”‚   â”‚   â””â”€ DF(t): Discount factor = 1 / (1 + EIR)^t
â”‚   â”‚
â”‚   â”œâ”€ PD Term Structure:
â”‚   â”‚   â”œâ”€ Marginal PD(t): Default probability in period t given survival to t-1
â”‚   â”‚   â”‚   â””â”€ Hazard rate Î»(t): Instantaneous default intensity
â”‚   â”‚   â”œâ”€ Survival Probability:
â”‚   â”‚   â”‚   â””â”€ S(t) = âˆ[Ï„=1 to t] (1 - PD(Ï„)) = exp(-âˆ«Î»(Ï„)dÏ„)
â”‚   â”‚   â”œâ”€ Cumulative PD:
â”‚   â”‚   â”‚   â””â”€ CPD(t) = 1 - S(t)
â”‚   â”‚   â”œâ”€ Term Structure Shapes:
â”‚   â”‚   â”‚   â”œâ”€ Flat: PD constant over time (simplest assumption)
â”‚   â”‚   â”‚   â”œâ”€ Increasing: PD rises with maturity (credit deterioration over time)
â”‚   â”‚   â”‚   â”œâ”€ Hump-Shaped: PD peaks mid-term (default risk highest Year 2-3)
â”‚   â”‚   â”‚   â””â”€ Reversion: Explicit forecast Years 1-3; revert to TTC thereafter
â”‚   â”‚   â””â”€ Example:
â”‚   â”‚       â”œâ”€ Flat: PD = 2% per year â†’ Lifetime PD (5 years) = 1 - (0.98)^5 = 9.6%
â”‚   â”‚       â””â”€ Increasing: PD = [1%, 1.5%, 2%, 2.5%, 3%] â†’ CPD = 9.5%
â”‚   â”‚
â”‚   â”œâ”€ EAD Term Structure (Amortization):
â”‚   â”‚   â”œâ”€ Amortizing Loans (Mortgages, Auto):
â”‚   â”‚   â”‚   â”œâ”€ Outstanding principal declines over time
â”‚   â”‚   â”‚   â”œâ”€ EAD(t) = Outstanding(t) from amortization schedule
â”‚   â”‚   â”‚   â””â”€ Example: $100k 10-year loan; Year 5 EAD = $50k
â”‚   â”‚   â”œâ”€ Bullet Loans (Corporate):
â”‚   â”‚   â”‚   â”œâ”€ Principal repaid at maturity
â”‚   â”‚   â”‚   â””â”€ EAD(t) = Constant (no amortization)
â”‚   â”‚   â”œâ”€ Revolving Credit (Credit Cards, Lines of Credit):
â”‚   â”‚   â”‚   â”œâ”€ EAD uncertain (future drawdowns)
â”‚   â”‚   â”‚   â”œâ”€ EAD = Drawn + Undrawn Ã— CCF
â”‚   â”‚   â”‚   â””â”€ CCF (Credit Conversion Factor): 20-50% (stressed = 100%)
â”‚   â”‚   â””â”€ Prepayments:
â”‚   â”‚       â”œâ”€ Mortgages: Voluntary prepayments reduce EAD
â”‚   â”‚       â””â”€ Model: Constant prepayment rate (CPR) or conditional (CPR varies with rates)
â”‚   â”‚
â”‚   â”œâ”€ Discounting:
â”‚   â”‚   â”œâ”€ Mandatory for Lifetime ECL (material impact over long horizons)
â”‚   â”‚   â”œâ”€ Discount Rate: Effective Interest Rate (EIR)
â”‚   â”‚   â”‚   â”œâ”€ Definition: Rate that discounts future cash flows to amortized cost
â”‚   â”‚   â”‚   â””â”€ Includes: Origination fees, transaction costs (not credit risk premium)
â”‚   â”‚   â”œâ”€ Example:
â”‚   â”‚   â”‚   â”œâ”€ Loss $10k in Year 5, EIR = 5%
â”‚   â”‚   â”‚   â””â”€ PV = $10k / (1.05)^5 = $7,835 (22% discount)
â”‚   â”‚   â””â”€ Stage 3 Debate:
â”‚   â”‚       â”œâ”€ IFRS 9 allows original EIR or credit-adjusted rate
â”‚   â”‚       â””â”€ Original EIR common (simpler; consistent with Stage 2)
â”‚   â”‚
â”‚   â”œâ”€ Example (5-Year Corporate Loan):
â”‚   â”‚   â”œâ”€ Loan: $1M, 5-year bullet, LGD = 40%, EIR = 5%
â”‚   â”‚   â”œâ”€ PD term structure (annual marginal): [1%, 1.5%, 2%, 2.5%, 3%]
â”‚   â”‚   â”œâ”€ Survival probabilities: [99%, 97.5%, 95.6%, 93.2%, 90.4%]
â”‚   â”‚   â”œâ”€ Year-by-year ECL:
â”‚   â”‚   â”‚   â”œâ”€ Year 1: $1M Ã— 1% Ã— 40% Ã— 0.9524 = $3,810
â”‚   â”‚   â”‚   â”œâ”€ Year 2: $1M Ã— 1.5% Ã— 40% Ã— 0.9070 = $5,442
â”‚   â”‚   â”‚   â”œâ”€ Year 3: $1M Ã— 2% Ã— 40% Ã— 0.8638 = $6,910
â”‚   â”‚   â”‚   â”œâ”€ Year 4: $1M Ã— 2.5% Ã— 40% Ã— 0.8227 = $8,227
â”‚   â”‚   â”‚   â””â”€ Year 5: $1M Ã— 3% Ã— 40% Ã— 0.7835 = $9,402
â”‚   â”‚   â””â”€ Total Lifetime ECL = $33,791 (vs 12-month ECL = $4,000; 8.4Ã— higher)
â”‚   â”‚
â”‚   â””â”€ Coverage Ratio:
â”‚       â”œâ”€ Stage 2: Typically 3%-15% of exposure
â”‚       â”œâ”€ Stage 3: Typically 30%-90% (LGD-driven; PD â‰ˆ 100%)
â”‚       â””â”€ Higher for longer maturities (more time for default; cumulative PD higher)
â”‚
â”œâ”€ Maturity Impact on Lifetime ECL:
â”‚   â”œâ”€ Short Maturity (< 2 years):
â”‚   â”‚   â”œâ”€ Lifetime ECL â‰ˆ 12-month ECL (horizons similar)
â”‚   â”‚   â””â”€ Stage 1â†’2 transfer: Modest ECL increase (1.5-2Ã—)
â”‚   â”œâ”€ Medium Maturity (2-5 years):
â”‚   â”‚   â”œâ”€ Lifetime ECL 3-8Ã— higher than 12-month ECL
â”‚   â”‚   â””â”€ Stage 1â†’2 transfer: Significant provision impact
â”‚   â”œâ”€ Long Maturity (> 10 years):
â”‚   â”‚   â”œâ”€ Lifetime ECL 10-20Ã— higher than 12-month ECL (if flat PD)
â”‚   â”‚   â”œâ”€ Discounting reduces impact (long-dated losses heavily discounted)
â”‚   â”‚   â””â”€ Reversion to TTC: Mitigates extreme long-term forecasts
â”‚   â””â”€ Example (Fixed $1M Loan, PD = 2%/year, LGD = 40%, EIR = 5%):
â”‚       â”œâ”€ 12-month ECL: $8,000 (constant)
â”‚       â”œâ”€ Lifetime ECL (2-year): $15,200 (1.9Ã— 12m)
â”‚       â”œâ”€ Lifetime ECL (5-year): $34,000 (4.3Ã— 12m)
â”‚       â”œâ”€ Lifetime ECL (10-year): $58,000 (7.3Ã— 12m)
â”‚       â””â”€ Lifetime ECL (30-year): $110,000 (13.8Ã— 12m; but discounting reduces)
â”‚
â”œâ”€ Stage 2 vs Stage 3 Lifetime ECL:
â”‚   â”œâ”€ Stage 2 (Performing Lifetime ECL):
â”‚   â”‚   â”œâ”€ PD < 100%: Credit risk elevated but not defaulted
â”‚   â”‚   â”œâ”€ Full PD term structure: Account for survival probabilities
â”‚   â”‚   â”œâ”€ Interest revenue: Calculated on gross carrying amount
â”‚   â”‚   â””â”€ Example: Lifetime PD = 10%, ECL = $40k on $1M loan
â”‚   â”‚
â”‚   â”œâ”€ Stage 3 (Impaired Lifetime ECL):
â”‚   â”‚   â”œâ”€ PD = 100% (default already occurred)
â”‚   â”‚   â”œâ”€ ECL = EAD Ã— LGD (no PD uncertainty; focus on recovery)
â”‚   â”‚   â”œâ”€ Recovery timing critical:
â”‚   â”‚   â”‚   â”œâ”€ Discount expected recoveries to present value
â”‚   â”‚   â”‚   â”œâ”€ Foreclosure: 2-3 years; Bankruptcy: 1-5 years
â”‚   â”‚   â”‚   â””â”€ Example: Recovery $60k in 2 years â†’ PV = $54.4k @ 5% EIR
â”‚   â”‚   â”œâ”€ Interest revenue: Calculated on net carrying amount (after ECL)
â”‚   â”‚   â””â”€ Example: $1M loan, LGD = 60%, ECL = $600k (recovery $400k)
â”‚   â”‚
â”‚   â””â”€ Key Difference:
â”‚       â”œâ”€ Stage 2: Probabilistic (PD Ã— LGD); uncertainty in timing/occurrence
â”‚       â””â”€ Stage 3: Deterministic (LGD only); uncertainty in recovery amount/timing
â”‚
â”œâ”€ Practical Simplifications:
â”‚   â”œâ”€ Flat PD Approximation:
â”‚   â”‚   â”œâ”€ Assume constant annual PD (simplifies calculation)
â”‚   â”‚   â”œâ”€ Lifetime ECL â‰ˆ EAD Ã— [1 - (1 - PD)^T] Ã— LGD Ã— avg DF
â”‚   â”‚   â””â”€ Acceptable for portfolios with stable credit risk
â”‚   â”‚
â”‚   â”œâ”€ Vintage Analysis:
â”‚   â”‚   â”œâ”€ Group loans by origination cohort
â”‚   â”‚   â”œâ”€ Apply cohort-specific default curves (based on historical performance)
â”‚   â”‚   â””â”€ Common for retail portfolios (auto, credit card)
â”‚   â”‚
â”‚   â”œâ”€ Roll Rates (Consumer Credit):
â”‚   â”‚   â”œâ”€ Migration through delinquency buckets (current â†’ 30 DPD â†’ 60 DPD â†’ default)
â”‚   â”‚   â”œâ”€ Lifetime ECL = Sum over buckets weighted by transition probabilities
â”‚   â”‚   â””â”€ Avoids explicit PD term structure modeling
â”‚   â”‚
â”‚   â””â”€ Portfolio-Level Models:
â”‚       â”œâ”€ Aggregate exposures by segment (product, rating, maturity)
â”‚       â”œâ”€ Apply segment-level PD/LGD; allocate back to loans
â”‚       â””â”€ Efficiency: Reduces computation for large portfolios
â”‚
â”œâ”€ Stage 1â†’2 Transfer Impact:
â”‚   â”œâ”€ Provisioning Cliff:
â”‚   â”‚   â”œâ”€ 12-month ECL â†’ Lifetime ECL transition causes large P&L charge
â”‚   â”‚   â”œâ”€ Magnitude depends on maturity (longer = larger jump)
â”‚   â”‚   â””â”€ Example: 10-year loan; 12m ECL = $5k â†’ Lifetime ECL = $50k (+$45k charge)
â”‚   â”‚
â”‚   â”œâ”€ Timeliness of SICR Detection:
â”‚   â”‚   â”œâ”€ Early SICR detection â†’ Gradual provisioning increase
â”‚   â”‚   â”œâ”€ Late SICR detection â†’ Sudden large charge (cliff effect)
â”‚   â”‚   â””â”€ Regulatory expectation: Timely SICR triggers; avoid delayed recognition
â”‚   â”‚
â”‚   â””â”€ P&L Volatility:
â”‚       â”œâ”€ Frequent Stage 1â†”2 oscillations â†’ Volatile provisions
â”‚       â”œâ”€ Mitigation: Hysteresis (different thresholds for upgrade vs downgrade)
â”‚       â””â”€ Cure probation: Require sustained improvement before Stage 2â†’1 transfer
â”‚
â””â”€ Regulatory & Disclosure:
    â”œâ”€ IFRS 7 Disclosure Requirements:
    â”‚   â”œâ”€ ECL breakdown: Stage 1 (12m) vs Stage 2/3 (lifetime)
    â”‚   â”œâ”€ Stage migrations: Transfers between stages; opening/closing balances
    â”‚   â”œâ”€ Maturity analysis: ECL by maturity bucket
    â”‚   â””â”€ Sensitivity: Impact of alternative scenarios/assumptions
    â”‚
    â”œâ”€ Basel IRB Alignment:
    â”‚   â”œâ”€ 12-month PD (Basel) â‰ˆ Stage 1 PD (IFRS 9)
    â”‚   â”œâ”€ Lifetime PD (downturn) â‰ˆ Stage 2 PD (IFRS 9 adverse scenario)
    â”‚   â””â”€ Efficiency: Use Basel models for IFRS 9 ECL (with adjustments)
    â”‚
    â””â”€ Audit Focus:
        â”œâ”€ PD term structure: Calibration; reasonableness of long-term PDs
        â”œâ”€ Discounting: EIR calculation; consistency across stages
        â”œâ”€ Maturity assumptions: Revolving credit expected life; prepayment rates
        â””â”€ SICR triggers: Timeliness; avoid delayed Stage 1â†’2 transfers
```

**Key Insight:** 12-month ECL (Stage 1) = short-horizon, low provisions (0.1-1%); Lifetime ECL (Stage 2/3) = full maturity, high provisions (3-90%); Lifetime ECL 3-20Ã— higher depending on maturity; Stage 1â†’2 transfer causes provisioning cliff; discounting reduces long-dated ECL impact.

## 5. Challenge Round
When 12-month vs lifetime ECL frameworks fail or introduce complexity:
- **Revolving Credit (Uncertain Maturity)**: Credit card with no fixed maturity â†’ Lifetime ECL horizon ambiguous; solution: Use expected behavioral life (e.g., 3-5 years based on historical usage); exclude contractual cancel-on-demand clauses if not exercised
- **Long-Dated Loans (30+ Years)**: Mortgage with 30-year term â†’ PD term structure extremely uncertain; solution: Revert to TTC mean after Year 5; flatten PD curve; rely on discounting to reduce far-future impact
- **Short Maturity (< 1 Year)**: 6-month loan â†’ Lifetime ECL â‰ˆ 12-month ECL â†’ Stage 1â†’2 transfer negligible impact; solution: Accept minimal Stage transfer effect; focus on Stage 3 (default) detection
- **Prepayments (Mortgages)**: Early repayment reduces EAD â†’ Lifetime ECL lower; solution: Model conditional prepayment rates (CPR); sensitivity to interest rates; stress test low prepayment scenario
- **Stage 2 Cure**: Loan in Stage 2 for 6 months; cures â†’ Revert to Stage 1 â†’ Provision release â†’ P&L volatility; solution: Require probation period (3-6 months current) before cure; avoid oscillation
- **Discounting Ambiguity (Stage 3)**: IFRS 9 allows original EIR or credit-adjusted rate â†’ Choice impacts ECL by 20-40%; solution: Consistent policy; original EIR common (simpler); disclose choice

## 6. Key References
- [IFRS 9 Financial Instruments (Section 5.5.3-5.5.5)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official standard; 12-month vs lifetime ECL definitions; measurement requirements
- [PwC IFRS 9: Expected Credit Loss (2019)](https://www.pwc.com/gx/en/audit-services/ifrs/publications/ifrs-9/expected-credit-loss-ifrs-9-practical-guide.pdf) - Practical guide; ECL calculation examples; maturity considerations; discount rate application
- [EY IFRS 9 Impairment Banking Survey (2020)](https://www.ey.com/en_gl/ifrs-technical-resources/ifrs-9-impairment-banking-survey-2020) - Industry practices; Stage 1/2 coverage ratios; lifetime ECL methodologies; benchmarking data

---
**Status:** IFRS 9 Core Concept | **Complements:** Three-Stage Approach, Expected Credit Loss Models, SICR, Forward-Looking Information
