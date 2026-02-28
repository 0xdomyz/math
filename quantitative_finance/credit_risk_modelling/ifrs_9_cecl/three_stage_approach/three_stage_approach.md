# Three-Stage Approach (IFRS 9)

## 1. Concept Skeleton
**Definition:** Classification framework for financial instruments into Stage 1 (12-month ECL), Stage 2 (lifetime ECL performing), Stage 3 (credit-impaired lifetime ECL) based on credit deterioration since origination  
**Purpose:** Timely recognition of credit losses; forward-looking expected loss accounting; replace incurred loss model (IAS 39); escalate provisioning as credit risk increases  
**Prerequisites:** Expected credit loss (ECL) concepts, significant increase in credit risk (SICR) criteria, default definition, probability of default (PD), loss given default (LGD)

## 2. Comparative Framing
| Stage | Credit Quality | ECL Horizon | Loss Recognition | Interest Revenue | Typical Example |
|-------|----------------|-------------|------------------|------------------|-----------------|
| **Stage 1** | No significant deterioration | 12-month ECL | Low (expected losses in next year) | Gross carrying amount Ã— EIR | New loan; current on payments |
| **Stage 2** | Significant increase in credit risk (SICR) | Lifetime ECL | Higher (all expected losses to maturity) | Gross carrying amount Ã— EIR | 30+ days past due (presumption); rating downgrade |
| **Stage 3** | Credit-impaired (default) | Lifetime ECL | Highest (impaired loss) | Net carrying amount Ã— EIR | 90+ days past due; bankruptcy; covenant breach |

## 3. Examples + Counterexamples

**Stage 1 Example:**  
Corporate loan $100M originated at BBB rating; current on payments; market conditions stable. 12-month PD = 0.5%, LGD = 40%. ECL = $100M Ã— 0.5% Ã— 40% = $200k provision.

**Stage 2 Trigger:**  
Same loan downgrades to BB after 1 year (3-notch downgrade = SICR). Lifetime PD = 5%, LGD = 40%, maturity 4 years remaining. Lifetime ECL = $100M Ã— 5% Ã— 40% = $2M provision (10Ã— Stage 1).

**Stage 3 Impairment:**  
Borrower files bankruptcy; 90+ days past due. Default certain (PD â‰ˆ 100%), LGD estimated 60% based on collateral. ECL = $100M Ã— 100% Ã— 60% = $60M provision.

**Edge Case (Cured Loan):**  
Stage 2 loan cures (payments resume; rating upgrade to BBB). SICR no longer present â†’ Transfer back to Stage 1 â†’ Reduce ECL to 12-month. Provision reversal through P&L.

**Failure Case (No SICR Detection):**  
Bank ignores rating downgrade (BBâ†’B), keeps loan in Stage 1. Borrower defaults 6 months later. Insufficient provisioning â†’ Sudden large loss; regulatory criticism.

## 4. Layer Breakdown
```
Three-Stage Model Framework:

â”œâ”€ Stage 1: Performing (No SICR):
â”‚   â”œâ”€ Definition: Credit risk has NOT increased significantly since initial recognition
â”‚   â”œâ”€ ECL Recognition: 12-month expected credit losses
â”‚   â”‚   â”œâ”€ Horizon: Expected losses from defaults in next 12 months only
â”‚   â”‚   â”œâ”€ Formula: ECL = EAD Ã— PD(12m) Ã— LGD
â”‚   â”‚   â””â”€ Rationale: Low deterioration risk; short-horizon focus
â”‚   â”œâ”€ Interest Revenue: Calculated on gross carrying amount
â”‚   â”‚   â””â”€ Gross amount = Amortized cost before ECL allowance
â”‚   â”œâ”€ Criteria for Stage 1:
â”‚   â”‚   â”œâ”€ No 30+ days past due (rebuttable presumption)
â”‚   â”‚   â”œâ”€ No significant rating downgrade (e.g., < 2 notches)
â”‚   â”‚   â”œâ”€ No adverse qualitative indicators (restructuring, covenant breach)
â”‚   â”‚   â””â”€ Low default probability (PD < threshold, e.g., 1%)
â”‚   â”œâ”€ Examples:
â”‚   â”‚   â”œâ”€ Newly originated loans (Day 1)
â”‚   â”‚   â”œâ”€ Current on payments; stable credit metrics
â”‚   â”‚   â”œâ”€ Investment-grade securities (AAA to BBB-)
â”‚   â”‚   â””â”€ No negative watchlist flags
â”‚   â””â”€ Provisioning: Typically 0.1%-1% of exposure (depends on PD/LGD)
â”‚
â”œâ”€ Stage 2: Performing (SICR):
â”‚   â”œâ”€ Definition: Credit risk HAS increased significantly since origination
â”‚   â”œâ”€ ECL Recognition: Lifetime expected credit losses
â”‚   â”‚   â”œâ”€ Horizon: All expected losses over remaining life of instrument
â”‚   â”‚   â”œâ”€ Formula: ECL = EAD Ã— PD(lifetime) Ã— LGD
â”‚   â”‚   â””â”€ Integration: Sum over all future periods weighted by PD(t)
â”‚   â”œâ”€ Interest Revenue: Still calculated on gross carrying amount
â”‚   â”‚   â””â”€ Not yet credit-impaired; revenue recognition continues
â”‚   â”œâ”€ Significant Increase in Credit Risk (SICR) Triggers:
â”‚   â”‚   â”œâ”€ Quantitative Indicators:
â”‚   â”‚   â”‚   â”œâ”€ 30+ days past due (rebuttable presumption per IFRS 9)
â”‚   â”‚   â”‚   â”œâ”€ Credit rating downgrade (e.g., 2+ notches)
â”‚   â”‚   â”‚   â”œâ”€ Relative PD increase: PD(current) / PD(origination) > threshold (e.g., 2Ã—)
â”‚   â”‚   â”‚   â”œâ”€ Absolute PD increase: PD > 5% (institution-specific)
â”‚   â”‚   â”‚   â””â”€ LTV deterioration: Loan-to-value > 100% (negative equity)
â”‚   â”‚   â”œâ”€ Qualitative Indicators:
â”‚   â”‚   â”‚   â”œâ”€ Borrower financial distress (covenant breach, restructuring request)
â”‚   â”‚   â”‚   â”œâ”€ Industry/sector deterioration (oil price collapse for energy loans)
â”‚   â”‚   â”‚   â”œâ”€ Economic downturn in borrower geography
â”‚   â”‚   â”‚   â”œâ”€ Management changes, litigation, regulatory action
â”‚   â”‚   â”‚   â””â”€ Adverse news (earnings warnings, credit watch negative)
â”‚   â”‚   â””â”€ Backstop: 30 days past due (mandatory unless rebutted)
â”‚   â”œâ”€ Stage Transfer Logic:
â”‚   â”‚   â”œâ”€ Stage 1 â†’ Stage 2: SICR criteria met
â”‚   â”‚   â”œâ”€ Stage 2 â†’ Stage 1: SICR no longer present (cure)
â”‚   â”‚   â”‚   â””â”€ Typically requires 6-12 months of satisfactory performance
â”‚   â”‚   â””â”€ Stage 2 â†’ Stage 3: Default occurs
â”‚   â”œâ”€ Provisioning: Typically 3%-15% of exposure (higher PD Ã— longer horizon)
â”‚   â””â”€ Key Challenge: Defining SICR thresholds (avoid cliff effects; balance timeliness vs stability)
â”‚
â”œâ”€ Stage 3: Non-Performing (Credit-Impaired):
â”‚   â”œâ”€ Definition: Objective evidence of impairment (default)
â”‚   â”œâ”€ ECL Recognition: Lifetime expected credit losses (default-adjusted)
â”‚   â”‚   â”œâ”€ Horizon: Remaining life (but default already occurred)
â”‚   â”‚   â”œâ”€ Formula: ECL = EAD Ã— PD(default = 100%) Ã— LGD = EAD Ã— LGD
â”‚   â”‚   â””â”€ Focus shifts to recovery estimation (collateral value, workout process)
â”‚   â”œâ”€ Interest Revenue: Calculated on net carrying amount (after ECL deduction)
â”‚   â”‚   â””â”€ Net amount = Amortized cost - ECL allowance
â”‚   â”‚   â””â”€ Lower interest revenue (reflects credit-impaired status)
â”‚   â”œâ”€ Default Triggers (IFRS 9 aligns with Basel):
â”‚   â”‚   â”œâ”€ 90+ days past due (presumption of default)
â”‚   â”‚   â”œâ”€ Bankruptcy, insolvency, administration
â”‚   â”‚   â”œâ”€ Covenant breach leading to acceleration
â”‚   â”‚   â”œâ”€ Distressed debt restructuring (concessions due to financial difficulty)
â”‚   â”‚   â”œâ”€ Sale of financial asset at material credit-related loss
â”‚   â”‚   â””â”€ Internal rating = Default grade (D)
â”‚   â”œâ”€ LGD Estimation:
â”‚   â”‚   â”œâ”€ Collateral valuation: Market value - costs to sell
â”‚   â”‚   â”œâ”€ Discounted cash flow: Expected recoveries from workout
â”‚   â”‚   â”œâ”€ Historical recovery rates: Industry-specific LGD (e.g., secured = 30%, unsecured = 70%)
â”‚   â”‚   â””â”€ Time to recovery: Discount recoveries to present value
â”‚   â”œâ”€ Stage Transfer Logic:
â”‚   â”‚   â”œâ”€ Stage 2 â†’ Stage 3: Default occurs
â”‚   â”‚   â”œâ”€ Stage 3 â†’ Stage 2: Cure (default status removed)
â”‚   â”‚   â”‚   â”œâ”€ Rare; requires full payment of arrears + probation period
â”‚   â”‚   â”‚   â””â”€ Typically 12+ months satisfactory performance
â”‚   â”‚   â””â”€ Direct Stage 1 â†’ Stage 3 possible (sudden default)
â”‚   â”œâ”€ Provisioning: Typically 30%-90% of exposure (high LGD; low recovery)
â”‚   â””â”€ Write-Off: When no reasonable expectation of recovery
â”‚       â”œâ”€ Remove from balance sheet
â”‚       â”œâ”€ ECL allowance utilized
â”‚       â””â”€ Continue collection efforts (off-balance sheet)
â”‚
â”œâ”€ Stage Transfer Mechanics:
â”‚   â”œâ”€ Monthly (or more frequent) assessment:
â”‚   â”‚   â”œâ”€ Evaluate SICR criteria for all Stage 1 exposures
â”‚   â”‚   â”œâ”€ Evaluate default criteria for Stage 2 exposures
â”‚   â”‚   â”œâ”€ Check cure criteria for Stage 2/3 exposures
â”‚   â”‚   â””â”€ Update ECL allowances for transfers
â”‚   â”œâ”€ Cliff Effects Mitigation:
â”‚   â”‚   â”œâ”€ Gradual PD increase approach (smooth transition)
â”‚   â”‚   â”œâ”€ Multiple SICR indicators (avoid single metric dominance)
â”‚   â”‚   â”œâ”€ Expert judgment overlay (qualitative factors)
â”‚   â”‚   â””â”€ Backstop prevents delayed recognition (30 DPD mandatory)
â”‚   â””â”€ P&L Impact:
â”‚       â”œâ”€ ECL increase: Provision expense (credit loss)
â”‚       â”œâ”€ ECL decrease: Provision release (gain)
â”‚       â”œâ”€ Stage 1â†’2 transfer: Lifetime ECL charge (significant impact)
â”‚       â””â”€ Stage 3 default: Large impairment loss (one-time)
â”‚
â”œâ”€ Practical Implementation:
â”‚   â”œâ”€ Data Requirements:
â”‚   â”‚   â”œâ”€ Origination data: PD, rating, LTV at inception
â”‚   â”‚   â”œâ”€ Current data: Payment status, rating, collateral value
â”‚   â”‚   â”œâ”€ Macroeconomic scenarios: GDP, unemployment, interest rates
â”‚   â”‚   â””â”€ Historical data: Default rates, recovery rates by segment
â”‚   â”œâ”€ Model Infrastructure:
â”‚   â”‚   â”œâ”€ PD models: Credit scoring, transition matrices, survival analysis
â”‚   â”‚   â”œâ”€ LGD models: Recovery rate estimation, collateral valuation
â”‚   â”‚   â”œâ”€ EAD models: Credit conversion factors, drawdown at default
â”‚   â”‚   â”œâ”€ SICR framework: Quantitative + qualitative rules engine
â”‚   â”‚   â””â”€ Scenario engine: Forward-looking macro scenarios weighted by probability
â”‚   â”œâ”€ Governance:
â”‚   â”‚   â”œâ”€ Model validation: Annual review; backtesting PD/LGD
â”‚   â”‚   â”œâ”€ SICR thresholds: Documented rationale; Board approval
â”‚   â”‚   â”œâ”€ Stage migration reports: Monthly monitoring; trend analysis
â”‚   â”‚   â””â”€ Management overlays: Expert adjustments for model limitations
â”‚   â””â”€ Systems:
â”‚       â”œâ”€ Data warehouse: Centralized exposure, payment, rating data
â”‚       â”œâ”€ ECL engine: Calculate 12-month and lifetime ECL by instrument
â”‚       â”œâ”€ Stage classification module: Apply SICR/default rules
â”‚       â”œâ”€ Reporting: Regulatory (EBA ITS 2018/1627); financial statements
â”‚       â””â”€ Audit trail: All stage transfers, model assumptions documented
â”‚
â””â”€ Comparison to IAS 39 (Incurred Loss Model):
    â”œâ”€ IAS 39: Recognized losses only when objective evidence of impairment (backward-looking)
    â”œâ”€ IFRS 9: Recognizes expected losses immediately (forward-looking)
    â”œâ”€ Impact: Earlier loss recognition; higher provisions in economic downturns
    â”œâ”€ Procyclicality: IFRS 9 more countercyclical (builds provisions in good times)
    â””â”€ Complexity: IFRS 9 requires sophisticated PD/LGD models; IAS 39 simpler (historical loss rates)
```

**Key Insight:** Stage 1 (12m ECL) = low risk; Stage 2 (lifetime ECL) = elevated risk; Stage 3 (impaired lifetime ECL) = default. SICR triggers Stage 1â†’2 transfer (major provisioning impact); early SICR detection critical to avoid sudden losses.

## 5. Challenge Round
When three-stage classification fails or introduces complexity:
- **Cliff Effects**: Single loan crosses 30 DPD threshold â†’ Immediate Stage 1â†’2 transfer â†’ Large ECL jump; solution: Use probationary period (multiple months SICR before transfer); smooth PD increase approach
- **Curing Instability**: Loan oscillates between Stage 1 and Stage 2 (volatile PD) â†’ Frequent ECL adjustments; P&L volatility; solution: Require sustained improvement (6-12 months) before cure; hysteresis in thresholds
- **SICR Threshold Sensitivity**: Small PD change near threshold â†’ Large ECL impact; solution: Multiple SICR indicators (rating, PD, DPD, qualitative); weight of evidence approach; avoid single metric dominance
- **New Originations (No Baseline)**: Loan originated today â†’ No "origination PD" for comparison; solution: Use underwriting PD as baseline; Stage 1 until SICR observed relative to underwriting
- **Purchased Credit-Impaired (PCI)**: Loan bought at discount (already impaired) â†’ Day 1 Stage 3? Solution: IFRS 9 has special PCI rules (recognize gross-up method; different impairment calc)
- **Model Risk**: PD model overstates deterioration â†’ Excessive Stage 2 migrations; solution: Backtesting PD models; management overlays; independent validation

## 6. Key References
- [IFRS 9 Financial Instruments (Full Standard)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official IFRS Foundation standard; classification and measurement; impairment
- [EBA Guidelines on Credit Institutions' Credit Risk Management (2017)](https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-credit-institutions-credit-risk-management-practices-and-accounting-for-expected-credit-losses) - European Banking Authority implementation guidance; SICR criteria; staging approaches
- [Deloitte IFRS 9 Impairment Guide (2019)](https://www2.deloitte.com/content/dam/Deloitte/global/Documents/Financial-Services/gx-fsi-ifrs9-guide-impairment.pdf) - Practical implementation; worked examples; model approaches; industry practices

---
**Status:** IFRS 9 Core Concept | **Complements:** Expected Credit Loss Models, SICR, Forward-Looking Information, Lifetime ECL
