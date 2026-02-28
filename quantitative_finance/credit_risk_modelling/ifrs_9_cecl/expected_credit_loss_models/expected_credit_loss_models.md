# Expected Credit Loss (ECL) Models

## 1. Concept Skeleton
**Definition:** Quantitative frameworks estimating probability-weighted present value of credit losses over instrument lifetime or 12 months; integrate PD (probability of default), LGD (loss given default), EAD (exposure at default), and macroeconomic scenarios  
**Purpose:** Calculate IFRS 9 / CECL provisions; forward-looking loss estimation; scenario-weighted risk quantification; regulatory capital calculation (aligns with Basel IRB)  
**Prerequisites:** Credit risk components (PD, LGD, EAD), survival analysis, logistic regression, macroeconomic scenario analysis, discounting, segmentation

## 2. Comparative Framing
| Model Type | Complexity | Data Requirements | Forward-Looking | Granularity | Use Case |
|------------|------------|-------------------|-----------------|-------------|----------|
| **Historical Loss Rate** | Low | Historical defaults | No (backward) | Portfolio-level | Simple portfolios; limited data |
| **Roll Rate (Migration)** | Medium | Payment status transitions | Implicit | Vintage/delinquency bucket | Consumer credit; arrears-based |
| **PD-LGD-EAD (IRB)** | High | Individual exposures; defaults | Yes (scenarios) | Loan-level | Corporate; Basel IRB; IFRS 9 |
| **Discounted Cash Flow** | Very High | Payment schedules; recovery timing | Yes | Instrument-level | Complex structures; Stage 3 |
| **Machine Learning** | Very High | Rich features (100+ variables) | Yes (if macro features) | Loan-level | Large datasets; non-linear patterns |

## 3. Examples + Counterexamples

**Simple Example:**  
Stage 1 loan: $100k, 12-month PD = 1%, LGD = 40%, EAD = $100k. ECL = $100k Ã— 1% Ã— 40% = $400. Stage 2 (lifetime): 5-year maturity, lifetime PD = 8%, ECL = $100k Ã— 8% Ã— 40% = $3,200.

**Scenario-Weighted ECL:**  
Base scenario (50% weight): PD = 2%, ECL = $800. Downturn (30% weight): PD = 5%, ECL = $2,000. Upturn (20% weight): PD = 1%, ECL = $400. Weighted ECL = 0.5Ã—$800 + 0.3Ã—$2,000 + 0.2Ã—$400 = $1,080.

**Discounted Cash Flow (Stage 3):**  
Defaulted loan $100k, expected recovery $40k in 2 years. Discount rate (EIR) = 5%. ECL = $100k - $40k/(1.05)Â² = $100k - $36.3k = $63.7k.

**Roll Rate Model:**  
Consumer portfolio: 1,000 loans current (bucket 0), 50 transition to 30 DPD (bucket 1), 10 from bucket 1 to 60 DPD (bucket 2), 5 from bucket 2 to default. Roll rates: 5% (0â†’1), 20% (1â†’2), 50% (2â†’default). ECL = sum over buckets weighted by exposure.

**Failure Case (No Scenarios):**  
Bank uses point estimate PD (mean) without scenario weighting. Economic downturn hits: Actual losses 3Ã— higher than ECL. Insufficient provisioning; regulatory criticism.

## 4. Layer Breakdown
```
Expected Credit Loss Model Framework:

â”œâ”€ ECL Formula (General):
â”‚   ECL = EAD Ã— PD Ã— LGD Ã— Discount Factor
â”‚   â”œâ”€ EAD: Exposure at Default (outstanding + undrawn commitments Ã— CCF)
â”‚   â”œâ”€ PD: Probability of Default (12-month or lifetime; scenario-adjusted)
â”‚   â”œâ”€ LGD: Loss Given Default (1 - recovery rate; collateral-adjusted)
â”‚   â””â”€ Discount Factor: Present value (discounted at effective interest rate, EIR)
â”‚
â”œâ”€ 12-Month ECL (Stage 1):
â”‚   â”œâ”€ Horizon: Next 12 months only
â”‚   â”œâ”€ PD: 12-month probability of default
â”‚   â”‚   â”œâ”€ Point-in-Time (PIT): Current economic conditions
â”‚   â”‚   â”œâ”€ Through-the-Cycle (TTC): Average over cycle (Basel IRB)
â”‚   â”‚   â””â”€ IFRS 9 requires PIT (forward-looking scenarios)
â”‚   â”œâ”€ Formula: ECL = EAD Ã— PD(12m) Ã— LGD
â”‚   â”œâ”€ Example: Loan $1M, PD(12m) = 0.5%, LGD = 45%
â”‚   â”‚   ECL = $1M Ã— 0.5% Ã— 45% = $2,250
â”‚   â””â”€ No discounting (materiality; 12m horizon short)
â”‚
â”œâ”€ Lifetime ECL (Stage 2 & 3):
â”‚   â”œâ”€ Horizon: Remaining contractual life of instrument
â”‚   â”œâ”€ PD: Cumulative probability of default over lifetime
â”‚   â”‚   â”œâ”€ Marginal PD(t): Default probability in period t (conditional on survival to t)
â”‚   â”‚   â”œâ”€ Survival probability: S(t) = âˆ(1 - PD(Ï„)) for Ï„ = 1..t-1
â”‚   â”‚   â”œâ”€ Cumulative PD: CPD = 1 - S(T) = 1 - âˆ(1 - PD(t))
â”‚   â”‚   â””â”€ Forward-looking: PD(t) varies by macroeconomic scenario path
â”‚   â”œâ”€ Integration Over Time:
â”‚   â”‚   ECL = âˆ‘[t=1 to T] { EAD(t) Ã— PD(t) Ã— LGD(t) Ã— DF(t) }
â”‚   â”‚   â”œâ”€ EAD(t): Exposure at time t (amortization, prepayments, drawdowns)
â”‚   â”‚   â”œâ”€ PD(t): Marginal default probability at time t
â”‚   â”‚   â”œâ”€ LGD(t): Loss given default (may vary with economic scenario)
â”‚   â”‚   â””â”€ DF(t): Discount factor = 1 / (1 + EIR)^t
â”‚   â”œâ”€ Example: 5-year loan $1M, annual PD = [1%, 1.5%, 2%, 2.5%, 3%]
â”‚   â”‚   Survival: S(1) = 99%, S(2) = 97.5%, ..., S(5) = 91.8%
â”‚   â”‚   ECL â‰ˆ $1M Ã— 8.2% Ã— 45% Ã— avg DF â‰ˆ $35,000 (lifetime)
â”‚   â””â”€ Stage 3: PD(default) = 100%; focus shifts to LGD estimation (recovery rate)
â”‚
â”œâ”€ PD Estimation Methods:
â”‚   â”œâ”€ Credit Scoring Models:
â”‚   â”‚   â”œâ”€ Logistic Regression: log(PD / (1-PD)) = Î²â‚€ + Î²â‚Xâ‚ + ... + Î²â‚™Xâ‚™
â”‚   â”‚   â”‚   â”œâ”€ Features: Debt-to-income, credit score, LTV, payment history
â”‚   â”‚   â”‚   â””â”€ Output: Point-in-time PD (12-month or 1-year)
â”‚   â”‚   â”œâ”€ Calibration: Map scores to PD using historical default rates
â”‚   â”‚   â””â”€ Segmentation: Separate models by product (mortgage, corporate, credit card)
â”‚   â”œâ”€ Transition Matrices (Rating Migration):
â”‚   â”‚   â”œâ”€ Markov chain: P(Rating_t+1 | Rating_t)
â”‚   â”‚   â”œâ”€ Transition probabilities: AAAâ†’AA, AAâ†’A, ..., CCCâ†’Default
â”‚   â”‚   â”œâ”€ Lifetime PD: Compound transitions over T periods
â”‚   â”‚   â””â”€ Scenario adjustment: Stress rating migration matrix (downturn = higher default rates)
â”‚   â”œâ”€ Survival Analysis (Hazard Models):
â”‚   â”‚   â”œâ”€ Hazard rate Î»(t): Instantaneous default rate at time t
â”‚   â”‚   â”œâ”€ Survival function: S(t) = exp(-âˆ«Î»(Ï„)dÏ„)
â”‚   â”‚   â”œâ”€ PD(t) = Î»(t) Ã— S(t-1)
â”‚   â”‚   â””â”€ Cox Proportional Hazards: Î»(t) = Î»â‚€(t) Ã— exp(Î²X)
â”‚   â”œâ”€ Structural Models (Merton):
â”‚   â”‚   â”œâ”€ Firm value follows geometric Brownian motion
â”‚   â”‚   â”œâ”€ Default when firm value < debt threshold
â”‚   â”‚   â”œâ”€ PD = N(-dâ‚‚) where dâ‚‚ = (ln(V/K) + (Î¼ - ÏƒÂ²/2)T) / (ÏƒâˆšT)
â”‚   â”‚   â””â”€ Calibrated to equity volatility, leverage
â”‚   â””â”€ Machine Learning (XGBoost, Neural Networks):
â”‚       â”œâ”€ Non-linear relationships; interaction effects
â”‚       â”œâ”€ Features: 100+ variables (payment history, macro, behavioral)
â”‚       â”œâ”€ Calibration: Ensure monotonicity; align with historical default rates
â”‚       â””â”€ Challenge: Interpretability; regulatory acceptance
â”‚
â”œâ”€ LGD Estimation Methods:
â”‚   â”œâ”€ Historical Recovery Rates:
â”‚   â”‚   â”œâ”€ LGD = 1 - (Recovery / EAD)
â”‚   â”‚   â”œâ”€ Segmentation: Secured vs unsecured; collateral type; seniority
â”‚   â”‚   â”œâ”€ Example: Secured mortgage LGD â‰ˆ 20-30%; unsecured credit card LGD â‰ˆ 70-90%
â”‚   â”‚   â””â”€ Downturn LGD: Adjust for economic stress (lower collateral values)
â”‚   â”œâ”€ Collateral Haircuts:
â”‚   â”‚   â”œâ”€ Market value of collateral Ã— (1 - haircut) - costs to sell
â”‚   â”‚   â”œâ”€ Real estate: 20-30% haircut; equipment: 40-60%; intangibles: 80%+
â”‚   â”‚   â””â”€ Stressed scenario: Higher haircuts (illiquid markets)
â”‚   â”œâ”€ Discounted Cash Flow (Workout LGD):
â”‚   â”‚   â”œâ”€ Estimate recovery timing: Foreclosure 2-3 years; bankruptcy 1-5 years
â”‚   â”‚   â”œâ”€ Discount recoveries at EIR
â”‚   â”‚   â””â”€ Example: Recovery $40k in 2 years, EIR 5% â†’ PV = $36.3k
â”‚   â””â”€ Regression Models:
â”‚       â”œâ”€ LGD = f(collateral value, seniority, industry, macro conditions)
â”‚       â””â”€ Calibrate to historical workout data
â”‚
â”œâ”€ EAD Estimation:
â”‚   â”œâ”€ On-Balance Sheet (Term Loans):
â”‚   â”‚   â”œâ”€ EAD = Outstanding principal + accrued interest
â”‚   â”‚   â””â”€ Amortization: EAD declines over time (scheduled repayments)
â”‚   â”œâ”€ Off-Balance Sheet (Commitments):
â”‚   â”‚   â”œâ”€ Credit Conversion Factor (CCF): Proportion drawn at default
â”‚   â”‚   â”œâ”€ EAD = Drawn + Undrawn Ã— CCF
â”‚   â”‚   â”œâ”€ Example: $50k drawn, $50k undrawn, CCF = 50% â†’ EAD = $50k + $25k = $75k
â”‚   â”‚   â””â”€ Stressed CCF: Higher drawdowns during crisis (liquidity stress)
â”‚   â””â”€ Derivatives (CVA):
â”‚       â”œâ”€ Expected Exposure (EE): Forward simulation of MTM
â”‚       â””â”€ EAD = Î± Ã— EE(t) (regulatory factor Î±)
â”‚
â”œâ”€ Forward-Looking Scenarios:
â”‚   â”œâ”€ Scenario Design:
â”‚   â”‚   â”œâ”€ Base (50-60% weight): Most likely economic path (consensus forecast)
â”‚   â”‚   â”œâ”€ Adverse/Downturn (20-30%): Recession scenario (GDP -2%, unemployment +3%)
â”‚   â”‚   â”œâ”€ Severe Adverse (5-10%): Tail risk (financial crisis; GDP -5%)
â”‚   â”‚   â””â”€ Upside (10-20%): Benign conditions (GDP +4%, low unemployment)
â”‚   â”œâ”€ Scenario Variables:
â”‚   â”‚   â”œâ”€ Macroeconomic: GDP growth, unemployment, inflation, interest rates
â”‚   â”‚   â”œâ”€ Market: Equity indices, commodity prices, FX rates
â”‚   â”‚   â””â”€ Sector-specific: Oil prices (energy loans), house prices (mortgages)
â”‚   â”œâ”€ PD/LGD Sensitivity to Scenarios:
â”‚   â”‚   â”œâ”€ Recession â†’ Higher PD (credit deterioration), Higher LGD (lower collateral values)
â”‚   â”‚   â”œâ”€ Boom â†’ Lower PD (stronger borrower finances), Lower LGD (asset price appreciation)
â”‚   â”‚   â””â”€ Econometric models: PD = f(GDP, unemployment); LGD = f(house prices)
â”‚   â””â”€ Probability Weighting:
â”‚       â”œâ”€ ECL = âˆ‘[s] w(s) Ã— ECL(s)
â”‚       â”œâ”€ Weights sum to 1; based on expert judgment or scenario probability
â”‚       â””â”€ IFRS 9 requires unbiased (not excessively prudent; not optimistic)
â”‚
â”œâ”€ Discounting:
â”‚   â”œâ”€ Effective Interest Rate (EIR):
â”‚   â”‚   â”œâ”€ Discount rate that equates present value of cash flows to amortized cost
â”‚   â”‚   â”œâ”€ Includes origination fees, transaction costs (not market risk premium)
â”‚   â”‚   â””â”€ Typical: 4-8% for corporate loans; 10-20% for credit cards
â”‚   â”œâ”€ Stage 1 & 2: Discount at EIR (original effective rate)
â”‚   â”œâ”€ Stage 3: Discount at EIR or credit-adjusted rate (debate; IFRS 9 allows both)
â”‚   â””â”€ Material Impact: Long maturity (10+ years) â†’ Discounting reduces ECL by 20-40%
â”‚
â”œâ”€ Segmentation:
â”‚   â”œâ”€ Why Segment:
â”‚   â”‚   â”œâ”€ Homogeneous risk within segment (similar PD/LGD drivers)
â”‚   â”‚   â”œâ”€ Reduces model complexity; improves calibration
â”‚   â”‚   â””â”€ Aligns with business practices (product types)
â”‚   â”œâ”€ Segmentation Dimensions:
â”‚   â”‚   â”œâ”€ Product: Mortgage, corporate term loan, revolving credit, credit card
â”‚   â”‚   â”œâ”€ Geography: Country, region (different economic conditions)
â”‚   â”‚   â”œâ”€ Industry: Energy, real estate, retail (sector risk)
â”‚   â”‚   â”œâ”€ Collateral: Secured vs unsecured; asset-backed
â”‚   â”‚   â””â”€ Vintage: Origination year (cohort analysis)
â”‚   â””â”€ Example: Mortgage ECL model separate from corporate loan model (different PD/LGD drivers)
â”‚
â”œâ”€ Model Calibration & Validation:
â”‚   â”œâ”€ Calibration:
â”‚   â”‚   â”œâ”€ Align modeled PD to historical default rates (by segment)
â”‚   â”‚   â”œâ”€ Central Tendency: Ensure long-run average PD = observed default rate
â”‚   â”‚   â”œâ”€ Adjust for economic cycle (TTC â†’ PIT conversion)
â”‚   â”‚   â””â”€ LGD: Map to historical recovery rates; adjust for downturn
â”‚   â”œâ”€ Backtesting:
â”‚   â”‚   â”œâ”€ Out-of-sample validation: Test PD model on holdout data
â”‚   â”‚   â”œâ”€ AUC (Area Under Curve): Discriminatory power (>0.7 acceptable; >0.8 good)
â”‚   â”‚   â”œâ”€ Gini coefficient: 2 Ã— AUC - 1 (alternative metric)
â”‚   â”‚   â””â”€ Calibration plots: Predicted PD vs observed default rate by decile
â”‚   â”œâ”€ Stress Testing:
â”‚   â”‚   â”œâ”€ Apply adverse scenarios; compare ECL to actual losses in crisis
â”‚   â”‚   â””â”€ Regulatory stress tests (CCAR, EBA): Validate scenario sensitivity
â”‚   â””â”€ Model Risk:
â”‚       â”œâ”€ Parameter uncertainty: PD/LGD estimates have confidence intervals
â”‚       â”œâ”€ Model misspecification: Logistic regression may miss non-linearities
â”‚       â””â”€ Management overlay: Expert adjustments for model limitations (e.g., COVID-19 shock)
â”‚
â””â”€ Practical Implementation:
    â”œâ”€ Systems Architecture:
    â”‚   â”œâ”€ Data warehouse: Loan-level exposures, payment history, collateral values
    â”‚   â”œâ”€ ECL engine: Calculate 12m and lifetime ECL by instrument Ã— scenario
    â”‚   â”œâ”€ Scenario platform: Generate macro paths; map to PD/LGD
    â”‚   â””â”€ Reporting: IFRS 9 disclosures; regulatory capital (Basel IRB)
    â”œâ”€ Computational Efficiency:
    â”‚   â”œâ”€ Monte Carlo simulation: For derivatives EAD (computationally expensive)
    â”‚   â”œâ”€ Closed-form approximations: For large portfolios (analytical ECL formula)
    â”‚   â””â”€ Parallel processing: Distribute calculations across computing cluster
    â”œâ”€ Governance:
    â”‚   â”œâ”€ Model documentation: Assumptions, data sources, validation results
    â”‚   â”œâ”€ Model approval: Risk committee, Board oversight
    â”‚   â”œâ”€ Annual review: Update PD/LGD models; recalibrate to new data
    â”‚   â””â”€ Audit trail: All ECL calculations; scenario assumptions logged
    â””â”€ Regulatory Alignment:
        â”œâ”€ IFRS 9: Forward-looking ECL; probability-weighted scenarios
        â”œâ”€ CECL (US GAAP): Similar to IFRS 9 lifetime ECL (all exposures)
        â””â”€ Basel IRB: PD/LGD/EAD models aligned with IFRS 9 (efficiency gain)
```

**Key Insight:** ECL = EAD Ã— PD Ã— LGD Ã— DF; 12-month ECL for Stage 1 (low risk); lifetime ECL for Stage 2/3 (elevated/impaired); scenario-weighted forward-looking estimates; calibrated to historical data; validated via backtesting.

## 5. Challenge Round
When ECL models fail or introduce complexity:
- **Data Scarcity (Low Default Portfolios)**: Investment-grade corporate portfolio; 0.1% annual default rate â†’ Insufficient defaults to calibrate PD; solution: External data (rating agency default rates); peer benchmarks; Bayesian priors
- **Scenario Weights Arbitrary**: Management assigns 50% base, 30% adverse, 20% upside â†’ Subjective; changes provisions significantly; solution: Historical frequency of macro regimes; expert panel consensus; sensitivity analysis
- **Long Maturity (30-year Mortgages)**: PD term structure extends 30 years â†’ High uncertainty; model risk; solution: Flatten PD curve after year 10; use TTC PD for long tail; sensitivity to maturity assumption
- **Revolving Credit (Credit Cards)**: EAD uncertain (drawdown behavior volatile); CCF model critical; solution: Behavioral scoring models; stress CCF (100% in crisis); vintage analysis
- **Model Risk (Overfitting)**: ML model achieves 95% AUC on historical data â†’ Overfit to noise; poor forward performance; solution: Regularization (L1/L2); cross-validation; simple model benchmark
- **Discount Rate Ambiguity (Stage 3)**: IFRS 9 allows original EIR or credit-adjusted rate â†’ Choice impacts ECL by 20-40%; solution: Consistent policy; disclose choice; sensitivity analysis

## 6. Key References
- [IFRS 9 Expected Credit Losses (EY Guide, 2020)](https://www.ey.com/en_gl/ifrs-technical-resources/ifrs-9-expected-credit-loss) - Practical implementation; ECL calculation methodologies; worked examples
- [Basel II: International Convergence of Capital Measurement (BIS, 2006)](https://www.bis.org/publ/bcbs128.pdf) - IRB approach; PD, LGD, EAD definitions; aligns with IFRS 9 ECL models
- [Loan Loss Provisioning and Economic Slowdowns (IMF Working Paper, 2018)](https://www.imf.org/en/Publications/WP/Issues/2018/07/23/Loan-Loss-Provisioning-and-Economic-Slowdowns-Too-Little-Too-Late-46053) - Empirical analysis; forward-looking ECL vs incurred loss; procyclicality

---
**Status:** IFRS 9 Core Methodology | **Complements:** Three-Stage Approach, SICR, Forward-Looking Information, PD/LGD/EAD Models
