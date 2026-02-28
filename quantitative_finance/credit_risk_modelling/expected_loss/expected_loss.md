# Expected Loss (EL)

## 1. Concept Skeleton

**Definition:** Expected Loss (EL) represents the forward-looking estimate of average credit loss calculated as the product of three fundamental risk components: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD), adjusted for portfolio correlation. Formally, at the individual loan level: $EL = PD \times LGD \times EAD$, while at the portfolio level, correlation introduces non-linearity requiring more sophisticated modeling approaches such as the Vasicek single-factor model or multi-factor credit models.

**Purpose:** EL serves multiple critical business functions in financial institutions:
1. **Loan Pricing & Profitability:** Set interest rate spreads to cover expected losses plus operating costs and capital targets. For example, a $100K commercial loan with EL = $800 (0.8%) requires a credit spread of at least 80-100 basis points above funding costs, depending on cost structure and return targets.
2. **Provisioning & Accounting:** Under IFRS 9 and CECL standards, banks establish allowance for credit losses (ACL) matching EL estimates over the loan life or 12-month forward period. A portfolio of $1B with average 1.2% EL requires $12M in ACL reserves, impacting P&L directly.
3. **Economic Capital Allocation:** Calculate how much shareholder capital to reserve for unexpected loss (UL) beyond EL. If EL = $12M and portfolio volatility suggests UL at 99.9% confidence = $45M, total economic capital requirement = $57M.
4. **Risk-Adjusted Performance Measurement:** Evaluate portfolio profitability net of credit losses. A portfolio earning $50M in revenue with $12M EL yields risk-adjusted return = $38M, or 3.8% on $1B exposure.
5. **Regulatory Capital Compliance:** Basel III requirements mandate capital ratios (Common Equity Tier 1, Tier 1, Total Capital) based on Risk-Weighted Assets (RWA). EL inputs directly influence RWA calculations under both Standardized and Advanced Internal Ratings-Based (IRB) approaches.

**Prerequisites:** Understanding of EL requires competency in:
- **Probability Theory:** Conditional probabilities, independence assumptions, Bayes' theorem for updating default probabilities with new information
- **Expected Value Calculations:** Linearity of expectation, portfolio aggregation techniques
- **Default Probability (PD) Estimation:** Through-the-cycle vs. point-in-time methods, migration matrix modeling, default curve extraction from bond spreads
- **Loss Given Default (LGD) Recovery:** Collateral valuation, seniority structure, economic cycle effects on recovery rates
- **Exposure at Default (EAD):** Credit line drawdown behavior, credit conversion factors (CCF), mark-to-market for derivatives
- **Portfolio Risk Models:** Correlation dynamics (Gaussian, Student-t, copula models), concentration risk measurement, stress testing methodologies

## 2. Comparative Framing
| Loss Metric | Expected Loss | Unexpected Loss | Credit VaR | Stress Loss |
|-------------|---------------|-----------------|-----------|-----------|
| **Statistic** | Mean/first moment | Std. deviation/volatility | Percentile (e.g., 99.9%) | Tail scenario |
| **Horizon** | Annual (typical) | Over same horizon | Over same horizon | Extreme scenario |
| **Use** | Pricing, provisioning | Capital requirement | Regulatory capital | Stress testing |
| **Correlation** | Portfolio-level average | Portfolio concentration | Full correlation structure | Assumes break |

## 3. Examples + Counterexamples

**Simple Example:**  
Loan characteristics: PD = 2%, LGD = 40%, EAD = $50K (principal outstanding). Calculation: $EL = 0.02 \times 0.40 \times $50K = $400$. 

Practical application: A bank with a portfolio of 1,000 similar loans ($50M aggregate exposure) expects to lose $400K annually. This reserve is charged to the P&L as an allowance for credit losses, and the capital allocation team sets aside economic capital to cover tail losses beyond $400K (e.g., 99.9% VaR might be $1.2M, requiring $800K economic capital buffer above EL).

**Failure Case:**  
Portfolio aggregation error. A bank evaluates its small business loan portfolio of 500 loans, each with $150K EAD. Naively, sum of individual ELs = $150K × 0.015 × 0.45 × 500 = $5.06M. However, assuming zero correlation is unrealistic; empirically, default correlations rise from 0.08 in normal times to 0.25 during recessions. 

Correct approach using Vasicek model with 15% correlation: Portfolio EL = $5.06M, but portfolio volatility increases to approximately $2.8M (std. dev.), requiring 99.9% VaR = $5.06M + 2.33 × $2.8M = $11.6M. The bank initially reserved only $5.06M, creating a $6.5M capital shortfall during downturn—exactly what occurred with commercial real estate portfolios in 2008-2009.

**Edge Case - Government Bond "Safety" Fallacy:**  
A portfolio manager holds $500M in AAA-rated sovereign bonds (Greece, Portugal) with PD = 0.05% based on historical default rates. Expected loss: $500M × 0.0005 × 0.60 = $150K. 

Apparent zero risk masks tail event: During the 2010-2012 eurozone crisis, Greek bonds lost 50% of value before recovery, equivalent to LGD ≈ 50%. Actual realized loss >> EL because tail probability spike from 0.05% to 5% within months was not captured in historical data. This illustrates the key limitation: EL = statistical mean, not a maximum loss bound.

### Technical Counterexample: Hidden Correlation Effects in "Diversified" Portfolios

**The Misconception:** Many practitioners believe that geographic or sector diversification eliminates correlation risk. A typical argument: "We have 40% in real estate, 30% in manufacturing, 20% in retail, and 10% in healthcare, across 15 countries. Correlation risk is minimal."

**Why This Fails:** Empirical evidence from credit events demonstrates that macro factors (interest rates, unemployment, equity volatility) create correlated defaults across seemingly uncorrelated segments.

**Concrete Numerical Example:**

*Scenario 1: Normal Economic Conditions (Correlation = 0.05)*
- Portfolio composition: 1,000 loans, $100M exposure, average PD = 1.2%, LGD = 45%
- Loan-level EL: $100M × 0.012 × 0.45 = $540K
- Portfolio EL (0% correlation assumed): $540K
- Portfolio volatility (σ): $450K
- 99.9% VaR: $540K + 3.09 × $450K = $2.93M
- Capital requirement: $2.39M ($2.93M - $540K)

*Scenario 2: Financial Crisis Spike (Correlation = 0.25)*
When the Fed raises rates by 200bp for inflation control (as in 2022), default correlations increase:
- PD increases: From 1.2% to 2.1% (75% jump due to cyclical deterioration)
- LGD increases: From 45% to 52% (collateral values decline, recovery efficiency worsens)
- Calculated EL: $100M × 0.021 × 0.52 = $1.09M (doubling from $540K)
- Portfolio volatility increases: To $1.8M (due to higher correlation and individual volatility)
- 99.9% VaR: $1.09M + 3.09 × $1.8M = $6.66M
- Capital requirement: $5.57M

**The Capital Gap:** Bank expected $2.39M economic capital but requires $5.57M—a **$3.18M shortfall (133% increase)**. The "diversified" portfolio compressed default correlations in normal times but experienced severe correlation clustering during the stress event.

**Real-World Parallel:** Banks in 2008 held portfolios marked 40-45% in commercial real estate loans across 50 properties thought to be geographically diversified (West Coast, Midwest, Northeast, South). When the housing market collapsed, default correlations spiked to 0.35-0.45, creating correlated losses in neighborhoods with supposedly independent risk drivers. A $2B portfolio with perceived 1.5% portfolio EL = $30M actually experienced losses approaching $85M (2.8% realized rate), validating the counterexample.

**Lesson:** EL estimation must explicitly model correlation dynamics and stress-test against tail scenarios where correlations increase 3-5x from baseline assumptions.

## 4. Layer Breakdown

### Expected Loss Framework Architecture
```
Expected Loss Framework:
├─ Foundational Components:
│  ├─ Probability of Default (PD): Likelihood of non-payment within horizon
│  │  ├─ Point-in-time (PIT): Current economic conditions (2-5%)
│  │  └─ Through-the-cycle (TTC): Average over 5-7 year cycle (1-3%)
│  ├─ Loss Given Default (LGD): Recovery shortfall after default
│  │  ├─ Senior secured: 15-30% LGD (80-85% recovery)
│  │  ├─ Senior unsecured: 35-50% LGD (50-65% recovery)
│  │  └─ Subordinated: 60-80% LGD (20-40% recovery)
│  └─ Exposure at Default (EAD): Amount outstanding at default
│
├─ Basic Formula (Loan-Level):
│  └─ $EL_{loan} = PD \times LGD \times EAD$
│     Example: $50K loan, PD=2%, LGD=40% → EL = $400
│
├─ Portfolio Aggregation:
│  ├─ Independent assumption: $EL_{portfolio} = \sum_{i=1}^{n} EL_i$
│  ├─ Negative correlation: $EL_{portfolio} < \sum EL_i$ (diversification benefit)
│  ├─ Positive correlation: $EL_{portfolio} > \sum EL_i$ (concentration risk)
│  └─ Correlation strength: Typically 0.05-0.15 normal times, 0.25-0.50 crisis
│
├─ Time Dimension & Discounting:
│  ├─ Single period (12-month): Standard regulatory horizon
│  ├─ Multi-period (loan life): Sum discounted losses year-by-year
│  │  └─ $EL_{total} = \sum_{t=1}^{T} \frac{EL_t}{(1+r)^t}$
│  │  └─ Discount rate typically 5-8% (cost of capital)
│  └─ Example: 5-year amortizing loan
│     ├─ Year 1: $EL_1 = $50K × 0.02 × 0.40 = $400
│     ├─ Year 2: $EL_2 = $40K × 0.025 × 0.41 = $410
│     ├─ Year 3: $EL_3 = $30K × 0.03 × 0.42 = $378
│     ├─ Year 4: $EL_4 = $20K × 0.025 × 0.41 = $205
│     ├─ Year 5: $EL_5 = $10K × 0.02 × 0.40 = $80
│     └─ PV(total EL) = $400/1.06 + $410/1.06² + ... = $1,340
│
├─ Parameterization Adjustments:
│  ├─ Economic cycle adjustment: Calibrate PD/LGD forward-looking
│  ├─ Stress scenarios: Recession (PD +100%, LGD +10%), Severe (-200%, +20%)
│  ├─ Model parameter uncertainty: ±0.5% absolute PD variance bands
│  └─ Macroeconomic overlays: Unemployment rate, GDP growth, credit spreads
│
├─ EL Components Interaction:
│  ├─ PD effect on EL: Linear (doubling PD doubles EL)
│  ├─ LGD effect on EL: Linear (increasing LGD to 60% increases EL by 50%)
│  └─ EAD effect on EL: Linear (decreasing drawdown CCF reduces EL)
│
├─ Accounting & Booking Standards:
│  ├─ IFRS 9 Stage Classification:
│  │  ├─ Stage 1 (Low risk): 12-month ECL, minimal loss provision
│  │  ├─ Stage 2 (Monitoring): Lifetime ECL at lower probability
│  │  └─ Stage 3 (Impaired): Lifetime ECL at higher probability, potential write-off
│  ├─ CECL (US Standard): Day-1 full lifetime ECL measurement
│  └─ Provisioning formula: Allowance for Credit Loss (ACL) = PV(All expected cash shortfalls)
│
├─ Pricing & Revenue Recognition:
│  ├─ Credit spread = EL + Operating costs + Target profit margin
│  │  └─ Formula: $Spread = EL_{yield} + Ops_{%} + ROE_{hurdle}$
│  │  └─ Example: 1% EL + 0.5% ops + 1% ROE hurdle = 250bp spread minimum
│  ├─ Loan pricing model:
│  │  ├─ Base rate (SOFR): 5.50%
│  │  ├─ Credit spread: 2.50% (from EL modeling)
│  │  ├─ Operating cost: 0.50%
│  │  └─ All-in rate: 8.50%
│  └─ Yield/return = All-in rate - EL realized = 8.5% - 1% = 7.5% (target)
│
├─ Capital Requirement Framework:
│  ├─ Economic Capital: UL + EL adjustment
│  │  └─ $EC_{99.9\%} = EL + (3.09 \times \sigma_{portfolio})$
│  ├─ Regulatory Capital (Basel III):
│  │  ├─ Standardized approach: EL → risk weight (20%-150%) → RWA
│  │  ├─ IRB approach: PD/LGD/EAD → Foundation (fixed maturity) → IRB Advanced
│  │  └─ Risk weight function: $RW = (PD, LGD, Maturity) \rightarrow$ Lookup table or formula
│  └─ Total Capital Requirement: RWA × 10.5% (8% Pillar 1 + 2.5% buffer)
│
└─ Uses & Applications:
   ├─ Loan Pricing: Set interest rate to recover costs + losses
   ├─ Provisioning: Book allowance on financial statements
   ├─ Capital Allocation: Assign economic capital to business units
   ├─ Portfolio Optimization: Adjust concentrations to reduce EL volatility
   ├─ Stress Testing: CCAR/DFAST scenarios mandate EL under recession
   ├─ Risk-Adjusted Performance: RAROC, RAROCE, EVA calculations
   └─ Regulatory Reporting: CRE templates, Pillar 3 disclosures
```

### Key Dependencies & Calculation Integration

The Expected Loss framework operates within a complex ecosystem of interconnected risk components that reinforce and amplify each other during stress periods. Understanding these dependencies is critical for robust credit risk management.

**PD-LGD Correlation:** Contrary to initial assumptions of independence, default probability and loss given default exhibit strong positive correlation. During recessions (negative GDP growth, rising unemployment), both increase simultaneously:
- In normal times: PD = 1.5%, LGD = 40%, Correlation(PD,LGD) ≈ 0.05
- In stress times: PD = 3.2%, LGD = 55%, Correlation(PD,LGD) ≈ 0.40
- Impact: A naive model assuming independence underestimates portfolio EL by 8-15% during downturns

**EAD-Default Correlation (Pro-Cyclical Drawdown):** Firms experiencing financial distress systematically draw on available credit lines as internal cash reserves deplete. Basel III parametrization recognizes this:
- Committed lines: CCF = 85% (firms draw 85% of undrawn amount near default)
- Uncommitted lines: CCF = 0% (banks can refuse to fund under stress)
- Example: $10M committed line, $6M drawn, CCF = 85% → EAD = $6M + 0.85 × $4M = $9.4M (94% utilization at default vs. initial 60%)

**Maturity Effect on Loss Severity:** Longer-dated exposures carry embedded losses from the trajectory of probability and recovery rates. A 5-year corporate loan shows increasing cumulative default probability, while recovery rates may decrease due to prolonged workout periods and collateral deterioration.

**Portfolio Concentration Risk:** The framework magnifies when correlation increases with portfolio concentration. Extreme concentration in single sector/geography can multiply expected losses:
- Diversified portfolio (40 sectors, 20 countries): Effective correlation = 0.08
- Concentrated portfolio (5 sectors, 3 countries): Effective correlation = 0.25
- Same average PD/LGD, but portfolio volatility increases 3.5x with concentration (same EL, higher capital requirement)

**Procyclical Feedback Loops:** EL estimates feed into capital requirements, which constrain lending capacity, potentially triggering supply-side credit contractions that further increase default probabilities—creating a reinforcing cycle that amplifies downturns (as experienced 2008-2009 and 2020 COVID crisis).

## 5. Challenge Round
When is EL calculation problematic?
- **Correlation instability**: PD/LGD correlation changes with economic cycle; historical correlations break down in crisis
- **Tail risk**: EL misses extreme losses (e.g., 99.9% VaR >> EL in crisis scenarios)
- **Model risk**: Small changes in PD assumption (1.5% → 2%) can significantly increase EL
- **Portfolio heterogeneity**: Single correlation assumption oversimplifies multi-segment portfolios with different default drivers
- **Forward guidance uncertainty**: Macro adjustments (recession probability) remain highly subjective despite stress testing frameworks
- **Parameter uncertainty**: Historical calibration windows miss structural breaks and regime changes in credit markets

## 6. Key References

1. **Basel Committee on Banking Supervision (2023).** "Capital Requirements - Credit Risk: Standardised and IRB Approaches." [*Basel III Framework*](https://www.bis.org/basel_framework/chapter/CRE/20.htm). Bank for International Settlements. Regulatory foundation for EL calculation in credit risk-weighted assets, covers Standardized and IRB methodologies with detailed CCF tables and risk weight formulas.

2. **Vasicek, O. (1991).** "Limiting Portfolio Losses." *Risk Management: Practices and Regulations*, JP Morgan. Foundational single-factor model for portfolio credit loss distributions, enabling portfolio-level EL and UL calculations with correlation assumptions (0.05-0.50 range).

3. **International Accounting Standards Board (2014).** "IFRS 9 Financial Instruments: Expected Credit Loss Impairment Model." [*IFRS 9 Standard*](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/), effective 2018. Accounting framework requiring day-1 ECL provisioning through 12-month and lifetime ECL stages, with forward-looking adjustments and macroeconomic scenarios.

4. **Federal Reserve Board (2021).** "Consolidated Supervision and Regulation of Bank Holding Companies." *Federal Reserve Supervisory Guidance on Stress Testing and Capital Planning*, FR 14-84. Covers CCAR/DFAST methodology requiring projected EL under recession/adverse/severely adverse scenarios, mandatory for large US bank holding companies (>$50B assets).

5. **Jorion, P. (2006).** *Value at Risk: The New Benchmark for Managing Financial Risk.* 3rd Edition, McGraw-Hill. Comprehensive treatment of portfolio loss distributions, VaR/CVaR methodology, and EL/UL separation with practical Monte Carlo simulation techniques and backtesting approaches.

6. **Credit Suisse Research (2017).** "Credit Risk Modeling: Theory and Applications." Credit Suisse Publications. Advanced multi-factor models extending Vasicek, addresses correlation dynamics and macroeconomic overlay approaches for stressed EL estimation in complex portfolios.

7. **Financial Stability Board (2019).** "Regulatory Developments in Credit Risk Modelling and Validation." *FSB Report to G20*. Emphasizes model risk in EL estimation, governance frameworks for PD/LGD calibration, and backtesting requirements for internal models under BCBS 239 principles.

---
**Status:** Primary risk metric for credit portfolio management | **Complements:** PD, LGD, EAD, Pricing, Provisioning
