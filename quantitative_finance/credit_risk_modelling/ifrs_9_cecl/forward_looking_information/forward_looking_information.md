# Forward-Looking Information & Macroeconomic Scenarios

## 1. Concept Skeleton
**Definition:** Incorporation of reasonable and supportable forecasts of future economic conditions into ECL estimates; macroeconomic scenario generation (base, adverse, upside) with probability weighting; aligns PD/LGD with forward economic paths  
**Purpose:** Distinguish IFRS 9 from incurred loss model (IAS 39); timely loss recognition in downturns; countercyclical provisioning; avoid "too little, too late"; regulatory compliance  
**Prerequisites:** Macro forecasting models, scenario design, econometric relationships (PD/LGD vs GDP, unemployment), probability weighting, Monte Carlo simulation, stress testing

## 2. Comparative Framing
| Scenario Type | Probability Weight | GDP Growth | Unemployment | Use Case | PD Impact | LGD Impact |
|---------------|-------------------|------------|--------------|----------|-----------|------------|
| **Base** | 50-60% | Consensus forecast (~2-3%) | Stable (~5%) | Most likely path | Baseline PD | Baseline LGD |
| **Adverse** | 20-30% | Recession (-2% to 0%) | Rising (+2-3pp) | Economic downturn | +50-100% PD | +20-40% LGD |
| **Severe Adverse** | 5-10% | Deep recession (-5%) | High (+5pp) | Tail risk (crisis) | +200-300% PD | +50-80% LGD |
| **Upside** | 10-20% | Boom (+4-5%) | Low (-2pp) | Benign conditions | -30-50% PD | -20-30% LGD |

## 3. Examples + Counterexamples

**Scenario-Weighted ECL:**  
Base (50%): PD = 2%, ECL = $1M. Adverse (30%): PD = 4%, ECL = $2M. Upside (20%): PD = 1%, ECL = $500k. Weighted ECL = 0.5Ã—$1M + 0.3Ã—$2M + 0.2Ã—$500k = $1.2M.

**Mortgage Portfolio (House Price Sensitivity):**  
Base: House prices +2%/yr, LGD = 30%. Adverse: House prices -10%, LGD = 50% (negative equity). Upside: House prices +5%, LGD = 20%. Weighted LGD = 0.5Ã—30% + 0.3Ã—50% + 0.2Ã—20% = 34%.

**Energy Loan (Oil Price Shock):**  
Base: Oil $80/bbl, PD = 3%. Adverse: Oil $40/bbl (collapse), PD = 15% (sector distress). Upside: Oil $100/bbl, PD = 1%. Management overlay: Weight adverse 50% (oil volatility high) â†’ Weighted PD = 8.3%.

**COVID-19 Example:**  
Pre-COVID (2019): Base PD = 1.5%, ECL = $500k. COVID scenario (March 2020): Adverse PD = 5%, weighted PD = 3%, ECL = $1M (doubled). Model shock; management overlays adjust sectors (airlines, retail hit hardest).

**Failure Case (No Forward-Looking):**  
Bank uses historical average PD (2% over 10 years). Recession hits; actual PD spikes to 6%. ECL understated by 3Ã— â†’ Regulatory criticism; late loss recognition.

## 4. Layer Breakdown
```
Forward-Looking Information Framework:

â”œâ”€ IFRS 9 Requirements:
â”‚   â”œâ”€ Principle: ECL must reflect reasonable and supportable information
â”‚   â”‚   â””â”€ IFRS 9.5.5.17: "Consider past events, current conditions, and forecasts"
â”‚   â”œâ”€ Forward-Looking Obligation:
â”‚   â”‚   â”œâ”€ Cannot rely solely on historical data (backward-looking)
â”‚   â”‚   â”œâ”€ Must incorporate forecasts of future economic conditions
â”‚   â”‚   â””â”€ Forecasts must be unbiased (not excessively prudent or optimistic)
â”‚   â”œâ”€ Reasonable and Supportable Horizon:
â”‚   â”‚   â”œâ”€ Explicit forecasts: Typically 3-5 years (consensus forecast horizon)
â”‚   â”‚   â”œâ”€ Beyond explicit: Revert to long-run average (through-the-cycle)
â”‚   â”‚   â””â”€ Example: Year 1-3 forecast; Year 4+ use historical mean PD
â”‚   â””â”€ Multiple Scenarios Required:
â”‚       â”œâ”€ Cannot use single point estimate (biased)
â”‚       â””â”€ Probability-weighted scenarios (base, adverse, upside minimum)
â”‚
â”œâ”€ Macroeconomic Scenario Design:
â”‚   â”œâ”€ Base Scenario:
â”‚   â”‚   â”œâ”€ Definition: Most likely economic path (modal forecast)
â”‚   â”‚   â”œâ”€ Source: Consensus economics; central bank forecasts; internal macro team
â”‚   â”‚   â”œâ”€ Horizon: 3-5 years explicit; beyond = revert to long-run mean
â”‚   â”‚   â”œâ”€ Variables:
â”‚   â”‚   â”‚   â”œâ”€ GDP growth: ~2-3% (developed markets)
â”‚   â”‚   â”‚   â”œâ”€ Unemployment: ~4-5% (natural rate)
â”‚   â”‚   â”‚   â”œâ”€ Interest rates: Central bank policy path
â”‚   â”‚   â”‚   â”œâ”€ Inflation: ~2% (central bank target)
â”‚   â”‚   â”‚   â””â”€ Asset prices: Equity indices, house prices, commodity prices
â”‚   â”‚   â””â”€ Probability Weight: Typically 50-60%
â”‚   â”‚
â”‚   â”œâ”€ Adverse Scenario:
â”‚   â”‚   â”œâ”€ Definition: Economic downturn (recession)
â”‚   â”‚   â”œâ”€ Severity: Moderate recession (GDP -2% to 0% for 1-2 years)
â”‚   â”‚   â”œâ”€ Variables:
â”‚   â”‚   â”‚   â”œâ”€ GDP growth: -1% to -2% (Year 1), 0% (Year 2), +1% (Year 3 recovery)
â”‚   â”‚   â”‚   â”œâ”€ Unemployment: +2-3 percentage points above base
â”‚   â”‚   â”‚   â”œâ”€ House prices: -10% to -15% (peak-to-trough)
â”‚   â”‚   â”‚   â”œâ”€ Equity markets: -20% to -30%
â”‚   â”‚   â”‚   â””â”€ Corporate profitability: Compressed margins; revenue decline
â”‚   â”‚   â”œâ”€ Probability Weight: 20-30% (plausible but not most likely)
â”‚   â”‚   â””â”€ Calibration: Historical recession frequency (1 in 10 years â†’ ~10-15% weight)
â”‚   â”‚
â”‚   â”œâ”€ Severe Adverse Scenario:
â”‚   â”‚   â”œâ”€ Definition: Deep recession / financial crisis (tail risk)
â”‚   â”‚   â”œâ”€ Severity: 2008-style crisis (GDP -5%, unemployment +5pp, house prices -30%)
â”‚   â”‚   â”œâ”€ Variables:
â”‚   â”‚   â”‚   â”œâ”€ GDP growth: -4% to -5% (Year 1), -2% (Year 2), slow recovery
â”‚   â”‚   â”‚   â”œâ”€ Unemployment: +5-7 percentage points
â”‚   â”‚   â”‚   â”œâ”€ House prices: -30% to -40%
â”‚   â”‚   â”‚   â”œâ”€ Equity markets: -50%+
â”‚   â”‚   â”‚   â””â”€ Credit spreads: Widen dramatically (risk aversion)
â”‚   â”‚   â”œâ”€ Probability Weight: 5-10% (rare but possible)
â”‚   â”‚   â””â”€ Use: Stress testing; regulatory capital adequacy (CCAR, EBA)
â”‚   â”‚
â”‚   â”œâ”€ Upside Scenario:
â”‚   â”‚   â”œâ”€ Definition: Economic boom (above-trend growth)
â”‚   â”‚   â”œâ”€ Variables:
â”‚   â”‚   â”‚   â”œâ”€ GDP growth: +4-5% (strong expansion)
â”‚   â”‚   â”‚   â”œâ”€ Unemployment: -1-2 percentage points (tight labor market)
â”‚   â”‚   â”‚   â”œâ”€ House prices: +10% (asset price appreciation)
â”‚   â”‚   â”‚   â”œâ”€ Equity markets: +20-30%
â”‚   â”‚   â”‚   â””â”€ Corporate profits: Strong earnings growth
â”‚   â”‚   â”œâ”€ Probability Weight: 10-20%
â”‚   â”‚   â””â”€ Impact: Lower PD (stronger borrower finances); lower LGD (higher collateral values)
â”‚   â”‚
â”‚   â””â”€ Scenario Path (Time Series):
â”‚       â”œâ”€ Quarter-by-quarter or year-by-year forecasts
â”‚       â”œâ”€ Example (Base GDP path): Year 1: 2.5%, Year 2: 2.8%, Year 3: 2.3%, Year 4+: 2.5% (long-run)
â”‚       â””â”€ Consistency: Scenarios must be internally consistent (e.g., GDP down â†’ unemployment up)
â”‚
â”œâ”€ Econometric Relationships (PD/LGD Sensitivity to Macro):
â”‚   â”œâ”€ PD Models:
â”‚   â”‚   â”œâ”€ Regression Specification: log(PD_t) = Î± + Î²â‚Â·GDP_t + Î²â‚‚Â·Unemp_t + Îµ_t
â”‚   â”‚   â”œâ”€ Expected Signs:
â”‚   â”‚   â”‚   â”œâ”€ Î²â‚ < 0: Higher GDP â†’ Lower PD (stronger economy reduces defaults)
â”‚   â”‚   â”‚   â””â”€ Î²â‚‚ > 0: Higher unemployment â†’ Higher PD (job losses increase defaults)
â”‚   â”‚   â”œâ”€ Estimation: Historical regression using panel data (PD vs macro over time)
â”‚   â”‚   â”œâ”€ Calibration: Ensure coefficients economically sensible (elasticity checks)
â”‚   â”‚   â””â”€ Example: 1% GDP decline â†’ +0.5pp PD increase (semi-elasticity)
â”‚   â”‚
â”‚   â”œâ”€ LGD Models:
â”‚   â”‚   â”œâ”€ Regression: LGD_t = Î± + Î²â‚Â·HousePrice_t + Î²â‚‚Â·RecoveryRate_t + Îµ_t
â”‚   â”‚   â”œâ”€ Expected Signs:
â”‚   â”‚   â”‚   â”œâ”€ Î²â‚ < 0: Higher house prices â†’ Lower LGD (collateral value higher)
â”‚   â”‚   â”‚   â””â”€ Recession indicator: Downturn â†’ +10-20pp LGD (fire sales)
â”‚   â”‚   â”œâ”€ Downturn LGD: Basel requirement; use adverse scenario LGD for capital
â”‚   â”‚   â””â”€ Example: House prices -10% â†’ LGD increases from 30% to 40%
â”‚   â”‚
â”‚   â”œâ”€ Segment-Specific Models:
â”‚   â”‚   â”œâ”€ Mortgages: PD = f(unemployment, house prices, interest rates)
â”‚   â”‚   â”œâ”€ Corporate: PD = f(GDP, sector performance, credit spreads)
â”‚   â”‚   â”œâ”€ Credit Cards: PD = f(unemployment, consumer confidence, delinquency rates)
â”‚   â”‚   â””â”€ Energy: PD = f(oil prices, capex, leverage)
â”‚   â”‚
â”‚   â””â”€ Non-Linearity:
â”‚       â”œâ”€ Tail Risk: Severe adverse scenario â†’ Disproportionate PD increase
â”‚       â”œâ”€ Threshold Effects: Unemployment > 10% â†’ PD spikes (mass layoffs)
â”‚       â””â”€ Modeling: Logistic transformation; quantile regression; stress multipliers
â”‚
â”œâ”€ Probability Weighting:
â”‚   â”œâ”€ Formula: ECL = âˆ‘[s] w(s) Ã— ECL(s)
â”‚   â”‚   â”œâ”€ s: Scenario index (base, adverse, upside)
â”‚   â”‚   â”œâ”€ w(s): Probability weight (sum to 1)
â”‚   â”‚   â””â”€ ECL(s): Expected credit loss under scenario s
â”‚   â”œâ”€ Weight Selection:
â”‚   â”‚   â”œâ”€ Expert Judgment: Risk committee consensus; macro team input
â”‚   â”‚   â”œâ”€ Historical Frequency: Recession occurs 1 in 10 years â†’ 10% weight
â”‚   â”‚   â”œâ”€ Market-Implied: Option prices, credit spreads imply risk-neutral probabilities
â”‚   â”‚   â””â”€ Unbiased Requirement: Cannot be excessively prudent (overweight adverse)
â”‚   â”œâ”€ Example Weights:
â”‚   â”‚   â”œâ”€ Stable Environment: Base 60%, Adverse 25%, Upside 15%
â”‚   â”‚   â”œâ”€ Uncertain Environment: Base 40%, Adverse 40%, Upside 20% (higher tail risk)
â”‚   â”‚   â””â”€ Crisis: Base 30%, Adverse 60%, Severe 10% (recession imminent)
â”‚   â”œâ”€ Sensitivity Analysis:
â”‚   â”‚   â”œâ”€ Test ECL with alternative weights (e.g., Â±10% shift)
â”‚   â”‚   â””â”€ Disclose sensitivity in financial statements (IFRS 7)
â”‚   â””â”€ Management Discretion:
â”‚       â”œâ”€ Overlay: Adjust weights for events not in models (e.g., COVID-19)
â”‚       â””â”€ Documentation: Rationale for weight changes; approved by risk committee
â”‚
â”œâ”€ Scenario Generation Process:
â”‚   â”œâ”€ Step 1: Define Horizon (typically 3-5 years explicit forecast)
â”‚   â”œâ”€ Step 2: Select Macro Variables:
â”‚   â”‚   â”œâ”€ Core: GDP, unemployment, inflation, interest rates
â”‚   â”‚   â”œâ”€ Asset Prices: Equity indices, house prices, commodity prices
â”‚   â”‚   â””â”€ Sector-Specific: Oil prices (energy), freight rates (shipping)
â”‚   â”œâ”€ Step 3: Source Forecasts:
â”‚   â”‚   â”œâ”€ Base: Consensus Economics; Bloomberg; central bank forecasts
â”‚   â”‚   â”œâ”€ Adverse: Historical recessions; stress test scenarios (CCAR, EBA)
â”‚   â”‚   â””â”€ Upside: Optimistic consensus; historical expansion periods
â”‚   â”œâ”€ Step 4: Ensure Internal Consistency:
â”‚   â”‚   â”œâ”€ GDP â†“ â†’ Unemployment â†‘, Corporate profits â†“, House prices â†“
â”‚   â”‚   â”œâ”€ Check: Okun's Law (GDP vs unemployment relationship)
â”‚   â”‚   â””â”€ Tools: VAR (Vector Autoregression) models; structural macro models
â”‚   â”œâ”€ Step 5: Map to PD/LGD:
â”‚   â”‚   â”œâ”€ Apply econometric models: PD(scenario) = f(GDP_scenario, Unemp_scenario)
â”‚   â”‚   â””â”€ Validate: Compare scenario PD to historical recession PDs
â”‚   â”œâ”€ Step 6: Calculate ECL by Scenario:
â”‚   â”‚   â”œâ”€ Run ECL model for each scenario s
â”‚   â”‚   â””â”€ ECL(s) = EAD Ã— PD(s) Ã— LGD(s) Ã— DF
â”‚   â”œâ”€ Step 7: Probability Weighting:
â”‚   â”‚   â””â”€ ECL_final = âˆ‘ w(s) Ã— ECL(s)
â”‚   â””â”€ Step 8: Governance:
â”‚       â”œâ”€ Risk committee approval of scenarios + weights
â”‚       â”œâ”€ Quarterly review; adjust if economic outlook shifts
â”‚       â””â”€ Document assumptions; audit trail
â”‚
â”œâ”€ Reasonable and Supportable Horizon:
â”‚   â”œâ”€ Explicit Forecasts (Years 1-3):
â”‚   â”‚   â”œâ”€ Use econometric models; external forecasts
â”‚   â”‚   â””â”€ High confidence; detailed quarter-by-quarter paths
â”‚   â”œâ”€ Beyond Explicit (Years 4+):
â”‚   â”‚   â”œâ”€ Revert to long-run average (through-the-cycle)
â”‚   â”‚   â”œâ”€ Rationale: Low confidence in long-term forecasts; avoid spurious precision
â”‚   â”‚   â””â”€ Implementation: Linear reversion over 1-2 years to TTC mean
â”‚   â”œâ”€ Example (PD Path):
â”‚   â”‚   â”œâ”€ Year 1: 2.5% (explicit forecast)
â”‚   â”‚   â”œâ”€ Year 2: 3.0% (explicit)
â”‚   â”‚   â”œâ”€ Year 3: 2.8% (explicit)
â”‚   â”‚   â”œâ”€ Year 4: 2.5% (revert to TTC mean = 2.2%)
â”‚   â”‚   â””â”€ Year 5+: 2.2% (TTC mean)
â”‚   â””â”€ IFRS 9 Guidance: "Consider uncertainty in longer horizons; revert to historical average"
â”‚
â”œâ”€ Management Overlays:
â”‚   â”œâ”€ Definition: Expert judgment adjustments to model-driven ECL
â”‚   â”œâ”€ Use Cases:
â”‚   â”‚   â”œâ”€ Model Limitations: Models miss new risk (e.g., pandemic not in historical data)
â”‚   â”‚   â”œâ”€ Emerging Risks: Geopolitical shocks, regulatory changes, climate risk
â”‚   â”‚   â”œâ”€ Data Quality: Sparse data segments (e.g., new product launches)
â”‚   â”‚   â””â”€ Scenario Inadequacy: Current scenarios don't capture ongoing developments
â”‚   â”œâ”€ Examples:
â”‚   â”‚   â”œâ”€ COVID-19: Models calibrated pre-pandemic; overlay to increase adverse weight
â”‚   â”‚   â”œâ”€ Brexit: UK exposures; overlay to reflect political uncertainty
â”‚   â”‚   â””â”€ Climate Risk: Real estate in flood zones; overlay to adjust LGD
â”‚   â”œâ”€ Governance:
â”‚   â”‚   â”œâ”€ Documented rationale; quantified impact
â”‚   â”‚   â”œâ”€ Approval by risk committee; audit scrutiny
â”‚   â”‚   â”œâ”€ Temporary: Review quarterly; remove when models updated
â”‚   â”‚   â””â”€ Disclosure: IFRS 7 requires overlay disclosure
â”‚   â””â”€ Challenges:
â”‚       â”œâ”€ Subjectivity: Risk of excessive conservatism or optimism
â”‚       â””â”€ Model Risk: Overreliance on overlays undermines model credibility
â”‚
â””â”€ Implementation & Systems:
    â”œâ”€ Scenario Platform:
    â”‚   â”œâ”€ Generate macro paths (base, adverse, upside)
    â”‚   â”œâ”€ Store historical scenarios; version control
    â”‚   â””â”€ Quarterly updates; approval workflow
    â”œâ”€ Econometric Models:
    â”‚   â”œâ”€ PD/LGD sensitivity models: Regression; calibration
    â”‚   â””â”€ Validation: Backtesting; compare forecast to actual
    â”œâ”€ ECL Engine:
    â”‚   â”œâ”€ Calculate ECL by instrument Ã— scenario
    â”‚   â”œâ”€ Probability weighting: Aggregate across scenarios
    â”‚   â””â”€ Output: Weighted ECL; scenario breakdown
    â”œâ”€ Reporting:
    â”‚   â”œâ”€ IFRS 7 Disclosure: Scenarios used; weights; sensitivity
    â”‚   â”œâ”€ Management Reports: ECL by scenario; scenario impact analysis
    â”‚   â””â”€ Regulatory: Stress test alignment (CCAR, EBA scenarios)
    â””â”€ Governance:
        â”œâ”€ Quarterly scenario review; update weights/paths if outlook changes
        â”œâ”€ Annual model validation; recalibrate econometric models
        â”œâ”€ Audit: External auditors review scenarios; challenge weights
        â””â”€ Regulatory: Supervisors assess forward-looking adequacy
```

**Key Insight:** IFRS 9 requires forward-looking ECL; multiple probability-weighted scenarios (base, adverse, upside); econometric models link PD/LGD to macro variables (GDP, unemployment, house prices); management overlays for model gaps; unbiased weighting; 3-5 year explicit forecast then revert to TTC.

## 5. Challenge Round
When forward-looking frameworks fail or introduce complexity:
- **Forecast Uncertainty (Long Horizon)**: 5-year GDP forecast highly uncertain â†’ Model generates spurious precision; solution: Revert to TTC mean after Year 3; widen confidence intervals; scenario range (not point estimates)
- **Model Misspecification**: PD model calibrated 2000-2019 (no pandemic); COVID hits â†’ Model useless; solution: Management overlays; update models with new data; stress test tail scenarios
- **Procyclicality**: Recession forecast â†’ Higher ECL â†’ Banks reduce lending â†’ Worsens recession (feedback loop); solution: Smoothing mechanisms (TTC overlays); regulatory forbearance (temporary); countercyclical buffers
- **Scenario Weights Arbitrary**: CFO increases adverse weight 30% â†’ 40% â†’ Provisions up 15%; subjective; solution: Document rationale; independent validation; sensitivity disclosure
- **Data Limitations (Emerging Markets)**: Limited historical recession data â†’ Cannot calibrate macro models; solution: Use developed market analogues; expert judgment; conservative assumptions
- **Cliff Effects at Horizon**: Explicit forecast ends Year 3 â†’ Sudden PD jump to TTC mean; solution: Smooth transition over 1-2 years; linear interpolation

## 6. Key References
- [IFRS 9 Financial Instruments (Section 5.5.17)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official standard; forward-looking information requirements; reasonable and supportable forecasts
- [IMF: Loan Loss Provisioning and Economic Slowdowns (2018)](https://www.imf.org/en/Publications/WP/Issues/2018/07/23/Loan-Loss-Provisioning-and-Economic-Slowdowns-Too-Little-Too-Late-46053) - Empirical analysis; forward-looking ECL vs incurred loss; procyclicality; policy implications
- [EBA Report on IFRS 9 Implementation (2019)](https://www.eba.europa.eu/eba-publishes-report-ifrs-9-implementation-eu-institutions) - European Banking Authority survey; scenario practices; weight selection; sensitivity analysis

---
**Status:** IFRS 9 Core Methodology | **Complements:** Three-Stage Approach, Expected Credit Loss Models, SICR
