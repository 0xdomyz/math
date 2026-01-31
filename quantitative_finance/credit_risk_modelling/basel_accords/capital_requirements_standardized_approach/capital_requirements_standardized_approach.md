# Capital Requirements & Standardized Approach (Basel III)

## 1. Concept Skeleton
**Definition:** Regulatory framework specifying minimum capital requirements for credit risk, market risk, operational risk; Standardized Approach uses external ratings (AAA-D) for credit risk; Advanced Internal Ratings-Based (IRB) allows bank models (PD/LGD/EAD); Market risk uses expected shortfall instead of VaR  
**Purpose:** Align bank incentives with actual risk exposure; prevent RWA gaming; apply consistent risk-weighting across institutions; capture tail risk in capital charges  
**Prerequisites:** Risk-Weighted Assets (RWA), credit risk fundamentals, market risk, operational risk measurement, external ratings, PD/LGD/EAD

## 2. Comparative Framing
| Aspect | Standardized Approach | Foundation IRB | Advanced IRB | Key Tradeoff |
|--------|---------------------|-----------------|---------------|------------|
| **Credit Risk RW Source** | External ratings (S&P, Moody's) | Bank estimates PD; regulator provides LGD/EAD | Bank estimates all (PD/LGD/EAD) | Complexity ↔ Risk-sensitivity |
| **Risk Weight** | 20%-150% by rating | Calculated formula (PD/LGD/EAD dependent) | Calculated formula (bank-specific) | Standardized ↔ Customized |
| **RWA Formula** | RW × Exposure | 1.06×12.5×(PD×LGD + correlation adj.) | 1.06×12.5×(PD×LGD×EAD + ... - discount) | Simple ↔ Refined |
| **Capital Requirement** | 8% × RWA | 8% × RWA (or lower if IRB floor) | 8% × RWA (bound by floor) | Easy compliance ↔ Optimized |
| **Regulatory Approval** | Automatic (published ratings) | Requires IRB model validation | Requires IRB + detailed validation | Fast ↔ Rigorous |
| **Data Requirements** | External ratings only | Internal default history + rating drift | Historical losses, correlation matrices | Low burden ↔ High burden |
| **Fit for Asset Class** | Standardized: Large corporates, sovereigns | Mid-market corporates, loans | Large banks (internal data available) | Cost-effective ↔ Accurate |
| **Game-Ability** | Cliff effects at rating boundaries | Parameter manipulation (PD/LGD underestimation) | Extensive model gaming potential | Transparent ↔ Exploitable |
| **Output Floor** | Baseline (100%) | 72.5% of SA RWA | 72.5% of SA RWA | —— |
| **Typical Capital** | 10-12% RWA | 8-11% RWA | 7-10% RWA | Conservative ↔ Efficient |

## 3. Examples + Counterexamples

**Standardized Approach Example:**  
Bank holds $100M loan to AAA-rated corporate. Risk weight = 20% (S&P AAA). RWA = 0.20 × $100M = $20M. Capital required = 8% × $20M = $1.6M.  
Rating downgrade to BBB: RW = 100%, RWA = $100M, capital = $8M (5x increase overnight). Called "cliff risk."

**Foundation IRB Example (Corporates):**  
Bank uses internal data; PD estimate = 1.5%, LGD = 45% (regulator provides), EAD = $100M.  
RW formula: 1.06 × 12.5 × [1.5% × 45% + √(correlation) × (default stress)] ≈ 60%.  
RWA = 0.60 × $100M = $60M, Capital = $4.8M (much lower than SA's $8M for BBB equivalent).

**Advanced IRB Example (Retail Mortgages):**  
Bank large mortgage portfolio; internal data shows 0.3% default rate, 30% LGD, 85% EAD (loan loss given default; 15% equity cushion).  
RW = 1.06 × 12.5 × [0.3% × 30%] ≈ 1.2% (extremely low—retail portfolios are safer).  
RWA = 0.012 × $500M = $6M, Capital = $0.48M (vs SA's standardized 35% for residential mortgages = $17.5M).  
**Counterexample:** 2008 crisis: Actual mortgage defaults 5%+, LGD 60%+. Advanced IRB models catastrophically underestimated. Lesson: Historical calibration fails in tail risk.

**Market Risk: VaR vs Expected Shortfall (CVaR):**  
Basel II: VaR(99%, 10-day) = $50M. Bank "99% confident losses won't exceed $50M."  
Basel III CVaR: ES(99%, 10-day) = $75M (average of losses in tail 1%). Captures tail severity.  
Result: Basel III capital ≈ 50% higher for trading desks.

**Operational Risk: Standardized Approach:**  
Bank revenue over 3 years: $500M average. OpRisk charge = 12% × $500M = $60M capital required.  
vs Advanced: Historical loss data, 15% loss ratio, internal models → $40M. Gaming: Shift business to lower-indicator revenue streams.

**Leverage Ratio as Binding Floor:**  
Bank: Tier 1 = $40B, Total Assets = $900B. Leverage ratio = $40B / $900B = 4.44%.  
RWA-based: If RWA = $300B (33% of assets, "low-risk"), Capital = 8% × $300B = $24B ✓ (easily met).  
But leverage ratio requires $27B (3% × $900B). Leverage ratio is binding → must hold extra capital.

**Output Floor (72.5%) Example:**  
Bank SA RWA = $400B (standardized). IRB model calculates RWA = $250B (37.5% reduction via gaming).  
Floor: IRB RWA ≥ 72.5% × $400B = $290B. Bank constrained to use $290B.  
**Effect:** Reduces incentive to game PD/LGD but doesn't eliminate. Floor is partial safeguard.

## 4. Layer Breakdown
```
Capital Requirement & Risk Calculation Framework:

├─ Credit Risk Capital Calculation
│  ├─ Standardized Approach (SA)
│  │   ├─ Asset Classification:
│  │   │   ├─ Central Government & Central Banks:
│  │   │   │   ├─ Domestic currency (country): RW = 0% (OeCD member)
│  │   │   │   ├─ Foreign currency: RW = 0-100% (country risk classification)
│  │   │   │   └─ Example: US Treasury 0%, Greece 100%+
│  │   │   ├─ Institutions (Banks, Investment Firms):
│  │   │   │   ├─ Domestic, central bank's home country: RW = 20%
│  │   │   │   ├─ AAA-AA rated: RW = 20%
│  │   │   │   ├─ A-rated: RW = 50%
│  │   │   │   ├─ BBB-rated: RW = 100%
│  │   │   │   ├─ Unrated: RW = 100%
│  │   │   │   └─ BB or below: RW = 150%
│  │   │   ├─ Corporates (Non-Financial):
│  │   │   │   ├─ AAA-AA rated: RW = 20%
│  │   │   │   ├─ A-rated: RW = 50%
│  │   │   │   ├─ BBB-rated: RW = 100%
│  │   │   │   ├─ BB-rated: RW = 100%
│  │   │   │   ├─ B or below: RW = 150%
│  │   │   │   └─ Unrated: RW = 100%
│  │   │   ├─ Retail Exposures:
│  │   │   │   ├─ Residential mortgages (RMBS): RW = 35%
│  │   │   │   ├─ Qualifying revolving (credit cards): RW = 75%
│  │   │   │   ├─ Other retail (auto loans, unsecured personal): RW = 75%
│  │   │   │   └─ All retail pools lower RW (lower individual defaults)
│  │   │   ├─ Equity Exposures:
│  │   │   │   ├─ Direct holdings: RW = 100%
│  │   │   │   ├─ Mutual funds (look-through): RW = rating-dependent 20-1250%
│  │   │   │   └─ Private equity: RW = 190%
│  │   │   └─ Off-Balance-Sheet Exposures:
│  │   │       ├─ Credit commitments: CCF (Credit Conversion Factor) 20%-100%
│  │   │       ├─ Guarantees: 100% CCF
│  │   │       └─ Letter of credit: 20%-100% CCF
│  │   ├─ Credit Risk Mitigation:
│  │   │   ├─ Collateral Haircuts: Adjust exposure for market fluctuations
│  │   │   │   ├─ Cash collateral: 0% haircut
│  │   │   │   ├─ Government bonds (AAA): 0-2% haircut
│  │   │   │   ├─ Investment-grade corporates: 2-4% haircut
│  │   │   │   ├─ Equity: 15-50% haircut (high volatility)
│  │   │   │   └─ Formula: Adjusted Exposure = Exposure - Collateral × (1 - Haircut)
│  │   │   ├─ Guarantees & Credit Derivatives:
│  │   │   │   ├─ Third-party guarantee: Risk weight → guarantor's rating
│  │   │   │   ├─ Example: BBB corporate guaranteed by AAA bank → use 20% RW
│  │   │   │   ├─ Partial coverage: Proportional risk weighting
│  │   │   │   └─ Counterparty concentration: Cap at 25% guarantee
│  │   │   └─ Netting: Reduce gross exposure by collateral/marks
│  │   └─ Capital Charge: 8% × (Adjusted RWA)
│  │
│  ├─ Internal Ratings-Based (IRB) Approach
│  │   ├─ Foundation IRB (F-IRB):
│  │   │   ├─ Bank estimates: Probability of Default (PD)
│  │   │   ├─ Regulator provides: Loss Given Default (LGD), Exposure at Default (EAD)
│  │   │   ├─ Formula (Corporates):
│  │   │   │   RW = 1.06 × 12.5 × {PD × LGD + √[R/(1-R)] × σ × N^{-1}(LGD)}
│  │   │   │   Where:
│  │   │   │   - N^{-1}(LGD) = inverse normal (portfolio tail stress)
│  │   │   │   - R = correlation factor (typically 0.12 for corporates, lower for retail)
│  │   │   │   - σ = asset volatility
│  │   │   ├─ Result: RW typically 15%-50% for corporates (vs 20%-150% SA)
│  │   │   ├─ LGD floor: 45% for corporate unsecured, 35% senior secured
│  │   │   ├─ EAD floor: Minimum 100% of loan balance
│  │   │   └─ Capital charge: 8% × RWA
│  │   │
│  │   ├─ Advanced IRB (A-IRB):
│  │   │   ├─ Bank estimates all: PD, LGD, EAD
│  │   │   ├─ Formula (Corporates - identical to F-IRB but bank-derived LGD/EAD):
│  │   │   │   RW = 1.06 × 12.5 × {PD × LGD + √[R/(1-R)] × σ × N^{-1}(LGD)}
│  │   │   ├─ LGD Estimation Methods:
│  │   │   │   ├─ Historical: Average LGD from bank's past defaults
│  │   │   │   ├─ Market-based: CDS spreads imply recovery rates
│  │   │   │   ├─ Collateral-adjusted: LGD = [Exposure - Collateral × (1-Haircut)] / Exposure
│  │   │   │   ├─ Sectoral: Vary by industry/collateral type
│  │   │   │   ├─ Maturity adjustment: Longer loans → higher LGD (less recovery time)
│  │   │   │   └─ Typical range: 10% (senior secured real estate) - 80% (unsecured)
│  │   │   ├─ EAD Estimation Methods:
│  │   │   │   ├─ Outstanding amount: Current loan balance
│  │   │   │   ├─ Undrawn commitments: Probability of drawdown (typical 20%-50%)
│  │   │   │   ├─ Derivative exposure: Current + potential future exposure (CVA methodology)
│  │   │   │   └─ Formula: EAD = Outstanding + Credit Conversion Factor × Undrawn
│  │   │   ├─ PD Estimation Methods (see below)
│  │   │   ├─ Capital charges: Same formula as F-IRB but can be 2-3x lower (if LGD/EAD underestimated)
│  │   │   └─ Output floor constraint: RWA ≥ 72.5% × SA RWA (prevents gaming)
│  │   │
│  │   ├─ PD (Probability of Default) Estimation:
│  │   │   ├─ Definition: Probability borrower defaults within 1-year horizon
│  │   │   ├─ Historical Approach:
│  │   │   │   ├─ Data: 5-10 years of default history
│  │   │   │   ├─ Calculation: PD = (Number of defaults in year) / (Number of borrowers at start)
│  │   │   │   ├─ Adjustment: Remove cyclical effects (normalize to "through-the-cycle" PD)
│  │   │   │   ├─ Example: 100 loans tracked 5 years → 2 defaults → PD = 2% / 5 = 0.4% annual
│  │   │   │   └─ Downside: Limited history, need stability
│  │   │   ├─ Rating System Approach:
│  │   │   │   ├─ Segment portfolio into rating grades (AAA → CCC)
│  │   │   │   ├─ Assign PD to each grade (based on historical default rates)
│  │   │   │   ├─ Rating grades defined by financial metrics:
│  │   │   │   │   ├─ Leverage (Debt/EBITDA): ↑ leverage → ↓ grade → ↑ PD
│  │   │   │   │   ├─ Profitability (EBITDA/Revenue): ↓ margin → ↓ grade
│  │   │   │   │   ├─ Interest coverage (EBITDA/Interest): Lower → worse grade
│  │   │   │   │   └─ Industry/Country: Adjust baseline PD by sector risk
│  │   │   │   ├─ Typical Mapping (Corporates):
│  │   │   │   │   ├─ Grade 1 (AAA equivalent): PD = 0.05%
│  │   │   │   │   ├─ Grade 3 (A equivalent): PD = 0.2%
│  │   │   │   │   ├─ Grade 5 (BBB equivalent): PD = 0.8%
│  │   │   │   │   ├─ Grade 7 (B equivalent): PD = 3%
│  │   │   │   │   └─ Grade 9 (CCC equivalent): PD = 10%+
│  │   │   │   └─ Validation: Compare grades to external ratings (should correlate)
│  │   │   ├─ Statistical Models:
│  │   │   │   ├─ Logistic regression: PD = 1 / (1 + e^{-[intercept + β1×Leverage + β2×ROE + ...]})
│  │   │   │   ├─ Merton model: Structural approach (firm value vs debt)
│  │   │   │   │   ├─ Firm equity value = Max(Assets - Debt, 0)
│  │   │   │   │   ├─ Default when Assets < Debt
│  │   │   │   │   ├─ PD = N[-DD] (distance to default)
│  │   │   │   │   └─ DD = [ln(Assets/Debt) + (μ - σ²/2)T] / (σ√T)
│  │   │   │   ├─ Machine learning: Gradient boosting, neural networks (increasing use)
│  │   │   │   └─ Advantages: Capture nonlinearities, incorporate many features
│  │   │   └─ Regulatory Criteria (IRB model approval):
│  │   │       ├─ At least 5 years historical data
│  │   │       ├─ Include stress periods (recessions, crises)
│  │   │       ├─ Testing: Backtesting (actual defaults vs model PD)
│  │   │       ├─ Stability: PD shouldn't change >50% year-to-year (unless business change)
│  │   │       ├─ Granularity: Enough observations per grade to be reliable
│  │   │       └─ Regulatory approval required before use
│  │   │
│  │   ├─ Retail IRB (Special treatment):
│  │   │   ├─ Lower correlation (0.05 vs 0.12 for corporates) → lower RW
│  │   │   ├─ Typical RW: 5-15% (vs 35%+ SA for mortgages)
│  │   │   ├─ Rationale: Individual defaults don't correlate highly (diversified pool)
│  │   │   └─ Constraint: Portfolio effects (recession → all retail defaults spike)
│  │   │
│  │   └─ IRB Floors & Output Floor:
│  │       ├─ IRB floor (pre-2023): Minimum RWA = 75% × SA RWA
│  │       ├─ Output floor (post-2023): Minimum RWA = 72.5% × SA RWA
│  │       ├─ Effect: Prevents IRB from reducing RWA >27.5% vs SA
│  │       ├─ Example: SA RWA = $400B, IRB calculated = $200B → Use $290B (72.5% floor)
│  │       └─ Phase-in: 72.5% (2023) → 72.5% (permanent after 2028)
│
├─ Market Risk Capital
│  ├─ Basel II Approach (VaR-based, now deprecated):
│  │   ├─ VaR(99%, 10-day) = 1% probability of loss > this amount
│  │   ├─ Capital = 3 × VaR + IdiosyncraticRisk + specific risk charge
│  │   ├─ Criticized: Doesn't capture tail severity (CVaR importance)
│  │   └─ Status: Phased out (replaced by FRTB)
│  │
│  ├─ Basel III/FRTB Approach (Expected Shortfall-based):
│  │   ├─ Expected Shortfall (CVaR) = Average loss in tail 1%
│  │   ├─ Calculation Steps:
│  │   │   ├─ Historical scenarios (250 days, last year of data)
│  │   │   ├─ Mark portfolio to each scenario
│  │   │   ├─ Calculate losses
│  │   │   ├─ Sort, select worst 1% of days
│  │   │   ├─ Average = ES
│  │   │   └─ Capital = 3 × ES (provides buffer)
│  │   ├─ Stressed ES: Same calculation but using pre-crisis market period
│  │   │   ├─ Captures regime where correlations spike
│  │   │   ├─ ES from 2008-2009 would be used if that's the worst period
│  │   │   └─ Capital charge = max(ES_current, ES_stressed)
│  │   ├─ Modeling Components:
│  │   │   ├─ Delta (linear sensitivity): ∂P/∂S × S change
│  │   │   ├─ Gamma (convexity): ½ × ∂²P/∂S² × (ΔS)²
│  │   │   ├─ Vega (volatility): ∂P/∂σ × Δσ
│  │   │   ├─ Rho (interest rate): ∂P/∂r × Δr
│  │   │   └─ Basis risk: Hedge doesn't perfectly offset (e.g., index vs individual stock)
│  │   └─ Capital charge (FRTB):
│  │       ├─ Sensitivities method (simplified): Fixed capital per dollar of delta/gamma/vega
│  │       ├─ Full revaluation (complex): Run model on scenarios
│  │       └─ Typical: 5-10% of notional for equity portfolios
│  │
│  ├─ Interest Rate Risk in Banking Book (IRRBB):
│  │   ├─ Non-trading positions (deposits, mortgages at fixed rates)
│  │   ├─ Pillar 2 capital add-on (not formulaic)
│  │   ├─ Measured as: Loss if rates move ±200 bps
│  │   ├─ Example: Deposit base $100B at 1%, mortgages $80B at 3.5%
│  │   │   Rate +200bps: Cost deposits ↑ by $2B, mortgage income ↑ $1.6B → Net loss $0.4B
│  │   ├─ Counterparty (counterparty risk): Derive exposure value using SA-CCR
│  │   └─ CVA (Credit Valuation Adjustment): Risk that counterparty defaults
│  │       ├─ Not just mark-to-market, but future exposure too
│  │       ├─ Capital charge on derivative portfolio
│  │       └─ Typically 2-5% of notional for active traders
│  │
│  └─ Concentration Risk (New):
│      ├─ Single counterparty large exposure limit
│      ├─ Exposure > 10% Tier 1 capital triggers capital charge
│      ├─ Formula: 0% if < 10%, scales to 100% if very large
│      ├─ Example: Bank Tier1 = $20B, exposure to client = $5B
│      │   Limit = 10% × $20B = $2B; $5B exceeds by $3B → capital charge on $3B
│      └─ Interconnectedness adds surcharge (systemically important counterparties)
│
├─ Operational Risk Capital
│  ├─ Standardized Approach (SA):
│  │   ├─ Capital Charge = 12% × Indicator (average 3-year)
│  │   ├─ Indicator typically = Gross revenue (adjusted for business lines)
│  │   ├─ Calculation:
│  │   │   ├─ Calculate indicator for each year (past 3 years)
│  │   │   ├─ Average the 3 years
│  │   │   ├─ Multiply by 12%
│  │   │   └─ Result = Capital required
│  │   ├─ Example:
│  │   │   ├─ Year 1 revenue: $100M, OpRisk indicator = $100M
│  │   │   ├─ Year 2 revenue: $120M, OpRisk indicator = $120M
│  │   │   ├─ Year 3 revenue: $110M, OpRisk indicator = $110M
│  │   │   ├─ Average = $110M
│  │   │   ├─ Capital required = 12% × $110M = $13.2M
│  │   └─ Simplified but pro-cyclical (revenue down in crisis → lower capital)
│  │
│  ├─ Advanced Approach (AA):
│  │   ├─ Used by large systemically important banks
│  │   ├─ Components:
│  │   │   ├─ Expected Loss (EL): E[Severity × Frequency]
│  │   │   │   ├─ Frequency: How many operational events per year (e.g., 5 events)
│  │   │   │   ├─ Severity: Average loss per event (e.g., $2M average)
│  │   │   │   └─ EL = 5 × $2M = $10M/year
│  │   │   ├─ Unexpected Loss (UL): Tail risk charge
│  │   │   │   ├─ Use CVaR or value-at-risk on loss distribution
│  │   │   │   ├─ UL = (99.9% VaR - EL) / 8 [convert to capital]
│  │   │   │   ├─ Typical: $30-50M (tail scenarios, rare large losses)
│  │   │   │   └─ Example: 0.1% tail = $100M loss, EL = $10M, UL = ($100M - $10M) / 8 = $11.25M
│  │   │   ├─ Internal Loss Multiplier (ILM): Adjustment for data quality/model risk
│  │   │   │   ├─ If severe crisis happened recently, data quality increases
│  │   │   │   ├─ ILM = [1 + (OpRisk events in crisis × weight)] / baseline
│  │   │   │   ├─ Range: 0.8 - 1.5
│  │   │   │   └─ Effect: Increase capital charge post-crisis (pro-cyclical tension)
│  │   │   └─ Diversification: Reduce capital by potential portfolio diversification
│  │   │       ├─ Different risk categories less than additive
│  │   │       ├─ Correlation < 1 → portfolio effect
│  │   │       └─ Divisor typically 1.5 - 2.5
│  │   ├─ Capital charge: OpRisk = [EL + UL] × ILM × 1/Diversification
│  │   ├─ Typical: $25-75M for large banks
│  │   └─ Regulatory approval required (strict model validation)
│  │
│  ├─ Loss Event Categories (Basel Operational Risk Framework):
│  │   ├─ Internal fraud: Employee theft, trade desk misconduct
│  │   ├─ External fraud: ATM theft, cyber attacks, client fraud
│  │   ├─ DLOA (Disruption of Business & System Failures): IT outages, power failures
│  │   ├─ EPCE (Employment Practices & Client Relations): Wrongful termination suits, discrimination
│  │   ├─ Damage to Physical Assets: Natural disasters, vandalism
│  │   ├─ Business Disruption & System Failures: Infrastructure failure
│  │   ├─ Execution, Delivery, Process Management: Trade errors, settlement fails
│  │   ├─ Client/Product line specific: Model risk, product defects
│  │   └─ Correlations: Typically low (diversified portfolio)
│  │
│  └─ Regulatory Scrutiny:
│      ├─ Stress scenarios including operational events
│      ├─ Enhanced monitoring for fraud/cyber risk
│      ├─ Technology resilience assessments
│      └─ Third-party risk management (outsourced functions)
│
├─ Total Capital Requirement Combination
│  ├─ Formula: Total Capital = Credit RW + Market RW + OpRisk
│  │   All converted to capital percentage of RWA
│  │   Capital = 8% × (Credit RWA + Market RWA + OpRisk RWA)
│  ├─ Interaction Effects:
│  │   ├─ Correlated risks: Market downturn + credit defaults + operational stress
│  │   ├─ Stress testing explicitly models combinations
│  │   ├─ No diversification benefit (regulatory conservative)
│  │   └─ Pillar 2 (supervisor) can add if combinations appear dangerous
│  ├─ Buffers (on top of 8% minimum):
│  │   ├─ Capital Conservation Buffer (CCB): 2.5%
│  │   ├─ Countercyclical Buffer (CyCB): 0-2.5%
│  │   ├─ G-SIB surcharge: 1-3.5%
│  │   └─ Total effective minimum: 12-17% for large banks
│  └─ Real Bank Example (Hypothetical Large Bank):
│      ├─ Credit RWA: $300B (75% of total)
│      ├─ Market RWA: $80B (20% of total)
│      ├─ OpRisk RWA: $20B (5% of total)
│      ├─ Total RWA: $400B
│      ├─ Minimum capital (8%): $32B
│      ├─ CCB (2.5%): $10B
│      ├─ CyCB (1%): $4B
│      ├─ G-SIB surcharge (2%): $8B
│      ├─ Total required: $54B (13.5% of RWA)
│      ├─ Typical buffer: Hold $60-70B (15-17.5%)
│      └─ Leverage ratio floor (3% of $1.5T assets) = $45B also binding
│
├─ Risk Calculation System Architecture
│  ├─ Data Pipeline:
│  │   ├─ Market data: Daily prices, rates, volatility (real-time)
│  │   ├─ Portfolio positions: Securities, derivatives, loans (daily)
│  │   ├─ Credit data: Ratings, default history, PD models (monthly updates)
│  │   ├─ Operational data: Loss events, audit reports (annual compilation)
│  │   └─ Regulatory data: Counterparty exposures, large exposures (monthly)
│  ├─ Model Components:
│  │   ├─ Credit risk models: PD/LGD/EAD for each borrower
│  │   ├─ Market risk models: VaR, ES, Greeks (delta/gamma/vega)
│  │   ├─ Operational risk models: Frequency/severity, loss events
│  │   ├─ Correlation models: How risks move together
│  │   └─ Scenario analysis: Tail events, stress testing
│  ├─ Computing:
│  │   ├─ Overnight: Full capital calculation (RWA recalculation)
│  │   ├─ Intraday: Market VaR updates (key for trading desks)
│  │   ├─ Monthly: Stress testing, regulatory reporting
│  │   ├─ Quarterly: Capital forecast, buffer testing
│  │   └─ Annual: IRB model backtesting, regulatory approval prep
│  └─ Audit & Governance:
│      ├─ Model risk management: Independent review of key models
│      ├─ Backtesting: Compare predicted vs actual losses
│      ├─ Sensitivity analysis: How capital changes with parameter shifts
│      ├─ Stress testing: Extreme scenarios
│      └─ Board oversight: Capital adequacy, strategic implications
```

**Interaction Example:**  
Bank portfolio: $100M in BBB corporate loans. Credit RW (SA) = 100%, RWA = $100M. Market position: Short $20M corporate bonds (hedge). Market RW (FRTB) per 2% ES charge = $0.4M. OpRisk allocated 5% → $5M. Total capital = 8% × ($100M + $0.4M + $5M) = $8.04M + buffers = ~$12M.

## 5. Mini-Project
Compare Standardized vs Advanced IRB capital requirements:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Bank corporate loan portfolio
portfolio = {
    'AAA': {'exposure': 50, 'rating': 'AAA'},
    'AA': {'exposure': 100, 'rating': 'AA'},
    'A': {'exposure': 150, 'rating': 'A'},
    'BBB': {'exposure': 300, 'rating': 'BBB'},
    'BB': {'exposure': 200, 'rating': 'BB'},
    'B': {'exposure': 100, 'rating': 'B'},
    'CCC': {'exposure': 50, 'rating': 'CCC'},
}

# 1. STANDARDIZED APPROACH (External ratings)
print("="*100)
print("STANDARDIZED APPROACH (SA) - Risk Weighted Assets Calculation")
print("="*100)

# Risk weights by rating
sa_rw = {
    'AAA': 0.20, 'AA': 0.20, 'A': 0.50,
    'BBB': 1.00, 'BB': 1.00, 'B': 1.50, 'CCC': 1.50,
}

sa_results = {}
total_sa_rwa = 0

for loan_type, data in portfolio.items():
    exposure = data['exposure']
    rating = data['rating']
    rw = sa_rw[rating]
    rwa = exposure * rw
    total_sa_rwa += rwa
    sa_results[loan_type] = {'exposure': exposure, 'rw': rw, 'rwa': rwa}
    print(f"{loan_type} {rating:>5}: ${exposure:>4.0f}M exposure, RW={rw*100:>5.0f}%, RWA=${rwa:>6.1f}M")

sa_capital = 0.08 * total_sa_rwa
print(f"\nTotal SA RWA: ${total_sa_rwa:.1f}M")
print(f"SA Capital (8%): ${sa_capital:.1f}M")

# 2. FOUNDATION IRB APPROACH (Bank estimates PD, regulator provides LGD/EAD)
print(f"\n" + "="*100)
print("FOUNDATION IRB - Bank-Estimated PD, Regulatory LGD/EAD")
print("="*100)

# Historical PD by rating (bank's data)
irb_pd = {
    'AAA': 0.02, 'AA': 0.05, 'A': 0.10,
    'BBB': 0.50, 'BB': 2.00, 'B': 5.00, 'CCC': 15.00,
}

# Regulatory LGD & EAD (foundation fixed)
irb_lgd = 0.45  # Regulatory floor for unsecured corporate
irb_ead = 1.00  # Regulatory EAD (100% outstanding)

# IRB formula parameters
maturity_adj = 1.0  # Assume 3-year loans → maturity adjustment ≈ 1.0
correlation = 0.12  # Corporate correlation

def calculate_irb_rw(pd, lgd, ead, correlation=0.12, maturity=1.0):
    """Calculate IRB risk weight using Basel formula."""
    # N(x) = cumulative normal
    # N^{-1}(x) = inverse normal
    
    pd_norm = norm.ppf(pd / 100)  # Convert PD % to decimal, then inverse normal
    
    # Correlation-adjusted maturity factor
    b = (0.11852 - 0.05478 * np.log(pd / 100)) ** 2
    maturity_factor = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    
    # N(.) calculation
    numerator = np.log(correlation / (1 - correlation)) + np.sqrt(correlation / (1 - correlation)) * norm.ppf(pd / 100)
    tail_quantile = norm.ppf(0.999)  # 99.9% confidence
    
    n_val = norm.cdf((numerator + np.sqrt(1 - correlation) * tail_quantile) / 
                     np.sqrt(correlation))
    
    # RW formula
    rw = 1.06 * 12.5 * (pd / 100 * lgd / 100 + np.sqrt(correlation / (1 - correlation)) * 
                        np.sqrt(1 - correlation) * (norm.ppf(pd / 100) - np.sqrt(correlation) * tail_quantile)) * maturity_factor
    
    # Simplified approximation
    rw_approx = 1.06 * 12.5 * (pd / 100 * lgd / 100)
    
    return max(min(rw_approx, 3.25), 0.001)  # Cap at 325%, floor at 0.1%

irb_results = {}
total_irb_rwa = 0

print(f"Using: LGD = {irb_lgd*100:.0f}%, EAD = {irb_ead*100:.0f}%, Correlation = {correlation*100:.0f}%\n")

for loan_type, data in portfolio.items():
    exposure = data['exposure']
    pd = irb_pd[loan_type]
    rw = calculate_irb_rw(pd, irb_lgd, irb_ead, correlation)
    rwa = exposure * rw
    total_irb_rwa += rwa
    irb_results[loan_type] = {'exposure': exposure, 'pd': pd, 'rw': rw, 'rwa': rwa}
    print(f"{loan_type} {data['rating']:>5}: PD={pd:>6.2f}%, RW={rw*100:>5.1f}%, RWA=${rwa:>6.1f}M")

irb_capital = 0.08 * total_irb_rwa
print(f"\nTotal IRB RWA: ${total_irb_rwa:.1f}M")
print(f"IRB Capital (8%): ${irb_capital:.1f}M")

# 3. ADVANCED IRB (Bank estimates PD, LGD, EAD - lower LGD for collateral)
print(f"\n" + "="*100)
print("ADVANCED IRB - Bank-Estimated PD, LGD, EAD (with Collateral)")
print("="*100)

# Bank estimates with collateral/loan structure
advanced_lgd = {
    'AAA': 0.20, 'AA': 0.25, 'A': 0.30,
    'BBB': 0.40, 'BB': 0.50, 'B': 0.60, 'CCC': 0.70,
}

# EAD accounts for undrawn commitments (30% drawdown probability)
advanced_ead_multiple = {
    'AAA': 0.90, 'AA': 0.92, 'A': 0.95,
    'BBB': 1.00, 'BB': 1.00, 'B': 1.00, 'CCC': 1.00,
}

adv_results = {}
total_adv_rwa = 0

print(f"Using: Collateral-adjusted LGD, Bank EAD estimates\n")

for loan_type, data in portfolio.items():
    exposure = data['exposure']
    rating = data['rating']
    pd = irb_pd[loan_type]
    lgd = advanced_lgd[rating]
    ead = advanced_ead_multiple[rating]
    rw = calculate_irb_rw(pd, lgd, ead, correlation)
    rwa = exposure * rw * ead
    total_adv_rwa += rwa
    adv_results[loan_type] = {'exposure': exposure, 'pd': pd, 'lgd': lgd, 'ead': ead, 'rw': rw, 'rwa': rwa}
    print(f"{loan_type} {rating:>5}: PD={pd:>6.2f}%, LGD={lgd*100:>5.0f}%, EAD={ead*100:>5.0f}%, RWA=${rwa:>6.1f}M")

adv_capital = 0.08 * total_adv_rwa
print(f"\nTotal Advanced IRB RWA: ${total_adv_rwa:.1f}M")
print(f"Advanced IRB Capital (8%): ${adv_capital:.1f}M")

# 4. OUTPUT FLOOR (72.5% of SA RWA)
print(f"\n" + "="*100)
print("OUTPUT FLOOR (72.5% constraint)")
print("="*100)

floor_value = 0.725 * total_sa_rwa
print(f"SA RWA: ${total_sa_rwa:.1f}M")
print(f"72.5% Floor: ${floor_value:.1f}M")
print(f"Advanced IRB RWA: ${total_adv_rwa:.1f}M")

if total_adv_rwa < floor_value:
    floored_rwa = floor_value
    print(f"Advanced IRB FLOORED to: ${floored_rwa:.1f}M (binding floor)")
else:
    floored_rwa = total_adv_rwa
    print(f"Advanced IRB above floor (no constraint)")

floored_capital = 0.08 * floored_rwa

# COMPARISON TABLE
print(f"\n" + "="*100)
print("CAPITAL REQUIREMENT COMPARISON")
print("="*100)

comparison = pd.DataFrame({
    'Approach': ['Standardized (SA)', 'Foundation IRB (F-IRB)', 'Advanced IRB (A-IRB)', 'Advanced IRB (w/ Floor)'],
    'Total RWA': [f'${total_sa_rwa:.1f}M', f'${total_irb_rwa:.1f}M', f'${total_adv_rwa:.1f}M', f'${floored_rwa:.1f}M'],
    'Capital (8%)': [f'${sa_capital:.1f}M', f'${irb_capital:.1f}M', f'${adv_capital:.1f}M', f'${floored_capital:.1f}M'],
    'Capital vs SA': ['Baseline', f'{(irb_capital/sa_capital - 1)*100:+.1f}%', 
                      f'{(adv_capital/sa_capital - 1)*100:+.1f}%',
                      f'{(floored_capital/sa_capital - 1)*100:+.1f}%'],
})

print(comparison.to_string(index=False))

rwa_reduction = (1 - total_adv_rwa / total_sa_rwa) * 100
print(f"\nAdvanced IRB RWA reduction vs SA: {rwa_reduction:.1f}%")
print(f"Output floor prevents RWA reduction >27.5% (current: {rwa_reduction:.1f}%)")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RWA by rating and approach
ax = axes[0, 0]
ratings = list(portfolio.keys())
sa_rwas = [sa_results[r]['rwa'] for r in ratings]
irb_rwas = [irb_results[r]['rwa'] for r in ratings]
adv_rwas = [adv_results[r]['rwa'] for r in ratings]

x = np.arange(len(ratings))
width = 0.25

ax.bar(x - width, sa_rwas, width, label='Standardized', alpha=0.8)
ax.bar(x, irb_rwas, width, label='Foundation IRB', alpha=0.8)
ax.bar(x + width, adv_rwas, width, label='Advanced IRB', alpha=0.8)

ax.set_ylabel('Risk-Weighted Assets ($M)')
ax.set_title('RWA by Rating and Approach')
ax.set_xticks(x)
ax.set_xticklabels(ratings)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Total capital requirements
ax = axes[0, 1]
approaches = ['SA', 'F-IRB', 'A-IRB', 'A-IRB\n(Floored)']
capitals = [sa_capital, irb_capital, adv_capital, floored_capital]
colors_cap = ['blue', 'orange', 'green', 'red']

bars = ax.bar(approaches, capitals, color=colors_cap, alpha=0.7)
ax.set_ylabel('Capital Required ($M)')
ax.set_title('Total Capital Requirement Comparison')
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar, cap in zip(bars, capitals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${cap:.1f}M',
            ha='center', va='bottom', fontweight='bold')

# Plot 3: RWA breakdown by rating
ax = axes[1, 0]
ax.barh(ratings, [adv_results[r]['rwa'] for r in ratings], color='teal', alpha=0.7)
ax.set_xlabel('RWA ($M)')
ax.set_title('Advanced IRB RWA Breakdown by Rating')
ax.grid(alpha=0.3, axis='x')

# Plot 4: Risk weight by rating
ax = axes[1, 1]
sa_rws = [sa_results[r]['rw']*100 for r in ratings]
irb_rws = [irb_results[r]['rw']*100 for r in ratings]
adv_rws = [adv_results[r]['rw']*100 for r in ratings]

x = np.arange(len(ratings))
ax.plot(x, sa_rws, 'o-', label='Standardized', linewidth=2, markersize=6)
ax.plot(x, irb_rws, 's-', label='Foundation IRB', linewidth=2, markersize=6)
ax.plot(x, adv_rws, '^-', label='Advanced IRB', linewidth=2, markersize=6)

ax.set_ylabel('Risk Weight (%)')
ax.set_title('Risk Weight Curves by Rating and Approach')
ax.set_xticks(x)
ax.set_xticklabels(ratings)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
plt.show()

# Gaming risk illustration
print(f"\n" + "="*100)
print("MODEL GAMING ILLUSTRATION")
print("="*100)
print(f"\nWhat if Advanced IRB underestimates LGD by 50%?")
gamed_lgd = {k: v * 0.5 for k, v in advanced_lgd.items()}
gamed_rwa = 0
for loan_type, data in portfolio.items():
    exposure = data['exposure']
    rating = data['rating']
    pd = irb_pd[loan_type]
    lgd = gamed_lgd[rating]
    ead = advanced_ead_multiple[rating]
    rw = calculate_irb_rw(pd, lgd, ead, correlation)
    rwa = exposure * rw * ead
    gamed_rwa += rwa
gamed_capital = 0.08 * gamed_rwa
gamed_reduction = (1 - gamed_rwa / total_sa_rwa) * 100

print(f"Gamed A-IRB RWA: ${gamed_rwa:.1f}M (reduction: {gamed_reduction:.1f}%)")
print(f"With 72.5% floor, gamed RWA capped at: ${floor_value:.1f}M")
print(f"Floor prevents capital arbitrage: ${gamed_capital:.1f}M → ${floored_capital:.1f}M (${floored_capital - gamed_capital:.1f}M difference)")
```

## 6. Challenge Round
- Map bank's $500M portfolio to Basel SA risk weights; calculate RWA
- Design F-IRB PD model for mid-market corporates (regression on financials)
- Estimate LGD for real estate collateral using historical recovery data
- Compare leverage ratio floor vs RW-based capital for leverage-heavy portfolio
- Run output floor test: Is 72.5% constraint binding for your IRB model?

## 7. Key References
- [BIS, "The Standardized Approach for Credit Risk" (2017)](https://www.bis.org/basel_framework/crossfunctional/output_floor.pdf) — Official regulation
- [BIS, "Internal Ratings-Based Approach" (2017)](https://www.bis.org/basel_framework/standard/crb.htm) — IRB formula and calibration
- [Federal Reserve, "CCAR 2024 Stress Test Scenarios"](https://www.federalreserve.gov/banking/ccar-capital-planning.htm) — Implementation example (US)
- [Gordy, "A Comparative Anatomy of Credit Risk Models" (2000), JFQA](https://www.jstor.org/) — Theoretical foundation

---
**Status:** Core regulatory framework (2008-present, continuously refined) | **Complements:** Basel III Framework, Liquidity Risk, Stress Testing, Market Risk
