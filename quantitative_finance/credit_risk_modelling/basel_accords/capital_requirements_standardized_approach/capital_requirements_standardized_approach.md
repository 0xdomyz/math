# Capital Requirements & Standardized Approach (Basel III)

## 1. Concept Skeleton
**Definition:** Regulatory framework specifying minimum capital requirements for credit risk, market risk, operational risk; Standardized Approach uses external ratings (AAA-D) for credit risk; Advanced Internal Ratings-Based (IRB) allows bank models (PD/LGD/EAD); Market risk uses expected shortfall instead of VaR  
**Purpose:** Align bank incentives with actual risk exposure; prevent RWA gaming; apply consistent risk-weighting across institutions; capture tail risk in capital charges  
**Prerequisites:** Risk-Weighted Assets (RWA), credit risk fundamentals, market risk, operational risk measurement, external ratings, PD/LGD/EAD

## 2. Comparative Framing
| Aspect | Standardized Approach | Foundation IRB | Advanced IRB | Key Tradeoff |
|--------|---------------------|-----------------|---------------|------------|
| **Credit Risk RW Source** | External ratings (S&P, Moody's) | Bank estimates PD; regulator provides LGD/EAD | Bank estimates all (PD/LGD/EAD) | Complexity â†” Risk-sensitivity |
| **Risk Weight** | 20%-150% by rating | Calculated formula (PD/LGD/EAD dependent) | Calculated formula (bank-specific) | Standardized â†” Customized |
| **RWA Formula** | RW Ã— Exposure | 1.06Ã—12.5Ã—(PDÃ—LGD + correlation adj.) | 1.06Ã—12.5Ã—(PDÃ—LGDÃ—EAD + ... - discount) | Simple â†” Refined |
| **Capital Requirement** | 8% Ã— RWA | 8% Ã— RWA (or lower if IRB floor) | 8% Ã— RWA (bound by floor) | Easy compliance â†” Optimized |
| **Regulatory Approval** | Automatic (published ratings) | Requires IRB model validation | Requires IRB + detailed validation | Fast â†” Rigorous |
| **Data Requirements** | External ratings only | Internal default history + rating drift | Historical losses, correlation matrices | Low burden â†” High burden |
| **Fit for Asset Class** | Standardized: Large corporates, sovereigns | Mid-market corporates, loans | Large banks (internal data available) | Cost-effective â†” Accurate |
| **Game-Ability** | Cliff effects at rating boundaries | Parameter manipulation (PD/LGD underestimation) | Extensive model gaming potential | Transparent â†” Exploitable |
| **Output Floor** | Baseline (100%) | 72.5% of SA RWA | 72.5% of SA RWA | â€”â€” |
| **Typical Capital** | 10-12% RWA | 8-11% RWA | 7-10% RWA | Conservative â†” Efficient |

## 3. Examples + Counterexamples

**Standardized Approach Example:**  
Bank holds $100M loan to AAA-rated corporate. Risk weight = 20% (S&P AAA). RWA = 0.20 Ã— $100M = $20M. Capital required = 8% Ã— $20M = $1.6M.  
Rating downgrade to BBB: RW = 100%, RWA = $100M, capital = $8M (5x increase overnight). Called "cliff risk."

**Foundation IRB Example (Corporates):**  
Bank uses internal data; PD estimate = 1.5%, LGD = 45% (regulator provides), EAD = $100M.  
RW formula: 1.06 Ã— 12.5 Ã— [1.5% Ã— 45% + âˆš(correlation) Ã— (default stress)] â‰ˆ 60%.  
RWA = 0.60 Ã— $100M = $60M, Capital = $4.8M (much lower than SA's $8M for BBB equivalent).

**Advanced IRB Example (Retail Mortgages):**  
Bank large mortgage portfolio; internal data shows 0.3% default rate, 30% LGD, 85% EAD (loan loss given default; 15% equity cushion).  
RW = 1.06 Ã— 12.5 Ã— [0.3% Ã— 30%] â‰ˆ 1.2% (extremely lowâ€”retail portfolios are safer).  
RWA = 0.012 Ã— $500M = $6M, Capital = $0.48M (vs SA's standardized 35% for residential mortgages = $17.5M).  
**Counterexample:** 2008 crisis: Actual mortgage defaults 5%+, LGD 60%+. Advanced IRB models catastrophically underestimated. Lesson: Historical calibration fails in tail risk.

**Market Risk: VaR vs Expected Shortfall (CVaR):**  
Basel II: VaR(99%, 10-day) = $50M. Bank "99% confident losses won't exceed $50M."  
Basel III CVaR: ES(99%, 10-day) = $75M (average of losses in tail 1%). Captures tail severity.  
Result: Basel III capital â‰ˆ 50% higher for trading desks.

**Operational Risk: Standardized Approach:**  
Bank revenue over 3 years: $500M average. OpRisk charge = 12% Ã— $500M = $60M capital required.  
vs Advanced: Historical loss data, 15% loss ratio, internal models â†’ $40M. Gaming: Shift business to lower-indicator revenue streams.

**Leverage Ratio as Binding Floor:**  
Bank: Tier 1 = $40B, Total Assets = $900B. Leverage ratio = $40B / $900B = 4.44%.  
RWA-based: If RWA = $300B (33% of assets, "low-risk"), Capital = 8% Ã— $300B = $24B âœ“ (easily met).  
But leverage ratio requires $27B (3% Ã— $900B). Leverage ratio is binding â†’ must hold extra capital.

**Output Floor (72.5%) Example:**  
Bank SA RWA = $400B (standardized). IRB model calculates RWA = $250B (37.5% reduction via gaming).  
Floor: IRB RWA â‰¥ 72.5% Ã— $400B = $290B. Bank constrained to use $290B.  
**Effect:** Reduces incentive to game PD/LGD but doesn't eliminate. Floor is partial safeguard.

## 4. Layer Breakdown
```
Capital Requirement & Risk Calculation Framework:

â”œâ”€ Credit Risk Capital Calculation
â”‚  â”œâ”€ Standardized Approach (SA)
â”‚  â”‚   â”œâ”€ Asset Classification:
â”‚  â”‚   â”‚   â”œâ”€ Central Government & Central Banks:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Domestic currency (country): RW = 0% (OeCD member)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Foreign currency: RW = 0-100% (country risk classification)
â”‚  â”‚   â”‚   â”‚   â””â”€ Example: US Treasury 0%, Greece 100%+
â”‚  â”‚   â”‚   â”œâ”€ Institutions (Banks, Investment Firms):
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Domestic, central bank's home country: RW = 20%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ AAA-AA rated: RW = 20%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ A-rated: RW = 50%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ BBB-rated: RW = 100%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Unrated: RW = 100%
â”‚  â”‚   â”‚   â”‚   â””â”€ BB or below: RW = 150%
â”‚  â”‚   â”‚   â”œâ”€ Corporates (Non-Financial):
â”‚  â”‚   â”‚   â”‚   â”œâ”€ AAA-AA rated: RW = 20%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ A-rated: RW = 50%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ BBB-rated: RW = 100%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ BB-rated: RW = 100%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ B or below: RW = 150%
â”‚  â”‚   â”‚   â”‚   â””â”€ Unrated: RW = 100%
â”‚  â”‚   â”‚   â”œâ”€ Retail Exposures:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Residential mortgages (RMBS): RW = 35%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Qualifying revolving (credit cards): RW = 75%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Other retail (auto loans, unsecured personal): RW = 75%
â”‚  â”‚   â”‚   â”‚   â””â”€ All retail pools lower RW (lower individual defaults)
â”‚  â”‚   â”‚   â”œâ”€ Equity Exposures:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Direct holdings: RW = 100%
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Mutual funds (look-through): RW = rating-dependent 20-1250%
â”‚  â”‚   â”‚   â”‚   â””â”€ Private equity: RW = 190%
â”‚  â”‚   â”‚   â””â”€ Off-Balance-Sheet Exposures:
â”‚  â”‚   â”‚       â”œâ”€ Credit commitments: CCF (Credit Conversion Factor) 20%-100%
â”‚  â”‚   â”‚       â”œâ”€ Guarantees: 100% CCF
â”‚  â”‚   â”‚       â””â”€ Letter of credit: 20%-100% CCF
â”‚  â”‚   â”œâ”€ Credit Risk Mitigation:
â”‚  â”‚   â”‚   â”œâ”€ Collateral Haircuts: Adjust exposure for market fluctuations
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Cash collateral: 0% haircut
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Government bonds (AAA): 0-2% haircut
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Investment-grade corporates: 2-4% haircut
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Equity: 15-50% haircut (high volatility)
â”‚  â”‚   â”‚   â”‚   â””â”€ Formula: Adjusted Exposure = Exposure - Collateral Ã— (1 - Haircut)
â”‚  â”‚   â”‚   â”œâ”€ Guarantees & Credit Derivatives:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Third-party guarantee: Risk weight â†’ guarantor's rating
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Example: BBB corporate guaranteed by AAA bank â†’ use 20% RW
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Partial coverage: Proportional risk weighting
â”‚  â”‚   â”‚   â”‚   â””â”€ Counterparty concentration: Cap at 25% guarantee
â”‚  â”‚   â”‚   â””â”€ Netting: Reduce gross exposure by collateral/marks
â”‚  â”‚   â””â”€ Capital Charge: 8% Ã— (Adjusted RWA)
â”‚  â”‚
â”‚  â”œâ”€ Internal Ratings-Based (IRB) Approach
â”‚  â”‚   â”œâ”€ Foundation IRB (F-IRB):
â”‚  â”‚   â”‚   â”œâ”€ Bank estimates: Probability of Default (PD)
â”‚  â”‚   â”‚   â”œâ”€ Regulator provides: Loss Given Default (LGD), Exposure at Default (EAD)
â”‚  â”‚   â”‚   â”œâ”€ Formula (Corporates):
â”‚  â”‚   â”‚   â”‚   RW = 1.06 Ã— 12.5 Ã— {PD Ã— LGD + âˆš[R/(1-R)] Ã— Ïƒ Ã— N^{-1}(LGD)}
â”‚  â”‚   â”‚   â”‚   Where:
â”‚  â”‚   â”‚   â”‚   - N^{-1}(LGD) = inverse normal (portfolio tail stress)
â”‚  â”‚   â”‚   â”‚   - R = correlation factor (typically 0.12 for corporates, lower for retail)
â”‚  â”‚   â”‚   â”‚   - Ïƒ = asset volatility
â”‚  â”‚   â”‚   â”œâ”€ Result: RW typically 15%-50% for corporates (vs 20%-150% SA)
â”‚  â”‚   â”‚   â”œâ”€ LGD floor: 45% for corporate unsecured, 35% senior secured
â”‚  â”‚   â”‚   â”œâ”€ EAD floor: Minimum 100% of loan balance
â”‚  â”‚   â”‚   â””â”€ Capital charge: 8% Ã— RWA
â”‚  â”‚   â”‚
â”‚  â”‚   â”œâ”€ Advanced IRB (A-IRB):
â”‚  â”‚   â”‚   â”œâ”€ Bank estimates all: PD, LGD, EAD
â”‚  â”‚   â”‚   â”œâ”€ Formula (Corporates - identical to F-IRB but bank-derived LGD/EAD):
â”‚  â”‚   â”‚   â”‚   RW = 1.06 Ã— 12.5 Ã— {PD Ã— LGD + âˆš[R/(1-R)] Ã— Ïƒ Ã— N^{-1}(LGD)}
â”‚  â”‚   â”‚   â”œâ”€ LGD Estimation Methods:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Historical: Average LGD from bank's past defaults
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Market-based: CDS spreads imply recovery rates
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Collateral-adjusted: LGD = [Exposure - Collateral Ã— (1-Haircut)] / Exposure
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Sectoral: Vary by industry/collateral type
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Maturity adjustment: Longer loans â†’ higher LGD (less recovery time)
â”‚  â”‚   â”‚   â”‚   â””â”€ Typical range: 10% (senior secured real estate) - 80% (unsecured)
â”‚  â”‚   â”‚   â”œâ”€ EAD Estimation Methods:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Outstanding amount: Current loan balance
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Undrawn commitments: Probability of drawdown (typical 20%-50%)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Derivative exposure: Current + potential future exposure (CVA methodology)
â”‚  â”‚   â”‚   â”‚   â””â”€ Formula: EAD = Outstanding + Credit Conversion Factor Ã— Undrawn
â”‚  â”‚   â”‚   â”œâ”€ PD Estimation Methods (see below)
â”‚  â”‚   â”‚   â”œâ”€ Capital charges: Same formula as F-IRB but can be 2-3x lower (if LGD/EAD underestimated)
â”‚  â”‚   â”‚   â””â”€ Output floor constraint: RWA â‰¥ 72.5% Ã— SA RWA (prevents gaming)
â”‚  â”‚   â”‚
â”‚  â”‚   â”œâ”€ PD (Probability of Default) Estimation:
â”‚  â”‚   â”‚   â”œâ”€ Definition: Probability borrower defaults within 1-year horizon
â”‚  â”‚   â”‚   â”œâ”€ Historical Approach:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Data: 5-10 years of default history
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Calculation: PD = (Number of defaults in year) / (Number of borrowers at start)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Adjustment: Remove cyclical effects (normalize to "through-the-cycle" PD)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Example: 100 loans tracked 5 years â†’ 2 defaults â†’ PD = 2% / 5 = 0.4% annual
â”‚  â”‚   â”‚   â”‚   â””â”€ Downside: Limited history, need stability
â”‚  â”‚   â”‚   â”œâ”€ Rating System Approach:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Segment portfolio into rating grades (AAA â†’ CCC)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Assign PD to each grade (based on historical default rates)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Rating grades defined by financial metrics:
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Leverage (Debt/EBITDA): â†‘ leverage â†’ â†“ grade â†’ â†‘ PD
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Profitability (EBITDA/Revenue): â†“ margin â†’ â†“ grade
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Interest coverage (EBITDA/Interest): Lower â†’ worse grade
â”‚  â”‚   â”‚   â”‚   â”‚   â””â”€ Industry/Country: Adjust baseline PD by sector risk
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Typical Mapping (Corporates):
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Grade 1 (AAA equivalent): PD = 0.05%
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Grade 3 (A equivalent): PD = 0.2%
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Grade 5 (BBB equivalent): PD = 0.8%
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Grade 7 (B equivalent): PD = 3%
â”‚  â”‚   â”‚   â”‚   â”‚   â””â”€ Grade 9 (CCC equivalent): PD = 10%+
â”‚  â”‚   â”‚   â”‚   â””â”€ Validation: Compare grades to external ratings (should correlate)
â”‚  â”‚   â”‚   â”œâ”€ Statistical Models:
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Logistic regression: PD = 1 / (1 + e^{-[intercept + Î²1Ã—Leverage + Î²2Ã—ROE + ...]})
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Merton model: Structural approach (firm value vs debt)
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Firm equity value = Max(Assets - Debt, 0)
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Default when Assets < Debt
â”‚  â”‚   â”‚   â”‚   â”‚   â”œâ”€ PD = N[-DD] (distance to default)
â”‚  â”‚   â”‚   â”‚   â”‚   â””â”€ DD = [ln(Assets/Debt) + (Î¼ - ÏƒÂ²/2)T] / (ÏƒâˆšT)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Machine learning: Gradient boosting, neural networks (increasing use)
â”‚  â”‚   â”‚   â”‚   â””â”€ Advantages: Capture nonlinearities, incorporate many features
â”‚  â”‚   â”‚   â””â”€ Regulatory Criteria (IRB model approval):
â”‚  â”‚   â”‚       â”œâ”€ At least 5 years historical data
â”‚  â”‚   â”‚       â”œâ”€ Include stress periods (recessions, crises)
â”‚  â”‚   â”‚       â”œâ”€ Testing: Backtesting (actual defaults vs model PD)
â”‚  â”‚   â”‚       â”œâ”€ Stability: PD shouldn't change >50% year-to-year (unless business change)
â”‚  â”‚   â”‚       â”œâ”€ Granularity: Enough observations per grade to be reliable
â”‚  â”‚   â”‚       â””â”€ Regulatory approval required before use
â”‚  â”‚   â”‚
â”‚  â”‚   â”œâ”€ Retail IRB (Special treatment):
â”‚  â”‚   â”‚   â”œâ”€ Lower correlation (0.05 vs 0.12 for corporates) â†’ lower RW
â”‚  â”‚   â”‚   â”œâ”€ Typical RW: 5-15% (vs 35%+ SA for mortgages)
â”‚  â”‚   â”‚   â”œâ”€ Rationale: Individual defaults don't correlate highly (diversified pool)
â”‚  â”‚   â”‚   â””â”€ Constraint: Portfolio effects (recession â†’ all retail defaults spike)
â”‚  â”‚   â”‚
â”‚  â”‚   â””â”€ IRB Floors & Output Floor:
â”‚  â”‚       â”œâ”€ IRB floor (pre-2023): Minimum RWA = 75% Ã— SA RWA
â”‚  â”‚       â”œâ”€ Output floor (post-2023): Minimum RWA = 72.5% Ã— SA RWA
â”‚  â”‚       â”œâ”€ Effect: Prevents IRB from reducing RWA >27.5% vs SA
â”‚  â”‚       â”œâ”€ Example: SA RWA = $400B, IRB calculated = $200B â†’ Use $290B (72.5% floor)
â”‚  â”‚       â””â”€ Phase-in: 72.5% (2023) â†’ 72.5% (permanent after 2028)
â”‚
â”œâ”€ Market Risk Capital
â”‚  â”œâ”€ Basel II Approach (VaR-based, now deprecated):
â”‚  â”‚   â”œâ”€ VaR(99%, 10-day) = 1% probability of loss > this amount
â”‚  â”‚   â”œâ”€ Capital = 3 Ã— VaR + IdiosyncraticRisk + specific risk charge
â”‚  â”‚   â”œâ”€ Criticized: Doesn't capture tail severity (CVaR importance)
â”‚  â”‚   â””â”€ Status: Phased out (replaced by FRTB)
â”‚  â”‚
â”‚  â”œâ”€ Basel III/FRTB Approach (Expected Shortfall-based):
â”‚  â”‚   â”œâ”€ Expected Shortfall (CVaR) = Average loss in tail 1%
â”‚  â”‚   â”œâ”€ Calculation Steps:
â”‚  â”‚   â”‚   â”œâ”€ Historical scenarios (250 days, last year of data)
â”‚  â”‚   â”‚   â”œâ”€ Mark portfolio to each scenario
â”‚  â”‚   â”‚   â”œâ”€ Calculate losses
â”‚  â”‚   â”‚   â”œâ”€ Sort, select worst 1% of days
â”‚  â”‚   â”‚   â”œâ”€ Average = ES
â”‚  â”‚   â”‚   â””â”€ Capital = 3 Ã— ES (provides buffer)
â”‚  â”‚   â”œâ”€ Stressed ES: Same calculation but using pre-crisis market period
â”‚  â”‚   â”‚   â”œâ”€ Captures regime where correlations spike
â”‚  â”‚   â”‚   â”œâ”€ ES from 2008-2009 would be used if that's the worst period
â”‚  â”‚   â”‚   â””â”€ Capital charge = max(ES_current, ES_stressed)
â”‚  â”‚   â”œâ”€ Modeling Components:
â”‚  â”‚   â”‚   â”œâ”€ Delta (linear sensitivity): âˆ‚P/âˆ‚S Ã— S change
â”‚  â”‚   â”‚   â”œâ”€ Gamma (convexity): Â½ Ã— âˆ‚Â²P/âˆ‚SÂ² Ã— (Î”S)Â²
â”‚  â”‚   â”‚   â”œâ”€ Vega (volatility): âˆ‚P/âˆ‚Ïƒ Ã— Î”Ïƒ
â”‚  â”‚   â”‚   â”œâ”€ Rho (interest rate): âˆ‚P/âˆ‚r Ã— Î”r
â”‚  â”‚   â”‚   â””â”€ Basis risk: Hedge doesn't perfectly offset (e.g., index vs individual stock)
â”‚  â”‚   â””â”€ Capital charge (FRTB):
â”‚  â”‚       â”œâ”€ Sensitivities method (simplified): Fixed capital per dollar of delta/gamma/vega
â”‚  â”‚       â”œâ”€ Full revaluation (complex): Run model on scenarios
â”‚  â”‚       â””â”€ Typical: 5-10% of notional for equity portfolios
â”‚  â”‚
â”‚  â”œâ”€ Interest Rate Risk in Banking Book (IRRBB):
â”‚  â”‚   â”œâ”€ Non-trading positions (deposits, mortgages at fixed rates)
â”‚  â”‚   â”œâ”€ Pillar 2 capital add-on (not formulaic)
â”‚  â”‚   â”œâ”€ Measured as: Loss if rates move Â±200 bps
â”‚  â”‚   â”œâ”€ Example: Deposit base $100B at 1%, mortgages $80B at 3.5%
â”‚  â”‚   â”‚   Rate +200bps: Cost deposits â†‘ by $2B, mortgage income â†‘ $1.6B â†’ Net loss $0.4B
â”‚  â”‚   â”œâ”€ Counterparty (counterparty risk): Derive exposure value using SA-CCR
â”‚  â”‚   â””â”€ CVA (Credit Valuation Adjustment): Risk that counterparty defaults
â”‚  â”‚       â”œâ”€ Not just mark-to-market, but future exposure too
â”‚  â”‚       â”œâ”€ Capital charge on derivative portfolio
â”‚  â”‚       â””â”€ Typically 2-5% of notional for active traders
â”‚  â”‚
â”‚  â””â”€ Concentration Risk (New):
â”‚      â”œâ”€ Single counterparty large exposure limit
â”‚      â”œâ”€ Exposure > 10% Tier 1 capital triggers capital charge
â”‚      â”œâ”€ Formula: 0% if < 10%, scales to 100% if very large
â”‚      â”œâ”€ Example: Bank Tier1 = $20B, exposure to client = $5B
â”‚      â”‚   Limit = 10% Ã— $20B = $2B; $5B exceeds by $3B â†’ capital charge on $3B
â”‚      â””â”€ Interconnectedness adds surcharge (systemically important counterparties)
â”‚
â”œâ”€ Operational Risk Capital
â”‚  â”œâ”€ Standardized Approach (SA):
â”‚  â”‚   â”œâ”€ Capital Charge = 12% Ã— Indicator (average 3-year)
â”‚  â”‚   â”œâ”€ Indicator typically = Gross revenue (adjusted for business lines)
â”‚  â”‚   â”œâ”€ Calculation:
â”‚  â”‚   â”‚   â”œâ”€ Calculate indicator for each year (past 3 years)
â”‚  â”‚   â”‚   â”œâ”€ Average the 3 years
â”‚  â”‚   â”‚   â”œâ”€ Multiply by 12%
â”‚  â”‚   â”‚   â””â”€ Result = Capital required
â”‚  â”‚   â”œâ”€ Example:
â”‚  â”‚   â”‚   â”œâ”€ Year 1 revenue: $100M, OpRisk indicator = $100M
â”‚  â”‚   â”‚   â”œâ”€ Year 2 revenue: $120M, OpRisk indicator = $120M
â”‚  â”‚   â”‚   â”œâ”€ Year 3 revenue: $110M, OpRisk indicator = $110M
â”‚  â”‚   â”‚   â”œâ”€ Average = $110M
â”‚  â”‚   â”‚   â”œâ”€ Capital required = 12% Ã— $110M = $13.2M
â”‚  â”‚   â””â”€ Simplified but pro-cyclical (revenue down in crisis â†’ lower capital)
â”‚  â”‚
â”‚  â”œâ”€ Advanced Approach (AA):
â”‚  â”‚   â”œâ”€ Used by large systemically important banks
â”‚  â”‚   â”œâ”€ Components:
â”‚  â”‚   â”‚   â”œâ”€ Expected Loss (EL): E[Severity Ã— Frequency]
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Frequency: How many operational events per year (e.g., 5 events)
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Severity: Average loss per event (e.g., $2M average)
â”‚  â”‚   â”‚   â”‚   â””â”€ EL = 5 Ã— $2M = $10M/year
â”‚  â”‚   â”‚   â”œâ”€ Unexpected Loss (UL): Tail risk charge
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Use CVaR or value-at-risk on loss distribution
â”‚  â”‚   â”‚   â”‚   â”œâ”€ UL = (99.9% VaR - EL) / 8 [convert to capital]
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Typical: $30-50M (tail scenarios, rare large losses)
â”‚  â”‚   â”‚   â”‚   â””â”€ Example: 0.1% tail = $100M loss, EL = $10M, UL = ($100M - $10M) / 8 = $11.25M
â”‚  â”‚   â”‚   â”œâ”€ Internal Loss Multiplier (ILM): Adjustment for data quality/model risk
â”‚  â”‚   â”‚   â”‚   â”œâ”€ If severe crisis happened recently, data quality increases
â”‚  â”‚   â”‚   â”‚   â”œâ”€ ILM = [1 + (OpRisk events in crisis Ã— weight)] / baseline
â”‚  â”‚   â”‚   â”‚   â”œâ”€ Range: 0.8 - 1.5
â”‚  â”‚   â”‚   â”‚   â””â”€ Effect: Increase capital charge post-crisis (pro-cyclical tension)
â”‚  â”‚   â”‚   â””â”€ Diversification: Reduce capital by potential portfolio diversification
â”‚  â”‚   â”‚       â”œâ”€ Different risk categories less than additive
â”‚  â”‚   â”‚       â”œâ”€ Correlation < 1 â†’ portfolio effect
â”‚  â”‚   â”‚       â””â”€ Divisor typically 1.5 - 2.5
â”‚  â”‚   â”œâ”€ Capital charge: OpRisk = [EL + UL] Ã— ILM Ã— 1/Diversification
â”‚  â”‚   â”œâ”€ Typical: $25-75M for large banks
â”‚  â”‚   â””â”€ Regulatory approval required (strict model validation)
â”‚  â”‚
â”‚  â”œâ”€ Loss Event Categories (Basel Operational Risk Framework):
â”‚  â”‚   â”œâ”€ Internal fraud: Employee theft, trade desk misconduct
â”‚  â”‚   â”œâ”€ External fraud: ATM theft, cyber attacks, client fraud
â”‚  â”‚   â”œâ”€ DLOA (Disruption of Business & System Failures): IT outages, power failures
â”‚  â”‚   â”œâ”€ EPCE (Employment Practices & Client Relations): Wrongful termination suits, discrimination
â”‚  â”‚   â”œâ”€ Damage to Physical Assets: Natural disasters, vandalism
â”‚  â”‚   â”œâ”€ Business Disruption & System Failures: Infrastructure failure
â”‚  â”‚   â”œâ”€ Execution, Delivery, Process Management: Trade errors, settlement fails
â”‚  â”‚   â”œâ”€ Client/Product line specific: Model risk, product defects
â”‚  â”‚   â””â”€ Correlations: Typically low (diversified portfolio)
â”‚  â”‚
â”‚  â””â”€ Regulatory Scrutiny:
â”‚      â”œâ”€ Stress scenarios including operational events
â”‚      â”œâ”€ Enhanced monitoring for fraud/cyber risk
â”‚      â”œâ”€ Technology resilience assessments
â”‚      â””â”€ Third-party risk management (outsourced functions)
â”‚
â”œâ”€ Total Capital Requirement Combination
â”‚  â”œâ”€ Formula: Total Capital = Credit RW + Market RW + OpRisk
â”‚  â”‚   All converted to capital percentage of RWA
â”‚  â”‚   Capital = 8% Ã— (Credit RWA + Market RWA + OpRisk RWA)
â”‚  â”œâ”€ Interaction Effects:
â”‚  â”‚   â”œâ”€ Correlated risks: Market downturn + credit defaults + operational stress
â”‚  â”‚   â”œâ”€ Stress testing explicitly models combinations
â”‚  â”‚   â”œâ”€ No diversification benefit (regulatory conservative)
â”‚  â”‚   â””â”€ Pillar 2 (supervisor) can add if combinations appear dangerous
â”‚  â”œâ”€ Buffers (on top of 8% minimum):
â”‚  â”‚   â”œâ”€ Capital Conservation Buffer (CCB): 2.5%
â”‚  â”‚   â”œâ”€ Countercyclical Buffer (CyCB): 0-2.5%
â”‚  â”‚   â”œâ”€ G-SIB surcharge: 1-3.5%
â”‚  â”‚   â””â”€ Total effective minimum: 12-17% for large banks
â”‚  â””â”€ Real Bank Example (Hypothetical Large Bank):
â”‚      â”œâ”€ Credit RWA: $300B (75% of total)
â”‚      â”œâ”€ Market RWA: $80B (20% of total)
â”‚      â”œâ”€ OpRisk RWA: $20B (5% of total)
â”‚      â”œâ”€ Total RWA: $400B
â”‚      â”œâ”€ Minimum capital (8%): $32B
â”‚      â”œâ”€ CCB (2.5%): $10B
â”‚      â”œâ”€ CyCB (1%): $4B
â”‚      â”œâ”€ G-SIB surcharge (2%): $8B
â”‚      â”œâ”€ Total required: $54B (13.5% of RWA)
â”‚      â”œâ”€ Typical buffer: Hold $60-70B (15-17.5%)
â”‚      â””â”€ Leverage ratio floor (3% of $1.5T assets) = $45B also binding
â”‚
â”œâ”€ Risk Calculation System Architecture
â”‚  â”œâ”€ Data Pipeline:
â”‚  â”‚   â”œâ”€ Market data: Daily prices, rates, volatility (real-time)
â”‚  â”‚   â”œâ”€ Portfolio positions: Securities, derivatives, loans (daily)
â”‚  â”‚   â”œâ”€ Credit data: Ratings, default history, PD models (monthly updates)
â”‚  â”‚   â”œâ”€ Operational data: Loss events, audit reports (annual compilation)
â”‚  â”‚   â””â”€ Regulatory data: Counterparty exposures, large exposures (monthly)
â”‚  â”œâ”€ Model Components:
â”‚  â”‚   â”œâ”€ Credit risk models: PD/LGD/EAD for each borrower
â”‚  â”‚   â”œâ”€ Market risk models: VaR, ES, Greeks (delta/gamma/vega)
â”‚  â”‚   â”œâ”€ Operational risk models: Frequency/severity, loss events
â”‚  â”‚   â”œâ”€ Correlation models: How risks move together
â”‚  â”‚   â””â”€ Scenario analysis: Tail events, stress testing
â”‚  â”œâ”€ Computing:
â”‚  â”‚   â”œâ”€ Overnight: Full capital calculation (RWA recalculation)
â”‚  â”‚   â”œâ”€ Intraday: Market VaR updates (key for trading desks)
â”‚  â”‚   â”œâ”€ Monthly: Stress testing, regulatory reporting
â”‚  â”‚   â”œâ”€ Quarterly: Capital forecast, buffer testing
â”‚  â”‚   â””â”€ Annual: IRB model backtesting, regulatory approval prep
â”‚  â””â”€ Audit & Governance:
â”‚      â”œâ”€ Model risk management: Independent review of key models
â”‚      â”œâ”€ Backtesting: Compare predicted vs actual losses
â”‚      â”œâ”€ Sensitivity analysis: How capital changes with parameter shifts
â”‚      â”œâ”€ Stress testing: Extreme scenarios
â”‚      â””â”€ Board oversight: Capital adequacy, strategic implications
```

**Interaction Example:**  
Bank portfolio: $100M in BBB corporate loans. Credit RW (SA) = 100%, RWA = $100M. Market position: Short $20M corporate bonds (hedge). Market RW (FRTB) per 2% ES charge = $0.4M. OpRisk allocated 5% â†’ $5M. Total capital = 8% Ã— ($100M + $0.4M + $5M) = $8.04M + buffers = ~$12M.

## 5. Challenge Round
- Map bank's $500M portfolio to Basel SA risk weights; calculate RWA
- Design F-IRB PD model for mid-market corporates (regression on financials)
- Estimate LGD for real estate collateral using historical recovery data
- Compare leverage ratio floor vs RW-based capital for leverage-heavy portfolio
- Run output floor test: Is 72.5% constraint binding for your IRB model?

## 6. Key References
- [BIS, "The Standardized Approach for Credit Risk" (2017)](https://www.bis.org/basel_framework/crossfunctional/output_floor.pdf) â€” Official regulation
- [BIS, "Internal Ratings-Based Approach" (2017)](https://www.bis.org/basel_framework/standard/crb.htm) â€” IRB formula and calibration
- [Federal Reserve, "CCAR 2024 Stress Test Scenarios"](https://www.federalreserve.gov/banking/ccar-capital-planning.htm) â€” Implementation example (US)
- [Gordy, "A Comparative Anatomy of Credit Risk Models" (2000), JFQA](https://www.jstor.org/) â€” Theoretical foundation

---
**Status:** Core regulatory framework (2008-present, continuously refined) | **Complements:** Basel III Framework, Liquidity Risk, Stress Testing, Market Risk
