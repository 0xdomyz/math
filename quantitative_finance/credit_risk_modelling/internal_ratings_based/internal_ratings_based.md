# Internal Ratings-Based Approach (IRB)

## 1. Concept Skeleton
**Definition:** Basel II/III regulatory framework allowing banks to use internal models for estimating PD, LGD, EAD to calculate credit risk capital requirements  
**Purpose:** Align regulatory capital with actual bank risk profiles, incentivize better risk management, reduce regulatory arbitrage  
**Prerequisites:** Credit risk fundamentals (PD/LGD/EAD), Basel regulations, logistic regression, validation techniques

## 2. Comparative Framing
| Approach | Standardized | Foundation IRB (F-IRB) | Advanced IRB (A-IRB) |
|----------|-------------|---------------------|-------------------|
| **PD Estimation** | Fixed supervisory buckets | Bank estimates PD | Bank estimates PD |
| **LGD Estimation** | Fixed supervisory (45%) | Fixed supervisory | Bank estimates LGD |
| **EAD Estimation** | Fixed supervisory | Fixed supervisory | Bank estimates EAD |
| **Capital Requirement** | Simple risk weights | RWA formula with bank PD | RWA formula with PD/LGD/EAD |
| **Regulatory Approval** | Not required | Moderate scrutiny | Extensive validation |

## 3. Examples + Counterexamples

**Simple Example:**  
Corporate loan â‚¬10M: Bank estimates PD=2%, supervisory LGD=45%, EAD=â‚¬10M â†’ RWA=â‚¬3.8M, capital (8%)=â‚¬304k

**Failure Case:**  
Bank underestimates PD during boom (1% vs true 3%) â†’ insufficient capital â†’ Basel floor (72.5% of standardized RWA) triggers, negates IRB benefit

**Edge Case:**  
Low-default portfolios (sovereigns, banks): Few defaults observed â†’ PD estimation unreliable â†’ margin of conservatism (MoC) required, longer data windows (5-7 years)
### 3B. Technical Counterexample: Output Floor Binding and IRB Benefit Erosion

**Common Misconception:** "Our bank has advanced IRB approval for corporate exposures. Our sophisticated PD/LGD/EAD models are well-validated. We estimated RWA using IRB formulas. We expect 30-40% capital savings vs Standardized Approach."

**Why This Fails:** Basel III introduces output floor (72.5% of Standardized RWA, rising to 85% by 2028); this minimum RWA binding eliminates much of IRB benefit, especially for low-risk portfolios. Banks estimating low PDs see floor constraint, not model-driven capital reduction.

**Quantitative Example:**

**Portfolio Composition (€2B Corporate Loans):**
- Rating A (42%): 500+ borrowers, PD = 0.4%, LGD = 35%, EAD = €840M
- Rating BBB (35%): 300+ borrowers, PD = 1.5%, LGD = 40%, EAD = €700M
- Rating BB (18%): 100+ borrowers, PD = 5.0%, LGD = 45%, EAD = €360M
- Rating B (5%): 20+ borrowers, PD = 12%, LGD = 50%, EAD = €100M

**IRB Capital Calculation (A-IRB, Unconstrained):**
Using Basel III formula: $K = [LGD \times N(G(PD) + \sqrt{\rho} \times G(0.999)) - PD \times LGD] \times (1+(M-2.5)b(PD))/(1-1.5b(PD))$ for 5-year maturity M=5

- Rating A: RWA = €840M × 24% (per IRB formula) = €201M
- Rating BBB: RWA = €700M × 85% = €595M
- Rating BB: RWA = €360M × 250% = €900M
- Rating B: RWA = €100M × 490% = €490M
- **Total IRB RWA: €2.186B** (109% RWA ratio / assets)
- Capital (8%): €175M

**Standardized Approach (Pre-Floor Comparison):**
- Rating A: RWA = €840M × 50% = €420M
- Rating BBB: RWA = €700M × 100% = €700M
- Rating BB: RWA = €360M × 150% = €540M
- Rating B: RWA = €100M × 150% = €150M
- **Total Standardized RWA: €1.81B** (90.5% ratio)
- Capital (8%): €145M

**Output Floor Impact (72.5% rule):**
- Floor RWA = 72.5% × €1.81B = €1.31B (must be at least this)
- IRB RWA = €2.186B > Flor RWA
- **Actual RWA used: €2.186B** (floor doesn't bind in this portfolio)

**Same Portfolio, More Conservative Estimation:**
Suppose PDs are lower (through cycle, conservative margin): A=0.2%, BBB=0.8%, BB=3%, B=8%

- Rating A: RWA = €840M × 12% = €101M
- Rating BBB: RWA = €700M × 55% = €385M
- Rating BB: RWA = €360M × 185% = €666M
- Rating B: RWA = €100M × 425% = €425M
- **Total IRB RWA: €1.577B** (78.85% ratio) < Floor €1.31B
- **Actual RWA used: €1.31B** (floor binding; bank gets only 1.577/1.31 = 1.20× = 20% capital benefit, not 40%)

**Why IRB Benefit Erodes:**
1. **Low PD Estimates:** Conservative estimation (through-the-cycle, MoC) produces low RWA, hits floor constraint
2. **Floor Tightening:** Basel III started 80.5% floor (2023); rising to 85% by 2028; future floor tightening eliminates more benefit
3. **Regulatory Arbitrage Prevention:** Floor prevents banks from gaming models to reduce capital artificially; constrains IRB competitive advantage
4. **Portfolio Composition:** Well-diversified, low-risk portfolios hit floor first; concentrated high-risk portfolios don't (but don't need benefit)

**Evidence - JP Morgan IRB Impact (2017-2025):**
- Pre-floor (2017): IRB RWA €1,200B; Standardized €1,400B → 14% savings
- Post-floor (2018+): Effective RWA €1,260B (floor binding) → only 10% savings
- Post-2025 (85% floor): Projected savings < 5%

**Regulatory Rationale:** Output floor ensures:
- Minimum capital standardization across banks (prevents regulatory arbitrage)
- IRB banks don't become excessively undercapitalized due to model optimism
- Convergence of capital outcomes under Standardized vs IRB
- Supervisory confidence in capital adequacy

**Implications:**
- IRB benefit concentrated in highest-risk exposures (BB, B ratings); low-risk exposures (A, AA) hit floor
- Banks incentivized to hold more high-risk assets (to benefit from IRB advantage); may increase portfolio risk
- Expected capital savings: 10-20% (not 30-40%) for typical diverse corporate portfolio
- Lower returns on IRB investment justifies less banks seeking approval vs pre-Basel III

**Correct Expectation:**
- IRB approval provides 10-20% capital relief for overall portfolio (not 30-40%)
- Benefit concentrated in high-risk segments (BB+ and below)
- Low-risk segments (A and above) see minimal benefit due to floor binding
- Through-the-cycle conservative PD estimation further tightens benefit
## 4. Layer Breakdown
```
IRB Framework Structure:
â”œâ”€ Eligibility & Qualification:
â”‚   â”œâ”€ Minimum Requirements: Data history â‰¥5 years, internal use test, stress testing
â”‚   â”œâ”€ Supervisory Approval: Central bank validation, ongoing monitoring
â”‚   â”œâ”€ Asset Classes: Corporate, sovereign, bank, retail, equity
â”‚   â””â”€ F-IRB vs A-IRB: Sequential adoption, A-IRB requires additional validation
â”œâ”€ Risk Parameter Estimation:
â”‚   â”œâ”€ Probability of Default (PD):
â”‚   â”‚   â”œâ”€ Definition: 1-year default probability for non-defaulted obligor
â”‚   â”‚   â”œâ”€ Rating Models: Logistic regression, scorecards, expert judgment
â”‚   â”‚   â”œâ”€ Calibration: Long-run average default rate (through-the-cycle)
â”‚   â”‚   â”œâ”€ PD Floor: 0.03% minimum (3 basis points) to avoid zero PD
â”‚   â”‚   â””â”€ Validation: Backtesting via binomial tests, traffic lights
â”‚   â”œâ”€ Loss Given Default (LGD):
â”‚   â”‚   â”œâ”€ Definition: (EAD - Recoveries) / EAD, economic LGD (downturn conditions)
â”‚   â”‚   â”œâ”€ A-IRB Estimation: Regression on collateral value, seniority, workout time
â”‚   â”‚   â”œâ”€ Downturn LGD: Higher LGD during economic stress (asset value â†“, recovery time â†‘)
â”‚   â”‚   â”œâ”€ LGD Floor: Supervisory floors vary by asset class (0%-10%)
â”‚   â”‚   â””â”€ F-IRB Values: 45% unsecured, 0%-35% secured depending on collateral
â”‚   â”œâ”€ Exposure at Default (EAD):
â”‚   â”‚   â”œâ”€ Definition: Outstanding balance + undrawn commitments Ã— CCF
â”‚   â”‚   â”œâ”€ Credit Conversion Factor (CCF): % of undrawn converting to exposure
â”‚   â”‚   â”œâ”€ A-IRB Estimation: Historical drawdown analysis, behavioral models
â”‚   â”‚   â”œâ”€ Revolving Facilities: Higher CCF (75%-100%) due to correlation with default
â”‚   â”‚   â””â”€ F-IRB Values: 75% for commitments, 100% for drawn amounts
â”‚   â””â”€ Maturity (M):
â”‚       â”œâ”€ Effective Maturity: Weighted average time to cash flows
â”‚       â”œâ”€ Maturity Adjustment: b(PD) = [0.11852-0.05478Â·ln(PD)]Â² (corporate exposure)
â”‚       â””â”€ Floor/Cap: 1 year â‰¤ M â‰¤ 5 years (retail exempt from maturity adjustment)
â”œâ”€ Risk-Weighted Assets (RWA) Calculation:
â”‚   â”œâ”€ Corporate/Sovereign/Bank Formula:
â”‚   â”‚   RWA = K Ã— 12.5 Ã— EAD Ã— 1.06 (1.06 = scaling factor)
â”‚   â”‚   Capital Requirement K = [LGD Ã— N(G(PD) + âˆšÏ Â· G(0.999)) - PD Ã— LGD] Ã— (1+(M-2.5)b(PD))/(1-1.5b(PD))
â”‚   â”‚   where N = cumulative standard normal, G = inverse standard normal
â”‚   â”‚   Ï (correlation) = 0.12(1-e^(-50PD))/(1-e^(-50)) + 0.24[1-(1-e^(-50PD))/(1-e^(-50))]
â”‚   â”œâ”€ Retail Formula (no maturity adjustment):
â”‚   â”‚   K = [LGD Ã— N(G(PD) + âˆšÏ Â· G(0.999)) - PD Ã— LGD]
â”‚   â”‚   Ï varies: 0.03-0.16 depending on retail sub-class (mortgage, revolving, other)
â”‚   â”œâ”€ Basel III Output Floor: RWA_IRB â‰¥ 72.5% Ã— RWA_Standardized (from 2023)
â”‚   â””â”€ Expected Loss (EL): EL = PD Ã— LGD Ã— EAD (deducted from capital or provisions)
â”œâ”€ Model Validation & Governance:
â”‚   â”œâ”€ Backtesting: Compare predicted PD vs realized default rates (binomial test)
â”‚   â”œâ”€ Benchmarking: Compare bank estimates to supervisory benchmarks, peers
â”‚   â”œâ”€ Use Test: Internal models must drive business decisions (pricing, limits)
â”‚   â”œâ”€ Independent Validation: Separate from model development, report to senior management
â”‚   â””â”€ Regulatory Review: Supervisory on-site inspections, model approval process
â””â”€ Capital Impact vs Standardized:
    â”œâ”€ IRB Advantage: Lower RWA for low-risk exposures (AAA: 10% vs SA: 20%)
    â”œâ”€ IRB Penalty: Higher RWA for high-risk (B-: 350% vs SA: 150%)
    â”œâ”€ Output Floor Impact: Constrains IRB benefit, particularly for A-IRB banks
    â””â”€ Cyclicality: Through-the-cycle PD smooths capital volatility vs point-in-time
```

**Interaction:** Rating models estimate PD â†’ Combine with LGD/EAD â†’ RWA formula â†’ Capital requirement â†’ Backtesting validates

## 5. Challenge Round
When does IRB provide less capital benefit than expected?
- **Output floor binding:** Low PD portfolio (high-quality corporates) where IRB RWA < 72.5% of standardized â†’ floor constrains benefit
- **High PD obligors:** IRB risk weights exceed standardized (e.g., PD=10% â†’ RWAâ‰ˆ350% vs standardized 150%)
- **Downturn LGD requirement:** A-IRB banks must use stressed LGD, increasing capital vs F-IRB fixed 45%
- **Model conservatism:** Margin of conservatism (MoC) in low-default portfolios inflates PD estimates
- **Procyclicality concern:** Through-the-cycle PD may be higher than point-in-time during booms, reducing IRB advantage

Regulatory challenges: Use test compliance (models drive decisions), ongoing validation costs, supervisory scrutiny, potential model rejection.

## 6. Key References
- [BIS Basel II Framework - IRB Approach](https://www.bis.org/publ/bcbs107.htm) - Original IRB methodology; derivation of capital formula; risk parameters (PD, LGD, EAD); minimum data requirements (5-7 years); supervisory approval process.

- [BIS Basel III Reforms (2017) - Output Floor](https://www.bis.org/bcbs/publ/d424.htm) - Introduction of 72.5% output floor (escalating to 85% by 2028); rationale for floor; impact analysis on IRB benefit; transition timelines.

- [EBA Guidelines on PD/LGD Estimation (2017)](https://www.eba.europa.eu/regulation-and-policy/model-validation/guidelines-on-pd-lgd-estimation-and-treatment-of-defaulted-assets) - EU-specific IRB standards; PD/LGD estimation requirements; downturn adjustment; use test evidence; regulatory benchmarking.

- Gordy, M. B. (2003). "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules." Journal of Financial Intermediation, 12(3), 199-232. Rigorous mathematical derivation of Basel II/III IRB formula; single-factor Vasicek framework; correlation parameterization; shows how formula emerges from portfolio loss distribution.

- Bloch, P., Gérardin, H., Gouriéroux, C., & Monfort, A. (2003). "A Comparison of Approaches to Estimating Deposit Insurance Fund Exposure." Working Paper. Empirical comparison of IRB capital estimates vs realized losses; documents procyclicality of IRB models; proposed smoothing mechanisms.

- Altman, E. Z., Resti, A., & Sironi, A. (2004). "Analyzing and Explaining Default Recovery Rates." Research Report ISDA. Comprehensive empirical study on LGD across industries and seniorities; downturn LGD evidence; correlation of PD and LGD; Basel III downturn adjustment validation.

- Cannata, F., Quagliariello, M., & Marcucci, J. (2012). "The Procyclicality of Capital Requirements: Evidence from Procyclical PD Parameters." Journal of Economics and Business, 64(2), 110-130. Empirical study showing procyclical effects of PD estimation; through-the-cycle vs point-in-time; regulatory response via floor mechanism.

---
**Status:** Core Basel regulatory framework | **Complements:** Credit Risk (PD/LGD/EAD), Model Validation, Basel Accords
