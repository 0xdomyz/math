# Default Probability (PD)

## 1. Concept Skeleton
**Definition:** Probability of default (PD) quantifies the likelihood that a borrower or counterparty fails to meet contractual payment obligations within a specified time horizon, typically expressed as an annual percentage (e.g., 2.3% one-year PD = 2.3% chance of default in next 12 months). PD serves as the cornerstone input for credit risk quantification across loan pricing, capital allocation, and loss provisioning.

**Purpose:** PD estimation fulfills critical functions across credit risk management, regulatory compliance, and business operations:
- **Credit decisioning & pricing**: Banks determine loan approval thresholds and risk-based interest rates. A mortgage applicant with 0.5% PD receives prime pricing (3.5% rate), while 3% PD borrower pays subprime rates (7.5% or declined). Automated underwriting engines use PD cutoffs: approve if PD < 5%, review manually if 5-10%, reject if >10%
- **Regulatory capital calculation**: Basel III IRB approach computes risk-weighted assets (RWA) using PD as key input: $\text{RWA} = \text{EAD} \times \text{RW}(PD, LGD, M)$ where risk weight increases nonlinearly with PD. A corporate loan with 2% PD requires ~60% RW vs 100%+ RW for 8% PD. Lower PD → lower capital required → higher ROE
- **IFRS 9 / CECL provisioning**: Expected credit loss = PD × LGD × EAD. Banks must book lifetime expected losses at origination (CECL) or Stage 2 transfer (IFRS 9). Forward-looking PD incorporating macroeconomic scenarios determines provision levels. US banks increased provisions by $50B+ in Q1 2020 due to COVID-driven PD increases
- **Portfolio risk management**: Aggregate portfolio PD drives credit VaR, economic capital, concentration limits, and stress testing. Chief Risk Officer monitors portfolio-weighted average PD trends: 2.1% (2019) → 5.3% (Q2 2020) → 2.8% (2023) reflecting credit cycle
- **Trading & CVA**: Counterparty credit risk for derivatives uses PD to calculate credit valuation adjustment (CVA). 5-year PD = 3% on $100M notional swap → CVA ≈ $500k assuming 40% LGD and exposure profile

**Prerequisites:** Mastering PD estimation requires grounding in complementary technical and domain knowledge:
- **Statistical foundations**: Logistic/probit regression for binary outcomes (default/no default), maximum likelihood estimation for parameter fitting, survival analysis for time-to-event modeling (Cox proportional hazards, Kaplan-Meier curves), handling of censored data (loans still performing at observation end)
- **Credit domain expertise**: Default definitions across jurisdictions (90 DPD in US retail, 180 DPD in some emerging markets, Basel definition of "unlikely to pay"), distinction between technical default (covenant breach) vs payment default, understanding of bankruptcy law (Chapter 11 reorganization vs Chapter 7 liquidation affects recovery timing)
- **Economic & market drivers**: Macro variables impacting default rates (unemployment rate elasticity: +1pp unemployment → +0.4pp default rate for prime mortgages), credit cycle behavior (PD mean-reversion, typical 5-7 year cycles), industry-specific factors (commodity price shocks for energy sector, seasonality in agriculture lending)
- **Quantitative finance**: Merton structural model linking equity volatility to default barrier, reduced-form intensity models (hazard rate calibration from CDS spreads: $\lambda(t) = \frac{\text{CDS spread}}{1-\text{LGD}}$), transition matrices for rating migration, copula models for joint default correlation
- **Related Topics to Cross-Reference**: Review [Credit Risk Definition](../credit_risk_definition/credit_risk_definition.md) for foundational concepts, [Loss Given Default](../loss_given_default/loss_given_default.md) for complementary LGD estimation, [Expected Loss](../expected_loss/expected_loss.md) for EL = PD × LGD × EAD calculations, [Transition Matrices](../transition_matrices/transition_matrices.md) for multi-period PD modeling, [Credit Scoring Models](../../scorecard_models/scorecard_models.md) for practical model implementation, [Model Validation](../model_validation/model_validation.md) for backtesting and performance monitoring

## 2. Comparative Framing
| Approach | Credit Scoring | Rating Agency | CDS Spreads | Structural Models |
|----------|---------------|--------------|------------|------------------|
| **Data** | Borrower financials, behavior | Qualitative + quantitative | Market prices | Firm value dynamics |
| **Horizon** | Typically 1 year | Through-the-cycle | Implied short-term | Depends on model |
| **Update Frequency** | Annual/quarterly | Periodic | Continuous | Model-dependent |
| **Calibration** | Historical defaults | Long history | Market-implied | Firm-specific |

## 3. Examples + Counterexamples

**Simple Example: Logistic Regression PD Calculation**  
A retail bank builds a PD model using logistic regression with coefficients: β₀=-3.2 (intercept), β₁=0.015 (debt-to-income), β₂=-0.008 (credit score). For borrower with DTI=45% and credit score=680:
- Linear predictor: $z = -3.2 + 0.015(45) - 0.008(680) = -3.2 + 0.675 - 5.44 = -7.965$
- PD: $PD = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{7.965}} = \frac{1}{1+2874.8} = 0.000348 = 0.0348\%$

This borrower has excellent credit profile (low PD). For comparison, borrower with DTI=60%, score=550:
- $z = -3.2 + 0.015(60) - 0.008(550) = -3.2 + 0.9 - 4.4 = -6.7$
- $PD = \frac{1}{1+e^{6.7}} = 0.0012 = 0.12\%$ (3.4x higher risk)

**Realistic Failure Case: Pro-Cyclicality in Crisis**  
Using fixed PD during crisis. Pre-2008, investment-grade corporate PD averaged 0.10%. During Q4 2008-Q1 2009, realized PD spiked to 1.2% (12x increase). Banks using static through-the-cycle PD estimates badly underestimated:
- **Capital shortfall**: RWA calculated with 0.10% PD → 8% capital ratio. Actual 1.2% PD required 2.5x more capital
- **Loan loss provisions**: Expected losses understated by 90% in IFRS 9 calculations
- **Portfolio concentration**: All "independent" borrowers defaulted together due to common macro shock (correlation jumped from 0.15 to 0.65)
- **Mitigation**: Point-in-time PD models that adjust for current economic conditions (unemployment rate, GDP growth, credit spreads); stress testing with severe but plausible scenarios

**Edge Case: Thin-File Borrowers**  
New borrower (age 22, first credit card) with no default history; cannot use empirical default rates. Traditional solutions:
- **Synthetic scoring**: Use alternative data (rent payments, utility bills, mobile phone payments, bank account history). Research shows mobile money transactions predict default with AUC=0.68
- **Peer comparison**: Assign PD based on cohort (age 18-25, income <$30k): historical 5-year PD = 2.3%
- **Conservative approach**: Assign worst decile PD (8-10%) until 12-month payment history established
- **Real example**: Lending Club's model uses 27 features including FICO, DTI, employment length. For thin files (FICO unavailable), uses DTI + income + inquiries with 30% higher PD estimate (calibration adjustment)

**Technical Counterexample: Confusing PD with Default Rate**  
Common misconception: "If portfolio default rate = 2%, then average PD = 2%"

**Why this fails**: Default rate is realized outcome for specific cohort in specific period; PD is forward-looking probability estimate. Key differences:
1. **Selection bias**: Approved loan portfolio excludes high-risk rejects. If bank approves 70% of applicants (rejecting worst 30%), observed default rate 2% does not represent true population PD. True population PD might be 5% including rejects
2. **Survivorship**: Default rate measures defaults among survivors. 1-year default rate = 2%, but conditional 2-year rate ≠ 4% due to survivorship (healthiest remain)
3. **Economic conditions**: Historical 2% rate during expansion; recession PD could be 6%. Default rate is backward-looking; PD should be forward-looking

**Correct approach**: PD = calibrated model output adjusted for:
- Economic cycle position (current conditions vs training period)
- Portfolio composition shifts (if customer mix changed)  
- Margin of conservatism (regulatory add-ons, model uncertainty)

**Real-world impact**: UK credit card issuer observed 3.2% account-level annual default rate (2016-2019). New regulations required forward-looking PD for IFRS 9. Adjusted PD = 4.1% after incorporating:
- Base PD from logistic model: 3.5%
- Macro adjustment for Brexit uncertainty: +0.4%
- Thin-file adjustment: +0.2%
- Model conservatism buffer: +0.0% (model well-calibrated)

Using observed 3.2% rate would have understated expected losses by 22%.

## 4. Layer Breakdown
```
Probability of Default Framework:
â”œâ”€ PD Definition:
â”‚   â”œâ”€ One-year PD: P(default within 12 months)
â”‚   â”œâ”€ Multi-year PD: Cumulative over T years
â”‚   â”œâ”€ Conditional PD: P(default in year t | survived to t)
â”‚   â””â”€ Lifetime PD: P(default at any point in contract
â”œâ”€ Calibration Methods:
â”‚   â”œâ”€ Empirical: Default rate from historical cohorts
â”‚   â”œâ”€ Regression-based: Logistic/probit model P(default) = f(covariates)
â”‚   â”œâ”€ Transition matrices: Rating migration to default
â”‚   â”œâ”€ CDS-implied: Backed out from market spreads
â”‚   â””â”€ Structural: From Merton model (asset value dynamics)
â”œâ”€ PD Levels by Credit Quality:
â”‚   â”œâ”€ AAA: 0.01% - 0.05% (excellent)
â”‚   â”œâ”€ A: 0.05% - 0.20% (good)
â”‚   â”œâ”€ BBB: 0.20% - 1.00% (investment grade)
â”‚   â”œâ”€ BB: 1.0% - 3.0% (speculative)
â”‚   â”œâ”€ B: 3.0% - 8.0% (high risk)
â”‚   â””â”€ D: 100% (default)
â”œâ”€ Point-in-Time vs Through-the-Cycle:
â”‚   â”œâ”€ PIT-PD: Reflects current economic conditions
â”‚   â”œâ”€ TTC-PD: Long-run average, smoothed across cycles
â”‚   â””â”€ Conversion: Adjust for cycle position
â””â”€ Term Structure:
    â”œâ”€ Survival probability: S(t) = 1 - âˆ‘_{i=1}^t PD_i
    â””â”€ Multi-year PD: 1 - S(t)
```

## 5. Challenge Round
When is PD estimation problematic?
- **Limited history**: New asset classes, rare events; can't rely on empirical frequencies
- **Structural breaks**: Credit wars, regulatory changes; historical patterns don't hold
- **Default clustering**: During crises, correlated defaults violate independence assumption
- **Rating inflation**: Models underestimate distress (Enron had AA rating before default)
- **Selection bias**: Observed defaults only from approved loans; rejected applicants' actual risk unknown

## 6. Key References
- **Basel Committee on Banking Supervision, "Basel III: IRB Approaches"** (2017) - [BIS Framework](https://www.bis.org/basel_framework/chapter/CRE/30.htm) - Regulatory standards for PD estimation, through-the-cycle requirements, minimum floors (0.03%), validation requirements, and treatment across asset classes. Essential for understanding IRB capital calculations
- **Merton, R.C., "On the Pricing of Corporate Debt"** (1974) - Journal of Finance - Foundational paper linking firm value dynamics to default probability via structural model. Shows PD = N(d₂) where d₂ depends on leverage ratio, asset volatility, and time horizon. Widely used for public corporates
- **Altman, E.I., "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy"** (1968) - Journal of Finance - Introduced Z-score model combining liquidity, profitability, leverage, solvency. Still used as benchmark; shows working capital/assets, retained earnings/assets predict default with 72% accuracy
- **Thomas, L.C., Edelman, D.B., Crook, J.N., "Credit Scoring and Its Applications"** (2002) - SIAM Monograph - Comprehensive textbook covering logistic regression, decision trees, neural networks for PD estimation. Chapters 4-5 detail model calibration, validation, and practical implementation
- **Löffler, G., Posch, P.N., "Credit Risk Modeling using Excel and VBA"** (2011) - Wiley - Practical guide with working examples of PD calibration from rating transitions, CDS spreads, and equity prices. Includes VBA code for Merton model, intensity models
- **Dwyer, D.W., "The Distribution of Defaults and Bayesian Model Validation"** (2007) - Journal of Risk Model Validation - Addresses how to validate PD models when defaults are rare. Proposes Bayesian approaches combining prior beliefs with sparse data
- **IFRS 9 Financial Instruments** (2014) - International Accounting Standards Board - Requires forward-looking PD for expected credit loss provisioning. Specifies lifetime PD for Stage 2/3 exposures, 12-month PD for Stage 1, incorporation of macroeconomic scenarios

---
**Status:** Core credit risk parameter | **Complements:** Credit Risk Definition, LGD, EAD, Expected Loss
