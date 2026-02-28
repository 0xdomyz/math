# Credit Risk Model Validation

## 1. Concept Skeleton
**Definition:** Independent assessment of credit risk model accuracy, stability, and regulatory compliance through backtesting, benchmarking, and sensitivity analysis  
**Purpose:** Ensure models produce reliable risk estimates, meet Basel requirements, identify model limitations before deployment  
**Prerequisites:** Statistical testing, credit risk models (PD/LGD/EAD), ROC curves, binomial tests, regulatory standards

## 2. Comparative Framing
| Validation Type | Backtesting | Benchmarking | Sensitivity Analysis | Regulatory Review |
|-----------------|-------------|--------------|---------------------|-------------------|
| **Focus** | Predicted vs realized outcomes | Model vs peers/alternatives | Parameter stability | Compliance with standards |
| **Metric** | Binomial test, traffic lights | Rank ordering, discrimination | Coefficient variation | Documentation, use test |
| **Frequency** | Annual minimum | Model development, review | Ongoing monitoring | Supervisory cycle (3-5 years) |
| **Outcome** | Pass/Fail calibration | Relative performance | Robustness assessment | Approval/rejection |

## 3. Examples + Counterexamples

**Simple Example:**  
PD model backtesting: Predicted 2% default rate, realized 2.3% over 1000 obligors â†’ binomial test p=0.62 (pass, within confidence interval)

**Failure Case:**  
LGD model: Predicted 40% LGD during boom, realized 65% in crisis â†’ no downturn adjustment â†’ fails regulatory validation, requires recalibration

**Edge Case:**  
Low-default portfolio (sovereigns): 3 defaults over 5 years from 200 obligors â†’ insufficient data for binomial test power â†’ requires qualitative validation, stress testing

## 4. Layer Breakdown
```
Model Validation Framework:
â”œâ”€ Quantitative Validation:
â”‚   â”œâ”€ Discriminatory Power (PD Models):
â”‚   â”‚   â”œâ”€ ROC Curve: Plot TPR vs FPR, visual assessment
â”‚   â”‚   â”œâ”€ AUC (Area Under Curve): 0.5 (random) to 1.0 (perfect)
â”‚   â”‚   â”‚   â””â”€ Thresholds: AUC < 0.60 poor, 0.70-0.80 acceptable, >0.80 strong
â”‚   â”‚   â”œâ”€ Gini Coefficient: Gini = 2Ã—AUC - 1, range [0, 1]
â”‚   â”‚   â”œâ”€ AR (Accuracy Ratio): AR = Gini_model / Gini_perfect
â”‚   â”‚   â””â”€ KS Statistic: max|CDF_defaults - CDF_non-defaults|
â”‚   â”œâ”€ Calibration (PD Models):
â”‚   â”‚   â”œâ”€ Binomial Test: For each rating grade, test Hâ‚€: realized DR = predicted PD
â”‚   â”‚   â”‚   â””â”€ Test statistic: (n_defaults - nÃ—PD) / âˆš(nÃ—PDÃ—(1-PD)) ~ N(0,1)
â”‚   â”‚   â”œâ”€ Traffic Light Approach: Green/Yellow/Red zones based on confidence bands
â”‚   â”‚   â”‚   â””â”€ Green: |realized - predicted| < 1.96Ïƒ (95% CI), Yellow: 1.96-2.58Ïƒ, Red: >2.58Ïƒ
â”‚   â”‚   â”œâ”€ Chi-Square Test: Î£(Observed - Expected)Â²/Expected ~ Ï‡Â²
â”‚   â”‚   â”œâ”€ Normal Test (large samples): z = (DR_realized - PD_predicted)/SE
â”‚   â”‚   â””â”€ Hosmer-Lemeshow: Group by predicted probability, test goodness-of-fit
â”‚   â”œâ”€ Stability Testing:
â”‚   â”‚   â”œâ”€ Out-of-Time Validation: Test on recent data not in development sample
â”‚   â”‚   â”œâ”€ Out-of-Sample: Holdout set (e.g., 30%) for validation
â”‚   â”‚   â”œâ”€ Population Stability Index (PSI):
â”‚   â”‚   â”‚   PSI = Î£(% actual - % expected) Ã— ln(% actual / % expected)
â”‚   â”‚   â”‚   â””â”€ Thresholds: PSI < 0.10 stable, 0.10-0.25 moderate shift, >0.25 significant shift
â”‚   â”‚   â”œâ”€ Coefficient Stability: Track parameter estimates over rolling windows
â”‚   â”‚   â””â”€ Rating Migration: Analyze upgrade/downgrade patterns (excessive migrations flag instability)
â”‚   â”œâ”€ LGD Validation:
â”‚   â”‚   â”œâ”€ Mean Absolute Error: MAE = (1/n)Î£|LGD_predicted - LGD_realized|
â”‚   â”‚   â”œâ”€ RÂ²: Goodness of fit for LGD regression models
â”‚   â”‚   â”œâ”€ Downturn Testing: Compare boom vs recession LGD estimates
â”‚   â”‚   â””â”€ Recovery Rate Analysis: Time-to-recovery, cure rates
â”‚   â””â”€ EAD Validation:
â”‚       â”œâ”€ CCF Comparison: Predicted vs realized credit conversion factors
â”‚       â”œâ”€ Utilization Rate: Drawdown behavior during stress
â”‚       â””â”€ Correlation with Default: Test independence assumption (violated â†’ higher CCF needed)
â”œâ”€ Benchmarking:
â”‚   â”œâ”€ External Rating Agencies: Compare bank PD to Moody's/S&P default rates by rating
â”‚   â”œâ”€ Peer Comparison: Regulatory benchmarking exercises (EBA, Fed stress tests)
â”‚   â”œâ”€ Alternative Models: Compare internal model to vendor models (Moody's RiskCalc, etc.)
â”‚   â””â”€ Historical Averages: Long-run default rates by industry/geography
â”œâ”€ Qualitative Validation:
â”‚   â”œâ”€ Model Documentation Review:
â”‚   â”‚   â”œâ”€ Development rationale, data sources, variable selection
â”‚   â”‚   â”œâ”€ Assumptions, limitations, known weaknesses
â”‚   â”‚   â””â”€ Governance: Model owner, approval dates, change log
â”‚   â”œâ”€ Use Test: Evidence model drives business decisions
â”‚   â”‚   â”œâ”€ Pricing: Risk-based loan pricing linked to PD estimates
â”‚   â”‚   â”œâ”€ Limit Setting: Credit limits calibrated to model outputs
â”‚   â”‚   â””â”€ Reporting: Senior management uses model in ICAAP, stress testing
â”‚   â”œâ”€ Data Quality Assessment:
â”‚   â”‚   â”œâ”€ Completeness: Missing values, data gaps
â”‚   â”‚   â”œâ”€ Accuracy: Data entry errors, reconciliation to source systems
â”‚   â”‚   â”œâ”€ Representativeness: Sample covers all material segments
â”‚   â”‚   â””â”€ Definition Consistency: Default, recovery definitions aligned with Basel
â”‚   â””â”€ Expert Judgment Validation:
â”‚       â”œâ”€ Face Validity: Do coefficients have expected signs, magnitudes?
â”‚       â”œâ”€ Economic Intuition: Does model respond logically to stress scenarios?
â”‚       â””â”€ Industry Knowledge: Are results consistent with domain expertise?
â”œâ”€ Regulatory Validation Requirements:
â”‚   â”œâ”€ Basel II/III Standards:
â”‚   â”‚   â”œâ”€ Minimum 5-year data requirement (7 for low-default)
â”‚   â”‚   â”œâ”€ PD models: Grade-level backtesting mandatory
â”‚   â”‚   â”œâ”€ LGD/EAD: Stress period coverage (downturn conditions)
â”‚   â”‚   â””â”€ Annual validation cycle minimum
â”‚   â”œâ”€ SR 11-7 (US Federal Reserve):
â”‚   â”‚   â”œâ”€ Independent validation function (separate from development)
â”‚   â”‚   â”œâ”€ Effective challenge: Rigorous testing, alternative approaches
â”‚   â”‚   â”œâ”€ Model risk management: Documentation, governance, ongoing monitoring
â”‚   â”‚   â””â”€ Validation findings: Severity ratings, remediation tracking
â”‚   â”œâ”€ EBA Guidelines (EU):
â”‚   â”‚   â”œâ”€ Margin of Conservatism (MoC): Additional buffer for model uncertainty
â”‚   â”‚   â”œâ”€ Benchmarking: Compare to supervisory/peer estimates
â”‚   â”‚   â”œâ”€ Downturn LGD: Dedicated validation of stressed scenarios
â”‚   â”‚   â””â”€ Use test evidence: Board reporting, business integration
â”‚   â””â”€ Supervisory Review Process (SRP):
â”‚       â”œâ”€ On-site inspections: Validator interviews, documentation review
â”‚       â”œâ”€ Model approval: Central bank must approve IRB models before use
â”‚       â”œâ”€ Ongoing monitoring: Supervisory letters, model change pre-approval
â”‚       â””â”€ Findings remediation: Timelines for addressing validation gaps
â””â”€ Outcome-Based Actions:
    â”œâ”€ Model Approved: Deploy for regulatory capital calculation
    â”œâ”€ Conditional Approval: Approve with remediation items, timeline <12 months
    â”œâ”€ Model Rejection: Revert to standardized approach, redevelopment required
    â”œâ”€ Add-Ons: Supervisory capital surcharge for unresolved validation issues
    â””â”€ Recalibration: Update parameters while maintaining model structure
```

**Interaction:** Backtest reveals calibration failure â†’ Investigate causes â†’ Recalibrate or redevelop â†’ Revalidate â†’ Deploy

## 5. Challenge Round
Why might a model pass development validation but fail in production?
- **Data drift:** Population characteristics shift post-deployment (new customer segments, economic regime change) â†’ PSI increases, AUC degrades
- **Stress period testing gap:** Model validated only in stable conditions â†’ downturn LGD underestimated by 30-50%
- **Use test failure:** Model outputs ignored by business (overrides, manual adjustments) â†’ defeats purpose of internal model
- **Low-default portfolios:** Insufficient defaults in validation period â†’ statistical tests lack power â†’ Type II error (fail to detect miscalibration)
- **Gaming/optimization:** Model developers optimize on validation set â†’ overfitting â†’ poor generalization to new data

Regulatory response: Ongoing monitoring requirements, annual validation cycles, supervisory stress testing (CCAR/EBA), capital add-ons for model uncertainty.

## 6. Key References
- [Basel Committee - Validation of Rating Systems (2005)](https://www.bis.org/publ/bcbs_wp14.htm) - Comprehensive validation standards; backtesting methodology; requirements for PD, LGD, EAD models; low-default portfolio treatments; regulatory expectations.

- [Federal Reserve SR 11-7 - Model Risk Management Guidance](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm) - Detailed US validation framework; independent validation function requirements; effective challenge expectations; governance structure; remediation protocols; severity ratings for findings.

- [EBA Guidelines on PD/LGD Estimation (2017)](https://www.eba.europa.eu/regulation-and-policy/model-validation) - EU regulatory validation requirements; margin of conservatism (MoC) concept; benchmarking expectations; downturn LGD validation; use test evidence standards.

- Tasche, D. (2003). "A Traffic Lights Approach to PD Validation." arXiv preprint cond-mat/0305038. Foundational paper on binomial testing methodology using traffic light zones; derives confidence bands for default rate outliers; practical backtesting framework widely adopted in Basel III validation.

- Jorion, P., Shi, Y., & Zhang, S. (2009). "Tightening Credit Standards: Loan-Level Evidence from the Discount Window." Journal of Finance, 64(1), 163-189. Empirical validation study; documents how credit standards change over cycle; validation methodologies for detecting model degradation in real-time.

- Abdymomunov, A., & Gerstenberger, J. (2013). "Backtesting and Stress Testing Credit Ratings under Asymmetric Loss." OCC Economics Working Paper 2013-2. Advanced validation techniques; asymmetric loss functions reflecting regulatory objectives; stress correction factors; validation under extreme scenarios.

---
**Status:** Critical regulatory requirement | **Complements:** Internal Ratings-Based (IRB), PD/LGD/EAD Estimation, Stress Testing
