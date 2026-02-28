# Scorecard Models

## 1. Concept Skeleton
**Definition:** Logistic regression-based credit scoring system converting borrower characteristics into probability of default; assigns weights to applicant features  
**Purpose:** Automate credit decisions, quantify borrower creditworthiness, ensure consistent underwriting, reduce human bias  
**Prerequisites:** Logistic regression, coefficient interpretation, credit data, feature engineering, calibration

## 2. Comparative Framing
| Model Type | Data | Transparency | Calibration | Speed | Regulatory Acceptance |
|------------|------|-------------|-------------|-------|----------------------|
| **Scorecard (Logistic)** | Historical defaults | High (weights visible) | Simple binomial | Real-time | High (Basel III IRB) |
| **Structural (Merton)** | Firm value, leverage | Medium (asset-based) | Equity vol → PD | Weekly | Medium |
| **Reduced-Form** | Market data (CDS) | Low (black box) | Market-implied | Daily | Low (research only) |
| **Machine Learning** | Extensive features | Very low (black box) | Complex optimization | Real-time | Growing acceptance |

## 3. Examples + Counterexamples

**Simple Example:**  
Credit score 720: Model outputs PD=1.2%. Score 650: PD=4.5%. Accept if PD < 3% → Approve first, decline second

**Failure Case:**  
Scorecard built on 2019 data performs poorly in 2024; economic conditions, lending practices changed. Model drift → recalibration needed

**Edge Case:**  
New applicant with limited credit history; few model variables populated. Use default assumptions or add synthetic data from peer comparison

## 4. Layer Breakdown
```
Scorecard Model Framework:
├─ Model Architecture:
│   ├─ Logistic function: PD = 1/(1+e^-z)
│   ├─ Linear predictor: z = β₀ + Σβᵢ×Xᵢ
│   ├─ Feature engineering: Raw → binned variables
│   └─ Coefficient interpretation: β > 0 → risk increases
├─ Key Features by Category:
│   ├─ Demographic: Age, income, employment tenure
│   ├─ Behavioral: Payment history, credit inquiries, delinquencies
│   ├─ Financial: Debt-to-income, savings, liquid assets
│   ├─ Loan characteristics: Amount, tenor, purpose
│   └─ Macroeconomic: Unemployment rate, interest rates
├─ Scorecard Development Steps:
│   ├─ 1. Data collection: Historical defaults + non-defaults
│   ├─ 2. Feature selection: Correlation, information value
│   ├─ 3. Binning: Convert continuous to categorical
│   ├─ 4. Weight-of-evidence: Transform bins → odds ratios
│   ├─ 5. Fit logistic regression: Estimate coefficients
│   ├─ 6. Calibration: Validate predictions vs actuals
│   └─ 7. Deployment: Apply to new applicants
├─ Score Card Interpretation:
│   ├─ Raw score: z = β₀ + Σβᵢ×Xᵢ
│   ├─ Scaled score: 300-850 (consumer friendly)
│   └─ Odds: odds = e^z = odds_at_base × odds_ratio
└─ Validation Metrics:
    ├─ AUC-ROC: Discrimination ability (0.5-1.0)
    ├─ Gini: Concentration of defaults (0-1)
    ├─ K-S statistic: Maximum separation
    └─ Calibration: Predicted vs actual default rate
```

## 5. Challenge Round
When are scorecards problematic?
- **Data quality**: Missing values, errors in historical data; model trained on garbage
- **Model drift**: Economic regime change (pandemic, rate shock); historical patterns don't hold
- **Adverse selection**: Approved applicants differ from rejected; model not predictive for approvals
- **Regulatory scrutiny**: Fair lending concerns (disparate impact); scorecard may embed demographic bias
- **Feature explosion**: Too many variables → overfitting; too few → poor discrimination

## 6. Key References
- [Logistic Regression Credit Scoring](https://en.wikipedia.org/wiki/Credit_scoring) - Classic approach, FICO score methodology
- [Weight-of-Evidence Binning](https://www.investopedia.com/terms/w/weight-of-evidence.asp) - Feature transformation technique
- [Basel III IRB Scorecards](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory framework for internal models

---
**Status:** Foundational credit modelling approach | **Complements:** PD estimation, validation, deployment
