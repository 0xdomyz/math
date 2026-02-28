# Scorecard Models

## 1. Concept Skeleton

**Definition:** A scorecard model is a logistic regression-based credit scoring system that transforms borrower characteristics into a probability of default (PD) or credit score, using interpretable statistical weights assigned to each applicant feature. The model converts raw borrower data into a simple, operationally deployable decision tool.

**Purpose:** Scorecard models serve critical functions in modern credit operations across multiple dimensions:
- **Credit Decision Automation**: Enable real-time loan approvals/rejections without manual underwriting review, reducing processing time from days to minutes while maintaining consistent criteria across thousands of applicants
- **Risk Quantification**: Translate creditworthiness into actionable probabilities and scores, enabling portfolio-level risk aggregation, pricing decisions, and capital allocation across lending products
- **Regulatory Compliance**: Satisfy Basel III IRB framework requirements for internal ratings-based capital calculations; provide auditable, explainable decision logic required by regulators (CECL, Fair Lending, Dodd-Frank)
- **Bias Reduction**: Minimize subjective human judgment through systematic, model-driven criteria; statistical weighting reduces but does not eliminate potential for disparate impact

**Prerequisites:** 
- **Statistical Foundation**: Logistic regression fundamentals, odds ratios, log-odds transformation ($logit(p) = \log(p/(1-p))$), maximum likelihood estimation, coefficient interpretation; understanding when logistic assumptions (linearity in log-odds, independence of observations) hold
- **Credit Domain Knowledge**: Default definition clarity (30/60/90+ DPD conventions), economic cycle behavior, feature/outcome relationship directionality (e.g., income negatively correlated with default), regulatory definitions of credit risk, Basel III/IFRS 9 requirements
- **Data Engineering**: Feature engineering from raw transaction data, binning/discretization of continuous variables, weight-of-evidence calculation, information value assessment, missing data imputation strategies, sample bias detection
- **Validation & Calibration**: Out-of-sample backtesting, population stability index, calibration plots, AUC-ROC interpretation, cutoff optimization for business objectives (approval rate vs. bad rate tradeoff)
- **Related Topics to Cross-Reference**: [Probability of Default (PD) Estimation](#), [Model Validation & Backtesting](#), [Credit Risk Data Preparation](#), [Regulatory Capital Frameworks (Basel III)](#)

## 2. Comparative Framing

| Dimension | Scorecard (Logistic) | Behavioral (XGBoost/RF) | Structural (Merton) | Reduced-Form (CDS) |
|-----------|---|---|---|---|
| **Data Source** | Loan application + historical defaults | Extensive transaction history | Firm financials, stock prices | Market CDS spreads |
| **Interpretability** | High (weights visible, odds ratios) | Low (feature importance estimates only) | Medium (asset-based logic) | Very Low (market-implied, black box) |
| **Implementation Speed** | Real-time (linear algebra) | Real-time (tree traversal) | Daily (calibration intensive) | Daily (market data dependent) |
| **Regulatory Acceptance** | High (Basel III IRB), transparent | Growing (governance required, explainability burden) | Medium (structural assumptions debated) | Low (research-only, confidence intervals wide) |
| **Calibration Complexity** | Simple (binomial optimization) | Complex (hyperparameter tuning, overfitting risk) | High (equity vol → PD conversion requires assumptions) | Implicit (market-driven daily updates) |
| **Sample Size Efficiency** | Good (works with 1000+ defaults) | Requires 5000+ defaults for robust trees | Works with public firms (sparse data) | Limited to traded entities |
| **Economic Cycle Sensitivity** | High (needs regular recalibration post-crisis) | High (trained on historical distribution) | Medium (Merton-type models assume stable leverage) | Real-time (reacts immediately to market sentiment) |

**Comparative Context**: Scorecards dominate retail/SME lending (credit cards, auto loans) because they balance interpretability, regulatory acceptance, and deployment speed. Tree-based models (XGBoost, random forests) capture nonlinearities and interaction effects that logistic regression misses, but create "black box" concerns in regulated industries where regulators demand explainability. Structural models (Merton) excel for large corporates but require balance sheet data that changes infrequently. Market-based models (CDS) reflect real-time credit views but only exist for publicly traded firms; they're best used as validation benchmarks rather than primary decision tools.

## 3. Examples + Counterexamples

### Simple Example: Loan Scorecard Decision
Consider a retail loan scorecard with four binned features (each coded 0-4, higher = more risk):
- Age group: 25-35 (low risk) → bin=0; 45-55 (higher risk) → bin=2
- Income quintile: Top quintile (low risk) → bin=0; Bottom quintile (high risk) → bin=4
- Employment tenure: 10+ years → bin=0; <1 year → bin=4
- Credit inquiries last 6 months: 0-1 (bin=0) vs. 5+ (bin=4)

Fitted logistic regression coefficients (log-odds):
- Intercept: β₀ = -2.1
- Age bin: β₁ = 0.15
- Income bin: β₂ = 0.35
- Tenure bin: β₃ = -0.20
- Inquiries bin: β₄ = 0.45

**Applicant A**: Age bin=0, Income bin=0, Tenure bin=0, Inquiries bin=0
- Linear predictor: $z_A = -2.1 + 0.15(0) + 0.35(0) - 0.20(0) + 0.45(0) = -2.1$
- PD: $PD_A = 1/(1+e^{2.1}) = 0.109 = 10.9\%$
- Decision at 15% threshold: **APPROVE**

**Applicant B**: Age bin=2, Income bin=4, Tenure bin=4, Inquiries bin=4
- Linear predictor: $z_B = -2.1 + 0.15(2) + 0.35(4) - 0.20(4) + 0.45(4) = -2.1 + 0.3 + 1.4 - 0.8 + 1.8 = 0.6$
- PD: $PD_B = 1/(1+e^{-0.6}) = 0.646 = 64.6\%$
- Decision at 15% threshold: **REJECT** (PD too high)

#### Realistic Failure Case: Model Drift Across Economic Regimes
A scorecard built on 2018-2019 retail lending data (unemployment ~4%, consumer confidence high) showed AUC=0.82 in validation. It was deployed in March 2020 as COVID-19 hit:
- **Training period default rate**: 2.3% (prevalence)
- **Deployment period (Q2 2020) default rate**: 8.1% (prevalence)
- **Assumption violated**: Model assumes stable relationship between applicant features and default outcomes
- **Problem manifestation**: 
  - Applicant income (self-employed, service workers) became volatile; historical bins no longer predictive of true risk
  - Employment tenure lost predictive power (mass furloughs, not captured in application data)
  - Behavioral variables (credit utilization) improved artificially due to government stimulus, masking growing default risk downstream
  - Calibration badly miscalibrated: Model predicting 5% PD for cohorts that defaulted at 12%+
- **Mitigation**: Model performance monitoring detected 30% AUC degradation within 2 weeks; emergency recalibration using recent defaults restored predictive power

#### Edge Case: Thin File / Limited Credit History
A 22-year-old applicant with no credit history, recent job start, applies for first credit card:
- **Problem**: Scorecard requires 7 features; applicant data sparse:
  - Age: 22 → quantile bin = calculated normally
  - Employment tenure: 3 months → raw value available
  - Credit inquiries: 0 → zero, but scoring algorithm expects distribution context
  - Payment history: **NULL** (no data) → imputation required
  - Debt-to-income ratio: Limited data, rough estimate possible
- **Solutions in practice**:
  - **Conservative imputation**: Assign worst-case bin for credit inquiries (assume high risk due to missing data)
  - **Peer comparison**: Use average for applicants aged 20-25, calculate PD relative to cohort
  - **Hybrid approval**: Use manual underwriting for incomplete profiles, alternative data (bank statements, utility payments, mobile money transactions)
  - **Alternative scorecard**: Deploy separate "thin file" model trained on limited-data applicants with shorter history windows

#### Technical Counterexample: Weight-of-Evidence Misinterpretation
**Common Misconception**: "Weight-of-evidence captures feature importance; higher WOE bin = stronger discriminatory power"

**Reality**: WOE measures distributional separation between goods and bads *within a feature only*; it does NOT account for correlation with other features or multicollinearity.

**Example of failure**:
- Feature 1 (Income): WOE bins range -0.5 to +0.8 (appears strong discrimination)
- Feature 2 (Age): WOE bins range -0.2 to +0.3 (appears weaker)
- However, if Income and Age are highly correlated (r=0.78), logistic regression fitting yields:
  - Income coefficient: β₁ = 0.08 (heavily penalized due to collinearity)
  - Age coefficient: β₂ = 0.35 (age becomes more important in model)
- **Implication**: Selection based purely on WOE without correlation analysis leads to:
  - Redundant variables (Income adds little after Age is in model)
  - Inflated perceived importance of WOE-weak features that are uncorrelated with others (Age)
  - Unstable coefficients that shift with small data changes

**Correct approach**: Use Information Value (IV = Σ WOE × Distribution %), then validate with partial correlation / VIF in full model to detect redundancy.

## 4. Layer Breakdown

**Scorecard Development Lifecycle:**

Scorecard models progress through three interconnected phases, each addressing distinct stakeholder concerns while building toward a production-ready credit decision system.

**PHASE 1: BUSINESS & DATA FOUNDATION**

Establishes regulatory requirements and assembles the data architecture needed for model development. This phase bridges business objectives (pricing, monitoring, capital allocation) with governance mandates (Basel III, IFRS 9, Fair Lending compliance).

```
BUSINESS & DATA FOUNDATION
├─ Regulatory & Business Context
│  ├─ Basel III IRB: PD validation, backtesting, independent governance
│  ├─ IFRS 9 / CECL: Expected Credit Loss calculations, forward-looking provisions
│  ├─ Fair Lending: Prohibited basis monitoring (race, religion, national origin), explainability
│  └─ Strategic Uses: Risk-based pricing, portfolio monitoring, capital allocation (RWA)
├─ Data Sources
│  ├─ Application: Demographics, financials (income, DTI), behavioral (inquiries, delinquencies)
│  ├─ Bureau: Credit reports (TransUnion, Equifax, Experian), public records, alternate data
│  └─ Internal: Loan performance (DPD status), origination cohorts, behavioral drift
└─ Data Quality Controls
   ├─ Completeness: Flag >30% missing rates per feature → triggers imputation strategy
   ├─ Validity: Range checks (age 18-100, ratios 0-1), outlier detection
   └─ Consistency: Cross-field validation (employment tenure ≤ age - 16), recency filters
```

**PHASE 2: FEATURE ENGINEERING & MODEL FITTING**

Transforms raw borrower attributes into predictive signals via binning, WOE transformation, and logistic regression. The core challenge is balancing predictive power against overfitting risk with limited default events.

```
FEATURE ENGINEERING & MODEL FITTING
├─ Feature Transformation Pipeline
│  ├─ Engineering: DTI ratios, utilization = Balance/Limit, aggregations (max delinquency, inquiry counts)
│  ├─ Binning Strategies:
│  │  ├─ Quantile-based (quintiles), optimal (CHAID, ExhaustiveChiMerge)
│  │  ├─ Domain-driven: Age bins [18-25, 26-35, 36-50, 51-65, 65+]
│  │  └─ Monotonic constraints: Bins ordered by default rate, handle rare categories (n<50)
│  └─ WOE Transformation: $WOE_i = \ln\left(\frac{\% \text{ Goods}_i}{\% \text{ Bads}_i}\right)$
│     └─ Information Value: $IV = \sum (\% \text{ Goods}_i - \% \text{ Bads}_i) \times WOE_i$ (threshold: IV > 0.02)
├─ Logistic Regression: $P(Y=1|X) = \frac{1}{1+e^{-(\beta_0 + \sum \beta_i X_i)}}$
│  ├─ Feature Selection: IV/WOE ranking, stepwise elimination (p<0.05), regularization (L1/L2)
│  ├─ Specification Choices:
│  │  ├─ Interaction terms: Include if domain-supported (high debt × low income)
│  │  ├─ Sample weights: Correct rare event bias (<5% default rate)
│  │  └─ Assumptions: Linearity in log-odds, independence, no perfect collinearity (VIF < 5)
│  └─ Interpretation: Odds ratio $OR = e^β$ (β=-0.05 → 5% decrease in odds per unit)
└─ Score Scaling: $\text{Score} = 500 + 75 \times \ln\left(\frac{\text{Odds}}{Odds_{\text{ref}}}\right)$ → consumer-friendly 300-850 range
   └─ Bucketing: Score < 600 (Reject), 600-700 (Review), 700+ (Approve)
```

**PHASE 3: VALIDATION, DEPLOYMENT & MONITORING**

Ensures model generalizes to unseen data and maintains performance in production. Continuous monitoring detects drift and triggers recalibration before decision quality degrades.

```
VALIDATION & PRODUCTION OPERATIONS
├─ Performance Validation
│  ├─ Discrimination Metrics:
│  │  ├─ AUC-ROC: 0.75-0.85 typical (probability random good > random bad)
│  │  ├─ Gini: $2 \times AUC - 1$ (0.50-0.70 typical)
│  │  └─ K-S Statistic: Max CDF separation (0.20-0.40 typical)
│  ├─ Calibration:
│  │  ├─ Curve: Plot predicted PD vs. actual default rate (deciles/quartiles)
│  │  ├─ Hosmer-Lemeshow: χ² test (p>0.05 adequate fit, combine with visual inspection)
│  │  └─ Brier Score: $\frac{1}{n}\sum (P_i - Y_i)^2$ (0.05-0.15 typical for imbalanced data)
│  └─ Stability: K-fold cross-validation (k=5), time-based splits (train 2018-19, validate 2020)
├─ Production Deployment
│  ├─ Architecture: Decision engine (<100ms latency), automated data feeds (application, bureau, internal)
│  ├─ Audit Trail: Log inputs/scores/decisions for FCRA/ECOA compliance, explainability
│  └─ Score Bands: Map PD to business rules (PD > 10% decline, 5-10% conditional, < 5% approve)
└─ Ongoing Monitoring
   ├─ Population Stability Index: $PSI = \sum (\% \text{Deploy}_i - \% \text{Train}_i) \times \ln(\% \text{Deploy}_i / \% \text{Train}_i)$
   │  └─ Thresholds: PSI < 0.10 (stable), 0.10-0.25 (minor drift), > 0.25 (recalibrate)
   ├─ Performance Triggers: AUC drop > 5%, calibration divergence > 2%, correlation weakening
   └─ Recalibration Cadence: Quarterly review, annual retraining (24m rolling data), emergency rebuild (2 weeks)
```

**Key Dependencies & Data Flow:**
- Phase 1 → Phase 2: Data quality (missing rates, validity) determines feature engineering feasibility; regulatory constraints limit feature usage
- Phase 2 → Phase 3: Model specification choices (regularization, sample weighting) affect overfitting risk and calibration stability
- Phase 3 → Phase 1: Monitoring feedback (PSI drift, AUC degradation) informs data requirements and recalibration for next model generation

**Mathematical Foundation Summary**:
- Log-odds form: $\text{logit}(p) = \beta_0 + \sum \beta_i X_i$
- Inverse transformation: $P(Y=1) = \frac{1}{1+e^{-(\beta_0 + \sum \beta_i X_i)}}$
- Odds interpretation: $\frac{P(Y=1)}{P(Y=0)} = e^{\beta_0 + \sum \beta_i X_i}$
- Log-likelihood: $\sum [Y_i \log(p_i) + (1-Y_i)\log(1-p_i)]$ (maximized by MLE)

## 5. Challenge Round

**1. Model Drift & Regime Change Risk**
Scorecards assume stable relationships between borrower characteristics and default outcomes, but economic regimes shift:
- **COVID-19 Example**: 2019 income became unstable for self-employed; credit utilization declined due to stimulus; unemployment jumped from 3.5% to 14.7% in 2 months
- **Financial Crisis (2008)**: Collateral values collapsed; feature dependencies inverted (falling HPI = rising default even for liquid applicants)
- **Regional Recessions**: Scorecard trained nationally fails in localized downturns (e.g., oil price shock hitting Texas in 2014-2016)
- **Mitigation Strategies**:
  - Monitor Population Stability Index monthly; flag if PSI > 0.10
  - Implement independent approval track (e.g., 20% random sample approved below scorecard threshold) to gather OOB performance data
  - Maintain quarterly performance dashboards with stratified default rates by vintages
  - Pre-plan recalibration workflow; retrain within 2 weeks of AUC degradation > 5%

**2. Adverse Selection & Sample Bias in Model Development**
Training data includes only applicants who were previously approved; rejected applicants never appear:
- **Problem**: Scorecard predicts PD for *marginally approvable* borrowers, not for applicants submitted by marketing in deployment
  - Training: n=10,000 approved applicants in 2018, default_rate=2.3%
  - Deployment: n=50,000 applicants from new origination channel (different marketing), same scorecard, observed default_rate=8.1%
- **Root Causes**:
  - Different applicant quality in new channel (e.g., online platform attracts lower-income borrowers)
  - Features used in previous approvals (human judgment) correlated with unobserved credit quality
  - Marketing targeting changed (e.g., "fast approval" messaging attracts subprime self-selected borrowers)
- **Consequences**:
  - Scorecard cutoff of 650 (trained on 2% population default rate) is miscalibrated for 8% population in deployment
  - Capital allocation incorrect; RWA calculations overstated
  - Fair Lending risk if demographic proxy features (e.g., postal code) were used and now apply to different demographic profile
- **Mitigation Strategies**:
  - A/B test new channels with broader scorecard bands before deploying to full population
  - Gather actual performance on rejected applicants (proxy: monitoring re-applicants, portfolio analytics post-crisis)
  - For new channels: Retrain scorecard on representative sample before full-scale deployment
  - Adjust intercept post-hoc if deployment population differs in prevalence from training (Platt scaling)

**3. Fair Lending & Disparate Impact**
Even without explicit demographic variables (age, race, religion, national origin), scorecards can embed discrimination:
- **Proxy Risk**: Features highly correlated with protected classes (e.g., zip code encodes racial composition → FHA redlining analogs)
- **Disparate Impact Example**:
  - Feature: "Years in current residence" → correlated with gentrification (younger minorities in gentrifying areas = low tenure)
  - Feature: "Employment industry" → retail, service-sector jobs overrepresented in minority populations
  - Result: Scorecard rejects minority applicants at 2× rate of White applicants (80% rule fails)
  - Liability: Even if no intent to discriminate, impact violation triggers FCRA/ECOA enforcement actions
- **Regulatory Scrutiny Triggers**:
  - OCC Bulletin 2007-20: Banks must monitor disparate impact periodically (now quarterly in exam guidance)
  - CFPB enforcement: $100M+ settlements for algorithmic discrimination (fair lending is primary focus post-2020)
  - State AGs: California Fair Lending Enforcement Act (2021); New York algorithmic accountability
- **Mitigation Strategies**:
  - Pre-deployment: Conduct disparate impact analysis (stratified default rates by protected characteristics + proxies)
  - Feature exclusion: Remove zip code, address, education level; use only income and credit metrics if possible
  - Neutral alternative: If disparate impact found, require documented business necessity or find less discriminatory alternative
  - Monitoring: Quarterly dashboard of approval rates / default rates by demographic bands; investigate > 20% gaps
  - Transparency: Maintain audit logs of score components; be prepared to explain any decision individual applicants

**4. Data Quality & Missing Values Mishandling**
Scorecard modeling with incomplete data can systematically bias risk estimates:
- **Problem Manifestation**:
  - Feature: "Years employed current job" (from application) → 28% of applicants missing
  - Quick fix: Impute with 0 (assume unemployed or refused to answer)
  - Reality: Missing often means "recent job change" or "self-employed" (different risk, not no employment)
  - Result: Model treats missing differently than actual unemployed (e.g., coefficient unstable; score has hidden variance)
- **Specific Risks**:
  - Missing not at random (MNAR): Missing data correlates with omitted variables (e.g., informal income, undocumented status)
  - Imputation bias: Mean/median of non-missing distorts distribution; example: If high-income applicants refuse to disclose, mean-imputation underestimates risk
  - Validation-deployment mismatch: If missing rates differ 2019 (model dev) vs. 2024 (deployment), coefficients unstable
- **Examples from practice**:
  - Credit bureau data age: If "last update" is >2 years old, historical delinquencies outdated / not reflects current risk
  - Income verification gaps: Some applicants lack recent paystubs (self-employed, gig workers) → imputation required
  - Thin files: Immigrants, recent arrivals → limited credit history; default imputation methods fail
- **Mitigation Strategies**:
  - Create missing-data indicators: Separate binary flag for "income missing", allows model to weight this separately
  - Use appropriate imputation: Multiple imputation, K-nearest neighbors (KNN), or predictive models (not just mean)
  - Monitor missing rates: Flag if missing rate for key feature jumps > 5% month-over-month
  - Data quality SLAs: Require credit bureau > 60 days freshness; escalate if < 40% of fields populated

**5. Overfitting & Unstable Feature Selection**
More features aren't better; spurious signals can inflate reported AUC but fail in production:
- **Mechanism**:
  - Sample size: n=5,000 applicants, 47 candidate features (engineer generates interaction terms, polynomial terms)
  - Without regularization: Logistic regression finds coefficients that perfectly separate in-sample goods from bads
  - Reported training AUC: 0.92; validation AUC: 0.68 (20+ percentage point cliff = massive overfitting)
- **Specific Pitfalls**:
  - Stepwise selection: Each variable selected by univariate tests, ignoring multicollinearity
  - Example: "Phone area code" has IV=0.08 (appears marginal) but is correlated with geography × income; once income included, phone code adds nothing but noise
  - Validation instability: Coefficients for weak features flip sign across train/val splits (β_phone = +0.12 train, -0.08 val)
- **Business Consequences**:
  - Score bounds unrealistic: Scorecard predicts PD=0.3% or PD=85% (too extreme)
  - Deployment shock: Model fails to generalize to new origination quarter; rapid AUC degradation
  - Computational cost: Each feature increases daily inference latency; 50 features vs. 15 features = 3× compute time for minimal AUC gain
  - Regulatory risk: Auditors question why "phone area code" predicts default; cannot defend economic rationale
- **Mitigation Strategies**:
  - Use regularization (Lasso, Ridge): Penalize coefficient magnitude; L1 shrinks weak features exactly to zero
  - Cross-validation: K-fold (k=5) on feature selection loop; only retain features significant in ≥4/5 folds
  - Information Value filtering: Pre-screen features with IV > 0.02; trim IV < 0.02 or highly correlated (VIF > 5)
  - Parsimony: Enforce max feature count (e.g., 15 features maximum); prefer domain-supported features over marginal statistical gains
  - Stability testing: Random split (50% train, 50% val) repeated 100 times; track coefficient ranges; flag if β range > 200% of median β

## 6. Key References

1. **Naeem Siddiqi. "Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring" (2006, Wiley). ISBN 978-0471754039.**
   - *Relevance*: Industry-standard textbook on scorecard development; covers WOE methodology, binning best practices, and production implementation. Siddiqi's framework is the de facto standard used in most banks globally.

2. **Basel Committee on Banking Supervision. "International Convergence of Capital Measurement and Capital Standards: A Revised Framework (Basel III)" (2010).**
   - *URL*: https://www.bis.org/bcbs/publ/d417.htm
   - *Relevance*: Authoritative regulatory framework defining PD estimation requirements for Internal Ratings-Based (IRB) approaches. Mandates independent validation, governance, and backtesting protocols for credit risk models used in capital calculations.

3. **Federal Reserve & OCC. "Guidance on Sound Practices for Model Risk Management" (SR Letter 11-7, 2011).**
   - *URL*: https://www.federalreserve.gov/bankinforeg/srletters/sr1107.htm
   - *Relevance*: Core risk governance guidance; requires independent model validation, documentation, monitoring, and escalation for model performance degradation. Defines expectations for production scorecard oversight.

4. **Financial Conduct Authority (FCA). "Algorithmic Decision-making & AI as a Financial Stability, Market Integrity & Consumer Protection Issue" (2022).**
   - *URL*: https://www.fca.org.uk/publication/consultation/cp22-5.html
   - *Relevance*: Modern regulatory perspective on algorithmic fairness, explainability, and auditability. Extends fair lending concerns to broader ESG/algorithmic bias frameworks; influences scorecard design for European and global banks.

5. **Board of Governors of the Federal Reserve System. "Interagency Guidance on Credit Risk Retention" (2014, amended 2020).**
   - *URL*: https://www.federalreserve.gov/newsevents/pressreleases/files/bcreg20141218a.pdf
   - *Relevance*: Establishes credit risk retention requirements and data quality standards for securitization; emphasizes accuracy of PD/LGD models underlying loan portfolios. Directly impacts scorecard validation rigor for loan-level data.

6. **Mays, Elizabeth (Editor). "Credit Risk Modelling: Theory and Applications" (2011, Oxford University Press). ISBN 978-0195372670.**
   - *Relevance*: Comprehensive academic treatment of credit risk models including logistic regression scorecards, structural models, and default intensities. Provides theoretical foundations and empirical calibration techniques.

7. **European Banking Authority (EBA). "Guidelines on PD Estimation, LGD Estimation and Discount Rate Estimation" (EBA/GL/2017/16, 2017).**
   - *URL*: https://www.eba.europa.eu/regulation-and-policy/model-validation
   - *Relevance*: IFRS 9 / ECB-aligned guidance on model validation, backtesting protocols, and recalibration triggers. Defines expected credit loss (ECL) requirements; scorecard PD outputs feed directly into ECL calculations.

8. **FAIR (Fair, Accurate, Informative, and Transparent) Machine Learning Initiative. "Fair Machine Learning Handbook" (2019-present).**
   - *URL*: https://fairmlbook.org/
   - *Relevance*: Contemporary resource addressing algorithmic fairness, disparate impact testing, and bias mitigation. While broader than credit scorecards, provides rigorous methods for fair lending compliance and monitoring.

---
**Status:** Foundational credit modelling approach | **Complements:** PD estimation, validation, deployment
