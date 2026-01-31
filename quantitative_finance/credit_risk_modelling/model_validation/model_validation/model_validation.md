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
PD model backtesting: Predicted 2% default rate, realized 2.3% over 1000 obligors → binomial test p=0.62 (pass, within confidence interval)

**Failure Case:**  
LGD model: Predicted 40% LGD during boom, realized 65% in crisis → no downturn adjustment → fails regulatory validation, requires recalibration

**Edge Case:**  
Low-default portfolio (sovereigns): 3 defaults over 5 years from 200 obligors → insufficient data for binomial test power → requires qualitative validation, stress testing

## 4. Layer Breakdown
```
Model Validation Framework:
├─ Quantitative Validation:
│   ├─ Discriminatory Power (PD Models):
│   │   ├─ ROC Curve: Plot TPR vs FPR, visual assessment
│   │   ├─ AUC (Area Under Curve): 0.5 (random) to 1.0 (perfect)
│   │   │   └─ Thresholds: AUC < 0.60 poor, 0.70-0.80 acceptable, >0.80 strong
│   │   ├─ Gini Coefficient: Gini = 2×AUC - 1, range [0, 1]
│   │   ├─ AR (Accuracy Ratio): AR = Gini_model / Gini_perfect
│   │   └─ KS Statistic: max|CDF_defaults - CDF_non-defaults|
│   ├─ Calibration (PD Models):
│   │   ├─ Binomial Test: For each rating grade, test H₀: realized DR = predicted PD
│   │   │   └─ Test statistic: (n_defaults - n×PD) / √(n×PD×(1-PD)) ~ N(0,1)
│   │   ├─ Traffic Light Approach: Green/Yellow/Red zones based on confidence bands
│   │   │   └─ Green: |realized - predicted| < 1.96σ (95% CI), Yellow: 1.96-2.58σ, Red: >2.58σ
│   │   ├─ Chi-Square Test: Σ(Observed - Expected)²/Expected ~ χ²
│   │   ├─ Normal Test (large samples): z = (DR_realized - PD_predicted)/SE
│   │   └─ Hosmer-Lemeshow: Group by predicted probability, test goodness-of-fit
│   ├─ Stability Testing:
│   │   ├─ Out-of-Time Validation: Test on recent data not in development sample
│   │   ├─ Out-of-Sample: Holdout set (e.g., 30%) for validation
│   │   ├─ Population Stability Index (PSI):
│   │   │   PSI = Σ(% actual - % expected) × ln(% actual / % expected)
│   │   │   └─ Thresholds: PSI < 0.10 stable, 0.10-0.25 moderate shift, >0.25 significant shift
│   │   ├─ Coefficient Stability: Track parameter estimates over rolling windows
│   │   └─ Rating Migration: Analyze upgrade/downgrade patterns (excessive migrations flag instability)
│   ├─ LGD Validation:
│   │   ├─ Mean Absolute Error: MAE = (1/n)Σ|LGD_predicted - LGD_realized|
│   │   ├─ R²: Goodness of fit for LGD regression models
│   │   ├─ Downturn Testing: Compare boom vs recession LGD estimates
│   │   └─ Recovery Rate Analysis: Time-to-recovery, cure rates
│   └─ EAD Validation:
│       ├─ CCF Comparison: Predicted vs realized credit conversion factors
│       ├─ Utilization Rate: Drawdown behavior during stress
│       └─ Correlation with Default: Test independence assumption (violated → higher CCF needed)
├─ Benchmarking:
│   ├─ External Rating Agencies: Compare bank PD to Moody's/S&P default rates by rating
│   ├─ Peer Comparison: Regulatory benchmarking exercises (EBA, Fed stress tests)
│   ├─ Alternative Models: Compare internal model to vendor models (Moody's RiskCalc, etc.)
│   └─ Historical Averages: Long-run default rates by industry/geography
├─ Qualitative Validation:
│   ├─ Model Documentation Review:
│   │   ├─ Development rationale, data sources, variable selection
│   │   ├─ Assumptions, limitations, known weaknesses
│   │   └─ Governance: Model owner, approval dates, change log
│   ├─ Use Test: Evidence model drives business decisions
│   │   ├─ Pricing: Risk-based loan pricing linked to PD estimates
│   │   ├─ Limit Setting: Credit limits calibrated to model outputs
│   │   └─ Reporting: Senior management uses model in ICAAP, stress testing
│   ├─ Data Quality Assessment:
│   │   ├─ Completeness: Missing values, data gaps
│   │   ├─ Accuracy: Data entry errors, reconciliation to source systems
│   │   ├─ Representativeness: Sample covers all material segments
│   │   └─ Definition Consistency: Default, recovery definitions aligned with Basel
│   └─ Expert Judgment Validation:
│       ├─ Face Validity: Do coefficients have expected signs, magnitudes?
│       ├─ Economic Intuition: Does model respond logically to stress scenarios?
│       └─ Industry Knowledge: Are results consistent with domain expertise?
├─ Regulatory Validation Requirements:
│   ├─ Basel II/III Standards:
│   │   ├─ Minimum 5-year data requirement (7 for low-default)
│   │   ├─ PD models: Grade-level backtesting mandatory
│   │   ├─ LGD/EAD: Stress period coverage (downturn conditions)
│   │   └─ Annual validation cycle minimum
│   ├─ SR 11-7 (US Federal Reserve):
│   │   ├─ Independent validation function (separate from development)
│   │   ├─ Effective challenge: Rigorous testing, alternative approaches
│   │   ├─ Model risk management: Documentation, governance, ongoing monitoring
│   │   └─ Validation findings: Severity ratings, remediation tracking
│   ├─ EBA Guidelines (EU):
│   │   ├─ Margin of Conservatism (MoC): Additional buffer for model uncertainty
│   │   ├─ Benchmarking: Compare to supervisory/peer estimates
│   │   ├─ Downturn LGD: Dedicated validation of stressed scenarios
│   │   └─ Use test evidence: Board reporting, business integration
│   └─ Supervisory Review Process (SRP):
│       ├─ On-site inspections: Validator interviews, documentation review
│       ├─ Model approval: Central bank must approve IRB models before use
│       ├─ Ongoing monitoring: Supervisory letters, model change pre-approval
│       └─ Findings remediation: Timelines for addressing validation gaps
└─ Outcome-Based Actions:
    ├─ Model Approved: Deploy for regulatory capital calculation
    ├─ Conditional Approval: Approve with remediation items, timeline <12 months
    ├─ Model Rejection: Revert to standardized approach, redevelopment required
    ├─ Add-Ons: Supervisory capital surcharge for unresolved validation issues
    └─ Recalibration: Update parameters while maintaining model structure
```

**Interaction:** Backtest reveals calibration failure → Investigate causes → Recalibrate or redevelop → Revalidate → Deploy

## 5. Mini-Project
Implement comprehensive PD model validation with backtesting and benchmarking:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# =====================================
# GENERATE CREDIT PORTFOLIO DATA
# =====================================
print("="*70)
print("CREDIT RISK MODEL VALIDATION")
print("="*70)

np.random.seed(42)
n_dev = 3000  # Development sample (historical)
n_val = 1000  # Validation sample (recent)

def generate_credit_data(n, time_period='stable', seed=None):
    """Generate synthetic credit data with defaults."""
    if seed:
        np.random.seed(seed)
    
    # Financial ratios
    leverage = np.random.normal(0, 1, n)
    profitability = np.random.normal(0, 1, n)
    liquidity = np.random.normal(0, 1, n)
    
    # True PD depends on time period (stress vs stable)
    if time_period == 'stable':
        z = -3.0 + 0.8*leverage - 0.6*profitability - 0.5*liquidity
    elif time_period == 'stress':
        # Higher intercept (baseline risk up), stronger sensitivity during crisis
        z = -2.5 + 1.0*leverage - 0.8*profitability - 0.7*liquidity
    
    prob_default = 1 / (1 + np.exp(-z))
    defaults = (np.random.uniform(0, 1, n) < prob_default).astype(int)
    
    return pd.DataFrame({
        'Leverage': leverage,
        'Profitability': profitability,
        'Liquidity': liquidity,
        'Default': defaults,
        'Period': time_period
    })

# Development sample (stable period)
data_dev = generate_credit_data(n_dev, time_period='stable', seed=42)

# Validation sample (recent period, slight stress)
data_val = generate_credit_data(n_val, time_period='stress', seed=123)

print(f"\nDevelopment Sample:")
print(f"   N = {len(data_dev)}, Defaults = {data_dev['Default'].sum()} ({data_dev['Default'].mean():.2%})")

print(f"\nValidation Sample:")
print(f"   N = {len(data_val)}, Defaults = {data_val['Default'].sum()} ({data_val['Default'].mean():.2%})")

# =====================================
# STEP 1: DEVELOP PD MODEL
# =====================================
print("\n" + "="*70)
print("STEP 1: PD MODEL DEVELOPMENT")
print("="*70)

X_dev = data_dev[['Leverage', 'Profitability', 'Liquidity']]
y_dev = data_dev['Default']

# Train logistic regression PD model
pd_model = LogisticRegression(random_state=42, max_iter=1000)
pd_model.fit(X_dev, y_dev)

# Predict PD on development sample
data_dev['PD'] = pd_model.predict_proba(X_dev)[:, 1]

print(f"\nModel Coefficients:")
for feature, coef in zip(X_dev.columns, pd_model.coef_[0]):
    print(f"   {feature}: {coef:.4f}")
print(f"   Intercept: {pd_model.intercept_[0]:.4f}")

# =====================================
# STEP 2: DISCRIMINATORY POWER (IN-SAMPLE)
# =====================================
print("\n" + "="*70)
print("STEP 2: DISCRIMINATORY POWER TESTING")
print("="*70)

# ROC Curve and AUC
fpr_dev, tpr_dev, thresholds_dev = roc_curve(y_dev, data_dev['PD'])
auc_dev = auc(fpr_dev, tpr_dev)
gini_dev = 2 * auc_dev - 1

print(f"\nDevelopment Sample (In-Sample):")
print(f"   AUC: {auc_dev:.4f}")
print(f"   Gini Coefficient: {gini_dev:.4f}")

# KS Statistic
def calculate_ks_statistic(y_true, y_pred):
    """Calculate Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks_stat = np.max(tpr - fpr)
    return ks_stat

ks_dev = calculate_ks_statistic(y_dev, data_dev['PD'])
print(f"   KS Statistic: {ks_dev:.4f}")

# Interpretation
if auc_dev >= 0.80:
    print(f"   → Strong discriminatory power (AUC ≥ 0.80)")
elif auc_dev >= 0.70:
    print(f"   → Acceptable discriminatory power (AUC ≥ 0.70)")
else:
    print(f"   → Weak discriminatory power (AUC < 0.70)")

# =====================================
# STEP 3: OUT-OF-TIME VALIDATION
# =====================================
print("\n" + "="*70)
print("STEP 3: OUT-OF-TIME VALIDATION (STABILITY)")
print("="*70)

X_val = data_val[['Leverage', 'Profitability', 'Liquidity']]
y_val = data_val['Default']

# Predict on validation sample
data_val['PD'] = pd_model.predict_proba(X_val)[:, 1]

# ROC and AUC on validation
fpr_val, tpr_val, thresholds_val = roc_curve(y_val, data_val['PD'])
auc_val = auc(fpr_val, tpr_val)
gini_val = 2 * auc_val - 1
ks_val = calculate_ks_statistic(y_val, data_val['PD'])

print(f"\nValidation Sample (Out-of-Time):")
print(f"   AUC: {auc_val:.4f} (Development: {auc_dev:.4f})")
print(f"   Gini: {gini_val:.4f} (Development: {gini_dev:.4f})")
print(f"   KS: {ks_val:.4f} (Development: {ks_dev:.4f})")

auc_drop = auc_dev - auc_val
if auc_drop < 0.05:
    print(f"   → Stable performance (AUC drop < 0.05)")
elif auc_drop < 0.10:
    print(f"   → Moderate degradation (AUC drop 0.05-0.10)")
else:
    print(f"   → Significant degradation (AUC drop ≥ 0.10) - requires investigation")

# =====================================
# STEP 4: CALIBRATION TESTING (BACKTESTING)
# =====================================
print("\n" + "="*70)
print("STEP 4: CALIBRATION TESTING (BACKTESTING)")
print("="*70)

def binomial_backtest(n_obligors, n_defaults, predicted_pd, confidence_level=0.95):
    """
    Perform binomial test for PD calibration.
    
    H₀: Realized default rate = Predicted PD
    Test statistic: z = (DR - PD) / √(PD(1-PD)/n)
    """
    realized_dr = n_defaults / n_obligors
    se = np.sqrt(predicted_pd * (1 - predicted_pd) / n_obligors)
    z_stat = (realized_dr - predicted_pd) / se if se > 0 else 0
    
    # Two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    # Confidence interval
    z_crit = stats.norm.ppf(1 - (1 - confidence_level)/2)
    ci_lower = predicted_pd - z_crit * se
    ci_upper = predicted_pd + z_crit * se
    
    # Traffic light: Green (pass), Yellow (marginal), Red (fail)
    if np.abs(z_stat) < 1.96:  # 95% CI
        traffic_light = 'Green'
    elif np.abs(z_stat) < 2.58:  # 99% CI
        traffic_light = 'Yellow'
    else:
        traffic_light = 'Red'
    
    return {
        'Realized_DR': realized_dr,
        'Predicted_PD': predicted_pd,
        'Z_Stat': z_stat,
        'P_Value': p_value,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Traffic_Light': traffic_light
    }

# Create PD rating buckets
data_val['PD_Bucket'] = pd.cut(data_val['PD'], 
                                 bins=[0, 0.01, 0.02, 0.05, 0.10, 1.0],
                                 labels=['<1%', '1-2%', '2-5%', '5-10%', '>10%'])

# Backtesting by bucket
backtest_results = []
for bucket in data_val['PD_Bucket'].unique():
    bucket_data = data_val[data_val['PD_Bucket'] == bucket]
    n_obs = len(bucket_data)
    n_def = bucket_data['Default'].sum()
    pred_pd = bucket_data['PD'].mean()
    
    result = binomial_backtest(n_obs, n_def, pred_pd)
    result['Bucket'] = bucket
    result['N'] = n_obs
    result['Defaults'] = n_def
    backtest_results.append(result)

backtest_df = pd.DataFrame(backtest_results)
backtest_df = backtest_df.sort_values('Predicted_PD')

print("\nBacktesting Results by PD Bucket:")
print(backtest_df[['Bucket', 'N', 'Defaults', 'Predicted_PD', 'Realized_DR', 
                   'Z_Stat', 'P_Value', 'Traffic_Light']].to_string(index=False))

# Overall calibration summary
n_green = (backtest_df['Traffic_Light'] == 'Green').sum()
n_yellow = (backtest_df['Traffic_Light'] == 'Yellow').sum()
n_red = (backtest_df['Traffic_Light'] == 'Red').sum()

print(f"\nTraffic Light Summary:")
print(f"   Green (Pass): {n_green}/{len(backtest_df)} buckets")
print(f"   Yellow (Marginal): {n_yellow}/{len(backtest_df)} buckets")
print(f"   Red (Fail): {n_red}/{len(backtest_df)} buckets")

if n_red > 0:
    print(f"   → Model calibration FAILS (Red zones present) - recalibration required")
elif n_yellow > len(backtest_df) / 2:
    print(f"   → Model calibration MARGINAL (many Yellow zones) - monitor closely")
else:
    print(f"   → Model calibration PASSES (mostly Green)")

# =====================================
# STEP 5: POPULATION STABILITY INDEX (PSI)
# =====================================
print("\n" + "="*70)
print("STEP 5: POPULATION STABILITY INDEX (PSI)")
print("="*70)

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index.
    
    PSI = Σ (% actual - % expected) × ln(% actual / % expected)
    """
    # Create distribution for expected and actual
    expected_percents, _ = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=buckets)
    
    # Convert to proportions
    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)
    
    # Avoid log(0) by adding small epsilon
    epsilon = 0.0001
    expected_percents = np.maximum(expected_percents, epsilon)
    actual_percents = np.maximum(actual_percents, epsilon)
    
    # PSI calculation
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi

psi_score = calculate_psi(data_dev['PD'], data_val['PD'], buckets=10)

print(f"\nPopulation Stability Index:")
print(f"   PSI = {psi_score:.4f}")

if psi_score < 0.10:
    print(f"   → Stable population (PSI < 0.10)")
elif psi_score < 0.25:
    print(f"   → Moderate shift (0.10 ≤ PSI < 0.25) - monitor")
else:
    print(f"   → Significant shift (PSI ≥ 0.25) - model may need recalibration")

# =====================================
# STEP 6: BENCHMARKING
# =====================================
print("\n" + "="*70)
print("STEP 6: BENCHMARKING")
print("="*70)

# Compare to external benchmark (simulated)
# Assume external rating agency provides default rates by risk category
external_benchmarks = {
    '<1%': 0.005,
    '1-2%': 0.015,
    '2-5%': 0.035,
    '5-10%': 0.075,
    '>10%': 0.150
}

print("\nBenchmark Comparison (Internal vs External):")
for idx, row in backtest_df.iterrows():
    bucket = row['Bucket']
    internal_dr = row['Realized_DR']
    external_dr = external_benchmarks.get(bucket, np.nan)
    diff = internal_dr - external_dr if not np.isnan(external_dr) else np.nan
    
    print(f"   {bucket}: Internal={internal_dr:.3%}, External={external_dr:.3%}, "
          f"Diff={diff:.3%}")

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ROC Curves (Development vs Validation)
axes[0, 0].plot(fpr_dev, tpr_dev, linewidth=2, label=f'Development (AUC={auc_dev:.3f})')
axes[0, 0].plot(fpr_val, tpr_val, linewidth=2, label=f'Validation (AUC={auc_val:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve: Development vs Validation')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Calibration Plot (Predicted vs Realized)
axes[0, 1].scatter(backtest_df['Predicted_PD']*100, backtest_df['Realized_DR']*100, 
                   s=100, alpha=0.7)
axes[0, 1].plot([0, 15], [0, 15], 'r--', linewidth=2, label='Perfect Calibration')

# Error bars for confidence intervals
for idx, row in backtest_df.iterrows():
    axes[0, 1].errorbar(row['Predicted_PD']*100, row['Realized_DR']*100,
                       yerr=[[row['Realized_DR']*100 - row['CI_Lower']*100],
                             [row['CI_Upper']*100 - row['Realized_DR']*100]],
                       fmt='none', ecolor='gray', alpha=0.5)

axes[0, 1].set_xlabel('Predicted PD (%)')
axes[0, 1].set_ylabel('Realized Default Rate (%)')
axes[0, 1].set_title('Calibration: Predicted vs Realized')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim(0, 15)
axes[0, 1].set_ylim(0, 15)

# Plot 3: Traffic Light Chart
colors = {'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'}
traffic_colors = [colors[tl] for tl in backtest_df['Traffic_Light']]

axes[1, 0].barh(range(len(backtest_df)), backtest_df['Z_Stat'], 
               color=traffic_colors, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(1.96, color='orange', linestyle='--', linewidth=2, label='95% CI')
axes[1, 0].axvline(-1.96, color='orange', linestyle='--', linewidth=2)
axes[1, 0].axvline(2.58, color='red', linestyle='--', linewidth=2, label='99% CI')
axes[1, 0].axvline(-2.58, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_yticks(range(len(backtest_df)))
axes[1, 0].set_yticklabels(backtest_df['Bucket'])
axes[1, 0].set_xlabel('Z-Statistic')
axes[1, 0].set_title('Traffic Light: Calibration Test Results')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='x')

# Plot 4: PD Distribution Comparison (Development vs Validation)
axes[1, 1].hist(data_dev['PD']*100, bins=30, alpha=0.5, label='Development', edgecolor='black')
axes[1, 1].hist(data_val['PD']*100, bins=30, alpha=0.5, label='Validation', edgecolor='black')
axes[1, 1].axvline(data_dev['PD'].mean()*100, color='blue', linestyle='--', 
                   linewidth=2, label=f'Dev Mean={data_dev["PD"].mean():.2%}')
axes[1, 1].axvline(data_val['PD'].mean()*100, color='orange', linestyle='--', 
                   linewidth=2, label=f'Val Mean={data_val["PD"].mean():.2%}')
axes[1, 1].set_xlabel('PD (%)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'PD Distribution (PSI={psi_score:.3f})')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print(f"✓ Discriminatory Power: AUC={auc_val:.3f} (Validation), {auc_dev:.3f} (Dev)")
print(f"✓ Stability: PSI={psi_score:.3f} ({'Stable' if psi_score<0.10 else 'Shifted'})")
print(f"✓ Calibration: {n_green}/{len(backtest_df)} Green, {n_yellow}/{len(backtest_df)} Yellow, {n_red}/{len(backtest_df)} Red")
print(f"✓ Benchmarking: Internal vs external default rates compared")

if n_red > 0 or psi_score >= 0.25 or auc_drop >= 0.10:
    print(f"\n⚠ VALIDATION FINDINGS: Model requires attention")
    if n_red > 0:
        print(f"  - Calibration failures in {n_red} bucket(s) → recalibrate")
    if psi_score >= 0.25:
        print(f"  - Significant population shift (PSI={psi_score:.2f}) → redevelop or adjust")
    if auc_drop >= 0.10:
        print(f"  - AUC degradation ({auc_drop:.2f}) → investigate stability")
else:
    print(f"\n✓ MODEL APPROVED: All validation criteria passed")
```

## 6. Challenge Round
Why might a model pass development validation but fail in production?
- **Data drift:** Population characteristics shift post-deployment (new customer segments, economic regime change) → PSI increases, AUC degrades
- **Stress period testing gap:** Model validated only in stable conditions → downturn LGD underestimated by 30-50%
- **Use test failure:** Model outputs ignored by business (overrides, manual adjustments) → defeats purpose of internal model
- **Low-default portfolios:** Insufficient defaults in validation period → statistical tests lack power → Type II error (fail to detect miscalibration)
- **Gaming/optimization:** Model developers optimize on validation set → overfitting → poor generalization to new data

Regulatory response: Ongoing monitoring requirements, annual validation cycles, supervisory stress testing (CCAR/EBA), capital add-ons for model uncertainty.

## 7. Key References
- [Basel Committee - Principles for the Validation of Rating Systems (2005)](https://www.bis.org/publ/bcbs_wp14.htm) - Validation standards
- [Federal Reserve SR 11-7 - Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm) - US validation framework
- [EBA Guidelines on PD/LGD Estimation (2017)](https://www.eba.europa.eu/regulation-and-policy/model-validation) - EU validation requirements
- [Tasche (2003) "A Traffic Lights Approach to PD Validation"](https://arxiv.org/abs/cond-mat/0305038) - Binomial testing methodology

---
**Status:** Critical regulatory requirement | **Complements:** Internal Ratings-Based (IRB), PD/LGD/EAD Estimation, Stress Testing
