# Default Probability (PD)

## 1. Concept Skeleton
**Definition:** Likelihood that a borrower fails to meet payment obligations within specified time horizon (typically one year)  
**Purpose:** Quantify borrower creditworthiness, price credit risk, estimate expected losses, determine capital requirements  
**Prerequisites:** Probability theory, logistic regression, time-to-event analysis, credit scoring

## 2. Comparative Framing
| Approach | Credit Scoring | Rating Agency | CDS Spreads | Structural Models |
|----------|---------------|--------------|------------|------------------|
| **Data** | Borrower financials, behavior | Qualitative + quantitative | Market prices | Firm value dynamics |
| **Horizon** | Typically 1 year | Through-the-cycle | Implied short-term | Depends on model |
| **Update Frequency** | Annual/quarterly | Periodic | Continuous | Model-dependent |
| **Calibration** | Historical defaults | Long history | Market-implied | Firm-specific |

## 3. Examples + Counterexamples

**Simple Example:**  
Credit score 750: Historical default rate 0.5% → PD₁yr = 0.005. Score 600: Default rate 3% → PD₁yr = 0.03

**Failure Case:**  
Using fixed PD during crisis. 2008: Investment-grade PD jumped 10x; static models massively underestimated losses

**Edge Case:**  
New borrower with no default history; cannot use empirical default rates. Use synthetic scoring or peer comparison

## 4. Layer Breakdown
```
Probability of Default Framework:
├─ PD Definition:
│   ├─ One-year PD: P(default within 12 months)
│   ├─ Multi-year PD: Cumulative over T years
│   ├─ Conditional PD: P(default in year t | survived to t)
│   └─ Lifetime PD: P(default at any point in contract
├─ Calibration Methods:
│   ├─ Empirical: Default rate from historical cohorts
│   ├─ Regression-based: Logistic/probit model P(default) = f(covariates)
│   ├─ Transition matrices: Rating migration to default
│   ├─ CDS-implied: Backed out from market spreads
│   └─ Structural: From Merton model (asset value dynamics)
├─ PD Levels by Credit Quality:
│   ├─ AAA: 0.01% - 0.05% (excellent)
│   ├─ A: 0.05% - 0.20% (good)
│   ├─ BBB: 0.20% - 1.00% (investment grade)
│   ├─ BB: 1.0% - 3.0% (speculative)
│   ├─ B: 3.0% - 8.0% (high risk)
│   └─ D: 100% (default)
├─ Point-in-Time vs Through-the-Cycle:
│   ├─ PIT-PD: Reflects current economic conditions
│   ├─ TTC-PD: Long-run average, smoothed across cycles
│   └─ Conversion: Adjust for cycle position
└─ Term Structure:
    ├─ Survival probability: S(t) = 1 - ∑_{i=1}^t PD_i
    └─ Multi-year PD: 1 - S(t)
```

## 5. Mini-Project
Build and validate a PD model:
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate borrower data
n_borrowers = 5000
data = pd.DataFrame({
    'debt_to_income': np.random.normal(0.4, 0.2, n_borrowers),
    'credit_score': np.random.normal(700, 100, n_borrowers),
    'employment_years': np.random.exponential(8, n_borrowers),
    'age': np.random.normal(45, 15, n_borrowers),
    'loan_amount': np.random.exponential(200000, n_borrowers)
})

# Generate synthetic defaults (more likely with high debt, low score, young age)
logit_score = (-3 + 
               1.5 * (data['debt_to_income'] - 0.4) + 
               -0.01 * (data['credit_score'] - 700) + 
               -0.05 * (data['employment_years'] - 5) +
               0.02 * (data['age'] - 45))

prob_default = 1 / (1 + np.exp(-logit_score))
default = (np.random.rand(n_borrowers) < prob_default).astype(int)
data['default'] = default

# Split into train/test
train_idx = np.random.rand(n_borrowers) < 0.7
X_train = data[train_idx].drop('default', axis=1)
y_train = data[train_idx]['default']
X_test = data[~train_idx].drop('default', axis=1)
y_test = data[~train_idx]['default']

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
pd_pred_train = model.predict_proba(X_train_scaled)[:, 1]
pd_pred_test = model.predict_proba(X_test_scaled)[:, 1]

# Model evaluation
print("=== PD Model Performance ===")
print(f"Training default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")
print(f"Mean predicted PD (train): {pd_pred_train.mean():.2%}")
print(f"Mean predicted PD (test): {pd_pred_test.mean():.2%}")

train_auc = roc_auc_score(y_train, pd_pred_train)
test_auc = roc_auc_score(y_test, pd_pred_test)
print(f"Training AUC: {train_auc:.3f}")
print(f"Test AUC: {test_auc:.3f}")

# Calibration check
print("\n=== PD Calibration ===")
pd_bins = np.linspace(0, 1, 11)
bin_centers = (pd_bins[:-1] + pd_bins[1:]) / 2
bin_actual = []
bin_predicted = []

for i in range(len(pd_bins)-1):
    mask = (pd_pred_test >= pd_bins[i]) & (pd_pred_test < pd_bins[i+1])
    if mask.sum() > 0:
        bin_actual.append(y_test[mask].mean())
        bin_predicted.append(pd_pred_test[mask].mean())
    else:
        bin_actual.append(np.nan)
        bin_predicted.append(np.nan)

print("PD Bin | Predicted | Actual | Calibration")
for j in range(len(pd_bins)-1):
    if not np.isnan(bin_actual[j]):
        print(f"{pd_bins[j]:5.1%}-{pd_bins[j+1]:5.1%} | {bin_predicted[j]:8.2%} | {bin_actual[j]:6.2%} | " + 
              ("✓" if abs(bin_predicted[j] - bin_actual[j]) < 0.02 else "✗"))

# Feature importance
print("\n=== PD Model Coefficients ===")
feature_names = X_train.columns
for feature, coef in zip(feature_names, model.coef_[0]):
    print(f"{feature:20s}: {coef:+.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ROC Curve
ax1 = axes[0, 0]
fpr, tpr, _ = roc_curve(y_test, pd_pred_test)
ax1.plot(fpr, tpr, linewidth=2, label=f'AUC = {test_auc:.3f}')
ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('PD Model ROC Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Calibration curve
ax2 = axes[0, 1]
valid_mask = ~np.isnan(bin_actual)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax2.plot(bin_predicted, bin_actual, 'o-', linewidth=2, markersize=8, 
         label='Observed calibration')
ax2.set_xlabel('Predicted PD')
ax2.set_ylabel('Actual Default Rate')
ax2.set_title('PD Model Calibration\n(Should lie on diagonal)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: PD distribution by default status
ax3 = axes[1, 0]
ax3.hist(pd_pred_test[y_test==0], bins=30, alpha=0.6, label='Non-default', edgecolor='black')
ax3.hist(pd_pred_test[y_test==1], bins=30, alpha=0.6, label='Default', edgecolor='black')
ax3.set_xlabel('Predicted PD')
ax3.set_ylabel('Count')
ax3.set_title('PD Distribution: Discriminatory Power')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Concentration of risk
ax4 = axes[1, 1]
sorted_idx = np.argsort(pd_pred_test)
cumulative_defaults = np.cumsum(y_test.iloc[sorted_idx].values)
cumulative_borrowers = np.arange(len(y_test))
cumulative_pct_defaults = cumulative_defaults / y_test.sum()
cumulative_pct_borrowers = cumulative_borrowers / len(y_test)

ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No discrimination')
ax4.plot(cumulative_pct_borrowers, cumulative_pct_defaults, linewidth=2, label='Model')
ax4.fill_between(cumulative_pct_borrowers, cumulative_pct_defaults, 
                 cumulative_pct_borrowers, alpha=0.2)
ax4.set_xlabel('Cumulative % of Borrowers (sorted by PD)')
ax4.set_ylabel('Cumulative % of Defaults')
ax4.set_title('Gini Coefficient\n(Ability to rank-order risk)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Multi-year PD calculation
print("\n=== Multi-Year PD ===")
annual_pd = 0.01  # 1% annual
years = [1, 2, 3, 5, 10]
for year in years:
    # Assuming independence: survival prob = (1-PD)^year
    survival_prob = (1 - annual_pd) ** year
    cumulative_pd = 1 - survival_prob
    print(f"{year:2d}-year PD: {cumulative_pd:.2%}")
```

## 6. Challenge Round
When is PD estimation problematic?
- **Limited history**: New asset classes, rare events; can't rely on empirical frequencies
- **Structural breaks**: Credit wars, regulatory changes; historical patterns don't hold
- **Default clustering**: During crises, correlated defaults violate independence assumption
- **Rating inflation**: Models underestimate distress (Enron had AA rating before default)
- **Selection bias**: Observed defaults only from approved loans; rejected applicants' actual risk unknown

## 7. Key References
- [Basel III PD Definition](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory standards, through-the-cycle interpretation
- [Logistic Regression for Credit Scoring](https://en.wikipedia.org/wiki/Credit_scoring) - Model fitting, interpretation
- [Structural PD Models](https://en.wikipedia.org/wiki/Merton_model) - Asset-based approach, market-implied PD

---
**Status:** Core credit risk parameter | **Complements:** Credit Risk Definition, LGD, EAD, Expected Loss
