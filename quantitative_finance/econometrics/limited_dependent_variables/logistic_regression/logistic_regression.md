# Logistic Regression

## 1. Concept Skeleton
**Definition:** Binary choice model estimating probability P(Y=1|X) using logistic function with linear index  
**Purpose:** Model binary outcomes when OLS inappropriate; interpret via odds ratios; classify observations  
**Prerequisites:** Maximum likelihood estimation, log-odds transformation, exponential function, asymptotic theory

## 2. Comparative Framing
| Method | Logistic (Logit) | Probit | Linear Probability (OLS) | Multinomial Logit |
|--------|------------------|--------|--------------------------|-------------------|
| **Link Function** | Logistic (log-odds) | Normal CDF (Φ) | Identity (linear) | Multiple categories |
| **Prob Range** | (0, 1) guaranteed | (0, 1) guaranteed | Can exceed [0,1] | Sum to 1 across J |
| **Interpretation** | Odds ratios | Marginal effects | Direct coefficients | Relative risk ratios |
| **Tail Behavior** | Fatter tails | Thinner tails | Unbounded | IIA assumption |

## 3. Examples + Counterexamples

**Classic Example:**  
Credit default: P(default=1) depends on debt-to-income, credit score. Logit coefficient 0.5 → 65% odds increase per unit change.

**Failure Case:**  
Perfect separation: All defaults have income < $30k, no defaults above. MLE doesn't converge (infinite coefficients). Use Firth penalized likelihood.

**Edge Case:**  
Rare events (1% positive): Standard logit underestimates probability. Use rare events correction (King & Zeng 2001) or case-control sampling adjustment.

## 4. Layer Breakdown
```
Logistic Regression Framework:
├─ Model Specification:
│   ├─ Latent Variable: Y* = X'β + ε, ε ~ Logistic
│   ├─ Observed: Y = 1[Y* > 0]
│   └─ Probability: P(Y=1|X) = Λ(X'β) = exp(X'β) / (1 + exp(X'β))
│       └─ Λ(·): Logistic CDF, S-shaped curve
├─ Log-Odds (Logit) Transformation:
│   ├─ Odds: P/(1-P) = exp(X'β)
│   ├─ Log-Odds: log[P/(1-P)] = X'β  (linear in parameters)
│   └─ Interpretation: β_j = change in log-odds per unit X_j
├─ Maximum Likelihood Estimation:
│   ├─ Likelihood: L(β) = ∏ᵢ P(Yᵢ|Xᵢ)^Yᵢ · (1-P(Yᵢ|Xᵢ))^(1-Yᵢ)
│   ├─ Log-Likelihood: ℓ(β) = Σᵢ[Yᵢ log Pᵢ + (1-Yᵢ) log(1-Pᵢ)]
│   │   └─ Maximize via Newton-Raphson, IRLS
│   ├─ Score: ∂ℓ/∂β = X'(Y - P)  (gradient)
│   └─ Hessian: ∂²ℓ/∂β∂β' = -X'WX, W = diag[Pᵢ(1-Pᵢ)]
├─ Interpretation:
│   ├─ Odds Ratio: OR = exp(β_j) for unit change in X_j
│   │   └─ OR > 1: Increases odds, OR < 1: Decreases odds
│   ├─ Marginal Effect: ∂P/∂X_j = β_j · P(1-P)  (not constant!)
│   │   └─ Average Marginal Effect (AME): Mean over sample
│   └─ Predicted Probability: P̂ = Λ(X'β̂)
├─ Goodness of Fit:
│   ├─ Pseudo-R²: McFadden = 1 - ℓ(β̂)/ℓ(β₀), Cox-Snell, Nagelkerke
│   ├─ Likelihood Ratio Test: LR = -2[ℓ(β₀) - ℓ(β̂)] ~ χ²(k)
│   ├─ Classification: Confusion matrix, ROC curve, AUC
│   └─ Hosmer-Lemeshow: Goodness-of-fit test (calibration)
└─ Diagnostics:
    ├─ Separation: Check for perfect/quasi-complete separation
    ├─ Influential Observations: Leverage, Cook's D for GLM
    ├─ Multicollinearity: VIF, condition number
    └─ Specification: Link test, Pregibon's test
```

**Interaction:** Linear index X'β → Logistic transformation → Probabilities in (0,1) → MLE estimation

## 5. Mini-Project
Estimate logistic regression with interpretation and diagnostics:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 1000

# ===== Data Generating Process =====
# Credit default example
# Predictors: debt-to-income ratio, credit score, loan amount
debt_to_income = np.random.uniform(0.1, 0.6, n)
credit_score = np.random.normal(700, 50, n)
loan_amount = np.random.uniform(10000, 100000, n)
age = np.random.uniform(22, 65, n)

# True logistic model
linear_index = (-5 + 8*debt_to_income - 0.01*credit_score + 
                0.00001*loan_amount - 0.02*age)
prob_default = 1 / (1 + np.exp(-linear_index))

# Generate binary outcome
default = (np.random.rand(n) < prob_default).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'default': default,
    'debt_to_income': debt_to_income,
    'credit_score': credit_score,
    'loan_amount': loan_amount,
    'age': age
})

print("="*70)
print("LOGISTIC REGRESSION: CREDIT DEFAULT PREDICTION")
print("="*70)
print(f"\nSample Size: {n}")
print(f"Default Rate: {default.mean():.1%}")
print("\nDescriptive Statistics:")
print(df.describe().round(2))

# ===== Train-Test Split =====
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, 
                                     stratify=df['default'])

print(f"\nTrain Size: {len(train_df)} | Test Size: {len(test_df)}")
print(f"Train Default Rate: {train_df['default'].mean():.1%}")
print(f"Test Default Rate: {test_df['default'].mean():.1%}")

# ===== Logistic Regression Estimation =====
X_train = sm.add_constant(train_df[['debt_to_income', 'credit_score', 
                                      'loan_amount', 'age']])
y_train = train_df['default']

logit_model = Logit(y_train, X_train).fit()

print("\n" + "="*70)
print("LOGISTIC REGRESSION RESULTS")
print("="*70)
print(logit_model.summary())

# ===== Interpretation: Odds Ratios =====
print("\n" + "="*70)
print("ODDS RATIOS (exp(β))")
print("="*70)
odds_ratios = np.exp(logit_model.params)
ci_lower = np.exp(logit_model.conf_int()[0])
ci_upper = np.exp(logit_model.conf_int()[1])

or_df = pd.DataFrame({
    'Odds Ratio': odds_ratios,
    '95% CI Lower': ci_lower,
    '95% CI Upper': ci_upper
})
print(or_df.round(4))

print("\nInterpretation:")
for var in ['debt_to_income', 'credit_score', 'loan_amount', 'age']:
    or_val = odds_ratios[var]
    pct_change = (or_val - 1) * 100
    if or_val > 1:
        print(f"  {var}: {pct_change:+.2f}% increase in odds per unit increase")
    else:
        print(f"  {var}: {-pct_change:.2f}% decrease in odds per unit increase")

# ===== Marginal Effects =====
print("\n" + "="*70)
print("AVERAGE MARGINAL EFFECTS (AME)")
print("="*70)

# Calculate AME manually
X_train_values = X_train.values
linear_pred = X_train_values @ logit_model.params.values
prob_pred = 1 / (1 + np.exp(-linear_pred))

# AME = mean[β * P(1-P)] for each variable
marginal_effects = {}
for i, var in enumerate(X_train.columns[1:], start=1):  # Skip constant
    me = logit_model.params[var] * prob_pred * (1 - prob_pred)
    marginal_effects[var] = me.mean()

me_df = pd.DataFrame({
    'Variable': list(marginal_effects.keys()),
    'AME': list(marginal_effects.values())
})
print(me_df.to_string(index=False))

print("\nInterpretation:")
print("  AME = Average change in P(default=1) for unit increase in X")
for var, ame in marginal_effects.items():
    print(f"  {var}: {ame:+.6f} change in probability")

# ===== Predictions and Classification =====
# Training predictions
X_test = sm.add_constant(test_df[['debt_to_income', 'credit_score', 
                                    'loan_amount', 'age']])
y_test = test_df['default']

prob_test = logit_model.predict(X_test)
pred_test = (prob_test >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, pred_test)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*70)
print("CLASSIFICATION PERFORMANCE (Test Set)")
print("="*70)
print("\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                  0      1")
print(f"Actual  0       {tn:4d}  {fp:4d}   (Specificity: {tn/(tn+fp):.1%})")
print(f"        1       {fn:4d}  {tp:4d}   (Sensitivity: {tp/(tp+fn):.1%})")
print(f"\nAccuracy: {(tp+tn)/(tp+tn+fp+fn):.1%}")
print(f"Precision: {tp/(tp+fp):.1%}")
print(f"Recall (Sensitivity): {tp/(tp+fn):.1%}")
print(f"F1-Score: {2*tp/(2*tp+fp+fn):.3f}")

# Classification Report
print("\n" + classification_report(y_test, pred_test, 
                                   target_names=['No Default', 'Default']))

# ===== ROC Curve and AUC =====
fpr, tpr, thresholds = roc_curve(y_test, prob_test)
auc = roc_auc_score(y_test, prob_test)

print(f"\nArea Under ROC Curve (AUC): {auc:.4f}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Predicted Probability Distribution
axes[0, 0].hist(prob_test[y_test==0], bins=30, alpha=0.6, 
                label='No Default', color='green', density=True)
axes[0, 0].hist(prob_test[y_test==1], bins=30, alpha=0.6, 
                label='Default', color='red', density=True)
axes[0, 0].axvline(0.5, color='black', linestyle='--', linewidth=2,
                   label='Threshold')
axes[0, 0].set_xlabel('Predicted Probability')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Predicted Probability Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: ROC Curve
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
axes[0, 2].set_ylabel('Actual')
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_title('Confusion Matrix (Threshold=0.5)')

# Plot 4: Logistic Curve (Debt-to-Income)
dti_range = np.linspace(0.1, 0.6, 100)
X_dti = np.column_stack([
    np.ones(100),  # constant
    dti_range,
    np.full(100, credit_score.mean()),
    np.full(100, loan_amount.mean()),
    np.full(100, age.mean())
])
prob_dti = logit_model.predict(X_dti)

axes[1, 0].scatter(train_df['debt_to_income'], train_df['default'], 
                   alpha=0.3, s=20, color='gray')
axes[1, 0].plot(dti_range, prob_dti, 'r-', linewidth=3,
                label='Logistic Curve')
axes[1, 0].set_xlabel('Debt-to-Income Ratio')
axes[1, 0].set_ylabel('P(Default = 1)')
axes[1, 0].set_title('Logistic Function: Debt-to-Income')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim(-0.05, 1.05)

# Plot 5: Marginal Effects across Probability Range
prob_range = np.linspace(0.01, 0.99, 100)
me_debt = logit_model.params['debt_to_income'] * prob_range * (1 - prob_range)
me_score = logit_model.params['credit_score'] * prob_range * (1 - prob_range)

axes[1, 1].plot(prob_range, me_debt, linewidth=2, 
                label='Debt-to-Income')
axes[1, 1].plot(prob_range, me_score, linewidth=2, 
                label='Credit Score')
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].set_xlabel('Predicted Probability P(Y=1)')
axes[1, 1].set_ylabel('Marginal Effect')
axes[1, 1].set_title('Marginal Effects (Non-Constant)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Odds Ratios with Confidence Intervals
or_plot = or_df.drop('const').sort_values('Odds Ratio')
y_pos = np.arange(len(or_plot))

axes[1, 2].errorbar(or_plot['Odds Ratio'], y_pos,
                    xerr=[or_plot['Odds Ratio'] - or_plot['95% CI Lower'],
                          or_plot['95% CI Upper'] - or_plot['Odds Ratio']],
                    fmt='o', markersize=8, capsize=5, linewidth=2)
axes[1, 2].axvline(1, color='red', linestyle='--', linewidth=2,
                   label='No Effect (OR=1)')
axes[1, 2].set_yticks(y_pos)
axes[1, 2].set_yticklabels(or_plot.index)
axes[1, 2].set_xlabel('Odds Ratio')
axes[1, 2].set_title('Odds Ratios with 95% CI')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ===== Goodness of Fit Tests =====
print("\n" + "="*70)
print("GOODNESS OF FIT")
print("="*70)

# McFadden's Pseudo R²
null_model = Logit(y_train, sm.add_constant(np.ones(len(y_train)))).fit(disp=0)
mcfadden_r2 = 1 - (logit_model.llf / null_model.llf)
print(f"McFadden's Pseudo R²: {mcfadden_r2:.4f}")

# Likelihood Ratio Test
lr_stat = -2 * (null_model.llf - logit_model.llf)
lr_pval = 1 - stats.chi2.cdf(lr_stat, df=len(logit_model.params)-1)
print(f"Likelihood Ratio Test: χ²({len(logit_model.params)-1}) = {lr_stat:.2f}, p < {lr_pval:.4f}")

# ===== Separation Check =====
print("\n" + "="*70)
print("DIAGNOSTICS: SEPARATION CHECK")
print("="*70)

# Check for complete separation (indicative, not exhaustive)
for var in ['debt_to_income', 'credit_score', 'loan_amount', 'age']:
    default_min = train_df[train_df['default']==1][var].min()
    default_max = train_df[train_df['default']==1][var].max()
    no_default_min = train_df[train_df['default']==0][var].min()
    no_default_max = train_df[train_df['default']==0][var].max()
    
    overlap = not (default_max < no_default_min or no_default_max < default_min)
    print(f"{var:20s}: {'Overlap ✓' if overlap else 'Separation ✗'}")

# ===== Varying Threshold Analysis =====
print("\n" + "="*70)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("="*70)

thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []

for thresh in thresholds_to_test:
    pred_thresh = (prob_test >= thresh).astype(int)
    cm_thresh = confusion_matrix(y_test, pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
    
    accuracy = (tp_t + tn_t) / len(y_test)
    precision = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    recall = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    
    threshold_results.append({
        'Threshold': thresh,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df.to_string(index=False, float_format='%.3f'))
```

## 6. Challenge Round
When does logistic regression fail or need modification?
- **Perfect separation**: All Y=1 have X>c, all Y=0 have X<c → MLE diverges (use Firth penalization)
- **Rare events**: P(Y=1) < 5% → Underestimates probabilities (use bias correction)
- **Imbalanced data**: 99:1 class ratio → Model predicts majority class always (use SMOTE, class weights)
- **Non-linear effects**: Relationship not S-shaped in X'β → Add polynomials, splines, or GAM
- **Unordered categories**: Y ∈ {A, B, C} → Use multinomial logit instead
- **Ordered categories**: Y ∈ {Low, Med, High} → Use ordered logit/probit instead

## 7. Key References
- [Hosmer, Lemeshow & Sturdivant - Applied Logistic Regression (3rd ed)](https://www.wiley.com/en-us/Applied+Logistic+Regression%2C+3rd+Edition-p-9780470582473)
- [Agresti - Categorical Data Analysis (3rd ed)](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635)
- [King & Zeng (2001) - Logistic Regression in Rare Events Data](https://doi.org/10.1093/oxfordjournals.pan.a004868)

---
**Status:** Core binary choice model | **Complements:** Probit, Multinomial Logit, GLM, Classification Trees
