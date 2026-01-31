# Generalized Linear Models (GLM)

## 1.1 Concept Skeleton
**Definition:** Extends linear regression to non-normal outcomes via link function connecting linear predictor to response  
**Purpose:** Model binary, count, or non-Gaussian outcomes while retaining regression framework  
**Prerequisites:** Linear regression, probability distributions, maximum likelihood estimation

## 1.2 Comparative Framing
| Model | Linear Regression | Logistic Regression (GLM) | Poisson Regression (GLM) |
|-------|------------------|--------------------------|------------------------|
| **Outcome Type** | Continuous, normal | Binary (0/1) | Count data (0,1,2,...) |
| **Link Function** | Identity (none) | Logit: log(p/(1-p)) | Log: log(λ) |
| **Distribution** | Normal (Gaussian) | Binomial | Poisson |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Disease occurrence (yes/no) ~ age: Logistic GLM predicts probability P(disease|age) via logit link

**Failure Case:**  
Count data with overdispersion (variance > mean): Poisson GLM underestimates uncertainty; use negative binomial instead

**Edge Case:**  
Perfect separation in logistic: All y=1 at x>5, all y=0 at x<5 → coefficients diverge to infinity

## 1.4 Layer Breakdown
```
GLM Components:
├─ Three Elements:
│   ├─ Random Component: Probability distribution (exponential family)
│   ├─ Systematic Component: Linear predictor η = β₀ + β₁x₁ + ... + βₖxₖ
│   └─ Link Function: g(μ) = η (connects mean to linear predictor)
├─ Common GLM Types:
│   ├─ Logistic Regression:
│   │   ├─ Outcome: Binary (0/1)
│   │   ├─ Link: Logit g(p) = log(p/(1-p))
│   │   └─ Interpretation: e^β = odds ratio
│   ├─ Poisson Regression:
│   │   ├─ Outcome: Count (0, 1, 2,...)
│   │   ├─ Link: Log g(λ) = log(λ)
│   │   └─ Interpretation: e^β = rate ratio
│   └─ Gamma Regression:
│       ├─ Outcome: Positive continuous (skewed)
│       └─ Link: Inverse or log
├─ Estimation:
│   ├─ Maximum Likelihood (not least squares)
│   ├─ Iteratively Reweighted Least Squares (IRLS)
│   └─ Deviance: Goodness-of-fit measure
└─ Assumptions:
    ├─ Linear relationship between link-transformed mean and predictors
    ├─ Independence of observations
    ├─ Correct distribution family
    └─ No perfect multicollinearity
```

## 1.5 Mini-Project
Fit and compare GLMs:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

np.random.seed(42)

# Generate binary outcome data (logistic)
n = 200
x = np.linspace(0, 10, n)
linear_pred = -3 + 0.6 * x
prob = 1 / (1 + np.exp(-linear_pred))  # Inverse logit
y_binary = np.random.binomial(1, prob)

# Generate count data (Poisson)
x_count = np.random.uniform(0, 5, n)
lambda_true = np.exp(0.5 + 0.3 * x_count)
y_count = np.random.poisson(lambda_true)

# Fit Logistic Regression
X_logit = sm.add_constant(x)
logit_model = sm.GLM(y_binary, X_logit, family=sm.families.Binomial())
logit_result = logit_model.fit()

print("Logistic Regression:")
print(logit_result.summary())
print(f"\nOdds Ratio for x: {np.exp(logit_result.params[1]):.3f}")

# Fit Poisson Regression
X_poisson = sm.add_constant(x_count)
poisson_model = sm.GLM(y_count, X_poisson, family=sm.families.Poisson())
poisson_result = poisson_model.fit()

print("\n\nPoisson Regression:")
print(poisson_result.summary())
print(f"\nRate Ratio for x: {np.exp(poisson_result.params[1]):.3f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Logistic: Raw data and fitted curve
axes[0, 0].scatter(x, y_binary, alpha=0.3, label='Observed')
x_pred = np.linspace(x.min(), x.max(), 100)
X_pred_logit = sm.add_constant(x_pred)
prob_pred = logit_result.predict(X_pred_logit)
axes[0, 0].plot(x_pred, prob_pred, 'r-', linewidth=2, label='Fitted P(Y=1)')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y (Binary)')
axes[0, 0].set_title('Logistic Regression')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Logistic: Residuals
residuals_logit = logit_result.resid_deviance
axes[0, 1].scatter(logit_result.fittedvalues, residuals_logit, alpha=0.5)
axes[0, 1].axhline(0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Fitted Values')
axes[0, 1].set_ylabel('Deviance Residuals')
axes[0, 1].set_title('Logistic: Residuals vs Fitted')
axes[0, 1].grid(True, alpha=0.3)

# Poisson: Raw data and fitted curve
axes[1, 0].scatter(x_count, y_count, alpha=0.3, label='Observed')
x_pred_count = np.linspace(x_count.min(), x_count.max(), 100)
X_pred_poisson = sm.add_constant(x_pred_count)
lambda_pred = poisson_result.predict(X_pred_poisson)
axes[1, 0].plot(x_pred_count, lambda_pred, 'r-', linewidth=2, label='Fitted λ')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y (Count)')
axes[1, 0].set_title('Poisson Regression')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Poisson: Residuals
residuals_poisson = poisson_result.resid_deviance
axes[1, 1].scatter(poisson_result.fittedvalues, residuals_poisson, alpha=0.5)
axes[1, 1].axhline(0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Fitted Values')
axes[1, 1].set_ylabel('Deviance Residuals')
axes[1, 1].set_title('Poisson: Residuals vs Fitted')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare predictions
print("\n\nExample Predictions:")
print(f"Logistic at x=5: P(Y=1) = {logit_result.predict(sm.add_constant([5]))[0]:.3f}")
print(f"Poisson at x=2: E[count] = {poisson_result.predict(sm.add_constant([2]))[0]:.3f}")
```

## 1.6 Challenge Round
When is GLM the wrong choice?
- **Non-exponential family distributions**: Use specialized models (e.g., beta regression for proportions)
- **Complex non-linear relationships**: Use GAM (generalized additive models) or machine learning
- **Hierarchical/clustered data**: Use GLMM (generalized linear mixed models)
- **Overdispersion in Poisson**: Use negative binomial or quasi-Poisson
- **Zero-inflated counts**: Use zero-inflated Poisson/negative binomial models

## 1.7 Key References
- [Wikipedia - Generalized Linear Model](https://en.wikipedia.org/wiki/Generalized_linear_model)
- [Statsmodels GLM Documentation](https://www.statsmodels.org/stable/glm.html)
- [McCullagh & Nelder - Generalized Linear Models (book)](https://www.utstat.toronto.edu/~brunner/oldclass/2201s11/readings/glmbook.pdf)
- Thinking: Link function transforms outcome scale to linear; MLE instead of OLS; Deviance replaces R²; Exponential family unifies distributions

---
**Status:** Flexible regression framework | **Complements:** Linear Regression, Maximum Likelihood, Hypothesis Testing
