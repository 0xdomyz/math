# Probit Model

## 1. Concept Skeleton
**Definition:** Binary choice model using cumulative standard normal distribution Φ(·) as link function for latent variable framework  
**Purpose:** Estimate P(Y=1|X) when assuming normally distributed errors; theoretical microeconomic foundation via utility maximization  
**Prerequisites:** Normal distribution, CDF properties, maximum likelihood, latent variable models, numerical integration

## 2. Comparative Framing
| Method | Probit | Logit | Linear Probability | Complementary Log-Log |
|--------|--------|-------|--------------------|-----------------------|
| **Link CDF** | Normal Φ(·) | Logistic Λ(·) | None (linear) | Gumbel (asymmetric) |
| **Tail Behavior** | Approaches 0/1 slower | Approaches 0/1 faster | Can exceed bounds | Heavy right tail |
| **Interpretation** | Marginal effects (ϕ) | Odds ratios natural | Direct β | Survival analysis |
| **Computation** | Requires integration | Closed form | OLS | Closed form |

## 3. Examples + Counterexamples

**Classic Example:**  
Labor force participation: Latent utility from working vs not working. Probit assumes unobserved heterogeneity (tastes, opportunities) is normally distributed.

**Failure Case:**  
Extreme value outcomes: Probit thin tails underfit rare events. Logit or t-distribution better for fat tails.

**Edge Case:**  
Identification with fixed effects: Incidental parameters problem in short panels. Probit estimates inconsistent (use conditional logit instead).

## 4. Layer Breakdown
```
Probit Model Framework:
├─ Latent Variable Specification:
│   Y*ᵢ = X'ᵢβ + εᵢ,  εᵢ ~ N(0, 1)
│   Yᵢ = 1[Y*ᵢ > 0]  (observe only binary outcome)
│   └─ Interpretation: Y* is latent utility/propensity
├─ Probability Model:
│   P(Yᵢ=1|Xᵢ) = P(Y*ᵢ > 0) = P(εᵢ > -X'ᵢβ)
│                = Φ(X'ᵢβ)
│   └─ Φ(·): Standard normal CDF, ∫₋∞^z φ(t)dt
├─ Maximum Likelihood:
│   ├─ Likelihood: L(β) = ∏ᵢ Φ(X'ᵢβ)^Yᵢ · [1-Φ(X'ᵢβ)]^(1-Yᵢ)
│   ├─ Log-Likelihood: ℓ(β) = Σᵢ[Yᵢ log Φ(X'ᵢβ) + (1-Yᵢ) log(1-Φ(X'ᵢβ))]
│   ├─ Score: ∂ℓ/∂β = Σᵢ[Yᵢφᵢ/Φᵢ - (1-Yᵢ)φᵢ/(1-Φᵢ)] · Xᵢ
│   │   └─ φᵢ = φ(X'ᵢβ): Standard normal PDF
│   └─ Optimization: Newton-Raphson (no closed form)
├─ Interpretation:
│   ├─ Index Coefficients: β_j = effect on latent Y* (not probabilities!)
│   ├─ Marginal Effect: ∂P/∂X_j = φ(X'β) · β_j  (evaluated at X)
│   │   ├─ ME at Mean (MEM): Evaluate at X̄
│   │   ├─ Average Marginal Effect (AME): Mean[φ(X'ᵢβ̂)·β̂_j]
│   │   └─ Marginal Effect at Representative (MER): Choose typical X₀
│   └─ Standardization: Scale β by √(1 + var(X'β)) for comparability
├─ Inference:
│   ├─ Asymptotic Normality: √n(β̂ - β) →^d N(0, I(β)⁻¹)
│   ├─ Standard Errors: I(β)⁻¹ = [-E(∂²ℓ/∂β∂β')]⁻¹
│   ├─ Wald Test: (Rβ̂ - r)'[RV̂R']⁻¹(Rβ̂ - r) ~ χ²(q)
│   └─ LR Test: 2[ℓ(β̂) - ℓ(β₀)] ~ χ²(k)
├─ Goodness of Fit:
│   ├─ Pseudo-R²: Various (McFadden, Efron, McKelvey-Zavoina)
│   ├─ Percent Correctly Predicted: Classification accuracy
│   ├─ AIC/BIC: Information criteria for model selection
│   └─ Brier Score: Mean squared error of probability predictions
└─ Extensions:
    ├─ Bivariate Probit: Two correlated binary outcomes
    ├─ Ordered Probit: Ordered categorical Y with multiple thresholds
    ├─ Multivariate Probit: J binary outcomes with correlation
    └─ Censored Probit (Tobit-like): Latent Y* observed when Y=1
```

**Interaction:** Normal distribution assumption → Latent variable framework → Probit link → MLE estimation

## 5. Mini-Project
Estimate probit model and compare with logit:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit

np.random.seed(123)
n = 1500

# ===== Data Generating Process: Probit DGP =====
# Labor force participation: Depends on wage offer (unobserved), education, age, children
education = np.random.normal(12, 3, n)  # Years of education
age = np.random.uniform(20, 60, n)
children = np.random.poisson(1.5, n)  # Number of children
non_labor_income = np.random.lognormal(10, 0.5, n)  # Partner income, etc.

# Latent variable: Utility difference from working
# True model: Probit with normal errors
latent_utility = (-2 + 0.3*education + 0.05*age - 0.15*age**2/100 - 
                  0.4*children - 0.0001*non_labor_income + 
                  np.random.normal(0, 1, n))

# Observed binary outcome
works = (latent_utility > 0).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'works': works,
    'education': education,
    'age': age,
    'age_squared': age**2 / 100,  # Scaled for numerical stability
    'children': children,
    'non_labor_income': non_labor_income / 1000  # Rescaled
})

print("="*70)
print("PROBIT MODEL: LABOR FORCE PARTICIPATION")
print("="*70)
print(f"\nSample Size: {n}")
print(f"Participation Rate: {works.mean():.1%}")
print("\nDescriptive Statistics:")
print(df.describe().round(2))

# ===== Probit Estimation =====
X = sm.add_constant(df[['education', 'age', 'age_squared', 
                         'children', 'non_labor_income']])
y = df['works']

probit_model = Probit(y, X).fit()

print("\n" + "="*70)
print("PROBIT MODEL RESULTS")
print("="*70)
print(probit_model.summary())

# ===== Logit Estimation (for comparison) =====
logit_model = Logit(y, X).fit(disp=0)

print("\n" + "="*70)
print("COMPARISON: PROBIT vs LOGIT COEFFICIENTS")
print("="*70)

comparison_df = pd.DataFrame({
    'Probit β': probit_model.params,
    'Logit β': logit_model.params,
    'Logit/1.6': logit_model.params / 1.6  # Approximate scaling
})
print(comparison_df.round(4))
print("\nNote: Logit coefficients ≈ 1.6 × Probit coefficients (rule of thumb)")

# ===== Marginal Effects =====
print("\n" + "="*70)
print("MARGINAL EFFECTS")
print("="*70)

# Get marginal effects using statsmodels
me_probit = probit_model.get_margeff(at='mean')  # MEM
ame_probit = probit_model.get_margeff(at='overall')  # AME

print("\nMarginal Effects at Mean (MEM):")
print(me_probit.summary())

print("\nAverage Marginal Effects (AME):")
print(ame_probit.summary())

# Manual calculation for transparency
X_values = X.values[:, 1:]  # Exclude constant
linear_index = X @ probit_model.params
phi_vals = stats.norm.pdf(linear_index)  # φ(X'β)

# AME for each variable
ame_manual = {}
for i, var in enumerate(X.columns[1:]):  # Skip constant
    me_i = phi_vals * probit_model.params[var]
    ame_manual[var] = me_i.mean()

print("\nManual AME Calculation:")
for var, ame in ame_manual.items():
    print(f"  {var:20s}: {ame:+.6f}")

# ===== Predictions =====
# Predicted probabilities
prob_probit = probit_model.predict(X)
prob_logit = logit_model.predict(X)

# Predicted outcomes (threshold 0.5)
pred_probit = (prob_probit >= 0.5).astype(int)
pred_logit = (prob_logit >= 0.5).astype(int)

# Accuracy
accuracy_probit = (pred_probit == y).mean()
accuracy_logit = (pred_logit == y).mean()

print("\n" + "="*70)
print("PREDICTION ACCURACY")
print("="*70)
print(f"Probit: {accuracy_probit:.1%}")
print(f"Logit:  {accuracy_logit:.1%}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Probit vs Logit Link Functions
z = np.linspace(-4, 4, 200)
probit_link = stats.norm.cdf(z)
logit_link = 1 / (1 + np.exp(-z))

axes[0, 0].plot(z, probit_link, linewidth=2, label='Probit (Normal CDF)')
axes[0, 0].plot(z, logit_link, linewidth=2, linestyle='--', 
                label='Logit (Logistic CDF)')
axes[0, 0].set_xlabel('Linear Index (X\'β)')
axes[0, 0].set_ylabel('P(Y = 1)')
axes[0, 0].set_title('Link Functions: Probit vs Logit')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)

# Plot 2: Difference in Link Functions
axes[0, 1].plot(z, probit_link - logit_link, linewidth=2, color='purple')
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].set_xlabel('Linear Index (X\'β)')
axes[0, 1].set_ylabel('Probit - Logit')
axes[0, 1].set_title('Difference: Probit vs Logit')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].fill_between(z, 0, probit_link - logit_link, 
                        alpha=0.3, color='purple')

# Plot 3: PDF (Marginal Effect Weights)
phi = stats.norm.pdf(z)
logistic_pdf = np.exp(-z) / (1 + np.exp(-z))**2

axes[0, 2].plot(z, phi, linewidth=2, label='Normal φ(z)')
axes[0, 2].plot(z, logistic_pdf, linewidth=2, linestyle='--',
                label='Logistic λ(z)(1-λ(z))')
axes[0, 2].set_xlabel('Linear Index (X\'β)')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_title('PDF: Marginal Effect Weights')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Predicted Probability Comparison
axes[1, 0].scatter(prob_probit, prob_logit, alpha=0.3, s=10)
axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='45° line')
axes[1, 0].set_xlabel('Probit Predicted Probability')
axes[1, 0].set_ylabel('Logit Predicted Probability')
axes[1, 0].set_title('Predicted Probabilities: Probit vs Logit')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(0, 1)

# Plot 5: Marginal Effect of Education (by age)
ages_to_plot = [25, 35, 45, 55]
education_range = np.linspace(8, 20, 50)

for age_val in ages_to_plot:
    me_by_ed = []
    for ed in education_range:
        X_temp = np.array([1, ed, age_val, age_val**2/100, 
                          df['children'].mean(), 
                          df['non_labor_income'].mean()])
        linear_idx = X_temp @ probit_model.params.values
        phi_temp = stats.norm.pdf(linear_idx)
        me = phi_temp * probit_model.params['education']
        me_by_ed.append(me)
    
    axes[1, 1].plot(education_range, me_by_ed, linewidth=2,
                    label=f'Age {age_val}')

axes[1, 1].set_xlabel('Education (Years)')
axes[1, 1].set_ylabel('Marginal Effect')
axes[1, 1].set_title('ME of Education by Age')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot 6: Participation Probability by Age
age_range = np.linspace(20, 60, 100)
prob_by_age_low_ed = []
prob_by_age_high_ed = []

for age_val in age_range:
    # Low education (10 years)
    X_low = np.array([1, 10, age_val, age_val**2/100,
                      df['children'].mean(), df['non_labor_income'].mean()])
    prob_by_age_low_ed.append(stats.norm.cdf(X_low @ probit_model.params.values))
    
    # High education (16 years)
    X_high = np.array([1, 16, age_val, age_val**2/100,
                       df['children'].mean(), df['non_labor_income'].mean()])
    prob_by_age_high_ed.append(stats.norm.cdf(X_high @ probit_model.params.values))

axes[1, 2].plot(age_range, prob_by_age_low_ed, linewidth=2,
                label='10 Years Education', color='red')
axes[1, 2].plot(age_range, prob_by_age_high_ed, linewidth=2,
                label='16 Years Education', color='blue')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('P(Works = 1)')
axes[1, 2].set_title('Participation Probability by Age & Education')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# ===== Hypothesis Tests =====
print("\n" + "="*70)
print("HYPOTHESIS TESTS")
print("="*70)

# Test: Age effect (age + age_squared jointly = 0)
hypothesis = '(age = 0), (age_squared = 0)'
wald_test = probit_model.wald_test(hypothesis)
print(f"\nWald Test: H₀: Age has no effect")
print(f"χ²(2) = {wald_test.statistic:.2f}, p-value = {wald_test.pvalue:.4f}")

# Test: All coefficients = 0 (overall model significance)
lr_test = probit_model.llr
lr_pval = probit_model.llr_pvalue
print(f"\nLikelihood Ratio Test: H₀: All β = 0")
print(f"χ²({len(probit_model.params)-1}) = {lr_test:.2f}, p-value < {lr_pval:.4f}")

# ===== Goodness of Fit =====
print("\n" + "="*70)
print("GOODNESS OF FIT")
print("="*70)

# Pseudo R-squared
print(f"McFadden's Pseudo R²: {probit_model.prsquared:.4f}")

# Count R-squared (fraction correctly predicted)
count_r2 = (pred_probit == y).mean()
print(f"Count R² (% Correct): {count_r2:.1%}")

# AIC and BIC
print(f"AIC: {probit_model.aic:.2f}")
print(f"BIC: {probit_model.bic:.2f}")

# ===== Heterogeneous Treatment Effects =====
print("\n" + "="*70)
print("HETEROGENEOUS MARGINAL EFFECTS")
print("="*70)

# Marginal effect of education varies across individuals
me_education_by_person = phi_vals * probit_model.params['education']

print(f"Education Marginal Effect:")
print(f"  Minimum: {me_education_by_person.min():.6f}")
print(f"  Mean (AME): {me_education_by_person.mean():.6f}")
print(f"  Median: {np.median(me_education_by_person):.6f}")
print(f"  Maximum: {me_education_by_person.max():.6f}")
print(f"  Std Dev: {me_education_by_person.std():.6f}")

print("\nInterpretation:")
print("  Marginal effects are heterogeneous across observations")
print("  Largest effects for those with predicted probability near 0.5")
print("  Smallest effects in tails (very high/low predicted probability)")

# ===== Specification Tests =====
print("\n" + "="*70)
print("SPECIFICATION TESTS")
print("="*70)

# Link test: Regress Y on ŷ and ŷ²
# If model correctly specified, ŷ² should be insignificant
y_hat = probit_model.predict(X)
y_hat_sq = y_hat ** 2

X_link = sm.add_constant(pd.DataFrame({
    'y_hat': y_hat,
    'y_hat_sq': y_hat_sq
}))
link_model = Probit(y, X_link).fit(disp=0)

print("Link Test (Specification):")
print(f"  Coefficient on ŷ²: {link_model.params['y_hat_sq']:.4f}")
print(f"  p-value: {link_model.pvalues['y_hat_sq']:.4f}")
if link_model.pvalues['y_hat_sq'] < 0.05:
    print("  ✗ Reject H₀: Specification may be inadequate")
else:
    print("  ✓ Fail to reject: Specification appears adequate")
```

## 6. Challenge Round
When does probit face challenges?
- **Fixed effects in panels**: Incidental parameters problem with short T → Use conditional logit or correlated random effects
- **Numerical integration**: Multivariate probit (>3 dimensions) computationally expensive → Use simulated MLE (GHK)
- **Identification**: Scale of β not identified (normalize var(ε)=1) → Coefficients not directly comparable across models
- **Interpretation**: No natural metric like odds ratios → Always report marginal effects
- **Tail sensitivity**: Thin tails may underfit extreme events → Consider logit or Student-t link
- **Non-normality**: If ε not normal, estimates inconsistent → Robust to moderate deviations but check residuals

## 7. Key References
- [Greene - Econometric Analysis (Ch 17)](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899)
- [Cameron & Trivedi - Microeconometrics (Ch 14)](https://www.cambridge.org/core/books/microeconometrics/1CA8C6B23A6FD32300AB5D5954CF1B73)
- [Wooldridge - Econometric Analysis of Cross Section and Panel Data (Ch 15)](https://mitpress.mit.edu/9780262232586/)

---
**Status:** Core binary choice model | **Complements:** Logit, Ordered Probit, Bivariate Probit, Heckman Selection
