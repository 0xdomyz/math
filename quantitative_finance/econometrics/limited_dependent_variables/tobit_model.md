# Tobit Model (Censored Regression)

## 1. Concept Skeleton
**Definition:** Regression model for censored/truncated continuous dependent variables where observations pile up at threshold(s)  
**Purpose:** Consistent estimation when dependent variable censored at known value(s); distinguish zero from true latent value  
**Prerequisites:** Latent variable models, maximum likelihood, truncated normal distribution, selection bias correction

## 2. Comparative Framing
| Method | Tobit (Censored) | Truncated Regression | Heckman Selection | OLS on Positives |
|--------|------------------|----------------------|-------------------|------------------|
| **Zeros Observed** | Yes (censored) | No (sample selection) | Yes (participation eq) | Discarded |
| **Bias if Misused** | None (consistent) | Inconsistent | None (two-step) | Selection bias |
| **Assumptions** | Normality, homoskedasticity | Normality | Bivariate normal | None relevant |
| **Use Case** | Corner solutions | Self-selection into sample | Labor supply, wages | Ad-hoc, biased |

## 3. Examples + Counterexamples

**Classic Example:**  
Charitable donations: Many give $0 (corner solution), others give Y* > 0. Tobit models latent generosity Y* censored at 0.

**Failure Case:**  
Hours worked: Using Tobit assumes non-workers want negative hours. Wrong! Use Heckman: participation decision + hours conditional on working.

**Edge Case:**  
Double censoring: Y ∈ [L, U] censored at both ends (e.g., test scores 0-100 with floor/ceiling effects). Extend Tobit with two-limit censoring.

## 4. Layer Breakdown
```
Tobit Model Framework:
├─ Latent Variable Specification:
│   Y*ᵢ = X'ᵢβ + εᵢ,  εᵢ ~ N(0, σ²)
│   Yᵢ = max(0, Y*ᵢ)  or  Yᵢ = Y*ᵢ if Y*ᵢ > 0, else 0
│   └─ Y* is latent desired/unobserved outcome
├─ Censoring Types:
│   ├─ Left Censoring: Y = max(L, Y*), typically L=0
│   ├─ Right Censoring: Y = min(U, Y*), e.g., top-coded income
│   └─ Two-Limit: Y = L if Y*≤L, Y* if L<Y*<U, U if Y*≥U
├─ Likelihood Function:
│   ├─ For Yᵢ = 0 (censored): P(Y*ᵢ ≤ 0) = Φ(-X'ᵢβ/σ)
│   ├─ For Yᵢ > 0 (uncensored): f(Yᵢ) = (1/σ)φ((Yᵢ - X'ᵢβ)/σ)
│   └─ Combined: L(β, σ) = ∏[censored] Φ(-X'β/σ) × ∏[uncensored] (1/σ)φ((Y-X'β)/σ)
├─ Log-Likelihood:
│   ℓ(β, σ) = Σ[Yᵢ=0] log Φ(-X'ᵢβ/σ) + 
│              Σ[Yᵢ>0] [log φ((Yᵢ-X'ᵢβ)/σ) - log σ]
│   └─ Maximize via Newton-Raphson (no closed form)
├─ Interpretation:
│   ├─ β: Effect on latent Y* (not observed Y!)
│   ├─ E[Y|X]: Unconditional expectation (includes zeros)
│   │   = Φ(X'β/σ) · (X'β + σλ),  λ = φ/Φ (inverse Mills ratio)
│   ├─ E[Y|Y>0, X]: Conditional on positive (truncated mean)
│   │   = X'β + σλ(X'β/σ)
│   └─ P(Y>0|X): Probability of uncensoring = Φ(X'β/σ)
├─ Marginal Effects:
│   ├─ On E[Y*|X]: ∂E[Y*]/∂X_j = β_j  (latent variable)
│   ├─ On E[Y|X]: ∂E[Y]/∂X_j = β_j · Φ(X'β/σ)  (observed, incl. zeros)
│   ├─ On E[Y|Y>0,X]: ∂E[Y|Y>0]/∂X_j = β_j[1 - λ(X'β/σ)(X'β/σ + λ)]
│   └─ On P(Y>0): ∂P(Y>0)/∂X_j = (β_j/σ) · φ(X'β/σ)  (extensive margin)
├─ McDonald-Moffitt Decomposition:
│   Total ME = Prob(uncensored) × ME(conditional) + E[Y|Y>0] × ME(prob)
│   └─ Separates intensive margin (how much) from extensive (whether)
├─ Extensions:
│   ├─ Heteroskedastic Tobit: σᵢ² = exp(Z'ᵢγ)
│   ├─ Type II Tobit (Heckman): Selection + outcome equations
│   ├─ Type III Tobit: Both Y₁ and Y₂ censored, bivariate
│   └─ Semiparametric: Powell's censored LAD, symmetrically trimmed LS
└─ Diagnostics:
    ├─ Normality: Residuals from uncensored; Q-Q plot, Jarque-Bera
    ├─ Homoskedasticity: Breusch-Pagan on uncensored observations
    ├─ Specification: RESET test, functional form tests
    └─ Corner Solution vs Selection: Compare Tobit to Heckman, Cragg model
```

**Interaction:** Censoring at threshold → Two-part likelihood → MLE → Marginal effects decomposition

## 5. Mini-Project
Estimate Tobit model with decomposition of effects:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm

np.random.seed(42)
n = 1000

# ===== Data Generating Process: Tobit =====
# Charitable donations example
# Covariates: income, age, education, religiosity
income = np.random.lognormal(10.5, 0.5, n)  # Income in thousands
age = np.random.uniform(25, 75, n)
education = np.random.normal(14, 3, n)
religiosity = np.random.uniform(0, 10, n)  # Scale 0-10

# Latent donation desire
sigma_true = 500
latent_donation = (-1000 + 0.02*income + 10*age + 50*education + 
                   100*religiosity + np.random.normal(0, sigma_true, n))

# Observed donation (censored at 0)
donation = np.maximum(0, latent_donation)

# Censoring indicator
censored = (donation == 0).astype(int)
censoring_rate = censored.mean()

# Create DataFrame
df = pd.DataFrame({
    'donation': donation,
    'income': income,
    'age': age,
    'education': education,
    'religiosity': religiosity,
    'censored': censored
})

print("="*70)
print("TOBIT MODEL: CHARITABLE DONATIONS")
print("="*70)
print(f"\nSample Size: {n}")
print(f"Censoring Rate (Y=0): {censoring_rate:.1%}")
print(f"Mean Donation (All): ${donation.mean():.2f}")
print(f"Mean Donation (Y>0): ${donation[donation>0].mean():.2f}")
print("\nDescriptive Statistics:")
print(df.describe().round(2))

# ===== Manual Tobit MLE Implementation =====
def tobit_loglik(params, y, X):
    """
    Tobit log-likelihood function
    params: [β₀, β₁, ..., βₖ, log(σ)]
    """
    n_vars = X.shape[1]
    beta = params[:n_vars]
    log_sigma = params[n_vars]
    sigma = np.exp(log_sigma)
    
    # Linear prediction
    xb = X @ beta
    
    # Censored observations (y=0)
    censored_mask = (y == 0)
    ll_censored = np.sum(stats.norm.logcdf(-xb[censored_mask] / sigma))
    
    # Uncensored observations (y>0)
    uncensored_mask = (y > 0)
    z = (y[uncensored_mask] - xb[uncensored_mask]) / sigma
    ll_uncensored = np.sum(stats.norm.logpdf(z) - log_sigma)
    
    return -(ll_censored + ll_uncensored)  # Negative for minimization

# Prepare data
X = sm.add_constant(df[['income', 'age', 'education', 'religiosity']])
y = df['donation'].values

# Initial values (OLS on positives + log(sd))
ols_pos = sm.OLS(y[y>0], X.loc[y>0]).fit()
init_params = np.append(ols_pos.params, np.log(ols_pos.resid.std()))

# Optimize
print("\n" + "="*70)
print("TOBIT MLE ESTIMATION")
print("="*70)
print("Optimizing... (this may take a moment)")

result = minimize(tobit_loglik, init_params, args=(y, X.values),
                 method='BFGS', options={'disp': False})

tobit_params = result.x
beta_hat = tobit_params[:X.shape[1]]
sigma_hat = np.exp(tobit_params[X.shape[1]])

# Standard errors (inverse Hessian)
from scipy.optimize import approx_fprime

def hessian_approx(params):
    """Approximate Hessian via finite differences"""
    eps = 1e-5
    n_params = len(params)
    hess = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        params_forward = params.copy()
        params_backward = params.copy()
        params_forward[i] += eps
        params_backward[i] -= eps
        
        grad_forward = approx_fprime(params_forward, tobit_loglik, eps, y, X.values)
        grad_backward = approx_fprime(params_backward, tobit_loglik, eps, y, X.values)
        
        hess[i, :] = (grad_forward - grad_backward) / (2 * eps)
    
    return hess

# Compute standard errors
hess = hessian_approx(tobit_params)
var_cov = np.linalg.inv(hess)
se_tobit = np.sqrt(np.diag(var_cov))

# Results table
var_names = list(X.columns) + ['log(sigma)']
results_df = pd.DataFrame({
    'Coefficient': tobit_params,
    'Std Error': se_tobit,
    't-stat': tobit_params / se_tobit,
    'p-value': 2 * (1 - stats.norm.cdf(np.abs(tobit_params / se_tobit)))
}, index=var_names)

print("\nTobit Coefficients:")
print(results_df.round(4))
print(f"\nσ (estimated): {sigma_hat:.2f}")
print(f"σ (true): {sigma_true:.2f}")

# ===== Compare with OLS (biased) =====
print("\n" + "="*70)
print("COMPARISON: TOBIT vs OLS (on all data)")
print("="*70)

ols_all = sm.OLS(y, X).fit()
print("\nOLS Results (All Observations):")
print(ols_all.summary().tables[1])

comparison_df = pd.DataFrame({
    'Tobit β': beta_hat,
    'OLS β (All)': ols_all.params,
    'OLS β (Y>0)': ols_pos.params
}, index=X.columns)
print("\nCoefficient Comparison:")
print(comparison_df.round(4))

# ===== Marginal Effects =====
print("\n" + "="*70)
print("MARGINAL EFFECTS DECOMPOSITION")
print("="*70)

# Evaluate at sample means
X_mean = X.mean().values
xb_mean = X_mean @ beta_hat

# Probability of uncensoring
prob_positive = stats.norm.cdf(xb_mean / sigma_hat)

# Inverse Mills ratio
lambda_imr = stats.norm.pdf(xb_mean / sigma_hat) / prob_positive

# Marginal effects
# 1. On latent Y*
me_latent = beta_hat

# 2. On unconditional E[Y|X] (includes zeros)
me_unconditional = beta_hat * prob_positive

# 3. On conditional E[Y|Y>0, X]
me_conditional = beta_hat * (1 - lambda_imr * (xb_mean/sigma_hat + lambda_imr))

# 4. On probability P(Y>0)
me_probability = (beta_hat / sigma_hat) * stats.norm.pdf(xb_mean / sigma_hat)

me_df = pd.DataFrame({
    'Latent Y*': me_latent,
    'E[Y|X] (uncond.)': me_unconditional,
    'E[Y|Y>0,X] (cond.)': me_conditional,
    'P(Y>0)': me_probability
}, index=X.columns)

print("\nMarginal Effects at Mean:")
print(me_df.round(6))

# McDonald-Moffitt decomposition
print("\n" + "="*70)
print("McDONALD-MOFFITT DECOMPOSITION")
print("="*70)
print("Total ME = Intensive Margin + Extensive Margin")

ey_pos = xb_mean + sigma_hat * lambda_imr  # E[Y|Y>0]

for var in X.columns[1:]:  # Skip constant
    idx = list(X.columns).index(var)
    total_me = me_unconditional[idx]
    intensive = prob_positive * me_conditional[idx]
    extensive = ey_pos * me_probability[idx]
    
    print(f"\n{var}:")
    print(f"  Total ME:      {total_me:+.6f}")
    print(f"  Intensive:     {intensive:+.6f}  ({intensive/total_me*100:.1f}%)")
    print(f"  Extensive:     {extensive:+.6f}  ({extensive/total_me*100:.1f}%)")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Distribution of Observed Y (with censoring)
axes[0, 0].hist(donation, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2,
                   label=f'Censored: {censoring_rate:.1%}')
axes[0, 0].set_xlabel('Donation ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Observed Donations (Censored at 0)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Latent vs Observed
axes[0, 1].scatter(latent_donation, donation, alpha=0.3, s=10)
axes[0, 1].plot([latent_donation.min(), latent_donation.max()],
                [0, 0], 'r-', linewidth=2, label='Censoring Threshold')
axes[0, 1].plot([0, latent_donation.max()],
                [0, latent_donation.max()], 'k--', linewidth=1,
                label='45° line')
axes[0, 1].set_xlabel('Latent Y* (Desired Donation)')
axes[0, 1].set_ylabel('Observed Y (Actual Donation)')
axes[0, 1].set_title('Censoring Mechanism')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Predicted vs Actual (Tobit)
y_pred_tobit = X.values @ beta_hat
y_pred_tobit_obs = prob_positive * (y_pred_tobit + sigma_hat * lambda_imr)

axes[0, 2].scatter(y_pred_tobit, y, alpha=0.3, s=10, label='Data')
axes[0, 2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--',
                linewidth=2, label='45° line')
axes[0, 2].set_xlabel('Tobit Predicted (X\'β)')
axes[0, 2].set_ylabel('Observed Y')
axes[0, 2].set_title('Tobit Fit')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Marginal Effect of Income (across income distribution)
income_range = np.linspace(income.min(), income.max(), 100)
me_income_unconditional = []
me_income_conditional = []

for inc in income_range:
    X_temp = np.array([1, inc, age.mean(), education.mean(), religiosity.mean()])
    xb_temp = X_temp @ beta_hat
    prob_temp = stats.norm.cdf(xb_temp / sigma_hat)
    lambda_temp = stats.norm.pdf(xb_temp / sigma_hat) / prob_temp
    
    me_unc = beta_hat[1] * prob_temp
    me_cond = beta_hat[1] * (1 - lambda_temp * (xb_temp/sigma_hat + lambda_temp))
    
    me_income_unconditional.append(me_unc)
    me_income_conditional.append(me_cond)

axes[1, 0].plot(income_range/1000, me_income_unconditional, linewidth=2,
                label='Unconditional E[Y|X]')
axes[1, 0].plot(income_range/1000, me_income_conditional, linewidth=2,
                label='Conditional E[Y|Y>0,X]')
axes[1, 0].set_xlabel('Income ($1000s)')
axes[1, 0].set_ylabel('Marginal Effect of Income')
axes[1, 0].set_title('Heterogeneous Marginal Effects')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Expected Donation by Religiosity
religiosity_range = np.linspace(0, 10, 100)
ey_unconditional = []
ey_conditional = []

for relig in religiosity_range:
    X_temp = np.array([1, income.mean(), age.mean(), education.mean(), relig])
    xb_temp = X_temp @ beta_hat
    prob_temp = stats.norm.cdf(xb_temp / sigma_hat)
    lambda_temp = stats.norm.pdf(xb_temp / sigma_hat) / prob_temp
    
    ey_unc = prob_temp * (xb_temp + sigma_hat * lambda_temp)
    ey_cond = xb_temp + sigma_hat * lambda_temp
    
    ey_unconditional.append(ey_unc)
    ey_conditional.append(ey_cond)

axes[1, 1].plot(religiosity_range, ey_unconditional, linewidth=2,
                label='E[Y|X] (includes zeros)', color='blue')
axes[1, 1].plot(religiosity_range, ey_conditional, linewidth=2,
                label='E[Y|Y>0,X] (positives only)', color='red')
axes[1, 1].set_xlabel('Religiosity (0-10 scale)')
axes[1, 1].set_ylabel('Expected Donation ($)')
axes[1, 1].set_title('Expected Donations by Religiosity')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Probability of Donating (P(Y>0))
prob_donate_by_income = []
for inc in income_range:
    X_temp = np.array([1, inc, age.mean(), education.mean(), religiosity.mean()])
    prob_donate_by_income.append(stats.norm.cdf((X_temp @ beta_hat) / sigma_hat))

prob_donate_by_relig = []
for relig in religiosity_range:
    X_temp = np.array([1, income.mean(), age.mean(), education.mean(), relig])
    prob_donate_by_relig.append(stats.norm.cdf((X_temp @ beta_hat) / sigma_hat))

ax1 = axes[1, 2]
ax2 = ax1.twinx()

line1 = ax1.plot(income_range/1000, prob_donate_by_income, 'b-',
                 linewidth=2, label='By Income')
ax1.set_xlabel('Income ($1000s) / Religiosity')
ax1.set_ylabel('P(Donate) by Income', color='b')
ax1.tick_params(axis='y', labelcolor='b')

line2 = ax2.plot(religiosity_range, prob_donate_by_relig, 'r-',
                 linewidth=2, label='By Religiosity')
ax2.set_ylabel('P(Donate) by Religiosity', color='r')
ax2.tick_params(axis='y', labelcolor='r')

axes[1, 2].set_title('Probability of Donating (Extensive Margin)')
axes[1, 2].grid(alpha=0.3)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

# ===== Residual Diagnostics (Uncensored) =====
print("\n" + "="*70)
print("RESIDUAL DIAGNOSTICS (Uncensored Observations)")
print("="*70)

uncensored_mask = (y > 0)
residuals_uncensored = y[uncensored_mask] - X.values[uncensored_mask] @ beta_hat
standardized_resid = residuals_uncensored / sigma_hat

# Normality test
_, p_jarque_bera = stats.jarque_bera(standardized_resid)
print(f"Jarque-Bera test for normality: p-value = {p_jarque_bera:.4f}")
if p_jarque_bera < 0.05:
    print("  ✗ Reject normality (may affect inference)")
else:
    print("  ✓ Normality assumption supported")

# Heteroskedasticity (Breusch-Pagan)
from statsmodels.stats.diagnostic import het_breuschpagan
_, p_bp, _, _ = het_breuschpagan(residuals_uncensored, 
                                 X.values[uncensored_mask])
print(f"Breusch-Pagan test: p-value = {p_bp:.4f}")
if p_bp < 0.05:
    print("  ✗ Heteroskedasticity detected (consider robust SE)")
else:
    print("  ✓ Homoskedasticity assumption supported")
```

## 6. Challenge Round
When is Tobit inappropriate or fails?
- **Sample selection vs corner solution**: Workers choose participation (selection) vs consumers buy 0 units (corner) → Use Heckman if selection
- **Heteroskedasticity**: Tobit assumes constant σ² → Use heteroskedastic Tobit or robust methods
- **Non-normality**: Heavy tails or skewness → Consider Box-Cox transformation or semiparametric estimators
- **Multiple censoring points**: Different thresholds for subgroups → Use two-limit or interval regression
- **Endogeneity**: Censored endogenous variable → Use IV-Tobit or control function approach
- **Count data at zero**: Y ∈ {0, 1, 2, ...} not continuous → Use zero-inflated Poisson/negative binomial

## 7. Key References
- [Tobin (1958) - Estimation of Relationships for Limited Dependent Variables](https://doi.org/10.2307/1907382)
- [Amemiya (1984) - Tobit Models: A Survey](https://doi.org/10.1016/0304-4076(84)90074-5)
- [Greene - Econometric Analysis (Ch 19)](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899)

---
**Status:** Censored regression model | **Complements:** Heckman Selection, Truncated Regression, Hurdle Models, Cragg Model
