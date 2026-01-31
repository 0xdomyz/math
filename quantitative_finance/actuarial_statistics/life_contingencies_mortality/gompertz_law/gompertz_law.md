# Gompertz Law

## 1. Concept Skeleton
**Definition:** Parametric mortality model μₓ = A·Bˣ; exponential force of mortality increasing with age  
**Purpose:** Fit adult mortality (ages 30-90) analytically, project survival curves, set assumptions for insurance pricing  
**Prerequisites:** Force of mortality, exponential functions, parametric curve fitting, maximum likelihood estimation

## 2. Comparative Framing
| Model | Gompertz (μₓ = AB^x) | Makeham (μₓ = A + BC^x) | Weibull (μₓ = λk(x)^{k-1}) |
|-------|----------------------|-------------------------|--------------------------|
| **Parameters** | 2 (A, B) | 3 (A, B, C) | 2 (λ, k) |
| **Ages 20-40** | Fits poorly (too high) | Better fit across all | Flexible, good fit |
| **Ages 60+** | Excellent fit | Excellent fit | Good but underestimates |
| **Interpretation** | Pure senescence | Senescence + accidents | Shape parameter flexibility |
| **Use** | Quick estimation | Standard actuarial | Advanced models |

## 3. Examples + Counterexamples

**Simple Example:**  
A = 0.0001, B = 1.075: μ₃₀ ≈ 0.0003, μ₆₀ ≈ 0.01, μ₈₀ ≈ 0.15; closely matches US male data ages 50-90

**Failure Case:**  
Gompertz alone on ages 0-30: Predicts infant mortality μ₀ ≈ 0.0001, actual ≈ 0.007; underestimates juvenile hazard by 70×

**Edge Case:**  
B very close to 1 (B = 1.01): Mortality increases very slowly; for 100+ years, μₓ barely changes; unrealistic

## 4. Layer Breakdown
```
Gompertz Model Structure:
├─ Core Equation:
│   ├─ μₓ = A · Bˣ  [force of mortality]
│   ├─ ln(μₓ) = ln(A) + x·ln(B)  [log-linear]
│   ├─ Survival: ₚₓ = exp[-A(Bˣ⁺ᵖ - Bˣ)/(ln B)]
│   └─ Life table: lₓ = l₀ · exp[-A(Bˣ - 1)/(ln B)]
├─ Parameters:
│   ├─ A (intercept): Baseline force at x=0; ~0.0001 to 0.001
│   ├─ B (rate): Age escalation factor; ~1.07 to 1.10 (7-10% annual increase)
│   └─ B-1: Annual proportional increase in force (~7%)
├─ Fitting Methods:
│   ├─ Linear regression: ln(qₓ) ≈ ln(A) + x·ln(B) on selected ages
│   ├─ Maximum likelihood: Binomial likelihood of deaths given exposures
│   ├─ Weighted least squares: Higher weight for larger exposures
│   └─ Optimization: Nelder-Mead, BFGS on log-likelihood
├─ Validation:
│   ├─ Chi-square: Observed vs expected deaths
│   ├─ Residual plots: Standardized residuals by age
│   ├─ Graphic inspection: Visual fit to empirical qₓ
│   └─ Forecast backtesting: Compare predicted to actual 5 years later
└─ Extensions:
    ├─ Gompertz-Makeham: Add constant A₀ for accident component
    ├─ Stochastic: Allow B(t) to vary with time (Lee-Carter)
    └─ Cohort: Apply improvement factors to age forward
```

**Interaction:** Raw death rates → Log transform → Linear fit → Extract A, B → Validate → Apply

## 5. Mini-Project
Fit Gompertz law to mortality data, compare fitting methods, and forecast:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.stats import linregress, chi2

# 1. SIMULATED MORTALITY DATA (realistic US male-like)
np.random.seed(42)

ages = np.arange(30, 96)
# True parameters
A_true = 0.00008
B_true = 1.075

# True force of mortality
mu_true = A_true * (B_true ** ages)

# Convert to annual probability (Gompertz survival formula)
# ₚₓ = exp[-A(B^{x+p} - B^x)/(ln B)]
# qₓ = 1 - ₚₓ
ln_B = np.log(B_true)
qx_true = 1 - np.exp(-A_true * (B_true**(ages+1) - B_true**ages) / ln_B)
qx_true = np.minimum(qx_true, 0.99999)

# Simulate deaths from exposures
exposures = np.random.poisson(10000, len(ages))  # ~10,000 per age
deaths = np.random.binomial(exposures, qx_true)

# Calculate empirical mortality rates
qx_empirical = deaths / exposures
mu_empirical = -np.log(1 - qx_empirical)

print("GOMPERTZ FITTING: Simulated Data")
print(f"True parameters: A = {A_true:.6f}, B = {B_true:.5f}")
print(f"Data: {len(ages)} ages, {exposures.sum():.0f} total exposures, {deaths.sum():.0f} deaths")
print()

# 2. LINEAR REGRESSION METHOD (traditional)
# ln(qₓ) ≈ ln(A) + x·ln(B)
log_qx = np.log(np.maximum(qx_empirical, 1e-6))

slope, intercept, r_value, p_value, std_err = linregress(ages, log_qx)

A_lin = np.exp(intercept)
B_lin = np.exp(slope)

print("METHOD 1: LINEAR REGRESSION (log qx vs age)")
print(f"Fitted A = {A_lin:.6f} (true: {A_true:.6f})")
print(f"Fitted B = {B_lin:.5f} (true: {B_true:.5f})")
print(f"R² = {r_value**2:.5f}, Slope std error = {std_err:.6f}")

# Predictions
qx_lin = 1 - np.exp(-A_lin * (B_lin**(ages+1) - B_lin**ages) / np.log(B_lin))
qx_lin = np.minimum(qx_lin, 0.99999)

print()

# 3. MAXIMUM LIKELIHOOD METHOD (statistically optimal)
def gompertz_qx(ages_data, A, B):
    """Gompertz mortality probability from parameters"""
    ln_B = np.log(B)
    qx = 1 - np.exp(-A * (B**(ages_data + 1) - B**ages_data) / ln_B)
    return np.minimum(np.maximum(qx, 1e-10), 1 - 1e-10)

def neg_log_likelihood(params, ages_data, deaths_data, exposures_data):
    """Negative log-likelihood for binomial model"""
    A, B = params
    
    if A <= 0 or B <= 1 or A > 0.1 or B > 1.2:
        return 1e10
    
    qx = gompertz_qx(ages_data, A, B)
    
    # Binomial log-likelihood
    ll = np.sum(deaths_data * np.log(qx) + (exposures_data - deaths_data) * np.log(1 - qx))
    return -ll

p0 = [0.0001, 1.07]
result_mle = minimize(neg_log_likelihood, p0,
                     args=(ages, deaths, exposures),
                     method='Nelder-Mead')

A_mle, B_mle = result_mle.x

print("METHOD 2: MAXIMUM LIKELIHOOD ESTIMATION")
print(f"Fitted A = {A_mle:.6f} (true: {A_true:.6f})")
print(f"Fitted B = {B_mle:.5f} (true: {B_true:.5f})")
print(f"Negative Log-Likelihood = {result_mle.fun:.2f}")

qx_mle = gompertz_qx(ages, A_mle, B_mle)
print()

# 4. WEIGHTED LEAST SQUARES
# Weight by exposure size (larger samples more reliable)
weights = np.sqrt(exposures)

def weighted_gompertz(params, ages_data, log_qx_data, weights_data):
    """Weighted least squares for log(qx)"""
    A, B = params
    
    if A <= 0 or B <= 1 or A > 0.1 or B > 1.2:
        return 1e10
    
    qx = gompertz_qx(ages_data, A, B)
    log_qx_pred = np.log(np.maximum(qx, 1e-6))
    
    residuals = (log_qx_data - log_qx_pred) * weights_data
    return np.sum(residuals**2)

result_wls = minimize(weighted_gompertz, p0,
                     args=(ages, log_qx, weights),
                     method='BFGS')

A_wls, B_wls = result_wls.x

print("METHOD 3: WEIGHTED LEAST SQUARES")
print(f"Fitted A = {A_wls:.6f} (true: {A_true:.6f})")
print(f"Fitted B = {B_wls:.5f} (true: {B_true:.5f})")

qx_wls = gompertz_qx(ages, A_wls, B_wls)
print()

# 5. GOODNESS-OF-FIT TESTS
def chi_square_gof(deaths_obs, deaths_exp, df_reduction=2):
    """Chi-square test for goodness of fit"""
    chi2_stat = np.sum((deaths_obs - deaths_exp)**2 / np.maximum(deaths_exp, 1))
    df = len(deaths_obs) - df_reduction
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p_value, df

deaths_exp_mle = exposures * qx_mle
chi2_stat, p_val, df = chi_square_gof(deaths, deaths_exp_mle)

print("GOODNESS-OF-FIT (MLE model):")
print(f"Chi-square statistic = {chi2_stat:.2f}")
print(f"Degrees of freedom = {df}")
print(f"P-value = {p_val:.4f}")
print(f"Conclusion: {'Good fit' if p_val > 0.05 else 'Poor fit'}")
print()

# 6. AGE-SPECIFIC RESIDUALS
residuals_std = (deaths - deaths_exp_mle) / np.sqrt(np.maximum(deaths_exp_mle, 1))

print("STANDARDIZED RESIDUALS (selected ages):")
print("Age\tObserved\tExpected\tResidual")
for age in [40, 50, 60, 70, 80]:
    idx = age - ages[0]
    print(f"{age}\t{deaths[idx]}\t\t{deaths_exp_mle[idx]:.1f}\t\t{residuals_std[idx]:.2f}")
print()

# 7. FORECAST TO FUTURE YEARS
future_ages = np.arange(96, 106)
qx_forecast_mle = gompertz_qx(future_ages, A_mle, B_mle)

print("FORECASTED MORTALITY (using MLE Gompertz):")
print("Age\tForecasted qx")
for age, qx in zip(future_ages[::2], qx_forecast_mle[::2]):
    print(f"{age}\t{qx:.5f}")
print()

# 8. FORCE OF MORTALITY COMPARISON
mu_gompertz_lin = A_lin * (B_lin ** ages)
mu_gompertz_mle = A_mle * (B_mle ** ages)
mu_gompertz_wls = A_wls * (B_wls ** ages)

# 9. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Fitted vs Empirical (log scale)
ax = axes[0, 0]
ax.semilogy(ages, qx_empirical, 'o', markersize=6, alpha=0.7, 
           label='Empirical (deaths/exposures)', color='black')
ax.semilogy(ages, qx_lin, '--', linewidth=2, label='Linear Regression', color='blue')
ax.semilogy(ages, qx_mle, '-', linewidth=2.5, label='Maximum Likelihood', color='red')
ax.semilogy(ages, qx_wls, ':', linewidth=2, label='Weighted LS', color='green')
ax.semilogy(ages, qx_true, '--', linewidth=1.5, alpha=0.5, 
           label='True Gompertz', color='gray')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Mortality Probability qx (log scale)', fontsize=11)
ax.set_title('Gompertz Fitting: Three Methods', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3, which='both')

# Plot 2: Force of mortality (linear scale)
ax = axes[0, 1]
ax.plot(ages, mu_empirical, 'o', markersize=4, alpha=0.5, 
       label='Empirical', color='black')
ax.plot(ages, mu_gompertz_mle, '-', linewidth=2.5, label='MLE Gompertz', color='red')
ax.plot(ages, mu_true, '--', linewidth=2, alpha=0.6, label='True', color='gray')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Force of Mortality μx', fontsize=11)
ax.set_title('Force of Mortality: Gompertz vs Empirical', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Standardized residuals
ax = axes[1, 0]
ax.scatter(ages, residuals_std, s=50, alpha=0.7, color='darkblue')
ax.axhline(0, color='r', linestyle='-', linewidth=2)
ax.axhline(2, color='orange', linestyle='--', alpha=0.7)
ax.axhline(-2, color='orange', linestyle='--', alpha=0.7)
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Standardized Residuals', fontsize=11)
ax.set_title('Goodness-of-Fit: Residual Analysis', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Parameter estimates across methods
ax = axes[1, 1]
methods = ['Linear Reg', 'MLE', 'Weighted LS', 'True']
A_vals = [A_lin, A_mle, A_wls, A_true]
B_vals = [B_lin, B_mle, B_wls, B_true]

x_pos = np.arange(len(methods))
ax_twin = ax.twinx()

bars1 = ax.bar(x_pos - 0.2, A_vals, 0.4, label='A', color='steelblue', edgecolor='black')
bars2 = ax_twin.bar(x_pos + 0.2, B_vals, 0.4, label='B', color='coral', edgecolor='black')

ax.set_ylabel('Parameter A', fontsize=11, color='steelblue')
ax_twin.set_ylabel('Parameter B', fontsize=11, color='coral')
ax.set_title('Parameter Estimates: Three Methods', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.tick_params(axis='y', labelcolor='steelblue')
ax_twin.tick_params(axis='y', labelcolor='coral')
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.5f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('gompertz_law_fitting.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 6. Challenge Round
When Gompertz law fails:
- **Ages 0-40**: Exponential growth doesn't fit; U-shaped infant/juvenile hazard; use Heligman-Pollard
- **Cohort heterogeneity**: Frail subgroup dies early, robust live long; single Gompertz masks bimodal distribution
- **Mortality improvement**: B changes over time (Lee-Carter needed); static B assumes no progress
- **Extreme tail (95+)**: Selection bias; healthiest survivors only; force increases slower than Gompertz predicts
- **Different causes**: Cancer accelerates, heart disease slows with age; mixture violates proportional structure

## 7. Key References
- [Gompertz Law (Wikipedia)](https://en.wikipedia.org/wiki/Gompertz%E2%80%93Makeham_law_of_mortality) - Mathematical background
- [Bowers et al., Actuarial Mathematics (Chapter 3)](https://www.soa.org/) - Parametric mortality models
- [CMI Analysis of Mortality Trends](https://www.cmi.ac.uk/) - Empirical Gompertz fits to UK data

---
**Status:** Classical parametric model | **Complements:** Force of Mortality, Makeham Law, Lee-Carter Stochastic Model
