# Force of Mortality

## 1. Concept Skeleton
**Definition:** Instantaneous rate of death μₓ at age x; continuous hazard function measuring force per unit time  
**Purpose:** Model age-dependent death intensity, fit parametric mortality laws, project future survival probabilities  
**Prerequisites:** Survival functions, probability density, hazard rates, continuous distributions

## 2. Comparative Framing
| Approach | Force of Mortality (μₓ) | Probability qₓ | Survival Sₓ |
|----------|-------------------------|----------------|------------|
| **Definition** | Instantaneous death rate | Probability of death in [x, x+1) | Cumulative survival to x |
| **Domain** | Continuous; per year infinitesimal | Discrete; annual probability | Cumulative 0→x |
| **Relationship** | μₓ = -d(ln Sₓ)/dx | qₓ ≈ μₓ (for small intervals) | Sₓ = exp(-∫μₜdt) |
| **Use Case** | Analytical mortality models | Life table calculation | Age-specific survival |

## 3. Examples + Counterexamples

**Simple Example:**  
Gompertz law μₓ = 0.0001·1.1^x fits human mortality well; force increases exponentially with age

**Failure Case:**  
Constant force μₓ = 0.05 (unrealistic): Predicts same death rate at age 20 and 80; doesn't capture age structure

**Edge Case:**  
Makeham law μₓ = A + B·C^x: Small constant A captures accidents/random deaths (independent of age), exponential BC^x for senescence

## 4. Layer Breakdown
```
Force of Mortality Structure:
├─ Definition & Relationships:
│   ├─ μₓ = lim(Δt→0) q_{x,Δt} / Δt
│   ├─ Sₓ(t) = exp(-∫₀^t μ_{x+u} du)
│   ├─ qₓ = 1 - exp(-∫₀^1 μ_{x+u} du)
│   └─ ₚₓ = exp(-∫₀^p μ_{x+u} du)  [survival p years]
├─ Parametric Models:
│   ├─ Exponential: μₓ = λ (constant, unrealistic)
│   ├─ Gompertz: μₓ = Ae^{Bx} (best fit adult mortality)
│   ├─ Makeham: μₓ = A + Be^{Cx} (adds accident rate A)
│   ├─ Weibull: μₓ = λk(x)^{k-1} (flexible shape)
│   └─ Lee-Carter: μₓ(t) = e^{αₓ + βₓκₜ} (stochastic trend)
├─ Estimation:
│   ├─ Maximum Likelihood: From life table deaths dₓ, exposures Eₓ
│   ├─ Graduation: Smooth empirical ₚₓ to parametric curve
│   ├─ Whittaker-Henderson: Penalized likelihood with smoothness
│   └─ Kernel Smoothing: Non-parametric local averaging
└─ Validation:
    ├─ Goodness-of-Fit: Chi-square on observed vs expected deaths
    ├─ Residual Analysis: Age-standardized deviations
    └─ Forecasting Backtests: Compare 5-year-ahead predictions to actual
```

**Interaction:** Life table dₓ → Estimate μₓ → Fit curve → Project future → Validate forecast

## 5. Mini-Project
Estimate force of mortality using parametric and non-parametric methods:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import chi2

# Simulated life table data (ages 0-100)
ages = np.arange(0, 101)
# Use Gompertz-Makeham parameters for realistic mortality
A = 0.0001  # Accident component
B = 0.00035  # Senescence coefficient
C = 1.075   # Age factor

# True force of mortality
mu_true = A + B * np.exp(C * ages)

# Generate life table: Start with 100,000 and apply force
lx = np.zeros(101)
lx[0] = 100000
dx = np.zeros(101)

for x in range(100):
    # Approximate qx from mu using: qx ≈ mu_x for small values
    # More accurate: qx = 1 - exp(-mu_x)
    qx = 1 - np.exp(-mu_true[x])
    dx[x] = lx[x] * qx
    lx[x+1] = lx[x] - dx[x]

# Calculate empirical qx
qx_empirical = dx[:-1] / lx[:-1]

# Convert to empirical mu (force of mortality)
mu_empirical = -np.log(1 - qx_empirical)

# 1. PARAMETRIC FITTING: Gompertz-Makeham
def gompertz_makeham(x, params):
    A, B, C = params
    return A + B * np.exp(C * x)

def negative_likelihood_gm(params, ages_data, dx_data, lx_data):
    """Negative log-likelihood for Gompertz-Makeham"""
    A, B, C = params
    
    # Avoid invalid parameters
    if A < 0 or B < 0 or C < 0 or A + B > 1:
        return 1e10
    
    mu = A + B * np.exp(C * ages_data)
    qx = 1 - np.exp(-mu)
    
    # Ensure qx is in (0, 1)
    qx = np.clip(qx, 1e-10, 1-1e-10)
    
    # Log-likelihood: deaths follow binomial
    ll = np.sum(dx_data * np.log(qx) + (lx_data - dx_data) * np.log(1 - qx))
    return -ll

# Fit on ages 20-95 (avoid extreme ages with sparse data)
mask = (ages >= 20) & (ages <= 95)
ages_fit = ages[mask]
dx_fit = dx[mask]
lx_fit = lx[mask]

# Initial guess
p0 = [0.0001, 0.0002, 1.07]

result_gm = minimize(negative_likelihood_gm, p0, 
                     args=(ages_fit, dx_fit, lx_fit),
                     method='Nelder-Mead')

params_gm = result_gm.x
A_est, B_est, C_est = params_gm

mu_gm_fitted = gompertz_makeham(ages, params_gm)

print("GOMPERTZ-MAKEHAM PARAMETER ESTIMATES:")
print(f"A (accident rate): {A_est:.6f} (true: {A:.6f})")
print(f"B (senescence coeff): {B_est:.6f} (true: {B:.6f})")
print(f"C (age factor): {C_est:.6f} (true: {C:.6f})")
print(f"Negative Log-Likelihood: {result_gm.fun:.2f}\n")

# 2. PARAMETRIC FITTING: Gompertz only (2 parameters)
def gompertz(x, params):
    B, C = params
    return B * np.exp(C * x)

def negative_likelihood_g(params, ages_data, dx_data, lx_data):
    """Negative log-likelihood for Gompertz"""
    B, C = params
    
    if B < 0 or C < 0 or B > 1:
        return 1e10
    
    mu = B * np.exp(C * ages_data)
    qx = 1 - np.exp(-mu)
    qx = np.clip(qx, 1e-10, 1-1e-10)
    
    ll = np.sum(dx_data * np.log(qx) + (lx_data - dx_data) * np.log(1 - qx))
    return -ll

p0_g = [0.0002, 1.07]
result_g = minimize(negative_likelihood_g, p0_g,
                   args=(ages_fit, dx_fit, lx_fit),
                   method='Nelder-Mead')

params_g = result_g.x
mu_g_fitted = gompertz(ages, params_g)

print("GOMPERTZ PARAMETER ESTIMATES (2-param):")
print(f"B (coeff): {params_g[0]:.6f} (true B: {B:.6f})")
print(f"C (factor): {params_g[1]:.6f} (true C: {C:.6f})")
print(f"Negative Log-Likelihood: {result_g.fun:.2f}\n")

# 3. NON-PARAMETRIC: Kernel smoothing (Epanechnikov kernel)
def kernel_smooth_mu(ages, mu_empirical, bandwidth=3):
    """Non-parametric smoothing of force of mortality"""
    mu_smooth = np.zeros_like(ages, dtype=float)
    
    for i, x in enumerate(ages):
        # Epanechnikov kernel: K(u) = 0.75(1 - u²) for |u| ≤ 1
        distances = np.abs(ages - x)
        u = distances / bandwidth
        
        # Apply kernel
        kernel_weights = np.where(u <= 1, 0.75 * (1 - u**2), 0)
        kernel_weights /= kernel_weights.sum()  # Normalize
        
        mu_smooth[i] = np.sum(kernel_weights * mu_empirical)
    
    return mu_smooth

mu_kernel = kernel_smooth_mu(ages, mu_empirical, bandwidth=3)

# 4. GOODNESS-OF-FIT TEST
def chi_square_gof(observed, expected):
    """Chi-square goodness-of-fit test"""
    # Combine small expected cells
    chi2_stat = np.sum((observed - expected)**2 / np.maximum(expected, 1))
    df = len(observed) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p_value, df

# Expected deaths under fitted model
qx_gm_fitted = 1 - np.exp(-mu_gm_fitted)
dx_gm_expected = lx * qx_gm_fitted

# Chi-square (on ages 20-95)
mask_chi2 = (ages >= 20) & (ages <= 95)
obs = dx[mask_chi2]
exp = dx_gm_expected[mask_chi2]
chi2_stat, p_val, df = chi_square_gof(obs, exp)

print(f"GOODNESS-OF-FIT TEST (Gompertz-Makeham):")
print(f"Chi-square statistic: {chi2_stat:.2f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_val:.4f}")
print(f"Conclusion: {'Good fit' if p_val > 0.05 else 'Poor fit'}\n")

# 5. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Force of mortality curves
ax = axes[0, 0]
ax.plot(ages, mu_true, 'k--', linewidth=2.5, label='True (Gompertz-Makeham)')
ax.plot(ages, mu_empirical, 'o', markersize=4, alpha=0.5, label='Empirical (from qx)')
ax.plot(ages, mu_gm_fitted, 'r-', linewidth=2, label='Fitted Gompertz-Makeham')
ax.plot(ages, mu_g_fitted, 'b-', linewidth=2, label='Fitted Gompertz')
ax.plot(ages, mu_kernel, 'g--', linewidth=2, label='Kernel Smoothing')
ax.set_xlabel('Age (x)', fontsize=11)
ax.set_ylabel('Force of Mortality μₓ', fontsize=11)
ax.set_title('Force of Mortality: True vs Fitted', fontsize=12, fontweight='bold')
ax.set_ylim([0, 0.3])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 2: Log scale (better visualization)
ax = axes[0, 1]
ax.semilogy(ages, mu_true, 'k--', linewidth=2.5, label='True')
ax.semilogy(ages, mu_empirical, 'o', markersize=4, alpha=0.5, label='Empirical')
ax.semilogy(ages, mu_gm_fitted, 'r-', linewidth=2, label='Gompertz-Makeham')
ax.semilogy(ages, mu_kernel, 'g--', linewidth=2, label='Kernel')
ax.set_xlabel('Age (x)', fontsize=11)
ax.set_ylabel('Force of Mortality μₓ (log scale)', fontsize=11)
ax.set_title('Force of Mortality (Log Scale)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')

# Plot 3: Residuals (observed - expected deaths)
ax = axes[1, 0]
residuals = dx - dx_gm_expected
residuals_std = residuals / np.sqrt(np.maximum(dx_gm_expected, 1))
ax.scatter(ages[mask_chi2], residuals_std[mask_chi2], alpha=0.6, s=40)
ax.axhline(0, color='r', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±2 SD')
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Age (x)', fontsize=11)
ax.set_ylabel('Standardized Residuals', fontsize=11)
ax.set_title('Goodness-of-Fit: Residual Analysis', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 4: Survival probability (derived from μ)
ax = axes[1, 1]
# Calculate survival from fitted models
Sx_true = np.exp(-np.cumsum(mu_true))
Sx_gm = np.exp(-np.cumsum(mu_gm_fitted))
Sx_kernel = np.exp(-np.cumsum(mu_kernel))

ax.plot(ages, Sx_true, 'k--', linewidth=2.5, label='True survival')
ax.plot(ages, Sx_gm, 'r-', linewidth=2, label='From GM fitted μ')
ax.plot(ages, Sx_kernel, 'g--', linewidth=2, label='From kernel μ')
ax.set_xlabel('Age (x)', fontsize=11)
ax.set_ylabel('Survival Probability Sₓ', fontsize=11)
ax.set_title('Derived Survival Curves', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('force_of_mortality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. PROJECTION DEMONSTRATION
print("MORTALITY PROJECTION (5 years forward):")
print("Age\tTrue μₓ\tEstimated μₓ\tError")
print("-" * 50)
for age in [30, 50, 70, 90]:
    idx = age
    print(f"{age}\t{mu_true[idx]:.5f}\t{mu_gm_fitted[idx]:.5f}\t{abs(mu_true[idx] - mu_gm_fitted[idx]):.5f}")
```

## 6. Challenge Round
When force of mortality models fail:
- **Cohort effects**: Gompertz-Makeham ignores generation-specific mortality improvements; use Lee-Carter or stochastic models
- **Young ages**: Infant/juvenile mortality has U-shape (Heligman-Pollard needed), not monotonic
- **Pandemic/war**: Force spikes unpredictably; historical data useless, scenario analysis required
- **Very old ages (95+)**: Sparse data, selection bias (healthy survivors); bootstrap confidence intervals wider
- **Heterogeneity**: Smokers, non-smokers different hazards; need stratified models or frailty terms

## 7. Key References
- [Actuarial Mathematics (Bowers et al.)](https://www.soa.org/) - Life contingencies fundamentals
- [Gompertz-Makeham Law (Wikipedia)](https://en.wikipedia.org/wiki/Gompertz%E2%80%93Makeham_law_of_mortality) - Parametric models
- [Lee-Carter Model (CMI Longevity)](https://www.cmi.ac.uk/) - Stochastic mortality projection

---
**Status:** Foundational actuarial model | **Complements:** Survival Analysis, Life Tables, Mortality Laws
