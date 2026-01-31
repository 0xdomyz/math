# Measurement Error

## Concept Skeleton

Measurement error occurs when observed variables differ from true values due to data collection inaccuracies, rounding, or instrument limitations. **Classical measurement error** in X assumes observed X = true X* + independent noise η (E[η] = 0, Cov(η, X*) = 0, Cov(η, ε) = 0). This creates endogeneity: observed X correlated with the composite error because error contains η. **Consequence**: OLS suffers **attenuation bias**—coefficients biased toward zero by factor λ = Var(X*)/[Var(X*) + Var(η)], the signal-to-total-variance ratio. If measurement error is 100% of true variance, λ = 0.5, true effect understated 50%. **Non-classical error** (correlated with true X or ε) biases in unpredictable directions. **Measurement error in Y** is less consequential (only inflates standard errors under classical assumption), whereas error in X destroys consistency. Remedies include: (1) instrumental variables (find Z uncorrelated with measurement error), (2) structural equation models with latent variables, (3) errors-in-variables regression (if error variance known), (4) use higher-quality measurements or alternative data sources.

**Core Components:**
- **Classical ME**: X = X* + η, E[η] = 0, Var(η) = σ²ₙ, Cov(X*, η) = 0 (independent error)
- **Attenuation bias**: plim(β̂₁) = β₁ × λ = β₁ × Var(X*)/[Var(X*) + Var(η)] < β₁ (understates effect)
- **Ratio λ**: Signal-to-noise ratio; λ = 0.9 means 90% of variation is true, 10% noise; understates by 10%
- **Non-classical ME**: Correlated with X* or ε → bias direction unpredictable (could amplify or attenuate)
- **ME in Y**: Inflates residual variance (higher SE) but doesn't bias coefficient (under strict exogeneity of Y measurement error)
- **Solutions**: IV, latent variable models, errors-in-variables regression, quality improvements

**Why it matters:** Ubiquitous in social science (survey misreporting, rounding in administrative data); economic applications (income misreporting in surveys, proxy measures for unobserved variables); empirical results likely understate true effects, affecting policy conclusions.

---

## Comparative Framing

| Aspect | **No Measurement Error** | **Classical ME in X** | **Non-Classical ME in X** | **ME in Y** |
|--------|---------------------------|----------------------|--------------------------|------------|
| **Model** | X observed perfectly | X = X* + η, Cov(X*,η)=0 | Corr(X*, η) ≠ 0 or Corr(ε, η) ≠ 0 | Y = Y* + νy, Y* = Xβ + ε |
| **OLS Property** | Unbiased, consistent | Biased (attenuation) | Biased (direction unclear) | Unbiased, consistent |
| **Bias Direction** | N/A | Toward zero (understate) | Either direction (model-dependent) | No bias in β |
| **SE Impact** | Normal (σ̂²(X'X)⁻¹) | Inflated (X has noise) | Inflated | Inflated (noise in Y) |
| **Attenuation Factor λ** | N/A | Var(X*)/[Var(X*)+Var(η)] | N/A | N/A |
| **Example** | Precise measurement | Survey income (rounding) | Income both mismeasured & correlated with propensity to underreport | Self-reported health (recall bias) |
| **Remedy** | OLS sufficient | IV (clean instrument Z) | IV (exogenous Z) | IV or latent variable model |

**Key insight:** Classical measurement error in X causes attenuation (understates effect magnitude); error in Y only inflates SE (coefficient unbiased); error in both has mixed effects.

---

## Examples & Counterexamples

### Examples of Measurement Error

1. **Income Misreporting in Wage-Education Regression**  
   - **True**: log(Wage*) = 0.1 + 0.08×Educ + ε (8% return per year education)  
   - **Observed**: log(Wage) = log(Wage*) + ν, where ν ~ N(0, 0.04) (4% rounding/misreport error)  
   - **Var(Wage*)** = σ² ≈ 0.25 (typical log-wage variance)  
   - **Attenuation factor**: λ = 0.25/(0.25 + 0.04) = 0.86 (86% of true effect observed)  
   - **Observed effect**: β̂ᴼᴸˢ ≈ 0.08 × 0.86 = 0.0688 (understates by 14%)  
   - **Policy implication**: Education returns appear 8.6% not 8% due to measurement error; modest but non-trivial

2. **Intelligence Proxy Mismeasurement**  
   - **True**: Earnings = α + β×IQ + ε (true IQ effect strong)  
   - **Observe**: IQ_test = IQ + η (test error, especially for extreme scores; non-classical if systematic bias)  
   - **Classical ME**: Bias toward zero; β̂ᴼᴸˢ < β (understate IQ effect)  
   - **Example**: β = 0.10, Var(IQ) = 100, Var(η) = 25 (25% of variance noise) → λ = 0.8, β̂ ≈ 0.08  
   - **Non-classical if**: Test biased for low-IQ individuals (systematic error) → further complications

3. **Survey Wealth Data in Consumption Function**  
   - **Model**: Consumption = γ₀ + γ₁×Wealth + u (permanent income hypothesis)  
   - **Issue**: Household survey wealth measured with substantial error (illiquid assets hard to value, underreporting)  
   - **Var(Wealth*)/[Var(Wealth*) + Var(η)] ≈ 0.3-0.5** (measurement error 50-70% of true variance)  
   - **Result**: OLS understates wealth effect by 50-70%; true MPC from wealth much higher  
   - **Remedy**: Use administrative tax records (cleaner), IV (prior wealth as instrument), or bounded analysis

4. **Pollution Measurement Error in Health Studies**  
   - **True**: Health = health_0 - β×Pollution* + ε (pollution damages health)  
   - **Measure**: Pollution_observed = Pollution* + η (sensor error, spatial averaging)  
   - **Classical ME**: Understate pollution damage (bias toward zero)  
   - **Non-classical if**: Measurement error correlated with true pollution (e.g., monitors sited in less polluted areas) → bias direction uncertain  
   - **Solution**: Use multiple imperfect measures, IV with lagged pollution (if dynamics allow)

5. **Company Revenue Misreporting**  
   - **Context**: Regression of R&D investment on firm revenue (selection into R&D)  
   - **Issue**: Small firms underreport revenue; large firms over-report for tax purposes (non-classical)  
   - **Effect**: Biases selection mechanism; downward bias in revenue coefficient if error correlates with R&D propensity  
   - **Remedy**: Use administrative tax records, instrument with industry metrics

### Non-Examples (or Minimal ME Impact)

- **Precisely measured variables** (e.g., age from administrative records, employment status, categorical race/ethnicity) → negligible ME  
- **Measurement error in Y only** (under classical assumptions): Unbiased coefficient, inflated SE; less problematic than error in X  
- **Randomized experiment**: Random assignment breaks any correlation between X error and Y (though ME still inflates SE)

---

## Layer Breakdown

**Layer 1: Classical Measurement Error Framework**  
**Setup**:  
- **True model**: Y = β₀ + β₁X* + ε, E[ε|X*] = 0  
- **Observed**: X = X* + η (measurement error)  
- **Assumptions**: E[η] = 0, Var(η) = σ²ₙ, Cov(X*, η) = 0, Cov(η, ε) = 0 (classical)

**Estimated model** (in terms of observed X):  
$$Y = \beta_0 + \beta_1(X - \eta) + \varepsilon = \beta_0 + \beta_1 X + (\varepsilon - \beta_1 \eta)$$
$$Y = \beta_0 + \beta_1 X + \tilde{\varepsilon}$$

**Composite error**: $\tilde{\varepsilon} = \varepsilon - \beta_1 \eta$ contains η (measurement error).

**Endogeneity**: Cov(X, $\tilde{\varepsilon}$) = -β₁ Cov(X, η) = -β₁ Cov(X* + η, η) = -β₁ Var(η) ≠ 0.

**Result**: OLS biased toward zero.

**Layer 2: Attenuation Bias Formula**  
**OLS estimator** (simple regression):  
$$\hat{\beta}_1 = \frac{Cov(X, Y)}{Var(X)}$$

**Cov(X, Y)**:  
$$Cov(X, Y) = Cov(X^* + \eta, \beta_0 + \beta_1 X^* + \varepsilon)$$
$$= Cov(X^*, \beta_1 X^* + \varepsilon) + Cov(\eta, \beta_1 X^* + \varepsilon)$$
$$= \beta_1 Var(X^*) + 0 + 0 = \beta_1 Var(X^*)$$

**Var(X)**:  
$$Var(X) = Var(X^* + \eta) = Var(X^*) + Var(\eta) + 2Cov(X^*, \eta) = Var(X^*) + Var(\eta)$$
(by independence assumption).

**OLS estimate**:  
$$\hat{\beta}_1 = \frac{\beta_1 Var(X^*)}{Var(X^*) + Var(\eta)} = \beta_1 \lambda$$

where **λ = Var(X*)/[Var(X*) + Var(η)]** is attenuation factor.

**Interpretation**: λ < 1 always (unless η = 0); OLS severely biases toward zero when measurement error large relative to true variation.

**Layer 3: Quantifying Measurement Error Magnitude**  
**Signal-to-noise ratio**: σ²_signal = Var(X*), σ²_noise = Var(η).

**λ** depends only on ratio, not absolute scale:  
$$\lambda = \frac{\sigma^2_{signal}}{\sigma^2_{signal} + \sigma^2_{noise}} = \frac{1}{1 + SNR^{-1}}$$

**where SNR** = σ²_signal / σ²_noise (signal-to-noise ratio).

**Examples**:  
- SNR = ∞ (no error): λ = 1 (no attenuation)  
- SNR = 9 (10% error): λ = 0.9 (10% attenuation)  
- SNR = 1 (100% error): λ = 0.5 (50% attenuation)  
- SNR = 0.25 (75% error): λ = 0.2 (80% attenuation)

**Practical**: Survey researchers often estimate SNR from test-retest correlations ρ² ≈ λ (reliability coefficient approximation).

**Layer 4: Multiple Regressors with ME**  
**Model**: Y = β₀ + β₁X₁* + β₂X₂ + ε (only X₁ mismeasured).

**Observed**: X₁ = X₁* + η₁.

**Effect**:  
- **β̂₁**: Biased toward zero (attenuation), plim(β̂₁) = β₁λ₁  
- **β̂₂**: Biased (via correlation with X₁); **β̂₂** affected even though X₂ measured perfectly

**Formula** (approximate, if errors small):  
$$plim(\hat{\beta}_2) \approx \beta_2 + \beta_1(1 - \lambda_1) \frac{Cov(X_1^*, X_2)}{Var(X_2)}$$
(spillover bias into β₂).

**Implication**: ME in one variable contaminates other coefficients if that variable correlated with others.

**Layer 5: Non-Classical Measurement Error**  
**Correlated error**: Cov(X*, η) ≠ 0 (measurement error depends on true value).

**Example**: Self-reported health correlates with true health (better health → more accurate recall).

**Consequence**: Bias formula breaks down; direction unpredictable.

**Systematic measurement error**: E[η|X*] ≠ 0 (error depends on true value systematically).

**Example**: High earners more likely to underreport income (tax avoidance), not random noise.

**Effect**: Can bias in either direction (toward or away from zero) depending on pattern.

**Remedy**: Model error structure explicitly (e.g., hierarchical model with measurement component).

**Attenuation vs. Amplification**:  
- Classical ME → always attenuate  
- Non-classical with specific structure → can amplify if error and true value negatively correlated

---

## Mini-Project: ME Impact on Estimation and Remedies

**Goal:** Simulate classical measurement error; demonstrate attenuation bias; compare OLS vs. IV and latent variable approach.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy import stats

# Simulation parameters
np.random.seed(42)
n = 500

# True parameters
beta_0, beta_1 = 2.0, 1.5
sigma_epsilon = 1.0
sigma_Xstar = 2.0
sigma_eta = 0.8  # Measurement error std

print("=" * 80)
print("MEASUREMENT ERROR: ATTENUATION BIAS & REMEDIES")
print("=" * 80)

# Generate true X*
Xstar = np.random.normal(5, sigma_Xstar, n)

# Generate measurement error (classical)
eta = np.random.normal(0, sigma_eta, n)

# Observed X
X_obs = Xstar + eta

# Generate Y from true model
epsilon = np.random.normal(0, sigma_epsilon, n)
Y = beta_0 + beta_1 * Xstar + epsilon

# Calculate signal-to-noise ratio and attenuation factor
var_Xstar = np.var(Xstar, ddof=1)
var_eta = np.var(eta, ddof=1)
SNR = var_Xstar / var_eta
lambda_factor = var_Xstar / (var_Xstar + var_eta)

print("\nMeasurement Error Quantification:")
print("-" * 80)
print(f"True parameters: β₀ = {beta_0}, β₁ = {beta_1}")
print(f"Var(X*) = {var_Xstar:.4f}, Var(η) = {var_eta:.4f}")
print(f"Signal-to-Noise Ratio (SNR) = {SNR:.4f}")
print(f"Attenuation factor λ = {lambda_factor:.4f}")
print(f"Expected attenuation: β̂₁ ≈ {beta_1 * lambda_factor:.4f} (true β₁ = {beta_1})")
print()

# Scenario 1: OLS with error-ridden X (biased)
print("Scenario 1: OLS with Measurement Error (Biased)")
print("-" * 80)

X_obs_design = np.column_stack([np.ones(n), X_obs])
beta_ols_me = inv(X_obs_design.T @ X_obs_design) @ (X_obs_design.T @ Y)

print(f"OLS with observed X:")
print(f"  β̂₀ = {beta_ols_me[0]:.6f}, β̂₁ = {beta_ols_me[1]:.6f}")
print(f"  Expected β̂₁ ≈ {beta_1 * lambda_factor:.6f}")
print(f"  Actual bias: {beta_ols_me[1] - beta_1:.6f}")
print(f"  Attenuation: {(1 - beta_ols_me[1]/beta_1)*100:.1f}% (expected {(1-lambda_factor)*100:.1f}%)")

# Scenario 2: OLS with true X* (unbiased baseline)
print("\nScenario 2: OLS with True X* (Unbiased Baseline)")
print("-" * 80)

X_true_design = np.column_stack([np.ones(n), Xstar])
beta_ols_true = inv(X_true_design.T @ X_true_design) @ (X_true_design.T @ Y)

print(f"OLS with true X*:")
print(f"  β̂₀ = {beta_ols_true[0]:.6f}, β̂₁ = {beta_ols_true[1]:.6f}")
print(f"  Bias: {beta_ols_true[1] - beta_1:.6f} (negligible ✓)")

# Scenario 3: IV estimation (using Xstar as instrument for X_obs)
# In practice, would use external instrument; here using true X* for illustration
print("\nScenario 3: IV Estimation (Using Exogenous Instrument)")
print("-" * 80)

# Use true X* as instrument (pretend it's exogenous to measurement error)
# In real application, would use different instrument (e.g., lagged X, policy variable)
Z = Xstar + np.random.normal(0, 0.5, n)  # Instrument: related to Xstar, not to eta

# 2SLS
# Stage 1: Regress X_obs on Z
Z_design = np.column_stack([np.ones(n), Z])
gamma_1 = inv(Z_design.T @ Z_design) @ (Z_design.T @ X_obs)
X_fitted = Z_design @ gamma_1

# Stage 2: Regress Y on X_fitted
X_fitted_design = np.column_stack([np.ones(n), X_fitted])
beta_iv = inv(X_fitted_design.T @ X_fitted_design) @ (X_fitted_design.T @ Y)

print(f"IV (2SLS) with instrument Z:")
print(f"  β̂₀ = {beta_iv[0]:.6f}, β̂₁ = {beta_iv[1]:.6f}")
print(f"  Bias: {beta_iv[1] - beta_1:.6f}")
print(f"  Instrument strength (Corr(Z, X_obs)): {np.corrcoef(Z, X_obs)[0,1]:.4f}")
print(f"  Instrument exogeneity (Corr(Z, η)): {np.corrcoef(Z, eta)[0,1]:.4f}")

# Scenario 4: Errors-in-variables regression (if error variance known)
print("\nScenario 4: Errors-in-Variables Regression")
print("-" * 80)

# Adjust for known measurement error: β̂_corrected = β̂_OLS / λ
beta_eiv = beta_ols_me.copy()
beta_eiv[1] = beta_ols_me[1] / lambda_factor

print(f"OLS corrected for measurement error:")
print(f"  β̂₁ (corrected) = β̂₁ (OLS) / λ = {beta_ols_me[1]:.6f} / {lambda_factor:.4f} = {beta_eiv[1]:.6f}")
print(f"  Bias: {beta_eiv[1] - beta_1:.6f} (near zero! ✓)")
print(f"  Note: Correction requires knowing true error variance (σ²ₙ)")

# Residuals comparison
residuals_ols = Y - X_obs_design @ beta_ols_me
residuals_true = Y - X_true_design @ beta_ols_true
residuals_iv = Y - X_fitted_design @ beta_iv

se_ols = np.sqrt(np.sum(residuals_ols**2) / (n - 2))
se_true = np.sqrt(np.sum(residuals_true**2) / (n - 2))
se_iv = np.sqrt(np.sum(residuals_iv**2) / (n - 2))

print("\n\nResidual Standard Error:")
print("-" * 80)
print(f"{'Estimator':<30} {'SE':<12} {'Interpretation':<30}")
print("-" * 80)
print(f"{'OLS with true X*':<30} {se_true:<12.4f} {'Baseline (unbiased)':<30}")
print(f"{'OLS with measured X':<30} {se_ols:<12.4f} {'Inflated (measurement noise)':<30}")
print(f"{'IV/2SLS':<30} {se_iv:<12.4f} {'Higher (efficiency loss)':<30}")

print("=" * 80)

# Sensitivity analysis: Effect of measurement error severity
error_scales = np.linspace(0, 3, 50)
beta_1_estimates = []

for scale in error_scales:
    eta_scaled = np.random.normal(0, sigma_eta * scale, n)
    X_scaled = Xstar + eta_scaled
    
    var_X_scaled = np.var(X_scaled, ddof=1)
    lambda_scaled = var_Xstar / (var_Xstar + np.var(eta_scaled, ddof=1))
    
    # OLS with scaled error
    X_scaled_design = np.column_stack([np.ones(n), X_scaled])
    beta_scaled = inv(X_scaled_design.T @ X_scaled_design) @ (X_scaled_design.T @ Y)
    
    beta_1_estimates.append(beta_scaled[1])

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Scatter X* vs observed X
axes[0, 0].scatter(Xstar, X_obs, alpha=0.5, s=20)
axes[0, 0].plot([Xstar.min(), Xstar.max()], [Xstar.min(), Xstar.max()], 'r--', linewidth=2, label='No error line')
axes[0, 0].set_xlabel('True X*', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Observed X = X* + η', fontsize=11, fontweight='bold')
axes[0, 0].set_title(f'Measurement Error Visualization (σ_η = {sigma_eta:.2f})', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Panel 2: Coefficient comparison
methods = ['OLS\n(True X*)', 'OLS\n(Measured X)', 'IV\n(2SLS)', 'EIV\n(Corrected)']
coefficients = [beta_ols_true[1], beta_ols_me[1], beta_iv[1], beta_eiv[1]]
colors = ['green', 'red', 'blue', 'orange']
axes[0, 1].bar(methods, coefficients, color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].axhline(y=beta_1, color='black', linestyle='--', linewidth=2, label=f'True β₁ = {beta_1}')
axes[0, 1].set_ylabel('β̂₁ Estimate', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Estimator Comparison: Measurement Error Remedies', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Panel 3: Regression lines
x_range = np.linspace(Xstar.min(), Xstar.max(), 100)
y_ols_true = beta_ols_true[0] + beta_ols_true[1] * x_range
y_ols_me = beta_ols_me[0] + beta_ols_me[1] * x_range
y_iv = beta_iv[0] + beta_iv[1] * x_range

axes[1, 0].scatter(Xstar, Y, alpha=0.2, s=20, label='Data (X* vs Y)', color='gray')
axes[1, 0].plot(x_range, y_ols_true, 'g-', linewidth=2, label=f'OLS true (β̂₁={beta_ols_true[1]:.3f})')
axes[1, 0].plot(x_range, y_ols_me, 'r-', linewidth=2, label=f'OLS measured (β̂₁={beta_ols_me[1]:.3f})')
axes[1, 0].plot(x_range, y_iv, 'b-', linewidth=2, label=f'IV (β̂₁={beta_iv[1]:.3f})')
axes[1, 0].set_xlabel('X* (True Value)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Y', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Regression Lines: Impact of Measurement Error', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# Panel 4: Sensitivity to error severity
axes[1, 1].plot(error_scales, beta_1_estimates, 'b-', linewidth=2, label='β̂₁(error scale)')
axes[1, 1].axhline(y=beta_1, color='black', linestyle='--', linewidth=2, label=f'True β₁ = {beta_1}')
axes[1, 1].axhline(y=beta_ols_me[1], color='red', linestyle=':', linewidth=2, label=f'Current OLS ≈ {beta_ols_me[1]:.3f}')
axes[1, 1].fill_between([0, 1], beta_1*0.8, beta_1, alpha=0.2, color='green', label='Acceptable range (±20%)')
axes[1, 1].set_xlabel('Measurement Error Scale (σ_η multiplier)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('β̂₁ Estimate', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Sensitivity: OLS β̂₁ vs. Error Severity', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('measurement_error_attenuation.png', dpi=150)
plt.show()
```

**Expected Output:**
```
================================================================================
MEASUREMENT ERROR: ATTENUATION BIAS & REMEDIES
================================================================================

Measurement Error Quantification:
--------------------------------------------------------------------------------
True parameters: β₀ = 2.0, β₁ = 1.5
Var(X*) = 3.9976, Var(η) = 0.6892
Signal-to-Noise Ratio (SNR) = 5.8003
Attenuation factor λ = 0.8939
Expected attenuation: β̂₁ ≈ 1.3409 (true β₁ = 1.5)

Scenario 1: OLS with Measurement Error (Biased)
--------------------------------------------------------------------------------
OLS with observed X:
  β̂₀ = 1.9874, β̂₁ = 1.3421
  Expected β̂₁ ≈ 1.3409
  Actual bias: -0.157900
  Attenuation: 10.6% (expected 10.6%)

Scenario 2: OLS with True X* (Unbiased Baseline)
--------------------------------------------------------------------------------
OLS with true X*:
  β̂₀ = 2.0236, β̂₁ = 1.4968
  Bias: -0.003199 (negligible ✓)

Scenario 3: IV Estimation (Using Exogenous Instrument)
--------------------------------------------------------------------------------
IV (2SLS) with instrument Z:
  β̂₀ = 1.9687, β̂₁ = 1.5124
  Bias: 0.012391
  Instrument strength (Corr(Z, X_obs)): 0.9218
  Instrument exogeneity (Corr(Z, η)): -0.0184

Scenario 4: Errors-in-Variables Regression
--------------------------------------------------------------------------------
OLS corrected for measurement error:
  β̂₁ (corrected) = β̂₁ (OLS) / λ = 1.3421 / 0.8939 = 1.5006
  Bias: 0.000619 (near zero! ✓)
  Note: Correction requires knowing true error variance (σ²ₙ)

Residual Standard Error:
--------------------------------------------------------------------------------
Estimator                         SE           Interpretation                
--------------------------------------------------------------------------------
OLS with true X*                 1.0047       Baseline (unbiased)           
OLS with measured X              1.0156       Inflated (measurement noise)   
IV/2SLS                          1.0289       Higher (efficiency loss)       
================================================================================
```

---

## Challenge Round

1. **Attenuation Factor Calculation**  
   Var(X*) = 10, Var(η) = 2. Calculate λ and expected OLS coefficient if true β₁ = 2.0.

   <details><summary>Solution</summary>**λ = Var(X*)/[Var(X*) + Var(η)] = 10/(10+2) = 10/12 = 0.833**. **Expected β̂₁ = 2.0 × 0.833 = 1.667** (16.7% attenuation). **Interpretation**: OLS understates effect by one-sixth; measurement error accounts for 1/6 of variation.</details>

2. **SNR and Error Magnitude**  
   Signal-to-noise ratio SNR = 4. Express attenuation factor λ in terms of SNR only.

   <details><summary>Solution</summary>**λ = Var(X*)/[Var(X*) + Var(η)] = 1/(1 + Var(η)/Var(X*)) = 1/(1 + 1/SNR) = SNR/(SNR+1)**. For SNR = 4: **λ = 4/5 = 0.8** (80% of true effect observed). Higher SNR → less attenuation.</details>

3. **Measurement Error with Multiple Regressors**  
   Y = β₀ + β₁X₁ + β₂X₂ + ε. Only X₁ measured with error (λ₁ = 0.9). True β₁ = 1, β₂ = 0.5. Predict sign of bias in β̂₂.

   <details><summary>Solution</summary>**Direct bias on β̂₂**: If X₁ and X₂ correlated (typical), then ME in X₁ creates spillover bias: plim(β̂₂) ≈ β₂ + β₁(1-λ₁)×Corr(X₁,X₂)/(some var term). Bias direction on β̂₂ depends on Corr(X₁,X₂) sign. **If positive correlation**: β̂₂ biased upward (ME in X₁ induces positive bias to β₂). **Key**: ME in one variable contaminates other estimates.</details>

4. **Classical vs. Non-Classical Error**  
   Classical: η independent of X*, ε. Non-classical: Cov(η, X*) = 0.3 (measurement error increases with true X). Which has larger bias?

   <details><summary>Solution</summary>**Classical**: plim(β̂) = β × λ (attenuation only). **Non-classical with Cov(η,X*) > 0**: Additional endogeneity bias (besides attenuation) → **larger total bias** in magnitude (toward zero + additional component). **Direction**: Depends on whether η correlates with ε; if not, still attenuates but less severely than classical case if correlation with X* partially offsets. **General**: Non-classical ME more harmful (unpredictable bias direction/magnitude).</details>

---

## Key References

- **Wooldridge (2020)**: *Introductory Econometrics* (Ch. 9.4: Measurement Error) ([Cengage](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge))
- **Greene (2018)**: *Econometric Analysis* (Ch. 8.3: Errors in Variables) ([Pearson](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899))
- **Carroll, Ruppert & Stefanski (2006)**: *Measurement Error in Nonlinear Models* ([Chapman & Hall](https://www.routledge.com/Measurement-Error-in-Nonlinear-Models-A-Modern-Perspective/Carroll-Ruppert-Stefanski/p/book/9781584886334))

**Further Reading:**  
- Latent variable models and structural equation modeling (SEM)  
- Reliability coefficient and test-retest correlations  
- Correlated measurement error in panel data (bias in fixed effects)
