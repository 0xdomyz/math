# Specification Tests

## 1. Concept Skeleton
**Definition:** Diagnostic procedures testing whether econometric model functional form, distributional assumptions, or variable inclusion satisfy data requirements  
**Purpose:** Detect misspecification (omitted nonlinearities, wrong functional form, omitted variables), ensure model validity before inference  
**Prerequisites:** OLS regression, hypothesis testing, F-tests, asymptotic theory

## 2. Comparative Framing
| Test | RESET | Linktest | Hausman | Davidson-MacKinnon |
|------|-------|----------|---------|-------------------|
| **Null Hypothesis** | Correct functional form | Linear specification adequate | Random effects valid | Model A correctly specified |
| **Test Statistic** | F-test on powers of ŷ | t-test on ŷ² | (β̂₁-β̂₂)'V⁻¹(β̂₁-β̂₂) ~ χ² | t-test on Model B fitted values |
| **Rejection** | Nonlinearity present | Misspecification detected | Use fixed effects | Model A misspecified |
| **Computation** | Fast (auxiliary regression) | Fast (one regression) | Moderate (two estimators) | Fast (encompassing test) |

## 3. Examples + Counterexamples

**Simple Example:**  
RESET test: Wage on Education linear model rejected (p=0.003) → add Education² term, RESET p=0.42 (pass)

**Failure Case:**  
Linktest on logged dependent variable: May falsely reject due to heteroscedasticity, not misspecification → use robust standard errors

**Edge Case:**  
Multiple specification issues: RESET detects nonlinearity + omitted variable simultaneously → cannot distinguish root cause, need theory-driven tests

## 4. Layer Breakdown
```
Specification Testing Framework:
├─ Functional Form Tests:
│   ├─ RESET (Ramsey 1969): Add powers of ŷ to regression
│   │   ├─ Step 1: Estimate Y = Xβ + ε, obtain ŷ
│   │   ├─ Step 2: Auxiliary regression Y = Xβ + γ₂ŷ² + γ₃ŷ³ + ... + u
│   │   ├─ Test: H₀: γ₂=γ₃=...=0 via F-test
│   │   └─ Rejection: Nonlinearity or omitted variables present
│   ├─ Linktest: Predict using ŷ and ŷ²
│   │   ├─ Auxiliary: Y = α + β₁ŷ + β₂ŷ² + ε
│   │   ├─ Expectation under H₀: β₁≈1, β₂≈0
│   │   └─ Rejection if β₂ significantly ≠ 0
│   └─ Box-Cox transformation: Test λ in Y^(λ) via MLE
├─ Variable Omission Tests:
│   ├─ Wald test: Test restrictions Rβ=r jointly
│   ├─ LM test (Lagrange Multiplier): Score test under H₀
│   └─ LR test (Likelihood Ratio): -2log(L₀/L₁) ~ χ²
├─ Encompassing Tests:
│   ├─ Davidson-MacKinnon J-test: Nest competing non-nested models
│   │   ├─ Model A: Y = X₁β₁ + ε₁
│   │   ├─ Model B: Y = X₂β₂ + ε₂
│   │   ├─ Test A: Y = X₁β₁ + γŷ_B + u (H₀: γ=0)
│   │   └─ Test B: Y = X₂β₂ + δŷ_A + v (H₀: δ=0)
│   └─ Cox test: Variance-adjusted comparison of likelihoods
└─ Exogeneity Tests:
    ├─ Durbin-Wu-Hausman: Test endogeneity of regressors
    ├─ Step 1: Regress suspect X on instruments Z, get residuals v̂
    ├─ Step 2: Include v̂ in main regression Y = Xβ + γv̂ + ε
    └─ H₀: γ=0 (exogeneity); rejection → endogeneity, use IV
```

**Interaction:** Misspecification → biased β̂, invalid inference → specification tests diagnose → respecify model → retest

## 5. Mini-Project
Implement specification tests to diagnose model misspecification:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.tools.eval_measures import rmse

# Generate data with nonlinear true relationship
np.random.seed(42)
n = 500

# True model: Y = β₀ + β₁X + β₂X² + β₃Z + ε (quadratic in X)
X = np.random.uniform(0, 10, n)
Z = np.random.normal(5, 2, n)
epsilon = np.random.normal(0, 2, n)

# True parameters
beta_0, beta_1, beta_2, beta_3 = 5, 2, -0.15, 1.5

# Generate Y with nonlinearity
Y = beta_0 + beta_1*X + beta_2*X**2 + beta_3*Z + epsilon

# Create DataFrame
data = pd.DataFrame({'Y': Y, 'X': X, 'Z': Z})

# =====================================
# RESET Test Implementation
# =====================================
print("="*70)
print("RAMSEY RESET TEST")
print("="*70)

def reset_test(y, X, powers=[2, 3], add_intercept=True):
    """
    Ramsey RESET test for functional form misspecification.
    
    H₀: E[Y|X] correctly specified (linear)
    H₁: Nonlinear terms improve fit
    """
    if add_intercept:
        X = add_constant(X)
    
    # Step 1: Estimate original model
    model_original = OLS(y, X).fit()
    y_hat = model_original.fittedvalues
    
    # Step 2: Create auxiliary regression with powers of ŷ
    X_augmented = X.copy()
    for p in powers:
        X_augmented = np.column_stack([X_augmented, y_hat**p])
    
    model_augmented = OLS(y, X_augmented).fit()
    
    # F-test: Compare restricted (original) vs unrestricted (augmented)
    # H₀: coefficients on ŷ², ŷ³, ... are all zero
    RSS_r = model_original.ssr  # Restricted sum of squared residuals
    RSS_u = model_augmented.ssr  # Unrestricted
    
    q = len(powers)  # Number of restrictions
    n = len(y)
    k = X.shape[1]  # Parameters in restricted model
    
    F_stat = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k - q))
    p_value = 1 - stats.f.cdf(F_stat, q, n - k - q)
    
    return {
        'F_statistic': F_stat,
        'p_value': p_value,
        'df_numerator': q,
        'df_denominator': n - k - q,
        'RSS_restricted': RSS_r,
        'RSS_unrestricted': RSS_u,
        'decision': 'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'
    }

# Test 1: Misspecified linear model (omits X²)
X_linear = data[['X', 'Z']].values
result_misspec = reset_test(data['Y'], X_linear, powers=[2, 3])

print("\n1. LINEAR MODEL (MISSPECIFIED): Y ~ X + Z")
print(f"   F-statistic: {result_misspec['F_statistic']:.4f}")
print(f"   p-value: {result_misspec['p_value']:.6f}")
print(f"   Decision: {result_misspec['decision']}")
print(f"   → Indicates functional form misspecification (expected, true model quadratic)")

# Test 2: Correctly specified quadratic model
X_quad = np.column_stack([data['X'], data['X']**2, data['Z']])
result_correct = reset_test(data['Y'], X_quad, powers=[2, 3])

print("\n2. QUADRATIC MODEL (CORRECT SPECIFICATION): Y ~ X + X² + Z")
print(f"   F-statistic: {result_correct['F_statistic']:.4f}")
print(f"   p-value: {result_correct['p_value']:.6f}")
print(f"   Decision: {result_correct['decision']}")
print(f"   → Passes RESET (correct functional form)")

# =====================================
# Linktest Implementation
# =====================================
print("\n" + "="*70)
print("LINKTEST")
print("="*70)

def linktest(y, X, add_intercept=True):
    """
    Linktest for model specification.
    
    Auxiliary regression: Y = α + β₁ŷ + β₂ŷ² + ε
    H₀: β₁ = 1 and β₂ = 0 (correct specification)
    """
    if add_intercept:
        X = add_constant(X)
    
    # Estimate original model
    model = OLS(y, X).fit()
    y_hat = model.fittedvalues
    
    # Auxiliary regression: Y on ŷ and ŷ²
    X_link = add_constant(np.column_stack([y_hat, y_hat**2]))
    model_link = OLS(y, X_link).fit()
    
    # Test H₀: coefficient on ŷ² equals zero
    beta_2 = model_link.params[2]
    se_2 = model_link.bse[2]
    t_stat = beta_2 / se_2
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), model_link.df_resid))
    
    return {
        'beta_yhat': model_link.params[1],
        'beta_yhat2': beta_2,
        'se_yhat2': se_2,
        't_statistic': t_stat,
        'p_value': p_value,
        'decision': 'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'
    }

# Linktest on misspecified model
result_link_mis = linktest(data['Y'], X_linear)

print("\n1. LINKTEST ON LINEAR MODEL (MISSPECIFIED)")
print(f"   Coefficient on ŷ: {result_link_mis['beta_yhat']:.4f}")
print(f"   Coefficient on ŷ²: {result_link_mis['beta_yhat2']:.4f} (SE: {result_link_mis['se_yhat2']:.4f})")
print(f"   t-statistic: {result_link_mis['t_statistic']:.4f}")
print(f"   p-value: {result_link_mis['p_value']:.6f}")
print(f"   Decision: {result_link_mis['decision']}")
print(f"   → ŷ² significant indicates misspecification")

# Linktest on correctly specified model
result_link_cor = linktest(data['Y'], X_quad)

print("\n2. LINKTEST ON QUADRATIC MODEL (CORRECT)")
print(f"   Coefficient on ŷ: {result_link_cor['beta_yhat']:.4f}")
print(f"   Coefficient on ŷ²: {result_link_cor['beta_yhat2']:.4f} (SE: {result_link_cor['se_yhat2']:.4f})")
print(f"   t-statistic: {result_link_cor['t_statistic']:.4f}")
print(f"   p-value: {result_link_cor['p_value']:.6f}")
print(f"   Decision: {result_link_cor['decision']}")
print(f"   → ŷ² insignificant confirms correct specification")

# =====================================
# Davidson-MacKinnon J-Test (Non-nested models)
# =====================================
print("\n" + "="*70)
print("DAVIDSON-MACKINNON J-TEST (Non-nested Models)")
print("="*70)

def davidson_mackinnon_test(y, X_A, X_B, add_intercept=True):
    """
    J-test for non-nested models.
    
    Model A: Y = X_A β_A + ε_A
    Model B: Y = X_B β_B + ε_B
    
    Test A encompasses B: Y = X_A β_A + γŷ_B + u (H₀: γ=0)
    Test B encompasses A: Y = X_B β_B + δŷ_A + v (H₀: δ=0)
    """
    if add_intercept:
        X_A = add_constant(X_A)
        X_B = add_constant(X_B)
    
    # Estimate both models
    model_A = OLS(y, X_A).fit()
    model_B = OLS(y, X_B).fit()
    
    y_hat_A = model_A.fittedvalues
    y_hat_B = model_B.fittedvalues
    
    # Test: Does Model A encompass Model B?
    X_A_encompass = np.column_stack([X_A, y_hat_B])
    model_A_encompass = OLS(y, X_A_encompass).fit()
    
    gamma = model_A_encompass.params[-1]
    se_gamma = model_A_encompass.bse[-1]
    t_A = gamma / se_gamma
    p_A = 2 * (1 - stats.t.cdf(np.abs(t_A), model_A_encompass.df_resid))
    
    # Test: Does Model B encompass Model A?
    X_B_encompass = np.column_stack([X_B, y_hat_A])
    model_B_encompass = OLS(y, X_B_encompass).fit()
    
    delta = model_B_encompass.params[-1]
    se_delta = model_B_encompass.bse[-1]
    t_B = delta / se_delta
    p_B = 2 * (1 - stats.t.cdf(np.abs(t_B), model_B_encompass.df_resid))
    
    return {
        'model_A_encompass': {
            'gamma': gamma, 'se': se_gamma, 't_stat': t_A, 'p_value': p_A,
            'decision': 'Model A does NOT encompass B' if p_A < 0.05 else 'Model A encompasses B'
        },
        'model_B_encompass': {
            'delta': delta, 'se': se_delta, 't_stat': t_B, 'p_value': p_B,
            'decision': 'Model B does NOT encompass A' if p_B < 0.05 else 'Model B encompasses A'
        }
    }

# Competing models: Linear vs Log-linear
X_model_A = data[['X', 'Z']].values  # Linear: Y ~ X + Z
X_model_B = np.column_stack([np.log(data['X'] + 1), data['Z']])  # Log: Y ~ log(X) + Z

result_jtest = davidson_mackinnon_test(data['Y'], X_model_A, X_model_B)

print("\nModel A: Y ~ X + Z (linear)")
print("Model B: Y ~ log(X+1) + Z (log-linear)")
print("\nTest 1: Can Model A encompass Model B?")
print(f"   Coefficient on ŷ_B: {result_jtest['model_A_encompass']['gamma']:.4f}")
print(f"   t-statistic: {result_jtest['model_A_encompass']['t_stat']:.4f}")
print(f"   p-value: {result_jtest['model_A_encompass']['p_value']:.6f}")
print(f"   {result_jtest['model_A_encompass']['decision']}")

print("\nTest 2: Can Model B encompass Model A?")
print(f"   Coefficient on ŷ_A: {result_jtest['model_B_encompass']['delta']:.4f}")
print(f"   t-statistic: {result_jtest['model_B_encompass']['t_stat']:.4f}")
print(f"   p-value: {result_jtest['model_B_encompass']['p_value']:.6f}")
print(f"   {result_jtest['model_B_encompass']['decision']}")

print("\n   → If both rejected: Neither model adequate (true model is quadratic)")

# =====================================
# Durbin-Wu-Hausman Endogeneity Test
# =====================================
print("\n" + "="*70)
print("DURBIN-WU-HAUSMAN ENDOGENEITY TEST")
print("="*70)

# Generate data with endogeneity
np.random.seed(123)
n = 300

# Instrumental variable
Z_instrument = np.random.normal(0, 1, n)

# Endogenous regressor: X correlated with error
u = np.random.normal(0, 1, n)  # Structural error
X_endog = 2 + 0.7*Z_instrument + 0.5*u + np.random.normal(0, 0.5, n)  # X correlated with u

# Outcome: Y depends on X and u (endogeneity)
Y_endog = 3 + 1.5*X_endog + u

data_endog = pd.DataFrame({'Y': Y_endog, 'X': X_endog, 'Z': Z_instrument})

def durbin_wu_hausman_test(y, X_endog, Z_instrument, add_intercept=True):
    """
    Test for endogeneity of regressor X.
    
    H₀: X exogenous (OLS consistent)
    H₁: X endogenous (need IV)
    
    Procedure:
    1. First stage: X = Z'π + v
    2. Auxiliary regression: Y = Xβ + γv̂ + ε
    3. Test H₀: γ=0
    """
    if add_intercept:
        Z_instrument = add_constant(Z_instrument if Z_instrument.ndim == 2 else Z_instrument.reshape(-1, 1))
        X_endog = X_endog.reshape(-1, 1) if X_endog.ndim == 1 else X_endog
    
    # First stage: Regress X on instruments
    first_stage = OLS(X_endog, Z_instrument).fit()
    v_hat = first_stage.resid  # Residuals
    
    # Auxiliary regression: Include v̂ in main equation
    X_augmented = add_constant(np.column_stack([X_endog, v_hat]))
    model_aux = OLS(y, X_augmented).fit()
    
    # Test coefficient on v̂
    gamma = model_aux.params[-1]
    se_gamma = model_aux.bse[-1]
    t_stat = gamma / se_gamma
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), model_aux.df_resid))
    
    # F-stat from first stage (instrument relevance)
    F_first_stage = first_stage.fstat
    
    return {
        'gamma': gamma,
        'se_gamma': se_gamma,
        't_statistic': t_stat,
        'p_value': p_value,
        'decision': 'Reject H₀: X is endogenous' if p_value < 0.05 else 'Fail to reject: X likely exogenous',
        'first_stage_F': F_first_stage,
        'instrument_relevance': 'Strong' if F_first_stage > 10 else 'Weak'
    }

result_dwh = durbin_wu_hausman_test(
    data_endog['Y'], 
    data_endog['X'].values, 
    data_endog['Z'].values
)

print("\nTesting endogeneity of X (true model: X endogenous)")
print(f"   Coefficient on v̂: {result_dwh['gamma']:.4f} (SE: {result_dwh['se_gamma']:.4f})")
print(f"   t-statistic: {result_dwh['t_statistic']:.4f}")
print(f"   p-value: {result_dwh['p_value']:.6f}")
print(f"   Decision: {result_dwh['decision']}")
print(f"\n   First-stage F-statistic: {result_dwh['first_stage_F']:.2f}")
print(f"   Instrument relevance: {result_dwh['instrument_relevance']}")
print(f"   → Reject H₀ confirms endogeneity; use IV/2SLS estimation")

# =====================================
# Visualization: Specification Testing
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: True vs Fitted (Misspecified Linear)
model_linear = OLS(data['Y'], add_constant(X_linear)).fit()
axes[0, 0].scatter(data['X'], data['Y'], alpha=0.3, s=20, label='Data')
X_sort = np.sort(data['X'])
axes[0, 0].plot(X_sort, beta_0 + beta_1*X_sort + beta_2*X_sort**2 + beta_3*data['Z'].mean(), 
                'r-', linewidth=2, label='True (Quadratic)')
axes[0, 0].scatter(data['X'], model_linear.fittedvalues, alpha=0.3, s=20, 
                  color='orange', label='Linear Fit')
axes[0, 0].set_title(f'Misspecified Linear Model\nRESET p={result_misspec["p_value"]:.4f} (Reject)')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].legend()

# Plot 2: True vs Fitted (Correct Quadratic)
model_quad = OLS(data['Y'], add_constant(X_quad)).fit()
axes[0, 1].scatter(data['X'], data['Y'], alpha=0.3, s=20, label='Data')
axes[0, 1].plot(X_sort, beta_0 + beta_1*X_sort + beta_2*X_sort**2 + beta_3*data['Z'].mean(), 
                'r-', linewidth=2, label='True (Quadratic)')
axes[0, 1].scatter(data['X'], model_quad.fittedvalues, alpha=0.3, s=20, 
                  color='green', label='Quadratic Fit')
axes[0, 1].set_title(f'Correct Quadratic Model\nRESET p={result_correct["p_value"]:.4f} (Pass)')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].legend()

# Plot 3: Linktest Visualization
y_hat_linear = model_linear.fittedvalues
X_link_visual = add_constant(np.column_stack([y_hat_linear, y_hat_linear**2]))
model_link_visual = OLS(data['Y'], X_link_visual).fit()

axes[1, 0].scatter(y_hat_linear, data['Y'], alpha=0.3, s=20)
y_hat_range = np.linspace(y_hat_linear.min(), y_hat_linear.max(), 100)
y_link_pred = (model_link_visual.params[0] + 
               model_link_visual.params[1]*y_hat_range + 
               model_link_visual.params[2]*y_hat_range**2)
axes[1, 0].plot(y_hat_range, y_link_pred, 'r-', linewidth=2, 
                label=f'Y = α + {model_link_visual.params[1]:.2f}ŷ + {model_link_visual.params[2]:.2f}ŷ²')
axes[1, 0].plot(y_hat_range, y_hat_range, 'k--', alpha=0.5, label='45° line (perfect fit)')
axes[1, 0].set_title('Linktest: Y vs ŷ (Misspecified Model)')
axes[1, 0].set_xlabel('Predicted ŷ')
axes[1, 0].set_ylabel('Actual Y')
axes[1, 0].legend()

# Plot 4: RESET Test Power Simulation
np.random.seed(42)
n_sims = 200
reset_pvalues = []

for _ in range(n_sims):
    X_sim = np.random.uniform(0, 10, 300)
    Z_sim = np.random.normal(5, 2, 300)
    Y_sim = 5 + 2*X_sim - 0.15*X_sim**2 + 1.5*Z_sim + np.random.normal(0, 2, 300)
    
    X_linear_sim = np.column_stack([X_sim, Z_sim])
    result_sim = reset_test(Y_sim, X_linear_sim, powers=[2, 3])
    reset_pvalues.append(result_sim['p_value'])

axes[1, 1].hist(reset_pvalues, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(0.05, color='r', linestyle='--', linewidth=2, label='α=0.05')
power = np.mean(np.array(reset_pvalues) < 0.05)
axes[1, 1].set_title(f'RESET Test Power Simulation\nPower = {power:.2%} (200 simulations)')
axes[1, 1].set_xlabel('p-value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Specification tests diagnose model misspecification:")
print("• RESET: Detects nonlinearity via powers of ŷ (quadratic detected ✓)")
print("• Linktest: Tests linear vs nonlinear via ŷ² coefficient")
print("• J-Test: Compares non-nested models (neither encompassed → need quadratic)")
print("• DWH: Identifies endogeneity (instrumentation required ✓)")
print(f"\nSimulation demonstrates high power ({power:.1%}) to detect misspecification")
```

## 6. Challenge Round
Why might specification tests fail to detect misspecification?
- **Low power:** Small samples, weak nonlinearity → tests lack power to reject H₀
- **Multiple issues:** Simultaneous endogeneity + nonlinearity → tests conflate sources, unclear remediation
- **Wrong test:** RESET assumes polynomial nonlinearity, misses threshold/interaction effects
- **Overfitting risk:** Adding powers of ŷ in auxiliary regression may fit noise in small samples
- **Conditional heteroscedasticity:** Affects test size, requires robust standard errors in linktest/RESET

When to prioritize theory over tests: Economics suggests log transformation but RESET passes linear → prefer log for interpretability and economic reasoning.

## 7. Key References
- [Ramsey (1969) "Tests for Specification Errors in Classical Linear Least Squares Regression Analysis"](https://doi.org/10.1111/j.2517-6161.1969.tb00796.x) - Original RESET test
- [Davidson & MacKinnon (1981) "Several Tests for Model Specification"](https://doi.org/10.2307/1911522) - J-test for non-nested models
- [Hausman (1978) "Specification Tests in Econometrics"](https://doi.org/10.2307/1913827) - Exogeneity testing framework
- [Wooldridge (2010) Econometric Analysis of Cross Section and Panel Data, Ch. 5](https://mitpress.mit.edu/9780262232586/) - Comprehensive specification testing

---
**Status:** Essential diagnostic tool | **Complements:** Model Selection (Information Criteria), Residual Analysis, Hypothesis Testing
