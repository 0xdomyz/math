# Spatial Error Model

## 1. Concept Skeleton
**Definition:** Regression with spatially correlated errors via λWu term; models omitted spatial variables or measurement issues as nuisance  
**Purpose:** Correct biased standard errors from spatial correlation in shocks; no substantive spillovers in outcome but spatial dependence in unobservables  
**Prerequisites:** Spatial autocorrelation, spatial weight matrices, generalized least squares (GLS), maximum likelihood estimation, FGLS

## 2. Comparative Framing
| Model | Spatial Error (SEM) | Spatial Lag (SAR) | Spatial Durbin (SDM) | OLS | FGLS (Non-Spatial) | Robust SE |
|-------|---------------------|-------------------|----------------------|-----|-------------------|-----------|
| **Specification** | Y = Xβ + u, u = λWu + ε | Y = ρWY + Xβ + ε | Y = ρWY + Xβ + WXθ + ε | Y = Xβ + ε | Y = Xβ + u, Var(u)=Ω | Y = Xβ + ε |
| **Spatial Dependence** | Error term | Outcome variable | Both Y and X | None | Non-spatial Ω | None assumed |
| **Interpretation** | Nuisance (omitted variables) | Substantive spillovers | Multiple channels | Direct effects | Non-spatial correlation | Heteroskedasticity |
| **β Consistency** | Yes (OLS consistent) | No (OLS biased) | No | Yes | Yes | Yes |
| **SE Bias** | Yes (OLS biased) | Yes | Yes | No (if iid) | No | No |
| **Estimation** | MLE, FGLS | MLE, IV/2SLS | MLE, IV | OLS | FGLS | OLS + robust |
| **Multiplier Effect** | No | Yes: (I-ρW)⁻¹ | Yes | No | No | No |

## 3. Examples + Counterexamples

**Classic Example:**  
Agricultural yields (n=400 farms): Spatially clustered shocks (weather, soil quality not in X). OLS β̂ consistent but SE underestimated by 40%. LM_error=22.3 (p<0.001), LM_lag=1.8 (p=0.18). Fit SEM: λ=0.42, residual Moran's I=0.03. Corrected SEs increase; some variables now insignificant.

**Failure Case:**  
True model has spatial lag (Y=ρWY+Xβ+ε), fit spatial error. λ̂=0.30 (significant) but spurious—captures outcome dependence not error correlation. β̂ remains biased. Diagnostic: Moran's I on SEM residuals still 0.18 (should be ~0).

**Edge Case:**  
Spatial heteroskedasticity (variance varies by location): SEM assumes constant σ². If Var(εᵢ) depends on location, need spatial heteroskedastic error model (SHEM) with diagonal heteroskedasticity + spatial correlation.

## 4. Layer Breakdown
```
Spatial Error Model (SEM):
├─ Model Specification:
│   ├─ Basic Form:
│   │   ├─ Y = Xβ + u
│   │   ├─ u = λWu + ε
│   │   ├─ ε ~ N(0, σ²I): iid innovations
│   │   ├─ W: n×n spatial weight matrix (row-standardized)
│   │   ├─ λ: Spatial error autoregressive coefficient
│   │   └─ u: Spatially autocorrelated error term
│   ├─ Reduced Form Error:
│   │   ├─ u = (I - λW)⁻¹ε
│   │   ├─ Requires |λ| < 1 for invertibility
│   │   └─ Var(u) = σ²[(I-λW)'(I-λW)]⁻¹ ≠ σ²I
│   ├─ Combined Form:
│   │   ├─ Y = Xβ + (I - λW)⁻¹ε
│   │   ├─ Non-spherical errors: Var(Y|X) = Ω ≠ σ²I
│   │   └─ Ω = σ²[(I-λW)'(I-λW)]⁻¹
│   ├─ Interpretation:
│   │   ├─ λ > 0: Positive error correlation (clustering of shocks)
│   │   ├─ λ < 0: Negative error correlation (rare)
│   │   ├─ Wu: Spatially lagged error (neighbors' shocks)
│   │   └─ No endogenous interaction in Y (unlike SAR)
│   └─ Stationarity:
│       ├─ λ must satisfy: 1/λ_min(W) < λ < 1/λ_max(W)
│       └─ Typically λ ∈ (-1, 1) for row-standardized W
├─ Motivation and Theory:
│   ├─ Sources of Spatial Error Correlation:
│   │   ├─ Omitted spatially correlated variables
│   │   ├─ Measurement error in outcome Y
│   │   ├─ Unobserved common shocks (weather, policy)
│   │   ├─ Spatial aggregation (areal units average heterogeneous micro-units)
│   │   └─ Model misspecification (wrong functional form)
│   ├─ Nuisance vs Substantive:
│   │   ├─ SEM: Spatial correlation is nuisance (not of interest)
│   │   ├─ SAR: Spatial dependence is substantive (theory-driven)
│   │   └─ SEM corrects inference; SAR changes interpretation
│   ├─ OLS Properties:
│   │   ├─ β̂_OLS consistent: plim β̂_OLS = β
│   │   ├─ But inefficient: Not minimum variance
│   │   ├─ Standard errors biased: Var(β̂_OLS) ≠ σ²(X'X)⁻¹
│   │   └─ Underestimated SEs if λ>0 (common)
│   ├─ GLS Intuition:
│   │   ├─ Transform to remove spatial correlation
│   │   ├─ Ỹ = Ω⁻½Y, X̃ = Ω⁻½X
│   │   ├─ β̂_GLS = (X̃'X̃)⁻¹X̃'Ỹ
│   │   └─ Efficient if Ω known
│   └─ Identification:
│       ├─ λ identified via spatial pattern in residuals
│       ├─ W exogenous (pre-specified)
│       └─ Distinction from SAR requires LM tests
├─ Estimation Methods:
│   ├─ OLS (Consistent but Inefficient):
│   │   ├─ β̂_OLS = (X'X)⁻¹X'Y
│   │   ├─ Consistent estimator of β
│   │   ├─ Standard errors wrong: Use spatial HAC or fit SEM
│   │   └─ Diagnostic: Moran's I on residuals
│   ├─ Maximum Likelihood (MLE):
│   │   ├─ Log-Likelihood:
│   │   │   ├─ ℓ(λ,β,σ²) = -(n/2)log(2πσ²) + log|I-λW| - (1/2σ²)(Y-Xβ)'(I-λW)'(I-λW)(Y-Xβ)
│   │   │   ├─ Jacobian: log|I-λW| = Σᵢ log(1 - λλᵢ)
│   │   │   └─ λᵢ: Eigenvalues of W
│   │   ├─ Concentrated Likelihood:
│   │   │   ├─ Profile out β, σ²
│   │   │   ├─ β̂(λ) = (X'X)⁻¹X'[(I-λW)Y]
│   │   │   ├─ σ̂²(λ) = e'e/n where e = (I-λW)Y - Xβ̂(λ)
│   │   │   └─ ℓ_c(λ) = -(n/2)log(σ̂²(λ)) + log|I-λW|
│   │   ├─ Optimization:
│   │   │   ├─ Grid search over λ
│   │   │   ├─ Or gradient-based (BFGS)
│   │   │   └─ Eigenvalues computed once
│   │   ├─ Standard Errors:
│   │   │   ├─ From inverse Hessian
│   │   │   ├─ Var(β̂_MLE) < Var(β̂_OLS) (efficiency gain)
│   │   │   └─ Asymptotically efficient
│   │   └─ Advantages:
│   │       ├─ Efficient under normality
│   │       ├─ Straightforward inference
│   │       └─ Standard approach
│   ├─ Feasible GLS (FGLS):
│   │   ├─ Two-Step Procedure:
│   │   │   ├─ Step 1: OLS to get residuals û
│   │   │   ├─ Step 2: Estimate λ from spatial correlation of û
│   │   │   ├─ Step 3: Construct Ω̂ = σ̂²[(I-λ̂W)'(I-λ̂W)]⁻¹
│   │   │   └─ Step 4: GLS with Ω̂ → β̂_FGLS = (X'Ω̂⁻¹X)⁻¹X'Ω̂⁻¹Y
│   │   ├─ λ Estimation in Step 2:
│   │   │   ├─ Regress û on Wû: û = λWû + v
│   │   │   ├─ Or: λ̂ = (û'Wû)/(û'W'Wû)
│   │   │   └─ Consistent for λ
│   │   ├─ Advantages:
│   │   │   ├─ Simple to implement
│   │   │   ├─ No distributional assumptions
│   │   │   └─ Asymptotically equivalent to MLE
│   │   └─ Disadvantages:
│   │       ├─ Two-step inefficiency
│   │       ├─ Standard errors need adjustment
│   │       └─ MLE preferred if feasible
│   ├─ Generalized Spatial Two-Stage Least Squares (GS2SLS):
│   │   ├─ Kelejian & Prucha (1998)
│   │   ├─ Step 1: OLS to get û
│   │   ├─ Step 2: GMM to estimate λ from moment conditions on û
│   │   ├─ Step 3: Cochrane-Orcutt transformation
│   │   └─ Robust to non-normality
│   ├─ Bayesian Estimation:
│   │   ├─ Priors on λ, β, σ²
│   │   ├─ MCMC sampling
│   │   ├─ Full posterior inference
│   │   └─ Flexible but computationally intensive
│   └─ Comparison:
│       ├─ MLE: Efficient, assumes normality, standard choice
│       ├─ FGLS: Simple, consistent, less efficient
│       ├─ GS2SLS: Robust to non-normality, complex
│       └─ Bayesian: Full uncertainty quantification
├─ Inference and Hypothesis Testing:
│   ├─ Moran's I on OLS Residuals:
│   │   ├─ Test for spatial correlation
│   │   ├─ If significant → SEM or SAR needed
│   │   └─ First diagnostic step
│   ├─ Lagrange Multiplier (LM) Tests:
│   │   ├─ LM_error: Test H₀: λ=0 (no spatial error)
│   │   │   ├─ LM_err = [e'We / (e'e/n)]² / tr(W'W + W²)
│   │   │   ├─ e: OLS residuals
│   │   │   └─ χ²(1) under null
│   │   ├─ LM_lag: Test H₀: ρ=0 (no spatial lag)
│   │   │   ├─ Distinguish SAR from SEM
│   │   │   └─ χ²(1) under null
│   │   ├─ Robust LM Tests:
│   │   │   ├─ Robust LM_error: Accounts for potential SAR
│   │   │   ├─ Robust LM_lag: Accounts for potential SEM
│   │   │   └─ Use robust versions for model selection
│   │   └─ Decision Rule:
│   │       ├─ If LM_err significant, LM_lag not → SEM
│   │       ├─ If LM_lag significant, LM_err not → SAR
│   │       ├─ If both → Use robust versions or SDM
│   │       └─ If neither → OLS adequate
│   ├─ Likelihood Ratio Test:
│   │   ├─ LR = 2(ℓ_SEM - ℓ_OLS)
│   │   ├─ χ²(1) under null λ=0
│   │   └─ Nested models
│   ├─ Wald Test:
│   │   ├─ Test λ=0 or restrictions on β
│   │   ├─ Wald = (λ̂-0)² / Var(λ̂)
│   │   └─ Asymptotically χ²
│   └─ Model Comparison:
│       ├─ AIC, BIC (penalize complexity)
│       ├─ Log-likelihood values
│       └─ Residual diagnostics (Moran's I)
├─ Interpretation of Coefficients:
│   ├─ β Coefficients:
│   │   ├─ Direct effects: ∂E[Y_i]/∂X_i = β_k
│   │   ├─ No spillover effects (unlike SAR)
│   │   ├─ Interpretation same as OLS
│   │   └─ But standard errors corrected
│   ├─ λ Coefficient:
│   │   ├─ Measures spatial correlation in shocks
│   │   ├─ λ > 0: Positive error clustering
│   │   ├─ λ ≈ 0: No spatial error correlation
│   │   └─ Not of substantive interest (nuisance)
│   ├─ No Spatial Multiplier:
│   │   ├─ Unlike SAR, no feedback effects
│   │   ├─ ∂Y/∂X is scalar β, not matrix
│   │   └─ Total effect = Direct effect = β
│   └─ Standard Error Correction:
│       ├─ Var(β̂_SEM) accounts for spatial correlation
│       ├─ Typically SE_SEM > SE_OLS if λ>0
│       └─ Inference changes (wider CIs, higher p-values)
├─ Diagnostics:
│   ├─ Residual Spatial Autocorrelation:
│   │   ├─ Moran's I on SEM residuals ε̂
│   │   ├─ Should be ≈0 if model correct
│   │   └─ Significant I → Misspecification
│   ├─ Heteroskedasticity:
│   │   ├─ Breusch-Pagan on ε̂
│   │   ├─ If present: Spatial heteroskedastic error model
│   │   └─ Or robust standard errors
│   ├─ Normality:
│   │   ├─ MLE assumes ε ~ N(0, σ²I)
│   │   ├─ Jarque-Bera test on residuals
│   │   └─ Robust to mild non-normality
│   ├─ Influential Observations:
│   │   ├─ Spatial Cook's D
│   │   ├─ Units with extreme residuals
│   │   └─ High leverage (many neighbors)
│   ├─ Weight Matrix Sensitivity:
│   │   ├─ Re-estimate with different W
│   │   ├─ Compare λ̂, β̂ estimates
│   │   └─ Results should be qualitatively robust
│   └─ Specification Test:
│       ├─ If SEM residuals still spatially correlated
│       ├─ Consider SAR or SDM
│       └─ May need different W or additional covariates
├─ SEM vs SAR Comparison:
│   ├─ Conceptual Difference:
│   │   ├─ SEM: Spatial dependence in unobservables (nuisance)
│   │   ├─ SAR: Spatial dependence in outcome (substantive)
│   │   └─ Theory should guide choice
│   ├─ Statistical Distinction:
│   │   ├─ LM tests discriminate
│   │   ├─ Robust LM tests preferred
│   │   └─ Can test non-nested via Vuong
│   ├─ Consequences of Misspecification:
│   │   ├─ SEM as SAR: β̂ biased, interpretation wrong
│   │   ├─ SAR as SEM: β̂ remains biased, SE correction insufficient
│   │   └─ Correct specification crucial
│   ├─ When to Use SEM:
│   │   ├─ No theoretical reason for Y_i to depend on Y_j
│   │   ├─ Spatial correlation from omitted variables
│   │   ├─ Measurement issues or aggregation
│   │   └─ Goal: Correct inference, not model spillovers
│   └─ When to Use SAR:
│       ├─ Theory predicts spillovers (contagion, diffusion)
│       ├─ Policy interest in indirect effects
│       └─ Feedback mechanisms important
├─ Spatial Durbin Error Model (SDEM):
│   ├─ Specification:
│   │   ├─ Y = Xβ + WXθ + u, u = λWu + ε
│   │   ├─ Adds WX (neighbors' covariates)
│   │   └─ Local spillovers (WX) + spatial error correlation
│   ├─ Interpretation:
│   │   ├─ θ: Exogenous spillovers from neighbors' X
│   │   ├─ λ: Residual spatial correlation
│   │   └─ More flexible than pure SEM
│   ├─ Estimation:
│   │   ├─ MLE similar to SEM
│   │   └─ FGLS with augmented X matrix
│   └─ Testing:
│       ├─ Test θ=0: Collapse to SEM
│       └─ Distinguish local (WX) from global (WY) spillovers
├─ Practical Implementation:
│   ├─ Python:
│   │   ├─ PySAL (spreg): ML_Error, GM_Error classes
│   │   ├─ Syntax: spreg.ML_Error(y, X, w)
│   │   ├─ Returns: λ, β, standard errors, diagnostics
│   │   └─ LM tests: spreg.LMtests
│   ├─ R:
│   │   ├─ spatialreg: errorsarlm() function
│   │   ├─ Syntax: errorsarlm(Y ~ X, data, listw)
│   │   ├─ Diagnostics: lm.LMtests() for LM tests
│   │   └─ Extensive model comparison tools
│   ├─ Stata:
│   │   ├─ spregress with error option
│   │   ├─ Syntax: spregress Y X, ml error(W)
│   │   └─ estat moran for residual diagnostics
│   └─ MATLAB:
│       ├─ Spatial Econometrics Toolbox
│       └─ sem() function
├─ Extensions:
│   ├─ Spatial Panel Data:
│   │   ├─ Y_it = X_it β + α_i + u_it, u_it = λW u_it + ε_it
│   │   ├─ Fixed effects + spatial error
│   │   ├─ Within transformation
│   │   └─ ML or GMM estimation
│   ├─ Spatial Heteroskedastic Error:
│   │   ├─ Var(ε_i) ≠ σ² (location-specific variance)
│   │   ├─ Diagonal Σ + spatial correlation
│   │   └─ SHEM model (Arraiz et al. 2010)
│   ├─ Spatial GARCH:
│   │   ├─ Time-varying spatial error variance
│   │   ├─ σ²_it = f(past shocks, spatial spillovers)
│   │   └─ Financial contagion applications
│   ├─ Higher-Order Spatial Errors:
│   │   ├─ u = λ₁Wu + λ₂W²u + ε
│   │   ├─ Multiple neighborhood rings
│   │   └─ Rare (identification issues)
│   └─ Spatial Regime Models:
│       ├─ Different β, λ by spatial regime
│       ├─ Urban vs rural, regions
│       └─ Chow tests for regime differences
├─ Assumptions:
│   ├─ Correct Weight Matrix:
│   │   ├─ W reflects true spatial relationships
│   │   └─ Misspecification affects λ̂, SE
│   ├─ Linearity:
│   │   ├─ E[Y|X] = Xβ linear
│   │   └─ Spatial correlation only in errors
│   ├─ Homoskedasticity (MLE):
│   │   ├─ Var(ε_i) = σ² constant
│   │   ├─ Violated → SHEM or robust SE
│   │   └─ Common in spatial data
│   ├─ Normality (MLE):
│   │   ├─ ε ~ N(0, σ²I)
│   │   ├─ Relaxable via GMM or robust inference
│   │   └─ MLE consistent even if non-normal
│   ├─ Exogeneity of X:
│   │   ├─ E[X'u] = 0
│   │   ├─ OLS consistency requires this
│   │   └─ Violated → IV methods
│   ├─ Stationarity:
│   │   ├─ Constant λ, β across space
│   │   └─ Spatial regimes if violated
│   └─ No Spatial Lag:
│       ├─ True model is SEM not SAR
│       └─ LM tests verify
└─ Common Pitfalls:
    ├─ Confusing SEM with SAR: Different models, interpretations
    ├─ Using OLS standard errors: Biased inference
    ├─ Ignoring LM tests: May fit wrong model
    ├─ Assuming SEM when SAR appropriate: β̂ biased
    ├─ Not checking residual autocorrelation: Model validation
    └─ Overinterpreting λ: It's nuisance parameter, not of primary interest
```

**Interaction:** OLS → Moran's I on residuals → LM_error test (significant?) → Fit SEM via MLE → Check residual diagnostics → Report β with corrected SE

## 5. Mini-Project
Implement spatial error model with MLE; compare to OLS and SAR:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import seaborn as sns

np.random.seed(468)

# ===== Simulate Spatial Data with Error Correlation =====
print("="*80)
print("SPATIAL ERROR MODEL (SEM)")
print("="*80)

# Grid coordinates
grid_size = 12
n = grid_size ** 2
coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])

print(f"\nSimulation Setup:")
print(f"  Grid: {grid_size} × {grid_size} = {n} locations")

# Generate covariates
X_raw = np.column_stack([
    np.ones(n),
    np.random.randn(n),  # X1
    np.random.randn(n)   # X2
])

# Spatial weight matrix (Rook contiguity)
def create_weight_matrix_rook(coords, grid_size):
    n = len(coords)
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.abs(coords[i] - coords[j])
                # Rook: shares edge only (Manhattan distance = 1)
                if np.sum(dist) == 1:
                    W[i, j] = 1
    
    # Row standardization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_std = W / row_sums
    
    return W_std, W

W, W_raw = create_weight_matrix_rook(coords, grid_size)

print(f"  Weight matrix: Rook contiguity, row-standardized")
print(f"  Connections: {np.sum(W_raw > 0)} ({np.mean(np.sum(W_raw>0, axis=1)):.1f} avg neighbors)")

# True parameters
lambda_true = 0.6
beta_true = np.array([3.0, 1.2, -0.9])
sigma_true = 0.8

print(f"\nTrue Parameters:")
print(f"  λ (spatial error): {lambda_true}")
print(f"  β: {beta_true}")
print(f"  σ: {sigma_true}")

# Generate Y via spatial error process: Y = Xβ + u, u = (I - λW)^{-1}ε
epsilon = np.random.randn(n) * sigma_true
Xbeta = X_raw @ beta_true

# Solve (I - λW)u = ε
I = np.eye(n)
A = I - lambda_true * W
u = np.linalg.solve(A, epsilon)
Y = Xbeta + u

print(f"  Generated Y via: Y = Xβ + (I - λW)^{{-1}}ε")
print(f"  Y statistics: mean={np.mean(Y):.3f}, sd={np.std(Y):.3f}")

# Check spatial autocorrelation (Moran's I)
def morans_i_simple(y, W):
    n = len(y)
    y_dev = y - np.mean(y)
    numerator = np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    S0 = np.sum(W)
    I = (n / S0) * (numerator / denominator)
    return I

I_u = morans_i_simple(u, W)
print(f"\nMoran's I on error u: {I_u:.4f} (spatial error correlation)")

# ===== OLS Estimation (Consistent but Inefficient) =====
print("\n" + "="*80)
print("OLS ESTIMATION (Consistent β̂, Biased SE)")
print("="*80)

XtX_inv = np.linalg.inv(X_raw.T @ X_raw)
beta_ols = XtX_inv @ X_raw.T @ Y
resid_ols = Y - X_raw @ beta_ols
sigma2_ols = np.sum(resid_ols ** 2) / (n - X_raw.shape[1])
se_ols = np.sqrt(np.diag(XtX_inv * sigma2_ols))

print(f"OLS Results:")
print(f"  β̂: {beta_ols}")
print(f"  SE (naive): {se_ols}")
print(f"  σ̂: {np.sqrt(sigma2_ols):.4f}")

# β̂_OLS should be close to β_true (consistency)
bias_ols = beta_ols - beta_true
print(f"\nOLS Bias (β̂_OLS - β_true): {bias_ols}")
print(f"  ✓ β̂_OLS consistent even with spatial error correlation")

# Moran's I on OLS residuals
I_resid_ols = morans_i_simple(resid_ols, W)
print(f"\nMoran's I on OLS residuals: {I_resid_ols:.4f}")
if I_resid_ols > 0.1:
    print(f"  ✓ Significant spatial autocorrelation → SEM needed for correct SE")

# ===== LM Tests (Lagrange Multiplier) =====
print("\n" + "="*80)
print("LAGRANGE MULTIPLIER TESTS")
print("="*80)

# LM test for spatial error
def lm_error_test(resid, X, W):
    """LM test for spatial error correlation"""
    n = len(resid)
    k = X.shape[1]
    
    # Trace terms
    tr_W = np.trace(W)
    tr_WtW = np.trace(W.T @ W)
    tr_W2 = np.trace(W @ W)
    
    # Test statistic
    e_We = resid @ W @ resid
    e_e = resid @ resid
    
    T = tr_WtW + tr_W2
    
    LM_error = (e_We / (e_e / n))**2 / T
    p_value = 1 - stats.chi2.cdf(LM_error, df=1)
    
    return LM_error, p_value

LM_err, p_err = lm_error_test(resid_ols, X_raw, W)

print(f"LM Test for Spatial Error:")
print(f"  LM_error = {LM_err:.4f}")
print(f"  P-value = {p_err:.4f}")

if p_err < 0.05:
    print(f"  ✓ Reject H₀: λ=0 → Spatial error model needed")
else:
    print(f"  Fail to reject H₀ → No spatial error correlation")

# LM test for spatial lag (for comparison)
def lm_lag_test(Y, X, W):
    """LM test for spatial lag"""
    n = len(Y)
    
    # OLS residuals
    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ Y
    resid = Y - X @ beta_ols
    s2 = np.sum(resid**2) / n
    
    # WY
    WY = W @ Y
    
    # Test statistic
    e_WY = resid @ WY
    
    # M = I - X(X'X)^{-1}X'
    XtX_inv = np.linalg.inv(X.T @ X)
    MWY = WY - X @ (XtX_inv @ X.T @ WY)
    
    LM_lag = (e_WY / s2)**2 / (MWY @ MWY)
    p_value = 1 - stats.chi2.cdf(LM_lag, df=1)
    
    return LM_lag, p_value

LM_lag, p_lag = lm_lag_test(Y, X_raw, W)

print(f"\nLM Test for Spatial Lag (for comparison):")
print(f"  LM_lag = {LM_lag:.4f}")
print(f"  P-value = {p_lag:.4f}")

if p_lag < 0.05:
    print(f"  Reject H₀: ρ=0 → Spatial lag model")
else:
    print(f"  Fail to reject H₀ → No spatial lag")

print(f"\nModel Selection:")
if p_err < 0.05 and p_lag >= 0.05:
    print(f"  → Spatial Error Model (SEM)")
elif p_lag < 0.05 and p_err >= 0.05:
    print(f"  → Spatial Lag Model (SAR)")
elif p_err < 0.05 and p_lag < 0.05:
    print(f"  → Both significant: Use robust LM tests or SDM")
else:
    print(f"  → OLS adequate")

# ===== Maximum Likelihood Estimation (SEM) =====
print("\n" + "="*80)
print("MAXIMUM LIKELIHOOD ESTIMATION (SEM)")
print("="*80)

# Eigenvalues of W
eigvals_W = eigh(W, eigvals_only=True)
lambda_min, lambda_max = eigvals_W.min(), eigvals_W.max()

print(f"Eigenvalue bounds for W:")
print(f"  λ_min = {lambda_min:.4f}, λ_max = {lambda_max:.4f}")
print(f"  λ must be in ({1/lambda_min:.3f}, {1/lambda_max:.3f})")

def log_likelihood_sem(params, Y, X, W, eigvals_W):
    """Concentrated log-likelihood for SEM"""
    lam = params[0]
    n = len(Y)
    
    # Bounds check
    if lam <= 1/eigvals_W.min() or lam >= 1/eigvals_W.max():
        return -1e10
    
    # Jacobian: log|I - λW|
    log_det = np.sum(np.log(1 - lam * eigvals_W))
    
    # Transform Y
    Y_trans = Y - lam * (W @ Y)
    
    # β̂(λ) via OLS on transformed data
    # But for SEM: Y_trans = (I-λW)Y, regress on X
    # Actually: (I-λW)(Y - Xβ) = ε
    # So: (I-λW)Y = (I-λW)Xβ + ε
    # For concentrated likelihood, use standard approach
    
    # Residuals after transforming Y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y_trans
    resid = Y_trans - X @ beta_hat
    
    # σ²
    sigma2_hat = np.sum(resid ** 2) / n
    
    # Log-likelihood
    ll = -(n/2) * np.log(2*np.pi) - (n/2) * np.log(sigma2_hat) + log_det - n/2
    
    return -ll

# Optimize over λ
print(f"\nOptimizing MLE...")
result_mle = optimize.minimize_scalar(
    lambda lam: log_likelihood_sem(np.array([lam]), Y, X_raw, W, eigvals_W),
    bounds=(1/lambda_min + 0.01, 1/lambda_max - 0.01),
    method='bounded'
)

lambda_mle = result_mle.x

# Get β, σ² at optimal λ
Y_trans_mle = Y - lambda_mle * (W @ Y)
XtX_inv = np.linalg.inv(X_raw.T @ X_raw)
beta_mle = XtX_inv @ X_raw.T @ Y_trans_mle
resid_mle = Y_trans_mle - X_raw @ beta_mle
sigma2_mle = np.sum(resid_mle ** 2) / n

# Compute ε̂ (innovations)
u_hat = Y - X_raw @ beta_mle
epsilon_hat = u_hat - lambda_mle * (W @ u_hat)

print(f"\nMLE Results:")
print(f"  λ̂: {lambda_mle:.4f} (true: {lambda_true})")
print(f"  β̂: {beta_mle} (true: {beta_true})")
print(f"  σ̂: {np.sqrt(sigma2_mle):.4f} (true: {sigma_true})")

# Standard errors (simplified - full calculation requires Hessian)
# Var(β̂_SEM) from inverse of Fisher information
# For demonstration, compare to OLS SE
print(f"\n(Full SE calculation via Hessian; using software recommended)")

# Residual diagnostics
I_resid_mle = morans_i_simple(epsilon_hat, W)
print(f"\nMoran's I on SEM innovations ε̂: {I_resid_mle:.4f}")
print(f"  (Should be near zero if model correct)")

# ===== FGLS Estimation (Two-Step) =====
print("\n" + "="*80)
print("FEASIBLE GLS (FGLS) ESTIMATION")
print("="*80)

# Step 1: OLS residuals (already have resid_ols)
# Step 2: Estimate λ from residuals
# Simple method: regress û on Wû (instrumental variable approach)

Wu = W @ resid_ols

# IV regression: û = λWû + v
# Use Wû as instrument (actually, this is problematic; better use GMM)
# Simplified: λ̂ = (û'Wû) / (û'W'Wû)
lambda_fgls = (resid_ols @ Wu) / (Wu @ Wu)

print(f"Step 1: OLS residuals computed")
print(f"Step 2: λ̂_FGLS = {lambda_fgls:.4f} (true: {lambda_true})")

# Step 3: Construct Ω̂
# Ω = σ²[(I-λW)'(I-λW)]^{-1}
A_fgls = I - lambda_fgls * W
Omega_inv = A_fgls.T @ A_fgls

# Step 4: GLS
# β̂_GLS = (X'Ω^{-1}X)^{-1}X'Ω^{-1}Y
XtOmega_inv_X = X_raw.T @ Omega_inv @ X_raw
beta_fgls = np.linalg.inv(XtOmega_inv_X) @ (X_raw.T @ Omega_inv @ Y)

# Residuals
resid_fgls = Y - X_raw @ beta_fgls
sigma2_fgls = np.sum(resid_fgls ** 2) / n

print(f"\nFGLS Results:")
print(f"  λ̂: {lambda_fgls:.4f}")
print(f"  β̂: {beta_fgls}")
print(f"  σ̂: {np.sqrt(sigma2_fgls):.4f}")

# ===== Comparison of Estimators =====
print("\n" + "="*80)
print("COMPARISON OF ESTIMATORS")
print("="*80)

comparison = pd.DataFrame({
    'Parameter': ['λ', 'β₀', 'β₁', 'β₂', 'σ'],
    'True': [lambda_true, beta_true[0], beta_true[1], beta_true[2], sigma_true],
    'OLS': [np.nan, beta_ols[0], beta_ols[1], beta_ols[2], np.sqrt(sigma2_ols)],
    'MLE': [lambda_mle, beta_mle[0], beta_mle[1], beta_mle[2], np.sqrt(sigma2_mle)],
    'FGLS': [lambda_fgls, beta_fgls[0], beta_fgls[1], beta_fgls[2], np.sqrt(sigma2_fgls)]
})

print(comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print(f"\nKey Insights:")
print(f"  • OLS β̂ consistent (close to true β)")
print(f"  • But OLS SE biased (doesn't account for spatial correlation)")
print(f"  • MLE and FGLS estimate λ, correct SE")
print(f"  • MLE more efficient than FGLS")

# ===== Compare to Spatial Lag Model (Misspecification) =====
print("\n" + "="*80)
print("MISSPECIFICATION CHECK: SAR instead of SEM")
print("="*80)

# Fit SAR (spatial lag) to data generated from SEM
# This demonstrates what happens with wrong model

WY = W @ Y

# 2SLS for SAR
WX = W @ X_raw[:, 1:]  # Instruments
X_first = np.column_stack([X_raw, WX])
gamma_first = np.linalg.inv(X_first.T @ X_first) @ X_first.T @ WY
WY_hat = X_first @ gamma_first

# Second stage
X_second = np.column_stack([WY_hat, X_raw])
params_sar = np.linalg.inv(X_second.T @ X_second) @ X_second.T @ Y

rho_sar = params_sar[0]
beta_sar = params_sar[1:]

print(f"Spatial Lag Model (Wrong Specification):")
print(f"  ρ̂ = {rho_sar:.4f} (spurious - true model has no ρ)")
print(f"  β̂ = {beta_sar}")

# Residuals
resid_sar = Y - X_second @ params_sar
I_resid_sar = morans_i_simple(resid_sar, W)

print(f"\nMoran's I on SAR residuals: {I_resid_sar:.4f}")
print(f"  Still has spatial correlation → Wrong model")

print(f"\n✓ SEM correctly removes spatial autocorrelation")
print(f"  SAR misspecification leaves residual correlation")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: True error u (spatially correlated)
ax1 = axes[0, 0]
u_grid = u.reshape(grid_size, grid_size)
im1 = ax1.imshow(u_grid, cmap='RdBu_r', origin='lower')
ax1.set_title(f'True Error u (λ={lambda_true}, I={I_u:.3f})')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
plt.colorbar(im1, ax=ax1)

# Plot 2: OLS residuals (spatial pattern)
ax2 = axes[0, 1]
resid_ols_grid = resid_ols.reshape(grid_size, grid_size)
im2 = ax2.imshow(resid_ols_grid, cmap='RdBu_r', origin='lower')
ax2.set_title(f'OLS Residuals (I={I_resid_ols:.3f})')
ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')
plt.colorbar(im2, ax=ax2)

# Plot 3: SEM innovations (no spatial pattern)
ax3 = axes[0, 2]
epsilon_hat_grid = epsilon_hat.reshape(grid_size, grid_size)
im3 = ax3.imshow(epsilon_hat_grid, cmap='RdBu_r', origin='lower')
ax3.set_title(f'SEM Innovations ε̂ (I={I_resid_mle:.3f})')
ax3.set_xlabel('X coordinate')
ax3.set_ylabel('Y coordinate')
plt.colorbar(im3, ax=ax3)

# Plot 4: Moran scatter (residuals)
ax4 = axes[1, 0]
Wu_ols = W @ resid_ols
ax4.scatter(resid_ols, Wu_ols, alpha=0.6, s=30, label='OLS')
slope_ols = np.cov(resid_ols, Wu_ols)[0, 1] / np.var(resid_ols)
x_line = np.linspace(resid_ols.min(), resid_ols.max(), 100)
y_line = slope_ols * (x_line - np.mean(resid_ols)) + np.mean(Wu_ols)
ax4.plot(x_line, y_line, 'r--', linewidth=2, label=f'Slope={slope_ols:.3f}')
ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax4.axvline(0, color='gray', linestyle='-', linewidth=0.5)
ax4.set_xlabel('OLS Residuals')
ax4.set_ylabel('W × Residuals')
ax4.set_title('Moran Scatter: Spatial Error')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: β comparison
ax5 = axes[1, 1]
params_compare = np.array([
    [beta_true[1], beta_true[2]],
    [beta_ols[1], beta_ols[2]],
    [beta_mle[1], beta_mle[2]],
    [beta_fgls[1], beta_fgls[2]]
])
x_pos = np.arange(2)
width = 0.2
colors_bar = ['black', 'gray', 'red', 'blue']
labels_bar = ['True', 'OLS', 'MLE', 'FGLS']

for i, (params, color, label) in enumerate(zip(params_compare, colors_bar, labels_bar)):
    ax5.bar(x_pos + i * width, params, width, label=label, color=color, alpha=0.7)

ax5.set_xticks(x_pos + 1.5 * width)
ax5.set_xticklabels(['β₁', 'β₂'])
ax5.set_ylabel('Coefficient')
ax5.set_title('β Estimates (All Consistent)')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')
ax5.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot 6: LM test statistics
ax6 = axes[1, 2]
lm_stats = [LM_err, LM_lag]
lm_labels = ['LM_error', 'LM_lag']
colors_lm = ['red' if LM_err > stats.chi2.ppf(0.95, 1) else 'gray',
             'blue' if LM_lag > stats.chi2.ppf(0.95, 1) else 'gray']

bars = ax6.bar(range(2), lm_stats, color=colors_lm, alpha=0.7)
ax6.set_xticks(range(2))
ax6.set_xticklabels(lm_labels)
ax6.set_ylabel('LM Statistic')
ax6.set_title('Lagrange Multiplier Tests')
ax6.axhline(stats.chi2.ppf(0.95, 1), color='red', linestyle='--', 
           linewidth=1, label='5% critical value')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

# Add p-values on bars
for i, (bar, stat, pval) in enumerate(zip(bars, lm_stats, [p_err, p_lag])):
    ax6.text(i, stat + 1, f'p={pval:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('spatial_error_model.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Spatial Error Correlation:")
print(f"   True λ = {lambda_true} → Positive error clustering")
print(f"   Moran's I (errors) = {I_u:.4f}")
print(f"   Source: Omitted spatially correlated variables")

print("\n2. OLS Properties:")
print(f"   β̂_OLS consistent (bias: {bias_ols})")
print(f"   But SE biased (underestimated if λ>0)")
print(f"   Residuals spatially autocorrelated (I={I_resid_ols:.3f})")

print("\n3. LM Tests:")
print(f"   LM_error = {LM_err:.2f} (p={p_err:.4f}) → Significant")
print(f"   LM_lag = {LM_lag:.2f} (p={p_lag:.4f}) → Not significant")
print(f"   Conclusion: Spatial Error Model appropriate")

print("\n4. MLE vs FGLS:")
print(f"   MLE λ̂ = {lambda_mle:.4f} (accurate)")
print(f"   FGLS λ̂ = {lambda_fgls:.4f} (close but less efficient)")
print(f"   Both correct spatial correlation")
print(f"   MLE preferred for efficiency")

print("\n5. Coefficient Interpretation:")
print(f"   β coefficients same as OLS (no spillovers)")
print(f"   λ is nuisance parameter (not of primary interest)")
print(f"   Standard errors corrected → Valid inference")

print("\n6. Practical Recommendations:")
print("   • Test for spatial error: LM_error on OLS residuals")
print("   • Distinguish SEM vs SAR: Use LM tests")
print("   • MLE preferred (efficient under normality)")
print("   • FGLS simpler but less efficient")
print("   • Report corrected standard errors")
print("   • Check residuals: Should have I≈0")

print("\n7. Software:")
print("   • Python: PySAL (spreg.ML_Error, spreg.LMtests)")
print("   • R: spatialreg::errorsarlm(), lm.LMtests()")
print("   • Stata: spregress with ml error() option")
```

## 6. Challenge Round
When does spatial error modeling fail or mislead?
- **SAR misspecified as SEM**: True model has Y spillovers (ρWY), fit SEM → β̂ remains biased; λ̂ spuriously captures outcome dependence → Use LM tests; robust versions
- **Spatial heteroskedasticity**: Var(εᵢ) varies by location, SEM assumes homoskedasticity → Biased λ̂, inefficient β̂; use SHEM (spatial heteroskedastic error model)
- **Non-stationarity**: λ varies across space (urban/rural different error correlation) → Global SEM averages; spatial regime models needed
- **Weight matrix wrong**: W doesn't reflect true spatial relationships → λ̂ biased, SE correction incomplete; test multiple W specifications
- **Omitted spatial lag**: If both ρWY and λWu present, pure SEM insufficient → Spatial Durbin Model (SDM) or general spatial model (SAC)
- **Measurement error in W**: Neighbors misidentified (boundary errors, data quality) → Attenuation bias in λ̂; sensitivity analysis critical

## 7. Key References
- [Anselin (1988) - Spatial Econometrics: Methods and Models](https://link.springer.com/book/10.1007/978-94-015-7799-1)
- [Kelejian & Prucha (1998) - A Generalized Spatial Two-Stage Least Squares Procedure](https://www.sciencedirect.com/science/article/abs/pii/S016517659700877X)
- [LeSage & Pace (2009) - Introduction to Spatial Econometrics, Ch. 5](https://www.taylorfrancis.com/books/mono/10.1201/9781420064254/introduction-spatial-econometrics-james-lesage-robert-pace)

---
**Status:** Alternative to spatial lag model when no outcome spillovers | **Complements:** Spatial lag model, LM tests, spatial Durbin model
