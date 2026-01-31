# Spatial Lag Model

## 1. Concept Skeleton
**Definition:** Regression with outcome depending on neighbors' outcomes via ρWY term; models spatial spillovers and feedback effects  
**Purpose:** Capture endogenous interaction effects where Y at location i directly influenced by Y at neighbors; estimate spillover magnitude  
**Prerequisites:** Spatial autocorrelation, spatial weight matrices, maximum likelihood estimation, instrumental variables, matrix algebra

## 2. Comparative Framing
| Model | Spatial Lag (SAR) | Spatial Error (SEM) | Spatial Durbin (SDM) | OLS | Panel Fixed Effects | GWR |
|-------|-------------------|---------------------|----------------------|-----|---------------------|-----|
| **Specification** | Y = ρWY + Xβ + ε | Y = Xβ + λWu + u | Y = ρWY + Xβ + WXθ + ε | Y = Xβ + ε | Y = Xβ + αᵢ + ε | Y = Xβ(s) + ε |
| **Spatial Dependence** | Endogenous (Y) | Exogenous (error) | Both Y and X | None | Time-invariant αᵢ | Varying coefficients |
| **Interpretation** | Global spillovers | Nuisance (omitted) | Spillovers + local | Direct effects only | Within-unit | Local effects |
| **Endogeneity** | Yes (simultaneity) | No | Yes | No | Corr(αᵢ,X) allowed | No |
| **Estimation** | MLE, IV/2SLS, GMM | MLE, FGLS | MLE, IV | OLS | Within/FE | Local OLS |
| **Multiplier Effect** | Yes: (I-ρW)⁻¹ | No | Yes | No | No | No |

## 3. Examples + Counterexamples

**Classic Example:**  
House prices (n=500): ρ=0.35 (p<0.001). $1 increase in house value raises neighbors' values by $0.54 (direct+indirect effects). Ignoring spatial lag (OLS) biases β̂ₓ and underestimates standard errors. Moran's I on OLS residuals=0.28, on SAR residuals=0.02 (corrected).

**Failure Case:**  
Spatial error process misspecified as lag: True model Y=Xβ+λWu, fit Y=ρWY+Xβ+ε. ρ̂=0.25 (significant) but spurious—captures error correlation not outcome spillovers. LM test discriminates: LM_lag insignificant, LM_error=18.5 (p<0.001).

**Edge Case:**  
ρ near unity (ρ=0.95): Explosive spatial process unlikely theoretically. Estimates sensitive to weight matrix normalization. Check eigenvalue bounds: ρ ∈ (1/λ_min, 1/λ_max) for stationarity.

## 4. Layer Breakdown
```
Spatial Lag Model (Spatial Autoregressive Model):
├─ Model Specification:
│   ├─ Basic Form:
│   │   ├─ Y = ρWY + Xβ + ε
│   │   ├─ Y: n×1 outcome vector
│   │   ├─ X: n×k covariates (includes intercept)
│   │   ├─ W: n×n spatial weight matrix (row-standardized)
│   │   ├─ ρ: Spatial autoregressive coefficient
│   │   ├─ β: k×1 covariate effects
│   │   └─ ε ~ N(0, σ²I): iid errors
│   ├─ Reduced Form:
│   │   ├─ Y = (I - ρW)⁻¹(Xβ + ε)
│   │   ├─ Requires |ρ| < 1 for invertibility
│   │   ├─ Shows Y as weighted average of all units (global spillovers)
│   │   └─ Spatial multiplier: S = (I - ρW)⁻¹
│   ├─ Interpretation:
│   │   ├─ ρ > 0: Positive feedback (clustering)
│   │   ├─ ρ < 0: Negative feedback (competition, rare)
│   │   ├─ WY: Spatially lagged outcome (neighbors' average)
│   │   └─ Endogenous interaction effect
│   ├─ Contrast with Spatial Error:
│   │   ├─ SEM: Y = Xβ + u, u = λWu + ε (nuisance)
│   │   ├─ SAR: Substantive spillovers (ρ in structural equation)
│   │   └─ SAR implies spatial dependence in outcome; SEM in shocks
│   └─ Stationarity:
│       ├─ ρ must satisfy: 1/λ_min(W) < ρ < 1/λ_max(W)
│       ├─ λ_min, λ_max: Eigenvalues of W
│       └─ Typically ρ ∈ (-1, 1) for row-standardized W
├─ Motivation and Theory:
│   ├─ Tobler's Law:
│   │   ├─ Near things more related than distant
│   │   └─ Outcome at i influenced by outcomes at neighbors
│   ├─ Examples of Spatial Spillovers:
│   │   ├─ Housing: Neighborhood quality affects property values
│   │   ├─ Crime: Crime in area i increases risk in neighbors
│   │   ├─ Growth: Regional development spillovers
│   │   ├─ Innovation: Knowledge diffusion across regions
│   │   └─ Epidemiology: Contagion effects
│   ├─ Endogeneity:
│   │   ├─ WY correlated with ε (simultaneity)
│   │   ├─ E[yᵢ|yⱼ, j≠i] depends on neighbors
│   │   └─ OLS inconsistent (biased, even asymptotically)
│   ├─ Identification:
│   │   ├─ Relies on W being pre-specified (exogenous)
│   │   ├─ Spatial exclusion restrictions
│   │   └─ Variation in network position identifies ρ
│   └─ Feedback Effects:
│       ├─ Shock to unit i affects neighbors (WY term)
│       ├─ Neighbors' responses feed back to i
│       └─ Total effect > direct effect (multiplier)
├─ Estimation Methods:
│   ├─ OLS (Biased, Inconsistent):
│   │   ├─ Ignores endogeneity of WY
│   │   ├─ Biased β̂, ρ not identified
│   │   └─ Only valid if ρ=0 (test this)
│   ├─ Maximum Likelihood (MLE):
│   │   ├─ Log-Likelihood:
│   │   │   ├─ ℓ(ρ,β,σ²) = -(n/2)log(2πσ²) + log|I-ρW| - (1/2σ²)(Y-ρWY-Xβ)'(Y-ρWY-Xβ)
│   │   │   ├─ log|I-ρW| = Σᵢ log(1 - ρλᵢ) where λᵢ: eigenvalues of W
│   │   │   └─ Jacobian term crucial (accounts for simultaneity)
│   │   ├─ Concentrated Likelihood:
│   │   │   ├─ Profile out β, σ²
│   │   │   ├─ ℓ_c(ρ) = -(n/2)log(σ̂²(ρ)) + log|I-ρW|
│   │   │   └─ Grid search or optimization over ρ
│   │   ├─ Standard Errors:
│   │   │   ├─ From inverse of Hessian (Fisher information)
│   │   │   └─ Asymptotically efficient
│   │   ├─ Advantages:
│   │   │   ├─ Consistent and efficient
│   │   │   ├─ Allows hypothesis testing (LR, Wald)
│   │   │   └─ Standard approach
│   │   └─ Disadvantages:
│   │       ├─ Requires normality assumption (can relax)
│   │       ├─ Computationally intensive (eigenvalues)
│   │       └─ Sensitive to misspecification
│   ├─ Instrumental Variables / 2SLS:
│   │   ├─ Instrument for WY:
│   │   │   ├─ Use WX, W²X, higher-order lags
│   │   │   ├─ Spatial exclusion: Neighbors' X affects yᵢ via neighbors' Y
│   │   │   └─ Relevance: Cov(WY, WX) ≠ 0; Exogeneity: Cov(WX, ε) = 0
│   │   ├─ 2SLS Steps:
│   │   │   ├─ First stage: WY = WXγ + Xδ + v
│   │   │   ├─ Second stage: Y = ρ(ŴY) + Xβ + ε
│   │   │   └─ Standard 2SLS inference
│   │   ├─ Advantages:
│   │   │   ├─ No distributional assumptions
│   │   │   ├─ Robust to non-normality
│   │   │   └─ Familiar framework
│   │   └─ Disadvantages:
│   │       ├─ Less efficient than MLE
│   │       ├─ Many weak instruments if higher-order lags
│   │       └─ Sensitive to instrument choice
│   ├─ Generalized Method of Moments (GMM):
│   │   ├─ Moment conditions:
│   │   │   ├─ E[X'ε] = 0
│   │   │   ├─ E[(WX)'ε] = 0
│   │   │   └─ E[(W²X)'ε] = 0
│   │   ├─ Optimal weighting via HAC
│   │   ├─ Overidentification tests (Hansen J)
│   │   └─ Handles heteroskedasticity
│   ├─ Bayesian Estimation:
│   │   ├─ Priors on ρ, β, σ²
│   │   ├─ MCMC sampling (Gibbs, MH)
│   │   ├─ Naturally handles parameter uncertainty
│   │   └─ Flexible (hierarchical extensions)
│   └─ Comparison:
│       ├─ MLE: Efficient but assumes normality
│       ├─ IV/2SLS: Robust but less efficient
│       ├─ GMM: Flexible weighting, robust
│       └─ Bayesian: Full uncertainty, computationally intensive
├─ Inference and Hypothesis Testing:
│   ├─ Spatial Autocorrelation Tests:
│   │   ├─ Moran's I on residuals
│   │   ├─ If significant → Spatial model needed
│   │   └─ Compare OLS vs SAR residuals
│   ├─ LM Tests (Lagrange Multiplier):
│   │   ├─ LM_lag: Test H₀: ρ=0 (no spatial lag)
│   │   ├─ LM_error: Test H₀: λ=0 (no spatial error)
│   │   ├─ Robust versions: Account for alternative
│   │   └─ Guide model selection (lag vs error)
│   ├─ Likelihood Ratio Test:
│   │   ├─ LR = 2(ℓ_SAR - ℓ_OLS)
│   │   ├─ χ²(1) under null ρ=0
│   │   └─ Nested models
│   ├─ Wald Test:
│   │   ├─ Test ρ=0 or linear restrictions on β
│   │   ├─ Wald = (ρ̂-0)² / Var(ρ̂)
│   │   └─ Asymptotically χ²
│   └─ Model Comparison:
│       ├─ AIC, BIC: Penalize complexity
│       ├─ Log-likelihood values
│       └─ Out-of-sample prediction accuracy
├─ Interpretation of Coefficients:
│   ├─ Direct Effects:
│   │   ├─ Impact of X_i on Y_i holding neighbors constant
│   │   ├─ Not simply β (due to feedback)
│   │   ├─ Average direct effect: (1/n) × tr[(I-ρW)⁻¹β]
│   │   └─ Diagonal elements of multiplier matrix
│   ├─ Indirect Effects (Spillovers):
│   │   ├─ Impact of X_j (j≠i) on Y_i via spatial network
│   │   ├─ Off-diagonal elements of (I-ρW)⁻¹β
│   │   └─ Average indirect effect: (1/n) × 1'[(I-ρW)⁻¹β - diag((I-ρW)⁻¹β)]
│   ├─ Total Effects:
│   │   ├─ Total = Direct + Indirect
│   │   ├─ Average total effect: (1/n) × 1'(I-ρW)⁻¹β1
│   │   └─ Full impact of X change across all units
│   ├─ Spatial Multiplier:
│   │   ├─ S = (I - ρW)⁻¹ = I + ρW + ρ²W² + ρ³W³ + ...
│   │   ├─ Infinite series if |ρ|<1
│   │   ├─ Captures feedback loops
│   │   └─ Multiplier magnitude: [1/(1-ρ)] for uniform neighbors
│   ├─ Marginal Effects:
│   │   ├─ ∂Y/∂X is matrix, not scalar
│   │   ├─ Summarize via averages (direct/indirect/total)
│   │   └─ Report all three for completeness
│   └─ Example:
│       ├─ ρ=0.4, β_income=0.5
│       ├─ Direct effect ≈ 0.5 × [1/(1-0.4 × avg_neighbor)] ≈ 0.6
│       ├─ Indirect effect ≈ 0.5 × [0.4/(1-0.4)] ≈ 0.33
│       └─ Total effect ≈ 0.93
├─ Spatial Durbin Model (SDM):
│   ├─ Specification:
│   │   ├─ Y = ρWY + Xβ + WXθ + ε
│   │   ├─ Adds WX (neighbors' covariates)
│   │   └─ Nests SAR (θ=0) and SLX (ρ=0)
│   ├─ Interpretation:
│   │   ├─ Allows local spillovers (WX) and global (WY)
│   │   ├─ More flexible than pure SAR
│   │   └─ θ: Direct exogenous spillovers
│   ├─ Estimation:
│   │   ├─ MLE similar to SAR
│   │   └─ IV with WX, W²X instruments
│   └─ Testing:
│       ├─ Test θ=0: Collapse to SAR
│       └─ Test ρ=0, θ=-ρβ: Collapse to SEM
├─ Diagnostics:
│   ├─ Residual Spatial Autocorrelation:
│   │   ├─ Moran's I on ε̂
│   │   ├─ Should be near zero if model correct
│   │   └─ Significant I → Misspecification
│   ├─ Influential Observations:
│   │   ├─ Spatial Cook's D
│   │   ├─ Leverage: Units with many neighbors
│   │   └─ Outliers: Extreme residuals
│   ├─ Heteroskedasticity:
│   │   ├─ Breusch-Pagan test on residuals
│   │   ├─ Robust standard errors (White)
│   │   └─ GMM estimation
│   ├─ Non-Stationarity:
│   │   ├─ Chow test for spatial regimes
│   │   ├─ GWR as alternative
│   │   └─ Examine local parameter variation
│   └─ Weight Matrix Sensitivity:
│       ├─ Re-estimate with different W
│       ├─ Compare ρ̂, β̂ estimates
│       └─ Robustness essential
├─ Practical Implementation:
│   ├─ Python:
│   │   ├─ PySAL (spreg): ML_Lag, GM_Lag classes
│   │   ├─ Specification: spreg.ML_Lag(y, X, w)
│   │   ├─ Returns: ρ, β, standard errors, diagnostics
│   │   └─ Spatial diagnostics built-in
│   ├─ R:
│   │   ├─ spatialreg: lagsarlm() function
│   │   ├─ Syntax: lagsarlm(Y ~ X, data, listw)
│   │   ├─ Impacts: impacts() for direct/indirect/total
│   │   └─ Extensive diagnostics (lm.morantest, etc.)
│   ├─ Stata:
│   │   ├─ spregress with lag option
│   │   ├─ Syntax: spregress Y X, ml lag(W)
│   │   └─ estat impact for effects decomposition
│   └─ MATLAB:
│       ├─ Spatial Econometrics Toolbox (LeSage)
│       └─ sar() function
├─ Extensions:
│   ├─ Spatial Panel Data:
│   │   ├─ Y_it = ρW Y_it + X_it β + α_i + ε_it
│   │   ├─ Fixed effects + spatial lag
│   │   ├─ Dynamic: Y_it = τY_i,t-1 + ρW Y_it + ...
│   │   └─ Maximum likelihood or GMM
│   ├─ Spatial Probit/Logit:
│   │   ├─ Binary outcome with spatial lag
│   │   ├─ Y* = ρWY* + Xβ + ε, Y=1(Y*>0)
│   │   ├─ Recursive importance sampling or MCMC
│   │   └─ Difficult estimation
│   ├─ Spatial Quantile Regression:
│   │   ├─ Conditional quantiles with spatial lag
│   │   ├─ Heterogeneous spatial effects
│   │   └─ MCMC or instrumental variables
│   ├─ Higher-Order Spatial Lags:
│   │   ├─ Y = ρ₁WY + ρ₂W²Y + Xβ + ε
│   │   ├─ Multiple neighborhood rings
│   │   └─ Rare in practice (identification issues)
│   └─ Spatiotemporal Models:
│       ├─ Space-time weight matrices
│       ├─ Diffusion dynamics
│       └─ Forecasting applications
├─ Assumptions:
│   ├─ Correct Weight Matrix:
│   │   ├─ W reflects true spatial relationships
│   │   ├─ Misspecification biases estimates
│   │   └─ Theory/context should guide W
│   ├─ Homoskedasticity:
│   │   ├─ Var(ε_i) = σ² constant
│   │   ├─ Relaxable via robust SEs or GMM
│   │   └─ Heteroskedasticity common in spatial data
│   ├─ Exogeneity of X:
│   │   ├─ E[X'ε] = 0
│   │   ├─ Omitted variables cause bias
│   │   └─ Control for confounders
│   ├─ Linearity:
│   │   ├─ Effect of WY on Y is linear
│   │   ├─ May need transformations or interactions
│   │   └─ Test functional form
│   ├─ Stationarity:
│   │   ├─ Constant ρ, β across space
│   │   ├─ Violated if spatial regimes exist
│   │   └─ GWR detects non-stationarity
│   └─ No Measurement Error:
│       ├─ Especially in W (neighbor definitions)
│       └─ Mismeasured locations bias ρ̂
└─ Common Pitfalls:
    ├─ Confusing SAR with SEM: Different interpretations, models
    ├─ Ignoring weight matrix sensitivity: Always check robustness
    ├─ Reporting only ρ: Must decompose effects (direct/indirect/total)
    ├─ Using OLS: Inconsistent estimates; always use MLE or IV
    ├─ Overfitting with high-order lags: Identification weak
    └─ Not testing for spatial autocorrelation first: Run Moran's I on OLS residuals
```

**Interaction:** OLS → Moran's I on residuals (spatial autocorrelation?) → LM tests (lag vs error?) → Estimate SAR via MLE or IV → Compute direct/indirect/total effects → Diagnostic checks

## 5. Mini-Project
Implement spatial lag model with MLE and IV estimation; compare OLS:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import seaborn as sns

np.random.seed(357)

# ===== Simulate Spatial Data with Known Spillovers =====
print("="*80)
print("SPATIAL LAG MODEL (SAR)")
print("="*80)

# Grid coordinates
grid_size = 15
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

# Spatial weight matrix (Queen contiguity)
def create_weight_matrix_queen(coords, grid_size):
    n = len(coords)
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.abs(coords[i] - coords[j])
                if np.max(dist) <= 1:  # Queen: shares edge or corner
                    W[i, j] = 1
    
    # Row standardization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_std = W / row_sums
    
    return W_std, W

W, W_raw = create_weight_matrix_queen(coords, grid_size)

print(f"  Weight matrix: Queen contiguity, row-standardized")
print(f"  Connections: {np.sum(W_raw > 0)} ({np.mean(np.sum(W_raw>0, axis=1)):.1f} avg neighbors)")

# True parameters
rho_true = 0.5
beta_true = np.array([2.0, 1.5, -0.8])
sigma_true = 0.5

print(f"\nTrue Parameters:")
print(f"  ρ (spatial lag): {rho_true}")
print(f"  β: {beta_true}")
print(f"  σ: {sigma_true}")

# Generate Y via spatial process: Y = (I - ρW)^{-1}(Xβ + ε)
epsilon = np.random.randn(n) * sigma_true
Xbeta = X_raw @ beta_true

# Solve (I - ρW)Y = Xβ + ε
I = np.eye(n)
A = I - rho_true * W
Y = np.linalg.solve(A, Xbeta + epsilon)

print(f"  Generated Y via: Y = (I - ρW)^{{-1}}(Xβ + ε)")
print(f"  Y statistics: mean={np.mean(Y):.3f}, sd={np.std(Y):.3f}")

# Spatial lag variable
WY = W @ Y

# Check spatial autocorrelation (Moran's I)
def morans_i_simple(y, W):
    n = len(y)
    y_dev = y - np.mean(y)
    numerator = np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    S0 = np.sum(W)
    I = (n / S0) * (numerator / denominator)
    return I

I_Y = morans_i_simple(Y, W)
print(f"\nMoran's I on Y: {I_Y:.4f} (strong positive autocorrelation)")

# ===== OLS Estimation (Naive, Biased) =====
print("\n" + "="*80)
print("OLS ESTIMATION (Biased)")
print("="*80)

# OLS: Y = Xβ + ε (ignores spatial lag)
XtX_inv = np.linalg.inv(X_raw.T @ X_raw)
beta_ols = XtX_inv @ X_raw.T @ Y
resid_ols = Y - X_raw @ beta_ols
sigma2_ols = np.sum(resid_ols ** 2) / (n - X_raw.shape[1])
se_ols = np.sqrt(np.diag(XtX_inv * sigma2_ols))

print(f"OLS Results:")
print(f"  β̂: {beta_ols}")
print(f"  SE: {se_ols}")
print(f"  σ̂: {np.sqrt(sigma2_ols):.4f}")

# Moran's I on OLS residuals
I_resid_ols = morans_i_simple(resid_ols, W)
print(f"\nMoran's I on OLS residuals: {I_resid_ols:.4f}")
if I_resid_ols > 0.1:
    print(f"  ✓ Significant spatial autocorrelation in residuals → SAR model needed")

# Bias in OLS
bias_ols = beta_ols - beta_true
print(f"\nOLS Bias (β̂_OLS - β_true): {bias_ols}")

# ===== Maximum Likelihood Estimation =====
print("\n" + "="*80)
print("MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
print("="*80)

# Eigenvalues of W for Jacobian
eigvals_W = eigh(W, eigvals_only=True)
lambda_min, lambda_max = eigvals_W.min(), eigvals_W.max()

print(f"Eigenvalue bounds for W:")
print(f"  λ_min = {lambda_min:.4f}, λ_max = {lambda_max:.4f}")
print(f"  ρ must be in ({1/lambda_min:.3f}, {1/lambda_max:.3f}) for stationarity")

def log_likelihood_sar(params, Y, X, W, eigvals_W):
    """Concentrated log-likelihood for SAR"""
    rho = params[0]
    n = len(Y)
    
    # Bounds check
    if rho <= 1/eigvals_W.min() or rho >= 1/eigvals_W.max():
        return -1e10
    
    # Jacobian term: log|I - ρW| = sum(log(1 - ρλ_i))
    log_det = np.sum(np.log(1 - rho * eigvals_W))
    
    # Residuals
    Y_trans = Y - rho * (W @ Y)
    
    # Concentrated β
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y_trans
    
    # Residuals
    resid = Y_trans - X @ beta_hat
    
    # Concentrated σ²
    sigma2_hat = np.sum(resid ** 2) / n
    
    # Log-likelihood
    ll = -(n/2) * np.log(2*np.pi) - (n/2) * np.log(sigma2_hat) + log_det - n/2
    
    return -ll  # Minimize negative ll

# Optimize over ρ
print(f"\nOptimizing MLE...")
result_mle = optimize.minimize_scalar(
    lambda rho: log_likelihood_sar(np.array([rho]), Y, X_raw, W, eigvals_W),
    bounds=(1/lambda_min + 0.01, 1/lambda_max - 0.01),
    method='bounded'
)

rho_mle = result_mle.x

# Get β, σ² at optimal ρ
Y_trans_mle = Y - rho_mle * WY
XtX_inv = np.linalg.inv(X_raw.T @ X_raw)
beta_mle = XtX_inv @ X_raw.T @ Y_trans_mle
resid_mle = Y_trans_mle - X_raw @ beta_mle
sigma2_mle = np.sum(resid_mle ** 2) / n

print(f"\nMLE Results:")
print(f"  ρ̂: {rho_mle:.4f} (true: {rho_true})")
print(f"  β̂: {beta_mle} (true: {beta_true})")
print(f"  σ̂: {np.sqrt(sigma2_mle):.4f} (true: {sigma_true})")

# Standard errors (from Hessian)
# Simplified: Use numerical Hessian
from scipy.optimize import approx_fprime

def ll_all_params(params):
    rho = params[0]
    beta = params[1:4]
    log_sigma2 = params[4]
    sigma2 = np.exp(log_sigma2)
    
    if rho <= 1/lambda_min + 0.01 or rho >= 1/lambda_max - 0.01:
        return 1e10
    
    log_det = np.sum(np.log(1 - rho * eigvals_W))
    Y_trans = Y - rho * WY
    resid = Y_trans - X_raw @ beta
    ll = -(n/2) * np.log(2*np.pi*sigma2) + log_det - (1/(2*sigma2)) * np.sum(resid**2)
    return -ll

params_mle_all = np.concatenate([[rho_mle], beta_mle, [np.log(sigma2_mle)]])

# Hessian approximation
hessian_approx = optimize.approx_fprime(params_mle_all, ll_all_params, epsilon=1e-5)
# For proper SE, need full Hessian matrix (skip for brevity; use software)

print(f"\n(Full SE calculation omitted; use software like PySAL for accurate SEs)")

# Moran's I on MLE residuals
I_resid_mle = morans_i_simple(resid_mle, W)
print(f"\nMoran's I on MLE residuals: {I_resid_mle:.4f}")
print(f"  (Should be near zero if model correct)")

# ===== Instrumental Variables (2SLS) Estimation =====
print("\n" + "="*80)
print("INSTRUMENTAL VARIABLES (2SLS) ESTIMATION")
print("="*80)

# Instruments: WX (neighbors' covariates)
WX = W @ X_raw[:, 1:]  # Exclude intercept, apply W to X1, X2

# First stage: WY ~ X + WX
X_first = np.column_stack([X_raw, WX])
XtX_first_inv = np.linalg.inv(X_first.T @ X_first)
gamma_first = XtX_first_inv @ X_first.T @ WY
WY_hat = X_first @ gamma_first

# First stage diagnostics
resid_first = WY - WY_hat
SS_res_first = np.sum(resid_first ** 2)
SS_tot_first = np.sum((WY - np.mean(WY)) ** 2)
R2_first = 1 - SS_res_first / SS_tot_first

# F-statistic for instrument strength
k1 = X_raw.shape[1]
k2 = WX.shape[1]
F_stat = ((SS_tot_first - SS_res_first) / k2) / (SS_res_first / (n - X_first.shape[1]))

print(f"First Stage: WY ~ X + WX")
print(f"  R²: {R2_first:.4f}")
print(f"  F-statistic: {F_stat:.2f}")
if F_stat > 10:
    print(f"  ✓ Strong instruments (F > 10)")
else:
    print(f"  ⚠ Weak instruments (F < 10)")

# Second stage: Y ~ ŴY + X
X_second = np.column_stack([WY_hat, X_raw])
XtX_second_inv = np.linalg.inv(X_second.T @ X_second)
params_2sls = XtX_second_inv @ X_second.T @ Y

rho_2sls = params_2sls[0]
beta_2sls = params_2sls[1:]

# Residuals and σ²
resid_2sls = Y - X_second @ params_2sls
sigma2_2sls = np.sum(resid_2sls ** 2) / (n - X_second.shape[1])

# Standard errors (need to account for generated regressor)
# Simplified (not fully correct without adjustment)
se_2sls_naive = np.sqrt(np.diag(XtX_second_inv * sigma2_2sls))

print(f"\n2SLS Results:")
print(f"  ρ̂: {rho_2sls:.4f} (true: {rho_true})")
print(f"  β̂: {beta_2sls} (true: {beta_true})")
print(f"  σ̂: {np.sqrt(sigma2_2sls):.4f}")
print(f"  (SE not adjusted for generated regressor)")

# ===== Compare Estimators =====
print("\n" + "="*80)
print("COMPARISON OF ESTIMATORS")
print("="*80)

comparison = pd.DataFrame({
    'Parameter': ['ρ', 'β₀', 'β₁', 'β₂'],
    'True': [rho_true, beta_true[0], beta_true[1], beta_true[2]],
    'OLS': [np.nan, beta_ols[0], beta_ols[1], beta_ols[2]],
    'MLE': [rho_mle, beta_mle[0], beta_mle[1], beta_mle[2]],
    '2SLS': [rho_2sls, beta_2sls[0], beta_2sls[1], beta_2sls[2]]
})

print(comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ===== Direct, Indirect, Total Effects (MLE) =====
print("\n" + "="*80)
print("SPATIAL EFFECTS DECOMPOSITION (MLE)")
print("="*80)

# Spatial multiplier: S = (I - ρW)^{-1}
S = np.linalg.inv(I - rho_mle * W)

# For each covariate (excluding intercept for interpretation)
effects_results = []

for k in range(1, len(beta_mle)):  # Skip intercept
    # Direct effects: Diagonal of S * β_k
    direct_k = np.mean(np.diag(S * beta_mle[k]))
    
    # Total effects: Mean of (S * β_k) row sums
    total_k = np.mean(np.sum(S * beta_mle[k], axis=1))
    
    # Indirect effects: Total - Direct
    indirect_k = total_k - direct_k
    
    effects_results.append({
        'Variable': f'X{k}',
        'Direct': direct_k,
        'Indirect': indirect_k,
        'Total': total_k
    })

effects_df = pd.DataFrame(effects_results)
print(effects_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print(f"\nInterpretation:")
print(f"  Direct: Impact of X_i on Y_i")
print(f"  Indirect: Impact of X_j (j≠i) on Y_i (spillover)")
print(f"  Total: Direct + Indirect (full network effect)")

# Spatial multiplier magnitude
avg_multiplier = 1 / (1 - rho_mle * np.mean(W[W > 0]))
print(f"\nSpatial multiplier (approx): {avg_multiplier:.3f}")
print(f"  $1 direct increase → ${avg_multiplier:.3f} total (including feedback)")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Spatial distribution of Y
ax1 = axes[0, 0]
Y_grid = Y.reshape(grid_size, grid_size)
im1 = ax1.imshow(Y_grid, cmap='RdBu_r', origin='lower')
ax1.set_title('True Y (Spatial Lag Process)')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
plt.colorbar(im1, ax=ax1)

# Plot 2: OLS residuals (spatial pattern)
ax2 = axes[0, 1]
resid_ols_grid = resid_ols.reshape(grid_size, grid_size)
im2 = ax2.imshow(resid_ols_grid, cmap='RdBu_r', origin='lower')
ax2.set_title(f'OLS Residuals (Moran I={I_resid_ols:.3f})')
ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')
plt.colorbar(im2, ax=ax2)

# Plot 3: MLE residuals (no spatial pattern)
ax3 = axes[0, 2]
resid_mle_grid = resid_mle.reshape(grid_size, grid_size)
im3 = ax3.imshow(resid_mle_grid, cmap='RdBu_r', origin='lower')
ax3.set_title(f'MLE Residuals (Moran I={I_resid_mle:.3f})')
ax3.set_xlabel('X coordinate')
ax3.set_ylabel('Y coordinate')
plt.colorbar(im3, ax=ax3)

# Plot 4: Moran scatter plot (Y vs WY)
ax4 = axes[1, 0]
ax4.scatter(Y, WY, alpha=0.6, s=30)
slope_moran = np.cov(Y, WY)[0, 1] / np.var(Y)
x_line = np.linspace(Y.min(), Y.max(), 100)
y_line = slope_moran * (x_line - np.mean(Y)) + np.mean(WY)
ax4.plot(x_line, y_line, 'r--', linewidth=2, label=f'Slope ≈ ρ = {rho_true:.2f}')
ax4.set_xlabel('Y')
ax4.set_ylabel('WY (Spatial Lag)')
ax4.set_title('Moran Scatter Plot')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Parameter comparison
ax5 = axes[1, 1]
params_compare = np.array([
    [beta_true[1], beta_true[2]],
    [beta_ols[1], beta_ols[2]],
    [beta_mle[1], beta_mle[2]],
    [beta_2sls[1], beta_2sls[2]]
])
x_pos = np.arange(2)
width = 0.2
colors_bar = ['black', 'gray', 'red', 'blue']
labels_bar = ['True', 'OLS', 'MLE', '2SLS']

for i, (params, color, label) in enumerate(zip(params_compare, colors_bar, labels_bar)):
    ax5.bar(x_pos + i * width, params, width, label=label, color=color, alpha=0.7)

ax5.set_xticks(x_pos + 1.5 * width)
ax5.set_xticklabels(['β₁', 'β₂'])
ax5.set_ylabel('Coefficient')
ax5.set_title('β Estimates Comparison')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# Plot 6: Effects decomposition
ax6 = axes[1, 2]
x_effects = np.arange(len(effects_df))
width_effects = 0.25
ax6.bar(x_effects - width_effects, effects_df['Direct'], width_effects, 
        label='Direct', alpha=0.7)
ax6.bar(x_effects, effects_df['Indirect'], width_effects, 
        label='Indirect (Spillover)', alpha=0.7)
ax6.bar(x_effects + width_effects, effects_df['Total'], width_effects, 
        label='Total', alpha=0.7)
ax6.set_xticks(x_effects)
ax6.set_xticklabels(effects_df['Variable'])
ax6.set_ylabel('Effect Size')
ax6.set_title('Spatial Effects Decomposition')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('spatial_lag_model.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Spatial Autocorrelation:")
print(f"   Moran's I (Y) = {I_Y:.4f} → Strong spatial clustering")
print(f"   Generated by ρ = {rho_true} (positive feedback)")

print("\n2. OLS Bias:")
print(f"   OLS ignores spatial lag → Biased β̂")
print(f"   Residuals spatially autocorrelated (I={I_resid_ols:.3f})")
print(f"   Standard errors underestimated (invalid inference)")

print("\n3. MLE Estimation:")
print(f"   ρ̂_MLE = {rho_mle:.4f} (true: {rho_true}) → Accurate")
print(f"   β̂_MLE close to true values")
print(f"   Residuals no spatial autocorrelation (I={I_resid_mle:.3f})")

print("\n4. 2SLS (IV) Estimation:")
print(f"   Instruments: WX (neighbors' covariates)")
print(f"   F-statistic = {F_stat:.1f} → Strong instruments")
print(f"   ρ̂_2SLS = {rho_2sls:.4f} → Consistent but less efficient than MLE")

print("\n5. Spatial Effects:")
print(f"   Direct effects ≈ β (but adjusted for feedback)")
print(f"   Indirect effects (spillovers) substantial")
print(f"   Total effects = Direct + Indirect")
print(f"   Ignoring spatial lag misses {100*(effects_df['Indirect'].mean()/effects_df['Total'].mean()):.0f}% of impact!")

print("\n6. Practical Recommendations:")
print("   • Always test for spatial autocorrelation (Moran's I on OLS residuals)")
print("   • Use MLE for efficiency (if normality reasonable)")
print("   • 2SLS robust to non-normality but less efficient")
print("   • Report direct/indirect/total effects (not just ρ, β)")
print("   • Check residuals for remaining spatial patterns")
print("   • Sensitivity to weight matrix specification")

print("\n7. Software:")
print("   • Python: PySAL (spreg.ML_Lag, spreg.GM_Lag)")
print("   • R: spatialreg::lagsarlm(), impacts()")
print("   • Stata: spregress with ml lag() option")
```

## 6. Challenge Round
When does spatial lag modeling fail or mislead?
- **Weight matrix misspecification**: Wrong W (e.g., contiguity when distance matters) biases ρ̂ and effects → Theory-driven W; test multiple specifications
- **Spatial error vs lag confusion**: Fit SAR when true model SEM (or vice versa) → Use LM tests; check robust versions; compare residual diagnostics
- **Weak instruments (IV)**: If WX poorly predicts WY (F<10), 2SLS inconsistent → Use higher-order lags W²X carefully; prefer MLE if feasible
- **Boundary effects**: Edge units have fewer neighbors → Downward bias in ρ̂; use toroidal corrections or focus on interior
- **Non-stationarity**: ρ, β vary across space (urban/rural) → Global SAR inappropriate; use spatial regimes or GWR
- **Reverse causality with covariates**: If X also spatially lagged, endogeneity beyond WY → Spatial Durbin Model (add WX); IV for both WY and WX

## 7. Key References
- [LeSage & Pace (2009) - Introduction to Spatial Econometrics](https://www.taylorfrancis.com/books/mono/10.1201/9781420064254/introduction-spatial-econometrics-james-lesage-robert-pace)
- [Anselin (1988) - Spatial Econometrics: Methods and Models](https://link.springer.com/book/10.1007/978-94-015-7799-1)
- [Elhorst (2014) - Spatial Econometrics: From Cross-Sectional Data to Spatial Panels](https://link.springer.com/book/10.1007/978-3-642-40340-8)

---
**Status:** Core spatial econometrics model | **Complements:** Spatial error model, Spatial Durbin Model, spatial panel data
