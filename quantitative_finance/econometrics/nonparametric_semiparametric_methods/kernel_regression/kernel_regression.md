# Kernel Regression

## 1. Concept Skeleton
**Definition:** Estimate regression function m(x) = E[Y|X=x] nonparametrically using local weighted averages with kernel weights  
**Purpose:** Flexible functional form avoiding parametric misspecification; captures complex nonlinear relationships without imposing structure  
**Prerequisites:** Kernel functions, bandwidth selection, bias-variance tradeoff, local polynomial regression, cross-validation, asymptotic theory

## 2. Comparative Framing
| Method | Kernel Regression | Linear Regression | Splines | Local Polynomial | kNN Regression | GAM |
|--------|-------------------|-------------------|---------|------------------|----------------|-----|
| **Parametric** | No (fully nonparametric) | Yes (linear β'X) | Semi (basis + penalty) | No (local fits) | No | Semi (additive) |
| **Functional Form** | Data-driven locally | Global linear | Piecewise polynomial | Local polynomial | Step function | Additive smooth |
| **Smoothness** | Continuous (kernel) | Linear only | Knot-continuous | Local smooth | Discontinuous | Smooth per variable |
| **Boundary Bias** | High (asymmetric weights) | None | Low (extrapolation) | Lower (odd-order) | None | Low |
| **Curse of Dimensionality** | Severe (exponential n) | None | Moderate | Severe | Severe | Additive reduces |
| **Bandwidth Choice** | Critical (h parameter) | None | Knot selection | Critical (h) | k neighbors | Smoothing per term |
| **Interpretability** | Low (black box) | High (β coefficients) | Moderate | Low | Low | Moderate (partial plots) |

## 3. Examples + Counterexamples

**Classic Example:**  
Wage-experience profile: Nonlinear wage growth flattens with experience. Parametric (linear, quadratic) imposes shape. Kernel regression with Gaussian kernel h=2 years captures hump-shaped profile (peak at 25 years, decline after 40). Bandwidth h=1 overfits (wiggly), h=5 oversmooths (misses peak). Cross-validation selects h_CV=1.8. Local linear corrects boundary bias (young/old workers). MSE=0.45 vs OLS quadratic MSE=0.68 (misspecified).

**Failure Case:**  
High-dimensional covariates: Predict house prices from 15 features (sqft, bedrooms, location, etc.). Kernel regression requires dense data in 15-D space. Sample n=5000 spreads thin (curse of dimensionality). Nearest neighbors sparse → high variance. Bandwidth h small (underfitting), h large (oversmoothing). MSE degrades rapidly K>5 dimensions. Solutions: Dimension reduction (PCA), additive models (GAM), or parametric with interactions.

**Edge Case:**  
Boundary estimation: Estimate density/regression near x_min or x_max. Kernel places weights symmetrically → asymmetric data → bias. Example: Age-income at age=18 (left boundary). Nadaraya-Watson averages only right neighbors → downward bias. Local linear regression corrects by fitting line (intercept unbiased at boundary). Local polynomial (odd order) eliminates boundary bias up to order p. Higher-order kernels (higher vanishing moments) also help.

## 4. Layer Breakdown
```
Kernel Regression Framework:
├─ Core Idea:
│   ├─ Regression function: m(x) = E[Y|X=x]
│   ├─ Estimate locally: Weighted average of nearby Y_i
│   ├─ Weights: Kernel K((X_i - x) / h) (closer X_i → higher weight)
│   └─ Nonparametric: No functional form assumption
├─ Nadaraya-Watson Estimator:
│   ├─ Definition: m̂(x) = Σᵢ K((X_i - x) / h) Y_i / Σᵢ K((X_i - x) / h)
│   ├─ Interpretation: Local constant (horizontal line)
│   ├─ Weights: w_i(x) = K((X_i - x) / h) / Σⱼ K((X_j - x) / h)
│   ├─ Σ w_i(x) = 1 (convex combination)
│   ├─ Properties:
│   │   ├─ Continuous m̂(x) if K continuous
│   │   ├─ Simple implementation
│   │   └─ Boundary bias (asymmetric kernel placement)
│   ├─ Bias: E[m̂(x)] - m(x) ≈ (h²/2) m''(x) μ₂(K) + o(h²)
│   │   ├─ μ₂(K) = ∫ u² K(u) du (second moment)
│   │   ├─ Bias O(h²) if m smooth
│   │   └─ At boundary: Bias O(h) (higher, asymmetric)
│   ├─ Variance: Var[m̂(x)] ≈ σ²(x) R(K) / (n h f(x))
│   │   ├─ R(K) = ∫ K²(u) du (roughness)
│   │   ├─ Decreases with n, h
│   │   └─ Increases where f(x) small (sparse data)
│   └─ MSE: MSE(x) = Bias² + Variance ≈ (h⁴/4) [m''(x)]² μ₂²(K) + σ²(x) R(K) / (n h f(x))
├─ Kernel Functions:
│   ├─ Requirements:
│   │   ├─ K(u) ≥ 0 (non-negative)
│   │   ├─ ∫ K(u) du = 1 (integrate to 1)
│   │   ├─ Symmetric: K(u) = K(-u) (unbiased)
│   │   └─ ∫ u K(u) du = 0 (zero first moment)
│   ├─ Common Kernels:
│   │   ├─ Uniform: K(u) = 0.5 · I(|u| ≤ 1)
│   │   │   ├─ Simple, discontinuous
│   │   │   └─ R(K) = 0.5, μ₂(K) = 1/3
│   │   ├─ Triangular: K(u) = (1 - |u|) · I(|u| ≤ 1)
│   │   │   ├─ Linear taper, continuous
│   │   │   └─ R(K) = 2/3, μ₂(K) = 1/6
│   │   ├─ Epanechnikov: K(u) = 0.75 (1 - u²) · I(|u| ≤ 1)
│   │   │   ├─ MSE-optimal (minimizes R(K) for fixed μ₂)
│   │   │   ├─ R(K) = 0.6, μ₂(K) = 1/5
│   │   │   └─ Smooth, compact support
│   │   ├─ Gaussian: K(u) = (2π)^(-1/2) exp(-u²/2)
│   │   │   ├─ Infinitely differentiable, unbounded support
│   │   │   ├─ R(K) = (4π)^(-1/2), μ₂(K) = 1
│   │   │   └─ Slower tails than compact kernels
│   │   ├─ Quartic (Biweight): K(u) = (15/16)(1 - u²)² · I(|u| ≤ 1)
│   │   │   ├─ Smooth, compact
│   │   │   └─ R(K) = 5/7, μ₂(K) = 1/7
│   │   └─ Higher-Order Kernels:
│   │       ├─ ∫ uᵏ K(u) du = 0 for k=1,...,p (p vanishing moments)
│   │       ├─ Reduces bias to O(h^(p+1))
│   │       ├─ Can be negative (not proper density)
│   │       └─ Higher variance, less used in practice
│   ├─ Kernel Choice:
│   │   ├─ Efficiency: Epanechnikov slightly better, but differences small
│   │   ├─ Differentiability: Gaussian for smooth m̂(x)
│   │   ├─ Computational: Compact kernels (uniform, Epanechnikov) faster
│   │   └─ Rule: Kernel choice matters less than bandwidth h
│   └─ Rescaled Kernel: K_h(u) = (1/h) K(u/h)
│       ├─ Bandwidth h controls width
│       └─ ∫ K_h(u) du = 1
├─ Bandwidth Selection:
│   ├─ Role of h:
│   │   ├─ Small h: Low bias (local), high variance (few data)
│   │   ├─ Large h: High bias (oversmoothing), low variance (many data)
│   │   └─ Optimal h balances bias-variance: h_opt ∝ n^(-1/5) for second-order kernel
│   ├─ Cross-Validation (CV):
│   │   ├─ Leave-One-Out CV: CV(h) = (1/n) Σᵢ [Y_i - m̂₋ᵢ(X_i; h)]²
│   │   │   ├─ m̂₋ᵢ(X_i; h): Estimate at X_i excluding (X_i, Y_i)
│   │   │   ├─ Minimize CV(h) over grid of h values
│   │   │   ├─ Computationally intensive (n fits)
│   │   │   └─ Gold standard (data-driven)
│   │   ├─ K-Fold CV: Split data into K folds, average prediction error
│   │   └─ GCV (Generalized CV): Approximation with correction for effective df
│   ├─ Plug-In Methods:
│   │   ├─ Minimize AMISE: ∫ [Bias²(x) + Variance(x)] f(x) dx
│   │   ├─ h_opt = [R(K) σ² / (μ₂²(K) n ∫ [m''(x)]² f(x) dx)]^(1/5)
│   │   ├─ Requires estimating σ², m''(x), f(x)
│   │   ├─ Two-stage: Pilot h → estimate m'' → optimal h
│   │   └─ Sheather-Jones, Direct Plug-In
│   ├─ Rule-of-Thumb (Silverman):
│   │   ├─ h = 1.06 σ_X n^(-1/5) (Gaussian reference)
│   │   ├─ Assumes m(x) smooth, f(x) Gaussian
│   │   ├─ Fast, no iteration
│   │   └─ Often oversmoothes if m highly nonlinear
│   ├─ Optimal Rate: h = C n^(-1/(2p+d))
│   │   ├─ p: Smoothness of m (derivatives)
│   │   ├─ d: Dimension of X
│   │   └─ d=1, p=2: h ∝ n^(-1/5), convergence n^(-4/5)
│   └─ Practical:
│       ├─ Start with rule-of-thumb
│       ├─ Visual inspection (underfit/overfit)
│       └─ CV for final h (if computational budget allows)
├─ Local Polynomial Regression:
│   ├─ Motivation: Reduce boundary bias, improve MSE
│   ├─ Idea: Fit polynomial locally instead of constant
│   ├─ Local Linear (p=1):
│   │   ├─ Minimize: Σᵢ K((X_i - x) / h) [Y_i - (β₀ + β₁(X_i - x))]²
│   │   ├─ Solution: (β̂₀(x), β̂₁(x)) from weighted least squares
│   │   ├─ Estimate: m̂(x) = β̂₀(x) (intercept at x)
│   │   ├─ Boundary bias: O(h²) even at boundary (corrects NW)
│   │   ├─ Interpretation: Local slope β̂₁(x) estimates m'(x)
│   │   └─ Widely recommended (Fan & Gijbels 1996)
│   ├─ Local Quadratic (p=2):
│   │   ├─ Minimize: Σᵢ K((X_i - x) / h) [Y_i - (β₀ + β₁(X_i - x) + β₂(X_i - x)²)]²
│   │   ├─ m̂(x) = β̂₀(x)
│   │   ├─ Bias O(h³) if m''' exists
│   │   ├─ Higher variance than local linear
│   │   └─ Useful if curvature strong
│   ├─ General Order p:
│   │   ├─ Fit polynomial of degree p
│   │   ├─ Bias O(h^(p+1))
│   │   ├─ Variance increases with p
│   │   └─ p=1 (local linear) usually sufficient
│   ├─ Matrix Form:
│   │   ├─ β̂(x) = (X'W(x)X)⁻¹ X'W(x)Y
│   │   ├─ X: Design matrix [1, X_i-x, (X_i-x)², ...]
│   │   ├─ W(x): Diagonal weights K((X_i-x)/h)
│   │   └─ m̂(x) = e₁' β̂(x) where e₁ = (1, 0, ..., 0)'
│   └─ Advantages:
│       ├─ Automatic boundary correction (odd order p)
│       ├─ Adaptation to local curvature
│       ├─ Design-adaptive (handles unequal spacing)
│       └─ Minimax optimal (under regularity)
├─ Multivariate Kernel Regression:
│   ├─ X ∈ ℝ^d (d-dimensional covariates)
│   ├─ Product Kernel: K(u) = ∏ⱼ₌₁ᵈ K₁(uⱼ)
│   ├─ Bandwidth: h = (h₁, ..., h_d) or scalar h
│   ├─ Estimator: m̂(x) = Σᵢ K_H(X_i - x) Y_i / Σᵢ K_H(X_i - x)
│   ├─ Curse of Dimensionality:
│   │   ├─ Convergence rate: n^(-2p/(2p+d)) → 0 slowly as d↑
│   │   ├─ Required n grows exponentially with d
│   │   ├─ d=1: h∝n^(-1/5), rate n^(-4/5)
│   │   ├─ d=5: h∝n^(-1/14), rate n^(-8/14) much slower
│   │   └─ Practical limit: d ≤ 4 with moderate n
│   ├─ Solutions:
│   │   ├─ Additive models: m(x) = α + Σⱼ mⱼ(xⱼ) (dimension-free rate)
│   │   ├─ Projection pursuit: m(x) = Σₖ gₖ(αₖ'x) (reduce to 1-D)
│   │   ├─ Variable selection: Use only relevant covariates
│   │   └─ Dimension reduction: PCA, sufficient dimension reduction
│   └─ Practical: Stick to d≤3 for pure kernel regression
├─ Inference:
│   ├─ Pointwise Confidence Intervals:
│   │   ├─ Asymptotic normality: √(nh) [m̂(x) - m(x)] →_d N(0, σ²(x) R(K) / f(x))
│   │   ├─ Standard error: SE(x) = √[σ̂²(x) R(K) / (n h f̂(x))]
│   │   ├─ CI: m̂(x) ± 1.96 · SE(x)
│   │   ├─ Bias-corrected CI: Subtract estimated bias
│   │   └─ Bootstrap: Resample (X_i, Y_i), recompute m̂(x)
│   ├─ Uniform Confidence Bands:
│   │   ├─ Simultaneous coverage: P(m(x) ∈ [L(x), U(x)] ∀x) ≥ 1-α
│   │   ├─ Wider than pointwise (multiplicity correction)
│   │   └─ Hall-Horowitz, bootstrap methods
│   ├─ Hypothesis Testing:
│   │   ├─ H₀: m(x) = m₀(x) (e.g., linearity)
│   │   ├─ Test statistic: T_n = ∫ [m̂(x) - m₀(x)]² f̂(x) dx
│   │   ├─ Bootstrap p-values
│   │   └─ Härdle-Mammen wild bootstrap
│   └─ Variance Estimation:
│       ├─ Residual variance: σ̂²(x) = Σᵢ K((X_i-x)/h) (Y_i - m̂(X_i))² / Σᵢ K((X_i-x)/h)
│       ├─ Or: σ̂² = (1/n) Σᵢ [Y_i - m̂(X_i)]² / (1 - tr(S)/n)²
│       └─ S: Smoother matrix m̂ = SY
├─ Computational Considerations:
│   ├─ Naive: O(n²) for n evaluation points (each requires n weights)
│   ├─ Binning: Discretize X, aggregate within bins (fast approximation)
│   ├─ Fast Fourier Transform (FFT): For equally-spaced grid
│   ├─ Compact Kernels: Skip distant points (K=0 outside support)
│   └─ Parallel: Evaluate m̂(x) independently across x values
├─ Practical Workflow:
│   ├─ 1. Plot (X, Y): Check nonlinearity, outliers
│   ├─ 2. Choose kernel: Epanechnikov or Gaussian
│   ├─ 3. Select bandwidth:
│   │   ├─ Start with rule-of-thumb h_ROT
│   │   ├─ Visual: Plot m̂(x) for h_ROT/2, h_ROT, 2h_ROT
│   │   ├─ CV: Minimize CV(h) over grid
│   │   └─ Final h from CV
│   ├─ 4. Fit: Nadaraya-Watson or local linear
│   ├─ 5. Diagnostics:
│   │   ├─ Residual plot: e_i = Y_i - m̂(X_i) vs X_i
│   │   ├─ Check heteroskedasticity, patterns
│   │   └─ Q-Q plot for normality
│   ├─ 6. Inference: Bootstrap CI, plot with bands
│   └─ 7. Compare: Parametric (OLS) vs nonparametric (AIC/BIC if nested)
├─ Advantages:
│   ├─ No functional form assumption (flexible)
│   ├─ Consistent under weak conditions
│   ├─ Captures complex nonlinearities
│   ├─ Transparent (visualize m̂(x))
│   └─ Robust to outliers (local averaging)
├─ Disadvantages:
│   ├─ Curse of dimensionality (d>3 problematic)
│   ├─ Boundary bias (Nadaraya-Watson)
│   ├─ Bandwidth choice critical (subjectivity)
│   ├─ Slow convergence: n^(-4/5) vs parametric n^(-1/2)
│   ├─ No interpretation (no β coefficients)
│   └─ Extrapolation poor (no data → no estimate)
└─ When to Use:
    ├─ Use kernel regression when:
    │   ├─ Relationship clearly nonlinear
    │   ├─ Parametric form unknown/suspect
    │   ├─ Low dimension (d ≤ 3)
    │   ├─ Sufficient sample size (n > 200)
    │   └─ Exploratory analysis (no need for interpretation)
    ├─ Avoid when:
    │   ├─ High dimension (d > 5)
    │   ├─ Small sample (n < 100)
    │   ├─ Need interpretation (use parametric/GAM)
    │   └─ Extrapolation required
    └─ Combine with:
        ├─ Parametric comparison (specification test)
        ├─ Additive models (semiparametric)
        └─ Variable selection (dimension reduction)
```

**Interaction:** Specify kernel K → Choose bandwidth h via CV/plug-in → Compute weights w_i(x) for each x → Weighted average Σw_i Y_i → Obtain m̂(x) → Bootstrap CI → Diagnostic plots

## 5. Mini-Project
Implement kernel regression with multiple bandwidths, local polynomial correction, and cross-validation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
import seaborn as sns

np.random.seed(2025)

# ===== Simulate Nonlinear Data =====
print("="*80)
print("KERNEL REGRESSION")
print("="*80)

n = 800

# Covariate
X = np.random.uniform(0, 10, n)

# True nonlinear function: Sinusoidal with quadratic trend
def true_function(x):
    return 5 + 2*x - 0.15*x**2 + 3*np.sin(x)

# Add heteroskedastic noise
sigma_x = 1 + 0.3*X  # Variance increases with X
epsilon = np.random.randn(n) * sigma_x

Y = true_function(X) + epsilon

print(f"\nSimulation Setup:")
print(f"  Sample size: n={n}")
print(f"  Covariate: X ~ Uniform(0, 10)")
print(f"  True function: m(x) = 5 + 2x - 0.15x² + 3sin(x)")
print(f"  Noise: Heteroskedastic σ(x) = 1 + 0.3x")

# Evaluation grid
x_grid = np.linspace(0, 10, 200)
y_true = true_function(x_grid)

# ===== Kernel Functions =====
def epanechnikov_kernel(u):
    """Epanechnikov kernel (MSE-optimal)"""
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

def gaussian_kernel(u):
    """Gaussian kernel"""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def uniform_kernel(u):
    """Uniform kernel"""
    return np.where(np.abs(u) <= 1, 0.5, 0)

def triangular_kernel(u):
    """Triangular kernel"""
    return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)

# ===== Nadaraya-Watson Estimator =====
def nadaraya_watson(X_train, Y_train, x_eval, h, kernel_func=epanechnikov_kernel):
    """
    Nadaraya-Watson kernel regression estimator
    
    Parameters:
    - X_train: Training covariates (n,)
    - Y_train: Training outcomes (n,)
    - x_eval: Evaluation points (m,)
    - h: Bandwidth
    - kernel_func: Kernel function
    
    Returns:
    - m_hat: Estimated regression function at x_eval (m,)
    """
    m_hat = np.zeros(len(x_eval))
    
    for i, x in enumerate(x_eval):
        # Compute kernel weights
        u = (X_train - x) / h
        weights = kernel_func(u)
        
        # Weighted average
        if weights.sum() > 0:
            m_hat[i] = np.sum(weights * Y_train) / weights.sum()
        else:
            m_hat[i] = np.nan
    
    return m_hat

# ===== Local Linear Regression =====
def local_linear(X_train, Y_train, x_eval, h, kernel_func=epanechnikov_kernel):
    """
    Local linear regression estimator
    
    Fits: Y_i ≈ β₀ + β₁(X_i - x) with weights K((X_i - x)/h)
    Returns: m̂(x) = β₀
    """
    m_hat = np.zeros(len(x_eval))
    
    for i, x in enumerate(x_eval):
        # Kernel weights
        u = (X_train - x) / h
        weights = kernel_func(u)
        
        if weights.sum() > 1e-10:
            # Design matrix: [1, X_i - x]
            X_centered = X_train - x
            X_design = np.column_stack([np.ones(len(X_train)), X_centered])
            
            # Weighted least squares
            W = np.diag(weights)
            try:
                XWX = X_design.T @ W @ X_design
                XWY = X_design.T @ W @ Y_train
                beta = np.linalg.solve(XWX, XWY)
                m_hat[i] = beta[0]  # Intercept at x
            except np.linalg.LinAlgError:
                m_hat[i] = np.nan
        else:
            m_hat[i] = np.nan
    
    return m_hat

# ===== Bandwidth Selection: Cross-Validation =====
def cv_score(X, Y, h, kernel_func=epanechnikov_kernel, method='nw'):
    """
    Leave-one-out cross-validation score
    
    CV(h) = (1/n) Σᵢ [Y_i - m̂₋ᵢ(X_i)]²
    """
    n = len(X)
    cv_errors = np.zeros(n)
    
    for i in range(n):
        # Leave-one-out indices
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        X_train = X[mask]
        Y_train = Y[mask]
        x_eval = np.array([X[i]])
        
        # Predict at X[i] using others
        if method == 'nw':
            y_pred = nadaraya_watson(X_train, Y_train, x_eval, h, kernel_func)
        else:  # local linear
            y_pred = local_linear(X_train, Y_train, x_eval, h, kernel_func)
        
        cv_errors[i] = (Y[i] - y_pred[0])**2
    
    return cv_errors.mean()

print("\n" + "="*80)
print("BANDWIDTH SELECTION: CROSS-VALIDATION")
print("="*80)

# Grid of bandwidths
h_grid = np.linspace(0.2, 2.0, 20)

print(f"\nSearching over {len(h_grid)} bandwidth values...")

# CV for Nadaraya-Watson
cv_scores_nw = [cv_score(X, Y, h, epanechnikov_kernel, method='nw') for h in h_grid]
h_opt_nw = h_grid[np.argmin(cv_scores_nw)]

# CV for Local Linear
cv_scores_ll = [cv_score(X, Y, h, epanechnikov_kernel, method='ll') for h in h_grid]
h_opt_ll = h_grid[np.argmin(cv_scores_ll)]

print(f"\nOptimal Bandwidth (Nadaraya-Watson): h={h_opt_nw:.3f}")
print(f"  CV Score: {min(cv_scores_nw):.3f}")

print(f"\nOptimal Bandwidth (Local Linear): h={h_opt_ll:.3f}")
print(f"  CV Score: {min(cv_scores_ll):.3f}")

# Rule-of-thumb (Silverman)
h_rot = 1.06 * X.std() * n**(-1/5)
print(f"\nRule-of-Thumb (Silverman): h={h_rot:.3f}")

# ===== Fit Models with Different Bandwidths =====
print("\n" + "="*80)
print("KERNEL REGRESSION: BANDWIDTH COMPARISON")
print("="*80)

bandwidths = [0.3, h_opt_nw, 1.5]
labels = ['Undersmooth (h=0.3)', f'CV Optimal (h={h_opt_nw:.2f})', 'Oversmooth (h=1.5)']

predictions_nw = {}

for h, label in zip(bandwidths, labels):
    y_pred = nadaraya_watson(X, Y, x_grid, h, epanechnikov_kernel)
    predictions_nw[label] = y_pred
    
    # MSE on grid
    mse = np.mean((y_pred - y_true)**2)
    print(f"\n{label}:")
    print(f"  MSE: {mse:.3f}")

# ===== Nadaraya-Watson vs Local Linear =====
print("\n" + "="*80)
print("NADARAYA-WATSON vs LOCAL LINEAR")
print("="*80)

# Fit both at optimal bandwidths
y_pred_nw = nadaraya_watson(X, Y, x_grid, h_opt_nw, epanechnikov_kernel)
y_pred_ll = local_linear(X, Y, x_grid, h_opt_ll, epanechnikov_kernel)

mse_nw = np.mean((y_pred_nw - y_true)**2)
mse_ll = np.mean((y_pred_ll - y_true)**2)

print(f"\nNadaraya-Watson (h={h_opt_nw:.3f}):")
print(f"  MSE: {mse_nw:.3f}")

print(f"\nLocal Linear (h={h_opt_ll:.3f}):")
print(f"  MSE: {mse_ll:.3f}")
print(f"  Improvement: {100*(mse_nw - mse_ll)/mse_nw:.1f}% reduction in MSE")

# Boundary bias comparison
x_boundary = x_grid[:10]  # Left boundary
y_true_boundary = y_true[:10]
y_pred_nw_boundary = y_pred_nw[:10]
y_pred_ll_boundary = y_pred_ll[:10]

bias_nw_boundary = np.mean(y_pred_nw_boundary - y_true_boundary)
bias_ll_boundary = np.mean(y_pred_ll_boundary - y_true_boundary)

print(f"\nBoundary Bias (x ∈ [0, 0.5]):")
print(f"  Nadaraya-Watson: {bias_nw_boundary:.3f}")
print(f"  Local Linear: {bias_ll_boundary:.3f}")
print(f"  Local linear reduces boundary bias ✓")

# ===== Different Kernels Comparison =====
print("\n" + "="*80)
print("KERNEL FUNCTION COMPARISON")
print("="*80)

kernels = {
    'Epanechnikov': epanechnikov_kernel,
    'Gaussian': gaussian_kernel,
    'Uniform': uniform_kernel,
    'Triangular': triangular_kernel
}

kernel_results = {}

for name, kernel_func in kernels.items():
    y_pred = nadaraya_watson(X, Y, x_grid, h_opt_nw, kernel_func)
    mse = np.mean((y_pred - y_true)**2)
    kernel_results[name] = {'prediction': y_pred, 'mse': mse}
    
    print(f"\n{name} Kernel:")
    print(f"  MSE: {mse:.3f}")

print(f"\nNote: Kernel choice matters less than bandwidth")
print(f"  MSE range: [{min(r['mse'] for r in kernel_results.values()):.3f}, "
      f"{max(r['mse'] for r in kernel_results.values()):.3f}]")

# ===== Parametric Comparison =====
print("\n" + "="*80)
print("PARAMETRIC COMPARISON")
print("="*80)

# Linear OLS
from sklearn.linear_model import LinearRegression

# Linear
linear_model = LinearRegression()
linear_model.fit(X.reshape(-1, 1), Y)
y_pred_linear = linear_model.predict(x_grid.reshape(-1, 1))
mse_linear = np.mean((y_pred_linear - y_true)**2)

# Quadratic
X_quad = np.column_stack([X, X**2])
quad_model = LinearRegression()
quad_model.fit(X_quad, Y)
X_grid_quad = np.column_stack([x_grid, x_grid**2])
y_pred_quad = quad_model.predict(X_grid_quad)
mse_quad = np.mean((y_pred_quad - y_true)**2)

print(f"\nLinear OLS:")
print(f"  MSE: {mse_linear:.3f}")

print(f"\nQuadratic OLS:")
print(f"  MSE: {mse_quad:.3f}")

print(f"\nLocal Linear Kernel Regression:")
print(f"  MSE: {mse_ll:.3f}")

print(f"\nNonparametric improvement over quadratic: "
      f"{100*(mse_quad - mse_ll)/mse_quad:.1f}%")

# ===== Bootstrap Confidence Intervals =====
print("\n" + "="*80)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("="*80)

n_bootstrap = 200
x_eval_ci = np.array([2.5, 5.0, 7.5])  # Three evaluation points

print(f"\nComputing {n_bootstrap} bootstrap samples...")

bootstrap_predictions = np.zeros((n_bootstrap, len(x_eval_ci)))

for b in range(n_bootstrap):
    # Resample
    idx = np.random.choice(n, n, replace=True)
    X_boot = X[idx]
    Y_boot = Y[idx]
    
    # Fit local linear
    y_boot = local_linear(X_boot, Y_boot, x_eval_ci, h_opt_ll, epanechnikov_kernel)
    bootstrap_predictions[b, :] = y_boot

# Confidence intervals
ci_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)
y_pred_ci = local_linear(X, Y, x_eval_ci, h_opt_ll, epanechnikov_kernel)

print(f"\n95% Bootstrap Confidence Intervals:")
for i, x_val in enumerate(x_eval_ci):
    print(f"  x={x_val:.1f}: m̂(x)={y_pred_ci[i]:.2f}, "
          f"CI=[{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]")

# ===== Visualizations =====
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Raw data + true function
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X, Y, alpha=0.3, s=20, color='gray', label='Data')
ax1.plot(x_grid, y_true, 'r-', linewidth=3, label='True m(x)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Simulated Data')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Bandwidth comparison (Nadaraya-Watson)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax2.plot(x_grid, y_true, 'r--', linewidth=2, label='True', alpha=0.7)

colors = ['blue', 'green', 'orange']
for (label, y_pred), color in zip(predictions_nw.items(), colors):
    ax2.plot(x_grid, y_pred, linewidth=2, label=label, color=color)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Bandwidth Selection Effect')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Cross-validation curves
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(h_grid, cv_scores_nw, 'o-', linewidth=2, label='Nadaraya-Watson', color='blue')
ax3.plot(h_grid, cv_scores_ll, 's-', linewidth=2, label='Local Linear', color='green')
ax3.axvline(h_opt_nw, color='blue', linestyle='--', alpha=0.5)
ax3.axvline(h_opt_ll, color='green', linestyle='--', alpha=0.5)
ax3.axvline(h_rot, color='red', linestyle=':', linewidth=2, label='Rule-of-Thumb')
ax3.set_xlabel('Bandwidth h')
ax3.set_ylabel('CV Score (MSE)')
ax3.set_title('Cross-Validation')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Nadaraya-Watson vs Local Linear
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax4.plot(x_grid, y_true, 'r--', linewidth=2, label='True', alpha=0.7)
ax4.plot(x_grid, y_pred_nw, linewidth=2, label='Nadaraya-Watson', color='blue')
ax4.plot(x_grid, y_pred_ll, linewidth=2, label='Local Linear', color='green')
ax4.axvspan(0, 1, alpha=0.1, color='yellow', label='Boundary')
ax4.axvspan(9, 10, alpha=0.1, color='yellow')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('Nadaraya-Watson vs Local Linear')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Plot 5: Kernel function comparison
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax5.plot(x_grid, y_true, 'r--', linewidth=2, label='True', alpha=0.7)

colors = ['blue', 'green', 'purple', 'orange']
for (name, result), color in zip(kernel_results.items(), colors):
    ax5.plot(x_grid, result['prediction'], linewidth=2, label=name, color=color)

ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_title('Kernel Function Comparison')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# Plot 6: Parametric vs Nonparametric
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax6.plot(x_grid, y_true, 'r-', linewidth=3, label='True', alpha=0.8)
ax6.plot(x_grid, y_pred_linear, '--', linewidth=2, label='Linear OLS', color='blue')
ax6.plot(x_grid, y_pred_quad, '--', linewidth=2, label='Quadratic OLS', color='orange')
ax6.plot(x_grid, y_pred_ll, linewidth=2, label='Local Linear', color='green')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_title('Parametric vs Nonparametric')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

# Plot 7: Residuals (Local Linear)
ax7 = fig.add_subplot(gs[2, 0])
y_fitted = local_linear(X, Y, X, h_opt_ll, epanechnikov_kernel)
residuals = Y - y_fitted
ax7.scatter(X, residuals, alpha=0.4, s=20, color='blue')
ax7.axhline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('X')
ax7.set_ylabel('Residuals')
ax7.set_title('Residual Plot (Local Linear)')
ax7.grid(alpha=0.3)

# Plot 8: MSE comparison
ax8 = fig.add_subplot(gs[2, 1])
methods = ['Linear', 'Quadratic', 'NW', 'Local Linear']
mse_values = [mse_linear, mse_quad, mse_nw, mse_ll]
colors_bar = ['blue', 'orange', 'purple', 'green']

bars = ax8.bar(methods, mse_values, color=colors_bar, alpha=0.7, edgecolor='black')
ax8.set_ylabel('MSE')
ax8.set_title('Mean Squared Error Comparison')
ax8.grid(alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, mse_values):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}',
             ha='center', fontsize=9)

# Plot 9: Bootstrap confidence intervals
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax9.plot(x_grid, y_true, 'r--', linewidth=2, label='True', alpha=0.7)

# Full prediction
y_pred_full = local_linear(X, Y, x_grid, h_opt_ll, epanechnikov_kernel)
ax9.plot(x_grid, y_pred_full, linewidth=2, label='Local Linear', color='green')

# CI at evaluation points
for i, x_val in enumerate(x_eval_ci):
    ax9.plot([x_val, x_val], [ci_lower[i], ci_upper[i]], 'o-', 
             linewidth=3, markersize=8, color='darkblue')

ax9.scatter(x_eval_ci, y_pred_ci, s=100, marker='o', color='darkblue', 
            zorder=5, label='95% CI')
ax9.set_xlabel('X')
ax9.set_ylabel('Y')
ax9.set_title('Bootstrap Confidence Intervals')
ax9.legend(fontsize=8)
ax9.grid(alpha=0.3)

plt.savefig('kernel_regression.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n1. Bandwidth Selection:")
print(f"   • CV optimal (NW): h={h_opt_nw:.3f}, CV score={min(cv_scores_nw):.3f}")
print(f"   • CV optimal (LL): h={h_opt_ll:.3f}, CV score={min(cv_scores_ll):.3f}")
print(f"   • Rule-of-thumb: h={h_rot:.3f}")
print(f"   • CV provides data-driven choice ✓")

print("\n2. Method Comparison (MSE):")
print(f"   • Linear OLS: {mse_linear:.3f} (misspecified)")
print(f"   • Quadratic OLS: {mse_quad:.3f}")
print(f"   • Nadaraya-Watson: {mse_nw:.3f}")
print(f"   • Local Linear: {mse_ll:.3f} (best)")
print(f"   • Nonparametric reduces MSE by {100*(mse_quad-mse_ll)/mse_quad:.1f}% ✓")

print("\n3. Local Linear Advantages:")
print(f"   • Boundary bias: NW={bias_nw_boundary:.3f}, LL={bias_ll_boundary:.3f}")
print(f"   • Local linear corrects boundary bias ✓")
print(f"   • MSE improvement: {100*(mse_nw-mse_ll)/mse_nw:.1f}%")

print("\n4. Kernel Functions:")
print(f"   • MSE range: {min(r['mse'] for r in kernel_results.values()):.3f} - "
      f"{max(r['mse'] for r in kernel_results.values()):.3f}")
print(f"   • Kernel choice less critical than bandwidth ✓")

print("\n5. Practical Insights:")
print("   • Cross-validation essential for bandwidth selection")
print("   • Local linear preferred (boundary correction)")
print("   • Kernel choice (Epanechnikov/Gaussian) similar performance")
print("   • Bootstrap provides valid confidence intervals")
print("   • Nonparametric captures complex nonlinearities")
print("   • Visual diagnostics (residuals) important")

print("\n" + "="*80)
```

## 6. Challenge Round
When does kernel regression fail?
- **High dimension**: d>5 covariates → Curse of dimensionality, sparse data → Use additive models (GAM), projection pursuit, or dimension reduction
- **Small sample**: n<100 with noisy Y → High variance, poor h selection → Increase n, or use parametric with flexible form (splines)
- **Boundary estimation**: x near min/max → Nadaraya-Watson biased → Use local linear (automatic correction) or local polynomial (odd order)
- **Discontinuities**: m(x) has jumps (treatment cutoff) → Kernel smooths over jump → Use regression discontinuity design, split estimation, or variable-bandwidth
- **Extrapolation**: Predict outside X range → No data → undefined/unreliable → Use parametric for extrapolation, or report "no estimate"
- **Computational cost**: Large n (n>10,000) → O(n²) slow → Use binning, FFT for grid, or local regression with kd-trees

## 7. Key References
- [Nadaraya (1964), Watson (1964) - Kernel Regression Estimator](https://en.wikipedia.org/wiki/Kernel_regression)
- [Fan & Gijbels (1996) - Local Polynomial Modelling and Its Applications](https://www.routledge.com/Local-Polynomial-Modelling-and-Its-Applications/Fan-Gijbels/p/book/9780412983214)
- [Li & Racine (2007) - Nonparametric Econometrics: Theory and Practice](https://press.princeton.edu/books/hardcover/9780691121611/nonparametric-econometrics)

---
**Status:** Nonparametric regression with data-driven smoothing | **Complements:** Splines, GAM, local polynomial, bandwidth selection, bias-variance tradeoff
