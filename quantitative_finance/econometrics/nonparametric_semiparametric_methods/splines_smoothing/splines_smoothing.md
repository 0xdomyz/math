# Splines and Smoothing

## 1. Concept Skeleton
**Definition:** Piecewise polynomial functions with smooth joins (knots); estimate regression via basis expansion with penalty for roughness  
**Purpose:** Flexible nonparametric regression with automatic smoothness; balance fit and complexity via penalization  
**Prerequisites:** Basis functions, knots/breakpoints, B-splines, natural splines, penalized regression, smoothing parameter λ, cross-validation, degrees of freedom

## 2. Comparative Framing
| Method | Regression Splines | Smoothing Splines | Kernel Regression | Polynomial | P-splines | GAM |
|--------|-------------------|-------------------|-------------------|------------|-----------|-----|
| **Basis** | Truncated power/B-spline | Reproducing kernel | Kernel weights | Global polynomial | B-spline + penalty | Multiple bases |
| **Knots** | Fixed (user choice) | All data points | Bandwidth h | None | Fixed + penalty | Per term |
| **Smoothness** | Continuous derivatives | Minimizes ∫[m''(x)]² | Kernel-dependent | Degree p | Penalty on differences | Component-wise |
| **Penalty** | None (basis only) | Integrated squared 2nd deriv | None | None | Difference penalty | Per smooth term |
| **Degrees of Freedom** | # basis functions | Trace of smoother | Effective n | p+1 | λ-dependent | Sum across terms |
| **Extrapolation** | Polynomial (caution) | Linear beyond range | Poor (undefined) | Dangerous | Linear/constant | Component-specific |
| **Computation** | Linear regression | GCV/REML for λ | Local weighted avg | OLS | Penalized LS | Backfitting |

## 3. Examples + Counterexamples

**Classic Example:**  
Temperature-ozone relationship: Nonlinear ozone concentration increases with temperature (threshold effects, saturation). Cubic splines with 5 interior knots (K=8 basis functions) capture smooth S-shaped curve. Natural splines constrain linear beyond boundaries (no wild extrapolation). Smoothing spline with λ=0.01 (via GCV) balances fit and roughness (effective df=6.2). Regression spline MSE=12.3 vs linear MSE=45.7 (73% improvement). Knot placement at quantiles (20%, 40%, 60%, 80%) distributes flexibility.

**Failure Case:**  
Stock returns prediction: Highly volatile with no smooth trend (random walk + jumps). Splines overfit noise, fitting every wiggle. λ=0 (interpolation) → MSE_train=0.1, MSE_test=8.5 (massive overfitting). Optimal λ=5 (heavy smoothing) → nearly linear fit, but still MSE_test=7.2 vs linear MSE_test=7.0. No predictable smooth function exists. Splines inappropriate—returns not smooth conditional on observables.

**Edge Case:**  
Boundary extrapolation: Predict income for age=75 when data X∈[18,65]. Cubic spline extrapolates cubically (income→∞ absurd). Natural spline extrapolates linearly (better, but still risky outside data range). Solution: (1) Natural splines for safety, (2) Report extrapolation explicitly, (3) Use domain knowledge constraints (income flattens/declines post-65), (4) Truncate predictions or expand data range.

## 4. Layer Breakdown
```
Splines and Smoothing Framework:
├─ Core Idea:
│   ├─ Spline: Piecewise polynomial with smooth joins
│   ├─ Knots: Breakpoints where polynomial pieces connect
│   ├─ Smoothness: Continuous function + derivatives up to order
│   └─ Basis representation: m(x) = Σⱼ βⱼ Bⱼ(x) (linear in parameters)
├─ Regression Splines:
│   ├─ Definition: Linear regression on basis functions
│   ├─ Model: Y = Σⱼ βⱼ Bⱼ(X) + ε
│   ├─ Estimate: β̂ via OLS on design matrix [B₁(X), ..., Bₖ(X)]
│   ├─ Knot Selection:
│   │   ├─ User-specified: Domain knowledge (changepoints)
│   │   ├─ Quantiles: K knots at sample quantiles (equal bins)
│   │   ├─ Uniform: Evenly spaced over X range
│   │   └─ Adaptive: More knots where m(x) varies (rarely used)
│   ├─ Number of Knots:
│   │   ├─ Too few: Underfit (bias)
│   │   ├─ Too many: Overfit (variance)
│   │   ├─ Rule-of-thumb: K ≈ min(n/4, 35)
│   │   └─ Select via AIC/BIC or cross-validation
│   └─ Advantages:
│       ├─ Fast (OLS on fixed basis)
│       ├─ Interpretable (coefficients)
│       └─ Standard inference (t-tests, CIs)
├─ Truncated Power Basis:
│   ├─ Definition: Polynomial + truncated polynomials at knots
│   ├─ Basis (cubic): {1, x, x², x³, (x-κ₁)₊³, ..., (x-κₖ)₊³}
│   │   ├─ (x-κ)₊ = max(x-κ, 0) (positive part)
│   │   ├─ Continuous 2nd derivative at knots
│   │   └─ K+4 basis functions (cubic with K interior knots)
│   ├─ Interpretation: β₀+β₁x+β₂x²+β₃x³ globally, adjustments at knots
│   ├─ Problem: Numerical instability (high powers)
│   └─ Solution: Use B-splines instead
├─ B-splines (Basis Splines):
│   ├─ Definition: Locally supported basis functions
│   ├─ Properties:
│   │   ├─ Non-negative: Bⱼ(x) ≥ 0
│   │   ├─ Compact support: Bⱼ(x) = 0 outside [τⱼ, τⱼ₊ₚ₊₁]
│   │   ├─ Partition of unity: Σⱼ Bⱼ(x) = 1 for all x
│   │   └─ Numerically stable
│   ├─ Construction:
│   │   ├─ Degree p (cubic: p=3)
│   │   ├─ Knot sequence: τ = {τ₁, ..., τₘ} (augmented with boundaries)
│   │   ├─ Recursive definition (Cox-de Boor):
│   │   │   ├─ B_{i,0}(x) = I(τᵢ ≤ x < τᵢ₊₁)
│   │   │   └─ B_{i,p}(x) = [(x-τᵢ)/(τᵢ₊ₚ-τᵢ)]B_{i,p-1}(x) + [(τᵢ₊ₚ₊₁-x)/(τᵢ₊ₚ₊₁-τᵢ₊₁)]B_{i+1,p-1}(x)
│   │   └─ Number of basis functions: K + p + 1
│   ├─ Advantages:
│   │   ├─ Numerically stable (no high powers)
│   │   ├─ Local influence (change βⱼ affects only local region)
│   │   ├─ Efficient computation (sparse design matrix)
│   │   └─ Standard in software (R: bs(), Python: scipy.interpolate)
│   └─ Implementation: scipy.interpolate.BSpline, patsy.bs()
├─ Natural Cubic Splines:
│   ├─ Definition: Cubic spline with linear extrapolation beyond boundaries
│   ├─ Constraints:
│   │   ├─ m''(x_min) = 0 (second derivative zero at left boundary)
│   │   ├─ m''(x_max) = 0 (second derivative zero at right boundary)
│   │   └─ Forces linear beyond data range
│   ├─ Basis Functions: K interior knots → K+2 basis functions
│   │   ├─ 2 fewer df than cubic B-splines (boundary constraints)
│   │   └─ Reduces extrapolation variance
│   ├─ Advantages:
│   │   ├─ Safe extrapolation (linear, not cubic)
│   │   ├─ Lower variance at boundaries
│   │   └─ Preferred for boundary regions
│   ├─ Disadvantages: Less flexible at boundaries (by design)
│   └─ Use: Default choice for regression splines
├─ Smoothing Splines:
│   ├─ Definition: Minimize penalized residual sum of squares
│   ├─ Objective: min_m Σᵢ [Y_i - m(X_i)]² + λ ∫ [m''(x)]² dx
│   │   ├─ First term: Fidelity to data (RSS)
│   │   ├─ Second term: Roughness penalty (integrated squared 2nd derivative)
│   │   └─ λ: Smoothing parameter (controls tradeoff)
│   ├─ Solution: Natural cubic spline with knots at all data points X_i
│   │   ├─ m̂(x) = Σᵢ βᵢ N_i(x) (natural spline basis at each X_i)
│   │   ├─ β̂ = (N'N + λΩ)⁻¹ N'Y
│   │   └─ Ω: Penalty matrix (∫ N_i''(x) N_j''(x) dx)
│   ├─ Smoothing Parameter λ:
│   │   ├─ λ=0: Interpolation (passes through all points, m̂(X_i)=Y_i)
│   │   ├─ λ→∞: Linear fit (m''→0, straight line)
│   │   ├─ Effective degrees of freedom: df_λ = tr(S_λ)
│   │   │   ├─ S_λ = N(N'N + λΩ)⁻¹N' (smoother matrix)
│   │   │   ├─ λ↓ → df↑ (more flexible)
│   │   │   └─ λ↑ → df↓ (smoother)
│   │   └─ Selection: GCV or REML
│   ├─ Generalized Cross-Validation (GCV):
│   │   ├─ GCV(λ) = n · RSS(λ) / [n - df_λ]²
│   │   ├─ Minimize GCV(λ) over λ
│   │   ├─ Approximates leave-one-out CV efficiently
│   │   └─ Automatic (no manual tuning)
│   ├─ REML (Restricted Maximum Likelihood):
│   │   ├─ Treat spline as random effects model
│   │   ├─ Estimate λ via REML (unbiased for variance components)
│   │   └─ mgcv::gam() uses REML by default
│   ├─ Advantages:
│   │   ├─ Automatic smoothness (λ via GCV/REML)
│   │   ├─ Optimal among linear smoothers (Reinsch theorem)
│   │   ├─ Bayesian interpretation (posterior mean under prior)
│   │   └─ No knot selection needed
│   └─ Disadvantages:
│       ├─ Computational: O(n³) for n knots (large n slow)
│       ├─ Overfitting risk if λ small
│       └─ Less interpretable than regression splines
├─ Penalized Splines (P-splines):
│   ├─ Idea: Combine B-spline basis with difference penalty
│   ├─ Model: Y = B(X)β + ε, minimize ||Y - Bβ||² + λ β'D'Dβ
│   │   ├─ B: B-spline basis (moderate number of knots, e.g., 10-40)
│   │   ├─ D: Difference matrix (penalizes Δᵈβ)
│   │   ├─ d=2: Second differences (Δ²βⱼ = βⱼ₊₂ - 2βⱼ₊₁ + βⱼ)
│   │   └─ λ: Smoothing parameter
│   ├─ Solution: β̂ = (B'B + λD'D)⁻¹ B'Y
│   ├─ Advantages:
│   │   ├─ Fast: Moderate basis size (not n knots)
│   │   ├─ Penalty on coefficients (simple, computationally efficient)
│   │   ├─ Flexible: Works with any basis (B-splines typical)
│   │   └─ Eilers & Marx (1996) popularized
│   ├─ Knot Selection: 10-40 knots (more than regression splines, fewer than smoothing splines)
│   ├─ λ Selection: GCV, AIC, or REML
│   └─ Use: Modern default (mgcv, gam packages)
├─ Degrees of Freedom:
│   ├─ Regression Splines: df = # basis functions = K + p + 1
│   ├─ Smoothing Splines: df_λ = tr(S_λ) (depends on λ)
│   │   ├─ λ=0: df=n (interpolation)
│   │   ├─ λ→∞: df=2 (linear)
│   │   └─ Effective df (not integer)
│   ├─ P-splines: df_λ = tr(B(B'B + λD'D)⁻¹B')
│   ├─ Interpretation: "Equivalent number of parameters"
│   └─ Use: Model selection (AIC, BIC with df), inference
├─ Model Selection:
│   ├─ Cross-Validation:
│   │   ├─ K-fold CV: Split data, minimize prediction error
│   │   ├─ LOO-CV for smoothing parameter λ
│   │   └─ CV for knot number K in regression splines
│   ├─ Information Criteria:
│   │   ├─ AIC = -2 log L + 2·df
│   │   ├─ BIC = -2 log L + log(n)·df
│   │   ├─ Use df_λ for penalized methods
│   │   └─ Compare models (lower better)
│   ├─ GCV (for λ): Efficient approximate CV
│   ├─ REML (for λ): Maximum likelihood for variance components
│   └─ Visual: Plot fit for different λ or K (check over/underfit)
├─ Inference:
│   ├─ Regression Splines:
│   │   ├─ Standard OLS inference (t-tests, F-tests)
│   │   ├─ CIs: β̂ⱼ ± t_{n-df} · SE(β̂ⱼ)
│   │   ├─ Pointwise CI for m̂(x): Ŷ(x) ± t · SE[Ŷ(x)]
│   │   └─ Simultaneous bands: Wider (Scheffé, Bonferroni)
│   ├─ Smoothing Splines:
│   │   ├─ Bayesian interpretation: m̂(x) is posterior mean
│   │   ├─ Posterior variance: Var[m̂(x)] = σ̂² · S_λ(x, x)
│   │   ├─ Pointwise CI: m̂(x) ± z_{α/2} · √Var[m̂(x)]
│   │   └─ Underestimates variability (doesn't account for λ uncertainty)
│   ├─ Bootstrap:
│   │   ├─ Resample (X_i, Y_i), refit spline
│   │   ├─ Compute percentile CIs
│   │   └─ Accounts for all uncertainty (λ, knots, coefficients)
│   └─ Hypothesis Tests:
│       ├─ H₀: m(x) is linear vs H₁: nonlinear
│       ├─ Likelihood ratio test (nested models)
│       ├─ F-test: Compare RSS (linear vs spline)
│       └─ p-value from F or bootstrap distribution
├─ Multivariate Extensions:
│   ├─ Tensor Product Splines:
│   │   ├─ Basis: B(x₁, x₂) = B₁(x₁) ⊗ B₂(x₂) (Kronecker product)
│   │   ├─ Model: m(x₁, x₂) = Σᵢⱼ βᵢⱼ B₁ᵢ(x₁) B₂ⱼ(x₂)
│   │   ├─ Knots: Grid in (x₁, x₂) space
│   │   └─ Curse of dimensionality: # basis grows as K₁·K₂·...·Kₚ
│   ├─ Thin-Plate Splines:
│   │   ├─ Penalty: ∫∫ [∂²m/∂x₁² + 2∂²m/∂x₁∂x₂ + ∂²m/∂x₂²]² dx₁ dx₂
│   │   ├─ Isotropic (rotation invariant)
│   │   ├─ Radial basis functions
│   │   └─ mgcv::s(x1, x2, bs="tp")
│   ├─ Additive Models (GAM):
│   │   ├─ m(x₁, ..., xₚ) = α + m₁(x₁) + ... + mₚ(xₚ)
│   │   ├─ Each mⱼ(xⱼ) is spline
│   │   ├─ Avoids curse of dimensionality
│   │   └─ See GAM section
│   └─ Practical: Use tensor products for d≤3, GAM for d>3
├─ Computational:
│   ├─ R:
│   │   ├─ bs(): B-splines (splines package)
│   │   ├─ ns(): Natural splines (splines package)
│   │   ├─ smooth.spline(): Smoothing splines (automatic λ via GCV)
│   │   ├─ mgcv::gam(): P-splines, smoothing splines (REML)
│   │   └─ gss: Smoothing spline ANOVA
│   ├─ Python:
│   │   ├─ scipy.interpolate: BSpline, CubicSpline
│   │   ├─ patsy: bs(), cr() for formula interface
│   │   ├─ pygam: Generalized additive models
│   │   └─ statsmodels: GLM with spline basis
│   └─ Julia:
│       ├─ BSplines.jl
│       └─ SmoothingSplines.jl
├─ Practical Workflow:
│   ├─ 1. Exploratory plot: Scatter Y vs X (assess nonlinearity)
│   ├─ 2. Choose spline type:
│   │   ├─ Regression splines: If know where knots needed
│   │   ├─ Smoothing splines: Automatic λ, no knot choice
│   │   └─ P-splines: Modern default (fast, flexible)
│   ├─ 3. Fit model:
│   │   ├─ Regression: Select K (AIC/BIC/CV)
│   │   ├─ Smoothing: λ via GCV or REML
│   │   └─ P-splines: λ via REML
│   ├─ 4. Diagnostics:
│   │   ├─ Residual plot: Check patterns
│   │   ├─ Q-Q plot: Normality
│   │   ├─ Effective df: Reasonable? (2 < df < 10 typical)
│   │   └─ Visual: Fit looks smooth, not wiggly?
│   ├─ 5. Inference: CIs (bootstrap preferred)
│   └─ 6. Sensitivity: Try different K or λ (robustness check)
├─ Advantages:
│   ├─ Flexible: Captures complex nonlinearity
│   ├─ Smooth: Continuous derivatives (not jumpy like binning)
│   ├─ Interpretable: Visualize m̂(x)
│   ├─ Inference: CIs, hypothesis tests available
│   ├─ Computationally efficient (especially P-splines)
│   └─ Automatic: λ selection via GCV/REML (no manual tuning)
├─ Disadvantages:
│   ├─ Knot selection: Regression splines require choice
│   ├─ Curse of dimensionality: Pure tensor products fail d>3
│   ├─ Extrapolation: Polynomial beyond range (use natural splines)
│   ├─ Overfitting: If λ too small or K too large
│   └─ Interpretation: No simple coefficients (unlike linear)
└─ When to Use:
    ├─ Use splines when:
    │   ├─ Relationship clearly nonlinear
    │   ├─ Want smooth function (not piecewise)
    │   ├─ Low-moderate dimension (d ≤ 3 for tensor, higher with GAM)
    │   ├─ Need inference (CIs, tests)
    │   └─ Sufficient sample (n > 100)
    ├─ Avoid when:
    │   ├─ Linear sufficient (Occam's razor)
    │   ├─ High dimension without additive structure
    │   ├─ Small sample (overfitting risk)
    │   └─ Discontinuities expected (use regression discontinuity)
    └─ Combine with:
        ├─ Parametric terms (semiparametric)
        ├─ Multiple predictors (GAM)
        └─ Penalty methods (ridge, lasso on spline coefficients)
```

**Interaction:** Choose spline type → Select knots K (regression) or λ (smoothing/P-splines) → Construct basis B(X) → Estimate β̂ via OLS or penalized LS → Compute m̂(x) = B(x)'β̂ → Diagnostics → Inference (CIs)

## 5. Mini-Project
Implement regression splines, natural splines, smoothing splines, and P-splines with automatic smoothing parameter selection:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from scipy.linalg import solve
from sklearn.linear_model import Ridge
import seaborn as sns

np.random.seed(2026)

# ===== Simulate Nonlinear Data =====
print("="*80)
print("SPLINES AND SMOOTHING")
print("="*80)

n = 600

# Covariate
X = np.random.uniform(0, 10, n)
X_sorted_idx = np.argsort(X)

# True nonlinear function: Complex shape with multiple features
def true_function(x):
    return 10 + 5*np.sin(x) + 0.5*x**2 - 0.02*x**3

# Heteroskedastic noise
sigma_x = 1.5 + 0.2*X
epsilon = np.random.randn(n) * sigma_x

Y = true_function(X) + epsilon

print(f"\nSimulation Setup:")
print(f"  Sample size: n={n}")
print(f"  Covariate: X ~ Uniform(0, 10)")
print(f"  True function: m(x) = 10 + 5sin(x) + 0.5x² - 0.02x³")
print(f"  Noise: Heteroskedastic σ(x) = 1.5 + 0.2x")

# Evaluation grid
x_grid = np.linspace(0, 10, 300)
y_true = true_function(x_grid)

# ===== B-Spline Basis Construction =====
def create_bspline_basis(x, knots, degree=3):
    """
    Create B-spline basis matrix
    
    Parameters:
    - x: Evaluation points (n,)
    - knots: Interior knots (K,)
    - degree: Spline degree (3 for cubic)
    
    Returns:
    - B: Basis matrix (n, K+degree+1)
    """
    # Augment knots with boundaries
    x_min, x_max = x.min(), x.max()
    # Add degree+1 boundary knots on each side
    knots_augmented = np.concatenate([
        [x_min]*(degree+1),
        knots,
        [x_max]*(degree+1)
    ])
    
    # Number of basis functions
    n_basis = len(knots) + degree + 1
    
    # Construct B-spline basis using scipy
    basis_matrix = np.zeros((len(x), n_basis))
    
    for i in range(n_basis):
        # Create B-spline for basis function i
        c = np.zeros(n_basis)
        c[i] = 1
        bspl = interpolate.BSpline(knots_augmented, c, degree, extrapolate=True)
        basis_matrix[:, i] = bspl(x)
    
    return basis_matrix

# ===== 1. Regression Splines (Cubic B-splines) =====
print("\n" + "="*80)
print("REGRESSION SPLINES (Cubic B-splines)")
print("="*80)

# Knot selection: Quantiles
K_values = [4, 8, 12]

regression_spline_results = {}

for K in K_values:
    # Interior knots at quantiles
    quantiles = np.linspace(0, 1, K+2)[1:-1]  # Exclude 0 and 1
    knots = np.quantile(X, quantiles)
    
    # Construct basis
    B_train = create_bspline_basis(X, knots, degree=3)
    B_grid = create_bspline_basis(x_grid, knots, degree=3)
    
    # Fit via OLS
    beta_hat = np.linalg.lstsq(B_train, Y, rcond=None)[0]
    
    # Predictions
    y_pred_train = B_train @ beta_hat
    y_pred_grid = B_grid @ beta_hat
    
    # MSE
    mse_train = np.mean((Y - y_pred_train)**2)
    mse_grid = np.mean((y_pred_grid - y_true)**2)
    
    # Degrees of freedom
    df = B_train.shape[1]
    
    # AIC / BIC
    rss = np.sum((Y - y_pred_train)**2)
    sigma2_hat = rss / (n - df)
    aic = n * np.log(rss/n) + 2*df
    bic = n * np.log(rss/n) + np.log(n)*df
    
    regression_spline_results[K] = {
        'prediction': y_pred_grid,
        'mse_train': mse_train,
        'mse_grid': mse_grid,
        'df': df,
        'aic': aic,
        'bic': bic,
        'knots': knots
    }
    
    print(f"\nK={K} interior knots:")
    print(f"  Degrees of freedom: {df}")
    print(f"  MSE (train): {mse_train:.3f}")
    print(f"  MSE (grid): {mse_grid:.3f}")
    print(f"  AIC: {aic:.1f}")
    print(f"  BIC: {bic:.1f}")

# Best by BIC
best_K = min(K_values, key=lambda k: regression_spline_results[k]['bic'])
print(f"\nBest K by BIC: {best_K}")

# ===== 2. Natural Cubic Splines =====
print("\n" + "="*80)
print("NATURAL CUBIC SPLINES")
print("="*80)

# Use scipy's CubicSpline with natural boundaries
from scipy.interpolate import CubicSpline

# Sort data for interpolation
X_sorted = X[X_sorted_idx]
Y_sorted = Y[X_sorted_idx]

# Fit natural cubic spline (interpolating through means in bins)
# For practical natural spline, use basis with boundary constraints
# Here we'll demonstrate concept with interpolation

# Create bins for smoother fit
n_bins = 20
bins = np.linspace(X.min(), X.max(), n_bins+1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_means = np.array([Y[(X >= bins[i]) & (X < bins[i+1])].mean() 
                      if np.sum((X >= bins[i]) & (X < bins[i+1])) > 0 
                      else np.nan 
                      for i in range(n_bins)])

# Remove NaN bins
valid = ~np.isnan(bin_means)
bin_centers_valid = bin_centers[valid]
bin_means_valid = bin_means[valid]

# Fit natural cubic spline
cs_natural = CubicSpline(bin_centers_valid, bin_means_valid, bc_type='natural')
y_pred_natural = cs_natural(x_grid)

mse_natural = np.mean((y_pred_natural - y_true)**2)

print(f"\nNatural Cubic Spline (via binning + interpolation):")
print(f"  MSE: {mse_natural:.3f}")
print(f"  Boundary condition: Second derivative = 0 (linear extrapolation)")

# ===== 3. Smoothing Splines with GCV =====
print("\n" + "="*80)
print("SMOOTHING SPLINES (GCV)")
print("="*80)

def smoothing_spline_gcv(X, Y, lambdas):
    """
    Smoothing spline with GCV for lambda selection
    
    Minimize: ||Y - m(X)||² + λ ∫ [m''(x)]² dx
    """
    n = len(X)
    
    # Sort data
    sorted_idx = np.argsort(X)
    X_sorted = X[sorted_idx]
    Y_sorted = Y[sorted_idx]
    
    # Natural spline basis at data points
    # Use cubic splines with knots at all X_i
    # Construct basis via B-splines
    knots = X_sorted[1:-1]  # Interior knots at all but boundary points
    B = create_bspline_basis(X_sorted, knots, degree=3)
    
    # Penalty matrix (second derivative inner products)
    # Approximate Ω via finite differences
    K_basis = B.shape[1]
    Omega = np.zeros((K_basis, K_basis))
    
    # Numerical integration of second derivatives
    x_fine = np.linspace(X.min(), X.max(), 1000)
    B_fine = create_bspline_basis(x_fine, knots, degree=3)
    dx = x_fine[1] - x_fine[0]
    
    # Second derivatives (numerical)
    B_second_deriv = np.gradient(np.gradient(B_fine, dx, axis=0), dx, axis=0)
    Omega = B_second_deriv.T @ B_second_deriv * dx
    
    gcv_scores = []
    fits = []
    dfs = []
    
    for lam in lambdas:
        # Penalized least squares
        beta_hat = solve(B.T @ B + lam * Omega, B.T @ Y_sorted, assume_a='pos')
        
        # Fitted values
        y_fitted = B @ beta_hat
        
        # Smoother matrix
        S = B @ solve(B.T @ B + lam * Omega, B.T, assume_a='pos')
        df_lambda = np.trace(S)
        
        # Residual sum of squares
        rss = np.sum((Y_sorted - y_fitted)**2)
        
        # GCV score
        gcv = (n * rss) / (n - df_lambda)**2
        
        gcv_scores.append(gcv)
        fits.append((beta_hat, B, knots))
        dfs.append(df_lambda)
    
    # Optimal lambda
    opt_idx = np.argmin(gcv_scores)
    lambda_opt = lambdas[opt_idx]
    beta_opt, B_opt, knots_opt = fits[opt_idx]
    df_opt = dfs[opt_idx]
    
    # Predict on grid
    B_grid = create_bspline_basis(x_grid, knots_opt, degree=3)
    y_pred_grid = B_grid @ beta_opt
    
    return {
        'lambda_opt': lambda_opt,
        'df': df_opt,
        'prediction': y_pred_grid,
        'gcv_scores': gcv_scores,
        'dfs': dfs
    }

# Lambda grid (log scale)
lambdas = np.logspace(-2, 3, 30)

print(f"\nSearching over {len(lambdas)} lambda values...")

smoothing_result = smoothing_spline_gcv(X, Y, lambdas)

lambda_opt = smoothing_result['lambda_opt']
df_opt = smoothing_result['df']
y_pred_smooth = smoothing_result['prediction']

mse_smooth = np.mean((y_pred_smooth - y_true)**2)

print(f"\nOptimal λ (GCV): {lambda_opt:.4f}")
print(f"  Effective df: {df_opt:.2f}")
print(f"  MSE: {mse_smooth:.3f}")

# ===== 4. P-splines (Penalized B-splines) =====
print("\n" + "="*80)
print("P-SPLINES (Penalized B-splines)")
print("="*80)

def pspline(X, Y, n_knots=20, lambdas=None, difference_order=2):
    """
    P-splines: B-spline basis with difference penalty
    
    Minimize: ||Y - Bβ||² + λ β'D'Dβ
    where D is difference matrix
    """
    # Interior knots
    knots = np.linspace(X.min(), X.max(), n_knots)[1:-1]
    
    # B-spline basis
    B = create_bspline_basis(X, knots, degree=3)
    K = B.shape[1]
    
    # Difference matrix
    if difference_order == 1:
        D = np.diff(np.eye(K), n=1, axis=0)
    elif difference_order == 2:
        D = np.diff(np.eye(K), n=2, axis=0)
    else:
        raise ValueError("difference_order must be 1 or 2")
    
    # GCV for lambda selection
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 30)
    
    n = len(X)
    gcv_scores = []
    fits = []
    dfs = []
    
    for lam in lambdas:
        # Ridge regression with penalty D'D
        beta_hat = solve(B.T @ B + lam * (D.T @ D), B.T @ Y, assume_a='pos')
        
        # Fitted values
        y_fitted = B @ beta_hat
        
        # Effective df
        S = B @ solve(B.T @ B + lam * (D.T @ D), B.T, assume_a='pos')
        df_lambda = np.trace(S)
        
        # GCV
        rss = np.sum((Y - y_fitted)**2)
        gcv = (n * rss) / (n - df_lambda)**2
        
        gcv_scores.append(gcv)
        fits.append(beta_hat)
        dfs.append(df_lambda)
    
    # Optimal
    opt_idx = np.argmin(gcv_scores)
    lambda_opt = lambdas[opt_idx]
    beta_opt = fits[opt_idx]
    df_opt = dfs[opt_idx]
    
    # Predict on grid
    B_grid = create_bspline_basis(x_grid, knots, degree=3)
    y_pred_grid = B_grid @ beta_opt
    
    return {
        'lambda_opt': lambda_opt,
        'df': df_opt,
        'prediction': y_pred_grid,
        'gcv_scores': gcv_scores,
        'dfs': dfs,
        'knots': knots
    }

pspline_result = pspline(X, Y, n_knots=20)

lambda_opt_ps = pspline_result['lambda_opt']
df_opt_ps = pspline_result['df']
y_pred_ps = pspline_result['prediction']

mse_ps = np.mean((y_pred_ps - y_true)**2)

print(f"\nP-splines (20 knots, 2nd order difference penalty):")
print(f"  Optimal λ (GCV): {lambda_opt_ps:.4f}")
print(f"  Effective df: {df_opt_ps:.2f}")
print(f"  MSE: {mse_ps:.3f}")

# ===== Comparison with Parametric =====
print("\n" + "="*80)
print("PARAMETRIC COMPARISON")
print("="*80)

# Linear
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X.reshape(-1, 1), Y)
y_pred_linear = linear_model.predict(x_grid.reshape(-1, 1))
mse_linear = np.mean((y_pred_linear - y_true)**2)

# Cubic polynomial
X_poly = np.column_stack([X, X**2, X**3])
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)
X_grid_poly = np.column_stack([x_grid, x_grid**2, x_grid**3])
y_pred_poly = poly_model.predict(X_grid_poly)
mse_poly = np.mean((y_pred_poly - y_true)**2)

print(f"\nLinear: MSE={mse_linear:.3f}")
print(f"Cubic Polynomial: MSE={mse_poly:.3f}")
print(f"Regression Spline (K={best_K}): MSE={regression_spline_results[best_K]['mse_grid']:.3f}")
print(f"Smoothing Spline: MSE={mse_smooth:.3f}")
print(f"P-spline: MSE={mse_ps:.3f}")

print(f"\nBest spline method: P-spline (lowest MSE)")
print(f"Improvement over cubic polynomial: {100*(mse_poly - mse_ps)/mse_poly:.1f}%")

# ===== Visualizations =====
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Data + true function
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X, Y, alpha=0.3, s=15, color='gray', label='Data')
ax1.plot(x_grid, y_true, 'r-', linewidth=3, label='True m(x)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Simulated Data')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Regression splines with different K
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax2.plot(x_grid, y_true, 'r--', linewidth=2, label='True', alpha=0.7)

colors = ['blue', 'green', 'orange']
for K, color in zip(K_values, colors):
    result = regression_spline_results[K]
    ax2.plot(x_grid, result['prediction'], linewidth=2, 
             label=f'K={K} (df={result["df"]})', color=color)
    # Mark knots
    knots = result['knots']
    ax2.scatter(knots, [ax2.get_ylim()[0]]*len(knots), marker='|', 
                s=100, color=color, alpha=0.5)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Regression Splines (varying K)')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: GCV curves for smoothing spline
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(smoothing_result['dfs'], smoothing_result['gcv_scores'], 'o-', 
         linewidth=2, color='blue')
ax3.axvline(df_opt, color='red', linestyle='--', linewidth=2, 
            label=f'Optimal df={df_opt:.1f}')
ax3.set_xlabel('Effective Degrees of Freedom')
ax3.set_ylabel('GCV Score')
ax3.set_title('Smoothing Spline: GCV')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xscale('log')
ax3.set_yscale('log')

# Plot 4: Method comparison
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax4.plot(x_grid, y_true, 'r-', linewidth=3, label='True', alpha=0.8)
ax4.plot(x_grid, regression_spline_results[best_K]['prediction'], 
         linewidth=2, label=f'Reg. Spline (K={best_K})', color='blue')
ax4.plot(x_grid, y_pred_smooth, linewidth=2, label='Smoothing Spline', color='green')
ax4.plot(x_grid, y_pred_ps, linewidth=2, label='P-spline', color='purple')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('Spline Methods Comparison')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Plot 5: Natural spline vs regular
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax5.plot(x_grid, y_true, 'r--', linewidth=2, label='True', alpha=0.7)
ax5.plot(x_grid, y_pred_natural, linewidth=2, label='Natural Cubic Spline', color='green')
ax5.plot(x_grid, regression_spline_results[8]['prediction'], linewidth=2, 
         label='Regular Cubic Spline (K=8)', color='blue', linestyle='--')

# Highlight boundaries
ax5.axvspan(0, 0.5, alpha=0.1, color='yellow', label='Boundary')
ax5.axvspan(9.5, 10, alpha=0.1, color='yellow')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_title('Natural vs Regular Splines')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# Plot 6: Parametric vs nonparametric
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(X, Y, alpha=0.2, s=10, color='lightgray')
ax6.plot(x_grid, y_true, 'r-', linewidth=3, label='True', alpha=0.8)
ax6.plot(x_grid, y_pred_linear, '--', linewidth=2, label='Linear', color='blue')
ax6.plot(x_grid, y_pred_poly, '--', linewidth=2, label='Cubic Poly', color='orange')
ax6.plot(x_grid, y_pred_ps, linewidth=2, label='P-spline', color='green')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_title('Parametric vs Splines')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

# Plot 7: Residuals (P-spline)
ax7 = fig.add_subplot(gs[2, 0])
y_fitted_ps = pspline(X, Y, n_knots=20, lambdas=[lambda_opt_ps])['prediction']
# Need to interpolate to get fitted values at X
from scipy.interpolate import interp1d
f_interp = interp1d(x_grid, y_pred_ps, kind='cubic', fill_value='extrapolate')
y_fitted_X = f_interp(X)
residuals_ps = Y - y_fitted_X

ax7.scatter(X, residuals_ps, alpha=0.4, s=20, color='blue')
ax7.axhline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('X')
ax7.set_ylabel('Residuals')
ax7.set_title('Residual Plot (P-spline)')
ax7.grid(alpha=0.3)

# Plot 8: MSE comparison
ax8 = fig.add_subplot(gs[2, 1])
methods = ['Linear', 'Cubic\nPoly', f'Reg\nSpline\nK={best_K}', 'Smooth\nSpline', 'P-spline']
mse_values = [mse_linear, mse_poly, regression_spline_results[best_K]['mse_grid'], 
              mse_smooth, mse_ps]
colors_bar = ['blue', 'orange', 'cyan', 'green', 'purple']

bars = ax8.bar(methods, mse_values, color=colors_bar, alpha=0.7, edgecolor='black')
ax8.set_ylabel('MSE')
ax8.set_title('Mean Squared Error Comparison')
ax8.grid(alpha=0.3, axis='y')
ax8.tick_params(axis='x', labelsize=8)

# Add value labels
for bar, val in zip(bars, mse_values):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.2f}',
             ha='center', fontsize=9)

# Plot 9: P-spline GCV curve
ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(pspline_result['dfs'], pspline_result['gcv_scores'], 'o-', 
         linewidth=2, color='purple')
ax9.axvline(df_opt_ps, color='red', linestyle='--', linewidth=2,
            label=f'Optimal df={df_opt_ps:.1f}')
ax9.set_xlabel('Effective Degrees of Freedom')
ax9.set_ylabel('GCV Score')
ax9.set_title('P-spline: GCV')
ax9.legend()
ax9.grid(alpha=0.3)
ax9.set_xscale('log')
ax9.set_yscale('log')

plt.savefig('splines_smoothing.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n1. Regression Splines:")
print(f"   • K=4: df={regression_spline_results[4]['df']}, MSE={regression_spline_results[4]['mse_grid']:.3f}, BIC={regression_spline_results[4]['bic']:.1f}")
print(f"   • K=8: df={regression_spline_results[8]['df']}, MSE={regression_spline_results[8]['mse_grid']:.3f}, BIC={regression_spline_results[8]['bic']:.1f} ✓ (best)")
print(f"   • K=12: df={regression_spline_results[12]['df']}, MSE={regression_spline_results[12]['mse_grid']:.3f}, BIC={regression_spline_results[12]['bic']:.1f}")
print(f"   • BIC selects K={best_K} (balances fit and complexity)")

print("\n2. Smoothing Splines:")
print(f"   • Optimal λ={lambda_opt:.4f} (via GCV)")
print(f"   • Effective df={df_opt:.2f}")
print(f"   • MSE={mse_smooth:.3f}")
print(f"   • Automatic smoothness selection ✓")

print("\n3. P-splines:")
print(f"   • 20 knots, 2nd order difference penalty")
print(f"   • Optimal λ={lambda_opt_ps:.4f} (via GCV)")
print(f"   • Effective df={df_opt_ps:.2f}")
print(f"   • MSE={mse_ps:.3f} (best among splines) ✓")

print("\n4. Method Comparison:")
print(f"   • Linear: MSE={mse_linear:.3f} (misspecified)")
print(f"   • Cubic Polynomial: MSE={mse_poly:.3f}")
print(f"   • Best Spline: MSE={mse_ps:.3f}")
print(f"   • Improvement: {100*(mse_poly - mse_ps)/mse_poly:.1f}% reduction ✓")

print("\n5. Practical Insights:")
print("   • GCV provides automatic λ selection (no manual tuning)")
print("   • P-splines: Modern default (fast, flexible, robust)")
print("   • Natural splines: Safe extrapolation (linear beyond range)")
print("   • BIC useful for regression spline knot selection")
print("   • Effective df interpretable (2=linear, higher=more flexible)")
print("   • All splines substantially outperform global polynomial")

print("\n" + "="*80)
```

## 6. Challenge Round
When do splines fail?
- **Overfitting**: λ→0 or K→n → Interpolation through noise → Use GCV/REML for automatic λ, BIC for K selection
- **Boundary extrapolation**: Cubic splines → wild predictions outside range → Use natural splines (linear extrapolation), or avoid extrapolation
- **Discontinuities**: True m(x) has jumps (policy changes, thresholds) → Splines smooth over → Use regression discontinuity, piecewise models, or adaptive knots
- **High dimension**: d>3 with tensor products → Curse of dimensionality → Use additive models (GAM), projection pursuit, or variable selection
- **Small sample**: n<50 with many knots → Overfitting, unstable → Reduce K, increase λ (smoothing), or use parametric
- **Computational**: Very large n (>100k) with smoothing splines → O(n³) slow → Use P-splines (moderate K), or local regression

## 7. Key References
- [Eilers & Marx (1996) - Flexible Smoothing with B-splines and Penalties](https://projecteuclid.org/euclid.ss/1038425655)
- [Wahba (1990) - Spline Models for Observational Data](https://epubs.siam.org/doi/book/10.1137/1.9781611970128)
- [Wood (2017) - Generalized Additive Models: An Introduction with R (2nd ed)](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781498728331)

---
**Status:** Flexible regression via piecewise polynomials with smoothness penalties | **Complements:** GAM, kernel regression, polynomial regression, penalized methods
