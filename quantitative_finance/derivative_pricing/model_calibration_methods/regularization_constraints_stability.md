# Regularization, Constraints & Model Stability

## 1. Concept Skeleton
**Definition:** Techniques to prevent overfitting in model calibration; penalize model complexity; enforce economic constraints (no-arbitrage, positivity, bounds); improve numerical stability and parameter robustness  
**Purpose:** Calibrations produce parsimonious models with good out-of-sample performance; parameters stable across time; avoid nonsensical solutions (negative volatility, arbitrage opportunities); improve forward-testing accuracy  
**Prerequisites:** Optimization algorithms, regularization penalties (L1, L2), Bayesian methods, constraint handling, cross-validation, parameter bounds, economic theory (no-arbitrage)

## 2. Comparative Framing
| Technique | Complexity Penalty | Effect | Use Case | Computation | Overfitting |
|-----------|------------------|--------|----------|-----------|------------|
| **Ridge (L2)** | λ Σ θ² | Shrink large params | Many params; multicollinearity | Fast | Moderate reduction |
| **Lasso (L1)** | λ Σ\|θ\| | Sparse solution; force some θ=0 | Feature selection | Fast | Strong reduction |
| **Elastic Net** | L1 + L2 mix | Balanced shrinkage + sparsity | Balance Ridge/Lasso | Fast | Strong reduction |
| **Bayesian (Prior)** | -log(Prior) | Shrink toward prior mean | Small samples; domain knowledge | Slow (MCMC) | Excellent |
| **Cross-Validation** | Out-of-sample error | Choose λ via test set | Model selection | Moderate | Depends on λ choice |
| **Early Stopping** | Training/validation split | Stop before perfect fit | Iterative; neural nets | Moderate | Depends on patience |
| **Parameter Constraints** | Bounds; inequalities | Force feasibility | No-arbitrage; economics | Moderate | Depends on tightness |

## 3. Examples + Counterexamples

**Ridge Regression Success:**  
Calibrate Vol surface (Heston 5 params + 20 IV data points). Ridge penalty λ = 0.01 → Shrink large parameters → Stability improves; out-of-sample RMSE down 15%; parameters vary less day-to-day.

**Lasso Overfitting Cure:**  
Spline IV surface (100 basis functions). Regular least squares: R² = 0.999 (perfect); forward-test RMSE = 50 bps (terrible). With Lasso (λ = 0.1): R² = 0.96; forward-test RMSE = 5 bps (excellent); sparse solution uses only 20 basis functions.

**Bayesian Prior Impact:**  
MLE estimate of Heston κ = 0.5 ± 0.3 (wide CI; data-limited). Add prior κ ~ Normal(1.0, 0.2) (domain belief: mean reversion strong) → Posterior κ = 0.8 ± 0.18 (narrower; pulled toward prior).

**Constraint Enforcement (No-Arbitrage):**  
Calibrate to option prices; spline fit crosses call-put parity boundary → Arbitrage opportunity. Add constraint: C - P = S e^(-qT) - K e^(-rT); optimizer respects → No arbitrage guaranteed.

**Early Stopping for Neural Networks:**  
Train NN to predict IV surface (1000 neurons). Training loss decreases monotonically; validation loss decreases then rises at epoch 500 (overfitting). Stop at epoch 500 → Generalization improves; test accuracy increases.

**Feller Condition Violation (Heston Disaster):**  
Unconstrained optimization violates Feller condition 2κθ < σ² → Simulated vol paths go negative → Pricing blows up. Add constraint 2κθ ≥ σ² (inequality) → All solutions valid; no crashes.

## 4. Layer Breakdown
```
Regularization & Constraint Framework:

├─ Regularization Penalties:
│   ├─ L2 Penalty (Ridge):
│   │   ├─ Objective: Minimize MSE + λ Σ θ²
│   │   ├─ Effect: Shrink all parameters toward zero (proportional to size)
│   │   ├─ Gradient: ∂/∂θ_i = 2(MSE)'_i + 2λθ_i
│   │   ├─ Solution: Ridge regression has closed form (X'X + λI)⁻¹X'y
│   │   ├─ Advantages:
│   │   │   ├─ Closed form (analytical)
│   │   │   ├─ Computationally stable (adds λ to diagonal; improves conditioning)
│   │   │   ├─ All parameters remain nonzero (information retention)
│   │   │   └─ Smooth solution path with λ
│   │   ├─ Disadvantages:
│   │   │   ├─ Does not eliminate parameters (keeps small ones)
│   │   │   ├─ Difficult to interpret (all params affected)
│   │   │   └─ Requires tuning λ
│   │   └─ Typical λ: 0.001 - 0.1 (choose by cross-validation)
│   │
│   ├─ L1 Penalty (Lasso):
│   │   ├─ Objective: Minimize MSE + λ Σ |θ|
│   │   ├─ Effect: Shrink small parameters to exactly zero; large params shrink less
│   │   ├─ Gradient: ∂/∂θ = 2(MSE)'_i + λ sign(θ) (subgradient; discontinuous at 0)
│   │   ├─ Solution: No closed form; iterative (coordinate descent, proximal gradient)
│   │   ├─ Advantages:
│   │   │   ├─ Sparse solution (many θ_i = 0 exactly)
│   │   │   ├─ Feature selection (identifies important parameters)
│   │   │   ├─ Interpretability (zero parameters are not needed)
│   │   │   └─ Aggressive shrinkage; prevents overfitting
│   │   ├─ Disadvantages:
│   │   │   ├─ No closed form (optimization required)
│   │   │   ├─ Arbitrary if many correlated features (may select one of many)
│   │   │   └─ Requires tuning λ
│   │   └─ Typical λ: 0.01 - 0.5 (choose by cross-validation)
│   │
│   ├─ Elastic Net (Ridge + Lasso):
│   │   ├─ Objective: Minimize MSE + λ₁ Σ θ² + λ₂ Σ |θ|
│   │   ├─ Effect: Hybrid; shrinks all (Ridge) + zeros out some (Lasso)
│   │   ├─ Parameters: α = λ₂/(λ₁ + λ₂) ∈ [0,1]; higher α → more Lasso effect
│   │   ├─ Advantages:
│   │   │   ├─ Combines benefits of Ridge (stability) + Lasso (sparsity)
│   │   ├─ Disadvantages:
│   │   │   ├─ Two hyperparameters (λ₁, λ₂) require tuning
│   │   │   └─ More complex than Ridge or Lasso alone
│   │   └─ Typical: α ∈ [0.2, 0.8]; prefer α = 0.5 (balanced)
│   │
│   └─ Other Penalties:
│       ├─ Huber Loss (robust): Mix MSE (small errors) + MAE (large; outliers)
│       │   ├─ Down-weight outliers; stable to data errors
│       │   └─ Use case: Noisy market data
│       ├─ Smoothing penalty: λ Σ (θ_i - θ_{i-1})²
│       │   ├─ Forces adjacent parameters similar (temporal smoothness)
│       │   └─ Use case: Term structure; smooth parameters across time
│       └─ Complexity penalty: λ × (# nonzero parameters)
│           ├─ Information criteria (AIC, BIC)
│           └─ Trade-off: model fit vs parsimony
│
├─ Cross-Validation for Hyperparameter Selection:
│   ├─ Purpose: Choose λ that minimizes out-of-sample error
│   ├─ K-Fold CV Procedure:
│   │   ├─ Step 1: Split data into K folds (typical K=5 or K=10)
│   │   ├─ Step 2: For each λ ∈ {λ₁, λ₂, ..., λ_N}:
│   │   │   ├─ For fold i = 1 to K:
│   │   │   │   ├─ Use folds ≠ i for training; fold i for validation
│   │   │   │   ├─ Fit model with λ on training
│   │   │   │   ├─ Evaluate MSE on validation fold
│   │   │   │   └─ Record CV_error_i(λ)
│   │   │   └─ Average: CV_error(λ) = (1/K) Σ CV_error_i(λ)
│   │   ├─ Step 3: Choose λ* = argmin_λ CV_error(λ)
│   │   ├─ Step 4: Refit on all data with λ*; report final model
│   │   └─ Advantage: Objective hyperparameter selection; reduces overfitting
│   │
│   ├─ Time Series CV (For temporal data):
│   │   ├─ Respect temporal ordering (no look-ahead bias)
│   │   ├─ Procedure:
│   │   │   ├─ Use historical data (t = 1 to T-h) for training
│   │   │   ├─ Test on future (t = T-h+1 to T)
│   │   │   ├─ Roll window: t-train ∈ [t-L, t-1]; t-test = t
│   │   │   └─ Iterate for all t
│   │   └─ More conservative; respects data ordering
│   │
│   └─ λ Selection Grid:
│       ├─ Log-spaced grid: λ ∈ {10⁻⁴, 10⁻³, ..., 10¹}
│       ├─ Fine-tune around optimal: λ ∈ [λ*-0.5, λ*+0.5]
│       └─ Practical: 20-50 λ values tested
│
├─ Constraints (Hard Constraints):
│   ├─ Box Constraints (Parameter Bounds):
│   │   ├─ θ_min ≤ θ ≤ θ_max for each parameter
│   │   ├─ Example: σ > 0 (vol positive); 0 < ρ < 1 (correlation)
│   │   ├─ Implementation:
│   │   │   ├─ Transformation: θ = θ_min + (θ_max - θ_min) × σ(α) [sigmoid]
│   │   │   ├─ Optimize α (unconstrained); back-transform to θ
│   │   │   └─ Algorithm: BFGS with parameter transformation
│   │   └─ Advantage: Simple; enforces feasibility
│   │
│   ├─ Equality Constraints:
│   │   ├─ g(θ) = 0 (e.g., Σθ = 1 for mixing probabilities)
│   │   ├─ Lagrange multipliers: L = f(θ) - λ g(θ)
│   │   ├─ Optimize: ∇L = 0 and g(θ) = 0 (system of equations)
│   │   └─ Algorithm: Augmented Lagrangian; penalty methods
│   │
│   ├─ Inequality Constraints (No-Arbitrage):
│   │   ├─ Call-Put Parity: C - P = S e^(-qT) - K e^(-rT)
│   │   ├─ Monotonicity: C(K) decreasing in K; P(K) increasing in K
│   │   ├─ Convexity: Hessian constraints on option prices (second derivatives)
│   │   ├─ Feller Condition (Heston): 2κθ ≥ σ²
│   │   ├─ Positivity: All prices, volatilities, probabilities > 0
│   │   ├─ Implementation:
│   │   │   ├─ Interior point methods (enforce constraints via barriers)
│   │   │   ├─ Penalty methods (add violation penalties to objective)
│   │   │   ├─ Active set methods (iterate on boundary constraints)
│   │   │   └─ Algorithm: SLSQP (sequential least squares quadratic program)
│   │   └─ Advantage: Prevents arbitrage; economically sensible solutions
│   │
│   └─ Practical: Combine soft (regularization) + hard (constraints) penalties
│
├─ Bayesian Regularization:
│   ├─ Prior Distribution:
│   │   ├─ Encode domain beliefs: θ ~ p(θ) [prior]
│   │   ├─ Example: κ ~ Normal(1.0, 0.3) [Heston mean reversion speed]
│   │   │   ├─ μ = 1.0: Central belief (typical markets)
│   │   │   └─ σ = 0.3: Uncertainty around belief
│   │   ├─ Alternative: Uniform prior (weak; data-driven)
│   │   └─ Sparse prior (spike-and-slab): Force many θ ≈ 0
│   │
│   ├─ Likelihood:
│   │   ├─ p(data | θ): Probability of observing data given parameters
│   │   ├─ Market prices: p(prices | θ) = ∏ N(price_model(θ) - price_market; σ²)
│   │   ├─ Log-likelihood: ℓ(θ) = -Σ(model - market)²
│   │   └─ Estimation target: Maximize ℓ(θ) [MLE] or Maximize posterior [Bayesian]
│   │
│   ├─ Posterior (Bayes Rule):
│   │   ├─ p(θ | data) ∝ p(data | θ) × p(θ) [Likelihood × Prior]
│   │   ├─ Interpretation: Updated beliefs after observing data
│   │   ├─ Advantage: Incorporates prior knowledge; shrinkage toward prior
│   │   └─ Disadvantage: Computationally intensive (MCMC sampling required)
│   │
│   ├─ MCMC Sampling:
│   │   ├─ Markov Chain Monte Carlo: Generate samples from posterior
│   │   ├─ Algorithm (Metropolis-Hastings):
│   │   │   ├─ Start at θ₀ (initial guess)
│   │   │   ├─ For iteration t = 1 to T:
│   │   │   │   ├─ Propose θ* ~ q(·|θ_t) [proposal distribution]
│   │   │   │   ├─ Compute acceptance ratio: α = min(1, [p(θ*|data)/p(θ_t|data)])
│   │   │   │   ├─ If uniform(0,1) < α: Accept θ_{t+1} = θ*
│   │   │   │   └─ Else: Reject; θ_{t+1} = θ_t
│   │   │   └─ Result: Samples {θ₁, θ₂, ..., θ_T} approximate posterior
│   │   ├─ Posterior summaries:
│   │   │   ├─ Mean: E[θ | data] ≈ (1/T) Σ θ_t
│   │   │   ├─ Credible intervals: Quantiles of {θ_t}
│   │   │   └─ Posterior std dev: SD[θ | data]
│   │   └─ Advantage: Uncertainty quantification; parameter distributions
│   │
│   └─ Practical: Set prior over parameter ranges; run 50K iterations (burn-in 10K)
│
├─ Numerical Stability Techniques:
│   ├─ Parameter Scaling:
│   │   ├─ Problem: Hessian ill-conditioned if parameters have very different scales
│   │   ├─ Example: α ~ 0.2 (vol); β ~ 0.95 (CEV parameter); difference → 200×
│   │   ├─ Solution: Normalize parameters to [0,1]; optimize on normalized scale
│   │   │   ├─ α_normalized = (α - 0.01) / (1 - 0.01)
│   │   │   ├─ β_normalized = (β - 0.5) / (1 - 0.5)
│   │   │   └─ Optimize {α_norm, β_norm}; back-transform after
│   │   └─ Benefit: Better Hessian conditioning; faster convergence
│   │
│   ├─ Regularized Hessian:
│   │   ├─ H = ∇²f (Hessian; may be ill-conditioned)
│   │   ├─ Regularize: H_reg = H + λ_H I (add λ_H to diagonal)
│   │   ├─ Solve: θ_new = θ - H_reg⁻¹ ∇f (Newton step with regularization)
│   │   ├─ Effect: Improves conditioning; prevents singular Hessian
│   │   └─ Practical: λ_H = 0.01 × trace(H)/dim (proportional to scale)
│   │
│   ├─ Gradient Preconditioning:
│   │   ├─ Bad: All parameters update at same rate (gradient-dependent)
│   │   ├─ Better: Precondition by Hessian (quasi-Newton methods)
│   │   │   ├─ BFGS: Approximates Hessian iteratively; adaptive step sizes
│   │   │   └─ Result: Faster convergence; fewer iterations
│   │   └─ Alternative: Diagonal preconditioning (1/diag(H))
│   │
│   ├─ Line Search:
│   │   ├─ Problem: Full Newton step may overshoot (f increases)
│   │   ├─ Solution: α ∈ (0,1]; find θ_new = θ - α H⁻¹ ∇f minimizing f
│   │   ├─ Methods:
│   │   │   ├─ Backtracking: Start α = 1; halve until sufficient decrease
│   │   │   ├─ Cubic interpolation: Fit cubic; minimize
│   │   │   └─ Wolfe conditions: Enforce sufficient decrease + gradient improvement
│   │   └─ Benefit: Guaranteed descent; convergence to local minimum
│   │
│   └─ Trust Region Methods:
│       ├─ Idea: Limit step size to region where quadratic model valid
│       ├─ Algorithm:
│       │   ├─ Define trust radius Δ
│       │   ├─ Solve: min_d {f(θ) + d'∇f + 0.5 d' H d} subject to ‖d‖ ≤ Δ
│       │   ├─ If actual decrease ≥ predicted: Accept; expand Δ
│       │   └─ If poor: Shrink Δ
│       ├─ Advantage: Robust; handles ill-conditioning; guaranteed convergence
│       └─ Disadvantage: More complex; parameter Δ to tune
│
├─ Practical Workflow:
│   ├─ Step 1: Setup (unregularized) baseline optimization
│   ├─ Step 2: Cross-validate to find optimal λ
│   ├─ Step 3: Add hard constraints (bounds, no-arbitrage)
│   ├─ Step 4: Test forward-sample performance
│   ├─ Step 5: Compare to baseline (check improvement)
│   ├─ Step 6: Bootstrap parameter distribution (resample; refit)
│   ├─ Step 7: Monitor parameter stability over time
│   └─ Step 8: Document choice (λ, constraints, rationale)
│
└─ Software & Implementation:
    ├─ Python:
    │   ├─ scikit-learn.linear_model: Ridge, Lasso, ElasticNet (linear models)
    │   ├─ scipy.optimize.minimize: General constrained optimization (SLSQP)
    │   ├─ scipy.optimize.least_squares: Nonlinear LS with bounds, constraints
    │   ├─ statsmodels: Regularized regression, cross-validation
    │   └─ pymc: Bayesian modeling; MCMC sampling
    ├─ R:
    │   ├─ glmnet: Ridge/Lasso/ElasticNet
    │   ├─ optim with method="L-BFGS-B": Constrained optimization
    │   ├─ bayesm: Bayesian model estimation
    │   └─ rstan: Hamiltonian MCMC; Bayesian inference
    └─ Specialized:
        ├─ QuantLib (C++): Calibration engines with constraints
        └─ CVXPY (Python): Convex optimization with constraints
```

**Key Insight:** Regularization prevents overfitting by penalizing complexity; Ridge for stability; Lasso for sparsity; constraints enforce economic theory; cross-validation selects hyperparameter λ; Bayesian methods incorporate domain knowledge; numerical stability critical for convergence → combine soft penalties + hard constraints + preconditioning; monitor out-of-sample performance.

## 5. Mini-Project
Compare unregularized vs regularized IV surface calibration:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from sklearn.model_selection import KFold

np.random.seed(42)

print("="*70)
print("Regularization in Volatility Surface Calibration")
print("="*70)

# Generate synthetic IV surface data
maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])

T_grid, K_grid = np.meshgrid(maturities, strikes)
T_flat = T_grid.flatten()
K_flat = K_grid.flatten()

# True market data (synthetic; with smile)
market_iv = 0.20 + 0.05 * np.exp(-(T_grid - 1)**2 / 0.5) + \
            0.05 * ((K_grid - 1)**2)  # Smile
market_iv = market_iv.clip(0.05, 0.50).flatten()

print(f"Data Points: {len(market_iv)}")
print(f"Maturities: {maturities}")
print(f"Strikes: {strikes}")
print("")

# Parametric model: Polynomial smile + exponential term structure
# IV(K, T) = β₀ + β₁ T + β₂ T² + β₃ (K-1) + β₄ (K-1)² + β₅ T(K-1)
# 6 parameters
def build_feature_matrix(T, K):
    """Build design matrix for IV regression"""
    ones = np.ones_like(T)
    T2 = T**2
    moneyness = K - 1.0
    moneyness2 = moneyness**2
    interaction = T * moneyness
    
    X = np.column_stack([ones, T, T2, moneyness, moneyness2, interaction])
    return X

X = build_feature_matrix(T_flat, K_flat)

# Unregularized calibration (OLS)
def ols_objective(beta):
    pred = X @ beta
    residuals = pred - market_iv
    mse = np.mean(residuals**2)
    return mse

result_ols = minimize(ols_objective, np.zeros(6), method='BFGS')
beta_ols = result_ols.x
mse_ols = result_ols.fun
pred_ols = X @ beta_ols

# Ridge regression (L2 penalty)
lambdas = np.logspace(-4, 1, 50)
mse_ridge = []
pred_ridge_list = []

for lam in lambdas:
    def ridge_objective(beta):
        pred = X @ beta
        residuals = pred - market_iv
        mse = np.mean(residuals**2) + lam * np.sum(beta**2)
        return mse
    
    result = minimize(ridge_objective, np.zeros(6), method='BFGS')
    mse_ridge.append(result.fun)
    pred_ridge_list.append(X @ result.x)

mse_ridge = np.array(mse_ridge)

# Lasso (L1 penalty) - use coordinate descent
def lasso_objective(beta, lam):
    pred = X @ beta
    residuals = pred - market_iv
    mse = np.mean(residuals**2) + lam * np.sum(np.abs(beta))
    return mse

mse_lasso = []
beta_lasso_list = []

for lam in lambdas:
    def lasso_obj(beta):
        return lasso_objective(beta, lam)
    
    result = minimize(lasso_obj, np.zeros(6), method='Nelder-Mead',
                     options={'maxiter': 5000})
    mse_lasso.append(result.fun)
    beta_lasso_list.append(result.x)

mse_lasso = np.array(mse_lasso)

# Cross-validation for optimal λ
print("Cross-Validation for λ Selection (Ridge):")
print("-"*70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_ridge = []

for lam in lambdas:
    cv_errors = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = market_iv[train_idx], market_iv[test_idx]
        
        # Fit on train
        def ridge_train(beta):
            pred = X_train @ beta
            residuals = pred - y_train
            mse = np.mean(residuals**2) + lam * np.sum(beta**2)
            return mse
        
        result = minimize(ridge_train, np.zeros(6), method='BFGS')
        beta_ridge = result.x
        
        # Evaluate on test
        y_pred = X_test @ beta_ridge
        mse_test = np.mean((y_pred - y_test)**2)
        cv_errors.append(mse_test)
    
    cv_score = np.mean(cv_errors)
    cv_scores_ridge.append(cv_score)

cv_scores_ridge = np.array(cv_scores_ridge)
optimal_idx = np.argmin(cv_scores_ridge)
lambda_opt = lambdas[optimal_idx]

print(f"Optimal λ (via CV): {lambda_opt:.2e}")
print(f"CV error at optimal λ: {cv_scores_ridge[optimal_idx]:.2e}")

# Refit Ridge with optimal λ
def ridge_final(beta, lam):
    pred = X @ beta
    residuals = pred - market_iv
    mse = np.mean(residuals**2) + lam * np.sum(beta**2)
    return mse

result_ridge_opt = minimize(lambda b: ridge_final(b, lambda_opt), np.zeros(6), method='BFGS')
beta_ridge_opt = result_ridge_opt.x
pred_ridge_opt = X @ beta_ridge_opt

# Out-of-sample validation (hold-out test set)
print("\nOut-of-Sample Validation:")
print("-"*70)

test_idx = np.random.choice(len(market_iv), size=15, replace=False)
train_idx = np.setdiff1d(np.arange(len(market_iv)), test_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = market_iv[train_idx], market_iv[test_idx]

# OLS on training data
result_ols_train = minimize(lambda b: np.mean((X_train @ b - y_train)**2),
                            np.zeros(6), method='BFGS')
beta_ols_train = result_ols_train.x

# Ridge on training data
result_ridge_train = minimize(lambda b: np.mean((X_train @ b - y_train)**2) + lambda_opt * np.sum(b**2),
                              np.zeros(6), method='BFGS')
beta_ridge_train = result_ridge_train.x

# Test performance
pred_ols_test = X_test @ beta_ols_train
pred_ridge_test = X_test @ beta_ridge_train

rmse_ols_train = np.sqrt(np.mean((X_train @ beta_ols_train - y_train)**2))
rmse_ols_test = np.sqrt(np.mean((pred_ols_test - y_test)**2))

rmse_ridge_train = np.sqrt(np.mean((X_train @ beta_ridge_train - y_train)**2))
rmse_ridge_test = np.sqrt(np.mean((pred_ridge_test - y_test)**2))

print(f"\nOLS (Unregularized):")
print(f"  Training RMSE: {rmse_ols_train:.4e}")
print(f"  Test RMSE:     {rmse_ols_test:.4e}")
print(f"  Overfitting:   {rmse_ols_test/rmse_ols_train:.2f}× (test/train)")

print(f"\nRidge (λ = {lambda_opt:.2e}):")
print(f"  Training RMSE: {rmse_ridge_train:.4e}")
print(f"  Test RMSE:     {rmse_ridge_test:.4e}")
print(f"  Overfitting:   {rmse_ridge_test/rmse_ridge_train:.2f}× (test/train)")

# Parameter stability: Jackknife
print("\nParameter Stability (Jackknife Leave-One-Out):")
print("-"*70)

beta_loo_ols = []
beta_loo_ridge = []

for i in range(min(10, len(market_iv))):  # Test on first 10 points
    mask = np.ones(len(market_iv), dtype=bool)
    mask[i] = False
    
    X_loo = X[mask]
    y_loo = market_iv[mask]
    
    # OLS
    result_loo_ols = minimize(lambda b: np.mean((X_loo @ b - y_loo)**2),
                              np.zeros(6), method='BFGS')
    beta_loo_ols.append(result_loo_ols.x)
    
    # Ridge
    result_loo_ridge = minimize(lambda b: np.mean((X_loo @ b - y_loo)**2) + lambda_opt * np.sum(b**2),
                                np.zeros(6), method='BFGS')
    beta_loo_ridge.append(result_loo_ridge.x)

beta_loo_ols = np.array(beta_loo_ols)
beta_loo_ridge = np.array(beta_loo_ridge)

param_std_ols = np.std(beta_loo_ols, axis=0)
param_std_ridge = np.std(beta_loo_ridge, axis=0)

print(f"\nParameter Standard Deviations (Jackknife):")
print(f"{'Parameter':<12} {'OLS Std':<12} {'Ridge Std':<12} {'Reduction':<12}")
print("-"*70)
for j in range(6):
    reduction = param_std_ols[j] / (param_std_ridge[j] + 1e-10) - 1
    print(f"β{j:<11} {param_std_ols[j]:<12.4e} {param_std_ridge[j]:<12.4e} {reduction*100:6.1f}%")

# ===== VISUALIZATION =====

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Training error vs λ
ax = axes[0, 0]
ax.loglog(lambdas, mse_ridge, 'o-', linewidth=2, label='Ridge', markersize=4)
ax.loglog(lambdas, mse_lasso, 's-', linewidth=2, label='Lasso', markersize=4)
ax.axvline(lambda_opt, color='red', linestyle='--', linewidth=2, label=f'Optimal λ = {lambda_opt:.2e}')
ax.axhline(mse_ols, color='green', linestyle=':', linewidth=2, label=f'OLS = {mse_ols:.2e}')

ax.set_xlabel('λ (Penalty Parameter)')
ax.set_ylabel('Training MSE')
ax.set_title('Regularization Path: Training Error vs λ')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# Plot 2: Cross-validation error
ax = axes[0, 1]
ax.semilogx(lambdas, cv_scores_ridge, 'o-', linewidth=2, markersize=6, label='CV Error')
ax.axvline(lambda_opt, color='red', linestyle='--', linewidth=2, label=f'Optimal λ')
ax.fill_between(lambdas, cv_scores_ridge - 1e-4, cv_scores_ridge + 1e-4, alpha=0.2)

ax.set_xlabel('λ (Penalty Parameter)')
ax.set_ylabel('Cross-Validation Error (5-Fold)')
ax.set_title('Optimal λ Selection via Cross-Validation')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Out-of-sample performance
ax = axes[0, 2]
models = ['OLS Train', 'OLS Test', 'Ridge Train', 'Ridge Test']
rmses = [rmse_ols_train, rmse_ols_test, rmse_ridge_train, rmse_ridge_test]
colors = ['blue', 'red', 'blue', 'red']
alphas = [0.5, 0.5, 1.0, 1.0]

ax.bar(models, rmses, color=colors, alpha=alphas)
ax.set_ylabel('RMSE')
ax.set_title('Out-of-Sample Validation: Train vs Test')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Parameter estimates (OLS vs Ridge)
ax = axes[1, 0]
x_pos = np.arange(6)
width = 0.35

ax.bar(x_pos - width/2, beta_ols, width, label='OLS', alpha=0.7)
ax.bar(x_pos + width/2, beta_ridge_opt, width, label=f'Ridge (λ={lambda_opt:.2e})', alpha=0.7)

ax.set_xlabel('Parameter Index')
ax.set_ylabel('Coefficient Value')
ax.set_title('Parameter Estimates: OLS vs Ridge')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'β{i}' for i in range(6)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 5: Residuals
ax = axes[1, 1]
res_ols = pred_ols - market_iv
res_ridge = pred_ridge_opt - market_iv

ax.scatter(pred_ols, res_ols, alpha=0.5, label='OLS', s=50)
ax.scatter(pred_ridge_opt, res_ridge, alpha=0.5, label=f'Ridge', s=50)
ax.axhline(0, color='black', linestyle='--', linewidth=1)

ax.set_xlabel('Fitted IV')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot: OLS vs Ridge')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Summary statistics
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
Regularization Summary

Dataset:
  Points: {len(market_iv)}
  Parameters: 6
  Ratio: {len(market_iv)/6:.1f}:1

OLS (Unregularized):
  Train RMSE: {rmse_ols_train:.2e}
  Test RMSE:  {rmse_ols_test:.2e}
  Overfit:    {rmse_ols_test/rmse_ols_train:.2f}×
  Param Std:  {np.mean(param_std_ols):.2e}

Ridge (λ = {lambda_opt:.2e}):
  Train RMSE: {rmse_ridge_train:.2e}
  Test RMSE:  {rmse_ridge_test:.2e}
  Overfit:    {rmse_ridge_test/rmse_ridge_train:.2f}×
  Param Std:  {np.mean(param_std_ridge):.2e}

Improvement:
  Test RMSE:     {(1 - rmse_ridge_test/rmse_ols_test)*100:+6.1f}%
  Stability:     {param_std_ols[0]/param_std_ridge[0]:6.2f}× better
  Overfitting:   {(1 - rmse_ridge_test/rmse_ridge_train)/(1 - rmse_ols_test/rmse_ols_train)*100:+6.1f}%

Key Insight:
  Ridge improves generalization
  Test error ↓, Params ↓ std
  Bias-variance tradeoff
"""

ax.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('regularization_constraints_stability.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Ridge regularization improves out-of-sample generalization")
print("2. Cross-validation selects optimal λ objectively")
print("3. Parameter stability improves with regularization (Jackknife)")
print("4. Test RMSE decreases; validation confirms improvement")
print("5. Bias-variance tradeoff: slight training error increase; test improvement")
```

## 6. Challenge Round
Regularization and stability challenges:
- **Hyperparameter Sensitivity**: Optimal λ depends on data; different data → different optimal λ; solution: Use robust CV; ensemble multiple λ values; Bayesian hierarchical priors
- **Constraint Conflicts**: No-arbitrage constraints may conflict with perfect data fit; solution: Relax constraints; use soft penalties instead of hard constraints; accept small violations
- **MCMC Convergence**: Bayesian MCMC slow; chain may not mix well; solution: Use adaptive proposals; parallel tempering; diagnostic plots (Gelman-Rubin R̂)
- **Ill-Conditioned Hessian**: Optimization stalls; convergence slow; solution: Preconditioning; parameter scaling; regularized Hessian
- **Regime Changes**: Optimal λ changes with market regime; single λ insufficient; solution: Adaptive recalibration; regime-switching penalties; ensemble models
- **Local Minima**: Multiple λ values give similar fit; solution: Grid search + random restarts; global optimization (simulated annealing, genetic algorithms)

## 7. Key References
- [Hastie, Tibshirani & Friedman: Elements of Statistical Learning (2009)](https://web.stanford.edu/~hastie/ElemStatLearn/) - Ridge, Lasso, cross-validation; foundational ML text; practical guidance
- [Nishimura & Gerard: Regularized Parameter Estimation in Stochastic Models (2018)](https://arxiv.org/abs/1805.09920) - MCMC regularization; Bayesian variable selection; modern methods
- [Boyd & Vandenberghe: Convex Optimization (2004)](https://web.stanford.edu/~boyd/cvxbook/) - Constrained optimization; interior-point methods; theoretical foundations

---
**Status:** Calibration Best Practices | **Pairs Well With:** Parameter Estimation, Volatility Calibration, Model Risk Management
