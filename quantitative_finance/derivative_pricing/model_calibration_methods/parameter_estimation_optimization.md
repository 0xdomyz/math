# Parameter Estimation & Optimization Methods

## 1. Concept Skeleton
**Definition:** Techniques to estimate model parameters (volatility, mean reversion, jump intensities) from market data; optimization algorithms minimize objective functions (least squares, maximum likelihood) to fit model predictions to observed prices  
**Purpose:** Calibrate pricing models to market reality; determine input parameters from liquid market data (option prices, historical returns); balance fit quality vs model complexity; enable consistent pricing across instruments  
**Prerequisites:** Optimization algorithms (gradient descent, Newton-Raphson), likelihood functions, regression analysis, numerical methods, model formulations (Black-Scholes, Heston, jump-diffusion)

## 2. Comparative Framing
| Method | Objective Function | Data Input | Stability | Computation | Best For |
|--------|-------------------|-----------|-----------|-----------|----------|
| **Least Squares (OLS)** | Minimize MSE | Time series returns | High | Fast | Historical volatility; simple models |
| **Maximum Likelihood (MLE)** | Maximize likelihood | Returns/prices | Medium | Moderate | Parameter uncertainty quantification; testing |
| **Least Squares (Options)** | Minimize IV differences | Option prices | Low (ill-posed) | Slow | IV surface calibration; smile fitting |
| **Regularization (Ridge/Lasso)** | MSE + λ·penalty | Time series | Very High | Fast | Avoid overfitting; sparse models |
| **Bayesian Methods** | Posterior likelihood | Prices + priors | High | Slow | Parameter uncertainty; small samples |
| **Machine Learning (NN, XGBoost)** | Custom loss | Large datasets | Medium | Moderate-Fast | Complex nonlinear relationships |

## 3. Examples + Counterexamples

**Simple: Volatility from Returns**  
Historical daily returns: σ_sample = 0.015 (1.5%). Realized volatility = annualized σ = 0.015 × √252 = 23.8%. Simple; works for liquid assets.

**Least Squares (Smile Fitting):**  
Market IVs: [15%, 18%, 20%, 22%, 25%] across strikes. SABR model fit minimizes Σ(IV_model - IV_market)². Achieves RMSE = 0.5% IV (good fit).

**MLE Example (Jump Intensity):**  
Historical log-returns; estimate jump intensity λ and jump size μ_J via MLE. Likelihood function incorporates jump component; likelihood maximized at λ = 0.05/year.

**Overfitting Trap:**  
Fit 10-parameter model to 50 option prices → Perfect in-sample fit. Forward-test on new options → Terrible performance. Solution: Regularization; Bayesian priors; cross-validation.

**Numerical Instability:**  
Calibrate stochastic volatility model by minimizing (Model IV - Market IV)² without constraints. Optimization diverges; volatility of volatility → ∞. Solution: Add constraints; use robust optimizer.

**Historical vs Implied Volatility:**  
Realized volatility from last year = 18%. Implied volatility from 1M ATM option = 22%. Discrepancy: Future vol expectations > historical. Use implied for pricing (forward-looking).

## 4. Layer Breakdown
```
Parameter Estimation & Optimization Framework:

├─ Problem Setup:
│   ├─ Objective Function: L(θ) = Loss metric to minimize/maximize
│   │   ├─ Least Squares: L(θ) = Σ(y_obs - y_model(θ))²
│   │   ├─ Maximum Likelihood: L(θ) = -∑log(f(y|θ)) (negative log-likelihood)
│   │   ├─ Regularized: L(θ) = MSE + λ × Penalty(θ)
│   │   └─ Bayesian: L(θ) = -log(Prior(θ)) - log(Likelihood(data|θ))
│   ├─ Data: Observations to fit (historical returns, option prices, time series)
│   └─ Parameters θ: Unknowns to estimate (volatility σ, mean μ, jump intensity λ, etc.)
│
├─ Least Squares Optimization:
│   ├─ Ordinary Least Squares (OLS):
│   │   ├─ Linear regression: y = β₀ + β₁x₁ + ... + βₙxₙ + ε
│   │   ├─ Objective: Minimize Σ(y_i - ŷ_i)²
│   │   ├─ Closed-form solution: β = (X'X)⁻¹X'y
│   │   ├─ Advantages: Fast; analytical solution; interpretable
│   │   ├─ Disadvantages: Assumes linear relationship; sensitive to outliers
│   │   └─ Application: Historical volatility; simple regression models
│   │
│   ├─ Nonlinear Least Squares (NLS):
│   │   ├─ Objective: Minimize Σ(y_i - f(x_i, θ))² for nonlinear f
│   │   ├─ Example: Volatility smile fit; y = IV_market, f(θ) = SABR IV
│   │   ├─ Algorithms: Gauss-Newton, Levenberg-Marquardt
│   │   │   ├─ Gauss-Newton: Fast; assumes small residuals
│   │   │   └─ Levenberg-Marquardt: Robust; damping parameter balances Newton/steepest descent
│   │   ├─ Gradient: ∇L = -2X'(y - y_model(θ))
│   │   ├─ Hessian: H ≈ 2X'X (Gauss-Newton approximation)
│   │   └─ Advantages: Nonlinear flexibility; still relatively fast
│   │
│   ├─ Weighted Least Squares:
│   │   ├─ Objective: Minimize Σ w_i(y_i - ŷ_i)²
│   │   ├─ Weights w_i: Higher weight for more reliable observations
│   │   ├─ Example: Option price calibration; weight ATM higher (more liquid)
│   │   └─ Benefit: Down-weight outliers; improve fit on important data
│   │
│   └─ Robust Regression:
│       ├─ Objective: Minimize robust loss (Huber loss, MAE instead of MSE)
│       ├─ Advantage: Less sensitive to outliers than OLS
│       └─ Example: Calibrate to option prices with bid-ask bounce noise
│
├─ Maximum Likelihood Estimation (MLE):
│   ├─ Likelihood Function:
│   │   ├─ L(θ) = ∏ f(y_i|θ) = Joint probability of observing data given θ
│   │   ├─ Log-Likelihood: ℓ(θ) = Σ log(f(y_i|θ))
│   │   └─ Goal: Maximize ℓ (or minimize -ℓ)
│   ├─ For Normal Distribution:
│   │   ├─ f(y|μ,σ) = (1/(σ√(2π))) × exp(-(y-μ)²/(2σ²))
│   │   ├─ ℓ(μ,σ) = -n/2 × log(2π) - n × log(σ) - Σ(y_i - μ)²/(2σ²)
│   │   ├─ MLE: μ̂ = ȳ (sample mean); σ̂ = √(Σ(y_i - ȳ)²/n)
│   │   └─ Matches OLS for normal errors
│   ├─ For Jump-Diffusion Process:
│   │   ├─ dy = μdt + σdW + dJ (with Poisson jump component)
│   │   ├─ ℓ(μ, σ, λ, μ_J) = Σ log(f(Δy_i | parameters))
│   │   ├─ f incorporates: Diffusion part + jump probability × jump size distribution
│   │   └─ MLE numerically solves for parameters
│   ├─ Advantages:
│   │   ├─ Efficient estimators (lowest variance among unbiased estimators)
│   │   ├─ Asymptotically normal (enables confidence intervals)
│   │   ├─ Generalizes to any distribution (not just normal)
│   │   └─ Allows hypothesis testing (likelihood ratio tests)
│   └─ Disadvantages:
│       ├─ Computationally expensive (numerical optimization required)
│       ├─ Requires likelihood specification (misspecification → bias)
│       └─ May have multiple local maxima
│
├─ Optimization Algorithms:
│   ├─ Gradient Descent:
│   │   ├─ Iterative: θ_{n+1} = θ_n - α ∇L(θ_n)
│   │   ├─ Step size α: Learning rate (controls convergence speed)
│   │   ├─ Convergence: Slow but guaranteed for convex L
│   │   └─ Variants: SGD (stochastic), momentum, Adam (adaptive learning)
│   │
│   ├─ Newton-Raphson (Second-Order):
│   │   ├─ Iterative: θ_{n+1} = θ_n - H⁻¹∇L (H = Hessian)
│   │   ├─ Convergence: Fast (quadratic) near optimum
│   │   ├─ Disadvantage: Hessian computation expensive; inversion unstable
│   │   └─ Practical: Use BFGS approximation (quasi-Newton)
│   │
│   ├─ Gauss-Newton (for least squares):
│   │   ├─ Uses Hessian approximation: H ≈ 2J'J (Jacobian-based)
│   │   ├─ Fast; avoids second derivatives
│   │   └─ Common for nonlinear regression problems
│   │
│   ├─ Levenberg-Marquardt:
│   │   ├─ Hybrid: Gauss-Newton + gradient descent
│   │   ├─ Damping parameter λ: Increases during difficult regions
│   │   ├─ Robust; handles near-singular Hessian
│   │   └─ Standard for nonlinear least squares (least-sq calibration)
│   │
│   ├─ Simulated Annealing / Genetic Algorithms:
│   │   ├─ Global optimization; avoids local minima
│   │   ├─ Slower but handles non-smooth objectives
│   │   └─ Use when objective has many local minima
│   │
│   └─ Constraint Handling:
│       ├─ Unconstrained: Standard optimization
│       ├─ Box constraints: θ_min < θ < θ_max (e.g., σ > 0, 0 < ρ < 1)
│       │   └─ Solution: Parameter transformation or interior-point methods
│       ├─ Equality constraints: g(θ) = 0 (rare in calibration)
│       └─ Inequality constraints: g(θ) ≤ 0 (e.g., no-arbitrage)
│
├─ Volatility Estimation:
│   ├─ Historical Volatility (From Returns):
│   │   ├─ Daily log-returns: r_t = log(P_t / P_{t-1})
│   │   ├─ Sample std dev: σ̂ = √(Σ(r_t - r̄)² / (n-1))
│   │   ├─ Annualized: σ_annual = σ_daily × √252 (trading days/year)
│   │   ├─ Advantages: Simple; data readily available
│   │   ├─ Disadvantages: Backward-looking; ignores future expectations
│   │   └─ Estimator variance: Var(σ̂) ≈ σ²/(2n) for normal data
│   │
│   ├─ Implied Volatility (From Option Prices):
│   │   ├─ Inverse problem: Given option price, solve for σ via Black-Scholes
│   │   ├─ C_BS(S, K, T, σ, r) = market_price → Solve for σ numerically
│   │   ├─ Algorithms: Bisection (robust), Newton-Raphson (fast), Brent (best)
│   │   ├─ Advantages: Forward-looking; reflects market expectations
│   │   ├─ Disadvantages: Assumes BS model (misspecification); bid-ask noise
│   │   └─ Volatility smile: IV varies by strike (BS model inadequacy)
│   │
│   ├─ Realized Volatility (High-Frequency):
│   │   ├─ RV = √(Σ(r_{i,intraday})²) over day
│   │   ├─ More accurate than daily close-to-close (captures intraday moves)
│   │   ├─ Challenge: Microstructure noise (bid-ask bounce, staleness)
│   │   └─ Solution: Two-scale RV (Two-scales Realized Variance; Zhang et al.)
│   │
│   ├─ GARCH & Stochastic Volatility:
│   │   ├─ GARCH(1,1): σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
│   │   ├─ MLE: Maximize joint likelihood of returns (time-varying σ)
│   │   ├─ Advantages: Captures volatility clustering; dynamic estimates
│   │   └─ Application: Risk models; volatility forecasting
│   │
│   └─ Jump vs Diffusion:
│       ├─ Decompose return: R = Diffusion + Jump component
│       ├─ Jump detection: Realized vol >> IV (suggests jumps)
│       └─ MLE: Estimate jump intensity λ; jump size distribution (normal, double exponential)
│
├─ Calibration Targets:
│   ├─ Vanilla Option Prices:
│   │   ├─ Minimize: Σ w_i (C_model(K_i, T_i, θ) - C_market(K_i, T_i))²
│   │   ├─ Typical: Calibrate to ATM + 5 strikes + 3-5 maturities
│   │   ├─ Advantages: Direct prices; no IV inversion needed
│   │   └─ Disadvantages: Prices have bid-ask bounce; lower precision
│   │
│   ├─ Implied Volatility (IV) Surface:
│   │   ├─ Minimize: Σ w_i (IV_model(K_i, T_i) - IV_market(K_i, T_i))²
│   │   ├─ IV = sigma(K, T, θ) from model (e.g., SABR, local vol, Heston)
│   │   ├─ Advantages: Normalize prices; emphasize smile structure
│   │   └─ Disadvantages: IV inversion numerical; sensitive near ATM
│   │
│   ├─ Historical Time Series:
│   │   ├─ Minimize: -ℓ(returns | θ) = MLE approach
│   │   ├─ Data: Daily/weekly/monthly returns; potentially long sample
│   │   ├─ Advantages: Stable estimates; parameter uncertainty quantifiable
│   │   └─ Disadvantages: Historical vol ≠ future vol; regime change risk
│   │
│   └─ Exotic Prices:
│       ├─ Use exotic as calibration target → Test model realism
│       ├─ Example: Autocallable calibration to barrier knock-out prices
│       └─ Advantage: Validates model beyond vanilla options
│
├─ Model Risk & Stability:
│   ├─ Ill-Posed Inverse Problem:
│   │   ├─ Multiple parameter sets θ → Same option prices (non-uniqueness)
│   │   ├─ Small price changes → Large parameter changes (sensitivity)
│   │   ├─ Example: Volatility smile; multiple models fit equally well
│   │   └─ Mitigation: Regularization; Bayesian priors; stability constraints
│   │
│   ├─ Overfitting:
│   │   ├─ Too many parameters vs data points → Fit noise, not signal
│   │   ├─ In-sample fit excellent; out-of-sample terrible
│   │   ├─ Detection: Cross-validation; forward-test on new data
│   │   └─ Remedy: Regularization (Ridge/Lasso); Bayesian shrinkage; fewer parameters
│   │
│   ├─ Parameter Uncertainty:
│   │   ├─ Confidence intervals: θ̂ ± z_{α/2} SE(θ̂)
│   │   ├─ For MLE: SE(θ̂) ≈ √(Var(θ̂)) from Fisher information matrix
│   │   │   └─ I(θ) = -E[∂²ℓ/∂θ²]; Var ≈ I(θ)⁻¹
│   │   └─ Bootstrap: Resample data; reestimate θ; empirical distribution of θ̂
│   │
│   └─ Regime Changes:
│       ├─ Market structure shifts (financial crisis, volatility regime changes)
│       ├─ Fixed parameter model → Misses new regime
│       └─ Solution: Time-varying parameters; rolling window calibration; regime-switching models
│
├─ Practical Workflow:
│   ├─ Step 1: Choose Model (BS, SABR, Heston, local vol)
│   ├─ Step 2: Select Calibration Data (option prices or IV surface)
│   ├─ Step 3: Define Objective Function (MSE, MLE, weighted)
│   ├─ Step 4: Set Constraints (parameter bounds; no-arbitrage)
│   ├─ Step 5: Choose Optimizer (Levenberg-Marquardt for LS; BFGS for MLE)
│   ├─ Step 6: Optimize θ to minimize objective
│   ├─ Step 7: Validate:
│   │   ├─ In-sample fit (residual plots, R² / AIC)
│   │   ├─ Out-of-sample test (forward-test on new prices)
│   │   ├─ Stability (resample data; reestimate; compare θ)
│   │   └─ Price exotics with calibrated model; compare to market
│   └─ Step 8: Deploy (use calibrated θ for pricing)
│
└─ Software & Tools:
    ├─ Python:
    │   ├─ scipy.optimize.minimize: General nonlinear optimization
    │   ├─ scipy.optimize.least_squares: Nonlinear least squares
    │   ├─ statsmodels: MLE, GARCH, time series models
    │   └─ scikit-learn: Regularization, cross-validation
    ├─ R:
    │   ├─ optim(): General optimization
    │   ├─ fGarch: GARCH models
    │   └─ bbmle: MLE framework
    └─ Specialized:
        ├─ QuantLib (C++): Calibration engines
        └─ MATLAB: Optimization Toolbox
```

**Key Insight:** Parameter estimation = minimize objective function (LS or MLE) subject to constraints; trade-off between fit quality and stability; forward-test on new data; avoid overfitting via regularization; multiple models may fit equally → model risk; use robust algorithms (Levenberg-Marquardt, BFGS).

## 5. Mini-Project
Calibrate volatility smile using SABR model:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.stats import norm

# Set seed
np.random.seed(42)

# Generate synthetic option data (ATM IV = 20%, smile structure)
K = np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])  # Moneyness
T = 1.0  # 1 year
r = 0.03
S0 = 100
F = S0 * np.exp(r * T)  # Forward price

# True market IVs (synthetic; with smile)
atm_vol = 0.20
smile_curvature = 0.02
market_iv = atm_vol + smile_curvature * ((K/F - 1)**2)  # Parabolic smile
market_iv = market_iv.clip(0.05, 0.50)

print("="*70)
print("SABR Model Calibration: Volatility Smile Fitting")
print("="*70)
print(f"Forward Price: ${F:.2f}")
print(f"Time to Maturity: {T} year")
print(f"ATM IV: {atm_vol:.1%}")
print("")
print("Market Data (Strikes and IVs):")
print("-"*70)
for k, iv in zip(K, market_iv):
    print(f"Strike {k*F:7.1f} (Moneyness {k:4.2f}): IV = {iv:.2%}")
print("")

# SABR Model
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """
    SABR model volatility (simplified; asymptotic formula)
    alpha: ATM volatility
    beta: CEV parameter (0=normal, 1=lognormal)
    rho: correlation (FX vol)
    nu: volatility of volatility
    """
    if K == F:
        # ATM: Simplifies to alpha
        return alpha
    
    # Log-moneyness
    x = np.log(F / K)
    
    # SABR formula (Hagan et al. 2002)
    # Approximation: σ_SAB ≈ α / (F*K)^((1-β)/2) × z / χ(z) × [ 1 + term2 + term3 ]
    
    # Simplified formula for beta close to 1 (lognormal)
    if beta >= 0.99:
        # Pure lognormal approximation
        z = nu / alpha * F**beta * np.log(F / K)
        if abs(z) < 1e-6:
            chi_z = 1.0
        else:
            chi_z = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        vol = alpha * chi_z / (F*K)**(0.5 * (1 - beta)) * (
            1 +
            ((1-beta)**2 / 24) * (alpha / (F*K)**(1-beta))**2 * np.log(F/K)**2 +
            (rho * beta * nu * alpha / (4 * (F*K)**(0.5*(1-beta)))) * np.log(F/K) +
            ((2 - 3*rho**2) / 24) * nu**2 * T
        )
    else:
        # General CEV case (simpler approximation)
        vol = alpha
    
    return max(vol, 0.01)  # Floor at 1%

# Vectorized version
def sabr_vol_vec(K, F, T, alpha, beta, rho, nu):
    return np.array([sabr_vol(F, k, T, alpha, beta, rho, nu) for k in K])

# Calibration: Minimize MSE between model IV and market IV
def objective(params):
    alpha, beta, rho, nu = params
    
    # Constraints
    if alpha <= 0 or alpha > 1:
        return 1e10
    if beta < 0.5 or beta > 1:
        return 1e10
    if rho < -0.99 or rho > 0.99:
        return 1e10
    if nu <= 0 or nu > 2:
        return 1e10
    
    model_iv = sabr_vol_vec(K, F, T, alpha, beta, rho, nu)
    mse = np.mean((model_iv - market_iv)**2)
    return mse

# Initial guess
x0 = [0.20, 0.95, -0.2, 0.4]

# Optimize
result = minimize(objective, x0, method='Nelder-Mead', 
                  options={'maxiter': 5000, 'xatol': 1e-6})

alpha_opt, beta_opt, rho_opt, nu_opt = result.x

print("Optimization Results:")
print("-"*70)
print(f"Converged: {result.success}")
print(f"Iterations: {result.nit}")
print(f"Objective (MSE): {result.fun:.2e}")
print("")
print("Calibrated SABR Parameters:")
print("-"*70)
print(f"α (ATM vol):        {alpha_opt:.4f} ({alpha_opt:.2%})")
print(f"β (CEV parameter):  {beta_opt:.4f}")
print(f"ρ (correlation):    {rho_opt:.4f}")
print(f"ν (vol of vol):     {nu_opt:.4f}")
print("")

# Fit quality
model_iv_opt = sabr_vol_vec(K, F, T, alpha_opt, beta_opt, rho_opt, nu_opt)
residuals = model_iv_opt - market_iv
rmse = np.sqrt(np.mean(residuals**2))

print("Fit Quality:")
print("-"*70)
print(f"RMSE (IV):          {rmse:.2%}")
print(f"Max Error:          {np.max(np.abs(residuals)):.2%}")
print("")

# Detailed comparison
print("Calibration Results (Strike-by-Strike):")
print("-"*70)
print(f"{'Strike':<12} {'Moneyness':<12} {'Market IV':<12} {'Model IV':<12} {'Error':<12}")
print("-"*70)
for k, m_iv, md_iv, err in zip(K, market_iv, model_iv_opt, residuals):
    print(f"${k*F:8.1f}   {k:8.2f}       {m_iv:8.2%}     {md_iv:8.2%}     {err:+8.2%}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Smile fitting
ax = axes[0, 0]
strike_fine = np.linspace(0.70, 1.30, 100)
market_iv_fine = atm_vol + smile_curvature * ((strike_fine/F - 1)**2)
market_iv_fine = market_iv_fine.clip(0.05, 0.50)
model_iv_fine = sabr_vol_vec(strike_fine, F, T, alpha_opt, beta_opt, rho_opt, nu_opt)

ax.plot(strike_fine, market_iv_fine * 100, 'b-', linewidth=2, label='Market IV')
ax.plot(strike_fine, model_iv_fine * 100, 'r--', linewidth=2, label='SABR Fit')
ax.scatter(K, market_iv * 100, color='blue', s=100, marker='o', zorder=5, label='Market Data Points')
ax.scatter(K, model_iv_opt * 100, color='red', s=100, marker='x', zorder=5, label='Model Fit')

ax.set_xlabel('Moneyness (K/F)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('SABR Model Calibration to Volatility Smile')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Residuals
ax = axes[0, 1]
ax.scatter(K, residuals * 100, color='green', s=100, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axhline(rmse * 100, color='red', linestyle=':', linewidth=1, label=f'RMSE: {rmse:.2%}')
ax.axhline(-rmse * 100, color='red', linestyle=':', linewidth=1)

ax.set_xlabel('Moneyness (K/F)')
ax.set_ylabel('Residuals (%)')
ax.set_title('Calibration Residuals (Model - Market)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Parameter sensitivity (vega)
ax = axes[1, 0]
alpha_range = np.linspace(0.15, 0.30, 20)
rmse_by_alpha = []

for alpha_test in alpha_range:
    iv_test = sabr_vol_vec(K, F, T, alpha_test, beta_opt, rho_opt, nu_opt)
    rmse_test = np.sqrt(np.mean((iv_test - market_iv)**2))
    rmse_by_alpha.append(rmse_test)

ax.plot(alpha_range * 100, np.array(rmse_by_alpha) * 100, 'purple', linewidth=2)
ax.axvline(alpha_opt * 100, color='red', linestyle='--', linewidth=2, label=f'Optimal: {alpha_opt:.2%}')

ax.set_xlabel('α (ATM Volatility, %)')
ax.set_ylabel('RMSE (%)')
ax.set_title('Objective Function Sensitivity to α')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Model parameters summary
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
SABR Model Calibration Summary

Calibrated Parameters:
  α (ATM volatility)      = {alpha_opt:.4f}
  β (CEV parameter)       = {beta_opt:.4f}
  ρ (correlation)         = {rho_opt:.4f}
  ν (volatility of vol)   = {nu_opt:.4f}

Fit Quality:
  RMSE (IV)               = {rmse:.2%}
  Max Absolute Error      = {np.max(np.abs(residuals)):.2%}
  R² (approx)             = {1 - np.sum(residuals**2)/np.sum((market_iv - np.mean(market_iv))**2):.4f}

Observations:
  Data Points             = {len(K)}
  Maturities              = {T} year
  Strike Range (K/F)      = {K.min():.2f} to {K.max():.2f}

Interpretation:
  β ≈ 1: Lognormal dynamics
  ρ < 0: Negative skew (volatility smile)
  ν > 0: Stochastic volatility regime
"""

ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('parameter_estimation_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("Key Insights:")
print("="*70)
print("1. SABR calibration captures volatility smile structure")
print("2. Negative ρ reflects skew (lower strikes higher IV)")
print("3. Fit quality (RMSE) indicates model adequacy")
print("4. Parameter sensitivity important for risk management")
print("5. Leverage calibrated params for exotic pricing")
```

## 6. Challenge Round
When parameter estimation fails or introduces complexity:
- **Ill-Posed Inverse Problem**: Multiple parameter sets fit equally well → Non-uniqueness; solution: Regularization (Bayesian priors); use longer data series; add constraints
- **Local Minima**: Optimization converges to local minimum not global → Wrong parameters; solution: Multiple initial guesses; global optimizers (simulated annealing, genetic algorithms); verify with alternative data
- **Overfitting**: Model fits historical data perfectly; forward-test fails; solution: Regularization; cross-validation; simpler model; fewer parameters
- **Numerical Instability**: Hessian singular; optimization diverges; solution: Robust algorithms (Levenberg-Marquardt); parameter bounds; scaling
- **Model Risk**: Calibrated to vanilla options; exotics misprice; solution: Calibrate to exotic prices too; multiple calibration targets; stress test
- **Regime Changes**: Parameters estimated from pre-crisis data; crisis hits; parameters wrong; solution: Rolling window calibration; regime-switching models; frequent recalibration

## 7. Key References
- [Hagan et al.: Managing Smile Risk (2002)](https://arxiv.org/abs/math/0504418) - SABR model; volatility smile calibration; widely-used asymptotic formula
- [Press et al.: Numerical Recipes (2007)](http://numerical.recipes/) - Optimization algorithms; least squares; MLE; practical implementations
- [Wilmott: Quantitative Finance (2000)](https://www.paulwilmott.com/) - Model calibration theory; optimization methods; practical considerations

---
**Status:** Derivative Pricing Core Methodology | **Complements:** Volatility Surface, Implied Volatility, Stochastic Models
