# Volatility Surface Calibration & Model Fitting

## 1. Concept Skeleton
**Definition:** Process of fitting derivative pricing models to market implied volatility data across strikes and maturities; captures smile/skew (IV varying by strike) and term structure (IV varying by time)  
**Purpose:** Ensure pricing model reflects market realities; price exotics consistently with vanilla market; detect arbitrage; quantify model adequacy; enable risk-neutral valuation  
**Prerequisites:** Implied volatility concept, Black-Scholes formula, forward-looking expectations, local volatility vs stochastic volatility, interpolation and extrapolation techniques

## 2. Comparative Framing
| Approach | Surface Shape | Flexibility | Arbitrage | Smoothness | Extrapolation | Best For |
|----------|---------------|-----------|-----------|-----------|-------------|----------|
| **Parametric (SABR)** | Smooth | Low | Enforced | High | Stable | Liquid markets; risk management |
| **Local Volatility** | Any shape | High | May violate | Medium | Unstable | Vanilla calibration; short-term |
| **Stochastic Vol (Heston)** | Smooth | Medium | Via parameters | High | Stable | Vol clustering; path-dependent |
| **Spline/Interpolation** | Exact (data points) | Very High | None | Low | Extrapolates poorly | Liquid spot markets |
| **ML/Neural Networks** | Any shape | Very High | None | Medium | Risky | Large data; nonlinear patterns |

## 3. Examples + Counterexamples

**Simple: Flat Volatility**  
All strikes, all maturities: σ = 18%. Black-Scholes works everywhere. Reality: Never true; observable smile/skew even in most liquid markets.

**Local Volatility Calibration:**  
Dupire formula: σ_local(K,T) = √[(∂C/∂T + r·K·∂C/∂K) / (0.5·K²·∂²C/∂K²)]  
Fit local vol surface to market option prices → Exact fit guaranteed. Forward-test: Terrible performance on exotics (local vol assumption fails).

**SABR Calibration Success:**  
Interest rate swaptions: Calibrate α, β, ρ, ν to implied vols across strikes and tenors → Good fit; prices exotics (Bermudan swaptions) reasonably well → Stable parameters.

**Extrapolation Failure:**  
Spline fit to 1M-5Y maturities; extrapolate to 10Y and 30Y → Nonsensical IV values (negative or >100%); breaks pricing. Solution: Use parametric models that extrapolate reasonably.

**Arbitrage from Poor Fit:**  
Smile too steep; butterfly arbitrage opportunity (buy OTM puts + calls; sell ATM call) → Positive PnL with zero cost. Solution: Better calibration; smooth smile; adjust bid-ask.

**Smile Reversal Issue:**  
USD/EUR FX: Smile during normal times; during crisis, reversal (ATM lowest IV, OTM highest). Same model fails both regimes. Solution: Regime-switching model; frequent recalibration; multiple models.

## 4. Layer Breakdown
```
Volatility Surface Calibration Framework:

├─ Data Preparation:
│   ├─ Market Option Data:
│   │   ├─ Input: Bid/ask prices for vanilla options
│   │   ├─ Strikes: ATM ± multiple stds (e.g., 0.70 - 1.30 moneyness)
│   │   ├─ Maturities: 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y (or market available)
│   │   ├─ Implied Volatility: Invert BS to get IV for each option
│   │   └─ Data cleaning: Remove stale prices; bid-ask bounce; outliers
│   │
│   ├─ IV Surface:
│   │   ├─ 2D grid: Strikes (x-axis) × Maturities (y-axis)
│   │   ├─ Values: Implied volatilities (z-axis)
│   │   ├─ Smoothness: Often bumpy due to bid-ask noise
│   │   ├─ Smile/Skew: IV not flat across strikes (violation of BS assumption)
│   │   │   ├─ Smile: IV higher OTM both sides (e.g., currency options)
│   │   │   ├─ Skew: IV higher on one side (e.g., equity puts higher; volatility smile)
│   │   │   └─ Term structure: IV varies with maturity (often U-shaped short-term, flat long-term)
│   │   └─ Data quality: Liquidity drops at extreme strikes/maturities; bid-ask widens
│   │
│   └─ Preprocessing:
│       ├─ Outlier removal: Flag IVs > 150% or < 5% (likely errors)
│       ├─ Bid-ask averaging: Use mid (not bid or ask separately)
│       ├─ Strike grouping: Bucket into liquid region (core) vs sparse (wings)
│       └─ Maturity alignment: Standardize to common dates (quarterly, annual, etc.)
│
├─ Calibration Models:
│   ├─ Parametric Models (Smooth Surface):
│   │   ├─ SABR Model:
│   │   │   ├─ Parameters: α (ATM vol), β (CEV), ρ (correlation), ν (vol-of-vol)
│   │   │   ├─ Volatility formula: σ_SABR(F, K, T, params) [Hagan asymptotic formula]
│   │   │   ├─ Calibration: Minimize Σ w_i (IV_model - IV_market)² → Solve for 4 params
│   │   │   ├─ Advantages: Smooth extrapolation; stable; interpretable parameters
│   │   │   ├─ Disadvantages: 4 params may not capture steep smiles; β fixed often
│   │   │   └─ Best for: Interest rate derivatives; currencies
│   │   │
│   │   ├─ Stochastic Volatility (Heston):
│   │   │   ├─ SDE: dS = μS dt + √v S dW¹; dv = κ(θ - v) dt + σ√v dW²
│   │   │   ├─ Parameters: κ (mean reversion), θ (long-run vol), σ (vol-of-vol), ρ (correlation)
│   │   │   ├─ Calibration: Minimize Σ (Price_model - Price_market)² → Monte Carlo or closed-form
│   │   │   ├─ Advantages: Vol clustering; dynamic vol; realistic; path-dependent options
│   │   │   ├─ Disadvantages: Computational cost; multiple local minima; parameter stability
│   │   │   └─ Best for: Equity derivatives; long-dated; exotic options
│   │   │
│   │   ├─ Polynomial Smile Models:
│   │   │   ├─ Formula: σ(K) = α + β(K - F) + γ(K - F)² + δ(K - F)³
│   │   │   ├─ Fitted per maturity T → Surface from polynomial + term structure
│   │   │   ├─ Advantages: Simple; fast; flexible shape
│   │   │   ├─ Disadvantages: Extrapolates poorly; no economc theory; may create arbs
│   │   │   └─ Best for: Quick fits; small adjustments to market
│   │   │
│   │   └─ Mixture Models:
│   │       ├─ Mixture of lognormals; mixture of Poisson jump sizes; etc.
│   │       ├─ Flexibility to fit fat tails + skew + kurtosis
│   │       └─ Disadvantage: Many parameters → overfitting risk
│   │
│   ├─ Local Volatility:
│   │   ├─ Concept: σ_local(S, T) time-dependent and spot-dependent
│   │   ├─ Dupire Formula:
│   │   │   ├─ σ_local²(K, T) = [∂C/∂T + r·K·∂C/∂K] / [0.5·K²·∂²C/∂K²]
│   │   │   ├─ Inputs: Market option prices C(K, T) across strikes and maturities
│   │   │   ├─ Computation: Numerically differentiate C (numerical derivatives; noisy)
│   │   │   ├─ Result: Exact fit to all vanilla options (by construction)
│   │   │   └─ Downside: Non-unique; forward-test poor; extrapolation dangerous
│   │   │
│   │   ├─ Practical Implementation:
│   │   │   ├─ Interpolate IV surface (spline or parametric model)
│   │   │   ├─ Convert IV → prices via BS formula
│   │   │   ├─ Compute derivatives (finite difference or spline slopes)
│   │   │   ├─ Apply Dupire formula → σ_local(K, T)
│   │   │   └─ Build PDE/Monte Carlo solver with local vol surface
│   │   │
│   │   ├─ Advantages: Exact calibration; simple concept
│   │   ├─ Disadvantages: Overfitting; extrapolation fails; forward-test poor
│   │   └─ Best for: Short-term barrier options; European exotics; intrinsic fitting
│   │
│   └─ Machine Learning / Neural Networks:
│       ├─ Input layer: Strike, maturity, moneyness, historical vol, vega, etc.
│       ├─ Hidden layers: Learn nonlinear IV surface mapping
│       ├─ Output layer: Implied volatility prediction
│       ├─ Advantages: Capture complex nonlinearities; fast predictions; leverage big data
│       ├─ Disadvantages: Black box; poor extrapolation; slow training; model risk
│       └─ Best for: High-dimensional data; real-time adjustments to large datasets
│
├─ Interpolation & Extrapolation:
│   ├─ Strikes:
│   │   ├─ Liquid core: ATM ± 10% moneyness → Use all data
│   │   ├─ Semi-liquid: ± 10-20% → Sparse data; may need interpolation
│   │   ├─ Illiquid wings: > ± 20% → Extrapolate; model-dependent; risk
│   │   ├─ Methods:
│   │   │   ├─ Linear interpolation: IV connects data points; not smooth
│   │   │   ├─ Cubic spline: Smooth; may oscillate (Runge's phenomenon); need damping
│   │   │   ├─ Parametric (SABR): Smooth; stable extrapolation; less oscillation
│   │   │   └─ Convex combinations: Mix liquid core with model extrapolation
│   │   └─ Extrapolation safeguards: IV floor (5%); ceiling (100%); smoothness limits
│   │
│   ├─ Maturities:
│   │   ├─ Short-term (< 1M): High noise; bid-ask bounce; sparse; careful fit
│   │   ├─ Medium (1M - 5Y): Liquid; good data; standard calibration
│   │   ├─ Long-term (> 5Y): Sparse; low liquidity; model-dependent; extrapolate
│   │   ├─ Term structure patterns:
│   │   │   ├─ Contango: IV decreasing with T (normal)
│   │   │   ├─ Backwardation: IV increasing with T (crisis)
│   │   │   ├─ Hump: IV rises then falls (intermediate volatility spike)
│   │   │   └─ Flat: Similar IV across all T (unusual; either liquid or model-fitted)
│   │   └─ Methods:
│   │       ├─ Spline along time dimension (smooth term structure)
│   │       ├─ Parametric (SABR κ parameter controls term structure)
│   │       └─ Regression (IV regressed on T; polynomial or exponential fit)
│   │
│   └─ Spot Dynamics (Forward Looking):
│       ├─ ATM IV often leads spot moves (volatility forecasting)
│       ├─ Sticky strike: IV moves with underlying spot (old assumption; less used)
│       ├─ Sticky delta: IV moves with moneyness (K/S, not absolute K); preferred
│       ├─ Sticky vega: Vega-weighted portfolio IV moves (advanced)
│       └─ Dynamic volatility surface: Reprices as spot changes; critical for risk management
│
├─ Calibration Workflow:
│   ├─ Step 1: Collect Market Data
│   │   ├─ Option prices across strikes and maturities
│   │   ├─ Compute implied volatility (via Newton-Raphson)
│   │   ├─ Inspect IV surface (plot; check for smile, skew, term structure)
│   │   └─ Quality check (outliers, stale prices, illiquid regions)
│   │
│   ├─ Step 2: Select Calibration Model
│   │   ├─ SABR: Interest rates, commodities (simple; stable)
│   │   ├─ Heston: Equities, long-dated (realistic vol dynamics)
│   │   ├─ Local vol: Quick fits; short-term exotics
│   │   ├─ Neural networks: Large datasets; real-time
│   │   └─ Consider: Asset class, data quality, exotic types to price
│   │
│   ├─ Step 3: Set Calibration Targets
│   │   ├─ Option 1: Implied volatilities (minimize IV errors)
│   │   │   ├─ Advantage: Normalized; emphasizes smile
│   │   │   ├─ Disadvantage: IV inversion numerical; sensitive near ATM
│   │   │   └─ Formula: Σ w_i(IV_model - IV_market)²
│   │   ├─ Option 2: Option prices (minimize price errors)
│   │   │   ├─ Advantage: Direct; prices have bid-ask; data-driven
│   │   │   ├─ Disadvantage: Prices scale with spot; hard to compare across strikes
│   │   │   └─ Formula: Σ w_i(Price_model - Price_market)²
│   │   ├─ Option 3: Hybrid (IV + prices + exotics)
│   │   │   ├─ Advantage: Comprehensive; tests against exotics
│   │   │   └─ Disadvantage: Multi-objective; requires balancing
│   │   └─ Weighting: w_i = 1 (equal); w_i ∝ 1/Bid_Ask (inverse bid-ask); w_i ∝ Vega (option sensitivity)
│   │
│   ├─ Step 4: Optimize Parameters
│   │   ├─ Objective: Minimize Σ w_i(model - market)²
│   │   ├─ Constraints:
│   │   │   ├─ Parameter bounds: α > 0; 0.5 < β < 1; -0.99 < ρ < 0.99; ν > 0
│   │   │   ├─ Arbitrage: Call prices decreasing in K; put-call parity; no-arbitrage vol bounds
│   │   │   └─ Smoothness: Vol surface smooth (no sudden jumps)
│   │   ├─ Algorithm:
│   │   │   ├─ Levenberg-Marquardt (for least squares; robust; fast)
│   │   │   ├─ BFGS (quasi-Newton; general optimization)
│   │   │   ├─ Simulated annealing (global search; handles local minima)
│   │   │   └─ Genetic algorithms (parallel search; exploratory)
│   │   └─ Convergence: Monitor objective; gradient norm; parameter stability
│   │
│   ├─ Step 5: Validate Fit
│   │   ├─ In-sample diagnostics:
│   │   │   ├─ R² statistic: Goodness of fit; target > 0.95
│   │   │   ├─ RMSE (residuals): RMS error vs market data
│   │   │   ├─ Max error: Check outliers; acceptable threshold set by business
│   │   │   └─ Residual plot: Look for patterns (systematic misfit; regional issues)
│   │   ├─ Out-of-sample test:
│   │   │   ├─ Hold out 20% of options as test set
│   │   │   ├─ Calibrate on remaining 80%
│   │   │   ├─ Price held-out options with calibrated model
│   │   │   ├─ Compare model prices to actual market prices
│   │   │   └─ If forward-test poor → Overfitting; reduce parameters; add regularization
│   │   ├─ Stability:
│   │   │   ├─ Recalibrate on data excluding one day → Parameters stable (not too sensitive)
│   │   │   ├─ If highly unstable → Non-unique; poor calibration target; consider different model
│   │   │   └─ Bootstrap: Resample data; reestimate; compare parameter distributions
│   │   └─ Exotic pricing:
│   │       ├─ Price exotic option with calibrated model
│   │       ├─ Compare to market quotes (if available) or dealer prices
│   │       ├─ If exotic prices off → Model inadequate; may need switching
│   │       └─ Document tracking error; set PnL limits
│   │
│   ├─ Step 6: Deploy & Monitor
│   │   ├─ Use calibrated parameters in pricing engine
│   │   ├─ Update calibration frequency:
│   │   │   ├─ Liquid markets: Daily (end-of-day)
│   │   │   ├─ Semi-liquid: Weekly
│   │   │   └─ Illiquid: Monthly or event-driven
│   │   ├─ Monitor parameter stability:
│   │   │   ├─ Plot α, β, ρ, ν over time
│   │   │   ├─ Flag jumps (sudden parameter changes; possible model failure)
│   │   │   └─ Investigate anomalies (market stress; data error; model inadequacy)
│   │   └─ Recalibration workflow:
│   │       ├─ If new options listed → Add to calibration target
│   │       ├─ If market structure changes → Consider model switch
│   │       └─ If parameters drift → Investigate cause (regime change; liquidity shift)
│   │
│   └─ Step 7: Risk Management
│       ├─ Model risk charge: Reserve capital for model error
│       ├─ Scenarios: Stress test portfolio under calibrated model with perturbed parameters
│       ├─ Scenario analysis:
│       │   ├─ Increase ν (vol-of-vol) → Greater tail risk
│       │   ├─ Decrease ρ (correlation) → More negative skew
│       │   └─ Stress parameters ± 20%
│       └─ Alternative models: Price with different model; compare; identify hedge
│
├─ Advanced Topics:
│   ├─ Multi-Curve Calibration:
│   │   ├─ Calibrate to multiple asset classes simultaneously
│   │   ├─ Interest rates (discounting); FX (translation); equities (dividends)
│   │   ├─ Correlation structure → Affects exotic prices
│   │   └─ Example: Cross-currency swaption calibration
│   │
│   ├─ Time-Varying Calibration:
│   │   ├─ Intraday: Recalibrate hourly or on major news
│   │   ├─ Parameter time-dependence: σ(t) changes with regime
│   │   ├─ Regime detection: Hidden Markov Model (HMM) to identify market states
│   │   └─ Adaptive recalibration: Switch models if regime changes
│   │
│   ├─ Regularization & Penalization:
│   │   ├─ Ridge (L2): Σ(error)² + λ Σ θ² → Shrink large parameters
│   │   ├─ Lasso (L1): Σ(error)² + λ Σ |θ| → Sparsify (force small parameters to zero)
│   │   ├─ Elastic Net: Mix L1 + L2 → Balance shrinkage + sparsity
│   │   ├─ Purpose: Reduce overfitting; improve stability; forward-test performance
│   │   └─ λ (penalty strength): Cross-validation to choose optimal λ
│   │
│   └─ Ensemble Methods:
│       ├─ Calibrate multiple models (SABR, Heston, local vol, spline)
│       ├─ Weighted average prices: w₁ Price₁ + w₂ Price₂ + ... (model averaging)
│       ├─ Advantage: Reduces model-specific risk; robust to misspecification
│       └─ Disadvantage: Slower; requires multiple calibrations
│
└─ Practical Considerations:
    ├─ Data Quality:
    │   ├─ Bid-ask bounce: Use mid prices; lag adjustment if stale
    │   ├─ Outliers: Remove or downweight (1.5 × IQR rule; domain knowledge)
    │   ├─ Missing data: Interpolate carefully or exclude (don't invent)
    │   └─ Time zones: Align times (FX trades 24/5; equity trading hours)
    │
    ├─ Computational Efficiency:
    │   ├─ Cache IV computations (don't re-invert BS repeatedly)
    │   ├─ Parallel optimization: Use multiple cores; grid search alternatives
    │   ├─ Incremental updates: Recalibrate only changed options (not all)
    │   └─ Approximations: Use asymptotic formulas (SABR); avoid Monte Carlo if possible
    │
    ├─ Governance & Audit:
    │   ├─ Document calibration: Model choice, data, objectives, constraints
    │   ├─ Backtesting: Compare model prices to subsequent market prices; track errors
    │   ├─ Independent validation: Model validation team; stress testing
    │   └─ Escalation: Alert on unusual residuals; investigate; approve recalibrations
    │
    └─ Technology Stack:
        ├─ Languages: C++ (speed); Python (flexibility); R (statistics)
        ├─ Libraries: scipy.optimize (Python); QuantLib (C++); fOptions (R)
        ├─ Visualization: Matplotlib, Plotly; 3D surface plots
        └─ Production: Real-time data feeds; calibration pipelines; pricing engines
```

**Key Insight:** IV surface calibration balances fidelity (fit market data) with stability (extrapolate reasonably); parametric models provide smooth stable extrapolation; local vol fits perfectly but fails forward; model choice drives exotic pricing → validate on forward prices; multi-objective optimization (vanilla + exotic targets) improves robustness.

## 5. Mini-Project
Calibrate and compare multiple models to interest rate swaption IV surface:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

print("="*70)
print("Interest Rate Swaption IV Surface Calibration")
print("="*70)

# Synthetic swaption IV surface (realistic shape)
# Tenors (years to expiration) and underlying swap lengths
tenors = np.array([0.5, 1, 2, 5, 10])  # T (years)
swaps = np.array([1, 2, 5, 10, 30])    # τ (swap length, years)

# Create mesh grid
T_grid, tau_grid = np.meshgrid(tenors, swaps)

# Synthetic IV surface (ATM implied vols)
# U-shaped term structure (short: high, medium: low, long: rising)
# Smile structure (OTM skew)
iv_atm = 0.15 + 0.05 * np.exp(-(T_grid - 1)**2 / 0.5) + 0.02 / np.sqrt(tau_grid)
atm_iv_flat = iv_atm.flatten()

print(f"ATM Swaption IVs (Sample):")
print("-"*70)
print(f"{'Tenor (y)':<12} {'5Y Swap':<12} {'10Y Swap':<12} {'30Y Swap':<12}")
print("-"*70)
for i, t in enumerate(tenors):
    iv_5y = iv_atm[2, i]  # 5Y swap row
    iv_10y = iv_atm[3, i]  # 10Y swap row
    iv_30y = iv_atm[4, i]  # 30Y swap row
    print(f"{t:<12.1f} {iv_5y:<12.2%} {iv_10y:<12.2%} {iv_30y:<12.2%}")
print("")

# Model 1: SABR (parametric)
def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """SABR implied vol formula"""
    if K == F:
        return alpha
    
    x = np.log(F / K)
    z = nu / alpha * F**beta * x
    
    if abs(z) < 1e-6:
        chi_z = 1.0
    else:
        chi_z = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    
    if beta >= 0.99:
        vol = alpha * chi_z / (F*K)**(0.5*(1-beta)) * (
            1 + ((1-beta)**2/24) * np.log(F/K)**2 +
            (rho * beta * nu * alpha / 4) * np.log(F/K) +
            ((2 - 3*rho**2)/24) * nu**2 * T
        )
    else:
        vol = alpha
    
    return max(vol, 0.01)

def sabr_surface(T, tau, params):
    """SABR surface (ATM, beta fixed at 1)"""
    alpha, rho, nu = params
    beta = 1.0
    F = 0.03 + 0.01 * np.log(tau)  # Forward rate term-dependent
    return sabr_iv(F, F, T, alpha, rho, nu, beta)

# Model 2: Parametric smile (polynomial in log-moneyness + term structure)
def poly_smile_surface(T, tau, params):
    """Polynomial smile + exponential term structure"""
    a, b, c, d, e = params
    
    # Base ATM vol
    atm = a * np.exp(-b * T) + c / np.sqrt(tau) + d
    
    # Add smile (second-order)
    smile_effect = e * 0.01 * T * (0.1**2)  # Minimal smile for demo
    
    return np.maximum(atm + smile_effect, 0.05)

# Model 3: Spline interpolation (least smooth; highest fidelity)
def spline_surface(T, tau, T_data, tau_data, iv_data):
    """Interpolate via bivariate spline (simplified: separate splines per maturity)"""
    # For each tenor, fit tau structure
    try:
        cs = CubicSpline(tau_data, iv_data)
        return np.maximum(cs(tau), 0.05)
    except:
        return iv_data.mean() * np.ones_like(T)

# Calibration data: Market IVs at grid points
market_iv_data = iv_atm.flatten()
market_T = T_grid.flatten()
market_tau = tau_grid.flatten()

# Calibrate SABR (3 parameters: α, ρ, ν; β fixed = 1)
def sabr_objective(params):
    try:
        alpha, rho, nu = params
        if alpha <= 0 or alpha > 1:
            return 1e10
        if rho < -0.99 or rho > 0.99:
            return 1e10
        if nu <= 0 or nu > 2:
            return 1e10
        
        model_iv = np.array([sabr_surface(T, tau, params) 
                             for T, tau in zip(market_T, market_tau)])
        
        mse = np.mean((model_iv - market_iv_data)**2)
        return mse
    except:
        return 1e10

x0_sabr = [0.15, -0.3, 0.4]
result_sabr = minimize(sabr_objective, x0_sabr, method='Nelder-Mead',
                       options={'maxiter': 5000})
params_sabr = result_sabr.x

# Calibrate Polynomial
def poly_objective(params):
    try:
        a, b, c, d, e = params
        if a <= 0 or b < 0 or c < 0 or d < 0 or e < 0:
            return 1e10
        
        model_iv = np.array([poly_smile_surface(T, tau, params)
                             for T, tau in zip(market_T, market_tau)])
        
        mse = np.mean((model_iv - market_iv_data)**2)
        return mse
    except:
        return 1e10

x0_poly = [0.15, 0.1, 0.02, 0.12, 0.01]
result_poly = minimize(poly_objective, x0_poly, method='Nelder-Mead',
                       options={'maxiter': 5000})
params_poly = result_poly.x

# Fit spline
params_spline = None  # No parameters; just interpolation

print("Calibration Results:")
print("-"*70)
print(f"SABR MSE:         {result_sabr.fun:.2e}")
print(f"Polynomial MSE:   {result_poly.fun:.2e}")
print("")
print(f"SABR params:")
print(f"  α = {params_sabr[0]:.4f}, ρ = {params_sabr[1]:.4f}, ν = {params_sabr[2]:.4f}")
print(f"Poly params:")
print(f"  a={params_poly[0]:.4f}, b={params_poly[1]:.4f}, c={params_poly[2]:.4f}, " +
      f"d={params_poly[3]:.4f}, e={params_poly[4]:.4f}")
print("")

# Generate predictions on fine grid
T_fine = np.linspace(0.5, 10, 30)
tau_fine = np.linspace(1, 30, 30)
T_fine_grid, tau_fine_grid = np.meshgrid(T_fine, tau_fine)

# SABR surface
iv_sabr_fine = np.array([[sabr_surface(T, tau, params_sabr) 
                          for T in T_fine] for tau in tau_fine])

# Polynomial surface
iv_poly_fine = np.array([[poly_smile_surface(T, tau, params_poly)
                          for T in T_fine] for tau in tau_fine])

# Spline (interpolate only on original grid)
iv_spline_fine = np.array([[np.interp(tau, swaps, iv_atm[:, np.argmin(np.abs(tenors - T))])
                            for T in T_fine] for tau in tau_fine])

# Compute fit quality
def compute_metrics(model_iv_points, market_iv_points):
    """Residual statistics"""
    residuals = model_iv_points - market_iv_points
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    r2 = 1 - np.sum(residuals**2) / np.sum((market_iv_points - np.mean(market_iv_points))**2)
    return rmse, mae, max_error, r2

# Evaluate at original data points
iv_sabr_data = np.array([sabr_surface(T, tau, params_sabr) for T, tau in zip(market_T, market_tau)])
iv_poly_data = np.array([poly_smile_surface(T, tau, params_poly) for T, tau in zip(market_T, market_tau)])

rmse_sabr, mae_sabr, max_err_sabr, r2_sabr = compute_metrics(iv_sabr_data, market_iv_data)
rmse_poly, mae_poly, max_err_poly, r2_poly = compute_metrics(iv_poly_data, market_iv_data)
rmse_spline, mae_spline, max_err_spline, r2_spline = compute_metrics(market_iv_data, market_iv_data)

print("Fit Quality Metrics:")
print("-"*70)
print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'Max Error':<12} {'R²':<12}")
print("-"*70)
print(f"{'SABR':<15} {rmse_sabr:<12.4%} {mae_sabr:<12.4%} {max_err_sabr:<12.4%} {r2_sabr:<12.4f}")
print(f"{'Polynomial':<15} {rmse_poly:<12.4%} {mae_poly:<12.4%} {max_err_poly:<12.4%} {r2_poly:<12.4f}")
print(f"{'Spline':<15} {rmse_spline:<12.4%} {mae_spline:<12.4%} {max_err_spline:<12.4%} {r2_spline:<12.4f}")
print("")

# 3D Visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Market IV surface
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot_surface(T_fine_grid, tau_fine_grid, iv_atm / iv_atm.max() * 100, cmap='viridis', alpha=0.8)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Swap Length (years)')
ax.set_zlabel('IV (%)')
ax.set_title('Market IV Surface (Original)')

# Plot 2: SABR fit
ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.plot_surface(T_fine_grid, tau_fine_grid, iv_sabr_fine * 100, cmap='plasma', alpha=0.8)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Swap Length (years)')
ax.set_zlabel('IV (%)')
ax.set_title(f'SABR Calibration (RMSE: {rmse_sabr:.2%})')

# Plot 3: Polynomial fit
ax = fig.add_subplot(2, 3, 3, projection='3d')
ax.plot_surface(T_fine_grid, tau_fine_grid, iv_poly_fine * 100, cmap='cool', alpha=0.8)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Swap Length (years)')
ax.set_zlabel('IV (%)')
ax.set_title(f'Polynomial Calibration (RMSE: {rmse_poly:.2%})')

# Plot 4: SABR residuals
ax = fig.add_subplot(2, 3, 4)
residuals_sabr = iv_sabr_data - market_iv_data
ax.scatter(market_T, residuals_sabr * 100, c=market_tau, cmap='RdYlBu', s=50, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Residuals (bps)')
ax.set_title('SABR Residuals')
ax.grid(alpha=0.3)
cb = plt.colorbar(ax.collections[0], ax=ax)
cb.set_label('Swap Length (years)')

# Plot 5: Polynomial residuals
ax = fig.add_subplot(2, 3, 5)
residuals_poly = iv_poly_data - market_iv_data
ax.scatter(market_T, residuals_poly * 100, c=market_tau, cmap='RdYlBu', s=50, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Residuals (bps)')
ax.set_title('Polynomial Residuals')
ax.grid(alpha=0.3)
cb = plt.colorbar(ax.collections[0], ax=ax)
cb.set_label('Swap Length (years)')

# Plot 6: Comparison summary
ax = fig.add_subplot(2, 3, 6)
ax.axis('off')

summary_text = f"""
Swaption IV Surface Calibration Comparison

Market Data:
  Tenors (T)       : {', '.join([f'{x:.1f}y' for x in tenors])}
  Swap Lengths (τ) : {', '.join([f'{x:.0f}y' for x in swaps])}
  Total Points     : {len(market_iv_data)}

Model Performance:
┌────────────────────────────────────────┐
│ Model       │ RMSE    │ MAE     │ R²   │
├────────────────────────────────────────┤
│ SABR        │ {rmse_sabr:6.2%}  │ {mae_sabr:6.2%}  │ {r2_sabr:6.4f} │
│ Polynomial  │ {rmse_poly:6.2%}  │ {mae_poly:6.2%}  │ {r2_poly:6.4f} │
│ Spline      │ {rmse_spline:6.2%}  │ {mae_spline:6.2%}  │ {r2_spline:6.4f} │
└────────────────────────────────────────┘

SABR Calibrated Parameters:
  α (ATM vol)  = {params_sabr[0]:8.4f}
  ρ (correlation) = {params_sabr[1]:8.4f}
  ν (vol-of-vol)  = {params_sabr[2]:8.4f}
  β (beta)     = 1.0000 (fixed; lognormal)

Interpretation:
  SABR: Smooth, stable extrapolation
  Poly: Good fit to term structure
  Spline: Perfect fit; risky extrapolation
"""

ax.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('volatility_surface_calibration.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("Key Insights:")
print("="*70)
print("1. SABR provides smooth, stable extrapolation with 3 parameters")
print("2. Polynomial fits term structure well; may fail at extremes")
print("3. Spline perfect fit but poor extrapolation to new tenors/swaps")
print("4. Trade-off: Fidelity vs stability; choose based on use case")
print("5. Validation: Forward-test on new options; monitor residuals")
```

## 6. Challenge Round
IV surface calibration complexities:
- **Non-Uniqueness**: Multiple parameter sets fit market equally well (ill-posed inverse problem); solution: Bayesian priors; regularization; domain constraints
- **Extrapolation Failure**: Spline fits perfectly in liquid core; wings extrapolate nonsensically; solution: Parametric models; data boundary constraints; soft boundaries
- **Bid-Ask Bounce**: Outlier prices distort calibration; solution: Robust loss functions (Huber loss); down-weight illiquid; median instead of mean
- **Regime Shifts**: Crisis-era smile different from normal times; fixed parameters fail; solution: Regime-switching models; frequent recalibration; multiple-model ensemble
- **Computational Cost**: IV inversion slow; optimization stalls; solution: Cached computations; parallel optimization; approximate formulas (SABR asymptotic)
- **Model Risk**: Calibrated to vanillas; exotics misprice; solution: Calibrate to exotics too; forward-test; price with multiple models; hedge model risk

## 7. Key References
- [Hagan et al.: Managing Smile Risk (2002)](https://arxiv.org/abs/math/0504418) - SABR model; widely used in practice; asymptotic formula
- [Rebonato: Volatility and Correlation (2004)](https://www.wiley.com/en-us/The+Volatility+Smile-p-9780471973935) - Smile modeling; smile dynamics; practical frameworks
- [Dupire: Pricing with a Smile (1994)](https://www.risk.net/our-publications/magazine/risk-magazine) - Local volatility; Dupire formula; foundational model

---
**Status:** Derivative Pricing Core | **Pairs Well With:** Implied Volatility, Exotic Pricing, Model Risk
