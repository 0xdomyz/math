# Model Calibration Methods

## 1. Concept Skeleton
**Definition:** Estimate model parameters by minimizing distance between theoretical and market prices, ensuring pricing model matches observable derivative prices  
**Purpose:** Calibrate Black-Scholes volatility surface, Heston parameters, SABR coefficients to liquid market instruments for consistent derivative valuation  
**Prerequisites:** Optimization theory, implied volatility, options pricing models (Black-Scholes, Heston, SABR)

## 2. Comparative Framing
| Method | Least Squares | Weighted Least Squares | Maximum Likelihood | Moment Matching |
|--------|---------------|------------------------|---------------------|-----------------|
| **Objective** | min Σ(V_model-V_mkt)² | min Σ wᵢ(V_model-V_mkt)² | max Π f(data\|θ) | E[g(X)] = theoretical moments |
| **Weights** | Equal (1) | Vega-weighted or bid-ask | Probability-based | N/A |
| **Use Case** | Simple IV surface | Liquid options (more weight) | Time series calibration | Quick approximation |
| **Robustness** | Sensitive to outliers | Reduces outlier impact | Statistical foundation | Fast but approximate |

## 3. Examples + Counterexamples

**Simple Example:**  
Calibrate Black-Scholes IV surface: 20 call options with strikes 90-110, maturities 1M-1Y → minimize Σ(C_BS(σ(K,T))-C_mkt)² → piecewise linear σ(K,T) surface

**Failure Case:**  
Unconstrained Heston calibration: negative variance parameters (v₀<0, θ<0) violate Feller condition 2κθ≥σ_v² → unstable variance process, negative volatility

**Edge Case:**  
Deep OTM options with wide bid-ask spreads: market price $0.05±$0.10 → calibration overfits noise → use vega weights wᵢ=1/vega² to reduce impact

## 4. Layer Breakdown
```
Model Calibration:
├─ Calibration Framework:
│   ├─ Target Instruments: Market-quoted liquid options
│   │   ├─ Equity: ATM/OTM puts and calls across maturities
│   │   ├─ FX: 25-delta risk reversals, 25-delta butterflies
│   │   ├─ Interest Rates: Cap/floor vols, swaption matrix
│   │   └─ Selection: Most liquid strikes/tenors, exclude stale quotes
│   ├─ Calibration Problem:
│   │   min_θ Loss(θ) = Σᵢ wᵢ [V_model(θ; Kᵢ,Tᵢ) - V_mkt(Kᵢ,Tᵢ)]²
│   │   subject to: θ_min ≤ θ ≤ θ_max (parameter constraints)
│   ├─ Parameter Vector θ: Depends on model
│   │   ├─ Black-Scholes: θ = {σ(K,T)} (volatility surface)
│   │   ├─ Heston: θ = {v₀, κ, θ_v, σ_v, ρ} (5 parameters)
│   │   └─ SABR: θ = {α, β, ρ, ν} (4 parameters per tenor)
│   └─ Frequency: Daily for active trading desks, weekly for risk management
├─ Loss Functions:
│   ├─ Price-Based:
│   │   Loss_price = Σ wᵢ (V_model - V_mkt)²
│   │   ├─ Simple, direct market fit
│   │   └─ Issue: Deep OTM low prices dominate (small absolute errors)
│   ├─ Implied Volatility-Based:
│   │   Loss_IV = Σ wᵢ (σ_model - σ_mkt)²
│   │   ├─ More stable (IV roughly same order of magnitude)
│   │   └─ Preferred for smile calibration
│   ├─ Relative Error:
│   │   Loss_rel = Σ wᵢ [(V_model - V_mkt)/V_mkt]²
│   │   └─ Normalizes for option value scale
│   └─ Mixed:
│       Loss = α·Loss_price + β·Loss_IV (hybrid approach)
├─ Weighting Schemes:
│   ├─ Uniform: wᵢ = 1 (equal weight all instruments)
│   ├─ Vega-Weighted:
│   │   wᵢ = vega(Kᵢ,Tᵢ)² (weight by sensitivity to vol changes)
│   │   ├─ ATM options get higher weight (largest vega)
│   │   └─ Reduces OTM noise impact
│   ├─ Inverse Bid-Ask:
│   │   wᵢ = 1/(bid-ask spread)² (tighter markets = higher weight)
│   │   └─ Accounts for liquidity
│   ├─ Volume/OI-Weighted:
│   │   wᵢ = √(volume) or √(open_interest)
│   │   └─ Emphasize actively traded strikes
│   └─ Combined:
│       wᵢ = vega × (1/bid-ask) (multiple criteria)
├─ Optimization Methods:
│   ├─ Local Optimization:
│   │   ├─ Levenberg-Marquardt (LM):
│   │   │   ├─ Update: θ_(k+1) = θ_k - [J'J + λI]^(-1) J'r (damped Gauss-Newton)
│   │   │   ├─ J = Jacobian matrix (∂V/∂θ), r = residuals (V_model - V_mkt)
│   │   │   ├─ λ controls interpolation: Gauss-Newton (λ→0) vs Gradient Descent (λ→∞)
│   │   │   └─ Fast convergence for well-behaved surfaces
│   │   ├─ BFGS (Quasi-Newton):
│   │   │   ├─ Approximates Hessian without computing second derivatives
│   │   │   └─ Good for smooth objective functions
│   │   └─ Constrained: L-BFGS-B (box constraints), SQP (general constraints)
│   ├─ Global Optimization:
│   │   ├─ Differential Evolution:
│   │   │   ├─ Population-based, mutation + crossover
│   │   │   └─ Robust to local minima (Heston multi-modal surface)
│   │   ├─ Particle Swarm Optimization (PSO):
│   │   │   └─ Swarm intelligence, fast for low-dimensional problems
│   │   ├─ Basin Hopping:
│   │   │   └─ Random perturbations + local minimization (escape local minima)
│   │   └─ Genetic Algorithms:
│   │       └─ Evolutionary search, handles discrete parameter spaces
│   └─ Hybrid:
│       Global search (DE) → Local refinement (LM) (best of both)
├─ Model-Specific Calibration:
│   ├─ Black-Scholes Implied Volatility Surface:
│   │   ├─ Parametric Form:
│   │   │   σ(K,T) = f(m,T; θ) where m = ln(K/F) (log-moneyness)
│   │   │   ├─ SVI (Stochastic Volatility Inspired):
│   │   │     σ²(k) = a + b[ρ(k-m) + √((k-m)²+σ²)]
│   │   │     Parameters: {a, b, ρ, m, σ} (5 per tenor)
│   │   │   └─ Arbitrage-free conditions: No calendar/butterfly arbitrage
│   │   ├─ Non-Parametric: Cubic spline interpolation on (K,T) grid
│   │   └─ Calibration: Minimize Loss_IV = Σ(σ_SVI - σ_mkt)²
│   ├─ Heston Model:
│   │   ├─ Parameters: θ = {v₀, κ, θ_v, σ_v, ρ}
│   │   │   v₀: Initial variance, κ: Mean reversion speed
│   │   │   θ_v: Long-term variance, σ_v: Vol-of-vol, ρ: Spot-vol correlation
│   │   ├─ Pricing: Characteristic function + FFT for European options
│   │   ├─ Constraints:
│   │   │   ├─ Feller condition: 2κθ_v ≥ σ_v² (variance stays positive)
│   │   │   ├─ Bounds: 0<v₀<1, 0<κ<10, -1<ρ<0 (typical equity)
│   │   │   └─ Stability: κ large enough for mean reversion
│   │   └─ Calibration: Minimize Loss_IV or Loss_price to vanilla options
│   ├─ SABR Model:
│   │   ├─ Forward LIBOR/FX dynamics: dF = α F^β dW₁
│   │   │   dα = ν α dW₂, Corr(dW₁,dW₂) = ρ
│   │   ├─ Parameters: θ = {α, β, ρ, ν}
│   │   │   α: Initial volatility, β: Elasticity (0=normal, 1=lognormal)
│   │   │   ρ: Forward-vol correlation, ν: Vol-of-vol
│   │   ├─ Calibration per Tenor: Separate {α,ρ,ν} for each expiry (β often fixed)
│   │   └─ Use: FX options, swaptions (better smile fit than Black)
│   └─ Local Volatility:
│       ├─ Dupire Formula: σ_LV²(K,T) = [∂C/∂T + rK∂C/∂K] / [½K²∂²C/∂K²]
│       ├─ Input: Full IV surface C(K,T)
│       ├─ Output: Local vol function σ(S,t) deterministic
│       └─ Advantage: Exact fit to vanilla surface, but wrong forward smile
├─ Regularization Techniques:
│   ├─ Parameter Penalties:
│   │   Loss_reg = Loss_cal + λ_reg Σ(θⱼ - θⱼ_prior)² (Tikhonov)
│   │   └─ Keeps parameters near reasonable priors (historical estimates)
│   ├─ Smoothness Penalties:
│   │   Penalize ∂²σ/∂K² (avoid artificial wiggles in IV surface)
│   ├─ Arbitrage Constraints:
│   │   ├─ Calendar spread: C(T₁) ≤ C(T₂) for T₁<T₂
│   │   ├─ Butterfly spread: ∂²C/∂K² ≥ 0 (call price convex in strike)
│   │   └─ Enforce during optimization via inequality constraints
│   └─ Stability Checks:
│       Monitor condition number of Jacobian (ill-conditioning → regularize)
├─ Calibration Validation:
│   ├─ In-Sample Fit:
│   │   ├─ RMSE: √[Σ(V_model-V_mkt)²/N] (price or IV)
│   │   ├─ Mean Absolute Error: Σ|V_model-V_mkt|/N
│   │   └─ Max Error: max|V_model-V_mkt| (worst-case check)
│   ├─ Out-of-Sample:
│   │   Calibrate to liquid strikes, test on illiquid → assess extrapolation
│   ├─ Time Stability:
│   │   Track parameter drift day-to-day (large jumps indicate overfitting)
│   └─ P&L Explain:
│       Mark-to-market with calibrated model vs actual P&L (model risk metric)
└─ Practical Considerations:
    ├─ Bid-Ask Handling:
    │   ├─ Use mid prices for calibration
    │   ├─ Ensure model prices within bid-ask spread
    │   └─ Flag violations (model outside bid-ask → recalibration needed)
    ├─ Computation Speed:
    │   ├─ Analytic gradients (∂V/∂θ) vs finite differences (10x speedup)
    │   ├─ Parallel pricing across strikes/maturities
    │   └─ Cache repeated calculations (Greeks, characteristic functions)
    ├─ Model Selection:
    │   ├─ Simple products: Black-Scholes IV surface sufficient
    │   ├─ Path-dependent/barriers: Local vol or Heston
    │   └─ Forward smile dynamics: Stochastic vol (Heston, SABR)
    └─ Recalibration Frequency:
        Daily for active desks, intraday for high-frequency market makers
```

**Interaction:** Market data → Select liquid instruments → Choose loss function + weights → Optimize with constraints → Validate fit → Use calibrated model for pricing illiquid derivatives

## 5. Mini-Project
Calibrate Heston stochastic volatility model to market option prices:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.integrate import quad

# =====================================
# HESTON MODEL CALIBRATION
# =====================================
print("="*70)
print("HESTON MODEL CALIBRATION TO MARKET OPTIONS")
print("="*70)

# =====================================
# HESTON PRICING via Characteristic Function
# =====================================
def heston_characteristic_function(u, S0, v0, kappa, theta, sigma_v, rho, r, T):
    """
    Heston characteristic function φ(u) for log(S_T).
    Used in Fourier inversion for option pricing.
    """
    # Parameters
    lam = 0  # Risk premium (often set to 0)
    d = np.sqrt((rho * sigma_v * u * 1j - kappa)**2 + sigma_v**2 * (u * 1j + u**2))
    g = (kappa - rho * sigma_v * u * 1j - d) / (kappa - rho * sigma_v * u * 1j + d)
    
    # Characteristic exponents
    C = r * u * 1j * T + (kappa * theta / sigma_v**2) * \
        ((kappa - rho * sigma_v * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    
    D = ((kappa - rho * sigma_v * u * 1j - d) / sigma_v**2) * \
        ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    phi = np.exp(C + D * v0 + 1j * u * np.log(S0))
    return phi

def heston_call_price(S0, K, v0, kappa, theta, sigma_v, rho, r, T):
    """
    Price European call option under Heston model using Fourier inversion.
    
    C = S0 * P1 - K * exp(-rT) * P2
    where P1, P2 are probabilities computed via characteristic function.
    """
    # P1: Probability under stock measure
    def integrand_P1(u):
        phi = heston_characteristic_function(u - 1j, S0, v0, kappa, theta, sigma_v, rho, r, T)
        numerator = np.exp(-1j * u * np.log(K)) * phi
        denominator = 1j * u * S0
        return np.real(numerator / denominator)
    
    P1 = 0.5 + (1 / np.pi) * quad(integrand_P1, 0, 100, limit=100)[0]
    
    # P2: Probability under money market measure
    def integrand_P2(u):
        phi = heston_characteristic_function(u, S0, v0, kappa, theta, sigma_v, rho, r, T)
        numerator = np.exp(-1j * u * np.log(K)) * phi
        denominator = 1j * u
        return np.real(numerator / denominator)
    
    P2 = 0.5 + (1 / np.pi) * quad(integrand_P2, 0, 100, limit=100)[0]
    
    # Call price
    call_price = S0 * P1 - K * np.exp(-r * T) * P2
    return max(call_price, 0)  # Ensure non-negative

# =====================================
# MARKET DATA (Synthetic)
# =====================================
# Simulate "market" data with known Heston parameters
np.random.seed(42)

S0_true = 100.0
r_true = 0.05
v0_true = 0.04       # Initial variance (σ₀² = 20%²)
kappa_true = 2.0     # Mean reversion speed
theta_true = 0.04    # Long-term variance
sigma_v_true = 0.3   # Vol-of-vol
rho_true = -0.7      # Negative correlation (leverage effect)

# Generate market option grid
strikes = np.array([80, 90, 95, 100, 105, 110, 120])
maturities = np.array([0.25, 0.5, 1.0])

market_data = []
for T in maturities:
    for K in strikes:
        true_price = heston_call_price(S0_true, K, v0_true, kappa_true, theta_true, 
                                        sigma_v_true, rho_true, r_true, T)
        # Add small noise to simulate market imperfections
        market_price = true_price + np.random.normal(0, 0.05)
        
        # Calculate vega for weighting (approximate)
        vega_weight = S0_true * np.sqrt(T) * norm.pdf(norm.ppf(0.5))  # Simplified
        
        market_data.append({
            'Strike': K,
            'Maturity': T,
            'Market_Price': max(market_price, 0.01),
            'True_Price': true_price,
            'Vega_Weight': vega_weight
        })

market_df = pd.DataFrame(market_data)

print("\nMarket Option Data (Sample):")
print(market_df.head(10).to_string(index=False))
print(f"\nTotal instruments: {len(market_df)}")

# =====================================
# CALIBRATION OBJECTIVE FUNCTION
# =====================================
def calibration_objective(params, market_df, S0, r, weight_type='uniform'):
    """
    Calibration loss function: Sum of squared pricing errors.
    
    Parameters to calibrate: [v0, kappa, theta, sigma_v, rho]
    """
    v0, kappa, theta, sigma_v, rho = params
    
    # Feller condition check
    if 2 * kappa * theta < sigma_v**2:
        return 1e10  # Penalty for violating Feller condition
    
    total_error = 0
    for idx, row in market_df.iterrows():
        K = row['Strike']
        T = row['Maturity']
        market_price = row['Market_Price']
        
        try:
            model_price = heston_call_price(S0, K, v0, kappa, theta, sigma_v, rho, r, T)
            error = (model_price - market_price)**2
            
            # Apply weights
            if weight_type == 'vega':
                weight = row['Vega_Weight']
            else:
                weight = 1.0
            
            total_error += weight * error
        except:
            return 1e10  # Return large penalty for numerical failures
    
    return total_error

# =====================================
# CALIBRATION (Global Optimization)
# =====================================
print("\n" + "="*70)
print("CALIBRATION: Differential Evolution (Global Optimizer)")
print("="*70)

# Parameter bounds [v0, kappa, theta, sigma_v, rho]
bounds = [
    (0.01, 0.5),    # v0: Initial variance
    (0.1, 5.0),     # kappa: Mean reversion speed
    (0.01, 0.5),    # theta: Long-term variance
    (0.05, 1.0),    # sigma_v: Vol-of-vol
    (-0.99, -0.01)  # rho: Negative correlation (equity typical)
]

print("\nTrue Parameters (used to generate market data):")
print(f"   v0     = {v0_true:.4f}")
print(f"   kappa  = {kappa_true:.4f}")
print(f"   theta  = {theta_true:.4f}")
print(f"   sigma_v= {sigma_v_true:.4f}")
print(f"   rho    = {rho_true:.4f}")

print("\nCalibrating... (this may take 30-60 seconds)")

result = differential_evolution(
    calibration_objective,
    bounds,
    args=(market_df, S0_true, r_true, 'uniform'),
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    tol=1e-6,
    seed=42,
    disp=False
)

v0_cal, kappa_cal, theta_cal, sigma_v_cal, rho_cal = result.x

print("\nCalibrated Parameters:")
print(f"   v0     = {v0_cal:.4f}  (true: {v0_true:.4f}, error: {abs(v0_cal-v0_true):.4f})")
print(f"   kappa  = {kappa_cal:.4f}  (true: {kappa_true:.4f}, error: {abs(kappa_cal-kappa_true):.4f})")
print(f"   theta  = {theta_cal:.4f}  (true: {theta_true:.4f}, error: {abs(theta_cal-theta_true):.4f})")
print(f"   sigma_v= {sigma_v_cal:.4f}  (true: {sigma_v_true:.4f}, error: {abs(sigma_v_cal-sigma_v_true):.4f})")
print(f"   rho    = {rho_cal:.4f}  (true: {rho_true:.4f}, error: {abs(rho_cal-rho_true):.4f})")

print(f"\nFeller Condition Check: 2κθ = {2*kappa_cal*theta_cal:.4f} vs σ_v² = {sigma_v_cal**2:.4f}")
print(f"   {'✓ PASS' if 2*kappa_cal*theta_cal >= sigma_v_cal**2 else '✗ FAIL'}")

# =====================================
# VALIDATION: Model vs Market Prices
# =====================================
print("\n" + "="*70)
print("CALIBRATION FIT ANALYSIS")
print("="*70)

market_df['Model_Price'] = market_df.apply(
    lambda row: heston_call_price(S0_true, row['Strike'], v0_cal, kappa_cal, theta_cal,
                                   sigma_v_cal, rho_cal, r_true, row['Maturity']),
    axis=1
)

market_df['Pricing_Error'] = market_df['Model_Price'] - market_df['Market_Price']
market_df['Relative_Error'] = (market_df['Pricing_Error'] / market_df['Market_Price']) * 100

print("\nPricing Errors (Sample):")
print(market_df[['Strike', 'Maturity', 'Market_Price', 'Model_Price', 'Pricing_Error']].head(10).to_string(index=False))

rmse = np.sqrt(np.mean(market_df['Pricing_Error']**2))
mae = np.mean(np.abs(market_df['Pricing_Error']))
max_error = np.max(np.abs(market_df['Pricing_Error']))

print(f"\nCalibration Metrics:")
print(f"   RMSE:        ${rmse:.4f}")
print(f"   MAE:         ${mae:.4f}")
print(f"   Max Error:   ${max_error:.4f}")
print(f"   Mean Rel Error: {np.mean(np.abs(market_df['Relative_Error'])):.2f}%")

# =====================================
# IMPLIED VOLATILITY SURFACE
# =====================================
def black_scholes_call(S, K, r, sigma, T):
    """Black-Scholes call price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_volatility(market_price, S, K, r, T):
    """Compute implied volatility via Newton-Raphson."""
    sigma = 0.2  # Initial guess
    for _ in range(50):
        price = black_scholes_call(S, K, r, sigma, T)
        vega = S * np.sqrt(T) * norm.pdf((np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
        if vega < 1e-10:
            break
        diff = market_price - price
        if abs(diff) < 1e-6:
            break
        sigma += diff / vega
    return sigma

market_df['IV_Market'] = market_df.apply(
    lambda row: implied_volatility(row['Market_Price'], S0_true, row['Strike'], r_true, row['Maturity']),
    axis=1
)

market_df['IV_Model'] = market_df.apply(
    lambda row: implied_volatility(row['Model_Price'], S0_true, row['Strike'], r_true, row['Maturity']),
    axis=1
)

print("\nImplied Volatility Smile (T=1.0 year):")
T_slice = 1.0
iv_slice = market_df[market_df['Maturity'] == T_slice][['Strike', 'IV_Market', 'IV_Model']].copy()
iv_slice['Moneyness'] = iv_slice['Strike'] / S0_true
print(iv_slice.to_string(index=False))

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model vs Market Prices
axes[0, 0].scatter(market_df['Market_Price'], market_df['Model_Price'], alpha=0.6, s=50)
price_range = [market_df['Market_Price'].min(), market_df['Market_Price'].max()]
axes[0, 0].plot(price_range, price_range, 'r--', linewidth=2, label='Perfect Fit')
axes[0, 0].set_xlabel('Market Price ($)')
axes[0, 0].set_ylabel('Model Price ($)')
axes[0, 0].set_title(f'Calibration Fit (RMSE=${rmse:.3f})')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Pricing Errors by Strike
for T in maturities:
    subset = market_df[market_df['Maturity'] == T]
    axes[0, 1].plot(subset['Strike'], subset['Pricing_Error'], marker='o', label=f'T={T}Y')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Strike')
axes[0, 1].set_ylabel('Pricing Error ($)')
axes[0, 1].set_title('Pricing Errors by Strike and Maturity')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Implied Volatility Smile
for T in maturities:
    subset = market_df[market_df['Maturity'] == T].copy()
    subset = subset.sort_values('Strike')
    axes[1, 0].plot(subset['Strike'], subset['IV_Market']*100, marker='o', linestyle='--', 
                    label=f'Market T={T}Y', alpha=0.7)
    axes[1, 0].plot(subset['Strike'], subset['IV_Model']*100, marker='s', linestyle='-', 
                    label=f'Model T={T}Y')
axes[1, 0].set_xlabel('Strike')
axes[1, 0].set_ylabel('Implied Volatility (%)')
axes[1, 0].set_title('Implied Volatility Smile (Market vs Model)')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Parameter Convergence Visualization
param_names = ['v₀', 'κ', 'θ', 'σᵥ', 'ρ']
param_true = [v0_true, kappa_true, theta_true, sigma_v_true, rho_true]
param_cal = [v0_cal, kappa_cal, theta_cal, sigma_v_cal, rho_cal]

x_pos = np.arange(len(param_names))
width = 0.35
axes[1, 1].bar(x_pos - width/2, param_true, width, label='True', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, param_cal, width, label='Calibrated', alpha=0.7)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(param_names)
axes[1, 1].set_ylabel('Parameter Value')
axes[1, 1].set_title('Calibrated vs True Parameters')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Heston calibration successful:")
print(f"• Parameters recovered within mean error: {np.mean([abs(v0_cal-v0_true), abs(kappa_cal-kappa_true), abs(theta_cal-theta_true), abs(sigma_v_cal-sigma_v_true), abs(rho_cal-rho_true)]):.4f}")
print(f"• Pricing RMSE: ${rmse:.4f} (vs typical option values $5-$20)")
print(f"• IV surface captured: Smile and term structure fitted")
print(f"• Method: Differential Evolution (global) avoided local minima")
print(f"• Feller condition: {'Satisfied' if 2*kappa_cal*theta_cal >= sigma_v_cal**2 else 'Violated'}")
```

## 6. Challenge Round
Why does Heston calibration often produce multiple local minima?
- **Parameter correlation:** High correlation between (v₀,θ) and (κ,σ_v) → many combinations yield similar option prices (degeneracy)
- **Objective landscape:** Non-convex loss surface with flat regions → gradient-based methods get stuck
- **Short-dated dominance:** Short maturity options (most liquid) insensitive to θ and κ → long-term parameters poorly identified
- **Vol-of-vol ambiguity:** σ_v affects smile curvature, but similar curvature achievable with different (σ_v,ρ) pairs
- **Solutions:** Use global optimizer (differential evolution, basin hopping), add regularization toward historical parameter estimates, calibrate to both vanilla options and variance swaps (better identify θ), use sequential calibration (fix some parameters from historical data)

Modern practice: Calibrate to liquid vanillas + variance swap term structure + historical time series (hybrid approach combines market and statistical information).

## 7. Key References
- [Cont & Tankov (2004) Financial Modelling with Jump Processes, Ch. 10](https://www.routledge.com/Financial-Modelling-with-Jump-Processes/Cont-Tankov/p/book/9781584884132) - Calibration methodology for jump-diffusions
- [Gatheral (2006) The Volatility Surface: A Practitioner's Guide](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119202073) - SVI parameterization, arbitrage-free constraints
- [Rouah (2013) The Heston Model and its Extensions in Matlab and C#](https://www.wiley.com/en-us/The+Heston+Model+and+Its+Extensions+in+Matlab+and+C%23-p-9781118548257) - Practical Heston calibration with code
- [Andersen (2007) Efficient Simulation of the Heston Stochastic Volatility Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405) - QE scheme for Monte Carlo calibration

---
**Status:** Bridges market data and pricing models | **Complements:** Heston model, SABR, Local volatility
