# Model Calibration Methods

## Concept Skeleton
**Definition:** Estimate model parameters by minimizing distance between theoretical and market prices, ensuring pricing model matches observable derivative prices  
**Purpose:** Calibrate Black-Scholes volatility surface, Heston parameters, SABR coefficients to liquid market instruments for consistent derivative valuation  
**Prerequisites:** Optimization theory, implied volatility, options pricing models (Black-Scholes, Heston, SABR)

## Comparative Framing
| Method | Least Squares | Weighted Least Squares | Maximum Likelihood | Moment Matching |
|--------|---------------|------------------------|---------------------|-----------------|
| **Objective** | min Σ(V_model-V_mkt)² | min Σ wᵢ(V_model-V_mkt)² | max Π f(data\|θ) | E[g(X)] = theoretical moments |
| **Weights** | Equal (1) | Vega-weighted or bid-ask | Probability-based | N/A |
| **Use Case** | Simple IV surface | Liquid options (more weight) | Time series calibration | Quick approximation |
| **Robustness** | Sensitive to outliers | Reduces outlier impact | Statistical foundation | Fast but approximate |

## Examples + Counterexamples
**Simple Example:**  
Calibrate Black-Scholes IV surface: 20 call options with strikes 90-110, maturities 1M-1Y → minimize Σ(C_BS(σ(K,T))-C_mkt)² → piecewise linear σ(K,T) surface

**Failure Case:**  
Unconstrained Heston calibration: negative variance parameters (v₀<0, θ<0) violate Feller condition 2κθ≥σ_v² → unstable variance process, negative volatility

**Edge Case:**  
Deep OTM options with wide bid-ask spreads: market price $0.05±$0.10 → calibration overfits noise → use vega weights wᵢ=1/vega² to reduce impact

## Layer Breakdown
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

## Challenge Round
Why does Heston calibration often produce multiple local minima?
- **Parameter correlation:** High correlation between (v₀,θ) and (κ,σ_v) → many combinations yield similar option prices (degeneracy)
- **Objective landscape:** Non-convex loss surface with flat regions → gradient-based methods get stuck
- **Short-dated dominance:** Short maturity options (most liquid) insensitive to θ and κ → long-term parameters poorly identified
- **Vol-of-vol ambiguity:** σ_v affects smile curvature, but similar curvature achievable with different (σ_v,ρ) pairs
- **Solutions:** Use global optimizer (differential evolution, basin hopping), add regularization toward historical parameter estimates, calibrate to both vanilla options and variance swaps (better identify θ), use sequential calibration (fix some parameters from historical data)

Modern practice: Calibrate to liquid vanillas + variance swap term structure + historical time series (hybrid approach combines market and statistical information).

## Key References
- [Cont & Tankov (2004) Financial Modelling with Jump Processes, Ch. 10](https://www.routledge.com/Financial-Modelling-with-Jump-Processes/Cont-Tankov/p/book/9781584884132) - Calibration methodology for jump-diffusions
- [Gatheral (2006) The Volatility Surface: A Practitioner's Guide](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119202073) - SVI parameterization, arbitrage-free constraints
- [Rouah (2013) The Heston Model and its Extensions in Matlab and C#](https://www.wiley.com/en-us/The+Heston+Model+and+Its+Extensions+in+Matlab+and+C%23-p-9781118548257) - Practical Heston calibration with code
- [Andersen (2007) Efficient Simulation of the Heston Stochastic Volatility Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405) - QE scheme for Monte Carlo calibration

---
**Status:** Bridges market data and pricing models | **Complements:** Heston model, SABR, Local volatility
