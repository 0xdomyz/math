# Parameter Estimation & Optimization Methods

## Concept Skeleton
**Definition:** Techniques to estimate model parameters (volatility, mean reversion, jump intensities) from market data; optimization algorithms minimize objective functions (least squares, maximum likelihood) to fit model predictions to observed prices  
**Purpose:** Calibrate pricing models to market reality; determine input parameters from liquid market data (option prices, historical returns); balance fit quality vs model complexity; enable consistent pricing across instruments  
**Prerequisites:** Optimization algorithms (gradient descent, Newton-Raphson), likelihood functions, regression analysis, numerical methods, model formulations (Black-Scholes, Heston, jump-diffusion)

## Comparative Framing
| Method | Objective Function | Data Input | Stability | Computation | Best For |
|--------|-------------------|-----------|-----------|-----------|----------|
| **Least Squares (OLS)** | Minimize MSE | Time series returns | High | Fast | Historical volatility; simple models |
| **Maximum Likelihood (MLE)** | Maximize likelihood | Returns/prices | Medium | Moderate | Parameter uncertainty quantification; testing |
| **Least Squares (Options)** | Minimize IV differences | Option prices | Low (ill-posed) | Slow | IV surface calibration; smile fitting |
| **Regularization (Ridge/Lasso)** | MSE + λ·penalty | Time series | Very High | Fast | Avoid overfitting; sparse models |
| **Bayesian Methods** | Posterior likelihood | Prices + priors | High | Slow | Parameter uncertainty; small samples |
| **Machine Learning (NN, XGBoost)** | Custom loss | Large datasets | Medium | Moderate-Fast | Complex nonlinear relationships |

## Examples + Counterexamples
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

## Layer Breakdown
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

## Challenge Round
When parameter estimation fails or introduces complexity:
- **Ill-Posed Inverse Problem**: Multiple parameter sets fit equally well → Non-uniqueness; solution: Regularization (Bayesian priors); use longer data series; add constraints
- **Local Minima**: Optimization converges to local minimum not global → Wrong parameters; solution: Multiple initial guesses; global optimizers (simulated annealing, genetic algorithms); verify with alternative data
- **Overfitting**: Model fits historical data perfectly; forward-test fails; solution: Regularization; cross-validation; simpler model; fewer parameters
- **Numerical Instability**: Hessian singular; optimization diverges; solution: Robust algorithms (Levenberg-Marquardt); parameter bounds; scaling
- **Model Risk**: Calibrated to vanilla options; exotics misprice; solution: Calibrate to exotic prices too; multiple calibration targets; stress test
- **Regime Changes**: Parameters estimated from pre-crisis data; crisis hits; parameters wrong; solution: Rolling window calibration; regime-switching models; frequent recalibration

## Key References
- [Hagan et al.: Managing Smile Risk (2002)](https://arxiv.org/abs/math/0504418) - SABR model; volatility smile calibration; widely-used asymptotic formula
- [Press et al.: Numerical Recipes (2007)](http://numerical.recipes/) - Optimization algorithms; least squares; MLE; practical implementations
- [Wilmott: Quantitative Finance (2000)](https://www.paulwilmott.com/) - Model calibration theory; optimization methods; practical considerations

---
**Status:** Derivative Pricing Core Methodology | **Complements:** Volatility Surface, Implied Volatility, Stochastic Models
