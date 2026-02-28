# Stochastic Model Calibration: Heston & Jump-Diffusion

## Concept Skeleton
**Definition:** Fitting stochastic differential equation models (Heston, jump-diffusion) to market data; parameters govern volatility clustering, mean reversion, jump intensity/size; enables realistic pricing of path-dependent and exotic options  
**Purpose:** Capture dynamic volatility behavior (not constant like BS); model realistic return distributions (fat tails, skewness); enable consistent exotic pricing; risk management under stress  
**Prerequisites:** Stochastic calculus, SDEs, Monte Carlo simulation, Fourier methods, characteristic functions, likelihood estimation, MLE computation

## Comparative Framing
| Model | Volatility Type | Parameters | Calibration Cost | Exotics | Tail Risk | Best For |
|-------|-----------------|-----------|-----------------|---------|-----------|----------|
| **Heston** | Stochastic | 5 (κ, θ, σ, ρ, v₀) | Moderate | Good | Fat tails | Equities; long-dated |
| **Jump-Diffusion** | Constant + jumps | 5 (σ, λ, μ_J, σ_J, γ) | Moderate | Better | Jump tail risk | Crisis periods; gap risk |
| **Bates (Heston+Jumps)** | Stochastic + jumps | 8 | High | Excellent | Very realistic | Comprehensive pricing |
| **SVJ (Stochastic Vol Jump)** | Stochastic + correlated jumps | 10 | Very High | Excellent | Best | Academic; computationally demanding |
| **GARCH** | Time-varying; discrete | 3-5 | Low | Poor | Moderate | Risk forecasting; short-term |
| **Local Vol** | Spot/time dependent | Surface (100+) | High | Good | Limited | Short-term barrier options |

## Examples + Counterexamples
**Heston Success (Equity Options):**  
Calibrate Heston to 1M-5Y S&P 500 option prices → κ = 1.5 (fast mean reversion), θ = 0.04 (20% long-run vol), σ = 0.6 (moderate vol-of-vol). Price 3Y Bermudan swaption → Good match to market quotes → Parameters stable week-to-week.

**Jump-Diffusion for Tail Events:**  
Pre-2008, jump intensity λ = 0.05/year (small jumps rare). Post-crisis market data: λ = 0.5/year, jump size μ_J = -0.05 (5% down jumps). Heston alone would drastically underprice OTM puts → Jump diffusion captures tail risk.

**GARCH Forecasting:**  
Daily returns show volatility clustering. Fit GARCH(1,1) to 2Y history → Forecast 10-day volatility. Short-term predictions good; 3-month+ predictions revert to long-run vol (poor for pricing long-dated exotics).

**Monte Carlo Bias (Wrong Model):**  
Calibrate Heston; simulate 10,000 paths; price exotic option. But true market is jump-diffusion → Simulated exotic prices too low (missing jump tail risk) → Risk-adjusted value >> model price → Mismatch signals model inadequacy.

**Parameter Uncertainty (Confidence Interval):**  
Calibrate Heston on daily closings; κ = 1.5 ± 0.3 (wide 95% CI). Recalibrate on intraday 1-hour data → κ = 1.6 ± 0.1 (narrower). Data frequency affects parameter precision significantly.

**Regime Change Disaster:**  
Calibrate on 2015-2019 (low vol regime) → Heston θ = 0.12 (12% long-run vol). 2020 COVID crash → θ suddenly insufficient; model severely underprice volatility. Solution: Recalibrate immediately; use longer historical data; regime-switching models.

## Layer Breakdown
```
Stochastic Model Calibration Framework:

├─ Heston Model:
│   ├─ Dynamics:
│   │   ├─ dS/S = μ dt + √v dW¹(t)     [Stock price; geometric Brownian with stochastic vol]
│   │   ├─ dv = κ(θ - v) dt + σ√v dW²(t)  [Variance; mean-reversion to θ with speed κ]
│   │   ├─ dW¹·dW² = ρ dt                [Correlation between stock and vol shocks]
│   │   └─ Parameters:
│   │       ├─ κ (mean reversion speed): κ > 0; fast κ → vol mean-reverts quickly
│   │       ├─ θ (long-run variance): Steady-state variance (expected long-term vol)
│   │       ├─ σ (volatility of volatility): σ > 0; controls vol clustering intensity
│   │       ├─ ρ ∈ [-1, 1]: Leverage effect; typically ρ < 0 (up moves → lower vol)
│   │       └─ v₀ ≥ 0: Initial variance (calibrated to current IV)
│   │
│   ├─ Option Pricing (Heston):
│   │   ├─ Closed-form formula exists (Heston 1993)
│   │   │   ├─ C(S, K, T, r, q, parameters) via characteristic function
│   │   │   ├─ C = S e^(-qT) P₁ - K e^(-rT) P₂
│   │   │   ├─ P₁, P₂ obtained from Fourier inversion (numerical integration)
│   │   │   └─ Fast (< 1ms per option); numerically stable
│   │   ├─ Advantages: Analytical solution; fast computation; calibration efficient
│   │   ├─ Disadvantages: Complex formula; numerical integration required; 5 parameters
│   │   └─ Implementation: Use COS method or FFT for speed (standard practice)
│   │
│   └─ Calibration Targets:
│       ├─ Objective: Minimize Σ w_i (IV_model - IV_market)²
│       ├─ IV_model computed via Heston formula → converted back to IV (Newton-Raphson)
│       ├─ Constraints:
│       │   ├─ Feller condition: 2κθ ≥ σ² (ensures v > 0 always; stability)
│       │   ├─ Parameter bounds: κ > 0.1; θ ∈ [0.01, 1]; σ ∈ [0.1, 3]; ρ ∈ [-0.99, 0.99]
│       │   └─ No arbitrage: Prices decrease in strike; put-call parity (implicit in Heston)
│       └─ Optimization: BFGS or Levenberg-Marquardt; multiple initial guesses
│
├─ Jump-Diffusion Model (Merton):
│   ├─ Dynamics:
│   │   ├─ dS/S = (μ - λμ_J) dt + σ dW(t) + (e^J - 1) dN(t)
│   │   ├─ N(t): Poisson process with intensity λ (jump probability per unit time)
│   │   ├─ J ~ N(log(1 + γ), σ_J²): Log-normal jump size (γ = mean jump %; σ_J = std dev)
│   │   └─ Parameters:
│   │       ├─ σ: Diffusion volatility (between jumps; typically < total vol)
│   │       ├─ λ: Jump intensity (jumps/year; e.g., λ = 0.1 → 1 jump per 10 years)
│   │       ├─ μ_J: Mean log-jump size (e.g., -0.05 for -5% downside jumps)
│   │       ├─ σ_J: Jump size volatility (cluster size uncertainty)
│   │       └─ Leverage effect typically captured via negative μ_J
│   │
│   ├─ Option Pricing (Merton):
│   │   ├─ Series expansion (Merton 1976):
│   │   │   ├─ C = Σ_{n=0}^∞ e^(-λT) (λT)^n / n! × C_BS(σ_n)
│   │   │   ├─ σ_n: Adjusted volatility incorporating n jumps
│   │   │   ├─ Truncate series at n = 100 (convergence fast)
│   │   │   └─ Fast; numerically stable
│   │   ├─ Fourier methods (alternative):
│   │   │   ├─ Characteristic function: φ(u; J, σ, λ, μ_J, σ_J)
│   │   │   ├─ COS method for efficient inversion
│   │   │   └─ Preferred for exotic pricing (variance reduction)
│   │   └─ Advantages: Closed form (series); fast; thin-tailed alternative to Heston
│       └─ Disadvantages: Series truncation; simplistic jump structure
│   │
│   └─ Calibration Targets:
│       ├─ Objective: Minimize Σ w_i (IV_model - IV_market)²
│       ├─ IV_model computed via Merton series (30-50 terms typical)
│       ├─ Constraints:
│       │   ├─ λ ≥ 0, μ_J ∈ [-0.3, 0.1], σ_J ∈ [0.01, 1]
│       │   ├─ σ > 0; typically σ < σ_total (diffusion vol < observed vol)
│       │   └─ Parameter combinations must price observable exotics
│       └─ Optimization: Brent search or BFGS; global optimizer preferred (multiple minima)
│
├─ Bates Model (Heston + Jumps):
│   ├─ Combines stochastic volatility (Heston) + jump component (Merton)
│   ├─ Dynamics:
│   │   ├─ dS/S = (r - q - λμ_J) dt + √v dW¹ + (e^J - 1) dN
│   │   ├─ dv = κ(θ - v) dt + σ√v dW²
│   │   └─ dW¹·dW² = ρ dt
│   ├─ Parameters: 8 (κ, θ, σ_vol, ρ, v₀, λ, μ_J, σ_J)
│   ├─ Advantages: Comprehensive model; stochastic + jump tail risk; excellent fit
│   ├─ Disadvantages: 8 parameters → overfitting risk; slow calibration; numerical complexity
│   └─ Use case: High-frequency trading; crisis periods; comprehensive risk modeling
│
├─ Calibration Workflow:
│   ├─ Step 1: Data Preparation
│   │   ├─ Collect option prices (ATM, 5 strikes, 5 maturities minimum)
│   │   ├─ Compute implied volatility (Newton-Raphson inversion)
│   │   ├─ Inspect IV surface (smile structure, term structure, outliers)
│   │   └─ Filter: Remove illiquid, extreme moneyness, bid-ask spread > 5%
│   │
│   ├─ Step 2: Model Selection
│   │   ├─ Simple: Heston (good balance complexity/fit)
│   │   ├─ Alternative: Jump-Diffusion (for fat tails, crisis)
│   │   ├─ Comprehensive: Bates (both Vol + Jump risks)
│   │   └─ Consider: Asset class, exotic types, computational budget
│   │
│   ├─ Step 3: Initial Guess & Bounds
│   │   ├─ Heston typical ranges:
│   │   │   ├─ κ ∈ [0.1, 5] (mean reversion speed; daily data → κ ~ 0.5-2)
│   │   │   ├─ θ ∈ [0.01, 0.5] (long-run vol; typically √(θ) ∈ [10%, 70%])
│   │   │   ├─ σ ∈ [0.1, 2] (vol of vol; typically σ ∈ [0.3, 1])
│   │   │   ├─ ρ ∈ [-0.99, -0.3] (leverage; typically negative)
│   │   │   └─ v₀ ∈ [0.01, 0.5] (start from current realized vol)
│   │   ├─ Jump-Diffusion:
│   │   │   ├─ λ ∈ [0, 0.5] (0 → no jumps; 0.5 → one jump per 2 years)
│   │   │   ├─ μ_J ∈ [-0.1, 0.05] (negative = downside bias)
│   │   │   └─ σ_J ∈ [0.05, 0.5] (size volatility)
│   │   └─ Use domain knowledge / prior estimates
│   │
│   ├─ Step 4: Optimization
│   │   ├─ Objective function: MSE of implied vols (or prices)
│   │   ├─ Algorithm:
│   │   │   ├─ Levenberg-Marquardt: Local; fast; good for smooth objectives
│   │   │   ├─ BFGS: Quasi-Newton; good balance speed/robustness
│   │   │   ├─ Simulated Annealing: Global search; handles multiple minima; slow
│   │   │   └─ Particle Swarm: Newer; parallel; exploratory
│   │   ├─ Multiple restarts: Try 5-10 different initial guesses
│   │   ├─ Convergence criteria:
│   │   │   ├─ Gradient norm < 1e-5
│   │   │   ├─ Parameter changes < 1e-6
│   │   │   └─ Objective improvements < 1e-8
│   │   └─ Typical runtime: Heston 1-5 seconds; Bates 5-30 seconds (per reoptimization)
│   │
│   ├─ Step 5: Validation (In-Sample)
│   │   ├─ Residuals:
│   │   │   ├─ R²: >0.95 (good fit); 0.90-0.95 (acceptable); <0.90 (poor)
│   │   │   ├─ RMSE (IV basis points): <5 bps (excellent); 5-20 bps (good); >50 bps (poor)
│   │   │   └─ Max error: Monitor tail residuals; flag outliers
│   │   ├─ Smile capture:
│   │   │   ├─ ATM fit: Essential; residuals here < 1 bps
│   │   │   ├─ Wing fit: OTM puts/calls; residuals < 3 bps
│   │   │   └─ Plot residuals by strike; look for systematic patterns
│   │   ├─ Term structure:
│   │   │   ├─ Check fit across maturities (1M, 3M, 6M, 1Y, 2Y, 5Y)
│   │   │   └─ Verify model captures term structure shape (U-shape, hump, etc.)
│   │   └─ Parameter stability:
│   │       ├─ Daily calibration: Parameters change gradually (not erratically)
│   │       ├─ Jackknife: Exclude one data point; recalibrate; parameters should be similar
│   │       └─ If unstable: Non-unique; ill-conditioned; need regularization or constraints
│   │
│   ├─ Step 6: Out-of-Sample Validation
│   │   ├─ Forward-test: Hold out 20% of options
│   │   ├─ Calibrate on remaining 80%; test on held-out
│   │   ├─ Compare model prices to forward prices
│   │   ├─ If forward-test errors >> in-sample errors → Overfitting
│   │   ├─ Remedy:
│   │   │   ├─ Reduce parameters (simpler model)
│   │   │   ├─ Add regularization penalty
│   │   │   ├─ Use Bayesian priors (shrinkage)
│   │   │   └─ Increase data (if possible)
│   │   └─ Iterate until forward-test satisfactory
│   │
│   ├─ Step 7: Exotic Pricing Test
│   │   ├─ Price exotic option with calibrated model (e.g., Bermudan swaption)
│   │   ├─ Compare to market quotes (if available) or dealer consensus
│   │   ├─ If model prices off by >5-10% → Model inadequate
│   │   ├─ Debug:
│   │   │   ├─ Switch model (Heston → Bates)
│   │   │   ├─ Recalibrate on exotic data too
│   │   │   ├─ Check Monte Carlo convergence (if using simulation)
│   │   │   └─ Verify no-arbitrage (put-call parity, etc.)
│   │   └─ If mismatch persists → Model risk; document; hedge externally
│   │
│   └─ Step 8: Deploy & Monitor
│       ├─ Recalibration frequency:
│       │   ├─ Liquid markets: Daily (after market close)
│       │   ├─ Semi-liquid: Weekly
│       │   ├─ Crisis: Intraday (every hour) to track rapidly changing IV
│       │   └─ Monitoring alert: Parameter change > 10% from previous → Investigate
│       ├─ Parameter tracking:
│       │   ├─ Plot κ, θ, σ, ρ over time (time series charts)
│       │   ├─ Flagged jumps (market shocks; model failure; data error)
│       │   ├─ Rolling diagnostics: Last 30 days fit quality, residuals
│       │   └─ PnL attribution: Track pricing errors; loss from model risk
│       └─ Governance:
│           ├─ Independent validation: Separate team checks calibrations
│           ├─ Backtesting: Compare model prices to subsequent market prices
│           ├─ Escalation: Alert if calibration fails to converge; residuals spike
│           └─ Model change approval: Require sign-off before switching models
│
├─ Parameter Interpretation & Insight:
│   ├─ Heston κ (Mean Reversion Speed):
│   │   ├─ κ = 1: Half-life of vol shocks ≈ 0.7 years ≈ 260 trading days
│   │   ├─ κ = 5: Half-life ≈ 55 trading days
│   │   ├─ High κ: Volatility spikes revert quickly (mean-reverting market)
│   │   ├─ Low κ: Volatility persistent (trending market)
│   │   └─ Typical equity: κ ∈ [0.5, 2] (moderate reversion)
│   │
│   ├─ Heston θ (Long-Run Variance):
│   │   ├─ θ = 0.04 → √θ = 20% (long-term expected volatility)
│   │   ├─ Often calibrated > current vol (forward-looking; expects vol rise)
│   │   ├─ θ = current vol²: Flat term structure expected
│   │   ├─ θ > current vol²: Rising term structure (vol spike expected)
│   │   └─ Used for forecasting; 3Y+ vols revert to √θ
│   │
│   ├─ Heston σ (Volatility of Volatility):
│   │   ├─ σ = 0: No vol-of-vol; ATM IV constant (severe model misspecification)
│   │   ├─ σ = 0.5: Moderate volatility clustering; typical range
│   │   ├─ σ > 1: Extreme volatility regime changes; crisis dynamics
│   │   ├─ Higher σ → Fatter tails; higher kurtosis
│   │   └─ OTM option prices sensitive to σ; market risk
│   │
│   ├─ Heston ρ (Leverage Correlation):
│   │   ├─ ρ = -0.5: Downside large moves → Vol spikes (typical equities)
│   │   ├─ ρ = 0: No leverage; vol independent of returns (rare; FX sometimes)
│   │   ├─ ρ = +0.5: Upside moves → Vol rises (structural change; anti-leverage)
│   │   └─ Determines skew shape: ρ < 0 → Put skew; ρ > 0 → Call skew
│   │
│   └─ Jump Parameters:
│       ├─ λ (Jump Intensity):
│       │   ├─ λ = 0.05: ~1 jump per 20 years (low frequency)
│       │   ├─ λ = 0.20: ~1 jump per 5 years (moderate)
│       │   ├─ λ = 1.0: ~1 jump per year (very active; crisis regime)
│       │   └─ Pre-crisis typical λ ≈ 0.05-0.10; crisis λ > 0.5
│       ├─ μ_J (Mean Jump Size):
│       │   ├─ μ_J = -0.05: Downside bias; -5% average jump
│       │   ├─ μ_J > 0: Upside bias (rare unless rally regime)
│       │   └─ Magnitude determines tail skewness
│       └─ σ_J (Jump Size Volatility):
│           ├─ σ_J = 0.1: Precise jumps; e.g., exactly -5% ± 10%
│           ├─ σ_J = 0.3: Dispersed jump sizes; -5% ± 30% (more realistic)
│           └─ Larger σ_J → Fatter tails
│
└─ Advanced Topics:
    ├─ Regime-Switching Calibration:
    │   ├─ Model: Two regimes (normal + crisis) with different parameters
    │   ├─ Markov chain: Probability transitions between regimes each day
    │   ├─ Calibration: MLE on joint probability of returns + regime sequence
    │   ├─ Advantage: Captures structural changes without complete recalibration
    │   └─ Challenge: More parameters; harder optimization
    │
    ├─ Bayesian Calibration:
    │   ├─ Prior: Expert beliefs on parameter ranges
    │   ├─ Likelihood: Market prices (data)
    │   ├─ Posterior: Updated beliefs (combining prior + data)
    │   ├─ MCMC sampling: Draw parameter samples from posterior distribution
    │   ├─ Advantage: Parameter uncertainty quantified; small samples robust
    │   └─ Disadvantage: Computationally intensive (100K+ iterations)
    │
    ├─ Particle Filter Calibration:
    │   ├─ Sequential estimation: Recalibrate as new data arrives
    │   ├─ Bayesian update: p(θ|data) updated incrementally
    │   ├─ Advantage: Real-time tracking; adaptive to regime changes
    │   └─ Practical: Used in risk management; algorithmic trading
    │
    └─ Ensemble Calibration:
        ├─ Fit multiple models (Heston, Jump-Diffusion, Local Vol)
        ├─ Average pricing: Price exotic with each model; take weighted average
        ├─ Advantage: Reduces model-specific risk; robust
        └─ Trade-off: Slower; requires maintaining multiple models
```

**Key Insight:** Stochastic model calibration aims to recover parameters capturing market risk (volatility clustering, jumps, leverage); Heston good for smooth vol; Jumps for tail events; Bates comprehensive but 8 parameters increase overfitting risk; always validate out-of-sample and on exotics; parameters can be unstable (non-unique) → use regularization or Bayesian priors; recalibration frequency depends on market stability.

## Challenge Round
Stochastic model calibration difficulties:
- **Local Minima**: Optimizer converges to local min; parameters suboptimal; solution: Multiple restarts; global optimizers; parameter bounds; domain constraints
- **Parameter Non-Uniqueness**: Multiple (κ, θ, σ, ρ) combos fit equally well; forward-test fails; solution: Bayesian priors; regularization; additional exotic constraints
- **Overfitting (Too Many Parameters)**: 5 (Heston) or 8 (Bates) parameters vs 15 market points → Overfit in-sample; poor forward-test; solution: Fewer parameters; cross-validation; simplify model
- **Feller Condition Violation**: 2κθ < σ² → Volatility can go negative → Pricing breaks; solution: Enforce constraint; reparameterize; use transformed parameters (always > 0)
- **Numerical Instability**: Heston pricing via Fourier inversion unstable at extreme strikes; solution: Use robust integration methods (COS, FFT); characteristic function pole handling
- **Regime Change**: Pre-crisis parameters (low λ) fail during crisis; recalibration essential; solution: Intraday recalibration; regime-switching model; rolling windows

## Key References
- [Heston: A Closed-Form Solution for Options with Stochastic Volatility (1993)](https://www.jstor.org/stable/2328279) - Seminal work; characteristic function approach; foundational formula
- [Broadie & Kaya: Exact Simulation of Stochastic Volatility and Jump Diffusion Models (2006)](https://pubsonline.informs.org/doi/abs/10.1287/opre.1050.0223) - Bates model; exact simulation methods; practical implementation
- [Singleton & Umantsev: Calibrating LIBOR and Swap Curves (2002)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=339126) - Interest rate calibration; curve fitting; practical considerations

---
**Status:** Derivative Pricing Advanced | **Pairs Well With:** Exotic Options Pricing, Model Risk, Monte Carlo Methods
