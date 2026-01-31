# Monte Carlo Pricing Topics Guide

**Complete reference of Monte Carlo simulation concepts, pricing methods, and variance reduction techniques.**

---

## I. Monte Carlo Fundamentals

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Random Number Generation** | N/A | Pseudo-random, quasi-random (Sobol, Halton); uniform [0,1] seeds | [Wiki - RNG](https://en.wikipedia.org/wiki/Random_number_generation) |
| **Pseudorandom Sequences** | N/A | Deterministic sequences with statistical properties of randomness | [Wiki - Pseudorandom](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) |
| **Quasi-Random Numbers (Low-Discrepancy)** | N/A | Sobol, Halton, Niederreiter; uniform coverage, faster convergence | [Wiki - Low-Discrepancy](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) |
| **Box-Muller Transform** | N/A | Convert uniform RN to normal RN; pairs of standard normals | [Wiki - Box-Muller](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) |
| **Inverse Transform Method** | N/A | Generate RN from any distribution using inverse CDF | [Wiki - Inverse Transform](https://en.wikipedia.org/wiki/Inverse_transform_sampling) |
| **Acceptance-Rejection Method** | N/A | Generate samples from complex distributions via rejection | [Wiki - Rejection Sampling](https://en.wikipedia.org/wiki/Rejection_sampling) |

---

## II. Stochastic Processes for Asset Prices

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Geometric Brownian Motion (GBM)** | ✓ monte_carlo_pricing.md | dS = μS dt + σS dW; lognormal price evolution | [Wiki - GBM](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) |
| **Brownian Motion (Wiener Process)** | N/A | Continuous-time random walk; dW ~ N(0, dt) | [Wiki - Brownian](https://en.wikipedia.org/wiki/Wiener_process) |
| **Drift & Volatility** | N/A | μ (expected return), σ (price uncertainty); calibrated from data | [Wiki - Drift](https://en.wikipedia.org/wiki/Drift_(random_walk)) |
| **Ito's Lemma** | N/A | Stochastic calculus chain rule; derivatives of random processes | [Wiki - Ito](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma) |
| **Jump Diffusion Models** | N/A | Merton model; continuous drift + discontinuous jumps | [Wiki - Jump Diffusion](https://en.wikipedia.org/wiki/Jump_process) |
| **Ornstein-Uhlenbeck Process** | N/A | Mean-reverting process; interest rates, volatility | [Wiki - OU](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) |
| **Vasicek Model** | N/A | Interest rate model; mean-reversion for rates | [Wiki - Vasicek](https://en.wikipedia.org/wiki/Vasicek_model) |
| **Hull-White Model** | N/A | Extended Vasicek; matches yield curve | [Wiki - Hull-White](https://en.wikipedia.org/wiki/Hull%E2%80%93White_model) |
| **Cox-Ingersoll-Ross (CIR) Model** | N/A | Mean-reverting square-root diffusion; non-negative rates | [Wiki - CIR](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model) |
| **Heston Model** | N/A | Stochastic volatility; ξ evolves separately; calibrated to options | [Wiki - Heston](https://en.wikipedia.org/wiki/Heston_model) |

---

## III. Monte Carlo Simulation Framework

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Simulation Setup** | ✓ monte_carlo_pricing.md | Define asset model, time steps, paths, payoff function | Internal |
| **Path Generation** | ✓ monte_carlo_pricing.md | Discrete-time discretization; Euler, Milstein schemes | [Wiki - Path](https://en.wikipedia.org/wiki/Gillespie_algorithm) |
| **Time Discretization (Δt)** | ✓ monte_carlo_pricing.md | Smaller Δt → more accurate, higher computation cost | [Wiki - Discretization](https://en.wikipedia.org/wiki/Discretization) |
| **Euler Scheme** | ✓ monte_carlo_pricing.md | S_{n+1} = S_n + μS_n Δt + σS_n √Δt Z_n; order O(Δt) | [Wiki - Euler](https://en.wikipedia.org/wiki/Euler_method) |
| **Milstein Scheme** | N/A | Higher-order; includes σ² term; order O((Δt)²) | [Wiki - Milstein](https://en.wikipedia.org/wiki/Milstein_method) |
| **Payoff Function** | ✓ monte_carlo_pricing.md | Option value at maturity T; European: max(S_T - K, 0) for call | Internal |
| **Price Estimation** | ✓ monte_carlo_pricing.md | Average discounted payoff over N paths; e^{-rT} E[payoff] | Internal |
| **Convergence** | N/A | Standard error decreases as O(1/√N); law of large numbers | [Wiki - LLN](https://en.wikipedia.org/wiki/Law_of_large_numbers) |
| **Standard Error** | ✓ monte_carlo_pricing.md | SE = σ_{payoff} / √N; depends on payoff variance | Internal |
| **Confidence Intervals** | ✓ monte_carlo_pricing.md | Price ± 1.96 × SE for 95% CI | [Wiki - CI](https://en.wikipedia.org/wiki/Confidence_interval) |

---

## IV. European Options Pricing

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **European Call Option** | N/A | Payoff: max(S_T - K, 0); exercisable only at maturity | [Wiki - Call](https://en.wikipedia.org/wiki/Call_option) |
| **European Put Option** | N/A | Payoff: max(K - S_T, 0); exercisable only at maturity | [Wiki - Put](https://en.wikipedia.org/wiki/Put_option) |
| **Black-Scholes Closed Form** | N/A | Analytic solution for European options under GBM; benchmark | [Wiki - Black-Scholes](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) |
| **Monte Carlo vs Black-Scholes** | N/A | MC approximates closed form; flexible for complex payoffs | Internal |
| **Vanilla Options** | N/A | Standard European/American calls and puts | [Wiki - Vanilla](https://en.wikipedia.org/wiki/Vanilla_option) |

---

## V. American & Exotic Options Pricing

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **American Option Pricing** | N/A | Early exercise possible; requires backward induction or binomial | [Wiki - American](https://en.wikipedia.org/wiki/American_option) |
| **Longstaff-Schwartz Algorithm** | N/A | MC method for American options; regression on in-the-money paths | [Paper - LS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=191649) |
| **Asian Options** | N/A | Payoff based on average price; path-dependent | [Wiki - Asian](https://en.wikipedia.org/wiki/Asian_option) |
| **Barrier Options** | N/A | Knock-in/knock-out when price crosses barrier; path-dependent | [Wiki - Barrier](https://en.wikipedia.org/wiki/Barrier_option) |
| **Lookback Options** | N/A | Payoff depends on min/max price over period | [Wiki - Lookback](https://en.wikipedia.org/wiki/Lookback_option) |
| **Basket Options** | N/A | Payoff on portfolio of assets; multivariate correlation | [Wiki - Basket](https://en.wikipedia.org/wiki/Basket_option) |
| **Bermudan Options** | N/A | Exercise on specific dates (between European & American) | [Wiki - Bermudan](https://en.wikipedia.org/wiki/Bermudan_option) |
| **Rainbow Options** | N/A | Multiple underlying assets; complex correlation structure | [Wiki - Rainbow](https://en.wikipedia.org/wiki/Rainbow_option) |

---

## VI. Variance Reduction Techniques

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Antithetic Variates** | N/A | Use Z and -Z pairs; halves variance; highly effective | [Wiki - Antithetic](https://en.wikipedia.org/wiki/Antithetic_variates) |
| **Control Variates** | N/A | Subtract correlated known quantity; reduces payoff variance | [Wiki - Control](https://en.wikipedia.org/wiki/Control_variate) |
| **Importance Sampling** | N/A | Oversample relevant regions (in-the-money paths) | [Wiki - Importance](https://en.wikipedia.org/wiki/Importance_sampling) |
| **Stratified Sampling** | N/A | Divide sample space into strata; sample each uniformly | [Wiki - Stratified](https://en.wikipedia.org/wiki/Stratified_sampling) |
| **Moment Matching** | N/A | Force sample mean/variance to match theoretical values | [Wiki - Moment Matching](https://en.wikipedia.org/wiki/Moment_matching) |
| **Latin Hypercube Sampling** | N/A | Quasi-random for multivariate; better coverage than random | [Wiki - LHS](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) |
| **Conditional MC** | N/A | Condition on known quantities; reduce effective dimensionality | Internal |

---

## VII. Multivariate & Correlated Paths

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Correlation Matrix** | N/A | ρ ∈ [-1, 1]; pairwise asset dependence | [Wiki - Correlation](https://en.wikipedia.org/wiki/Correlation_and_dependence) |
| **Cholesky Decomposition** | N/A | Factor correlation matrix; generate correlated normal RVs | [Wiki - Cholesky](https://en.wikipedia.org/wiki/Cholesky_decomposition) |
| **Principal Component Analysis (PCA)** | N/A | Reduce correlation matrix to principal components | [Wiki - PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) |
| **Copulas** | N/A | Separate marginals from joint dependence; Gaussian, Student-t | [Wiki - Copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)) |
| **Multi-Asset Simulation** | N/A | Simulate basket of stocks with realistic correlations | Internal |

---

## VIII. Model Calibration & Parameters

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Volatility Estimation** | N/A | Historical volatility (std of returns); implied (from option prices) | [Wiki - Volatility](https://en.wikipedia.org/wiki/Volatility_(finance)) |
| **Historical Volatility** | N/A | σ = std(ln(S_{t+1}/S_t)); backward-looking | Internal |
| **Implied Volatility** | N/A | Inverse BS: find σ matching observed option price | [Wiki - IV](https://en.wikipedia.org/wiki/Implied_volatility) |
| **GARCH Models** | N/A | Heteroskedastic volatility; ξ_t ~ N(0, σ_t²); σ_t evolves | [Wiki - GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) |
| **Smile & Skew** | N/A | IV varies by strike; volatility surface calibration | [Wiki - Volatility Smile](https://en.wikipedia.org/wiki/Volatility_smile) |
| **Interest Rate Curve** | N/A | Discount factors; zero-coupon bond prices by maturity | [Wiki - Yield Curve](https://en.wikipedia.org/wiki/Yield_curve) |
| **Repo Rates & Dividends** | N/A | Borrow costs, dividend yields; adjust forward price | Internal |

---

## IX. Greeks & Sensitivities (Delta Hedging)

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Delta (Δ)** | N/A | ∂Price/∂S; hedge ratio; probability of ITM (call) | [Wiki - Delta](https://en.wikipedia.org/wiki/Delta_(finance)) |
| **Gamma (Γ)** | N/A | ∂²Price/∂S²; convexity; delta changes with S | [Wiki - Gamma](https://en.wikipedia.org/wiki/Gamma_(finance)) |
| **Vega (ν)** | N/A | ∂Price/∂σ; volatility sensitivity; key risk in options | [Wiki - Vega](https://en.wikipedia.org/wiki/Vega_(finance)) |
| **Theta (θ)** | N/A | ∂Price/∂T; time decay; positive for long puts/calls near maturity | [Wiki - Theta](https://en.wikipedia.org/wiki/Theta_(finance)) |
| **Rho (ρ)** | N/A | ∂Price/∂r; interest rate sensitivity; minimal for short-dated | [Wiki - Rho](https://en.wikipedia.org/wiki/Rho_(finance)) |
| **Greek Computation** | N/A | Finite differences from MC paths; pathwise derivative | Internal |
| **Hedging Strategy** | N/A | Dynamic rebalancing; delta-neutral portfolio maintenance | [Wiki - Hedging](https://en.wikipedia.org/wiki/Hedge_(finance)) |

---

## X. Computational Methods & Implementation

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Vectorization** | N/A | NumPy/array operations; efficient large-scale matrix computation | [NumPy Docs](https://numpy.org/doc/stable/) |
| **Parallel Computing** | N/A | GPU/multicore; embarrassingly parallel path generation | [CUDA Docs](https://docs.nvidia.com/cuda/) |
| **Numba JIT Compilation** | N/A | Speed up Python MC; compile to machine code | [Numba Docs](https://numba.readthedocs.io/) |
| **Memory Efficiency** | N/A | Avoid storing all paths; stream-process discounted payoffs | Internal |
| **Execution Speed Trade-offs** | N/A | Speed vs accuracy; SE depends on N; diminishing returns | Internal |

---

## XI. Validation & Benchmarking

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Convergence Testing** | N/A | SE → 0 as N → ∞; verify O(1/√N) empirically | Internal |
| **Comparison to Closed Form** | ✓ monte_carlo_pricing.md | Price vs Black-Scholes for European options; validate code | Internal |
| **Sensitivity Analysis** | N/A | Vary inputs (S, K, T, r, σ); check expected directional changes | Internal |
| **P&L Verification** | N/A | Back-test on historical data; delta hedge profitability | Internal |
| **Model Risk** | N/A | Assumptions (normality, constant σ); stress test edge cases | [Wiki - Model Risk](https://en.wikipedia.org/wiki/Model_risk) |

---

## XII. Advanced Topics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Quasi-MC (QMC) Integration** | N/A | Sobol sequences; O(log^d N) vs O(1/√N); high dimensions | [Wiki - QMC](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method) |
| **Multilevel Monte Carlo (MLMC)** | N/A | Combine coarse & fine grids; O(ε^{-2}) vs O(ε^{-3}); efficiency gain | [Paper - MLMC](https://people.maths.ox.ac.uk/gilesm/mlmc_review.pdf) |
| **Machine Learning for Pricing** | N/A | Neural networks replace MC; fast inference post-training | Internal |
| **Adjoint AD (Automatic Differentiation)** | N/A | Compute Greeks via reverse-mode AD; O(1) cost per path | [Wiki - AD](https://en.wikipedia.org/wiki/Automatic_differentiation) |
| **Rare Event Simulation** | N/A | Importance sampling for tail risk; VaR, expected shortfall | [Wiki - Rare Events](https://en.wikipedia.org/wiki/Importance_sampling) |

---

## XIII. Practical Considerations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Bid-Ask Spread** | N/A | Transaction costs; reduce MC price by half-spread for profit margin | [Wiki - Spread](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread) |
| **Liquidity Risk** | N/A | Slippage; assume prices realizable at quoted levels | [Wiki - Liquidity](https://en.wikipedia.org/wiki/Liquidity_risk) |
| **Counterparty Risk** | N/A | Credit exposure; adjust fair value by counterparty default probability | [Wiki - CVA](https://en.wikipedia.org/wiki/Credit_valuation_adjustment) |
| **Regulatory Capital** | N/A | Basel III; VaR, Stressed VaR, IRC | [Wiki - Basel](https://en.wikipedia.org/wiki/Basel_III) |
| **Scenario Analysis** | N/A | Stress test; extreme moves, correlation breaks | Internal |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **Wikipedia Finance & Pricing** | https://en.wikipedia.org/wiki/Mathematical_finance | Stochastic models, derivatives, options |
| **Paul Wilmott QF** | https://www.paulwilmott.com | Advanced derivatives, MC, calibration |
| **Numerical Recipes** | http://numerical.recipes | Algorithms, random numbers, integration |
| **Hull: Options, Futures, & Derivatives** | https://www-2.rotman.utoronto.ca/~hull | Comprehensive derivatives textbook |
| **Giles: Multilevel MC** | https://people.maths.ox.ac.uk/gilesm | MLMC efficiency, complexity analysis |

---

## Quick Stats

- **Total Topics Documented**: 85+
- **Workspace Files Created**: 1
- **Categories**: 13
- **MC-Specific Concepts**: 40+
- **Coverage**: Fundamentals → Simulation → Greeks → Implementation

