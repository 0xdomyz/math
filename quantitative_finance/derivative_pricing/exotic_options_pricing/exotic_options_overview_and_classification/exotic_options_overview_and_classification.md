# Exotic Options: Overview & Classification

## Concept Skeleton
**Definition:** Non-standard options with payoff structures or exercise features departing from vanilla calls/puts; includes barriers, lookbacks, Asians, baskets, and exotics with path-dependency  
**Purpose:** Provide customized risk exposure; reduce hedging costs vs vanilla replication; enable structured products; match specific market views  
**Prerequisites:** Vanilla option pricing, Monte Carlo methods, PDE solvers, path-dependent payoffs, risk-neutral valuation

## Comparative Framing
| Category | Type | Payoff | Path-Dependent | Pricing |
|----------|------|--------|----------------|---------|
| **Barrier** | Down-and-out call | Standard if S>L | Yes | Lattice/MC |
| **Lookback** | Float strike call | max(S)-K | Yes | MC/FD |
| **Asian** | Arithmetic average | A(T)-K | Yes (full path) | MC (excellent) |
| **Basket** | Multi-asset call | max(w₁S₁+...,K) | Weakly | MC |
| **Chooser** | Choose call/put | max(C,P) | No | Closed-form |
| **Quanto** | FX-adjusted | S_foreign × FX | Weakly | Closed-form variant |
| **Cliquet** | Reset coupon | ∑max(S_i/S_{i-1}-K,0) | Reset dates | Closed-form each |
| **Variance Swap** | Realized variance | (σ_realized - σ_fixed)² | Yes | Model-dependent |

## Examples + Counterexamples
**Simple Barrier Example:**  
Down-and-out call: S=$110, K=$100, r=5%, σ=20%, T=1yr, barrier L=$95. As long as S>$95, acts like vanilla. If S touches $95, option worthless. Premium less than vanilla (protection removed).

**Lookback Call Success:**  
S(T)=$110, max(S)=$120. Payoff = $120-K = higher than vanilla $110-K. Lookback max captures best price along path.

**Asian Put Advantage:**  
Volatile stock ending in-the-money. Asian (average) dampens volatility vs vanilla; cheaper hedge for corporations averaging cash flows.

**Basket Call Correlation Impact:**  
Two stocks, each σ=25%. Uncorrelated (ρ=0): Basket call moderate volatility. Perfectly correlated (ρ=1): Basket volatility ≈ individual stock volatility. Price difference ~10-15%.

**Quanto Forex Timing:**  
Japanese stock, USD investor. If JPY weakens, even if Nikkei rises, investor returns reduced. Quanto call guarantees FX rate, removes currency risk.

## Layer Breakdown
```
Exotic Options Classification & Framework:

├─ Main Categories (by mechanism):
│  ├─ Barrier Options:
│  │   ├─ Knock-in (activated if S crosses level)
│  │   ├─ Knock-out (eliminated if S crosses level)
│  │   ├─ Barriers: Down (S < L), Up (S > U)
│  │   └─ Usage: Reduce cost vs vanilla, specify scenarios
│  ├─ Path-Dependent (Lookback):
│  │   ├─ Payoff uses max/min of S(t) over [0,T]
│  │   ├─ Float strike: max(S) - K (better than vanilla for buyer)
│  │   └─ Fixed strike: (max(S) - strike) as call; (strike - min(S)) as put
│  ├─ Average-Based (Asian):
│  │   ├─ Arithmetic: (1/n)∑S_i (path-dependent, full history)
│  │   ├─ Geometric: (∏S_i)^(1/n) (analytic in BS framework)
│  │   ├─ Weighted average: ∑w_i*S_i
│  │   └─ Used: Corporate hedges (COGS averaging), reduce volatility
│  ├─ Multi-Asset:
│  │   ├─ Basket: Linear combination ∑w_i*S_i (correlated stock bundle)
│  │   ├─ Best-of: max(S₁, S₂, ...) as payoff
│  │   ├─ Worst-of: min(S₁, S₂, ...)
│  │   ├─ Spread: S₁ - S₂ (commodity spread, crack spread)
│  │   └─ Correlation-sensitive: Pricing depends on ρ_ij
│  ├─ Timing/Optionality:
│  │   ├─ Chooser: Decide later (call or put)
│  │   ├─ Swing: Multiple exercise dates within window
│  │   ├─ Bermuda: Discrete dates (between European ↔ American)
│  │   └─ Callable: Issuer can redeem early
│  ├─ Structural Variants:
│  │   ├─ Quanto: FX-adjusted payoff (currency locked)
│  │   ├─ Cliquet: Reset coupon (best-of period returns)
│  │   ├─ Pyramid: Layered strikes, rebates
│  │   └─ Range accrual: Coupon if S stays in band
│  └─ Volatility-Linked:
│      ├─ Variance swap: Realized vs fixed volatility
│      ├─ Volatility swap: Realized vs fixed vol (linear)
│      ├─ VIX options: Options on volatility index
│      └─ Dispersion trades: Corr arbitrage
├─ Pricing Complexity Ladder:
│  ├─ Level 1 (Semi-Analytical):
│  │   ├─ Barrier (analytical bounds via reflection principle)
│  │   ├─ Asian geometric mean (closed-form in BS)
│  │   ├─ Chooser (simple decomposition)
│  │   └─ Cliquet (series of vanilla options)
│  ├─ Level 2 (Numerical - Lattice/FD):
│  │   ├─ Arithmetic Asian (FD or trinomial tree)
│  │   ├─ Lookback (state space augmented)
│  │   ├─ Barrier with rebates (tracking nodes)
│  │   └─ Swing options (dynamic programming)
│  ├─ Level 3 (Monte Carlo Essential):
│  │   ├─ Multi-dimensional basket
│  │   ├─ Lookback (large number of observations)
│  │   ├─ Complex path dependencies
│  │   └─ Stochastic volatility + exotic
│  └─ Level 4 (Advanced):
│      ├─ Variance swaps (realized vol theory)
│      ├─ Dispersion trades (correlation model)
│      ├─ Exotic American (perpetual optimal stopping)
│      └─ Counterparty risk adjustment (CVA)
├─ Payoff Mechanics:
│  ├─ European-style exotics:
│  │   ├─ Exercise only at T
│  │   ├─ Payoff determined by path S(t) ∀t∈[0,T]
│  │   ├─ Easier to value (known future boundary)
│  │   └─ Example: Asian, lookback
│  ├─ American-style exotics:
│  │   ├─ Exercise at any time τ ≤ T
│  │   ├─ Optimal stopping + path dependence
│  │   ├─ Harder to value (free boundary)
│  │   └─ Example: Bermuda, swing
│  └─ Early Exercise Features:
│      ├─ Knock-out: Automatic termination (exogenous)
│      ├─ Knock-in: Activation (threshold-triggered)
│      └─ Callable: Issuer discretion (optimal stopping)
├─ Valuation Approaches:
│  ├─ Analytical (rare):
│  │   ├─ Closed-form: Chooser, some barriers, cliquet
│  │   ├─ Reflection principle: Some barrier payoffs
│  │   └─ Integral form: Lookback bounds
│  ├─ Semi-Analytical:
│  │   ├─ Characteristic function (FFT): Swing options
│  │   ├─ Laplace transform: Barrier with rebates
│  │   └─ Eigenfunction expansion: Some PDE exotics
│  ├─ Numerical (Deterministic):
│  │   ├─ Finite difference (multi-dimensional PDE)
│  │   ├─ Binomial/Trinomial (extended state space)
│  │   ├─ Finite element: Unstructured boundaries
│  │   └─ Lattice methods: General framework
│  ├─ Numerical (Stochastic):
│  │   ├─ Monte Carlo (embarrassingly parallel)
│  │   ├─ Antithetic variates (variance reduction)
│  │   ├─ Control variates (vs vanilla)
│  │   └─ Quasi-random (deterministic sampling, faster)
│  └─ Approximation Methods:
│      ├─ Perturbation: Approximate closed-form
│      ├─ Taylor expansion: Around vanilla
│      └─ Binomial "superposition": Combine single-step solutions
├─ Implementation Considerations:
│  ├─ Path Sampling:
│  │   ├─ Continuous path: Brownian bridge for barrier
│  │   ├─ Discrete observation: Asian average dates
│  │   ├─ Lookback minimum: Track running minimum
│  │   └─ Grid refinement: Adaptive vs uniform
│  ├─ Boundary Treatment:
│  │   ├─ Absorbing barrier: Option terminates
│  │   ├─ Reflecting barrier: Path bounces (knock-ins)
│  │   ├─ Sticky strikes: Rebates depend on path to barrier
│  │   └─ Continuous monitoring: Vs discrete observation dates
│  ├─ Computational Efficiency:
│  │   ├─ Parallelization: MC paths independent
│  │   ├─ GPU acceleration: CUDA for large simulations
│  │   ├─ Antithetic + stratified: Cut variance 50-90%
│  │   └─ Adaptive sampling: Spend effort on high-variance regions
│  └─ Calibration & Validation:
│      ├─ Greeks: Numerical differentiation (bump-and-reprice)
│      ├─ Convergence: Mesh refinement until stable
│      ├─ Benchmark: Compare to analytical bounds where available
│      └─ Stress testing: Extreme parameters, barrier proximity
└─ Market Applications:
   ├─ Structured Products:
   │   ├─ Capital-protected notes (barrier puts embedded)
   │   ├─ Autocallable (cliquet + barrier)
   │   ├─ Range accruals (corporate hedges)
   │   └─ Participation notes (basket calls)
   ├─ Equity/Commodity:
   │   ├─ Knock-out calls (reduce cost)
   │   ├─ Knock-in puts (downside protection)
   │   ├─ Asian puts (hedge averaging expenses)
   │   └─ Barrier call spreads (cap risk)
   ├─ FX/Rates:
   │   ├─ Quanto calls (currency-hedged)
   │   ├─ Swing options (energy trading)
   │   ├─ Range notes (interest rate notes)
   │   └─ Binary options (one-touch)
   └─ Risk Management:
       ├─ Volatility arbitrage (variance swaps)
       ├─ Correlation trading (basket spreads)
       ├─ Exotic gamma management (barrier hedge)
       └─ Synthetic replication (exotic as vanilla portfolio)
```

**Interaction:** Exotic payoff structure → path-dependent valuation → numerical method choice → Greeks & Greeks of Greeks → risk management.

## Challenge Round
- Why are Asian options cheaper than vanilla options?
- Explain reflection principle for barrier options (boundary condition)
- How does correlation affect basket option pricing?
- Design a structured product combining barrier + cliquet
- Compare variance swap pricing vs volatility swap

## Key References
- [Zhang, P. G. "Exotic Options" (2nd ed., 1998)](https://www.wiley.com/en-us/Exotic+Options%2C+2nd+Edition-p-9780471975946) — Comprehensive exotic reference
- [Haug, E. G. "Complete Guide to Option Pricing Formulas"](https://www.wiley.com/en-us/Complete+Guide+to+Option+Pricing+Formulas%2C+2nd+Edition-p-9780071389976) — Formulas & examples
- [Wilmott, P. "On Quant Finance" (Volume 2)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119287742) — Theory & implementation

---
**Status:** Comprehensive exotic taxonomy | **Complements:** Barrier Options, Asian Options, Lookback Options, Monte Carlo Methods
