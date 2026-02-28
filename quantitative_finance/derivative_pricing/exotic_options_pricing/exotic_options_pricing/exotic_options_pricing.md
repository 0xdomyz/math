# Exotic Options Pricing

## Concept Skeleton
**Definition:** Non-standard derivatives with payoffs depending on path, multiple assets, or complex conditions beyond vanilla call/put structures  
**Purpose:** Tailor risk exposure to specific market views; reduce hedging costs; exploit market inefficiencies; enhance yield or protection  
**Prerequisites:** Black-Scholes framework, risk-neutral valuation, Monte Carlo methods, numerical PDE, path-dependent option mechanics

## Comparative Framing
| Option Type | Vanilla (European) | Barrier | Asian | Lookback | Digital |
|-------------|-------------------|---------|-------|----------|---------|
| **Payoff Depends On** | Terminal price | Path + barrier | Path average | Path extreme | Binary outcome |
| **Complexity** | Simple | Moderate | Moderate | High | Discontinuous |
| **Price vs Vanilla** | Benchmark | Cheaper | Cheaper | More expensive | Varies |
| **Hedging** | Delta-Gamma | Jump risk at barrier | Continuous | Requires path | Infinite gamma |
| **Closed-Form** | Yes (BS) | Sometimes | Geometric only | Rarely | Yes |

| Pricing Method | Monte Carlo | PDE/Finite Diff | Tree Methods | Analytical | Closed-Form Approx |
|----------------|-------------|-----------------|--------------|------------|--------------------|
| **Path-Dependent** | Excellent | Difficult | Non-recombining | Rare | Limited |
| **Multi-Asset** | Excellent | Curse of dim. | Infeasible | Very rare | Very limited |
| **Barriers** | Good (monitoring) | Good | Moderate | Some cases | Some cases |
| **Accuracy** | O(1/√n) | O(h²) | O(1/n) | Exact | Approximate |

## Examples + Counterexamples
**Simple Example:**  
Barrier knock-out call (barrier=$110, strike=$100, spot=$100): If price hits $110 before expiry, option worthless. Cheaper than vanilla since might knock out.

**Perfect Fit:**  
Asian option for commodity hedger: Payoff based on average oil price over quarter matches physical delivery pattern. Reduces manipulation risk at single fixing.

**Digital/Binary:**  
All-or-nothing call: Pays $100 if S_T>K, else $0. Used in structured notes. Infinite gamma near K at expiry → difficult to hedge.

**Lookback Call:**  
Payoff = S_T - min(S_t over [0,T]). Always ITM at expiry. Expensive (guarantees best execution). Popular in FX for importers/exporters.

**Basket Option:**  
Call on weighted portfolio of 5 stocks: Payoff = max(Σw_i S_i - K, 0). Cheaper than sum of individual calls due to diversification/correlation.

**Poor Fit:**  
Using Black-Scholes for barrier: Continuous monitoring assumption vs reality of discrete checks. Can misprice by 5-10% depending on frequency.

## Layer Breakdown
```
Exotic Options Framework:

├─ Path-Dependent Options:
│  ├─ Asian Options (Average Price):
│  │   ├─ Payoff Structures:
│  │   │   ├─ Arithmetic average: Payoff = max((1/n)Σ S_ti - K, 0)
│  │   │   ├─ Geometric average: Payoff = max(∏S_ti^(1/n) - K, 0)
│  │   │   ├─ Fixed strike: K predetermined
│  │   │   └─ Floating strike: K = average, payoff on terminal S_T
│  │   ├─ Pricing:
│  │   │   ├─ Geometric: Closed-form (adjusted BS)
│  │   │   │   Parameters: σ_geo = σ/√3, r_geo adjusted
│  │   │   ├─ Arithmetic: No closed-form, use Monte Carlo
│  │   │   ├─ Control variate: Use geometric as control
│  │   │   └─ PDE: Requires state variable for running average
│  │   ├─ Advantages:
│  │   │   ├─ Lower volatility → cheaper than vanilla
│  │   │   ├─ Manipulation-resistant (average vs single fixing)
│  │   │   └─ Matches cash flows for physical delivery
│  │   └─ Uses:
│  │       Commodities, currencies, equity compensation
│  ├─ Barrier Options:
│  │   ├─ Types:
│  │   │   ├─ Knock-out: Dies if barrier hit
│  │   │   │   ├─ Down-and-out: Lower barrier
│  │   │   │   └─ Up-and-out: Upper barrier
│  │   │   ├─ Knock-in: Activates if barrier hit
│  │   │   │   ├─ Down-and-in: Lower barrier
│  │   │   │   └─ Up-and-in: Upper barrier
│  │   │   ├─ Double barrier: Two barriers (in or out)
│  │   │   └─ Partial barriers: Active only during period
│  │   ├─ In-Out Parity:
│  │   │   Knock-in + Knock-out = Vanilla
│  │   │   Arbitrage relationship
│  │   ├─ Pricing:
│  │   │   ├─ Closed-form: Some cases (Merton, Reiner-Rubinstein)
│  │   │   ├─ Reflection principle: Mirror image method
│  │   │   ├─ Monte Carlo: Track barrier breaches
│  │   │   │   Continuous monitoring: Brownian bridge
│  │   │   │   Discrete monitoring: Actual path checks
│  │   │   └─ PDE: Boundary condition at barrier (value=0 or rebate)
│  │   ├─ Greeks:
│  │   │   ├─ Delta: Discontinuous at barrier
│  │   │   ├─ Gamma: Spikes near barrier
│  │   │   └─ Vega: Different behavior vs vanilla
│  │   └─ Practical Considerations:
│  │       ├─ Monitoring frequency: Daily, continuous, specific times
│  │       ├─ Rebate: Payment if knocked out
│  │       ├─ Hedging difficulty: Jump risk at barrier
│  │       └─ Used to cheapen vanilla (OTM barrier less likely)
│  ├─ Lookback Options:
│  │   ├─ Fixed Strike Lookback:
│  │   │   ├─ Call: max(max(S_t) - K, 0)
│  │   │   └─ Put: max(K - min(S_t), 0)
│  │   ├─ Floating Strike Lookback:
│  │   │   ├─ Call: S_T - min(S_t) (always ITM)
│  │   │   └─ Put: max(S_t) - S_T (always ITM)
│  │   ├─ Pricing:
│  │   │   ├─ Closed-form exists (Goldman et al.)
│  │   │   ├─ Involves cumulative normal integrals
│  │   │   ├─ Monte Carlo: Track running max/min
│  │   │   └─ PDE: Two state variables (S and max/min)
│  │   ├─ Value:
│  │   │   Expensive (guarantees best execution)
│  │   │   Floating strike: Worth more than fixed
│  │   └─ Uses:
│  │       FX (best rate), performance measurement
│  └─ Ladder Options:
│      ├─ Lock in profits at rungs (price levels)
│      ├─ Payoff = max of (locked gains, terminal payoff)
│      └─ Path-dependent with discrete memory points
├─ Multi-Asset Options:
│  ├─ Basket Options:
│  │   ├─ Payoff: max(Σ w_i S_i - K, 0)
│  │   │   Weighted sum of assets
│  │   ├─ Pricing:
│  │   │   ├─ No closed-form (non-lognormal sum)
│  │   │   ├─ Monte Carlo: Simulate correlated assets
│  │   │   │   Use Cholesky decomposition for correlation
│  │   │   ├─ Approximations: Moment-matching to lognormal
│  │   │   └─ Tree: Tensor product (infeasible for many assets)
│  │   ├─ Correlation Impact:
│  │   │   ├─ Higher correlation → closer to single asset
│  │   │   ├─ Lower correlation → diversification benefit
│  │   │   └─ Dispersion trade: Long basket, short components
│  │   └─ Uses:
│  │       Index options (custom), portfolio hedging
│  ├─ Rainbow Options:
│  │   ├─ Best-of / Worst-of:
│  │   │   ├─ Best-of call: max(max(S₁, S₂, ...) - K, 0)
│  │   │   ├─ Worst-of put: max(K - min(S₁, S₂, ...), 0)
│  │   │   └─ Best/worst of multiple assets
│  │   ├─ Pricing:
│  │   │   ├─ 2-asset: Closed-form (Stulz)
│  │   │   ├─ n-asset: Monte Carlo
│  │   │   └─ Correlation crucial: Determines spread
│  │   ├─ Value:
│  │   │   ├─ Best-of: More valuable than individual
│  │   │   ├─ Worst-of: Less valuable
│  │   │   └─ Correlation effect opposite for calls vs puts
│  │   └─ Uses:
│  │       Employee stock options (best of company/index)
│  │       Currency hedging (best rate of multiple pairs)
│  ├─ Spread Options:
│  │   ├─ Payoff: max(S₁ - S₂ - K, 0)
│  │   │   Difference between two assets
│  │   ├─ Exchange Option (Margrabe):
│  │   │   K=0: max(S₁ - S₂, 0)
│  │   │   Closed-form solution
│  │   ├─ Pricing:
│  │   │   ├─ Margrabe formula (K=0)
│  │   │   ├─ Kirk approximation (K>0)
│  │   │   └─ Monte Carlo for general case
│  │   └─ Uses:
│  │       Commodities (crack spreads), pairs trading
│  ├─ Quanto Options:
│  │   ├─ Payoff in different currency from underlying
│  │   ├─ Example: Nikkei option paying in USD
│  │   ├─ Pricing: Adjust drift for correlation
│  │   │   μ_quanto = μ - ρ σ_asset σ_FX
│  │   └─ Uses:
│  │       International investments without FX risk
│  └─ Correlation Options:
│      Direct bets on correlation between assets
│      Dispersion trading, correlation swaps
├─ Digital / Binary Options:
│  ├─ Cash-or-Nothing:
│  │   ├─ Call: Pays fixed amount C if S_T > K, else 0
│  │   ├─ Put: Pays C if S_T < K, else 0
│  │   ├─ Pricing: C × e^(-rT) × N(±d₂)
│  │   └─ Derivative of vanilla call w.r.t. K
│  ├─ Asset-or-Nothing:
│  │   ├─ Pays S_T if S_T > K (call) or S_T < K (put)
│  │   ├─ Pricing: S₀ × N(±d₁)
│  │   └─ Building block for vanillas
│  ├─ Greeks:
│  │   ├─ Delta: Spikes near strike at expiry
│  │   ├─ Gamma: Dirac delta function (infinite at K)
│  │   └─ Vega: Also spikes, changes sign near K
│  ├─ Hedging:
│  │   ├─ Extremely difficult near expiry
│  │   ├─ Small move → large delta change
│  │   └─ Often hedged with vanilla spreads
│  └─ Uses:
│      Structured products, binary bets, FX barriers
├─ Chooser / Compound Options:
│  ├─ Chooser:
│  │   ├─ Holder chooses call or put at future date
│  │   ├─ Simple chooser: Same K, T for both
│  │   ├─ Complex chooser: Different parameters
│  │   └─ Pricing: Closed-form for simple (Rubinstein)
│  ├─ Compound Options:
│  │   ├─ Option on an option
│  │   ├─ Call-on-call, put-on-put, call-on-put, put-on-call
│  │   ├─ Two strikes, two expiries (T₁ < T₂)
│  │   ├─ Pricing: Nested expectations, bivariate normal
│  │   └─ Uses: Real options (staged investment), volatility bets
│  └─ Value:
│      Optionality to wait → time value premium
├─ Variance / Volatility Products:
│  ├─ Variance Swaps:
│  │   ├─ Payoff: N × (σ²_realized - K_var)
│  │   │   N = notional per variance point
│  │   ├─ Realized variance: σ²_real = (252/n) Σ ln²(S_t/S_{t-1})
│  │   ├─ Fair strike: K_var = E[σ²_realized]
│  │   ├─ Pricing: Replication with log-contract
│  │   │   K_var = (2/T) ∫ C(K)/K² dK + put integral
│  │   │   Model-free using strip of options
│  │   └─ Properties:
│  │       ├─ Pure volatility exposure (convex in vol)
│  │       ├─ Vega: Constant across strikes
│  │       └─ Path-dependent (realized vol over period)
│  ├─ Volatility Swaps:
│  │   ├─ Payoff: N × (σ_realized - K_vol)
│  │   │   Linear in vol, not variance
│  │   ├─ Approximation: K_vol ≈ K_var - σ³/(8×K_var)
│  │   │   Convexity adjustment
│  │   └─ Less liquid than variance swaps
│  ├─ VIX Options:
│  │   ├─ Underlying: VIX index (30-day implied vol)
│  │   ├─ Pricing: Not lognormal (mean-reverting)
│  │   │   Use VIX futures as forward
│  │   └─ Hedging: Tail risk, vol spike protection
│  └─ Corridor Variance Swaps:
│      Only accrues when spot in corridor [L, H]
│      Reduces cost, targets specific scenarios
├─ Forward-Start / Cliquet Options:
│  ├─ Forward-Start:
│  │   ├─ Option granted now, strike set at future date
│  │   ├─ Typically K = S_T1 (at-the-money forward)
│  │   ├─ Pricing: Closed-form (homogeneity property)
│  │   │   V = S₀ × BS(1, 1, r, T₂-T₁, σ) / B(0,T₁)
│  │   └─ Uses: Employee stock options (ESO)
│  ├─ Cliquet (Ratchet):
│  │   ├─ Series of forward-start options
│  │   ├─ Locks in periodic gains (sum of returns)
│  │   ├─ Payoff: Σ max(α × return_i, floor)
│  │   │   α = participation rate, may have caps/floors
│  │   └─ Pricing: Sum of forward-starts with caps/floors
│  └─ Value:
│      Protection against vol spikes in future
│      Popular in structured products
├─ Other Exotic Structures:
│  ├─ Himalaya Options:
│  │   ├─ Basket with best performer removed each period
│  │   ├─ Payoff = sum of best assets at each date
│  │   └─ Reduces concentration risk
│  ├─ Napoleon Options:
│  │   Like Himalaya but worst performer removed
│  ├─ Shout Options:
│  │   Holder can "shout" once to lock in intrinsic value
│  │   Combines lookback and call features
│  ├─ Parisian Options:
│  │   Barrier triggered only if breached for continuous period
│  │   Less sensitive to brief spikes than standard barriers
│  └─ Power Options:
│      Payoff = (S_T)^α - K
│      Non-linear exposure, higher moments matter
└─ Pricing Considerations:
   ├─ Model Selection:
   │   ├─ GBM: Standard, may underprice barriers/digitals
   │   ├─ Jump-diffusion: Better for discontinuous payoffs
   │   ├─ Stochastic vol: Smile/skew dependent payoffs
   │   └─ Local vol: Path-dependent, barriers
   ├─ Numerical Methods:
   │   ├─ Monte Carlo: Path-dependent, multi-asset
   │   │   ├─ Variance reduction crucial (antithetic, control)
   │   │   ├─ Barriers: Brownian bridge for continuous monitoring
   │   │   └─ Discretization error: Euler vs Milstein
   │   ├─ PDE/Finite Difference:
   │   │   ├─ Low-dimensional (<3 assets)
   │   │   ├─ Barriers natural as boundary conditions
   │   │   └─ Stability, convergence issues for discontinuous payoffs
   │   ├─ Trees:
   │   │   ├─ Path-dependent: Non-recombining (exponential)
   │   │   └─ Better for American-style exotics
   │   └─ Semi-Analytical:
   │       Fourier methods, Laplace transforms for special cases
   ├─ Hedging Challenges:
   │   ├─ Path-dependence: Greeks change with history
   │   ├─ Barriers: Jump risk, discontinuous deltas
   │   ├─ Digitals: Infinite gamma at strike
   │   ├─ Multi-asset: Correlation risk (vega, vanna)
   │   └─ Dynamic replication often imperfect
   ├─ Market Practices:
   │   ├─ Bid-ask spreads: Wider than vanilla (illiquidity)
   │   ├─ Valuation adjustments: Model risk, liquidity
   │   ├─ Hedging costs: Built into price
   │   └─ Regulatory capital: Higher risk weights
   └─ Applications:
      ├─ Structured products: Tailored payoffs for retail
      ├─ Corporate hedging: Match cash flow patterns
      ├─ Trading strategies: Express specific views
      └─ Cost reduction: Barriers cheaper than vanillas
```

**Interaction:** Exotic payoff structure → Select pricing method → Model calibration → Risk management (Greeks, scenarios) → Dynamic hedging strategy.

## Challenge Round
1. **Parisian Barrier:** Implement Parisian option (barrier triggered only after continuous breach for time τ). How does window size τ affect price vs standard barrier?

2. **Cliquet Ratchet:** Price cliquet with annual resets, 100% participation, 0% floor, 10% cap per period (5 years). How sensitive to forward volatility term structure?

3. **Variance Swap Replication:** Replicate variance swap using strip of OTM calls and puts. Calculate fair strike. Compare to realized variance. Why difference?

4. **Himalaya Option:** Price 3-asset Himalaya (best performer removed each year). Use nested Monte Carlo for dynamic selection. Compare to sum of individual lookbacks.

5. **Smile Impact on Digitals:** Price digital call using flat vol vs volatility smile. How much difference? Explain via risk-neutral density impact.

## Key References
- [Haug, The Complete Guide to Option Pricing Formulas (Part II)](https://www.mhprofessional.com/the-complete-guide-to-option-pricing-formulas-9780071389976-usa)
- [Gatheral, The Volatility Surface (Chapter 6 - Exotic Options)](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792529)
- [Wilmott, Paul Wilmott Introduces Quantitative Finance (Chapter 13)](https://www.wiley.com/en-us/Paul+Wilmott+Introduces+Quantitative+Finance-p-9780470319581)
- [Joshi, The Concepts and Practice of Mathematical Finance (Chapter 16)](https://www.cambridge.org/core/books/concepts-and-practice-of-mathematical-finance/2B5B6F1C2B3D0F8E5E7E7F8E9E9E9E9E)

---
**Status:** Advanced derivative structures | **Complements:** Monte Carlo, PDE Methods, Greeks, Volatility Surface, Multi-Asset Models
