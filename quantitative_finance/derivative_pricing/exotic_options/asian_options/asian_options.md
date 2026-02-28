# Asian Options

## Concept Skeleton
**Definition:** Path-dependent options where payoff depends on average asset price over specified period  
**Purpose:** Reduce manipulation risk; smooth out volatility; hedging applications; commodity derivatives  
**Prerequisites:** Monte Carlo simulation, path generation, time discretization, variance reduction techniques

## Comparative Framing
| Feature | Arithmetic Asian | Geometric Asian | European Vanilla | Lookback |
|---------|------------------|-----------------|------------------|----------|
| **Payoff** | max(Avg_A(S) - K, 0) | max(Avg_G(S) - K, 0) | max(S_T - K, 0) | max(max(S) - K, 0) |
| **Pricing** | Monte Carlo | Closed-form available | Black-Scholes | Monte Carlo |
| **Variance** | Low (averaging effect) | Lower (geometric) | High | Highest |
| **Value** | Lower than European | Lower than arithmetic | Highest | Higher than Asian |
| **Manipulation** | Hard (avg over time) | Hard | Easy (spot at T) | Impossible |

## Examples + Counterexamples
**Simple Example:**  
Arithmetic Asian call: S = [100, 105, 110, 102, 108]; Avg = 105; K = 100 → Payoff = 5 (vs European on S_T=108 → 8)

**Failure Case:**  
Discrete monitoring (monthly) vs continuous: Large price movements between observations → average misrepresents true path

**Edge Case:**  
Single observation: Asian → European (average of one point = terminal price); control variate works perfectly

## Layer Breakdown
```
Asian Option Pricing Pipeline:
├─ Option Types:
│   ├─ Arithmetic Average Asian:
│   │   ├─ Average: A = (1/n) Σ S_i for observations at t_1, ..., t_n
│   │   ├─ Call Payoff: max(A - K, 0)
│   │   ├─ Put Payoff: max(K - A, 0)
│   │   └─ No Closed-Form: Monte Carlo required
│   ├─ Geometric Average Asian:
│   │   ├─ Average: G = (Π S_i)^(1/n) = exp((1/n) Σ ln(S_i))
│   │   ├─ Call Payoff: max(G - K, 0)
│   │   ├─ Closed-Form Available: Under GBM, G is lognormal
│   │   └─ Used as Control Variate for Arithmetic
│   ├─ Fixed Strike vs Floating Strike:
│   │   ├─ Fixed Strike: max(Average - K, 0) - K predetermined
│   │   └─ Floating Strike: max(S_T - Average, 0) - Strike = average
│   └─ Average Price vs Average Strike:
│       ├─ Average Price: Payoff based on average, strike fixed
│       └─ Average Strike: Strike = average, payoff based on terminal price
├─ Monte Carlo Simulation:
│   ├─ Path Generation (n_steps observations):
│   │   ├─ Euler Scheme: S_{i+1} = S_i exp((r - σ²/2)Δt + σ√Δt Z_i)
│   │   ├─ Store all prices: S_0, S_1, ..., S_n
│   │   └─ Monitoring: Daily, weekly, monthly (affects n)
│   ├─ Average Computation:
│   │   ├─ Arithmetic: A = (1/n) Σ S_i
│   │   ├─ Geometric: G = exp((1/n) Σ ln(S_i))
│   │   └─ Weighted: A_w = Σ w_i S_i (non-uniform weights)
│   ├─ Payoff Calculation:
│   │   ├─ Call: C = max(Average - K, 0)
│   │   ├─ Put: P = max(K - Average, 0)
│   │   └─ Discount: PV = e^(-rT) × Payoff
│   └─ Price Estimation:
│       ├─ Mean: V = (1/N) Σ PV_i over N paths
│       └─ Standard Error: SE = σ_payoff / √N
├─ Variance Reduction (Critical for Asians):
│   ├─ Control Variate (Geometric Asian):
│   │   ├─ Simulate both A and G on same paths
│   │   ├─ Known: E[Payoff_geometric] from closed-form
│   │   ├─ Correlation: ρ(Payoff_A, Payoff_G) ≈ 0.95-0.99
│   │   └─ Adjusted Estimator: V* = Payoff_A - β(Payoff_G - E[Payoff_G])
│   ├─ Antithetic Variates:
│   │   ├─ Use Z and -Z for path generation
│   │   └─ Correlation: Cov(A(Z), A(-Z)) < 0
│   ├─ Stratified Sampling:
│   │   ├─ Partition [0, T] into equal intervals
│   │   └─ Uniform coverage of averaging period
│   └─ Moment Matching:
│       ├─ Force sample mean of S_i to match E[S_i] = S_0 e^(r t_i)
│       └─ Reduces path-to-path variance
├─ Closed-Form Geometric Asian:
│   ├─ Under GBM: ln(G) ~ Normal distribution
│   ├─ Adjusted Parameters:
│   │   ├─ σ_G = σ / √3 (variance reduces due to averaging)
│   │   ├─ r_G = (r - σ²/2) / 2 + σ²/6 (drift adjustment)
│   │   └─ μ_G = ln(S_0) + r_G T
│   └─ Price: Use Black-Scholes formula with (S_0, σ_G, r_G)
└─ Properties:
    ├─ Value: Asian < European (averaging reduces volatility → lower optionality)
    ├─ Vega: Lower than European (less sensitive to volatility changes)
    ├─ Theta: More gradual decay (averaging smooths time effect)
    └─ Delta: Time-dependent (early: high; late: low as average locked in)
```

**Interaction:** Generate paths with many time steps → Compute average (arithmetic/geometric) → Payoff on average → Discount to present

## Challenge Round
**Q1:** Why is arithmetic Asian always worth more than geometric Asian? Prove using Jensen's inequality.  
**A1:** Arithmetic average A = (1/n)Σ S_i; Geometric G = (Π S_i)^(1/n). Since exp(x) is convex: A = (1/n)Σ S_i > exp((1/n)Σ ln(S_i)) = G by Jensen. For call payoff max(Avg - K, 0), higher average → higher value. Geometric Asian is lower bound.

**Q2:** Control variate: Explain why ρ(Payoff_arith, Payoff_geo) ≈ 0.99 for Asian options.  
**A2:** Both payoffs depend on same underlying paths; differ only in averaging method (arithmetic vs geometric). Correlation high because both increase/decrease together with S. Arithmetic slightly higher but highly correlated → β ≈ 1 → variance reduction ≈ (1 - ρ²) ≈ 98%.

**Q3:** Discrete vs continuous monitoring: How does monitoring frequency affect Asian option value?  
**A3:** More frequent monitoring (higher n) → average converges to continuous case → slightly lower value (less variance in average). Difference typically < 1% for daily vs continuous. Continuous Asian has closed-form for geometric; arithmetic requires Monte Carlo even for continuous.

**Q4:** Fixed strike max(A - K, 0) vs floating strike max(S_T - A, 0): Which is more valuable?  
**A4:** Depends on relationship between S_T and A. Fixed strike: Benefits from high average. Floating strike: Benefits from S_T >> A (strong finish). Under GBM, both have different Greeks; floating strike has path-dependent strike → more complex hedging.

**Q5:** Why does Asian option have lower Vega than European option?  
**A5:** Vega = ∂V/∂σ. Averaging reduces effective volatility: Var(Average) < Var(S_T). Lower volatility exposure → less sensitivity to σ changes. Quantitatively: Vega_Asian ≈ Vega_European / √n for n observations.

**Q6:** Implement weighted Asian option where recent prices have higher weight. How to modify MC code?  
**A6:** Replace uniform weights (1/n) with exponential decay w_i = e^(-λ(T - t_i)) / Σ e^(-λ(T - t_j)). Higher λ → more weight on recent prices. In code: `weighted_avg = np.sum(weights * prices, axis=1)` where weights sum to 1. Mimics recency bias in certain markets.

**Q7:** Asian basket option: Payoff = max(Avg(Basket) - K, 0) for weighted basket. How does correlation affect value?  
**A7:** Higher correlation → basket behaves like single asset → higher volatility → higher Asian value. Lower correlation → diversification reduces variance → lower value. Control variate: Geometric average of basket (multi-dimensional). Cholesky decomposition for correlated paths.

**Q8:** Prove Asian option value decreases as averaging period progresses (for fixed strike, in-the-money).  
**A8:** As time passes, more observations accumulated → average "locks in" → less uncertainty in final average → option becomes more like fixed payoff → theta negative but volatility sensitivity decreases. Delta changes: Early (high delta, like European); Late (low delta, average mostly determined).

## Key References
**Primary Sources:**
- Kemna, A. & Vorst, T. "A Pricing Method for Options Based on Average Asset Values" (1990) - Geometric Asian closed-form
- [Asian Option Wikipedia](https://en.wikipedia.org/wiki/Asian_option) - Comprehensive overview
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 27: Exotic Options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - Asian options (pp. 327-365)
- Curran, M. "Valuing Asian and Portfolio Options by Conditioning on the Geometric Mean Price" (1994) - Control variate method

**Thinking Steps:**
1. Generate Monte Carlo paths with n_steps time discretization
2. Compute average: Arithmetic (mean) or Geometric (exp of mean log)
3. Calculate payoff based on average: max(Avg - K, 0) for call
4. Use geometric Asian as control variate (closed-form available)
5. Optimal β = Cov(Arith, Geo) / Var(Geo) for variance reduction
6. Discount expected payoff to present value
