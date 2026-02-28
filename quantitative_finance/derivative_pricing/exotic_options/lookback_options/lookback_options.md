# Lookback Options

## Concept Skeleton
**Definition:** Path-dependent options with payoff based on maximum or minimum asset price observed over option's life  
**Purpose:** Hindsight perfection; capture best price; speculation on extreme moves; no regret trading  
**Prerequisites:** Path-dependent payoffs, extreme value statistics, Monte Carlo simulation, min/max tracking

## Comparative Framing
| Feature | Fixed Strike Lookback | Floating Strike Lookback | Asian Option | European Vanilla |
|---------|----------------------|--------------------------|--------------|------------------|
| **Payoff** | max(M - K, 0) or max(K - m, 0) | max(S_T - m, 0) or max(M - S_T, 0) | max(Avg - K, 0) | max(S_T - K, 0) |
| **Strike** | Fixed K | Floating (min or max) | Fixed K | Fixed K |
| **Value** | Very high | Highest | Moderate | Lowest |
| **Vega** | High (extreme sensitivity) | Very high | Low (averaging) | Moderate |
| **Manipulation** | Impossible (max/min) | Impossible | Hard | Easy (spot at T) |

## Examples + Counterexamples
**Simple Example:**  
Fixed strike lookback call: K=$100, Path=[95, 110, 105, 98, 108], Max=110 → Payoff = 110 - 100 = $10

**Failure Case:**  
Single observation: Lookback → European (max of one point = terminal price); loses lookback premium

**Edge Case:**  
Zero volatility: Max = Min = S₀ → Lookback call = max(S₀ - K, 0); floating strike call = 0 (S_T - m = 0)

## Layer Breakdown
```
Lookback Option Classification & Pricing:
├─ Fixed Strike Lookback:
│   ├─ Fixed Strike Call:
│   │   ├─ Payoff: max(M - K, 0) where M = max(S_t) over t ∈ [0, T]
│   │   ├─ Exercise: Optimal hindsight (buy at strike, sell at max)
│   │   ├─ Value: Higher than European (M ≥ S_T always)
│   │   └─ Use Case: Capture upside extremes without timing risk
│   ├─ Fixed Strike Put:
│   │   ├─ Payoff: max(K - m, 0) where m = min(S_t) over t ∈ [0, T]
│   │   ├─ Exercise: Hindsight sell at strike, buy at min
│   │   ├─ Value: Higher than European (m ≤ S_T always)
│   │   └─ Use Case: Capture downside protection at best price
│   └─ Properties:
│       ├─ Always ITM: M > K for call (if ever crosses), m < K for put
│       ├─ No Regret: Guarantees best price over period
│       └─ Premium: Expensive (captures tail value)
├─ Floating Strike Lookback:
│   ├─ Floating Strike Call:
│   │   ├─ Payoff: S_T - m where m = min(S_t) over t ∈ [0, T]
│   │   ├─ Strike: Dynamically set to minimum observed price
│   │   ├─ Exercise: Buy at min, sell at terminal price
│   │   └─ Value: Highest (always positive if S_T > m)
│   ├─ Floating Strike Put:
│   │   ├─ Payoff: M - S_T where M = max(S_t) over t ∈ [0, T]
│   │   ├─ Strike: Dynamically set to maximum observed price
│   │   ├─ Exercise: Sell at max, buy at terminal price
│   │   └─ Value: Highest (always positive if M > S_T)
│   └─ Properties:
│       ├─ Always Positive Payoff: No strike to overcome
│       ├─ Perfect Timing: Captures full range of movement
│       └─ Most Expensive: Maximum optionality
├─ Monte Carlo Pricing:
│   ├─ Path Generation:
│   │   ├─ Fine Time Steps: Daily or finer discretization
│   │   ├─ Euler Scheme: S_{t+1} = S_t exp((r - σ²/2)Δt + σ√Δt Z_t)
│   │   └─ Store All Prices: Need min/max over entire path
│   ├─ Extrema Tracking:
│   │   ├─ Maximum: M = max(S_0, S_1, ..., S_n)
│   │   ├─ Minimum: m = min(S_0, S_1, ..., S_n)
│   │   └─ Running Min/Max: Update at each time step
│   ├─ Payoff Calculation:
│   │   ├─ Fixed Call: max(M - K, 0)
│   │   ├─ Fixed Put: max(K - m, 0)
│   │   ├─ Floating Call: S_T - m (always ≥ 0)
│   │   └─ Floating Put: M - S_T (always ≥ 0)
│   └─ Pricing:
│       ├─ Discount: PV = e^(-rT) × Payoff
│       └─ Average: V = (1/N) Σ PV_i over N paths
├─ Closed-Form Solutions (Limited):
│   ├─ Continuous Monitoring: Available under GBM (Goldman et al.)
│   ├─ Floating Strike: Simpler formulas (no strike parameter)
│   ├─ Fixed Strike: More complex (involves strike position)
│   └─ Discrete Monitoring: No closed-form; use MC or trees
├─ Variance Reduction:
│   ├─ Antithetic Variates: Z and -Z paths give correlated M, m
│   ├─ Control Variate: Use Asian option (correlated but cheaper)
│   ├─ Stratified: Stratify on final price S_T
│   └─ Moment Matching: Force paths to have correct E[M], E[m]
├─ Greeks & Hedging:
│   ├─ Delta: Time-dependent (high early, decreases as M/m established)
│   ├─ Gamma: Positive but decreases over time
│   ├─ Vega: Very high (extreme prices sensitive to volatility)
│   ├─ Theta: Negative (time decay as monitoring period shortens)
│   └─ Hedging: Difficult (path-dependent; delta changes with M/m)
└─ Discrete vs Continuous Monitoring:
    ├─ Continuous: True lookback (all times checked)
    ├─ Discrete: Check at specific times (daily, weekly)
    ├─ Bias: Discrete < Continuous (misses intraperiod extremes)
    └─ Adjustment: Broadie-Glasserman correction for discrete
```

**Interaction:** Generate paths → Track running min/max → Compute payoff on extremes → Discount to present

## Challenge Round
**Q1:** Why is floating strike lookback always worth more than fixed strike lookback?  
**A1:** Floating strike payoff = S_T - m (call) or M - S_T (put) → always captures full range. Fixed strike = max(M - K, 0) → limited by strike; if M < K, payoff = 0. Floating has no strike barrier → higher optionality → higher value.

**Q2:** Prove lookback call > European call. What's intuition via hindsight?  
**A2:** Fixed lookback: max(M - K, 0) ≥ max(S_T - K, 0) since M ≥ S_T always. Strict inequality unless S_T = M (terminal price is maximum). Hindsight: Lookback exercises at best price; European at terminal only. No-regret premium built into price.

**Q3:** Why does lookback option have extremely high Vega (volatility sensitivity)?  
**A3:** Higher volatility → larger price swings → higher max/min spread → larger payoffs. For floating strike call: Payoff = S_T - m = range. Var(Range) increases with σ² → lookback value highly sensitive to vol. Vega_lookback >> Vega_European.

**Q4:** Discrete monitoring: How does sampling frequency affect lookback value?  
**A4:** Higher frequency (more observations) → closer to continuous → finds true extremes → higher value. Daily vs weekly: Daily captures more extremes → higher max, lower min → larger payoff → higher price. Limit: Continuous monitoring gives upper bound.

**Q5:** Delta of lookback option: How does it change over time?  
**A5:** Early: High delta (max/min not established; sensitive to S movements). Late: Lower delta (extremes already observed; locked in). If current S near established max/min, delta changes (max affects call, min affects put). Time-dependent and path-dependent delta.

**Q6:** Partial lookback: Payoff = max(M - K, 0) but M = max over [t*, T] for t* < T. How to price?  
**A6:** Lookback starts at t*; ignore prices before. MC: Generate full path, compute max only from t* onward. Cheaper than full lookback (less monitoring → smaller max). Used for cheapening premium while keeping lookback feature.

**Q7:** Lookback on portfolio: Max of weighted basket max(w₁M₁ + w₂M₂ - K, 0). Challenge?  
**A7:** Component maximums occur at different times: M₁ at t₁, M₂ at t₂. Weighted sum max(w₁S₁ + w₂S₂) ≠ w₁M₁ + w₂M₂. Must track portfolio value at each time step, find its maximum. More complex; correlation matters less than individual extremes.

**Q8:** Hedging lookback option: Why is it nearly impossible near extrema?  
**A8:** Near established max (for call): Small move up → new max → delta jumps. Small move down → max unchanged → delta near zero. Gamma infinite at max crossing point. Hedging requires continuous rebalancing with infinite frequency → impossible in practice. Transaction costs prohibitive.

## Key References
**Primary Sources:**
- Goldman, M.B., Sosin, H.B., Gatto, M.A. "Path Dependent Options" (1979) - Original lookback pricing
- [Lookback Option Wikipedia](https://en.wikipedia.org/wiki/Lookback_option) - Overview and types
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 27: Lookback Options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Lookback simulation (pp. 392-405)
- Conze, A. & Viswanathan "Path Dependent Options: The Case of Lookback Options" (1991) - Closed-form formulas

**Thinking Steps:**
1. Generate Monte Carlo paths with fine time discretization
2. Track running maximum and minimum along each path
3. Compute payoff: Fixed (max/min vs strike K) or Floating (S_T vs min/max)
4. Discount expected payoff; note always positive for floating strike
5. Higher variance than European due to extreme value dependence
6. Floating strike always more valuable (captures full range, no strike barrier)
