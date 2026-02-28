# Binomial Tree Model

## Concept Skeleton
**Definition:** Discrete-time option pricing model representing stock price evolution as recombining tree with up/down moves; enables valuation via backward induction and risk-neutral probability  
**Purpose:** Price American and exotic options where closed-forms don't exist; intuitive framework for understanding option dynamics; converges to Black-Scholes as steps increase  
**Prerequisites:** Option pricing basics, risk-neutral valuation, no-arbitrage, discrete probability, dynamic programming

## Comparative Framing
| Feature | Binomial Tree | Black-Scholes | Trinomial Tree | Monte Carlo |
|---------|---------------|---------------|----------------|-------------|
| **Time** | Discrete | Continuous | Discrete | Discrete paths |
| **Complexity** | O(n²) | O(1) | O(n²) | O(m) paths |
| **American Options** | Natural | Not applicable | Natural | Approximate |
| **Convergence** | To BS as n→∞ | Exact limit | Faster than binomial | Slow for early exercise |
| **Intuition** | High (visual tree) | Medium | Medium | Low |
| **Dividends** | Easy to add | Extension needed | Easy | Easy |
| **Path-Dependent** | Limited | N/A | Limited | Excellent |
| **Calibration** | Simple parameters | Single volatility | More parameters | Flexible |

## Examples + Counterexamples
**Simple Example:**  
Stock $100, 2 periods, up factor u=1.1, down factor d=0.9. At expiry: $121, $99, $81. Risk-neutral probability calculated from no-arbitrage.

**Perfect Fit:**  
American put option: Can exercise early if deeply in-the-money. Binomial tree naturally handles early exercise decision at each node via backward induction.

**Convergence:**  
Single-period tree: rough approximation. 100 periods: converges closely to Black-Scholes. 1000 periods: essentially identical to BS but slower.

**Exotic Options:**  
Barrier option (knock-out at $110): Easy to implement—just zero value at nodes ≥$110. Path-dependent payoffs naturally captured.

**Poor Fit:**  
Very long-dated options with many paths: Monte Carlo often faster. Highly path-dependent (Asian average): tree becomes non-recombining → exponential explosion.

## Layer Breakdown
```
Binomial Tree Framework:

├─ Tree Construction:
│  ├─ Time discretization: Divide T into n steps, Δt = T/n
│  ├─ Stock price nodes: S_{i,j} = S₀ × u^j × d^(i-j)
│  │   where i = time step (0 to n), j = number of up moves
│  ├─ Up factor: u = e^(σ√Δt)
│  ├─ Down factor: d = 1/u = e^(-σ√Δt)
│  ├─ Recombining property: u × d = 1 (nodes merge)
│  │   → Total nodes = (n+1)(n+2)/2 instead of 2^n
│  └─ Alternative parameterizations:
│      ├─ Cox-Ross-Rubinstein (CRR): u = e^(σ√Δt), d = 1/u
│      ├─ Jarrow-Rudd (JR): Matches first 2 moments
│      └─ Leisen-Reimer (LR): Improves convergence to BS
├─ Risk-Neutral Probability:
│  ├─ Derivation from no-arbitrage:
│  │   ├─ Portfolio: Δ shares + option replicates risk-free
│  │   ├─ Expected return under Q = risk-free rate r
│  │   └─ Solve: p×u + (1-p)×d = e^(rΔt)
│  ├─ Formula: p = (e^(rΔt) - d) / (u - d)
│  ├─ Properties:
│  │   ├─ 0 < p < 1 (valid probability)
│  │   ├─ Ensures no arbitrage
│  │   ├─ Not actual probability (risk-adjusted)
│  │   └─ Same p used at all nodes (time-homogeneous)
│  └─ With dividends (yield q):
│      p = (e^((r-q)Δt) - d) / (u - d)
├─ Option Valuation (Backward Induction):
│  ├─ Terminal payoff at expiry (step n):
│  │   ├─ Call: max(S_n - K, 0)
│  │   ├─ Put: max(K - S_n, 0)
│  │   └─ Exotic: Custom payoff
│  ├─ Recursive valuation (step i to i-1):
│  │   V_{i-1,j} = e^(-rΔt) × [p×V_{i,j+1} + (1-p)×V_{i,j}]
│  ├─ Discounting: Apply e^(-rΔt) each step
│  ├─ European: Only backward induction
│  ├─ American: At each node:
│  │   V_{i,j} = max(Intrinsic, Continuation)
│  │   = max(Payoff if exercise, Discounted expected value)
│  └─ Early exercise optimal when:
│      Intrinsic > Continuation (e.g., deep ITM American put)
├─ Greeks Calculation:
│  ├─ Delta: ΔV/ΔS from first two nodes
│  │   Δ = (V_{1,1} - V_{1,0}) / (S_{1,1} - S_{1,0})
│  ├─ Gamma: ΔΔ/ΔS from first three nodes
│  │   Use finite differences on delta
│  ├─ Theta: (V_{1,0} - V_{0,0}) / Δt
│  │   Time decay from root to next step
│  ├─ Vega: Re-run with σ±ε, measure ΔV
│  ├─ Rho: Re-run with r±ε, measure ΔV
│  └─ Finite-difference approximations
├─ Convergence & Accuracy:
│  ├─ Convergence to Black-Scholes:
│  │   As n→∞, binomial → continuous (BS limit)
│  ├─ Rate: Typically O(1/√n) error
│  ├─ Oscillation: Even/odd n can oscillate around true value
│  ├─ Smoothing techniques:
│  │   ├─ Average n and n+1 results
│  │   ├─ Richardson extrapolation
│  │   └─ Control variate (use BS as benchmark)
│  └─ Stability: Requires 0 < p < 1
│      Violated if Δt too large or r too different from implied drift
├─ Extensions:
│  ├─ Dividends:
│  │   ├─ Discrete: Reduce stock price by dividend at ex-date
│  │   ├─ Continuous yield q: Adjust p formula
│  │   └─ Known dollar amount: S drops by D at specific node
│  ├─ Barrier Options:
│  │   ├─ Knock-out: Set value=0 if barrier hit
│  │   ├─ Knock-in: Activate only if barrier hit
│  │   └─ Check at each node or path
│  ├─ Path-Dependent:
│  │   ├─ Lookback: Track max/min along paths (non-recombining)
│  │   ├─ Asian: Track average (requires state expansion)
│  │   └─ Computational explosion without approximation
│  └─ Multi-Asset: Tensor product of trees (d dimensions → n^d nodes)
└─ Practical Considerations:
   ├─ Number of steps: n=50-500 typical (tradeoff speed/accuracy)
   ├─ Step alignment: Match tree nodes to dividend dates, barriers
   ├─ Numerical stability: Check p bounds, condition number
   ├─ Memory: O(n²) storage if store full tree, O(n) if only current layer
   └─ Vectorization: Batch operations for speed
```

**Interaction:** Up/down moves encode volatility; risk-neutral probability encodes drift adjustment; backward induction implements dynamic programming for optimal exercise.

## Challenge Round
1. **Trinomial Tree:** Implement 3-branch tree (up, middle, down). Compare convergence speed to binomial. Which is more stable?

2. **Barrier Option:** Implement knock-out call (barrier=$110). How does step size affect accuracy near barrier?

3. **Dividend Impact:** Add discrete dividend of $5 at t=0.5. Compare American call value with/without dividend. When is early exercise optimal?

4. **Greeks Stability:** Calculate delta, gamma for different n. Plot stability/convergence. Which Greek converges fastest?

5. **Control Variate:** Use Black-Scholes as control for variance reduction in American option pricing. How much improvement?

## Key References
- [Cox, Ross, Rubinstein (1979) - Binomial Option Pricing](https://www.jstor.org/stable/2327557)
- [Hull, Options, Futures, and Other Derivatives (Chapter 13)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000006649)
- [Leisen & Reimer (1996) - Improved Binomial](https://www.sciencedirect.com/science/article/abs/pii/0165188995008846)
- [Shreve, Stochastic Calculus for Finance I (Chapter 1)](https://www.springer.com/series/3401)

---
**Status:** Practical tree-based pricing | **Complements:** Black-Scholes Model, American Options, Numerical Methods, Monte Carlo
