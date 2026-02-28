# Greek Computation

## Concept Skeleton
**Definition:** Numerical and analytical methods to compute Greeks (Delta, Gamma, Vega, Theta, Rho) from option pricing models  
**Purpose:** Obtain derivatives for hedging and risk management; compare analytical (BS closed-form) vs. numerical (MC, finite difference) accuracy  
**Prerequisites:** Greeks definitions, option pricing, numerical methods, finite differences, Monte Carlo

## Comparative Framing
| Method | Analytical | Finite Difference | Pathwise Derivative | Likelihood Ratio |
|--------|-----------|-------------------|-------------------|-----------------|
| **Accuracy** | O(0) exact (if model exact) | O(ε²) central | O(1/N) MC error | O(1/N) MC error |
| **Computation** | Fast (formula) | Moderate (3 prices) | Fast per path | Moderate (weights) |
| **Model Required** | Closed-form BS | Any pricer | Differentiable payoff | Continuous distribution |
| **Discontinuous Payoffs** | N/A | Prone to noise | Breaks (jumps) | Applicable |
| **Dimension Scaling** | Excellent | Poor (curse of dimensionality) | Good | Good |

## Examples + Counterexamples
**Simple Example:**  
European call via BS: Analytical delta = N(d1); exact, O(1) time; finite difference: [V(S+ε) - V(S-ε)]/(2ε) ≈ O(ε²) error

**Practical Case:**  
Basket option via MC: Analytical unavailable; pathwise derivative tracks dPayoff/dS along paths; efficient Greek computation

**Limitation Case:**  
Digital option at strike: Analytical payoff discontinuous; finite difference delta jumps violently; pathwise derivative fails

**Trade-off Case:**  
Bermuda option via binomial: Analytical impossible; finite difference slow in high dimension; pathwise AD (automatic differentiation) efficient

## Layer Breakdown
```
Greek Computation Methods:
├─ Analytical Approach (Closed-Form):
│   ├─ Black-Scholes Greeks (direct formulas):
│   │   ├─ Delta = N(d1)
│   │   ├─ Gamma = N'(d1) / (S σ √T)
│   │   ├─ Vega = S N'(d1) √T
│   │   ├─ Theta = -S N'(d1) σ / (2√T) - r K e^{-rT} N(d2)
│   │   └─ Rho = K T e^{-rT} N(d2)
│   ├─ Advantages:
│   │   ├─ Exact (no numerical error)
│   │   ├─ Fast (O(1) evaluation)
│   │   └─ Stable (no conditioning issues)
│   └─ Limitations:
│       ├─ Only for simple payoffs (European options)
│       ├─ Assumes constant volatility (violated in practice)
│       └─ No path-dependent options
├─ Numerical Differentiation (Finite Differences):
│   ├─ Forward difference: δ_f ≈ [V(S+ε) - V(S)] / ε
│   ├─ Backward difference: δ_b ≈ [V(S) - V(S-ε)] / ε
│   ├─ Central difference: δ_c ≈ [V(S+ε) - V(S-ε)] / (2ε)
│   ├─ Error analysis:
│   │   ├─ Forward/backward: O(ε) error
│   │   ├─ Central: O(ε²) error (preferred)
│   │   ├─ Optimal ε: sqrt(machine precision) × |V| (balances discretization + rounding)
│   │   └─ Typical ε: 1e-4 to 1e-6 of spot price
│   ├─ Applications:
│   │   ├─ Any pricing model (no formula needed)
│   │   ├─ Barrier, American, path-dependent options
│   │   └─ Stochastic vol, jump diffusion
│   └─ Challenges:
│       ├─ Requires 3 price evaluations (central diff) vs. 1 (analytical)
│       ├─ Noise amplification for small ε
│       ├─ High dimensions expensive (M parameters × 3 prices each)
│       └─ Discontinuous payoffs cause noise spikes
├─ Pathwise Derivative (For Monte Carlo):
│   ├─ Method:
│   │   ├─ Price = E[e^{-rT} × f(S(T))]
│   │   ├─ Delta = E[e^{-rT} × f'(S(T)) × dS(T)/dS(0)]
│   │   ├─ dS(T)/dS(0) = e^{integral of drift rates}
│   │   └─ Compute via pathwise derivative: track sensitivity along each path
│   ├─ Advantages:
│   │   ├─ Single MC run (no extra paths needed)
│   │   ├─ O(1/N) MC error only (same as price)
│   │   ├─ Efficient for Greeks in high dimensions
│   │   └─ Scales well to portfolio Greeks (many Greeks, 1 run)
│   └─ Limitations:
│       ├─ Requires differentiable payoff (fails for digital, barriers at strike)
│       ├─ Implementation: Requires coding dPayoff/dS
│       └─ Not applicable to discontinuous payoffs
├─ Likelihood Ratio Method:
│   ├─ Concept:
│   │   ├─ Price = E[e^{-rT} × f(S(T; θ))]
│   │   ├─ Greek = E[e^{-rT} × f(S) × (d ln p(S|θ) / dθ)]
│   │   ├─ Reweight paths by likelihood gradient
│   │   └─ Also called "score function" method
│   ├─ Advantage:
│   │   ├─ Works for discontinuous payoffs
│   │   ├─ Applicable to barriers, digitals, Asian discontinuities
│   │   └─ Single MC run
│   └─ Disadvantage:
│       ├─ High variance (likelihood ratio can be large)
│       ├─ Requires careful tuning
│       └─ Often combined with variance reduction
├─ Automatic Differentiation (AD):
│   ├─ Concept:
│   │   ├─ Compute derivatives via chain rule on computation graph
│   │   ├─ Forward-mode: Track df/dθ through operations
│   │   ├─ Reverse-mode (backprop): Compute all Greeks in 1 pass
│   │   └─ Exact derivatives (to machine precision)
│   ├─ Advantages:
│   │   ├─ Exact (no numerical error from discretization)
│   │   ├─ All Greeks in single evaluation
│   │   ├─ Efficient scaling in high dimensions
│   │   └─ Works for any differentiable computation
│   └─ Implementation: JAX, PyTorch, TensorFlow autodiff
└─ Comparison Summary:
    ├─ Analytical: Fast, exact, limited applicability
    ├─ Finite Diff: Flexible, slow, discretization error
    ├─ Pathwise: Efficient MC, requires smooth payoff
    ├─ Likelihood: MC, works for discontinuous
    └─ AD: Exact, efficient, modern (code-heavy)
```

**Interaction:** Choose computation method → balance speed, accuracy, applicability → compute portfolio Greeks → rebalance hedges

## Challenge Round
When does each method perform poorly?
- **Analytical**: Model assumptions violated (constant vol, no jumps); exotic options no closed form
- **Finite Difference**: Noisy payoffs (barriers, digitals); high dimensions (curse of dimensionality)
- **Pathwise**: Discontinuous payoffs (digital options); barriers at-the-money (no smooth boundary)
- **Likelihood Ratio**: High variance; requires careful tuning; slow convergence vs. pathwise
- **AD**: Implementation complexity; requires differentiable computation graph

## Key References
- [Glasserman - Monte Carlo Methods (Chapters 7-8)](https://www.springer.com/gp/book/9780387004519)
- [Broadie & Glasserman - Estimating Security Prices (1996)](https://www.jstor.org/stable/1088739)
- [Jäckel - Monte Carlo Methods (Chapter 13)](https://www.wiley.com/en-us/Monte+Carlo+Methods+in+Finance-p-9780471497417)

---
**Status:** Operational Greek computation | **Complements:** Delta, Gamma, Vega, Greek Portfolio Management
