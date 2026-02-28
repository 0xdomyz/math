# Control Variates

## Concept Skeleton
**Definition:** Variance reduction technique using a correlated random variable with known expectation to reduce estimator variance  
**Purpose:** Minimize estimator variance by subtracting off correlated "control" that explains variation; effective when control ρ > 0.5  
**Prerequisites:** Correlation concepts, covariance, linear regression, Monte Carlo basics

## Comparative Framing
| Aspect | Control Variates | Antithetic Variates | Importance Sampling |
|--------|-----------------|-------------------|-------------------|
| **Variance Reduction** | ρ²-dependent; ~80% if ρ ≈ 1 | ~50% always | Problem-specific |
| **Requires Known E[Control]** | Yes, essential | No | No, but alters distribution |
| **Coefficient Optimization** | Regression; β̂ = Cov/Var | Fixed (-1 pair) | Not applicable |
| **Complexity** | Moderate | Trivial | High |
| **Best Use Case** | When close proxy exists | Monotone payoffs | Rare events, discontinuities |

## Examples + Counterexamples
**Simple Example:**  
Option pricing: Use European option price (closed-form) as control for Asian option → strong correlation → ~70% variance reduction

**Strong Case:**  
Geometric Asian option: Control = Arithmetic Asian (easy to price); controls 80% of variance in true price  

**Weak Case:**  
Lookback option with European control: Payoff dependence weak (lookback max is unrelated to European final price) → ρ ≈ 0.2 → minimal benefit

**Failure Case:**  
Using arithmetic average as control for knockout barrier option: If barrier hit, both knockouts together → control provides no variance reduction

## Layer Breakdown
```
Control Variates Process:
├─ Control Selection:
│   ├─ Choose control Y with E[Y] = μ_Y known
│   ├─ Y highly correlated with payoff X (ρ >> 0)
│   └─ Example: Simpler analytical proxy or closed-form benchmark
├─ Coefficient Optimization:
│   ├─ Simulate N paths:
│   │   ├─ Payoff X_i, Control Y_i for each path
│   │   └─ Covariance matrix: Cov(X, Y), Var(Y)
│   ├─ Optimal coefficient: β* = Cov(X,Y) / Var(Y)
│   ├─ Regression: β* ≈ Cov(X̂,Ŷ) / Var(Ŷ)
│   └─ Alternative: Use predetermined β = 1 for simplicity
├─ Adjusted Estimator:
│   ├─ Standard payoff: X̄ = (1/N) ΣXᵢ
│   ├─ Control adjustment: (1/N) Σ(Xᵢ - β(Yᵢ - μ_Y))
│   ├─ Expanded: (1/N) ΣXᵢ - β((1/N) ΣYᵢ - μ_Y)
│   └─ Simplified: X̄ - β(Ȳ - μ_Y)
├─ Variance Reduction:
│   ├─ Original Var(X̄) = σ²_X / N
│   ├─ Controlled Var = (σ²_X(1 - ρ²)) / N
│   ├─ Reduction Factor = (1 - ρ²)
│   └─ Example: ρ = 0.9 → ~19% variance; ρ = 0.99 → ~2% variance
└─ Final Price Estimate:
    ├─ Discounted: e^{-rT} × [X̄ - β(Ȳ - μ_Y)]
    └─ SE: e^{-rT} × σ_controlled / √N
```

**Interaction:** Strong correlation Y→X + known E[Y] + optimized β → most effective variance reduction available

## Challenge Round
When are controls ineffective?
- Low correlation (ρ < 0.5): Control adds noise; use simpler methods
- Control expectation unknown: Can't apply correction term properly
- High-dimensional problems: Harder to find correlated controls in 10+ dimensions
- Discontinuous payoffs: Lookup barrier versus European control unrelated → ρ ≈ 0
- Multiple controls needed: Requires multivariate regression; can introduce overfitting

## Key References
- [Wikipedia - Control Variates](https://en.wikipedia.org/wiki/Variance_reduction)
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)
- [Broadie & Glasserman - Estimating Security Prices](https://www.jstor.org/stable/1088739)

---
**Status:** Advanced variance reduction | **Complements:** Antithetic Variates, Importance Sampling, Regression Analysis
