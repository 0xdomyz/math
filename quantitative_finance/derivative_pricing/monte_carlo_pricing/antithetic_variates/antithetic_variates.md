# Antithetic Variates

## Concept Skeleton
**Definition:** Variance reduction technique pairing random samples with their negatives to create negatively correlated payoffs, halving estimator variance  
**Purpose:** Reduce standard error without increasing number of simulations; improve convergence rate from O(1/√N) baseline  
**Prerequisites:** Monte Carlo fundamentals, random number generation, variance concepts, correlation

## Comparative Framing
| Technique | Variance Reduction | Correlation | Implementation | Overhead |
|-----------|-------------------|-------------|-----------------|-----------|
| **Antithetic Variates** | ~50% | Negative (ρ ≈ -1) | Trivial (use Z, -Z) | Negligible |
| **Control Variates** | Depends on ρ | Must be chosen | Moderate | Compute control |
| **Importance Sampling** | Problem-dependent | N/A | Complex | Reweight entire sample |
| **No Reduction** | Baseline 1.0 | N/A | Simplest | None |

## Examples + Counterexamples
**Simple Example:**  
European call option pricing: N=10,000 paths with antithetic sampling yields SE ≈ 0.5 × (SE without technique)

**Success Case:**  
Asian option (payoff average of prices): Payoff monotone in S → antithetic pairs negatively correlated → strong variance reduction (~60%)

**Failure Case:**  
Barrier option knocked out early: If barrier hit, payoff = 0 regardless of Z or -Z → no correlation, no benefit

**Counterintuitive Case:**  
Non-monotone payoff (butterfly spread): Max payoff interior; high-S and low-S paths both hit limits → weaker pairing benefit (~20% reduction)

## Layer Breakdown
```
Antithetic Variates Process:
├─ Sample Generation:
│   ├─ Generate N/2 uniform random vectors U ~ U(0,1)ᵈ
│   ├─ Create pairs: (U, 1-U)
│   └─ Transform to normals: Z = Φ⁻¹(U), Z' = -Z
├─ Path Simulation (for each pair):
│   ├─ Path 1: S(T; Z)
│   ├─ Path 2: S(T; -Z) [symmetric under negation]
│   ├─ Payoff 1: f(S(T; Z))
│   └─ Payoff 2: f(S(T; -Z))
├─ Averaging:
│   ├─ Pair Average: [f(S(T; Z)) + f(S(T; -Z))] / 2
│   ├─ Price Estimate: (1/N) Σ[pairs averaged]
│   └─ Expected Result: E[Payoff(Z)] ≈ E[Payoff(-Z)]
└─ Variance Reduction:
    ├─ Standard Estimator Var: σ²_f / (N/2)
    ├─ Pair Correlations: ρ(f(Z), f(-Z)) << 0
    ├─ Variance of Average: (σ²_f / (N/2)) × (1 + ρ) ≈ σ²_f / N × (1 - 1)
    └─ Optimal Benefit: Var_antithetic ≈ 0.5 × Var_standard
```

**Interaction:** Negative correlation between paired paths → lower sample variance → tighter CI for same N

## Challenge Round
When does antithetic variates underperform?
- Monotone payoff functions: High correlation → strong reduction (~50%)
- Non-monotone payoffs (e.g., straddles): Symmetric up/down moves hit different regions → weaker reduction (~10-20%)
- Discontinuous payoffs (barriers): Early knock-out breaks symmetry → payoff(Z) ≠ -payoff(-Z)
- Multi-dimensional: Negating all Z dimensions may not preserve correlation structure; works best for 1D
- Correlated underlyings: Basket options need coordinated negation; requires careful construction

## Key References
- [Wikipedia - Antithetic Variates](https://en.wikipedia.org/wiki/Antithetic_variates)
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)
- [Numerical Recipes - Antithetic Sampling](http://numerical.recipes)

---
**Status:** Fundamental variance reduction | **Complements:** Control Variates, Importance Sampling, Convergence Analysis
