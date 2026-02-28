# Moment Matching

## Concept Skeleton
**Definition:** Variance reduction technique forcing simulated sample moments (mean, variance, skewness) to match theoretical moments analytically  
**Purpose:** Remove systematic bias in finite samples; reduce discrepancy between empirical and true distributions  
**Prerequisites:** Moments (mean, variance, skewness, kurtosis), empirical distributions, bootstrap

## Comparative Framing
| Technique | Moment Matching | Antithetic | Importance | Stratified |
|-----------|-----------------|-----------|-----------|-----------|
| **Variance Reduction** | ~20-40% | ~50% | Problem-specific | ~30-50% |
| **Removes Bias** | Yes | No | Yes (if correct) | No |
| **Affects Mean** | Yes | No | Yes | No |
| **Computational Cost** | Low | Negligible | High | Low |
| **Best For** | Finite N bias correction | Simple payoffs | Rare events | Coverage guarantee |

## Examples + Counterexamples
**Simple Example:**  
N=100 Monte Carlo paths of normal returns: Sample mean ≠ 0, sample variance ≠ 1; moment matching standardizes to (0,1)

**Success Case:**  
Pricing Asian options: Moment matching ensures average price matches theoretical mean → reduced bias in estimated expected payoff

**Limitation Case:**  
Skewed/kurtotic payoffs: Matching first 2 moments misses tail behavior; higher moments needed → complexity increases

**Trade-off Case:**  
Latin Hypercube + Moment Matching: LHS already covers space uniformly; MM improves further ~10%; combined effect better than separate

## Layer Breakdown
```
Moment Matching Process:
├─ Sample Generation:
│   ├─ Generate N samples X₁, ..., Xₙ ~ q (e.g., N(0,1))
│   └─ Compute empirical moments:
│       ├─ X̄ = (1/N) ΣXᵢ (empirical mean)
│       ├─ Ŝ² = (1/N) Σ(Xᵢ - X̄)² (empirical variance)
│       └─ Ĝ₃ = (1/N) Σ(Xᵢ - X̄)³/Ŝ³, Ĝ₄ = ... (skewness, kurtosis)
├─ Theoretical Moments:
│   ├─ μ = theoretical mean
│   ├─ σ² = theoretical variance
│   └─ γ₃, γ₄ = theoretical skewness, kurtosis
├─ Adjustment (Standardization):
│   ├─ Center: Yᵢ = Xᵢ - X̄ + μ
│   ├─ Scale: Yᵢ ← Yᵢ × (σ/Ŝ)
│   ├─ Alternative (more aggressive):
│   │   ├─ Iterative regression: Fit lower-order polynomial to transform X to match moments
│   │   └─ Cornish-Fisher: Match skewness & kurtosis via quantile adjustments
│   └─ Result: Adjusted sample Yᵢ has moments ≈ theoretical
├─ Path Simulation:
│   ├─ Use adjusted samples Yᵢ as random shocks
│   ├─ S(T; Yᵢ) = GBM paths with corrected Gaussian innovations
│   └─ Payoff_i = f(S(T; Yᵢ))
├─ Aggregation:
│   ├─ Price = (1/N) Σ e^{-rT} × Payoff_i
│   └─ Variance: Reduced by eliminating sampling error in moments
└─ Benefits:
    ├─ Empirical distribution matches theoretical
    ├─ Reduced bias for finite N
    ├─ Better tail behavior (if higher moments matched)
    └─ Often combined with antithetic/stratified for stacking effects
```

**Interaction:** Generate paths → compute moments → adjust to match theory → reduced finite-sample bias

## Challenge Round
When does moment matching underperform?
- Higher moments unknown: Can't match skewness/kurtosis without estimation; adds complexity
- Iterative methods required: Scaling/centering is order-dependent; convergence not guaranteed
- Multivariate: Matching joint moments in high dimensions exponentially complex
- Non-monotone payoffs: Adjusting shocks changes path structure; may not preserve correlations
- Computational cost vs. benefit: ~10-20% improvement often modest; antithetic/stratified simpler

## Key References
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)
- [Acerbi - Moment Matching Methods](https://ssrn.com/abstract=1107960)
- [Cornish & Fisher - Moments and Cumulants](https://www.jstor.org/stable/2332539)

---
**Status:** Bias reduction technique | **Complements:** Antithetic Variates, Latin Hypercube, Stratified Sampling
