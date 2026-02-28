# Stratified Sampling

## Concept Skeleton
**Definition:** Variance reduction technique dividing sample space into non-overlapping strata and sampling uniformly from each stratum  
**Purpose:** Ensure coverage of entire domain; reduce variance ~30-50% by eliminating clustering gaps  
**Prerequisites:** Probability distributions, partition concepts, uniform sampling

## Comparative Framing
| Aspect | Stratified | Latin Hypercube | Antithetic | Importance |
|--------|-----------|-----------------|-----------|-----------|
| **Variance Reduction** | ~30-50% | ~40-60% | ~50% | Problem-dependent |
| **Dimension Scaling** | Linear; manageable to 10D | Excellent; handles 50+D | Excellent; applies to any | Poor; curse of dimensionality |
| **Computational Cost** | Low | Moderate | Negligible | High |
| **Best For** | Uniform distributions | Latin hypercube sampling | 1D or 2D | Rare events, tails |

## Examples + Counterexamples
**Simple Example:**  
Uniform integral over [0,1]: Divide into 10 strata [0, 0.1], [0.1, 0.2], ... [0.9, 1.0]; sample 1 point uniformly in each; variance reduction ~30%

**Success Case:**  
Portfolio value distribution: Return space partitioned into quintiles; stratified ensures each percentile represented → distribution estimates accurate

**Limitation Case:**  
1D vs 2D: Stratified reduces variance by ~30% for 1D; but same strata in 2D only covers 1/100 of domain per stratum; inefficient

**Failure Case:**  
Non-uniform payoff density: Most value in one region; equal strata allocation wastes samples in low-value regions → importance sampling better

## Layer Breakdown
```
Stratified Sampling Process:
├─ Stratum Definition:
│   ├─ Partition sample space Ω into k non-overlapping strata: Ω = ⋃Sᵢ
│   ├─ Probability mass: pᵢ = P(X ∈ Sᵢ)
│   └─ Examples:
│   │   ├─ Uniform [0,1]: k equal intervals
│   │   ├─ Normal: quantile-based strata (e.g., by 0.1 increments)
│   │   └─ Portfolio returns: percentile-based bins
├─ Allocation:
│   ├─ Proportional: nᵢ = N × pᵢ (matches population)
│   ├─ Optimal (Neyman): nᵢ = N × (pᵢ σᵢ) / Σ(pⱼ σⱼ) (allocate to high-variance strata)
│   └─ Goal: N = Σnᵢ total paths
├─ Sampling Within Strata:
│   ├─ For each stratum i:
│   │   ├─ Generate nᵢ uniform samples uᵢⱼ ~ U(0,1)
│   │   ├─ Transform to stratum: Xᵢⱼ = Fᵢ⁻¹(pᵢ⁻¹ × uᵢⱼ + pᵢ⁻¹_start)
│   │   └─ Evaluate payoff: Payoffᵢⱼ = f(Xᵢⱼ)
├─ Aggregation:
│   ├─ Stratum average: μ̂ᵢ = (1/nᵢ) ΣPayoffᵢⱼ
│   ├─ Stratum estimate: Eᵢ = pᵢ × μ̂ᵢ
│   ├─ Overall estimate: E[f] = ΣEᵢ
│   └─ Variance: Var_stratified = Σ pᵢ²(σᵢ²/nᵢ)
└─ Variance Reduction:
    ├─ Standard MC: σ² / N
    ├─ Stratified: Σ pᵢ(σᵢ²/N) = (Σ pᵢσᵢ²) / N
    ├─ Reduction: 1 - (Σ pᵢσᵢ²) / σ² ≈ 30-50% if σᵢ ≈ σ uniform
    └─ Optimal: Reduction up to (Σ pᵢσᵢ)² / σ² if Neyman allocation
```

**Interaction:** Partition domain → ensure coverage → allocate samples → aggregate → guaranteed variance reduction

## Challenge Round
When is stratified sampling suboptimal?
- High dimensions (d >> 10): Strata exponentially increase; infeasible k^d strata
- Non-uniform payoff: Some strata contribute more; better to use importance sampling
- Path-dependent payoffs: Stratification of endpoints doesn't reflect intermediate path variation
- Computational overhead: Setting up strata boundaries, allocating samples adds cost
- Thin-tailed distributions: All strata equally important; no advantage over antithetic

## Key References
- [Wikipedia - Stratified Sampling](https://en.wikipedia.org/wiki/Stratified_sampling)
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)
- [Scikit-Learn - Stratified K-Fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

---
**Status:** Fundamental variance reduction | **Complements:** Latin Hypercube Sampling, Antithetic Variates, Quasi-Random Numbers
