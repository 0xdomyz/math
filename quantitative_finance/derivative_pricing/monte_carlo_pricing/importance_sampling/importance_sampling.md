# Importance Sampling

## Concept Skeleton
**Definition:** Variance reduction technique sampling from alternative distribution q(x) and reweighting samples to estimate expectations under original distribution p(x)  
**Purpose:** Shift sampling focus to high-contribution regions (rare events, tails); critical for small probability estimation  
**Prerequisites:** Probability distributions, likelihood ratios, Bayes' theorem, change of measure

## Comparative Framing
| Technique | Variance Reduction | Requires Analytical Form | Bias Possible | Computation |
|-----------|-------------------|--------------------------|--------------|-------------|
| **Importance Sampling** | Problem-specific; ~100x for rare events | No, only p/q ratio | Yes, if q ≈ 0 | Reweighting |
| **Antithetic Variates** | ~50% always | No | No | Pair negation |
| **Stratified Sampling** | ~30-50% uniform | No | No | Stratify domain |
| **Standard MC** | Baseline 1.0 | N/A | No | None |

## Examples + Counterexamples
**Simple Example:**  
Out-of-the-money (OTM) call: Strike K >> S₀; rare ITM scenarios; importance sample overweight OTM paths → 50-100x variance reduction

**Extreme Case:**  
VaR at 99.9% confidence: Only ~1 in 1,000 paths matter; standard MC wastes 999 paths; importance sampling concentrates sampling in tail → 1000x efficiency gain

**Failure Case:**  
Mis-specified importance density q << p in support region: Sample outside relevant domain → likelihood ratio explodes → high variance, biased estimates (Kish effective sample size collapse)

**Tricky Case:**  
Path-dependent knock-in barrier: Importance shift high prices early, but most value from delayed crossings → suboptimal q; requires careful tuning

## Layer Breakdown
```
Importance Sampling Process:
├─ Problem Setup:
│   ├─ Original distribution: p(x)
│   ├─ Expectation of interest: E_p[f(X)] = ∫ f(x)p(x) dx
│   └─ Goal: Concentrate sampling in high-payoff regions
├─ Choose Importance Density q(x):
│   ├─ Properties:
│   │   ├─ Support: q(x) > 0 wherever p(x)f(x) ≠ 0
│   │   ├─ Easy to sample: Tractable q ~ Normal, Exponential
│   │   └─ Covers tails: Broader than p to capture events
│   ├─ Shift parameter tuning:
│   │   ├─ Optimal q*: Minimizes Var[f(X) w(X)]
│   │   └─ Optimal: q*(x) ∝ |f(x)|p(x)
│   └─ Practical choices: Nudge mean, increase variance, change distribution family
├─ Likelihood Ratio (Importance Weight):
│   ├─ w(x) = p(x) / q(x)
│   ├─ Adjusted estimator: (1/N) Σ f(Xᵢ) w(Xᵢ) where Xᵢ ~ q
│   └─ Unbiased: E_q[f(X) w(X)] = E_p[f(X)]
├─ Reweighting & Normalization:
│   ├─ Self-normalized IS: Σ[f(Xᵢ) w(Xᵢ)] / Σ w(Xᵢ)
│   ├─ Biased but lower variance for extreme weight ratios
│   └─ Kish ESS = (Σ wᵢ)² / (Σ wᵢ²) ≤ N; efficiency = ESS/N
├─ Variance Analysis:
│   ├─ Standard estimator var: σ²_p / N
│   ├─ IS estimator var: E_q[(f w)²] - (E_q[f w])²) / N
│   ├─ Can be worse if q misspecified (variance → ∞ if w → ∞)
│   └─ Degeneracy: Few paths dominate; effective sample size collapses
└─ Final Estimate:
    ├─ Price: e^{-rT} × (1/N) Σ f(Xᵢ) w(Xᵢ)
    └─ SE: estimate via bootstrap of reweighted samples
```

**Interaction:** Right choice of q → massive variance reduction; wrong choice → high variance, exploding weights

## Challenge Round
When does importance sampling degrade performance?
- Poor q choice: q doesn't cover p support → weights explode → high variance
- Kish ESS collapse: Few dominant paths; effective N << actual N
- Curse of dimensionality: Hard to design good q in 10+ dimensions simultaneously
- Derivative discontinuities: Barrier options; changing q doesn't help with structural jumps
- Computational cost: Reweighting, weight computation overhead vs variance savings

## Key References
- [Wikipedia - Importance Sampling](https://en.wikipedia.org/wiki/Importance_sampling)
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)
- [Owen & Zhou - Safe & Effective Importance Sampling](https://arxiv.org/abs/1906.07701)

---
**Status:** Advanced variance reduction for rare events | **Complements:** Antithetic Variates, Multilevel MC, Tail Risk Estimation
