# Bayesian Updating

## 1. Concept Skeleton
**Definition:** Update beliefs about probabilities as evidence arrives  
**Purpose:** Refine win-rate estimates, detect bias in games, adapt strategies  
**Prerequisites:** Bayes’ theorem, prior/posterior, likelihoods

## 2. Comparative Framing
| Approach | Prior | Data Need | Use Case |
|---|---|---|---|
| Frequentist | None | Large | Long-run rates |
| Bayesian | Explicit | Moderate | Early inference |
| Empirical Bayes | Data-driven | Medium | Shrinkage |

## 3. Examples + Counterexamples
**Simple Example:** Update belief that a coin is biased after 20 flips.  
**Failure Case:** Overconfident prior overwhelms real data.  
**Edge Case:** Non-stationary games violate fixed-probability assumption.

## 4. Layer Breakdown
```
Bayesian Loop:
├─ Choose prior distribution
├─ Observe outcomes
├─ Compute likelihood
├─ Update posterior
└─ Use posterior mean for decisions
```

## 5. Mini-Project
Estimate win probability with a Beta prior:
```python
from math import comb

alpha, beta = 2, 2  # prior
wins, losses = 12, 8
posterior_alpha = alpha + wins
posterior_beta = beta + losses
posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

print("Posterior win rate:", round(posterior_mean, 3))
```

## 6. Challenge Round
- Priors must be justified or sensitivity-tested.  
- Strategy shifts can invalidate earlier data.  
- Small sample posteriors still carry high uncertainty.

## 7. Key References
- [Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_inference)
- [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution)
- [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
