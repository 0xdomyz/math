# Probability Distributions in Games

## 1. Concept Skeleton
**Definition:** Common distributions (binomial, Poisson, normal, geometric) model gambling outcomes  
**Purpose:** Predict streaks, bankroll variance, and payout dispersion  
**Prerequisites:** Random variables, expectation, variance

## 2. Comparative Framing
| Distribution | Use Case | Parameter | Example |
|---|---|---|---|
| Binomial | Win/loss over n bets | p | 55% win rate over 100 bets |
| Geometric | Time to first win | p | Spins until first bonus |
| Poisson | Rare events | λ | Jackpot hits per month |
| Normal | Aggregate results | μ, σ | Session profit distribution |

## 3. Examples + Counterexamples
**Simple Example:** Binomial models number of roulette wins in 100 spins.  
**Failure Case:** Using normal for small samples misestimates tail risk.  
**Edge Case:** Heavy-tailed payouts (jackpots) need lognormal or Pareto.

## 4. Layer Breakdown
```
Distributions in Gambling:
├─ Define event and trial structure
├─ Choose distribution (discrete vs continuous)
├─ Estimate parameters from data
├─ Validate fit (QQ plots, KS test)
└─ Use for risk and EV projections
```

## 5. Mini-Project
Simulate binomial win counts and compare to normal approximation:
```python
import numpy as np

p = 0.52
n = 100
trials = 5000
wins = np.random.binomial(n, p, size=trials)

mean = wins.mean()
std = wins.std()
print("Mean wins:", round(mean, 2), "Std:", round(std, 2))
```

## 6. Challenge Round
- Wrong distribution choice underestimates variance.  
- Parameter drift breaks historical fits.  
- Rare-event tails dominate bankroll outcomes.

## 7. Key References
- [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
- [Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
