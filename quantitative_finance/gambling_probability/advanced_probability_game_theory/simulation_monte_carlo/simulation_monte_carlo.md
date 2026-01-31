# Simulation & Monte Carlo

## 1. Concept Skeleton
**Definition:** Use random sampling to approximate game probabilities and EV  
**Purpose:** Estimate complex odds (poker equity, blackjack EV) when formulas are hard  
**Prerequisites:** Random sampling, law of large numbers, variance

## 2. Comparative Framing
| Method | Accuracy | Speed | Use Case |
|---|---|---|---|
| Exact enumeration | High | Slow | Small state spaces |
| Monte Carlo | Medium-high | Fast | Large state spaces |
| Quasi-MC | Higher | Medium | Smooth payoffs |

## 3. Examples + Counterexamples
**Simple Example:** Simulate roulette spins to approximate house edge.  
**Failure Case:** Too few samples yields noisy estimates.  
**Edge Case:** Rare-event jackpots need variance-reduction techniques.

## 4. Layer Breakdown
```
Monte Carlo Workflow:
├─ Define model and payoff
├─ Randomly sample outcomes
├─ Aggregate outcomes to estimate EV
├─ Compute confidence intervals
└─ Increase samples until stable
```

## 5. Mini-Project
Estimate roulette EV by simulation:
```python
import numpy as np

spins = 100000
payout = 35
win_prob = 1/37  # single number European roulette
wins = np.random.binomial(spins, win_prob)

ev = (wins * payout - (spins - wins)) / spins
print("Estimated EV per $1 bet:", round(ev, 4))
```

## 6. Challenge Round
- RNG bias can skew results.  
- Convergence can be slow for heavy-tailed games.  
- Model errors dominate sampling errors.

## 7. Key References
- [Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)
- [Simulation in Gambling](https://www.wizardofodds.com/gambling/)
