# Discrete vs Continuous Variables in Gambling

## 1. Concept Skeleton
**Definition:** Discrete outcomes are countable (dice, cards); continuous outcomes are measurable (time, bankroll growth)  
**Purpose:** Choose correct models for probabilities and risk  
**Prerequisites:** Random variables, distributions

## 2. Comparative Framing
| Type | Example | Distribution | Use |
|---|---|---|---|
| Discrete | Dice outcomes | Binomial | Win counts |
| Continuous | Time between wins | Exponential | Waiting time |
| Mixed | Payout size | Mixture | Jackpot modeling |

## 3. Examples + Counterexamples
**Simple Example:** Roulette outcome is discrete (0–36).  
**Failure Case:** Modeling discrete wins with a continuous normal at small n.  
**Edge Case:** Slot jackpots mix discrete wins with continuous bankroll effects.

## 4. Layer Breakdown
```
Variable Selection:
├─ Identify outcome type
├─ Match to distribution family
├─ Estimate parameters
├─ Validate with data
└─ Use in EV/variance calculations
```

## 5. Mini-Project
Simulate discrete dice wins and continuous waiting times:
```python
import numpy as np

wins = np.random.binomial(100, 1/6, size=1000)
waiting = np.random.exponential(scale=6, size=1000)

print("Avg dice wins:", wins.mean())
print("Avg waiting time:", round(waiting.mean(), 2))
```

## 6. Challenge Round
- Mixed distributions need careful modeling.  
- Continuous approximations can understate tail risk.  
- Small samples distort distribution choice.

## 7. Key References
- [Discrete and Continuous Variables](https://en.wikipedia.org/wiki/Random_variable)
- [Exponential Distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
- [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
