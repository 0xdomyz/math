# Statistics Applied to Gambling

## 1. Concept Skeleton
**Definition:** Statistical tools quantify edge, variance, and confidence in gambling results  
**Purpose:** Distinguish skill from luck; validate profitability  
**Prerequisites:** Estimation, confidence intervals, hypothesis testing

## 2. Comparative Framing
| Tool | Question Answered | Output | Use Case |
|---|---|---|---|
| Confidence interval | Is win-rate above 50%? | Range | Edge validation |
| Hypothesis test | Is edge real? | p-value | Skill confirmation |
| Regression | What predicts wins? | Coefficients | Strategy tuning |

## 3. Examples + Counterexamples
**Simple Example:** 60 wins in 100 bets yields a 95% CI around 50–70%.  
**Failure Case:** Declaring skill after 10 wins in a row (small sample).  
**Edge Case:** Selection bias inflates apparent win rates.

## 4. Layer Breakdown
```
Statistical Workflow:
├─ Collect unbiased results
├─ Estimate win-rate and variance
├─ Compute confidence intervals
├─ Test null (no edge)
└─ Decide to scale or stop
```

## 5. Mini-Project
Compute a Wilson confidence interval for win rate:
```python
import math

wins, n = 60, 100
z = 1.96
phat = wins / n
center = (phat + z*z/(2*n)) / (1 + z*z/n)
margin = z * math.sqrt((phat*(1-phat)+z*z/(4*n)) / n) / (1 + z*z/n)

print("95% CI:", round(center - margin, 3), "to", round(center + margin, 3))
```

## 6. Challenge Round
- Multiple testing without correction yields false positives.  
- Survivor bias hides failed systems.  
- Non-stationary odds invalidate earlier estimates.

## 7. Key References
- [Confidence Interval](https://en.wikipedia.org/wiki/Confidence_interval)
- [Hypothesis Testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
- [Wilson Score Interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
