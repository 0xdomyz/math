# Over/Under (Total) Betting

## 1. Concept Skeleton
**Definition:** Bet on total points being over or under a posted line  
**Purpose:** Evaluate total-score distribution vs bookmaker line  
**Prerequisites:** Probability, distributions, odds conversion

## 2. Comparative Framing
| Market | Typical Line | Drivers | Volatility |
|---|---|---|---|
| NBA total | 210–240 | Pace, efficiency | Medium |
| NFL total | 38–52 | Weather, tempo | High |
| Soccer total | 2–3 | Styles, finishing | High |

## 3. Examples + Counterexamples
**Simple Example:** Total 48.5, model mean 51 → over is positive EV.  
**Failure Case:** Ignoring pace changes from injuries leads to wrong totals.  
**Edge Case:** Extreme weather can collapse totals quickly.

## 4. Layer Breakdown
```
Total Betting:
├─ Model expected points
├─ Estimate variance of totals
├─ Convert line to implied probability
├─ Compare to model probability
└─ Place bet or pass
```

## 5. Mini-Project
Compute over probability with normal model:
```python
import math

mean_total = 52
std_total = 10
line = 48.5
z = (line - mean_total) / std_total
p_over = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
print("Over probability:", round(p_over, 3))
```

## 6. Challenge Round
- Totals are sensitive to tempo assumptions.  
- Late-game strategies distort totals.  
- Public overs bias lines upward.

## 7. Key References
- [Over/Under Betting](https://www.gambling.com/over-under-betting)
- [Totals Betting Basics](https://www.actionnetwork.com/education/over-under-betting)
- [Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
