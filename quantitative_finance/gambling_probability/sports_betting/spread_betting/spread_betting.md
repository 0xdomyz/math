# Spread Betting

## 1. Concept Skeleton
**Definition:** Point spread handicaps the favorite to balance odds  
**Purpose:** Evaluate true win probability against adjusted score  
**Prerequisites:** Probability, odds conversion, margin distributions

## 2. Comparative Framing
| Spread | Favorite Needs | Typical Line | Note |
|---|---|---|---|
| -3.5 | Win by 4+ | -110 | Common in NFL |
| +7.5 | Lose by ≤7 | -110 | Underdog cushion |
| -1.5 | Win by 2+ | -120 | Tight matchup |

## 3. Examples + Counterexamples
**Simple Example:** Team -3.5 must win by 4 to cover.  
**Failure Case:** Ignoring key numbers (3, 7) in NFL reduces value.  
**Edge Case:** Late-game variance (garbage time) flips spreads more than moneylines.

## 4. Layer Breakdown
```
Spread Evaluation:
├─ Model score distribution
├─ Convert spread to win probability
├─ Compare to implied probability
├─ Adjust for key numbers
└─ Decide bet sizing
```

## 5. Mini-Project
Estimate cover probability with a normal margin:
```python
import math

mean_margin = 4
std_margin = 10
spread = 3.5

z = (spread - mean_margin) / std_margin
p_cover = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
print("Cover probability:", round(p_cover, 3))
```

## 6. Challenge Round
- Score distributions are not perfectly normal.  
- Key-number pricing can be asymmetric.  
- Line movement after injuries can erase edge.

## 7. Key References
- [Point Spread](https://en.wikipedia.org/wiki/Point_spread)
- [Sports Betting Guide](https://www.draftkings.com/help/article/beginner-sports-betting-guide)
- [Implied Probability](https://www.aceodds.com/bet-calculator/implied-probability)
