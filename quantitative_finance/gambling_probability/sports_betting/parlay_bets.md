# Parlay Bets

## 1. Concept Skeleton
**Definition:** Combine multiple legs; all must win for payout  
**Purpose:** Understand compounded odds and correlation risk  
**Prerequisites:** Probability multiplication, odds conversion

## 2. Comparative Framing
| Legs | Win Prob (50% each) | Fair Payout | Typical Payout |
|---:|---:|---:|---:|
| 2 | 25% | 3.0x | 2.6–2.8x |
| 3 | 12.5% | 7.0x | 6.0–6.5x |
| 4 | 6.25% | 15.0x | 12–13x |

## 3. Examples + Counterexamples
**Simple Example:** Two 50% legs → 25% parlay win rate.  
**Failure Case:** Assuming independence when legs are correlated.  
**Edge Case:** Correlated legs can be positive EV if book ignores correlation.

## 4. Layer Breakdown
```
Parlay Logic:
├─ Convert each leg to probability
├─ Multiply probabilities (adjust for correlation)
├─ Compare to offered payout
├─ Compute EV
└─ Decide if value exists
```

## 5. Mini-Project
Compute parlay EV with independent legs:
```python
p1, p2, payout = 0.52, 0.52, 2.7
p_win = p1 * p2
 ev = p_win * payout - (1 - p_win)
print("Parlay EV:", round(ev, 4))
```

## 6. Challenge Round
- Sportsbooks bake extra margin into parlays.  
- Correlation misestimation is the biggest trap.  
- High variance makes long losing streaks likely.

## 7. Key References
- [Parlay Betting](https://www.actionnetwork.com/education/parlay)
- [Sports Betting](https://en.wikipedia.org/wiki/Sports_betting)
- [Implied Probability](https://www.aceodds.com/bet-calculator/implied-probability)
