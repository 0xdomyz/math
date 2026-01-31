# Vigorish (Vig) in Sports Betting

## 1. Concept Skeleton
**Definition:** Vig is the bookmaker fee embedded in odds (e.g., -110 lines)  
**Purpose:** Quantify break-even win rates and true EV  
**Prerequisites:** Odds conversion, implied probability

## 2. Comparative Framing
| Line | Implied Prob | Break-Even | Vig |
|---|---:|---:|---:|
| -110 / -110 | 52.38% | 52.38% | ~4.76% |
| -105 / -105 | 51.22% | 51.22% | ~2.44% |
| -120 / +100 | 54.55% / 50% | 54.55% | ~4.55% |

## 3. Examples + Counterexamples
**Simple Example:** At -110, you must win 52.38% to break even.  
**Failure Case:** Winning 51% still loses money due to vig.  
**Edge Case:** Reduced vig lines can turn a small edge profitable.

## 4. Layer Breakdown
```
Vig Analysis:
├─ Convert odds to implied probabilities
├─ Sum both sides to find overround
├─ Compute break-even rate
├─ Compare to your model edge
└─ Line shop to reduce vig
```

## 5. Mini-Project
Compute break-even win rate:
```python
line = -110
break_even = abs(line) / (abs(line) + 100)
print("Break-even win rate:", round(break_even, 4))
```

## 6. Challenge Round
- Market moves quickly erase thin edges.  
- Public bias inflates lines near big events.  
- Small samples mask true win rate.

## 7. Key References
- [Sports Betting Odds](https://www.espn.com/betting/)
- [Implied Probability](https://www.aceodds.com/bet-calculator/implied-probability)
- [Overround](https://en.wikipedia.org/wiki/Overround)
