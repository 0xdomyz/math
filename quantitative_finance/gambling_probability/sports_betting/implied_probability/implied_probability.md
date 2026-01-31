# Implied Probability

## 1. Concept Skeleton
**Definition:** Implied probability converts odds into the bookmaker’s estimated chance  
**Purpose:** Compare to your model probability to determine edge  
**Prerequisites:** Odds formats, probability arithmetic

## 2. Comparative Framing
| Odds Format | Example | Implied Prob |
|---|---|---:|
| American | -120 | 54.55% |
| American | +150 | 40.00% |
| Decimal | 2.25 | 44.44% |

## 3. Examples + Counterexamples
**Simple Example:** +150 implies 40% win rate; if you estimate 45%, it’s +EV.  
**Failure Case:** Comparing raw odds across books without conversion.  
**Edge Case:** Two-way markets have overround; implied probs sum >100%.

## 4. Layer Breakdown
```
Implied Probability:
├─ Identify odds format
├─ Convert to probability
├─ Adjust for overround
├─ Compare to model probability
└─ Compute expected value
```

## 5. Mini-Project
Convert odds and compute edge:
```python
def implied_prob(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

odds = +150
true_p = 0.45
imp_p = implied_prob(odds)
edge = true_p - imp_p
print("Edge:", round(edge, 4))
```

## 6. Challenge Round
- Removing vig requires normalizing both sides.  
- Market moves fast; stale odds ruin edges.  
- Model error dominates when edges are small.

## 7. Key References
- [Implied Probability](https://www.aceodds.com/bet-calculator/implied-probability)
- [Sports Betting Odds](https://www.espn.com/betting/)
- [Overround](https://en.wikipedia.org/wiki/Overround)
