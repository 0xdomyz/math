# EV in Sports Betting

## 1. Concept Skeleton
**Definition:** Expected value combines win probability and payout minus stake  
**Purpose:** Identify profitable bets and reject negative EV offers  
**Prerequisites:** Implied probability, odds conversion, EV formula

## 2. Comparative Framing
| Scenario | True Win Prob | Odds | EV |
|---|---:|---|---:|
| Small edge | 54% | -110 | +1.3% |
| No edge | 52% | -110 | -0.4% |
| Large edge | 60% | -110 | +7.6% |

## 3. Examples + Counterexamples
**Simple Example:** If true win rate is 55% at -110, EV is positive.  
**Failure Case:** Using public picks with no edge yields negative EV.  
**Edge Case:** Promo boosts can temporarily create high EV.

## 4. Layer Breakdown
```
EV Calculation:
├─ Estimate true probability
├─ Convert odds to payout
├─ Compute EV = p×win - (1-p)×stake
├─ Adjust for vig
└─ Size bet (Kelly or fractional)
```

## 5. Mini-Project
Compute EV for a given bet:
```python
def ev(prob, odds):
    if odds < 0:
        win = 100 / abs(odds)
    else:
        win = odds / 100
    return prob * win - (1 - prob) * 1

print("EV:", round(ev(0.55, -110), 4))
```

## 6. Challenge Round
- Probability estimates are noisy; small edges vanish.  
- Limits and bans restrict scaling.  
- Correlated bets inflate risk.

## 7. Key References
- [Expected Value](https://en.wikipedia.org/wiki/Expected_value)
- [Betting Mathematics](https://www.pinnacle.com/en/betting-resources/betting-strategy/expected-value)
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)
