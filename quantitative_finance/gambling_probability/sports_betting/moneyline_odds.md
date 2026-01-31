# Moneyline Odds

## 1. Concept Skeleton
**Definition:** Moneyline odds represent payout without point spreads (American, decimal, fractional)  
**Purpose:** Convert odds to implied probability and compare to true win rates  
**Prerequisites:** Odds conversion, probability

## 2. Comparative Framing
| Format | Example | Implied Prob | Note |
|---|---|---:|---|
| American | -150 | 60% | Favorite |
| American | +200 | 33.3% | Underdog |
| Decimal | 2.50 | 40% | Total return |

## 3. Examples + Counterexamples
**Simple Example:** +150 implies 40%? No, it implies 40%? Actually 100/(100+150)=40%.  
**Failure Case:** Mixing decimal and American without conversion leads to wrong EV.  
**Edge Case:** Promotions (boosted odds) can flip EV positive.

## 4. Layer Breakdown
```
Moneyline Workflow:
├─ Read odds format
├─ Convert to implied probability
├─ Estimate true win probability
├─ Compare to break-even
└─ Decide to bet or pass
```

## 5. Mini-Project
Convert American to implied probability:
```python
def implied_prob_american(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

for o in [-150, +200]:
    print(o, round(implied_prob_american(o), 4))
```

## 6. Challenge Round
- Odds vary across books; line shopping matters.  
- Injury/news impacts probability quickly.  
- Public bias skews favorites and big-market teams.

## 7. Key References
- [Betting Odds Guide](https://www.espn.com/betting/)
- [Implied Probability](https://www.aceodds.com/bet-calculator/implied-probability)
- [Sports Betting](https://en.wikipedia.org/wiki/Sports_betting)
