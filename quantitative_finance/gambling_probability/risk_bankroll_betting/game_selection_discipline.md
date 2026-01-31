# Game Selection Discipline

## 1. Concept Skeleton
**Definition:** Systematic selection of +EV games and avoidance of negative EV tables  
**Purpose:** Preserve edge by choosing favorable rules, limits, and conditions  
**Prerequisites:** House edge, EV analysis, bankroll management

## 2. Comparative Framing
| Choice | Expected EV | Variance | Risk |
|---|---:|---:|---|
| Best rules table | Positive | Medium | Low |
| Average rules | Near zero | Medium | Medium |
| Worst rules | Negative | High | High |

## 3. Examples + Counterexamples
**Simple Example:** Choose 3:2 blackjack over 6:5 even if limits higher.  
**Failure Case:** Playing poor games to avoid waiting reduces EV to negative.  
**Edge Case:** Short-term promos can justify marginal tables temporarily.

## 4. Layer Breakdown
```
Selection Discipline:
├─ Compare rule sets
├─ Evaluate payout tables
├─ Assess competition and heat
├─ Estimate edge and variance
└─ Sit only when +EV
```

## 5. Mini-Project
Compare house edges across rules:
```python
rules = {"3:2_S17": 0.5, "6:5_H17": 1.9, "3:2_H17": 0.7}
for r, edge in rules.items():
    print(r, "house edge", edge, "%")
```

## 6. Challenge Round
- Waiting costs time; impatience reduces EV.  
- Promotions can hide worse base rules.  
- Overconfidence leads to playing suboptimal games.

## 7. Key References
- [House Edge](https://en.wikipedia.org/wiki/House_edge)
- [Wizard of Odds](https://www.wizardofodds.com/games/)
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)
