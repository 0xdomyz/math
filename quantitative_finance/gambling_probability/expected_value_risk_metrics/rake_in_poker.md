# Rake in Poker

## 1. Concept Skeleton
**Definition:** Rake is the casino’s fee from each pot or time-based charge  
**Purpose:** Measure how rake reduces player EV and game viability  
**Prerequisites:** Expected value, win-rate, pot size

## 2. Comparative Framing
| Rake Type | Typical Rate | Impact | Best For |
|---|---:|---|---|
| Percentage | 2–10% | High on small pots | Low-stakes rooms |
| Cap-based | 5% capped | Medium | Mid-stakes |
| Time charge | Fixed per hour | Predictable | High-stakes |

## 3. Examples + Counterexamples
**Simple Example:** 5% rake with $5 cap on $50 pot = $2.50 fee.  
**Failure Case:** Small stakes with large rake cap can make all players -EV.  
**Edge Case:** Time rake can be cheaper for aggressive players.

## 4. Layer Breakdown
```
Rake Impact:
├─ Estimate pots/hour
├─ Calculate rake per pot
├─ Convert to $/hour cost
├─ Compare to win-rate
└─ Decide game viability
```

## 5. Mini-Project
Estimate hourly rake cost:
```python
pots_per_hour = 25
avg_pot = 60
rake_rate = 0.05
cap = 5

rake_per_pot = min(avg_pot * rake_rate, cap)
 hourly_cost = pots_per_hour * rake_per_pot
print("Rake per hour:", round(hourly_cost, 2))
```

## 6. Challenge Round
- Rake scales against beginners more than pros.  
- High rake makes marginal edges worthless.  
- Promotions can offset rake but are variable.

## 7. Key References
- [Poker Rake](https://en.wikipedia.org/wiki/Rake_(poker))
- [Poker Room Economics](https://www.pokerstrategy.com/)
- [Poker Math](https://www.pokermath.com/)
