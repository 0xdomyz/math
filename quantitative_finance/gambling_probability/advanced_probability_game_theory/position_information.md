# Position & Information Advantage

## 1. Concept Skeleton
**Definition:** Acting later provides more information about opponents’ actions  
**Purpose:** Quantify positional EV; justify looser ranges in late position  
**Prerequisites:** Poker mechanics, conditional probability, expected value

## 2. Comparative Framing
| Position | Information | Typical Range | EV |
|---|---|---|---|
| Early | Low | Tight | Lower |
| Middle | Medium | Balanced | Medium |
| Late/Button | High | Wide | Highest |

## 3. Examples + Counterexamples
**Simple Example:** Button raises wider because blinds already acted.  
**Failure Case:** Over-loosening in position against aggressive blinds loses EV.  
**Edge Case:** Short-handed tables compress position gaps.

## 4. Layer Breakdown
```
Position Advantage:
├─ Observe prior actions
├─ Update beliefs on ranges
├─ Adjust bet sizing and bluff frequency
├─ Control pot size (check back)
└─ Realize equity more effectively
```

## 5. Mini-Project
Simulate positional EV shift with a simple information bonus:
```python
import numpy as np

positions = ["Early", "Middle", "Late"]
base_ev = np.array([-0.02, 0.00, 0.02])
info_bonus = np.array([0.00, 0.01, 0.03])

ev = base_ev + info_bonus

for p, v in zip(positions, ev):
    print(f"{p} EV per hand: {v:.3f}")
```

## 6. Challenge Round
- Information advantage shrinks against strong opponents.  
- Table dynamics (limps, 3-bets) can invert positional advantage.  
- Multiway pots reduce ability to control outcomes.

## 7. Key References
- [Poker Position](https://en.wikipedia.org/wiki/Poker_strategy)
- [Pot Odds and Position](https://www.pokerstrategy.com/)
- [Information Sets](https://en.wikipedia.org/wiki/Information_set_(game_theory))
