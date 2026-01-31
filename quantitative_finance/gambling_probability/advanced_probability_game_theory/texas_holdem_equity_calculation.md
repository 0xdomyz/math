# Texas Hold'em Equity Calculation

## 1. Concept Skeleton
**Definition:** Equity = probability a hand wins (or ties) against a range over random future cards  
**Purpose:** Compare hands, decide calls/raises, price draws vs pot odds  
**Prerequisites:** Combinatorics, conditional probability, hand ranking basics

## 2. Comparative Framing
| Scenario | Exact Equity | Simulation | Use Case |
|---|---:|---:|---|
| Preflop AA vs KK | 81.9% | ~82% | Quick baseline |
| Flop draw vs made hand | Exact when small tree | ~Approx | Fast decision |
| Range vs range | Intractable | Monte Carlo | Practical play |

## 3. Examples + Counterexamples
**Simple Example:** AA vs KK preflop ≈ 82% equity; profitable to stack off.  
**Failure Case:** Using raw win-rate without ties overstates EV in split-pot games.  
**Edge Case:** Multiway pots reduce equity and increase variance for drawing hands.

## 4. Layer Breakdown
```
Equity Calculation:
├─ Define hand(s) and ranges
├─ Enumerate or sample remaining board cards
├─ Evaluate best 5-card hand per player
├─ Count wins, losses, ties
├─ Convert to equity = (wins + 0.5×ties)/total
└─ Compare to pot odds and bet sizing
```

## 5. Mini-Project
Estimate equity for two fixed hands via simplified Monte Carlo (toy evaluator):
```python
import random
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "shdc"

def deck():
    return [r+s for r in RANKS for s in SUITS]

def hand_score(cards):
    """Toy evaluator: counts pairs/trips/quads, then high-card fallback."""
    ranks = [c[0] for c in cards]
    counts = Counter(ranks)
    counts_sorted = sorted(counts.values(), reverse=True)
    high = max([RANKS.index(r) for r in ranks])
    return (counts_sorted, high)

def equity(hand_a, hand_b, trials=5000):
    wins = ties = 0
    for _ in range(trials):
        d = deck()
        for c in hand_a + hand_b:
            d.remove(c)
        board = random.sample(d, 5)
        score_a = hand_score(hand_a + board)
        score_b = hand_score(hand_b + board)
        if score_a > score_b:
            wins += 1
        elif score_a == score_b:
            ties += 1
    return (wins + 0.5 * ties) / trials

print("AA vs KK equity (toy):", equity(["As","Ah"],["Ks","Kh"]))
```

## 6. Challenge Round
- Simplified evaluators can mis-rank hands; use full evaluators for accuracy.  
- Range vs range requires careful weighting; naive averages mislead.  
- Multiway equity can flip decisions; heads-up intuition fails.

## 7. Key References
- [Poker Hand Rankings](https://en.wikipedia.org/wiki/Poker_hand)
- [Texas Hold'em Odds](https://www.pokerstrategy.com/)
- [Monte Carlo Methods](https://en.wikipedia.org/wiki/Monte_Carlo_method)
