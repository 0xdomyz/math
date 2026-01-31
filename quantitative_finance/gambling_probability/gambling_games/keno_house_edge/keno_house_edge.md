# Keno House Edge

## 1. Concept Skeleton
**Definition:** The casino’s built-in advantage in keno, typically 25–40% depending on payout table  
**Purpose:** Quantify why keno is among the worst-value casino games  
**Prerequisites:** Expected value, combinatorics, house edge meaning

## 2. Comparative Framing
| Game | Typical House Edge | Variance | Payout Style | Skill Impact |
|------|--------------------|----------|--------------|--------------|
| **Keno** | 25–40% | Very high | Top-heavy jackpots | None |
| **Slots** | 5–15% | High | Jackpots | None |
| **Roulette (EU)** | 2.7% | Medium | Even payouts | None |
| **Blackjack (basic)** | 0.5–2% | Medium | Skill affects EV | Moderate |

## 3. Examples + Counterexamples

**Example (Bad Value):**  
$10 keno bet with 30% house edge → expected loss ≈ $3.

**Example (Misleading):**  
Large jackpot advertised but probability near zero.

**Counterexample (Better Value):**  
European roulette bet loses ~$0.27 per $10 on average.

## 4. Layer Breakdown
```
Keno House Edge:
├─ Game Structure:
│  ├─ Player selects 1–20 numbers
│  ├─ House draws 20 numbers from 80
│  └─ Payout depends on match count
├─ Probability Drivers:
│  ├─ Hypergeometric distribution governs outcomes
│  ├─ Most probability mass on low matches
│  └─ Jackpot outcomes extremely rare
├─ Payout Table Effects:
│  ├─ Operator sets payouts (not fixed by math)
│  ├─ Lower payouts increase house edge
│  └─ Different casinos → different edges
├─ Expected Value:
│  ├─ EV = Σ(probability × payout) - 1
│  ├─ Typical EV between -0.25 and -0.40
│  └─ House edge highest among mainstream games
└─ Behavioral Factors:
   ├─ Lottery-like appeal
   ├─ Slow pace reduces perceived loss rate
   └─ Jackpot focus hides negative EV
```

**Interaction:** Fixed combinatorics + adjustable payouts → persistent high edge.

## 5. Mini-Project
Compute keno EV for a simplified payout table:
```python
import math
import numpy as np
import matplotlib.pyplot as plt

# Hypergeometric probability
def hypergeom_prob(N, K, n, k):
    return math.comb(K, k) * math.comb(N-K, n-k) / math.comb(N, n)

N = 80   # total numbers
K = 20   # drawn numbers
n = 10   # chosen numbers (spot 10)

# Example payout table for 10-spot (simplified)
payouts = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 5,
    6: 20,
    7: 100,
    8: 500,
    9: 2000,
    10: 100000
}

# Calculate EV
EV = 0
probs = []
for k in range(0, n+1):
    p = hypergeom_prob(N, K, n, k)
    probs.append(p)
    EV += p * payouts.get(k, 0)

EV = EV - 1  # subtract $1 wager
house_edge = -EV

print(f"Expected Value per $1 bet: {EV:.4f}")
print(f"House Edge: {house_edge:.2%}")

# Plot probabilities and payouts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(range(n+1), probs, color='steelblue')
axes[0, 0].set_title('Probability by Matches (10-spot)')
axes[0, 0].set_xlabel('Matches')
axes[0, 0].set_ylabel('Probability')

axes[0, 1].bar(range(n+1), [payouts.get(k,0) for k in range(n+1)], color='orange')
axes[0, 1].set_title('Payout Table')
axes[0, 1].set_xlabel('Matches')
axes[0, 1].set_ylabel('Payout')

# Expected value contribution by match count
ev_contrib = [probs[k] * payouts.get(k,0) for k in range(n+1)]
axes[1, 0].bar(range(n+1), ev_contrib, color='green')
axes[1, 0].set_title('EV Contribution by Matches')
axes[1, 0].set_xlabel('Matches')
axes[1, 0].set_ylabel('Contribution')

# Cumulative probability
axes[1, 1].plot(np.cumsum(probs), marker='o')
axes[1, 1].set_title('Cumulative Probability of Matches')
axes[1, 1].set_xlabel('Matches')
axes[1, 1].set_ylabel('Cumulative Probability')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
Why does keno remain popular despite one of the worst house edges?
- Jackpot framing creates lottery-like excitement
- Slow pace hides true loss rate
- Payout tables are opaque to most players
- Small wins provide frequent reinforcement despite negative EV

## 7. Key References
- [Keno (Wikipedia)](https://en.wikipedia.org/wiki/Keno)
- [Wizard of Odds: Keno](https://www.wizardofodds.com/games/keno/)
- [House Edge (Wikipedia)](https://en.wikipedia.org/wiki/House_edge)

---
**Status:** Game-specific house edge | **Complements:** Expected Value, Combinatorics, Risk of Ruin
