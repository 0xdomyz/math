# Baccarat House Edge

## 1. Concept Skeleton
**Definition:** Built-in casino advantage in baccarat, varying by bet type (Banker, Player, Tie)  
**Purpose:** Compare betting options and quantify expected loss per wager  
**Prerequisites:** Expected value, probability basics, house edge meaning

## 2. Comparative Framing
| Bet Type | Typical Payout | Win Probability (approx) | House Edge | Risk Profile |
|----------|----------------|---------------------------|------------|--------------|
| **Banker** | 0.95:1 (5% commission) | ~45.86% | ~1.06% | Lowest edge |
| **Player** | 1:1 | ~44.62% | ~1.24% | Slightly higher edge |
| **Tie** | 8:1 or 9:1 | ~9.52% | ~14.36% | Very high edge |

## 3. Examples + Counterexamples

**Example (Best Choice):**  
Banker bet for $100: expected loss ≈ $1.06 per hand.

**Example (Common Mistake):**  
Tie bet looks attractive at 8:1 but loses ≈ $14.36 per $100 over time.

**Counterexample (Misconception):**  
“Player bet is safer because no commission” → false; higher house edge than Banker.

## 4. Layer Breakdown
```
Baccarat House Edge:
├─ Game Mechanics:
│  ├─ Three outcomes: Banker, Player, Tie
│  ├─ Dealing rules fixed by table
│  └─ Natural totals end hand (8 or 9)
├─ Probability Structure:
│  ├─ Banker wins slightly more often due to drawing rules
│  ├─ Commission offsets Banker advantage
│  └─ Tie is rare but heavily advertised
├─ Expected Value:
│  ├─ EV = Σ(probability × net payout)
│  ├─ Banker: commission reduces payout to 0.95
│  ├─ Player: full payout but lower win probability
│  └─ Tie: high payout but very low probability
├─ Practical Implications:
│  ├─ Banker bet minimizes loss rate
│  ├─ Player bet acceptable if commission inconvenient
│  └─ Tie bet is mathematically poor
└─ Casino Strategy:
   ├─ Promote Tie with large payout signage
   ├─ Maintain low edge on main bets to sustain play
   └─ Earn profit through volume and time played
```

**Interaction:** Fixed rules → Banker wins more → commission neutralizes → lowest edge remains Banker.

## 5. Mini-Project
Compute expected loss and simulate outcomes:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Approximate baccarat outcome probabilities (8-deck)
P_BANKER = 0.4586
P_PLAYER = 0.4462
P_TIE = 0.0952

# Payouts
PAYOUT_BANKER = 0.95  # after 5% commission
PAYOUT_PLAYER = 1.0
PAYOUT_TIE = 8.0

# Expected value per $1 bet
EV_BANKER = P_BANKER * PAYOUT_BANKER - (1 - P_BANKER)
EV_PLAYER = P_PLAYER * PAYOUT_PLAYER - (1 - P_PLAYER)
EV_TIE = P_TIE * PAYOUT_TIE - (1 - P_TIE)

print(f"Banker EV: {EV_BANKER:.4f} (House Edge {abs(EV_BANKER):.2%})")
print(f"Player EV: {EV_PLAYER:.4f} (House Edge {abs(EV_PLAYER):.2%})")
print(f"Tie EV: {EV_TIE:.4f} (House Edge {abs(EV_TIE):.2%})")

# Simulate 10,000 hands for each bet type
n = 10000

# Non-obvious design: treat each bet independently for comparable loss distributions
banker_results = np.random.choice([PAYOUT_BANKER, -1], size=n, p=[P_BANKER, 1-P_BANKER])
player_results = np.random.choice([PAYOUT_PLAYER, -1], size=n, p=[P_PLAYER, 1-P_PLAYER])
tie_results = np.random.choice([PAYOUT_TIE, -1], size=n, p=[P_TIE, 1-P_TIE])

# Cumulative profit
banker_cum = np.cumsum(banker_results)
player_cum = np.cumsum(player_results)
tie_cum = np.cumsum(tie_results)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(banker_cum, label='Banker')
axes[0, 0].set_title('Cumulative Profit (Banker)')
axes[0, 0].set_xlabel('Hand')
axes[0, 0].set_ylabel('Profit')

axes[0, 1].plot(player_cum, color='orange', label='Player')
axes[0, 1].set_title('Cumulative Profit (Player)')
axes[0, 1].set_xlabel('Hand')
axes[0, 1].set_ylabel('Profit')

axes[1, 0].plot(tie_cum, color='red', label='Tie')
axes[1, 0].set_title('Cumulative Profit (Tie)')
axes[1, 0].set_xlabel('Hand')
axes[1, 0].set_ylabel('Profit')

axes[1, 1].hist([banker_results, player_results, tie_results], bins=30, label=['Banker','Player','Tie'], alpha=0.7)
axes[1, 1].set_title('Outcome Distributions')
axes[1, 1].set_xlabel('Profit per Hand')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
If Banker has the lowest house edge, why do some players avoid it?
- Commission feels like a “tax” even if EV is better
- Players prefer even-money payouts for simplicity
- Superstitions (e.g., “Player streaks”) override math
- Some casinos cap Banker winnings, creating perceived friction

## 7. Key References
- [Baccarat (Wikipedia)](https://en.wikipedia.org/wiki/Baccarat)
- [Wizard of Odds: Baccarat](https://www.wizardofodds.com/games/baccarat/)
- [House Edge (Wikipedia)](https://en.wikipedia.org/wiki/House_edge)

---
**Status:** Game-specific house edge | **Complements:** Expected Value, House Edge, Bankroll Management
