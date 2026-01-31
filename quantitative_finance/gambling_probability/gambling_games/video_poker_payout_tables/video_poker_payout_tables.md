# Video Poker Payout Tables

## 1. Concept Skeleton
**Definition:** The payoff schedule in video poker that determines return-to-player (RTP) and house edge  
**Purpose:** Show how payout table variations dominate long-run profitability  
**Prerequisites:** Expected value, probability of poker hands, house edge

## 2. Comparative Framing
| Paytable (Jacks or Better) | Typical RTP | House Edge | Key Feature | Player Skill Impact |
|----------------------------|-------------|------------|-------------|---------------------|
| **9/6 JoB** | ~99.54% | ~0.46% | Full house 9, flush 6 | High |
| **8/5 JoB** | ~97.30% | ~2.70% | Lower full house/flush | High |
| **7/5 JoB** | ~96.15% | ~3.85% | Worse payouts | High |
| **Deuces Wild** | ~98–100%+ | 0–2% | Wildcards shift odds | Very high |

## 3. Examples + Counterexamples

**Example (Good Paytable):**  
9/6 Jacks or Better → skilled player RTP ≈ 99.5%.

**Example (Bad Paytable):**  
8/5 Jacks or Better → same strategy loses ~2.7% long run.

**Counterexample (Misconception):**  
“Any video poker is better than slots” → false if paytable is poor.

## 4. Layer Breakdown
```
Video Poker Payout Tables:
├─ Paytable Components:
│  ├─ Hand ranks: Pair, Two Pair, Trips, Straight, Flush, Full House, etc.
│  ├─ Payout units: Coins returned per coin wagered
│  └─ Max-coin bonus: Extra for royal flush
├─ RTP Mechanics:
│  ├─ RTP = Σ(probability × payout)
│  ├─ Optimal play maximizes RTP
│  └─ Small paytable changes shift RTP materially
├─ Key Variations:
│  ├─ 9/6 JoB: Full house 9, flush 6 (best common)
│  ├─ 8/5 JoB: Reduced payouts → big RTP drop
│  ├─ Bonus Poker: Increased four-of-a-kind payouts
│  └─ Deuces Wild: Wildcards → different optimal strategy
├─ Player Implications:
│  ├─ Paytable selection often more important than speed or volatility
│  ├─ Skill mistakes drop RTP 1–3%+ easily
│  └─ Max coins required to earn top royal flush payout
└─ Casino Strategy:
   ├─ Offer lower paytables in high-traffic locations
   ├─ Market “same game” while hiding RTP difference
   └─ Use progressive jackpots to offset lower base payouts
```

**Interaction:** Paytable choice → RTP shift → long-run outcome dominated by payout schedule, not short streaks.

## 5. Mini-Project
Compare RTP for simplified paytables:
```python
import numpy as np
import matplotlib.pyplot as plt

# Approximate hand probabilities for Jacks or Better (per hand)
# Source: standard estimates; precise values require full strategy tables
hand_probs = {
    "Royal Flush": 0.000025,
    "Straight Flush": 0.000108,
    "Four of a Kind": 0.002363,
    "Full House": 0.011512,
    "Flush": 0.010949,
    "Straight": 0.011214,
    "Three of a Kind": 0.074449,
    "Two Pair": 0.129318,
    "Jacks or Better": 0.214585,
    "Nothing": 0.545477
}

# Two paytables (per coin)
paytable_9_6 = {
    "Royal Flush": 250,   # 800 for max coins; per coin base
    "Straight Flush": 50,
    "Four of a Kind": 25,
    "Full House": 9,
    "Flush": 6,
    "Straight": 4,
    "Three of a Kind": 3,
    "Two Pair": 2,
    "Jacks or Better": 1,
    "Nothing": 0
}

paytable_8_5 = paytable_9_6.copy()
paytable_8_5["Full House"] = 8
paytable_8_5["Flush"] = 5

# Expected return per coin
rtp_9_6 = sum(hand_probs[h] * paytable_9_6[h] for h in hand_probs)
rtp_8_5 = sum(hand_probs[h] * paytable_8_5[h] for h in hand_probs)

print(f"RTP 9/6 JoB: {rtp_9_6:.4f}")
print(f"RTP 8/5 JoB: {rtp_8_5:.4f}")

# Non-obvious: scale RTP to show long-run loss per $1000 wagered
loss_9_6 = (1 - rtp_9_6) * 1000
loss_8_5 = (1 - rtp_8_5) * 1000

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RTP comparison
axes[0, 0].bar(['9/6', '8/5'], [rtp_9_6, rtp_8_5], color=['green','red'])
axes[0, 0].axhline(1.0, color='black', linestyle='--')
axes[0, 0].set_title('RTP by Paytable')
axes[0, 0].set_ylabel('Return to Player')

# Plot 2: Loss per $1000 wagered
axes[0, 1].bar(['9/6', '8/5'], [loss_9_6, loss_8_5], color=['blue','orange'])
axes[0, 1].set_title('Expected Loss per $1000')
axes[0, 1].set_ylabel('Expected Loss ($)')

# Plot 3: Hand probability distribution
axes[1, 0].bar(range(len(hand_probs)), list(hand_probs.values()), color='purple')
axes[1, 0].set_xticks(range(len(hand_probs)))
axes[1, 0].set_xticklabels(list(hand_probs.keys()), rotation=45, ha='right')
axes[1, 0].set_title('Hand Probability Distribution')
axes[1, 0].set_ylabel('Probability')

# Plot 4: Payout comparison for key hands
key_hands = ["Full House","Flush","Four of a Kind","Royal Flush"]
axes[1, 1].bar(key_hands, [paytable_9_6[h] for h in key_hands], label='9/6')
axes[1, 1].bar(key_hands, [paytable_8_5[h] for h in key_hands], label='8/5', alpha=0.7)
axes[1, 1].set_title('Key Payout Differences')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
If video poker can offer near break-even RTP, why do casinos still profit from it?
- Most players deviate from optimal strategy, lowering RTP
- Many machines use reduced paytables without clear disclosure
- Speed of play (hands per hour) scales small edge into large loss
- Max-coin requirement for royal flush lowers effective RTP if ignored

## 7. Key References
- [Video Poker (Wikipedia)](https://en.wikipedia.org/wiki/Video_poker)
- [Wizard of Odds: Jacks or Better](https://www.wizardofodds.com/games/video-poker/jacks-or-better/)
- [House Edge (Wikipedia)](https://en.wikipedia.org/wiki/House_edge)

---
**Status:** Game payout analysis | **Complements:** Expected Value, Poker Probabilities, House Edge
