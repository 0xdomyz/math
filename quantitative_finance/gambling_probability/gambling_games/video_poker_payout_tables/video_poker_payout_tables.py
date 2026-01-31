"""
Extracted from: video_poker_payout_tables.md
"""

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
