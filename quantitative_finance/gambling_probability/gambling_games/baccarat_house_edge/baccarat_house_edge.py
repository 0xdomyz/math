"""
Extracted from: baccarat_house_edge.md
"""

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
