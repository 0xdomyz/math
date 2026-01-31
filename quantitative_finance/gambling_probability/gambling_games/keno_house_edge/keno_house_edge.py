"""
Extracted from: keno_house_edge.md
"""

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
