"""
Extracted from: sunk_cost_fallacy.md
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_session(chase_losses=False, max_rounds=200, bankroll=500, bet=10, p_win=0.48, payout=1.0):
    history = []
    b = bankroll
    for _ in range(max_rounds):
        if b <= 0:
            break
        wager = bet
        # Non-obvious concept: chasing behavior escalates bets after drawdowns
        if chase_losses and len(history) > 0 and history[-1] < 0:
            wager = min(b, bet * 2)
        outcome = np.random.rand() < p_win
        if outcome:
            b += wager * payout
            history.append(wager * payout)
        else:
            b -= wager
            history.append(-wager)
    return b, history

# Run simulations
n_sessions = 500
finals_normal = []
finals_chase = []

for _ in range(n_sessions):
    b1, _ = simulate_session(chase_losses=False)
    b2, _ = simulate_session(chase_losses=True)
    finals_normal.append(b1)
    finals_chase.append(b2)

# Single illustrative path
_, path_normal = simulate_session(chase_losses=False)
_, path_chase = simulate_session(chase_losses=True)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sample bankroll paths
axes[0, 0].plot(np.cumsum(path_normal), label='No Chase')
axes[0, 0].plot(np.cumsum(path_chase), label='Chase Losses')
axes[0, 0].set_title('Bankroll Change (Single Session)')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Cumulative Profit')
axes[0, 0].legend()

# Plot 2: Final bankroll distribution
axes[0, 1].hist(finals_normal, bins=30, alpha=0.7, label='No Chase')
axes[0, 1].hist(finals_chase, bins=30, alpha=0.7, label='Chase Losses')
axes[0, 1].set_title('Final Bankroll Distribution')
axes[0, 1].set_xlabel('Bankroll')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Plot 3: Probability of ruin
ruin_normal = np.mean(np.array(finals_normal) <= 0)
ruin_chase = np.mean(np.array(finals_chase) <= 0)
axes[1, 0].bar(['No Chase', 'Chase Losses'], [ruin_normal, ruin_chase], color=['blue','red'])
axes[1, 0].set_title('Probability of Ruin')
axes[1, 0].set_ylabel('Probability')

# Plot 4: Average bankroll by round
max_len = min(len(path_normal), len(path_chase))
axes[1, 1].plot(np.cumsum(path_normal[:max_len]), label='No Chase')
axes[1, 1].plot(np.cumsum(path_chase[:max_len]), label='Chase Losses')
axes[1, 1].set_title('Average Trend (Sample Paths)')
axes[1, 1].set_xlabel('Round')
axes[1, 1].set_ylabel('Cumulative Profit')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Ruin probability (no chase): {ruin_normal:.2%}")
print(f"Ruin probability (chase losses): {ruin_chase:.2%}")
