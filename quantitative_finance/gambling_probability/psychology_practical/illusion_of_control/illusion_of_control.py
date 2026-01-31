"""
Extracted from: illusion_of_control.md
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate slot outcomes with and without "control" action
n_spins = 10000
p_win = 0.45  # illustrative (still negative EV with payouts)

# Control action does not change probability
wins_control = np.random.binomial(1, p_win, n_spins)
wins_no_control = np.random.binomial(1, p_win, n_spins)

# Rolling win rate
window = 200
roll_control = np.convolve(wins_control, np.ones(window)/window, mode='valid')
roll_no_control = np.convolve(wins_no_control, np.ones(window)/window, mode='valid')

# Simulate perceived control effect (subjective)
perceived_boost = roll_control - roll_no_control

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Win outcomes
axes[0, 0].plot(wins_control[:200], drawstyle='steps-post', label='Control')
axes[0, 0].plot(wins_no_control[:200], drawstyle='steps-post', label='No Control', alpha=0.6)
axes[0, 0].set_title('Outcome Sequences (First 200 Spins)')
axes[0, 0].set_xlabel('Spin')
axes[0, 0].set_ylabel('Win (1) / Loss (0)')
axes[0, 0].legend()

# Plot 2: Rolling win rates
axes[0, 1].plot(roll_control, label='Control')
axes[0, 1].plot(roll_no_control, label='No Control', alpha=0.7)
axes[0, 1].axhline(p_win, color='red', linestyle='--', label='True p')
axes[0, 1].set_title('Rolling Win Rate')
axes[0, 1].set_xlabel('Spin')
axes[0, 1].set_ylabel('Win Rate')
axes[0, 1].legend()

# Plot 3: Difference in rolling rates
axes[1, 0].plot(perceived_boost, color='purple')
axes[1, 0].axhline(0, color='black', linestyle='--')
axes[1, 0].set_title('Perceived Control Effect (Difference)')
axes[1, 0].set_xlabel('Spin')
axes[1, 0].set_ylabel('Rate Difference')

# Plot 4: Distribution of win rates
axes[1, 1].hist(roll_control, bins=30, alpha=0.7, label='Control')
axes[1, 1].hist(roll_no_control, bins=30, alpha=0.7, label='No Control')
axes[1, 1].set_title('Win Rate Distribution (Rolling Window)')
axes[1, 1].set_xlabel('Win Rate')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Mean win rate (control): {wins_control.mean():.3f}")
print(f"Mean win rate (no control): {wins_no_control.mean():.3f}")
