"""
Extracted from: hot_hand_fallacy.md
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate coin flips (fair)
num_trials = 1000
flips = np.random.binomial(1, 0.5, num_trials)

# Identify streaks
streak_lengths = []
current = 1
for i in range(1, num_trials):
    if flips[i] == flips[i-1]:
        current += 1
    else:
        streak_lengths.append(current)
        current = 1
streak_lengths.append(current)

# Conditional probability after streaks
max_k = 6
cond_probs = []
for k in range(1, max_k + 1):
    indices = []
    count = 0
    for i in range(k, num_trials):
        if np.all(flips[i-k:i] == 1):  # k wins in a row
            count += 1
            indices.append(i)
    if count > 0:
        cond_prob = flips[indices].mean()
    else:
        cond_prob = np.nan
    cond_probs.append(cond_prob)

# Simulate skill-based streaks (slight edge)
skill_flips = np.random.binomial(1, 0.55, num_trials)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sequence of flips
axes[0, 0].plot(flips[:200], drawstyle='steps-post')
axes[0, 0].set_title('Random Outcomes (First 200)')
axes[0, 0].set_xlabel('Trial')
axes[0, 0].set_ylabel('Outcome')

# Plot 2: Streak length distribution
axes[0, 1].hist(streak_lengths, bins=range(1, 15), alpha=0.7, color='orange')
axes[0, 1].set_title('Streak Length Distribution')
axes[0, 1].set_xlabel('Streak Length')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Conditional probability after k wins
axes[1, 0].plot(range(1, max_k + 1), cond_probs, marker='o')
axes[1, 0].axhline(0.5, color='red', linestyle='--', label='True p=0.5')
axes[1, 0].set_title('P(Win | k Wins in a Row)')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('Conditional Probability')
axes[1, 0].legend()

# Plot 4: Random vs skill-based win rates
window = 50
random_rate = np.convolve(flips, np.ones(window)/window, mode='valid')
skill_rate = np.convolve(skill_flips, np.ones(window)/window, mode='valid')

axes[1, 1].plot(random_rate, label='Random (p=0.5)')
axes[1, 1].plot(skill_rate, label='Skill (p=0.55)')
axes[1, 1].set_title('Rolling Win Rate (50-trial window)')
axes[1, 1].set_xlabel('Trial')
axes[1, 1].set_ylabel('Win Rate')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Average streak length: {np.mean(streak_lengths):.2f}")
print(f"Max streak length observed: {max(streak_lengths)}")
print(f"Conditional probs after k wins: {cond_probs}")
