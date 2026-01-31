"""
Extracted from: law_of_large_numbers_detailed.md
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Non-obvious: small edge requires massive N to detect with confidence
games = {
    "Roulette (2.7%)": 0.027,
    "Blackjack (0.5%)": 0.005,
    "Craps (1.4%)": 0.014,
    "Slots (10%)": 0.10
}

n_simulations = 100
max_hands = 1000000

results = {}

for game_name, house_edge in games.items():
    trajectories = []
    
    for sim in range(n_simulations):
        outcomes = np.random.binomial(1, 1 - house_edge, max_hands)
        cumsum = np.cumsum(outcomes - (1 - house_edge))
        trajectories.append(cumsum)
    
    results[game_name] = trajectories

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (game_name, house_edge) in enumerate(games.items()):
    trajectories = results[game_name]
    
    # Plot sample paths
    for traj in trajectories[:10]:
        axes[idx].plot(traj, alpha=0.1, color='blue')
    
    # Plot mean trajectory
    mean_traj = np.mean(trajectories, axis=0)
    axes[idx].plot(mean_traj, color='red', linewidth=2, label='Mean')
    
    # Theoretical expectation
    theoretical = np.arange(max_hands) * (-house_edge)
    axes[idx].plot(theoretical, color='green', linewidth=2, linestyle='--', label='Theory')
    
    axes[idx].set_title(f'{game_name}')
    axes[idx].set_xlabel('Number of Hands')
    axes[idx].set_ylabel('Cumulative Profit')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)
    axes[idx].set_xscale('log')

plt.tight_layout()
plt.show()

# Convergence rate
print("Sample sizes for 95% confidence in detecting house edge:")
print("(Using standard error = σ / √N with σ ≈ 1)")
for game_name, house_edge in games.items():
    z_score = 1.96
    acceptable_error = house_edge * 0.1  # 10% of edge
    n_required = (z_score / acceptable_error) ** 2
    print(f"{game_name}: {n_required:,.0f} hands")
