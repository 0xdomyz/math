"""
Extracted from: house_edge.md
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to calculate house edge
def house_edge(payout_ratio, true_probability):
    """
    Calculate house edge given payout ratio and true probability
    payout_ratio: amount won per unit bet
    true_probability: true probability of winning
    """
    expected_win = (1 + payout_ratio) * true_probability - 1
    return -expected_win * 100

# Example 1: Common Casino Games
print("=== House Edges in Popular Games ===\n")

games = {
    "Blackjack (basic strategy)": {"edge": 0.5, "rtp": 99.5},
    "Craps (pass/don't pass)": {"edge": 1.4, "rtp": 98.6},
    "Baccarat": {"edge": 1.06, "rtp": 98.94},
    "Roulette (European)": {"edge": 2.7, "rtp": 97.3},
    "Roulette (American)": {"edge": 5.26, "rtp": 94.74},
    "Slots (average)": {"edge": 8.0, "rtp": 92.0},
    "Keno": {"edge": 30.0, "rtp": 70.0},
}

for game, values in games.items():
    print(f"{game:30s} | Edge: {values['edge']:5.2f}% | RTP: {values['rtp']:5.2f}%")

# Example 2: Roulette detailed calculation
print("\n\n=== Roulette House Edge Calculation ===")
print("American Roulette (18 red, 18 black, 2 green = 38 total):")

# Betting on red
true_prob_red = 18/38
payout_on_red = 1  # Bet $1, win $1 (get $2 total)
edge_red = house_edge(payout_on_red, true_prob_red)

print(f"  True probability of red: {true_prob_red:.4f} ({18}/38)")
print(f"  Payout if red wins: 1:1")
print(f"  House edge: {edge_red:.2f}%")
print(f"  Expected loss per $100 bet: ${abs(edge_red):.2f}")

# Betting on a number
true_prob_number = 1/38
payout_on_number = 35  # Bet $1, win $35
edge_number = house_edge(payout_on_number, true_prob_number)

print(f"\n  True probability of single number: {true_prob_number:.4f} (1/38)")
print(f"  Payout if wins: 35:1")
print(f"  House edge: {edge_number:.2f}%")
print(f"  Expected loss per $100 bet: ${abs(edge_number):.2f}")

# Example 3: Simulate outcomes
print("\n\n=== Long-Term Simulation ===")

def simulate_roulette(bet_size, num_bets, edge_pct, seed=42):
    """Simulate roulette betting with house edge"""
    np.random.seed(seed)
    cumulative_loss = 0
    
    for _ in range(num_bets):
        if np.random.random() < (1 - edge_pct/100):
            win = bet_size  # Win bet
        else:
            win = -bet_size  # Lose bet
        cumulative_loss -= win
    
    return cumulative_loss

# Compare games
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: House edge comparison bar chart
game_names = list(games.keys())
edges = [games[g]["edge"] for g in game_names]
colors = ['green' if e < 2 else 'orange' if e < 5 else 'red' for e in edges]

axes[0, 0].barh(game_names, edges, color=colors, alpha=0.7)
axes[0, 0].set_xlabel('House Edge (%)')
axes[0, 0].set_title('Casino House Edges')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Expected loss vs number of bets
bet_size = 10
num_bets_range = np.array([10, 50, 100, 500, 1000, 5000, 10000])
edge_pct = 5.26  # Roulette

expected_losses = [(edge_pct/100) * bet_size * n for n in num_bets_range]

axes[0, 1].plot(num_bets_range, expected_losses, 'o-', linewidth=2, markersize=8, color='darkred')
axes[0, 1].fill_between(num_bets_range, expected_losses, alpha=0.3, color='red')
axes[0, 1].set_xlabel('Number of $10 Bets')
axes[0, 1].set_ylabel('Expected Loss ($)')
axes[0, 1].set_title(f'Expected Loss Over Time (Roulette, {edge_pct}% edge)')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Distribution of outcomes (many simulations)
num_simulations = 1000
outcomes_100bets = [simulate_roulette(10, 100, 5.26) for _ in range(num_simulations)]
outcomes_1000bets = [simulate_roulette(10, 1000, 5.26) for _ in range(num_simulations)]

axes[1, 0].hist(outcomes_100bets, bins=50, alpha=0.5, label='100 bets', color='blue')
axes[1, 0].hist(outcomes_1000bets, bins=50, alpha=0.5, label='1000 bets', color='red')
axes[1, 0].axvline(np.mean(outcomes_100bets), color='blue', linestyle='--', linewidth=2)
axes[1, 0].axvline(np.mean(outcomes_1000bets), color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Net Loss ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Outcomes\n(1000 simulations of $10 bets)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Game comparison expected loss
num_bets = 100
bet_size = 10
games_compare = ['Blackjack', 'Craps', 'Roulette (EU)', 'Roulette (US)', 'Slots']
edges_compare = [0.5, 1.4, 2.7, 5.26, 8.0]
expected_losses_compare = [(e/100) * bet_size * num_bets for e in edges_compare]

axes[1, 1].bar(games_compare, expected_losses_compare, color=['green', 'yellowgreen', 'yellow', 'orange', 'red'], alpha=0.7)
axes[1, 1].set_ylabel('Expected Loss ($)')
axes[1, 1].set_title(f'Expected Loss Comparison\n({num_bets} × ${bet_size} bets)')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, loss in enumerate(expected_losses_compare):
    axes[1, 1].text(i, loss + 0.1, f'${loss:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\n=== Expected Loss Example ===")
print(f"If you bet $100 on roulette 100 times:")
print(f"  Expected loss (American): ${(5.26/100) * 100 * 100:.2f}")
print(f"  Expected loss (European): ${(2.7/100) * 100 * 100:.2f}")
print(f"\nOver a lifetime of gambling, this compounds dramatically.")
