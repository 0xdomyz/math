"""
Extracted from: heat_table_selection.md
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simplified EV adjustments (illustrative)
# Non-obvious: use rule deltas to approximate EV without full blackjack simulator
base_edge = -0.005  # -0.5% for good rules

rule_sets = {
    "Good Rules (3:2, S17, DAS)": base_edge,
    "Average Rules (3:2, H17, no DAS)": base_edge - 0.004,
    "Bad Rules (6:5, H17, no DAS)": base_edge - 0.014,
}

n_hands = 50000
bet = 10

results = {}
for name, edge in rule_sets.items():
    # simulate profit as EV + noise
    outcomes = np.random.normal(loc=edge * bet, scale=bet * 1.15, size=n_hands)
    results[name] = np.cumsum(outcomes)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative profit
for name, series in results.items():
    axes[0, 0].plot(series, label=name)
axes[0, 0].set_title('Cumulative Profit by Rule Set')
axes[0, 0].set_xlabel('Hand')
axes[0, 0].set_ylabel('Profit')
axes[0, 0].legend()

# Plot 2: Distribution of final profit
finals = [series[-1] for series in results.values()]
axes[0, 1].bar(list(results.keys()), finals)
axes[0, 1].set_title('Final Profit (50,000 Hands)')
axes[0, 1].set_ylabel('Profit')
axes[0, 1].tick_params(axis='x', rotation=20)

# Plot 3: Rule-set EV bars
axes[1, 0].bar(list(rule_sets.keys()), [e*100 for e in rule_sets.values()])
axes[1, 0].set_title('Approx House Edge by Rule Set')
axes[1, 0].set_ylabel('House Edge (%)')
axes[1, 0].tick_params(axis='x', rotation=20)

# Plot 4: Heat vs EV trade-off (illustrative)
heat = [2, 5, 1]  # arbitrary heat score
axes[1, 1].scatter([e*100 for e in rule_sets.values()], heat)
for i, name in enumerate(rule_sets.keys()):
    axes[1, 1].annotate(name, (list(rule_sets.values())[i]*100, heat[i]))
axes[1, 1].set_title('Heat vs EV Trade-off')
axes[1, 1].set_xlabel('House Edge (%)')
axes[1, 1].set_ylabel('Heat Score (lower is better)')

plt.tight_layout()
plt.show()
