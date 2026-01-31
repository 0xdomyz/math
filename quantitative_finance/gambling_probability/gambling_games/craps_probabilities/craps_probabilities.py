"""
Extracted from: craps_probabilities.md
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def roll_dice():
    """Return sum of two dice"""
    return np.random.randint(1, 7) + np.random.randint(1, 7)

def simulate_craps_session(num_rolls=1000, bets=['pass'], odds_multiplier=1):
    """
    Simulate craps session
    bets: list of bet types
    odds_multiplier: how much to back pass bet with odds (1 = 1x odds)
    """
    bankroll = 1000
    point = 0
    
    for _ in range(num_rolls):
        roll = roll_dice()
        
        if point == 0:
            # Come-out roll
            if roll in [7, 11]:
                bankroll += 10  # Pass wins
            elif roll in [2, 3, 12]:
                bankroll -= 10  # Pass loses
            else:
                point = roll  # Establish point
                # Place odds bet
                if 'pass' in bets:
                    bankroll -= 10 * odds_multiplier
        else:
            # Point established
            if roll == point:
                # Point wins
                bankroll += 10  # Pass wins
                # Pay odds
                if roll in [4, 10]:
                    bankroll += 10 * odds_multiplier * 2  # 2:1 odds
                elif roll in [5, 9]:
                    bankroll += 10 * odds_multiplier * 1.5  # 3:2 odds
                elif roll in [6, 8]:
                    bankroll += 10 * odds_multiplier * 1.2  # 6:5 odds
                point = 0
            elif roll == 7:
                # 7 out
                bankroll -= 10  # Pass loses
                bankroll -= 10 * odds_multiplier  # Odds lost
                point = 0
    
    return bankroll

def calculate_craps_probabilities():
    """Calculate key craps probabilities"""
    outcomes = Counter()
    
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            sum_dice = die1 + die2
            outcomes[sum_dice] += 1
    
    return outcomes

# Example 1: Dice roll probabilities
print("=== Dice Roll Probabilities ===\n")

outcomes = calculate_craps_probabilities()

print(f"{'Sum':<6} {'Count':<8} {'Probability':<15} {'Interpretation':<30}")
print("-" * 60)

for s in range(2, 13):
    count = outcomes[s]
    prob = count / 36
    if s == 7:
        interp = "Most common (16.67%)"
    elif s in [2, 12]:
        interp = "Least common (2.78%)"
    else:
        interp = ""
    print(f"{s:<6} {count:<8} {prob:<15.4f} {interp:<30}")

# Example 2: Come-out roll outcomes
print("\n\n=== Come-Out Roll Outcomes ===\n")

natural = outcomes[7] + outcomes[11]
craps = outcomes[2] + outcomes[3] + outcomes[12]
point_est = sum(outcomes[i] for i in [4, 5, 6, 8, 9, 10])

print(f"Natural (7 or 11): {natural}/36 = {natural/36:.1%} → Pass wins")
print(f"Craps (2, 3, 12): {craps}/36 = {craps/36:.1%} → Pass loses")
print(f"Point (4-10): {point_est}/36 = {point_est/36:.1%} → Continue")

# Example 3: Pass line win probability by point
print("\n\n=== Pass Line Win Probability by Point ===\n")

points = [4, 5, 6, 8, 9, 10]
sevens = 6

print(f"{'Point':<10} {'Ways to Make':<18} {'Ways to Make 7':<18} {'Win %':<15}")
print("-" * 60)

for point in points:
    ways_to_make = outcomes[point]
    win_prob = ways_to_make / (ways_to_make + sevens)
    print(f"{point:<10} {ways_to_make:<18} {sevens:<18} {win_prob:<15.1%}")

# Example 4: Odds bet payouts
print("\n\n=== Odds Bet Payouts (True Probability) ===\n")

print(f"{'Point':<10} {'Odds':<15} {'Payout on $10':<15}")
print("-" * 40)

odds_dict = {4: 2, 5: 1.5, 6: 1.2, 8: 1.2, 9: 1.5, 10: 2}

for point in points:
    odds = odds_dict[point]
    payout = 10 * odds
    print(f"{point:<10} {odds:.1f}:1 {payout:<15.1f}")

# Example 5: Session simulation
print("\n\n=== Session Simulation Comparison ===\n")

np.random.seed(42)

strategies = [
    ("Pass line only", ['pass'], 0),
    ("Pass + 1x odds", ['pass'], 1),
    ("Pass + 2x odds", ['pass'], 2),
    ("Pass + 5x odds", ['pass'], 5),
]

results_summary = []

for strat_name, bets, odds_mult in strategies:
    results = []
    for _ in range(100):
        final_bank = simulate_craps_session(num_rolls=1000, bets=bets, odds_multiplier=odds_mult)
        results.append(final_bank)
    
    avg = np.mean(results)
    std = np.std(results)
    min_val = np.min(results)
    max_val = np.max(results)
    
    results_summary.append({
        'strategy': strat_name,
        'avg': avg,
        'std': std,
        'min': min_val,
        'max': max_val
    })

print(f"{'Strategy':<20} {'Avg Bankroll':<18} {'Std Dev':<15} {'Min':<12} {'Max':<12}")
print("-" * 77)

for r in results_summary:
    print(f"{r['strategy']:<20} ${r['avg']:<17,.0f} ${r['std']:<14,.0f} ${r['min']:<11,.0f} ${r['max']:<11,.0f}")

# Example 6: Field bet vs pass line
print("\n\n=== Field Bet vs Pass Line Over 100 Rolls ===\n")

field_covered = set([2, 3, 4, 5, 9, 10, 11, 12])
uncovered = set([6, 7, 8])

field_hits = sum(1 for s in range(2, 13) if s in field_covered)
field_losses = sum(1 for s in range(2, 13) if s in uncovered)

print(f"Field covered: {field_covered} ({field_hits}/36 = {field_hits/36:.1%})")
print(f"Field uncovered: {uncovered} ({field_losses}/36 = {field_losses/36:.1%})")
print(f"House edge on field: ~5.56%")
print(f"House edge on pass: ~1.41%")
print(f"\nRecommendation: Avoid field bet; use pass + odds")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Dice roll distribution
sums = list(range(2, 13))
counts = [outcomes[s] for s in sums]
colors_dist = ['red' if s == 7 else 'orange' if s in [2, 12] else 'blue' for s in sums]

axes[0, 0].bar(sums, counts, color=colors_dist, alpha=0.7)
axes[0, 0].set_xlabel('Sum')
axes[0, 0].set_ylabel('Frequency out of 36')
axes[0, 0].set_title('Dice Sum Distribution')
axes[0, 0].set_xticks(sums)
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Come-out roll outcomes
come_out = ['Natural\n(7,11)', 'Craps\n(2,3,12)', 'Point\n(4-10)']
come_out_counts = [natural, craps, point_est]
colors_come = ['green', 'red', 'blue']

axes[0, 1].bar(come_out, come_out_counts, color=colors_come, alpha=0.7)
axes[0, 1].set_ylabel('Frequency out of 36')
axes[0, 1].set_title('Come-Out Roll Outcomes')
axes[0, 1].set_ylim([0, 30])
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Pass line win rate by point
point_win_rates = [outcomes[p] / (outcomes[p] + 6) * 100 for p in points]

axes[1, 0].plot(points, point_win_rates, 'o-', linewidth=2, markersize=8, color='darkblue')
axes[1, 0].axhline(50, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Point Value')
axes[1, 0].set_ylabel('Pass Line Win Rate (%)')
axes[1, 0].set_title('Pass Line Win Probability by Point')
axes[1, 0].set_xticks(points)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Strategy comparison (bankroll after 1000 rolls)
strats = [r['strategy'] for r in results_summary]
avgs = [r['avg'] for r in results_summary]
stds = [r['std'] for r in results_summary]

x_pos = np.arange(len(strats))
axes[1, 1].bar(x_pos, avgs, yerr=stds, capsize=5, alpha=0.7, color=['red', 'orange', 'yellow', 'green'])
axes[1, 1].axhline(1000, color='black', linestyle='--', linewidth=2, label='Starting')
axes[1, 1].set_ylabel('Final Bankroll ($)')
axes[1, 1].set_title('Strategy Performance (1000 rolls, 100 sims)')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(strats, rotation=15, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
