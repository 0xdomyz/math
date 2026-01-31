"""
Gambler's Fallacy Analysis
Extracted from: gamblers_fallacy.md
Simulate random sequences to demonstrate independence
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def simulate_roulette_fallacy(num_spins=10000):
    """
    Simulate roulette spins and track "due" betting
    Red = 1, Black = 0 (simplified, ignore green)
    """
    red_numbers = set(
        [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
    )

    spins = []
    streak_lengths = []
    current_color = None
    current_streak = 0

    for _ in range(num_spins):
        spin = np.random.randint(0, 37)  # European roulette
        if spin in red_numbers:
            color = "red"
        elif spin == 0:
            color = "green"
        else:
            color = "black"

        spins.append(color)

        if color == current_color and color != "green":
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_color = color
            current_streak = 1

    return spins, streak_lengths


def test_independence(spins, color_to_test="red"):
    """
    Test if outcome is independent of previous outcome
    """
    # Count transitions
    prev_color = None
    after_red = {"red": 0, "black": 0, "green": 0}
    after_black = {"red": 0, "black": 0, "green": 0}

    for spin in spins:
        if prev_color == "red":
            after_red[spin] += 1
        elif prev_color == "black":
            after_black[spin] += 1
        prev_color = spin

    total_after_red = sum(after_red.values())
    total_after_black = sum(after_black.values())

    return after_red, after_black, total_after_red, total_after_black


# Example 1: Demonstrate independence
print("=== Gambler's Fallacy: Independence Test ===\n")

np.random.seed(42)

spins, streaks = simulate_roulette_fallacy(num_spins=100000)

after_red, after_black, total_red, total_black = test_independence(spins)

print(f"After RED spin:")
print(f"  Next RED: {after_red['red']/total_red:.1%} (Expected: ~48.6%)")
print(f"  Next BLACK: {after_black['black']/total_black:.1%} (Expected: ~48.6%)")
print(f"  Next GREEN: {after_red['green']/total_red:.1%} (Expected: ~2.7%)\n")

print(f"After BLACK spin:")
print(f"  Next RED: {after_black['red']/total_black:.1%} (Expected: ~48.6%)")
print(f"  Next BLACK: {after_black['black']/total_black:.1%} (Expected: ~48.6%)")
print(f"  Next GREEN: {after_black['green']/total_black:.1%} (Expected: ~2.7%)\n")

print("Conclusion: Previous spin does NOT affect next spin probability")

# Example 2: Streak analysis
print("\n\n=== Streak Length Distribution ===\n")

streak_counts = np.bincount(streaks)

print(f"{'Streak Length':<20} {'Count':<12} {'Probability':<15}")
print("-" * 47)

for length in range(1, min(11, len(streak_counts))):
    if length < len(streak_counts):
        count = streak_counts[length]
        prob = count / len(streaks)
        theoretical = (0.486**length) * (1 - 0.486)  # Geometric distribution
        print(f"{length:<20} {count:<12} {prob:<15.4f} (Theory: {theoretical:.4f})")

# Example 3: "Due" number fallacy
print("\n\n=== 'Due' Number Fallacy ===\n")

# Track numbers that haven't appeared
np.random.seed(42)
num_spins = 1000
roulette_spins = [np.random.randint(0, 37) for _ in range(num_spins)]

# Find longest gap for each number
number_gaps = {i: [] for i in range(37)}
last_seen = {i: -1 for i in range(37)}

for spin_idx, number in enumerate(roulette_spins):
    for num in range(37):
        if num == number:
            gap = spin_idx - last_seen[num] - 1
            number_gaps[num].append(gap)
            last_seen[num] = spin_idx

# Calculate max gap for each number
max_gaps = {num: max(gaps) if gaps else num_spins for num, gaps in number_gaps.items()}

print(f"Number with longest gap: {max(max_gaps, key=max_gaps.get)}")
print(f"Max gap: {max(max_gaps.values())} spins")
print(f"\nNext spin probability for 'overdue' number: 1/37 = 2.7%")
print(f"Next spin probability for any other number: 1/37 = 2.7%")
print(f"\nConclusion: 'Overdue' numbers have SAME probability as any other")

# Example 4: Betting on "due" outcomes (loss simulation)
print("\n\n=== Cost of Gambler's Fallacy ===\n")


def simulate_fallacy_betting(num_sessions=1000):
    """
    Simulate betting on 'due' outcomes vs random
    """
    fallacy_bankroll = 1000
    random_bankroll = 1000

    for _ in range(num_sessions):
        spins = [np.random.randint(0, 37) for _ in range(100)]

        # Gambler's fallacy: Bet on color after 3-streak
        current_streak = 0
        prev_color = None

        for spin in spins:
            red_set = set(
                [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
            )
            if spin in red_set:
                color = "red"
            elif spin == 0:
                color = "green"
            else:
                color = "black"

            # Fallacy bettor: Bet against streak of 3+
            if current_streak >= 3 and prev_color in ["red", "black"]:
                opposite_color = "black" if prev_color == "red" else "red"
                if color == opposite_color:
                    fallacy_bankroll += 10
                else:
                    fallacy_bankroll -= 10

            # Random bettor: Bet on random color each time
            if np.random.random() < 0.5:
                if color == "red":
                    random_bankroll += 10
                else:
                    random_bankroll -= 10
            else:
                if color == "black":
                    random_bankroll += 10
                else:
                    random_bankroll -= 10

            # Update streak
            if color == prev_color and color != "green":
                current_streak += 1
            else:
                current_streak = 1
                prev_color = color

    return fallacy_bankroll, random_bankroll


fallacy_final, random_final = simulate_fallacy_betting(num_sessions=1000)

print(f"Starting bankroll: $1000\n")
print(f"Fallacy bettor (bets against streaks):")
print(f"  Final bankroll: ${fallacy_final:,.0f}")
print(f"  Loss: ${1000 - fallacy_final:,.0f}\n")

print(f"Random bettor (no pattern):")
print(f"  Final bankroll: ${random_final:,.0f}")
print(f"  Loss: ${1000 - random_final:,.0f}\n")

print(f"Conclusion: Both lose to house edge; fallacy provides NO advantage")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Independence verification
after_red_probs = [
    after_red["red"] / total_red,
    after_red["black"] / total_red,
    after_red["green"] / total_red,
]
after_black_probs = [
    after_black["red"] / total_black,
    after_black["black"] / total_black,
    after_black["green"] / total_black,
]
expected_probs = [18 / 37, 18 / 37, 1 / 37]

x_pos = np.arange(3)
width = 0.25

axes[0, 0].bar(
    x_pos - width, after_red_probs, width, label="After RED", alpha=0.7, color="red"
)
axes[0, 0].bar(
    x_pos, after_black_probs, width, label="After BLACK", alpha=0.7, color="black"
)
axes[0, 0].bar(
    x_pos + width, expected_probs, width, label="Expected", alpha=0.7, color="green"
)
axes[0, 0].set_ylabel("Probability")
axes[0, 0].set_title("Independence: Next Outcome Given Previous Color")
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(["Next RED", "Next BLACK", "Next GREEN"])
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis="y")

# Plot 2: Streak length distribution
streak_lengths_plot = [len([s for s in streaks if s == i]) for i in range(1, 11)]
theoretical_probs = [(0.486**i) * (1 - 0.486) * len(streaks) for i in range(1, 11)]

x_streaks = np.arange(1, 11)
axes[0, 1].bar(
    x_streaks - 0.2, streak_lengths_plot, 0.4, label="Observed", alpha=0.7, color="blue"
)
axes[0, 1].bar(
    x_streaks + 0.2,
    theoretical_probs,
    0.4,
    label="Theoretical",
    alpha=0.7,
    color="orange",
)
axes[0, 1].set_xlabel("Streak Length")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Streak Length Distribution (100k spins)")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis="y")

# Plot 3: Gap distribution for single number
num_appearances = roulette_spins.count(17)  # Track number 17
gaps_17 = number_gaps[17]

if gaps_17:
    axes[1, 0].hist(gaps_17, bins=20, alpha=0.7, color="purple", edgecolor="black")
    axes[1, 0].axvline(
        np.mean(gaps_17),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean gap: {np.mean(gaps_17):.1f}",
    )
    axes[1, 0].axvline(
        37, color="green", linestyle="--", linewidth=2, label="Expected: 37"
    )
    axes[1, 0].set_xlabel("Gap Between Appearances")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Number 17 Gap Distribution (1000 spins)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis="y")

# Plot 4: Cumulative outcomes (law of large numbers)
np.random.seed(42)
cumulative_red = []
cumulative_black = []
red_count = 0
black_count = 0

for i, spin in enumerate(spins[:1000]):
    if spin == "red":
        red_count += 1
    elif spin == "black":
        black_count += 1

    cumulative_red.append(red_count / (i + 1) * 100 if i > 0 else 0)
    cumulative_black.append(black_count / (i + 1) * 100 if i > 0 else 0)

spin_range = np.arange(1, 1001)
axes[1, 1].plot(spin_range, cumulative_red, label="Red %", color="red", linewidth=2)
axes[1, 1].plot(
    spin_range, cumulative_black, label="Black %", color="black", linewidth=2
)
axes[1, 1].axhline(
    48.6, color="green", linestyle="--", linewidth=2, label="Expected 48.6%"
)
axes[1, 1].fill_between(spin_range, 45, 52, alpha=0.1, color="gray")
axes[1, 1].set_xlabel("Number of Spins")
axes[1, 1].set_ylabel("Cumulative Percentage")
axes[1, 1].set_title("Law of Large Numbers (Converges Over Time)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim([40, 60])

plt.tight_layout()
plt.savefig("gamblers_fallacy_analysis.png", dpi=100, bbox_inches="tight")
print("\n✓ Visualization saved: gamblers_fallacy_analysis.png")
plt.show()
