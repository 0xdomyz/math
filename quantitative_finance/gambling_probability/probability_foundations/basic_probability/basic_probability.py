"""
Extracted from: basic_probability.md
"""

import numpy as np
from itertools import combinations, permutations
import matplotlib.pyplot as plt

# Example 1: Basic Probability (Dice)
print("=== Basic Probability ===")
print("Fair 6-sided die:")
print(f"  P(rolling 3): 1/6 = {1/6:.4f}")
print(f"  P(rolling even): 3/6 = {3/6:.4f}")
print(f"  P(rolling > 4): 2/6 = {2/6:.4f}")

# Example 2: Compound Events
print("\n=== Compound Events ===")
print("Two fair dice:")
p_sum_7 = len([(i,j) for i in range(1,7) for j in range(1,7) if i+j==7]) / 36
p_both_even = (3/6) * (3/6)  # Independent
print(f"  P(sum = 7): {p_sum_7:.4f}")
print(f"  P(both even): {p_both_even:.4f}")
print(f"  P(sum ≥ 10): {len([(i,j) for i in range(1,7) for j in range(1,7) if i+j>=10])/36:.4f}")

# Example 3: Conditional Probability
print("\n=== Conditional Probability ===")
print("Card drawing (without replacement):")
print(f"  P(1st card Ace): 4/52 = {4/52:.4f}")
print(f"  P(2nd card Ace | 1st was Ace): 3/51 = {3/51:.4f}")
print(f"  P(both Aces): (4/52) × (3/51) = {(4/52)*(3/51):.4f}")

# Example 4: Odds Conversion
print("\n=== Odds Conversion ===")

def prob_to_odds(p):
    """Convert probability to decimal odds"""
    return 1 / p

def odds_to_prob(odds):
    """Convert decimal odds to probability"""
    return 1 / odds

def prob_to_american(p):
    """Convert probability to American odds"""
    if p >= 0.5:
        return -100 / (1/p - 1)
    else:
        return 100 * (1/p - 1)

def american_to_prob(american):
    """Convert American odds to implied probability"""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)

probabilities = [0.25, 0.5, 0.75]
for p in probabilities:
    decimal = prob_to_odds(p)
    american = prob_to_american(p)
    print(f"  P = {p:.2f} → Decimal: {decimal:.2f} → American: {american:+.0f}")

# Example 5: Combinatorics
print("\n=== Combinatorics ===")
# Poker: 5 cards from 52
from math import comb
total_hands = comb(52, 5)
pair_hands = comb(4, 2) * comb(48, 3) * comb(13, 1) * comb(12, 2)
p_pair = pair_hands / total_hands
print(f"  Total 5-card hands: {total_hands:,}")
print(f"  One pair hands: {pair_hands:,}")
print(f"  P(exactly one pair): {p_pair:.4f}")

# Example 6: Bayes' Theorem
print("\n=== Bayes' Theorem ===")
print("Disease testing:")
p_disease = 0.01  # 1% have disease
p_positive_given_disease = 0.95  # 95% sensitivity
p_positive_given_healthy = 0.05  # 5% false positive rate

p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_healthy * (1 - p_disease))
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"  P(disease): {p_disease:.4f}")
print(f"  P(positive | disease): {p_positive_given_disease:.4f}")
print(f"  P(positive | healthy): {p_positive_given_healthy:.4f}")
print(f"  P(disease | positive): {p_disease_given_positive:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Dice outcomes
outcomes_1d = list(range(1, 7))
probs_1d = [1/6] * 6
axes[0, 0].bar(outcomes_1d, probs_1d, color='steelblue', alpha=0.7)
axes[0, 0].set_title('Single Die Probability')
axes[0, 0].set_xlabel('Outcome')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].set_ylim([0, 0.2])

# Plot 2: Two dice sum distribution
sums = [i+j for i in range(1,7) for j in range(1,7)]
sum_counts = {s: sums.count(s) for s in range(2, 13)}
axes[0, 1].bar(sum_counts.keys(), [v/36 for v in sum_counts.values()], 
               color='coral', alpha=0.7)
axes[0, 1].set_title('Sum of Two Dice')
axes[0, 1].set_xlabel('Sum')
axes[0, 1].set_ylabel('Probability')

# Plot 3: Odds conversion
probs = np.linspace(0.01, 0.99, 100)
decimal_odds = 1 / probs
axes[1, 0].plot(probs, decimal_odds, linewidth=2)
axes[1, 0].set_title('Probability vs Decimal Odds')
axes[1, 0].set_xlabel('Probability')
axes[1, 0].set_ylabel('Decimal Odds')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Bayes visualization
categories = ['Disease\n(Positive Test)', 'False Positive\n(Healthy)']
probabilities_bayes = [p_positive_given_disease * p_disease,
                       p_positive_given_healthy * (1 - p_disease)]
colors = ['red', 'orange']
axes[1, 1].bar(categories, probabilities_bayes, color=colors, alpha=0.7)
axes[1, 1].set_title("Bayes' Theorem: Disease Test Result")
axes[1, 1].set_ylabel('P(Positive)')
total_positive = sum(probabilities_bayes)
for i, (cat, prob) in enumerate(zip(categories, probabilities_bayes)):
    pct = prob / total_positive * 100
    axes[1, 1].text(i, prob/2, f'{pct:.1f}%\nof positives', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
