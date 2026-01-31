"""
Extracted from: statistics_applied_to_gambling.md
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate poker player: claims 55% win rate
true_win_prob = 0.55
n_hands = 10000

# Generate outcomes
outcomes = np.random.binomial(1, true_win_prob, n_hands)
win_count = np.sum(outcomes)
observed_rate = win_count / n_hands

# Statistical test: binomial test against null (50%)
p_value = stats.binom_test(win_count, n_hands, 0.5, alternative='two-sided')

# Confidence interval: Wilson score interval
from scipy.stats import binom
ci_lower = binom.ppf(0.025, n_hands, observed_rate) / n_hands
ci_upper = binom.ppf(0.975, n_hands, observed_rate) / n_hands

# Non-obvious: sample size determines detectability
sample_sizes = np.logspace(1, 5, 50).astype(int)
detectable_edges = []

for n in sample_sizes:
    # Edge detectable if 95% CI excludes 0.50
    se = np.sqrt(true_win_prob * (1 - true_win_prob) / n)
    ci_width = 1.96 * se
    # For detection, need observed > 0.5 + ci_width (approximately)
    min_observed = 0.5 + ci_width
    detectable = (true_win_prob > min_observed)
    detectable_edges.append(1 if detectable else 0)

# Convergence of rolling average
window = 100
rolling_avg = np.convolve(outcomes, np.ones(window)/window, mode='valid')
cumulative_rate = np.cumsum(outcomes) / np.arange(1, n_hands+1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative win rate convergence
axes[0, 0].plot(cumulative_rate, label='Cumulative Rate', linewidth=1.5)
axes[0, 0].axhline(true_win_prob, color='red', linestyle='--', label='True p=0.55')
axes[0, 0].axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Null p=0.50')
axes[0, 0].fill_between(range(n_hands), 0.5-0.05, 0.5+0.05, alpha=0.2, color='gray', label='95% CI if no edge')
axes[0, 0].set_title('Cumulative Win Rate Convergence')
axes[0, 0].set_xlabel('Hands')
axes[0, 0].set_ylabel('Win Rate')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Rolling window average
axes[0, 1].plot(rolling_avg, label='Rolling Avg (100 hands)', alpha=0.7)
axes[0, 1].axhline(true_win_prob, color='red', linestyle='--', label='True p=0.55')
axes[0, 1].axhline(0.5, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Rolling 100-Hand Win Rate')
axes[0, 1].set_xlabel('Hands')
axes[0, 1].set_ylabel('Win Rate')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Confidence intervals narrow with sample size
axes[1, 0].semilogx(sample_sizes, [0.55]*len(sample_sizes), 'r-', linewidth=2, label='True p=0.55')
ci_widths = [1.96*np.sqrt(0.55*0.45/n) for n in sample_sizes]
axes[1, 0].fill_between(sample_sizes, 0.55-np.array(ci_widths), 0.55+np.array(ci_widths), 
                        alpha=0.3, color='green', label='95% CI')
axes[1, 0].axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Null (p=0.50)')
axes[1, 0].set_title('Confidence Interval Convergence')
axes[1, 0].set_xlabel('Sample Size (hands)')
axes[1, 0].set_ylabel('Win Rate')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Statistical power vs sample size for edge detection
axes[1, 1].semilogx(sample_sizes, detectable_edges, 'b-', linewidth=2)
axes[1, 1].set_title('Statistical Power (95% CI excludes 0.50)')
axes[1, 1].set_xlabel('Sample Size (hands)')
axes[1, 1].set_ylabel('Edge Detectable')
axes[1, 1].set_ylim(-0.1, 1.1)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Observed win rate: {observed_rate:.4f} ({win_count}/{n_hands})")
print(f"Binomial test p-value: {p_value:.6f}")
print(f"Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"\nInterpretation: {'-%.2f%% edge cannot be ruled out' % ((0.5-ci_upper)*100) if ci_upper < 0.5 else '+%.2f%% edge with 95%% confidence' % ((ci_lower-0.5)*100)}")
