# Statistics Applied to Gambling

## 1. Concept Skeleton
**Definition:** Using statistical inference to detect true edges, confidence intervals, and win rate validation  
**Purpose:** Distinguish real skill/advantage from luck-driven variance  
**Prerequisites:** Hypothesis testing, confidence intervals, sample size calculations, statistical power

## 2. Comparative Framing
| Concept | Statistical Test | Sample Size | Decision Rule | Risk Level |
|---------|-----------------|------------|---------------|-----------|
| **Null Hypothesis** | No edge (EV=0) | Varies | Reject if p<0.05 | 5% Type I |
| **Confidence Interval** | 95% range for true EV | 10K+ hands | If CI > 0%, likely edge | Captures uncertainty |
| **Significant Result** | EV significantly ≠ 0 | 50K+ hands | p-value < 0.05 | 5% false positive |
| **Power Analysis** | Probability detect true edge | 100K+ hands | 80-90% detection power | Minimize Type II |

## 3. Examples + Counterexamples

**Example (Detection):**  
Poker player with 55% win rate over 10,000 hands → p < 0.05 → statistically significant edge detected.

**Example (Insufficient):**  
50 hands at 60% win rate → could be luck (95% CI includes 50%) → insufficient data.

**Counterexample (Misinterpretation):**  
100 hands at -2% rate → assume "bad run" without statistical testing → could be true edge.

## 4. Layer Breakdown
```
Statistics Applied to Gambling:
├─ Hypothesis Testing Framework:
│  ├─ H₀ (null): No edge, p = 0.5 (or house edge assumed)
│  ├─ H₁ (alternative): Edge exists, p ≠ 0.5
│  ├─ Test statistic: (observed - expected) / SE
│  ├─ p-value: Probability of observing result by chance
│  └─ Decision: Reject H₀ if p < α (typically 0.05)
├─ Sample Size Calculations:
│  ├─ Rule of thumb: N ≥ (z_α/2 + z_β)² × p(1-p) / e²
│  ├─ Detecting 1% edge: ~25,000 hands (80% power)
│  ├─ Detecting 5% edge: ~1,000 hands (80% power)
│  └─ Smaller edge = exponentially more hands needed
├─ Confidence Intervals:
│  ├─ 95% CI on win rate = observed ± 1.96 × SE
│  ├─ If CI includes 50%, cannot reject no-edge hypothesis
│  ├─ Narrower interval = larger sample or lower variance
│  └─ Interpretation: 95% confident true value in range
├─ Common Statistical Errors:
│  ├─ Type I (false positive): Declare edge when none exists
│  ├─ Type II (false negative): Miss true edge due to insufficient power
│  ├─ P-hacking: Multiple tests until p < 0.05 (fraudulent)
│  ├─ Selection bias: Only report winning games
│  └─ Regression to mean: Extreme results partially revert
├─ Variance vs Signal:
│  ├─ Coefficient of variation: σ / EV determines sample size
│  ├─ High-variance games: Need larger N to isolate signal
│  ├─ Low-variance games: Signal detectable faster
│  └─ Example: Coin flip variance ≈ 0.25; poker variance >> 1
├─ Running Rates & Win Rate Stability:
│  ├─ Calculate rolling win percentage over fixed windows
│  ├─ Convergence to true EV evident in long-run plot
│  ├─ Significant deviations (>3σ) investigate causation
│  └─ Session-to-session comparison reveals consistency
└─ Meta-Analysis:
   ├─ Combine multiple players/games to boost statistical power
   ├─ Aggregate data across variations to detect general principle
   ├─ Account for multiple testing via Bonferroni correction
   └─ Publication bias: Positive results reported, null results buried
```

**Interaction:** Collect data → calculate statistics → test hypothesis → determine confidence in edge → adjust strategy.

## 5. Mini-Project
Statistical validation of claimed edge:
```python
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
```

## 6. Challenge Round
Why do most gamblers avoid rigorous statistical tracking?
- Confirmation bias: Unconsciously remember wins, forget losses
- Selection bias: Stop tracking when "running bad" or after big losses
- Emotional resistance: Data may confirm you lack true edge
- Computational burden: Tracking, calculating, interpreting data is tedious
- Painful truth: Statistics often show negative EV that contradicts self-image

## 7. Key References
- [Hypothesis Testing (Wikipedia)](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
- [Sample Size Calculation](https://en.wikipedia.org/wiki/Sample_size_determination)
- [Poker Statistics & Variance](https://www.twoplustwo.com/books/)

---
**Status:** Validation methodology | **Complements:** Expected Value, Hypothesis Testing, Data Analysis
