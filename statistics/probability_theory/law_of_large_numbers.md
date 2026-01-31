# Law of Large Numbers

## 4.1 Concept Skeleton
**Definition:** Sample mean converges to population mean as sample size approaches infinity; x̄ₙ → μ as n → ∞  
**Purpose:** Justify using sample statistics to estimate population parameters; foundation for frequentist inference  
**Prerequisites:** Expected value, variance, convergence concepts, basic probability

## 4.2 Comparative Framing
| Theorem | Weak LLN | Strong LLN | Central Limit Theorem | Chebyshev's Inequality |
|---------|----------|-----------|---------------------|---------------------|
| **Statement** | x̄ₙ →ᵖ μ (convergence in probability) | x̄ₙ →ᵃ·ˢ· μ (almost sure convergence) | √n(x̄ₙ-μ) →ᵈ N(0,σ²) | P(\|X-μ\|≥kσ) ≤ 1/k² |
| **Speed** | Not specified | Not specified | √n rate | Bound only |
| **Distribution** | Any (finite variance) | Any (finite mean) | Approximate normal | Any (finite variance) |
| **Use** | Theoretical guarantee | Stronger guarantee | Practical inference | Probability bounds |

## 4.3 Examples + Counterexamples

**Simple Example:**  
Fair coin: After 10 flips → 6 heads (60%). After 10,000 flips → 5,023 heads (50.23%). Converges to p=0.5

**Failure Case:**  
Cauchy distribution has no mean; LLN doesn't apply. Sample mean doesn't converge to any value

**Edge Case:**  
Law of small numbers fallacy: Expecting convergence with tiny samples. n=10 can show large deviations from μ

## 4.4 Layer Breakdown
```
Law of Large Numbers Framework:
├─ Weak Law (WLLN):
│   ├─ For any ε > 0: lim_{n→∞} P(|x̄ₙ - μ| > ε) = 0
│   ├─ Sample mean converges in probability to μ
│   ├─ Guarantees: Can make P(close to μ) arbitrarily high
│   └─ Requirements: Finite mean E[X] = μ, finite variance Var(X)
├─ Strong Law (SLLN):
│   ├─ P(lim_{n→∞} x̄ₙ = μ) = 1
│   ├─ Sample mean converges almost surely to μ
│   ├─ Stronger: Convergence happens with probability 1
│   └─ Requirements: Finite mean E[X] = μ (variance can be infinite)
├─ Convergence Mechanisms:
│   ├─ Variance of x̄ₙ: Var(x̄ₙ) = σ²/n → 0 as n → ∞
│   ├─ Standard error: SE(x̄ₙ) = σ/√n
│   └─ Decreases at rate 1/√n
├─ Practical Implications:
│   ├─ Sample mean is consistent estimator of μ
│   ├─ Larger samples → better estimates
│   ├─ Justifies long-run frequency interpretation
│   └─ Foundation for Monte Carlo simulation
└─ Variants:
    ├─ Bernoulli LLN: Sample proportion → true probability
    ├─ Continuous version: Sample mean of continuous r.v.
    └─ Multivariate: Vector of means converges
```

## 4.5 Mini-Project
Simulate and visualize Law of Large Numbers:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Simulation 1: Fair coin flips
print("=== Fair Coin: Convergence to p=0.5 ===")
n_max = 10000
true_p = 0.5
flips = np.random.choice([0, 1], size=n_max, p=[0.5, 0.5])

# Calculate running mean
cumsum = np.cumsum(flips)
running_mean = cumsum / (np.arange(n_max) + 1)

# Sample sizes to check
check_points = [10, 100, 1000, 10000]
for n in check_points:
    prop = running_mean[n-1]
    print(f"n={n:5d}: proportion = {prop:.4f}, error = {abs(prop-true_p):.4f}")

# Simulation 2: Dice rolls
print("\n=== Dice Rolls: Convergence to μ=3.5 ===")
true_mean = 3.5
rolls = np.random.randint(1, 7, size=n_max)
cumsum_dice = np.cumsum(rolls)
running_mean_dice = cumsum_dice / (np.arange(n_max) + 1)

for n in check_points:
    mean_val = running_mean_dice[n-1]
    print(f"n={n:5d}: mean = {mean_val:.4f}, error = {abs(mean_val-true_mean):.4f}")

# Simulation 3: Different distributions
print("\n=== LLN Across Distributions ===")
distributions = {
    'Normal(5, 2)': {'data': np.random.normal(5, 2, n_max), 'true_mean': 5},
    'Exponential(λ=2)': {'data': np.random.exponential(0.5, n_max), 'true_mean': 0.5},
    'Uniform(0, 10)': {'data': np.random.uniform(0, 10, n_max), 'true_mean': 5},
    'Poisson(λ=7)': {'data': np.random.poisson(7, n_max), 'true_mean': 7}
}

running_means = {}
for dist_name, dist_info in distributions.items():
    data = dist_info['data']
    cumsum_dist = np.cumsum(data)
    running_means[dist_name] = cumsum_dist / (np.arange(n_max) + 1)
    final_mean = running_means[dist_name][-1]
    true_mean = dist_info['true_mean']
    print(f"{dist_name:20s}: final mean = {final_mean:.4f}, true = {true_mean:.2f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Coin flip convergence
ax1 = axes[0, 0]
n_range = np.arange(1, n_max+1)
ax1.plot(n_range, running_mean, linewidth=1.5, alpha=0.8)
ax1.axhline(true_p, color='r', linestyle='--', linewidth=2, label=f'True p={true_p}')
ax1.set_xlabel('Number of Flips')
ax1.set_ylabel('Proportion of Heads')
ax1.set_title('Law of Large Numbers: Coin Flips')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Multiple trials showing variability
ax2 = axes[0, 1]
n_trials = 20
for trial in range(n_trials):
    flips_trial = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])
    cumsum_trial = np.cumsum(flips_trial)
    running_mean_trial = cumsum_trial / (np.arange(1000) + 1)
    ax2.plot(range(1, 1001), running_mean_trial, alpha=0.3, linewidth=1)

ax2.axhline(0.5, color='r', linestyle='--', linewidth=2, label='True p=0.5')
ax2.set_xlabel('Number of Flips')
ax2.set_ylabel('Proportion of Heads')
ax2.set_title(f'LLN: {n_trials} Independent Trials\n(All converge to 0.5)')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Multiple distributions
ax3 = axes[0, 2]
colors = ['blue', 'green', 'orange', 'purple']
for (dist_name, _), color in zip(distributions.items(), colors):
    true_mean = distributions[dist_name]['true_mean']
    ax3.plot(n_range, running_means[dist_name], label=dist_name, 
             linewidth=1.5, alpha=0.7, color=color)
    ax3.axhline(true_mean, color=color, linestyle='--', alpha=0.5, linewidth=1)

ax3.set_xlabel('Sample Size')
ax3.set_ylabel('Sample Mean')
ax3.set_title('LLN Works for Any Distribution\n(with finite mean)')
ax3.set_xscale('log')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Error magnitude vs sample size
ax4 = axes[1, 0]
sample_sizes = np.logspace(1, 4, 50, dtype=int)
errors_theoretical = 2 / np.sqrt(sample_sizes)  # σ/√n for coin (σ=0.5)
errors_empirical = []

for n in sample_sizes:
    # Average error across 100 trials
    trial_errors = []
    for _ in range(100):
        sample = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        sample_mean = np.mean(sample)
        trial_errors.append(abs(sample_mean - 0.5))
    errors_empirical.append(np.mean(trial_errors))

ax4.plot(sample_sizes, errors_theoretical, 'r--', linewidth=2, label='Theoretical SE = σ/√n')
ax4.plot(sample_sizes, errors_empirical, 'b-', linewidth=2, label='Empirical MAE')
ax4.set_xlabel('Sample Size')
ax4.set_ylabel('Mean Absolute Error')
ax4.set_title('Error Decreases at 1/√n Rate')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Probability of deviation (Chebyshev)
ax5 = axes[1, 1]
# P(|x̄ - μ| > ε) ≤ σ²/(n·ε²) (Chebyshev's inequality)
n_values = [10, 50, 100, 500, 1000]
epsilon_range = np.linspace(0.01, 0.5, 100)

for n in n_values:
    # Coin: σ² = 0.25
    prob_bound = 0.25 / (n * epsilon_range**2)
    prob_bound = np.minimum(prob_bound, 1)  # Cap at 1
    ax5.plot(epsilon_range, prob_bound, label=f'n={n}', linewidth=2)

ax5.set_xlabel('Deviation ε')
ax5.set_ylabel('P(|x̄ - μ| > ε)')
ax5.set_title('Chebyshev Bound: Probability of Deviation\n(Upper bound decreases with n)')
ax5.set_yscale('log')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Failure case - Cauchy distribution
ax6 = axes[1, 2]
print("\n=== Cauchy Distribution: LLN Fails ===")
# Cauchy has no defined mean; sample mean doesn't converge
n_cauchy = 10000
cauchy_samples = stats.cauchy.rvs(size=n_cauchy)
cumsum_cauchy = np.cumsum(cauchy_samples)
running_mean_cauchy = cumsum_cauchy / (np.arange(n_cauchy) + 1)

# For comparison, also show normal
normal_samples = np.random.normal(0, 1, n_cauchy)
cumsum_normal = np.cumsum(normal_samples)
running_mean_normal = cumsum_normal / (np.arange(n_cauchy) + 1)

ax6.plot(range(1, n_cauchy+1), running_mean_cauchy, label='Cauchy (no mean)', 
         linewidth=1.5, alpha=0.7)
ax6.plot(range(1, n_cauchy+1), running_mean_normal, label='Normal (μ=0)', 
         linewidth=1.5, alpha=0.7)
ax6.axhline(0, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax6.set_xlabel('Sample Size')
ax6.set_ylabel('Running Mean')
ax6.set_title('LLN Failure: Cauchy Distribution\n(Infinite variance, no convergence)')
ax6.set_xscale('log')
ax6.set_ylim(-5, 5)
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstration: Monte Carlo integration using LLN
print("\n=== Monte Carlo Integration via LLN ===")
# Estimate π by random sampling
# Area of circle: π, Area of square: 4
# π/4 = P(point in circle) → π = 4·(proportion in circle)

n_points = 100000
x = np.random.uniform(-1, 1, n_points)
y = np.random.uniform(-1, 1, n_points)
inside_circle = (x**2 + y**2) <= 1

running_estimates = []
for i in range(1, n_points+1):
    prop_inside = np.sum(inside_circle[:i]) / i
    pi_estimate = 4 * prop_inside
    if i in [100, 1000, 10000, 100000]:
        running_estimates.append((i, pi_estimate))
        print(f"n={i:6d}: π estimate = {pi_estimate:.6f}, error = {abs(pi_estimate - np.pi):.6f}")

print(f"True π = {np.pi:.6f}")
print("LLN ensures convergence to true value as n → ∞")

# Weak vs Strong LLN illustration
print("\n=== Weak vs Strong Law ===")
print("Weak Law: For any ε>0, P(|x̄ₙ - μ| > ε) → 0 as n → ∞")
print("  Meaning: Can make probability of large error arbitrarily small")
print("\nStrong Law: P(lim_{n→∞} x̄ₙ = μ) = 1")
print("  Meaning: Convergence happens with probability 1 (almost sure)")
print("\nIntuition: Strong law says 'will converge', weak says 'likely to be close'")
```

## 4.6 Challenge Round
When does LLN not apply?
- **Infinite mean**: Cauchy distribution; sample mean never stabilizes
- **Heavy-tailed distributions**: Pareto with α≤1 has infinite mean; extreme values dominate
- **Dependent samples**: Time series with strong autocorrelation; effective sample size < n
- **Changing distribution**: Non-stationary processes; parameters shift over time
- **Small samples**: Law requires n→∞; practical samples finite, convergence incomplete

## 4.7 Key References
- [Law of Large Numbers Explained](https://en.wikipedia.org/wiki/Law_of_large_numbers) - Weak vs strong versions, proofs, historical context
- [LLN Interactive Visualization](https://seeing-theory.brown.edu/basic-probability/index.html#section3) - Dynamic simulation showing convergence
- [Monte Carlo Methods](https://en.wikipedia.org/wiki/Monte_Carlo_method) - Applications of LLN in computation and simulation

---
**Status:** Fundamental justification for statistics | **Complements:** Central Limit Theorem, Consistency, Monte Carlo Methods
