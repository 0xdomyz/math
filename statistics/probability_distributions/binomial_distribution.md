# Binomial Distribution

## 1. Concept Skeleton
**Definition:** Discrete probability distribution for n independent binary trials with probability p of success each; P(X=k) = C(n,k) × p^k × (1-p)^(n-k)  
**Purpose:** Model count of successes in fixed number of independent trials; foundation for hypothesis tests on proportions; basis for exact and approximate inference  
**Prerequisites:** Probability of independent events, combinations, discrete distributions, binary outcomes

## 2. Comparative Framing
| Property | Binomial | Normal | Poisson | Hypergeometric |
|----------|----------|--------|---------|-----------------|
| **Type** | Discrete | Continuous | Discrete | Discrete |
| **Parameters** | n, p | μ, σ | λ | N, K, n |
| **Support** | 0,1,...,n | (-∞,+∞) | 0,1,2,... | 0 to min(n,K) |
| **Trials** | Fixed (n) | N/A | Infinite/continuous time | Fixed without replacement |
| **Independence** | Yes | N/A | Yes | No (dependent) |
| **Use** | Success/fail trials | Natural phenomena | Rare events | Sampling w/o replacement |
| **Approximation** | Approx normal if np≥5 | Exact | When n large, p small |  N/A |

## 3. Examples + Counterexamples

**Simple Example:**  
Coin flip: n=10 trials, p=0.5 (fair coin). What's P(exactly 7 heads)? P(X=7) = C(10,7) × 0.5^7 × 0.5^3 = 0.117 or 11.7%.

**Perfect Fit:**  
Quality control: Inspect 20 items, each defective with probability 0.05. Count defects. Exact binomial (fixed n, independent trials, constant p).

**Approximation Example:**  
Election poll: n=1000 voters, p=0.48 (support estimate). Use normal approximation (np=480, n(1-p)=520 both >> 5). Approximate confidence interval via normal.

**Poor Fit:**  
Sampling without replacement: Draw 5 cards from deck. Success (Ace) probability changes after each draw. Use hypergeometric, not binomial.

**Edge Case:**  
n=1, p=0.5: Binomial becomes Bernoulli (single coin flip). P(X=0)=0.5, P(X=1)=0.5.

## 4. Layer Breakdown
```
Binomial Distribution:

├─ Probability Mass Function (PMF):
│  ├─ P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
│  ├─ C(n,k) = n! / (k!(n-k)!) = combinations
│  ├─ Valid for k = 0, 1, 2, ..., n
│  └─ Sum over all k: ΣP(X=k) = 1
├─ Parameters and Properties:
│  ├─ n = number of trials (fixed, positive integer)
│  ├─ p = probability of success per trial (0 ≤ p ≤ 1)
│  ├─ q = 1-p = probability of failure
│  ├─ Mean: E[X] = np
│  ├─ Variance: Var(X) = np(1-p) = npq
│  ├─ SD: σ = √(npq)
│  ├─ Skewness: (q-p) / √(npq)
│  │   ├─ Skew = 0 if p = 0.5 (symmetric)
│  │   ├─ Skew > 0 if p < 0.5 (right-skewed)
│  │   └─ Skew < 0 if p > 0.5 (left-skewed)
│  └─ Kurtosis: (1 - 6pq) / npq
├─ Related Distributions:
│  ├─ Bernoulli: Binomial(1, p)
│  ├─ Sum of Binomials: X₁+X₂ ~ Binomial(n₁+n₂, p) if independent
│  ├─ Normal Approximation: If np≥5 and n(1-p)≥5
│  │   └─ X ≈ N(np, npq) for large n
│  ├─ Poisson Approximation: If n large, p small, np moderate
│  │   └─ Binomial(n,p) ≈ Poisson(np)
│  └─ Negative Binomial: Inverse question—how many trials until r successes?
├─ Cumulative Distribution Function (CDF):
│  ├─ F(k) = P(X ≤ k) = Σⱼ₌₀ᵏ P(X=j)
│  ├─ No closed form; requires summation or tables
│  ├─ P(X > k) = 1 - F(k)
│  └─ P(a ≤ X ≤ b) = F(b) - F(a-1)
├─ Inference on p:
│  ├─ Point Estimate: p̂ = (# successes) / n
│  ├─ Exact Confidence Interval: Clopper-Pearson (conservative)
│  ├─ Normal Approximation CI: p̂ ± z* × √(p̂(1-p̂)/n)
│  │   └─ Valid when np̂≥5 and n(1-p̂)≥5
│  ├─ Wilson Score Interval: Often preferred (accounts for uncertainty)
│  └─ Hypothesis Test: H₀: p=p₀ vs H₁: p≠p₀
│      └─ Use z-test (normal approx) or exact binomial test
└─ Simulation and Modeling:
   ├─ Generate random samples: X ~ Binomial(n, p)
   ├─ Estimate parameters from data
   └─ Goodness-of-fit testing
```

**Interaction:** p controls center/skew; n controls spread (larger n → tighter distribution). Product np = mean level.

## 5. Mini-Project
Binomial distribution applications and approximations:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("BINOMIAL DISTRIBUTION EXPLORATION")
print("="*60)

# Scenario 1: Coin flips - effect of n and p
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

scenarios = [
    (10, 0.5, 'n=10, p=0.5'),
    (20, 0.5, 'n=20, p=0.5'),
    (10, 0.3, 'n=10, p=0.3'),
    (20, 0.3, 'n=20, p=0.3'),
    (50, 0.5, 'n=50, p=0.5'),
    (100, 0.1, 'n=100, p=0.1'),
]

for idx, (n, p, title) in enumerate(scenarios):
    ax = axes[idx // 3, idx % 3]
    
    # Calculate PMF
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    
    # Calculate statistics
    mean = n * p
    variance = n * p * (1 - p)
    std = np.sqrt(variance)
    
    # Plot PMF
    ax.bar(x, pmf, alpha=0.6, color='blue', edgecolor='black')
    
    # Overlay normal approximation if valid
    if n*p >= 5 and n*(1-p) >= 5:
        x_cont = np.linspace(0, n, 200)
        normal_approx = norm.pdf(x_cont, mean, std)
        ax.plot(x_cont, normal_approx, 'r-', linewidth=2, label='Normal Approx')
    
    ax.axvline(mean, color='g', linestyle='--', linewidth=2, label=f'Mean={mean:.1f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Number of Successes')
    ax.set_ylabel('Probability')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Scenario 2: Confidence intervals for proportion
print("\n" + "="*60)
print("CONFIDENCE INTERVALS FOR PROPORTION")
print("="*60)

n = 100
successes = 35
p_hat = successes / n

# Method 1: Normal approximation
se_normal = np.sqrt(p_hat * (1 - p_hat) / n)
ci_normal_lower = p_hat - 1.96 * se_normal
ci_normal_upper = p_hat + 1.96 * se_normal

# Method 2: Wilson Score (more accurate)
z_alpha = 1.96
denominator = 1 + z_alpha**2 / n
center = (p_hat + z_alpha**2 / (2*n)) / denominator
margin = z_alpha * np.sqrt((p_hat * (1-p_hat) / n) + (z_alpha**2 / (4*n**2))) / denominator
ci_wilson_lower = center - margin
ci_wilson_upper = center + margin

print(f"\nObserved: {successes} successes in {n} trials")
print(f"Sample proportion: p̂ = {p_hat:.3f}")
print(f"\n1. Normal Approximation 95% CI:")
print(f"   [{ci_normal_lower:.3f}, {ci_normal_upper:.3f}]")
print(f"   (Valid if np̂≥5 and n(1-p̂)≥5: {n*p_hat:.0f}≥5 ✓ and {n*(1-p_hat):.0f}≥5 ✓)")

print(f"\n2. Wilson Score 95% CI (recommended):")
print(f"   [{ci_wilson_lower:.3f}, {ci_wilson_upper:.3f}]")

# Scenario 3: Exact vs Approximate Hypothesis Test
print("\n" + "="*60)
print("HYPOTHESIS TEST: H₀: p=0.5 vs H₁: p≠0.5")
print("="*60)

n_test = 20
successes_test = 15
p0 = 0.5

# Exact binomial test
prob_exact = binom.pmf(successes_test, n_test, p0)
# P-value: sum of probabilities as extreme or more extreme
p_value_exact = 2 * min(
    binom.cdf(successes_test, n_test, p0),  # Left tail
    1 - binom.cdf(successes_test - 1, n_test, p0)  # Right tail
)

# Normal approximation z-test
z_stat = (successes_test - n_test * p0) / np.sqrt(n_test * p0 * (1-p0))
p_value_normal = 2 * (1 - norm.cdf(abs(z_stat)))

print(f"\nObserved: {successes_test} successes in {n_test} trials")
print(f"Test statistic (# successes): {successes_test}")
print(f"\n1. Exact Binomial Test:")
print(f"   P(X = {successes_test} | H₀) = {prob_exact:.4f}")
print(f"   Two-tailed p-value = {p_value_exact:.4f}")
print(f"   Decision: {'Reject H₀' if p_value_exact < 0.05 else 'Fail to reject H₀'} (α=0.05)")

print(f"\n2. Normal Approximation z-test:")
print(f"   z = ({successes_test} - {n_test*p0}) / √({n_test*p0*(1-p0)}) = {z_stat:.3f}")
print(f"   Two-tailed p-value = {p_value_normal:.4f}")
print(f"   Decision: {'Reject H₀' if p_value_normal < 0.05 else 'Fail to reject H₀'} (α=0.05)")

# Scenario 4: Simulation - verification via sampling
print("\n" + "="*60)
print("MONTE CARLO SIMULATION")
print("="*60)

n_sim = 10000
n_trials = 100
p_true = 0.4

# Generate random binomial samples
simulated_successes = np.random.binomial(n_trials, p_true, n_sim)

print(f"\nSimulating {n_sim} samples from Binomial(n={n_trials}, p={p_true}):")
print(f"Theoretical mean: np = {n_trials * p_true}")
print(f"Simulated mean: {simulated_successes.mean():.2f}")
print(f"Theoretical variance: np(1-p) = {n_trials * p_true * (1-p_true)}")
print(f"Simulated variance: {simulated_successes.var():.2f}")
print(f"Theoretical SD: {np.sqrt(n_trials * p_true * (1-p_true)):.2f}")
print(f"Simulated SD: {simulated_successes.std():.2f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Simulated distribution vs theoretical
ax = axes[0]
ax.hist(simulated_successes, bins=30, alpha=0.6, density=True, 
        label='Simulated', color='blue', edgecolor='black')
x_range = np.arange(0, n_trials+1)
theoretical_pmf = binom.pmf(x_range, n_trials, p_true)
ax.plot(x_range, theoretical_pmf, 'ro-', markersize=4, linewidth=2, label='Theoretical')
ax.set_title('Simulated vs Theoretical Distribution')
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Probability')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Confidence interval coverage
ax = axes[1]
coverage_count = 0
ci_width_list = []

for i in range(100):  # Repeat CI calculation
    successes = np.random.binomial(n_trials, p_true, 1)[0]
    p_hat = successes / n_trials
    se = np.sqrt(p_hat * (1-p_hat) / n_trials)
    ci_lower = p_hat - 1.96 * se
    ci_upper = p_hat + 1.96 * se
    
    if ci_lower <= p_true <= ci_upper:
        coverage_count += 1
    ci_width_list.append(ci_upper - ci_lower)

ax.hist(ci_width_list, bins=20, alpha=0.7, color='green', edgecolor='black')
ax.set_title(f'CI Width Distribution\n(Coverage: {coverage_count}% of 100 samples)')
ax.set_xlabel('Confidence Interval Width')
ax.set_ylabel('Frequency')
ax.grid(alpha=0.3)

# Plot 3: Effect of sample size on CI width
ax = axes[2]
sample_sizes = np.arange(20, 501, 20)
avg_ci_widths = []

p_hat_fixed = 0.4
for n in sample_sizes:
    se = np.sqrt(p_hat_fixed * (1-p_hat_fixed) / n)
    width = 2 * 1.96 * se
    avg_ci_widths.append(width)

ax.plot(sample_sizes, avg_ci_widths, 'o-', linewidth=2, markersize=6, color='purple')
ax.set_title('95% CI Width vs Sample Size')
ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('Confidence Interval Width')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Rare Event Approximation:** If n=1000, p=0.001, use Poisson approximation with λ=np. Compare Binomial(1000,0.001) to Poisson(1) PMF.

2. **Sequential Testing:** As you collect data (flips), track running proportion. When does it converge to true p? Law of large numbers visualization.

3. **Power Calculation:** Want 90% power to detect p=0.6 vs p=0.5 with α=0.05. How large must n be? (Use cumulative binomial probabilities or G*Power.)

4. **Underdispersion vs Overdispersion:** Real data sometimes more variable than binomial predicts (overdispersed). Beta-binomial model accounts for this; compare fit.

5. **Multiple Comparisons in Binomial:** Run 100 binomial tests, α=0.05 each. Expected false positives ≈ 5. Observe inflated Type I error; use Bonferroni correction.

## 7. Key References
- [Binomial Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Binomial_distribution)
- [Wilson Score Intervals (Brown et al.)](https://www.jstor.org/stable/2685470)
- [Normal Approximation Conditions](https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation)
- [Exact Binomial Tests (R Documentation)](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html)

---
**Status:** Core discrete distribution | **Complements:** Hypothesis Testing on Proportions, Normal Approximation, Confidence Intervals
