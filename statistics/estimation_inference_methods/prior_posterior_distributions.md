# Prior & Posterior Distributions

## 1. Concept Skeleton
**Definition:** Prior is initial belief about parameter before observing data; posterior is updated belief after incorporating observed data through Bayes' theorem  
**Purpose:** Formalize learning mechanism; combine domain knowledge with evidence; quantify uncertainty updates  
**Prerequisites:** Probability distributions, Bayes' theorem, likelihood functions, conditional probability

## 2. Comparative Framing
| Aspect | Prior P(θ) | Likelihood P(Data\|θ) | Posterior P(θ\|Data) |
|--------|-----------|----------------------|----------------------|
| **Timing** | Before data observed | Given observed data | After updating with data |
| **Role** | Initial belief/assumption | Evidence from observations | Final inference |
| **Source** | Expert opinion, historical data, convenience | Data collected in study | Combination of prior + evidence |
| **Updates** | Static; doesn't change | Fixed once data collected | Changes with new data |
| **Conjugacy** | Matched with likelihood for tractability | Determines posterior form | Proportional to prior × likelihood |

## 3. Examples + Counterexamples

**Simple Example:**  
Medical test for rare disease (0.1% prevalence). Prior P(Disease) = 0.001. Test accuracy 95%. Posterior dramatically different from 95% despite high test accuracy—due to low prior.

**Prior Selection Example:**  
Coin flip fairness: Uninformed prior Beta(1,1) updates with 10 heads, 0 tails → Posterior Beta(11,1) → E[p] = 0.917 (less extreme than MLE=1.0)

**Failure Case:**  
Overconfident prior: Set P(θ ∈ [4.9, 5.1]) = 0.99, collect data indicating θ = 10. Posterior heavily influenced by prior stubbornness; slow to learn truth.

**Edge Case:**  
Improper prior (e.g., uniform on ℝ): Technically not a valid probability distribution but results in valid posterior when likelihood is sufficiently informative.

## 4. Layer Breakdown
```
Bayesian Learning Process:

├─ Prior Specification: P(θ)
│  ├─ Informative Prior: Incorporates domain knowledge
│  │   ├─ Expert opinion (subjective)
│  │   ├─ Historical data (objective)
│  │   └─ Previous studies
│  ├─ Weakly Informative Prior: Minimal assumptions
│  │   ├─ Regularization effect (prevents extreme values)
│  │   ├─ Improved stability
│  │   └─ Still allows data to dominate
│  ├─ Uninformative Prior: Minimal knowledge
│  │   ├─ Uniform on parameter space
│  │   ├─ Jeffreys prior (scale-invariant)
│  │   └─ Reference prior
│  └─ Conjugate Prior: Math convenience
│      ├─ Prior ⊗ Likelihood → Posterior (same family)
│      ├─ Beta-Binomial
│      ├─ Normal-Normal
│      └─ Gamma-Poisson
├─ Data Collection & Likelihood: P(Data|θ)
│  ├─ Assume data generation model
│  ├─ Compute likelihood for observed data
│  ├─ Likelihood acts as evidence strength
│  └─ Independent of prior (conceptually)
├─ Bayesian Update: P(θ|Data) ∝ P(Data|θ) × P(θ)
│  ├─ Bayes' Theorem application
│  ├─ Normalizing constant ensures valid probability
│  ├─ Prior strength × likelihood strength = posterior
│  ├─ Large n: Likelihood dominates prior
│  └─ Small n: Prior influences posterior strongly
└─ Inference:
   ├─ Point Estimate: MAP (Maximum A Posteriori)
   ├─ Credible Intervals: Range covering 95% posterior mass
   ├─ Predictive Distribution: P(New Data | Observed Data)
   └─ Sequential Updating: Today's posterior = tomorrow's prior
```

**Interaction:** Prior encodes initial uncertainty; likelihood provides evidence; posterior balances both.

## 5. Mini-Project
Compare prior influence on posterior inference:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import beta as beta_function

# Scenario: Estimating probability of coin bias
# True underlying p = 0.6 (slightly biased toward heads)
# We observe data and use different priors

# Generate observed data
np.random.seed(42)
n_flips = 20
true_p = 0.6
observed_heads = np.random.binomial(n_flips, true_p)
observed_tails = n_flips - observed_heads

print(f"Observed: {observed_heads} heads, {observed_tails} tails")

# Define different priors (all Beta distributions for conjugacy)
priors = {
    'Uninformed (Beta(1,1))': (1, 1),
    'Weak Info (Beta(2,2))': (2, 2),
    'Moderately Info (Beta(10,10))': (10, 10),
    'Strong Fair Coin (Beta(100,100))': (100, 100),
    'Skeptical Bias (Beta(20,5))': (20, 5),
}

# Posterior is Beta(α + heads, β + tails)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

p_range = np.linspace(0, 1, 200)

# Plot 1: Data visualization
axes[0].bar(['Heads', 'Tails'], [observed_heads, observed_tails], color=['blue', 'red'], alpha=0.7)
axes[0].set_title(f'Observed Data (n={n_flips})')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, n_flips + 2)

# Plot posteriors with different priors
for idx, (prior_name, (alpha, beta)) in enumerate(priors.items(), 1):
    alpha_posterior = alpha + observed_heads
    beta_posterior = beta + observed_tails
    
    prior_dist = stats.beta(alpha, beta)
    posterior_dist = stats.beta(alpha_posterior, beta_posterior)
    
    # Plot
    axes[idx].plot(p_range, prior_dist.pdf(p_range), 'g--', linewidth=2, label='Prior', alpha=0.7)
    axes[idx].plot(p_range, posterior_dist.pdf(p_range), 'b-', linewidth=2.5, label='Posterior')
    
    # Mark credible interval
    ci_lower, ci_upper = posterior_dist.ppf([0.025, 0.975])
    axes[idx].axvline(ci_lower, color='r', linestyle=':', alpha=0.7, label='95% CI')
    axes[idx].axvline(ci_upper, color='r', linestyle=':', alpha=0.7)
    
    # Mark true value
    axes[idx].axvline(true_p, color='k', linestyle='-', alpha=0.5, linewidth=1)
    
    posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    axes[idx].axvline(posterior_mean, color='purple', linestyle='--', alpha=0.7, label=f'Mean={posterior_mean:.2f}')
    
    axes[idx].set_title(prior_name)
    axes[idx].set_xlabel('p (probability of heads)')
    axes[idx].set_ylabel('Density')
    axes[idx].legend(fontsize=8)
    axes[idx].set_xlim(0, 1)
    
    print(f"\n{prior_name}:")
    print(f"  Posterior mean: {posterior_mean:.3f}")
    print(f"  95% Credible Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Posterior SD: {posterior_dist.std():.3f}")

plt.tight_layout()
plt.show()

# Demonstrate sequential updating
print("\n" + "="*60)
print("Sequential Updating: Posterior becomes prior for next update")
print("="*60)

# Start with uninformed prior
alpha, beta = 1, 1

observations = [
    (5, 5, "First batch: 5H, 5T"),
    (8, 2, "Second batch: 8H, 2T"),
    (12, 3, "Third batch: 12H, 3T"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for batch_idx, (heads, tails, desc) in enumerate(observations):
    alpha = alpha + heads
    beta = beta + tails
    
    posterior_dist = stats.beta(alpha, beta)
    posterior_mean = alpha / (alpha + beta)
    
    # Plot
    axes[batch_idx].plot(p_range, posterior_dist.pdf(p_range), 'b-', linewidth=2.5)
    axes[batch_idx].fill_between(p_range, posterior_dist.pdf(p_range), alpha=0.3)
    axes[batch_idx].axvline(posterior_mean, color='r', linestyle='--', linewidth=2, label=f'Mean={posterior_mean:.3f}')
    axes[batch_idx].axvline(true_p, color='k', linestyle='-', linewidth=1.5, alpha=0.5, label=f'True={true_p}')
    
    ci_lower, ci_upper = posterior_dist.ppf([0.025, 0.975])
    axes[batch_idx].axvline(ci_lower, color='g', linestyle=':', alpha=0.7)
    axes[batch_idx].axvline(ci_upper, color='g', linestyle=':', alpha=0.7)
    
    axes[batch_idx].set_title(desc)
    axes[batch_idx].set_xlabel('p')
    axes[batch_idx].set_ylabel('Density')
    axes[batch_idx].legend()
    axes[batch_idx].set_xlim(0, 1)
    
    print(f"\n{desc}")
    print(f"  Cumulative: {alpha-1} heads, {beta-1} tails")
    print(f"  Posterior mean: {posterior_mean:.3f}")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Posterior precision: {posterior_dist.std():.3f}")

plt.tight_layout()
plt.show()

# Compare prior strength vs sample size
print("\n" + "="*60)
print("Prior Strength vs Sample Size Impact")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scenario: Different sample sizes, strong prior for bias
strong_prior = (50, 50)  # Strong belief in fairness
heads_observed = 12
tails_observed = 8

sample_sizes = [20, 50, 100, 500, 1000]
posteriors = []

for n in sample_sizes:
    # Scale heads/tails proportionally
    h = int(heads_observed * (n / 20))
    t = int(tails_observed * (n / 20))
    
    alpha_post = strong_prior[0] + h
    beta_post = strong_prior[1] + t
    
    posteriors.append((n, stats.beta(alpha_post, beta_post)))

colors = plt.cm.viridis(np.linspace(0, 1, len(posteriors)))

for (n, dist), color in zip(posteriors, colors):
    axes[0].plot(p_range, dist.pdf(p_range), linewidth=2, label=f'n={n}', color=color)

axes[0].set_title('Strong Prior (Beta(50,50)): Posterior w/ Increasing Sample Size')
axes[0].set_xlabel('p')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Extract posterior means
means = [dist.mean() for _, dist in posteriors]
sds = [dist.std() for _, dist in posteriors]

axes[1].errorbar(sample_sizes, means, yerr=[1.96*s for s in sds], fmt='o-', linewidth=2, markersize=8)
axes[1].axhline(true_p, color='r', linestyle='--', label=f'True p={true_p}')
axes[1].axhline(0.5, color='g', linestyle=':', label='Prior mean (fair coin)')
axes[1].set_xlabel('Total Sample Size')
axes[1].set_ylabel('Posterior Mean')
axes[1].set_title('Posterior Mean Convergence to True Value')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Prior Selection Dilemma:** You have strong historical data suggesting θ = 0.3, but new preliminary study hints θ = 0.7. How do you encode this conflict in the prior? (Mixture prior solution: 0.7 × Beta(10,20) + 0.3 × Beta(20,10))

2. **Sensitivity Analysis:** Run your analysis with 5 different reasonable priors. If all yield same conclusion, robust. If not, prior sensitive—report uncertainty honestly.

3. **Prior Misspecification:** Deliberately use wrong prior (e.g., strong bias when data suggests opposite). Observe how large sample size eventually dominates prior. Quantify how large n must be.

4. **Objective vs Subjective:** Jeffreys prior (scale-invariant) vs expert opinion. When ethical/practical to use which?

5. **Predictions vs Parameters:** Posterior over θ different from posterior predictive for new data. Integrate out θ to get predictive distribution—captures both parameter uncertainty and data variability.

## 7. Key References
- [Gelman et al., Bayesian Data Analysis (Chapter 2-3)](http://www.stat.columbia.edu/~gelman/book/)
- [Prior Elicitation Review (O'Hagan et al.)](https://www.jstor.org/stable/2245144)
- [Conjugate Prior Catalog](https://en.wikipedia.org/wiki/Conjugate_prior)
- [Richard McElreath, Statistical Rethinking (Chapter 2-3)](https://xcelab.net/rm/statistical-rethinking/)

---
**Status:** Core Bayesian inference | **Complements:** Bayesian Inference, Maximum Likelihood, Posterior Predictive Checks
