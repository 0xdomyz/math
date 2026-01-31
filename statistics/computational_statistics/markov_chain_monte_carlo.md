# Markov Chain Monte Carlo (MCMC)

## 1. Concept Skeleton
**Definition:** Algorithm for sampling from complex probability distributions by constructing a Markov chain with desired distribution as equilibrium  
**Purpose:** Bayesian posterior sampling, parameter estimation when analytical solutions intractable  
**Prerequisites:** Probability theory, Markov chains, Bayesian inference, computational skills

## 2. Comparative Framing
| Method | MCMC | Direct Sampling | Grid Approximation |
|--------|------|-----------------|-------------------|
| **Use Case** | High-dimensional posteriors | Simple distributions | Low-dimensional only |
| **Efficiency** | Explores intelligently | Random exploration | Exhaustive evaluation |
| **Convergence** | Requires burn-in | Immediate | Immediate but costly |
| **Scalability** | Scales to 100+ params | Limited dimensions | Infeasible beyond 3D |

## 3. Examples + Counterexamples

**Simple Example:**  
Estimate mean of bimodal distribution: Chain explores both peaks, samples proportional to density

**Failure Case:**  
Poor proposal distribution causes chain to get stuck in local mode, never exploring full posterior

**Edge Case:**  
Highly correlated parameters create narrow ridges → slow mixing, autocorrelated samples require thinning

## 4. Layer Breakdown
```
MCMC Components:
├─ Markov Chain: Sequence where next state depends only on current
├─ Target Distribution: π(θ) posterior we want to sample from
├─ Proposal Distribution: q(θ'|θ) suggests next candidate
├─ Acceptance Criterion:
│   ├─ Metropolis-Hastings: Accept with prob min(1, π(θ')/π(θ) × q(θ|θ')/q(θ'|θ))
│   └─ Gibbs Sampling: Always accept, sample from conditional p(θᵢ|θ₋ᵢ)
├─ Convergence Diagnostics:
│   ├─ Trace plots: Visual inspection of mixing
│   ├─ R̂ statistic: Compare within/between chain variance
│   └─ Effective Sample Size: Accounts for autocorrelation
└─ Output: Representative samples from posterior distribution
```

**Interaction:** Proposal → Evaluation → Accept/Reject → Update chain → Repeat until convergence

## 5. Mini-Project
Implement Metropolis-Hastings to sample from mixture of normals:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Target: mixture of two normals
def target_pdf(x):
    return 0.3 * norm.pdf(x, -2, 0.5) + 0.7 * norm.pdf(x, 3, 1.0)

# Metropolis-Hastings algorithm
def metropolis_hastings(n_samples, proposal_std=1.0):
    samples = np.zeros(n_samples)
    current = 0.0  # Starting point
    
    for i in range(n_samples):
        # Propose new state
        proposed = current + np.random.normal(0, proposal_std)
        
        # Acceptance ratio (symmetric proposal cancels)
        acceptance_ratio = target_pdf(proposed) / target_pdf(current)
        
        # Accept or reject
        if np.random.rand() < acceptance_ratio:
            current = proposed
        
        samples[i] = current
    
    return samples

# Run MCMC
np.random.seed(42)
n_samples = 10000
burn_in = 1000
samples = metropolis_hastings(n_samples)
samples_post_burnin = samples[burn_in:]

# Diagnostics
print(f"Acceptance rate: {np.mean(np.diff(samples) != 0):.2%}")
print(f"Mean: {samples_post_burnin.mean():.2f}")
print(f"Std: {samples_post_burnin.std():.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Trace plot
axes[0, 0].plot(samples[:1000])
axes[0, 0].set_title('Trace Plot (First 1000)')
axes[0, 0].set_xlabel('Iteration')

# Histogram vs true density
x_range = np.linspace(-5, 6, 500)
axes[0, 1].hist(samples_post_burnin, bins=50, density=True, alpha=0.6, label='MCMC samples')
axes[0, 1].plot(x_range, target_pdf(x_range), 'r-', linewidth=2, label='True density')
axes[0, 1].set_title('Sample Distribution')
axes[0, 1].legend()

# Autocorrelation
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(samples_post_burnin, ax=axes[1, 0])
axes[1, 0].set_title('Autocorrelation')
axes[1, 0].set_xlim(0, 100)

# Running mean
running_mean = np.cumsum(samples) / np.arange(1, n_samples + 1)
axes[1, 1].plot(running_mean)
axes[1, 1].axhline(y=samples_post_burnin.mean(), color='r', linestyle='--')
axes[1, 1].set_title('Running Mean (Convergence)')
axes[1, 1].set_xlabel('Iteration')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is MCMC the wrong tool?
- Simple distributions with analytical solutions (use direct calculation)
- Discrete low-dimensional space (use exact enumeration)
- Real-time applications (too slow, need variational inference)
- Multimodal posteriors with isolated modes (chain may never jump between modes)
- Computational constraints (consider variational Bayes or Laplace approximation)

## 7. Key References
- [MCMC Interactive Visualization](https://chi-feng.github.io/mcmc-demo/)
- [PyMC3 Documentation](https://docs.pymc.io/)
- [Gelman et al., Bayesian Data Analysis (Chapter 11)](http://www.stat.columbia.edu/~gelman/book/)

---
**Status:** Core computational Bayesian method | **Complements:** Bayesian Inference, Bootstrapping, Maximum Likelihood
