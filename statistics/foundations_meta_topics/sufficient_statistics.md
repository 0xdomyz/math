# Sufficient Statistics

## 1.1 Concept Skeleton
**Definition:** Statistic T(X) capturing all information in data X about parameter θ; no information loss  
**Purpose:** Data reduction without losing inferential power, simplify analysis  
**Prerequisites:** Probability distributions, likelihood function, factorization theorem

## 1.2 Comparative Framing
| Statistic | Sufficient | Minimal Sufficient | Complete |
|-----------|-----------|-------------------|----------|
| **Information** | All parameter info | No further reduction | Unique in expectation |
| **Dimension** | May be multi-dimensional | Lowest dimension | N/A |
| **Example (Normal)** | (x̄, s²) | (x̄, s²) | (x̄, s²) |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Bernoulli trials: n=10, data={1,0,1,1,0,1,1,1,0,1}. Sufficient: T=Σxᵢ=7 (sum sufficient, not individual values)

**Failure Case:**  
Sample median for normal data: Not sufficient (loses information about variance and tails)

**Edge Case:**  
Entire dataset always sufficient but defeats purpose of data reduction

## 1.4 Layer Breakdown
```
Sufficient Statistics Framework:
├─ Factorization Theorem:
│   ├─ T(X) sufficient for θ ⟺ L(θ|x) = g(T(x),θ) · h(x)
│   ├─ g: Depends on data only through T(X)
│   └─ h: Independent of θ
├─ Common Examples:
│   ├─ Normal(μ,σ²) unknown μ: T(X) = x̄ sufficient
│   ├─ Normal(μ,σ²) both unknown: T(X) = (x̄, s²) sufficient
│   ├─ Bernoulli(p): T(X) = Σxᵢ sufficient
│   ├─ Poisson(λ): T(X) = Σxᵢ sufficient
│   └─ Exponential(λ): T(X) = Σxᵢ sufficient
├─ Minimal Sufficient:
│   ├─ Definition: Further reduction impossible without information loss
│   ├─ Uniqueness: Essentially unique (up to bijection)
│   └─ Criterion: T(x)/T(y) independent of θ ⟹ minimal sufficient
├─ Complete Statistic:
│   ├─ Definition: E[g(T)] = 0 for all θ ⟹ P(g(T)=0) = 1
│   ├─ Use: Ensures UMVU estimators (Lehmann-Scheffé theorem)
│   └─ Exponential families: Sufficient statistics are complete
└─ Properties:
    ├─ MLE function of sufficient statistic
    ├─ Rao-Blackwell: Conditioning on sufficient stat improves estimators
    └─ Data reduction: n observations → k-dimensional T (k << n)
```

## 1.5 Mini-Project
Verify sufficiency and compare estimators:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate data from Exponential distribution
true_lambda = 2.0
n = 100
data = np.random.exponential(1/true_lambda, n)

# Sufficient statistic for Exponential: T = sum(X) or equivalently mean(X)
T_sufficient = np.sum(data)  # or equivalently n * np.mean(data)

# MLE for exponential: λ̂ = n / sum(X) = 1 / mean(X)
mle_lambda = n / T_sufficient

print(f"True λ: {true_lambda}")
print(f"Sufficient statistic T = Σxᵢ: {T_sufficient:.3f}")
print(f"MLE λ̂ = n/T: {mle_lambda:.3f}")
print(f"Sample mean x̄: {np.mean(data):.3f}")
print(f"1/x̄: {1/np.mean(data):.3f}")

# Demonstrate Rao-Blackwell improvement
# Inefficient estimator: use only first observation
def inefficient_estimator(data):
    """Use only first observation: 1/X₁"""
    return 1 / data[0] if data[0] > 0 else np.inf

# Improved estimator using sufficient statistic
def rao_blackwell_estimator(data):
    """Condition on sufficient statistic: 1/x̄"""
    return 1 / np.mean(data)

# Simulation to compare estimators
n_sims = 1000
inefficient_estimates = []
rb_estimates = []
mle_estimates = []

for _ in range(n_sims):
    sim_data = np.random.exponential(1/true_lambda, n)
    inefficient_estimates.append(inefficient_estimator(sim_data))
    rb_estimates.append(rao_blackwell_estimator(sim_data))
    mle_estimates.append(n / np.sum(sim_data))

inefficient_estimates = np.array(inefficient_estimates)
rb_estimates = np.array(rb_estimates)
mle_estimates = np.array(mle_estimates)

# Remove extreme outliers for visualization
inefficient_estimates = inefficient_estimates[inefficient_estimates < 10]

print(f"\nSimulation Results (n={n_sims}):")
print(f"Inefficient estimator:")
print(f"  Mean: {np.mean(inefficient_estimates):.3f}")
print(f"  Variance: {np.var(inefficient_estimates):.3f}")
print(f"Rao-Blackwell (sufficient stat):")
print(f"  Mean: {np.mean(rb_estimates):.3f}")
print(f"  Variance: {np.var(rb_estimates):.3f}")
print(f"  Variance reduction: {(1 - np.var(rb_estimates)/np.var(inefficient_estimates))*100:.1f}%")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Original data with true distribution
axes[0, 0].hist(data, bins=30, density=True, alpha=0.7, edgecolor='black', label='Data')
x = np.linspace(0, data.max(), 100)
axes[0, 0].plot(x, stats.expon.pdf(x, scale=1/true_lambda), 'r-', 
                linewidth=2, label=f'True Exp({true_lambda})')
axes[0, 0].plot(x, stats.expon.pdf(x, scale=1/mle_lambda), 'g--', 
                linewidth=2, label=f'MLE Exp({mle_lambda:.2f})')
axes[0, 0].legend()
axes[0, 0].set_title('Data with True and MLE Distributions')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')

# 2. Compare estimator distributions
axes[0, 1].hist(inefficient_estimates, bins=50, alpha=0.5, density=True, 
                label='Inefficient (X₁)', edgecolor='black')
axes[0, 1].hist(rb_estimates, bins=50, alpha=0.5, density=True, 
                label='Rao-Blackwell (suff stat)', edgecolor='black')
axes[0, 1].axvline(true_lambda, color='r', linestyle='--', linewidth=2, label='True λ')
axes[0, 1].legend()
axes[0, 1].set_title('Estimator Distributions')
axes[0, 1].set_xlabel('Estimated λ')
axes[0, 1].set_ylabel('Density')

# 3. Convergence with sample size
sample_sizes = np.arange(10, 500, 10)
mse_inefficient = []
mse_rb = []

for n_test in sample_sizes:
    estimates_ineff = []
    estimates_rb = []
    for _ in range(500):
        sim_data = np.random.exponential(1/true_lambda, n_test)
        if sim_data[0] > 0:
            estimates_ineff.append(1/sim_data[0])
        estimates_rb.append(1/np.mean(sim_data))
    
    mse_inefficient.append(np.mean((np.array(estimates_ineff) - true_lambda)**2))
    mse_rb.append(np.mean((np.array(estimates_rb) - true_lambda)**2))

axes[1, 0].plot(sample_sizes, mse_inefficient, 'b-', label='Inefficient', linewidth=2)
axes[1, 0].plot(sample_sizes, mse_rb, 'r-', label='Sufficient Stat', linewidth=2)
axes[1, 0].set_xlabel('Sample Size n')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('Mean Squared Error vs Sample Size')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 4. Factorization demonstration
# Show likelihood factors through sufficient statistic
test_lambdas = np.linspace(0.5, 4, 100)
ll_full = []
ll_sufficient = []

for lam in test_lambdas:
    # Full likelihood
    ll_full.append(np.sum(stats.expon.logpdf(data, scale=1/lam)))
    # Likelihood through sufficient statistic (proportional)
    ll_sufficient.append(n * np.log(lam) - lam * T_sufficient)

# Normalize for comparison
ll_full = np.array(ll_full) - np.max(ll_full)
ll_sufficient = np.array(ll_sufficient) - np.max(ll_sufficient)

axes[1, 1].plot(test_lambdas, ll_full, 'b-', linewidth=2, label='Full Likelihood')
axes[1, 1].plot(test_lambdas, ll_sufficient, 'r--', linewidth=2, label='Via Sufficient Stat')
axes[1, 1].axvline(true_lambda, color='g', linestyle=':', label='True λ')
axes[1, 1].axvline(mle_lambda, color='orange', linestyle=':', label='MLE')
axes[1, 1].set_xlabel('λ')
axes[1, 1].set_ylabel('Normalized Log-Likelihood')
axes[1, 1].set_title('Likelihood Factorization')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 1.6 Challenge Round
When is sufficiency not useful?
- **Non-parametric inference**: No finite-dimensional θ, sufficiency concept breaks down
- **Complex models**: Sufficient statistic may be as complex as original data (no reduction)
- **Robust estimation**: Sufficient stats optimal under model, but model may be wrong
- **Computational ease**: Sometimes easier to compute with raw data than extract sufficient stat
- **Model misspecification**: Sufficiency assumes correct distributional family

## 1.7 Key References
- [Wikipedia - Sufficient Statistic](https://en.wikipedia.org/wiki/Sufficient_statistic)
- [Factorization Theorem](https://en.wikipedia.org/wiki/Sufficient_statistic#Fisher%E2%80%93Neyman_factorization_theorem)
- [Rao-Blackwell Theorem](https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem)
- Thinking: Sufficiency enables dimension reduction without information loss; MLE always function of sufficient statistic; Exponential families have nice sufficient statistics

---
**Status:** Theoretical foundation for optimal inference | **Complements:** Likelihood, MLE, Rao-Blackwell, Completeness
