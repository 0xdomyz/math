# Jackknife Resampling

## 1. Concept Skeleton
**Definition:** Resampling technique systematically omitting one observation at a time to estimate bias and variance of statistic  
**Purpose:** Bias reduction, variance estimation, influence diagnostics for individual observations  
**Prerequisites:** Basic statistics, sampling distributions, understanding of estimators

## 2. Comparative Framing
| Method | Jackknife | Bootstrap | Delta Method |
|--------|-----------|-----------|--------------|
| **Samples Created** | n (leave-one-out) | 1000+ (resample) | 0 (analytic) |
| **Primary Use** | Bias estimation | Confidence intervals | Quick approximation |
| **Computational Cost** | Low (n iterations) | High (B iterations) | Instant |
| **Robustness** | Sensitive to outliers | Handles non-smooth | Requires differentiability |

## 3. Examples + Counterexamples

**Simple Example:**  
Estimate bias of sample variance: S² slightly underestimates σ², jackknife quantifies this bias

**Failure Case:**  
Median of small sample: Non-smooth statistic, jackknife gives poor variance estimate (use bootstrap)

**Edge Case:**  
Influential outlier: Jackknife identifies observation with disproportionate impact when excluded

## 4. Layer Breakdown
```
Jackknife Process:
├─ Original Sample: {x₁, x₂, ..., xₙ}
├─ Leave-One-Out Samples:
│   ├─ J₁ = {x₂, x₃, ..., xₙ} (omit x₁)
│   ├─ J₂ = {x₁, x₃, ..., xₙ} (omit x₂)
│   └─ Jₙ = {x₁, x₂, ..., xₙ₋₁} (omit xₙ)
├─ Compute Statistics: θ̂₍ᵢ₎ on each Jᵢ
├─ Jackknife Estimates:
│   ├─ Bias: (n-1) × (mean(θ̂₍ᵢ₎) - θ̂)
│   ├─ Variance: [(n-1)/n] × Σ(θ̂₍ᵢ₎ - mean(θ̂₍ᵢ₎))²
│   └─ Pseudo-values: n×θ̂ - (n-1)×θ̂₍ᵢ₎
└─ Bias-Corrected Estimator: θ̂_BC = n×θ̂ - (n-1)×mean(θ̂₍ᵢ₎)
```

**Interaction:** Systematic exclusion → Influence quantification → Bias/variance decomposition

## 5. Mini-Project
Jackknife variance estimation for sample skewness:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate skewed data
np.random.seed(42)
n = 30
data = np.random.exponential(scale=2, size=n)

# Original statistic
theta_hat = skew(data)
print(f"Sample skewness: {theta_hat:.3f}")

# Jackknife resampling
n_samples = len(data)
theta_jackknife = np.zeros(n_samples)

for i in range(n_samples):
    # Leave out observation i
    jackknife_sample = np.delete(data, i)
    theta_jackknife[i] = skew(jackknife_sample)

# Jackknife estimates
theta_jack_mean = theta_jackknife.mean()
bias_jack = (n_samples - 1) * (theta_jack_mean - theta_hat)
var_jack = ((n_samples - 1) / n_samples) * np.sum((theta_jackknife - theta_jack_mean)**2)
se_jack = np.sqrt(var_jack)

# Bias-corrected estimate
theta_bc = n_samples * theta_hat - (n_samples - 1) * theta_jack_mean

print(f"\nJackknife Results:")
print(f"Mean of jackknife estimates: {theta_jack_mean:.3f}")
print(f"Bias: {bias_jack:.4f}")
print(f"Standard Error: {se_jack:.3f}")
print(f"Bias-corrected estimate: {theta_bc:.3f}")

# Pseudo-values
pseudo_values = n_samples * theta_hat - (n_samples - 1) * theta_jackknife

# Confidence interval from pseudo-values
ci_lower = theta_bc - 1.96 * se_jack
ci_upper = theta_bc + 1.96 * se_jack
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original data histogram
axes[0].hist(data, bins=15, edgecolor='black', alpha=0.7)
axes[0].set_title(f'Original Data (n={n})')
axes[0].set_xlabel('Value')
axes[0].axvline(data.mean(), color='r', linestyle='--', label='Mean')
axes[0].legend()

# Jackknife estimates distribution
axes[1].hist(theta_jackknife, bins=15, edgecolor='black', alpha=0.7)
axes[1].axvline(theta_hat, color='r', linewidth=2, label='Original')
axes[1].axvline(theta_jack_mean, color='g', linestyle='--', label='Jack Mean')
axes[1].set_title('Jackknife Estimates Distribution')
axes[1].set_xlabel('Skewness')
axes[1].legend()

# Influence plot
influence = theta_hat - theta_jackknife
axes[2].scatter(range(n_samples), influence, alpha=0.6)
axes[2].axhline(0, color='r', linestyle='--')
axes[2].set_title('Influence of Each Observation')
axes[2].set_xlabel('Observation Index')
axes[2].set_ylabel('Influence (θ̂ - θ̂₍ᵢ₎)')

plt.tight_layout()
plt.show()

# Compare to bootstrap
from scipy.stats import bootstrap
result = bootstrap((data,), skew, n_resamples=10000, random_state=42)
print(f"\nBootstrap 95% CI: [{result.confidence_interval.low:.3f}, {result.confidence_interval.high:.3f}]")
```

## 6. Challenge Round
When is jackknife the wrong tool?
- Non-smooth statistics (median, mode): Jackknife variance unreliable, use bootstrap
- Very large datasets (n>10000): Computational burden, consider subsampling
- Complex sampling designs: Simple jackknife invalid, need specialized methods
- Time series: Temporal dependence broken, use block methods
- Strong outlier influence: Variance estimate unstable, consider robust alternatives

## 7. Key References
- [Jackknife Resampling (Wikipedia)](https://en.wikipedia.org/wiki/Jackknife_resampling)
- [Efron, The Jackknife, the Bootstrap and Other Resampling Plans](https://www.jstor.org/stable/2987009)
- [Cross-Validation vs Jackknife in Model Selection](https://stats.stackexchange.com/questions/tagged/jackknife)

---
**Status:** Classic resampling for bias estimation | **Complements:** Bootstrap, Cross-Validation, Influence Analysis
