# Kernel Density Estimation (KDE)

## 1. Concept Skeleton
**Definition:** Non-parametric method estimating probability density function by placing smooth kernels on each data point and summing  
**Purpose:** Visualize distribution shape, estimate densities without assuming parametric form (e.g., normal)  
**Prerequisites:** Probability distributions, calculus (integration), understanding of histograms

## 2. Comparative Framing
| Method | KDE | Histogram | Parametric Fit |
|--------|-----|-----------|---------------|
| **Smoothness** | Continuous curve | Discrete bins | Smooth (if correct) |
| **Parameters** | Bandwidth h | Bin width/count | Distribution-specific |
| **Flexibility** | Adapts to any shape | Sensitive to bin choice | Constrained to family |
| **Computation** | O(n) per eval point | O(n) once | Fast after fitting |

## 3. Examples + Counterexamples

**Simple Example:**  
Bimodal data (two peaks): Histogram shows bins, KDE reveals smooth double-peaked density

**Failure Case:**  
Sparse data with large bandwidth: Over-smooths, hides real structure, creates false unimodal appearance

**Edge Case:**  
Bounded support (e.g., [0,∞)): Standard KDE places mass below zero → use boundary-corrected kernels

## 4. Layer Breakdown
```
KDE Components:
├─ Data Points: {x₁, x₂, ..., xₙ}
├─ Kernel Function K(u):
│   ├─ Gaussian: (1/√2π) exp(-u²/2) [most common]
│   ├─ Epanechnikov: (3/4)(1-u²) if |u|<1 [optimal MSE]
│   ├─ Uniform: 1/2 if |u|<1 [boxcar]
│   └─ Properties: ∫K(u)du=1, symmetric, typically centered at 0
├─ Bandwidth h: Controls smoothness
│   ├─ Small h: Wiggly, overfits, high variance
│   ├─ Large h: Over-smooths, underfits, high bias
│   └─ Selection: Scott's rule h = n^(-1/5)×σ, cross-validation
├─ Density Estimate:
│   f̂(x) = (1/nh) × Σᵢ K((x - xᵢ)/h)
└─ Output: Smooth continuous PDF estimate
```

**Interaction:** Each data point contributes kernel → Sum scaled by bandwidth → Normalized density

## 5. Mini-Project
Implement and visualize KDE with different bandwidths:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

# Generate mixture data
np.random.seed(42)
n1, n2 = 100, 80
data = np.concatenate([
    np.random.normal(-2, 0.8, n1),
    np.random.normal(3, 1.2, n2)
])

# Manual KDE implementation
def kde_gaussian(data, x_eval, bandwidth):
    """Gaussian kernel density estimate"""
    n = len(data)
    density = np.zeros_like(x_eval)
    
    for xi in data:
        kernel_values = norm.pdf(x_eval, loc=xi, scale=bandwidth)
        density += kernel_values
    
    return density / n

# Evaluation points
x_range = np.linspace(-6, 7, 500)

# Try different bandwidths
bandwidths = [0.1, 0.5, 1.0]
scott_bw = len(data)**(-1/5) * data.std()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Different bandwidths (manual)
for bw in bandwidths:
    density = kde_gaussian(data, x_range, bw)
    axes[0, 0].plot(x_range, density, label=f'h={bw}', linewidth=2)

axes[0, 0].hist(data, bins=30, density=True, alpha=0.3, color='gray', label='Histogram')
axes[0, 0].set_title('KDE with Different Bandwidths')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

# Plot 2: Scott's rule (scipy)
kde_scott = gaussian_kde(data, bw_method='scott')
kde_silverman = gaussian_kde(data, bw_method='silverman')

axes[0, 1].plot(x_range, kde_scott(x_range), label='Scott\'s rule', linewidth=2)
axes[0, 1].plot(x_range, kde_silverman(x_range), label='Silverman\'s rule', linewidth=2)
axes[0, 1].hist(data, bins=30, density=True, alpha=0.3, color='gray')
axes[0, 1].set_title('Automatic Bandwidth Selection')
axes[0, 1].legend()
print(f"Scott's bandwidth: {scott_bw:.3f}")
print(f"Silverman's bandwidth: {kde_silverman.factor * data.std() * len(data)**(-1/5):.3f}")

# Plot 3: Kernel contribution visualization
axes[1, 0].hist(data, bins=30, density=True, alpha=0.3, color='gray')
sample_points = data[:10]  # Show first 10 kernels
for xi in sample_points:
    kernel = norm.pdf(x_range, loc=xi, scale=scott_bw)
    axes[1, 0].plot(x_range, kernel, 'r-', alpha=0.3, linewidth=0.8)

kde_final = gaussian_kde(data, bw_method='scott')
axes[1, 0].plot(x_range, kde_final(x_range), 'b-', linewidth=3, label='Sum (KDE)')
axes[1, 0].set_title('Individual Kernel Contributions')
axes[1, 0].legend()

# Plot 4: Comparison with true distribution
true_density = 0.556 * norm.pdf(x_range, -2, 0.8) + 0.444 * norm.pdf(x_range, 3, 1.2)
axes[1, 1].plot(x_range, true_density, 'g--', linewidth=2, label='True density')
axes[1, 1].plot(x_range, kde_final(x_range), 'b-', linewidth=2, label='KDE estimate')
axes[1, 1].hist(data, bins=30, density=True, alpha=0.3, color='gray')
axes[1, 1].set_title('KDE vs True Density')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Sample from KDE
samples_from_kde = kde_final.resample(size=1000).flatten()
print(f"\nOriginal data mean: {data.mean():.2f}")
print(f"KDE samples mean: {samples_from_kde.mean():.2f}")
```

## 6. Challenge Round
When is KDE the wrong tool?
- High dimensions (>3): Curse of dimensionality, need exponentially more data
- Heavy tails/outliers: Standard kernels underestimate tail density
- Discrete data: KDE assumes continuity, use probability mass functions
- Need exact probabilities: KDE is estimate, parametric model may be better if valid
- Real-time constraints: O(n) evaluation cost per point, consider binning approximations

## 7. Key References
- [KDE Interactive Demo](https://mathisonian.github.io/kde/)
- [scipy.stats.gaussian_kde Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
- [Silverman, Density Estimation for Statistics and Data Analysis](https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203)

---
**Status:** Essential non-parametric visualization tool | **Complements:** Histograms, Parametric Distributions, Bootstrap
