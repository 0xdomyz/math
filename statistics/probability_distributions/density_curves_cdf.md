# Density Curves & CDF

## 1.1 Concept Skeleton
**Definition:** PDF/PMF describes probability at each value; CDF gives cumulative probability up to value  
**Purpose:** Fully characterize probability distributions, compute probabilities, visualize distributions  
**Prerequisites:** Probability basics, integration (for continuous), summation (for discrete)

## 1.2 Comparative Framing
| Function | PDF (Probability Density Function) | PMF (Probability Mass Function) | CDF (Cumulative Distribution Function) |
|----------|----------------------------------|--------------------------------|---------------------------------------|
| **Type** | Continuous | Discrete | Both |
| **Output** | Density (not probability) | Probability P(X=x) | Probability P(X≤x) |
| **Properties** | ∫f(x)dx = 1, f(x) ≥ 0 | Σp(x) = 1, p(x) ≥ 0 | Non-decreasing, F(-∞)=0, F(∞)=1 |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Normal(0,1): PDF peaks at 0, CDF is S-curve with F(0)=0.5 (50% below mean)

**Failure Case:**  
P(X=x) from PDF directly: PDF values aren't probabilities; must integrate P(a<X<b) = ∫f(x)dx

**Edge Case:**  
Uniform[0,1]: PDF constant (f=1), CDF linear (F(x)=x); simplest continuous distribution

## 1.4 Layer Breakdown
```
Probability Functions Framework:
├─ Probability Mass Function (PMF) - Discrete:
│   ├─ Definition: p(x) = P(X = x)
│   ├─ Properties:
│   │   ├─ 0 ≤ p(x) ≤ 1 for all x
│   │   ├─ Σ_all_x p(x) = 1
│   │   └─ Bar chart representation
│   ├─ Examples:
│   │   ├─ Binomial: p(k) = C(n,k)p^k(1-p)^(n-k)
│   │   └─ Poisson: p(k) = (λ^k e^-λ)/k!
│   └─ Use: P(X in A) = Σ_{x in A} p(x)
├─ Probability Density Function (PDF) - Continuous:
│   ├─ Definition: f(x) such that P(a ≤ X ≤ b) = ∫_a^b f(x)dx
│   ├─ Properties:
│   │   ├─ f(x) ≥ 0 for all x (but can be >1)
│   │   ├─ ∫_{-∞}^{∞} f(x)dx = 1
│   │   ├─ P(X = x) = 0 for any single x
│   │   └─ Curve representation
│   ├─ Examples:
│   │   ├─ Normal: f(x) = (1/√(2πσ²))e^(-(x-μ)²/(2σ²))
│   │   └─ Exponential: f(x) = λe^(-λx) for x≥0
│   └─ Use: Area under curve = probability
├─ Cumulative Distribution Function (CDF):
│   ├─ Definition: F(x) = P(X ≤ x)
│   ├─ Discrete: F(x) = Σ_{t≤x} p(t)
│   ├─ Continuous: F(x) = ∫_{-∞}^x f(t)dt
│   ├─ Properties:
│   │   ├─ Non-decreasing: x₁ < x₂ ⟹ F(x₁) ≤ F(x₂)
│   │   ├─ lim_{x→-∞} F(x) = 0
│   │   ├─ lim_{x→∞} F(x) = 1
│   │   ├─ Right-continuous
│   │   └─ P(a < X ≤ b) = F(b) - F(a)
│   └─ Inverse: Quantile function F^(-1)(p) gives x where F(x)=p
├─ Relationships:
│   ├─ PDF from CDF: f(x) = dF(x)/dx
│   ├─ CDF from PDF: F(x) = ∫_{-∞}^x f(t)dt
│   ├─ PMF from CDF: p(x) = F(x) - F(x-)
│   └─ Survival Function: S(x) = 1 - F(x) = P(X > x)
└─ Applications:
    ├─ Probability Calculation: Use CDF for P(X≤x)
    ├─ Percentiles/Quantiles: Inverse CDF
    ├─ Random Number Generation: Inverse transform sampling
    └─ Distribution Comparison: Overlaying CDFs
```

## 1.5 Mini-Project
Visualize and compute with PDF/PMF/CDF:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# === DISCRETE: Binomial Distribution ===
n_trials = 20
p_success = 0.3
binom = stats.binom(n_trials, p_success)

x_discrete = np.arange(0, n_trials + 1)
pmf = binom.pmf(x_discrete)
cdf = binom.cdf(x_discrete)

# 1. PMF (Bar chart)
axes[0, 0].bar(x_discrete, pmf, alpha=0.7, edgecolor='black')
axes[0, 0].set_title(f'PMF: Binomial(n={n_trials}, p={p_success})')
axes[0, 0].set_xlabel('k (Number of Successes)')
axes[0, 0].set_ylabel('P(X = k)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. CDF (Step function)
axes[0, 1].step(x_discrete, cdf, where='post', linewidth=2)
axes[0, 1].scatter(x_discrete, cdf, color='red', zorder=5)
axes[0, 1].set_title(f'CDF: Binomial(n={n_trials}, p={p_success})')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('P(X ≤ k)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Probability calculation example
k1, k2 = 5, 10
prob_range = binom.cdf(k2) - binom.cdf(k1-1)
axes[0, 2].bar(x_discrete, pmf, alpha=0.3, edgecolor='black')
mask = (x_discrete >= k1) & (x_discrete <= k2)
axes[0, 2].bar(x_discrete[mask], pmf[mask], alpha=0.7, color='red', 
               edgecolor='black', label=f'P({k1}≤X≤{k2}) = {prob_range:.3f}')
axes[0, 2].set_title('Probability from PMF')
axes[0, 2].set_xlabel('k')
axes[0, 2].set_ylabel('P(X = k)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')

# === CONTINUOUS: Normal Distribution ===
mu, sigma = 0, 1
normal = stats.norm(mu, sigma)

x_continuous = np.linspace(-4, 4, 1000)
pdf = normal.pdf(x_continuous)
cdf = normal.cdf(x_continuous)

# 4. PDF (Density curve)
axes[1, 0].plot(x_continuous, pdf, 'b-', linewidth=2)
axes[1, 0].fill_between(x_continuous, 0, pdf, alpha=0.3)
axes[1, 0].set_title(f'PDF: Normal(μ={mu}, σ={sigma})')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('f(x) (Density)')
axes[1, 0].grid(True, alpha=0.3)

# 5. CDF (S-curve)
axes[1, 1].plot(x_continuous, cdf, 'r-', linewidth=2)
axes[1, 1].axhline(0.5, color='g', linestyle='--', alpha=0.5, label='F(μ) = 0.5')
axes[1, 1].axvline(mu, color='g', linestyle='--', alpha=0.5)
axes[1, 1].set_title(f'CDF: Normal(μ={mu}, σ={sigma})')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('F(x) = P(X ≤ x)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Probability as area under PDF
a, b = -1, 1
x_shaded = x_continuous[(x_continuous >= a) & (x_continuous <= b)]
prob_area = normal.cdf(b) - normal.cdf(a)

axes[1, 2].plot(x_continuous, pdf, 'b-', linewidth=2)
axes[1, 2].fill_between(x_shaded, 0, normal.pdf(x_shaded), 
                        alpha=0.7, color='red', 
                        label=f'P({a}≤X≤{b}) = {prob_area:.3f}')
axes[1, 2].axvline(a, color='k', linestyle='--', alpha=0.5)
axes[1, 2].axvline(b, color='k', linestyle='--', alpha=0.5)
axes[1, 2].set_title('Probability as Area Under PDF')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('f(x)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# === COMPARISON: Multiple Distributions ===
# 7. PDF comparison
x_range = np.linspace(-5, 10, 1000)
axes[2, 0].plot(x_range, stats.norm(0, 1).pdf(x_range), label='N(0,1)', linewidth=2)
axes[2, 0].plot(x_range, stats.norm(2, 1.5).pdf(x_range), label='N(2,1.5)', linewidth=2)
axes[2, 0].plot(x_range, stats.expon(scale=1).pdf(x_range), label='Exp(1)', linewidth=2)
axes[2, 0].set_title('PDF Comparison')
axes[2, 0].set_xlabel('x')
axes[2, 0].set_ylabel('f(x)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 8. CDF comparison
axes[2, 1].plot(x_range, stats.norm(0, 1).cdf(x_range), label='N(0,1)', linewidth=2)
axes[2, 1].plot(x_range, stats.norm(2, 1.5).cdf(x_range), label='N(2,1.5)', linewidth=2)
axes[2, 1].plot(x_range, stats.expon(scale=1).cdf(x_range), label='Exp(1)', linewidth=2)
axes[2, 1].set_title('CDF Comparison')
axes[2, 1].set_xlabel('x')
axes[2, 1].set_ylabel('F(x)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 9. Quantile function (Inverse CDF)
p_range = np.linspace(0.01, 0.99, 100)
quantiles = normal.ppf(p_range)

axes[2, 2].plot(p_range, quantiles, 'purple', linewidth=2)
for p in [0.25, 0.5, 0.75]:
    q = normal.ppf(p)
    axes[2, 2].plot(p, q, 'ro', markersize=8)
    axes[2, 2].text(p, q, f'  {p*100:.0f}%: {q:.2f}', fontsize=9)
axes[2, 2].set_title('Quantile Function (Inverse CDF)')
axes[2, 2].set_xlabel('Probability p')
axes[2, 2].set_ylabel('x where F(x) = p')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Computational examples
print("=== Discrete Distribution (Binomial) ===")
print(f"Distribution: Binomial(n={n_trials}, p={p_success})")
print(f"\nPMF Examples:")
print(f"  P(X = 5) = {binom.pmf(5):.4f}")
print(f"  P(X = 10) = {binom.pmf(10):.4f}")
print(f"\nCDF Examples:")
print(f"  P(X ≤ 5) = {binom.cdf(5):.4f}")
print(f"  P(X ≤ 10) = {binom.cdf(10):.4f}")
print(f"\nProbability Calculations:")
print(f"  P(5 ≤ X ≤ 10) = CDF(10) - CDF(4) = {binom.cdf(10) - binom.cdf(4):.4f}")
print(f"  P(X > 8) = 1 - CDF(8) = {1 - binom.cdf(8):.4f}")

print("\n=== Continuous Distribution (Normal) ===")
print(f"Distribution: Normal(μ={mu}, σ={sigma})")
print(f"\nPDF Examples (Density, not probability!):")
print(f"  f(0) = {normal.pdf(0):.4f}")
print(f"  f(1) = {normal.pdf(1):.4f}")
print(f"\nCDF Examples:")
print(f"  P(X ≤ 0) = {normal.cdf(0):.4f}")
print(f"  P(X ≤ 1) = {normal.cdf(1):.4f}")
print(f"\nProbability Calculations:")
print(f"  P(-1 ≤ X ≤ 1) = CDF(1) - CDF(-1) = {normal.cdf(1) - normal.cdf(-1):.4f}")
print(f"  P(X > 1.96) = 1 - CDF(1.96) = {1 - normal.cdf(1.96):.4f}")
print(f"\nQuantiles (Percentiles):")
print(f"  25th percentile (Q1): {normal.ppf(0.25):.4f}")
print(f"  50th percentile (Median): {normal.ppf(0.5):.4f}")
print(f"  75th percentile (Q3): {normal.ppf(0.75):.4f}")
print(f"  95th percentile: {normal.ppf(0.95):.4f}")

# Relationship demonstration
print("\n=== Relationship: PDF and CDF ===")
x_test = 0.5
# CDF from PDF (numerical integration)
cdf_from_pdf = np.trapz(normal.pdf(x_continuous[x_continuous <= x_test]), 
                        x_continuous[x_continuous <= x_test])
print(f"At x = {x_test}:")
print(f"  CDF(x) directly: {normal.cdf(x_test):.4f}")
print(f"  ∫_{-∞}^x PDF(t)dt: {cdf_from_pdf:.4f}")

# PDF from CDF (numerical derivative)
h = 0.001
pdf_from_cdf = (normal.cdf(x_test + h) - normal.cdf(x_test - h)) / (2 * h)
print(f"\n  PDF(x) directly: {normal.pdf(x_test):.4f}")
print(f"  dCDF(x)/dx: {pdf_from_cdf:.4f}")
```

## 1.6 Challenge Round
When are PDF/PMF/CDF the wrong tools?
- **Need exact probability from PDF**: Can't get P(X=x) from PDF alone; must integrate interval
- **Comparing distributions visually**: Overlapping PDFs hard to compare; use CDF or Q-Q plots
- **Multivariate distributions**: Need joint PDF/CDF (higher-dimensional)
- **Empirical data**: Use empirical CDF (ECDF) or kernel density estimation instead
- **Non-standard distributions**: May lack closed-form PDF/CDF; use numerical methods

## 1.7 Key References
- [Khan Academy - Density Curves](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/density-curve)
- [Wikipedia - Probability Density Function](https://en.wikipedia.org/wiki/Probability_density_function)
- [Wikipedia - Cumulative Distribution Function](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
- Thinking: PDF/PMF describes shape; CDF gives cumulative probability; PDF values aren't probabilities (must integrate); CDF always between 0 and 1

---
**Status:** Fundamental distribution description | **Complements:** All Probability Distributions, Quantiles, Random Variables
