# Central Limit Theorem (CLT)

## 6.1 Concept Skeleton
**Definition:** Distribution of sample means approaches normal distribution as n increases, regardless of population distribution shape  
**Purpose:** Justifies using normal distribution for inference about means  
**Prerequisites:** Distributions, sampling, normal distribution

## 6.2 Comparative Framing
| Aspect | CLT | Law of Large Numbers | Normal Distribution |
|--------|-----|-------------------|-------------------|
| **Concerns** | Distribution of means | Sample mean convergence | Population shape |
| **Implication** | Means ~ Normal | Mean → μ as n→∞ | Predictions about individuals |
| **Sample size** | n ≥ 30 usually works | Needs very large n | Applies to single observations |

## 6.3 Examples + Counterexamples

**CLT Success:**  
Population: Exponential (right-skewed). Draw 100 samples of n=50, plot sample means → Bell curve! (Despite population skewed)

**Where CLT Fails:**  
Heavy-tailed distribution (Cauchy): Sample means don't converge. Undefined mean/variance break CLT assumptions.

**Edge Case:**  
Skewed populations: n ≥ 30 rule of thumb breaks down. With heavy skew, need n > 100 for approximate normality

## 6.4 Layer Breakdown
```
CLT Components:
├─ Original Population: Any distribution, mean μ, std σ
├─ Sampling Process: Draw samples of size n, compute mean x̄
├─ Distribution of x̄: 
│   ├─ Mean: μ (unbiased)
│   ├─ Std Dev: σ/√n (decreases with n!)
│   └─ Shape: Approaches Normal as n→∞
└─ Convergence Rate: Faster for symmetric populations
```

## 6.5 Mini-Project
Visualize CLT:
```python
import numpy as np
import matplotlib.pyplot as plt

# Exponential population (skewed!)
population = np.random.exponential(scale=2, size=100000)

# Draw 1000 samples, compute means
sample_means = []
for i in range(1000):
    sample = np.random.choice(population, size=50)
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=50, density=True, alpha=0.7, label='Sample Means')
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.title('CLT: Sample Means Approach Normal (Despite Exponential Population)')
plt.legend()
plt.show()

# Result: Bell curve despite exponential population!
print(f"Sample means mean: {np.mean(sample_means):.2f}")
print(f"Sample means std: {np.std(sample_means):.2f}")
```

## 6.6 Challenge Round
When does CLT break down?
- Extremely heavy-tailed distributions (Cauchy, some financial data)
- Dependent observations (time series, spatial data)
- Finite population, sampling without replacement significantly

When is CLT not needed?
- Non-parametric tests (don't assume normality)
- Direct probability calculation (don't need normal approx)
- Modern computing (bootstrap instead)

## 6.7 Key References
- [CLT Interactive Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html)
- [Formal Statement & Proof](https://en.wikipedia.org/wiki/Central_limit_theorem)
- [Convergence Rate Discussion](https://stats.stackexchange.com/questions/146920/)

---
**Status:** Fundamental theorem | **Complements:** Probability Distributions, Confidence Intervals, Hypothesis Testing
