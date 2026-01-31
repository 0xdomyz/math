# Probability Distributions

## 2.1 Concept Skeleton
**Definition:** Mathematical function describing likelihood of outcomes in a random experiment  
**Purpose:** Model real-world phenomena, calculate probabilities, estimate parameters  
**Prerequisites:** Probability basics, functions, calculus

## 2.2 Comparative Framing
| Distribution | Use Case | Shape | Parameters | Domain |
|--------------|----------|-------|-----------|--------|
| **Normal** | Natural measurements, errors | Bell curve (symmetric) | μ, σ | (-∞, ∞) |
| **Binomial** | Repeated binary trials | Discrete, peaked | n, p | {0,1,...,n} |
| **Exponential** | Time until next event | Right-skewed decay | λ | (0, ∞) |
| **Poisson** | Count of events in time | Discrete, peaked | λ | {0,1,2,...} |
| **Uniform** | Equal probability outcomes | Flat | a, b | [a, b] |
| **Chi-square** | Variance of normal data | Right-skewed | k | (0, ∞) |

## 2.3 Examples + Counterexamples

**Normal Distribution Example:**  
Heights in population: μ=170cm, σ=10cm → most people 160-180cm

**Failure Case:**  
Assuming normality when data is right-skewed (income, wait times). Model predictions fail at extremes.

**Edge Case:**  
Central Limit Theorem: Sample means are approximately normal EVEN if population isn't, if n is large enough

## 2.4 Layer Breakdown
```
Distribution Components:
├─ PDF/PMF: Probability density (continuous) or mass (discrete)
├─ CDF: Cumulative probability up to x
├─ Mean (μ): Center of distribution
├─ Variance (σ²): Spread around mean
├─ Skewness: Asymmetry (left/right)
└─ Kurtosis: Tail heaviness
```

## 2.5 Mini-Project
Visualize and compare distributions:
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 1000)
plt.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), label='Normal')
plt.plot(x, 1/(1 + x**2) / np.pi, label='Cauchy')
plt.legend()
plt.title('Normal vs Cauchy (undefined mean!)')
plt.show()

# Note: Cauchy has undefined mean/variance!
```

## 2.6 Challenge Round
When does distribution choice matter least?
- Large samples (CLT makes everything approximately normal)
- When only testing for shift in location (rank tests robust)
- When using methods robust to outliers (median, IQR)

When does it matter MOST?
- Small samples (distribution shape critical)
- Extreme value prediction (tail behavior matters)
- Hypothesis testing about variance/shape

## 2.7 Key References
- [Probability Distributions Guide](https://en.wikipedia.org/wiki/Probability_distribution)
- [Interactive Distribution Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html)
- [When to use which distribution](https://www.khanacademy.org/math/statistics-probability)

---
**Status:** Core concept | **Complements:** Central Limit Theorem, Hypothesis Testing
