# Hypothesis Testing

## 1.1 Concept Skeleton
**Definition:** Testing whether observed data provides sufficient evidence to reject a null hypothesis (H₀)  
**Purpose:** Make probabilistic decisions about population parameters from sample data  
**Prerequisites:** Normal distributions, sampling, statistical inference

## 1.2 Comparative Framing
| Concept | vs. Confidence Intervals | vs. Bayesian | vs. Effect Size |
|---------|------------------------|-------------|-----------------|
| **Hypothesis Test** | Tests specific claim (binary decision) | Prior beliefs included | Ignores practical significance |
| **Confidence Interval** | Estimates range of parameter values | No prior | Shows magnitude + uncertainty |
| **Bayesian** | Frequentist approach | Incorporates prior beliefs | Can estimate any parameter |
| **Effect Size** | Tests presence of effect | - | Measures practical importance |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Drug company tests if new medicine improves recovery: H₀: μ = 50%, H₁: μ ≠ 50%

**Failure Case:**  
Using hypothesis test on ALL data collected (data dredging). Multiple tests inflate false positive rate → need correction (Bonferroni)

**Edge Case:**  
Very large sample: Even tiny, negligible effects become statistically significant (p < 0.05). Paradox: statistical significance ≠ practical importance

## 1.4 Layer Breakdown
```
Hypothesis Test Structure:
├─ Null Hypothesis (H₀): Status quo, no effect
├─ Alternative Hypothesis (H₁): Effect exists
├─ Test Statistic: t, z, χ², F (depending on data type)
├─ p-value: Probability of observing data IF H₀ true
├─ Significance Level (α): Decision threshold (usually 0.05)
└─ Decision: Reject H₀ if p < α
```

## 1.5 Mini-Project
Implement: t-test comparing two group means
```python
from scipy import stats

group1 = [2, 4, 6, 8, 10]
group2 = [3, 5, 7, 9, 11]
t_stat, p_value = stats.ttest_ind(group1, group2)

# Decision: reject H₀ if p_value < 0.05
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

## 1.6 Challenge Round
When is hypothesis testing the WRONG choice?
- Exploratory analysis (use confidence intervals instead)
- Multiple comparisons without correction
- When you really care about effect size, not just existence
- Small samples with multiple testing (high false positive rate)

## 1.7 Key References
- [Hypothesis Testing Intuition](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
- [p-values explained](https://www.statisticsdonewrong.com/)
- [Type I/II errors and power](https://www.khanacademy.org/math/statistics-probability/test-of-significance)

---
**Status:** Core concept | **Complements:** Type I/II Errors, p-values
