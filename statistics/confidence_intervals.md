# Confidence Intervals

## 7.1 Concept Skeleton
**Definition:** Range of values containing true parameter with specified probability (e.g., 95% CI)  
**Purpose:** Quantify uncertainty about parameter estimates from sample data  
**Prerequisites:** Sampling distributions, standard error, normal distribution

## 7.2 Comparative Framing
| Concept | Confidence Interval | Prediction Interval | Credible Interval |
|---------|------------------|------------------|------------------|
| **What it bounds** | Population parameter | Future individual value | Parameter (Bayesian) |
| **Wider** | Narrower | Much wider (includes variation) | Depends on prior |
| **Interpretation** | Procedure covers true param 95% of samples | Future obs falls in range 95% of time | 95% probability param in range |
| **Frequentist** | Yes | Yes | No (Bayesian) |

## 7.3 Examples + Counterexamples

**Simple Example:**  
Sample of 100 people: mean height 170cm, SE=2cm → 95% CI = [166, 174]cm  
Interpretation: If we repeated sampling, 95% of such intervals would contain true mean

**Misinterpretation:**  
NOT "95% probability true mean in [166, 174]" (frequentist: param is fixed, not random)

**Edge Case:**  
Confidence level vs coverage: Nominal 95% CI may not have exactly 95% coverage (depends on assumptions)

## 7.4 Layer Breakdown
```
CI Construction:
├─ Point Estimate: Sample statistic (mean, proportion)
├─ Standard Error: Std dev of sampling distribution
├─ Critical Value: z or t value for desired confidence
├─ Margin of Error: CV × SE
└─ Interval: Point ± Margin of Error

Formula: Estimate ± (Critical Value × Standard Error)
```

## 7.5 Mini-Project
Calculate confidence intervals:
```python
import scipy.stats as stats
import numpy as np

# Sample data
data = [5, 7, 8, 6, 9, 7, 8, 7]
n = len(data)
mean = np.mean(data)
se = np.std(data, ddof=1) / np.sqrt(n)

# 95% CI using t-distribution (correct for small samples)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_lower = mean - t_crit * se
ci_upper = mean + t_crit * se

print(f"Mean: {mean:.2f}")
print(f"Standard Error: {se:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Bootstrap CI (non-parametric alternative)
bootstrap_means = []
for _ in range(10000):
    bootstrap_sample = np.random.choice(data, size=n, replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"95% Bootstrap CI: {bootstrap_ci}")
```

## 7.6 Challenge Round
When is confidence interval better than hypothesis test?
- Estimation focus (how big is the effect?)
- Multiple parameters (shows all simultaneously)
- Effect size matters (CI shows magnitude)

When might CI fail?
- Non-coverage: Nominal 95% CI has < 95% coverage (small n, non-normal)
- Misinterpretation: Leads to Bayesian conclusions
- Multiple comparisons: Need adjustment

## 7.7 Key References
- [CI Interpretation Guide](https://en.wikipedia.org/wiki/Confidence_interval)
- [Why not use CI for everything](https://www.nature.com/articles/d41586-019-00857-7)
- [Bootstrap Confidence Intervals](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

---
**Status:** Core estimation tool | **Complements:** Hypothesis Testing, Central Limit Theorem, Statistical Inference
