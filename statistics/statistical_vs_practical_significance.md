# Statistical Significance vs Practical Significance

## 4.1 Concept Skeleton
**Definition:**  
- Statistical: Result unlikely if null hypothesis true (p < α)
- Practical: Effect size large enough to matter in real world

**Purpose:** Avoid misleading conclusions from large samples  
**Prerequisites:** Effect size, hypothesis testing, sample size

## 4.2 Comparative Framing
| Factor | Statistical Sig. | Practical Sig. | Confidence Interval |
|--------|-----------------|---------------|--------------------|
| **Depends on** | p-value, sample size | Effect size magnitude | Both: effect + uncertainty |
| **"A bigger jump"** | Small sample needed | Large effect needed | Shows plausible range |
| **Paradox** | Large n: trivial effect sig. | Small n: large effect not sig. | Always informative |

## 4.3 Examples + Counterexamples

**Large Sample, No Practical Significance:**  
Study of 1,000,000 people: Weight loss program gives 0.1 kg average loss (p=0.01). Statistically significant but clinically irrelevant.

**Small Sample, Practical Difference, Not Sig.:**  
10 patients: New treatment 50% vs old 30% success (p=0.20). Practical difference but low power.

**Edge Case:**  
Equivalence testing: Proving treatments are approximately equal (inverse of typical hypothesis test)

## 4.4 Layer Breakdown
```
Comprehensive Analysis:
├─ Point Estimate: Best guess for effect (mean difference)
├─ Confidence Interval: Plausible range (95% CI)
├─ Effect Size: Standardized measure (Cohen's d, r, odds ratio)
├─ Hypothesis Test: p-value
├─ Sample Size: Power to detect meaningful effect
└─ Clinical/Practical Threshold: "What size difference matters?"
```

## 4.5 Mini-Project
Compare statistical vs practical significance:
```python
from scipy import stats
import numpy as np

# Scenario 1: Large effect, small sample
effect1 = [10, 12, 11, 13, 12]
baseline1 = [8, 9, 7, 8, 9]
t1, p1 = stats.ttest_ind(effect1, baseline1)
# Large effect (2 units), maybe p > 0.05 (not sig)

# Scenario 2: Tiny effect, huge sample
effect2 = np.random.normal(100.1, 1, 10000)
baseline2 = np.random.normal(100, 1, 10000)
t2, p2 = stats.ttest_ind(effect2, baseline2)
# Tiny effect (0.1 units), likely p < 0.05 (sig)

print(f"Scenario 1: t={t1:.3f}, p={p1:.3f} (large effect, small n)")
print(f"Scenario 2: t={t2:.3f}, p={p2:.3f} (tiny effect, large n)")
```

## 4.6 Challenge Round
When is statistical significance sufficient?
- Early exploratory phase (screening for candidates)
- Consistent with multiple studies
- Clear theoretical mechanism

When do you NEED practical significance?
- Clinical/health decisions (patients care about real improvement)
- Policy/business decisions (costs must justify benefits)
- Any applied research impacting real world

## 4.7 Key References
- [Effect Size Interpretation Guide](https://en.wikipedia.org/wiki/Effect_size)
- [Why p < 0.05 is problematic](https://www.nature.com/articles/d41586-019-00857-7)
- [Moving to practical significance](https://www.statisticsdonewrong.com/)

---
**Status:** Critical distinction | **Complements:** Hypothesis Testing, Effect Size, Confidence Intervals
