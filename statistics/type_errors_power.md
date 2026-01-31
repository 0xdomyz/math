# Type I & Type II Errors, Statistical Power

## 8.1 Concept Skeleton
**Definition:**  
- Type I (α): Reject H₀ when true (false positive)
- Type II (β): Fail to reject H₀ when false (false negative)
- Power: 1-β (probability of detecting true effect)

**Purpose:** Balance risk of false conclusions  
**Prerequisites:** Hypothesis testing, decision rules

## 8.2 Comparative Framing
| Error Type | Cost if Occurs | Typical α | Typical β | Example |
|-----------|----------------|----------|----------|---------|
| **Type I** | False alarm, wasted resources | 0.05 | - | Claiming cure doesn't work (stops good treatment) |
| **Type II** | Miss real effect, miss opportunity | - | 0.20 | Missing actual cure (patients suffer) |
| **Power** | - | - | 0.80 typical | Probability of finding cure if it exists |

## 8.3 Examples + Counterexamples

**Type I Error (α=0.05):**  
Testing 100 true null hypotheses → expect 5 false positives (random noise)

**Type II Error (β):**  
Small sample size → low power → likely miss real effect even if present

**Edge Case:**  
Very low α (0.001) reduces Type I but increases Type II. Tradeoff exists!

## 8.4 Layer Breakdown
```
Decision Matrix:
            Reality: H₀ True  | Reality: H₀ False
Decide: H₀  ├─ Correct (1-α)  | Type II Error (β)
Decide: H₁  ├─ Type I Error (α)| Correct (Power=1-β)

Power Determinants:
├─ Effect Size: Larger effect → higher power
├─ Sample Size: Larger n → higher power
├─ α level: Higher α → higher power (but more Type I errors)
├─ Variability: Lower σ → higher power
└─ Test Type: One-tailed > two-tailed power
```

## 8.5 Mini-Project
Calculate statistical power:
```python
from scipy.stats import norm
import numpy as np

# Parameters
effect_size = 0.5  # Cohen's d
alpha = 0.05
sample_size = 64

# Power calculation (simplified for 2-sample t-test)
# Power ≈ Φ(√(n/2) * effect_size - z_α)
z_alpha = norm.ppf(1 - alpha/2)
z_power = np.sqrt(sample_size/2) * effect_size - z_alpha
power = norm.cdf(z_power)

print(f"Effect size (d): {effect_size}")
print(f"Sample size: {sample_size}")
print(f"Alpha: {alpha}")
print(f"Power (1-β): {power:.3f}")  # Usually ~0.80 is target

# Interpretation: 80% chance of detecting effect if it exists
```

Alternative using statsmodels:
```python
from statsmodels.stats.power import tt_solve_power

power = tt_solve_power(effect_size=0.5, nobs1=32, alpha=0.05, 
                       power=None, alternative='two-sided')
print(f"Power: {power:.3f}")

# Or solve for sample size needed for 80% power
n = tt_solve_power(effect_size=0.5, nobs1=None, alpha=0.05, 
                   power=0.8, alternative='two-sided')
print(f"Sample size needed: {n:.0f}")
```

## 8.6 Challenge Round
When do you accept high Type II error?
- Expensive interventions (low α acceptable to avoid false positives)
- Regulatory approval (side effects feared)

When do you prioritize detecting effects?
- Cheap screening (false positives caught later)
- Medical diagnosis (missing disease = big cost)
- Exploratory research (low bar to investigate further)

## 8.7 Key References
- [Type I/II Error Visualization](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
- [Power Analysis Primer](https://www.khanacademy.org/math/statistics-probability/test-of-significance)
- [Sample Size/Power Calculators](https://www.biostathandbook.com/power.html)

---
**Status:** Critical for study design | **Complements:** Hypothesis Testing, Study Planning, Significance
