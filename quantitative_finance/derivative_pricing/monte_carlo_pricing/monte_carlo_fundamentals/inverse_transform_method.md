# Inverse Transform Method

## 1. Concept Skeleton
**Definition:** Generate samples by applying inverse CDF to uniform draws: $X=F^{-1}(U)$  
**Purpose:** Sample from arbitrary distributions with known inverse CDF  
**Prerequisites:** CDFs, monotonic functions, numerical inversion

## 2. Comparative Framing
| Method | Inverse Transform | Acceptance-Rejection | Box-Muller |
|---|---|---|---|
| **Requirement** | Invertible CDF | Envelope distribution | Uniform pairs |
| **Speed** | Fast if closed-form | Variable | Moderate |
| **Use** | Exponential, uniform | Complex PDFs | Normal |

## 3. Examples + Counterexamples

**Simple Example:**  
Exponential $X=-\ln(1-U)/\lambda$ from $U(0,1)$.

**Failure Case:**  
CDF without closed form requires numerical inversion; can be slow or unstable.

**Edge Case:**  
Discrete distributions: use inverse CDF on cumulative probabilities.

## 4. Layer Breakdown
```
Inverse Transform Sampling:
├─ Draw U ~ Uniform(0,1)
├─ Compute X = F^{-1}(U)
├─ Validate:
│   ├─ Histogram vs target PDF
│   └─ Compare moments
└─ Use in MC
```

**Interaction:** Uniform sample → inverse CDF → target distribution sample

## 5. Mini-Project
Sample exponential and verify mean:
```python
import numpy as np

np.random.seed(42)

lam = 2.0
U = np.random.rand(100000)
X = -np.log(1-U)/lam

print(f"Sample mean: {X.mean():.4f}, True mean: {1/lam:.4f}")
```

## 6. Challenge Round

**Q1:** Why use $1-U$ instead of $U$?  
**A1:** It avoids $\log(0)$ when $U$ is extremely close to 0; both are valid since $1-U$ is uniform.

**Q2:** When is numerical inversion required?  
**A2:** For distributions with no closed-form inverse CDF (e.g., normal); use root-finding.

**Q3:** How do you sample discrete distributions?  
**A3:** Compute cumulative probabilities and pick the first bin where $U$ falls.

**Q4:** Why is this method exact?  
**A4:** If $U$ is uniform and $F$ is CDF, then $F^{-1}(U)$ has CDF $F$.

## 7. Key References
- [Inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)

---
**Status:** Fundamental sampling method | **Complements:** Acceptance-rejection
