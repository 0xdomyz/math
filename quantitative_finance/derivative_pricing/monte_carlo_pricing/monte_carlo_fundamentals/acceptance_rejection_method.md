# Acceptance-Rejection Method

## 1. Concept Skeleton
**Definition:** Sample from target PDF $f(x)$ using a proposal PDF $g(x)$ and accept with probability $f(x)/(M g(x))$  
**Purpose:** Generate samples from distributions with complex or non-invertible CDFs  
**Prerequisites:** PDF bounds, proposal distributions, uniform RNG

## 2. Comparative Framing
| Method | Acceptance-Rejection | Inverse Transform | Box-Muller |
|---|---|---|---|
| **CDF needed** | No | Yes | No |
| **Efficiency** | Depends on $M$ | High if closed-form | High |
| **Use** | Complex PDFs | Simple CDFs | Normal |

## 3. Examples + Counterexamples

**Simple Example:**  
Sample from a triangular distribution using a uniform proposal with $M=2$.

**Failure Case:**  
Poor proposal $g(x)$ → large $M$ → low acceptance, slow simulation.

**Edge Case:**  
Target distribution with heavy tails requires heavy-tailed proposal to ensure finite $M$.

## 4. Layer Breakdown
```
Acceptance-Rejection:
├─ Choose proposal g(x) and constant M with f(x) ≤ M g(x)
├─ Sample Y ~ g(x)
├─ Sample U ~ Uniform(0,1)
├─ Accept if U ≤ f(Y) / (M g(Y))
└─ Repeat until accepted
```

**Interaction:** Propose → accept/reject → build target samples

## 5. Mini-Project
Sample from a triangular distribution:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Target: triangular on [0,1] with peak at 0.5: f(x)=4x for x<=0.5, f(x)=4(1-x) for x>0.5

def f(x):
    return np.where(x <= 0.5, 4*x, 4*(1-x))

# Proposal: uniform g(x)=1 on [0,1]
M = 2.0

samples = []
while len(samples) < 50000:
    y = np.random.rand()
    u = np.random.rand()
    if u <= f(y) / (M * 1.0):
        samples.append(y)

samples = np.array(samples)

plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
xs = np.linspace(0,1,200)
plt.plot(xs, f(xs), 'r-', label='Target pdf')
plt.legend()
plt.title('Acceptance-Rejection Sampling')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** How does acceptance rate relate to $M$?  
**A1:** Acceptance rate is $1/M$ for a tight bound; larger $M$ means more rejections.

**Q2:** Why is choice of $g(x)$ critical?  
**A2:** A good proposal closely matches $f$ to keep $M$ small and improve efficiency.

**Q3:** What if $f(x)$ has unbounded support?  
**A3:** Choose a heavy-tailed $g(x)$ (e.g., Cauchy) to ensure $f \le M g$.

**Q4:** How do you verify samples?  
**A4:** Compare sample histogram and moments to the target distribution.

## 7. Key References
- [Rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)

---
**Status:** General-purpose sampler | **Complements:** Inverse transform
