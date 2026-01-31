# Random Number Generation

## 1. Concept Skeleton
**Definition:** Algorithms to produce sequences of numbers that approximate i.i.d. draws from a target distribution  
**Purpose:** Drive Monte Carlo simulations for pricing and risk measurement  
**Prerequisites:** Uniform distribution, transformations, PRNG concepts

## 2. Comparative Framing
| Type | Pseudo-Random | Quasi-Random | True Random |
|---|---|---|---|
| **Source** | Deterministic algorithm | Low-discrepancy sequences | Physical entropy |
| **Use** | General MC | Variance reduction | Security/crypto |
| **Repeatable** | Yes | Yes | No |

## 3. Examples + Counterexamples

**Simple Example:**  
Use a PRNG to generate uniform $U(0,1)$ samples, then map to normals for GBM.

**Failure Case:**  
Poor PRNG with short period → repeating patterns → biased option prices.

**Edge Case:**  
Using true random sources makes results irreproducible; debugging becomes hard.

## 4. Layer Breakdown
```
RNG Workflow:
├─ Uniform Generator:
│   ├─ PRNG state and seed
│   └─ Produce U ~ Uniform(0,1)
├─ Transformation:
│   ├─ Inverse CDF
│   ├─ Box-Muller
│   └─ Acceptance-Rejection
├─ Validation:
│   ├─ Mean/variance tests
│   └─ Autocorrelation checks
└─ Simulation:
    ├─ Feed into SDE discretization
    └─ Aggregate payoffs
```

**Interaction:** Generate uniform draws → transform to target distribution → validate → use in MC

## 5. Mini-Project
Test uniformity and autocorrelation of a PRNG stream:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
U = np.random.rand(100000)

# Uniformity check
print(f"Mean: {U.mean():.4f}, Var: {U.var():.4f}")

# Autocorrelation
lag = 1
autocorr = np.corrcoef(U[:-lag], U[lag:])[0, 1]
print(f"Lag-1 autocorrelation: {autocorr:.6f}")

plt.figure(figsize=(6,4))
plt.hist(U, bins=50, density=True, alpha=0.7)
plt.title('Uniform(0,1) Histogram')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** Why prefer PRNGs for MC over true random?  
**A1:** Repeatability and speed; reproducible results are essential for debugging and validation.

**Q2:** How does period length matter?  
**A2:** Too short a period causes cycles within large MC runs, biasing estimates.

**Q3:** Why test autocorrelation?  
**A3:** Dependencies violate i.i.d. assumptions and distort variance estimates.

**Q4:** When is quasi-random better?  
**A4:** For smooth integrands in moderate dimensions; convergence improves vs $O(1/\sqrt{N})$.

## 7. Key References
- [Random number generation](https://en.wikipedia.org/wiki/Random_number_generation)  
- [Pseudo-random number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)

---
**Status:** Core MC driver | **Complements:** Box-Muller, inverse transform
