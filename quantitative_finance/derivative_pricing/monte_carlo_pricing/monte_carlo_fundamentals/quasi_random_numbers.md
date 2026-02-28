# Quasi-Random Numbers (Low-Discrepancy)

## Concept Skeleton
**Definition:** Deterministic sequences that fill the unit cube more uniformly than pseudo-random samples  
**Purpose:** Improve convergence of Monte Carlo integration for smooth payoffs  
**Prerequisites:** Discrepancy, dimensionality, scrambling

## Comparative Framing
| Sequence | Strength | Weakness | Use |
|---|---|---|---|
| **Sobol** | High-dimensional uniformity | Needs scrambling | QMC pricing |
| **Halton** | Simple | Correlations in high dims | Low-dim integrals |
| **Niederreiter** | Strong theoretical | Complex | Research |

## Examples + Counterexamples
**Simple Example:**  
Sobol points reduce variance in Asian option pricing vs PRNG.

**Failure Case:**  
High dimension (d>100) without scrambling → correlations degrade performance.

**Edge Case:**  
Non-smooth payoff (digital) reduces QMC advantage; variance reduction minimal.

## Layer Breakdown
```
QMC Workflow:
├─ Generate Low-Discrepancy Points
│   ├─ Sobol or Halton sequence
│   └─ Apply scrambling if needed
├─ Map to Distribution
│   └─ Apply inverse CDF to each dimension
├─ Compute Payoff
├─ Estimate Price
│   └─ Average discounted payoffs
└─ Compare to PRNG
```

**Interaction:** Generate quasi-random points → transform to target distribution → simulate → estimate

## Challenge Round
**Q1:** Why does QMC converge faster?  
**A1:** Low-discrepancy points reduce integration error for smooth functions; error decreases closer to $O(1/N)$.

**Q2:** Why use scrambling?  
**A2:** It randomizes QMC for error estimation while preserving low discrepancy.

**Q3:** When does QMC fail?  
**A3:** High-dimensional or non-smooth payoffs reduce the uniformity advantage.

**Q4:** How to choose dimension ordering?  
**A4:** Place most important factors in early dimensions; QMC is most uniform there.

## Key References
- [Low-discrepancy sequence](https://en.wikipedia.org/wiki/Low-discrepancy_sequence)  
- [Quasi-Monte Carlo method](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method)

---
**Status:** Variance reduction via sampling | **Complements:** RNG, variance reduction techniques
