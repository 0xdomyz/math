# Random Number Generation

## Concept Skeleton
**Definition:** Algorithms to produce sequences of numbers that approximate i.i.d. draws from a target distribution  
**Purpose:** Drive Monte Carlo simulations for pricing and risk measurement  
**Prerequisites:** Uniform distribution, transformations, PRNG concepts

## Comparative Framing
| Type | Pseudo-Random | Quasi-Random | True Random |
|---|---|---|---|
| **Source** | Deterministic algorithm | Low-discrepancy sequences | Physical entropy |
| **Use** | General MC | Variance reduction | Security/crypto |
| **Repeatable** | Yes | Yes | No |

## Examples + Counterexamples
**Simple Example:**  
Use a PRNG to generate uniform $U(0,1)$ samples, then map to normals for GBM.

**Failure Case:**  
Poor PRNG with short period → repeating patterns → biased option prices.

**Edge Case:**  
Using true random sources makes results irreproducible; debugging becomes hard.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why prefer PRNGs for MC over true random?  
**A1:** Repeatability and speed; reproducible results are essential for debugging and validation.

**Q2:** How does period length matter?  
**A2:** Too short a period causes cycles within large MC runs, biasing estimates.

**Q3:** Why test autocorrelation?  
**A3:** Dependencies violate i.i.d. assumptions and distort variance estimates.

**Q4:** When is quasi-random better?  
**A4:** For smooth integrands in moderate dimensions; convergence improves vs $O(1/\sqrt{N})$.

## Key References
- [Random number generation](https://en.wikipedia.org/wiki/Random_number_generation)  
- [Pseudo-random number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)

---
**Status:** Core MC driver | **Complements:** Box-Muller, inverse transform
