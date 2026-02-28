# Acceptance-Rejection Method

## Concept Skeleton
**Definition:** Sample from target PDF $f(x)$ using a proposal PDF $g(x)$ and accept with probability $f(x)/(M g(x))$  
**Purpose:** Generate samples from distributions with complex or non-invertible CDFs  
**Prerequisites:** PDF bounds, proposal distributions, uniform RNG

## Comparative Framing
| Method | Acceptance-Rejection | Inverse Transform | Box-Muller |
|---|---|---|---|
| **CDF needed** | No | Yes | No |
| **Efficiency** | Depends on $M$ | High if closed-form | High |
| **Use** | Complex PDFs | Simple CDFs | Normal |

## Examples + Counterexamples
**Simple Example:**  
Sample from a triangular distribution using a uniform proposal with $M=2$.

**Failure Case:**  
Poor proposal $g(x)$ → large $M$ → low acceptance, slow simulation.

**Edge Case:**  
Target distribution with heavy tails requires heavy-tailed proposal to ensure finite $M$.

## Layer Breakdown
```
Acceptance-Rejection:
├─ Choose proposal g(x) and constant M with f(x) ≤ M g(x)
├─ Sample Y ~ g(x)
├─ Sample U ~ Uniform(0,1)
├─ Accept if U ≤ f(Y) / (M g(Y))
└─ Repeat until accepted
```

**Interaction:** Propose → accept/reject → build target samples

## Challenge Round
**Q1:** How does acceptance rate relate to $M$?  
**A1:** Acceptance rate is $1/M$ for a tight bound; larger $M$ means more rejections.

**Q2:** Why is choice of $g(x)$ critical?  
**A2:** A good proposal closely matches $f$ to keep $M$ small and improve efficiency.

**Q3:** What if $f(x)$ has unbounded support?  
**A3:** Choose a heavy-tailed $g(x)$ (e.g., Cauchy) to ensure $f \le M g$.

**Q4:** How do you verify samples?  
**A4:** Compare sample histogram and moments to the target distribution.

## Key References
- [Rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)

---
**Status:** General-purpose sampler | **Complements:** Inverse transform
