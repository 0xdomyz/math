# Execution Speed Trade-offs

## Concept Skeleton
**Definition:** Balancing runtime, accuracy, and resource usage in Monte Carlo implementations  
**Purpose:** Achieve target error tolerance with minimal compute cost  
**Prerequisites:** Standard error, complexity, convergence $O(1/\sqrt{N})$

## Comparative Framing
| Lever | Effect on Error | Effect on Time | Typical Choice |
|---|---|---|---|
| **Paths $N$** | $\downarrow$ as $1/\sqrt{N}$ | $\uparrow$ linearly | Increase until SE target |
| **Timesteps $M$** | $\downarrow$ discretization | $\uparrow$ linearly | Minimal for accuracy |
| **Variance Reduction** | $\downarrow$ strongly | Small overhead | High ROI |

## Examples + Counterexamples
**Simple Example:**  
Doubling paths reduces SE by $\sqrt{2}$ but doubles runtime.

**Failure Case:**  
Reducing timesteps too much causes bias in path-dependent options.

**Edge Case:**  
Using antithetic variates yields same SE at half the runtime.

## Layer Breakdown
```
Speed-Accuracy Optimization:
├─ Error Budget:
│   ├─ Statistical error (MC) ~ O(1/√N)
│   └─ Discretization error ~ O(Δt)
├─ Decide Targets:
│   ├─ Target SE (e.g., 0.01)
│   └─ Acceptable bias
├─ Optimize Levers:
│   ├─ Increase N until SE met
│   ├─ Increase M until bias acceptable
│   ├─ Add variance reduction
│   └─ Use vectorization/JIT/parallel
└─ Validate:
    ├─ Convergence plots
    └─ Benchmark against closed-form
```

**Interaction:** Set accuracy → choose N, M, VR → measure runtime → iterate

## Challenge Round
**Q1:** Why does $O(1/\sqrt{N})$ convergence make MC expensive?  
**A1:** Halving error requires 4× more paths, so costs grow quickly for high precision.

**Q2:** How do variance reduction techniques alter speed?  
**A2:** They reduce variance per path, effectively lowering required N for same SE.

**Q3:** Why is discretization error different from MC error?  
**A3:** Discretization is bias from time-stepping; MC error is statistical noise from finite N.

**Q4:** When should you increase timesteps?  
**A4:** For path-dependent payoffs or models with jumps/mean reversion where coarse steps bias results.

## Key References
- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)  
- [Variance reduction](https://en.wikipedia.org/wiki/Variance_reduction)

---
**Status:** Implementation planning guide | **Complements:** Variance reduction, benchmarking
