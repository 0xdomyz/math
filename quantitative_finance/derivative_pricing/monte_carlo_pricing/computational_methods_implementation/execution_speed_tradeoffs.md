# Execution Speed Trade-offs

## 1. Concept Skeleton
**Definition:** Balancing runtime, accuracy, and resource usage in Monte Carlo implementations  
**Purpose:** Achieve target error tolerance with minimal compute cost  
**Prerequisites:** Standard error, complexity, convergence $O(1/\sqrt{N})$

## 2. Comparative Framing
| Lever | Effect on Error | Effect on Time | Typical Choice |
|---|---|---|---|
| **Paths $N$** | $\downarrow$ as $1/\sqrt{N}$ | $\uparrow$ linearly | Increase until SE target |
| **Timesteps $M$** | $\downarrow$ discretization | $\uparrow$ linearly | Minimal for accuracy |
| **Variance Reduction** | $\downarrow$ strongly | Small overhead | High ROI |

## 3. Examples + Counterexamples

**Simple Example:**  
Doubling paths reduces SE by $\sqrt{2}$ but doubles runtime.

**Failure Case:**  
Reducing timesteps too much causes bias in path-dependent options.

**Edge Case:**  
Using antithetic variates yields same SE at half the runtime.

## 4. Layer Breakdown
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

## 5. Mini-Project
Estimate runtime vs error tradeoff:
```python
import numpy as np
import time

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def mc_price(N):
    Z = np.random.randn(N)
    ST = S0 * np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * payoff.mean()
    se = np.exp(-r*T) * payoff.std(ddof=1) / np.sqrt(N)
    return price, se

Ns = [1_000, 5_000, 20_000, 100_000, 500_000]

for N in Ns:
    start = time.time()
    price, se = mc_price(N)
    elapsed = time.time() - start
    print(f"N={N:>7}, Price={price:.4f}, SE={se:.4f}, Time={elapsed:.3f}s")
```

## 6. Challenge Round

**Q1:** Why does $O(1/\sqrt{N})$ convergence make MC expensive?  
**A1:** Halving error requires 4× more paths, so costs grow quickly for high precision.

**Q2:** How do variance reduction techniques alter speed?  
**A2:** They reduce variance per path, effectively lowering required N for same SE.

**Q3:** Why is discretization error different from MC error?  
**A3:** Discretization is bias from time-stepping; MC error is statistical noise from finite N.

**Q4:** When should you increase timesteps?  
**A4:** For path-dependent payoffs or models with jumps/mean reversion where coarse steps bias results.

## 7. Key References
- [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)  
- [Variance reduction](https://en.wikipedia.org/wiki/Variance_reduction)

---
**Status:** Implementation planning guide | **Complements:** Variance reduction, benchmarking
