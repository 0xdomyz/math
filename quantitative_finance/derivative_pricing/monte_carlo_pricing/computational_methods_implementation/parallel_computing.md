# Parallel Computing

## 1. Concept Skeleton
**Definition:** Distributing Monte Carlo paths across CPU cores or GPUs  
**Purpose:** Reduce wall-clock time for large simulations  
**Prerequisites:** Embarrassingly parallel workloads, multiprocessing, RNG streams

## 2. Comparative Framing
| Method | Multiprocessing | Threading | GPU (CUDA) |
|---|---|---|---|
| **Use** | CPU cores | IO-bound | Massive parallel |
| **Speedup** | Moderate | Limited (GIL) | High |
| **Complexity** | Medium | Low | High |

## 3. Examples + Counterexamples

**Simple Example:**  
Split 10 million paths across 8 cores → near 8× speedup.

**Failure Case:**  
Shared RNG without seeding → correlated paths and biased estimates.

**Edge Case:**  
Too few paths per core → overhead dominates, speedup < 1.

## 4. Layer Breakdown
```
Parallel MC Workflow:
├─ Partition Work:
│   ├─ N paths → chunks per worker
│   └─ Each worker computes payoffs independently
├─ RNG Strategy:
│   ├─ Independent seeds or jump-ahead streams
│   └─ Avoid overlap of random numbers
├─ Aggregate Results:
│   ├─ Sum payoffs, sum squares
│   └─ Compute mean and SE globally
└─ Validate:
    ├─ Compare with single-thread result
    └─ Ensure reproducibility
```

**Interaction:** Split paths → run in parallel → aggregate statistics

## 5. Mini-Project
Parallelize Monte Carlo with multiprocessing:
```python
import numpy as np
from multiprocessing import Pool

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def worker(seed_n):
    seed, n = seed_n
    np.random.seed(seed)
    Z = np.random.randn(n)
    ST = S0 * np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    return payoff.sum(), (payoff**2).sum(), n

N = 2_000_000
n_workers = 4
chunk = N // n_workers
seeds = [42 + i for i in range(n_workers)]

with Pool(n_workers) as pool:
    results = pool.map(worker, [(seeds[i], chunk) for i in range(n_workers)])

sum_payoff = sum(r[0] for r in results)
sum_sq = sum(r[1] for r in results)
count = sum(r[2] for r in results)

mean = sum_payoff / count
var = sum_sq / count - mean**2
se = np.sqrt(var / count)
price = np.exp(-r*T) * mean

print(f"Price: {price:.6f}, SE: {se:.6f}")
```

## 6. Challenge Round

**Q1:** Why avoid shared RNG state?  
**A1:** It creates correlation across workers, invalidating variance estimates.

**Q2:** How do you scale to GPUs?  
**A2:** Use CUDA kernels (Numba/CuPy) with thousands of threads and batched RNG.

**Q3:** Why can parallel be slower?  
**A3:** Worker startup, IPC, and memory overhead dominate for small jobs.

**Q4:** How to combine standard errors?  
**A4:** Aggregate sums and sums of squares across workers; compute global SE.

## 7. Key References
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)  
- [CUDA](https://docs.nvidia.com/cuda/)

---
**Status:** Scalability technique | **Complements:** Vectorization, Numba
