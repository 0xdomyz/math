# Parallel Computing

## Concept Skeleton
**Definition:** Distributing Monte Carlo paths across CPU cores or GPUs  
**Purpose:** Reduce wall-clock time for large simulations  
**Prerequisites:** Embarrassingly parallel workloads, multiprocessing, RNG streams

## Comparative Framing
| Method | Multiprocessing | Threading | GPU (CUDA) |
|---|---|---|---|
| **Use** | CPU cores | IO-bound | Massive parallel |
| **Speedup** | Moderate | Limited (GIL) | High |
| **Complexity** | Medium | Low | High |

## Examples + Counterexamples
**Simple Example:**  
Split 10 million paths across 8 cores → near 8× speedup.

**Failure Case:**  
Shared RNG without seeding → correlated paths and biased estimates.

**Edge Case:**  
Too few paths per core → overhead dominates, speedup < 1.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why avoid shared RNG state?  
**A1:** It creates correlation across workers, invalidating variance estimates.

**Q2:** How do you scale to GPUs?  
**A2:** Use CUDA kernels (Numba/CuPy) with thousands of threads and batched RNG.

**Q3:** Why can parallel be slower?  
**A3:** Worker startup, IPC, and memory overhead dominate for small jobs.

**Q4:** How to combine standard errors?  
**A4:** Aggregate sums and sums of squares across workers; compute global SE.

## Key References
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)  
- [CUDA](https://docs.nvidia.com/cuda/)

---
**Status:** Scalability technique | **Complements:** Vectorization, Numba
