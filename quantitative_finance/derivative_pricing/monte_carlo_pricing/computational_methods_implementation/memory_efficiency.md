# Memory Efficiency

## Concept Skeleton
**Definition:** Reducing memory usage during Monte Carlo by avoiding storing full paths  
**Purpose:** Enable large-scale simulations without swapping or crashes  
**Prerequisites:** Streaming statistics, Welford algorithm, chunking

## Comparative Framing
| Strategy | Full Path Storage | Streaming Stats | Chunked Simulation |
|---|---|---|---|
| **Memory** | High | Low | Medium |
| **Speed** | Moderate | High | High |
| **Use** | Path-dependent payoffs | Vanilla payoffs | Mixed |

## Examples + Counterexamples
**Simple Example:**  
Compute payoff mean on the fly using Welford; no need to keep all payoffs.

**Failure Case:**  
Storing 1e6 paths × 1e3 steps → 8e11 bytes → memory crash.

**Edge Case:**  
Path-dependent option (Asian) needs partial path storage; use rolling sums.

## Layer Breakdown
```
Memory-Efficient MC:
├─ Streaming Estimation:
│   ├─ Update mean and variance incrementally
│   └─ No full payoff array
├─ Chunking:
│   ├─ Simulate in batches of size B
│   └─ Aggregate batch statistics
├─ Path Compression:
│   ├─ Store only required state (running average)
│   └─ Discard intermediate prices
└─ Validation:
    ├─ Compare with full storage for small N
    └─ Check numerical stability
```

**Interaction:** Simulate in chunks → update mean/variance → discard data

## Challenge Round
**Q1:** Why is Welford numerically stable?  
**A1:** It avoids catastrophic cancellation by updating mean and variance incrementally.

**Q2:** How do you handle Asian options without storing all paths?  
**A2:** Keep running sums (or geometric sums) and update per timestep.

**Q3:** When does chunking help?  
**A3:** When memory is limited but you need large $N$ for small standard error.

**Q4:** What is the tradeoff of streaming?  
**A4:** Less flexible for path-dependent diagnostics; you lose path history.

## Key References
- [Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)  
- [Welford's algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

---
**Status:** Large-scale MC enabler | **Complements:** Vectorization, parallel computing
