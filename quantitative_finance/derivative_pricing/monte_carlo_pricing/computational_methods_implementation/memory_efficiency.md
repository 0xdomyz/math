# Memory Efficiency

## 1. Concept Skeleton
**Definition:** Reducing memory usage during Monte Carlo by avoiding storing full paths  
**Purpose:** Enable large-scale simulations without swapping or crashes  
**Prerequisites:** Streaming statistics, Welford algorithm, chunking

## 2. Comparative Framing
| Strategy | Full Path Storage | Streaming Stats | Chunked Simulation |
|---|---|---|---|
| **Memory** | High | Low | Medium |
| **Speed** | Moderate | High | High |
| **Use** | Path-dependent payoffs | Vanilla payoffs | Mixed |

## 3. Examples + Counterexamples

**Simple Example:**  
Compute payoff mean on the fly using Welford; no need to keep all payoffs.

**Failure Case:**  
Storing 1e6 paths × 1e3 steps → 8e11 bytes → memory crash.

**Edge Case:**  
Path-dependent option (Asian) needs partial path storage; use rolling sums.

## 4. Layer Breakdown
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

## 5. Mini-Project
Streaming payoff statistics for a call option:
```python
import numpy as np

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
N = 5_000_000
chunk = 200_000

mean = 0.0
M2 = 0.0
count = 0

np.random.seed(42)

for _ in range(N // chunk):
    Z = np.random.randn(chunk)
    ST = S0 * np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    for x in payoff:
        count += 1
        delta = x - mean
        mean += delta / count
        M2 += delta * (x - mean)

variance = M2 / (count - 1)
se = np.sqrt(variance / count)
price = np.exp(-r*T) * mean

print(f"Price: {price:.6f}, SE: {se:.6f}")
```

## 6. Challenge Round

**Q1:** Why is Welford numerically stable?  
**A1:** It avoids catastrophic cancellation by updating mean and variance incrementally.

**Q2:** How do you handle Asian options without storing all paths?  
**A2:** Keep running sums (or geometric sums) and update per timestep.

**Q3:** When does chunking help?  
**A3:** When memory is limited but you need large $N$ for small standard error.

**Q4:** What is the tradeoff of streaming?  
**A4:** Less flexible for path-dependent diagnostics; you lose path history.

## 7. Key References
- [Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)  
- [Welford's algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

---
**Status:** Large-scale MC enabler | **Complements:** Vectorization, parallel computing
