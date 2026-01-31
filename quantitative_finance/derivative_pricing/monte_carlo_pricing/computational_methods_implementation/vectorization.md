# Vectorization

## 1. Concept Skeleton
**Definition:** Rewriting computations to use array operations instead of Python loops  
**Purpose:** Speed up Monte Carlo by leveraging optimized low-level numerical libraries  
**Prerequisites:** NumPy broadcasting, memory layout, basic complexity

## 2. Comparative Framing
| Approach | Python Loops | Vectorized NumPy | Numba JIT |
|---|---|---|---|
| **Speed** | Slow | Fast | Fastest |
| **Readability** | High | Medium | Medium |
| **Setup** | None | None | Compile step |

## 3. Examples + Counterexamples

**Simple Example:**  
Simulate $S_T$ for 1e6 paths using one vectorized call to `np.exp`.

**Failure Case:**  
Vectorizing with huge intermediate arrays → memory blowout and slower runtime.

**Edge Case:**  
Small $N$ (e.g., 1e3) → loop overhead minimal; vectorization gains small.

## 4. Layer Breakdown
```
Vectorization Workflow:
├─ Identify Inner Loop:
│   ├─ Path simulation
│   ├─ Payoff calculation
│   └─ Discounting
├─ Replace Loops:
│   ├─ Pre-allocate arrays
│   ├─ Use broadcasting for parameters
│   └─ Use ufuncs (np.exp, np.maximum)
├─ Avoid Python Conditionals:
│   └─ Use boolean masks
└─ Validate:
    ├─ Compare to loop baseline
    └─ Check numerical stability
```

**Interaction:** Batch computations → fewer Python calls → faster MC

## 5. Mini-Project
Loop vs vectorized GBM terminal prices:
```python
import numpy as np
import time

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
N = 1_000_000
np.random.seed(42)

# Loop
start = time.time()
ST_loop = np.empty(N)
for i in range(N):
    z = np.random.randn()
    ST_loop[i] = S0 * np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
loop_time = time.time() - start

# Vectorized
start = time.time()
Z = np.random.randn(N)
ST_vec = S0 * np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
vec_time = time.time() - start

# Price comparison
price_loop = np.exp(-r*T) * np.mean(np.maximum(ST_loop - K, 0))
price_vec = np.exp(-r*T) * np.mean(np.maximum(ST_vec - K, 0))

print(f"Loop time: {loop_time:.3f}s")
print(f"Vectorized time: {vec_time:.3f}s")
print(f"Price diff: {abs(price_loop-price_vec):.6f}")
```

## 6. Challenge Round

**Q1:** Why can vectorization be slower for tiny arrays?  
**A1:** Overhead of creating arrays and calling ufuncs can exceed loop cost for small $N$.

**Q2:** When does vectorization hurt memory?  
**A2:** If it creates large intermediate arrays (e.g., storing all paths for all timesteps).

**Q3:** How do you avoid temporary arrays?  
**A3:** Use in-place operations (`out=`), chunking, or stream computation.

**Q4:** Why is broadcasting critical?  
**A4:** It applies operations across axes without explicit loops, reducing code and runtime.

## 7. Key References
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)  
- [NumPy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html)

---
**Status:** Core speed technique | **Complements:** Numba, memory efficiency
