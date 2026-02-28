# Vectorization

## Concept Skeleton
**Definition:** Rewriting computations to use array operations instead of Python loops  
**Purpose:** Speed up Monte Carlo by leveraging optimized low-level numerical libraries  
**Prerequisites:** NumPy broadcasting, memory layout, basic complexity

## Comparative Framing
| Approach | Python Loops | Vectorized NumPy | Numba JIT |
|---|---|---|---|
| **Speed** | Slow | Fast | Fastest |
| **Readability** | High | Medium | Medium |
| **Setup** | None | None | Compile step |

## Examples + Counterexamples
**Simple Example:**  
Simulate $S_T$ for 1e6 paths using one vectorized call to `np.exp`.

**Failure Case:**  
Vectorizing with huge intermediate arrays → memory blowout and slower runtime.

**Edge Case:**  
Small $N$ (e.g., 1e3) → loop overhead minimal; vectorization gains small.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why can vectorization be slower for tiny arrays?  
**A1:** Overhead of creating arrays and calling ufuncs can exceed loop cost for small $N$.

**Q2:** When does vectorization hurt memory?  
**A2:** If it creates large intermediate arrays (e.g., storing all paths for all timesteps).

**Q3:** How do you avoid temporary arrays?  
**A3:** Use in-place operations (`out=`), chunking, or stream computation.

**Q4:** Why is broadcasting critical?  
**A4:** It applies operations across axes without explicit loops, reducing code and runtime.

## Key References
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)  
- [NumPy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html)

---
**Status:** Core speed technique | **Complements:** Numba, memory efficiency
