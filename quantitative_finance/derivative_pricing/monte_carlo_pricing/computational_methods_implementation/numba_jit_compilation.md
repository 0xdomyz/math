# Numba JIT Compilation

## Concept Skeleton
**Definition:** Just-in-time compilation of Python/NumPy code into machine code using Numba  
**Purpose:** Accelerate Monte Carlo loops while keeping Python syntax  
**Prerequisites:** Array programming, function purity, type stability

## Comparative Framing
| Approach | Pure Python | NumPy Vectorized | Numba JIT |
|---|---|---|---|
| **Speed** | Slow | Fast | Fastest for loops |
| **Flexibility** | High | Medium | Medium |
| **Setup** | None | None | JIT compile |

## Examples + Counterexamples
**Simple Example:**  
Compile a path simulation loop to achieve >10× speedup.

**Failure Case:**  
Using unsupported Python objects in JIT function → falls back to object mode.

**Edge Case:**  
First run slower due to compilation; repeated runs benefit.

## Layer Breakdown
```
Numba Acceleration:
├─ Identify Hot Loop:
│   ├─ Path simulation loop
│   └─ Payoff accumulation
├─ Refactor Function:
│   ├─ Use NumPy arrays
│   ├─ Avoid Python objects
│   └─ Use fixed types
├─ Decorate:
│   └─ @njit for nopython mode
└─ Validate:
    ├─ Compare outputs
    └─ Measure speedup
```

**Interaction:** JIT compile loop → run in machine code → faster MC

## Challenge Round
**Q1:** Why does Numba require type stability?  
**A1:** It compiles to machine code with fixed types; dynamic typing prevents optimization.

**Q2:** When is vectorization better than Numba?  
**A2:** When array operations dominate and fit in memory; vectorization leverages optimized BLAS.

**Q3:** Why is the first call slow?  
**A3:** Numba compiles the function on first execution; subsequent calls reuse machine code.

**Q4:** How to generate RNG in Numba safely?  
**A4:** Use `np.random.randn()` inside JIT; for reproducibility, use Numba’s random seeding strategies.

## Key References
- [Numba documentation](https://numba.readthedocs.io/)  
- [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)

---
**Status:** Loop acceleration tool | **Complements:** Vectorization, parallel computing
