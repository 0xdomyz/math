# Pseudorandom Sequences

## 1. Concept Skeleton
**Definition:** Deterministic sequences that mimic randomness with long period and good statistical properties  
**Purpose:** Provide repeatable, high-quality random inputs for simulation  
**Prerequisites:** Seeds, period, statistical testing

## 2. Comparative Framing
| PRNG | Period | Quality | Typical Use |
|---|---|---|---|
| **LCG** | Short | Low | Teaching/demo |
| **Mersenne Twister** | $2^{19937}-1$ | High | General MC |
| **PCG** | Long | High | Modern MC |

## 3. Examples + Counterexamples

**Simple Example:**  
Seeded Mersenne Twister produces identical sequences across runs.

**Failure Case:**  
LCG with poor parameters fails spectral tests → visible lattice structure.

**Edge Case:**  
Seeding with time only can cause repeated streams in parallel runs.

## 4. Layer Breakdown
```
PRNG Lifecycle:
├─ Seed Initialization
├─ State Update Rule
│   ├─ LCG: x_{n+1} = (ax_n + c) mod m
│   └─ MT/PCG: complex recurrence
├─ Output Transformation
│   └─ Convert to U(0,1)
└─ Testing
    ├─ Diehard tests
    └─ Spectral tests
```

**Interaction:** Seed → generate sequence → test → use in MC

## 5. Mini-Project
Compare LCG vs NumPy Mersenne Twister:
```python
import numpy as np
import matplotlib.pyplot as plt

# Simple LCG
m, a, c = 2**31-1, 1103515245, 12345
x = 42
lcg = []
for _ in range(10000):
    x = (a*x + c) % m
    lcg.append(x/m)

lcg = np.array(lcg)
mt = np.random.RandomState(42).rand(10000)

# Scatter test
plt.figure(figsize=(6,4))
plt.scatter(lcg[:-1], lcg[1:], s=3, alpha=0.3, label='LCG')
plt.scatter(mt[:-1], mt[1:], s=3, alpha=0.3, label='MT')
plt.legend()
plt.title('Lag-1 Scatter: LCG vs MT')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** Why are LCGs discouraged for pricing?  
**A1:** They have lattice structure in higher dimensions, causing biased integrals.

**Q2:** How to avoid seed collisions in parallel runs?  
**A2:** Use independent streams or jump-ahead methods; assign unique, spaced seeds.

**Q3:** Why use PCG or MT in MC?  
**A3:** Long period, good equidistribution, and widely tested statistical quality.

**Q4:** Can a PRNG be truly random?  
**A4:** No, it is deterministic; “randomness” is statistical approximation.

## 7. Key References
- [Pseudorandom number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)  
- [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister)

---
**Status:** PRNG foundations | **Complements:** RNG, quasi-random
