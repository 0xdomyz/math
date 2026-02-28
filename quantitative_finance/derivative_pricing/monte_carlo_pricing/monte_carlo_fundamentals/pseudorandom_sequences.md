# Pseudorandom Sequences

## Concept Skeleton
**Definition:** Deterministic sequences that mimic randomness with long period and good statistical properties  
**Purpose:** Provide repeatable, high-quality random inputs for simulation  
**Prerequisites:** Seeds, period, statistical testing

## Comparative Framing
| PRNG | Period | Quality | Typical Use |
|---|---|---|---|
| **LCG** | Short | Low | Teaching/demo |
| **Mersenne Twister** | $2^{19937}-1$ | High | General MC |
| **PCG** | Long | High | Modern MC |

## Examples + Counterexamples
**Simple Example:**  
Seeded Mersenne Twister produces identical sequences across runs.

**Failure Case:**  
LCG with poor parameters fails spectral tests → visible lattice structure.

**Edge Case:**  
Seeding with time only can cause repeated streams in parallel runs.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why are LCGs discouraged for pricing?  
**A1:** They have lattice structure in higher dimensions, causing biased integrals.

**Q2:** How to avoid seed collisions in parallel runs?  
**A2:** Use independent streams or jump-ahead methods; assign unique, spaced seeds.

**Q3:** Why use PCG or MT in MC?  
**A3:** Long period, good equidistribution, and widely tested statistical quality.

**Q4:** Can a PRNG be truly random?  
**A4:** No, it is deterministic; “randomness” is statistical approximation.

## Key References
- [Pseudorandom number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator)  
- [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister)

---
**Status:** PRNG foundations | **Complements:** RNG, quasi-random
