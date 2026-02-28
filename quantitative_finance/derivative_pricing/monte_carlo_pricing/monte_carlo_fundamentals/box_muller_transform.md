# Box-Muller Transform

## Concept Skeleton
**Definition:** Transformation of two independent $U(0,1)$ variables into two independent standard normal variables  
**Purpose:** Generate normal random variables for Gaussian-driven models  
**Prerequisites:** Uniform RNG, logarithms, trigonometry

## Comparative Framing
| Method | Box-Muller | Polar (Marsaglia) | Ziggurat |
|---|---|---|---|
| **Speed** | Moderate | Faster | Fastest |
| **Complexity** | Simple | Moderate | Higher |
| **Quality** | Exact | Exact | Approx/Exact |

## Examples + Counterexamples
**Simple Example:**  
Transform two uniform samples into two normals for a GBM step.

**Failure Case:**  
Using $U=0$ causes $\log(0)$; must avoid 0 exactly.

**Edge Case:**  
Vectorized Box-Muller may create large arrays; watch memory usage.

## Layer Breakdown
```
Box-Muller:
├─ Draw U1, U2 ~ Uniform(0,1)
├─ Compute:
│   ├─ Z1 = sqrt(-2 ln U1) * cos(2π U2)
│   └─ Z2 = sqrt(-2 ln U1) * sin(2π U2)
├─ Output: Z1, Z2 ~ N(0,1)
└─ Use in simulation
```

**Interaction:** Uniform inputs → transform → normals → drive GBM

## Challenge Round
**Q1:** Why generate two normals at once?  
**A1:** The transform is based on polar coordinates; each uniform pair yields two independent normals.

**Q2:** Why avoid $U1=0$?  
**A2:** $\log(0)$ is undefined; use $U1=\max(U1,\epsilon)$.

**Q3:** When is Box-Muller slower?  
**A3:** Trig functions are costly compared to Ziggurat or Polar methods.

**Q4:** Is Box-Muller exact?  
**A4:** Yes, the transformation yields exact standard normal variables.

## Key References
- [Box–Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)

---
**Status:** Normal generator | **Complements:** Inverse transform, acceptance-rejection
