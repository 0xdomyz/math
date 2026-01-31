# Classification

## 1. Concept Skeleton
**Definition:** Taxonomy of processes by time, state, dependence, and increments  
**Purpose:** Select appropriate models and inference tools  
**Prerequisites:** Discrete/continuous variables, Markov property

## 2. Comparative Framing
| Dimension | Option A | Option B | Implication |
|----------|----------|----------|-------------|
| **Time** | Discrete | Continuous | Difference equations vs SDEs |
| **State** | Discrete | Continuous | Chains vs diffusions |
| **Memory** | Markov | Non-Markov | Simple vs history-dependent |
| **Increments** | Independent | Dependent | Renewal vs long-memory |

## 3. Examples + Counterexamples

**Simple Example:**  
Discrete-time, discrete-state: Markov chain

**Failure Case:**  
Treating long-memory series as Markov → misestimated risk

**Edge Case:**  
Continuous-time, jump process: compound Poisson

## 4. Layer Breakdown
```
Classification Map:
├─ Time index: {0,1,2,...} or [0,∞)
├─ State space: finite, countable, or ℝ^d
├─ Dependence: Markov vs higher-order
├─ Stationarity: strict, weak, none
└─ Path properties: continuous, càdlàg, jumpy
```

**Interaction:** Classify → pick model class → choose estimation method

## 5. Mini-Project
Identify process class from simulated paths:
```python
import numpy as np

x = np.cumsum(np.random.normal(size=1000))
print("Continuous state, discrete time, dependent increments")
```

## 6. Challenge Round
Where does classification fail?
- Mixed processes (regime-switching) blur categories
- Nonstationarity invalidates standard tools
- Sampling at irregular times changes apparent class

## 7. Key References
- [Stochastic Process (Wikipedia)](https://en.wikipedia.org/wiki/Stochastic_process)
- [Continuous-time Markov Chain (Wikipedia)](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain)
- [Stochastic Differential Equation (Wikipedia)](https://en.wikipedia.org/wiki/Stochastic_differential_equation)

---
**Status:** Model selection guide | **Complements:** Definition, Applications
