# Stationary Distribution

## 1. Concept Skeleton
**Definition:** Distribution $\pi$ with $\pi = \pi P$ for a Markov chain  
**Purpose:** Long-run state probabilities and equilibrium analysis  
**Prerequisites:** Linear algebra, Markov chains, transition matrices

## 2. Comparative Framing
| Concept | Stationary Distribution | Limiting Distribution | Initial Distribution |
|--------|-------------------------|-----------------------|----------------------|
| **Depends on $P$** | Yes | Yes | No |
| **Depends on $\pi_0$** | No | Sometimes | Yes |
| **Existence** | Not always | Requires ergodicity | Always |

## 3. Examples + Counterexamples

**Simple Example:**  
For $P=\begin{bmatrix}0.8&0.2\\0.3&0.7\end{bmatrix}$, $\pi=[0.6,0.4]$

**Failure Case:**  
Periodic chain (e.g., two-state flip) has stationary distribution but no convergence

**Edge Case:**  
Multiple communicating classes → multiple stationary distributions

## 4. Layer Breakdown
```
Stationary Distribution Workflow:
├─ Solve π = πP
├─ Add constraint: sum(π)=1
├─ Check: π ≥ 0
└─ Verify conditions: irreducible + aperiodic → convergence to π
```

**Interaction:** Solve system → Validate constraints → Check convergence conditions

## 5. Mini-Project
Estimate stationary distribution by simulation:
```python
import numpy as np

P = np.array([
    [0.9, 0.1, 0.0],
    [0.2, 0.6, 0.2],
    [0.1, 0.2, 0.7]
])

state = 0
counts = np.zeros(3, dtype=int)

for _ in range(20000):
    counts[state] += 1
    state = np.random.choice([0, 1, 2], p=P[state])

print(counts / counts.sum())
```

## 6. Challenge Round
When is $\pi$ unreliable?
- Chain not irreducible → multiple invariant distributions
- Time-inhomogeneous transitions → no single stationary $\pi$
- Short horizons → transient behavior dominates

## 7. Key References
- [Stationary Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Stationary_distribution)
- [Markov Chain (Wikipedia)](https://en.wikipedia.org/wiki/Markov_chain)
- [Mixing Times (Levin & Peres)](https://bookstore.ams.org/mbk-47)

---
**Status:** Equilibrium concept | **Complements:** Transition Matrix, Ergodic Theorem
