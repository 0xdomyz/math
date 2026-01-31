# Transition Matrix

## 1. Concept Skeleton
**Definition:** Matrix $P$ where entry $P_{ij}$ is the probability of moving from state $i$ to $j$  
**Purpose:** Compactly encode Markov dynamics and enable multi-step forecasting  
**Prerequisites:** Matrix multiplication, conditional probability, Markov chains

## 2. Comparative Framing
| Object | Transition Matrix | Adjacency Matrix | Stochastic Matrix |
|--------|--------------------|------------------|-------------------|
| **Entries** | Probabilities | 0/1 or weights | Probabilities |
| **Row Sum** | 1 | Not required | 1 (row-stochastic) |
| **Use** | Dynamics | Graph structure | Probabilistic transitions |

## 3. Examples + Counterexamples

**Simple Example:**  
2-state matrix $P=\begin{bmatrix}0.9&0.1\\0.4&0.6\end{bmatrix}$

**Failure Case:**  
Negative entries or rows not summing to 1 → invalid transition matrix

**Edge Case:**  
Identity matrix → no transitions, states are absorbing

## 4. Layer Breakdown
```
Transition Matrix Operations:
├─ One-step: π1 = π0 P
├─ k-step: πk = π0 P^k
├─ Path probability: product of transitions
└─ Steady state: solve π = πP with sum(π)=1
```

**Interaction:** Encode transitions → Multiply for horizon → Analyze steady-state

## 5. Mini-Project
Compute 5-step transition probabilities and steady-state:
```python
import numpy as np

P = np.array([
    [0.8, 0.2],
    [0.3, 0.7]
])

P5 = np.linalg.matrix_power(P, 5)
print("P^5=\n", P5)

# steady-state solve (π = πP)
A = np.vstack([P.T - np.eye(2), np.ones(2)])
b = np.array([0, 0, 1])
pi = np.linalg.lstsq(A, b, rcond=None)[0]
print("pi=", pi)
```

## 6. Challenge Round
Common pitfalls:
- Using column-stochastic matrices while assuming row-stochastic
- Applying $P^k$ when transitions are time-varying
- Forgetting to verify irreducibility or aperiodicity

## 7. Key References
- [Transition Matrix (Wikipedia)](https://en.wikipedia.org/wiki/Transition_matrix)
- [Stochastic Matrix (Wikipedia)](https://en.wikipedia.org/wiki/Stochastic_matrix)
- [Markov Chains (Kemeny & Snell)](https://www.sciencedirect.com/book/9780124049411/finite-markov-chains)

---
**Status:** Core representation tool | **Complements:** Stationary Distribution, Spectral Analysis
