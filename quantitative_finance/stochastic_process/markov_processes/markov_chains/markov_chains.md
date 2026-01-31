# Markov Chains

## 1. Concept Skeleton
**Definition:** Stochastic process with the Markov property: future depends only on present state  
**Purpose:** Model systems with memoryless transitions (queues, credit states, regime switches)  
**Prerequisites:** Conditional probability, matrices, discrete-time processes

## 2. Comparative Framing
| Process | Markov Chain | IID Sequence | Hidden Markov Model |
|--------|---------------|--------------|----------------------|
| **Dependence** | Depends on current state | Independent | Depends on hidden state |
| **State Observed** | Yes | Yes | No (hidden) |
| **Dynamics** | Transition matrix | Fixed distribution | Emissions + transitions |

## 3. Examples + Counterexamples

**Simple Example:**  
Weather states {Sunny, Rainy} with fixed transition probabilities

**Failure Case:**  
Stock returns with volatility clustering → violates memoryless assumption

**Edge Case:**  
Absorbing state (e.g., default) where once entered, process never leaves

## 4. Layer Breakdown
```
Markov Chain Structure:
├─ State Space S = {s1, s2, ..., sk}
├─ Transition Matrix P
│   ├─ Pij = P(X_{t+1}=sj | X_t=si)
│   └─ Rows sum to 1
├─ Initial Distribution π0
├─ Evolution: πt+1 = πt P
└─ Long-Run Behavior: stationary distribution π* (if exists)
```

**Interaction:** Initialize → Apply transitions → Evolve distribution → Check stability

## 5. Mini-Project
Simulate a 3-state Markov chain and estimate long-run probabilities:
```python
import numpy as np

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

n_steps = 10000
state = 0
counts = np.zeros(3, dtype=int)

for _ in range(n_steps):
    counts[state] += 1
    state = np.random.choice([0, 1, 2], p=P[state])

print(counts / n_steps)
```

## 6. Challenge Round
When do Markov chains mislead?
- Hidden dependencies (seasonality, regime shifts) break memoryless property
- Non-stationary transitions require time-varying matrices
- Continuous state spaces need different models (e.g., diffusion)

## 7. Key References
- [Markov Chain (Wikipedia)](https://en.wikipedia.org/wiki/Markov_chain)
- [Markov Chains Intro (Khan Academy)](https://www.khanacademy.org/math/statistics-probability/probability-library/markov-chains/a/markov-chains-intro)
- [Markov Chains (Norris, Cambridge)](https://www.cambridge.org/core/books/markov-chains/6A2F6C4A7040A1E8C2EEC6AACE1FEAE1)

---
**Status:** Core stochastic modeling tool | **Complements:** Poisson Process, HMMs, Time Series
