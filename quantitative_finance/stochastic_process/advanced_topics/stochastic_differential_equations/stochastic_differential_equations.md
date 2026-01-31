# Stochastic Differential Equations

## 1. Concept Skeleton
**Definition:** Differential equations driven by stochastic processes, typically Brownian motion  
**Purpose:** Model random dynamics in finance, physics, and biology  
**Prerequisites:** Ito calculus, Brownian motion, ODEs

## 2. Comparative Framing
| Equation | ODE | SDE | Jump SDE |
|---------|-----|-----|----------|
| **Noise** | None | Brownian | Brownian + jumps |
| **Solution** | Deterministic | Random | Random |
| **Use Case** | Deterministic systems | Diffusion models | Crash/arrival models |

## 3. Examples + Counterexamples

**Simple Example:**  
Geometric Brownian motion: $dS_t=\mu S_t dt+\sigma S_t dW_t$

**Failure Case:**  
Applying Euler–Maruyama with large step size → biased paths

**Edge Case:**  
Mean-reverting SDE: $dX_t=\kappa(\theta-X_t)dt+\sigma dW_t$

## 4. Layer Breakdown
```
SDE Modeling Workflow:
├─ Specify drift μ(t, X_t)
├─ Specify diffusion σ(t, X_t)
├─ Choose interpretation (Ito/Stratonovich)
├─ Solve analytically or simulate numerically
└─ Validate with data or moments
```

**Interaction:** Define drift/diffusion → simulate/solve → validate

## 5. Mini-Project
Simulate an SDE with Euler–Maruyama:
```python
import numpy as np

T, n = 1.0, 1000
 dt = T / n

mu, sigma = 0.1, 0.3
X = np.zeros(n)

for t in range(1, n):
    X[t] = X[t-1] + mu*X[t-1]*dt + sigma*X[t-1]*np.sqrt(dt)*np.random.normal()

print(X[:5])
```

## 6. Challenge Round
When do SDE models break?
- Coefficients violate existence/uniqueness conditions
- Discrete observations mask continuous-time dynamics
- Heavy tails require jump or Lévy models

## 7. Key References
- [Stochastic Differential Equation (Wikipedia)](https://en.wikipedia.org/wiki/Stochastic_differential_equation)
- [Euler–Maruyama Method (Wikipedia)](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
- [Øksendal, Stochastic Differential Equations](https://link.springer.com/book/10.1007/978-3-642-14394-6)

---
**Status:** Core continuous-time modeling | **Complements:** Ito Calculus, Brownian Motion
