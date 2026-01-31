# Ito Calculus

## 1. Concept Skeleton
**Definition:** Stochastic calculus for Brownian motion with Ito’s formula  
**Purpose:** Differentiate/integrate functions of stochastic processes  
**Prerequisites:** Brownian motion, limits, multivariable calculus

## 2. Comparative Framing
| Calculus | Riemann–Stieltjes | Ito | Stratonovich |
|---------|--------------------|-----|--------------|
| **Chain Rule** | Standard | Ito correction term | Standard-like |
| **Interpretation** | Pathwise | Mean-square | Physical modeling |
| **Use Case** | Deterministic paths | Finance | Engineering |

## 3. Examples + Counterexamples

**Simple Example:**  
For $X_t=W_t$, $d(W_t^2)=2W_t dW_t + dt$

**Failure Case:**  
Using standard calculus chain rule on $W_t$ without correction

**Edge Case:**  
Itô integral of adapted process with finite variance

## 4. Layer Breakdown
```
Ito Calculus Core:
├─ Ito integral: ∫ H_t dW_t
├─ Ito isometry: E[(∫ H_t dW_t)^2] = E[∫ H_t^2 dt]
├─ Ito formula: d f(t, X_t) = f_t dt + f_x dX_t + 0.5 f_{xx} (dX_t)^2
└─ Quadratic variation: (dW_t)^2 = dt
```

**Interaction:** Define integrand → construct Ito integral → apply Ito formula

## 5. Mini-Project
Simulate Ito correction in $W_t^2$:
```python
import numpy as np

T, n = 1.0, 10000
 dt = T / n

W = np.cumsum(np.sqrt(dt) * np.random.normal(size=n))
W2 = W**2
approx = np.diff(W2) - 2*W[:-1]*np.diff(W)
print(approx.mean(), "~", dt)
```

## 6. Challenge Round
Common pitfalls:
- Confusing Ito vs Stratonovich conventions
- Ignoring quadratic variation
- Using non-adapted integrands

## 7. Key References
- [Itô Calculus (Wikipedia)](https://en.wikipedia.org/wiki/It%C5%8D_calculus)
- [Itô’s Lemma (Wikipedia)](https://en.wikipedia.org/wiki/It%C5%8D%27s_lemma)
- [Øksendal, Stochastic Differential Equations](https://link.springer.com/book/10.1007/978-3-642-14394-6)

---
**Status:** Core diffusion calculus | **Complements:** Brownian Motion, SDEs
