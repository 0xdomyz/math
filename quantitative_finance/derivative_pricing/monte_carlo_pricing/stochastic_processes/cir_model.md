# Cox–Ingersoll–Ross (CIR) Model

## 1. Concept Skeleton
**Definition:** Mean-reverting square-root diffusion: $dr_t = a(b-r_t)dt + \sigma\sqrt{r_t} dW_t$  
**Purpose:** Model non-negative interest rates or variances  
**Prerequisites:** SDEs, mean reversion, noncentral chi-square

## 2. Comparative Framing
| Model | CIR | Vasicek | Heston Variance |
|---|---|---|---|
| **Positivity** | Yes (if Feller) | No | Yes |
| **Distribution** | Noncentral chi-square | Normal | Noncentral chi-square |
| **Use** | Rates | Rates | Volatility |

## 3. Examples + Counterexamples

**Simple Example:**  
Use CIR to prevent negative rates when volatility is high.

**Failure Case:**  
If Feller condition fails, $r_t$ can hit 0 frequently; discretization bias grows.

**Edge Case:**  
$\sigma=0$ reduces to deterministic mean reversion.

## 4. Layer Breakdown
```
CIR Dynamics:
├─ SDE: dr = a(b-r)dt + σ√r dW
├─ Feller condition:
│   └─ 2ab ≥ σ² ensures strict positivity
├─ Exact transition:
│   └─ Noncentral chi-square distribution
└─ Simulation:
    ├─ Exact sampling or full truncation Euler
    └─ Use for rates/variance
```

**Interaction:** Simulate non-negative r_t → discount payoffs or drive volatility

## 5. Mini-Project
CIR simulation with full truncation:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

r0, a, b, sigma = 0.02, 0.8, 0.04, 0.1
T, n, paths = 5.0, 500, 5

dt = T/n
r = np.zeros((paths, n+1))
r[:,0] = r0

for i in range(n):
    Z = np.random.randn(paths)
    r_pos = np.maximum(r[:, i], 0)
    r[:, i+1] = r[:, i] + a*(b - r_pos)*dt + sigma*np.sqrt(r_pos*dt)*Z

for p in range(paths):
    plt.plot(np.linspace(0,T,n+1), r[p])

plt.title('CIR Short Rate')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** What is the Feller condition?  
**A1:** $2ab \ge \sigma^2$; ensures the process stays strictly positive.

**Q2:** Why use full truncation Euler?  
**A2:** It avoids negative values under discretization by truncating $r_t$ at 0.

**Q3:** Why is CIR preferred over Vasicek for rates?  
**A3:** It enforces non-negativity, matching economic constraints.

**Q4:** What is the stationary mean?  
**A4:** $E[r_t] \to b$ as $t \to \infty$.

## 7. Key References
- [Cox–Ingersoll–Ross model](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model)

---
**Status:** Positive mean-reverting model | **Complements:** Vasicek, Heston
