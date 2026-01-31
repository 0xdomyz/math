# Heston Model

## 1. Concept Skeleton
**Definition:** Stochastic volatility model with mean-reverting variance:  
$\begin{aligned}
 dS_t &= rS_t dt + \sqrt{v_t} S_t dW^S_t\\
 dv_t &= \kappa(\theta - v_t)dt + \xi\sqrt{v_t} dW^v_t,\quad dW^S dW^v=\rho dt
\end{aligned}$  
**Purpose:** Capture volatility smile/skew and stochastic variance dynamics  
**Prerequisites:** CIR process, correlation, Monte Carlo simulation

## 2. Comparative Framing
| Model | GBM | Heston | Jump Diffusion |
|---|---|---|---|
| **Volatility** | Constant | Stochastic | Constant |
| **Skew/Smile** | Poor | Good | Good |
| **Complexity** | Low | Medium | Medium |

## 3. Examples + Counterexamples

**Simple Example:**  
Negative $\rho$ generates equity skew (vol rises when price falls).

**Failure Case:**  
Euler discretization can make variance negative; bias results if not corrected.

**Edge Case:**  
$\xi=0$ reduces Heston to GBM with constant variance.

## 4. Layer Breakdown
```
Heston Simulation:
├─ Variance Process:
│   ├─ dv = κ(θ-v)dt + ξ√v dW_v
│   ├─ Correlate dW_v with dW_S (ρ)
│   └─ Use full truncation or QE scheme
├─ Price Process:
│   └─ dS = rS dt + √v S dW_S
├─ Correlation:
│   └─ dW_S = ρ dW_v + √(1-ρ²) dW_⊥
└─ Pricing:
    ├─ Monte Carlo
    └─ Characteristic function methods
```

**Interaction:** Simulate v_t → drive S_t with correlated shocks → price options

## 5. Mini-Project
Heston simulation with full truncation:
```python
import numpy as np

np.random.seed(42)

S0, v0, r = 100, 0.04, 0.05
kappa, theta, xi, rho = 1.5, 0.04, 0.5, -0.7
T, n, N = 1.0, 252, 200000

dt = T/n
S = np.full(N, S0, dtype=float)
v = np.full(N, v0, dtype=float)

for _ in range(n):
    Z1 = np.random.randn(N)
    Z2 = np.random.randn(N)
    dWv = np.sqrt(dt)*Z1
    dWs = np.sqrt(dt)*(rho*Z1 + np.sqrt(1-rho**2)*Z2)
    
    v_pos = np.maximum(v, 0)
    v = v + kappa*(theta - v_pos)*dt + xi*np.sqrt(v_pos)*dWv
    v_pos = np.maximum(v, 0)
    S = S * np.exp((r-0.5*v_pos)*dt + np.sqrt(v_pos)*dWs)

print(f"Mean S_T: {S.mean():.2f}")
```

## 6. Challenge Round

**Q1:** Why does negative $\rho$ create skew?  
**A1:** Negative correlation makes volatility rise when price falls, steepening downside skew.

**Q2:** What is the Feller condition?  
**A2:** $2\kappa\theta \ge \xi^2$ ensures variance stays positive in continuous time.

**Q3:** Why not use Euler for variance?  
**A3:** It can create negative variance; use full truncation or QE scheme.

**Q4:** When is Heston preferred?  
**A4:** When smile/skew is material and GBM misprices OTM options.

## 7. Key References
- [Heston model](https://en.wikipedia.org/wiki/Heston_model)  
- [Stochastic volatility](https://en.wikipedia.org/wiki/Stochastic_volatility)

---
**Status:** Stochastic volatility benchmark | **Complements:** GBM, CIR
