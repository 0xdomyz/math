# GARCH Models

## 1. Concept Skeleton
**Definition:** Volatility models where conditional variance depends on past shocks and past variance  
**Purpose:** Forecast time-varying volatility; improve MC inputs and risk estimates  
**Prerequisites:** Time series, conditional variance, stationarity

## 2. Comparative Framing
| Model | GARCH(1,1) | EWMA | Stochastic Volatility |
|---|---|---|---|
| **Dynamics** | $\sigma_t^2=\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2$ | $\sigma_t^2=\lambda\sigma_{t-1}^2+(1-\lambda)\epsilon_{t-1}^2$ | Separate latent process |
| **Mean Reversion** | Yes | No (implicit) | Yes |
| **Use** | Forecasting | Risk metrics | Option pricing |

## 3. Examples + Counterexamples

**Simple Example:**  
GARCH(1,1) fitted to returns gives $\alpha=0.07, \beta=0.90$ → persistent volatility.

**Failure Case:**  
Assuming constant σ in crisis → underestimates near-term risk; GARCH adjusts faster.

**Edge Case:**  
If $\alpha+\beta \ge 1$, variance is non-stationary; forecasts explode.

## 4. Layer Breakdown
```
GARCH(1,1) Calibration:
├─ Model:
│   ├─ r_t = μ + ε_t
│   ├─ ε_t = σ_t z_t, z_t ~ N(0,1)
│   └─ σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2
├─ Constraints:
│   ├─ ω > 0
│   ├─ α ≥ 0, β ≥ 0
│   └─ α + β < 1 (stationary)
├─ Estimation:
│   ├─ Maximum likelihood
│   └─ Optimize log-likelihood over ω,α,β
├─ Forecast:
│   └─ E[σ_{t+h}^2] = ω(1- (α+β)^h)/(1-α-β) + (α+β)^h σ_t^2
└─ Use in MC:
    ├─ Simulate σ_t over horizon
    └─ Feed σ_t into return simulation
```

**Interaction:** Fit parameters → forecast σ_t → simulate returns with time-varying volatility

## 5. Mini-Project
Fit GARCH(1,1) and compare to EWMA:
```python
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

np.random.seed(42)

# Simulated returns
n = 2000
returns = np.random.normal(0, 0.01, n)

# Fit GARCH(1,1)
model = arch_model(returns*100, vol='GARCH', p=1, q=1, mean='Zero')
res = model.fit(disp='off')

print(res.params)

# Conditional volatility
cond_vol = res.conditional_volatility / 100

# EWMA
lam = 0.94
sigma2 = np.zeros(n)
sigma2[0] = np.var(returns)
for t in range(1, n):
    sigma2[t] = lam * sigma2[t-1] + (1-lam) * returns[t-1]**2

ewma = np.sqrt(sigma2)

plt.figure(figsize=(10,4))
plt.plot(cond_vol, label='GARCH σ')
plt.plot(ewma, label='EWMA σ', alpha=0.7)
plt.legend()
plt.title('Conditional Volatility: GARCH vs EWMA')
plt.grid(alpha=0.3)
plt.show()
```

## 6. Challenge Round

**Q1:** Why is $\alpha+\beta$ called persistence?  
**A1:** It measures how long shocks affect variance; values near 1 imply slow decay.

**Q2:** When should you prefer GARCH over historical σ?  
**A2:** When volatility clusters and recent shocks matter; GARCH adapts dynamically.

**Q3:** Why does GARCH often underestimate tails?  
**A3:** It assumes conditional normality; heavy tails require t-distribution or EGARCH.

**Q4:** How does GARCH affect option prices?  
**A4:** It produces forward-looking σ_t paths; pricing depends on expected future variance.

## 7. Key References
- [GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity)  
- Bollerslev, T. “Generalized autoregressive conditional heteroskedasticity” (1986)

---
**Status:** Dynamic volatility model | **Complements:** Historical volatility, implied volatility
