# GARCH Models

## Concept Skeleton
**Definition:** Volatility models where conditional variance depends on past shocks and past variance  
**Purpose:** Forecast time-varying volatility; improve MC inputs and risk estimates  
**Prerequisites:** Time series, conditional variance, stationarity

## Comparative Framing
| Model | GARCH(1,1) | EWMA | Stochastic Volatility |
|---|---|---|---|
| **Dynamics** | $\sigma_t^2=\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2$ | $\sigma_t^2=\lambda\sigma_{t-1}^2+(1-\lambda)\epsilon_{t-1}^2$ | Separate latent process |
| **Mean Reversion** | Yes | No (implicit) | Yes |
| **Use** | Forecasting | Risk metrics | Option pricing |

## Examples + Counterexamples
**Simple Example:**  
GARCH(1,1) fitted to returns gives $\alpha=0.07, \beta=0.90$ → persistent volatility.

**Failure Case:**  
Assuming constant σ in crisis → underestimates near-term risk; GARCH adjusts faster.

**Edge Case:**  
If $\alpha+\beta \ge 1$, variance is non-stationary; forecasts explode.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why is $\alpha+\beta$ called persistence?  
**A1:** It measures how long shocks affect variance; values near 1 imply slow decay.

**Q2:** When should you prefer GARCH over historical σ?  
**A2:** When volatility clusters and recent shocks matter; GARCH adapts dynamically.

**Q3:** Why does GARCH often underestimate tails?  
**A3:** It assumes conditional normality; heavy tails require t-distribution or EGARCH.

**Q4:** How does GARCH affect option prices?  
**A4:** It produces forward-looking σ_t paths; pricing depends on expected future variance.

## Key References
- [GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity)  
- Bollerslev, T. “Generalized autoregressive conditional heteroskedasticity” (1986)

---
**Status:** Dynamic volatility model | **Complements:** Historical volatility, implied volatility
