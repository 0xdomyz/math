# Heston Model

## Concept Skeleton
**Definition:** Stochastic volatility model with mean-reverting variance:  
$\begin{aligned}
 dS_t &= rS_t dt + \sqrt{v_t} S_t dW^S_t\\
 dv_t &= \kappa(\theta - v_t)dt + \xi\sqrt{v_t} dW^v_t,\quad dW^S dW^v=\rho dt
\end{aligned}$  
**Purpose:** Capture volatility smile/skew and stochastic variance dynamics  
**Prerequisites:** CIR process, correlation, Monte Carlo simulation

## Comparative Framing
| Model | GBM | Heston | Jump Diffusion |
|---|---|---|---|
| **Volatility** | Constant | Stochastic | Constant |
| **Skew/Smile** | Poor | Good | Good |
| **Complexity** | Low | Medium | Medium |

## Examples + Counterexamples
**Simple Example:**  
Negative $\rho$ generates equity skew (vol rises when price falls).

**Failure Case:**  
Euler discretization can make variance negative; bias results if not corrected.

**Edge Case:**  
$\xi=0$ reduces Heston to GBM with constant variance.

## Layer Breakdown
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

## Challenge Round
**Q1:** Why does negative $\rho$ create skew?  
**A1:** Negative correlation makes volatility rise when price falls, steepening downside skew.

**Q2:** What is the Feller condition?  
**A2:** $2\kappa\theta \ge \xi^2$ ensures variance stays positive in continuous time.

**Q3:** Why not use Euler for variance?  
**A3:** It can create negative variance; use full truncation or QE scheme.

**Q4:** When is Heston preferred?  
**A4:** When smile/skew is material and GBM misprices OTM options.

## Key References
- [Heston model](https://en.wikipedia.org/wiki/Heston_model)  
- [Stochastic volatility](https://en.wikipedia.org/wiki/Stochastic_volatility)

---
**Status:** Stochastic volatility benchmark | **Complements:** GBM, CIR
