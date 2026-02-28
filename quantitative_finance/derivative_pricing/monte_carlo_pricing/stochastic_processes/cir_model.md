# Cox–Ingersoll–Ross (CIR) Model

## Concept Skeleton
**Definition:** Mean-reverting square-root diffusion: $dr_t = a(b-r_t)dt + \sigma\sqrt{r_t} dW_t$  
**Purpose:** Model non-negative interest rates or variances  
**Prerequisites:** SDEs, mean reversion, noncentral chi-square

## Comparative Framing
| Model | CIR | Vasicek | Heston Variance |
|---|---|---|---|
| **Positivity** | Yes (if Feller) | No | Yes |
| **Distribution** | Noncentral chi-square | Normal | Noncentral chi-square |
| **Use** | Rates | Rates | Volatility |

## Examples + Counterexamples
**Simple Example:**  
Use CIR to prevent negative rates when volatility is high.

**Failure Case:**  
If Feller condition fails, $r_t$ can hit 0 frequently; discretization bias grows.

**Edge Case:**  
$\sigma=0$ reduces to deterministic mean reversion.

## Layer Breakdown
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

## Challenge Round
**Q1:** What is the Feller condition?  
**A1:** $2ab \ge \sigma^2$; ensures the process stays strictly positive.

**Q2:** Why use full truncation Euler?  
**A2:** It avoids negative values under discretization by truncating $r_t$ at 0.

**Q3:** Why is CIR preferred over Vasicek for rates?  
**A3:** It enforces non-negativity, matching economic constraints.

**Q4:** What is the stationary mean?  
**A4:** $E[r_t] \to b$ as $t \to \infty$.

## Key References
- [Cox–Ingersoll–Ross model](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model)

---
**Status:** Positive mean-reverting model | **Complements:** Vasicek, Heston
