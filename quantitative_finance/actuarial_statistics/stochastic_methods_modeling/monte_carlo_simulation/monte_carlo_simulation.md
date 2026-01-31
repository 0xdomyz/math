# Monte Carlo Simulation

## 1. Concept Skeleton
**Definition:** Generate random scenarios to estimate distributions of outcomes and tail risks  
**Purpose:** Value complex products, compute risk metrics (VaR, ES), and test reserves  
**Prerequisites:** Random number generation, path simulation, convergence

## 2. Comparative Framing
| Method | Monte Carlo | Closed-Form | Tree/Lattice |
|--------|------------|-------------|--------------|
| **Flexibility** | High | Low | Moderate |
| **Speed** | Slow | Fast | Moderate |
| **Accuracy** | Converges slowly | Exact (if exists) | Grid-dependent |

## 3. Examples + Counterexamples

**Simple Example:**  
Simulate 10,000 equity paths to value variable annuity guarantee

**Failure Case:**  
Too few simulations → unstable tail risk estimates

**Edge Case:**  
Variance reduction (antithetic variates, control variates)

## 4. Layer Breakdown
```
Monte Carlo Workflow:
├─ Define risk factors and models
├─ Generate correlated scenarios
├─ Project cash flows per path
├─ Discount to present value
├─ Aggregate statistics (mean, VaR, ES)
└─ Confidence intervals
```

**Interaction:** Model → simulate → project → aggregate → report

## 5. Mini-Project
Estimate option value via MC:
```python
import numpy as np

S0 = 100
K = 110
r = 0.03
sigma = 0.20
T = 1
n_sim = 10000

Z = np.random.normal(size=n_sim)
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoff = np.maximum(ST - K, 0)
price = np.exp(-r*T) * payoff.mean()
print("Call price:", price)
```

## 6. Challenge Round
Common pitfalls:
- Insufficient simulations for tail estimates
- Not using variance reduction techniques
- Ignoring computational cost vs. accuracy tradeoff

## 7. Key References
- [Monte Carlo Method (Wikipedia)](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Variance Reduction (Wikipedia)](https://en.wikipedia.org/wiki/Variance_reduction)
- [Monte Carlo Finance (Wikipedia)](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance)

---
**Status:** Simulation foundation | **Complements:** All stochastic models
