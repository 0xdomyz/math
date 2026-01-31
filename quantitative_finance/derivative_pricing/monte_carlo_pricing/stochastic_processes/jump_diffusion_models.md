# Jump Diffusion Models

## 1. Concept Skeleton
**Definition:** SDE with continuous diffusion plus Poisson-driven jumps  
**Purpose:** Capture sudden price moves and heavy tails missing in GBM  
**Prerequisites:** Poisson processes, GBM, Ito–Lévy

## 2. Comparative Framing
| Model | GBM | Merton Jump Diffusion | Heston |
|---|---|---|---|
| **Jumps** | No | Yes | No |
| **Volatility** | Constant | Constant | Stochastic |
| **Tails** | Thin | Fat | Moderate |

## 3. Examples + Counterexamples

**Simple Example:**  
$S_t$ jumps with intensity $\lambda=0.5$ and lognormal jump sizes.

**Failure Case:**  
Ignoring jumps underprices short-dated deep OTM options.

**Edge Case:**  
$\lambda \to 0$ → model reduces to GBM.

## 4. Layer Breakdown
```
Merton Jump Diffusion:
├─ SDE: dS/S = (μ - λκ)dt + σ dW + (J-1) dN
│   ├─ N: Poisson process with intensity λ
│   ├─ J: jump multiplier (lognormal)
│   └─ κ = E[J-1]
├─ Simulation:
│   ├─ Draw Poisson k ~ Poisson(λΔt)
│   ├─ If k>0, apply k jumps
│   └─ Combine with diffusion
└─ Pricing:
    ├─ Monte Carlo or Fourier
    └─ Captures skew/smile
```

**Interaction:** Simulate jumps + diffusion → compute payoffs → discount

## 5. Mini-Project
Simulate jump diffusion terminal prices:
```python
import numpy as np

np.random.seed(42)

S0, T, r, sigma = 100, 1.0, 0.05, 0.2
lam = 0.5
muJ, sigJ = -0.1, 0.2
N = 200000

# Poisson jumps
k = np.random.poisson(lam*T, size=N)

# Sum of jump sizes
J = np.exp(muJ + sigJ*np.random.randn(N))

# Diffusion part
Z = np.random.randn(N)
ST = S0 * np.exp((r - 0.5*sigma**2 - lam*(np.exp(muJ+0.5*sigJ**2)-1))*T + sigma*np.sqrt(T)*Z) * (J**k)

print(f"Mean ST: {ST.mean():.2f}")
```

## 6. Challenge Round

**Q1:** Why subtract $\lambda\kappa$ from drift?  
**A1:** To preserve the expected growth under risk-neutral measure when jumps are added.

**Q2:** How do jumps affect option prices?  
**A2:** They increase tail risk, raising OTM option values and skew.

**Q3:** Why use Poisson?  
**A3:** It models random arrival of discrete jump events with constant intensity.

**Q4:** When is jump diffusion essential?  
**A4:** For markets with frequent discontinuities (earnings, defaults, macro shocks).

## 7. Key References
- [Jump process](https://en.wikipedia.org/wiki/Jump_process)  
- [Merton model](https://en.wikipedia.org/wiki/Jump_diffusion)

---
**Status:** Discontinuous price dynamics | **Complements:** GBM, Heston
