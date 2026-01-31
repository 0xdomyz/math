# Implied Volatility

## 1. Concept Skeleton
**Definition:** Volatility input that makes a pricing model (e.g., Black-Scholes) match observed option prices  
**Purpose:** Market-consensus risk estimate used for pricing, hedging, and smile surfaces  
**Prerequisites:** Option pricing model, root-finding, no-arbitrage

## 2. Comparative Framing
| Aspect | Historical σ | Implied σ | Realized σ |
|---|---|---|---|
| **Source** | Past returns | Option prices | Future realized |
| **Forward-looking** | No | Yes | Yes (ex post) |
| **Use** | Risk | Pricing | Validation |

## 3. Examples + Counterexamples

**Simple Example:**  
Call price $C=5.20$ at $S_0=100, K=100, T=1$ → solve $\sigma_{imp}=0.22$.

**Failure Case:**  
Deep ITM/OTM options with low liquidity → implied σ noisy or arbitrary.

**Edge Case:**  
Inconsistent quotes yield arbitrage; implied vol not unique without smoothing.

## 4. Layer Breakdown
```
Implied Volatility Extraction:
├─ Inputs:
│   ├─ Observed option price C_mkt
│   ├─ Model pricing function C(σ)
│   └─ Bounds: σ ∈ [1e-4, 5.0]
├─ Root-Finding:
│   ├─ Solve f(σ) = C(σ) - C_mkt = 0
│   ├─ Methods: Newton-Raphson, bisection, Brent
│   └─ Need vega for Newton step
├─ Convergence:
│   ├─ Newton: σ_{n+1} = σ_n - f(σ)/Vega(σ)
│   └─ Stop when |f(σ)| < tolerance
└─ Surface:
    ├─ Repeat over strikes/maturities
    ├─ Interpolate for smile/skew
    └─ Check no-arbitrage constraints
```

**Interaction:** Price model → root-find σ → build surface → validate arbitrage

## 5. Mini-Project
Solve implied vol using bisection and compare to Newton:
```python
import numpy as np
from scipy.stats import norm

# Black-Scholes call

def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Vega

def bs_vega(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S0*norm.pdf(d1)*np.sqrt(T)

# Bisection implied vol

def implied_vol_bisect(C_mkt, S0, K, T, r, low=1e-4, high=5.0, tol=1e-6):
    for _ in range(200):
        mid = 0.5*(low + high)
        price = bs_call(S0, K, T, r, mid)
        if abs(price - C_mkt) < tol:
            return mid
        if price > C_mkt:
            high = mid
        else:
            low = mid
    return mid

# Newton implied vol

def implied_vol_newton(C_mkt, S0, K, T, r, sigma0=0.2, tol=1e-6):
    sigma = sigma0
    for _ in range(50):
        price = bs_call(S0, K, T, r, sigma)
        vega = bs_vega(S0, K, T, r, sigma)
        sigma -= (price - C_mkt) / vega
        if abs(price - C_mkt) < tol:
            return max(sigma, 1e-6)
    return max(sigma, 1e-6)

S0, K, T, r = 100, 100, 1, 0.05
C_mkt = 5.20

iv_bi = implied_vol_bisect(C_mkt, S0, K, T, r)
iv_nt = implied_vol_newton(C_mkt, S0, K, T, r)

print(f"Implied vol (bisection): {iv_bi:.4f}")
print(f"Implied vol (Newton): {iv_nt:.4f}")
```

## 6. Challenge Round

**Q1:** Why does Newton sometimes fail?  
**A1:** Vega can be tiny (deep ITM/OTM), causing large steps; bisection is more stable.

**Q2:** How to ensure no-arbitrage in vol surface?  
**A2:** Enforce monotonicity in strike, convexity of prices, and calendar spread constraints.

**Q3:** Why is implied vol forward-looking?  
**A3:** It reflects market expectations and risk premia embedded in option prices.

**Q4:** How do dividends affect implied vol?  
**A4:** They change forward price; incorrect dividend assumptions bias implied vol.

## 7. Key References
- [Implied volatility](https://en.wikipedia.org/wiki/Implied_volatility)  
- [Black–Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)

---
**Status:** Primary market calibration target | **Complements:** Volatility smile, surface
