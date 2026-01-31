# Smile & Skew

## 1. Concept Skeleton
**Definition:** Pattern where implied volatility varies with strike and maturity (smile or skew)  
**Purpose:** Capture market pricing of tail risk and asymmetry; build volatility surface  
**Prerequisites:** Implied volatility, option moneyness, arbitrage constraints

## 2. Comparative Framing
| Pattern | Smile | Skew | Flat Surface |
|---|---|---|---|
| **Shape** | U-shaped | Downward sloping | Constant |
| **Markets** | FX | Equities | Idealized BS |
| **Cause** | Symmetric tail risk | Crash risk | Constant σ assumption |

## 3. Examples + Counterexamples

**Simple Example:**  
Equity index options: deep OTM puts have high implied σ → negative skew.

**Failure Case:**  
Assuming flat σ in skewed market → mispriced OTM puts and poor hedging.

**Edge Case:**  
Near-zero rates and low vol: smile flattens; surface almost constant.

## 4. Layer Breakdown
```
Smile/Skew Construction:
├─ Inputs:
│   ├─ Option prices by strike and maturity
│   └─ Forward price F and discount factor
├─ Convert to Implied Vols:
│   └─ σ_imp(K,T) from each quote
├─ Fit Surface:
│   ├─ Parametric: SVI, SABR
│   ├─ Non-parametric: splines
│   └─ Enforce no-arbitrage constraints
├─ Diagnostics:
│   ├─ Risk reversal (skew measure)
│   └─ Butterfly (smile curvature)
└─ Use:
    ├─ Price exotic options
    └─ Hedge with local/volatility models
```

**Interaction:** Market quotes → implied vols → fit smooth surface → validate

## 5. Mini-Project
Construct a simple smile from synthetic quotes:
```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic implied vols for a single maturity
K = np.array([80, 90, 100, 110, 120])
imp_vol = np.array([0.35, 0.28, 0.22, 0.24, 0.30])

plt.figure(figsize=(6,4))
plt.plot(K, imp_vol, 'o-', linewidth=2)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('Volatility Smile')
plt.grid(alpha=0.3)
plt.show()

# Simple skew measure: risk reversal
rr = imp_vol[1] - imp_vol[3]
print(f"Risk reversal (90-110): {rr:.4f}")
```

## 6. Challenge Round

**Q1:** Why does equity skew slope downward?  
**A1:** Markets price crash risk; demand for OTM puts increases implied volatility on the downside.

**Q2:** What arbitrage constraints must a surface satisfy?  
**A2:** Monotonicity and convexity of option prices in strike, and calendar spread constraints in maturity.

**Q3:** Why is local volatility derived from the smile?  
**A3:** The Dupire formula maps implied vol surface to a local volatility function consistent with observed prices.

**Q4:** How does skew affect delta-hedging?  
**A4:** Delta depends on implied vol; skew changes delta sensitivity (vanna/vomma), affecting hedging P&L.

## 7. Key References
- [Volatility smile](https://en.wikipedia.org/wiki/Volatility_smile)  
- [SVI parameterization](https://en.wikipedia.org/wiki/Stochastic_volatility-inspired)

---
**Status:** Market-implied surface feature | **Complements:** Implied volatility, local vol
