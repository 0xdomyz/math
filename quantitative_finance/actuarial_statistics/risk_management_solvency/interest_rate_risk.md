# Interest Rate Risk

## 1. Concept Skeleton
**Definition:** Risk from mismatches between asset yields and liability discount rates  
**Purpose:** Manage asset-liability duration, reinvestment, and valuation volatility  
**Prerequisites:** Interest rate term structure, duration, discounting

## 2. Comparative Framing
| Risk Type | Interest Rate Risk | Credit Risk | Liquidity Risk |
|----------|--------------------|-------------|----------------|
| **Driver** | Rate movements | Defaults, spreads | Funding constraints |
| **Impact** | PV of liabilities | Asset losses | Forced sales |
| **Mitigation** | ALM, hedging | Diversification | Liquidity buffers |

## 3. Examples + Counterexamples

**Simple Example:**  
Rates fall → liabilities increase more than assets

**Failure Case:**  
Assuming static yield curve for long-term guarantees

**Edge Case:**  
Negative rates in short maturities

## 4. Layer Breakdown
```
Interest Rate Risk Flow:
├─ Measure durations and convexity
├─ Project yield curve scenarios
├─ Revalue assets and liabilities
├─ Assess capital sensitivity
└─ Hedge with swaps or duration matching
```

**Interaction:** Measure → scenario → revalue → hedge

## 5. Mini-Project
Compute duration-weighted mismatch:
```python
import numpy as np

asset_dur = np.array([3.0, 7.0])
asset_weights = np.array([0.6, 0.4])
liability_dur = 5.5

port_dur = (asset_dur * asset_weights).sum()
print("Duration gap:", port_dur - liability_dur)
```

## 6. Challenge Round
Common pitfalls:
- Ignoring convexity in large rate moves
- Assuming perfect hedge under basis risk
- Overlooking reinvestment risk in long run

## 7. Key References
- [Interest Rate Risk (Wikipedia)](https://en.wikipedia.org/wiki/Interest_rate_risk)
- [Duration (Wikipedia)](https://en.wikipedia.org/wiki/Duration_(finance))
- [ALM (Wikipedia)](https://en.wikipedia.org/wiki/Asset%E2%80%93liability_management)

---
**Status:** Core ALM risk | **Complements:** Solvency Capital, Stochastic Modeling
