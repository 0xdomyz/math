# Pension Funding

## 1. Concept Skeleton
**Definition:** Strategy to contribute assets to meet future pension obligations  
**Purpose:** Ensure plan solvency and manage contribution volatility  
**Prerequisites:** Liability valuation, actuarial cost methods, cash-flow forecasts

## 2. Comparative Framing
| Method | Frozen Attained | Entry Age Normal | Unit Credit |
|--------|-----------------|-----------------|------------|
| **Cost Base** | Past service | Career earnings | Service accrued |
| **Volatility** | Low | Moderate | High |
| **Use** | Mature plans | Career plans | Smaller plans |

## 3. Examples + Counterexamples

**Simple Example:**  
Determine annual employer contribution to avoid underfunding

**Failure Case:**  
Contribution volatility causes budget strain

**Edge Case:**  
Plan amendments retroactively increase liability

## 4. Layer Breakdown
```
Funding Workflow:
├─ Determine liability (ABO or PBO)
├─ Value assets
├─ Compute unfunded amount
├─ Choose actuarial cost method
├─ Set contribution schedule
└─ Monitor and adjust
```

**Interaction:** Measure → cost → contribute → monitor

## 5. Mini-Project
Compute level contribution amortizing unfunded liability:
```python
unfunded = 100000
years = 20
rate = 0.04
contribution = unfunded * rate / (1 - (1 + rate) ** (-years))
print("Annual contribution:", contribution)
```

## 6. Challenge Round
Common pitfalls:
- Overly aggressive assumptions leading to underfunding
- Contribution spikes from assumption changes
- Not adjusting for actual vs expected experience

## 7. Key References
- [Pension Funding (Wikipedia)](https://en.wikipedia.org/wiki/Pension_fund)
- [Actuarial Cost Methods (SOA)](https://www.soa.org/)
- [FASB Pension Accounting](https://www.fasb.org/)

---
**Status:** Liability management tool | **Complements:** Actuarial Cost Methods, Asset Allocation
