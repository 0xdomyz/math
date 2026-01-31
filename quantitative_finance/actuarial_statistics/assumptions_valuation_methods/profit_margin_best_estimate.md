# Profit Margin & Best Estimate

## 1. Concept Skeleton
**Definition:** Assumptions embedded in pricing for profit vs. best-estimate assumptions for reserves  
**Purpose:** Balance profitability, conservatism, and competitive pricing  
**Prerequisites:** Assumption ranges, regulatory guidance, market conditions

## 2. Comparative Framing
| Approach | Profit Margin | Best Estimate | Conservative |
|----------|---------------|---------------|--------------|
| **Bias** | Favorable | Neutral | Unfavorable |
| **Use** | Pricing | IFRS 17 | Statutory |
| **Risk** | Over-optimistic | Fair | Safe |

## 3. Examples + Counterexamples

**Simple Example:**  
Best-estimate mortality = experience A/E; pricing mortality = 90% of best-estimate

**Failure Case:**  
Excessive margins leading to uncompetitive pricing

**Edge Case:**  
Onerous contract detection when assumptions become unfavorable

## 4. Layer Breakdown
```
Profit vs. Best-Estimate:
├─ Define best-estimate assumptions
├─ Layer profit/safety margins
├─ Monitor actual experience
├─ Unlock if actual diverges
└─ Adjust ongoing pricing
```

**Interaction:** Best-estimate → add margin → price → monitor

## 5. Mini-Project
Model profit margin impact:
```python
best_estimate_cost = 100
best_estimate_return = 20
profit_margin_pct = 0.15

gross_premium = best_estimate_cost * (1 + profit_margin_pct)
profit = gross_premium - best_estimate_cost
print("Gross premium:", gross_premium, "Profit:", profit)
```

## 6. Challenge Round
Common pitfalls:
- Double-layering margins (pricing + reserving) biases earnings
- Over-margins lose competitiveness; under-margins risk loss
- Not updating margins as experience changes

## 7. Key References
- [Profit Margin (Wikipedia)](https://en.wikipedia.org/wiki/Profit_margin)
- [Best Estimate (IFRS 17)](https://www.ifrs.org/)
- [Solvency II (EIOPA)](https://www.eiopa.europa.eu/)

---
**Status:** Pricing-reserving bridge | **Complements:** Assumption Setting, Risk Management
