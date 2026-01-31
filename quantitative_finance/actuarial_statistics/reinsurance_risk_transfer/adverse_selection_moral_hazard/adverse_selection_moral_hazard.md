# Adverse Selection & Moral Hazard

## 1. Concept Skeleton
**Definition:** Adverse selection: high-risk individuals self-select into insurance; moral hazard: insured behavior changes post-purchase  
**Purpose:** Understand behavioral risks and design underwriting/claims controls  
**Prerequisites:** Information asymmetry, behavioral economics, incentive structures

## 2. Comparative Framing
| Risk Type | Adverse Selection | Moral Hazard | Systemic Risk |
|----------|------------------|--------------|--------------|
| **Timing** | Before purchase | After purchase | Market-wide |
| **Driver** | Hidden information | Hidden action | Correlation |
| **Mitigation** | Underwriting | Claims management | Capital adequacy |

## 3. Examples + Counterexamples

**Simple Example:**  
Smokers more likely to buy life insurance; claim rates higher than population average

**Failure Case:**  
Ignoring moral hazard in workers' comp → excessive claim inflation

**Edge Case:**  
Ex-ante moral hazard: people take risks because insured

## 4. Layer Breakdown
```
Adverse Selection & Moral Hazard:
├─ Information asymmetry
├─ Underwriting to reveal risk
├─ Incentive alignment (deductibles, copays)
├─ Claims management and investigation
└─ Monitoring and predictive modeling
```

**Interaction:** Inform → underwrite → monitor → adjust

## 5. Mini-Project
Compare claim rates by selection:
```python
selected_claims = 120
selected_count = 800
population_claims = 100
population_count = 10000

selected_rate = selected_claims / selected_count
pop_rate = population_claims / population_count
print("Selection ratio:", selected_rate / pop_rate)
```

## 6. Challenge Round
Common pitfalls:
- Conflating adverse selection with moral hazard
- Over-relying on underwriting to eliminate adverse selection
- Ignoring long-tail moral hazard (e.g., lifestyle changes post-claim)

## 7. Key References
- [Adverse Selection (Wikipedia)](https://en.wikipedia.org/wiki/Adverse_selection)
- [Moral Hazard (Wikipedia)](https://en.wikipedia.org/wiki/Moral_hazard)
- [Insurance Economics (SOA)](https://www.soa.org/)

---
**Status:** Behavioral risk framework | **Complements:** Underwriting, Claims Management
