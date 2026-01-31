# Commutation & Retroactive Covers

## 1. Concept Skeleton
**Definition:** Commutation settles remaining liabilities; retroactive covers past-period losses  
**Purpose:** Finalize claims, manage tail risk, and transfer historical exposures  
**Prerequisites:** Outstanding claims, historical exposure data, settlement negotiations

## 2. Comparative Framing
| Concept | Commutation | Retroactive Cover | Normal Treaty |
|---------|------------|-------------------|--------------|
| **Timing** | End of relationship | Past periods | Ongoing |
| **Liability** | Settled upfront | Coverage begins backward | Forward-looking |
| **Risk** | Finality | Tail risk transfer | Ongoing |

## 3. Examples + Counterexamples

**Simple Example:**  
Commutation: insurer pays lump sum to close outstanding claims

**Failure Case:**  
Retroactive cover mispriced without adequate historical data

**Edge Case:**  
Claims-made retroactive covering latency (e.g., asbestos)

## 4. Layer Breakdown
```
Commutation & Retroactive:
├─ Identify outstanding claims
├─ Actuarial reserve estimate
├─ Commutation settlement value
├─ Retroactive cover attachment point
└─ Implementation and monitoring
```

**Interaction:** Measure reserves → negotiate settlement → activate retro

## 5. Mini-Project
Estimate commutation value:
```python
reserves = 500000
discount_rate = 0.05
settlement_discount = 0.10

commutation_value = reserves * (1 - settlement_discount) / (1 + discount_rate)
print("Commutation value:", commutation_value)
```

## 6. Challenge Round
Common pitfalls:
- Significantly under/overestimating tail reserves
- Inadequate definition of covered periods in retroactive
- Moral hazard if retroactive too generous

## 7. Key References
- [Commutation (Insurance) (Wikipedia)](https://en.wikipedia.org/wiki/Commutation)
- [Retroactive Coverage (IAA)](https://www.actuaries.org/)
- [Claims Settlement (SOA)](https://www.soa.org/)

---
**Status:** Tail management tool | **Complements:** Reserves, Reinsurance Treaties
