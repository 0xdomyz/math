# Critical Illness Insurance

## 1. Concept Skeleton
**Definition:** Lump-sum product triggered by specified severe illnesses (cancer, MI, stroke)  
**Purpose:** Cover financial impact of major health events  
**Prerequisites:** Incidence rates, survival rates, diagnosis timing

## 2. Comparative Framing
| Insurance | Critical Illness | Disability Income | Life Insurance |
|-----------|-----------------|------------------|----------------|
| **Trigger** | Diagnosis | Disability | Death |
| **Benefit** | Lump sum | Monthly income | Death payout |
| **Recovery** | Variable | Yes | N/A |

## 3. Examples + Counterexamples

**Simple Example:**  
CI benefit of $100,000 if diagnosed with cancer age 45-65

**Failure Case:**  
Ignoring survival after diagnosis; high claims after payout

**Edge Case:**  
Pre-existing condition exclusions vs. accelerated underwriting

## 4. Layer Breakdown
```
CI Product Mechanics:
├─ Covered conditions (cancer, MI, stroke, etc.)
├─ Incidence rates by age/sex
├─ Waiting period (e.g., 30 days)
├─ Multiple illnesses once per policy
└─ Cancellation risk if recovered
```

**Interaction:** Diagnose → wait → claim → pay → monitor

## 5. Mini-Project
Compute CI premium estimate:
```python
incidence = 0.003  # 0.3% annually
benefit = 100000
admin_rate = 0.15
profit_margin = 0.10
gross_premium = (incidence * benefit * (1 + admin_rate)) / (1 - profit_margin)
print("Premium:", gross_premium)
```

## 6. Challenge Round
Common pitfalls:
- Underestimating reincidence risk
- Not adjusting incidence rates by geographic/population factors
- Overlapping claims with disability or life insurance

## 7. Key References
- [Critical Illness Insurance (Wikipedia)](https://en.wikipedia.org/wiki/Critical_illness_insurance)
- [Health Event Incidence (CDC)](https://www.cdc.gov/)
- [SOA CI Studies](https://www.soa.org/)

---
**Status:** Lump-sum health product | **Complements:** Disability Insurance, Life Insurance
