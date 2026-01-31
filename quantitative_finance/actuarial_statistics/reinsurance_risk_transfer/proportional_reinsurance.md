# Proportional Reinsurance

## 1. Concept Skeleton
**Definition:** Reinsurer shares losses and premiums pro-rata with ceding insurer  
**Purpose:** Proportionally reduce exposure while sharing premium income  
**Prerequisites:** Premium structures, loss distribution, contract terms

## 2. Comparative Framing
| Reinsurance Type | Proportional | Non-Proportional | Retrocession |
|------------------|-------------|-----------------|--------------|
| **Risk Sharing** | Pro-rata | Layer-based | Reinsurer protection |
| **Premium** | Shared | Limited | Reinsured premium |
| **Use Case** | Balance sheet relief | Tail risk | Risk transfer chain |

## 3. Examples + Counterexamples

**Simple Example:**  
Quota share: insurer retains 60%, reinsurer takes 40% of premiums and losses

**Failure Case:**  
Proportional treaty with inadequate expense loading

**Edge Case:**  
Surplus share with variable retention by risk size

## 4. Layer Breakdown
```
Proportional Reinsurance:
├─ Quota share: fixed % transfer
├─ Surplus share: by excess of limit
├─ Facultative: per-risk basis
└─ Settlement and experience accounting
```

**Interaction:** Define share → apply to risks → settle claims

## 5. Mini-Project
Compute shared premium and loss:
```python
premium = 100000
loss = 50000
reinsurer_share = 0.30

reinsured_premium = premium * reinsurer_share
reinsured_loss = loss * reinsurer_share
print("Premium:", reinsured_premium, "Loss:", reinsured_loss)
```

## 6. Challenge Round
Common pitfalls:
- Proportional treaty on small portfolio → high volatility
- Not adjusting expenses for smaller retained book
- Adverse selection in facultative placement

## 7. Key References
- [Proportional Reinsurance (Wikipedia)](https://en.wikipedia.org/wiki/Reinsurance)
- [Quota Share (IAA)](https://www.actuaries.org/)
- [Reinsurance Fundamentals (Swiss Re)](https://www.swissre.com/)

---
**Status:** Balance sheet tool | **Complements:** Non-Proportional, Retrocession
