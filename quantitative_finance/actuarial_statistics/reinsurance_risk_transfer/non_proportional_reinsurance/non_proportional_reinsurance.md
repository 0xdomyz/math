# Non-Proportional Reinsurance

## 1. Concept Skeleton
**Definition:** Reinsurer covers losses above a retention threshold (excess-of-loss)  
**Purpose:** Transfer tail risk and catastrophic losses  
**Prerequisites:** Attachment points, loss distributions, severity modeling

## 2. Comparative Framing
| Structure | Excess-of-Loss | Proportional | Stop-Loss |
|-----------|----------------|-------------|-----------|
| **Trigger** | Loss threshold | Premium/loss % | Annual aggregate |
| **Coverage** | Layer above attachment | Pro-rata | Total loss above limit |
| **Use Case** | Tail risk | Balance relief | Portfolio protection |

## 3. Examples + Counterexamples

**Simple Example:**  
Insurer retains first $1M per occurrence, reinsurer covers $1M–$5M excess

**Failure Case:**  
Attachment too high → tail risk exposure remains unmanaged

**Edge Case:**  
Multiple layer towers with different reinsurers

## 4. Layer Breakdown
```
Non-Proportional Structure:
├─ Attachment (deductible from insurer's view)
├─ Limit (coverage amount)
├─ Rate on line (premium ÷ limit)
├─ Loss adjustment expenses
└─ Claims settlement
```

**Interaction:** Define layers → price → monitor claims

## 5. Mini-Project
Compute reinsurer payout:
```python
loss = 3000000
attachment = 1000000
limit = 4000000

reinsured_loss = max(0, min(loss - attachment, limit))
print("Reinsurer pays:", reinsured_loss)
```

## 6. Challenge Round
Common pitfalls:
- Basis risk: attachment too low for efficient pricing
- Catastrophe accumulation across layers
- Reinstatement clauses increasing effective coverage

## 7. Key References
- [Excess-of-Loss Reinsurance (Wikipedia)](https://en.wikipedia.org/wiki/Reinsurance)
- [Non-Proportional Treaty (IAA)](https://www.actuaries.org/)
- [Catastrophe Reinsurance (Swiss Re)](https://www.swissre.com/)

---
**Status:** Tail risk tool | **Complements:** Catastrophe Bonds, Proportional Reinsurance
