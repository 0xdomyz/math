# Retrocession

## 1. Concept Skeleton
**Definition:** Reinsurance of reinsurance; a reinsurer cedes risk to a retrocedent  
**Purpose:** Allow reinsurers to manage their own exposure and capital  
**Prerequisites:** Reinsurance contracts, layered risk transfer

## 2. Comparative Framing
| Party | Primary Insurer | Reinsurer | Retrocedent |
|-------|-----------------|-----------|------------|
| **Role** | Sells to policyholders | Buys from insurer | Buys from reinsurer |
| **Exposure** | Original | Secondary | Tertiary |
| **Use** | Manage portfolio | Manage portfolio | Manage capital |

## 3. Examples + Counterexamples

**Simple Example:**  
Ceding insurer → reinsurer (primary treaty) → retrocedent (secondary treaty)

**Failure Case:**  
Excessive layering increases complexity and credit risk

**Edge Case:**  
Retrocession aggregating multiple underlying treaties

## 4. Layer Breakdown
```
Retrocession Chain:
├─ Primary insurance policy
├─ Primary reinsurance treaty
├─ Secondary (retro) treaty
└─ Potential tertiary structures
```

**Interaction:** Primary risk flows through layers

## 5. Mini-Project
Track risk flow through retrocession:
```python
primary_loss = 100000
primary_reins_recovery = 60000
retro_recovery = max(0, primary_reins_recovery - 30000)

net_to_primary = primary_loss - primary_reins_recovery
net_to_reinsurer = primary_reins_recovery - retro_recovery
print("Primary net:", net_to_primary, "Reinsurer net:", net_to_reinsurer)
```

## 6. Challenge Round
Common pitfalls:
- Credit concentration across layers
- Complexity leading to pricing errors
- Inadequate documentation of layered structures

## 7. Key References
- [Retrocession (Wikipedia)](https://en.wikipedia.org/wiki/Reinsurance#Retrocession)
- [Reinsurance Chains (IAA)](https://www.actuaries.org/)
- [Market Practice (Lloyd's)](https://www.lloyds.com/)

---
**Status:** Capital management tool | **Complements:** Reinsurance, Credit Risk
