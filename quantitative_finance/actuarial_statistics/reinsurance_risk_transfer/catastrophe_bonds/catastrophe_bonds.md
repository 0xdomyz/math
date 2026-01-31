# Catastrophe Bonds

## 1. Concept Skeleton
**Definition:** Insurance-linked securities transferring catastrophic risk to capital markets  
**Purpose:** Diversify insurance risk beyond traditional reinsurance  
**Prerequisites:** Catastrophe modeling, bond pricing, trigger mechanisms

## 2. Comparative Framing
| Instrument | Cat Bond | Reinsurance | Insurance | Credit Bond |
|-----------|----------|------------|-----------|------------|
| **Trigger** | Catastrophe | Claims | Policy events | Credit event |
| **Investor** | Capital market | Reinsurers | N/A | Bondholders |
| **Return** | Premium + spread | Underwriting margin | N/A | Coupon |

## 3. Examples + Counterexamples

**Simple Example:**  
$500M bond issued; coupon suspended if earthquake losses exceed $1B

**Failure Case:**  
Poorly defined trigger leading to disputes or moral hazard

**Edge Case:**  
Parametric trigger (magnitude/intensity) vs. indemnity trigger

## 4. Layer Breakdown
```
Cat Bond Structure:
├─ Originating insurer/sponsor
├─ SPV (special purpose vehicle)
├─ Trigger mechanism (parametric/indemnity)
├─ Investor principal at risk
└─ Insurance-linked securities
```

**Interaction:** Issue bond → collect premiums → trigger if loss → settle

## 5. Mini-Project
Estimate cat bond risk transfer:
```python
bond_size = 500000000
trigger_loss = 1000000000
expected_loss = 100000000
pricing_spread = 0.04

coupon = bond_size * (expected_loss / trigger_loss + pricing_spread)
print("Annual coupon:", coupon)
```

## 6. Challenge Round
Common pitfalls:
- Basis risk: trigger doesn't match actual losses
- Moral hazard from sponsor incentives
- Oversimplifying catastrophe model assumptions

## 7. Key References
- [Catastrophe Bond (Wikipedia)](https://en.wikipedia.org/wiki/Catastrophe_bond)
- [Cat Securities (Swiss Re)](https://www.swissre.com/)
- [Insurance-Linked Securities (IAA)](https://www.actuaries.org/)

---
**Status:** Capital market risk transfer | **Complements:** Non-Proportional Reinsurance, Tail Risk
