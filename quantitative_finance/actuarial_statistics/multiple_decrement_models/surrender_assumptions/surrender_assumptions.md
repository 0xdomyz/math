# Surrender Assumptions

## 1. Concept Skeleton
**Definition:** Assumptions on policyholders taking cash value and terminating coverage  
**Purpose:** Forecast cash flows and liquidity needs for surrender options  
**Prerequisites:** Policy values, lapse modeling, cash flow projection

## 2. Comparative Framing
| Concept | Lapse | Surrender | Partial Withdrawal |
|--------|------|-----------|--------------------|
| **Cash Value** | Not necessarily | Yes | Yes |
| **Effect** | Termination | Termination + payout | Reduced account value |
| **Modeling** | Rates | Rates + value | Utilization rate |

## 3. Examples + Counterexamples

**Simple Example:**  
Surrender spike after guaranteed rate period ends

**Failure Case:**  
Ignoring surrender charges in behavior modeling

**Edge Case:**  
Dynamic behavior driven by account value vs guarantee

## 4. Layer Breakdown
```
Surrender Modeling:
├─ Base surrender curve
├─ Charge schedule and incentives
├─ Dynamic response to market rates
└─ Impact on liquidity and reserves
```

**Interaction:** Set base → adjust for incentives → project cash outflows

## 5. Mini-Project
Compute surrender payout after charge:
```python
account = 10000
charge = 0.05
payout = account * (1 - charge)
print(payout)
```

## 6. Challenge Round
Common pitfalls:
- Overlooking partial withdrawals
- Treating surrender and lapse as identical
- Underestimating policyholder option value

## 7. Key References
- [Surrender Value (Wikipedia)](https://en.wikipedia.org/wiki/Surrender_value)
- [Lapse (Insurance) (Wikipedia)](https://en.wikipedia.org/wiki/Lapse_(insurance))
- [SOA Surrender Studies](https://www.soa.org/)

---
**Status:** Liquidity-sensitive assumption | **Complements:** Lapse Rates, Profit Testing
