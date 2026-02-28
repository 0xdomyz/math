# Repo Rates & Dividends

## Concept Skeleton
**Definition:** Financing rate (repo) and cash distributions (dividends) affecting forward prices  
**Purpose:** Adjust drift and forward price in option pricing and Monte Carlo simulation  
**Prerequisites:** Forward pricing, carry costs, dividend yield

## Comparative Framing
| Input | Repo Rate | Dividend Yield | Risk-Free Rate |
|---|---|---|---|
| **Role** | Funding cost | Cash outflow | Discounting |
| **Impact on Forward** | Lowers carry if higher | Lowers forward | Raises forward |
| **Typical in** | Equity financing | Equity indices | Discounting |

## Examples + Counterexamples
**Simple Example:**  
$S_0=100$, $r=5%$, dividend yield $q=2%$ → $F=S_0 e^{(r-q)T}$.

**Failure Case:**  
Ignoring discrete dividends → mispriced short-dated options around ex-dividend dates.

**Edge Case:**  
Special dividends (large one-off) can violate smooth yield assumptions.

## Layer Breakdown
```
Carry Adjustment:
├─ Inputs:
│   ├─ Risk-free rate r
│   ├─ Dividend yield q (continuous) or discrete dividends
│   └─ Repo/financing rate r_repo
├─ Forward Price:
│   ├─ Continuous yield: F = S0 e^{(r-q)T}
│   ├─ With repo: F = S0 e^{(r-r_repo-q)T}
│   └─ Discrete dividends: F = (S0 - PV(dividends)) e^{rT}
├─ Drift for MC:
│   ├─ Risk-neutral drift: (r - q - r_repo)
│   └─ Use in GBM simulation
└─ Calibration:
    ├─ Estimate q from dividend futures or index data
    ├─ Repo from financing markets
    └─ Validate against forward prices
```

**Interaction:** Estimate carry inputs → adjust forward and drift → price options

## Challenge Round
**Q1:** Why does dividend yield reduce call prices?  
**A1:** Dividends lower forward prices and reduce expected terminal spot, decreasing call value.

**Q2:** When is repo rate relevant?  
**A2:** For financed positions or securities lending; it alters carry and forward pricing.

**Q3:** How do discrete dividends affect early exercise?  
**A3:** Calls may be optimally exercised just before ex-dividend to capture cash flow.

**Q4:** Can dividend yield be negative?  
**A4:** Effective negative yield can occur with short rebates or special financing conditions.

## Key References
- [Dividend yield](https://en.wikipedia.org/wiki/Dividend_yield)  
- [Forward price](https://en.wikipedia.org/wiki/Forward_price)

---
**Status:** Carry adjustments for forwards | **Complements:** Interest rate curve, implied volatility
