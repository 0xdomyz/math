# Theta (θ)

## Concept Skeleton
**Definition:** First-order partial derivative of option price with respect to time; measures the rate of change in option value as time passes (time decay)  
**Purpose:** Quantify temporal decay of option value; monitor erosion of premium; identify time decay profit/loss in positions  
**Prerequisites:** Partial derivatives, option pricing, time value, decay concepts

## Comparative Framing
| Aspect | Theta | Delta | Gamma | Vega |
|--------|-------|-------|-------|------|
| **Measure** | ∂V/∂T (time) | ∂V/∂S (spot) | ∂²V/∂S² | ∂V/∂σ (vol) |
| **Call Sign** | Usually < 0 (decay) | 0 to +1 | Always > 0 | Always > 0 |
| **Put Sign** | Usually > 0 (wait value) | -1 to 0 | Always > 0 | Always > 0 |
| **Most Negative** | OTM call, short maturity | N/A | ATM, short maturity | N/A |
| **Hedging Vehicle** | Calendar spread | Underlying | Options | Vol derivatives |
| **Time Dependency** | Accelerates as T → 0 | N/A | Accelerates as T → 0 | Linear in √T |

## Examples + Counterexamples
**Simple Example:**  
OTM call 1 day to expiry, worth $0.01: Theta ≈ -$0.01/day; expires worthless tomorrow

**Practical Case:**  
ATM straddle seller: Short theta benefits from daily time decay; collects premium as both calls and puts lose value

**Counterintuitive Case:**  
Deep ITM put: Positive theta; benefits from time decay (wait value decreases, intrinsic value + decay benefit = erosion of discount)

**Edge Case:**  
Dividend-paying stock: Call theta can be positive pre-dividend (dividend yield benefit > decay); changes sign

## Layer Breakdown
```
Theta Concept & Dynamics:
├─ Mathematical Definition:
│   ├─ Theta: θ = ∂V/∂T (sometimes denoted as -∂V/∂t to show decay)
│   ├─ Daily theta: θ/365 (annualized theta divided by trading days)
│   ├─ Interpretation: Option value changes by θ dollars per year (or θ/365 per day)
│   └─ Conventionally: Theta usually negative for long calls (decay erodes value)
├─ Black-Scholes Formula:
│   ├─ Call theta = -S×N'(d1)×σ/(2√T) - r×K×e^{-rT}×N(d2)
│   ├─ Put theta = -S×N'(d1)×σ/(2√T) + r×K×e^{-rT}×N(-d2)
│   ├─ Where N'(d1) = normal PDF
│   ├─ Properties:
│   │   ├─ Usually negative for calls (time decay)
│   │   ├─ Usually positive for ITM puts (intrinsic value dominates)
│   │   ├─ Accelerates near expiry (√T → 0)
│   │   └─ Zero for ATM options at very long maturity (time effect minimal)
├─ Gamma-Theta Relationship:
│   ├─ Theta ≈ -γ×S²×σ²/2 - r×V (approximation; gamma decay + interest cost)
│   ├─ For ATM option: γ effect dominates; theta ≈ -γ×S²×σ²/2
│   ├─ Long gamma → negative theta (pay for convexity)
│   ├─ Short gamma → positive theta (collect for risk)
│   └─ Breakeven: At what realized vol does gamma P&L offset theta loss?
├─ Time Dependencies:
│   ├─ Acceleration: |θ| increases as T → 0 (√T in denominator)
│   ├─ Scaling: √T term → theta ∝ 1/√T near expiry
│   ├─ ATM vs OTM: ATM theta most negative (highest time value at risk)
│   ├─ ITM options: Decay slower than OTM (less time value)
│   └─ Deep ITM: Theta → intrinsic value (little decay left)
├─ Hedging Theta:
│   ├─ Long theta strategies: Sell premium (short straddles, call spreads)
│   ├─ Calendar spreads: Buy long-dated, sell short-dated → harvest theta differential
│   ├─ Cost: Long theta = short gamma (negative convexity)
│   ├─ Monitoring: Daily theta tracking; rebalance if position theta drifts
│   └─ Greeks change: Theta constantly changes with spot, vol, time
└─ Practical Metrics:
    ├─ Theta decay curve: Shows θ vs time; accelerates exponentially
    ├─ Dollar theta: Actual daily decay in position value
    ├─ Percentage theta: θ as % of option price (decay rate)
    ├─ Theta per gamma: Efficiency of theta collection relative to gamma risk
    └─ Rolling strategies: Extend theta by rolling positions forward
```

**Interaction:** Time passes → option loses time value → theta decay → selling premium profitable if vol stays low

## Challenge Round
When does theta analysis fail?
- Dividend drops: Ex-dividend date → sudden price drop; call theta can spike negative
- Earnings announcements: Volatility jumps break theta model (BS assumes constant vol)
- Interest rate changes: Theta = f(r); rate changes affect call theta calculation
- Non-calendar spreads: Rolling positions changes moneyness; theta is path-dependent
- Bid-ask spread: Theta profit can be erased by transaction costs on frequent rehedging

## Key References
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Natenberg - Option Volatility & Pricing (Chapter 12)](https://www.amazon.com/Option-Volatility-Pricing-Advanced-Strategies/dp/1557784124)
- [Wilmott - Quantitative Finance (Chapter 7)](https://www.paulwilmott.com)

---
**Status:** Time decay metric | **Complements:** Gamma, Vega, Calendar Spreads, Greeks Dynamics
