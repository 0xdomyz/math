# Option Pricing Basics

## Concept Skeleton
**Definition:** Foundation for valuing financial derivatives; establishes framework connecting spot price, strike price, time value, and volatility to determine fair value of options  
**Purpose:** Understand intrinsic vs time value; establish pricing bounds (no-arbitrage); introduce concepts used in advanced models; practical valuation framework  
**Prerequisites:** Financial derivatives, option types (call/put), payoff diagrams, no-arbitrage principle, interest rates, spot-forward relationship

## Comparative Framing
| Concept | American Option | European Option | Forward Contract | Futures |
|---------|-----------------|-----------------|-----------------|---------|
| **Exercise** | Any time to expiry | Only at expiry | Obligation at maturity | Daily settlement |
| **Early Exercise** | Possible | Not possible | N/A | N/A |
| **Value** | ≥ European | Baseline | Linear in spot | Marked-to-market |
| **Pricing** | No simple formula | Closed forms available | F = S × e^(rT) | Similar to forward |
| **Holder Optionality** | Yes | Limited | None | Limited |
| **Typical Use** | Stock options | Easier to analyze theoretically | Hedging | Speculation |
| **Complexity** | Higher | Moderate | Simple | Moderate |

## Examples + Counterexamples
**Simple Example:**  
Stock at $100, call option with strike $105, 1 year to expiry. Intrinsic value = max(100-105, 0) = $0. Time value encodes probability stock rises above $105.

**Arbitrage Bound:**  
Call value C must satisfy: max(S-K, 0) ≤ C ≤ S. If C > S, arbitrage: buy stock, sell call. If C < max(S-K, 0), arbitrage: buy call, immediate exercise if positive intrinsic.

**Put-Call Parity:**  
C - P = S - Ke^(-rT). If violated, arbitrage. Example: S=$50, K=$50, r=5%, T=1, C=$3, P=$1. Check: 3-1=2 vs 50-50e^(-0.05)≈2.44. Mispriced; arbitrage opportunity.

**Time Decay:**  
At-the-money option loses value as expiration approaches (theta decay), all else equal. Far out-of-the-money: minimal time value, negligible decay.

**Early Exercise Value:**  
American call on non-dividend stock: never optimal to exercise early (maintain flexibility). American call with dividends: may exercise just before ex-dividend date.

## Layer Breakdown
```
Option Pricing Foundations:

├─ Payoff Structures:
│  ├─ Call Option Payoff:
│  │   ├─ At expiry: max(S_T - K, 0)
│  │   ├─ Value components: Intrinsic + Time Value
│  │   ├─ Intrinsic: max(S - K, 0) (immediate exercise value)
│  │   └─ Time Value: Option Price - Intrinsic
│  ├─ Put Option Payoff:
│  │   ├─ At expiry: max(K - S_T, 0)
│  │   ├─ Intrinsic: max(K - S, 0)
│  │   └─ Time value (similar decay as calls)
│  └─ Portfolio Combinations:
│      ├─ Straddle: Buy call + put (bet on volatility)
│      ├─ Strangle: OTM call + OTM put (cheaper straddle)
│      ├─ Spread: Long call + short call (limited profit/loss)
│      └─ Collar: Long stock + long put + short call (downside protection)
├─ No-Arbitrage Bounds:
│  ├─ Call Bounds:
│  │   ├─ Lower: max(S - Ke^(-rT), 0) ≤ C
│  │   ├─ Upper: C ≤ S (must not exceed stock value)
│  │   └─ For American: C_American ≥ max(S - K, 0)
│  ├─ Put Bounds:
│  │   ├─ Lower: max(Ke^(-rT) - S, 0) ≤ P
│  │   ├─ Upper: P ≤ Ke^(-rT)
│  │   └─ For American: P_American ≥ max(K - S, 0)
│  └─ Arbitrage Strategies:
│      ├─ Conversion: Long stock + long put + short call
│      │   (Creates synthetic risk-free bond)
│      ├─ Reversal: Short stock + short put + long call
│      └─ Box Spread: Call spread + put spread (synthesizes bond)
├─ Put-Call Parity (European):
│  ├─ C - P = S - Ke^(-rT)
│  ├─ Derivation: Construct two portfolios with same payoff
│  │   ├─ Portfolio A: Long call + cash Ke^(-rT)
│  │   ├─ Portfolio B: Long stock + long put
│  │   └─ Both worth max(S_T, K) at expiry
│  ├─ Rearrangement: P = C - S + Ke^(-rT)
│  ├─ Implications: Call and put are related; can't price independently
│  └─ American Exception: C_Am - P_Am ≤ S - Ke^(-rT) (inequality, not equality)
├─ Greeks & Sensitivities:
│  ├─ Delta (∂C/∂S): Change in option price per $1 stock move
│  │   ├─ Range: 0 to 1 for calls, -1 to 0 for puts
│  │   ├─ Interpretation: Equivalent shares of stock
│  │   └─ Hedging: Delta-neutral portfolio
│  ├─ Gamma (∂²C/∂S²): Delta sensitivity to stock price
│  │   ├─ Highest at-the-money
│  │   ├─ Increases near expiry
│  │   └─ Risk for hedgers: Delta becomes inaccurate
│  ├─ Theta (∂C/∂t): Time decay (usually negative for buyer)
│  │   ├─ Long calls/puts: Lose time value
│  │   ├─ Near expiry: Steep decay
│  │   └─ Short gamma = earn theta
│  ├─ Vega (∂C/∂σ): Volatility sensitivity
│  │   ├─ Long options: Positive vega (profit from vol increases)
│  │   ├─ At-the-money: Highest vega
│  │   └─ Volatility traders main concern
│  └─ Rho (∂C/∂r): Interest rate sensitivity
│      ├─ Weaker effect than other Greeks
│      ├─ Long options: Positive rho for calls, negative for puts
│      └─ More important for bonds/long-dated options
├─ Factors Affecting Option Value:
│  ├─ Stock Price (S): ↑S → ↑Call, ↓Put
│  ├─ Strike Price (K): ↑K → ↓Call, ↑Put
│  ├─ Time to Expiry (T): Usually ↑T → ↑Option value (more time = more optionality)
│  │   Exception: Deep in-the-money put value might decrease
│  ├─ Volatility (σ): ↑σ → ↑Call value, ↑Put value
│  │   (Both benefit from increased uncertainty)
│  ├─ Risk-free Rate (r): ↑r → ↑Call, ↓Put
│  │   (Higher discount rate affects forward price)
│  ├─ Dividends (D): ↑D → ↓Call, ↑Put
│  │   (Reduces expected stock price at expiry)
│  └─ Early Exercise (American): Adds value due to flexibility
└─ Early Exercise Decisions:
   ├─ American Call (no dividends): Never optimal early
   │   (Intrinsic max(S-K,0) < option value due to time value)
   ├─ American Call (with dividends): May exercise before ex-date
   ├─ American Put: Always has early exercise value
   │   (Can lock in intrinsic max(K-S, 0))
   ├─ Trigger: Exercise if S - D*e^(rT) > C_continuation
   └─ Binomial model easily captures this optionality
```

**Interaction:** Spot price determines intrinsic value; volatility and time drive time value; interest rates affect discounting and forward pricing.

## Challenge Round
1. **Early Exercise:** American call on dividend-paying stock with D paid just before expiry. When is early exercise optimal? Derive condition.

2. **Put-Call Parity Violations:** Find or construct real market prices violating parity. Execute arbitrage; calculate risk-free profit.

3. **Straddle Pricing:** Buy call + put at same strike. How does value vary with volatility? What's breakeven on moves?

4. **Volatility Surface:** Different strikes/expirations have different implied volatilities. Plot surface; identify "smile" or "skew."

5. **Approximations:** For small T, use Taylor expansion of Black-Scholes. First-order approximation: C ≈ max(S-K,0) + (time value term). Compare to exact.

## Key References
- [Hull, Options, Futures, and Other Derivatives (Chapter 9-10)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000006649)
- [Black-Scholes Model (Wikipedia)](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Put-Call Parity (Investopedia)](https://www.investopedia.com/terms/p/putcallparity.asp)
- [Option Greeks Explained](https://www.investopedia.com/terms/g/greeks.asp)

---
**Status:** Foundation derivative pricing | **Complements:** Black-Scholes Model, Binomial Trees, Implied Volatility, Risk Measures
