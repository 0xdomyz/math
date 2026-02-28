# Interest Rate Derivatives

## Concept Skeleton
**Definition:** Financial contracts whose value depends on future interest rates, including caps, floors, swaptions, and bonds, priced under term structure models  
**Purpose:** Hedge interest rate risk, speculate on rate movements, manage duration exposure in portfolios  
**Prerequisites:** Term structure models, bond math, Black formula for caps/floors, LIBOR market model

## Comparative Framing
| Product | Cap | Floor | Swaption | Interest Rate Swap |
|---------|-----|-------|----------|-------------------|
| **Payoff** | Max(L-K,0) per period | Max(K-L,0) per period | Option to enter swap | Fixed vs floating exchange |
| **Use Case** | Protect against rising rates | Protect against falling rates | Hedge future swap entry | Convert fixed/floating exposure |
| **Pricing Model** | Black '76 formula (caplet sum) | Black '76 formula (floorlet sum) | Black formula or LMM | Discount cash flows to PV |
| **Volatility Input** | Cap volatility surface | Floor volatility surface | Swaption vol matrix | N/A (deterministic) |

## Examples + Counterexamples
**Simple Example:**  
3-year cap on 3M LIBOR, strike 3%, notional $10M → quarterly payoffs Max(LIBOR_t - 3%, 0) × 0.25 × $10M, priced as sum of 12 caplets

**Failure Case:**  
Using Black-Scholes for swaptions with constant volatility → ignores term structure dynamics, volatility smile → SABR or LMM required for accuracy

**Edge Case:**  
Negative interest rates (EUR, JPY 2015-2020): Black formula breaks down (assumes lognormal rates) → shifted lognormal or normal model needed

## Layer Breakdown
```
Interest Rate Derivatives:
├─ Foundational Products:
│   ├─ Interest Rate Swap (IRS):
│   │   ├─ Fixed Leg: ∑ K × τᵢ × DF(tᵢ) (K = fixed rate)
│   │   ├─ Floating Leg: ∑ L(tᵢ₋₁,tᵢ) × τᵢ × DF(tᵢ) (L = LIBOR/SOFR)
│   │   ├─ Swap Value: V_swap = V_float - V_fixed (receiver pays fixed)
│   │   ├─ Par Swap Rate: K such that V_swap = 0 at inception
│   │   └─ Use: Convert floating exposure to fixed (or vice versa)
│   ├─ Forward Rate Agreement (FRA):
│   │   ├─ Payoff: (L - K) × τ × N × DF(T) (settled at T)
│   │   ├─ Forward Rate: F(t,T,T+τ) = [DF(T)/DF(T+τ) - 1]/τ
│   │   └─ Use: Lock in future borrowing/lending rate
│   └─ Zero-Coupon Bond:
│       ├─ Price: P(t,T) = DF(T) = e^(-r(T-t)(T-t)) (continuous)
│       └─ Yield: y = -ln(P)/T (zero rate)
├─ Caps and Floors:
│   ├─ Interest Rate Cap:
│   │   ├─ Definition: Portfolio of caplets, each pays Max(L-K,0) × τ × N
│   │   ├─ Caplet Pricing (Black '76):
│   │   │   Caplet(T) = N × τ × DF(T) × [F × Φ(d₁) - K × Φ(d₂)]
│   │   │   where F = forward LIBOR, d₁ = [ln(F/K) + ½σ²T]/(σ√T), d₂ = d₁ - σ√T
│   │   ├─ Cap Value: Sum of all caplet values
│   │   ├─ Cap Volatility Surface: σ_cap(K,T) varies by strike and maturity
│   │   └─ Use: Protection against rising rates (borrower buys cap)
│   ├─ Interest Rate Floor:
│   │   ├─ Floorlet: Max(K-L,0) × τ × N per period
│   │   ├─ Pricing: Similar to caplet, Φ(-d₂) and Φ(-d₁) in formula
│   │   └─ Use: Protection against falling rates (lender buys floor)
│   └─ Collar:
│       ├─ Long Cap + Short Floor (or vice versa)
│       ├─ Zero-Cost Collar: Choose strikes so cap premium = floor premium
│       └─ Limits interest rate exposure to [K_floor, K_cap] range
├─ Swaptions:
│   ├─ Definition: Option to enter interest rate swap at future date
│   │   ├─ Payer Swaption: Right to pay fixed (receive floating)
│   │   ├─ Receiver Swaption: Right to receive fixed (pay floating)
│   │   └─ European vs Bermudan: Single vs multiple exercise dates
│   ├─ Pricing (Black Formula):
│   │   V_payer = A × DF(T₀) × [S × Φ(d₁) - K × Φ(d₂)]
│   │   where S = forward swap rate, K = strike, A = annuity factor
│   │   A = ∑ τᵢ × DF(tᵢ) (PV of $1 per period)
│   ├─ Swaption Volatility Matrix:
│   │   ├─ Rows: Option expiry (1Y, 2Y, 5Y, 10Y, ...)
│   │   ├─ Columns: Swap tenor (1Y, 5Y, 10Y, 30Y, ...)
│   │   └─ Notation: 2Y5Y swaption = option expiring in 2Y on 5Y swap
│   ├─ Use Cases:
│   │   ├─ Hedge callable bonds (issuer has option to prepay)
│   │   ├─ Monetize rate views without immediate swap commitment
│   │   └─ Portfolio immunization strategies
│   └─ Advanced Pricing: LIBOR Market Model for Bermudan swaptions
├─ Term Structure Models:
│   ├─ Short-Rate Models:
│   │   ├─ Vasicek: dr = a(b-r)dt + σdW (mean-reverting, Gaussian)
│   │   ├─ Cox-Ingersoll-Ross (CIR): dr = a(b-r)dt + σ√r dW (non-negative)
│   │   ├─ Hull-White: dr = [θ(t) - ar]dt + σdW (time-dependent mean reversion)
│   │   └─ Calibration: Fit to current yield curve, then price derivatives
│   ├─ LIBOR Market Model (LMM):
│   │   ├─ Model forward LIBOR rates directly: dLᵢ/Lᵢ = σᵢdWᵢ
│   │   ├─ Advantages: Market-consistent (match cap/floor prices), lognormal rates
│   │   ├─ Calibration: Match cap volatilities across strikes and maturities
│   │   └─ Simulation: Monte Carlo for exotic derivatives, Bermudan swaptions
│   └─ Heath-Jarrow-Morton (HJM):
│       ├─ Model forward rate curve evolution: df(t,T) = α(t,T)dt + σ(t,T)dW
│       ├─ No-arbitrage drift: α(t,T) = σ(t,T)∫ᵗᵀ σ(t,s)ds
│       └─ Flexible but high-dimensional (infinite-dimensional system)
├─ Convexity Adjustments:
│   ├─ Timing Mismatch: Payment at T₂, fixing at T₁ (T₁ < T₂)
│   │   ├─ Adjustment: E^T₂[L(T₁,T₂)] ≠ F(0,T₁,T₂) (forward rate)
│   │   └─ Formula: Adjusted forward = F × (1 + ½σ²T₁τ/(1+Fτ))
│   ├─ Constant Maturity Swap (CMS):
│   │   ├─ Floating leg pays swap rate (not LIBOR)
│   │   └─ Requires convexity adjustment due to nonlinearity
│   └─ In-Arrears Swaps: Fixing and payment same date → convexity correction
└─ Risk Management:
    ├─ Duration: ∂P/∂y sensitivity to parallel yield shift
    ├─ Key Rate Duration: Sensitivity to specific maturity buckets
    ├─ DV01 (Dollar Value of 1bp): Change in value for 1bp rate change
    └─ Vega: Sensitivity to volatility changes (caps, swaptions)
```

**Interaction:** Construct yield curve → Calculate forward rates → Price caplets/swaptions using Black '76 → Sum to get cap/swaption value

## Challenge Round
When does Black '76 formula fail for interest rate derivatives?
- **Negative rates:** Lognormal assumption breaks (ln(negative) undefined) → shifted lognormal or normal (Bachelier) model required
- **Volatility smile:** Flat vol assumption inaccurate, especially for deep OTM/ITM → SABR model captures smile dynamics
- **Bermudan swaptions:** Multiple exercise dates require Monte Carlo + optimal stopping (Longstaff-Schwartz) or tree methods
- **Correlation structure:** Multi-factor products (e.g., CMS spread options) need full LIBOR Market Model with factor correlations
- **Long-dated products:** Term structure of volatility matters, constant vol inadequate → time-dependent volatility σ(t,T)

Modern approaches: SABR for smile, LMM for complex path-dependent payoffs, Hull-White for callable bonds needing analytical tractability.

## Key References
- [Brigo & Mercurio (2006) Interest Rate Models – Theory and Practice](https://link.springer.com/book/10.1007/978-3-540-34604-3) - Comprehensive LIBOR Market Model
- [Rebonato (2004) Volatility and Correlation in Interest Rate Derivatives](https://www.wiley.com/en-us/Volatility+and+Correlation%3A+The+Perfect+Hedger+and+the+Fox%2C+2nd+Edition-p-9780470091395) - Swaption volatility surface modeling
- [Hull (2018) Options, Futures, and Other Derivatives, Ch. 32](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) - Interest rate derivatives overview
- [Andersen & Piterbarg (2010) Interest Rate Modeling](https://www.atlanticfinancial.com/interest-rate-modeling-volumes-1-2-and-3-lech-grzelak-cornelis-oosterlee) - Advanced topics including CVA for rates

---
**Status:** Core fixed income derivative class | **Complements:** Black-Scholes, Monte Carlo, Term Structure Models
