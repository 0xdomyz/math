# Vanilla Options

## Concept Skeleton
**Definition:** Standard European or American call/put options with no exotic features; most liquid exchange-traded contracts  
**Purpose:** Speculation on directional moves, hedging stock positions, income generation (covered calls), basis for complex strategies  
**Prerequisites:** Option payoff functions, intrinsic vs time value, moneyness concepts, exercise mechanics

## Comparative Framing
| Feature | European Call/Put | American Call/Put | Digital (Binary) | Exotic (Asian, Barrier) |
|---------|-------------------|-------------------|------------------|-------------------------|
| **Exercise** | Maturity T only | Anytime ≤ T | Maturity T only | Varies (path-dependent) |
| **Payoff** | Continuous (linear) | Continuous | Discontinuous (0 or 1) | Custom (average, max, etc.) |
| **Pricing** | BS closed-form | Binomial/LSM | Closed-form | Monte Carlo, PDE |
| **Liquidity** | Very high | Very high | Low (OTC) | Low (OTC, structured) |
| **Greeks** | Standard Δ, Γ, ν | Early ex. premium | Large Γ near barrier | Path-dependent Greeks |

## Examples + Counterexamples
**Simple Example:**  
Buy SPY $450 call, expiry 30 days, premium $3.50; if SPY closes at $460 → profit = $10 - $3.50 = $6.50/share

**Failure Case:**  
Sell naked puts (unlimited downside): Stock crashes from $100 to $50 → owe $50/share; margin called

**Edge Case:**  
American call on non-dividend stock: Early exercise never optimal (time value > intrinsic); effectively European

## Layer Breakdown
```
Vanilla Option Classification & Mechanics:
├─ Option Types:
│   ├─ Call Option:
│   │   ├─ Right to BUY asset at strike K
│   │   ├─ Payoff (European): max(S_T - K, 0)
│   │   ├─ Buyer: Bullish; limited loss (premium), unlimited upside
│   │   └─ Seller (writer): Bearish; limited gain (premium), unlimited downside
│   └─ Put Option:
│       ├─ Right to SELL asset at strike K
│       ├─ Payoff (European): max(K - S_T, 0)
│       ├─ Buyer: Bearish; limited loss (premium), high upside (max = K)
│       └─ Seller (writer): Bullish; limited gain (premium), high downside
├─ Exercise Style:
│   ├─ European:
│   │   ├─ Exercise ONLY at maturity T
│   │   ├─ Priced via Black-Scholes (closed-form)
│   │   └─ Common in index options (SPX, cash-settled)
│   └─ American:
│       ├─ Exercise ANYTIME on or before T
│       ├─ Early Exercise Value: American ≥ European (optionality premium)
│       ├─ Optimal Early Exercise: American put (when deep ITM); call without dividends (never)
│       └─ Priced via binomial tree, Longstaff-Schwartz MC
├─ Moneyness (Intrinsic Value):
│   ├─ In-The-Money (ITM):
│   │   ├─ Call: S > K (positive intrinsic value)
│   │   ├─ Put: S < K (positive intrinsic value)
│   │   └─ High delta (call: 0.6-1.0; put: -1.0 to -0.6)
│   ├─ At-The-Money (ATM):
│   │   ├─ Call/Put: S ≈ K (zero intrinsic value)
│   │   ├─ Maximum time value (theta highest)
│   │   └─ Delta ≈ 0.5 (call) or -0.5 (put)
│   └─ Out-Of-The-Money (OTM):
│       ├─ Call: S < K (zero intrinsic value)
│       ├─ Put: S > K (zero intrinsic value)
│       ├─ Pure time value (speculative)
│       └─ Low delta (call: 0-0.4; put: -0.4 to 0)
├─ Value Components:
│   ├─ Intrinsic Value:
│   │   ├─ Call: max(S - K, 0)
│   │   ├─ Put: max(K - S, 0)
│   │   └─ Realized if exercised immediately
│   └─ Time Value (Extrinsic):
│       ├─ Option Price - Intrinsic Value
│       ├─ Reflects uncertainty (volatility) and time to expiry
│       ├─ Decays to zero at maturity (theta effect)
│       └─ Maximum for ATM options
├─ Market Conventions:
│   ├─ Contract Size: Typically 100 shares per contract (equity options)
│   ├─ Expiration: Monthly (3rd Friday), weekly, quarterly
│   ├─ Strike Spacing: $1, $2.50, $5, $10 depending on underlying price
│   ├─ Settlement: Physical delivery (equity) or cash (index options)
│   └─ Trading Hours: Regular market hours + extended for some products
├─ Basic Strategies (Single Option):
│   ├─ Long Call: Buy call; bullish; limited risk, unlimited reward
│   ├─ Short Call: Sell call; bearish/neutral; unlimited risk, limited reward
│   ├─ Long Put: Buy put; bearish; limited risk, high reward (max K)
│   ├─ Short Put: Sell put; bullish; high risk (max K), limited reward
│   ├─ Covered Call: Long stock + Short call; income generation, cap upside
│   └─ Protective Put: Long stock + Long put; insurance, floor downside at K
└─ Greeks Summary:
    ├─ Delta (Δ): Hedge ratio; ∂V/∂S; range [0,1] call, [-1,0] put
    ├─ Gamma (Γ): Delta sensitivity; ∂²V/∂S²; peak at ATM
    ├─ Vega (ν): Volatility sensitivity; ∂V/∂σ; positive for long options
    ├─ Theta (θ): Time decay; ∂V/∂t; negative for long options (ATM peak)
    └─ Rho (ρ): Rate sensitivity; ∂V/∂r; minor except long-dated options
```

**Interaction:** Classify option type → Determine moneyness → Separate intrinsic/time value → Price via BS → Analyze Greeks

## Challenge Round
**Q1:** Why is American call on non-dividend stock worth same as European call?  
**A1:** Early exercise call: Receive S - K today. Defer to T: Receive (S_T - K)e^(-r(T-t)) expected value > S - K (time value of money on strike payment). Always better to sell option (recover time value) than exercise early. With dividends: Early exercise may be optimal just before ex-dividend date.

**Q2:** Prove intrinsic value is lower bound for option price (no arbitrage).  
**A2:** Call price C ≥ max(S - Ke^(-rT), 0). If C < S - Ke^(-rT), arbitrage: Buy call for C, short stock for S, invest K at rate r. At T, exercise call (pay K, get S_T), close short (deliver S_T), profit = S - C - K > 0. Put similar with S/K swapped.

**Q3:** Compare theta decay for ATM vs deep ITM option. Which decays faster?  
**A3:** ATM option has higher theta (faster decay) in absolute dollars. ATM has maximum time value; as T → 0, entire value evaporates. Deep ITM has mostly intrinsic value (stable) + small time value → lower theta. Percentage decay may be higher for OTM (all time value).

**Q4:** Explain why put-call parity C - P = S - Ke^(-rT) holds only for European options.  
**A4:** Parity derived from portfolio equivalence at maturity T: (Call + Ke^(-rT) cash) = (Put + Stock). American options can be exercised early → portfolios differ before T. American put may be exercised early (capture K), breaking parity.

**Q5:** Covered call vs collar: How does collar reduce cost?  
**A5:** Covered call: Long stock + Short call at K_high; collect premium P_call. Collar: Long stock + Long put at K_low + Short call at K_high; net cost = P_put - P_call (often near zero). Collar sacrifices upside to finance downside protection.

**Q6:** Why do OTM options have higher percentage gains/losses than ITM options?  
**A6:** Leverage effect. OTM call costs $1, ITM call costs $10 (more intrinsic). If stock moves $5, OTM may go $1 → $5 (400% gain), ITM $10 → $15 (50% gain). OTM has higher gamma → larger delta changes → explosive percentage moves.

**Q7:** When does American put have optimal early exercise boundary?  
**A7:** Deep ITM: When time value < interest on strike (opportunity cost). If S → 0, put worth K - S ≈ K. Early exercise captures K to invest at rate r. Boundary S* depends on (K, r, σ, T): Typically S* < K; exercise when S < S*.

**Q8:** Compute breakeven prices for long straddle (buy ATM call + ATM put). When is this profitable?  
**A8:** Cost = C_ATM + P_ATM. Payoff at T: max(S_T - K, 0) + max(K - S_T, 0) = |S_T - K|. Breakeven: |S_T - K| = Cost → S_T = K ± Cost. Profitable if stock moves beyond K ± Cost (high realized volatility). Loses if stock stays near K (volatility below implied).

## Key References
**Primary Sources:**
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapters 9-12: Option Properties
- [Options (finance) Wikipedia](https://en.wikipedia.org/wiki/Option_(finance)) - Comprehensive overview
- CBOE *Options 101* - Market conventions and terminology

**Technical Details:**
- Cox, J.C. & Rubinstein, M. *Options Markets* (1985) - Vanilla option theory (Chapters 3-7)
- Natenberg, S. *Option Volatility and Pricing* (2014) - Trading strategies (Chapters 4-10)

**Thinking Steps:**
1. Classify option: Call vs put, European vs American
2. Determine moneyness: ITM (has intrinsic), ATM (max time value), OTM (pure speculation)
3. Separate intrinsic value (exercise today) from time value (uncertainty premium)
4. Price using Black-Scholes (European) or binomial tree (American)
5. Compute delta for hedge ratio (how many shares to replicate option)
6. Analyze strategy: Directional (long call/put) vs hedged (covered call, protective put)
