# Risk-Neutral Valuation

## Concept Skeleton
**Definition:** Derivative pricing framework where discounted expected payoffs under risk-neutral measure equal current price; eliminates need to estimate real-world drift  
**Purpose:** Value options and derivatives without knowing investors' risk preferences; enables arbitrage-free pricing through replication arguments  
**Prerequisites:** No-arbitrage principle, probability theory, martingales, change of measure (Radon-Nikodym), stochastic calculus basics

## Comparative Framing
| Measure | Risk-Neutral (Q) | Real-World (P) | Forward Measure | T-Forward Measure |
|---------|------------------|----------------|-----------------|-------------------|
| **Drift** | Risk-free rate r | Actual μ | Zero (martingale) | Specific numeraire |
| **Purpose** | Pricing | Forecasting | Simplify formulas | Bond options |
| **Discount** | e^(-rT) | e^(-μT) | No discount needed | T-bond numeraire |
| **Volatility** | Same as P | Historical/forecast | Same | Same |

| Concept | Risk-Neutral | No-Arbitrage | Replication | Martingale |
|---------|--------------|--------------|-------------|------------|
| **Foundation** | Q-measure pricing | Law of one price | Synthetic portfolio | Mathematical tool |
| **Application** | All derivatives | Any asset | Complete markets | Pricing framework |
| **Assumption** | Q exists | Markets efficient | Hedgeable | Q is EMM |
| **Limitation** | Model-dependent | Frictionless | May not exist | Technical |

## Examples + Counterexamples
**Simple Example:**  
Stock $100, grows 15%/year real-world. Risk-neutral: grows at r=5%. Option prices using r=5%, not 15%. Investors' risk preferences embedded in current stock price already.

**Perfect Fit:**  
European call pricing: Simulate under Q (drift=r), calculate E^Q[max(S_T-K,0)], discount at r → matches Black-Scholes exactly. No need to know real drift μ.

**Replication Argument:**  
Delta-hedge portfolio: Δ shares + B in bank replicates option. Portfolio grows at r (self-financing). Option must also grow at r under Q → risk-neutral drift emerges naturally.

**Real-World vs Risk-Neutral:**  
Real-world P: E^P[S_T]=S_0 e^(μT), μ=15%. Risk-neutral Q: E^Q[S_T]=S_0 e^(rT), r=5%. Different expectations, same price S_0 due to risk adjustment.

**Incomplete Market:**  
Jump-diffusion with unhedgeable jumps: Multiple risk-neutral measures exist (Q not unique). Range of arbitrage-free prices, not single value. Need additional pricing principle (utility, etc.).

**Poor Fit:**  
Long-dated equity options (10+ years): Discount factor e^(-rT) dominates, small vol changes huge impact. Real-world default risk, model risk become significant → Q-measure approximation breaks down.

## Layer Breakdown
```
Risk-Neutral Valuation Framework:

├─ Fundamental Principle:
│  ├─ No-Arbitrage: Cannot create riskless profit from nothing
│  │   ├─ Law of one price: Same payoff → same price
│  │   ├─ Implies: Discounted price process is martingale
│  │   └─ Consequence: Risk-neutral measure Q exists
│  ├─ Pricing Formula:
│  │   V_0 = E^Q[e^(-rT) Payoff(S_T)]
│  │   Expectation under Q, discount at risk-free rate
│  ├─ Key Insight:
│  │   Current price S_0 already reflects risk premium
│  │   Option value depends only on S_0, not future drift μ
│  └─ Why It Works:
│      Replication: Can hedge continuously → return must be r
│      Alternative: Arbitrage opportunity exists
├─ Risk-Neutral Measure (Q):
│  ├─ Definition:
│  │   Probability measure where discounted asset prices are martingales
│  │   E^Q[S_T | F_t] = S_t e^(r(T-t))
│  ├─ Construction (Girsanov Theorem):
│  │   ├─ Real-world: dS = μS dt + σS dW^P
│  │   ├─ Risk-neutral: dS = rS dt + σS dW^Q
│  │   ├─ Change of measure: dW^Q = dW^P + ((μ-r)/σ)dt
│  │   └─ Market price of risk: λ = (μ-r)/σ
│  ├─ Radon-Nikodym Derivative:
│  │   dQ/dP = exp(-λW^P_T - ½λ²T)
│  │   Converts probabilities: P → Q
│  ├─ Properties:
│  │   ├─ Q is EMM (Equivalent Martingale Measure)
│  │   ├─ Same null sets as P (equivalent)
│  │   ├─ Volatility unchanged: σ^Q = σ^P
│  │   └─ Only drift shifts: μ → r
│  └─ Existence & Uniqueness:
│      ├─ Complete market: Unique Q
│      ├─ Incomplete: Multiple Q's (bounds on price)
│      └─ Arbitrage exists: No Q exists
├─ Derivation via Replication:
│  ├─ Self-Financing Portfolio:
│  │   ├─ Hold Δ_t shares of stock
│  │   ├─ Hold B_t in bank account (bond)
│  │   ├─ Portfolio value: Π_t = Δ_t S_t + B_t
│  │   └─ Replicates option: Π_T = Payoff(S_T)
│  ├─ Dynamics:
│  │   dΠ = Δ dS + r B dt
│  │   No cash injection (self-financing)
│  ├─ Hedging Condition:
│  │   Choose Δ such that dΠ has no dW term
│  │   → Π grows at rate r (riskless)
│  ├─ Result:
│  │   Π_t = e^(-r(T-t)) E^Q[Payoff | F_t]
│  │   Discounted portfolio is Q-martingale
│  └─ Conclusion:
│      Option value = Replication cost = Risk-neutral expectation
├─ Black-Scholes via Risk-Neutral:
│  ├─ Under Q:
│  │   S_T = S_0 exp((r - ½σ²)T + σ√T Z)
│  │   where Z ~ N(0,1) under Q
│  ├─ Call payoff:
│  │   C(S_T) = max(S_T - K, 0)
│  ├─ Expected payoff:
│  │   E^Q[C(S_T)] = ∫ max(S_T - K, 0) φ(z) dz
│  │   Integral over lognormal distribution
│  ├─ Analytical evaluation:
│  │   Yields: S_0 N(d_1) - K e^(-rT) N(d_2)
│  │   Black-Scholes formula
│  └─ No μ appears: Only S_0, K, r, T, σ
├─ Risk-Neutral Probability:
│  ├─ Interpretation:
│  │   NOT real probability of outcomes
│  │   Mathematical construct for pricing
│  ├─ Example (Binomial):
│  │   ├─ Real-world: P(up)=0.6, P(down)=0.4
│  │   ├─ Risk-neutral: Q(up)=(e^(rΔt)-d)/(u-d)
│  │   │   Typically Q(up) < P(up) if μ > r
│  │   └─ Risk adjustment: Reduces probability of good outcomes
│  ├─ Intuition:
│  │   Q-probabilities price risk aversion into expectations
│  │   Equivalent to using P-probabilities with risk-adjusted discount
│  └─ Connection:
│      E^Q[X] = E^P[X × (dQ/dP)]
│      Expectation under Q = Weighted expectation under P
├─ Numeraire Change:
│  ├─ General Principle:
│  │   Any tradable asset can be numeraire (unit of account)
│  │   Relative prices in numeraire units are martingales
│  ├─ Bank Account Numeraire:
│  │   ├─ N_t = e^(rt) (money market account)
│  │   ├─ Measure: Risk-neutral Q
│  │   ├─ Result: S_t / N_t = S_t e^(-rt) is Q-martingale
│  │   └─ Standard pricing: V_0 = E^Q[e^(-rT) V_T]
│  ├─ Stock as Numeraire:
│  │   ├─ N_t = S_t (stock price)
│  │   ├─ Measure: Stock measure Q^S
│  │   ├─ Result: V_t / S_t is Q^S-martingale
│  │   └─ Use: Simplifies exchange options (Margrabe)
│  ├─ Zero-Coupon Bond Numeraire:
│  │   ├─ N_t = P(t,T) (T-bond price)
│  │   ├─ Measure: T-forward measure Q^T
│  │   ├─ Result: Forward prices are martingales
│  │   └─ Use: Interest rate derivatives (caps, swaptions)
│  └─ Conversion (Fundamental Theorem):
│      V_0/N_0 = E^Q^N[V_T / N_T]
│      Change numeraire → change measure → simplify calculations
├─ Applications:
│  ├─ European Options:
│  │   ├─ Calls, puts: Direct expected value calculation
│  │   ├─ Digitals: Q(S_T > K) under lognormal
│  │   └─ Any terminal payoff: E^Q[g(S_T)]
│  ├─ Path-Dependent Options:
│  │   ├─ Asians: E^Q[max(Avg(S)-K, 0)]
│  │   ├─ Barriers: E^Q[Payoff × Indicator(no breach)]
│  │   ├─ Lookbacks: E^Q[max over path - K]
│  │   └─ Monte Carlo: Simulate under Q, average payoffs
│  ├─ Multi-Asset Options:
│  │   ├─ Correlation enters through joint distribution under Q
│  │   ├─ Baskets: E^Q[max(w·S_T - K, 0)]
│  │   └─ Spreads: E^Q[max(S₁_T - S₂_T - K, 0)]
│  ├─ Interest Rate Derivatives:
│  │   ├─ Caps/Floors: Use forward measure
│  │   ├─ Swaptions: Swap measure (annuity numeraire)
│  │   └─ Exotic rates: HJM, LMM frameworks
│  └─ Credit Derivatives:
│      Default-adjusted Q: Intensity models, hazard rates
├─ Real-World vs Risk-Neutral:
│  ├─ Real-World (P-measure):
│  │   ├─ Purpose: Forecasting, risk management, VaR
│  │   ├─ Drift: Historical μ or estimated
│  │   ├─ Probabilities: True likelihood
│  │   └─ Example: 20% chance stock below $90
│  ├─ Risk-Neutral (Q-measure):
│  │   ├─ Purpose: Pricing derivatives
│  │   ├─ Drift: Risk-free rate r
│  │   ├─ Probabilities: Risk-adjusted (not real)
│  │   └─ Example: 35% "risk-neutral probability" below $90
│  ├─ Relationship:
│  │   ├─ Q puts more weight on bad outcomes
│  │   ├─ Reflects risk aversion in market prices
│  │   └─ Connected via market price of risk λ
│  └─ When to Use Which:
│      ├─ Pricing: Always use Q
│      ├─ Hedging: Can use either (both give same hedge ratio)
│      ├─ Risk assessment: Use P (real probabilities)
│      └─ Scenario analysis: Use P (realistic outcomes)
├─ Market Price of Risk:
│  ├─ Definition:
│  │   λ = (μ - r) / σ
│  │   Excess return per unit volatility
│  ├─ Interpretation:
│  │   ├─ Compensation for bearing risk
│  │   ├─ Higher λ → higher risk premium
│  │   └─ Market determines λ via supply/demand
│  ├─ Girsanov Connection:
│  │   dW^Q = dW^P + λ dt
│  │   Shift Brownian motion by λ
│  ├─ Multi-Factor:
│  │   Vector λ = [λ₁, ..., λ_n]
│  │   One λ per risk factor
│  └─ Calibration:
│      Extract λ from option prices (implied)
│      Or estimate from time series (historical)
├─ Completeness:
│  ├─ Complete Market:
│  │   ├─ Every contingent claim can be replicated
│  │   ├─ Unique risk-neutral measure Q
│  │   ├─ Example: Black-Scholes (1 stock, 1 bond)
│  │   └─ Consequence: Unique arbitrage-free price
│  ├─ Incomplete Market:
│  │   ├─ Some claims cannot be hedged
│  │   ├─ Multiple Q's exist (set of EMMs)
│  │   ├─ Example: Jump-diffusion, stochastic vol
│  │   └─ Consequence: Price bounds, not unique price
│  ├─ No-Arbitrage vs Completeness:
│  │   ├─ No-arbitrage: Q exists (feasible prices)
│  │   ├─ Completeness: Q unique (unique price)
│  │   └─ Both: Harrison-Pliska fundamental theorem
│  └─ Practical Impact:
│      Incompleteness → Model risk, calibration challenges
├─ Limitations & Caveats:
│  ├─ Continuous Trading:
│  │   Assumes infinite rebalancing (unrealistic)
│  │   Transaction costs violate perfect replication
│  ├─ No-Arbitrage Assumption:
│  │   Requires liquid, efficient markets
│  │   Breaks during crises, illiquidity
│  ├─ Model Risk:
│  │   Q depends on chosen model (GBM, jumps, etc.)
│  │   Wrong model → wrong Q → wrong price
│  ├─ Real-World Drift Irrelevant:
│  │   True for pricing, NOT for risk management
│  │   P&L depends on real outcomes under P
│  └─ Long Maturity:
│      Model assumptions degrade over long horizons
│      Discount factors compound small errors
└─ Practical Implementation:
   ├─ Monte Carlo:
   │   ├─ Simulate paths under Q (drift=r)
   │   ├─ Calculate payoff each path
   │   ├─ Average and discount: V_0 = e^(-rT) × mean(payoffs)
   │   └─ Variance reduction: Same as before
   ├─ Trees:
   │   ├─ Risk-neutral probabilities at each node
   │   ├─ Backward induction with discount
   │   └─ Matches risk-neutral expectation
   ├─ PDE Approach:
   │   Black-Scholes PDE derived from risk-neutral argument
   │   Solve PDE with boundary conditions
   ├─ Closed-Form:
   │   Evaluate E^Q[Payoff] analytically if possible
   │   Black-Scholes, Bachelier, etc.
   └─ Calibration:
      ├─ Extract Q from liquid option prices
      ├─ Use calibrated Q to price illiquid derivatives
      └─ Ensure consistency across products
```

**Interaction:** No-arbitrage → Q exists → Pricing via E^Q[Discounted payoff] → Model choice determines Q → Calibration to market → Exotic valuation.

## Challenge Round
1. **Heston Model Q-Measure:** Derive risk-neutral dynamics for Heston stochastic volatility. What's market price of volatility risk? How does it affect option prices?

2. **Incomplete Market:** Model jump-diffusion (Merton). Show multiple EMMs exist. Calculate bounds on option price. What additional principle determines unique price?

3. **Foreign Exchange:** Derive risk-neutral measure for FX option (two interest rates). Show how domestic/foreign rate enter. What's Garman-Kohlhagen formula?

4. **Change of Numeraire:** Price Margrabe exchange option using stock as numeraire. Verify simpler than bank account numeraire. What's new risk-neutral measure?

5. **Long-Dated Options:** Price 20-year equity call. How sensitive to model assumptions? Compare real-world P&L distribution to Q-measure pricing. Why diverge?

## Key References
- [Harrison & Pliska (1981) - Fundamental Theorem of Asset Pricing](https://www.jstor.org/stable/3689775)
- [Cox & Ross (1976) - Risk-Neutral Pricing](https://www.jstor.org/stable/2978261)
- [Shreve, Stochastic Calculus for Finance II (Chapters 1-5)](https://www.springer.com/series/3401)
- [Baxter & Rennie, Financial Calculus (Chapter 3)](https://www.cambridge.org/core/books/financial-calculus/33D34BFC5A07FA0DBEF0D4A39A0C1B13)

---
**Status:** Foundation of derivative pricing | **Complements:** Black-Scholes, Replication, Martingales, Girsanov Theorem, No-Arbitrage Principle
