# Black-Scholes Model

## Concept Skeleton
**Definition:** Closed-form mathematical model for pricing European options on non-dividend-paying stocks; assumes log-normal stock price distribution, constant volatility, no arbitrage, continuous trading  
**Purpose:** Foundation for derivatives pricing; enables rapid option valuation; basis for implied volatility calculations; practical benchmark despite unrealistic assumptions  
**Prerequisites:** Option pricing basics, stochastic calculus, log-normal distribution, risk-neutral valuation, no-arbitrage principle

## Comparative Framing
| Model | Black-Scholes | Binomial | Monte Carlo | Numerical PDE |
|-------|---------------|----------|-------------|---------------|
| **Type** | Closed-form | Tree-based | Simulation | Grid-based |
| **Speed** | Instant | Fast (recombines) | Slow (many paths) | Moderate |
| **Accuracy** | Good (standard assumptions) | Exact (for grid) | Improves with paths | Accurate (fine grid) |
| **Dividends** | Extension available | Easy to add | Straightforward | Easy to add |
| **American Options** | N/A | Natural fit | Approximate needed | Natural fit |
| **Exotics** | Limited | Limited | Excellent | Good |
| **Intuition** | Medium | High | Medium | Low |
| **Computation** | Analytical | Numerical recurrence | Sampling | Numerical solver |

## Examples + Counterexamples
**Simple Example:**  
Stock $100, strike $100, r=5%, T=1 year, σ=20%. BS formula gives C≈$10.45, P≈$5.57. Matches market prices well for liquid, vanilla options.

**Excellent Fit:**  
Short-dated ATM options on large-cap stocks: Continuous trading, low transaction costs, relatively stable volatility. BS very accurate.

**Poor Fit:**  
Very volatile stocks (σ changes), illiquid markets (bid-ask spreads large), short-dated options deep OTM (jumps matter, assumes continuous paths), or when large discrete dividends occur.

**American Options:**  
BS doesn't handle early exercise. Must use binomial, Monte Carlo, or numerical PDE. For calls on non-dividend stocks, BS bound is tight (American=European).

**Extension - With Dividends:**  
Modify to S*e^(-q*T) where q is continuous dividend yield. Reduces call value, increases put value.

## Layer Breakdown
```
Black-Scholes Framework:

├─ Assumptions (Critical):
│  ├─ Stock price follows geometric Brownian motion:
│  │   dS = μS dt + σS dW
│  │   (constant volatility σ, drift μ)
│  ├─ Continuous trading (no bid-ask spreads)
│  ├─ No arbitrage (can replicate option payoff)
│  ├─ Risk-free rate r constant
│  ├─ No dividends (or constant dividend yield)
│  ├─ Frictionless market (no taxes, costs)
│  ├─ No restrictions on short-selling
│  ├─ Log-normal distribution for S_T
│  └─ European exercise only
├─ Mathematical Derivation:
│  ├─ Setup: Option value C(S,t) dependent on spot, time
│  ├─ Replicating portfolio: Hold Δ shares + bond
│  ├─ No-arbitrage condition: Portfolio return = r
│  ├─ Ito's Lemma applied to C(S,t)
│  ├─ Results in PDE:
│  │   ∂C/∂t + rS(∂C/∂S) + (1/2)σ²S²(∂²C/∂S²) = rC
│  ├─ Boundary conditions:
│  │   C(S,T) = max(S-K, 0) [call payoff at expiry]
│  │   C(0,t) = 0 [worthless if S→0]
│  │   C(S,t) ≈ S as S→∞ [behaves like stock]
│  └─ Solution: Closed-form formulas
├─ Black-Scholes Formulas:
│  ├─ Call Price:
│  │   C = S₀ N(d₁) - K e^(-rT) N(d₂)
│  ├─ Put Price:
│  │   P = K e^(-rT) N(-d₂) - S₀ N(-d₁)
│  ├─ Where:
│  │   d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
│  │   d₂ = d₁ - σ√T
│  ├─ Components:
│  │   S₀: Current spot price
│  │   K: Strike price
│  │   r: Risk-free rate
│  │   T: Time to expiry
│  │   σ: Volatility (annualized)
│  │   N(.): Standard normal CDF
│  └─ Put-Call Parity: C - P = S₀ - K e^(-rT)
├─ Greeks Derivation:
│  ├─ Delta (∂C/∂S):
│  │   Δ_call = N(d₁) ∈ [0, 1]
│  │   Δ_put = -N(-d₁) ∈ [-1, 0]
│  │   Hedging interpretation: Buy Δ shares per short call
│  ├─ Gamma (∂²C/∂S²):
│  │   Γ = N'(d₁) / (S σ√T) > 0
│  │   Peaks near ATM, highest near expiry
│  │   Hedging cost: Must rebalance as S moves
│  ├─ Theta (∂C/∂t):
│  │   Θ_call = -S N'(d₁) σ/(2√T) - r K e^(-rT) N(d₂)
│  │   Negative for long calls (time decay)
│  │   Θ_put = -S N'(d₁) σ/(2√T) + r K e^(-rT) N(-d₂)
│  ├─ Vega (∂C/∂σ):
│  │   ν = S N'(d₁) √T > 0 (same for calls and puts)
│  │   Peak ATM, decreases near expiry or deep in/out
│  └─ Rho (∂C/∂r):
│      ρ_call = K T e^(-rT) N(d₂)
│      ρ_put = -K T e^(-rT) N(-d₂)
├─ Implied Volatility:
│  ├─ Invert BS formula: Given market price → find σ
│  ├─ No closed form; use Newton-Raphson
│  ├─ Volatility smile/skew: IV varies by strike
│  ├─ Term structure: IV varies by expiry
│  └─ Used for comparisons, quoting, risk assessment
├─ Violations & Reality Checks:
│  ├─ Constant volatility assumption violated:
│  │   ├─ Volatility changes over time (stochastic vol)
│  │   ├─ Different strikes imply different vols (smile)
│  │   └─ Solution: Heston model, SABR, etc.
│  ├─ Jump risk:
│  │   ├─ Stock prices jump (gap events, earnings)
│  │   ├─ Log-normal doesn't account for gaps
│  │   └─ Solution: Jump-diffusion models
│  ├─ Discrete rebalancing costs:
│  │   ├─ Hedging only possible at discrete times
│  │   ├─ Gamma risk accumulates
│  │   └─ Transaction costs eat into profits
│  └─ Liquidity/bid-ask spreads:
│      ├─ Pricing models assume perfect markets
│      ├─ Real bid-ask affects profitable hedging
│      └─ Adjustment: Add transaction cost factor
└─ Extensions:
   ├─ With dividends (yield q):
   │   C = S₀ e^(-qT) N(d₁) - K e^(-rT) N(d₂)
   │   d₁ = [ln(S₀/K) + (r-q+σ²/2)T] / (σ√T)
   ├─ With foreign exchange (similar form)
   ├─ With futures (different drift term)
   └─ Approximations for small moves (delta/gamma terms)
```

**Interaction:** BS price sensitive to all five parameters (S, K, r, T, σ); Greeks measure each sensitivity; Vega largest uncertainty (volatility hardest to estimate).

## Challenge Round
1. **Implied Volatility:** Market call price $11. Calculate IV using Newton-Raphson. What if IV fluctuates?

2. **Model Violations:** Compare BS price to actual market data (stock with jump risk, high volatility). Explore which Greeks most affected.

3. **Discretization Impact:** Implement discrete delta-hedging over time. Compare P&L to theoretical BS (continuous hedging).

4. **Dividend Yields:** Add continuous dividend q. How does it affect delta, theta? Special cases: deep ITM/OTM.

5. **Extreme Moves:** Sample from actual log-returns (likely fatter tails than normal). Price options; compare to BS (should overprice/underprice depending on tail).

## Key References
- [Black, Scholes, Merton (1973) - Original Paper](https://www.jstor.org/stable/3003143)
- [Hull, Options, Futures, and Other Derivatives (Chapter 15)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000006649)
- [Stochastic Calculus for Finance (Shreve)](https://www.springer.com/series/3401)
- [Greeks Explained (Fisher, 2000)](https://www.jstor.org/stable/2676670)

---
**Status:** Foundation model in derivatives pricing | **Complements:** Option Pricing Basics, Implied Volatility, Greeks, Monte Carlo Pricing
