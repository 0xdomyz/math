# Real-World vs Risk-Neutral Pricing

## Concept Skeleton
**Definition:** Distinguish physical probability measure ℙ (real-world) from risk-neutral measure ℚ (pricing), related via Girsanov theorem and market price of risk  
**Purpose:** Price derivatives using risk-neutral probabilities (no-arbitrage), while forecasting/risk management uses real-world probabilities  
**Prerequisites:** Stochastic calculus, Brownian motion, Radon-Nikodym derivative, martingale theory, arbitrage-free pricing

## Comparative Framing
| Measure | Real-World (ℙ) | Risk-Neutral (ℚ) |
|---------|---------------|------------------|
| **Purpose** | Forecasting, risk analysis, expected returns | Derivative pricing (no-arbitrage) |
| **Drift** | μ (expected return, e.g., 8-12% equity) | r (risk-free rate, e.g., 5%) |
| **Volatility** | σ_physical (calibrate to historical) | σ_implied (calibrate to option prices) |
| **Discounting** | Risk-adjusted rate (μ or WACC) | Risk-free rate r |
| **Example** | E^ℙ[S_T] = S_0 e^{μT} (forecast stock price) | E^ℚ[e^{-rT} Payoff] (option price) |

## Examples + Counterexamples
**Simple Example:**  
Stock S₀=$100, μ=10%, σ=20%, r=5% → Real-world forecast E^ℙ[S₁Y]=$110, but risk-neutral E^ℚ[S₁Y]=$105.13 (used for option pricing)

**Failure Case:**  
Using real-world μ to price options → Call overpriced (drift too high) → arbitrage via delta-hedging exploits mispricing

**Edge Case:**  
Derivatives on non-traded assets (e.g., temperature, electricity): No unique ℚ (incomplete market) → multiple equivalent martingale measures, model choice matters

## Layer Breakdown
```
Real-World vs Risk-Neutral Pricing:
├─ Probability Measures:
│   ├─ Physical Measure ℙ (Real-World):
│   │   ├─ Definition: True probabilities governing asset dynamics in reality
│   │   ├─ Purpose: Forecasting, VaR, stress testing, expected portfolio returns
│   │   ├─ Equity Dynamics: dS/S = μ dt + σ dW^ℙ
│   │   │   where μ = expected return (historical: 8-12% for equity)
│   │   ├─ Estimation: Historical calibration (regression, MLE from time series)
│   │   └─ Example: Forecast S₁Y ~ LogNormal(S₀ e^{μT}, σ²T)
│   ├─ Risk-Neutral Measure ℚ (Pricing Measure):
│   │   ├─ Definition: Artificial probabilities making discounted assets martingales
│   │   ├─ Purpose: Derivative pricing via no-arbitrage (Law of One Price)
│   │   ├─ Equity Dynamics: dS/S = r dt + σ dW^ℚ
│   │   │   where r = risk-free rate (replace μ with r)
│   │   ├─ Key Property: e^{-rt} S_t is a ℚ-martingale → E^ℚ[S_T | F_t] = S_t e^{r(T-t)}
│   │   └─ Pricing Formula: V₀ = E^ℚ[e^{-rT} Payoff(S_T)]
│   └─ Relationship:
│       Change of measure via Radon-Nikodym derivative (Girsanov theorem)
├─ Girsanov Theorem:
│   ├─ Statement: Change Brownian motion drift without changing volatility
│   │   dW^ℚ = dW^ℙ + θ dt (θ = market price of risk)
│   │   where θ = (μ - r) / σ (Sharpe ratio adjusted for risk-free rate)
│   ├─ Measure Change:
│   │   dℚ/dℙ = exp(-½∫₀ᵀ θ²dt - ∫₀ᵀ θ dW^ℙ) (Radon-Nikodym derivative)
│   │   ├─ Under ℚ: W^ℚ_t = W^ℙ_t + ∫₀ᵗ θ ds is ℚ-Brownian motion
│   │   └─ Novikov Condition: E[exp(½∫₀ᵀ θ²dt)] < ∞ (ensures valid measure)
│   ├─ Application to Equity:
│   │   ℙ: dS = μ S dt + σ S dW^ℙ
│   │   ℚ: dS = r S dt + σ S dW^ℚ (replace μ with r, W^ℙ with W^ℚ)
│   ├─ Market Price of Risk:
│   │   θ = (μ - r) / σ (compensation per unit of volatility)
│   │   ├─ Positive θ: Risk-averse investors demand premium over r
│   │   └─ Empirical: θ ≈ 0.2-0.4 for equity (Sharpe ratio)
│   └─ Multi-Dimensional:
│       For n risk factors: dW^ℚ = dW^ℙ + Θ dt (vector θ, correlation preserved)
├─ Fundamental Theorem of Asset Pricing:
│   ├─ First FTAP: No arbitrage ⟺ ∃ equivalent martingale measure ℚ
│   │   Under ℚ: Discounted asset prices are martingales
│   ├─ Second FTAP: Market completeness ⟺ unique ℚ
│   │   ├─ Complete: All payoffs replicable (e.g., Black-Scholes with stock + bond)
│   │   └─ Incomplete: Multiple ℚ (e.g., stochastic vol, jump models)
│   └─ Implications:
│       Derivative price = Expected payoff under ℚ, discounted at r
├─ Black-Scholes Framework:
│   ├─ ℙ-Dynamics: dS = μ S dt + σ S dW^ℙ
│   ├─ ℚ-Dynamics: dS = r S dt + σ S dW^ℚ
│   ├─ Option Pricing:
│   │   C(S,t) = E^ℚ[e^{-r(T-t)} max(S_T - K, 0) | S_t = S]
│   │   = S Φ(d₁) - K e^{-r(T-t)} Φ(d₂) (Black-Scholes formula)
│   │   where d₁ = [ln(S/K) + (r+½σ²)(T-t)] / (σ√(T-t))
│   ├─ Implied Volatility:
│   │   σ_implied calibrated to option prices under ℚ (forward-looking)
│   │   ≠ σ_historical estimated from returns under ℙ
│   └─ PDE Derivation:
│       Δ-hedging eliminates risk → risk-free return → no μ appears in PDE
│       ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
├─ Forward vs Spot Measure:
│   ├─ Spot Measure (ℚ): Numeraire = money market account B_t = e^{rt}
│   │   Price: V₀ = E^ℚ[e^{-rT} Payoff]
│   ├─ Forward Measure (ℚ^T): Numeraire = zero-coupon bond P(t,T)
│   │   Price: V₀ = P(0,T) E^{ℚ^T}[Payoff]
│   │   ├─ Advantage: No discounting needed under ℚ^T
│   │   └─ Use: Interest rate derivatives (caps, swaptions)
│   ├─ Stock Measure (ℚ^S): Numeraire = stock S_t
│   │   Use: Quantos, change-of-numeraire techniques
│   └─ Change of Numeraire Formula:
│       dℚ^T/dℚ = [B_T / P(T,T)] / [B_0 / P(0,T)] (Radon-Nikodym derivative)
├─ Incomplete Markets:
│   ├─ Sources of Incompleteness:
│   │   ├─ Stochastic Volatility: σ_t random, not traded → can't hedge vol risk
│   │   ├─ Jumps: Infinite hedging frequency required → impractical
│   │   ├─ Non-Traded Assets: Weather, credit, energy → no replicating portfolio
│   │   └─ Transaction Costs: Continuous hedging not feasible
│   ├─ Consequence: Multiple ℚ satisfy no-arbitrage → derivative price range
│   │   V_min = inf_{ℚ∈M} E^ℚ[e^{-rT} Payoff]
│   │   V_max = sup_{ℚ∈M} E^ℚ[e^{-rT} Payoff]
│   │   where M = set of equivalent martingale measures
│   ├─ Approaches:
│   │   ├─ Utility Indifference: Price such that agent indifferent to trading
│   │   ├─ Good-Deal Bounds: Restrict Sharpe ratio to reasonable range
│   │   └─ Minimal Entropy: Choose ℚ closest to ℙ (minimize KL divergence)
│   └─ Example (Stochastic Vol):
│       Heston model has non-unique ℚ → specify market price of vol risk λ_v
│       dv = κ(θ - v)dt + σ_v √v (dW^ℙ_v + λ_v dt) (ℙ-dynamics)
│       dv = κ(θ - v - λ_v σ_v √v)dt + σ_v √v dW^ℚ_v (ℚ-dynamics)
├─ Practical Differences:
│   ├─ Option Pricing:
│   │   ℚ: Use implied vol σ_implied, drift = r
│   │   Example: σ_implied = 25% (from market), σ_historical = 20% (from data)
│   │   → Price with 25% vol, not 20%
│   ├─ VaR / Risk Management:
│   │   ℙ: Use historical μ and σ, simulate realistic scenarios
│   │   Example: 1-day 99% VaR uses ℙ to forecast portfolio loss distribution
│   ├─ Monte Carlo:
│   │   ℙ: dS = μ S dt + σ S dW^ℙ (forecast paths)
│   │   ℚ: dS = r S dt + σ S dW^ℚ (pricing paths)
│   │   → Same volatility σ, different drift
│   └─ Hedging:
│       Delta Δ = ∂V/∂S same under ℙ and ℚ (local property)
│       But expected P&L differs: E^ℙ[ΔP&L] ≠ 0, E^ℚ[ΔP&L] = 0 (ℚ is pricing measure)
├─ Quanto Adjustments:
│   ├─ Problem: Derivative payoff in foreign currency, settle in domestic
│   │   Example: European call on Nikkei (¥ payoff) settled in USD
│   ├─ Standard Pricing: E^{ℚ_¥}[e^{-r_¥ T} Payoff_¥] converted at spot FX
│   ├─ Quanto Pricing: Fix FX rate at inception (no FX risk)
│   │   Quanto_Call = E^{ℚ_$}[e^{-r_$ T} max(S_T^{¥/$} - K, 0)]
│   │   Requires change of measure ℚ_¥ → ℚ_$
│   ├─ Quanto Adjustment:
│   │   Drift adjusted by: μ_quanto = μ - ρ σ_S σ_FX
│   │   where ρ = correlation(Stock, FX), σ_FX = FX volatility
│   └─ Application: Quanto CDS, quanto equity options common in structured products
└─ Calibration:
    ├─ ℙ-Calibration (Historical):
    │   ├─ Data: Time series of returns r_t = ln(S_{t+1}/S_t)
    │   ├─ μ estimate: Sample mean μ̂ = (1/n)Σ r_t
    │   ├─ σ estimate: Sample std dev σ̂ = √Var(r_t)
    │   └─ Adjustment: Annualize (μ_annual = μ_daily × 252, σ_annual = σ_daily × √252)
    ├─ ℚ-Calibration (Market):
    │   ├─ Input: Option prices across strikes and maturities
    │   ├─ Implied Vol: Invert Black-Scholes to extract σ_implied(K,T)
    │   ├─ Model Parameters: Calibrate Heston, SABR to match option surface
    │   └─ Risk-Free Rate: Use OIS or Treasury curve (r from market)
    └─ Hybrid:
        Use ℙ for μ, ℚ for σ → convert between measures via Girsanov
```

**Interaction:** Calibrate ℙ from historical data (μ, σ) → Define market price of risk θ = (μ-r)/σ → Apply Girsanov to obtain ℚ-dynamics (replace μ with r) → Price derivatives under ℚ using Monte Carlo or PDE

## Challenge Round
Why can't we use real-world probabilities ℙ to price options?
- **Arbitrage:** Delta-hedging creates risk-free portfolio → must earn r, not μ → pricing PDE has no μ term (eliminated by hedging argument)
- **Risk preferences:** Different investors have different μ expectations (bulls vs bears) → no unique price without no-arbitrage framework
- **Replication:** Option payoff replicated by dynamic trading (stock + bond) → value = cost of replication independent of μ
- **Martingale property:** Under ℚ, discounted prices are martingales (fair game) → E^ℚ[e^{-rT} S_T] = S_0, simplifies math
- **Market completeness:** Black-Scholes assumes complete market → unique ℚ, any derivative replicable
- **Empirical failure:** Using ℙ with historical μ=10% drastically overprices calls (arbitrageurs would sell), market converges to ℚ-price

Reality: ℚ is artificial construct ensuring no-arbitrage, ℙ retained for risk management (VaR, stress tests, expected returns forecasting), but pricing must use ℚ.

## Key References
- [Shreve (2004) Stochastic Calculus for Finance II, Ch. 5](https://link.springer.com/book/10.1007/978-1-4939-2867-8) - Girsanov theorem, change of measure, FTAP proofs
- [Björk (2009) Arbitrage Theory in Continuous Time, Ch. 10-11](https://oxford.universitypressscholarship.com/view/10.1093/oso/9780198851615.001.0001/oso-9780198851615) - Change of numeraire, forward measures
- [Musiela & Rutkowski (2005) Martingale Methods in Financial Modelling](https://link.springer.com/book/10.1007/b137866) - Advanced measure theory, incomplete markets
- [Duffie (2001) Dynamic Asset Pricing Theory, Ch. 6](https://press.princeton.edu/books/hardcover/9780691139852/dynamic-asset-pricing-theory) - Risk-neutral valuation foundations

---
**Status:** Foundational pricing theory | **Complements:** Black-Scholes, Monte Carlo, Stochastic calculus
