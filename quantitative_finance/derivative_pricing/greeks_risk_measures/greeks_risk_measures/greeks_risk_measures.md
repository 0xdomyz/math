# Greeks and Risk Measures

## Concept Skeleton
**Definition:** First-order and higher-order partial derivatives of option value with respect to underlying parameters; quantify sensitivity and risk exposure  
**Purpose:** Measure price sensitivity to spot, volatility, time, rates; enable delta hedging, portfolio risk management, and dynamic replication strategies  
**Prerequisites:** Option pricing basics, partial derivatives, Black-Scholes model, portfolio theory, calculus of variations

## Comparative Framing
| Greek | Definition | Interpretation | Hedge Frequency |
|-------|------------|----------------|-----------------|
| **Delta (Δ)** | ∂V/∂S | Shares to hedge | Continuous (practical: daily) |
| **Gamma (Γ)** | ∂²V/∂S² = ∂Δ/∂S | Delta stability | When large (nonlinear) |
| **Vega (ν)** | ∂V/∂σ | Volatility exposure | Weekly/event-driven |
| **Theta (Θ)** | ∂V/∂t | Time decay | Cannot hedge (time passes) |
| **Rho (ρ)** | ∂V/∂r | Interest rate risk | Rarely (stable rates) |

| Risk Measure | Greeks Used | Purpose | Timescale |
|--------------|-------------|---------|-----------|
| **Delta Risk** | Δ | Directional exposure | Intraday |
| **Gamma Risk** | Γ | Curvature, gap risk | Daily |
| **Vega Risk** | ν | Vol surface changes | Weekly |
| **Higher-Order** | Vanna, Volga | Cross-sensitivities | Strategic |

## Examples + Counterexamples
**Simple Example:**  
ATM call Δ≈0.5: Hedge 100 calls with -50 shares. Stock up $1 → call gains ~$50, shares lose $50 → net flat.

**Perfect Fit:**  
Market maker: Sells 1000 calls (Δ=0.6 each), immediately buys 600 shares to delta-hedge. Rebalances daily as Δ changes.

**Gamma Bomb:**  
Large short ATM options near expiry: Γ explodes → Δ changes rapidly → hedging requires massive stock trades → potential losses from slippage.

**Vega Exposure:**  
VIX spike from 15% to 30%: Long straddle (high +vega) profits significantly even if spot unchanged. Short vol position bleeds.

**Theta Decay:**  
OTM option 1 week to expiry: Θ≈-0.05 per day. Each day loses $5 in time value even if spot/vol stable. Cannot hedge, only by offsetting position.

**Poor Fit:**  
Using BS Greeks for deep OTM options near barriers: Model assumes continuous prices, ignores jump risk → Greeks misleading.

## Layer Breakdown
```
Greeks Framework:

├─ First-Order Greeks:
│  ├─ Delta (Δ): ∂V/∂S
│  │   ├─ Call Delta: 0 < Δ < 1 (typically 0.5 ATM)
│  │   ├─ Put Delta: -1 < Δ < 0 (typically -0.5 ATM)
│  │   ├─ Interpretation: Hedge ratio (shares per option)
│  │   ├─ Portfolio Delta: Σ(Δᵢ × Positionᵢ)
│  │   ├─ Delta-neutral: Portfolio Δ = 0
│  │   └─ Delta hedging: Buy/sell stock to maintain Δ=0
│  ├─ Vega (ν): ∂V/∂σ
│  │   ├─ Always positive for vanilla options
│  │   ├─ Maximum for ATM options
│  │   ├─ Interpretation: P&L change per 1% vol shift
│  │   ├─ Vega hedging: Use other options (calendar spreads)
│  │   └─ Volatility smile: Different strikes have different vegas
│  ├─ Theta (Θ): ∂V/∂t (time decay)
│  │   ├─ Negative for long options (lose time value)
│  │   ├─ Positive for short options (earn premium decay)
│  │   ├─ Accelerates near expiry (convex decay)
│  │   ├─ ATM options: Highest absolute theta
│  │   └─ Cannot hedge except with offsetting options
│  └─ Rho (ρ): ∂V/∂r
│      ├─ Call Rho: Positive (higher r → higher call value)
│      ├─ Put Rho: Negative (higher r → lower put value)
│      ├─ Long maturity: Higher sensitivity
│      └─ Rarely hedged in practice (stable rates)
├─ Second-Order Greeks:
│  ├─ Gamma (Γ): ∂²V/∂S² = ∂Δ/∂S
│  │   ├─ Measures delta stability (convexity)
│  │   ├─ Always positive for long options
│  │   ├─ Maximum for ATM options
│  │   ├─ Spikes near expiry for ATM
│  │   ├─ Interpretation: How fast delta changes
│  │   ├─ High Γ → Requires frequent delta rebalancing
│  │   ├─ Gamma P&L: ½Γ(ΔS)² (from realized volatility)
│  │   └─ Gamma risk: Large moves cause hedging losses
│  ├─ Vanna: ∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S
│  │   ├─ Cross-sensitivity: Spot-vol interaction
│  │   ├─ Important for skew dynamics
│  │   └─ Used in advanced vol surface modeling
│  ├─ Volga (Vomma): ∂²V/∂σ² = ∂ν/∂σ
│  │   ├─ Vega convexity (vega stability)
│  │   ├─ Positive for vanilla options
│  │   └─ Relevant for large vol changes
│  └─ Charm: ∂²V/∂S∂t = ∂Δ/∂t
│      ├─ Delta decay over time
│      └─ Useful for forward delta projections
├─ Greeks in Black-Scholes:
│  ├─ Call Delta: Δ_call = N(d₁)
│  ├─ Put Delta: Δ_put = N(d₁) - 1 = -N(-d₁)
│  ├─ Gamma (same for call/put):
│  │   Γ = φ(d₁) / (S σ √T)
│  │   where φ(x) = (1/√(2π)) exp(-x²/2)
│  ├─ Vega (same for call/put):
│  │   ν = S φ(d₁) √T
│  ├─ Theta:
│  │   Θ_call = -S φ(d₁) σ/(2√T) - rK e^(-rT) N(d₂)
│  │   Θ_put = -S φ(d₁) σ/(2√T) + rK e^(-rT) N(-d₂)
│  └─ Rho:
│      ρ_call = K T e^(-rT) N(d₂)
│      ρ_put = -K T e^(-rT) N(-d₂)
├─ Delta Hedging:
│  ├─ Objective: Eliminate directional risk (Δ=0)
│  ├─ Static hedge: Set at inception, don't adjust
│  │   → Only works for linear payoffs or perfect replication
│  ├─ Dynamic hedge: Continuously rebalance
│  │   ├─ Frequency tradeoff: Transaction costs vs tracking error
│  │   ├─ Discrete rehedging: Gamma risk accumulates
│  │   └─ P&L attribution: Theta + ½Γ(ΔS)²
│  ├─ Delta-hedging portfolio:
│  │   Portfolio = Options + Δ_hedge × Stock
│  │   where Δ_hedge = -Σ(Δ_option × Quantity)
│  ├─ Gamma scalping:
│  │   ├─ Profit from realized vol > implied vol
│  │   ├─ Long gamma (long options): Profit from rebalancing
│  │   └─ Short gamma (short options): Lose from rebalancing
│  └─ Hedging errors:
│      ├─ Discrete rebalancing → Gamma P&L
│      ├─ Model risk (wrong σ) → Vega P&L
│      └─ Transaction costs → Drag on performance
├─ Portfolio-Level Risk:
│  ├─ Net Greeks:
│  │   Δ_portfolio = Σᵢ Δᵢ × Positionᵢ
│  │   Γ_portfolio = Σᵢ Γᵢ × Positionᵢ
│  │   ν_portfolio = Σᵢ νᵢ × Positionᵢ
│  ├─ Risk limits:
│  │   ├─ Delta limit: Maximum directional exposure
│  │   ├─ Gamma limit: Maximum convexity risk
│  │   ├─ Vega limit: Maximum vol exposure
│  │   └─ Aggregated across underlyings, maturities
│  ├─ Greeks by strike/maturity:
│  │   Grid of exposures across vol surface
│  ├─ Greeks ladder (term structure):
│  │   Vega by maturity bucket (1m, 3m, 6m, 1y, etc.)
│  └─ Correlation risk:
│      Greeks across multiple underlyings → correlation matrix
├─ Advanced Topics:
│  ├─ Forward Greeks:
│  │   Greeks at future date (for forward-starting options)
│  ├─ Greeks with dividends:
│  │   Modify formulas for continuous yield q
│  │   Δ_call = e^(-qT) N(d₁)
│  ├─ Greeks for exotic options:
│  │   ├─ Barrier options: Discontinuous delta at barrier
│  │   ├─ Digital options: Dirac delta (infinite gamma)
│  │   └─ Asian options: Path-dependent → weighted Greeks
│  ├─ Model-free Greeks:
│  │   Extract from option prices without assuming model
│  ├─ Greeks hedging with options:
│  │   ├─ Vega hedge: Use different strike/maturity options
│  │   ├─ Gamma hedge: Buy/sell options (cannot hedge with stock)
│  │   └─ Multi-Greek hedging: System of equations
│  └─ Jump risk (beyond Greeks):
│      Greeks assume continuous paths → fail for gaps
└─ Practical Considerations:
   ├─ Bid-ask spreads: Affect hedging costs
   ├─ Discrete rebalancing: Optimal frequency analysis
   ├─ Greeks aggregation: By book, desk, firm-wide
   ├─ Real-time Greeks: Streaming calculations for traders
   ├─ Greeks reporting: Daily risk reports for management
   └─ Regulatory capital: Risk-weighted assets based on Greeks
```

**Interaction:** Delta measures slope, Gamma measures curvature; hedging Delta creates Gamma exposure; Theta compensates Gamma in delta-hedged portfolio (Black-Scholes PDE).

## Challenge Round
1. **Vanna & Volga:** Implement second-order cross Greeks. How do they change with skew? When are they significant?

2. **Gamma Scalping P&L:** Simulate discrete rehedging (daily vs hourly). Calculate realized Gamma P&L. How does frequency affect profit?

3. **Multi-Greek Hedging:** Build portfolio that is delta-neutral, gamma-neutral, and vega-neutral using 3 different options. Solve system of equations.

4. **Greeks at Barriers:** Calculate Greeks for knock-out option. How does Delta behave near barrier? What's discontinuity at barrier?

5. **Optimal Rehedging:** Find optimal delta-hedge rebalancing frequency minimizing sum of transaction costs and tracking error. Is there a closed-form solution?

## Key References
- [Hull, Options, Futures, and Other Derivatives (Chapter 19)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000006649)
- [Taleb, Dynamic Hedging (Chapter 7-9)](https://www.wiley.com/en-us/Dynamic+Hedging-p-9780471152804)
- [Wilmott, Paul Wilmott on Quantitative Finance (Greeks Section)](https://www.wiley.com/en-us/Paul+Wilmott+on+Quantitative+Finance-p-9781118836798)
- [Haug, The Complete Guide to Option Pricing Formulas (Appendix A)](https://www.mhprofessional.com/the-complete-guide-to-option-pricing-formulas-9780071389976-usa)

---
**Status:** Core risk management tool | **Complements:** Black-Scholes Model, Delta Hedging, Portfolio Risk Management, Option Trading Strategies
