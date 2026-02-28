# Hedging Strategy

## Concept Skeleton
**Definition:** Dynamic portfolio rebalancing strategy using Greeks to neutralize directional and other risks; maintaining delta-neutral (or target delta) position while monitoring higher-order Greeks  
**Purpose:** Isolate specific risks; reduce or eliminate exposure to spot, volatility, time decay; profit from realized vs. implied volatility differential  
**Prerequisites:** Greeks (Delta, Gamma, Vega, Theta), portfolio management, rebalancing mechanics, transaction costs

## Comparative Framing
| Strategy | Delta-Neutral | Gamma-Neutral | Vega-Neutral | Theta-Neutral |
|----------|---------------|---------------|-------------|--------------|
| **What's Hedged** | Directional spot | Convexity (non-linearity) | Volatility changes | Time decay |
| **Typical Position** | Long option + short delta shares | Long + short gamma instruments | Long + short vega | Long + short theta |
| **Profit Source** | Theta decay (if short vega) | Realized vol > implied | Volatility contraction | Daily decay collection |
| **Loss Source** | Vega exposure | Realized vol < implied | Realized vol spike | Gamma cost |
| **Rehedging Frequency** | Daily/continuous | Continuous | Daily/weekly | Triggered |
| **Transaction Costs** | Moderate | High | Low | Low |

## Examples + Counterexamples
**Simple Example:**  
Long 1 call with Δ = 0.6: Short 0.6 shares to delta-hedge; spot move ↑$1 → call ↑$0.60 - hedge loss $0.60 = $0 (approximately)

**Practical Case:**  
Market maker: Sells volatility (short straddle, negative vega); delta-hedges continuously; profits if realized vol < implied vol; loses if volatility spikes

**Gamma Trap:**  
Delta-hedged long call in falling spot market: Rehedge repeatedly at higher spot levels (buy high, sell low); gamma loss amplifies; total loss = realized vol gain - theta decay

**Calendar Spread Hedging:**  
Buy long-dated call, sell short-dated call: Delta-hedge the short call; long calendar theta; expires in 3mo; roll forward to repeat

## Layer Breakdown
```
Hedging Strategy Framework:
├─ Objective Definition:
│   ├─ Primary risk to eliminate: Delta (directional), Vega (volatility), Gamma (convexity)
│   ├─ Secondary risks: Theta (time decay), higher-order Greeks
│   ├─ Constraints:
│   │   ├─ Budget: Cash for hedges
│   │   ├─ Liquidity: Available instruments
│   │   ├─ Costs: Transaction fees, bid-ask spreads
│   │   └─ Regulatory: Capital requirements, position limits
├─ Delta Hedging (Most Common):
│   ├─ Setup:
│   │   ├─ Initial: Long option, short Δ × N shares (N = option contracts)
│   │   ├─ Δ = ∂V/∂S; negative for puts, positive for calls
│   │   └─ Result: Portfolio ≈ market-neutral (directional insensitive)
│   ├─ Rebalancing:
│   │   ├─ Frequency: Continuous (ideal) vs. discrete (practical)
│   │   ├─ Trigger: Fixed delta threshold (e.g., |Δ| > 0.05) or time interval
│   │   ├─ Rebalance cost: Proportional to γ × (ΔS)² (gamma P&L from spot move)
│   │   └─ Optimal frequency: Balances hedging cost vs. drift risk
│   ├─ P&L Analysis:
│   │   ├─ Spot P&L: ~0 (delta-hedged)
│   │   ├─ Vega P&L: Σ νᵢ × Δσ (volatility exposure remains)
│   │   ├─ Theta P&L: Σ θᵢ / 365 × days (daily time decay)
│   │   ├─ Gamma P&L: Σ γᵢ / 2 × (ΔSᵢ)² (convexity profit/loss)
│   │   └─ Total: θ + γ/2 × (realized vol)² - ν × (implied vol changes)
├─ Vega Hedging:
│   ├─ Goal: Eliminate volatility exposure
│   ├─ Instruments:
│   │   ├─ Opposite-signed vega options (calls/puts)
│   │   ├─ Variance swaps (direct vol hedging)
│   │   ├─ VIX futures (implied vol index)
│   │   └─ Volatility spreads (calendar, diagonal)
│   ├─ Implementation:
│   │   ├─ Compute portfolio vega: Σ νᵢ × Qᵢ (Greeks × quantities)
│   │   ├─ Hedge instrument vega: νₕ
│   │   ├─ Hedge ratio: |Vega_portfolio| / νₕ
│   │   ├─ Short/Long: Opposite sign to portfolio vega
│   │   └─ Monitor: Rebalance if vega drifts beyond threshold
├─ Gamma Management:
│   ├─ Strategy Choices:
│   │   ├─ Long gamma: Long options; profit from moves; pay vega/theta
│   │   ├─ Short gamma: Sell options; collect premium; lose on large moves
│   │   └─ Gamma-neutral: Use spreads (long near-dated, short far-dated)
│   ├─ Risk Profile:
│   │   ├─ Delta-hedged gamma position: P&L = θ + γ/2 × (ΔS)²
│   │   ├─ Long gamma expected profit: ≈ γ / 2 × (realized vol)² × T (annualized)
│   │   ├─ Breakeven vol: Where gamma P&L = theta decay cost
│   │   └─ Exposure: Daily gamma × spot move² → cumulative P&L
├─ Theta Harvesting:
│   ├─ Strategy:
│   │   ├─ Sell short-dated options (high theta decay)
│   │   ├─ Delta-hedge to neutralize spot exposure
│   │   ├─ Collect daily theta as time passes
│   │   ├─ Roll position forward weekly/monthly
│   │   └─ Repeat to scale portfolio
│   ├─ Dynamics:
│   │   ├─ Daily P&L: +θ/365 (decay benefit) - γ/2 × (ΔS)² (if moves occur)
│   │   ├─ Breakeven: Realized vol must be < implied vol (theta benefit > gamma loss)
│   │   ├─ Margin requirement: theta strategies need buffer for adverse moves
│   │   └─ Gamma drag: High gamma near expiry accelerates cost
├─ Practical Considerations:
│   ├─ Costs:
│   │   ├─ Bid-ask spread: Entry + exit cost per rehedge
│   │   ├─ Commissions: Fixed or proportional
│   │   ├─ Slippage: Execution risk; actual price vs. quoted
│   │   └─ Market impact: Large hedges move prices
│   ├─ Frequency Optimization:
│   │   ├─ Daily: Standard (most liquid times)
│   │   ├─ Weekly: Lower cost; more drift risk
│   │   ├─ On-demand: Triggered by threshold breach
│   │   └─ Optimal: Minimizes hedging cost + drift cost
│   ├─ Discretionary Overrides:
│   │   ├─ Vol forecasts: If expecting vol spike, reduce hedging
│   │   ├─ Spot forecasts: If expecting directional move, adjust delta target
│   │   ├─ Liquidity: Reduce rehedging if market stress
│   │   └─ Risk limits: Never exceed position limits despite Greeks
└─ Greeks Portfolio View:
    ├─ Position Greeks: Sum of all Greeks across book
    ├─ Greek limits: Max allowed delta, gamma, vega, theta
    ├─ Rebalancing: Coordinate across all positions
    ├─ Monitoring dashboard: Real-time Greeks with thresholds
    └─ Risk committee: Escalate if Greeks breach limits
```

**Interaction:** Compute portfolio Greeks → identify dominant risks → execute hedges → rebalance on schedule/threshold → repeat

## Challenge Round
When does hedging fail?
- Model risk: Greeks assume BS model; reality has jumps, stochastic vol, correlations
- Execution delays: Can't hedge instantly; gap risk overnight or during market stress
- Liquidity crunch: Can't execute hedges due to illiquidity; forced to hold unhedged
- Correlation breakdowns: Assumed hedges become imperfect (e.g., basis risk)
- Tail events: Greeks linear approximations; massive moves break assumptions

## Key References
- [Hull - Options, Futures & Derivatives (Chapters 19-20)](https://www-2.rotman.utoronto.ca/~hull)
- [Taleb - Dynamic Hedging (Complete)](https://www.paulwilmott.com)
- [Natenberg - Option Volatility & Pricing (Chapters 15-16)](https://www.amazon.com/Option-Volatility-Pricing-Advanced-Strategies/dp/1557784124)

---
**Status:** Core portfolio management technique | **Complements:** Greeks Framework, Risk Management, Portfolio Greeks
