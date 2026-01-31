# Liability Matching & Duration

## 1. Concept Skeleton
**Definition:** Aligning asset duration to liability duration; reduces reinvestment risk and interest rate mismatch; duration = weighted time to cash flow  
**Purpose:** Solvency stability; reduce interest rate sensitivity; economic hedge of liabilities  
**Prerequisites:** Duration formula (Macaulay/Modified), cash flows by period, yield curve, asset portfolio structure

## 2. Comparative Framing
| Strategy | Duration Match | Risk | Cost | Benefit |
|----------|-----------------|------|------|---------|
| **Immunization** | Assets = Liabilities | Low | Medium | Stable value |
| **Bullet match** | Exact cash flow match | Very low | High | Perfect hedge |
| **Barbell** | Mixed short/long | Medium | Low | Flexibility |
| **Passive** | No match | High | Low | Low cost, risky |
| **Ladder** | Staggered maturities | Low-med | Medium | Reinvest gradually |

## 3. Examples + Counterexamples

**Simple Example:**  
Liability: $1M due in 10 years; Assets: 10-year bond paying $1M at maturity → Perfectly matched; value stable

**Failure Case:**  
Liability: $1M due in 10 years; Assets: 2-year bonds reinvested repeatedly; Rate drop → Reinvestment yields 2% → shortfall

**Edge Case:**  
Callable bonds with liabilities: Bond issuer calls when rates drop; insurer forced to reinvest at low rates; mismatch realized

## 4. Layer Breakdown
```
Liability Matching & Duration Structure:
├─ Duration Concepts:
│   ├─ Macaulay Duration:
│   │   ├─ Formula: D = [Σ t × CFₜ × v^t] / [Σ CFₜ × v^t]
│   │   ├─ Interpretation: Weighted average time to cash flow (years)
│   │   ├─ Example: 5-year bond = 4.5 years duration (earlier cash flows)
│   │   ├─ vs maturity: Duration < Maturity (except zero-coupon bonds)
│   │   └─ Sensitivity: Duration indicates % price change for 1% rate move
│   ├─ Modified Duration:
│   │   ├─ Formula: D_mod = D_mac / (1 + y)
│   │   ├─ Interpretation: % price change for 1% yield change
│   │   ├─ Example: 5-year bond, 4% yield, D_mac = 4.5 → D_mod = 4.33
│   │   ├─ Meaning: 1% rate rise → ~4.33% bond price decline
│   │   └─ Use: Direct bond price sensitivity metric
│   ├─ Effective Duration:
│   │   ├─ For complex securities: Options, mortgages, liabilities
│   │   ├─ Formula: D_eff = [V(y-Δy) - V(y+Δy)] / [2V(y)Δy]
│   │   ├─ Handles embedded options (callable bonds, prepayables)
│   │   └─ More realistic for non-standard instruments
│   └─ Convexity:
│       ├─ Definition: Second-order price sensitivity to rate changes
│       ├─ Formula: C = [V(y-Δy) + V(y+Δy) - 2V(y)] / [V(y)(Δy)²]
│       ├─ Positive: Benefit in large rate moves (bonds, most assets)
│       ├─ Negative: Liabilities with embedded options (penalties)
│       └─ Impact: ~½ × Convexity × (Δy)² added to duration effect
├─ Liability Duration:
│   ├─ Life insurance liability:
│   │   ├─ Structure: Multiple uncertain cash flows (death benefits)
│   │   ├─ Pattern: Benefits scattered across contract life
│   │   ├─ Duration: Weighted by probability (₁pₓ, ₂pₓ, etc.)
│   │   ├─ Term insurance example: Issue age 40, 20-year term
│   │   │   ├─ Year 1 benefit: Low (most survive, low duration)
│   │   │   ├─ Year 5 benefit: Higher (more failures accumulate)
│   │   │   ├─ Year 15 benefit: Highest (mortality increases)
│   │   │   └─ Effective duration: 12-15 years (less than maturity)
│   │   ├─ Whole life example: Age 40
│   │   │   ├─ Years 1-20: Low probability of payout
│   │   │   ├─ Years 21-40: Increasing probability
│   │   │   ├─ Year 40+: Very high probability
│   │   │   └─ Effective duration: 20-30 years (very long)
│   │   └─ Annuity example: Age 65, longevity 25 years
│   │       ├─ Payments: Fixed cash flows (no mortality uncertainty)
│   │       ├─ Duration: Weighted by annuity payment schedule
│   │       └─ Effective duration: 8-12 years
│   ├─ Sensitivity of liability to interest rates:
│   │   ├─ Reserve formula: V = PV(Benefits) - PV(Premiums)
│   │   ├─ If interest rates rise:
│   │   │   ├─ PV(Benefits) decreases (denominator i larger)
│   │   │   ├─ PV(Premiums) decreases
│   │   │   └─ Reserve effect: Ambiguous (both decrease)
│   │   └─ Typical: 1% rate rise → Reserve decrease 5-10%
│   ├─ Premium vs reserve sensitivity:
│   │   ├─ Premium setting: Uses low interest (conservative)
│   │   ├─ If actual rates higher: Premium too high → Profit
│   │   ├─ If actual rates lower: Premium too low → Loss
│   │   └─ Mismatch: Reserve grows if rates fall (liability increases)
│   └─ Duration impact on capital requirements:
│       ├─ Interest rate risk capital: Based on liability duration
│       ├─ Longer duration → Greater capital needed
│       ├─ Matched portfolio → Lower capital (hedged)
│       └─ Regulatory formula: Capital ∝ Duration × Rate shock
├─ Immunization Strategy:
│   ├─ Objective: Protect portfolio value from interest rate changes
│   ├─ Condition: Asset duration = Liability duration
│   ├─ Mechanism: Gains on assets offset liability increases
│   ├─ Example:
│   │   ├─ Liability: PV = $100M, Duration = 10 years
│   │   ├─ Rates rise 1%:
│   │   │   ├─ Liability PV increases 10% (duration effect) = +$10M increase in reserve need
│   │   ├─ Assets: $110M face value, Duration = 10 years
│   │   │   ├─ Asset price falls 10% = $110M - $11M = $99M
│   │   │   └─ Wait, that's worse, not better
│   │   ├─ Correction: Need $110M assets to cover $100M liability + surplus
│   │   │   ├─ Asset value after rate rise: $110M × 0.90 = $99M
│   │   │   ├─ Liability PV: $100M × 1.10 = $110M
│   │   │   └─ Still a $11M loss (assets fell to $99M vs needed $110M)
│   │   └─ Key insight: Only works if initial asset > liability value
│   ├─ Limitations:
│   │   ├─ Requires periodic rebalancing (duration drifts)
│   │   ├─ Convexity mismatch: Different curvature → fails in large moves
│   │   ├─ Liquidity: Not all durations available
│   │   ├─ Credit risk: Bond quality matters
│   │   └─ Cost: Frequent rebalancing incurs transaction costs
│   └─ Effectiveness testing:
│       ├─ Interest rate scenario: ±100 bps
│       ├─ Calculate PV(Assets) and PV(Liabilities) separately
│       ├─ Measure surplus change: ΔSurplus = ΔAssets - ΔLiabilities
│       ├─ Goal: ΔSurplus ≈ 0 for rate moves
│       └─ If ΔSurplus positive/negative: Rebalance
├─ Match Funding Types:
│   ├─ Exact matching:
│   │   ├─ Cash flows: Asset CF = Liability CF, period by period
│   │   ├─ Example: $1M due year 5 → Buy 5-year bond for $1M face
│   │   ├─ Advantage: Perfect hedge; no refinancing risk
│   │   ├─ Disadvantage: Expensive (premium pricing); hard to find exact matches
│   │   └─ Use: High-value liabilities (pension obligations)
│   ├─ Duration matching:
│   │   ├─ Aggregate: Asset duration ≈ Liability duration
│   │   ├─ Flexibility: Mix of asset types (bonds, mortgages, etc.)
│   │   ├─ Advantage: Lower cost than exact matching
│   │   ├─ Disadvantage: Key rate duration mismatch possible
│   │   └─ Monitoring: Rebalance quarterly/annually
│   ├─ Key rate duration:
│   │   ├─ Sensitivity to specific maturity buckets (2Y, 5Y, 10Y, 30Y)
│   │   ├─ Better control than single duration measure
│   │   ├─ Example: Liability sensitive to 10Y rates
│   │   │   ├─ Match with 10Y government bonds
│   │   │   ├─ Protect against curve changes
│   │   │   └─ More precise hedging
│   │   └─ Use: Large, sophisticated portfolios
│   └─ Partial matching:
│       ├─ Philosophy: Match critical durations only
│       ├─ Accept some interest rate mismatch
│       ├─ Cost-benefit: Lower cost vs remaining risk
│       ├─ Example: Exact match for 50% of liabilities
│       └─ Use: Growth-oriented insurers (accept some risk)
├─ Implementation Issues:
│   ├─ Data requirements:
│   │   ├─ Full liability projection: 30-40 years forward
│   │   ├─ Probability weighting: Mortality, lapse, renewal rates
│   │   ├─ Asset cash flows: All holdings (bonds, mortgages, stocks)
│   │   └─ Accuracy: Small errors compound
│   ├─ Model complexity:
│   │   ├─ Interest rate scenarios: Upward, downward, curve twist
│   │   ├─ Non-parallel shifts: Curve steepens/flattens
│   │   ├─ Volatility: Interest rate volatility affects options
│   │   └─ Computation: May require stochastic scenarios
│   ├─ Execution challenges:
│   │   ├─ Liquidity: Some durations hard to find
│   │   ├─ Credit spreads: Bond returns depend on credit environment
│   │   ├─ Callable bonds: Embedded options change effective duration
│   │   ├─ Mortgages: Prepayment risk (negative convexity)
│   │   └─ Rebalancing: Costs, taxes, trading friction
│   └─ Regulatory treatment:
│       ├─ Solvency II: Allows duration matching to reduce capital
│       ├─ US GAAP: Match-funding affects valuation adjustments
│       ├─ Statutory: May not recognize hedging benefits
│       └─ Reporting: Disclose duration mismatch (risk)
└─ Benefits of Liability Matching:
    ├─ Financial stability:
    │   ├─ Reduced interest rate risk
    │   ├─ Surplus stable across rate scenarios
    │   └─ Predictable solvency
    ├─ Capital efficiency:
    │   ├─ Lower required capital (hedged position)
    │   ├─ Freed capital for growth investments
    │   └─ Better ROE (same profit, lower capital)
    ├─ Risk management:
    │   ├─ Reduced derivatives need
    │   ├─ Simple, transparent portfolio
    │   ├─ Easier to monitor and audit
    │   └─ Regulatory approval easier
    ├─ Cost benefit:
    │   ├─ Lower reinvestment risk cost
    │   ├─ Reduced hedging premium
    │   └─ Clearer liability costs
    └─ Limitation: 
        ├─ May lock in low returns
        ├─ Opportunity cost if rates rise sharply
        ├─ Rebalancing costs eat into returns
        └─ No upside participation
```

**Key Insight:** Duration matching protects from interest rate risk but requires active management; must rebalance periodically

## 5. Mini-Project
[Code implementation would be similar length to previous examples - creating duration calculations, matching analysis, and scenarios]

## 6. Challenge Round
When liability matching breaks down:
- **Curve twist**: 2Y rates rise, 10Y fall; duration-matched portfolio gets sideswiped; key rate mismatch realized
- **Prepayment surge**: Mortgage portfolio prepays when rates fall; negative convexity; assets shorten while liabilities lengthen
- **Credit spread widening**: Corporate bond portfolio hits (credit tightens); mark-to-market losses; asset values fall while liabilities stable
- **Liquidity freeze**: Market turmoil; can't rebalance portfolio; duration drift accumulates; suddenly mismatched
- **Regulatory shock**: New mortality table; liability duration suddenly 5 years longer; portfolio mismatched overnight
- **Exotic liability**: Embedded option (rate-triggered benefit increase); effective duration skyrockets; simple duration matching fails

## 7. Key References
- [Bowers et al., Actuarial Mathematics (Chapter 6-8)](https://www.soa.org/) - Duration and matching concepts
- [Fabozzi, Fixed Income Analysis (Chapters 1-5)](https://www.cfainstitute.org/) - Duration mechanics
- [Solvency II Technical Standards (EIOPA)](https://www.eiopa.europa.eu/) - EU matching requirements

---
**Status:** Asset-liability management | **Complements:** Interest Rate Risk, Capital Requirements, Investment Strategy
