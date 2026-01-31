# Dynamic Hedging & Derivatives Strategy

## 1. Concept Skeleton
**Definition:** Using derivatives (swaps, forwards, options) to hedge changing interest rate, mortality, or market risks; continuously rebalanced  
**Purpose:** Reduce interest rate/longevity/equity exposure; lock in returns; manage tail risks; reduce capital requirements  
**Prerequisites:** Derivative pricing, mark-to-market valuation, counterparty credit assessment, operational systems for rebalancing

## 2. Comparative Framing
| Strategy | Instrument | Payoff | Cost | Duration |
|----------|-----------|--------|------|----------|
| **Interest rate swap** | Swap | Fixed vs floating exchange | 0 upfront | 5-30Y |
| **Longevity swap** | Swap | Fixed vs actual mortality | 0-1% | 10-50Y |
| **Swaption** | Option | Right to enter swap | 1-3% | 1-10Y |
| **Collar** | Option combo | Floor + cap on rate | 0-0.5% (cost-free) | 1-10Y |
| **Bond futures** | Futures | Hedge portfolio duration | 0 margin | 3-12M |
| **Mortality forward** | Forward | Fixed mortality rate | 0 upfront | 1-5Y |
| **CAT bond** | Securitization | Transfer tail risk | 15-25% yield | 3-5Y |

## 3. Examples + Counterexamples

**Simple Example:**  
Insurer: $500M annuity portfolio, interest rate risk; buys 10Y interest rate swap (pay fixed 3%, receive floating); if rates fall, receive compensates annuity reserve loss

**Failure Case:**  
Hedged with 5Y swap; rates were expected stable; instead rates fall 200 bps; swap value increases $60M; good hedge! But then counterparty (Lehman Bros) defaults; swap value evaporates; no benefit received

**Edge Case:**  
Mortality swap collars: Insurer pays if mortality < 90% of assumption; receives if > 110%; paradoxical: pays to protect on both sides; designed to transfer tail risk above collar, not average risk

## 4. Layer Breakdown
```
Dynamic Hedging & Derivatives Strategy Structure:
├─ Interest Rate Hedging:
│   ├─ Interest rate swap mechanics:
│   │   ├─ Definition: Exchange fixed rate payments for floating (or vice versa)
│   │   ├─ Parties: Insurer (fixed payer) vs Bank (floating payer)
│   │   ├─ Cash flows:
│   │   │   ├─ Insurer pays: Fixed rate × Notional × Period (e.g., 3% × $100M × 0.5Y = $1.5M/semi-annual)
│   │   │   ├─ Bank pays: LIBOR × Notional × Period (e.g., 2.8% × $100M × 0.5Y = $1.4M)
│   │   │   ├─ Net: Insurer pays 0.2%, Bank pays 0.8% (if LIBOR > 3%)
│   │   │   └─ Notional never exchanged (only interest)
│   │   ├─ Purpose for insurer:
│   │   │   ├─ Asset side: Fixed rate assets (bonds) generate fixed income
│   │   │   ├─ Liability side: Reserve declines if rates fall (bad for insurer)
│   │   │   ├─ Swap: Lock in fixed liability cost; if rates fall, swap gain offsets
│   │   │   └─ Duration matching: Pay fixed = extend duration of liability match
│   │   ├─ Pricing:
│   │   │   ├─ At initiation: Fixed rate = current LIBOR + tenor spread
│   │   │   ├─ Example: 10Y swap rate = LIBOR + 0.5% = 3.0% (if LIBOR 2.5%)
│   │   │   ├─ Mark-to-market: Daily valuation as rates change
│   │   │   └─ Gain/loss: If rates rise (swap rate ↑ to 3.5%), fixed payer loses (locked at 3%)
│   │   ├─ Accounting treatment:
│   │   │   ├─ Hedge accounting (if qualified): Gains/losses offset liability changes
│   │   │   ├─ Mark-to-market (if not): Swap gain/loss flows to income statement
│   │   │   ├─ Volatility: Can swing ±10-20% of notional annually
│   │   │   └─ Strategic: Hedge accounting reduces earnings volatility
│   │   └─ Duration of swap:
│   │       ├─ 2Y swap: Duration ~1.8Y
│   │       ├─ 5Y swap: Duration ~4.4Y
│   │       ├─ 10Y swap: Duration ~8.5Y
│   │       ├─ 20Y swap: Duration ~16.5Y
│   │       └─ Matches liability if durations aligned
│   ├─ Swaption (option on swap):
│   │   ├─ Definition: Right (not obligation) to enter swap at fixed rate
│   │   ├─ Payer swaption: Right to pay fixed (buy if expect rates to fall)
│   │   ├─ Receiver swaption: Right to receive fixed (buy if expect rates to rise)
│   │   ├─ Premium: Typically 1-3% of notional
│   │   ├─ Example:
│   │   │   ├─ Buy 5Y payer swaption on 10Y swap
│   │   │   ├─ Strike: 3.0% (right to pay 3% in 5 years for 10Y swap)
│   │   │   ├─ Cost: $1.5M upfront (1.5% × $100M notional)
│   │   │   ├─ Scenario 1 (Rates fall to 2.5%):
│   │   │   │   ├─ Exercise: Enter swap to pay 3.0% (vs market 2.5%)
│   │   │   │   ├─ Value: 0.5% × remaining duration × $100M ≈ $4M
│   │   │   │   ├─ Profit: $4M - $1.5M premium = $2.5M net gain
│   │   │   │   └─ Or: Don't exercise; lose premium ($1.5M loss)
│   │   │   ├─ Scenario 2 (Rates rise to 3.5%):
│   │   │   │   ├─ Don't exercise: Swap in market at 3.5%, avoid paying 3.0%
│   │   │   │   ├─ Loss: Premium only ($1.5M)
│   │   │   │   └─ Optionality: Pays off if rates move favorably
│   │   │   └─ Use: Asymmetric protection; pay premium for upside potential
│   │   └─ Valuation:
│   │       ├─ Black-Karasinski model: Common for swaption pricing
│   │       ├─ Inputs: Spot yield curve, volatility, maturity, strike
│   │       └─ Output: Fair value of premium
│   ├─ Collar strategy (cost-neutral hedge):
│   │   ├─ Definition: Buy floor (receive fixed if rates fall), sell cap (pay fixed if rates rise)
│   │   ├─ Result: Cost neutral (premium received = premium paid)
│   │   ├─ Payoff:
│   │   │   ├─ Floor strike: 2.5% (insurer protected if rates fall below)
│   │   │   ├─ Cap strike: 3.5% (insurer accepts rates up to this level)
│   │   │   ├─ Rates at 2.0%: Receive 0.5% from floor
│   │   │   ├─ Rates at 3.0%: No payment (within collar)
│   │   │   ├─ Rates at 4.0%: Pay cap; capped at 0.5% (3.5% - 4.0%)
│   │   │   └─ Range: Protected if rates 0.5% below/above collar
│   │   └─ Application:
│   │       ├─ Zero cost attractive for insurance companies
│   │       ├─ Downside protection + limit upside benefit
│   │       ├─ Common structure for 5-10Y periods
│   │       └─ Trade-off: Miss gains above cap; protected below floor
│   └─ Bond futures (short-term hedge):
│       ├─ Definition: Exchange-traded futures contract on bond portfolio
│       ├─ Characteristics:
│       │   ├─ Standardized: US Treasury notes (CTD "Cheapest to Deliver")
│       │   ├─ Notional: $100K, quoted as % of face (e.g., 98-10 = 98 + 10/32)
│       │   ├─ Margin: 2-3% of notional (required for position)
│       │   ├─ Settlement: Physical delivery or cash at expiration
│       │   └─ Liquidity: Extremely high (1000s of contracts trade per day)
│       ├─ Hedging mechanism:
│       │   ├─ Portfolio: $50M of 10Y bonds
│       │   ├─ Futures hedge: Sell 500 TNote futures ($100K each)
│       │   ├─ Hedge ratio: Duration-adjusted
│       │   │   ├─ Portfolio duration: 8 years
│       │   │   ├─ Futures contract duration: 6 years
│       │   │   ├─ Contracts needed: (8/6) × ($50M / $100K) = 667 contracts
│       │   │   └─ Actual: Use 650-670 to match exposure
│       │   ├─ Rate scenario: Rates rise 1%
│       │   │   ├─ Portfolio loss: -8% × $50M = -$4M
│       │   │   ├─ Futures gain: +6% × 667 × $100K = +$4M (approx)
│       │   │   └─ Net: ~$0 loss (protected)
│       │   └─ Rebalancing: Adjust hedge weekly/monthly as rates change
│       ├─ Advantages:
│       │   ├─ Low cost: Margin only 2-3% (no upfront premium)
│       │   ├─ Liquid: Easy to adjust size
│       │   ├─ Transparent: Price visible real-time
│       │   └─ Flexible: Can hedge partial portfolio
│       ├─ Disadvantages:
│       │   ├─ Basis risk: Futures don't track exact portfolio (CTD changes, curve twist)
│       │   ├─ Rebalancing: Requires active management
│       │   ├─ Margin call: Adverse moves require cash posting
│       │   ├─ Accounting: Can force mark-to-market in earnings
│       │   └─ Tail risk: Very large rate moves break hedge
│       └─ Use case: Short-term tactical hedges (weeks to months)
├─ Mortality & Longevity Hedging:
│   ├─ Longevity swap mechanics:
│   │   ├─ Parties: Insurer (pays fixed mortality cost) vs Reinsurer/Bank (receives fixed, pays actual)
│   │   ├─ Cash flows:
│   │   │   ├─ Insurer pays: Fixed PV of assumed mortality benefit stream
│   │   │   │   ├─ Example: Age 65+ annuity, 1,000 lives, $20K annual payout
│   │   │   │   ├─ Expected payouts (mortality assumption): $14M/year
│   │   │   │   ├─ PV at 3%: $14M / 3% = $467M (perpetuity assumption)
│   │   │   │   └─ Insurer pays: ~$467M upfront (or annual installments)
│   │   │   ├─ Bank pays: Actual benefit payout (survivors continue)
│   │   │   │   ├─ If actual mortality > assumption: Bank pays more (good for insurer)
│   │   │   │   ├─ If actual mortality < assumption: Bank pays less (bad for insurer, but infrequent)
│   │   │   │   └─ Insurer benefits: Insured against longevity trend
│   │   │   └─ Net: Insurer transfers longevity risk to reinsurer
│   │   ├─ Duration of longevity swap:
│   │   │   ├─ Portfolio: 1,000 lives, age 65, female
│   │   │   ├─ Expected payouts: Years 1-30 (some to age 95+)
│   │   │   ├─ Effective duration: 15-20 years
│   │   │   └─ Notional: $467M on $20B company = 2.3% of balance sheet
│   │   ├─ Counterparty risk:
│   │   │   ├─ If bank fails: No more longevity insurance; insurer bears risk
│   │   │   ├─ Mitigation: Insurance on reinsurer; diversify counterparties
│   │   │   └─ Credit spread: Bank includes 100-150 bps for credit risk
│   │   └─ Accounting treatment:
│   │       ├─ Hedge accounting: Longevity gain/loss offsets liability changes
│   │       ├─ Effect: Reduces earnings volatility from mortality changes
│   │       └─ Regulators: Generally accept longevity swaps as hedge
│   ├─ Mortality forward (simpler instrument):
│   │   ├─ Definition: Agreement to pay fixed mortality cost for fixed period
│   │   ├─ Duration: Typically 1-5 years (shorter than longevity swap)
│   │   ├─ Example:
│   │   │   ├─ 3-year mortality forward on life insurance block
│   │   │   ├─ Notional benefit: $50M/year
│   │   │   ├─ Fixed rate: 103% of expected deaths (1.5M cost)
│   │   │   ├─ If actual mortality 110%: Insurer saves 7% × $50M = $3.5M
│   │   │   ├─ If actual mortality 100%: Insurer loses 3% × $50M = $1.5M
│   │   │   └─ Use: Protect against short-term mortality volatility
│   │   └─ Cost: 0-0.5% of premium volume (cheaper than longevity swap)
│   ├─ Catastrophe bond (tail risk hedge):
│   │   ├─ Definition: Bond that pays coupon + principal only if catastrophe doesn't occur
│   │   ├─ Structure:
│   │   │   ├─ Insurer: Issues CAT bond $100M, 3% coupon, 5Y maturity
│   │   │   ├─ Investor (hedge fund, pension): Buys bond for $100M
│   │   │   ├─ Coupon: 3% + spread (typically 8-15% = 11-18% total yield)
│   │   │   ├─ Trigger: If catastrophe hits (e.g., major hurricane, pandemic)
│   │   │   │   ├─ Trigger amount: $500M+ insured loss
│   │   │   │   ├─ Bond principal written down (e.g., $100M → $60M)
│   │   │   │   ├─ Insurer keeps: $40M of loss absorption
│   │   │   │   └─ Investor loses: $40M principal (but saved coupon in interim)
│   │   │   └─ No trigger: Bond pays full $100M + interest at maturity
│   │   ├─ Economics:
│   │   │   ├─ Insurer cost: 11% yield on $100M = $11M/year (expensive!)
│   │   │   ├─ But: Hedges $500M potential loss
│   │   │   ├─ Fair trade if: P(catastrophe) > $11M / $500M = 2.2% annually
│   │   │   ├─ Typical cat probability: 1-5%/year (varies by region)
│   │   │   └─ Decision: If P > 2.2%, CAT bond makes sense
│   │   ├─ Advantages:
│   │   │   ├─ Transfers tail risk: Catastrophe absorbed by capital market
│   │   │   ├─ No counterparty risk: Collateral held (reduces credit risk)
│   │   │   ├─ Capital relief: Hedge not counted as asset; capital freed
│   │   │   └─ Diversification: External capital takes risk
│   │   ├─ Disadvantages:
│   │   │   ├─ Expensive: 8-15% yield = high insurance cost
│   │   │   ├─ Basis risk: Trigger may not match actual loss
│   │   │   ├─ Liquidity: Can't easily unwind early
│   │   │   └─ Complexity: Complex structure; investor education needed
│   │   └─ Market: $200B+ annual issuance; growing steadily
│   └─ Mortality swap collar:
│       ├─ Structure: Buy floor (receive if mortality < 90% assumption) + Sell cap (pay if mortality > 110%)
│       ├─ Cost: Zero (floor premium = cap premium)
│       ├─ Payoff:
│       │   ├─ If mortality 85%: Receive 5% of assumption cost (floor value)
│       │   ├─ If mortality 100%: No payment (within collar)
│       │   ├─ If mortality 115%: Pay 5% of assumption cost (capped at 110%)
│       │   └─ Protection: ±10% of assumption
│       ├─ Use: Cost-neutral, but limits benefit to tail protection
│       └─ Strategic: Protect against extreme mortality events (pandemic, terrorism)
├─ Equity Hedging:
│   ├─ For variable annuity portfolios (equity exposure):
│   ├─ Put option: Right to sell equity at strike price
│   │   ├─ Cost: 1-3% of equity value per year
│   │   ├─ Example: $300M equity, buy 6-month puts at 90% of current value
│   │   ├─ Premium: $4.5M (1.5% × $300M)
│   │   ├─ If equity falls 20%: Put value = 10% × $300M = $30M (good hedge!)
│   │   ├─ If equity rises 10%: Let puts expire; lose premium $4.5M
│   │   └─ Net: Limits downside; pays for insurance
│   ├─ Collar (cost-neutral):
│   │   ├─ Buy put at 90% (downside protection)
│   │   ├─ Sell call at 110% (cap upside)
│   │   ├─ Cost: Zero (balanced premiums)
│   │   ├─ Payoff: Protected if equity falls below 90%; capped if rises above 110%
│   │   └─ Use: When cost is concern; accept limited upside
│   └─ Dynamic replication:
│       ├─ Track equity option payoff via synthetic hedging
│       ├─ Buy/sell equity proportional to delta (sensitivity to price)
│       ├─ Cost: Lower than buying actual put options
│       ├─ Complexity: Requires daily rebalancing
│       └─ Use: Large portfolios with operational capability
├─ Implementation & Operational Issues:
│   ├─ System requirements:
│   │   ├─ Pricing: Ability to value derivatives daily (Black-Scholes, Hull-White, etc.)
│   │   ├─ Risk measurement: Greeks (delta, gamma, vega, rho, theta)
│   │   ├─ Accounting: Hedge accounting rules (ASC 815, IFRS 9)
│   │   ├─ Collateral: Margin posting, settlement infrastructure
│   │   └─ Reporting: Fair value, hedge effectiveness testing
│   ├─ Hedging effectiveness:
│   │   ├─ Requirement: Documented method; >80% effectiveness in regulation
│   │   ├─ Testing: Regression analysis (derivative gain vs liability loss)
│   │   ├─ Rebalancing: Quarterly or when hedge effectiveness drops below threshold
│   │   └─ Adjustment: Add/remove hedges to maintain >80%
│   ├─ Accounting treatment:
│   │   ├─ Hedge accounting:
│   │   │   ├─ Derivative gain/loss deferred to OCI (Other Comprehensive Income)
│   │   │   ├─ Offsets liability change; net impact minimal
│   │   │   ├─ Earnings volatility reduced
│   │   │   └─ Requirement: >80% effectiveness; documented strategy
│   │   ├─ Mark-to-market (if not qualified):
│   │   │   ├─ Derivative gain/loss flows to current earnings
│   │   │   ├─ Can swing ±$10-50M per quarter (high volatility)
│   │   │   ├─ Mismatches: Derivative gain but liability stable (earnings spike)
│   │   │   └─ Accounting "noise": Obscures true performance
│   │   └─ Decision: Qualifies for hedge accounting? Major P&L impact
│   ├─ Basis risk:
│   │   ├─ Definition: Hedge doesn't perfectly track liability
│   │   ├─ Example: Interest rate swap on LIBOR; liabilities tied to broader index
│   │   ├─ Mismatch: Swap gains don't exactly offset liability changes
│   │   ├─ Impact: Residual risk 10-30% of unhedged risk
│   │   └─ Mitigation: Use longer duration swaps; accept partial hedge
│   ├─ Liquidity risk:
│   │   ├─ Mark-to-market hedge: Must post margin if hedge value falls
│   │   ├─ Example: Interest rate swap on $100M, rate rises
│   │   │   ├─ Swap value: Falls $3M (fixed payer loses value)
│   │   │   ├─ Margin call: $3M cash required
│   │   │   ├─ If market illiquid: May force early termination (bad price)
│   │   │   └─ Risk: Forced to realize loss; defeats hedge purpose
│   │   ├─ Duration of hedge: Longer hedges = higher liquidity risk
│   │   └─ Mitigation: Maintain cash reserves; diversify counterparties
│   └─ Counterparty risk:
│       ├─ Reinsurer fails: Longevity hedge disappears
│       ├─ Bank fails (e.g., Lehman 2008): Swap value unrealized
│       ├─ CVA (Credit Valuation Adjustment): Price reflects failure probability
│       │   ├─ Example: 10Y swap marked at $5M value
│       │   ├─ But counterparty 2% default probability → $100K CVA cost
│       │   ├─ Hedge value effectively $4.9M
│       │   └─ Monitoring: Quarterly counterparty review
│       ├─ Mitigation:
│       │   ├─ Diversify counterparties (no single large position)
│       │   ├─ Collateral: CSA (Credit Support Annex) requires margining
│       │   ├─ Insurance: CDS on reinsurer (hedge the hedge)
│       │   └─ Stress test: What if counterparty fails?
│       └─ Regulatory: Consider counterparty risk in capital
├─ Dynamic Rebalancing Strategy:
│   ├─ Frequency: Daily, weekly, monthly (depends on volatility)
│   ├─ Triggers:
│   │   ├─ Duration drift: If portfolio duration > target + 0.5Y → rebalance
│   │   ├─ Basis widening: If basis risk > threshold → adjust hedge ratio
│   │   ├─ Market event: Major move (>100 bps) → immediate rebalancing
│   │   └─ Volatility spike: If realized vol > 2× expected → increase hedge
│   ├─ Optimization:
│   │   ├─ Cost minimization: Choose cheapest hedge vehicle
│   │   ├─ Tax efficiency: Time tax losses/gains
│   │   ├─ Accounting: Manage P&L impact; timing of rebalancing
│   │   └─ Trade-off: Frequent rebalancing reduces residual risk but increases cost
│   ├─ Monitoring:
│   │   ├─ Daily: Position reconciliation, P&L tracking
│   │   ├─ Weekly: Effectiveness testing, basis risk assessment
│   │   ├─ Monthly: Strategic review, hedge adequacy
│   │   └─ Quarterly: Formal effectiveness audit, rebalancing decision
│   └─ Documentation:
│       ├─ Hedge strategy: Objectives, target hedge ratio, instruments, frequency
│       ├─ Testing method: Regression, dollar-offset, critical measure
│       ├─ Effectiveness: Historical >80%? Current below 80%?
│       ├─ Rebalancing: When triggered? Why? What adjusted?
│       └─ Audit trail: Full record for regulatory review
└─ Cost-Benefit Analysis:
    ├─ Hedging costs:
    │   ├─ Swap: 0% upfront (but mark-to-market volatility)
    │   ├─ Swaption: 1-3% premium
    │   ├─ CAT bond: 8-15% yield (very expensive)
    │   ├─ Reinsurance: 5-15% of premium
    │   └─ Dynamic replication: 0.5-2% annualized
    ├─ Benefits:
    │   ├─ Capital reduction: 20-50% lower requirement if fully hedged
    │   ├─ Risk reduction: Surplus volatility ↓ 50-80% if effective
    │   ├─ Earnings stability: Reduced P&L volatility (if hedge accounting)
    │   ├─ ROE improvement: Same profit on lower capital = higher return
    │   └─ Strategic flexibility: Can take more risk elsewhere
    ├─ Break-even analysis:
    │   ├─ Hedge cost: 5% annually on $100M = $5M/year
    │   ├─ Capital freed: 30% of $50M requirement = $15M
    │   ├─ Cost of capital: 10% (discount rate) on $15M = $1.5M savings
    │   ├─ Net benefit: $1.5M savings - $5M cost = -$3.5M (don't hedge!)
    │   ├─ But if capital constrained: ROE boost makes it worthwhile
    │   └─ Decision: Model-dependent; context matters
    ├─ Scenario testing:
    │   ├─ Base case: Rates stable; hedge costs $5M; no benefit realized; decision: don't hedge
    │   ├─ Bull case: Rates fall 2%; unhedged loss $50M; hedged loss $2M; benefit $48M (do hedge)
    │   ├─ Bear case: Rates rise 2%; unhedged gain $30M; hedged gain $3M; cost $27M (regret hedge)
    │   └─ Decision: Expected value calculation; probability-weighted
    └─ Strategic balance:
        ├─ Core business: Hedge interest rate risk on annuities (high risk if unhedged)
        ├─ Longevity: Selective hedging on tail risks (don't hedge all longevity)
        ├─ Growth business: Minimal hedging (maintain risk; assume profitability)
        └─ Result: Balanced portfolio; manageable capital; acceptable volatility
```

**Key Insight:** Derivatives powerful tools but expensive; perfect hedging is economically irrational; selective hedging of tail risks optimal

## 5. Mini-Project
[Would include swap pricing calculations, hedge ratio optimization, scenario-based cost-benefit analysis, and visualization of hedged vs unhedged positions]

## 6. Challenge Round
When hedging fails:
- **Basis blowout**: Swap should hedge interest rate risk; instead basis widens 50 bps; residual loss $10M; hedge breaks down in crisis
- **Counterparty spiral**: Bank hedges with AIG; AIG's credit spreads widen; bank marks derivative value down; but needs bank to pay if rates move; two-way failure mode
- **Hedge ineffectiveness**: Thought hedge was 90% effective; actually 60%; quarterly testing shows; must deactivate hedge accounting; all deferred gains flow to earnings; $50M earnings shock
- **Gamma blow-up**: Swaption loses gamma as expiration approaches; position initially profitable; suddenly becomes liability; forced to unwind at bad prices
- **Wrong instrument**: Bought interest rate swaption; actual risk was duration mismatch; rates rise but swaption expires worthless; no protection realized
- **Collateral drain**: Swap initially cost zero; but mark-to-market gains $80M in insurer's favor; but counterparty posts margin of $80M, draining liquidity; crisis: can't get cash back (counterparty default)

## 7. Key References
- [Interest Rate Markets: A Practical Introduction (Fabozzi, Bhasin, 2019)](https://www.cfainstitute.org/) - Derivative mechanics and pricing
- [Longevity Swaps: An Overview (Life Insurance Settlement Solutions, 2019)](https://www.soa.org/) - Mortality risk transfer
- [Hull-White Interest Rate Model (Hull, White, 1990)](https://scholar.google.com/) - Stochastic rate modeling for hedging

---
**Status:** Risk transfer | **Complements:** Interest Rate Risk, Liability Matching, Capital Requirements
