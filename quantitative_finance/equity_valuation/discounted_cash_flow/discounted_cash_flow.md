# Discounted Cash Flow (DCF) Valuation

## 1. Concept Skeleton
**Definition:** Enterprise value equals present value of all future free cash flows; fundamental valuation approach; sum of explicit forecast period + terminal value  
**Purpose:** Determine intrinsic value independent of market price; test if security fairly valued; support investment decisions  
**Prerequisites:** Financial statement analysis, free cash flow calculation, discount rate (WACC), terminal value estimation, financial projections

## 2. Comparative Framing
| Method | DCF | Comparable Multiples | Asset-Based | Dividend Discount Model |
|--------|-----|---------------------|-------------|------------------------|
| **Basis** | Cash flows | Market comps | Net assets | Dividends paid |
| **Time Horizon** | Multi-period explicit | Snapshot | Book value | Perpetuity |
| **Accuracy** | High (if assumptions correct) | Quick; subject to market sentiment | Lower (illiquid assets) | Limited (non-dividend payers) |
| **Complexity** | High (requires projections) | Moderate | Low | Moderate |
| **Best Use** | Mature companies; clear FCF | Quick screening; benchmarking | Asset-heavy; real estate | Dividend-growth investors |
| **Sensitivity** | High (WACC, terminal growth critical) | Moderate | Low | High (growth rate) |

## 3. Examples + Counterexamples

**Simple Example:**  
Mature software company; FCF year 1-5: $100M, $110M, $120M, $130M, $140M; terminal growth 3%; WACC 8%; terminal value ~$1.4B; present value of FCF ~$450M; enterprise value ~$1.85B

**Failure Case:**  
Overoptimistic projections (25% annual growth forever); WACC too low (6% instead of 10%); terminal growth too high (5% instead of 3%); valuation inflated 50%; investment loses money

**Edge Case:**  
Mature, declining company; FCF negative in years 2-5; cannot sustain; DCF suggests bankruptcy value; market still trades above liquidation; market wrong or turnaround expected

## 4. Layer Breakdown
```
DCF Valuation Structure:
├─ Free Cash Flow Calculation:
│   ├─ Starting Point (EBIT-Based):
│   │   ├─ EBIT (Earnings Before Interest & Tax)
│   │   │   ├─ = Revenue - Operating Expenses
│   │   │   ├─ Excludes: Interest, taxes, one-time items
│   │   │   ├─ Example: $1,000M revenue, $600M COGS, $200M OpEx = $200M EBIT
│   │   │   └─ Often proxy: Operating Income or EBITDA (add back D&A)
│   │   ├─ Less: Taxes
│   │   │   ├─ NOPAT = EBIT × (1 - Tax Rate)
│   │   │   ├─ Example: $200M × (1 - 0.25) = $150M NOPAT
│   │   │   ├─ Note: Use marginal tax rate; not effective rate
│   │   │   └─ Adjustment: Add back value of tax shields (if debt financed)
│   │   ├─ Plus: Depreciation & Amortization (non-cash)
│   │   │   ├─ Add back to NOPAT
│   │   │   ├─ Example: +$40M D&A
│   │   │   └─ Rationale: Deducted for tax; doesn't affect cash
│   │   ├─ Less: Capital Expenditure (cash out)
│   │   │   ├─ Investment in PP&E; plant, equipment, infrastructure
│   │   │   ├─ Example: -$60M Capex
│   │   │   ├─ Sustaining capex: Maintains current production
│   │   │   ├─ Growth capex: Enables expansion
│   │   │   └─ Estimate: Often % of revenue or growth capex separately
│   │   ├─ Less: Increase in Net Working Capital
│   │   │   ├─ NWC = Current Assets - Current Liabilities
│   │   │   ├─ Increase in NWC ties up cash (inventory, receivables)
│   │   │   ├─ Example: Inventory up $10M, AR up $5M, AP up $3M → NWC increase $12M
│   │   │   ├─ Cash impact: -$12M (less free cash)
│   │   │   └─ Decrease: Cash release (not common; usually growth phase ties up capital)
│   │   └─ Formula: FCF = NOPAT + D&A - Capex - Δ(NWC)
│   │       ├─ Example: $150M + $40M - $60M - $12M = $118M FCF
│   │       ├─ Alternative: Start with Net Income (requires adjustments for financing)
│   │       └─ Key: Unlevered FCF (assumes no debt); used for Enterprise Value
│   ├─ Projecting Future FCF (Explicit Forecast Period):
│   │   ├─ Historical analysis:
│   │   │   ├─ Review prior 3-5 years FCF
│   │   │   ├─ Calculate growth rates; identify trends
│   │   │   ├─ Example: FCF grew 5%, 7%, 6%, 8% annually; average ~6.5%
│   │   │   └─ Baseline: Use historical as anchor
│   │   ├─ Revenue projection:
│   │   │   ├─ Market size & growth; company market share
│   │   │   ├─ Unit volume × price per unit
│   │   │   ├─ Example: Industry grows 3% annually; company gains share → 5% revenue growth
│   │   │   ├─ Year 1: $1,050M; Year 2: $1,103M; Year 3: $1,158M (5% CAGR)
│   │   │   ├─ Reality check: Compare to consensus; peer growth; macro outlook
│   │   │   └─ Variation: Can decline (competitive pressure) or accelerate (new products)
│   │   ├─ Margin projection:
│   │   │   ├─ Gross margin: COGS / Revenue; subject to input cost inflation, pricing power
│   │   │   ├─ Operating margin: OpEx scale; leverage (fixed vs variable)
│   │   │   ├─ Example: Gross margin 40% (stable); OpEx $200M (fixed) + 5% of revenue (variable)
│   │   │   │   ├─ Year 1 OpEx: $200M + 5% × $1,050M = $252.5M
│   │   │   │   ├─ Year 2 OpEx: $200M + 5% × $1,103M = $255.15M
│   │   │   │   ├─ EBIT margin improves: OpEx deleverages as revenue grows
│   │   │   │   └─ Impact: 1% margin expansion per $100M revenue growth
│   │   │   ├─ Margin maturity: Eventually stabilize (diminishing returns)
│   │   │   └─ Trend: If improving, project continued but at slowing rate
│   │   ├─ Capex projection:
│   │   │   ├─ Sustaining capex: Maintenance level; usually 70-80% of D&A
│   │   │   ├─ Growth capex: To support revenue growth; 2-3% of incremental revenue
│   │   │   ├─ Example: Sustaining $40M (D&A level); Growth capex 2.5% × $50M revenue growth = $1.25M
│   │   │   ├─ Total capex: $41.25M
│   │   │   └─ Variation: Cyclical industries (construction) have variable capex
│   │   ├─ NWC projection:
│   │   │   ├─ NWC as % of revenue (common proxy)
│   │   │   ├─ Example: Typical 10% of revenue; includes inventory, AR, AP
│   │   │   ├─ Year 1: 10% × $1,050M = $105M
│   │   │   ├─ Year 2: 10% × $1,103M = $110.3M
│   │   │   ├─ Δ NWC: $110.3M - $105M = $5.3M (cash tie-up)
│   │   │   ├─ Efficiency: Better working capital management → lower % (8% vs 10%)
│   │   │   └─ Scaling: Early-stage needs more; mature less (30% vs 5%)
│   │   ├─ Forecast period length:
│   │   │   ├─ Typical: 5-10 years (explicit projection)
│   │   │   ├─ Longer period: Better captures full cycle; but uncertainty increases
│   │   │   ├─ Shorter period: Less detail; more reliance on terminal value
│   │   │   ├─ Example: Tech startup (uncertain) → 5 years; utility (stable) → 10 years
│   │   │   └─ Beyond: Terminal value (constant growth perpetuity)
│   │   └─ Example Multi-Year Projection:
│   │       ├─ Year 1: Revenue $1,050M, NOPAT $158M, D&A $42M, Capex $62M, Δ NWC $5M → FCF $133M
│   │       ├─ Year 2: Revenue $1,103M, NOPAT $168M, D&A $44M, Capex $63M, Δ NWC $5M → FCF $144M
│   │       ├─ Year 3: Revenue $1,158M, NOPAT $178M, D&A $46M, Capex $64M, Δ NWC $5M → FCF $155M
│   │       ├─ ... (Years 4-5 similar growth pattern)
│   │       └─ Pattern: Revenue grows 5% annually; NOPAT grows faster (margin expansion); FCF robust
│   └─ Terminal Value Calculation:
│       ├─ Perpetuity Growth Method (most common):
│       │   ├─ Assumption: Company grows at constant rate forever (g%)
│       │   ├─ Formula: Terminal Value = FCF(final year) × (1+g) / (WACC - g)
│       │   ├─ Example: Year 5 FCF $180M; WACC 8%; growth rate 3%
│       │   │   ├─ Terminal Value = $180M × 1.03 / (0.08 - 0.03) = $184.2M / 0.05 = $3,684M
│       │   │   └─ Interpretation: Perpetuity value of company post-year-5
│       │   ├─ Growth rate selection:
│       │   │   ├─ GDP growth typical proxy: 2-3% for developed economies
│       │   │   ├─ Industry growth: If above GDP, use industry rate (premium justified)
│       │   │   ├─ Below GDP: Mature/declining industries; 1-2%
│       │   │   ├─ Constraint: g < WACC (or infinite value; nonsensical)
│       │   │   ├─ Sensitivity: 3% vs 2.5% growth → 20% valuation difference
│       │   │   └─ Conservative: Use lower growth (better downside protection)
│       │   ├─ WACC selection:
│       │   │   ├─ Cost of Equity (Re): CAPM; β, risk-free rate, market risk premium
│       │   │   ├─ Cost of Debt (Rd): Interest rate on debt; tax-adjusted
│       │   │   ├─ WACC = E/V × Re + D/V × Rd × (1-Tax)
│       │   │   ├─ Example: 70% equity (Re 10%), 30% debt (Rd 5%, tax 25%)
│       │   │   │   ├─ WACC = 0.7 × 0.10 + 0.3 × 0.05 × 0.75 = 0.07 + 0.0113 = 8.13%
│       │   │   └─ Sensitivity: 8% vs 10% WACC → 50% valuation difference (huge!)
│       │   └─ Critique: Assumes perpetuity unrealistic; management may change; technology obsolete
│       ├─ Exit Multiple Method:
│       │   ├─ Assumption: Exit at year N at certain multiple (e.g., 15× FCF)
│       │   ├─ Formula: Terminal Value = FCF(year N) × Exit Multiple
│       │   ├─ Example: Year 5 FCF $180M; Exit at 15× FCF → Terminal Value = $2,700M
│       │   ├─ Multiple selection:
│       │   │   ├─ Industry average P/FCF ratio; historical precedent
│       │   │   ├─ If industry trades 12-18× FCF, use 15× (midpoint)
│       │   │   ├─ Conservative: Use lower multiple (industry trough)
│       │   │   └─ Optimistic: Use higher multiple (peak valuations)
│       │   ├─ Comparison to perpetuity:
│       │   │   ├─ Perpetuity method: More theoretically sound; but sensitive to growth assumption
│       │   │   ├─ Exit multiple: More practical; uses market data; but tied to current market multiples
│       │   │   └─ Triangulation: Use both; compare results; should be reasonable
│       │   └─ Advantage: Avoids perpetuity growth assumption
│       └─ Terminal Value as % of Total Value:
│           ├─ Typical split: 60-80% of total value from terminal value
│           ├─ Example: PV(FCF years 1-5) = $600M; PV(Terminal) = $1,200M; Total = $1,800M
│           │   ├─ Terminal = 67% of value (typical)
│           │   ├─ Implication: Terminal value dominates; small changes → big impact
│           │   └─ Risk: If terminal assumptions wrong, entire valuation wrong
│           ├─ Variation by industry:
│           │   ├─ High-growth tech: Terminal = 50% (explicit period captures most of value)
│           │   ├─ Mature utility: Terminal = 80% (nearly all value post-period)
│           │   └─ Implication: Different industries have different value drivers
│           └─ Sensitivity: Terminal growth from 2% to 4% → +40% valuation impact
├─ Discounting to Present Value:
│   ├─ Present Value Calculation:
│   │   ├─ PV = Sum of [FCF(t) / (1 + WACC)^t] for all periods
│   │   ├─ Example: Year 1 FCF $133M at WACC 8%
│   │   │   ├─ PV = $133M / 1.08 = $123.1M
│   │   ├─ Year 2 FCF $144M
│   │   │   ├─ PV = $144M / (1.08^2) = $144M / 1.1664 = $123.5M
│   │   ├─ Year 3-5: Similar calculations
│   │   └─ Total PV(FCF years 1-5) ≈ $600M (from earlier example)
│   ├─ Discount Rate Application:
│   │   ├─ WACC discount rate (cost of capital)
│   │   ├─ If higher WACC → Lower present value (riskier company)
│   │   ├─ If lower WACC → Higher present value (safer, lower risk)
│   │   ├─ Implication: Small changes in WACC have large impact
│   │   └─ Adjustment for risk: Some use hurdle rate > WACC (adds safety margin)
│   ├─ Terminal Value Discounting:
│   │   ├─ Terminal value occurs at year N (e.g., year 5)
│   │   ├─ Discount back: PV(TV) = Terminal Value / (1 + WACC)^N
│   │   ├─ Example: Terminal Value $3,684M, discount to present at year 5
│   │   │   ├─ PV(TV) = $3,684M / (1.08^5) = $3,684M / 1.4693 = $2,507M
│   │   ├─ Impact: Discounting has big impact (year 5 discount ~32%)
│   │   └─ Sensitivity: Change WACC 1% → ~8% change in terminal value present value
│   └─ Enterprise Value:
│       ├─ Sum: Enterprise Value = PV(FCF explicit) + PV(Terminal Value)
│       ├─ Example: $600M + $2,507M = $3,107M Enterprise Value
│       ├─ Interpretation: Value of operating business (debt + equity)
│       └─ Adjustment: Subtract net debt to get Equity Value
├─ Bridge to Equity Value:
│   ├─ Enterprise Value to Equity Value:
│   │   ├─ Equity Value = Enterprise Value - Net Debt
│   │   ├─ Net Debt = Total Debt - Cash & Equivalents
│   │   ├─ Example: EV $3,107M - Net Debt $400M = Equity Value $2,707M
│   │   └─ Interpretation: Shareholders' residual claim on assets
│   ├─ Per Share Value:
│   │   ├─ Intrinsic Value per Share = Equity Value / Diluted Shares Outstanding
│   │   ├─ Example: $2,707M / 500M shares = $5.41/share
│   │   ├─ Comparison: Current market price (e.g., $6.00) → undervalued or overvalued?
│   │   └─ Investment decision: If $5.41 > market price, consider buying
│   ├─ Fully Diluted Shares:
│   │   ├─ Include: Issued shares + stock options + restricted stock + convertible bonds
│   │   ├─ Impact: More shares → lower per-share value
│   │   ├─ Example: Basic shares 500M; options 50M; total diluted 550M
│   │   │   ├─ Value per share: $2,707M / 550M = $4.92/share (lower than $5.41)
│   │   │   └─ Significant impact if high option grants
│   │   └─ Prudence: Always use fully diluted (conservative)
│   └─ Minority Interest Adjustment:
│       ├─ If subsidiary: Adjust for minority holders' stake
│       ├─ Example: 75% owned subsidiary; parent's share = EV × 0.75
│       └─ Not typically issue for parent company valuations
├─ Sensitivity Analysis:
│   ├─ One-Way Sensitivity:
│   │   ├─ Change one variable; observe impact on value
│   │   ├─ Example: WACC sensitivity
│   │   │   ├─ WACC 7% → EV $3,450M → $6.90/share
│   │   │   ├─ WACC 8% → EV $3,107M → $5.41/share (base case)
│   │   │   ├─ WACC 9% → EV $2,800M → $5.09/share
│   │   │   ├─ WACC 10% → EV $2,520M → $4.58/share
│   │   │   └─ Range: $4.58 to $6.90 (50% swing for 3% WACC range)
│   │   ├─ Growth rate sensitivity:
│   │   │   ├─ Terminal growth 2% → EV $2,750M → $5.00/share
│   │   │   ├─ Terminal growth 3% → EV $3,107M → $5.41/share (base)
│   │   │   ├─ Terminal growth 4% → EV $3,550M → $6.45/share
│   │   │   └─ Implication: Terminal growth critical driver
│   │   └─ Revenue growth sensitivity:
│   │       ├─ 3% annual (vs 5% base) → Lower FCF → Lower EV
│   │       ├─ 7% annual (vs 5% base) → Higher FCF → Higher EV
│   │       └─ Impact depends on margin expansion assumptions
│   ├─ Two-Way Sensitivity Table:
│   │   ├─ Format: Rows = WACC (7-10%), Columns = Terminal Growth (2-4%)
│   │   ├─ Table entries: Equity value per share
│   │   ├─ Example:
│   │   │   ├─        | 2% Growth | 3% Growth | 4% Growth
│   │   │   ├─ 7%    | $6.10 | $6.90 | $8.20
│   │   │   ├─ 8%    | $4.95 | $5.41 | $6.45
│   │   │   ├─ 9%    | $4.20 | $4.60 | $5.30
│   │   │   ├─ 10%   | $3.65 | $3.95 | $4.50
│   │   │   └─ Interpretation: Base case ($5.41) with range $3.65-$8.20
│   │   └─ Investment decision: If current price in lower half of range → buy; upper half → sell
│   ├─ Scenario Analysis:
│   │   ├─ Bull case: Optimistic growth (7%), margin expansion, lower WACC (7%)
│   │   │   ├─ Valuation: $8.20/share
│   │   ├─ Base case: 5% growth, stable margins, 8% WACC
│   │   │   ├─ Valuation: $5.41/share
│   │   ├─ Bear case: Declining growth (2%), margin compression, higher WACC (10%)
│   │   │   ├─ Valuation: $3.65/share
│   │   ├─ Probability-weighted value:
│   │   │   ├─ P(Bull) 25% × $8.20 + P(Base) 50% × $5.41 + P(Bear) 25% × $3.65
│   │   │   ├─ = $2.05 + $2.71 + $0.91 = $5.67/share
│   │   │   └─ Expected value: $5.67 vs current market price guides investment
│   │   └─ Risk assessment: Wide range indicates high uncertainty
│   └─ Sensitivity Limits:
│       ├─ Reasonableness check: If doubling WACC halves value, intuitive but risky assumption
│       ├─ Terminal growth bounds: Must be < WACC (or g → ∞)
│       ├─ Revenue growth: Must be compatible with market size (can't grow forever > industry)
│       └─ Confidence: Valuations within ±20% of base case reasonable; >50% indicates high sensitivity
├─ Valuation Quality Assessment:
│   ├─ Key Drivers:
│   │   ├─ Terminal value: Dominates; small assumption errors → large value errors
│   │   ├─ WACC: Small changes have large impact; 1% change → 8-12% value change
│   │   ├─ Revenue growth: Early period critical; drives later period FCF
│   │   └─ Margin assumptions: Sustainability; realistic based on competitive position
│   ├─ Sanity Checks:
│   │   ├─ Implied multiples: Check if resulting P/E, EV/EBITDA reasonable vs peers
│   │   ├─ FCF yield: FCF / Enterprise Value; should be positive; typically 5-15%
│   │   ├─ Growth multiples: High growth firms trade at high multiples; realistic?
│   │   ├─ ROI: Implied return on investments; >WACC (or destroy value)
│   │   └─ Market sentiment: Compare valuation to analyst consensus; if 50% different, investigate
│   ├─ Sensitivity to Assumptions:
│   │   ├─ List key assumptions: Terminal growth 3%, WACC 8%, margin 18%
│   │   ├─ Priority: Which drive most value?
│   │   ├─ Confidence: How confident in each assumption?
│   │   └─ Recommendation: If high sensitivity to low-confidence assumption, valuation risky
│   └─ Limitation Recognition:
│       ├─ 5-10 year projection unrealistic; assumptions will prove wrong
│       ├─ Terminal value perpetuity simplistic; companies don't last forever
│       ├─ WACC constant throughout unrealistic; changes with leverage, risk profile
│       ├─ Competition & disruption: Tech obsolescence not captured
│       └─ Market can stay irrational longer than valuation investor stays solvent
└─ Comparative Valuation:
    ├─ DCF vs Market Price:
    │   ├─ If DCF > Market: Undervalued; potential buying opportunity
    │   ├─ If DCF < Market: Overvalued; potential selling opportunity
    │   ├─ Margin of safety: Typical 20-30% discount to DCF value before committing
    │   └─ Example: DCF $5.41, market $6.00; only 11% overvalued; not enough margin
    ├─ DCF vs Comparable Multiples:
    │   ├─ Calculate implied P/E from DCF
    │   ├─ Example: DCF EPS $1.08, market $6.00 → Implied P/E = 5.56×
    │   ├─ Compare to peers: If peers trade 15-20× P/E, company cheap
    │   └─ Reconciliation: If different valuations, investigate why
    ├─ Peer Benchmarking:
    │   ├─ Use DCF as intrinsic value
    │   ├─ Compare to peer average trading multiples
    │   ├─ Identify outliers; investigate valuation differences
    │   └─ Decision: If DCF reasonable vs peers, use for investment decision
    └─ Multiple Valuation Methods:
        ├─ Use DCF, comparable multiples, precedent transactions together
        ├─ Triangulation: If all three methods suggest similar value, high confidence
        ├─ Divergence: If methods suggest different values, investigate assumptions
        ├─ Weighting: DCF 50%, multiples 30%, precedent 20% (typical allocation)
        └─ Final valuation: Weighted average; balances theoretical with practical
```

**Key Insight:** DCF theoretically pure; but garbage in→garbage out; small assumption changes drive huge value swings; sensitivity analysis essential

## 5. Mini-Project
[Code would include: multi-year FCF projection, WACC calculation, terminal value methods, sensitivity tables, scenario analysis]

## 6. Challenge Round
When DCF fails:
- **Terminal value trap**: Terminal value = 80% of total; small growth assumption change (3%→3.5%) → 20% value change
- **WACC manipulation**: Artificially low WACC to inflate valuation; overlooks leverage risk; hidden risk
- **Projection overconfidence**: Forecast $500M revenue year 5; actual $250M; loss materializes; valuation was wrong
- **Margin assumption break**: Assume 18% EBIT margin sustained; competition commoditizes; margins fall to 12%; equity value halves
- **Perpetuity nonsense**: Terminal growth assumed 4%; but GDP growth only 2%; company growing faster than entire economy forever (impossible)
- **Debt ignored**: DCF assumes stable leverage; company takes on debt post-valuation; WACC rises; equity value destroyed

## 7. Key References
- [Damodaran - DCF Valuation Models](https://pages.stern.nyu.edu/~adamodar/pdfiles/valn2ed/ch2.pdf) - DCF mechanics, tutorials
- [CFA Institute - Valuation Methods](https://www.cfainstitute.org/) - Professional standards, best practices
- [Graham & Dodd - Security Analysis](https://www.investopedia.com/terms/g/graham-and-doddsville.asp) - Fundamental valuation principles

---
**Status:** Absolute valuation | **Complements:** WACC, Terminal Value, Comparable Multiples, Sensitivity Analysis
