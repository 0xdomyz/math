# Comparable Company Analysis (Trading Multiples)

## 1. Concept Skeleton
**Definition:** Valuation based on trading multiples of comparable companies; P/E, EV/EBITDA, P/B ratios; relative valuation approach  
**Purpose:** Quick market-based valuation; identify outliers; benchmark company vs peers; supplement DCF  
**Prerequisites:** Public company multiples, industry definitions, financial metrics (earnings, EBITDA, revenue, book value), adjustments for differences

## 2. Comparative Framing
| Metric | P/E Ratio | EV/EBITDA | P/B Ratio | EV/Revenue | P/FCF Ratio |
|--------|-----------|-----------|----------|-----------|-----------|
| **Numerator** | Market Cap | Enterprise Value | Market Cap | Enterprise Value | Market Cap |
| **Denominator** | Net Income | EBITDA | Book Value | Revenue | Free Cash Flow |
| **Best For** | Profitable firms | Asset-heavy; leverage | Book-heavy (banks) | Revenue model (growth) | Cash flow proxy |
| **Range** | 10-25× typical | 8-15× typical | 1-4× typical | 2-8× typical | 15-25× typical |
| **Advantages** | Widely available | Removes leverage | Asset value | Includes unprofitable | Cash reality check |
| **Disadvantages** | Sensitive to capital structure | Excludes D&A tax effects | Less forward-looking | Ignores profitability | Cyclicality variation |
| **Earnings quality risk** | High (accounting manipulation) | Medium (EBITDA adjustable) | Medium | Low | Low (actual cash) |

## 3. Examples + Counterexamples

**Simple Example:**  
Tech peer group: P/E multiples 18×, 20×, 22×, 19× (average 19.75×); target company earnings $1.00/share; comparable value = $1.00 × 19.75 = $19.75/share; market price $20.00 → fairly valued

**Failure Case:**  
High-growth startup; peer average P/E 15×; target company P/E 40× (not an outlier if hypergrowth); using peer multiple grossly undervalues (should apply higher multiple for growth); misapplication of comparable methodology

**Edge Case:**  
Acquisition target; peer P/E 15×; acquirer pays 30% premium; effective multiple 19.5×; comparable methodology shows 15× but market prices control premium (M&A dynamics override trading multiples)

## 4. Layer Breakdown
```
Comparable Company Analysis Structure:
├─ Selection of Comparables:
│   ├─ Peer Group Definition:
│   │   ├─ Industry/Sector alignment: Same industry; similar business model
│   │   ├─ Example: Target = Apple (consumer tech hardware)
│   │   │   ├─ Peers: Samsung, Microsoft, Google, Meta (tech companies)
│   │   │   ├─ NOT: Ford (auto, different product), Exxon (energy, different industry)
│   │   │   └─ Rationale: Similar competitive dynamics, customer base, margins
│   │   ├─ Market Cap range: Similar size; large cap vs mid-cap have different multiples
│   │   │   ├─ Target: $500B market cap
│   │   │   ├─ Peers: $400B-$700B (similar size)
│   │   │   ├─ Avoid: $50B company (different growth profile) or $2T company (mature giant)
│   │   │   └─ Impact: Market cap affects valuation multiple significantly
│   │   ├─ Geography: Consider regional differences; US vs China different valuations
│   │   │   ├─ US tech: Higher valuation multiples (higher WACC component)
│   │   │   ├─ China tech: Government risk premium; lower multiples
│   │   │   └─ Emerging markets: Different risk adjustments
│   │   ├─ Growth profile: High-growth peers vs mature peers
│   │   │   ├─ Target: Growth 15% annually
│   │   │   ├─ Peers: Growth 10-20% (aligned)
│   │   │   ├─ Don't compare to: Peers growing 2% (value companies; lower multiples)
│   │   │   └─ Impact: Growth peers command 20-30% higher multiples
│   │   ├─ Business model: Single-product vs diversified; recurring revenue vs project-based
│   │   │   ├─ Software (high-margin, recurring): Higher multiples (20-30× EBITDA)
│   │   │   ├─ Manufacturing (low-margin, cyclical): Lower multiples (5-8× EBITDA)
│   │   │   └─ Implication: Can't compare SaaS to hardware on same multiple basis
│   │   ├─ Profitability: Profitable vs unprofitable; quality of earnings
│   │   │   ├─ Target: 20% net margin
│   │   │   ├─ Peers: 15-25% net margin (aligned quality)
│   │   │   ├─ Unprofitable startup: Can't use P/E (denominator zero)
│   │   │   └─ Impact: Use EV/Revenue for unprofitable; but lower multiple than profitable
│   │   ├─ Leverage: Debt levels impact multiples; deleveraged vs leveraged peers
│   │   │   ├─ Target: D/E = 0.5 (moderate leverage)
│   │   │   ├─ Peers: D/E = 0.3-0.7 (similar)
│   │   │   ├─ Overleveraged peer: Discount multiple by 10-20% (risk premium)
│   │   │   └─ Impact: Debt affects both numerator (EV includes debt) and denominator (interest expense)
│   │   ├─ Data sources:
│   │   │   ├─ Bloomberg Terminal; FactSet; S&P Capital IQ
│   │   │   ├─ Yahoo Finance; Google Finance (free; limited)
│   │   │   ├─ Company filings: SEC 10-K (annual); 10-Q (quarterly)
│   │   │   └─ Broker research: Goldman, JPM, Deutsche Bank (institutional)
│   │   └─ Typical peer count:
│   │       ├─ Narrow group: 3-5 peers (focused comparison; less data noise)
│   │       ├─ Broad group: 8-15 peers (more robust; various sub-segments)
│   │       └─ Trade-off: Narrow easier to control; broad more representative
│   ├─ Historical vs. Trailing Multiples:
│   │   ├─ Trailing (TTM - Last Twelve Months):
│   │   │   ├─ Uses most recent 12 months actual data
│   │   │   ├─ Advantage: Most recent; reflects current performance
│   │   │   ├─ Disadvantage: Noise from cyclicality; temporary issues
│   │   │   └─ Example: P/E (TTM) = Market Cap / LTM Earnings
│   │   ├─ Historical (Prior Year):
│   │   │   ├─ Prior year actual reported earnings
│   │   │   ├─ Advantage: Full-year audited; clean
│   │   │   ├─ Disadvantage: Stale if business changing rapidly
│   │   │   └─ Example: P/E (LY) = Market Cap / Prior Year Earnings
│   │   ├─ Forward (Consensus Estimate):
│   │   │   ├─ Next year analyst consensus earnings estimate
│   │   │   ├─ Advantage: Incorporates growth; future-looking
│   │   │   ├─ Disadvantage: Analyst estimates biased (typically too optimistic)
│   │   │   ├─ Bias amount: Typically 10-15% overestimate for growth stocks
│   │   │   └─ Example: P/E (Forward) = Market Cap / Next Year Consensus Earnings
│   │   └─ Timing impact:
│   │       ├─ Same company different multiples on TTM vs Forward (especially cyclical)
│   │       ├─ Cyclical: Earnings peaked → high TTM P/E; forecast decline → low Forward P/E
│   │       ├─ Recommendation: Use all three; triangulate; understand differences
│   │       └─ Example: Auto company (cyclical)
│   │           ├─ TTM: $80B cap / $8B earnings = 10× P/E (peak earnings)
│   │           ├─ Forward: $80B / $6B estimate = 13.3× P/E (earnings decline expected)
│   │           └─ Interpretation: Multiples differ; methodology matters
│   └─ Multiple Calculation Methodology:
│       ├─ For each peer, calculate relevant multiples:
│       │   ├─ P/E = Market Capitalization / Net Income
│       │   ├─ EV/EBITDA = Enterprise Value / EBITDA
│       │   ├─ P/B = Market Cap / Book Value of Equity
│       │   ├─ EV/Revenue = EV / Total Revenue
│       │   └─ P/FCF = Market Cap / Free Cash Flow
│       ├─ Example calculation (Apple):
│       │   ├─ Market Cap: $2,000B
│       │   ├─ Net Income: $100B
│       │   ├─ P/E = $2,000B / $100B = 20×
│       │   ├─ EBITDA: $130B (operating income + D&A adjustment)
│       │   ├─ Enterprise Value: $2,000B (market cap) - $30B (cash) + $5B (debt) = $1,975B
│       │   ├─ EV/EBITDA = $1,975B / $130B = 15.2×
│       │   ├─ Book Value of Equity: $60B
│       │   ├─ P/B = $2,000B / $60B = 33.3× (high; intangible-heavy company)
│       │   ├─ Revenue: $380B
│       │   ├─ EV/Revenue = $1,975B / $380B = 5.2×
│       │   └─ FCF: $110B
│       └─ Organize by peer:
│           ├─ Peer | P/E | EV/EBITDA | P/B | EV/Revenue
│           ├─ Apple | 20.0× | 15.2× | 33.3× | 5.2×
│           ├─ Microsoft | 28.5× | 22.1× | 12.4× | 10.5×
│           ├─ Google | 22.3× | 18.6× | 5.8× | 7.1×
│           ├─ Meta | 18.9× | 16.2× | 6.2× | 3.5×
│           ├─ Average | 22.4× | 18.0× | 14.4× | 6.6×
│           ├─ Median | 21.2× | 17.4× | 9.3× | 6.2×
│           └─ Interpretation: Use median (more robust to outliers than mean)
├─ Adjustment Factors:
│   ├─ Profit Margin Adjustment:
│   │   ├─ If target lower margin than peers, apply discount
│   │   ├─ Target margin: 15%; Peer avg: 20%
│   │   │   ├─ Discount: (15-20)/20 = -25% margin shortfall
│   │   │   ├─ Multiple adjustment: 22.4× × (1 - 0.25) = 16.8× (reduced multiple)
│   │   │   └─ Rationale: Lower margin = lower profit per revenue = lower valuation
│   │   ├─ Advantage: Adjusts for operational efficiency differences
│   │   ├─ Counterpoint: If low margin is temporary, shouldn't discount forever
│   │   └─ Applied to: P/E primarily; can adjust revenue multiples
│   ├─ Growth Rate Adjustment:
│   │   ├─ PEG Ratio = P/E / Growth Rate (%)
│   │   ├─ Target: P/E 18×, growth 20%
│   │   │   ├─ PEG = 18 / 20 = 0.9 (relatively cheap for growth)
│   │   ├─ Peer avg: P/E 22.4×, growth 12%
│   │   │   ├─ PEG = 22.4 / 12 = 1.87 (expensive for lower growth)
│   │   ├─ Interpretation: Target cheaper on growth-adjusted basis
│   │   ├─ Adjustment: Apply peer P/E but scale for growth difference
│   │   │   ├─ Formula: Adjusted P/E = Peer P/E × (Target Growth / Peer Growth)
│   │   │   ├─ = 22.4× × (20 / 12) = 37.3× (premium for faster growth)
│   │   │   └─ Note: Assumes growth drives multiples; not always true
│   │   └─ Use: Growth-adjusted multiples for fast-growing companies
│   ├─ Leverage Adjustment:
│   │   ├─ Debt impacts WACC; riskier company deserves lower multiple
│   │   ├─ Target: D/E = 0.5; Peer avg: D/E = 0.3
│   │   │   ├─ Target more leveraged; higher financial risk
│   │   │   ├─ Multiple discount: 5-10% (depends on debt level magnitude)
│   │   │   └─ Adjusted P/E: 22.4× × 0.95 = 21.3× (slightly discounted)
│   │   ├─ Alternative: Use both P/E (equity value) and EV/EBITDA (enterprise value)
│   │   │   ├─ P/E captures leverage differences in numerator (market cap)
│   │   │   ├─ EV/EBITDA controls for leverage (uses enterprise value, not equity)
│   │   │   └─ Implication: Different leverage profiles use different multiples
│   │   └─ Application: More important for capital-intensive industries
│   ├─ Size Adjustment:
│   │   ├─ Larger companies often trade at lower multiples (maturity premium)
│   │   ├─ Target: $200B market cap (mid-cap); Peer avg: $50B (small-cap)
│   │   │   ├─ Peers might trade 25× P/E; target deserves 20× (size discount)
│   │   │   └─ Rationale: Larger = mature = lower growth = lower multiple
│   │   ├─ Empirical: Size effect ~1-2% per doubling of market cap
│   │   └─ Application: Less critical if peer group properly sized
│   ├─ Profitability Quality Adjustment:
│   │   ├─ Peer earnings quality varies; some use aggressive accounting
│   │   ├─ If peer has high accruals (aggressive), discount their multiple
│   │   │   ├─ High accruals = earnings not cash; risky
│   │   │   ├─ Discount: 10-15% if accounting quality questioned
│   │   │   └─ Metric: Accruals / Earnings; >0.3 flagged as aggressive
│   │   ├─ If target has clean earnings (conservative), maybe premium
│   │   └─ Impact: Normalized multiples less volatile
│   └─ Geographic/Market Risk Adjustment:
│       ├─ Emerging market company: Apply risk premium
│       ├─ US tech peer: 20× P/E
│       ├─ China tech: Same business model; higher political risk
│       │   ├─ Risk adjustment: -10% to -20% multiple discount
│       │   ├─ Multiple: 20× × 0.85 = 17× (lower multiple)
│       │   └─ Rationale: Government intervention risk, currency risk, opacity
│       ├─ Europe peer: Different regulatory environment; maybe -5%
│       └─ Typical adjustments: US baseline; EM -15% to -25%; Europe -5% to -10%
├─ Valuation from Comparable Multiples:
│   ├─ Simple Application:
│   │   ├─ Use peer median/average P/E
│   │   ├─ Target company: EPS $2.50
│   │   ├─ Peer median P/E: 21×
│   │   ├─ Valuation: $2.50 × 21 = $52.50/share
│   │   ├─ Shares outstanding: 500M
│   │   ├─ Equity value: $52.50 × 500M = $26.25B
│   │   └─ Interpretation: Comparable companies suggest $52.50/share fair value
│   ├─ Multiple Metrics Approach:
│   │   ├─ Use several multiples; get range of values
│   │   ├─ Example target company financials:
│   │   │   ├─ Net Income: $2.5B (EPS $2.50/share × 1B shares)
│   │   │   ├─ EBITDA: $4.0B
│   │   │   ├─ Revenue: $20B
│   │   │   ├─ Book Value: $6B
│   │   │   ├─ FCF: $2.8B
│   │   │   └─ Market Cap (unknown; to be valued)
│   │   ├─ Valuation using different multiples:
│   │   │   ├─ P/E (21×): $2.5B × 21 = $52.5B equity value
│   │   │   ├─ EV/EBITDA (18×): EV = $4.0B × 18 = $72B
│   │   │   │   ├─ Equity = $72B - Net Debt $2B = $70B
│   │   │   │   ├─ Per share: $70B / 1B = $70/share
│   │   │   ├─ EV/Revenue (6.6×): EV = $20B × 6.6 = $132B
│   │   │   │   ├─ Equity = $132B - $2B = $130B
│   │   │   │   ├─ Per share: $130/share (high; revenue multiple volatile)
│   │   │   ├─ P/B (14.4×): Equity = $6B × 14.4 = $86.4B
│   │   │   │   └─ Per share: $86.40/share (asset-heavy valuation)
│   │   │   └─ P/FCF (20×): Equity = $2.8B × 20 = $56B
│   │   │       └─ Per share: $56/share
│   │   ├─ Range of values:
│   │   │   ├─ Low: $52.50/share (P/E)
│   │   │   ├─ Mid: $56-70/share (EV/EBITDA, P/FCF)
│   │   │   ├─ High: $86.40-130/share (P/B, EV/Revenue)
│   │   │   └─ Reasonable range: $52-70/share (exclude revenue multiple as too volatile)
│   │   └─ Interpretation: Multiple methods provide range; not single answer
│   ├─ Weighted Average Approach:
│   │   ├─ Assign weights based on confidence in each metric
│   │   ├─ Example weights:
│   │   │   ├─ P/E: 40% (most reliable for profitable company)
│   │   │   ├─ EV/EBITDA: 40% (clean of leverage effects)
│   │   │   ├─ P/FCF: 20% (cash reality check)
│   │   │   └─ (Exclude P/B, EV/Revenue as less appropriate for this company)
│   │   ├─ Weighted valuation:
│   │   │   ├─ = 0.40 × $52.50 + 0.40 × $70 + 0.20 × $56
│   │   │   ├─ = $21 + $28 + $11.20
│   │   │   ├─ = $60.20/share
│   │   │   └─ Fair value estimate: $60.20/share
│   │   └─ Sensitivity: If weights shifted to EV/EBITDA (higher valuation), fair value rises
│   └─ Enterprise Value Approach:
│       ├─ EV/EBITDA most common for enterprise value multiples
│       ├─ Target company: $4.0B EBITDA; Peer EV/EBITDA = 18×
│       ├─ Enterprise Value: $4.0B × 18 = $72B
│       ├─ Adjustments:
│       │   ├─ Less: Total Debt: $10B
│       │   ├─ Plus: Cash & Equivalents: $2B
│       │   ├─ Less: Preferred Stock: $1B
│       │   ├─ Plus: Minority Interests: $0.5B
│       │   └─ Net: Equity Value = $72B - $10B + $2B - $1B + $0.5B = $63.5B
│       ├─ Per share: $63.5B / 1B shares = $63.50/share
│       └─ Common adjustments:
│           ├─ Operating leases: Add to debt (liability nature)
│           ├─ Contingent liabilities: Reduce equity (potential claims)
│           ├─ Option value: Add (potential valuable asset; not in EBITDA)
│           └─ Off-balance sheet items: Normalize financials first
├─ Validation & Sanity Checks:
│   ├─ Implied Multiples Check:
│   │   ├─ If comparable valuation = $60.20/share, calculate implied multiples
│   │   ├─ Implied P/E: $60.20 / $2.50 EPS = 24.1× (vs peer avg 21×)
│   │   │   ├─ Interpretation: Using comparables valued target 15% higher than peer average
│   │   │   ├─ Reason: Maybe target higher growth; better margins; lower risk
│   │   │   └─ Validity check: Reasonable or suspicious?
│   │   ├─ Implied EV/EBITDA: $72B / $4B = 18× (exactly peer median; expected)
│   │   └─ Implied P/B: $63.5B / $6B = 10.6× (vs peer avg 14.4×; lower; maybe conservative)
│   ├─ Current Market Price Comparison:
│   │   ├─ Comparable valuation: $60.20/share
│   │   ├─ Current market price: $55.00/share
│   │   ├─ Discount: ($55 - $60.20) / $60.20 = -8.6% (undervalued)
│   │   ├─ Investment decision: If undervalued & confidence high, consider buying
│   │   ├─ Margin of safety: Typical 20-30% required; current 8.6% margin thin
│   │   └─ Conclusion: Modest undervaluation; not compelling unless other catalysts
│   ├─ Peer Outlier Analysis:
│   │   ├─ Is any peer significantly different? Investigate why
│   │   ├─ Example: One peer trades 28× P/E (vs avg 21×)
│   │   │   ├─ Reason: Higher growth? Acquisition premium? Earnings restatement?
│   │   │   ├─ Decision: Include or exclude from average?
│   │   │   ├─ Typical: Exclude if outlier >1.5 std dev from mean
│   │   │   └─ Impact: Outlier can skew average by 5-10%
│   │   └─ Mean vs Median: Use median (more robust to outliers)
│   ├─ Consistency with DCF:
│   │   ├─ If DCF valuation $70/share; comparable $60/share; 14% difference
│   │   ├─ Reconcile: Different assumptions or methodologies?
│   │   │   ├─ Maybe DCF assumes higher growth; comparables more conservative
│   │   │   ├─ Or comparables reflect market pessimism; DCF shows intrinsic value
│   │   ├─ Range: $60-70/share; true value likely within this band
│   │   └─ Use both: DCF + comparables for triangulation
│   └─ Historical Valuation Trend:
│       ├─ How did peer multiples change over time?
│       ├─ 5-year average P/E: 19×; current 21× (elevated)
│       │   ├─ Reason: Valuations expanded; multiples compression risk
│       │   ├─ Implication: Current comparable valuation may be high (peak multiple)
│       │   └─ Discount: Maybe apply 5-year avg vs current (more conservative)
│       ├─ Cyclical variation: Multiples expand in bull market; contract in bear
│       │   ├─ Current environment: Bull market; multiples inflated
│       │   ├─ Prudence: Use trough multiples for downside scenario
│       │   └─ Example: Use 18× instead of 21× for bear case
│       └─ Trend: If multiples compressing, recent valuations risky
└─ Limitations & Challenges:
    ├─ Market Sentiment Capture:
    │   ├─ Comparable multiples reflect market opinion; not always rational
    │   ├─ If market overly optimistic, all peer multiples inflated
    │   │   ├─ Using inflated multiples → inflated target valuation
    │   │   └─ Risk: Follow market down when it corrects
    │   ├─ Market pessimism: Low multiples; using them undervalues
    │   └─ Cyclical risk: Tech bubble (2000) multiples looked normal; crashed 70%
    ├─ Accounting Differences:
    │   ├─ GAAP vs IFRS differences in revenue recognition, asset valuation
    │   ├─ If peers use aggressive accounting, multiples inflated
    │   │   ├─ Example: Revenue recognition policies differ; one peer records sooner
    │   │   ├─ Same economic earnings; different multiples
    │   │   └─ Adjust: Normalize earnings for comparable analysis
    │   ├─ One-time items: Exclude from EBITDA for cleanness
    │   └─ Stock-based compensation: Add back as expense if not already included
    ├─ Peer Selection Bias:
    │   ├─ If peer group cherry-picked (easier comparables), introduces bias
    │   ├─ Narrow peer group (3-5) more subject to idiosyncratic differences
    │   │   ├─ Solution: Use broader group (10+) to average out noise
    │   │   └─ Trade-off: Broader group less focused; some peers not that comparable
    │   ├─ Survival bias: Only public companies included; private peers excluded
    │   │   ├─ Public peers may be larger/more successful → higher multiples
    │   │   ├─ Implication: Private company comparables would be lower
    │   │   └─ Adjustment: Apply discount if comparing to mostly public comps
    │   └─ Geographic bias: If only US peers, international risk not captured
    ├─ Forward Estimates Uncertainty:
    │   ├─ Forward multiples use analyst consensus; estimates often biased
    │   ├─ Consensus typically 10-15% too optimistic (especially growth)
    │   │   ├─ If using forward P/E 20×, true multiple may be 22× after revisions
    │   │   └─ Implication: Forward multiples overvalue companies
    │   ├─ Recommendation: Use both trailing and forward; triangulate
    │   └─ Reversion: Analyst estimates revert to reality; creates valuation reset
    ├─ Cyclicality Challenges:
    │   ├─ Cyclical industries: Earnings peak and trough
    │   ├─ Using peak earnings P/E low; using trough P/E high
    │   │   ├─ Example: Auto company TTM P/E 8× (peak cycle); forward P/E 12× (expected decline)
    │   │   ├─ Which multiple to use? Industry average may be 10×
    │   │   └─ Approach: Use normalized earnings (average cycle earnings)
    │   ├─ Formula: Normalized Earnings = Average Earnings over Economic Cycle
    │   │   ├─ Last 5 years avg; exclude exceptionals
    │   │   └─ Normalized P/E = Market Cap / Normalized Earnings
    │   └─ Impact: Proper adjustment critical for cyclical valuation
    └─ One-Off Events Distortion:
        ├─ Acquisition premium in one peer distorts multiple
        ├─ Restructuring charges in another; depressed earnings
        ├─ Spin-off creating trading anomalies
        ├─ Solution: Exclude outlier peers; normalize for one-offs
        └─ Recommendation: Audit each peer's financials for quality
```

**Key Insight:** Comparable multiples quick and intuitive; but captures market psychology more than intrinsic value; use to validate DCF, not replace it

## 5. Mini-Project
[Code would include: peer multiple calculation, outlier detection, adjustment factors, valuation range analysis, sensitivity to multiples]

## 6. Challenge Round
When comparables mislead:
- **Market irrationality peak**: Tech bubble 2000; comps trading 50-100× revenue; appear normal; crash 70% afterward
- **Accounting manipulation**: One peer aggressive revenue recognition; inflates EBITDA by 20%; higher multiple misleads; copy cat gets valuation inflated
- **Outlier confusion**: One acquisition target; trades 40× EBITDA (control premium); using 40× for minority stake overvalues by 30%
- **Cyclical peak mislabeling**: Auto company peak cycle earnings; multiples look cheap; earnings decline 40%; valuation destroyed
- **Forward estimate trap**: Analysts forecast 25% growth; company delivers 5%; forward multiple was 25×; actual trailing becomes 50×; loss crystallizes
- **Leverage hidden in comps**: Private equity owned competitor; artificially low equity multiples (high leverage); using that multiple undervalues unlevered company by 20%

## 7. Key References
- [Bloomberg Equity Research](https://www.bloomberg.com/) - Real-time multiples, peer analysis, valuation models
- [CFA Institute - Multiples Approach](https://www.cfainstitute.org/) - Professional standards, valuation best practices
- [Aswath Damodaran - Valuation Multiples Tutorial](https://pages.stern.nyu.edu/~adamodar/) - Academic treatment, industry multiples

---
**Status:** Relative valuation | **Complements:** Intrinsic Value & DCF, Financial Statement Analysis, Industry Benchmarking
