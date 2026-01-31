# Relative Valuation & Multiples

## 1. Concept Skeleton
**Definition:** Value equity by comparing price multiples (P/E, EV/EBITDA, P/B) to peer companies rather than discounting cash flows; assumes market prices similar companies similarly  
**Purpose:** Quick, market-based valuation; identify mispriced stocks vs peers; benchmark valuation reasonableness; practical for practitioners  
**Prerequisites:** Financial statement analysis, comparable company selection, multiple calculation, market microstructure understanding

## 2. Comparative Framing
| Multiple | Calculation | Use Case | Advantages | Disadvantages |
|----------|-------------|----------|-----------|----------------|
| **P/E Ratio** | Share Price / EPS | Profitability comparison | Simple, widely available | Sensitive to accounting (depreciation, stock-based comp) |
| **EV/EBITDA** | Enterprise Value / EBITDA | Leverage-neutral comparison | Removes debt impact, common in M&A | Ignores capex, working capital |
| **P/B Ratio** | Market Cap / Book Value | Asset-heavy companies | Useful for banks, insurers | Book value stale, understated intangibles |
| **P/S Ratio** | Market Cap / Revenue | Low/negative earning companies | Difficult to manipulate | Ignores margins; high for unprofitable firms |
| **PEG Ratio** | P/E / Growth Rate | Growth-adjusted valuation | Accounts for growth; identifies undervalued growers | Growth estimates subjective; future-dependent |
| **FCF Yield** | Operating FCF / Market Cap | Cash generation focus | Most theoretically sound; links to intrinsic value | Requires detailed cash flow projections |
| **EV/Sales** | Enterprise Value / Revenue | Stable revenue focus | Comparable to P/S; less sensitive to margins | Ignores profitability; high correlation with P/S |
| **Dividend Yield** | Annual Dividend / Share Price | Income-focused valuation | For mature dividend payers | Assumes dividend continuity; ignores growth |

## 3. Examples + Counterexamples

**P/E Valuation Win (Tech Sector Peer Comparison, 2024):**  
Comparable tech companies (Microsoft, Google, Meta): Average P/E = 30×.  
Target company XYZ: EPS = $5, Current price = $120 (P/E = 24×).  
Valuation: XYZ at industry P/E → $5 × 30 = $150 per share.  
Conclusion: XYZ undervalued by 25% (trading at $120 vs $150 fair value).  
Outcome: 6 months later, XYZ reaches $145 (fundamental improvement + multiple expansion).

**EV/EBITDA Trap (High Leverage Firm, 2023):**  
Comparable industrial companies: Average EV/EBITDA = 8×.  
Target company DEF: EBITDA = $100M, Enterprise Value = $750M (EV/EBITDA = 7.5×).  
Valuation: DEF appears undervalued at 7.5× vs 8× peer average.  
Reality: DEF has $500M debt (high leverage vs peers at 2x); interest coverage poor.  
Outcome: Credit downgrade → spreads widen → enterprise value compresses to $600M despite stable EBITDA.  
Lesson: Multiple valuation must adjust for leverage differences (use levered vs unlevered betas).

**P/B Mispricing (Bank Sector Crisis, 2008):**  
US banks pre-crisis: Average P/B = 1.5× (historical norm).  
Bank ABC: Trading at P/B = 1.3× (seemingly undervalued).  
Valuation: ABC's book value = $50/share → Fair value = $50 × 1.5 = $75/share.  
Reality: Book value overstated (hidden loan losses not yet recognized); ROE collapsing.  
Actual outcome: Book value fell to $20/share (write-downs); stock fell to $8/share (P/B = 0.4×).  
Lesson: P/B fails when ROE deteriorates or accounting quality poor.

**PEG Ratio Success (High-Growth Company, 2015-2020):**  
Company GHI: P/E = 50×, expected growth = 40%/year → PEG = 50/40 = 1.25.  
Peer comparables: Average PEG = 1.8 (GHI appears undervalued on growth basis).  
Valuation implication: GHI trading below "fair" PEG, suggesting growth optionality underpriced.  
Outcome: Next 5 years GHI maintained 35%+ growth; stock returned +400% (justified high P/E).

**FCF Yield Signaling (Mature Dividend Payer, 2022):**  
Dividend aristocrat MNO: Historical FCF yield = 4%, current = 6% (market cap fell 20%).  
Interpretation: Either FCF fell or stock became undervalued (relative to cash generation).  
Valuation: If FCF stable, 6% yield attractive vs 3% risk-free rate → Upside potential.  
Outcome: 18 months later, stock recovered +25% (dividend cut avoided, cash flow held).

**Revenue Multiple Compression (SaaS Downturn, 2022):**  
SaaS companies 2021: Average P/S = 8× (high growth, high multiples).  
Company JKL: P/S = 7× (appeared undervalued vs peers).  
Late 2022: SaaS sector rotation; multiples compress to 3-4× due to higher rates.  
Outcome: Even if revenue stable, stock fell -60% (multiple compression alone, no operational change).  
Lesson: Multiples highly sensitive to discount rates and growth expectations (cyclical).

## 4. Layer Breakdown
```
Relative Valuation Framework:

├─ Core Concept: Comparable Company Analysis
│  ├─ Selection Criteria:
│  │  ├─ Industry peers: Similar business model, revenue sources
│  │  │  ├─ Example: Compare Spotify to Apple Music, not to Netflix
│  │  │  ├─ Rationale: Different value drivers (music streaming vs content library)
│  │  │  └─ Data: Peer databases (CapIQ, FactSet, Bloomberg)
│  │  │
│  │  ├─ Size/Scale matching:
│  │  │  ├─ Revenue: ±30-50% range around target
│  │  │  ├─ Market cap: ±50% range (large caps can have different multiples)
│  │  │  ├─ Example: Compare Pfizer (pharma giant) to Moderna (smaller biotech separately)
│  │  │  └─ Rationale: Scale affects profitability (margins, capex intensity)
│  │  │
│  │  ├─ Geographic/Product focus:
│  │  │  ├─ Domestic vs international exposure
│  │  │  ├─ Product concentration: Single product vs diversified portfolio
│  │  │  ├─ Example: Compare Samsung (diversified tech) only to conglomerates, not focused competitors
│  │  │  └─ Adjustment: Different market growth rates, FX exposure
│  │  │
│  │  └─ Capital Structure:
│  │     ├─ Leverage: Debt/equity ratios should be comparable
│  │     ├─ Example: High-leverage LBO'd company → compare only to levered comps
│  │     ├─ Adjustment: Use unlevered multiples if leverage differs materially
│  │     └─ Lease obligations: Operating vs financial leases (IFRS 16 now capitalizes)
│  │
│  ├─ Multiple Calculation:
│  │  ├─ Earnings multiples (P/E, EV/EBITDA):
│  │  │  ├─ Trailing (last 12 months, LTM) vs Forward (next 12 months)
│  │  │  ├─ TTM approach: Use actual recent quarters to avoid accounting period effects
│  │  │  ├─ Forward: Based on analyst consensus estimates
│  │  │  ├─ Advantage of forward: Removes transitory earnings impacts
│  │  │  ├─ Risk of forward: Estimates prone to optimism bias
│  │  │  └─ Typical practice: Use trailing for mature firms, forward for growth firms
│  │  │
│  │  ├─ Normalization adjustments:
│  │  │  ├─ One-time items: Remove M&A charges, restructuring, asset sales
│  │  │  ├─ Stock-based compensation: Add back non-cash expense (affects after-tax earnings)
│  │  │  ├─ Unusual margins: Normalize for cyclical peaks/troughs
│  │  │  ├─ Example: P/E excluding one-time charges = "adjusted P/E" (common in investment banking)
│  │  │  └─ Warning: Over-adjustment can obscure operational issues
│  │  │
│  │  ├─ Mean vs Median:
│  │  │  ├─ Mean P/E: Average of peer multiples (vulnerable to outliers)
│  │  │  ├─ Median P/E: 50th percentile (robust to outliers)
│  │  │  ├─ Example: 5 peers at 12×, 14×, 15×, 16×, 25× (outlier)
│  │  │  ├─ Mean = 16.4×, Median = 15×
│  │  │  └─ Practice: Median preferred; remove obvious outliers first (bottom/top 1)
│  │  │
│  │  └─ Time horizon:
│  │     ├─ Current trading multiple
│  │     ├─ Historical range (1-3 year average)
│  │     ├─ Example: Stock trading at P/E 20×, historical avg 16× → premium or inflated expectations?
│  │     └─ Cyclical context: Peak vs trough multiples during cycle
│  │
│  ├─ Valuation Formula:
│  │  ├─ Intrinsic Value = Comparable Multiple × Target Company Metric
│  │  ├─ Example 1 (P/E): Fair Value = Peer P/E × Target EPS
│  │  │  ├─ Peer P/E = 20×, Target EPS = $5 → Fair Value = $100/share
│  │  │  ├─ Current price $90 → 11% upside
│  │  │  └─ Assumption: Target justified in trading at peer multiple
│  │  │
│  │  ├─ Example 2 (EV/EBITDA): Enterprise Value = Peer EV/EBITDA × Target EBITDA
│  │  │  ├─ Peer EV/EBITDA = 10×, Target EBITDA = $50M → EV = $500M
│  │  │  ├─ Less: Net Debt = $100M → Equity Value = $400M
│  │  │  ├─ Divided by: Shares outstanding 20M → Value/share = $20
│  │  │  └─ More precise than P/E (removes capital structure effects)
│  │  │
│  │  └─ Multiple adjustments (target-specific):
│  │     ├─ Quality premium: If target more profitable than peers, apply higher multiple
│  │     ├─ Growth adjustment: If faster-growing, warrant slightly higher multiple
│  │     ├─ Risk discount: If riskier (concentrated customer, new product), apply discount
│  │     ├─ Formula: Adjusted Multiple = Base Multiple × (Target Quality / Peer Quality)
│  │     └─ Example: Peers at 20× P/E, Target ROE 20% vs peer avg 15% → 20× × (20/15) = 26.7×
│  │
│  └─ Sensitivity & Range:
│     ├─ Conservative case: Use lower multiple (bottom quartile of peers)
│     ├─ Base case: Use median/mean multiple
│     ├─ Aggressive case: Use higher multiple (top quartile)
│     ├─ Example: Base $100 → Range $70-$130 across scenarios
│     └─ Communicate range, not single point estimate
│
├─ Individual Multiple Types (Detailed):
│  ├─ P/E Ratio (Price-to-Earnings):
│  │  ├─ Definition: Share price / Earnings per share (usually net income)
│  │  ├─ Calculation: P/E = Market Cap / Net Income
│  │  ├─ Interpretation:
│  │  │  ├─ P/E = 15: Investors pay $15 for $1 of annual earnings
│  │  │  ├─ High P/E (>20): Market expects growth; higher earnings multiple justified only if growth delivers
│  │  │  ├─ Low P/E (<10): Value trap territory; verify why market discounts
│  │  │  └─ Negative P/E (loss-making): Use other multiples (P/S, P/B)
│  │  │
│  │  ├─ Advantages:
│  │  │  ├─ Simple, intuitive, widely reported
│  │  │  ├─ Comparable across companies (unlike absolute profit figures)
│  │  │  ├─ Historical data readily available
│  │  │  └─ Used in most investment screening tools
│  │  │
│  │  ├─ Disadvantages:
│  │  │  ├─ Sensitive to capital structure (leverage increases EPS through financial risk)
│  │  │  ├─ One-time items distort earnings (stock-based comp, restructuring charges, gains/losses)
│  │  │  ├─ Accounting methods vary (depreciation, revenue recognition, reserves)
│  │  │  ├─ Ignores capex and working capital needs (cash flow more relevant)
│  │  │  └─ Different tax rates affect comparability (effective tax rate varies)
│  │  │
│  │  └─ Usage:
│  │     ├─ Best for: Mature, profitable companies with stable margins
│  │     ├─ Avoid for: Financial companies (non-linear earnings), utilities (capital-intensive), startups (unprofitable)
│  │     ├─ Adjustment: Use "normalized" or "adjusted" P/E (exclude one-time items)
│  │     └─ Typical range: 10-25× for developed market equities
│  │
│  ├─ EV/EBITDA (Enterprise Value-to-EBITDA):
│  │  ├─ Definition:
│  │  │  ├─ Enterprise Value = Market Cap + Total Debt - Cash & Equivalents
│  │  │  ├─ EBITDA = Earnings before Interest, Taxes, Depreciation, Amortization
│  │  │  ├─ Ratio = (Market Cap + Debt - Cash) / EBITDA
│  │  │  └─ Interpretation: "How many years of EBITDA does the market pay?"
│  │  │
│  │  ├─ Advantages (vs P/E):
│  │  │  ├─ Removes capital structure effect (debt, equity split doesn't affect valuation)
│  │  │  ├─ Removes tax effect (pre-tax, so tax rates don't distort comparison)
│  │  │  ├─ Removes accounting depreciation (non-cash; relevant for comparisons)
│  │  │  ├─ Used widely in M&A (transaction multiples based on EV/EBITDA)
│  │  │  └─ More stable than P/E in high-leverage situations
│  │  │
│  │  ├─ Disadvantages:
│  │  │  ├─ Ignores capex needs (EBITDA overstates cash available after reinvestment)
│  │  │  ├─ Ignores working capital requirements
│  │  │  ├─ Company with $100M EBITDA but needs $80M annual capex → Not $100M cash available
│  │  │  ├─ Requires careful Net Debt calculation (what counts as cash? restricted funds?)
│  │  │  └─ EBITDA itself can be manipulated (add-backs to normalize)
│  │  │
│  │  └─ Usage:
│  │     ├─ Best for: Capital-intensive industries (utilities, telecom, infrastructure), M&A comparison
│  │     ├─ Avoid for: Service companies with low capex (where EBITDA ≈ FCF, so P/E suffices)
│  │     ├─ Typical range: 6-12× for mature utilities, 10-15× for growth industrials
│  │     └─ M&A baseline: Transaction EV/EBITDA typically 8-12× in normal markets
│  │
│  ├─ P/B Ratio (Price-to-Book):
│  │  ├─ Definition: Market Cap / Total Shareholder Equity (Book Value)
│  │  ├─ Interpretation: "How much premium/discount to accounting value?"
│  │  ├─ P/B = 1.0: Trading at book value (no intangible premium)
│  │  ├─ P/B > 1.0: Investors expect ROE > cost of equity (intangibles, competitive advantage)
│  │  ├─ P/B < 1.0: Trading below book (either undervalued or expected ROE < cost of capital)
│  │  │
│  │  ├─ Advantages:
│  │  │  ├─ Less subject to manipulation than earnings (book value fairly stable)
│  │  │  ├─ Useful for asset-heavy companies (banks, insurers, REITs)
│  │  │  ├─ Works for unprofitable companies (P/E undefined)
│  │  │  └─ Long history of data available (P/B ratio one of oldest metrics)
│  │  │
│  │  ├─ Disadvantages:
│  │  │  ├─ Book value understates intangibles (brand, customer base, R&D built internally)
│  │  │  ├─ Tech/software companies: Book value minimal → P/B > 20× but not overvalued
│  │  │  ├─ Accounting changes: Asset revaluations, write-downs affect book value
│  │  │  ├─ ROE not always reflected in multiples (high ROE companies still trade at high P/B, low ROE at low)
│  │  │  └─ Not theoretically grounded (unlike P/E or DCF)
│  │  │
│  │  ├─ Usage:
│  │  │  ├─ Best for: Banks (tangible assets clear), insurers (asset-backed), utilities (assets critical)
│  │  │  ├─ Avoid for: Technology (intangible-heavy), pharma (R&D assets not on balance sheet)
│  │  │  ├─ Typical range: 1.0-3.0× for banks, >5× for tech firms
│  │  │  └─ Value indicator: P/B < 1.0 can signal value opportunity or distress
│  │  │
│  │  └─ Enhancement: Use "Price-to-Tangible Book Value"
│  │     ├─ Remove goodwill & intangibles from book value
│  │     ├─ More conservative than P/B (useful for banks with high goodwill)
│  │     └─ Formula: Market Cap / (Tangible Equity) where Tangible = Book Value - Goodwill - Intangibles
│  │
│  ├─ P/S Ratio (Price-to-Sales):
│  │  ├─ Definition: Market Cap / Total Revenue
│  │  ├─ Interpretation: "How many years of sales is market value?"
│  │  ├─ Advantages:
│  │  │  ├─ Difficult to manipulate (revenue recognized consistently)
│  │  │  ├─ Works for unprofitable companies (net income = 0, P/E undefined)
│  │  │  ├─ Less cyclical than earnings (sales more stable than profits)
│  │  │  └─ Useful for early-stage companies (revenue growth focus)
│  │  │
│  │  ├─ Disadvantages:
│  │  │  ├─ Ignores profitability (high-margin vs low-margin business look same by P/S)
│  │  │  ├─ Two companies at P/S 3×: One with 30% EBIT margin, one with 5% margin → very different value
│  │  │  ├─ Ignores capex requirements (revenue needed to support ongoing business)
│  │  │  ├─ Weak correlation to market returns (less predictive than P/E or P/B)
│  │  │  └─ High P/S not always mean overvalued (scale businesses have low P/S initially)
│  │  │
│  │  └─ Usage:
│  │     ├─ Best for: Loss-making growth companies, SaaS businesses (focus on top-line)
│  │     ├─ Secondary metric: Use with other multiples, not standalone
│  │     ├─ Typical range: 0.5-3× for mature businesses, 3-10× for growth SaaS
│  │     └─ Companion analysis: Margin improvement potential (turnaround story)
│  │
│  └─ PEG Ratio (Price/Earnings-to-Growth):
│     ├─ Definition: P/E Ratio / Expected Growth Rate (annual earnings growth %)
│     ├─ Example: P/E = 30, Expected growth = 30%/year → PEG = 30/30 = 1.0
│     ├─ Interpretation:
│     │  ├─ PEG = 1.0: Fair valued (growth reflected in multiple)
│     │  ├─ PEG < 1.0: Undervalued (high growth not fully priced in)
│     │  ├─ PEG > 1.0: Overvalued (growth expectations may be too high)
│     │  └─ Common rule: Buy at PEG < 1.0, sell at PEG > 1.5
│     │
│     ├─ Advantages:
│     │  ├─ Adjusts P/E for growth explicitly (addresses growth vs value tension)
│     │  ├─ Identifies undervalued growers (not all cheap stocks are good, not all expensive bad)
│     │  ├─ Simple to calculate if growth forecast available
│     │  └─ Intuitive to investors (lower PEG = better value)
│     │
│     ├─ Disadvantages:
│     │  ├─ Growth rate estimates highly subjective (analysts often too optimistic)
│     │  ├─ Assumes growth sustainable (high growth often mean-reverts, not permanent)
│     │  ├─ Doesn't account for risk (higher growth = higher volatility, not reflected in PEG)
│     │  ├─ Breaks down for very high growth (PEG = 50/100 = 0.5, but still expensive)
│     │  └─ Backward-looking growth might differ from forward expectations
│     │
│     └─ Usage:
│        ├─ Best for: Growth stock valuation (tech, biotech, emerging markets)
│        ├─ Screen for: PEG 0.5-1.5 range (moderate valuation)
│        ├─ Verify with: Other multiples (P/B, EV/EBITDA) to confirm story
│        └─ Typical range: 0.5-3.0 for growth stocks, <1.0 for strong opportunities
│
├─ Multiple Selection & Weighting:
│  ├─ Choose primary multiple:
│  │  ├─ P/E for profitable, stable-margin companies
│  │  ├─ EV/EBITDA for capital-intensive or leveraged firms
│  │  ├─ P/B for financial companies
│  │  ├─ P/S for unprofitable growth companies
│  │  └─ FCF yield for cash-generation focused analysis
│  │
│  ├─ Use multiple metrics:
│  │  ├─ Triangulate across 2-3 multiples (reduces single-metric bias)
│  │  ├─ Example: Calculate using P/E, EV/EBITDA, P/B → Average the 3 fair values
│  │  ├─ Confidence: Higher if multiple methods converge on similar value
│  │  └─ Divergence signals: Investigate if methods give very different valuations
│  │
│  ├─ Weight by relevance:
│  │  ├─ P/E: 50% weight (most reliable for stable business)
│  │  ├─ EV/EBITDA: 30% weight (validates EV impact)
│  │  ├─ P/B: 20% weight (asset value floor)
│  │  ├─ Example: (50% × P/E value) + (30% × EV/EBITDA value) + (20% × P/B value)
│  │  └─ Customize weights based on business characteristics
│  │
│  └─ Stress test assumptions:
│     ├─ Peer selection sensitivity (change peer set ±1-2 companies, remeasure)
│     ├─ Multiple sensitivity (±1 standard deviation, measure impact)
│     ├─ Example: P/E peers avg 20× → Test 18× and 22× scenarios
│     └─ Communicate range, not single value
│
├─ Limitations & When Multiples Fail:
│  ├─ Merger/restructuring year:
│  │  ├─ One-time charges depress earnings → Multiples distorted
│  │  ├─ Solution: Use normalized/adjusted multiples
│  │  └─ Example: P/E excludes restructuring charges for fair comparison
│  │
│  ├─ Accounting differences:
│  │  ├─ GAAP vs IFRS (depreciation methods, lease accounting, provisions)
│  │  ├─ US generally conservative, IFRS sometimes aggressive
│  │  ├─ International comparison: Standardize accounting basis
│  │  └─ Solution: Use EV/EBITDA (more accounting-neutral)
│  │
│  ├─ Cyclical industries:
│  │  ├─ Multiples swing dramatically through cycle
│  │  ├─ Example: Auto industry P/E = 4× at trough, 12× at peak (same companies)
│  │  ├─ Solution: Use normalized earnings or multiples (through-cycle average)
│  │  └─ Timing risk: Multiples expansion can be as profitable as earnings growth
│  │
│  ├─ Growth inflection:
│  │  ├─ Company transitioning: Startup → scaled company (growth rates change)
│  │  ├─ Multiples will adjust (likely upward if growth accelerates, downward if decelerates)
│  │  ├─ Solution: Assess inflection timing and magnitude (scenario analysis)
│  │  └─ Example: Netflix multiples compressed during transition from 30% to 10% growth
│  │
│  └─ Market sentiment shifts:
│     ├─ Multiple compression/expansion independent of fundamentals
│     ├─ Tech correction 2022: P/E fell from 28× to 18× (market multiple reset)
│     ├─ Defensive: Use DCF validation to ensure multiples justified
│     └─ Opportunity: Emotional moves can create mispricing
│
└─ Practical Integration with DCF:
   ├─ Relative valuation as sanity check:
   │  ├─ DCF gives $100/share → Comparable analysis gives $95/share
   │  ├─ Convergence suggests solid valuation; divergence requires investigation
   │  ├─ If DCF $100 but comps $150 → Either DCF assumptions conservative or comps high
   │  └─ Resolution: Adjust assumptions or peer selection
   │
   ├─ Terminal value linkage:
   │  ├─ DCF terminal value often derived from exit multiple
   │  ├─ Example: Terminal year EBITDA $50M × 8× EV/EBITDA exit multiple = $400M terminal value
   │  ├─ Sensitivity: Terminal value often 70-80% of total DCF value → Multiple selection critical
   │  └─ Peer comparison: Ensure exit multiple reasonable vs historical comps
   │
   └─ Hybrid approach:
      ├─ Weight DCF 50%, Relative Valuation 50% in final fairness opinion
      ├─ Reduces single-method risk (both methods have blind spots)
      ├─ Communicate: "Fair value $95-105, based on DCF + comps analysis"
      └─ Professional standard for M&A, IPO fairness opinions
```

**Interaction:** Compare target company metric (P/E, EBITDA, book value) to peer companies → Derive peer multiple → Apply to target → Valuation range. Rigor: Peer selection critical (garbage in, garbage out).

## 5. Mini-Project
Implement comparable company analysis for tech company valuation:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*100)
print("RELATIVE VALUATION: COMPARABLE COMPANY ANALYSIS")
print("="*100)

# Define peer companies and financials
peers_data = {
    'Company': ['Microsoft', 'Google', 'Meta', 'Amazon', 'Apple'],
    'Market_Cap_B': [2500, 1600, 900, 1400, 3000],
    'Net_Income_B': [75, 76, 23, -2.7, 99],
    'EBITDA_B': [120, 110, 50, 45, 130],
    'Revenue_B': [230, 280, 125, 470, 400],
    'Book_Value_B': [200, 180, 80, 150, 60],
    'Debt_B': [50, 30, 10, 35, 110],
    'Cash_B': [100, 110, 60, 80, 30],
}

peers_df = pd.DataFrame(peers_data)

# Calculate multiples for peers
peers_df['P/E'] = peers_df['Market_Cap_B'] / peers_df['Net_Income_B']
peers_df['EV'] = peers_df['Market_Cap_B'] + peers_df['Debt_B'] - peers_df['Cash_B']
peers_df['EV/EBITDA'] = peers_df['EV'] / peers_df['EBITDA_B']
peers_df['P/S'] = peers_df['Market_Cap_B'] / peers_df['Revenue_B']
peers_df['P/B'] = peers_df['Market_Cap_B'] / peers_df['Book_Value_B']

print("\nPEER COMPANIES - FINANCIAL SNAPSHOT")
print("-" * 100)
print(peers_df[['Company', 'Market_Cap_B', 'Net_Income_B', 'Revenue_B', 'EBITDA_B', 'Book_Value_B']].to_string(index=False))

print("\n\nVALUATION MULTIPLES - PEER GROUP")
print("-" * 100)
multiples_df = peers_df[['Company', 'P/E', 'EV/EBITDA', 'P/S', 'P/B']].copy()
for col in ['P/E', 'EV/EBITDA', 'P/S', 'P/B']:
    multiples_df[col] = multiples_df[col].apply(lambda x: f"{x:.2f}x" if x > 0 else "N/A")
print(multiples_df.to_string(index=False))

# Calculate peer multiples (median)
pe_multiples = peers_df['P/E'][peers_df['P/E'] > 0]
ev_ebitda_multiples = peers_df['EV/EBITDA'][peers_df['EV/EBITDA'] > 0]
ps_multiples = peers_df['P/S'][peers_df['P/S'] > 0]
pb_multiples = peers_df['P/B'][peers_df['P/B'] > 0]

print(f"\n\nPEER MULTIPLE STATISTICS")
print("-" * 100)
print(f"P/E Ratio:          Mean = {pe_multiples.mean():.2f}x | Median = {pe_multiples.median():.2f}x | StDev = {pe_multiples.std():.2f}x")
print(f"EV/EBITDA:          Mean = {ev_ebitda_multiples.mean():.2f}x | Median = {ev_ebitda_multiples.median():.2f}x | StDev = {ev_ebitda_multiples.std():.2f}x")
print(f"P/S Ratio:          Mean = {ps_multiples.mean():.2f}x | Median = {ps_multiples.median():.2f}x | StDev = {ps_multiples.std():.2f}x")
print(f"P/B Ratio:          Mean = {pb_multiples.mean():.2f}x | Median = {pb_multiples.median():.2f}x | StDev = {pb_multiples.std():.2f}x")

# Target company to value
target_data = {
    'Company': 'Target Tech Inc',
    'Net_Income_B': 8.5,
    'EBITDA_B': 15.0,
    'Revenue_B': 60.0,
    'Book_Value_B': 25.0,
    'Debt_B': 5.0,
    'Cash_B': 15.0,
    'Shares_M': 500,  # Million shares
}

print(f"\n\nTARGET COMPANY FINANCIALS")
print("-" * 100)
print(f"Company: {target_data['Company']}")
print(f"Net Income: ${target_data['Net_Income_B']:.2f}B")
print(f"EBITDA: ${target_data['EBITDA_B']:.2f}B")
print(f"Revenue: ${target_data['Revenue_B']:.2f}B")
print(f"Book Value: ${target_data['Book_Value_B']:.2f}B")
print(f"Shares Outstanding: {target_data['Shares_M']:.0f}M")

# Valuation using different multiples
print(f"\n\nVALUATION SCENARIOS")
print("-" * 100)

# Scenario 1: P/E Valuation
pe_conservative = pe_multiples.min()
pe_median = pe_multiples.median()
pe_aggressive = pe_multiples.max()

target_valuation_pe_cons = (target_data['Net_Income_B'] * pe_conservative * 1000) / target_data['Shares_M']
target_valuation_pe_med = (target_data['Net_Income_B'] * pe_median * 1000) / target_data['Shares_M']
target_valuation_pe_agg = (target_data['Net_Income_B'] * pe_aggressive * 1000) / target_data['Shares_M']

print(f"\n1. P/E VALUATION (Price-to-Earnings)")
print(f"   Conservative (Min P/E {pe_conservative:.2f}x): ${target_valuation_pe_cons:.2f}/share")
print(f"   Base Case (Median P/E {pe_median:.2f}x):     ${target_valuation_pe_med:.2f}/share")
print(f"   Aggressive (Max P/E {pe_aggressive:.2f}x):   ${target_valuation_pe_agg:.2f}/share")

# Scenario 2: EV/EBITDA Valuation
ev_conservative = ev_ebitda_multiples.min()
ev_median = ev_ebitda_multiples.median()
ev_aggressive = ev_ebitda_multiples.max()

target_ev_cons = target_data['EBITDA_B'] * ev_conservative
target_ev_med = target_data['EBITDA_B'] * ev_median
target_ev_agg = target_data['EBITDA_B'] * ev_aggressive

target_equity_cons = (target_ev_cons - target_data['Debt_B'] + target_data['Cash_B']) * 1000 / target_data['Shares_M']
target_equity_med = (target_ev_med - target_data['Debt_B'] + target_data['Cash_B']) * 1000 / target_data['Shares_M']
target_equity_agg = (target_ev_agg - target_data['Debt_B'] + target_data['Cash_B']) * 1000 / target_data['Shares_M']

print(f"\n2. EV/EBITDA VALUATION")
print(f"   Conservative (EV/EBITDA {ev_conservative:.2f}x): ${target_equity_cons:.2f}/share")
print(f"   Base Case (EV/EBITDA {ev_median:.2f}x):     ${target_equity_med:.2f}/share")
print(f"   Aggressive (EV/EBITDA {ev_aggressive:.2f}x):   ${target_equity_agg:.2f}/share")

# Scenario 3: P/S Valuation
ps_conservative = ps_multiples.min()
ps_median = ps_multiples.median()
ps_aggressive = ps_multiples.max()

target_valuation_ps_cons = (target_data['Revenue_B'] * ps_conservative * 1000) / target_data['Shares_M']
target_valuation_ps_med = (target_data['Revenue_B'] * ps_median * 1000) / target_data['Shares_M']
target_valuation_ps_agg = (target_data['Revenue_B'] * ps_aggressive * 1000) / target_data['Shares_M']

print(f"\n3. P/S VALUATION (Price-to-Sales)")
print(f"   Conservative (Min P/S {ps_conservative:.2f}x): ${target_valuation_ps_cons:.2f}/share")
print(f"   Base Case (Median P/S {ps_median:.2f}x):     ${target_valuation_ps_med:.2f}/share")
print(f"   Aggressive (Max P/S {ps_aggressive:.2f}x):   ${target_valuation_ps_agg:.2f}/share")

# Scenario 4: P/B Valuation
pb_conservative = pb_multiples.min()
pb_median = pb_multiples.median()
pb_aggressive = pb_multiples.max()

target_valuation_pb_cons = (target_data['Book_Value_B'] * pb_conservative * 1000) / target_data['Shares_M']
target_valuation_pb_med = (target_data['Book_Value_B'] * pb_median * 1000) / target_data['Shares_M']
target_valuation_pb_agg = (target_data['Book_Value_B'] * pb_aggressive * 1000) / target_data['Shares_M']

print(f"\n4. P/B VALUATION (Price-to-Book)")
print(f"   Conservative (Min P/B {pb_conservative:.2f}x): ${target_valuation_pb_cons:.2f}/share")
print(f"   Base Case (Median P/B {pb_median:.2f}x):     ${target_valuation_pb_med:.2f}/share")
print(f"   Aggressive (Max P/B {pb_aggressive:.2f}x):   ${target_valuation_pb_agg:.2f}/share")

# Summary valuation range
all_vals_base = [target_valuation_pe_med, target_equity_med, target_valuation_ps_med, target_valuation_pb_med]
all_vals_cons = [target_valuation_pe_cons, target_equity_cons, target_valuation_ps_cons, target_valuation_pb_cons]
all_vals_agg = [target_valuation_pe_agg, target_equity_agg, target_valuation_ps_agg, target_valuation_pb_agg]

print(f"\n\nFAIR VALUE SUMMARY")
print("-" * 100)
print(f"Conservative Case (Low multiples):  ${np.mean(all_vals_cons):.2f}/share (Range: ${np.min(all_vals_cons):.2f} - ${np.max(all_vals_cons):.2f})")
print(f"Base Case (Median multiples):       ${np.mean(all_vals_base):.2f}/share (Range: ${np.min(all_vals_base):.2f} - ${np.max(all_vals_base):.2f})")
print(f"Aggressive Case (High multiples):   ${np.mean(all_vals_agg):.2f}/share (Range: ${np.min(all_vals_agg):.2f} - ${np.max(all_vals_agg):.2f})")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Peer multiples comparison
ax = axes[0, 0]
multiples_labels = ['P/E', 'EV/EBITDA', 'P/S', 'P/B']
peer_means = [pe_multiples.mean(), ev_ebitda_multiples.mean(), ps_multiples.mean(), pb_multiples.mean()]
peer_stds = [pe_multiples.std(), ev_ebitda_multiples.std(), ps_multiples.std(), pb_multiples.std()]

ax.bar(multiples_labels, peer_means, yerr=peer_stds, capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
ax.set_title('Peer Multiple Averages (with Std Dev)')
ax.set_ylabel('Multiple (x)')
ax.grid(alpha=0.3, axis='y')
for i, (mean, std) in enumerate(zip(peer_means, peer_stds)):
    ax.text(i, mean + std + 0.3, f'{mean:.2f}x', ha='center', fontsize=9)

# Plot 2: Valuation by method (base case)
ax = axes[0, 1]
methods = ['P/E', 'EV/EBITDA', 'P/S', 'P/B']
values = all_vals_base
colors_vals = ['blue', 'green', 'orange', 'red']

ax.barh(methods, values, color=colors_vals, alpha=0.7)
ax.axvline(np.mean(values), color='black', linestyle='--', linewidth=2, label=f'Average: ${np.mean(values):.2f}')
ax.set_title('Target Fair Value by Method (Base Case)')
ax.set_xlabel('Value per Share ($)')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Plot 3: Valuation range (scenarios)
ax = axes[1, 0]
scenarios = ['Conservative', 'Base Case', 'Aggressive']
scenario_means = [np.mean(all_vals_cons), np.mean(all_vals_base), np.mean(all_vals_agg)]
scenario_mins = [np.min(all_vals_cons), np.min(all_vals_base), np.min(all_vals_agg)]
scenario_maxs = [np.max(all_vals_cons), np.max(all_vals_base), np.max(all_vals_agg)]
scenario_ranges = [scenario_maxs[i] - scenario_mins[i] for i in range(len(scenarios))]

ax.bar(scenarios, scenario_means, yerr=[np.array(scenario_means) - np.array(scenario_mins),
                                        np.array(scenario_maxs) - np.array(scenario_means)],
       capsize=5, alpha=0.7, color=['red', 'blue', 'green'])
ax.set_title('Fair Value Range Across Scenarios')
ax.set_ylabel('Value per Share ($)')
ax.grid(alpha=0.3, axis='y')
for i, (mean, low, high) in enumerate(zip(scenario_means, scenario_mins, scenario_maxs)):
    ax.text(i, high + 2, f'${high:.0f}', ha='center', fontsize=8)
    ax.text(i, mean, f'${mean:.2f}', ha='center', fontsize=9, fontweight='bold')

# Plot 4: Peer company multiples scatter
ax = axes[1, 1]
ax.scatter(peers_df['EV/EBITDA'], peers_df['P/E'], s=300, alpha=0.6, color='blue')
for idx, row in peers_df.iterrows():
    ax.annotate(row['Company'], (row['EV/EBITDA'], row['P/E']), 
                fontsize=9, ha='center', va='center')

ax.set_xlabel('EV/EBITDA')
ax.set_ylabel('P/E Ratio')
ax.set_title('Peer Multiple Relationship (EV/EBITDA vs P/E)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)
print(f"- Comparable company analysis depends critically on peer selection")
print(f"- Use multiple metrics (P/E, EV/EBITDA, P/S, P/B) to triangulate valuation")
print(f"- Median more robust than mean (removes outliers)")
print(f"- Valuation range communicates uncertainty (not single point estimate)")
print(f"- Convergence across methods increases confidence in valuation")
```

## 6. Challenge Round
- Build peer screening tool: Identify 10 comparable companies using quantitative criteria (size, ROE, growth), calculate multiples, determine outliers
- Sensitivity analysis: Create 3-way table (P/E multiple × EPS × Share count) showing valuation at different assumptions
- Historical multiple analysis: Compare current multiples to 3-year rolling average; assess if company trading at premium/discount to historical norm
- Transaction analysis: Research 5 recent M&A deals in sector, analyze EV/EBITDA multiples paid, identify synergy adjustments
- Valuation summary memo: Write 1-page fairness opinion using 4+ multiples, clearly state assumptions, quantify margin of safety

## 7. Key References
- [Damodaran (2012), "Valuation Approaches and Metrics: A Guide for Equity Valuation," Stern NYU](https://pages.stern.nyu.edu/~adamodar/pdfiles/eqnotes/eqval.pdf) — Comprehensive multiples framework
- [PwC (2023), "Multiples Valuation in M&A: A Practitioner's Guide," Deal Advisory](https://www.pwc.com/valuations) — Transaction multiples analysis
- [Rosenbaum & Pearl (2013), "Investment Banking: Valuation, Leveraged Buyouts, and Mergers & Acquisitions," Wiley](https://www.wiley.com/en-us/Investment+Banking%3A+Valuation%2C+Leveraged+Buyouts%2C+and+Mergers+and+Acquisitions%2C+3rd+Edition-p-9781118004518) — Practitioner M&A standard
- [Alford (1992), "The Effect of the Set of Comparable Firms on Estimates of Value," Journal of Finance](https://www.jstor.org/stable/2329066) — Peer selection impact on valuation accuracy

---
**Status:** Industry standard (widely used in M&A, IPOs, equity research) | **Complements:** DCF Analysis, Scenario Planning, Financial Statement Analysis, Market Multiples Tracking
