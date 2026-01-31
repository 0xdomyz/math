# Financial Statement Analysis & Ratio Interpretation: Extracting Value Signals from Financials

## I. Concept Skeleton

**Definition:** Financial statement analysis involves extracting quantitative metrics (ratios) from balance sheets, income statements, and cash flow statements to assess company profitability, efficiency, leverage, liquidity, and growth potential. Ratios normalize financials across time periods and company sizes, enabling comparative valuation and trend detection.

**Purpose:** Identify financial strength/weakness, detect manipulation or deterioration, compare peers at standardized metrics, project future performance based on historical trends, and validate DCF assumptions (margins, ROIC, growth rates).

**Prerequisites:** Financial statement structure (balance sheet, income statement, cash flow), accrual vs. cash accounting, double-entry bookkeeping, time-value of money, comparable company analysis framework.

---

## II. Comparative Framing

| **Ratio Category** | **Key Metrics** | **Calculation** | **Interpretation** | **Valuation Use** |
|-----------|----------|----------|----------|----------|
| **Profitability** | Gross margin, Operating margin, Net margin | (Gross Profit / Revenue), (EBIT / Revenue), (Net Income / Revenue) | Higher = Better efficiency; Compare to peers & historical | Project sustainable margins in DCF |
| **Efficiency** | Asset turnover, Receivables days, Inventory days | (Revenue / Avg. Assets), (AR / Daily Revenue), (Inventory / Daily COGS) | Higher turnover = Better capital deployment | Justify ROIC assumptions |
| **Leverage** | Debt-to-equity, Interest coverage, Debt-to-EBITDA | (Total Debt / Equity), (EBIT / Interest Exp), (Total Debt / EBITDA) | Lower = Less financial risk; Coverage >2x = Safe | Adjust cost of capital, validate solvency |
| **Liquidity** | Current ratio, Quick ratio, Working capital days | (Current Assets / Current Liabilities), (CA - Inventory) / CL, (AR+Inv) / (Daily Op. Exp) | >1.5 = Comfortable; <1 = Distress risk | Assess cash flow timing mismatches |
| **Growth** | Revenue growth, EPS growth, Free cash flow growth | (Revenue_t / Revenue_t-1 - 1), (EPS_t / EPS_t-1 - 1), (FCF_t / FCF_t-1 - 1) | Multi-year trends; Compare to GDP/industry | Terminal growth rate anchor (usually 2-3%) |

---

## III. Examples & Counterexamples

### Example 1: Margin Analysis - Why Higher Margins ≠ Always Better

**Setup:**
- Company A: Revenue $1B, Net margin 20%, Net income $200M
- Company B: Revenue $10B, Net margin 5%, Net income $500M
- Question: Which is more valuable?

**Naive Analysis (Misleading):**

```
Interpretation: "Company A is better! 20% margin vs 5%"

Problem with this logic:
├─ Margin in isolation ignores SCALE
├─ Company B generates 2.5x more absolute profit
├─ Valuation depends on both margin AND growth potential
├─
└─ Conclusion: Margin is a quality metric, NOT a value metric
```

**Sophisticated Analysis (Correct):**

```
Question 1: Why does Company B have lower margin?

Possibilities:
├─ High-volume, low-margin business model (e.g., retail)
│  └─ May be sustainable and very profitable
├─ Temporary margin compression (cyclical downturn)
│  └─ Margins may recover
├─ Structural weakness (losing competitive position)
│  └─ Future deterioration risk
└─ Mix shift (selling lower-margin products)
   └─ May indicate channel expansion

Check via trend analysis:
├─ Company A margins: 20%, 21%, 19%, 22% (stable, sustainable)
├─ Company B margins: 8%, 7%, 5%, 4% (declining, problematic!)
│  └─ Interpretation: Margin compression is accelerating
│
└─ Revised assessment: Company B at risk despite higher absolute profit
```

**Enterprise Value Comparison:**

```
Company A (Sustainable margins):
├─ Enterprise Value: $4B
├─ Multiples: $4B / $200M = 20x Net Income
├─ Reasoning: Stable 20% margins, justified 20x multiple
│
└─ ROE: $200M / $1B equity = 20% (excellent)

Company B (Deteriorating margins):
├─ Enterprise Value: $3B
├─ Multiples: $3B / $500M = 6x Net Income
├─ Reasoning: Declining margins → lower multiple despite higher earnings
│
└─ ROE: $500M / $2B equity = 25% (high, but declining margins concern)
```

**Key Insight:**

```
Valuation framework:
├─ Step 1: Calculate current profitability (margins, ROE, ROIC)
├─ Step 2: Assess trend (improving? Stable? Deteriorating?)
├─ Step 3: Identify drivers (structural or cyclical?)
├─ Step 4: Project sustainable level (normalized margins for DCF)
│
└─ Multiple valuation uses:
   ├─ Company A: 20x justified (high quality, stable)
   ├─ Company B: 6x justified (deteriorating quality)
   └─ Same earnings level ≠ same valuation
```

---

### Example 2: Efficiency Ratios - Asset Turnover & Capital Intensity

**Setup:**
- Company X (Efficient): Revenue $1B, Avg. assets $500M, ROIC 25%
- Company Y (Capital Heavy): Revenue $1B, Avg. assets $2B, ROIC 12.5%
- Question: Why different ROIC despite same revenue?

**Asset Turnover Calculation:**

```
Company X:
├─ Asset turnover: $1B / $500M = 2.0x
│  (Generates $2 of revenue per $1 of assets)
├─ ROIC: Net after-tax profit / Capital invested
│  ├─ If profit = $100M
│  ├─ ROIC = $100M / $500M = 20% 
│  └─ Actually: Can be higher if ROIC = Net profit / (equity + debt)
│
└─ Interpretation: Asset-light model (e.g., software, consulting)

Company Y:
├─ Asset turnover: $1B / $2B = 0.5x
│  (Generates $0.50 of revenue per $1 of assets)
├─ ROIC: 12.5% (same profit margins but lower leverage)
│
└─ Interpretation: Capital-intensive model (e.g., manufacturing, utilities)
```

**Valuation Impact:**

```
Company X (High turnover):
├─ Low capex required for growth
├─ Working capital: Minimal (~30 days receivables/payables)
├─ FCF = NI - Capex - ∆WC = Often high relative to NI
└─ Valuation: Higher FCF conversion → higher multiple justified

Company Y (Low turnover):
├─ High capex required for growth (asset maintenance + expansion)
├─ Working capital: Significant (~90 days operating cycle)
├─ FCF = NI - Capex - ∆WC = Often much lower than NI
└─ Valuation: Lower FCF conversion → lower multiple justified
```

**Why This Matters for DCF:**

```
Sustainable cash flow projection:

Company X:
├─ Revenue $1B growing 10% → $1.1B next year
├─ Capex required: $50M (5% of revenue change, asset-light)
├─ Working capital increase: $10M
├─ FCF = NI($110M) - Capex($50M) - ∆WC($10M) = $50M
│
└─ FCF conversion: 45% of net income

Company Y:
├─ Revenue $1B growing 10% → $1.1B next year
├─ Capex required: $200M (20% of revenue change, capital-intensive)
├─ Working capital increase: $50M
├─ FCF = NI($110M) - Capex($200M) - ∆WC($50M) = -$140M
│
└─ FCF conversion: Negative (reinvestment exceeds earnings!)
```

---

### Example 3: Leverage & Coverage - Safety Margin for Debt

**Setup:**
- Company P: EBIT $100M, Interest expense $20M, Debt $500M, Equity $300M
- Company Q: EBIT $100M, Interest expense $40M, Debt $800M, Equity $200M
- Question: Which is safer? Which should get lower cost of capital?

**Coverage Ratio Analysis:**

```
Company P:
├─ Interest coverage: EBIT / Interest = $100M / $20M = 5.0x
│  (Can pay interest 5 times over from operating earnings)
├─ Debt-to-equity: $500M / $300M = 1.67x
├─ Debt-to-EBITDA: $500M / $100M = 5.0x (assuming EBIT = EBITDA)
│
└─ Interpretation: Comfortable leverage, low default risk

Company Q:
├─ Interest coverage: EBIT / Interest = $100M / $40M = 2.5x
│  (Can pay interest 2.5 times over; tight)
├─ Debt-to-equity: $800M / $200M = 4.0x (very high)
├─ Debt-to-EBITDA: $800M / $100M = 8.0x (highly leveraged)
│
└─ Interpretation: Vulnerable to earnings shock, higher default risk
```

**Valuation Implication:**

```
Cost of Capital Adjustment:

Company P (Safe leverage):
├─ Risk-free rate: 5%
├─ Beta: 1.0 (market risk)
├─ Cost of equity = 5% + 1.0 × 5% = 10%
├─ Cost of debt (after-tax): 4% × (1 - 25% tax) = 3%
├─ Weighted avg: 60% × 10% + 40% × 3% = 7.2%
│
└─ Reasoning: Low leverage → lower cost of capital

Company Q (Risky leverage):
├─ Risk-free rate: 5%
├─ Beta: 1.5 (higher leverage increases equity risk)
├─ Cost of equity = 5% + 1.5 × 5% = 12.5%
├─ Cost of debt (after-tax): 6% × (1 - 25% tax) = 4.5% (higher = riskier)
├─ Weighted avg: 60% × 12.5% + 40% × 4.5% = 9.3%
│
└─ Reasoning: High leverage → higher cost of capital → lower valuation
```

**Impact on Enterprise Value:**

```
Scenario: Both companies have $100M FCF next year, growing 3% perpetually

Company P (WACC 7.2%):
├─ Enterprise Value = FCF / (WACC - g) = $100M / (7.2% - 3%) = $2,381M
└─ Multiple: 23.8x FCF (justified by safety)

Company Q (WACC 9.3%):
├─ Enterprise Value = FCF / (WACC - g) = $100M / (9.3% - 3%) = $1,515M
└─ Multiple: 15.2x FCF (penalized for risk)

Valuation difference: $2,381M - $1,515M = $866M (37% discount for Q)
```

**Counterexample: High Leverage Justified by Cash Flow**

```
Company R: EBIT $500M, Interest $100M, Debt $1,000M, Equity $500M

Coverage:
├─ Interest coverage: $500M / $100M = 5.0x (comfortable!)
├─ Debt-to-equity: 2.0x (high absolute ratio)
│
└─ Assessment: Despite high absolute leverage, coverage is safe
   → Justified if EBIT is stable and unlikely to decline
```

---

## IV. Layer Breakdown

```
FINANCIAL STATEMENT ANALYSIS & RATIO FRAMEWORK

┌─────────────────────────────────────────────────────┐
│  1. PROFITABILITY ANALYSIS                          │
│                                                     │
│  Gross Margin (Revenue - COGS) / Revenue:           │
│  ├─ Measures: Direct production efficiency          │
│  ├─ Drivers: Input costs, pricing power,            │
│  │  manufacturing automation                        │
│  ├─ Interpretation:                                 │
│  │  ├─ >50% = Strong pricing/low costs              │
│  │  ├─ 20-50% = Varies by industry                  │
│  │  ├─ <20% = Commoditized, low margin              │
│  │  └─ Trend: Rising = Improving; Falling = Risk   │
│  ├─ Across industries:                              │
│  │  ├─ Software: 70-80%+ (minimal COGS)            │
│  │  ├─ Retail: 30-40% (high COGS)                  │
│  │  ├─ Pharma: 50-60% (R&D intensive)              │
│  │  └─ Utilities: 20-30% (regulated, commodity)    │
│  └─ DCF use: Project sustainable gross margin      │
│     (typically industry + company competitive      │
│     position-based)                                 │
│                                                     │
│  Operating Margin (EBIT / Revenue):                 │
│  ├─ Measures: Operating leverage + efficiency      │
│  ├─ Captures: SG&A, R&D, depreciation              │
│  ├─ Interpretation:                                │
│  │  ├─ >20% = Highly efficient, strong moat       │
│  │  ├─ 10-20% = Healthy                            │
│  │  ├─ 5-10% = Competitive pressure                │
│  │  └─ <5% = Distressed or commodity               │
│  ├─ Operating leverage:                             │
│  │  ├─ High fixed costs → Margins expand with      │
│  │  │  revenue growth (e.g., software)             │
│  │  ├─ Low fixed costs → Margins stable             │
│  │  └─ Implication: Growth drives profit           │
│  │     acceleration in high fixed-cost             │
│  │     businesses                                   │
│  └─ DCF use: Project normalized operating margin   │
│     as company matures                              │
│                                                     │
│  Net Profit Margin (Net Income / Revenue):          │
│  ├─ Measures: Bottom-line profitability after      │
│  │  all expenses, taxes, interest                  │
│  ├─ Interpretation:                                │
│  │  ├─ >15% = Excellent (competitive advantage)   │
│  │  ├─ 5-15% = Normal (competitive market)        │
│  │  ├─ 0-5% = Thin (commodity or scale play)      │
│  │  └─ <0% = Loss-making (distressed)             │
│  ├─ Drivers of change:                              │
│  │  ├─ Operating margin expansion/contraction      │
│  │  ├─ Interest expense (leverage)                 │
│  │  ├─ Tax rate changes                            │
│  │  └─ Extraordinary items (one-time gains/losses) │
│  └─ DCF use: Validate against unlevered margins    │
│     + tax rate adjustment                          │
│                                                     │
│  Return on Equity (ROE = Net Income / Avg Equity): │
│  ├─ Measures: Shareholder return on invested      │
│  │  capital                                        │
│  ├─ DuPont decomposition: Net margin × Asset      │
│  │  turnover × Leverage ratio                      │
│  ├─ Interpretation:                                │
│  │  ├─ >15% = Excellent (beating cost of equity)  │
│  │  ├─ 10-15% = Good                               │
│  │  ├─ <10% = Underperforming (below cost of      │
│  │  │  capital typically)                          │
│  │  └─ Negative = Value destruction                │
│  ├─ Drivers: Efficiency, leverage, margins        │
│  └─ DCF use: Validate assumptions about future    │
│     profitability vs. cost of capital              │
│                                                     │
│  Return on Invested Capital (ROIC = NOPAT /        │
│  Invested Capital):                                 │
│  ├─ Measures: Return on ALL capital (equity +      │
│  │  debt), excludes financing effects              │
│  ├─ NOPAT = EBIT × (1 - Tax rate)                  │
│  ├─ Invested Capital = Equity + Debt - Cash        │
│  ├─ Interpretation:                                │
│  │  ├─ ROIC > WACC = Value creation               │
│  │  ├─ ROIC < WACC = Value destruction            │
│  │  └─ Gap magnitude = Economic profit per $      │
│  │     of capital invested                         │
│  ├─ Used in: EVA (Economic Value Added)            │
│  │  = NOPAT - (WACC × Invested Capital)           │
│  └─ DCF use: Critical for terminal value and      │
│     growth value assessment                        │
│                                                     │
│  Sustainable Growth Rate:                          │
│  ├─ = ROE × (1 - Payout ratio)                     │
│  ├─ = Retention ratio × ROE                        │
│  ├─ Interpretation: Maximum growth without        │
│  │  requiring external capital injection           │
│  ├─ Example: ROE 15%, Dividend 40% payout        │
│  │  → Sustainable growth = 15% × 60% = 9%        │
│  └─ DCF use: Anchor for terminal growth rate      │
│     (usually can't exceed 2-3% long-term)         │
│                                                     │
└──────────────────┬────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │  2. EFFICIENCY RATIOS & CAPITAL DEPLOYMENT  │
    │                                              │
    │  Asset Turnover (Revenue / Avg. Assets):     │
    │  ├─ Measures: How efficiently assets         │
    │  │  generate revenue                        │
    │  ├─ High turnover (>2.0):                   │
    │  │  ├─ Minimal capital needed for growth     │
    │  │  ├─ Examples: Retail, software (SaaS)    │
    │  │  └─ Valuation: Higher FCF margin         │
    │  ├─ Low turnover (<0.5):                    │
    │  │  ├─ Capital-intensive, heavy reinvestment│
    │  │  ├─ Examples: Manufacturing, utilities   │
    │  │  └─ Valuation: Adjust capex/WC impact   │
    │  └─ Trend: Declining = Worsening efficiency │
    │                                              │
    │  Receivables Days (AR / Daily Revenue):      │
    │  ├─ Measures: Average collection time        │
    │  │  from customers                          │
    │  ├─ Calculation: AR / (Revenue / 365)       │
    │  ├─ Interpretation:                         │
    │  │  ├─ <30 days = Efficient collection      │
    │  │  ├─ 30-60 days = Normal (B2B)            │
    │  │  ├─ >90 days = Credit risk or change     │
    │  │  └─ Trend: Rising = Quality deterioration│
    │  └─ Impact on cash flow: Longer collection  │
    │     → More working capital tied up          │
    │                                              │
    │  Inventory Days (Inventory / Daily COGS):    │
    │  ├─ Measures: How long inventory sits        │
    │  ├─ Calculation: Avg Inventory / (COGS / 365)│
    │  ├─ Interpretation:                         │
    │  │  ├─ <30 days = Fast-moving (retail,      │
    │  │  │  perishables)                         │
    │  │  ├─ 30-90 days = Normal (manufacturing)  │
    │  │  ├─ >120 days = Slow-moving or obsolete │
    │  │  └─ Trend: Rising = Demand weakness or   │
    │  │     obsolescence risk                    │
    │  └─ Impact: Slower inventory → cash tied up │
    │     & obsolescence risk increases           │
    │                                              │
    │  Payables Days (AP / Daily COGS):            │
    │  ├─ Measures: How long company takes to      │
    │  │  pay suppliers                           │
    │  ├─ High payables days = Supplier credit    │
    │  ├─ Trend: Rising = Stretched finances      │
    │  │  or improved supplier relationships      │
    │  └─ Working capital cycle = Receivables     │
    │     Days + Inventory Days - Payables Days  │
    │                                              │
    │  Cash Conversion Cycle (CCC):                │
    │  ├─ = (AR Days + Inv Days - AP Days)        │
    │  ├─ Measures: Days of working capital       │
    │  │  financing needed                        │
    │  ├─ Example:                                 │
    │  │  ├─ AR: 30 days, Inv: 45 days, AP: 20   │
    │  │  │  days                                 │
    │  │  └─ CCC = 30 + 45 - 20 = 55 days        │
    │  ├─ Negative CCC: Cash generated from       │
    │  │  operations before payment (best case)   │
    │  └─ DCF use: Adjust working capital impact  │
    │     on free cash flow                       │
    │                                              │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │  3. LEVERAGE & SOLVENCY RATIOS               │
    │                                              │
    │  Debt-to-Equity (Total Debt / Equity):       │
    │  ├─ Measures: Financial leverage and risk   │
    │  ├─ Interpretation:                         │
    │  │  ├─ <0.5x = Conservative                │
    │  │  ├─ 0.5-1.5x = Moderate                 │
    │  │  ├─ 1.5-2.5x = Elevated risk            │
    │  │  └─ >2.5x = Distressed                  │
    │  ├─ Industry-dependent:                      │
    │  │  ├─ Tech: <0.5x (high ROIC)             │
    │  │  ├─ Utilities: 1.5-2.5x (regulated)     │
    │  │  ├─ Real estate: 2-3x (asset-backed)    │
    │  │  └─ Banks: 10-15x (regulated leverage)   │
    │  └─ Trend: Rising = Deteriorating position  │
    │                                              │
    │  Debt-to-EBITDA (Total Debt / EBITDA):       │
    │  ├─ Measures: Years to repay debt from      │
    │  │  operating cash flow                     │
    │  ├─ Interpretation:                         │
    │  │  ├─ <2.0x = Very safe                    │
    │  │  ├─ 2-3x = Comfortable                   │
    │  │  ├─ 3-4x = Elevated                      │
    │  │  ├─ 4-5x = Stressed                      │
    │  │  └─ >5x = Distressed                     │
    │  ├─ More stable than D/E (earnings more     │
    │  │  volatile than cash flow)                │
    │  └─ Covenant trigger: Often <3.5x or 4x     │
    │                                              │
    │  Interest Coverage (EBIT / Interest Exp):    │
    │  ├─ Measures: Ability to pay interest from  │
    │  │  operating earnings                      │
    │  ├─ Interpretation:                         │
    │  │  ├─ >5.0x = Safe, low default risk      │
    │  │  ├─ 2.5-5.0x = Comfortable              │
    │  │  ├─ 1.5-2.5x = Vulnerable              │
    │  │  └─ <1.5x = Severe distress            │
    │  ├─ Covenant trigger: Often >2.5x           │
    │  └─ Forward-looking: Project coverage       │
    │     under different scenarios               │
    │                                              │
    │  Debt Service Coverage (FCF / (Interest +    │
    │  Principal)):                                │
    │  ├─ Measures: Can company pay debt          │
    │  │  service from operating cash?            │
    │  ├─ >1.25x = Safe margin                    │
    │  ├─ <1.0x = Cannot service without          │
    │  │  external financing                      │
    │  └─ Most reliable indicator of solvency     │
    │                                              │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │  4. LIQUIDITY & WORKING CAPITAL              │
    │                                              │
    │  Current Ratio (Current Assets / Current      │
    │  Liabilities):                                │
    │  ├─ Measures: Short-term solvency            │
    │  ├─ Interpretation:                         │
    │  │  ├─ >2.0x = Conservative, cash surplus   │
    │  │  ├─ 1.5-2.0x = Comfortable               │
    │  │  ├─ 1.0-1.5x = Tight but manageable      │
    │  │  └─ <1.0x = Distress, funding risk      │
    │  ├─ By industry:                             │
    │  │  ├─ Tech: 2-3x (high cash)              │
    │  │  ├─ Retail: 1-1.5x (rapid inventory     │
    │  │  │  turnover)                            │
    │  │  └─ Banks: <1.0x (normal due to         │
    │  │     deposits)                            │
    │  └─ Trend: Declining = Liquidity pressure   │
    │                                              │
    │  Quick Ratio (CA - Inventory / CL):          │
    │  ├─ Measures: Liquidity excluding slow       │
    │  │  assets                                  │
    │  ├─ More conservative than current ratio     │
    │  ├─ >1.0x = Safe; <0.5x = At risk          │
    │  └─ Used by creditors to assess real        │
    │     liquidity                                │
    │                                              │
    │  Operating Working Capital:                  │
    │  ├─ = Current Assets - Current Liabilities  │
    │  ├─ Measures: Capital tied up in daily ops  │
    │  ├─ Positive = Cash needed to fund growth   │
    │  ├─ Negative = Cash generated by growth     │
    │  │  (rare; usually retail with high        │
    │  │  payables)                               │
    │  └─ DCF use: Project ∆WC impact on FCF     │
    │     (working capital increases reduce FCF)  │
    │                                              │
    │  Cash Ratio (Cash / Current Liabilities):    │
    │  ├─ Most conservative liquidity measure      │
    │  ├─ <0.5x = Low cash reserves                │
    │  ├─ >1.0x = Excessive cash (opportunity     │
    │  │  cost)                                   │
    │  └─ Used for crisis scenarios (immediate    │
    │     liquidity needs)                         │
    │                                              │
    └────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### DuPont ROE Decomposition

$$\text{ROE} = \text{Net Margin} \times \text{Asset Turnover} \times \text{Equity Multiplier}$$

$$\text{ROE} = \frac{\text{Net Income}}{\text{Revenue}} \times \frac{\text{Revenue}}{\text{Avg Assets}} \times \frac{\text{Avg Assets}}{\text{Equity}}$$

### Cash Conversion Cycle

$$\text{CCC} = \text{DaysReceivables} + \text{DaysInventory} - \text{DaysPayable}$$

### Working Capital Change Impact on FCF

$$\text{FCF} = \text{Net Income} - \text{Capex} - \Delta\text{WorkingCapital}$$

Where $\Delta\text{WC} = \text{WC}_t - \text{WC}_{t-1}$ (increase in WC reduces FCF)

### Normalized Margin Projection

$$\text{Normalized Margin}_t = \text{Historical Average} + \text{Trend Adjustment} + \text{Cyclical Adjustment}$$

---

## VI. Python Mini-Project: Financial Statement Analyzer & Ratio Interpreter

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ============================================================================
# FINANCIAL STATEMENT RATIO ANALYZER
# ============================================================================

class FinancialStatementAnalyzer:
    """Parse financial statements and calculate key ratios"""
    
    def __init__(self, company_name, fiscal_year):
        self.company_name = company_name
        self.fiscal_year = fiscal_year
    
    @staticmethod
    def calculate_profitability_ratios(revenue, cogs, operating_income, 
                                       net_income, avg_equity, avg_assets):
        """
        Calculate profitability metrics
        """
        gross_margin = (revenue - cogs) / revenue if revenue > 0 else 0
        operating_margin = operating_income / revenue if revenue > 0 else 0
        net_margin = net_income / revenue if revenue > 0 else 0
        roe = net_income / avg_equity if avg_equity > 0 else 0
        roa = net_income / avg_assets if avg_assets > 0 else 0
        
        return {
            'Gross Margin': gross_margin,
            'Operating Margin': operating_margin,
            'Net Margin': net_margin,
            'ROE': roe,
            'ROA': roa
        }
    
    @staticmethod
    def calculate_efficiency_ratios(revenue, cogs, avg_assets, 
                                    avg_receivables, avg_inventory, 
                                    avg_payables):
        """
        Calculate operational efficiency metrics
        """
        asset_turnover = revenue / avg_assets if avg_assets > 0 else 0
        
        daily_revenue = revenue / 365
        receivables_days = avg_receivables / daily_revenue if daily_revenue > 0 else 0
        
        daily_cogs = cogs / 365
        inventory_days = avg_inventory / daily_cogs if daily_cogs > 0 else 0
        
        payables_days = avg_payables / daily_cogs if daily_cogs > 0 else 0
        
        cash_conversion_cycle = receivables_days + inventory_days - payables_days
        
        return {
            'Asset Turnover': asset_turnover,
            'Receivables Days': receivables_days,
            'Inventory Days': inventory_days,
            'Payables Days': payables_days,
            'Cash Conversion Cycle': cash_conversion_cycle
        }
    
    @staticmethod
    def calculate_leverage_ratios(total_debt, avg_equity, ebit, interest_expense, 
                                  ebitda, fcf):
        """
        Calculate financial leverage and solvency metrics
        """
        debt_to_equity = total_debt / avg_equity if avg_equity > 0 else 0
        
        interest_coverage = ebit / interest_expense if interest_expense > 0 else np.inf
        
        debt_to_ebitda = total_debt / ebitda if ebitda > 0 else 0
        
        # Approximate debt service = Interest + Principal (assume 10% principal repayment)
        principal_repayment = total_debt * 0.1
        debt_service_coverage = fcf / (interest_expense + principal_repayment) if \
                               (interest_expense + principal_repayment) > 0 else 0
        
        return {
            'Debt-to-Equity': debt_to_equity,
            'Interest Coverage': interest_coverage,
            'Debt-to-EBITDA': debt_to_ebitda,
            'Debt Service Coverage': debt_service_coverage
        }
    
    @staticmethod
    def calculate_liquidity_ratios(current_assets, current_liabilities, 
                                   inventory, cash):
        """
        Calculate short-term liquidity metrics
        """
        current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
        
        quick_ratio = (current_assets - inventory) / current_liabilities if \
                     current_liabilities > 0 else 0
        
        cash_ratio = cash / current_liabilities if current_liabilities > 0 else 0
        
        working_capital = current_assets - current_liabilities
        
        return {
            'Current Ratio': current_ratio,
            'Quick Ratio': quick_ratio,
            'Cash Ratio': cash_ratio,
            'Working Capital': working_capital
        }
    
    @staticmethod
    def dupont_roe_decomposition(net_income, revenue, avg_assets, avg_equity):
        """
        Decompose ROE into components
        """
        net_margin = net_income / revenue if revenue > 0 else 0
        asset_turnover = revenue / avg_assets if avg_assets > 0 else 0
        equity_multiplier = avg_assets / avg_equity if avg_equity > 0 else 0
        
        roe = net_margin * asset_turnover * equity_multiplier
        
        return {
            'ROE': roe,
            'Net Margin Component': net_margin,
            'Asset Turnover Component': asset_turnover,
            'Equity Multiplier (Leverage)': equity_multiplier
        }


class RatioTrendAnalyzer:
    """Analyze ratio trends over multiple years"""
    
    @staticmethod
    def trend_analysis(years, ratios, window=3):
        """
        Calculate trend in ratios over time
        """
        df = pd.DataFrame({'Year': years, 'Ratio': ratios})
        df['Trend'] = df['Ratio'].rolling(window=window, min_periods=1).mean()
        
        # Linear regression for slope (direction)
        x = np.arange(len(years))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ratios)
        
        return {
            'values': ratios,
            'trend': df['Trend'].tolist(),
            'slope': slope,
            'r_squared': r_value**2,
            'direction': 'Improving' if slope > 0 else 'Declining'
        }
    
    @staticmethod
    def quality_assessment(current_ratio, trend_slope, industry_benchmark):
        """
        Assess financial metric quality
        """
        if current_ratio > industry_benchmark * 1.2:
            quality = 'Excellent (Above Benchmark)'
        elif current_ratio > industry_benchmark * 0.9:
            quality = 'Good (At Benchmark)'
        elif current_ratio > industry_benchmark * 0.75:
            quality = 'Fair (Below Benchmark)'
        else:
            quality = 'Poor (Well Below Benchmark)'
        
        if trend_slope > 0:
            quality += ' + Improving'
        elif trend_slope < -0.01:
            quality += ' + Deteriorating'
        else:
            quality += ' + Stable'
        
        return quality


class DCFFinancialProjector:
    """Project financials for DCF using historical ratios"""
    
    @staticmethod
    def project_fcf(base_revenue, growth_rates, margin_trajectory, 
                    capex_pct_revenue, wc_change_pct_revenue, tax_rate):
        """
        Project free cash flow based on historical patterns
        
        margin_trajectory: [year1_margin, year2_margin, ...] (normalized estimates)
        """
        years = len(growth_rates)
        projected_fcf = []
        revenues = [base_revenue]
        
        for year in range(years):
            # Project revenue
            revenue_year = revenues[-1] * (1 + growth_rates[year])
            revenues.append(revenue_year)
            
            # Project NOPAT
            nopat = revenue_year * margin_trajectory[year] * (1 - tax_rate)
            
            # Project capex and working capital changes
            capex = revenue_year * capex_pct_revenue
            wc_change = (revenue_year - revenues[-2]) * wc_change_pct_revenue
            
            # FCF
            fcf = nopat - capex - wc_change
            projected_fcf.append(fcf)
        
        return {
            'revenues': revenues[1:],
            'fcf': projected_fcf,
            'avg_fcf': np.mean(projected_fcf),
            'fcf_to_revenue': np.mean(np.array(projected_fcf) / np.array(revenues[1:]))
        }


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FINANCIAL STATEMENT ANALYSIS & RATIO INTERPRETATION")
print("="*80)

# 1. Sample company financials (realistic values)
print(f"\n1. PROFITABILITY ANALYSIS")
print(f"{'-'*80}")

# Company A (Healthy)
revenue_a = 1000  # $M
cogs_a = 400
operating_income_a = 300
net_income_a = 200
avg_equity_a = 800
avg_assets_a = 1200

prof_ratios_a = FinancialStatementAnalyzer.calculate_profitability_ratios(
    revenue_a, cogs_a, operating_income_a, net_income_a, avg_equity_a, avg_assets_a
)

print(f"\nCompany A (Healthy):")
for metric, value in prof_ratios_a.items():
    print(f"  {metric}: {value*100:.1f}%")

# Company B (Deteriorating)
revenue_b = 1000
cogs_b = 450
operating_income_b = 200
net_income_b = 100
avg_equity_b = 800
avg_assets_b = 1200

prof_ratios_b = FinancialStatementAnalyzer.calculate_profitability_ratios(
    revenue_b, cogs_b, operating_income_b, net_income_b, avg_equity_b, avg_assets_b
)

print(f"\nCompany B (Deteriorating):")
for metric, value in prof_ratios_b.items():
    print(f"  {metric}: {value*100:.1f}%")

print(f"\nInterpretation:")
print(f"  - Company A: Gross margin 60%, Net margin 20% → Healthy")
print(f"  - Company B: Gross margin 55%, Net margin 10% → Margin compression")

# 2. Efficiency analysis
print(f"\n2. EFFICIENCY ANALYSIS")
print(f"{'-'*80}")

avg_receivables_a = 100  # $M (30 days)
avg_inventory_a = 80    # $M (30 days)
avg_payables_a = 60     # $M (20 days)

eff_ratios_a = FinancialStatementAnalyzer.calculate_efficiency_ratios(
    revenue_a, cogs_a, avg_assets_a, avg_receivables_a, avg_inventory_a, avg_payables_a
)

print(f"\nCompany A (Efficient):")
for metric, value in eff_ratios_a.items():
    if metric != 'Asset Turnover':
        print(f"  {metric}: {value:.1f} days" if 'Days' in metric else f"  {metric}: {value:.2f}x")
    else:
        print(f"  {metric}: {value:.2f}x")

# Company B (Less efficient)
avg_receivables_b = 120  # 43 days (collection slower)
avg_inventory_b = 150    # 55 days (inventory slower)
avg_payables_b = 50      # Payables declining (cash stress)

eff_ratios_b = FinancialStatementAnalyzer.calculate_efficiency_ratios(
    revenue_b, cogs_b, avg_assets_b, avg_receivables_b, avg_inventory_b, avg_payables_b
)

print(f"\nCompany B (Less Efficient):")
for metric, value in eff_ratios_b.items():
    if metric != 'Asset Turnover':
        print(f"  {metric}: {value:.1f} days" if 'Days' in metric else f"  {metric}: {value:.2f}x")
    else:
        print(f"  {metric}: {value:.2f}x")

print(f"\nInterpretation:")
print(f"  - Company A: 40-day cash cycle → Capital efficient")
print(f"  - Company B: 108-day cash cycle → Trapped cash, refinance risk")

# 3. Leverage analysis
print(f"\n3. LEVERAGE & SOLVENCY ANALYSIS")
print(f"{'-'*80}")

total_debt_a = 500
ebit_a = operating_income_a
interest_exp_a = 20  # Debt service
ebitda_a = operating_income_a + 50  # D&A
fcf_a = net_income_a - 100  # Capex - WC

lev_ratios_a = FinancialStatementAnalyzer.calculate_leverage_ratios(
    total_debt_a, avg_equity_a, ebit_a, interest_exp_a, ebitda_a, fcf_a
)

print(f"\nCompany A (Conservative Leverage):")
for metric, value in lev_ratios_a.items():
    if value < 100:
        print(f"  {metric}: {value:.1f}x")
    else:
        print(f"  {metric}: {value:.0f}x (Safe)")

# Company B (Higher leverage)
total_debt_b = 800
interest_exp_b = 60
ebitda_b = operating_income_b + 50
fcf_b = net_income_b - 80

lev_ratios_b = FinancialStatementAnalyzer.calculate_leverage_ratios(
    total_debt_b, avg_equity_b, ebit_b, interest_exp_b, ebitda_b, fcf_b
)

print(f"\nCompany B (Elevated Leverage):")
for metric, value in lev_ratios_b.items():
    if value < 100:
        print(f"  {metric}: {value:.1f}x")
    else:
        print(f"  {metric}: {value:.0f}x (Distressed)")

print(f"\nInterpretation:")
print(f"  - Company A: 5x coverage, 2.5x EBITDA → Safe")
print(f"  - Company B: 3.3x coverage, 4x EBITDA → Vulnerable")

# 4. DuPont Decomposition
print(f"\n4. DuPONT ROE DECOMPOSITION")
print(f"{'-'*80}")

dupont_a = FinancialStatementAnalyzer.dupont_roe_decomposition(
    net_income_a, revenue_a, avg_assets_a, avg_equity_a
)

print(f"\nCompany A:")
print(f"  ROE: {dupont_a['ROE']*100:.1f}%")
print(f"    = Net Margin ({dupont_a['Net Margin Component']*100:.1f}%)")
print(f"    × Asset Turnover ({dupont_a['Asset Turnover Component']:.2f}x)")
print(f"    × Equity Multiplier ({dupont_a['Equity Multiplier (Leverage)']:.2f}x)")

# 5. Trend analysis
print(f"\n5. RATIO TREND ANALYSIS (5-Year History)")
print(f"{'-'*80}")

years = np.array([2020, 2021, 2022, 2023, 2024])
company_a_net_margins = np.array([0.18, 0.19, 0.20, 0.21, 0.20])
company_b_net_margins = np.array([0.15, 0.14, 0.12, 0.11, 0.10])

trend_a = RatioTrendAnalyzer.trend_analysis(years, company_a_net_margins, window=2)
trend_b = RatioTrendAnalyzer.trend_analysis(years, company_b_net_margins, window=2)

print(f"\nCompany A Net Margin Trend:")
print(f"  Slope: {trend_a['slope']:.4f} (direction: {trend_a['direction']})")
print(f"  R²: {trend_a['r_squared']:.2f}")
print(f"  Quality: {RatioTrendAnalyzer.quality_assessment(0.20, trend_a['slope'], 0.18)}")

print(f"\nCompany B Net Margin Trend:")
print(f"  Slope: {trend_b['slope']:.4f} (direction: {trend_b['direction']})")
print(f"  R²: {trend_b['r_squared']:.2f}")
print(f"  Quality: {RatioTrendAnalyzer.quality_assessment(0.10, trend_b['slope'], 0.18)}")

# 6. FCF projection
print(f"\n6. DCF PROJECTION USING HISTORICAL RATIOS")
print(f"{'-'*80}")

growth_rates = np.array([0.08, 0.07, 0.06, 0.05, 0.03])
margin_trajectory_a = np.array([0.20, 0.21, 0.21, 0.20, 0.19])  # Normalized
capex_pct = 0.08
wc_change_pct = 0.03
tax_rate = 0.25

fcf_proj_a = DCFFinancialProjector.project_fcf(
    revenue_a, growth_rates, margin_trajectory_a, capex_pct, wc_change_pct, tax_rate
)

print(f"\nCompany A - 5 Year FCF Projection:")
for i, (rev, fcf) in enumerate(zip(fcf_proj_a['revenues'], fcf_proj_a['fcf']), 1):
    print(f"  Year {i}: Revenue ${rev:.0f}M, FCF ${fcf:.0f}M")

print(f"\nAverage FCF: ${fcf_proj_a['avg_fcf']:.0f}M")
print(f"FCF-to-Revenue Conversion: {fcf_proj_a['fcf_to_revenue']*100:.1f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Profitability Comparison
ax1 = axes[0, 0]

metrics = list(prof_ratios_a.keys())
company_a_values = [prof_ratios_a[m]*100 for m in metrics]
company_b_values = [prof_ratios_b[m]*100 for m in metrics]

x = np.arange(len(metrics))
width = 0.35

bars_a = ax1.bar(x - width/2, company_a_values, width, label='Company A (Healthy)',
                 color='green', alpha=0.7, edgecolor='black', linewidth=1)
bars_b = ax1.bar(x + width/2, company_b_values, width, label='Company B (Deteriorating)',
                 color='red', alpha=0.7, edgecolor='black', linewidth=1)

ax1.set_xlabel('Metric')
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Panel 1: Profitability Ratio Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' ', '\n') for m in metrics], fontsize=8)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Efficiency Metrics
ax2 = axes[0, 1]

eff_metrics = ['Receivables Days', 'Inventory Days', 'Payables Days', 'Cash Conversion Cycle']
eff_a_values = [eff_ratios_a['Receivables Days'], eff_ratios_a['Inventory Days'],
                eff_ratios_a['Payables Days'], eff_ratios_a['Cash Conversion Cycle']]
eff_b_values = [eff_ratios_b['Receivables Days'], eff_ratios_b['Inventory Days'],
                eff_ratios_b['Payables Days'], eff_ratios_b['Cash Conversion Cycle']]

x_eff = np.arange(len(eff_metrics))
bars_eff_a = ax2.bar(x_eff - width/2, eff_a_values, width, label='Company A',
                     color='green', alpha=0.7, edgecolor='black', linewidth=1)
bars_eff_b = ax2.bar(x_eff + width/2, eff_b_values, width, label='Company B',
                     color='red', alpha=0.7, edgecolor='black', linewidth=1)

ax2.set_ylabel('Days')
ax2.set_title('Panel 2: Operating Efficiency (Working Capital Days)')
ax2.set_xticks(x_eff)
ax2.set_xticklabels([m.replace(' ', '\n') for m in eff_metrics], fontsize=8)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Net Margin Trend
ax3 = axes[1, 0]

ax3.plot(years, company_a_net_margins*100, label='Company A (Stable)', 
        marker='o', linewidth=2.5, markersize=8, color='green')
ax3.plot(years, company_b_net_margins*100, label='Company B (Declining)',
        marker='s', linewidth=2.5, markersize=8, color='red')

ax3.fill_between(years, company_a_net_margins*100, alpha=0.2, color='green')
ax3.fill_between(years, company_b_net_margins*100, alpha=0.2, color='red')

ax3.set_xlabel('Year')
ax3.set_ylabel('Net Margin (%)')
ax3.set_title('Panel 3: Net Margin Trend Analysis')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Panel 4: Leverage & Coverage
ax4 = axes[1, 1]

lev_metrics = ['Debt-to-Equity', 'Debt-to-EBITDA', 'Interest Coverage']
lev_a_vals = [lev_ratios_a['Debt-to-Equity'], lev_ratios_a['Debt-to-EBITDA'],
              lev_ratios_a['Interest Coverage']]
lev_b_vals = [lev_ratios_b['Debt-to-Equity'], lev_ratios_b['Debt-to-EBITDA'],
              lev_ratios_b['Interest Coverage']]

x_lev = np.arange(len(lev_metrics))
bars_lev_a = ax4.bar(x_lev - width/2, lev_a_vals, width, label='Company A (Safe)',
                     color='green', alpha=0.7, edgecolor='black', linewidth=1)
bars_lev_b = ax4.bar(x_lev + width/2, lev_b_vals, width, label='Company B (Elevated)',
                     color='red', alpha=0.7, edgecolor='black', linewidth=1)

# Add benchmark lines for safety
ax4.axhline(y=2.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
           label='Coverage Threshold')
ax4.axhline(y=3.0, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)

ax4.set_ylabel('Ratio (x)')
ax4.set_title('Panel 4: Leverage Metrics & Coverage Ratios')
ax4.set_xticks(x_lev)
ax4.set_xticklabels(lev_metrics, fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 8)

plt.tight_layout()
plt.savefig('financial_statement_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• Margin trends > absolute margin levels (direction matters most)")
print("• Working capital efficiency directly impacts FCF vs net income")
print("• Leverage ratios must be viewed together (D/E AND coverage AND debt/EBITDA)")
print("• DuPont shows ROE drivers: margin, efficiency, and leverage contribute")
print("• FCF projections must adjust for historical working capital patterns")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Palepu, K. G., Healy, P. M., & Bernard, V. L. (2020).** "Business Analysis and Valuation: Using Financial Statements," 6th ed.
   - Comprehensive ratio framework, financial quality assessment

2. **Damodaran, A. (2012).** "Investment Valuation: Tools and Techniques for Determining Any Asset's Value," 3rd ed.
   - DCF integration with financial statement analysis, normalized metrics

3. **Simons, R. (1999).** "Levers of Control: How Managers Use Innovative Control Systems to Drive Strategic Renewal."
   - Financial metrics as control levers, linking strategy to execution

**Key Design Concepts:**

- **Trend > Level:** Margin of 15% is meaningless without trend (improving, stable, deteriorating).
- **Normalization Critical:** Use cyclic-adjusted, one-time-item-adjusted metrics for DCF input.
- **Working Capital = Hidden Value:** Small changes in days translate to massive FCF differences at scale.
- **Leverage Ratios Triangulate Risk:** Always use 3+ metrics (D/E, D/EBITDA, coverage) not just one.
- **DuPont Decomposition = Strategy Proxy:** Improvements in margin vs. efficiency signal competitive advantage type.

