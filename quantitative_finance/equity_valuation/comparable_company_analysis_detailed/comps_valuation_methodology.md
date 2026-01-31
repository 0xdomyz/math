# Comparable Company Analysis (Comps): Peer-Based Valuation Framework

## I. Concept Skeleton

**Definition:** Comparable Company Analysis (Comps) values a target firm by identifying peer companies in similar industries/markets with comparable fundamentals, then applying their valuation multiples (P/E, EV/EBITDA, Price/Sales) to the target's financials. The assumption: similar businesses should trade at similar valuation multiples; differences = growth/profitability/risk gaps.

**Purpose:** Market-based valuation (uses actual trading prices, not theoretical models), quick relative assessment (avoids complex DCF modeling), identify over/undervaluation (vs peers), triangulate with DCF, and support M&A pricing benchmarking.

**Prerequisites:** Peer firm identification, financial statement analysis, market multiples extraction, normalization techniques, risk/growth adjustment frameworks, statistical regression.

---

## II. Comparative Framing

| **Valuation Method** | **Comparable Comps** | **DCF (Intrinsic)** | **Precedent Transactions** | **Asset-Based** | **Industry Multiple** |
|-----------|----------|----------|----------|----------|-----------|
| **Data Source** | Current market prices | Company projections | Historical M&A deals | Balance sheet | Industry average |
| **Methodology** | Multiple × Metric | PV(Cash flows) | Recent deal multiples | Net asset value | Simple median |
| **Time Horizon** | Current day | 5-10 years forward | Recent past | Point-in-time | Market consensus |
| **Subjectivity** | Low (market observed) | High (projections) | Medium (outliers exist) | Low (audited) | Medium (cherry-pick peers) |
| **Sensitivity** | Peer selection | Growth/WACC assumptions | Deal specificity | Intangible assets | Sector normalization |
| **Best For** | Mature stable firms | High-growth companies | Deal benchmarking | Liquidation/breakup | Quick screening |
| **Output** | Valuation range | Point estimate | Price precedent | Residual value | Fair value band |
| **Typical Application** | IPO pricing, sell-side | M&A, VC/PE | Deal negotiations | Real estate, tangibles | Credit analysis |
| **Adjustment Factors** | Growth, leverage, margins | Terminal growth, WACC | Control premium, synergies | Obsolescence | Size, geography |

---

## III. Examples & Counterexamples

### Example 1: Comps vs DCF - The Growth Trap
**Setup:**
- Target: SaaS company, $100M revenue, 40% growth, unprofitable
- Peer 1 (Mature SaaS): $500M rev, 15% growth, 25% EBITDA margin, P/E = 30x
- Peer 2 (High-growth SaaS): $200M rev, 50% growth, 5% margin, P/E = 80x
- Task: Value the target

**Scenario A: Comps Approach (Naive - Apply Median Multiple)**

```
Step 1: Calculate peer multiples
Peer 1 (mature): P/E = 30x (lower growth)
Peer 2 (high-growth): P/E = 80x (higher growth)
Median: 55x

Step 2: Apply to target
Target EBIT: $0 (unprofitable)
Target Valuation: $0 × 55x = $0 ???
```

**Problem:** 
- Unprofitable company can't use P/E multiple
- Must normalize to comparable EBITDA or revenue

**Scenario A (Corrected): Use EV/Revenue Multiple**

```
Peer 1: EV/Revenue = 30x × 25% margin ÷ 15% growth = 50x revenue
Peer 2: EV/Revenue = 80x × 5% margin ÷ 50% growth = 8x revenue
Median: 29x revenue

Target Valuation: $100M × 29x = $2,900M
```

**Issue:** Median is distorted by divergent growth profiles.

**Better Comps Approach:**

```
Segment by growth rate:
High-growth peers (>40% growth): Peer 2 only (8x revenue)
Target (40% growth): Use 8x revenue comparable
Target Valuation: $100M × 8x = $800M
```

**Scenario B: DCF Approach (Intrinsic Value)**

```
Assumptions:
├─ Revenue CAGR (5 years): 40%
├─ Terminal growth: 3%
├─ EBITDA margin → 15% (reaches by year 5)
├─ WACC: 8%
└─ Tax rate: 25%

Projections:
Year 1-5: $100M → $100M × (1.4)^5 = $537M
Terminal EBITDA: $537M × 15% = $80.6M
Terminal value: $80.6M ÷ (8% - 3%) = $1,612M
PV(Terminal): $1,612M ÷ (1.08)^5 = $1,097M

Total DCF Valuation: ~$1,100M (average)
```

**Comparison:**
- Comps (high-growth peers): $800M
- DCF (intrinsic, growth-adjusted): $1,100M
- Gap: $300M (comps undervalues high-growth potential)

**Lesson:** Comps for high-growth firms = undervaluation risk; need DCF + sensitivity analysis.

---

### Example 2: Peer Selection Bias - Cherry-Picking Multiples
**Setup:**
- Target: Retail company, $500M revenue, 2% net margin
- Task: Find comparable retailers and apply multiples

**Scenario A: Cherry-Picked Comps (Bad Selection)**

| Company | Revenue | Growth | Margin | P/E | EV/Rev | Note |
|---------|---------|--------|--------|-----|--------|------|
| Amazon | $500B | 15% | 3% | Negative (loses money operationally) | 0.5x | Tech-enabled (not pure retail) |
| Costco | $250B | 10% | 2.8% | 50x | 1.2x | Warehouse (different model) |
| Best Buy | $50B | 2% | 2.1% | 8x | 0.3x | Electronics specialty (narrow) |

**Wrong Approach:** Pick median (0.3x rev) or cherry-pick (0.5x Costco) → $150M-$250M valuation (too low).

**Scenario B: Proper Peer Selection (Rigorous)**

Criteria for true comparables:
- Same revenue scale (±50%, not $50B vs $500B)
- Similar growth rates (±5%, not 2% vs 15%)
- Same business model (not tech-enabled vs traditional)
- Same customer base (not B2B wholesale vs B2C retail)

| Company | Revenue | Growth | Margin | P/E | EV/Rev | Reason |
|---------|---------|--------|--------|-----|--------|--------|
| Target | $500M | 2% | 2% | TBD | TBD | - |
| Peer A: Regional Retailer 1 | $600M | 2.5% | 2.2% | 12x | 0.5x | Scale match, growth match |
| Peer B: Regional Retailer 2 | $400M | 1.8% | 1.9% | 11x | 0.48x | Scale match, growth match |
| Peer C: National Retailer | $800M | 2.1% | 2.1% | 11.5x | 0.49x | Close match, similar profile |

**Proper Valuation:**

```
Median P/E: 11.5x
Target Net Income: $500M × 2% = $10M
Equity Value: $10M × 11.5x = $115M (conservative)

Median EV/Revenue: 0.49x
Target Revenue: $500M
Enterprise Value: $500M × 0.49x = $245M
Less: Net debt (assume $20M): $225M equity value
```

**Comparison:**
- Cherry-picked: $150-$250M (wrong)
- Rigorous selection: $115-$225M (reasonable range)
- Impact: Peer selection difference = ±50% valuation swing!

**Lesson:** Rigorous peer matching critical; cherry-picking peers = manipulation risk.

---

### Example 3: Market Sentiment vs Comps - The Bubble Problem
**Setup:**
- Sector: Tech startups, 2021 peak
- Target: Unprofitable fintech, -$50M EBIT, but "disruptive"
- Comps: Funded at $500M-$10B valuations despite losses

**Scenario A: Comps Valuation (Bubble Era)**

```
Recent comp valuation:
Peer 1 (similar fintech): Funded at $5B, negative EBIT
Peer 2 (adjacent fintech): Valued at $3B, negative EBIT
Implied valuation: $2-$5B (no EBITDA multiple works)

Target Valuation by Comps: $3B
Reasoning: Market sentiment + "growth narrative"
```

**Result (2021):** Target raised at $3B
**Result (2024):** Target down-round to $500M (80% haircut)

**Problem:** Comps captured inflated bubble multiples, not fundamentals.

**Scenario B: Sanity-Checked Comps + DCF**

```
Red flags:
├─ All peers unprofitable (can't use normal multiples)
├─ Peer multiples not tied to metrics (sentiment-driven)
├─ No revenue scale (all <$50M rev, high burn)
└─ Narrative-only justification (common bubble signal)

Conservative approach:
├─ Ignore pure sentiment comps
├─ Use pre-bubble comps (established fintech)
├─ Apply "unprofitability haircut" (-60% for burn rate)
└─ Validate with DCF path-to-profitability

Corrected Valuation: $300-500M (not $3B)
```

**Lesson:** Comps can be bubble-infected; need fundamental sanity check + DCF cross-validation.

---

## IV. Layer Breakdown

```
COMPARABLE COMPANY ANALYSIS FRAMEWORK

┌──────────────────────────────────────────────────┐
│       COMPARABLE COMPANY VALUATION PROCESS        │
│                                                   │
│ Core: Market multiples × Target metrics          │
│       Adjust for differences                      │
└────────────────────┬─────────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  1. PEER IDENTIFICATION & SCREENING        │
    │                                             │
    │  Step 1: Define universe                   │
    │  ├─ Industry/sector: Tech, Retail, etc.    │
    │  ├─ Geography: US, EU, Global             │
    │  ├─ Business model: B2B, B2C, marketplace  │
    │  └─ Size: Revenue scale range              │
    │                                             │
    │  Universe (initial): 200+ tech companies   │
    │                                             │
    │  Step 2: Financial metrics screening        │
    │  ├─ Profitability: Yes/No or scale        │
    │  ├─ Growth rate: 15-35% range (target)     │
    │  ├─ Margins: 10-25% EBITDA target          │
    │  ├─ Revenue scale: $100M-$5B (target)      │
    │  └─ Public trading (for multiples)         │
    │                                             │
    │  Filtered: 15-25 true comparables          │
    │                                             │
    │  Step 3: Qualitative review                 │
    │  ├─ Business model match                    │
    │  ├─ Customer base similarity                │
    │  ├─ Competitive positioning                │
    │  ├─ Growth drivers alignment                │
    │  └─ Risk profile comparability              │
    │                                             │
    │  Final Comp Set: 6-10 best matches          │
    │                                             │
    │  Red flags for exclusion:                   │
    │  ├─ Merger/spinoff announcement            │
    │  ├─ Major restructuring                     │
    │  ├─ One-time gains/charges (distort multiples)
    │  ├─ Recent M&A (contaminated financials)   │
    │  └─ Extreme leverage (distorts equity value)│
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  2. MULTIPLE EXTRACTION & NORMALIZATION     │
    │                                             │
    │  Key Multiples to Calculate:                │
    │                                             │
    │  Equity Multiples:                          │
    │  ├─ P/E (Price/Earnings)                    │
    │  │  = Market Cap / Net Income              │
    │  │  Problem: Sensitive to taxes, D&A       │
    │  │                                          │
    │  ├─ Price/Sales                             │
    │  │  = Market Cap / Revenue                 │
    │  │  Advantage: No accounting choice        │
    │  │  Problem: Ignores profitability         │
    │  │                                          │
    │  ├─ Price/Book                              │
    │  │  = Market Cap / Net Assets              │
    │  │  Good for: Asset-heavy industries       │
    │  │                                          │
    │  ├─ PEG Ratio                               │
    │  │  = P/E ÷ Growth Rate                    │
    │  │  Advantage: Growth-adjusted             │
    │  │  Use: Compare growth profiles            │
    │  │                                          │
    │  Enterprise Value Multiples (better):       │
    │  ├─ EV/EBITDA                               │
    │  │  = (Equity Value + Debt - Cash)         │
    │  │    / EBITDA (operating earnings)        │
    │  │  Advantage: Comparable across capital    │
    │  │  structures (leverage neutral)           │
    │  │                                          │
    │  ├─ EV/Revenue                              │
    │  │  Advantage: Works for unprofitable      │
    │  │  Disadvantage: Ignores costs             │
    │  │                                          │
    │  ├─ EV/EBIT                                 │
    │  │  = EV / Operating Income                │
    │  │  Middle ground: More leverage-neutral    │
    │  │  than P/E, more realistic than EV/Rev   │
    │  │                                          │
    │  Sample Extraction (3 peers):                │
    │  ┌─────────────────────────────────────┐   │
    │  │ Company    Market Cap  Revenue EBITDA│   │
    │  │ Peer A     $2,000M     $500M   $100M │   │
    │  │ Peer B     $1,500M     $400M    $75M │   │
    │  │ Peer C     $1,000M     $300M    $60M │   │
    │  │                                     │   │
    │  │ EV/Revenue Multiples:               │   │
    │  │ Peer A: $2,000M / $500M = 4.0x     │   │
    │  │ Peer B: $1,500M / $400M = 3.75x    │   │
    │  │ Peer C: $1,000M / $300M = 3.33x    │   │
    │  │                                     │   │
    │  │ EV/EBITDA Multiples:                │   │
    │  │ Peer A: $2,000M / $100M = 20x      │   │
    │  │ Peer B: $1,500M / $75M = 20x       │   │
    │  │ Peer C: $1,000M / $60M = 16.7x     │   │
    │  │                                     │   │
    │  │ Median EV/Revenue: 3.75x            │   │
    │  │ Median EV/EBITDA: 20x               │   │
    │  └─────────────────────────────────────┘   │
    │                                             │
    │  Normalization adjustments:                 │
    │  ├─ Remove one-time gains/charges           │
    │  ├─ Add back stock-based comp (SBC)         │
    │  ├─ Normalize EBITDA (normalize margin)     │
    │  ├─ Pro forma (remove discontinued ops)     │
    │  └─ Trailing 12-month (TTM) basis           │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  3. VALUATION CALCULATION                   │
    │                                             │
    │  Method 1: Apply median multiple            │
    │                                             │
    │  Target metrics:                            │
    │  ├─ Revenue: $400M                          │
    │  ├─ EBITDA: $80M                            │
    │  └─ Net Income: $32M                        │
    │                                             │
    │  Median multiples from comps:                │
    │  ├─ EV/Revenue: 3.75x                       │
    │  ├─ EV/EBITDA: 20x                          │
    │  └─ P/E: 25x                                │
    │                                             │
    │  Valuations:                                │
    │  ├─ Using Revenue: $400M × 3.75x            │
    │  │                = $1,500M                │
    │  │                                          │
    │  ├─ Using EBITDA: $80M × 20x                │
    │  │                = $1,600M                │
    │  │                                          │
    │  ├─ Using Net Income: $32M × 25x            │
    │  │                    = $800M              │
    │  │    (Note: Lower because P/E exposed to  │
    │  │     tax, D&A differences)               │
    │  │                                          │
    │  Method 2: Calculate mean & range            │
    │                                             │
    │  EV/EBITDA range across peers:              │
    │  Low: 16.7x (Peer C)                        │
    │  Median: 20x                                │
    │  High: 20x (Peers A, B)                     │
    │  Mean: 18.9x                                │
    │                                             │
    │  Valuations:                                │
    │  ├─ Conservative (16.7x): $1,336M          │
    │  ├─ Base case (18.9x): $1,512M              │
    │  ├─ Optimistic (20x): $1,600M               │
    │  └─ Range: $1,300M - $1,600M (±12%)         │
    │                                             │
    │  Method 3: Weighted average                 │
    │  (if some multiples more reliable):          │
    │                                             │
    │  40% weight EV/Revenue: $1,500M            │
    │  40% weight EV/EBITDA:  $1,600M            │
    │  20% weight P/E:         $800M             │
    │  Blended: 0.4×$1,500 + 0.4×$1,600         │
    │           + 0.2×$800 = $1,400M             │
    │                                             │
    │  Final Valuation Range: $1,300M - $1,600M   │
    │  Fair Value (midpoint): $1,450M             │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  4. ADJUSTMENT FOR DIFFERENCES              │
    │                                             │
    │  Comps methodology assumes peers = target    │
    │  In reality, differences exist → adjustments │
    │                                             │
    │  Adjustment Factor 1: Growth Rate            │
    │  ├─ Peer growth: 15% average                │
    │  ├─ Target growth: 25% (higher)             │
    │  ├─ Adjustment: +15% premium (faster growth)│
    │  │  (Use PEG ratio: multiply by growth %)   │
    │  └─ Adjusted value: $1,450M × 1.15          │
    │                   = $1,668M                 │
    │                                             │
    │  Adjustment Factor 2: Profitability          │
    │  ├─ Peer EBITDA margin: 20%                 │
    │  ├─ Target margin: 18% (slightly lower)     │
    │  ├─ Adjustment: -10% (margin compression)   │
    │  └─ Adjusted value: $1,668M × 0.90          │
    │                   = $1,501M                 │
    │                                             │
    │  Adjustment Factor 3: Size/Scale             │
    │  ├─ Peer avg revenue: $400M                 │
    │  ├─ Target revenue: $400M (same scale)      │
    │  ├─ Adjustment: 0% (no size premium)        │
    │  └─ Value stays: $1,501M                    │
    │                                             │
    │  Adjustment Factor 4: Risk/Leverage          │
    │  ├─ Peer avg leverage: 2.0x                 │
    │  ├─ Target leverage: 1.5x (lower risk)      │
    │  ├─ Adjustment: +5% (less risky)            │
    │  └─ Adjusted value: $1,501M × 1.05          │
    │                   = $1,576M                 │
    │                                             │
    │  Final Adjusted Valuation: $1,576M          │
    │  (vs unadjusted $1,450M = +9% impact)       │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  5. SENSITIVITY & SCENARIO ANALYSIS          │
    │                                             │
    │  Variable: Multiple selection               │
    │  ├─ Low scenario (16.7x EV/EBITDA)          │
    │  │  = $80M × 16.7x = $1,336M               │
    │  │                                          │
    │  ├─ Base case (18.9x)                       │
    │  │  = $80M × 18.9x = $1,512M               │
    │  │                                          │
    │  └─ High scenario (20x)                     │
    │     = $80M × 20x = $1,600M                 │
    │                                             │
    │  Variable: Growth adjustment                │
    │  ├─ Conservative growth: -10%               │
    │  │  = $1,512M × 0.90 = $1,361M             │
    │  │                                          │
    │  ├─ Base growth: 0%                         │
    │  │  = $1,512M                              │
    │  │                                          │
    │  └─ Optimistic growth: +15%                 │
    │     = $1,512M × 1.15 = $1,739M             │
    │                                             │
    │  Variable: Multiple type                    │
    │  ├─ EV/Revenue: $1,500M (broad valuation)   │
    │  ├─ EV/EBITDA: $1,600M (profitability adj)  │
    │  └─ P/E: $800M (tax-sensitive)              │
    │                                             │
    │  Valuation Range:                           │
    │  ├─ Low: $800M (P/E, conservative)          │
    │  ├─ Base: $1,500M (blend)                   │
    │  └─ High: $1,700M (growth adjusted)         │
    │                                             │
    │  Use in practice:                           │
    │  ├─ Conservative offer: Use low end         │
    │  ├─ Fair value negotiation: Use midpoint    │
    │  └─ Target asking price: Use high end       │
    └──────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Multiple Calculation

**Enterprise Value Multiple:**
$$\text{EV Multiple} = \frac{\text{Market Cap} + \text{Total Debt} - \text{Cash}}{\text{EBITDA}}$$

**Equity Multiple (P/E):**
$$\text{P/E} = \frac{\text{Market Cap}}{\text{Net Income}}$$

**PEG Ratio (Growth-adjusted):**
$$\text{PEG} = \frac{\text{P/E}}{\text{Growth Rate (\%)}}$$

Lower PEG = cheaper relative to growth.

### Valuation Using Multiple

**Target valuation:**
$$V_{\text{target}} = \text{Metric}_{\text{target}} \times \text{Multiple}_{\text{median}}$$

**Adjusted for differences:**
$$V_{\text{adjusted}} = V_{\text{base}} \times \prod_i (1 + \Delta_i)$$

where $\Delta_i$ = adjustment factors (growth, margin, leverage, etc.)

### Discount/Premium for Differences

**Growth adjustment:**
$$\text{Growth premium} = \frac{\text{Growth}_{\text{target}} - \text{Growth}_{\text{peers}}}{\text{Growth}_{\text{peers}}} \times \text{Sensitivity}$$

Typical sensitivity: 10-20% (1-2% growth difference = 1-2% value change).

---

## VI. Python Mini-Project: Comparable Company Valuation Engine

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ============================================================================
# COMPARABLE COMPANY VALUATION ENGINE
# ============================================================================

class Peer:
    """
    Represents a comparable company
    """
    
    def __init__(self, name, market_cap, revenue, ebitda, net_income, 
                 debt, cash, growth_rate):
        self.name = name
        self.market_cap = market_cap
        self.revenue = revenue
        self.ebitda = ebitda
        self.net_income = net_income
        self.debt = debt
        self.cash = cash
        self.growth_rate = growth_rate
    
    def calc_ev(self):
        """Calculate Enterprise Value"""
        return self.market_cap + self.debt - self.cash
    
    def calc_multiples(self):
        """Calculate key valuation multiples"""
        ev = self.calc_ev()
        
        return {
            'EV/Revenue': ev / self.revenue if self.revenue > 0 else None,
            'EV/EBITDA': ev / self.ebitda if self.ebitda > 0 else None,
            'P/E': self.market_cap / self.net_income if self.net_income > 0 else None,
            'Price/Sales': self.market_cap / self.revenue if self.revenue > 0 else None,
            'PEG': (self.market_cap / self.net_income) / self.growth_rate 
                   if self.net_income > 0 and self.growth_rate > 0 else None
        }


class CompsValuation:
    """
    Comparable company valuation engine
    """
    
    def __init__(self, target_name, target_revenue, target_ebitda, 
                 target_net_income, target_growth):
        self.target_name = target_name
        self.target_revenue = target_revenue
        self.target_ebitda = target_ebitda
        self.target_net_income = target_net_income
        self.target_growth = target_growth
        
        self.peers = []
        self.multiples_df = None
        self.valuations = {}
    
    def add_peer(self, peer):
        """Add comparable company"""
        self.peers.append(peer)
    
    def extract_multiples(self):
        """Extract multiples from all peers"""
        multiples_list = []
        
        for peer in self.peers:
            mults = peer.calc_multiples()
            mults['Company'] = peer.name
            mults['Growth'] = peer.growth_rate
            multiples_list.append(mults)
        
        self.multiples_df = pd.DataFrame(multiples_list)
        return self.multiples_df
    
    def calculate_valuation(self, multiple_type='EV/EBITDA', method='median'):
        """
        Calculate target valuation using specified multiple
        method: 'median', 'mean', 'lowQ1', 'highQ3'
        """
        
        if self.multiples_df is None:
            self.extract_multiples()
        
        # Get the multiple
        multiples = self.multiples_df[multiple_type].dropna()
        
        if method == 'median':
            multiple = multiples.median()
        elif method == 'mean':
            multiple = multiples.mean()
        elif method == 'lowQ1':
            multiple = multiples.quantile(0.25)
        elif method == 'highQ3':
            multiple = multiples.quantile(0.75)
        else:
            multiple = multiples.median()
        
        # Apply to target
        if 'EV/EBITDA' in multiple_type:
            valuation = self.target_ebitda * multiple
        elif 'EV/Revenue' in multiple_type:
            valuation = self.target_revenue * multiple
        elif 'P/E' in multiple_type or 'Price/Sales' in multiple_type:
            if 'Price/Sales' in multiple_type:
                valuation = self.target_revenue * multiple
            else:
                valuation = self.target_net_income * multiple
        else:
            valuation = None
        
        return {
            'multiple': multiple,
            'valuation': valuation,
            'multiple_type': multiple_type,
            'method': method
        }
    
    def growth_adjusted_valuation(self, base_valuation, peer_growth_median):
        """
        Adjust valuation for target growth vs peers
        """
        growth_diff = (self.target_growth - peer_growth_median) / peer_growth_median
        
        # PEG-style adjustment: +/-10% per percentage point growth difference
        adjustment = 1 + (growth_diff * 0.10)
        
        adjusted = base_valuation * adjustment
        
        return {
            'base': base_valuation,
            'adjusted': adjusted,
            'growth_adjustment': adjustment,
            'growth_diff_pct': growth_diff * 100
        }
    
    def valuation_range(self, multiple_type='EV/EBITDA'):
        """
        Generate valuation range (Q1, median, Q3)
        """
        
        multiples = self.multiples_df[multiple_type].dropna()
        
        low = self.calculate_valuation(multiple_type, method='lowQ1')['valuation']
        mid = self.calculate_valuation(multiple_type, method='median')['valuation']
        high = self.calculate_valuation(multiple_type, method='highQ3')['valuation']
        
        return {
            'low': low,
            'mid': mid,
            'high': high,
            'range': high - low,
            'range_pct': (high - low) / mid * 100
        }


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COMPARABLE COMPANY VALUATION ANALYSIS")
print("="*80)

# Create comparable companies
peers_data = [
    Peer('Peer A: Market Leader', 
         market_cap=2000, revenue=500, ebitda=100, net_income=50,
         debt=400, cash=100, growth_rate=0.12),
    
    Peer('Peer B: High-Growth',
         market_cap=1500, revenue=400, ebitda=75, net_income=35,
         debt=200, cash=50, growth_rate=0.20),
    
    Peer('Peer C: Mature',
         market_cap=1000, revenue=300, ebitda=60, net_income=30,
         debt=300, cash=100, growth_rate=0.08),
    
    Peer('Peer D: Emerging',
         market_cap=800, revenue=200, ebitda=40, net_income=15,
         debt=150, cash=30, growth_rate=0.25),
    
    Peer('Peer E: Stable',
         market_cap=1200, revenue=350, ebitda=70, net_income=35,
         debt=250, cash=80, growth_rate=0.10),
]

# Target company
target_revenue = 450
target_ebitda = 85
target_net_income = 40
target_growth = 0.15

# Create valuation engine
comps = CompsValuation('Target Company', target_revenue, target_ebitda,
                       target_net_income, target_growth)

for peer in peers_data:
    comps.add_peer(peer)

# Extract multiples
multiples_df = comps.extract_multiples()

print(f"\nPeer Multiples Summary:")
print(f"{'─'*80}")
print(multiples_df.to_string(index=False))

# Calculate valuations using different multiples
print(f"\n" + "="*80)
print(f"VALUATION USING DIFFERENT MULTIPLES")
print(f"="*80)

valuation_ev_rev = comps.calculate_valuation('EV/Revenue', 'median')
print(f"\n1. EV/Revenue Multiple:")
print(f"   ├─ Median multiple: {valuation_ev_rev['multiple']:.2f}x")
print(f"   ├─ Target revenue: ${target_revenue}M")
print(f"   └─ Valuation: ${valuation_ev_rev['valuation']:.0f}M")

valuation_ev_ebitda = comps.calculate_valuation('EV/EBITDA', 'median')
print(f"\n2. EV/EBITDA Multiple:")
print(f"   ├─ Median multiple: {valuation_ev_ebitda['multiple']:.2f}x")
print(f"   ├─ Target EBITDA: ${target_ebitda}M")
print(f"   └─ Valuation: ${valuation_ev_ebitda['valuation']:.0f}M")

valuation_pe = comps.calculate_valuation('P/E', 'median')
print(f"\n3. P/E Multiple:")
print(f"   ├─ Median multiple: {valuation_pe['multiple']:.1f}x")
print(f"   ├─ Target net income: ${target_net_income}M")
print(f"   └─ Valuation: ${valuation_pe['valuation']:.0f}M")

# Growth-adjusted valuation
peer_growth_median = multiples_df['Growth'].median()
print(f"\n" + "="*80)
print(f"GROWTH-ADJUSTED VALUATION")
print(f"="*80)

print(f"\nPeer growth (median): {peer_growth_median*100:.1f}%")
print(f"Target growth: {target_growth*100:.1f}%")
print(f"Growth differential: {(target_growth - peer_growth_median)*100:.1f}%")

growth_adj = comps.growth_adjusted_valuation(valuation_ev_ebitda['valuation'],
                                             peer_growth_median)

print(f"\nBase valuation (EV/EBITDA): ${growth_adj['base']:.0f}M")
print(f"Growth adjustment factor: {growth_adj['growth_adjustment']:.2f}x")
print(f"Adjusted valuation: ${growth_adj['adjusted']:.0f}M")
print(f"Adjustment impact: +${growth_adj['adjusted'] - growth_adj['base']:.0f}M")

# Valuation range
print(f"\n" + "="*80)
print(f"VALUATION RANGE ANALYSIS")
print(f"="*80)

range_ev_ebitda = comps.valuation_range('EV/EBITDA')
print(f"\nEV/EBITDA Range:")
print(f"├─ Conservative (Q1): ${range_ev_ebitda['low']:.0f}M")
print(f"├─ Base case (Median): ${range_ev_ebitda['mid']:.0f}M")
print(f"├─ Optimistic (Q3): ${range_ev_ebitda['high']:.0f}M")
print(f"└─ Range width: ${range_ev_ebitda['range']:.0f}M ({range_ev_ebitda['range_pct']:.1f}%)")

range_ev_rev = comps.valuation_range('EV/Revenue')
print(f"\nEV/Revenue Range:")
print(f"├─ Conservative (Q1): ${range_ev_rev['low']:.0f}M")
print(f"├─ Base case (Median): ${range_ev_rev['mid']:.0f}M")
print(f"├─ Optimistic (Q3): ${range_ev_rev['high']:.0f}M")
print(f"└─ Range width: ${range_ev_rev['range']:.0f}M ({range_ev_rev['range_pct']:.1f}%)")

# Blended valuation
print(f"\n" + "="*80)
print(f"BLENDED VALUATION (Multiple Methods)")
print(f"="*80)

ev_rev_val = comps.calculate_valuation('EV/Revenue', 'median')['valuation']
ev_ebitda_val = comps.calculate_valuation('EV/EBITDA', 'median')['valuation']
pe_val = comps.calculate_valuation('P/E', 'median')['valuation']

# Blend (weights: 40% EV/Revenue, 40% EV/EBITDA, 20% P/E)
blended = 0.4 * ev_rev_val + 0.4 * ev_ebitda_val + 0.2 * pe_val

print(f"\nValuation by method:")
print(f"├─ EV/Revenue (40%): ${ev_rev_val:.0f}M")
print(f"├─ EV/EBITDA (40%): ${ev_ebitda_val:.0f}M")
print(f"├─ P/E (20%): ${pe_val:.0f}M")
print(f"│")
print(f"└─ Blended valuation: ${blended:.0f}M")

# ============================================================================
# MONTE CARLO: VALUATION SENSITIVITY
# ============================================================================

print(f"\n" + "="*80)
print(f"SENSITIVITY ANALYSIS (Monte Carlo)")
print(f"="*80)

num_sims = 1000
valuations_sims = []

for sim in range(num_sims):
    # Random sampling of peer multiples
    if len(multiples_df['EV/EBITDA'].dropna()) > 0:
        multiple = np.random.choice(multiples_df['EV/EBITDA'].dropna(), 1)[0]
        
        # Random noise on target EBITDA (±10%)
        ebitda_noise = target_ebitda * np.random.normal(1.0, 0.05)
        
        val = ebitda_noise * multiple
        valuations_sims.append(val)

valuations_sims = np.array(valuations_sims)

print(f"\nValuation Distribution (1000 sims):")
print(f"├─ Mean: ${np.mean(valuations_sims):.0f}M")
print(f"├─ Median: ${np.median(valuations_sims):.0f}M")
print(f"├─ Std Dev: ${np.std(valuations_sims):.0f}M")
print(f"├─ 10th percentile: ${np.percentile(valuations_sims, 10):.0f}M")
print(f"├─ 90th percentile: ${np.percentile(valuations_sims, 90):.0f}M")
print(f"└─ 80% confidence interval: ${np.percentile(valuations_sims, 10):.0f}M - ${np.percentile(valuations_sims, 90):.0f}M")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: Peer multiples scatter
ax1 = axes[0, 0]
for idx, row in multiples_df.iterrows():
    ax1.scatter(row['Growth']*100, row['EV/EBITDA'], s=150, alpha=0.6)
    ax1.annotate(row['Company'].split(':')[1].strip(), 
                xy=(row['Growth']*100, row['EV/EBITDA']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.scatter(target_growth*100, valuation_ev_ebitda['multiple']*0.85, 
           s=300, marker='*', color='red', label='Target (approx pos)')
ax1.set_xlabel('Growth Rate (%)')
ax1.set_ylabel('EV/EBITDA Multiple')
ax1.set_title('Panel 1: Peer Growth vs EV/EBITDA\n(Target positioned among peers)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2: Valuation by method
ax2 = axes[0, 1]
methods = ['EV/Revenue', 'EV/EBITDA', 'P/E', 'Blended']
valuations = [ev_rev_val, ev_ebitda_val, pe_val, blended]
colors = ['blue', 'green', 'orange', 'red']

bars = ax2.bar(methods, valuations, color=colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, valuations):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:.0f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Valuation ($M)')
ax2.set_title('Panel 2: Valuation by Different Multiples')
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Valuation range (EV/EBITDA)
ax3 = axes[1, 0]
scenarios = ['Conservative\n(Q1)', 'Base\n(Median)', 'Optimistic\n(Q3)']
vals_range = [range_ev_ebitda['low'], range_ev_ebitda['mid'], range_ev_ebitda['high']]
colors_range = ['red', 'green', 'blue']

bars3 = ax3.bar(scenarios, vals_range, color=colors_range, alpha=0.7, edgecolor='black')
for bar, val in zip(bars3, vals_range):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:.0f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_ylabel('Valuation ($M)')
ax3.set_title('Panel 3: EV/EBITDA Valuation Range\n(Q1 to Q3)')
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Valuation distribution (MC)
ax4 = axes[1, 1]
ax4.hist(valuations_sims, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(np.mean(valuations_sims), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(valuations_sims):.0f}M')
ax4.axvline(np.median(valuations_sims), color='green', linestyle='--', linewidth=2, label=f'Median: ${np.median(valuations_sims):.0f}M')
ax4.set_xlabel('Valuation ($M)')
ax4.set_ylabel('Frequency')
ax4.set_title('Panel 4: Valuation Distribution\n(1000 MC simulations)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparable_company_valuation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• Peer selection critical: Similar growth/profitability/scale = better multiples")
print("• Multiple selection matters: EV/EBITDA > P/E for leverage-neutral comparison")
print("• Growth adjustment essential: Higher growth targets deserve premium multiples")
print("• Valuation range wide: 10-30% spread typical (use triangulation with DCF)")
print("• Blended approach robust: Combine multiple methodologies for reliability")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Damodaran, A. (2012).** "Investment Valuation: Tools and Techniques for Determining the Value of Any Asset." 3rd ed.
   - Multiples extraction; peer selection; valuation frameworks

2. **McKinsey & Company (2015).** "Valuation: Measuring and Managing the Value of Companies." 6th ed.
   - Comparable multiples; adjustment factors; practical applications

3. **Fernández, P. (2002).** "Valuation Methods and Shareholder Value Creation." Academic Press.
   - Comps vs DCF; methodology comparison; best practices

**Key Design Concepts:**

- **Multiple Extraction:** Ensure normalization (one-time items, SBC adjustments) for clean multiples.
- **Peer Matching:** Rigorous screening critical; cherry-picking peers = manipulation; must match growth/margin/scale.
- **Growth Adjustment:** PEG-style adjustments essential for high-growth targets (10-20% sensitivity per growth point).
- **Triangulation:** Comps not standalone; combine with DCF + precedent transactions for robust valuation.
- **Bubble Detection:** Monitor median multiples vs historical ranges; extreme spreads signal sentiment overvaluation.

