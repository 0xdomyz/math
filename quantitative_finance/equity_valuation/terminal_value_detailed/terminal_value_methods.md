# Terminal Value Methods

## 1. Concept Skeleton
**Definition:** Estimate of company value beyond explicit forecast period in DCF; represents continuing value assuming perpetual operations  
**Purpose:** Capture majority of firm value (typically 60-80% of enterprise value) with simplified steady-state assumptions  
**Prerequisites:** Gordon Growth Model, exit multiples, competitive equilibrium economics, WACC calculation, fade/convergence concepts

## 2. Comparative Framing
| Method | Perpetuity Growth (Gordon) | Exit Multiple | H-Model (Two-Stage) | Competitive Equilibrium |
|--------|---------------------------|---------------|---------------------|------------------------|
| **Formula** | TV = FCF_{n+1}/(WACC-g) | TV = EBITDA_n × Multiple | TV = FCF_n × [(1+g_L) + H×(g_S-g_L)]/(r-g_L) | TV assumes ROIC → WACC |
| **Key Input** | Terminal growth rate (g) | Comparable EV/EBITDA | Short/long-term growth, fade period | Reinvestment rate at WACC |
| **Assumption** | Constant perpetual growth | Market multiples reasonable | Linear decline in growth | Competition eliminates excess returns |
| **Sensitivity** | Extremely high to g | Moderate to multiple selection | High to fade period (H) | Moderate (growth = reinvestment × ROIC) |
| **Best For** | Mature, stable businesses | Cross-check, relative valuation | Transition from high to low growth | Theoretical validation |
| **Drawback** | Unrealistic constant growth | Market-dependent (defeats DCF purpose) | Still sensitive to parameters | May understate franchise value |

| Terminal Growth Rate | Justification | Typical Range | When Appropriate |
|---------------------|---------------|---------------|------------------|
| **GDP Growth** | Can't outgrow economy long-term | 2-3% (developed), 4-5% (EM) | Mature diversified companies |
| **Inflation** | Nominal revenue tracking prices | 2-2.5% | Low real growth businesses |
| **Zero Growth** | Ultra-conservative | 0% | Declining industries, liquidation |
| **Industry Growth** | Sector-specific | Varies (1-6%) | Industry leaders maintaining share |
| **Risk-Free Rate** | Upper bound (can't exceed) | Rf - 0.5% to 1% | Theoretical maximum check |

## 3. Examples + Counterexamples

**Simple Example:**  
Utility company: Year 5 FCF = $100M, WACC = 8%, terminal g = 2%. TV = $100M×1.02/(0.08-0.02) = $1,700M. Reasonable: Stable regulated business, GDP-like growth.

**Perfect Fit:**  
Consumer staples (P&G): Mature markets, pricing power matches inflation, 2.5% perpetual growth. Terminal value 70% of EV. Exit multiple cross-check: 12x EBITDA consistent with perpetuity calculation.

**Failure Case:**  
Biotech startup: Project 30% growth through Year 5, then jump to 3% terminal. Unrealistic discontinuity. Missing transition period (Years 6-10) where growth should fade gradually from 30% → 3%.

**Edge Case:**  
Cyclical manufacturer at trough: Year 5 FCF = $20M (depressed). Using directly gives TV = $333M at 8% WACC, 2% g. But normalized mid-cycle FCF = $50M → TV = $833M. Must normalize before applying perpetuity formula.

**Common Mistake:**  
Terminal g = 5% with WACC = 6%. TV = FCF/(0.06-0.05) = 100× FCF. Tiny denominator creates enormous, unrealistic valuation. Terminal g must be well below WACC (typically WACC - 2% to 4%).

**Counterexample:**  
Tech company: "We'll grow at 15% forever because we're innovative." But 15% perpetual growth implies doubling every 5 years forever → eventually exceeds global GDP. Unrealistic. Must converge to GDP+ rate.

## 4. Layer Breakdown
```
Terminal Value Methodologies:

├─ Perpetuity Growth Method (Gordon Growth Model):
│   ├─ Formula Derivation:
│   │   │ PV of perpetuity with growth:
│   │   │ TV = FCF_1/(r-g) + FCF_2/(r-g)² + FCF_3/(r-g)³ + ...
│   │   │ With constant growth: FCF_t = FCF_0 × (1+g)^t
│   │   │ Sum of geometric series: TV = FCF_{n+1}/(r-g)
│   │   │ Where:
│   │   │   FCF_{n+1} = Free cash flow in first year after forecast period
│   │   │              = FCF_n × (1 + g_terminal)
│   │   │   r = Discount rate (WACC)
│   │   │   g = Terminal perpetual growth rate
│   │   └─ Constraint: g < r (otherwise TV infinite/undefined)
│   ├─ Terminal Growth Rate Selection:
│   │   ├─ GDP-Based Approach:
│   │   │   │ Long-term nominal GDP growth ≈ Real GDP + Inflation
│   │   │   │ Developed markets: 2-3% (1% real + 2% inflation)
│   │   │   │ Emerging markets: 4-5% (3% real + 2% inflation)
│   │   │   └─ Rationale: Company can't outgrow economy forever (market share caps at 100%)
│   │   ├─ Inflation-Based:
│   │   │   │ Use expected long-term inflation (2-2.5%)
│   │   │   │ Assumes zero real growth (maintenance mode)
│   │   │   └─ Conservative for mature or declining industries
│   │   ├─ Industry Growth:
│   │   │   │ Sector-specific long-term growth
│   │   │   │ Healthcare: 4-5% (aging demographics)
│   │   │   │ Utilities: 1-2% (mature, regulated)
│   │   │   │ Tech: 3-4% (despite high short-term, long-term convergence)
│   │   │   └─ Must justify why company maintains share
│   │   ├─ Risk-Free Rate Ceiling:
│   │   │   │ Terminal g should not exceed Rf (or Rf - 50bps)
│   │   │   │ If g > Rf, implies perpetual above-market returns (unlikely)
│   │   │   └─ Use as sanity check/upper bound
│   │   └─ Reinvestment-Implied Growth:
│   │       │ Sustainable growth = Reinvestment Rate × ROIC
│   │       │ Reinvestment Rate = (CapEx - D&A + ΔNWC) / NOPAT
│   │       │ Terminal period: ROIC should fade toward WACC (competitive equilibrium)
│   │       └─ Example: Reinvestment 30%, ROIC 10%, WACC 8% → g = 30%×10% = 3%
│   ├─ Last-Year Cash Flow Calculation:
│   │   ├─ Mechanical Approach:
│   │   │   FCF_{n+1} = FCF_n × (1 + g_terminal)
│   │   │   Assumes Year n is representative steady-state
│   │   ├─ Normalized Approach:
│   │   │   If Year n is cyclical peak/trough:
│   │   │   Use normalized/mid-cycle FCF instead
│   │   │   Average of Years n-2, n-1, n
│   │   │   Or adjust margins/revenues to mid-cycle
│   │   └─ Steady-State Adjustments:
│   │       Ensure Year n assumptions are sustainable:
│   │       - Margins: Competitive equilibrium (not peak)
│   │       - CapEx: Maintenance level (CapEx ≈ D&A)
│   │       - NWC: Stable (ΔNWC ≈ 0 or proportional to growth)
│   │       - ROIC: Approaching WACC (excess returns fade)
│   ├─ Sensitivity and Reasonableness:
│   │   ├─ Sensitivity to g:
│   │   │   │ TV is hyperbolic in g (small changes → large impact)
│   │   │   │ Example: WACC 10%, FCF $100M
│   │   │   │   g = 2%: TV = $102M/0.08 = $1,275M
│   │   │   │   g = 3%: TV = $103M/0.07 = $1,471M (+15%)
│   │   │   │   g = 4%: TV = $104M/0.06 = $1,733M (+36% vs g=2%)
│   │   │   └─ Always run sensitivity: g ± 0.5% and ± 1%
│   │   ├─ Implied Terminal Multiple:
│   │   │   TV / EBITDA_n or TV / NOPAT_n
│   │   │   Compare to historical multiples, peers
│   │   │   If TV/EBITDA > 20x for mature business → likely too high
│   │   └─ TV as % of Enterprise Value:
│   │       Typical: 60-75%
│   │       If >80%: Forecast period too short or terminal assumptions too optimistic
│   │       If <50%: Forecast captures most value (OK for declining businesses)
│   └─ Discount to Present Value:
│       TV occurs at end of Year n
│       PV(TV) = TV / (1 + WACC)^n
│       Add to PV of explicit forecast cash flows
│
├─ Exit Multiple Method:
│   ├─ Formula:
│   │   TV = Metric_n × Multiple
│   │   Common metrics:
│   │   - EBITDA (most common): EV/EBITDA
│   │   - EBIT: EV/EBIT
│   │   - Revenue: EV/Sales (for growth companies)
│   │   - Book Value: P/B (for financials)
│   ├─ Multiple Selection:
│   │   ├─ Comparable Companies:
│   │   │   │ Identify 5-10 public peers
│   │   │   │ Calculate current EV/EBITDA for each
│   │   │   │ Use median or average (exclude outliers)
│   │   │   │ Typical ranges:
│   │   │   │   - Mature industrials: 8-12x EBITDA
│   │   │   │   - Growth tech: 15-25x EBITDA
│   │   │   │   - Cyclicals at peak: 6-9x, trough: 12-15x (inverse)
│   │   │   └─ Adjust for differences (size, growth, margins)
│   │   ├─ Precedent Transactions:
│   │   │   M&A transaction multiples (includes control premium)
│   │   │   Typically 10-30% higher than trading multiples
│   │   │   Use if exit scenario assumes acquisition
│   │   ├─ Historical Multiples:
│   │   │   Company's own historical EV/EBITDA
│   │   │   Identify typical range (e.g., 10-14x)
│   │   │   Adjust for current market environment
│   │   └─ Perpetuity-Implied Multiple:
│   │       Cross-check: What multiple does perpetuity method imply?
│   │       EBITDA multiple ≈ (1-t)/(WACC-g) × (1 - CapEx/EBITDA)
│   │       Where t = tax rate
│   │       Example: (1-0.25)/(0.10-0.03) × (1-0.30) = 7.5x
│   ├─ Metric Selection:
│   │   ├─ EBITDA (most common):
│   │   │   │ Proxy for operating cash flow
│   │   │   │ Independent of capital structure, D&A
│   │   │   │ Comparable across companies
│   │   │   └─ Issue: Ignores CapEx needs (high CapEx industries overstated)
│   │   ├─ EBIT:
│   │   │   │ Includes D&A (better for asset-heavy)
│   │   │   │ Less common (EBITDA standard)
│   │   ├─ Revenue:
│   │   │   For early-stage, high-growth, unprofitable
│   │   │   SaaS: EV/Sales 5-15x
│   │   │   Issue: Ignores profitability entirely
│   │   └─ FCF:
│   │       Rarely used (if you have FCF, use perpetuity method)
│   │       But can use EV/FCF multiple as sanity check
│   ├─ Advantages:
│   │   │ Market-based (reflects investor sentiment)
│   │   │ Simple, intuitive
│   │   │ Avoids perpetuity growth debate
│   │   └─ Useful cross-check to perpetuity method
│   ├─ Disadvantages:
│   │   │ Introduces market dependency (defeats intrinsic DCF purpose)
│   │   │ Multiples may be artificially high/low at valuation date
│   │   │ Less theoretically sound than perpetuity
│   │   └─ Peer selection subjective (comparability issues)
│   └─ Present Value:
│       Same as perpetuity: Discount TV by (1+WACC)^n
│
├─ Two-Stage (H-Model) and Fade Models:
│   ├─ H-Model (Half-Life Decay):
│   │   │ Linear decline from high growth (g_S) to mature growth (g_L)
│   │   │ Formula: TV = FCF × [(1+g_L) + H×(g_S - g_L)] / (r - g_L)
│   │   │ H = Half of fade period (e.g., 5 years → H=5)
│   │   │ 
│   │   │ Example:
│   │   │   Year 5 FCF = $50M
│   │   │   Current growth (g_S) = 20%
│   │   │   Terminal growth (g_L) = 3%
│   │   │   Fade period = 10 years (H = 5)
│   │   │   WACC = 10%
│   │   │   TV = $50M × [(1.03) + 5×(0.20-0.03)] / (0.10-0.03)
│   │   │      = $50M × [1.03 + 0.85] / 0.07
│   │   │      = $50M × 26.86 = $1,343M
│   │   │
│   │   ├─ When to Use:
│   │   │   High-growth companies (15%+ in Year n)
│   │   │   Need transition to steady-state
│   │   │   Avoids unrealistic discontinuity
│   │   ├─ Fade Period Selection:
│   │   │   Depends on competitive dynamics
│   │   │   Strong moat: 10-15 years
│   │   │   Moderate moat: 5-10 years
│   │   │   Weak moat: 3-5 years (fade quickly)
│   │   └─ Comparison to Standard Gordon:
│   │       H-model typically gives higher TV (captures transition value)
│   │       But requires more assumptions (fade period subjective)
│   ├─ Explicit Two-Stage:
│   │   │ Instead of formula, project additional years explicitly
│   │   │ Years 1-5: Detailed forecast
│   │   │ Years 6-15: High-level projections with fading growth
│   │   │ Year 15+: Perpetuity with mature growth
│   │   │
│   │   │ Example:
│   │   │   Years 1-5: 15%, 12%, 10%, 8%, 7% growth
│   │   │   Years 6-10: 6%, 5%, 4.5%, 4%, 3.5%
│   │   │   Years 11+: 3% perpetual
│   │   │
│   │   └─ More accurate but labor-intensive
│   │       Use for high-value transactions, complex businesses
│   └─ Three-Stage Models:
│       Growth → Transition → Maturity
│       Each stage has different growth rate, duration
│       Example: 20% (3y) → Linear decline (7y) → 2.5% perpetual
│       Rarely used (complexity doesn't justify incremental accuracy)
│
├─ Competitive Equilibrium / Fade to WACC:
│   ├─ Economic Theory:
│   │   │ Long-term, competition erodes excess returns
│   │   │ ROIC converges to WACC (zero economic profit)
│   │   │ Sustainable growth = Reinvestment × WACC (not ROIC)
│   │   └─ Exception: Durable competitive advantages (moats)
│   ├─ Fade Dynamics:
│   │   │ Strong moat (Coca-Cola, Microsoft):
│   │   │   ROIC may stay 2-5% above WACC for decades
│   │   │   Slow fade (15-20 years)
│   │   │
│   │   │ Moderate moat (Regional bank):
│   │   │   ROIC fades to WACC + 1-2% in 5-10 years
│   │   │
│   │   │ Weak/No moat (Commodity producer):
│   │   │   ROIC quickly approaches WACC (3-5 years)
│   │   │   May drop below WACC in downturns
│   │   │
│   │   └─ Empirical Evidence:
│   │       Studies show median ROIC regresses toward cost of capital
│   │       Only top quartile sustains excess returns >10 years
│   ├─ Calculation Approach:
│   │   │ Terminal ROIC = WACC (or WACC + spread for moat)
│   │   │ Terminal Reinvestment Rate = g / ROIC_terminal
│   │   │ Terminal FCF = NOPAT × (1 - Reinvestment Rate)
│   │   │
│   │   │ Example:
│   │   │   Terminal g = 3%, WACC = 10%
│   │   │   Conservative: ROIC_terminal = 10% (no moat)
│   │   │   Reinvestment = 3% / 10% = 30%
│   │   │   FCF = NOPAT × (1 - 0.30) = NOPAT × 0.70
│   │   │
│   │   │   Optimistic: ROIC_terminal = 12% (moat)
│   │   │   Reinvestment = 3% / 12% = 25%
│   │   │   FCF = NOPAT × 0.75 (higher FCF conversion)
│   │   │
│   │   └─ This approach explicitly links growth, ROIC, reinvestment
│   └─ Value Creation Test:
│       If assuming ROIC > WACC in perpetuity, justify:
│       - What is the moat? (brand, network effects, scale, regulation)
│       - Is moat durable 20+ years?
│       - Historical evidence of sustained excess returns?
│       If no clear answers → assume ROIC = WACC (conservative)
│
├─ Adjustments and Special Cases:
│   ├─ Cyclical Businesses:
│   │   │ Terminal year may be peak or trough
│   │   │ Use normalized/mid-cycle metrics
│   │   │ Average EBITDA over full cycle (e.g., 7-10 years)
│   │   │ Or adjust margins to mid-cycle levels
│   │   └─ Example: Year 5 EBITDA $200M (peak, 25% margin)
│   │       Mid-cycle margin: 20%
│   │       Normalized EBITDA: Revenue × 20% = $160M
│   │       Apply perpetuity or multiple to $160M, not $200M
│   ├─ Declining Industries:
│   │   │ Negative or zero growth (g ≤ 0%)
│   │   │ Perpetuity still applies: TV = FCF / (r - g)
│   │   │ Example: g = -2%, r = 10% → TV = FCF / 0.12
│   │   │ Or use liquidation value if decline terminal
│   │   └─ Be realistic: Don't assume perpetual life if industry dying
│   ├─ Capital Structure Changes:
│   │   │ If leverage expected to change in terminal period:
│   │   │ Adjust WACC (iterate on D/E ratio)
│   │   │ Or use APV (separate unlevered value + tax shields)
│   │   └─ Example: LBO → high leverage Years 1-10, then normalize
│   │       Model debt paydown explicitly; adjust WACC over time
│   ├─ Non-Operating Assets:
│   │   │ Terminal value is for operating business only
│   │   │ Real estate, investments, cash: Add separately to EV
│   │   │ Don't include in terminal FCF calculation
│   │   └─ Exception: If generating operating income, include
│   ├─ Tax Rate Changes:
│   │   │ If corporate tax rate expected to change long-term
│   │   │ Adjust terminal NOPAT and WACC accordingly
│   │   │ Example: Tax cut from 25% to 20% → higher NOPAT, lower WACC
│   │   └─ Use best estimate of long-term statutory rate
│   └─ Currency and Inflation:
│       Nominal vs Real:
│       - If using nominal WACC → must use nominal growth rate
│       - If using real WACC → must use real growth rate
│       Mismatch causes significant error
│       Emerging markets: Consider devaluation in real terms
│
└─ Cross-Checks and Validation:
    ├─ Perpetuity vs Exit Multiple Reconciliation:
    │   │ Calculate TV both ways
    │   │ If significantly different (>20%) → investigate
    │   │ Likely causes:
    │   │   - Growth rate too high/low
    │   │   - Multiple selection poor
    │   │   - Margin/ROIC assumptions inconsistent
    │   └─ Resolution: Adjust to make consistent, or use average
    ├─ Implied ROIC Check:
    │   │ Terminal ROIC = NOPAT / Invested Capital
    │   │ Should be near WACC (within 0-3%)
    │   │ If ROIC >> WACC perpetually → overvaluing (unsustainable)
    │   └─ Exception: Strong documented competitive advantages
    ├─ Sanity Check - Market Capitalization:
    │   │ For public company: Compare DCF equity value to market cap
    │   │ If DCF 2x market cap → likely too optimistic
    │   │ If DCF 0.5x market cap → likely too pessimistic or market sees issues
    │   └─ >30% difference warrants deep dive into assumptions
    ├─ Historical Multiple Comparison:
    │   │ Implied TV/EBITDA from perpetuity method
    │   │ Compare to company's 5-year average
    │   │ If outside historical range → justify or revise
    │   └─ Sector trends: Multiples compress/expand over time
    └─ Scenario Analysis:
        Base / Bull / Bear scenarios:
        - Base: GDP-level growth (2.5%), mid-cycle margins
        - Bull: GDP+ (3.5%), high margins, strong ROIC
        - Bear: Below GDP (1.5%), compressed margins
        Range of TV gives sense of uncertainty
```

**Interaction:** Project explicit forecast (Years 1-5) → Determine terminal year metrics (FCF, EBITDA, NOPAT) → Select terminal growth rate (GDP, inflation, industry) → Apply perpetuity formula OR select exit multiple → Calculate terminal value → Discount to present → Cross-check (perpetuity vs multiple, implied ROIC, TV/EBITDA) → Sensitivity analysis (g ± 0.5%, multiple ± 1x)

## 5. Mini-Project
Comprehensive terminal value calculation with sensitivity, fade models, and cross-validation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TERMINAL VALUE METHODS AND ANALYSIS")
print("="*80)

class TerminalValueCalculator:
    """Calculate and analyze terminal value using multiple methods"""
    
    def __init__(self, company_name="Example Corp"):
        self.company_name = company_name
        self.results = {}
        
    def perpetuity_growth_method(self, fcf_terminal_year, wacc, terminal_growth_rate):
        """
        Gordon Growth Model: TV = FCF_{n+1} / (WACC - g)
        """
        if wacc <= terminal_growth_rate:
            raise ValueError(f"WACC ({wacc:.2%}) must exceed terminal growth "
                           f"({terminal_growth_rate:.2%})")
        
        # First year of perpetuity
        fcf_perpetuity = fcf_terminal_year * (1 + terminal_growth_rate)
        
        # Terminal Value
        terminal_value = fcf_perpetuity / (wacc - terminal_growth_rate)
        
        self.results['perpetuity'] = {
            'Terminal_Value': terminal_value,
            'FCF_Terminal_Year': fcf_terminal_year,
            'FCF_First_Perpetuity': fcf_perpetuity,
            'WACC': wacc,
            'Terminal_Growth': terminal_growth_rate,
            'Spread': wacc - terminal_growth_rate
        }
        
        return terminal_value
    
    def exit_multiple_method(self, ebitda_terminal_year, exit_multiple):
        """
        Exit Multiple Method: TV = EBITDA_n × Multiple
        """
        terminal_value = ebitda_terminal_year * exit_multiple
        
        self.results['exit_multiple'] = {
            'Terminal_Value': terminal_value,
            'EBITDA_Terminal_Year': ebitda_terminal_year,
            'Exit_Multiple': exit_multiple
        }
        
        return terminal_value
    
    def h_model_two_stage(self, fcf_terminal_year, wacc, 
                         high_growth_rate, terminal_growth_rate, 
                         fade_period_years):
        """
        H-Model: Linear decline from high growth to terminal growth
        TV = FCF × [(1+g_L) + H×(g_S - g_L)] / (r - g_L)
        H = fade_period / 2
        """
        h = fade_period_years / 2
        
        numerator = (1 + terminal_growth_rate) + h * (high_growth_rate - terminal_growth_rate)
        denominator = wacc - terminal_growth_rate
        
        terminal_value = fcf_terminal_year * numerator / denominator
        
        self.results['h_model'] = {
            'Terminal_Value': terminal_value,
            'FCF_Terminal_Year': fcf_terminal_year,
            'High_Growth_Rate': high_growth_rate,
            'Terminal_Growth_Rate': terminal_growth_rate,
            'Fade_Period': fade_period_years,
            'H': h,
            'WACC': wacc
        }
        
        return terminal_value
    
    def competitive_equilibrium_method(self, nopat_terminal, terminal_growth, 
                                      wacc, terminal_roic=None):
        """
        Fade to WACC: Assume ROIC converges to WACC in terminal period
        Terminal FCF = NOPAT × (1 - Reinvestment Rate)
        Reinvestment Rate = g / ROIC
        """
        if terminal_roic is None:
            terminal_roic = wacc  # Conservative: ROIC = WACC (no excess returns)
        
        # Reinvestment rate needed to achieve terminal growth
        reinvestment_rate = terminal_growth / terminal_roic
        
        # Free cash flow conversion
        fcf_terminal = nopat_terminal * (1 - reinvestment_rate)
        
        # Apply perpetuity
        fcf_perpetuity = fcf_terminal * (1 + terminal_growth)
        terminal_value = fcf_perpetuity / (wacc - terminal_growth)
        
        self.results['competitive_equilibrium'] = {
            'Terminal_Value': terminal_value,
            'NOPAT_Terminal': nopat_terminal,
            'Terminal_ROIC': terminal_roic,
            'Reinvestment_Rate': reinvestment_rate,
            'FCF_Terminal': fcf_terminal,
            'FCF_Conversion': 1 - reinvestment_rate,
            'Terminal_Growth': terminal_growth,
            'WACC': wacc,
            'Economic_Profit': 'Positive' if terminal_roic > wacc else 'Zero' if terminal_roic == wacc else 'Negative'
        }
        
        return terminal_value
    
    def normalized_cyclical_adjustment(self, ebitda_years, margins_years=None):
        """
        Normalize cyclical business metrics to mid-cycle
        """
        # Average EBITDA over cycle
        normalized_ebitda = np.mean(ebitda_years)
        
        # If margins provided, calculate normalized margin
        if margins_years is not None:
            normalized_margin = np.mean(margins_years)
            return normalized_ebitda, normalized_margin
        
        return normalized_ebitda
    
    def sensitivity_analysis(self, fcf_terminal, wacc_base, growth_base,
                           wacc_range=None, growth_range=None):
        """
        Sensitivity of TV to WACC and terminal growth rate
        """
        if wacc_range is None:
            wacc_range = np.linspace(wacc_base - 0.02, wacc_base + 0.02, 9)
        if growth_range is None:
            growth_range = np.linspace(growth_base - 0.01, growth_base + 0.01, 9)
        
        # One-way sensitivities
        tv_wacc = []
        for w in wacc_range:
            if w > growth_base:
                tv = fcf_terminal * (1 + growth_base) / (w - growth_base)
                tv_wacc.append(tv)
            else:
                tv_wacc.append(np.nan)
        
        tv_growth = []
        for g in growth_range:
            if wacc_base > g:
                tv = fcf_terminal * (1 + g) / (wacc_base - g)
                tv_growth.append(tv)
            else:
                tv_growth.append(np.nan)
        
        # Two-way sensitivity table
        tv_2d = np.zeros((len(wacc_range), len(growth_range)))
        for i, w in enumerate(wacc_range):
            for j, g in enumerate(growth_range):
                if w > g:
                    tv_2d[i, j] = fcf_terminal * (1 + g) / (w - g)
                else:
                    tv_2d[i, j] = np.nan
        
        return {
            'wacc_range': wacc_range,
            'growth_range': growth_range,
            'tv_wacc_sensitivity': tv_wacc,
            'tv_growth_sensitivity': tv_growth,
            'tv_2d_table': tv_2d
        }
    
    def implied_multiple_from_perpetuity(self, terminal_value, ebitda_terminal,
                                         ebit_terminal=None, nopat_terminal=None):
        """
        Calculate implied multiples from perpetuity-derived TV
        """
        multiples = {}
        
        multiples['EV_EBITDA'] = terminal_value / ebitda_terminal
        
        if ebit_terminal is not None:
            multiples['EV_EBIT'] = terminal_value / ebit_terminal
        
        if nopat_terminal is not None:
            multiples['EV_NOPAT'] = terminal_value / nopat_terminal
        
        return multiples
    
    def reconcile_methods(self, perpetuity_tv, exit_multiple_tv):
        """
        Compare perpetuity and exit multiple terminal values
        """
        difference = perpetuity_tv - exit_multiple_tv
        pct_difference = (difference / perpetuity_tv) * 100
        
        reconciliation = {
            'Perpetuity_TV': perpetuity_tv,
            'Exit_Multiple_TV': exit_multiple_tv,
            'Difference': difference,
            'Percent_Difference': pct_difference,
            'Assessment': 'Reasonable' if abs(pct_difference) < 15 else 'Investigate'
        }
        
        return reconciliation

# Example 1: Base Case Perpetuity Method
print("\n" + "="*80)
print("EXAMPLE 1: PERPETUITY GROWTH METHOD (BASE CASE)")
print("="*80)

calc = TerminalValueCalculator("Stable Manufacturing Co.")

# Inputs
fcf_year5 = 100  # $100M in Year 5
wacc = 0.10      # 10% WACC
term_growth = 0.025  # 2.5% perpetual growth

tv_perpetuity = calc.perpetuity_growth_method(fcf_year5, wacc, term_growth)

print(f"\nInputs:")
print(f"  FCF (Year 5): ${fcf_year5:.1f}M")
print(f"  WACC: {wacc:.1%}")
print(f"  Terminal Growth Rate: {term_growth:.1%}")

print(f"\nCalculation:")
for key, val in calc.results['perpetuity'].items():
    if isinstance(val, (int, float)):
        if 'FCF' in key or 'Terminal_Value' in key:
            print(f"  {key}: ${val:.1f}M")
        else:
            print(f"  {key}: {val:.2%}")

# Example 2: Exit Multiple Method
print("\n" + "="*80)
print("EXAMPLE 2: EXIT MULTIPLE METHOD")
print("="*80)

ebitda_year5 = 150  # $150M EBITDA in Year 5
exit_multiple = 10.0  # 10x EV/EBITDA

tv_exit = calc.exit_multiple_method(ebitda_year5, exit_multiple)

print(f"\nInputs:")
print(f"  EBITDA (Year 5): ${ebitda_year5:.1f}M")
print(f"  Exit Multiple (EV/EBITDA): {exit_multiple:.1f}x")

print(f"\nTerminal Value (Exit Multiple): ${tv_exit:,.1f}M")

# Reconcile the two methods
print("\n" + "="*80)
print("RECONCILIATION: Perpetuity vs Exit Multiple")
print("="*80)

reconciliation = calc.reconcile_methods(tv_perpetuity, tv_exit)
for key, val in reconciliation.items():
    if isinstance(val, (int, float)):
        if 'TV' in key or 'Difference' in key:
            print(f"  {key}: ${val:,.1f}M")
        else:
            print(f"  {key}: {val:.1f}%")
    else:
        print(f"  {key}: {val}")

# Implied multiple from perpetuity
implied_mult = calc.implied_multiple_from_perpetuity(tv_perpetuity, ebitda_year5)
print(f"\nImplied EV/EBITDA from Perpetuity Method: {implied_mult['EV_EBITDA']:.1f}x")
print(f"Exit Multiple Used: {exit_multiple:.1f}x")
print(f"Difference: {implied_mult['EV_EBITDA'] - exit_multiple:.1f}x")

# Example 3: H-Model (Two-Stage Growth)
print("\n" + "="*80)
print("EXAMPLE 3: H-MODEL (TWO-STAGE WITH LINEAR FADE)")
print("="*80)

fcf_year5_growth = 80  # $80M (still growing fast)
high_growth = 0.15     # Currently growing 15%
term_growth_hmodel = 0.03  # Will fade to 3%
fade_period = 10       # Over 10 years

tv_hmodel = calc.h_model_two_stage(fcf_year5_growth, wacc, 
                                   high_growth, term_growth_hmodel, 
                                   fade_period)

print(f"\nInputs:")
print(f"  FCF (Year 5): ${fcf_year5_growth:.1f}M")
print(f"  High Growth Rate: {high_growth:.1%}")
print(f"  Terminal Growth Rate: {term_growth_hmodel:.1%}")
print(f"  Fade Period: {fade_period} years")
print(f"  WACC: {wacc:.1%}")

print(f"\nResults:")
print(f"  H (half-life): {calc.results['h_model']['H']:.1f} years")
print(f"  Terminal Value (H-Model): ${tv_hmodel:,.1f}M")

# Compare to standard perpetuity (no fade)
tv_standard = calc.perpetuity_growth_method(fcf_year5_growth, wacc, term_growth_hmodel)
print(f"  Terminal Value (Standard Perpetuity at 3%): ${tv_standard:,.1f}M")
print(f"  Premium from H-Model: ${tv_hmodel - tv_standard:,.1f}M ({(tv_hmodel/tv_standard - 1)*100:.1f}%)")

# Example 4: Competitive Equilibrium (Fade to WACC)
print("\n" + "="*80)
print("EXAMPLE 4: COMPETITIVE EQUILIBRIUM (ROIC → WACC)")
print("="*80)

nopat_year5 = 120  # $120M NOPAT
term_growth_comp = 0.025

# Scenario 1: No moat (ROIC = WACC)
tv_no_moat = calc.competitive_equilibrium_method(nopat_year5, term_growth_comp, 
                                                 wacc, terminal_roic=wacc)
print(f"\nScenario 1: No Moat (ROIC = WACC = {wacc:.1%})")
for key, val in calc.results['competitive_equilibrium'].items():
    if isinstance(val, (int, float)):
        if 'Terminal_Value' in key or 'NOPAT' in key or 'FCF' in key:
            print(f"  {key}: ${val:.1f}M")
        else:
            print(f"  {key}: {val:.2%}")
    else:
        print(f"  {key}: {val}")

# Scenario 2: Moderate moat (ROIC = WACC + 2%)
calc_moat = TerminalValueCalculator("Moat Company")
tv_moat = calc_moat.competitive_equilibrium_method(nopat_year5, term_growth_comp,
                                                   wacc, terminal_roic=wacc + 0.02)
print(f"\nScenario 2: Moderate Moat (ROIC = {wacc + 0.02:.1%})")
print(f"  Reinvestment Rate: {calc_moat.results['competitive_equilibrium']['Reinvestment_Rate']:.2%}")
print(f"  FCF Conversion: {calc_moat.results['competitive_equilibrium']['FCF_Conversion']:.2%}")
print(f"  Terminal Value: ${tv_moat:,.1f}M")
print(f"  Premium vs No Moat: ${tv_moat - tv_no_moat:,.1f}M ({(tv_moat/tv_no_moat - 1)*100:.1f}%)")

# Example 5: Cyclical Normalization
print("\n" + "="*80)
print("EXAMPLE 5: CYCLICAL BUSINESS NORMALIZATION")
print("="*80)

# Historical EBITDA over cycle
ebitda_history = np.array([140, 160, 180, 200, 160, 120, 140])  # 7 years
print(f"\nHistorical EBITDA (7 years): {ebitda_history}")
print(f"  Year 7 (Current): ${ebitda_history[-1]:.0f}M")
print(f"  Peak: ${ebitda_history.max():.0f}M")
print(f"  Trough: ${ebitda_history.min():.0f}M")

normalized_ebitda = calc.normalized_cyclical_adjustment(ebitda_history)
print(f"  Normalized (Mid-Cycle): ${normalized_ebitda:.1f}M")

# Terminal value using current vs normalized
exit_mult = 9.0
tv_current = ebitda_history[-1] * exit_mult
tv_normalized = normalized_ebitda * exit_mult

print(f"\nTerminal Value Comparison (using {exit_mult:.0f}x multiple):")
print(f"  Using Current EBITDA: ${tv_current:,.1f}M")
print(f"  Using Normalized EBITDA: ${tv_normalized:,.1f}M")
print(f"  Difference: ${tv_normalized - tv_current:,.1f}M ({(tv_normalized/tv_current - 1)*100:+.1f}%)")

# Example 6: Sensitivity Analysis
print("\n" + "="*80)
print("EXAMPLE 6: SENSITIVITY ANALYSIS")
print("="*80)

fcf_base = 100
wacc_base = 0.10
growth_base = 0.025

sensitivity = calc.sensitivity_analysis(fcf_base, wacc_base, growth_base)

print(f"\nBase Case:")
print(f"  FCF: ${fcf_base:.0f}M")
print(f"  WACC: {wacc_base:.1%}")
print(f"  Growth: {growth_base:.1%}")
base_tv = fcf_base * (1 + growth_base) / (wacc_base - growth_base)
print(f"  Terminal Value: ${base_tv:,.1f}M")

print(f"\nWACC Sensitivity (holding growth at {growth_base:.1%}):")
for w, tv in zip(sensitivity['wacc_range'], sensitivity['tv_wacc_sensitivity']):
    if not np.isnan(tv):
        change = (tv / base_tv - 1) * 100
        print(f"  WACC {w:.1%}: TV ${tv:,.0f}M ({change:+.1f}%)")

print(f"\nGrowth Rate Sensitivity (holding WACC at {wacc_base:.1%}):")
for g, tv in zip(sensitivity['growth_range'], sensitivity['tv_growth_sensitivity']):
    if not np.isnan(tv):
        change = (tv / base_tv - 1) * 100
        print(f"  Growth {g:.1%}: TV ${tv:,.0f}M ({change:+.1f}%)")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Method Comparison
ax = axes[0, 0]
methods = ['Perpetuity\n(2.5% g)', 'Exit\nMultiple', 'H-Model\n(15%→3%)', 'No Moat\n(ROIC=WACC)']
values = [tv_perpetuity, tv_exit, tv_hmodel, tv_no_moat]
colors = ['steelblue', 'coral', 'lightgreen', 'gold']
bars = ax.bar(methods, values, color=colors, alpha=0.7)
ax.set_ylabel('Terminal Value ($M)')
ax.set_title('Terminal Value Methods Comparison')
ax.axhline(base_tv, color='red', linestyle='--', linewidth=2, alpha=0.6)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 30, f'${val:.0f}M',
            ha='center', fontweight='bold', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Plot 2: Perpetuity Formula Sensitivity to Spread
ax = axes[0, 1]
spreads = np.linspace(0.03, 0.12, 50)  # WACC - g from 3% to 12%
tv_spreads = fcf_base * (1 + growth_base) / spreads
ax.plot(spreads * 100, tv_spreads, linewidth=2.5, color='darkred')
ax.axvline((wacc_base - growth_base) * 100, color='green', linestyle='--',
           linewidth=2, label=f'Base: {(wacc_base - growth_base)*100:.1f}%')
ax.fill_between(spreads * 100, 0, tv_spreads, alpha=0.2, color='red')
ax.set_xlabel('Spread: WACC - Growth (%)')
ax.set_ylabel('Terminal Value ($M)')
ax.set_title('TV Sensitivity to (WACC - g) Spread')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 3000)

# Plot 3: WACC Sensitivity
ax = axes[0, 2]
ax.plot(sensitivity['wacc_range'] * 100, sensitivity['tv_wacc_sensitivity'],
        'o-', linewidth=2, markersize=8, color='darkblue')
ax.axhline(base_tv, color='green', linestyle='--', label=f'Base: ${base_tv:.0f}M')
ax.axvline(wacc_base * 100, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel('WACC (%)')
ax.set_ylabel('Terminal Value ($M)')
ax.set_title('TV Sensitivity to WACC')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Growth Rate Sensitivity
ax = axes[1, 0]
ax.plot(sensitivity['growth_range'] * 100, sensitivity['tv_growth_sensitivity'],
        'o-', linewidth=2, markersize=8, color='darkgreen')
ax.axhline(base_tv, color='green', linestyle='--', label=f'Base: ${base_tv:.0f}M')
ax.axvline(growth_base * 100, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel('Terminal Growth Rate (%)')
ax.set_ylabel('Terminal Value ($M)')
ax.set_title('TV Sensitivity to Growth Rate')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: H-Model Growth Trajectory
ax = axes[1, 1]
years = np.arange(0, fade_period + 1)
growth_trajectory = np.linspace(high_growth, term_growth_hmodel, len(years))
ax.plot(years, growth_trajectory * 100, 'o-', linewidth=2.5, markersize=8, color='purple')
ax.axhline(term_growth_hmodel * 100, color='red', linestyle='--',
           label=f'Terminal: {term_growth_hmodel:.1%}')
ax.fill_between(years, growth_trajectory * 100, term_growth_hmodel * 100, alpha=0.3)
ax.set_xlabel('Years into Fade Period')
ax.set_ylabel('Growth Rate (%)')
ax.set_title('H-Model: Linear Growth Fade')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Cyclical EBITDA Normalization
ax = axes[1, 2]
years_hist = np.arange(1, len(ebitda_history) + 1)
ax.plot(years_hist, ebitda_history, 'o-', linewidth=2, markersize=8,
        label='Actual EBITDA', color='blue')
ax.axhline(normalized_ebitda, color='green', linestyle='--', linewidth=2.5,
           label=f'Normalized: ${normalized_ebitda:.0f}M')
ax.fill_between(years_hist, ebitda_history, normalized_ebitda, alpha=0.2)
ax.set_xlabel('Year')
ax.set_ylabel('EBITDA ($M)')
ax.set_title('Cyclical Business: EBITDA Normalization')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Two-Way Sensitivity Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
wacc_labels = [f"{w:.1%}" for w in sensitivity['wacc_range']]
growth_labels = [f"{g:.1%}" for g in sensitivity['growth_range']]
sns.heatmap(sensitivity['tv_2d_table'], annot=True, fmt='.0f',
            cmap='RdYlGn', center=base_tv, cbar_kws={'label': 'Terminal Value ($M)'},
            xticklabels=growth_labels, yticklabels=wacc_labels, ax=ax)
ax.set_xlabel('Terminal Growth Rate')
ax.set_ylabel('WACC')
ax.set_title('Two-Way Sensitivity: Terminal Value Heatmap')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. Perpetuity method: ${tv_perpetuity:,.0f}M vs Exit multiple: ${tv_exit:,.0f}M ({reconciliation['Assessment']})")
print(f"2. H-Model adds {(tv_hmodel/tv_standard - 1)*100:.1f}% premium for capturing transition growth")
print(f"3. WACC ±1% changes TV by ±{((sensitivity['tv_wacc_sensitivity'][0]/base_tv - 1)*100):.0f}% (extreme sensitivity)")
print(f"4. Growth ±0.5% changes TV by ±{((sensitivity['tv_growth_sensitivity'][-1]/base_tv - 1)*100):.0f}%")
print(f"5. Moat value: ROIC 2% above WACC → {(tv_moat/tv_no_moat - 1)*100:.1f}% higher TV")
print(f"6. Cyclical normalization critical: ${tv_normalized - tv_current:,.0f}M difference using trough vs mid-cycle")
print(f"7. Terminal value hypersensitive to (WACC - g) spread: Narrow spread → exploding valuations")
```

## 6. Challenge Round
Advanced terminal value problems and extensions:

1. **Optimal Fade Period:** Company currently at 18% ROIC (WACC 10%). Using H-model, determine optimal fade period that maximizes NPV while staying within competitive reality bounds. How does industry (tech vs utility) affect optimal fade?

2. **Implied Terminal Growth Reconciliation:** Public company trades at $50/share. Your DCF with 2.5% terminal growth gives $45/share. What terminal growth rate would reconcile to market price? Use goal-seek/solver. Is that growth rate defensible?

3. **Negative Growth Industries:** Newspaper publisher expects -5% perpetual revenue decline. Model terminal value with declining growth. How do you ensure FCF remains positive? What happens when FCF approaches zero?

4. **Variable WACC Over Time:** Company deleveraging (D/E from 2.0 to 0.5 over 10 years). WACC changes from 12% to 9%. Model terminal value with time-varying discount rate. Compare to constant WACC assumption.

5. **Real Options in Terminal Value:** Mining company: Terminal value includes option to expand if commodity prices rise. Use Black-Scholes to value expansion option. How much does optionality add to perpetuity-based TV?

6. **Monte Carlo Terminal Value:** Terminal growth ~ N(2.5%, 1%), WACC ~ N(10%, 1.5%), correlated (ρ=-0.3: high growth → high risk → high WACC). Simulate 10,000 scenarios. What is median TV? 90th percentile? How does this change investment decision vs point estimate?

7. **Market-Implied Terminal Growth:** For a public company, back-solve current market cap to implied terminal growth (fixing all other DCF assumptions). Compare to historical growth, analyst estimates. If market implies 8% but fundamentals suggest 3%, what explains the gap?

## 7. Key References
- [Damodaran, "The Dark Side of Valuation" (2nd Edition, 2009)](http://pages.stern.nyu.edu/~adamodar/) - terminal value estimation pitfalls, growth rate selection, competitive equilibrium concepts
- [Fuller & Hsia, "A Simplified Common Stock Valuation Model" (Financial Analysts Journal, 1984)](https://www.tandfonline.com/doi/abs/10.2469/faj.v40.n5.40) - H-model derivation and applications
- [Koller, Goedhart & Wessels, "Valuation Workbook" (Wiley, 6th Edition, 2015)](https://www.wiley.com/en-us/Valuation%3A+Measuring+and+Managing+the+Value+of+Companies%2C+Workbook%2C+6th+Edition-p-9781118874073) - practical terminal value exercises with industry benchmarks

---
**Status:** Critical DCF component (60-80% of enterprise value) | **Complements:** DCF Valuation, WACC Calculation, Growth Rate Estimation, Competitive Analysis, Comparable Company Multiples
