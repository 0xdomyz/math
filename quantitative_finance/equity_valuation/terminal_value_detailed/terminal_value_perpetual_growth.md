# Terminal Value & Perpetual Growth: Long-Term Value Drivers

## I. Concept Skeleton

**Definition:** Terminal value represents the present value of all cash flows beyond the explicit forecast period (typically years 6+). It captures long-term sustainable profitability and growth rate when company matures. Two methods: perpetuity growth model (FCF grows at constant rate forever) and exit multiple (company sold/valued at market multiple at forecast period end).

**Purpose:** Quantify value beyond 5-year forecast, determine long-term competitive positioning, test sustainability of assumptions, and identify terminal value sensitivity (often 60-80% of DCF value).

**Prerequisites:** FCF calculation, discount rate (WACC), perpetuity formulas, exit multiples (EV/EBITDA, P/E), normalized margins and growth rates, understanding of terminal competitive dynamics.

---

## II. Comparative Framing

| **Method** | **Formula** | **Assumption** | **Pros** | **Cons** |
|-----------|----------|----------|----------|----------|
| **Perpetuity Growth** | TV = FCF_terminal × (1+g) / (WACC - g) | Constant growth forever | Simple, theoretically sound | Sensitive to (WACC - g) spread |
| **Exit Multiple** | TV = Terminal EBITDA × Multiple | Sell/market at end | Market-based, conservative | Requires exit scenario forecast |
| **Two-Stage** | TV = Stable ROIC / WACC × Invested Capital | Normalize return to cost of capital | Reflects competitive equilibrium | Complex calculation |
| **Declining Growth** | TV = Fade from forecast growth to terminal | Gradual decline to equilibrium | More realistic (competition) | Calculation intensive |

---

## III. Examples & Counterexamples

### Example 1: Perpetuity Growth - The Core DCF Terminal Value

**Setup:**
- Year 5 FCF: $100M
- Terminal FCF growth: 3% annually forever
- WACC: 8%
- Question: What's terminal value at end of year 5?

**Perpetuity Calculation:**

```
Terminal Value (at end of Year 5):
├─ TV_5 = FCF_6 / (WACC - g)
├─ FCF_6 = $100M × 1.03 = $103M (first year of terminal)
├─ TV_5 = $103M / (8% - 3%) = $103M / 5% = $2,060M
│
└─ This represents the value of ALL cash flows from Year 6 onwards

Present Value of Terminal Value (at Year 0):
├─ PV(TV) = TV_5 / (1 + WACC)^5
├─ PV(TV) = $2,060M / (1.08)^5
├─ PV(TV) = $2,060M / 1.4693
└─ PV(TV) = $1,402M
```

**Terminal Value's Dominance:**

```
Complete DCF valuation:

Forecast Period (Years 1-5) FCF:
├─ Year 1: $80M, discounted: $80M / 1.08 = $74M
├─ Year 2: $87M, discounted: $87M / 1.08^2 = $75M
├─ Year 3: $94M, discounted: $94M / 1.08^3 = $75M
├─ Year 4: $100M, discounted: $100M / 1.08^4 = $74M
├─ Year 5: $100M, discounted: $100M / 1.08^5 = $68M
│
└─ PV(Forecast period) = $74 + $75 + $75 + $74 + $68 = $366M

Terminal Value:
├─ PV(TV) = $1,402M (calculated above)
│
└─ Total Enterprise Value:
   ├─ = PV(Forecast) + PV(TV)
   ├─ = $366M + $1,402M
   └─ = $1,768M
```

**Terminal Value as Percentage:**

```
Terminal Value Contribution:
├─ = $1,402M / $1,768M = 79.3%
│
└─ Interpretation:
   ├─ Terminal value represents 79% of valuation
   ├─ Small changes to terminal assumptions have HUGE impact
   ├─ Getting terminal value right is critical
   └─ Sensitivity focus: (WACC - g) spread, not just g
```

**Sensitivity to Assumptions:**

```
Impact of changing terminal assumptions:

Scenario A: g = 2.5% (vs 3% base)
├─ FCF_6 = $102.5M
├─ TV_5 = $102.5M / (8% - 2.5%) = $102.5M / 5.5% = $1,864M
├─ PV(TV) = $1,864M / 1.4693 = $1,268M
└─ Total EV = $366M + $1,268M = $1,634M (-7.6% vs base)

Scenario B: g = 3.5% (vs 3% base)
├─ FCF_6 = $103.5M
├─ TV_5 = $103.5M / (8% - 3.5%) = $103.5M / 4.5% = $2,300M
├─ PV(TV) = $2,300M / 1.4693 = $1,565M
└─ Total EV = $366M + $1,565M = $1,931M (+9.2% vs base)

Scenario C: WACC = 7% (lower risk, vs 8% base)
├─ TV_5 = $103M / (7% - 3%) = $103M / 4% = $2,575M
├─ PV(TV) = $2,575M / 1.07^5 = $2,575M / 1.4026 = $1,836M
└─ Total EV = $366M + $1,836M = $2,202M (+24.6% vs base)

Key insight:
├─ Terminal growth ±50 bps = ±10% valuation swing
├─ WACC ±100 bps = ±25% valuation swing
└─ Terminal value assumptions dominate valuation uncertainty
```

---

### Example 2: Exit Multiple Method - Comparable Exit

**Setup:**
- Forecast year 5 EBITDA: $150M
- Company will be valued at market multiples at exit
- Current market average EV/EBITDA: 12x
- Question: What's terminal value using exit multiple?

**Exit Multiple Terminal Value:**

```
Terminal Value (at end of Year 5):
├─ TV_5 = Terminal EBITDA × Exit Multiple
├─ TV_5 = $150M × 12x = $1,800M
│
└─ Represents value if company sold at market multiple

Present Value of Terminal Value:
├─ PV(TV) = $1,800M / (1.08)^5
├─ PV(TV) = $1,800M / 1.4693
└─ PV(TV) = $1,224M

Comparison to perpetuity:
├─ Perpetuity method gave TV = $1,402M
├─ Exit multiple method gave TV = $1,224M
├─ Difference: $178M (12.6% lower with exit method)
│
└─ Why lower? Exit multiple assumes competitive equilibrium
   (12x EV/EBITDA is market median, not premium)
```

**When to Use Exit Multiple:**

```
Scenarios where exit multiple is better:

1. Private equity investment (known exit timeframe)
   ├─ PE buys company, runs for 5-7 years, sells
   ├─ Exit multiple known from current market
   └─ More realistic than perpetuity assumption

2. Cyclical businesses (airlines, hotels, minerals)
   ├─ Terminal value highly dependent on exit cycle
   ├─ Company may be sold at peak/trough
   └─ Perpetuity assumes neutral cycle assumption

3. Risk of competitive disruption
   ├─ Assume company value normalizes to market average
   ├─ Don't assume perpetual competitive advantage
   └─ More conservative than perpetuity

4. Comparison to trading multiples
   ├─ If peers trade at 10x EBITDA
   ├─ Assume exited company gets similar multiple
   └─ Aligns valuation with market reality
```

**Counterexample: Exit Multiple Too Conservative**

```
Company with strong competitive advantages:

Perpetuity method:
├─ Terminal ROIC: 20% (vs WACC 8%)
├─ Sustainable premium to market average
├─ Terminal value justified at higher than 12x
└─ EV: $1,800M+ (using perpetuity)

Exit multiple method:
├─ Assume 12x market multiple (assumes no moat)
├─ Terminal value: $1,800M
├─ Undervalues competitive advantage by ~10-15%
└─ EV: $1,224M (too conservative)

Better approach:
├─ Use exit multiple at 15x (accounting for strength)
├─ TV_5 = $150M × 15x = $2,250M
├─ PV(TV) = $1,532M (between perpetuity and market)
└─ Blended approach captures both moat & reversion
```

---

### Example 3: Two-Stage Model - Explicit Fade to Terminal

**Setup:**
- Company growing 15% now (Stage 1: Years 1-5)
- Growth will fade to GDP rate (3%) by Year 10
- ROIC will decline from 18% to WACC (8%)
- Question: How to model realistic terminal value?

**Two-Stage Terminal Approach:**

```
Stage 1: High-growth period (Years 1-5)
├─ Revenue growth: 15% annually
├─ ROIC: 18% (above WACC, creating economic profit)
├─ Reinvestment rate: High (to achieve growth)
└─ Normalized EBIT margin: 25%

Transition: Explicit fade (Years 6-10)
├─ Growth gradually declines: 15% → 12% → 9% → 6% → 3%
├─ ROIC gradually declines: 18% → 15% → 12% → 10% → 8%
├─ Reinvestment rate decreases as growth normalizes
└─ Economic profit gradually erodes as competition intensifies

Stage 2: Terminal (Year 11+)
├─ Growth: 3% (GDP growth, terminal rate)
├─ ROIC: 8% (equals WACC, no economic profit)
├─ Reinvestment rate: 3% / 8% ≈ 37.5% of FCF
├─ No abnormal returns (purely perpetual at cost of capital)
│
└─ Terminal Value = FCF_11 / (WACC - g)
   where FCF_11 reflects normalized economics
```

**Financial Impact Comparison:**

```
One-stage perpetuity (assumes 15% growth continues):
├─ Terminal ROIC: 18% (unrealistic forever)
├─ Terminal value VERY high
└─ Overvalues by ignoring competitive fade

Two-stage with explicit fade:
├─ Stage 1: Premium valuations (ROIC > WACC)
├─ Stage 2: Normal valuations (ROIC = WACC)
├─ Terminal value lower than simple perpetuity
└─ More realistic valuation reflecting competition

Typical impact:
├─ One-stage perpetuity: EV $3,500M
├─ Two-stage with fade: EV $2,200M
└─ Difference: $1,300M (37% lower with realistic fade)
```

---

## IV. Layer Breakdown

```
TERMINAL VALUE & PERPETUAL GROWTH DYNAMICS

┌──────────────────────────────────────────────────┐
│  1. PERPETUITY GROWTH FUNDAMENTALS               │
│                                                  │
│  Formula: TV = FCF_terminal × (1 + g) / (WACC - g)
│                                                  │
│  Components breakdown:                           │
│  ├─ FCF_terminal: Year 5 FCF (end of forecast)  │
│  ├─ (1+g): Growth to next period (Year 6)       │
│  ├─ (WACC - g): Spread determines multiple      │
│  │  ├─ Wide spread (8% - 1%) = 7% → 14x multiple
│  │  ├─ Narrow spread (8% - 3%) = 5% → 20x       │
│  │  │   multiple                                │
│  │  └─ Critical: As g → WACC, TV → ∞            │
│  └─ PV: Discount back to present (Year 0)       │
│                                                  │
│  Perpetuity growth rate (g) constraints:        │
│  ├─ Realistic long-term (never >2-3%)          │
│  ├─ Usually anchored to GDP growth (2-3%)       │
│  ├─ Cannot exceed real wage growth               │
│  ├─ Cannot exceed real economy growth            │
│  └─ If company grows faster than economy        │
│     forever, must eventually be larger than     │
│     GDP (impossible)                            │
│                                                  │
│  Terminal WACC considerations:                   │
│  ├─ Usually assumed same as forecast WACC       │
│  ├─ But may differ in terminal (stable state)   │
│  ├─ Mature company = lower beta = lower WACC    │
│  ├─ Example: 9% WACC forecast → 7.5% terminal  │
│  └─ Impact: Lower terminal WACC = Higher TV    │
│                                                  │
│  Sensitivity to (WACC - g) spread:              │
│  ├─ Denominator is MOST sensitive input         │
│  ├─ ±25 bps g change = ±2-3 percentage point   │
│  │  change in denominator                       │
│  ├─ ±2-3 point denominator = ±40-60% TV change │
│  └─ Terminal value dominates valuation risk    │
│                                                  │
│  Terminal margin assumptions:                    │
│  ├─ Often normalize EBIT margin to industry     │
│  │  average (assumes no competitive advantage)  │
│  ├─ Example: If company has 30% terminal margin│
│  │  but peers average 15%, margin will compress │
│  ├─ Conservative: Use industry median, not peer │
│  │  high                                        │
│  ├─ Aggressive: Use company's current margin   │
│  │  (assumes moat persists forever)             │
│  └─ Reality: Margins fade toward industry normal│
│     as competition intensifies                  │
│                                                  │
│  Reinvestment in terminal period:                │
│  ├─ Terminal FCF = NOPAT - Capex - ∆WC         │
│  ├─ Terminal ROIC = NOPAT / Invested Capital   │
│  ├─ If ROIC = WACC (competitive equilibrium)   │
│  │  → Required capex = NOPAT × (g / ROIC)      │
│  ├─ Example: NOPAT $100M, ROIC = WACC = 8%,   │
│  │  g = 3%                                      │
│  │  → Capex = $100M × (3% / 8%) = $37.5M       │
│  │  → FCF = NOPAT - Capex = $62.5M             │
│  └─ Many analysts incorrectly use NOPAT as     │
│     terminal FCF (ignores capex, overstates TV) │
│                                                  │
└──────────────────┬────────────────────────────┘
                   │
    ┌──────────────▼────────────────────────────┐
    │  2. EXIT MULTIPLE METHOD                  │
    │                                            │
    │  Formula: TV = Terminal Multiple × Terminal │
    │  EBITDA (or EARNINGS)                      │
    │                                            │
    │  Exit multiple selection:                  │
    │  ├─ Use current market trading multiples   │
    │  ├─ Average of peer group (not highest)    │
    │  ├─ Example: Peer group trades 11-13x     │
    │  │  EBITDA → Use 12x                       │
    │  ├─ Industry cycle adjustment:             │
    │  │  ├─ If forecast year at trough → higher │
    │  │  │  multiple (reversion up)             │
    │  │  ├─ If forecast year at peak → lower    │
    │  │  │  multiple (reversion down)           │
    │  │  └─ Example: Hotel industry at          │
    │  │     recession → use 10x vs 14x cycle    │
    │  └─ Size adjustment:                       │
    │     ├─ Large companies: Lower multiple     │
    │     │  (more liquid, efficient markets)    │
    │     ├─ Mid-cap: Median                     │
    │     └─ Small company: Higher (illiquidity  │
    │        discount applied elsewhere)         │
    │                                            │
    │  Advantages of exit multiple:              │
    │  ├─ Market-based, not theoretical          │
    │  ├─ Conservative (avoids perpetuity        │
    │  │  infinity risk)                         │
    │  ├─ Easy to explain to board               │
    │  ├─ Aligns with market reality (IPO/sale   │
    │  │  at market multiple)                    │
    │  └─ Bounded valuation (not dependent on    │
    │     (WACC - g) spread)                     │
    │                                            │
    │  Disadvantages:                            │
    │  ├─ Ignores company competitive advantages │
    │  ├─ May be too conservative (undervalue)   │
    │  ├─ Market multiples cyclical (can be      │
    │  │  artificially high/low)                 │
    │  └─ Requires forecasting industry multiples│
    │     5-10 years in future (uncertain)      │
    │                                            │
    │  When exit multiple > perpetuity:          │
    │  ├─ Market multiples elevated (bubble)     │
    │  ├─ Perpetuity growth assumption too low  │
    │  │  (underestimate sustainable growth)     │
    │  └─ Take average of both methods for hedge │
    │                                            │
    │  When exit multiple < perpetuity:          │
    │  ├─ Perpetuity growth assumption too high  │
    │  ├─ Revert to exit multiple (more          │
    │  │  conservative)                          │
    │  ├─ OR reduce terminal growth/ROIC         │
    │  │  assumptions in perpetuity              │
    │  └─ Signal: Adjustment needed to perpetuity│
    │                                            │
    └──────────────────┬────────────────────────┘
                       │
    ┌──────────────────▼────────────────────────┐
    │  3. COMPETITIVE DYNAMICS & TERMINAL        │
    │  CONVERGENCE                               │
    │                                            │
    │  Economic moat erosion:                    │
    │  ├─ Year 1-5: Company earns premium returns│
    │  │  (ROIC > WACC, economic profit)         │
    │  ├─ Year 5-10: Competition intensifies     │
    │  │  (new entrants, substitutes, pricing    │
    │  │  pressure)                              │
    │  ├─ Year 10+: Moat eroded, convergence     │
    │  │  to market average                      │
    │  └─ Terminal: ROIC = WACC (no economic     │
    │     profit, purely competitive returns)    │
    │                                            │
    │  Sustainable growth in terminal:           │
    │  ├─ = Retention ratio × Terminal ROIC      │
    │  ├─ If terminal ROIC = WACC:               │
    │  │  ├─ Sustainable g = Retention × WACC    │
    │  │  └─ Limited by real economy growth      │
    │  ├─ If terminal ROIC > WACC (persists):    │
    │  │  ├─ Implies indefinite competitive      │
    │  │  │  advantage (rare)                    │
    │  │  ├─ Typical only for: Tech moats, brands│
    │  │  ├─ Use only if defensible (patents,    │
    │  │  │  switching costs)                    │
    │  │  └─ Reduce probability in scenarios     │
    │  └─ Declining ROIC approach:               │
    │     ├─ Project ROIC fade from Year 5→10   │
    │     ├─ Calculate NPV of explicit high-ROIC │
    │     │  years                               │
    │     ├─ Terminal value at ROIC = WACC       │
    │     └─ More realistic than perpetual moat  │
    │                                            │
    │  Terminal growth rate reality check:       │
    │  ├─ >4%: Very aggressive, requires         │
    │  │  indefinite competitive advantage       │
    │  ├─ 3-4%: At/above GDP, sustainable only  │
    │  │  with growing market share              │
    │  ├─ 2-3%: GDP range, achievable if company│
    │  │  grows with economy                     │
    │  ├─ <2%: Conservative, implies market     │
    │  │  share loss or contraction              │
    │  └─ Test: Can company grow faster than    │
    │     economy forever? (Answer should be no) │
    │                                            │
    │  Terminal margin normalization:             │
    │  ├─ Project EBIT margin fade to industry   │
    │  │  average over Years 5-10                │
    │  ├─ Avoid assuming current premium margins │
    │  │  persist forever                        │
    │  ├─ Reality: Scale and competition erode   │
    │  │  margins over time                      │
    │  └─ Example: If forecast margin 25% but    │
    │     industry 15%, fade in model            │
    │                                            │
    └────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Perpetuity Growth Terminal Value

$$TV_{\text{perpetuity}} = \frac{\text{FCF}_{\text{terminal}} \times (1 + g)}{\text{WACC} - g}$$

### Exit Multiple Terminal Value

$$TV_{\text{exit}} = \text{EBITDA}_{\text{terminal}} \times \text{Exit Multiple}$$

### Two-Stage with ROIC Fade

$$\text{Economic Profit}_t = (\text{ROIC}_t - \text{WACC}) \times \text{Invested Capital}_t$$

$$\text{Terminal EV} = \frac{\text{Invested Capital}_{\text{terminal}} \times \text{ROIC}_{\text{terminal}}}{\text{WACC}}$$

(When $\text{ROIC}_{\text{terminal}} = \text{WACC}$, terminal value equals book value of invested capital)

### Growth Sustainability Check

$$\text{Sustainable Growth} = \text{ROE} \times (1 - \text{Payout Ratio})$$

Terminal growth must not exceed sustainable growth indefinitely.

---

## VI. Python Mini-Project: Terminal Value Analyzer & Sensitivity Tester

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# TERMINAL VALUE ANALYZER
# ============================================================================

class TerminalValueCalculator:
    """Calculate terminal value using multiple methods"""
    
    @staticmethod
    def perpetuity_growth_tv(fcf_year5, wacc, terminal_growth):
        """
        Calculate terminal value using perpetuity growth method
        Returns TV at end of Year 5
        """
        if wacc <= terminal_growth:
            return np.inf
        
        tv = fcf_year5 * (1 + terminal_growth) / (wacc - terminal_growth)
        return tv
    
    @staticmethod
    def exit_multiple_tv(ebitda_year5, exit_multiple):
        """
        Calculate terminal value using exit multiple
        Returns TV at end of Year 5
        """
        tv = ebitda_year5 * exit_multiple
        return tv
    
    @staticmethod
    def two_stage_roic_fade(year5_invested_capital, year5_roic, terminal_roic, 
                           wacc, terminal_growth):
        """
        Calculate terminal value assuming ROIC fades to equilibrium
        """
        # Terminal ROIC = WACC for equilibrium
        # Terminal value = Invested capital grows with growth rate
        terminal_invested_capital = year5_invested_capital * (1 + terminal_growth)
        
        # Value at equilibrium (ROIC = WACC)
        if wacc == terminal_roic:
            # No economic profit; value = invested capital
            tv_equilibrium = terminal_invested_capital
        else:
            # With economic profit
            economic_profit = (terminal_roic - wacc) * terminal_invested_capital
            tv_equilibrium = terminal_invested_capital + (economic_profit / wacc)
        
        return tv_equilibrium
    
    @staticmethod
    def pv_terminal_value(tv_year5, wacc, forecast_years=5):
        """
        Discount terminal value back to present (Year 0)
        """
        discount_factor = (1 + wacc) ** forecast_years
        pv_tv = tv_year5 / discount_factor
        return pv_tv


class TerminalValueSensitivity:
    """Sensitivity analysis for terminal value assumptions"""
    
    @staticmethod
    def sensitivity_wacc_growth(fcf_year5, wacc_range, growth_range):
        """
        Create sensitivity matrix: WACC × Terminal Growth
        """
        matrix = []
        
        for g in growth_range:
            row = []
            for w in wacc_range:
                tv = TerminalValueCalculator.perpetuity_growth_tv(fcf_year5, w, g)
                row.append(min(tv, 1e10))  # Cap extreme values
            matrix.append(row)
        
        return np.array(matrix)
    
    @staticmethod
    def implied_wacc_from_multiple(fcf_year5, multiple, terminal_growth):
        """
        Reverse-engineer WACC given terminal value multiple
        
        Multiple = TV / FCF = (1+g) / (WACC - g)
        Solving: WACC = g + (1+g)/Multiple
        """
        implied_wacc = terminal_growth + (1 + terminal_growth) / multiple
        return implied_wacc
    
    @staticmethod
    def required_roic_for_growth(target_growth, wacc, reinvestment_rate):
        """
        Calculate required ROIC to sustain target growth
        
        Sustainable growth = ROIC × Retention ratio
        ROIC = Growth / Retention
        """
        if reinvestment_rate <= 0:
            return 0
        required_roic = target_growth / reinvestment_rate
        return required_roic


class TerminalValueValidator:
    """Sanity checks for terminal value assumptions"""
    
    @staticmethod
    def check_perpetuity_realism(wacc, terminal_growth, company_type=''):
        """
        Validate terminal growth assumption
        """
        flags = []
        
        if terminal_growth < 0:
            flags.append('⚠ Negative growth: Only for distressed scenarios')
        
        if terminal_growth > 0.04:
            flags.append('⚠ High terminal growth (>4%): Rare, requires strong moat')
        
        if terminal_growth > 0.03:
            flags.append('ℹ Terminal growth 3-4%: Above GDP, market share gains needed')
        
        if (wacc - terminal_growth) < 0.02:
            flags.append('⚠ Spread <2%: High sensitivity, small WACC change = large EV swing')
        
        if (wacc - terminal_growth) < 0.015:
            flags.append('🚨 CRITICAL: Spread <1.5%, valuation unstable')
        
        return flags
    
    @staticmethod
    def compare_methods(perpetuity_tv, exit_multiple_tv):
        """
        Compare perpetuity and exit multiple methods
        """
        difference = abs(perpetuity_tv - exit_multiple_tv)
        pct_diff = difference / min(perpetuity_tv, exit_multiple_tv) * 100
        
        assessment = {
            'perpetuity_tv': perpetuity_tv,
            'exit_tv': exit_multiple_tv,
            'difference': difference,
            'pct_difference': pct_diff
        }
        
        if pct_diff < 10:
            assessment['rating'] = 'Methods agree closely (good)'
        elif pct_diff < 25:
            assessment['rating'] = 'Moderate divergence (investigate)'
        else:
            assessment['rating'] = 'Large divergence (reconcile assumptions)'
        
        return assessment


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TERMINAL VALUE & PERPETUAL GROWTH ANALYSIS")
print("="*80)

# 1. Base case terminal value
print(f"\n1. BASE CASE - PERPETUITY GROWTH TV")
print(f"{'-'*80}")

fcf_year5 = 100  # $100M
wacc = 0.08
terminal_growth = 0.03

tv_perpetuity = TerminalValueCalculator.perpetuity_growth_tv(fcf_year5, wacc, terminal_growth)
pv_tv_perpetuity = TerminalValueCalculator.pv_terminal_value(tv_perpetuity, wacc)

print(f"\nAssumptions:")
print(f"  Year 5 FCF: ${fcf_year5}M")
print(f"  Terminal growth (g): {terminal_growth*100:.1f}%")
print(f"  WACC: {wacc*100:.1f}%")
print(f"  Spread (WACC - g): {(wacc - terminal_growth)*100:.1f}%")

print(f"\nTerminal Value Calculation:")
print(f"  TV_5 = $100M × (1.03) / (8% - 3%)")
print(f"  TV_5 = $103M / 5%")
print(f"  TV_5 = ${tv_perpetuity:.0f}M")

print(f"\nPresent Value (discounted to Year 0):")
print(f"  PV(TV) = ${tv_perpetuity:.0f}M / (1.08)^5")
print(f"  PV(TV) = ${pv_tv_perpetuity:.0f}M")

print(f"\nTerminal Value as % of Total EV (assuming $366M forecast period):")
pv_forecast = 366
total_ev = pv_forecast + pv_tv_perpetuity
tv_pct = pv_tv_perpetuity / total_ev * 100
print(f"  Total EV = ${pv_forecast}M + ${pv_tv_perpetuity:.0f}M = ${total_ev:.0f}M")
print(f"  TV % of EV: {tv_pct:.1f}%")

# 2. Exit multiple comparison
print(f"\n2. EXIT MULTIPLE METHOD")
print(f"{'-'*80}")

ebitda_year5 = 150  # $150M
exit_multiples = [10, 12, 14, 16]

print(f"\nYear 5 EBITDA: ${ebitda_year5}M")
print(f"Exit multiple sensitivity:")

for mult in exit_multiples:
    tv_exit = TerminalValueCalculator.exit_multiple_tv(ebitda_year5, mult)
    pv_tv_exit = TerminalValueCalculator.pv_terminal_value(tv_exit, wacc)
    total_ev_exit = pv_forecast + pv_tv_exit
    
    print(f"  {mult}x EBITDA: TV_5 = ${tv_exit:.0f}M, PV(TV) = ${pv_tv_exit:.0f}M, Total EV = ${total_ev_exit:.0f}M")

# 3. Perpetuity vs Exit comparison
print(f"\n3. METHOD COMPARISON: PERPETUITY vs EXIT MULTIPLE")
print(f"{'-'*80}")

tv_exit_12x = TerminalValueCalculator.exit_multiple_tv(ebitda_year5, 12)
comparison = TerminalValueValidator.compare_methods(tv_perpetuity, tv_exit_12x)

print(f"\nPerpethuity method: ${comparison['perpetuity_tv']:.0f}M")
print(f"Exit multiple (12x): ${comparison['exit_tv']:.0f}M")
print(f"Difference: ${comparison['difference']:.0f}M ({comparison['pct_difference']:.1f}%)")
print(f"Assessment: {comparison['rating']}")

# 4. Sensitivity analysis
print(f"\n4. SENSITIVITY ANALYSIS - WACC × TERMINAL GROWTH")
print(f"{'-'*80}")

wacc_range = np.array([0.06, 0.07, 0.08, 0.09, 0.10])
growth_range = np.array([0.015, 0.02, 0.025, 0.03, 0.035, 0.04])

matrix = TerminalValueSensitivity.sensitivity_wacc_growth(fcf_year5, wacc_range, growth_range)

print(f"\nTerminal Value Matrix (WACC rows × Growth columns):")
print(f"            Growth: ", end="")
for g in growth_range:
    print(f"{g*100:.1f}%   ", end="")
print()

for i, w in enumerate(wacc_range):
    print(f"WACC {w*100:.0f}%:     ", end="")
    for j in range(len(growth_range)):
        val = matrix[i, j]
        marker = " ← " if (w == wacc and growth_range[j] == terminal_growth) else ""
        print(f"${val:>5.0f}M{marker} ", end="")
    print()

# 5. Validation checks
print(f"\n5. TERMINAL VALUE REALITY CHECKS")
print(f"{'-'*80}")

validation_flags = TerminalValueValidator.check_perpetuity_realism(wacc, terminal_growth)

print(f"\nAssumption validation:")
for flag in validation_flags:
    print(f"  {flag}")

# 6. Implied metrics
print(f"\n6. IMPLIED METRICS & REVERSE ENGINEERING")
print(f"{'-'*80}")

# What WACC is implied by 20x terminal multiple?
implied_wacc = TerminalValueSensitivity.implied_wacc_from_multiple(fcf_year5, 20, terminal_growth)
print(f"\nIf terminal value multiple = 20x FCF:")
print(f"  Implied WACC = {terminal_growth*100:.1f}% + (1.03 / 20)")
print(f"  Implied WACC = {implied_wacc*100:.2f}%")
print(f"  Interpretation: 20x multiple requires very low WACC (~5.2%)")

# What ROIC is needed for sustainable 5% growth?
required_roic = TerminalValueSensitivity.required_roic_for_growth(0.05, wacc, 0.625)
print(f"\nIf sustainable growth target = 5%, reinvestment = 62.5%:")
print(f"  Required ROIC = 5% / 62.5% = {required_roic*100:.1f}%")
print(f"  Interpretation: Need 8% ROIC to sustain 5% growth (exceeds WACC)")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: EV Components
ax1 = axes[0, 0]

components = ['Forecast\nPeriod\n(Years 1-5)', 'Terminal\nValue\nComponent', 'Total\nEnterprise\nValue']
values = [pv_forecast, pv_tv_perpetuity, total_ev]
colors = ['lightblue', 'lightcoral', 'lightgreen']

bars = ax1.bar(components, values, color=colors, edgecolor='black', linewidth=1.5)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 30,
            f'${val:.0f}M', ha='center', fontweight='bold', fontsize=11)

# Add percentage labels
ax1.text(0, pv_forecast/2, f'{pv_forecast/total_ev*100:.0f}%', ha='center', fontweight='bold', fontsize=10)
ax1.text(1, pv_tv_perpetuity/2, f'{pv_tv_perpetuity/total_ev*100:.0f}%', ha='center', fontweight='bold', fontsize=10)

ax1.set_ylabel('Value ($M)')
ax1.set_title('Panel 1: DCF Components - Terminal Value Dominance')
ax1.set_ylim(0, 1800)
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Exit Multiple Impact
ax2 = axes[0, 1]

multiples = [10, 11, 12, 13, 14, 15, 16]
exit_tvs = []
pv_exit_tvs = []

for mult in multiples:
    tv = TerminalValueCalculator.exit_multiple_tv(ebitda_year5, mult)
    pv_tv = TerminalValueCalculator.pv_terminal_value(tv, wacc)
    exit_tvs.append(tv)
    pv_exit_tvs.append(pv_tv)

ax2.plot(multiples, pv_exit_tvs, linewidth=2.5, marker='o', markersize=8, color='red')
ax2.axhline(y=pv_tv_perpetuity, color='blue', linestyle='--', linewidth=2, 
           label=f'Perpetuity TV: ${pv_tv_perpetuity:.0f}M')
ax2.fill_between(multiples, 0, pv_exit_tvs, alpha=0.2, color='red')

ax2.set_xlabel('Exit Multiple (x EBITDA)')
ax2.set_ylabel('PV(TV) ($M)')
ax2.set_title('Panel 2: Terminal Value vs Exit Multiple')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel 3: Sensitivity Heatmap
ax3 = axes[1, 0]

im = ax3.imshow(matrix, cmap='RdYlGn', aspect='auto')
ax3.set_xticks(range(len(growth_range)))
ax3.set_yticks(range(len(wacc_range)))
ax3.set_xticklabels([f'{g*100:.1f}%' for g in growth_range])
ax3.set_yticklabels([f'{w*100:.0f}%' for w in wacc_range])
ax3.set_xlabel('Terminal Growth (%)')
ax3.set_ylabel('WACC (%)')
ax3.set_title('Panel 3: Terminal Value Sensitivity (WACC × Growth)')

# Add value labels
for i in range(len(wacc_range)):
    for j in range(len(growth_range)):
        text = ax3.text(j, i, f'${matrix[i, j]:.0f}',
                       ha='center', va='center', color='black', fontweight='bold', fontsize=8)

plt.colorbar(im, ax=ax3, label='TV_5 ($M)')

# Panel 4: WACC-Growth Spread Impact
ax4 = axes[1, 1]

spread_range = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
spread_tvs = []

for spread in spread_range:
    tv = fcf_year5 * 1.03 / spread
    spread_tvs.append(min(tv, 1e10))

ax4.plot(spread_range*100, spread_tvs, linewidth=2.5, marker='D', markersize=8, color='purple')
ax4.axvline(x=(wacc - terminal_growth)*100, color='red', linestyle='--', linewidth=2, 
           label=f'Base case: {(wacc - terminal_growth)*100:.0f}%')
ax4.fill_between(spread_range*100, 0, spread_tvs, alpha=0.2, color='purple')

ax4.set_xlabel('WACC - Terminal Growth Spread (%)')
ax4.set_ylabel('Terminal Value ($M)')
ax4.set_title('Panel 4: Spread Dominates Terminal Value\n(Infinite sensitivity as spread → 0)')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 5000)

plt.tight_layout()
plt.savefig('terminal_value_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• Terminal value typically 60-80% of total DCF value (dominates valuation)")
print("• (WACC - g) spread is critical: Small changes = huge EV swings")
print("• Check perpetuity vs exit multiple for consistency (should be within 20%)")
print("• Never assume company can grow faster than economy forever (test assumption)")
print("• Terminal ROIC should fade toward WACC (moat erodes with competition)")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Damodaran, A. (2012).** "Investment Valuation: Tools and Techniques for Determining Any Asset's Value," 3rd ed.
   - Terminal value framework, perpetuity sensitivity analysis

2. **Copeland, T., Koller, T., & Murrin, J. (2000).** "Valuation: Measuring and Managing the Value of Companies," 3rd ed.
   - Two-stage models, competitive dynamics in terminal value

3. **Palepu, K. G., & Healy, P. M. (2007).** "Business Analysis and Valuation," 3rd ed.
   - Terminal margin normalization, margin fade modeling

**Key Design Concepts:**

- **Terminal Value Dominates:** 60-80% of DCF typically; small assumption errors compound to massive valuation errors.
- **(WACC - g) Spread is Critical:** As spread shrinks below 2%, valuation becomes unstable and unreliable; spread <1% should trigger recheck.
- **Perpetuity Assumes Indefinite:** Implies company outpaces economy forever; only defensible for rare moats (strong brands, network effects).
- **Exit Multiple Reality-Check:** Compare perpetuity TV to exit multiple as sanity check; divergence >20% signals assumption misalignment.
- **Competitive Fade Required:** Model ROIC fade from forecast to terminal; realistic approach: ROIC → WACC as competition intensifies.

