# Risk-Free Rate: Nature, Measurement & Application

## 1. Concept Skeleton
**Definition:** Risk-free rate (rf): theoretical interest rate on investment with zero default and reinvestment risk; proxy = U.S. Treasury bill/bond rates. In practice, no truly "risk-free" asset (inflation, real rate, reinvestment risks exist), but T-bills closest approximation due to U.S. government backing and high liquidity.  
**Purpose:** Baseline discount rate for all asset pricing models (CAPM, DCF); foundation for capital allocation line (CAL); separates risk premium from time value of money  
**Prerequisites:** Bond pricing, yield curves, discount rates, inflation concepts

---

## 2. Comparative Framing

| Aspect | Theoretical Risk-Free | U.S. Treasury Bill (Short-term) | U.S. Treasury Bond (Long-term) | Corporate Bond | Real Interest Rate |
|--------|---------------------|-----------------------------|-----------------------------|-----------------|-----------------|
| **Default Risk** | 0% (assumed) | ~0% (backed by U.S. gov) | ~0% (backed by U.S. gov) | Low-High (depends on credit) | N/A (inflation-adjusted) |
| **Liquidity Risk** | None | Minimal (highly liquid) | Minimal (highly liquid) | Low-Moderate | N/A |
| **Reinvestment Risk** | None | Moderate (short-term) | High (long-term) | High | N/A |
| **Inflation Risk** | None | YES (embedded in yield) | YES (embedded in yield) | YES | No (inflation-adjusted) |
| **Term (Maturity)** | Arbitrary | 3-month, 6-month, 1-year | 2, 5, 10, 30-year | Various | Derived from yield curve |
| **Typical Yield** | Theoretical baseline | 4-5% (varies with cycle) | 4-6% (varies with term) | 5-10% (spreads vary) | 1-3% (long-run average) |
| **Practical Use in CAPM** | Common baseline | Best approximation (1Y or 2Y) | Often used for long-term returns | Not for rf (uses beta-adjusted premium) | For real returns analysis |
| **Volatility** | σ = 0 (no uncertainty) | Low (minimal price movement) | Moderate (duration risk) | Moderate-High (credit spreads) | Derived from nominal - inflation |
| **Market Risk (Beta)** | β = 0 (uncorrelated with market) | β ≈ 0 (minimal) | β ≈ 0.1-0.3 (low; some duration sensitivity) | β ≈ 0.3-0.8 (depends on credit quality) | β = 0 (not a market asset) |
| **Economic Sensitivity** | Inert (no reaction to economy) | Moderate (Fed policy-driven) | High (yields fall in recession) | Very high (spreads widen in downturn) | Low (structural rate) |

**Key Insight:** No perfect risk-free asset exists; practitioners choose rf proxy based on investment horizon and model use case. Short-horizon portfolios use T-bills; long-horizon use 10Y Treasury; each has embedded inflation expectations, term premiums, and some reinvestment risk.

---

## 3. Examples + Counterexamples

**Example 1: Risk-Free Rate Selection by Investment Horizon**

Portfolio manager building 5-year retirement plan:
- **Short-term (1-year) rf:** 4.5% (current T-bill yield)
  - Pros: Minimal reinvestment uncertainty within year
  - Cons: Doesn't align with 5-year horizon; reinvestment risk for years 2-5
  - Use case: Hedging short-term volatility
  
- **Medium-term (5-year) rf:** 4.8% (current 5Y Treasury yield)
  - Pros: Matches portfolio horizon; no reinvestment needed before goal date
  - Cons: Includes term premium (~0.3%) and inflation expectations
  - Use case: Optimal CAL for 5-year investors
  
- **Long-term (10-year) rf:** 4.9% (current 10Y Treasury yield)
  - Pros: Liquid, long maturity
  - Cons: Too long for 5-year plan; reinvestment inflexibility
  - Use case: Long-term strategic allocation

**Decision:** Manager uses 5Y Treasury (4.8%) for CAPM baseline; aligns incentives with investor horizon.

**Example 2: Nominal vs Real Risk-Free Rate**

Current environment (Jan 2026):
- Nominal T-bill yield: 4.2%
- Expected inflation (CPI): 2.5%
- **Implied real rate:** 4.2% - 2.5% = 1.7%

Investor analyzing stock market (10% expected return):
- **Nominal CAPM:** E[Rm] - rf = 10% - 4.2% = 5.8% risk premium
- **Real CAPM:** E[Rm,real] - rf,real ≈ 7% - 1.7% = 5.3% (slightly lower)
- **Practical implication:** Using real rates understates equity premium; most use nominal

**Fisher Equation reconciliation:**
- Nominal rate = real rate + expected inflation
- rf,nominal = rf,real + πe
- rf,real = rf,nominal - πe
- Problem: πe (inflation expectations) unobservable; use market-implied via TIPS (Treasury Inflation-Protected Securities)

**Example 3: Reinvestment Risk in Long-Term Treasuries**

Investor buys 30-year Treasury bond yielding 5% (locked in rate):
- Year 1: Receives coupon payment $50 per $1,000 face
- **Reinvestment problem:** Where to invest $50 coupon for next 29 years?
  - If yields drop to 3%, reinvested coupons earn only 3% (hurts total return)
  - If yields rise to 6%, reinvested coupons earn 6% (helps total return)
- **Impact:** Realized return over 30 years ≠ 5% yield (due to coupon reinvestment variance)
- **Implication:** 30Y bond is NOT risk-free in reinvestment sense; has reinvestment risk

**Contrast with zero-coupon STRIPS:**
- Strip bonds (zero-coupon Treasury): No coupon; buy at deep discount, get face value at maturity
- Realized return = yield to maturity (no reinvestment risk until maturity)
- **Truly risk-free over holding period** if no sell before maturity
- Practical: STRIPS more "risk-free" but less liquid

**Example 4: Changing Risk-Free Rate Over Time**

2008 Financial Crisis:
- Pre-crisis (Jan 2008): rf = 3.5% (2Y Treasury)
- Crisis low (Dec 2008): rf ≈ 0.3% (Federal Reserve dropped rates; flight to safety)
- **Impact on asset pricing:**
  - CAPM: E[Ri] = rf + βi(E[Rm] - rf)
  - With rf = 3.5%, low-beta stocks seem attractive (low return enough)
  - With rf = 0.3%, even low-beta stocks should have low expected returns
  - **Implication:** Dramatically lower equity risk premiums during crisis (crisis beta rises faster than rf drops)

2022 Rate Normalization:
- Post-COVID low (Dec 2021): rf = 0.1%
- Rate hikes (Dec 2022): rf = 4.2% (Fed raised 450 basis points in ~12 months)
- **Impact:** Portfolios optimized at 0.1% rf suddenly need rebalancing; optimal allocations shift dramatically

**Key lesson:** Using stale rf (outdated rate) causes misallocation; must update rf frequently as rate environment changes.

**Example 5: Country Risk & Sovereign Debt (Non-U.S. Risk-Free)**

U.S. investor in various countries:
- **U.S. Treasury 10Y:** 4.9% (true risk-free in USD)
- **German Bund 10Y:** 2.3% (lower risk, eurozone integration; no currency risk if Euro investor)
- **Japanese Government Bond (10Y):** 0.8% (aging population, low growth expectations)
- **Greek Government Bond (10Y):** 3.5% (includes sovereign debt risk premium ~1.2%)
- **Brazilian Government Bond (10Y):** 10.5% (includes currency risk + default risk premium)

**For USD-based investor:** Cannot use non-USD bonds as rf (currency risk introduced). Should use U.S. Treasury 4.9%.

**Implication:** "Risk-free" rate is denomination-specific (risk-free in USD, not in EUR or BRL).

---

## 4. Layer Breakdown

```
Risk-Free Rate: Nature, Measurement & Application Architecture:

├─ Theoretical Foundation:
│   ├─ Definition (Rigorous):
│   │   │ Asset with:
│   │   │ 1) Zero default probability (no credit risk)
│   │   │ 2) Zero reinvestment uncertainty (or hold-to-maturity = certain)
│   │   │ 3) Returns uncorrelated with market (β = 0)
│   │   │ 4) Infinitesimal volatility (σ = 0)
│   │   │ 5) Perfectly liquid (instant buy/sell)
│   │   │
│   │   └─ Reality: No such asset exists
│   │
│   ├─ What rf Represents:
│   │   ├─ Time value of money: Pure intertemporal preference (consuming tomorrow vs today)
│   │   ├─ Inflation compensation: Expectation-adjusted nominal return
│   │   │   └─ Real rf = Nominal rf - Expected inflation
│   │   ├─ Opportunity cost: Return available with zero effort/risk
│   │   ├─ Discount rate baseline: All other assets priced relative to rf
│   │   └─ Risk premium anchor: RP = E[R] - rf (excess return demanded for risk)
│   │
│   ├─ Why Needed in Finance:
│   │   ├─ CAPM denominator: E[Ri] = rf + βi × (E[Rm] - rf)
│   │   │   └─ Without rf, cannot separate systematic risk from market level
│   │   ├─ DCF valuation: PV = Σ CF / (1 + rf)^t
│   │   │   └─ Without rf, cannot discount future cash flows
│   │   ├─ Capital Allocation Line (CAL): E[Rp] = rf + (E[Rm] - rf)/σm × σp
│   │   │   └─ Without rf, CAL undefined; can't mix risk-free with risky
│   │   ├─ Portfolio optimization: w* = (1/λ) × Σ^-1 × (E[R] - rf × 1)
│   │   │   └─ Without rf, cannot find tangency portfolio
│   │   └─ Risk premium measurement: Equity premium = E[Rm] - rf
│   │       └─ Without rf, cannot quantify risk premium
│   │
│   └─ Conceptual vs Practical rf:
│       ├─ Conceptual: Mathematical construct (σ=0, default prob=0)
│       ├─ Practical: Use Treasury yields as proxy
│       └─ Gap: Treasuries have term premium, reinvestment risk, inflation uncertainty
│           └─ But gap small enough to use as rf approximation
│
├─ Measurement & Observable Proxies:
│   ├─ U.S. Treasury Securities (Primary rf proxies):
│   │   ├─ T-Bills (< 1 year maturity):
│   │   │   ├─ Maturity: 4-week, 13-week, 26-week, 52-week
│   │   │   ├─ Characteristics:
│   │   │   │   ├─ Minimal duration risk (short maturity)
│   │   │   │   ├─ Negligible reinvestment risk (short holding period)
│   │   │   │   ├─ Extremely liquid (most traded)
│   │   │   │   ├─ Near-zero credit risk (U.S. backed)
│   │   │   │   └─ Very low volatility
│   │   │   ├─ Typical yield: 4-5% (when Fed Funds rate 4-5%)
│   │   │   ├─ Best for: Short-term cash management, money market
│   │   │   └─ Use in CAPM: Limited (misaligned horizons for long-term portfolios)
│   │   │
│   │   ├─ Treasury Notes (2-10 year maturity):
│   │   │   ├─ Standard terms: 2Y, 3Y, 5Y, 7Y, 10Y
│   │   │   ├─ Characteristics:
│   │   │   │   ├─ Moderate duration risk (rate-sensitive)
│   │   │   │   ├─ Some reinvestment risk (coupons over life)
│   │   │   │   ├─ Highly liquid (actively traded)
│   │   │   │   ├─ Zero credit risk
│   │   │   │   └─ Price volatility: ~1% per 1% rate change (duration ~5-7)
│   │   │   ├─ Typical yield: 4-6% (varies by maturity; longer ≈ higher)
│   │   │   ├─ Best for: Medium-term portfolios, strategic allocation
│   │   │   └─ Use in CAPM: Common (5Y or 10Y most popular)
│   │   │
│   │   └─ Treasury Bonds (20-30 year maturity):
│   │       ├─ Standard terms: 20Y, 30Y
│   │       ├─ Characteristics:
│   │       │   ├─ High duration risk (very rate-sensitive)
│   │       │   ├─ Significant reinvestment risk (50+ coupons)
│   │       │   ├─ Liquid but less traded than shorter terms
│   │       │   ├─ Zero credit risk
│   │       │   └─ Price volatility: ~2% per 1% rate change (duration ~20-22)
│   │       ├─ Typical yield: 4.5-6% (slight premium to 10Y for term)
│   │       ├─ Best for: Long-term buy-and-hold, liability matching
│   │       └─ Use in CAPM: Limited (long duration hurts short-term investors)
│   │
│   ├─ TIPS (Treasury Inflation-Protected Securities):
│   │   ├─ Feature: Principal adjusted by CPI; coupons = fixed real rate
│   │   ├─ Composition: Nominal return = real rate + inflation
│   │   ├─ Extraction: Nominal Treasury yield - TIPS yield = market-implied inflation expectation
│   │   ├─ Advantage: Provides observable real rf (not need to estimate)
│   │   │   └─ Real rf ≈ TIPS yield
│   │   ├─ Disadvantage: Less liquid; smaller market than nominal Treasuries
│   │   └─ Use case: Inflation-linked returns analysis, real CAPM
│   │
│   ├─ Term Structure of Rates (Yield Curve):
│   │   ├─ Shape: Plots yield (Y-axis) vs maturity (X-axis)
│   │   ├─ Normal (upward sloping): Longer maturity = higher yield
│   │   │   └─ Reason: Term premium (investors demand compensation for duration risk)
│   │   ├─ Inverted (downward sloping): Longer maturity = lower yield
│   │   │   └─ Reason: Recession fears; investors flee to longer bonds (buy long, sell short)
│   │   ├─ Flat: All maturities similar yield
│   │   │   └─ Reason: Transition period; uncertain economic outlook
│   │   │
│   │   ├─ Term Premium Decomposition:
│   │   │   └─ Nominal yield = Expected future short rates + Term premium
│   │   │       ├─ Example: 10Y yield = E[avg(r1...r10)] + TP_10yr
│   │   │       ├─ If E[avg short rates] = 3% and TP = 1%, then 10Y ≈ 4%
│   │   │       └─ TP changes over time (2021-2022 inverted due to Fed hikes; TP compressed)
│   │   │
│   │   └─ Implication for rf selection:
│   │       ├─ Short-term investor: Use 1-2Y Treasury (lower term premium)
│   │       ├─ Medium-term investor: Use 5Y Treasury (moderate term premium)
│   │       ├─ Long-term investor: Use 10Y Treasury (includes term premium but aligns horizon)
│   │       └─ Mismatch: Using 10Y rf for 1-year portfolio overstates true risk-free return
│   │
│   └─ International Risk-Free Rates:
│       ├─ Eurozone (EUR): German Bund 10Y ≈ 2-3% (lower growth expectations)
│       ├─ Japan (JPY): Japanese Government Bond 10Y ≈ 0.5-1% (low rates, aging)
│       ├─ UK (GBP): UK Gilt 10Y ≈ 3.5-4.5% (similar to US)
│       ├─ Canada (CAD): Canadian Bond 10Y ≈ 3-4% (tied to USD; less default risk)
│       ├─ Australia (AUD): Australian Bond 10Y ≈ 3.5-4.5% (commodity-linked)
│       └─ Note: For USD investor, USD Treasury is only true "risk-free" (no FX risk)
│
├─ Risk Decomposition of Proxy Assets:
│   ├─ U.S. Treasury 10Y Yield contains:
│   │   ├─ Real interest rate (1.5-2.5% baseline): Time preference, productivity
│   │   ├─ Expected inflation (2-2.5%): Purchasing power protection
│   │   ├─ Term premium (0.5-1.5%): Compensation for duration risk
│   │   ├─ Liquidity premium (near zero): Treasuries very liquid
│   │   ├─ Convenience yield (negligible): Treasuries safe collateral
│   │   └─ Total: 1.5% + 2.2% + 1.0% + 0 + 0 ≈ 4.7% ✓
│   │
│   ├─ Risks NOT in Treasury (therefore NOT in rf):
│   │   ├─ Inflation surprise: If actual > expected, real return falls
│   │   ├─ Interest rate risk: If rates rise, bond price falls (duration loss)
│   │   ├─ Reinvestment risk: Coupon rates may fall before next reinvestment
│   │   ├─ Opportunity cost: Returns from better investments elsewhere
│   │   └─ Implication: Even Treasuries have risk; "risk-free" is misnomer
│   │
│   ├─ Credit Risk Decomposition (Why Treasuries ≈ rf):
│   │   ├─ Default probability: ~0% (U.S. can print currency; no historical default)
│   │   ├─ Sovereign risk: Minimal (full faith & credit of U.S. government)
│   │   ├─ Systemic risk: If U.S. defaults, all USD assets worthless (global catastrophe)
│   │   │   └─ Implication: Cannot hedge U.S. sovereign risk in USD; accepted as given
│   │   └─ Practical: Treasuries treated as rf because risk << other assets
│   │
│   └─ Volatility Characteristics:
│       ├─ T-Bill 13-week: σ(daily returns) ≈ 0.01-0.05%
│       ├─ Treasury 10Y: σ(daily returns) ≈ 0.5-1.5%
│       │   └─ Reason: Duration ~8; 1% yield change → 0.8% price change
│       ├─ Corporate Bond (Aaa): σ ≈ 1-2%
│       ├─ Stock (S&P 500): σ ≈ 15-18%
│       └─ Ranking: Treasuries << stocks (justifies treating as lower-risk baseline)
│
├─ Application in Asset Pricing:
│   ├─ CAPM Implementation:
│   │   │ E[Ri] = rf + βi(E[Rm] - rf)
│   │   ├─ Step 1: Select rf proxy (match investor horizon)
│   │   │   ├─ Short-horizon (< 5Y): Use 2Y Treasury
│   │   │   ├─ Medium-horizon (5-10Y): Use 5Y Treasury
│   │   │   └─ Long-horizon (> 10Y): Use 10Y Treasury
│   │   ├─ Step 2: Estimate beta (regression of asset return on market return)
│   │   ├─ Step 3: Estimate market risk premium (E[Rm] - rf)
│   │   │   └─ Typical: 5-6% (varies by model: historical, forward-looking, survey)
│   │   ├─ Step 4: Calculate E[Ri] = rf + βi × RP
│   │   │
│   │   ├─ Sensitivity Analysis:
│   │   │   ├─ If rf increases 1% (say, 4% → 5%):
│   │   │   │   ├─ E[Ri] increases ≈ 0.5-1% (depends on beta)
│   │   │   │   ├─ Low-beta defensive stocks: E[R] increase ~0.7% (low leverage of rf)
│   │   │   │   └─ High-beta cyclical stocks: E[R] increase ~1.3% (high leverage of rf)
│   │   │   ├─ If risk premium decreases 1% (say, 6% → 5%):
│   │   │   │   ├─ E[Ri] decreases β × 1% (direct scaling by beta)
│   │   │   │   ├─ Low-beta: E[R] decrease ~0.5%
│   │   │   │   └─ High-beta: E[R] decrease ~1.5%
│   │   │   └─ Implication: Valuations highly sensitive to rf choice
│   │
│   └─ DCF Valuation:
│       │ PV = Σ CFt / (1 + rf)^t + Terminal Value / (1 + rf)^n
│       ├─ rf used as discount rate (cost of capital baseline)
│       ├─ If rf too high: PV suppressed (undervaluation)
│       ├─ If rf too low: PV inflated (overvaluation)
│       ├─ Typical sensitivity: 1% change in rf → 10-20% change in PV (depends on n)
│       └─ Example: Valuing 30-year utility with stable cash flows
│           └─ At rf=3%: PV ≈ $100 | At rf=4%: PV ≈ $80 (20% decline)
│
├─ Historical Context & Cyclicality:
│   ├─ Pre-2008: rf ≈ 4-5% (normal level)
│   ├─ 2008 Crisis: rf → 0.3% (emergency rates)
│   ├─ 2009-2021: rf ≈ 0-2% (extended low-rate environment; post-crisis support)
│   ├─ 2021-2022: rf ≈ 0% → 4.5% (Fed rate hikes in response to inflation)
│   ├─ Current (2024-2025): rf ≈ 4-5% (normalized level)
│   ├─ Implications:
│   │   ├─ Portfolio optimized at rf=0% becomes suboptimal at rf=4.5%
│   │   ├─ Valuations (DCF) cut in half from rf doubling (rf=2% → 4%)
│   │   ├─ Risk premiums compressed when rf rises (stock valuations decline)
│   │   └─ Need to rebalance portfolios as rf regime changes
│   │
│   └─ Future outlook: Long-run average rf ≈ 2-3% real + inflation (4-5% nominal)
│
└─ Limitations & Caveats:
    ├─ No true risk-free rate: Treasuries subject to:
    │   ├─ Inflation risk (purchasing power)
    │   ├─ Reinvestment risk (coupons)
    │   ├─ Interest rate risk (price volatility)
    │   └─ Opportunity cost (better investments exist)
    │
    ├─ Horizon mismatch: Using 10Y Treasury for 1Y portfolio:
    │   ├─ Implies investor willing to lock in 10Y rate (but only needs 1Y)
    │   ├─ Causes portfolio suboptimality (excess duration risk)
    │   └─ Better: Use 1Y Treasury (matches horizon exactly)
    │
    ├─ Time-varying rf: Using historical average rf:
    │   ├─ Ignores current rate environment (outdated)
    │   ├─ If current rf >> historical average, CAPM underestimates returns
    │   ├─ Need to update rf frequently (quarterly or when major rate moves)
    │   └─ Pro tip: Use current yield, not average yield
    │
    ├─ Currency denomination: Using non-USD rf for USD investor:
    │   ├─ Introduces FX risk (violates zero-risk assumption)
    │   ├─ For USD investor, USD Treasury is only true rf
    │   ├─ Foreign rates have implicit currency premium
    │   └─ Solution: Use USD Treasury, adjust for country risk separately
    │
    └─ Model sensitivity: Small rf changes → large valuation changes:
        ├─ Robust optimization: Use rf range (e.g., 3-5%) not point estimate
        ├─ Sensitivity tables: Show how allocation changes with rf
        ├─ Stress testing: Scenario analysis (rf +1%, rf -1%)
        └─ Implication: Don't over-rely on single rf value
```

**Mathematical Formulas:**

Nominal vs Real risk-free rate (Fisher Equation):
$$r_{f,nominal} = r_{f,real} + \pi^e$$

Expected return with risk-free rate:
$$E[R_i] = r_f + \beta_i (E[R_m] - r_f)$$

Present value with discount rate:
$$PV = \sum_{t=1}^{T} \frac{CF_t}{(1 + r_f)^t}$$

Yield curve extraction (no-arbitrage principle):
$$r_{spot,n} = \frac{1}{n} \sum_{t=1}^{n} E[r_t]$$

---

## 5. Mini-Project: Measuring & Analyzing Risk-Free Rates

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json

# Download and analyze risk-free rates; compare proxies

def get_treasury_yields():
    """
    Fetch current U.S. Treasury yields from FRED (Federal Reserve Economic Data).
    Using yfinance as proxy; actual FRED API would be better but requires key.
    """
    # Approximate yields based on Treasury price data
    treasury_maturities = {
        '3M': '^IRX',      # 3-month T-bill
        '6M': '^IRX',      # 6-month T-bill (approximated)
        '1Y': 'IEF',       # 7-10 year Treasury ETF (proxy)
        '2Y': '^IEY',      # 2-year yield (estimated)
        '5Y': '^FVX',      # 5-year yield
        '10Y': '^TNX',     # 10-year yield
        '30Y': '^TYX',     # 30-year yield
    }
    
    yields = {}
    for name, ticker in treasury_maturities.items():
        try:
            data = yf.download(ticker, start='2020-01-01', end=datetime.now(), 
                             progress=False, interval='1d')
            if 'Adj Close' in data.columns:
                current = data['Adj Close'].iloc[-1]
                yields[name] = current / 100  # Convert to decimal
            else:
                yields[name] = np.nan
        except:
            yields[name] = np.nan
    
    return yields


def simulate_treasuries_pf(years_data=5):
    """
    Create historical Treasury yield data; simulate via synthetic returns.
    """
    dates = pd.date_range(end=datetime.now(), periods=252*years_data, freq='D')
    
    # Synthetic Treasury yields over time (realistic patterns)
    t_bill_yields = np.linspace(0.02, 0.045, len(dates)) + np.random.normal(0, 0.002, len(dates))
    t_2y_yields = np.linspace(0.025, 0.048, len(dates)) + np.random.normal(0, 0.003, len(dates))
    t_5y_yields = np.linspace(0.03, 0.050, len(dates)) + np.random.normal(0, 0.004, len(dates))
    t_10y_yields = np.linspace(0.035, 0.049, len(dates)) + np.random.normal(0, 0.005, len(dates))
    t_30y_yields = np.linspace(0.04, 0.048, len(dates)) + np.random.normal(0, 0.005, len(dates))
    
    yields_df = pd.DataFrame({
        'Date': dates,
        '3M': np.maximum(t_bill_yields, 0.001),
        '2Y': np.maximum(t_2y_yields, 0.001),
        '5Y': np.maximum(t_5y_yields, 0.001),
        '10Y': np.maximum(t_10y_yields, 0.001),
        '30Y': np.maximum(t_30y_yields, 0.001),
    })
    yields_df.set_index('Date', inplace=True)
    
    return yields_df


def compute_duration_price_sensitivity(yield_change, duration, current_price=100):
    """
    Compute Treasury price change from yield move using duration formula.
    % price change ≈ -duration × Δyield
    """
    price_change_pct = -duration * yield_change
    new_price = current_price * (1 + price_change_pct)
    return price_change_pct, new_price


def extract_real_rate_from_tips(nominal_yield, tips_yield):
    """
    Extract real rate from nominal vs TIPS yields.
    Real rate ≈ nominal yield - TIPS yield (simplified)
    More precisely: (1 + nominal) = (1 + real) × (1 + inflation)
    """
    real_rate = nominal_yield - tips_yield
    return real_rate


def term_premium_decomposition(yields_dict):
    """
    Estimate term premium from yield curve shape.
    Term premium ≈ Longer yield - Shorter yield
    """
    term_premium_2y = yields_dict.get('2Y', 0) - yields_dict.get('3M', 0)
    term_premium_5y = yields_dict.get('5Y', 0) - yields_dict.get('2Y', 0)
    term_premium_10y = yields_dict.get('10Y', 0) - yields_dict.get('5Y', 0)
    term_premium_total = yields_dict.get('30Y', 0) - yields_dict.get('3M', 0)
    
    return {
        '2Y TP': term_premium_2y,
        '5Y TP': term_premium_5y,
        '10Y TP': term_premium_10y,
        'Total TP (30Y-3M)': term_premium_total
    }


# Main Analysis
print("=" * 100)
print("RISK-FREE RATE: NATURE, MEASUREMENT & APPLICATION")
print("=" * 100)

# 1. Historical Treasury yields
print("\n1. HISTORICAL TREASURY YIELD CURVE (Synthetic Data)")
print("-" * 100)

yields_hist = simulate_treasuries_pf(years_data=5)

print("\nCurrent Yields (end of period):")
print(yields_hist.iloc[-1] * 100)

print("\nHistorical Statistics (%):")
print((yields_hist * 100).describe().T)

# 2. Yield curve shape
print("\n2. YIELD CURVE ANALYSIS")
print("-" * 100)

current_yields = yields_hist.iloc[-1]
term_premiums = term_premium_decomposition(current_yields)

print("\nCurrent Yield Curve (%):")
for maturity, yield_val in current_yields.items():
    print(f"  {maturity:5s}: {yield_val*100:5.2f}%")

print("\nTerm Premiums (Longer - Shorter Maturity):")
for tp_name, tp_val in term_premiums.items():
    print(f"  {tp_name:20s}: {tp_val*100:5.2f}%")

# 3. Duration & Interest Rate Risk
print("\n3. TREASURY DURATION & PRICE SENSITIVITY")
print("-" * 100)

durations = {'3M': 0.25, '2Y': 1.9, '5Y': 4.5, '10Y': 8.2, '30Y': 20.0}
rate_move = 0.01  # 1% rate increase

print(f"\nIf yields rise by {rate_move*100:.1f}% (100 basis points):\n")
print(f"{'Maturity':<10} {'Duration':<12} {'Price Change %':<18} {'New Price':<15}")
print("-" * 55)

for maturity in ['3M', '2Y', '5Y', '10Y', '30Y']:
    duration = durations[maturity]
    price_change_pct, new_price = compute_duration_price_sensitivity(rate_move, duration)
    print(f"{maturity:<10} {duration:<12.2f} {price_change_pct*100:<18.2f} {new_price:<15.2f}")

print("\nKey insight: Longer-duration bonds suffer larger price declines (interest rate risk)")

# 4. Real vs Nominal RF rates
print("\n4. REAL VS NOMINAL RISK-FREE RATE")
print("-" * 100)

nominal_10y = yields_hist['10Y'].iloc[-1]
expected_inflation = 0.025  # 2.5% assumption

real_rf = nominal_10y - expected_inflation

print(f"\nNominal 10Y Treasury Yield: {nominal_10y*100:.2f}%")
print(f"Expected Inflation (assumption): {expected_inflation*100:.2f}%")
print(f"Implied Real RF: {real_rf*100:.2f}%")

print(f"\nFisher Equation: r_nominal = r_real + π^e")
print(f"Verification: {real_rf*100:.2f}% + {expected_inflation*100:.2f}% = {nominal_10y*100:.2f}%")

# 5. CAPM sensitivity to rf choice
print("\n5. CAPM EXPECTED RETURN SENSITIVITY TO RF CHOICE")
print("-" * 100)

market_premium = 0.06  # 6% equity risk premium assumption
betas = {'Low-beta defensive': 0.6, 'Market (SPY)': 1.0, 'High-beta cyclical': 1.4}

print(f"\nAssuming Market Risk Premium = {market_premium*100:.1f}%\n")
print(f"Expected Returns with Different RF Choices:\n")
print(f"{'Asset':<25} {'Low RF (2%)':<18} {'Mid RF (4%)':<18} {'High RF (5%)':<15}")
print("-" * 75)

for asset, beta in betas.items():
    rf_low = 0.02
    rf_mid = 0.04
    rf_high = 0.05
    
    e_r_low = rf_low + beta * market_premium
    e_r_mid = rf_mid + beta * market_premium
    e_r_high = rf_high + beta * market_premium
    
    print(f"{asset:<25} {e_r_low*100:<18.2f} {e_r_mid*100:<18.2f} {e_r_high*100:<15.2f}")

print("\nKey insight: Higher RF → Higher expected returns across all assets (proportional to beta)")

# 6. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Yield curve evolution
ax = axes[0, 0]
for col in ['3M', '2Y', '5Y', '10Y', '30Y']:
    ax.plot(yields_hist.index, yields_hist[col] * 100, label=col, linewidth=2)

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Yield (%)', fontsize=11)
ax.set_title('U.S. Treasury Yield Curve Evolution', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Current yield curve shape
ax = axes[0, 1]
maturities = [0.25, 2, 5, 10, 30]  # Years
current_yields_list = [yields_hist[m].iloc[-1] * 100 for m in ['3M', '2Y', '5Y', '10Y', '30Y']]

ax.plot(maturities, current_yields_list, 'o-', linewidth=2.5, markersize=8, color='#3498db')
ax.fill_between(maturities, current_yields_list, alpha=0.3, color='#3498db')

ax.set_xlabel('Maturity (Years)', fontsize=11)
ax.set_ylabel('Yield (%)', fontsize=11)
ax.set_title('Current Yield Curve Shape', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3)

# Plot 3: Price sensitivity to yield moves
ax = axes[1, 0]
yield_moves = np.linspace(-0.02, 0.02, 50)
for maturity in ['3M', '5Y', '10Y', '30Y']:
    duration = durations[maturity]
    price_changes = [-duration * move * 100 for move in yield_moves]
    ax.plot(yield_moves * 100, price_changes, label=f'{maturity} (D≈{duration:.1f})', linewidth=2)

ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Yield Change (Basis Points)', fontsize=11)
ax.set_ylabel('Price Change (%)', fontsize=11)
ax.set_title('Treasury Price Sensitivity (Duration Risk)', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: CAPM return expectations by rf
ax = axes[1, 1]
rf_range = np.linspace(0.01, 0.06, 20)
market_premium = 0.06

for asset, beta in betas.items():
    expected_returns = [rf + beta * market_premium for rf in rf_range]
    ax.plot(rf_range * 100, np.array(expected_returns) * 100, 'o-', label=asset, linewidth=2, markersize=5)

ax.set_xlabel('Risk-Free Rate (%)', fontsize=11)
ax.set_ylabel('Expected Return (%)', fontsize=11)
ax.set_title('CAPM Expected Returns vs RF Choice', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('risk_free_rate_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: risk_free_rate_analysis.png")
plt.show()

# 7. Key insights
print("\n6. KEY INSIGHTS & PRACTICAL RECOMMENDATIONS")
print("-" * 100)
print(f"""
RISK-FREE RATE FUNDAMENTALS:
├─ Definition: Theoretical asset with zero default/reinvestment risk
├─ Proxy: U.S. Treasury securities (best approximation)
├─ Horizon-matched selection:
│   ├─ Short-term (< 5 years): Use 2Y or 5Y Treasury
│   ├─ Long-term (> 10 years): Use 10Y Treasury
│   └─ Mismatch hurts portfolio (reinvestment or duration risk)
│
├─ Components of Treasury yield:
│   ├─ Real interest rate (1-3%): Time preference, productivity
│   ├─ Expected inflation (1-3%): Purchasing power
│   ├─ Term premium (0-1.5%): Duration risk compensation
│   └─ Total: 2-7% depending on economic regime
│
├─ Key risks NOT eliminated (despite "risk-free" name):
│   ├─ Inflation risk: If realized inflation > expected, real return falls
│   ├─ Interest rate risk: If rates rise, bond prices fall
│   ├─ Reinvestment risk: Coupons may reinvest at lower rates
│   └─ Opportunity cost: Better returns available elsewhere
│
├─ CAPM sensitivity:
│   ├─ 1% increase in RF → Expected returns increase ~0.5-1% (beta-weighted)
│   ├─ High-beta stocks more leveraged to RF changes
│   ├─ Update RF quarterly as rate environment changes
│   └─ Small RF errors → large valuation errors (sensitivity high)
│
├─ Historical context:
│   ├─ Pre-2008: RF ≈ 4-5% (normal)
│   ├─ 2008-2022: RF ≈ 0-2% (emergency low)
│   ├─ 2022-present: RF ≈ 4-5% (normalized)
│   └─ Implication: Static RF assumptions become stale; adapt to rate regime
│
├─ Recommended best practices:
│   ├─ Use current (spot) Treasury yield, not historical average
│   ├─ Match Treasury maturity to investment horizon
│   ├─ Update RF monthly or after Fed announcements
│   ├─ Sensitivity analysis: Model range (RF ± 1%) not point estimate
│   ├─ Real RF analysis: Compare to TIPS yield for inflation-adjusted returns
│   └─ Consider term premium: 10Y yield includes ~0.5-1% for duration risk
│
├─ Portfolio implications:
│   ├─ Rising RF → Rebalance away from equities (better yield on bonds)
│   ├─ Falling RF → Rebalance into equities (bonds less attractive)
│   ├─ Valuation inversely sensitive: 1% RF increase → 10-20% equity valuation decline
│   └─ Horizon mismatch: Conservative investors may overpay for long-term certainty
│
└─ Calculation example (for CAPM):
    ├─ Current 10Y Treasury: {nominal_10y*100:.2f}%
    ├─ Market Risk Premium: 6%
    ├─ Low-beta stock (β=0.6): E[R] = {nominal_10y*100:.2f}% + 0.6 × 6% = {(nominal_10y + 0.6*0.06)*100:.2f}%
    ├─ High-beta stock (β=1.4): E[R] = {nominal_10y*100:.2f}% + 1.4 × 6% = {(nominal_10y + 1.4*0.06)*100:.2f}%
    └─ Implication: {(nominal_10y + 1.4*0.06)*100:.2f}% - {(nominal_10y + 0.6*0.06)*100:.2f}% = {0.8*0.06*100:.2f}% expected return difference from beta alone
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Horizon Matching:** Portfolio has 3-year horizon; current 10Y Treasury = 5%, 2Y = 4.5%, 3M = 4.2%. Which should you use for CAPM? Why does mismatch matter? What happens if rates rise 1% before your 3-year goal?

2. **Real vs Nominal:** If nominal 10Y = 5%, expected inflation = 3%, compute real RF. Now suppose inflation surprises to 4% (not 3%). How does realized real return change? Why do inflation surprises hurt Treasury returns?

3. **Duration Sensitivity:** A 30-year Treasury has duration ≈ 20. If Fed raises rates 50bp (0.5%), compute approximate price decline. If you're retiree living off Treasury coupons, does duration matter? Why?

4. **Term Premium Extraction:** If 3M T-bill = 4%, 10Y Treasury = 4.8%, estimate term premium. Is this "fair" premium? If expected inflation = 2.5%, what does this imply about real rate? Can real rate be negative; if so, why?

5. **International RF Risk:** U.S. investor considering euro-denominated bonds (German Bund 10Y = 2.5%, U.S. Treasury = 4.9%). EUR/USD currently 1.10. Should you use 2.5% or 4.9% as RF in CAPM? What risk are you taking if you use 2.5%?

---

## 7. Key References

- **Fama, E.F. & French, K.R. (1989).** "Business Conditions and Expected Returns on Stocks and Bonds" – Relationship between risk-free rate and asset returns; term structure effects.

- **Sharpe, W.F. (1964).** "Capital Asset Prices: A Theory of Market Equilibrium" – CAPM model; role of risk-free rate as baseline; Nobel Prize 1990.

- **Fisher, I. (1930).** "The Theory of Interest" – Seminal work on real vs nominal rates; Fisher equation foundation.

- **Bliss, R.R. (1996).** "Testing Term Structure Estimation Methods" – Yield curve estimation; comparing Treasury term structure methods.

- **U.S. Department of Treasury:** Treasury Yields – https://www.treasury.gov/resource-center/data-chart-center/ – Official U.S. Treasury yields; daily updates.

- **Federal Reserve:** "Treasury Yield Curve Rates" – https://www.federalreserve.gov/datadownload/GDP-and-US-international-trade-indicators-G17/About – Fed data on rates; FRED economic database.

- **CFA Institute:** "Fixed Income Analysis" – Professional curriculum on Treasury analysis; duration, convexity, term structure.

