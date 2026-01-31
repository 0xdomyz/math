# Cost of Capital & Weighted Average Cost of Capital (WACC)

## 1. Concept Skeleton
**Definition:** WACC = weighted average of cost of equity and cost of debt; represents required return on all capital; discount rate for DCF  
**Purpose:** Determine hurdle rate for investment decisions; discount future cash flows; adjust for financial risk; benchmark against expected returns  
**Prerequisites:** Cost of equity (CAPM), cost of debt, capital structure weights, tax effects, market data (risk-free rate, market risk premium, beta)

## 2. Comparative Framing
| Component | Cost of Equity | Cost of Debt | WACC | Hurdle Rate |
|-----------|----------------|-------------|------|-----------|
| **Basis** | Equity risk premium; beta | Interest rate; credit spread | Blend of equity + debt | Business-specific threshold |
| **Calculation** | CAPM: Rf + β(Rm - Rf) | Rd = Interest / Debt | E/V × Re + D/V × Rd(1-T) | Often WACC + buffer |
| **Typical range** | 8-15% (varies by risk) | 2-8% (varies by credit) | 5-10% (WACC blend) | 10-15% (added margin) |
| **Drivers** | Market conditions, beta | Credit quality, leverage | Capital mix, tax rate | Business risk, execution |
| **Sensitivity** | High (market movements) | Moderate (spread changes) | Moderate (both components) | Project-specific |
| **Use in DCF** | Primary (equity valuation) | Secondary (debt component) | Combined (enterprise value) | Override if conservative |

## 3. Examples + Counterexamples

**Simple Example:**  
Mature utility company: Cost of equity 7% (low beta 0.5, stable cash flows); Cost of debt 4% (investment grade); Capital structure 60% debt, 40% equity; WACC = 0.40 × 0.07 + 0.60 × 0.04 × (1-0.30) = 2.8% + 1.68% = 4.48%; use as DCF discount rate for stable, low-risk business

**Failure Case:**  
High-growth startup: Estimate cost of equity 20% (high beta due to volatility); Cost of debt 8% (risky); WACC calculated 15%; but actual company value depends on reaching profitability (binary outcome); WACC assumes continuous risk when true risk is binary (wrong model); valuation misleading

**Edge Case:**  
Acquisition target in leveraged buyout: Pre-acquisition WACC 8% (balanced capital structure); post-acquisition leverage rises (D/V increases to 70%); Cost of debt increases to 6% (higher default risk); new WACC = 0.30 × 10% + 0.70 × 0.06 × 0.70 = 3% + 2.94% = 5.94%; lower than pre-acquisition (changes only numerator, not underlying risk profile; leverage benefits decline as risk rises)

## 4. Layer Breakdown
```
WACC Structure:
├─ Cost of Equity (Re) - Equity Capital Component:
│   ├─ CAPM Framework:
│   │   ├─ Formula: Re = Rf + β × (Rm - Rf)
│   │   ├─ Where:
│   │   │   ├─ Rf = Risk-free rate (Treasury yield)
│   │   │   ├─ β = Beta (systematic risk relative to market)
│   │   │   ├─ Rm = Expected market return
│   │   │   ├─ (Rm - Rf) = Market risk premium (ERP)
│   │   │   └─ Example: Rf 2.5% + β 1.2 × (7% - 2.5%) = 2.5% + 1.2 × 4.5% = 2.5% + 5.4% = 7.9%
│   │   ├─ Intuition:
│   │   │   ├─ Investors require return ≥ risk-free rate (otherwise buy Treasuries)
│   │   │   ├─ Additional return required = Beta × Market Risk Premium
│   │   │   ├─ If β = 1.0: Stock moves with market; requires market return
│   │   │   ├─ If β > 1.0: Stock more volatile; requires higher return (risk premium)
│   │   │   ├─ If β < 1.0: Stock less volatile; requires lower return (lower risk)
│   │   │   └─ Example: Stable utility β 0.6; Cyclical auto β 1.4
│   │   └─ Example calculations:
│   │       ├─ Defensive stock: Rf 2%, β 0.7, ERP 5% → Re = 2% + 0.7×5% = 5.5%
│   │       ├─ Market stock: Rf 2%, β 1.0, ERP 5% → Re = 2% + 1.0×5% = 7%
│   │       ├─ Growth stock: Rf 2%, β 1.5, ERP 5% → Re = 2% + 1.5×5% = 9.5%
│   │       └─ Interpretation: Higher beta drives higher required return
│   ├─ Risk-Free Rate (Rf):
│   │   ├─ Definition: Yield on government bonds with no default risk
│   │   ├─ Proxy: US Treasury yield
│   │   │   ├─ Short-term (1-3 years): Used for short projects
│   │   │   ├─ Medium-term (5-10 years): Used for typical projects
│   │   │   ├─ Long-term (20-30 years): Used for very long-duration projects
│   │   │   └─ Typical choice: 10-year Treasury (matches typical valuation horizon)
│   │   ├─ Example data:
│   │   │   ├─ 10-year Treasury yield: 2.5% (low interest rate environment)
│   │   │   ├─ 10-year Treasury yield: 4.0% (normal environment)
│   │   │   ├─ 10-year Treasury yield: 5.5% (high interest rate environment)
│   │   │   └─ Valuation impact: 1% change in Rf → ~0.3% change in cost of equity
│   │   ├─ International consideration:
│   │   │   ├─ US company: Use US Treasury yield
│   │   │   ├─ UK company: Use UK Gilts yield
│   │   │   ├─ Emerging market: Use yield + country risk premium
│   │   │   │   ├─ Example: Brazil 10-year yield 12% = US Treasury 2.5% + Brazil risk premium 9.5%
│   │   │   │   └─ Implication: Emerging markets have higher risk-free rate (proxy)
│   │   │   └─ Note: If company international, blend Rf from relevant markets
│   │   └─ Timing: Use current yield; forward yields harder to estimate
│   ├─ Beta (β) - Systematic Risk:
│   │   ├─ Definition: Measure of stock volatility relative to market
│   │   ├─ Calculation methods:
│   │   │   ├─ Regression approach (most common):
│   │   │   │   ├─ Data: Monthly/weekly returns of stock vs. market index (3-5 years)
│   │   │   │   ├─ Regression: Stock return = α + β × Market return + ε
│   │   │   │   ├─ Slope coefficient = Beta
│   │   │   │   ├─ Example: β = 1.2 means stock rises 1.2% for each 1% market rise
│   │   │   ├─ Interpretation:
│   │   │   │   ├─ Market β = 1.0 (by definition)
│   │   │   │   ├─ β > 1.0: Aggressive stock (larger swings; higher leverage or cyclicality)
│   │   │   │   ├─ β < 1.0: Defensive stock (smaller swings; lower cyclicality)
│   │   │   │   ├─ β = 2.0: Stock swings double the market (e.g., leveraged stocks)
│   │   │   │   └─ β = 0.5: Stock swings half the market (e.g., utilities)
│   │   │   ├─ Examples by industry:
│   │   │   │   ├─ Utilities (defensive): β 0.6-0.9 (stable earnings; regulated returns)
│   │   │   │   ├─ Consumer staples (defensive): β 0.8-1.0 (resistant to recession)
│   │   │   │   ├─ Industrials (cyclical): β 1.0-1.3 (tied to economic cycle)
│   │   │   │   ├─ Technology (growth): β 1.2-1.6 (volatile; sensitive to sentiment)
│   │   │   │   ├─ Biotech (high-risk): β 1.3-1.8 (R&D uncertainty; regulatory risk)
│   │   │   │   └─ Leverage impact: Higher debt → higher β (financial risk added)
│   │   ├─ Unlevered vs. Levered Beta:
│   │   │   ├─ Levered β (equity beta): Reflects both business + financial risk
│   │   │   │   ├─ Used directly in CAPM for equity valuation
│   │   │   │   ├─ Affected by capital structure
│   │   │   │   └─ Example: Company with 50% leverage has higher β than same company unlevered
│   │   │   ├─ Unlevered β (asset beta): Business risk only; removes financial leverage
│   │   │   │   ├─ Formula: βu = βL / [1 + (1 - T) × D/E]
│   │   │   │   ├─ Example: Levered β 1.2, D/E 0.5, Tax 30%
│   │   │   │   │   ├─ βu = 1.2 / [1 + (1-0.3) × 0.5] = 1.2 / 1.35 = 0.89 (lower)
│   │   │   │   ├─ Use: Comparable company analysis; adjust for different capital structures
│   │   │   │   └─ Process: Unlever peer β → relever at target company's capital structure
│   │   │   ├─ Relevering:
│   │   │   │   ├─ If target company has different leverage than peer
│   │   │   │   ├─ Relever formula: βL = βu × [1 + (1 - T) × D/E]
│   │   │   │   ├─ Example: Unlever peer β 0.9, target D/E 1.0, Tax 30%
│   │   │   │   │   ├─ βL = 0.9 × [1 + (1-0.3) × 1.0] = 0.9 × 1.7 = 1.53 (higher; more leverage)
│   │   │   │   └─ Application: Ensure comparability across capital structures
│   │   └─ Data sources & adjustments:
│   │       ├─ Bloomberg, Yahoo Finance: Report levered β
│   │       ├─ Academic sources (Damodaran): Report by industry; good for comparison
│   │       ├─ Limitations of historical β:
│   │       │   ├─ Past may not predict future (market regime changes)
│   │       │   ├─ Short history (3 years) noisy; long history (10 years) stale
│   │       │   ├─ Merger/restructuring changes business; old β invalid
│   │       │   └─ Approach: Use historical as anchor; adjust for recent changes
│   │       ├─ Adjustment scenarios:
│   │       │   ├─ Company deleveraging: New beta lower than historical (less financial risk)
│   │       │   ├─ Company leveraging: New beta higher (more financial risk)
│   │       │   ├─ New product line: May increase/decrease systematic risk
│   │       │   └─ Recommendation: Unlever/relever to match expected future capital structure
│   │       └─ Example:
│   │           ├─ Historical levered β: 1.2 (at 40% debt ratio)
│   │           ├─ Unlever: βu = 1.2 / [1 + 0.7 × 0.67] = 0.85
│   │           ├─ Target capital structure: 50% debt (D/E = 1.0)
│   │           ├─ Relever: βL = 0.85 × [1 + 0.7 × 1.0] = 1.40 (reflects higher future leverage)
│   │           └─ Use new β = 1.40 in CAPM
│   ├─ Market Risk Premium (ERP):
│   │   ├─ Definition: Expected return on market portfolio minus risk-free rate
│   │   ├─ Formula: ERP = E[Rm] - Rf
│   │   ├─ Historical approach:
│   │   │   ├─ US stock market historical premium: ~5-6% (long-term average)
│   │   │   ├─ Data: 1926-present; different periods show variation
│   │   │   │   ├─ 1926-2000: ~6.5% (includes great depression, WWII, stable period)
│   │   │   │   ├─ 2000-2020: ~4.5% (lower; tech bubble aftermath, financial crisis)
│   │   │   │   ├─ 2020-present: ~5% (recovery; varied by year)
│   │   │   └─ Selection: Most analysts use 5-6% as normalized ERP
│   │   ├─ Forward-looking approach:
│   │   │   ├─ Use dividend growth model or survey data
│   │   │   ├─ Survey: Ask investors their expected market return
│   │   │   │   ├─ Typical range: 6-8% expected annual return
│   │   │   │   ├─ Minus current risk-free rate (e.g., 2.5%)
│   │   │   │   ├─ Implied ERP: 3.5-5.5%
│   │   │   │   └─ Adjustment: If economy expected to struggle, ERP might rise (demand higher return for risk)
│   │   │   └─ Dividend growth model:
│   │   │       ├─ Expected market return = Dividend Yield + Expected Growth
│   │   │       ├─ Example: S&P 500 dividend yield 2% + Expected growth 3% = 5% expected return
│   │   │       ├─ ERP = 5% - Risk-free rate 2.5% = 2.5%
│   │   │       └─ Lower than historical (suggests market fully valued)
│   │   ├─ Geographic variation:
│   │   │   ├─ US: 4-6% ERP (developed; lower risk)
│   │   │   ├─ Europe: 4-5% ERP (similar to US)
│   │   │   ├─ Japan: 3-4% ERP (lower; demographic headwinds)
│   │   │   ├─ Emerging markets: 6-8% ERP (higher; more volatility)
│   │   │   └─ Global portfolio: Blend by allocation
│   │   ├─ Sensitivity impact:
│   │   │   ├─ If ERP 5% vs. 4%: 1% difference
│   │   │   ├─ Cost of equity with β 1.5: Changes from 9.5% to 8.5% (1% change)
│   │   │   ├─ Valuation impact: 10-12% swing (moderate significance)
│   │   │   └─ Implication: ERP important; should not be carelessly assumed
│   │   └─ Practical recommendation:
│   │       ├─ Use 5% as baseline (middle of range; supported by long-term history)
│   │       ├─ Adjust for current environment: Rising rates → ERP lower; recession fears → ERP higher
│   │       ├─ Sensitivity: Test valuations at 4% and 6% ERP (±1% scenarios)
│   │       └─ Document assumption; revisit quarterly if market environment changes
│   └─ Cost of Equity Summary:
│       ├─ Calculation example:
│       │   ├─ Risk-free rate: 2.5% (10-year Treasury)
│       │   ├─ Beta: 1.2 (company risk profile)
│       │   ├─ Market risk premium: 5% (long-term average)
│       │   ├─ Cost of equity: 2.5% + 1.2 × 5% = 2.5% + 6% = 8.5%
│       │   └─ Interpretation: Equity investors require 8.5% annual return
│       ├─ Validation:
│       │   ├─ Compare to peer cost of equity (should be similar risk profile)
│       │   ├─ Industry check: Utility should be lower (5-6%); tech higher (10-12%)
│       │   ├─ Sanity check: >15% suggests very high risk or calculation error
│       │   └─ Reasonableness: Should be higher than cost of debt (equity more risky)
│       └─ Sensitivity: 
│           ├─ 1% change in Rf → ~1% change in Re
│           ├─ 0.1 change in β → ~0.5% change in Re (more impactful than Rf for volatile stocks)
│           ├─ 1% change in ERP → varies by β (higher β = higher sensitivity)
│           └─ Key: Beta and ERP are primary drivers
├─ Cost of Debt (Rd) - Debt Capital Component:
│   ├─ Definition: Interest rate investors require to hold company debt
│   ├─ Calculation:
│   │   ├─ Market approach: Yield to maturity of company's traded debt
│   │   │   ├─ If company has bonds trading: Use YTM (most accurate)
│   │   │   ├─ Example: Company issues 5-year bond; market YTM 4.5%
│   │   │   ├─ This is the cost of debt (market-determined)
│   │   │   └─ Advantage: Market-based; reflects current risk perception
│   │   ├─ Credit rating approach: If no traded debt
│   │   │   ├─ Determine company credit rating (AAA, AA, A, BBB, etc.)
│   │   │   ├─ Look up typical yield spread for that rating
│   │   │   ├─ Example: BBB rating = Risk-free 2.5% + Spread 2.0% = 4.5% YTM
│   │   │   ├─ Or use company's recent bank loan rate as proxy
│   │   │   └─ Less precise; but reasonable estimate
│   │   ├─ Interest expense approach:
│   │   │   ├─ Cost of debt = Total Interest Expense / Total Debt
│   │   │   ├─ Example: Interest expense $5M on $100M debt = 5% cost
│   │   │   ├─ Advantage: Uses actual company data
│   │   │   ├─ Disadvantage: Historical (past debt issuance rates); may not reflect current rates
│   │   │   └─ Adjustment: If issued at different rates historically, blend or use new rate
│   │   └─ Blended approach:
│   │       ├─ If company has multiple debt types (short-term, long-term, convertible)
│   │       ├─ Weight each by outstanding amount
│   │       ├─ Example: $50M short-term at 3% + $50M long-term at 5% = 4% blended
│   │       └─ Use 4% in WACC
│   ├─ Tax Effect on Debt:
│   │   ├─ Interest is tax-deductible; provides tax shield
│   │   ├─ After-tax cost: Rd × (1 - T)
│   │   ├─ Example: Rd 4.5%, Tax rate 30%
│   │   │   ├─ After-tax cost = 4.5% × (1 - 0.30) = 4.5% × 0.70 = 3.15%
│   │   │   ├─ Tax shield value = 4.5% × 0.30 = 1.35% (benefit)
│   │   │   └─ Interpretation: Company borrows at 4.5%; tax deduction saves 1.35%; net cost 3.15%
│   │   ├─ Tax rate determination:
│   │   │   ├─ Use company's effective tax rate (ETR) from financials
│   │   │   ├─ Example: Company pays 28% effective tax (lower than statutory 35%)
│   │   │   ├─ Use 28% in calculation
│   │   │   ├─ Conservative: If unsure, use lower tax rate (less shield; safer estimate)
│   │   │   └─ Forward: If expecting tax changes (new law), use forward rate
│   │   └─ Impact:
│   │       ├─ High tax rate companies: Larger tax shield; lower after-tax cost of debt
│   │       ├─ Low tax rate companies: Smaller tax shield; higher after-tax cost of debt
│   │       ├─ Tax-exempt entities (universities, nonprofits): Use Rd × (1 - 0) = Rd (no shield)
│   │       └─ Tax sensitivity: 1% change in T → ~0.045% change in after-tax Rd (small)
│   ├─ Credit Risk & Spreads:
│   │   ├─ Credit spread = Rd - Rf
│   │   ├─ Example: Rd 4.5%, Rf 2.5%, Spread 2%
│   │   ├─ Spread compensates for default risk
│   │   ├─ Historical spreads by rating:
│   │   │   ├─ AAA: 0.5-1.0% (highest quality; minimal spread)
│   │   │   ├─ AA: 1.0-1.5% (high quality)
│   │   │   ├─ A: 1.5-2.5% (medium quality)
│   │   │   ├─ BBB: 2.5-4.0% (borderline investment grade)
│   │   │   ├─ BB: 4.0-6.0% (below investment grade; speculative)
│   │   │   ├─ B: 6.0-10%+ (high default risk)
│   │   │   └─ Note: Spreads vary with economic conditions (widen in recession)
│   │   ├─ Forward adjustment:
│   │   │   ├─ If credit conditions tightening: Spreads widening; anticipate higher future Rd
│   │   │   ├─ If credit conditions easing: Spreads compressing; Rd may decline
│   │   │   └─ Volatility: Corporate spreads can swing 1-3% based on market conditions
│   │   └─ Leverage impact on spreads:
│   │       ├─ If company increasing debt: Risk rises; spreads widen; Rd increases
│   │       ├─ If company deleveraging: Risk falls; spreads narrow; Rd decreases
│   │       └─ Feedback: Higher leverage → Higher spread → Higher cost of debt (spiral)
│   └─ Cost of Debt Summary:
│       ├─ Calculation example:
│       │   ├─ Company's traded bond YTM: 4.5%
│       │   ├─ Tax rate: 30%
│       │   ├─ After-tax cost of debt: 4.5% × (1 - 0.30) = 3.15%
│       │   └─ Use 3.15% in WACC (not 4.5%)
│       ├─ Validation:
│       │   ├─ Compare to peer cost of debt (similar industry/credit quality)
│       │   ├─ Benchmark: Investment grade (BBB) typically 2-5% after-tax
│       │   ├─ Sanity check: Rd > Rf (debt more risky than Treasuries)
│       │   └─ Consistency: Rd < Re (equity more risky than debt; higher rate required)
│       └─ Sensitivity:
│           ├─ Changes in market rates affect all companies (parallel shift)
│           ├─ Changes in company credit quality affect idiosyncratic spread
│           ├─ Leverage changes affect company-specific spread
│           └─ Key: Monitor both market rates and company credit metrics
├─ Capital Structure Weights (V):
│   ├─ Definition: Market value proportions of equity and debt
│   ├─ Calculation:
│   │   ├─ Market value of equity: Stock price × Shares outstanding
│   │   │   ├─ Example: $50/share × 100M shares = $5B equity value
│   │   ├─ Market value of debt: Book value or market value (if traded)
│   │   │   ├─ Example: $2B debt outstanding
│   │   ├─ Total value: $5B + $2B = $7B
│   │   ├─ Weights:
│   │   │   ├─ E/V = $5B / $7B = 71.4% (equity weight)
│   │   │   ├─ D/V = $2B / $7B = 28.6% (debt weight)
│   │   │   └─ Total: 71.4% + 28.6% = 100%
│   │   └─ Note: Use market values, not book values (future cash flows based on market values)
│   ├─ Preferred stock & other:
│   │   ├─ If company has preferred stock:
│   │   │   ├─ Calculate preferred value separately
│   │   │   ├─ Estimate preferred cost (typically between debt and equity)
│   │   │   ├─ Include in WACC: WACC = E/V × Re + D/V × Rd(1-T) + P/V × Rp
│   │   │   └─ Usually small; often omitted for simplicity
│   │   └─ Minority interests & other:
│   │       ├─ If subsidiary: Include minority stake value
│   │       ├─ Operating leases: Some treat as debt-like
│   │       └─ Simplification: Focus on major equity and debt only
│   ├─ Target vs. Current Capital Structure:
│   │   ├─ Current structure: Actual market values today
│   │   │   ├─ Reflects historical decisions; may be temporary
│   │   │   ├─ Use if company in stable state
│   │   │   └─ Example: Company just completed financing; structure stable
│   │   ├─ Target structure: Long-term optimal capital structure
│   │   │   ├─ Company may be transitioning toward target
│   │   │   ├─ Use if expecting leverage changes (debt paydown or issuance)
│   │   │   ├─ Example: Company issued debt; plans to use for acquisition (temporary high leverage)
│   │   │   └─ Approach: Use target structure in WACC; phased transition
│   │   ├─ Industry average structure:
│   │   │   ├─ If company is outlier, may gravitate toward average
│   │   │   ├─ Example: Retail company with D/E 0.3 vs. industry average 0.6
│   │   │   ├─ May indicate underleverage; could increase debt (raise costs, increase leverage)
│   │   │   └─ Use peer average if company likely to normalize
│   │   └─ Typical choices:
│   │       ├─ Conservative: Use current structure (lower estimation error risk)
│   │       ├─ Base case: Use target/normalized structure
│   │       ├─ Aggressive: Use industry average or optimal structure
│   │       └─ Sensitivity: Calculate at ±10% debt level (bound analysis)
│   └─ Valuation Impact of Capital Structure:
│       ├─ As leverage increases: Re increases (equity more risky)
│       ├─ As leverage increases: Rd may increase (credit risk rises)
│       ├─ WACC often U-shaped (optimization exists):
│       │   ├─ Low leverage: Low after-tax cost (benefit of tax shield; minimal default risk)
│       │   ├─ Moderate leverage: Optimal (lowest WACC; tax benefit + manageable risk)
│       │   ├─ High leverage: Rising costs (default risk > tax benefit; equity requires high return)
│       │   └─ Implication: Company should optimize leverage for lowest WACC
│       ├─ Industry norms:
│       │   ├─ Capital-intensive (utilities, REITs): 50-70% debt acceptable
│       │   ├─ Tech/growth (limited cash flow): 0-20% debt typical
│       │   ├─ Financial (banks): 85-95% "leverage" normal (different context)
│       │   └─ Context: Interpretation differs by industry
│       └─ Stress scenario:
│           ├─ If recession likely: Leverage increases financial distress risk
│           ├─ Add buffer: Use higher leverage scenario in sensitivity
│           └─ Example: Current D/E 0.5; stress scenario D/E 0.75
├─ WACC Calculation & Application:
│   ├─ Formula:
│   │   ├─ WACC = (E/V × Re) + (D/V × Rd × (1 - T))
│   │   ├─ Example calculation:
│   │   │   ├─ E/V = 70% (equity weight)
│   │   │   ├─ Re = 9% (cost of equity from CAPM)
│   │   │   ├─ D/V = 30% (debt weight)
│   │   │   ├─ Rd = 5% (cost of debt)
│   │   │   ├─ T = 30% (tax rate)
│   │   │   ├─ WACC = 0.70 × 0.09 + 0.30 × 0.05 × (1 - 0.30)
│   │   │   ├─ WACC = 0.063 + 0.030 × 0.70
│   │   │   ├─ WACC = 0.063 + 0.021 = 0.084 = 8.4%
│   │   │   └─ Interpretation: Company should discount cash flows at 8.4%
│   │   └─ Variations:
│   │       ├─ If includes preferred: + (P/V × Rp)
│   │       ├─ If uses different tax rates: Adjust (1 - T) per component
│   │       └─ Simplification: Often tax rate assumed same for all debt
│   ├─ DCF application:
│   │   ├─ Enterprise Value = Sum of [FCF(t) / (1 + WACC)^t] + Terminal Value / (1 + WACC)^N
│   │   ├─ Example: FCF Y1 $100M, FCF Y2 $110M, Terminal Value $1,500M, WACC 8.4%
│   │   │   ├─ PV(Y1) = $100M / 1.084 = $92.3M
│   │   │   ├─ PV(Y2) = $110M / (1.084^2) = $93.6M
│   │   │   ├─ PV(TV) = $1,500M / (1.084^2) = $1,279M
│   │   │   ├─ Enterprise Value ≈ $1,465M
│   │   │   └─ WACC critical: 1% change in WACC → ~12% change in EV
│   │   └─ Sensitivity: WACC often largest value driver
│   ├─ Valuation impact scenarios:
│   │   ├─ If WACC 7.4% (vs. 8.4%): EV increases ~13% (lower discount rate)
│   │   ├─ If WACC 9.4% (vs. 8.4%): EV decreases ~11% (higher discount rate)
│   │   ├─ Range: 13% upside / 11% downside from WACC uncertainty
│   │   └─ Implication: WACC estimation critical to valuation
│   └─ Periodicity:
│       ├─ WACC recalculated:
│       │   ├─ At start of valuation (baseline)
│       │   ├─ Quarterly (if market rates change, leverage changes)
│       │   ├─ When company capital structure shifts significantly
│       │   └─ When strategic changes affect risk profile
│       ├─ Typical practice: Annual refresh; quarterly if major events
│       └─ Update components:
│           ├─ Rf: Update to current Treasury yield
│           ├─ β: Update if recent data warrants (usually annually)
│           ├─ ERP: Update to current market environment
│           ├─ Rd: Update based on company's current credit profile
│           ├─ E/V, D/V: Update to current market values
│           └─ T: Update if tax law changes expected
├─ Advanced Considerations:
│   ├─ Multi-period WACC:
│   │   ├─ If capital structure changing over time (e.g., company deleveraging)
│   │   ├─ Use different WACC each period:
│   │   │   ├─ Years 1-3: WACC 9% (high leverage)
│   │   │   ├─ Years 4-5: WACC 8% (moderate leverage)
│   │   │   ├─ Terminal: WACC 7.5% (target leverage, stable)
│   │   │   └─ DCF: Discount each period at appropriate WACC
│   │   └─ Use case: Leveraged buyout (high debt initially; pay down over time)
│   ├─ Unlevered vs. Levered WACC:
│   │   ├─ Unlevered WACC (no debt):
│   │   │   ├─ WACC_u = Re(unlevered) = Rf + β_u × ERP
│   │   │   ├─ Represents pure operating risk; no financial leverage
│   │   │   ├─ Use: Valuing unlevered businesses or comparing capital structures
│   │   │   └─ Example: Compare acquisition targets with different leverage
│   │   ├─ Levered WACC (with debt):
│   │   │   ├─ WACC_l = (E/V × Re) + (D/V × Rd × (1-T))
│   │   │   ├─ Reflects actual company capital structure
│   │   │   ├─ Use: Standard DCF valuation
│   │   │   └─ Relationship: WACC_l ≤ WACC_u (tax shield benefit)
│   │   └─ Flexibility:
│   │       ├─ Start with unlevered WACC (company agnostic)
│   │       ├─ Relever to company's target capital structure
│   │       └─ Allows comparability across different leverage levels
│   └─ Adjusted WACC (for special cases):
│       ├─ Distressed company: Increase WACC to reflect bankruptcy risk
│       ├─ Startup: Increase WACC for execution risk (often 2-5% premium)
│       ├─ Illiquid company: Increase WACC for lack of liquidity (illiquidity discount)
│       ├─ Foreign company: Adjust for country risk (add to Rf or increase ERP)
│       └─ Multiple: May be 2-5% higher than standard WACC
```

**Key Insight:** WACC reflects true cost of capital; small changes have large valuation impact; components (Rf, β, ERP, Rd, T, leverage) must be carefully estimated

## 5. Mini-Project
[Code would include: CAPM calculation, beta estimation, WACC computation, sensitivity to components, comparison across capital structures]

## 6. Challenge Round
When WACC analysis breaks down:
- **Beta obsolescence**: Company restructured; old β from pre-restructure no longer valid; using stale β overstates risk (if changed for the better) or understates (if worse)
- **Tax rate volatility**: Company assumed 30% normalized tax; but has $500M NOL carryforwards; true tax rate next 2 years = 8%; using 30% → WACC too high; undervalues near-term cash
- **Leverage trap**: Current leverage 60%; using current weights; but company plans to go private (100% leverage); future WACC undefined (all equity return required; infinity)
- **ERP bubble**: Historical 5% ERP; but current market P/E extreme (40×); forward returns may be 2%; using 5% ERP → WACC too low; valuation too high
- **Credit spread compression**: Bond spread 200 bps; appears stable; but true credit risk rising (leverage increasing); spread reversion to 400 bps likely next; current Rd too low
- **Multi-currency confusion**: Global company; revenue 60% USD, 40% EUR; using USD WACC; but EUR depreciation likely; all FCF in real EUR terms lower; model misses currency risk

## 7. Key References
- [CFA Institute - Cost of Capital](https://www.cfainstitute.org/) - CAPM, WACC best practices, standards
- [Aswath Damodaran - WACC Tutorial](https://pages.stern.nyu.edu/~adamodar/) - Detailed examples, beta estimation, tax effects
- [Ibbotson Associates - Equity Risk Premium](https://www.ibbotson.com/) - Historical data, forward estimates, professional grade

---
**Status:** Cost of capital | **Complements:** DCF Valuation, Beta Estimation, Capital Structure Optimization, Discount Rate Sensitivity
