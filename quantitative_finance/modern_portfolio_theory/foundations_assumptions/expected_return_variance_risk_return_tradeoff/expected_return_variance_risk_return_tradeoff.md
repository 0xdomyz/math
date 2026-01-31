# Expected Return, Variance & Risk-Return Tradeoff

## 1. Concept Skeleton
**Definition:** Expected Return: Weighted average of possible outcomes weighted by probability; μ = E[R] = Σ p_i r_i. Variance: Measure of dispersion around mean; σ² = E[(R - μ)²]. Risk-Return Tradeoff: Higher expected return requires accepting higher volatility; fundamental principle establishing all efficient portfolios along boundary curve (efficient frontier).  
**Purpose:** Quantify portfolio return expectations and risk; understand fundamental relationship (higher return → higher risk, not "free lunch"); foundation for all asset pricing and allocation decisions.  
**Prerequisites:** Probability, mean/variance (statistics), correlation, portfolio construction

---

## 2. Comparative Framing

| Concept | Expected Return | Variance | Risk-Return Tradeoff | Portfolio vs Individual Asset |
|---------|-----------------|----------|----------------------|------|
| **Definition** | E[R] = Σ wi E[Ri]; weighted avg future outcomes | σ² = E[(R-μ)²] = Σ (ri - μ)² pi; dispersion measure | Higher μ requires higher σ; trade-off is optimal choice | Portfolio: Weighted avg; Asset: Stand-alone |
| **Estimation** | Historical mean or forward-looking forecast | Historical std dev or model-based forecast | Efficient frontier shows trade-off curve | Individual weights sum to 1; correlation matters |
| **Predictability** | Disputed; historical mean poor predictor | More stable; mean reversion properties | Changes over time; regimes shift | Diversification reduces variance below weighted avg |
| **Horizon** | Longer horizon → More return needed (compensation) | Annualized; square-root-of-time scaling | Time-varying; lifecycle relevant | Multi-period optimization needed |
| **Input Sensitivity** | Highest sensitivity (small μ change → large weight) | Lower sensitivity; covariance more stable | Both inputs matter; forecasting hard | Portfolio less sensitive if diversified |
| **Interpretation** | Reward for investing; compensation for waiting | Risk measure; captured by variance in MPT | Investors choose point on frontier per λ | Covariance term creates diversification benefit |
| **Calculation Complexity** | Simple mean; complex if forecasting | Variance simple; covariance complex for many assets | Optimization needed; quadratic program | Matrix algebra; computational intensity |
| **Forward-looking** | Most uncertain; forward estimates diverge 2-3% | Moderately uncertain; regime changes affect σ | Regime-dependent; crisis regimes steep | Correlations unstable; decline in upmarkets |
| **Use in Practice** | CAPM, DCF, asset pricing models | Risk management, portfolio optimization, VaR | Asset allocation, rebalancing decisions | Fundamental to Markowitz; drives diversification |

**Key Insight:** Expected return hardest to estimate; variance somewhat easier; tradeoff is reality but not linear (diversification bends curve inward, improving risk-return).

---

## 3. Examples + Counterexamples

**Example 1: Computing Expected Return (Scenario Analysis)**

Investment: Single stock with three possible outcomes next year:

| Scenario | Probability | Return | Contribution |
|----------|------------|--------|--|
| Recession | 20% | -20% | 0.2 × (-0.20) = -0.04 |
| Base case | 60% | 12% | 0.6 × 0.12 = 0.072 |
| Boom | 20% | 35% | 0.2 × 0.35 = 0.07 |
| **Expected Return** | **100%** | **-** | **E[R] = 0.002 = 0.2%** |

**Variance calculation:**
- (R - E[R])² × p for each scenario:
  - Recession: (-0.20 - 0.002)² × 0.2 = 0.0404 × 0.2 = 0.00808
  - Base case: (0.12 - 0.002)² × 0.6 = 0.01398 × 0.6 = 0.00839
  - Boom: (0.35 - 0.002)² × 0.2 = 0.1212 × 0.2 = 0.02424
- **σ² = 0.00808 + 0.00839 + 0.02424 = 0.04071**
- **σ = √0.04071 = 20.2%**

**Risk-return trade-off:** Expected return only 0.2%, but volatility 20%! Negative expected value + high risk.

**Implication:** This stock is expensive; expected return too low for 20% volatility. Requires 0.2% + risk premium (20% × λ) to justify.

---

**Example 2: Portfolio Expected Return & Variance (Diversification Benefit)**

Two-asset portfolio:
- Stock A: E[RA] = 10%, σA = 20%
- Stock B: E[RB] = 6%, σB = 10%
- Correlation: ρ = 0.2
- Allocation: 50% A, 50% B

**Expected return:**
$$E[R_p] = 0.5 × 10% + 0.5 × 6% = 8%$$

**Variance:**
$$\sigma_p^2 = (0.5)^2 × (20%)^2 + (0.5)^2 × (10%)^2 + 2 × 0.5 × 0.5 × 0.2 × 20% × 10%$$
$$= 0.01 + 0.0025 + 0.002 = 0.0145$$
$$\sigma_p = 12.04%$$

**Without diversification benefit (if additive):**
- Equal-weighted volatility: (0.5 × 20% + 0.5 × 10%) = 15% (WRONG)

**Actual (with correlation benefit):**
- Portfolio volatility: 12.04% (ACTUAL)

**Diversification benefit:**
- Risk reduction: (15% - 12.04%) / 15% = **19.7% reduction!**
- Return reduction: (8% - 10%) / 10% = 20% reduction (small return cost for large risk reduction)
- **Trade-off favorable:** 20% return drop for 20% risk drop (same ratio; but from different levels)

**Sharpe ratio improvement:**
- Stock A alone: (10% - 2.5%) / 20% = 0.375
- Portfolio: (8% - 2.5%) / 12.04% = 0.456 (BETTER!)

**Implication:** Diversification can improve Sharpe ratio even if returns lower (covariance benefit dominates).

---

**Example 3: Risk-Return Tradeoff (Efficient Frontier)**

Market with three assets:
- Asset A: E[R] = 12%, σ = 25%, β = 1.5
- Asset B: E[R] = 8%, σ = 15%, β = 1.0
- Asset C: E[R] = 5%, σ = 8%, β = 0.5
- Risk-free: rf = 2.5%

**All-in strategies:**
- 100% A: Return 12%, risk 25%, Sharpe = (12% - 2.5%) / 25% = 0.38
- 100% B: Return 8%, risk 15%, Sharpe = (8% - 2.5%) / 15% = 0.37
- 100% C: Return 5%, risk 8%, Sharpe = (5% - 2.5%) / 8% = 0.31
- 100% rf: Return 2.5%, risk 0%, Sharpe = undefined (infinite for 0% risk!)

**Blended portfolio (60% A, 40% B):**
- Expected return: 0.6 × 12% + 0.4 × 8% = 10.4%
- Risk (assuming ρ = 0.6): √[(0.6)² × 25² + (0.4)² × 15² + 2 × 0.6 × 0.4 × 0.6 × 25 × 15] = 19.3%
- Sharpe: (10.4% - 2.5%) / 19.3% = 0.41

**Portfolio with rf (40% rf, 60% blended):**
- Expected return: 0.4 × 2.5% + 0.6 × 10.4% = 7.24%
- Risk: 0.6 × 19.3% = 11.58%
- Sharpe: (7.24% - 2.5%) / 11.58% = 0.41 (same!)

**Efficient frontier trade-offs:**
- Lower risk → Lower return (always trade off)
- Different correlations affect curvature
- Tangency portfolio best; all investors should hold it + rf

---

**Example 4: Expected Return Estimation (Forward-Looking vs Historical)**

S&P 500 historical (1926-2023):
- Mean return: ~10% p.a.
- Volatility: ~18% p.a.

Forecast for next 10 years (as of 2024):
- Earnings yield approach: Corporate earnings / market cap = 5% (lower than historical average)
- Implied return: 5% (base) + 2% (dividend yield) + 0% (multiple expansion) = 7%
- Capital gains only: 7% (no multiple expansion expected)
- Conservative estimate: 6-7% (below historical 10%)

**Reason for lower forecast:**
- Market valuations high (P/E ratios elevated)
- Low interest rates less likely (rates up from zero)
- Earnings growth modest (aging population, productivity)
- Historical mean-reversion suggests above-average returns unlikely

**Implication:** Using historical 10% for next decade would be optimistic; 6-7% more realistic.

**Portfolio allocation impact:**
- Using 10% forecast: Allocate 80% stocks, 20% bonds
- Using 7% forecast: Allocate 60% stocks, 40% bonds (more conservative)
- **Huge difference from small change in expected return!**

---

**COUNTEREXAMPLE 5: Risk NOT Constant (Heteroskedasticity)**

Assumption: σ² constant over time. Reality: Volatility clusters.

2008-2009 financial crisis:
- Pre-crisis (2007): σ_annual ≈ 15%
- Worst period (Sept-Oct 2008): σ_daily > 3% (equivalent to 50%+ annualized!)
- Recovery (2010): σ_annual ≈ 12%

**Implication:**
- Risk-return calculation assumes σ is stable
- During crises, actual σ surges (portfolio riskier than calculated)
- Back-testing with historical σ underestimates crisis risk
- Dynamic models (GARCH, jump diffusion) capture this

**COUNTEREXAMPLE 6: Non-Linear Risk-Return (Assets Outside Efficient Frontier)**

Theory: All efficient portfolios on frontier. Reality: Some portfolios not on frontier (inferior).

Scenario: Holding concentrated single stock.
- Stock A alone: E[R] = 15%, σ = 40%
- Portfolio (A + diversifiers): E[R] = 12%, σ = 18%

**Both on efficient frontier?**
- Stock A: Return higher, risk higher
- Portfolio: Return lower, risk lower (for given return, portfolio more efficient)

**True efficient frontier:** Portfolio beats stock on risk-return basis (higher Sharpe ratio).

**Why investors hold Stock A:**
- Behavioral (overconfidence, concentration bias)
- Constraints (inheritance, founder holding)
- Private information (believes higher return justified)

**Implication:** Markets NOT perfectly efficient; some investors hold inferior portfolios.

---

## 4. Layer Breakdown

```
Expected Return, Variance & Risk-Return Tradeoff: Comprehensive Analysis:

├─ Expected Return (Definition & Estimation):
│   ├─ Theoretical Definition:
│   │   │ E[R] = Σ p_i r_i (probability-weighted average of outcomes)
│   │   │
│   │   ├─ Properties:
│   │   │   ├─ Linear operator: E[aR + b] = aE[R] + b
│   │   │   ├─ Portfolio: E[Rp] = Σ w_i E[R_i] (weighted average)
│   │   │   ├─ Bounded: min(R_i) ≤ E[Rp] ≤ max(R_i)
│   │   │   └─ Unbiased: E[Rp] doesn't change with correlations
│   │   │
│   │   └─ Interpretation:
│   │       ├─ Compensation for investing (waiting, risk-taking)
│   │       ├─ Reward for deferring consumption today
│   │       ├─ Higher E[R] required to justify higher risk (risk-return tradeoff)
│   │       └─ Portfolio E[R] NOT helped by diversification (linear)
│   │
│   ├─ Estimation Methods:
│   │   ├─ (1) Historical average:
│   │   │   ├─ Data: Past returns (5-10 years typical)
│   │   │   ├─ Formula: μ_hist = (1/N) Σ r_t
│   │   │   ├─ Pros: Objective; uses actual data
│   │   │   ├─ Cons: Past ≠ future; subject to sample period choice
│   │   │   └─ Example: Averaging S&P 500 returns (1926-2024) = 10%; but current regime different
│   │   │
│   │   ├─ (2) Forward-looking (implied from current prices):
│   │   │   ├─ Dividend discount model: P = D/(r - g) → Solve for r
│   │   │   │   └─ r = D/P + g (dividend yield + growth rate)
│   │   │   ├─ Example: S&P 500 div yield 1.5% + growth 3% = 4.5%
│   │   │   ├─ Bond yield + equity risk premium:
│   │   │   │   └─ E[Rm] = rf + ERP (risk-free rate + equity premium)
│   │   │   ├─ Building-block approach (survey):
│   │   │   │   └─ Ask investors; use consensus forecast
│   │   │   ├─ Pros: Forward-looking; adjusts for current valuations
│   │   │   └─ Cons: Requires assumptions; sensitive to inputs
│   │   │
│   │   ├─ (3) Factor models:
│   │   │   ├─ CAPM: E[Ri] = rf + β(E[Rm] - rf)
│   │   │   ├─ Fama-French: E[Ri] = rf + βm(E[Rm] - rf) + βs(E[Rsmb] - rf) + βv(E[Rhml] - rf)
│   │   │   ├─ Pros: Decompose return into factors; update as market changes
│   │   │   └─ Cons: Requires factor forecasts (multiple inputs)
│   │   │
│   │   └─ (4) Scenario analysis:
│   │       ├─ Specify outcomes: Recession, base, expansion
│   │       ├─ Assign probabilities and returns for each
│   │       ├─ E[R] = Σ p_scenario × r_scenario
│   │       ├─ Pros: Explicit; intuitive; can include subjective views
│   │       └─ Cons: Subjective; prone to bias
│   │
│   ├─ Historical Mean Bias:
│   │   ├─ Problem: Past return ≠ future expected return
│   │   ├─ Reasons:
│   │   │   ├─ Regime changes: 1926-1945 (bull), 1965-1980 (bear), 1980-2000 (bull), etc.
│   │   │   ├─ Valuation reversion: High valuations → lower future returns (mean reversion)
│   │   │   ├─ Survivorship bias: Dead companies excluded (positive bias)
│   │   │   └─ Selection period: Cherry-picking favorable periods
│   │   │
│   │   ├─ Evidence:
│   │   │   ├─ Bogle (1991): Predicts 4-6% returns; S&P 500 averaged 10% (2000-2024)
│   │   │   ├─ But then predicted 5-7%; S&P 500 averaged 11% (2009-2019)
│   │   │   └─ Conclusion: Historical average not reliable; current valuations matter
│   │   │
│   │   └─ Implication:
│   │       ├─ Don't blindly use 10% historical average
│   │       ├─ Adjust for current valuations (lower valuations → higher forward returns)
│   │       ├─ Use building-block or dividend-based approaches
│   │       └─ Update expectations as market changes
│   │
│   ├─ Forward-Looking Expected Returns (Valuation-Based):
│   │   │ E[R] ≈ Dividend yield + Expected growth + Multiple expansion
│   │   │
│   │   ├─ Components:
│   │   │   ├─ Dividend yield (observable today):
│   │   │   │   ├─ 2024 S&P 500: ~1.5% (higher if buy value stocks; lower if growth)
│   │   │   │   ├─ Historical average: 2-3% (now below average)
│   │   │   │   └─ Impacts: Lower yield stocks → Lower expected return (if yield reverts)
│   │   │   │
│   │   │   ├─ Expected growth rate (earnings/dividends):
│   │   │   │   ├─ GDP growth proxy: ~2-3% long-term (mature economy)
│   │   │   │   ├─ Better estimates: Industry-specific; analyst consensus
│   │   │   │   ├─ Consensus for S&P 500: ~6-8% earnings growth
│   │   │   │   └─ Example: If earnings grow 7%, dividends can grow ~7%
│   │   │   │
│   │   │   └─ Multiple expansion (P/E ratio change):
│   │   │       ├─ 2024: P/E ~20-22 (near historical average)
│   │   │       ├─ If multiples expand to 25: Additional return; if contract to 18: Drag
│   │   │       ├─ Assumption for long-term: Multiples stable (no expansion/contraction)
│   │   │       └─ Long-term return from multiple expansion: Approximately zero
│   │   │
│   │   ├─ Calculation example:
│   │   │   ├─ Dividend yield: 1.5%
│   │   │   ├─ Growth rate: 6%
│   │   │   ├─ Multiple expansion: 0% (assumes stable valuation)
│   │   │   └─ E[R] = 1.5% + 6% + 0% = 7.5%
│   │   │
│   │   └─ Sensitivity:
│   │       ├─ If dividend yield up to 2% (valuations decline): E[R] = 8%+
│   │       ├─ If growth down to 4%: E[R] = 5.5%
│   │       └─ Valuation changes create wide forecast range
│   │
│   └─ Horizon-Dependent Expected Returns:
│       ├─ Short-term (1 year):
│       │   ├─ Influenced by business cycle, sentiment, momentum
│       │   ├─ Harder to forecast; high uncertainty
│       │   └─ Example: 2024 could be +20% or -10% (high dispersion)
│       │
│       ├─ Medium-term (3-5 years):
│       │   ├─ Business earnings trends; secular themes
│       │   ├─ Moderate uncertainty
│       │   └─ Example: Likely +5-8% annualized (narrower range)
│       │
│       └─ Long-term (10+ years):
│           ├─ Mean-reversion; dividend growth; economic growth
│           ├─ Lower uncertainty relative to expected return (higher Sharpe)
│           └─ Example: ~6-7% annualized (steady; less volatile)
│
├─ Variance (Definition & Estimation):
│   ├─ Theoretical Definition:
│   │   │ σ² = E[(R - E[R])²] = Σ p_i (r_i - E[R])²
│   │   │
│   │   ├─ Alternative formula (computational):
│   │   │   │ σ² = E[R²] - (E[R])²
│   │   │   │
│   │   │   ├─ Advantage: Two summations instead of squared deviations
│   │   │   ├─ Disadvantage: Sensitive to large values (numerical stability)
│   │   │   └─ Practice: Use deviation formula for stability
│   │   │
│   │   ├─ Properties:
│   │   │   ├─ Non-negative: σ² ≥ 0 (always)
│   │   │   ├─ Zero iff deterministic: σ² = 0 if R constant
│   │   │   ├─ Scale: Var(aR) = a² Var(R) (quadratic scaling)
│   │   │   │   └─ 2× leverage → 4× variance (big impact!)
│   │   │   ├─ Linearity broken: Var(R + S) ≠ Var(R) + Var(S) if correlated
│   │   │   │   └─ Var(R + S) = Var(R) + Var(S) + 2Cov(R,S)
│   │   │   └─ Independent: Var(R + S) = Var(R) + Var(S) if ρ = 0
│   │   │
│   │   └─ Interpretation:
│   │       ├─ Measures dispersion around mean
│   │       ├─ Higher variance = Riskier (more volatility)
│   │       ├─ Asymmetric: Extreme losses + extreme gains both increase variance
│   │       └─ Downside risk emphasis: Some prefer semi-variance (down-moves only)
│   │
│   ├─ Estimation Methods:
│   │   ├─ (1) Historical sample variance:
│   │   │   ├─ σ²_sample = (1/N) Σ (r_t - r_bar)²
│   │   │   ├─ Or unbiased: σ²_unbias = (1/(N-1)) Σ (r_t - r_bar)² (Bessel correction)
│   │   │   ├─ Pros: Objective; uses actual data
│   │   │   ├─ Cons: Past variance ≠ future (mean-reversion; regime changes)
│   │   │   └─ Example: S&P 500 historical σ ≈ 18%; but varies 12-30% by period
│   │   │
│   │   ├─ (2) GARCH models (volatility clustering):
│   │   │   ├─ Volatility itself random; changes over time
│   │   │   ├─ Model: σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
│   │   │   │   └─ Current vol depends on past shocks + past vol
│   │   │   ├─ Captures: Quiet periods → Violent periods → Calm again
│   │   │   └─ Used for: Risk forecasting, options pricing
│   │   │
│   │   ├─ (3) Implied volatility (from options):
│   │   │   ├─ Invert Black-Scholes; find σ consistent with market prices
│   │   │   ├─ Forward-looking (market's expectation of future vol)
│   │   │   ├─ Pros: Real-time; market consensus
│   │   │   ├─ Cons: Only available for optionable securities; varies by strike
│   │   │   └─ Example: VIX (S&P 500 implied vol) ranges 10-80
│   │   │
│   │   └─ (4) Factor models (systematic + idiosyncratic):
│   │       ├─ σ²_asset = β² σ²_market + σ²_idiosyncratic
│   │       ├─ Separate: Market risk (can't eliminate) + firm-specific (diversifiable)
│   │       └─ Useful: Understand risk decomposition
│   │
│   ├─ Volatility (Standard Deviation):
│   │   │ σ = √(σ²); more interpretable than variance (same units as return)
│   │   │
│   │   ├─ Interpretation:
│   │   │   ├─ ~68% of returns within 1σ (if normal distribution)
│   │   │   ├─ ~95% within 2σ
│   │   │   ├─ ~99.7% within 3σ
│   │   │   └─ S&P 500: σ = 18% annual → ~1.4% daily typical move
│   │   │
│   │   ├─ Annualization (from daily/monthly):
│   │   │   ├─ Daily → Annual: σ_annual = σ_daily × √252 (trading days)
│   │   │   ├─ Monthly → Annual: σ_annual = σ_monthly × √12
│   │   │   ├─ Assumes: Independence of periods (violations in practice)
│   │   │   └─ Example: Daily vol 1% → Annual 15.87% (1% × √252)
│   │   │
│   │   └─ Term structure (volatility across horizons):
│   │       ├─ Short-term (1 day): High variation relative to mean
│       ├─ Long-term (1 year): More stable; compounding reduces relative vol
│       └─ Volatility drag: (1+r-σ²/2) approximation (Jensen's inequality)
│
│   ├─ Covariance & Correlation (In Variance Context):
│   │   │ Portfolio variance includes cross-terms:
│   │   │ σ²_p = Σ Σ w_i w_j σ_i σ_j ρ_ij = Σ Σ w_i w_j σ_ij
│   │   │
│   │   ├─ Role of covariance:
│   │   │   ├─ If ρ = 1 (perfect correlation): No diversification benefit
│   │   │   │   └─ σ_p = Σ w_i σ_i (risk additive)
│   │   │   ├─ If ρ = 0 (uncorrelated): Significant benefit
│   │   │   │   └─ σ_p < Σ w_i σ_i (risk subadditive)
│   │   │   ├─ If ρ = -1 (perfectly inverse): Maximum benefit
│   │   │   │   └─ Can cancel risk (σ_p approaches zero)
│   │   │   └─ Typical stocks: ρ ≈ 0.6-0.8 (moderate correlation)
│   │   │
│   │   ├─ Covariance stationarity (assumption):
│   │   │   ├─ Covariances constant over time (usually violated)
│   │   │   ├─ 2008 crisis: Correlations spiked to near 1 (diversification failed)
│   │   │   ├─ 2022: Rate shocks created stock-bond correlation 0.8 (vs 0.3 normal)
│   │   │   └─ Implication: Use dynamic models for stress scenarios
│   │   │
│   │   └─ Correlation mean-reversion:
│   │       ├─ Long-term average: Correlations tend toward 0.5-0.7
│   │       ├─ Crisis spikes: Up to 0.9+
│   │       ├─ Recovery: Return toward long-term average
│       └─ Use: Can model correlation dynamics for scenarios
│
│   └─ Time-Varying Variance (Heteroskedasticity):
│       ├─ Calm markets: σ ≈ 12-15% (for stocks)
│       ├─ Crisis: σ ≈ 40-50% (10× increase possible)
│       ├─ Important for:
│       │   ├─ Risk management (variance model inadequate)
│       │   ├─ Rebalancing (frequency changes in crises)
│       │   └─ Margin requirements (must adjust as vol changes)
│       └─ Solutions: GARCH models, stress scenarios, adaptive rebalancing
│
├─ Risk-Return Tradeoff (Central Principle):
│   ├─ Statement:
│   │   │ To achieve higher expected return, investors must accept higher risk
│   │   │ No asset class offers high return without high volatility
│   │   │ Efficient frontier encodes this trade-off
│   │   │
│   │   └─ Quantitatively: E[Rp] increasing in σ_p (monotonic relationship)
│   │
│   ├─ Origin of Tradeoff (Why NOT free lunch?):
│   │   ├─ Competition: Investors all want high returns with low risk
│   │   ├─ Equilibrium: Assets with high return must have high risk
│   │   │   └─ If low-risk asset offered high return, everyone buys; price up; return down
│   │   ├─ Market clearing: Supply/demand equilibrate; no arbitrage
│   │   └─ Result: High-return assets risky; low-risk assets low-return
│   │
│   ├─ Efficient Frontier (Graphical Manifestation):
│   │   │ Curve in mean-std space; leftmost = minimum risk; rightmost = max return
│   │   │
│   │   ├─ Properties:
│   │   │   ├─ Concave (when starting from risk-free asset)
│   │   │   │   └─ Slope decreases as move right (diminishing risk-return ratio)
│   │   │   ├─ Upper envelope: Pareto optimal (best for each risk level)
│   │   │   ├─ All portfolios below frontier are suboptimal
│   │   │   └─ Any point on frontier = Different investor λ (different risk tolerance)
│   │   │
│   │   ├─ Curvature depends on correlations:
│   │   │   ├─ Low correlations (ρ ≈ 0): Curved frontier (big diversification benefit)
│   │   │   ├─ High correlations (ρ ≈ 1): Nearly linear (little benefit)
│   │   │   └─ Negative correlations (ρ < 0): Most curved (hedging)
│   │   │
│   │   └─ Capital Allocation Line (CAL):
│   │       ├─ Linear line tangent to frontier at one point (tangency/market portfolio)
│   │       ├─ Slope = Sharpe ratio = (E[Rm] - rf) / σ_m
│   │       ├─ All portfolios on CAL dominate interior portfolios
│   │       └─ All rational investors should use CAL (two-fund separation)
│   │
│   ├─ Quantifying the Tradeoff:
│   │   ├─ Sharpe Ratio:
│   │   │   │ SR = (E[Rp] - rf) / σ_p (return per unit risk)
│   │   │   │
│   │   │   ├─ Higher Sharpe = Better risk-adjusted return
│   │   │   ├─ Tangency portfolio maximizes Sharpe (best tradeoff)
│   │   │   ├─ All investors should hold tangency + risk-free (two-fund separation)
│   │   │   └─ If disagree on Sharpe, then different optimal portfolios
│   │   │
│   │   ├─ Marginal Rate of Substitution (MRS):
│   │   │   │ Indifference curve: U = E[R] - (λ/2)σ² = constant
│   │   │   │ Slope: dE[R]/dσ = λ σ (how much return needed per unit risk increase)
│   │   │   │
│   │   │   ├─ Higher λ: Steeper slope (more risk-averse; wants more return for risk)
│   │   │   ├─ Lower λ: Flatter slope (accepts risk for small return increase)
│   │   │   └─ At optimum: MRS = Sharpe ratio (indifference curve tangent to CAL)
│   │   │
│   │   └─ Elasticity of substitution:
│   │       ├─ If return increases 1%, how much can risk decrease?
│   │       ├─ Depends on frontier curvature (correlation structure)
│   │       └─ Diversification opportunities affect substitution rate
│   │
│   ├─ Evidence for Tradeoff:
│   │   ├─ Cross-asset class:
│   │   │   ├─ T-bills: ~2.5% return, ~0% risk
│   │   │   ├─ Investment-grade bonds: ~4-5% return, ~3-5% risk
│   │   │   ├─ Equities: ~7-10% expected return, ~15-18% risk
│   │   │   ├─ Small stocks: ~11% expected, ~20% risk
│   │   │   ├─ High-yield bonds: ~6-7% return, ~8-10% risk
│   │   │   └─ Crypto: ~15-20% expected(?), ~70-100% risk
│   │   │
│   │   ├─ Within equities:
│   │   │   ├─ Large-cap value: ~8% return, ~15% risk
│   │   │   ├─ Large-cap growth: ~9% return, ~18% risk
│   │   │   ├─ Small-cap: ~10% return, ~20% risk
│   │   │   └─ Emerging markets: ~10-12% return, ~20-25% risk
│   │   │
│   │   └─ Over time:
│   │       ├─ Bull markets (low vol): Lower returns relative to risk taken
│   │       ├─ Crash periods (high vol): Higher returns (mean reversion)
│   │       └─ Implication: Dynamic; changes over time
│   │
│   └─ Violations & Apparent Anomalies:
│       ├─ Low-risk anomaly: Stocks with low beta earned higher Sharpe than high-beta
│       │   └─ Violates CAPM prediction; leverage constraints explanation
│       ├─ Value premium: Low P/B stocks beat high P/B (different risk structure?)
│       │   └─ Related to distress risk, financial constraints
│       ├─ Momentum: Past winners keep winning (trend violates mean-reversion)
│       │   └─ Probably behavioral; herding, underreaction
│       └─ Size premium: Small stocks beat large (small-cap risk?)
│           └─ Mostly reversed; was 3% p.a.; now near zero
│
└─ Portfolio Construction Implications:
    ├─ Risk-Return is fundamental; optimizers use it
    ├─ Expected returns hardest to estimate; most sensitive
    ├─ Variance more stable; correlations least stable
    ├─ Time horizon matters: Longer → Different tradeoff
    ├─ Constraints affect tradeoff: Can't achieve theoretical optimum
    ├─ Costs reduce tradeoff: Must account in optimization
    ├─ Taxes move tradeoff: After-tax different from pre-tax
    └─ Behavioral factors: Investors not always on efficient frontier
```

---

## 5. Mini-Project: Estimating Expected Returns & Building Risk-Return Frontier

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Estimate forward-looking expected returns; construct risk-return relationship

def estimate_forward_returns(ticker, current_price, div_yield, earnings_growth, pe_ratio=None):
    """
    Estimate forward-looking expected return using dividend + growth approach.
    E[R] ≈ Dividend yield + Growth rate + Multiple expansion (assume 0)
    """
    
    total_return = div_yield + earnings_growth
    
    # Adjust for valuation (simple approach)
    historical_pe = 18  # Historical average
    valuation_adjustment = 0
    
    if pe_ratio:
        if pe_ratio > historical_pe:
            # Valuation above average; lower forward return (multiple compression risk)
            valuation_adjustment = -0.01  # -1% adjustment if elevated
        elif pe_ratio < historical_pe:
            # Valuation below average; higher forward return (multiple expansion)
            valuation_adjustment = 0.01   # +1% adjustment if depressed
    
    forward_return = total_return + valuation_adjustment
    
    return {
        'dividend_yield': div_yield,
        'growth_rate': earnings_growth,
        'valuation_adj': valuation_adjustment,
        'forward_return': forward_return
    }


def historical_expected_returns(tickers, start_date, end_date):
    """Estimate returns using historical averages."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns.mean() * 252  # Annualized


def historical_volatility(tickers, start_date, end_date):
    """Estimate volatility from historical data."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns.std() * np.sqrt(252)  # Annualized


def build_risk_return_frontier(expected_returns, cov_matrix, rf=0.025, num_points=50):
    """
    Construct efficient frontier by varying target return.
    Returns list of (volatility, return, weights) tuples.
    """
    
    n = len(expected_returns)
    frontier = []
    
    # Min and max returns
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    for target in target_returns:
        # Minimize variance subject to return and weight constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target}  # Target return
        ]
        
        bounds = tuple((0, 1) for _ in range(n))  # Long-only constraints
        
        # Initial guess: equal weights
        w0 = np.array([1/n] * n)
        
        # Objective: minimize variance
        def objective(w):
            return np.dot(w, np.dot(cov_matrix, w))
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            vol = np.sqrt(result.fun)
            frontier.append({
                'return': target,
                'volatility': vol,
                'weights': result.x
            })
    
    return pd.DataFrame(frontier)


def compute_expected_return_forward(ticker):
    """
    Forward-looking expected return estimation.
    Uses dividend yield + growth assumptions.
    """
    
    tick = yf.Ticker(ticker)
    info = tick.info
    
    # Get data
    div_yield = info.get('dividendYield', 0) or 0
    pe_ratio = info.get('trailingPE', None)
    peg_ratio = info.get('pegRatio', None)
    
    # Growth assumptions (conservative)
    if peg_ratio and peg_ratio > 0:
        implied_growth = info.get('epsTrailingTwelveMonths', 0) / pe_ratio if pe_ratio else 0.06
        earnings_growth = min(0.15, max(0.03, implied_growth))  # Bound between 3-15%
    else:
        earnings_growth = 0.06  # Default 6% growth
    
    return estimate_forward_returns(ticker, None, div_yield, earnings_growth, pe_ratio)


# Main Analysis
print("=" * 100)
print("EXPECTED RETURN, VARIANCE & RISK-RETURN TRADEOFF")
print("=" * 100)

# 1. Data Collection
print("\n1. PORTFOLIO ASSET SELECTION & RETURN ESTIMATION")
print("-" * 100)

tickers = ['SPY', 'QQQ', 'BND', 'AGG', 'GLD']
names = ['S&P 500', 'Tech (Nasdaq)', 'Bond ETF', 'Aggregate Bonds', 'Gold']

# Historical data
hist_returns = historical_expected_returns(tickers, '2015-01-01', '2024-01-01')
hist_vols = historical_volatility(tickers, '2015-01-01', '2024-01-01')

# Forward-looking estimates (adjusted)
# Based on current yields, growth expectations
forward_returns = pd.Series({
    'SPY': 0.075,      # 7.5% (lower than hist due to valuation)
    'QQQ': 0.085,      # 8.5% (tech premium)
    'BND': 0.045,      # 4.5% (bond yield approximately)
    'AGG': 0.042,      # 4.2% (agg bonds)
    'GLD': 0.030       # 3% (gold; minimal return, hedge value)
})

print("\nAsset Return Estimates:\n")
print(f"{'Asset':<20} {'Historical %':<18} {'Forward-Looking %':<20} {'Volatility %':<15}")
print("-" * 73)

for ticker, name in zip(tickers, names):
    print(f"{name:<20} {hist_returns[ticker]*100:<18.2f} {forward_returns[ticker]*100:<20.2f} "
          f"{hist_vols[ticker]*100:<15.2f}")

# 2. Covariance matrix and correlation
print("\n2. RISK STRUCTURE (COVARIANCE & CORRELATION)")
print("-" * 100)

# Calculate covariance matrix from historical data
data = yf.download(tickers, start='2015-01-01', end='2024-01-01', progress=False)['Adj Close']
returns = data.pct_change().dropna()

cov_matrix = returns.cov() * 252
corr_matrix = returns.corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

print(f"\nCovariance Matrix (Annualized):")
print((cov_matrix * 10000).round(0))  # Show in basis points

# 3. Individual asset risk-return
print("\n3. INDIVIDUAL ASSET RISK-RETURN METRICS")
print("-" * 100)

rf = 0.025

print(f"\n{'Asset':<20} {'Return %':<15} {'Volatility %':<18} {'Sharpe Ratio':<15}")
print("-" * 68)

for ticker, name in zip(tickers, names):
    ret = forward_returns[ticker]
    vol = hist_vols[ticker]
    sharpe = (ret - rf) / vol
    print(f"{name:<20} {ret*100:<15.2f} {vol*100:<18.2f} {sharpe:<15.3f}")

# 4. Build efficient frontier
print("\n4. EFFICIENT FRONTIER CONSTRUCTION")
print("-" * 100)

frontier = build_risk_return_frontier(forward_returns, cov_matrix, rf=rf, num_points=30)

print(f"\nGenerated {len(frontier)} efficient portfolios")
print(f"  Min volatility: {frontier['volatility'].min()*100:.2f}%")
print(f"  Max volatility: {frontier['volatility'].max()*100:.2f}%")
print(f"  Return range: {frontier['return'].min()*100:.2f}% - {frontier['return'].max()*100:.2f}%")

# Find maximum Sharpe portfolio
frontier['sharpe'] = (frontier['return'] - rf) / frontier['volatility']
max_sharpe_idx = frontier['sharpe'].idxmax()
max_sharpe_port = frontier.loc[max_sharpe_idx]

print(f"\nMaximum Sharpe Ratio Portfolio:")
print(f"  Return: {max_sharpe_port['return']*100:.2f}%")
print(f"  Volatility: {max_sharpe_port['volatility']*100:.2f}%")
print(f"  Sharpe Ratio: {max_sharpe_port['sharpe']:.3f}")

# 5. Risk-Return Tradeoff Analysis
print("\n5. RISK-RETURN TRADEOFF QUANTIFICATION")
print("-" * 100)

# Calculate incremental risk-return
frontier_sorted = frontier.sort_values('volatility')
frontier_sorted['return_diff'] = frontier_sorted['return'].diff()
frontier_sorted['vol_diff'] = frontier_sorted['volatility'].diff()
frontier_sorted['marginal_return_per_risk'] = frontier_sorted['return_diff'] / frontier_sorted['vol_diff']

print(f"\nMarginal Risk-Return Trade-off (Slope of frontier):")
print(f"{'Volatility %':<20} {'Return %':<15} {'MR/Risk':<15} {'Interpretation':<30}")
print("-" * 80)

sample_indices = [len(frontier_sorted)//4, len(frontier_sorted)//2, 3*len(frontier_sorted)//4]

for idx in sample_indices:
    if idx < len(frontier_sorted):
        row = frontier_sorted.iloc[idx]
        interp = "Steep" if row['marginal_return_per_risk'] > 2 else ("Moderate" if row['marginal_return_per_risk'] > 0.5 else "Flat")
        print(f"{row['volatility']*100:<20.2f} {row['return']*100:<15.2f} "
              f"{row['marginal_return_per_risk']:<15.3f} {interp:<30}")

# 6. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Efficient Frontier with Assets
ax = axes[0, 0]

# Frontier
ax.plot(frontier['volatility'] * 100, frontier['return'] * 100, 'o-', linewidth=2.5, 
       color='#2ecc71', markersize=4, label='Efficient Frontier', zorder=3)

# Individual assets
for ticker, name in zip(tickers, names):
    ret = forward_returns[ticker]
    vol = hist_vols[ticker]
    ax.scatter(vol * 100, ret * 100, s=200, alpha=0.7, label=name, zorder=4, edgecolors='black', linewidth=1)

# CAL (Capital Allocation Line)
cal_vols = np.linspace(0, frontier['volatility'].max() * 1.3 * 100, 100)
cal_returns = rf * 100 + max_sharpe_port['sharpe'] * cal_vols
ax.plot(cal_vols, cal_returns, 'k--', linewidth=1.5, alpha=0.6, label='CAL', zorder=2)

# Max Sharpe portfolio
ax.scatter(max_sharpe_port['volatility'] * 100, max_sharpe_port['return'] * 100, 
          s=300, marker='*', color='red', label='Max Sharpe', zorder=5, edgecolors='black', linewidth=1)

# Risk-free
ax.scatter(0, rf * 100, s=200, marker='s', color='gray', label='Risk-free', zorder=5, edgecolors='black', linewidth=1)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Efficient Frontier: Risk-Return Tradeoff', fontweight='bold', fontsize=13)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)
ax.set_ylim([0, 10])

# Plot 2: Correlation Heatmap
ax = axes[0, 1]

im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(tickers)))
ax.set_yticks(range(len(tickers)))
ax.set_xticklabels([t for t in tickers], rotation=45, ha='right')
ax.set_yticklabels([t for t in tickers])

# Add correlation values
for i in range(len(tickers)):
    for j in range(len(tickers)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', color='black', fontsize=9)

ax.set_title('Correlation Matrix (Diversification Drivers)', fontweight='bold', fontsize=13)
plt.colorbar(im, ax=ax, label='Correlation')

# Plot 3: Frontier Slope (Marginal Risk-Return)
ax = axes[1, 0]

frontier_sorted_plot = frontier_sorted.dropna()
ax.plot(frontier_sorted_plot['volatility'] * 100, 
       frontier_sorted_plot['marginal_return_per_risk'], 
       'o-', linewidth=2, markersize=6, color='#3498db')

ax.set_xlabel('Portfolio Volatility (%)', fontsize=12)
ax.set_ylabel('Marginal Return per % Risk', fontsize=12)
ax.set_title('Risk-Return Tradeoff Slope (Frontier Curvature)', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 4: Sharpe Ratio along Frontier
ax = axes[1, 1]

sharpes = (frontier['return'] - rf) / frontier['volatility']
ax.plot(frontier['volatility'] * 100, sharpes, 'o-', linewidth=2.5, markersize=6, color='#e74c3c')
ax.axhline(y=max_sharpe_port['sharpe'], color='green', linestyle='--', linewidth=2, alpha=0.7, 
          label=f'Max Sharpe: {max_sharpe_port["sharpe"]:.3f}')

ax.set_xlabel('Portfolio Volatility (%)', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Risk-Adjusted Return along Efficient Frontier', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('expected_return_variance_tradeoff.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: expected_return_variance_tradeoff.png")
plt.show()

# 7. Summary
print("\n" + "=" * 100)
print("KEY INSIGHTS:")
print("=" * 100)
print(f"""
EXPECTED RETURN ESTIMATION:
├─ Historical average unreliable (past ≠ future)
├─ Forward-looking approach: Dividend yield + Growth + Multiple expansion
├─ Current valuation matters: High P/E → Lower forward returns
└─ Consensus: S&P 500 ~7-8% forward (vs 10% historical average)

VARIANCE & RISK MEASUREMENT:
├─ Volatility ~15-18% for stocks (annualized)
├─ Bonds ~3-5% volatility (lower risk)
├─ Gold ~15% (hedge; low correlation to stocks)
└─ Time-varying: Volatility clusters; crisis periods 2-3× normal

RISK-RETURN TRADEOFF (FUNDAMENTAL):
├─ Higher expected return REQUIRES higher volatility (not free lunch)
├─ Efficient frontier shows: Optimal allocation by risk aversion
├─ Sharpe ratio: Return per unit risk; higher is better
├─ Two-fund separation: All investors hold market + risk-free
└─ Allocation depends on λ (risk tolerance), not on individual beliefs

DIVERSIFICATION BENEFIT:
├─ Portfolio volatility < weighted average of individual volatilities
├─ Correlation structure critical: Low ρ → More benefit
├─ Optimal allocation: Not equal-weight; considers risk & return
├─ Stocks-bonds: Correlation 0.3 (low); good diversifier
└─ Gold: Correlation 0 (uncorrelated); small allocation helps

FRONTIER PROPERTIES:
├─ Curved shape: Diminishing marginal return per unit risk
├─ Slope decreases moving right (more conservative portfolios have better Sharpe)
├─ Maximum Sharpe portfolio: Optimal risky asset for all rational investors
├─ CAL (Capital Allocation Line): Linear; connects rf + tangency portfolio
└─ All points above frontier: Impossible (violates optimization)

PRACTICAL IMPLICATIONS:
├─ Use forward-looking return estimates (not historical averages)
├─ Correlations change in stress; plan accordingly
├─ Rebalance toward lower vol in high-return environments (take profits)
├─ Tax-aware: After-tax expected returns differ materially
└─ Accept trade-off: Can't have high return with low risk
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Return Estimation Sensitivity:** If change expected return assumption from 8% to 6%, how much does optimal stock allocation change? Should all small return forecast changes change allocation materially?

2. **Variance Forecasting in Crisis:** If volatility jumps from 15% to 50% during crash, portfolio risk 3× higher than calculated. How do you rebalance? Exit positions (crystallize losses) or hold?

3. **Correlation Breakdown:** Stock-bond correlation typically 0.3; in 2022 rate shock it was 0.8. If portfolio designed for ρ=0.3, what was actual risk? How much diversification benefit lost?

4. **Risk-Return Tradeoff in Different Regimes:** In bull market (low vol, positive returns), marginal risk per return seems low (steep frontier). In bear market (high vol, negative returns), frontier flatter. Why? Should allocation change?

5. **Forward Returns vs Historical:** Company has 10% historical average return, but current valuation suggests 5% forward return. Which should use? When does historical make sense vs forward?

---

## 7. Key References

- **Markowitz, H.M. (1952).** "Portfolio Selection" *Journal of Finance* – Foundation of mean-variance optimization; expected return and variance central.

- **Tobin, J. (1958).** "Liquidity Preference as Behavior Towards Risk" – Capital allocation line; two-fund separation theorem.

- **Sharpe, W.F. (1964).** "Capital Asset Prices: A Theory of Market Equilibrium" – CAPM; risk-return equilibrium pricing.

- **Arnott, R.D. & Bernstein, P.L. (2002).** "What Risk Premium is 'Normal'?" *Research Affiliates* – Forward-looking expected returns; departures from historical averages.

- **Blitz, D., Hanauer, M.X., Vidojevic, M., & Zaremba, A. (2021).** "The Elasticity of Expected Returns" *Journal of Portfolio Management* – How expected returns vary with risk; curvature of frontier.

- **Campbell, J.Y. & Shiller, R.J. (1988).** "Stock Prices, Earnings, and Expected Dividends" *Journal of Finance* – Forward-looking returns; valuation metrics.

- **Damodaran, A. (2022).** "Damodaran Online: Expected Returns by Asset Class" – Comprehensive forward-return estimates; building-block approach.

