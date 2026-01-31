# Intertemporal CAPM (ICAPM) & Consumption CAPM (CCAPM)

## 1. Concept Skeleton
**ICAPM:** Multi-period extension of CAPM (Merton 1973). Investors care about both terminal wealth and future investment opportunities; optimal portfolios include **myopic demand** (standard CAPM risk-return tradeoff) plus **hedging demand** against changes in state variables (e.g., interest rates, inflation, volatility).  
**CCAPM:** Consumption-based model where asset prices depend on covariance with aggregate consumption growth (Breeden 1979). Risk is not variance of returns but how returns co-move with marginal utility of consumption.  
**Equity Premium Puzzle:** CCAPM with reasonable risk aversion (γ≈2-4) predicts much smaller equity premium than observed (6-7%).  
**Prerequisites:** CAPM, utility theory (CRRA), stochastic discount factor, dynamic programming

---

## 2. Comparative Framing

| Aspect | CAPM (Single-Period) | ICAPM (Intertemporal) | CCAPM (Consumption-Based) | APT (Arbitrage Pricing) | Fama-French (Empirical) |
|--------|----------------------|-----------------------|---------------------------|-------------------------|--------------------------|
| **Horizon** | One period | Multi-period | Multi-period with consumption | Multi-factor (static) | Empirical factor model |
| **Risk Factor** | Market return | Market + state variables | Covariance with consumption growth | Arbitrage factors | Size, value, profitability, investment |
| **Pricing Kernel** | $M=1-\beta R_m$ (implicit) | State-dependent SDF | $M_{t+1}=\beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma}$ | Linear factor pricing | Empirical regressions |
| **Hedging Demand** | None | Yes (state variables) | Yes (consumption risks) | Indirect (factor exposures) | Indirect |
| **Data Requirements** | Market returns | Returns + state variables | Consumption data (macro) | Factor returns | Factor returns |
| **Empirical Fit** | Weak (SML too flat) | Better (state variables help) | Weak (equity premium puzzle) | Moderate | Strong (explains anomalies) |
| **Interpretation** | Risk = market covariance | Risk = market + opportunity shifts | Risk = consumption covariance | Arbitrage relations | Empirical regularities |
| **Complexity** | Low | High | High (macro data) | Medium | Medium |
| **Practical Use** | Benchmark beta | Strategic asset allocation | Macro asset pricing research | Risk management | Portfolio construction |

**Key Insight:** ICAPM generalizes CAPM by adding hedging demands for changing investment opportunities; CCAPM provides deep theoretical foundation but struggles empirically due to equity premium puzzle.

---

## 3. Examples + Counterexamples

**Example 1: ICAPM Hedging Demand (Interest Rate Risk)**

Investor with long horizon cares about future interest rates (state variable). If rates fall, expected bond returns rise, and reinvestment opportunities improve.

- Asset A (long-term bonds): Positive exposure to falling rates
- Asset B (equities): Low exposure to rates

**ICAPM implication:** Investor holds **more bonds** than CAPM suggests because bonds hedge against rate declines, which hurt future investment opportunities.

**Formula:**
$$E[R_i] - r_f = \beta_{iM} \lambda_M + \beta_{iZ} \lambda_Z$$
- $Z$ = state variable (interest rates)
- $eta_{iZ}$ captures asset's sensitivity to rates

**Result:** Asset with negative exposure to state variables (bad hedges) earns higher expected return; good hedges earn lower expected returns.

---

**Example 2: CCAPM Pricing (Consumption Covariance)**

Assume two assets:
- Asset X: High returns in recessions (when consumption low)
- Asset Y: High returns in booms (when consumption high)

**Risk interpretation:**
- Asset X is **valuable insurance** (pays when marginal utility high)
- Asset Y is risky (pays when marginal utility low)

**CCAPM prediction:**
- Asset X should have **lower expected return** (investors accept lower premium)
- Asset Y should have **higher expected return** (requires compensation)

**Consumption beta:**
$$\beta^C_i = \frac{\text{Cov}(R_i, \Delta C)}{\text{Var}(\Delta C)}$$

**Expected return:**
$$E[R_i] - r_f = \gamma \cdot \text{Cov}(R_i, \Delta C)$$

**Implication:** Risk is covariance with consumption, not market variance.

---

**Example 3: Equity Premium Puzzle**

Data (US, 1889–2024):
- Equity premium ≈ 6-7% p.a.
- Consumption growth volatility ≈ 1-2% p.a.

**CCAPM predicts:**
$$E[R_m] - r_f = \gamma \sigma_C^2$$

If $\sigma_C = 0.02$, then:
$$6\% \approx \gamma (0.02)^2 = \gamma (0.0004)$$
$$\gamma \approx 150$$

**Problem:** Risk aversion 150 is implausibly high (typical estimates 2-10).  
**Conclusion:** CCAPM fails to explain equity premium with reasonable risk aversion.

---

**Example 4: ICAPM vs CAPM (Small Cap Value)**

Small-cap value stocks historically outperform CAPM predictions.

**CAPM view:**
- Beta ~1.1 → expected return slightly above market

**ICAPM view:**
- Small value stocks correlated with bad state variables (recessions, liquidity shocks)
- Negative hedging properties → require higher expected return

**Implication:** ICAPM can rationalize value premium by including state variables like liquidity, volatility, or credit spreads.

---

**COUNTEREXAMPLE 5: Consumption Data Issues**

Consumption data are:
- Low frequency (quarterly)
- Smoothed (durables; measurement error)
- Aggregated (not individual-level)

**Result:** Empirical CCAPM tests have low power; consumption betas are noisy. This weakens CCAPM fit despite theoretical appeal.

---

## 4. Layer Breakdown

```
ICAPM & CCAPM Architecture:

├─ CAPM Foundation (Single-Period):
│   ├─ Assumes one period, mean-variance investors
│   ├─ Expected return: E[R_i]-r_f = β_i (E[R_m]-r_f)
│   └─ Risk = covariance with market
│
├─ ICAPM (Intertemporal CAPM) - Merton 1973:
│   ├─ Motivation: Investors care about future investment opportunities
│   │   └─ State variables change expected returns, volatilities, interest rates
│   │
│   ├─ State Variables (Z_t):
│   │   ├─ Interest rates (yield curve level/slope)
│   │   ├─ Inflation (CPI, breakevens)
│   │   ├─ Volatility (VIX)
│   │   ├─ Credit spreads (Baa-Aaa)
│   │   └─ Dividend yield, term spread
│   │
│   ├─ ICAPM Expected Return:
│   │   ├─ E[R_i] - r_f = β_iM λ_M + Σ_k β_iZk λ_k
│   │   ├─ β_iM = Cov(R_i, R_m)/Var(R_m)
│   │   └─ β_iZk = Cov(R_i, Z_k)/Var(Z_k)
│   │
│   ├─ Decomposition of Demand:
│   │   ├─ Myopic demand (like CAPM): respond to market risk premium
│   │   └─ Hedging demand: protect against changes in investment opportunities
│   │
│   ├─ Implications:
│   │   ├─ Assets that hedge bad states (recession) earn lower returns
│   │   ├─ Assets correlated with worsening state variables require premiums
│   │   └─ Explains time-varying expected returns
│   │
│   └─ Empirical Evidence:
│       ├─ State variables predict returns (Cochrane 1999)
│       ├─ Term spread, dividend yield significant predictors
│       └─ ICAPM improves fit vs CAPM but not perfect
│
├─ CCAPM (Consumption CAPM) - Breeden 1979:
│   ├─ Core idea: Asset risk = covariance with marginal utility of consumption
│   │   └─ Bad times = low consumption; assets that pay then are valuable
│   │
│   ├─ Stochastic Discount Factor (SDF):
│   │   ├─ M_{t+1} = β (C_{t+1}/C_t)^(-γ)
│   │   ├─ Pricing: E[M_{t+1} R_{i,t+1}] = 1
│   │   └─ Expected return: E[R_i]-r_f = γ Cov(R_i, ΔC)
│   │
│   ├─ Empirical Challenges:
│   │   ├─ Consumption growth smooth → weak covariance
│   │   ├─ Equity premium puzzle (Mehra-Prescott 1985)
│   │   ├─ Risk-free rate puzzle (model predicts too high r_f)
│   │   └─ Measurement error (durables, imputation)
│   │
│   ├─ Extensions:
│   │   ├─ Habit formation (Campbell-Cochrane 1999)
│   │   ├─ Long-run risks (Bansal-Yaron 2004)
│   │   ├─ Rare disasters (Rietz 1988; Barro 2006)
│   │   └─ Heterogeneous agents (incomplete markets)
│   │
│   └─ Practical Use:
│       ├─ Macro asset pricing
│       ├─ Explains cross-asset risk premia theoretically
│       └─ Limited for short-term portfolio decisions
│
└─ Equity Premium Puzzle & Risk-Free Rate Puzzle:
    ├─ Equity premium too large for smooth consumption
    ├─ Risk-free rate too high in standard CCAPM
    ├─ Solutions:
    │   ├─ Habit formation → raises effective risk aversion in bad times
    │   ├─ Rare disasters → increases tail risk of equities
    │   ├─ Long-run risks → persistent consumption growth uncertainty
    │   └─ Time-varying risk aversion → explains premiums
    └─ Still open debate; models can fit but may require many parameters
```

---

## 5. Mini-Project: Testing ICAPM vs CCAPM with Market Data

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- Parameters ---
start = "2010-01-01"
end = "2024-12-31"

# Market and asset proxies
assets = {
    "SPY": "Market (S&P 500)",
    "IWM": "Small Cap",
    "EFA": "International",
    "AGG": "Bonds",
    "GLD": "Gold"
}

# State variable proxies (ICAPM)
state_vars = {
    "DGS10": "10Y Treasury Yield",  # Interest rates
    "T10Y2Y": "Term Spread",       # Term spread
}

# Consumption proxy (monthly retail sales growth)
# NOTE: FRED proxy via yfinance may be limited; fallback to SPY dividend yield proxy

print("=" * 100)
print("ICAPM vs CCAPM EMPIRICAL TEST")
print("=" * 100)

# 1. Download price data
prices = yf.download(list(assets.keys()), start=start, end=end, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()

# 2. Market and risk-free rate (proxy: 3M T-bill via ^IRX)
rf = yf.download("^IRX", start=start, end=end, auto_adjust=True)["Close"] / 100 / 12
rf = rf.reindex(returns.index).fillna(method='ffill')

# 3. ICAPM state variables (FRED data via yfinance)
try:
    rates_10y = yf.download("^TNX", start=start, end=end, auto_adjust=True)["Close"] / 100
    term_spread = rates_10y - rf * 12
    icapm_data = pd.DataFrame({"Rate10Y": rates_10y, "TermSpread": term_spread}).reindex(returns.index)
except:
    icapm_data = pd.DataFrame({"Rate10Y": np.nan, "TermSpread": np.nan}, index=returns.index)

icapm_data = icapm_data.fillna(method='ffill').fillna(method='bfill')

# 4. Consumption proxy (use retail sales growth if available)
# Use SPY dividend yield proxy (simplified)
spy_div = yf.download("SPY", start=start, end=end)["Dividends"]
cons_growth = spy_div.reindex(returns.index).fillna(0)
cons_growth = cons_growth / spy_div.rolling(12).sum().shift(1)  # Dividend yield proxy
cons_growth = cons_growth.replace([np.inf, -np.inf], 0).fillna(0)

# 5. Excess returns
excess_returns = returns.sub(rf, axis=0)
market_excess = excess_returns["SPY"]

# 6. CAPM regression (baseline)
print("\n1. CAPM Regression")
print("-" * 100)

capm_results = {}

for asset in assets.keys():
    if asset == "SPY":
        continue
    y = excess_returns[asset]
    X = sm.add_constant(market_excess)
    model = sm.OLS(y, X).fit()
    capm_results[asset] = model
    print(f"{asset} beta: {model.params[1]:.2f}, alpha: {model.params[0]*12*100:.2f}%")

# 7. ICAPM regression (market + state variables)
print("\n2. ICAPM Regression")
print("-" * 100)

icapm_results = {}

# Use changes in state variables as proxies
state_changes = icapm_data.diff().dropna()
state_changes = state_changes.reindex(excess_returns.index).fillna(0)

for asset in assets.keys():
    if asset == "SPY":
        continue
    y = excess_returns[asset]
    X = pd.concat([market_excess, state_changes], axis=1)
    X.columns = ["Market", "Rate10Y", "TermSpread"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    icapm_results[asset] = model
    print(f"{asset} beta_mkt: {model.params['Market']:.2f}, beta_rate: {model.params['Rate10Y']:.2f}, beta_spread: {model.params['TermSpread']:.2f}")

# 8. CCAPM regression (market vs consumption covariance)
print("\n3. CCAPM Regression")
print("-" * 100)

ccapm_results = {}

for asset in assets.keys():
    if asset == "SPY":
        continue
    y = excess_returns[asset]
    X = sm.add_constant(cons_growth)
    model = sm.OLS(y, X).fit()
    ccapm_results[asset] = model
    print(f"{asset} cons_beta: {model.params[1]:.4f}, alpha: {model.params[0]*12*100:.2f}%")

# 9. Compare model fit (R^2)
print("\n4. MODEL FIT COMPARISON (R^2)")
print("-" * 100)

print(f"{'Asset':<10} {'CAPM':<10} {'ICAPM':<10} {'CCAPM':<10}")
print("-" * 45)

for asset in assets.keys():
    if asset == "SPY":
        continue
    capm_r2 = capm_results[asset].rsquared
    icapm_r2 = icapm_results[asset].rsquared
    ccapm_r2 = ccapm_results[asset].rsquared
    print(f"{asset:<10} {capm_r2:<10.3f} {icapm_r2:<10.3f} {ccapm_r2:<10.3f}")

# 10. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Market vs Asset returns (CAPM)
ax = axes[0, 0]

for asset in ["IWM", "EFA", "AGG", "GLD"]:
    ax.scatter(market_excess, excess_returns[asset], alpha=0.4, label=asset)

ax.set_xlabel('Market Excess Return (SPY)', fontsize=12)
ax.set_ylabel('Asset Excess Return', fontsize=12)
ax.set_title('CAPM: Excess Returns vs Market', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: ICAPM State Variable (Rate changes)
ax = axes[0, 1]

ax.plot(state_changes.index, state_changes['Rate10Y'], label='Δ 10Y Yield', color='#e74c3c')
ax.plot(state_changes.index, state_changes['TermSpread'], label='Δ Term Spread', color='#3498db')
ax.set_title('State Variable Changes (ICAPM)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Consumption Proxy
ax = axes[1, 0]

ax.plot(cons_growth.index, cons_growth.values, color='#2ecc71')
ax.set_title('Consumption Proxy (Dividend Yield Growth)', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

# Plot 4: Model Fit Comparison
ax = axes[1, 1]

assets_list = ["IWM", "EFA", "AGG", "GLD"]
capm_r2s = [capm_results[a].rsquared for a in assets_list]
icapm_r2s = [icapm_results[a].rsquared for a in assets_list]
ccapm_r2s = [ccapm_results[a].rsquared for a in assets_list]

x = np.arange(len(assets_list))
width = 0.25

ax.bar(x - width, capm_r2s, width, label='CAPM', color='#3498db')
ax.bar(x, icapm_r2s, width, label='ICAPM', color='#2ecc71')
ax.bar(x + width, ccapm_r2s, width, label='CCAPM', color='#e74c3c')

ax.set_xticks(x)
ax.set_xticklabels(assets_list)
ax.set_ylabel('R^2')
ax.set_title('Model Fit: CAPM vs ICAPM vs CCAPM', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('icapm_ccapm_model_fit.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: icapm_ccapm_model_fit.png")
plt.show()

# 11. Key Insights
print("\n5. KEY INSIGHTS")
print("=" * 100)
print("""
ICAPM vs CAPM:
├─ ICAPM adds state variables (rates, spreads) → improves fit modestly
├─ Hedging demand explains why low-return assets can still be desirable
├─ Assets that hedge bad states earn lower expected returns
└─ Market beta alone insufficient for multi-period investors

CCAPM:
├─ Theoretical elegance: risk is consumption covariance
├─ Empirical weakness: consumption data smooth → low explanatory power
├─ Equity premium puzzle: requires γ ≈ 100+ to match observed 6-7% premium
└─ Extensions (habit, disasters, long-run risks) improve fit but add complexity

Practical Takeaways:
├─ Strategic allocation: consider macro state variables for long-term risk hedging
├─ ICAPM suggests demand for bonds, inflation hedges, volatility hedges
├─ CCAPM useful for macro pricing, less for tactical portfolio choice
└─ Market timing still difficult; state variables weak predictors short-term
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **ICAPM Hedging Demand:** Investor cares about inflation shocks. Which assets should have positive hedging demand? TIPS, commodities, equities? Explain using ICAPM logic.

2. **CCAPM Risk Interpretation:** Two assets have same market beta. Asset A pays during recessions; Asset B pays during booms. Which should have higher expected return? Use consumption beta reasoning.

3. **Equity Premium Puzzle Calculation:** Using $\sigma_C=0.02$, what risk aversion γ is required to justify a 5% equity premium? Is this plausible?

4. **State Variables:** Identify 3 state variables that predict equity returns. How would you incorporate them into ICAPM regression?

5. **Model Choice:** For a pension fund, is ICAPM or CCAPM more relevant? Consider liability hedging, long horizon, and data availability.

---

## 7. Key References

- **Merton, R.C. (1973).** "An Intertemporal Capital Asset Pricing Model" – ICAPM; hedging demands for changing investment opportunities.

- **Breeden, D.T. (1979).** "An Intertemporal Asset Pricing Model with Stochastic Consumption and Investment Opportunities" – CCAPM; consumption covariance pricing.

- **Mehra, R., & Prescott, E.C. (1985).** "The Equity Premium: A Puzzle" – Equity premium puzzle; CCAPM failure with reasonable risk aversion.

- **Campbell, J.Y., & Cochrane, J.H. (1999).** "By Force of Habit" – Habit formation model; resolves equity premium puzzle.

- **Bansal, R., & Yaron, A. (2004).** "Risks for the Long Run" – Long-run risks model; explains high equity premium.

- **Cochrane, J.H. (1999).** "New Facts in Finance" – State variables and predictability of returns.

- **Barro, R.J. (2006).** "Rare Disasters and Asset Markets" – Disaster risk explanation for high equity premium.

