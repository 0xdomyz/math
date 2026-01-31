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