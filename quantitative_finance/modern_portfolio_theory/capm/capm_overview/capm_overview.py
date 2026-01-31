import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Parameters
start = "2015-01-01"
end = "2024-12-31"

assets = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
market = "SPY"
rf_ticker = "^IRX"  # 3M T-bill proxy

# Download prices
prices = yf.download(assets + [market], start=start, end=end, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()

# Risk-free rate (monthly)
rf = yf.download(rf_ticker, start=start, end=end, auto_adjust=True)["Close"] / 100 / 12
rf = rf.reindex(returns.index).fillna(method="ffill")

# Excess returns
market_excess = returns[market] - rf

# Estimate CAPM betas
betas = {}
alphas = {}

for asset in assets:
    y = returns[asset] - rf
    X = sm.add_constant(market_excess)
    model = sm.OLS(y, X).fit()
    betas[asset] = model.params[1]
    alphas[asset] = model.params[0] * 12 * 100  # annualized alpha

# Print results
print("CAPM Beta & Alpha Estimates")
print("-" * 50)
for asset in assets:
    print(f"{asset}: beta={betas[asset]:.2f}, alpha={alphas[asset]:.2f}%")

# Plot SML approximation
risk_premium = market_excess.mean() * 12
rf_annual = rf.mean() * 12

beta_grid = np.linspace(0, 2, 50)
expected_returns = rf_annual + beta_grid * risk_premium

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter assets
for asset in assets:
    exp_return = (returns[asset].mean() * 12)
    ax.scatter(betas[asset], exp_return, s=80, label=asset)

# SML line
ax.plot(beta_grid, expected_returns, color="black", linewidth=2, label="SML")

ax.set_xlabel("Beta", fontsize=12)
ax.set_ylabel("Expected Return (annual)", fontsize=12)
ax.set_title("CAPM Security Market Line (Estimated)", fontweight="bold")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("capm_sml_estimated.png", dpi=300, bbox_inches="tight")
print("\n✓ Chart saved: capm_sml_estimated.png")
plt.show()