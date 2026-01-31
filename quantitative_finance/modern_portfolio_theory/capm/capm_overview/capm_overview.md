# Capital Asset Pricing Model (CAPM) — Core Framework

## 1. Concept Skeleton
**Definition:** CAPM is an equilibrium, single‑period model linking expected return to **systematic risk**. The only priced risk is market covariance (beta).  
**Core equation:**
$$E[R_i] = r_f + \beta_i (E[R_m]-r_f)$$
where $\beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$.  
**Purpose:** Provide a benchmark expected return, cost of equity, and risk‑adjusted performance baseline.  
**Prerequisites:** Mean‑variance optimization, risk‑free asset, covariance, utility theory

---

## 2. Comparative Framing

| Aspect | CAPM | ICAPM | Fama‑French | APT | Black‑Litterman |
|--------|------|-------|-------------|-----|----------------|
| **Horizon** | Single‑period | Multi‑period | Empirical (static) | Multi‑factor (static) | Multi‑factor (Bayesian) |
| **Risk** | Market beta only | Market + state variables | Market + size + value (+) | Linear factor betas | Market equilibrium + views |
| **Drivers** | Equilibrium, mean‑variance | Hedging demands | Empirical anomalies | No‑arbitrage | Investor views |
| **Output** | Expected return by beta | Expected return by betas | Expected return by factors | Expected return by factors | Optimized weights |
| **Fit** | Weak empirically | Better with state variables | Strong | Moderate | Strong (practical) |
| **Use** | Cost of equity, benchmark | Strategic allocation | Factor investing | Risk modeling | Institutional portfolios |

**Key Insight:** CAPM is elegant and foundational but empirically incomplete; it remains the simplest benchmark for expected returns and cost of equity.

---

## 3. Examples + Counterexamples

**Example 1: Expected Return from Beta**  
Assume $r_f=3\%$, market premium $E[R_m]-r_f=6\%$.  
- Stock A: $\beta=0.8$ → $E[R_A]=3\%+0.8\cdot6\%=7.8\%$  
- Stock B: $\beta=1.3$ → $E[R_B]=3\%+1.3\cdot6\%=10.8\%$  
**Implication:** Higher beta → higher expected return.

---

**Example 2: Portfolio Cost of Equity**  
Firm equity beta $\beta_e=1.1$, $r_f=4\%$, market premium $5\%$.  
$$k_e = 4\% + 1.1\cdot5\% = 9.5\%$$  
**Implication:** CAPM provides discount rate for valuation.

---

**Example 3: Defensive Asset Pricing**  
Low‑beta utility stock ($\beta=0.5$) should earn lower expected return. If it earns more, CAPM implies **positive alpha**.

---

**COUNTEREXAMPLE 4: Value Premium**  
Value stocks often earn higher returns than CAPM predicts despite similar beta. This violates CAPM and motivates multi‑factor models.

---

## 4. Layer Breakdown

```
CAPM Architecture:

├─ Assumptions:
│   ├─ Investors mean‑variance optimizing
│   ├─ Homogeneous expectations
│   ├─ Single period
│   ├─ No taxes/transaction costs
│   └─ Unlimited borrowing/lending at r_f
│
├─ Equilibrium Logic:
│   ├─ All investors hold market portfolio of risky assets
│   ├─ Differences only in risk‑free vs risky mix
│   └─ Market clears → single pricing relation
│
├─ Pricing Relation:
│   ├─ Expected return depends only on beta
│   └─ Idiosyncratic risk diversified away
│
├─ Empirical Tests:
│   ├─ SML slope too flat
│   ├─ Size, value, momentum anomalies
│   └─ Time‑varying beta and premiums
│
└─ Applications:
    ├─ Cost of equity
    ├─ Performance attribution
    └─ Benchmarking risk
```

---

## 5. Mini‑Project: CAPM Estimation with Market Data

```python
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
```

---

## 6. Challenge Round

1. **Beta Stability:** Estimate rolling beta for a stock. How does beta change in crises? What does that imply for CAPM use?  
2. **Alpha Interpretation:** If a stock has alpha +3% per year, is it skill or factor exposure? How would you test?  
3. **SML Slope:** If the observed SML is flatter than CAPM predicts, what does that imply about low‑beta vs high‑beta stocks?  
4. **Leverage Constraint:** How does limited borrowing at $r_f$ distort CAPM equilibrium?  
5. **Benchmark Choice:** What happens to beta estimates if the market proxy is incomplete (e.g., SPY only)?

---

## 7. Key References

- **Sharpe, W.F. (1964).** "Capital Asset Prices: A Theory of Market Equilibrium" – Original CAPM derivation.  
- **Lintner, J. (1965).** "The Valuation of Risk Assets" – CAPM extensions.  
- **Black, F. (1972).** "Capital Market Equilibrium with Restricted Borrowing" – Zero‑beta CAPM.  
- **Fama, E.F., & French, K.R. (1992).** "The Cross‑Section of Expected Stock Returns" – CAPM empirical failures.  
- **Bodie, Kane, Marcus (Investments).** CAPM applications and tests.  

