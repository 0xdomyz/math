# Currency Risk & Hedging in Global Portfolios

## 1. Concept Skeleton
**Definition:** Exchange rate fluctuations affecting returns on foreign currency investments; distinction between hedged (FX risk removed) and unhedged (FX exposure retained) portfolios  
**Purpose:** Quantify currency impact on international returns, determine optimal hedging ratios, manage multi-currency portfolio volatility  
**Prerequisites:** Foreign exchange markets, forward contracts, portfolio optimization, variance decomposition

---

## 2. Comparative Framing

| Aspect | Unhedged Foreign Assets | Hedged Foreign Assets | Domestic Assets |
|--------|--------------------------|----------------------|-----------------|
| **Currency Exposure** | Full FX volatility | Minimized (costs) | Zero (base currency) |
| **Return** | $R_{total} = (1 + r_{local})(1 + r_{FX}) - 1$ | $R_{hedged} \approx r_{local}$ | $R_{domestic}$ |
| **Volatility** | $\sigma_{unhedged}^2 = \sigma_{local}^2 + \sigma_{FX}^2 + 2\rho\sigma_{local}\sigma_{FX}$ | $\sigma_{hedged}^2 \approx \sigma_{local}^2$ | $\sigma_{domestic}^2$ |
| **Correlation Impact** | High if $\rho > 0$ (reinforces risk) | Low (hedging reduces) | N/A |
| **Hedging Cost** | 0% | 1-3% p.a. (interest differential) | N/A |
| **Use Case** | Benefit if currency appreciates; cost if depreciates | Reduces volatility; insurance | Baseline |
| **Long-term Investor** | May ignore transient FX noise | Hedge if currency volatile | Standard |

**Key Insight:** Currency return correlation with asset returns determines hedging benefit. Gold mines in Australia: if AUD depreciates when stocks fall, hedging reduces risk (positive correlation = bad). If AUD appreciates when stocks fall, don't hedge (negative correlation = diversification).

---

## 3. Examples + Counterexamples

**Example 1: Hedged vs Unhedged EUR Bonds**
- German 10Y Bund: 4.5% yield, EUR/USD spot 1.10, 1-year forward 1.09
- Unhedged return: 4.5% + (1.09 - 1.10)/1.10 = 4.5% - 0.91% = 3.59%
- Hedged return: 4.5% - (forward premium cost) ≈ 4.5% - 0.80% = 3.70%
- Conclusion: Hedging outperforms when EUR weakens; unhedged better if EUR strengthens

**Example 2: Japanese Equity + Strong Yen (2012-2014)**
- Nikkei 225: +58% (2012-2014 local currency)
- JPY strength: +13% (USD/JPY weakened from 80 to 105)
- Unhedged US investor: +58% + 13% = +71.5% (currency boost)
- Hedged US investor: +58% only (gave up currency gains)
- Conclusion: Positive correlation = hedging costly; unhedged captured full upside

**Counterexample: Emerging Market Crisis (2013 Taper Tantrum)**
- Turkish equity: -15% local currency (EM outflow)
- TRY weakness: -25% vs USD (flight to safety)
- Unhedged US investor: -15% × (1 - 0.25) = -36.25% (currency amplifies loss)
- Hedged US investor: -15% only (hedging acts as insurance)
- Conclusion: Negative correlation = hedging valuable; provides downside protection

**Edge Case: Uncovered Interest Rate Parity Failure**
- Interest differential: USD 5%, EUR 3% → 2% forward premium
- Forward predicts: EUR will depreciate 2% annually
- Reality: EUR appreciates 3% (carry trade blowup)
- Unhedged EUR borrower: Profits from surprise appreciation (unlikely outcome captured)
- Explanation: Investors underestimate FX risk; carry trade carries disaster risk

---

## 4. Layer Breakdown

```
Currency Risk Architecture:
├─ Return Decomposition:
│   ├─ Total Return (unhedged): R_total = (1 + R_local)(1 + R_FX) - 1
│   ├─ Components:
│   │   ├─ Local Currency Return: R_local (bond yield, stock dividend + price appreciation)
│   │   ├─ FX Return: R_FX = (S_T - S_0) / S_0 (spot rate appreciation)
│   │   └─ Cross-Term: R_local × R_FX (small for developed currencies)
│   └─ Hedged Approximation: R_hedged ≈ R_local - Cost_forward
│
├─ Volatility Decomposition:
│   ├─ Unhedged Variance: σ²_total = σ²_local + σ²_FX + 2ρ σ_local σ_FX
│   │   ├─ Asset volatility contribution: σ²_local
│   │   ├─ Currency volatility contribution: σ²_FX
│   │   ├─ Correlation effect: ρ (positive = amplification, negative = diversification)
│   │   └─ Typical: 30-40% of total volatility from FX in developed markets
│   │
│   ├─ Hedged Variance: σ²_hedged ≈ σ²_local (removes FX term)
│   │   └─ Hedging ratio h: σ²_portfolio = σ²_unhedged(1-h)² + σ²_FX h² (partial hedge optimization)
│   │
│   └─ Correlation Regime (key driver):
│       ├─ Positive ρ (stocks and local currency fall together):
│       │   └─ Hedging reduces compound risk; recommended for risk-averse
│       ├─ Negative ρ (stocks fall but currency strengthens → safe haven):
│       │   └─ Natural hedge; unhedged better for risk/return
│       └─ Near-zero ρ (independent movements):
│           └─ Hedging marginal benefit; decision depends on cost vs minor smoothing
│
├─ Forward Contract Mechanics:
│   ├─ Forward Rate: F = S₀ × (1 + r_domestic) / (1 + r_foreign)
│   │   └─ Reflects interest differential via covered interest parity
│   ├─ Forward Premium/Discount: (F - S₀) / S₀ = (r_domestic - r_foreign)
│   ├─ Annual Cost of Hedging: 1-3% typically (USD ~2%, EUR ~1%, JPY ~-1% carry)
│   └─ Basis Risk: Forward rate diverges from spot at maturity if rates move
│
├─ Portfolio Construction:
│   ├─ Unhedged Global Portfolio:
│   │   ├─ Weights: w_domestic + w_foreign_stocks + w_foreign_bonds = 1
│   │   ├─ Returns: Mix of local returns + FX exposure
│   │   └─ Volatility: Elevated by independent FX shocks (tail risk events)
│   │
│   ├─ Partially Hedged Portfolio:
│   │   ├─ Selective hedging: Currency pairs with positive correlation only
│   │   ├─ Weights: w_unhedged_EUR + w_hedged_JPY + w_domestic
│   │   └─ Optimization: Min variance subject to hedging cost constraints
│   │
│   └─ Fully Hedged Portfolio:
│       ├─ All FX converted back to base currency via forwards
│       ├─ Returns: Local currency performance + hedging costs
│       └─ Volatility: Reduced to underlying asset volatility (removes FX term)
│
└─ Dynamic Hedging Decisions:
    ├─ Tactical (market view): Reduce hedge when currency undervalued (strong forward)
    ├─ Strategic (correlation regime): Increase hedge if ρ turns positive (crisis mode)
    ├─ Mechanical (rules-based): Rehedge when spot drifts >5% from forward
    └─ Volatility-driven: Hedge more when FX volatility spikes (Vega risk)
```

**Mathematical Formulas:**

Mean-Variance for Unhedged Global Portfolio:
$$E[R_p] = w_d E[R_d] + w_f E[(1+r_f)(1+r_{fx})-1]$$
$$E[R_p] = w_d E[R_d] + w_f [E[r_f] + E[r_{fx}] + E[r_f \cdot r_{fx}]]$$

Hedged Portfolio (approximate):
$$E[R_{p,hedged}] = w_d E[R_d] + w_f [E[r_f] - C_{hedge}]$$

Variance with Correlation:
$$\sigma^2_{unhedged} = w_f^2[\sigma^2_f + \sigma^2_{fx} + 2\rho_{f,fx}\sigma_f\sigma_{fx}] + w_d^2\sigma^2_d + \text{cross terms}$$

Optimal Hedge Ratio (minimizes variance):
$$h^* = -\rho_{asset,fx} \frac{\sigma_{asset}}{\sigma_{fx}}$$
If $\rho = +0.5$, $\sigma_{asset} = 15\%$, $\sigma_{fx} = 10\%$:
$$h^* = -0.5 \times \frac{0.15}{0.10} = -0.75$$ (negative = should hedge 75%)

---

## 5. Mini-Project: Currency Risk Analysis with Dynamic Hedging

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Download international asset returns and FX data
# US 60/40 portfolio + International assets (EUR, JPY, GBP)

def get_international_data(start_date, end_date):
    """
    Fetch returns for domestic and international assets + FX rates.
    """
    # Asset tickers
    assets = {
        'SPY': 'US Stocks',           # S&P 500
        'AGG': 'US Bonds',            # Bond Aggregate
        'EAFE': 'International Equity', # Developed ex-US
        'PWE': 'Emerging Markets',    # EM stocks
    }
    
    # FX tickers (to get currency appreciation)
    # Note: yfinance doesn't have direct FX; use currency ETFs as proxy
    fx_proxy = {
        'FXE': 'EUR/USD',
        'FXY': 'JPY/USD',
        'FXB': 'GBP/USD'
    }
    
    data = yf.download(list(assets.keys()) + list(fx_proxy.keys()), 
                       start=start_date, end=end_date, progress=False)['Adj Close']
    
    returns = data.pct_change().dropna()
    return returns, assets, fx_proxy


def analyze_currency_impact(returns, assets, fx_proxy):
    """
    Decompose international asset returns into local + FX components.
    """
    # Correlations: assets vs FX
    analysis = pd.DataFrame()
    
    for asset, asset_name in assets.items():
        analysis.loc[asset_name, 'Volatility'] = returns[asset].std() * np.sqrt(252)
        analysis.loc[asset_name, 'Return (Annual)'] = returns[asset].mean() * 252
        
        # Correlations with each currency
        for fx, fx_name in fx_proxy.items():
            corr = returns[asset].corr(returns[fx])
            analysis.loc[asset_name, f'Corr w/ {fx_name}'] = corr
    
    return analysis


def unhedged_vs_hedged_portfolio(returns, weights, fx_ticker='FXE'):
    """
    Compare portfolio with and without FX hedge.
    Assume we hedge EUR exposure fully.
    """
    # Simulated hedge cost: interest differential (e.g., -1.5% annual on EUR carry)
    hedge_cost = -0.015 / 252  # Daily cost
    
    # Portfolio return (unhedged)
    portfolio_returns = (returns[['SPY', 'AGG', 'EAFE', 'PWE']] * weights).sum(axis=1)
    
    # Assume EAFE = ~70% EUR exposure, so hedging ~70% removes that FX
    fx_component = returns[fx_ticker] * 0.7  # Simplified: FX captured in currency move
    
    # Hedged version: remove FX volatility but pay cost
    portfolio_hedged = portfolio_returns - fx_component * weights[2] - hedge_cost
    
    return portfolio_returns, portfolio_hedged, fx_component


def optimize_hedge_ratio(returns, asset_returns, fx_returns):
    """
    Find optimal hedge ratio minimizing portfolio variance.
    
    h* = -ρ(asset, FX) × σ(asset) / σ(FX)
    """
    corr = asset_returns.corr(fx_returns)
    sigma_asset = asset_returns.std()
    sigma_fx = fx_returns.std()
    
    h_optimal = -corr * (sigma_asset / sigma_fx)
    h_optimal = np.clip(h_optimal, -1, 1)  # Bound hedge ratio [0, 100%]
    
    return h_optimal, corr


# Main Analysis
print("=" * 80)
print("INTERNATIONAL CURRENCY RISK ANALYSIS")
print("=" * 80)

# Get data (3 years of daily returns)
returns, assets, fx_proxy = get_international_data('2021-01-01', '2024-01-01')

# 1. Analyze currency impact
print("\n1. ASSET × CURRENCY CORRELATIONS")
print("-" * 80)
analysis = analyze_currency_impact(returns, assets, fx_proxy)
print(analysis.round(3))

# 2. Correlation analysis: When do we want to hedge?
print("\n2. OPTIMAL HEDGE RATIOS")
print("-" * 80)
eafe_returns = returns['EAFE'].dropna()
fxe_returns = returns['FXE'].dropna()

h_opt, corr_eafe_fxe = optimize_hedge_ratio(
    eafe_returns, eafe_returns, fxe_returns
)

print(f"EAFE × EUR/USD Correlation: {corr_eafe_fxe:.3f}")
print(f"Optimal Hedge Ratio (h*): {h_opt:.1%}")
print(f"  → Positive corr + high vol → hedge {max(0, h_opt):.1%} of exposure")
print(f"  → Negative corr → natural diversification, don't hedge")

# 3. Simulate unhedged vs hedged portfolios
print("\n3. PORTFOLIO COMPARISON (60% EAFE INT'L / 40% US BONDS)")
print("-" * 80)

weights_intl = np.array([0.00, 0.40, 0.60, 0.00])  # 40% bonds, 60% EAFE
portfolio_unhedged, portfolio_hedged, fx_component = unhedged_vs_hedged_portfolio(
    returns, weights_intl, fx_ticker='FXE'
)

comparison = pd.DataFrame({
    'Unhedged Portfolio': [
        portfolio_unhedged.mean() * 252,
        portfolio_unhedged.std() * np.sqrt(252),
        portfolio_unhedged.mean() * 252 / portfolio_unhedged.std() / np.sqrt(252)
    ],
    'Hedged Portfolio': [
        portfolio_hedged.mean() * 252,
        portfolio_hedged.std() * np.sqrt(252),
        portfolio_hedged.mean() * 252 / portfolio_hedged.std() / np.sqrt(252) if portfolio_hedged.std() > 0 else 0
    ],
    'FX Component': [
        fx_component.mean() * 252,
        fx_component.std() * np.sqrt(252),
        0  # N/A for FX component
    ]
}, index=['Annual Return', 'Annual Volatility', 'Sharpe Ratio'])

print(comparison.round(4))
print(f"\nHedging Cost: {0.015:.1%} p.a. (interest differential)")
print(f"FX Volatility Contribution: {fx_component.std() * np.sqrt(252) / portfolio_unhedged.std() / np.sqrt(252):.1%} of portfolio vol")

# 4. Rolling correlation: when does ρ(asset, FX) turn positive/negative?
print("\n4. ROLLING CORRELATION (120-DAY WINDOW)")
print("-" * 80)

window = 120
rolling_corr = eafe_returns.rolling(window).corr(fxe_returns)

print(f"Current Correlation: {rolling_corr.iloc[-1]:.3f}")
print(f"Mean Correlation: {rolling_corr.mean():.3f}")
print(f"Min Correlation (tail risk): {rolling_corr.min():.3f}")
print(f"Max Correlation (normal times): {rolling_corr.max():.3f}")
print(f"\nInterpretation: {rolling_corr.iloc[-1]:.2f} → ", end="")
if rolling_corr.iloc[-1] > 0.3:
    print("Strong positive → HEDGE to reduce amplified risk")
elif rolling_corr.iloc[-1] > 0:
    print("Weak positive → Selective hedging (minor benefit)")
else:
    print("Negative → Natural diversification, avoid hedging")

# 5. Visualization: Cumulative returns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative returns
ax = axes[0, 0]
cum_unhedged = (1 + portfolio_unhedged).cumprod()
cum_hedged = (1 + portfolio_hedged).cumprod()
ax.plot(cum_unhedged.index, cum_unhedged, label='Unhedged (EAFE exposure)', linewidth=2)
ax.plot(cum_hedged.index, cum_hedged, label='Hedged (FX removed)', linewidth=2, linestyle='--')
ax.set_title('Cumulative Returns: Unhedged vs Hedged Portfolios', fontweight='bold')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Currency component over time
ax = axes[0, 1]
ax.bar(fx_component.index, fx_component, alpha=0.6, label='EUR/USD Return', color='steelblue')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_title('EUR/USD Contribution to Portfolio', fontweight='bold')
ax.set_ylabel('Daily Return %')
ax.grid(alpha=0.3)

# Plot 3: Rolling correlation
ax = axes[1, 0]
ax.plot(rolling_corr.index, rolling_corr, linewidth=2, color='darkred')
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.fill_between(rolling_corr.index, 0, rolling_corr, alpha=0.3, color='red', label='Positive (hedge beneficial)')
ax.fill_between(rolling_corr.index, 0, rolling_corr, alpha=0.3, color='green', where=(rolling_corr < 0), label='Negative (diversification)')
ax.set_title('Rolling Correlation: EAFE × EUR/USD (120-day)', fontweight='bold')
ax.set_ylabel('Correlation')
ax.set_ylim(-1, 1)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Variance decomposition
ax = axes[1, 1]
total_var = portfolio_unhedged.std() ** 2 * 252
fx_var_contribution = (fx_component.std() ** 2 * 252) / (total_var) * 100
asset_var_contribution = 100 - fx_var_contribution

categories = ['Asset\nVolatility', 'FX\nVolatility']
values = [asset_var_contribution, fx_var_contribution]
colors = ['steelblue', 'orange']
ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('% of Total Variance')
ax.set_title('Variance Decomposition: Unhedged Portfolio', fontweight='bold')
ax.set_ylim(0, 100)
for i, v in enumerate(values):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('currency_risk_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: currency_risk_analysis.png")
plt.show()

# 6. Decision framework
print("\n5. HEDGING DECISION FRAMEWORK")
print("-" * 80)

print("""
IF Correlation(Asset, FX) > +0.3 AND FX Volatility > Asset Volatility:
  → HEDGE AGGRESSIVELY (positive correlation amplifies risk)
  → Hedge ratio: 75-100%
  → Example: Australian mining stocks + AUD decline during commodity crash
  → Rationale: Correlation becomes positive when both hit tail risk

IF Correlation(Asset, FX) ≈ 0 AND Hedging Cost < 2% p.a.:
  → SELECTIVE HEDGE (minor benefit; depends on cost)
  → Hedge ratio: 25-50%
  → Rationale: Independent movements don't amplify; cost may dominate

IF Correlation(Asset, FX) < -0.3:
  → NO HEDGE (natural diversification; hedging expensive)
  → Hedge ratio: 0%
  → Example: US recession → USD strengthens (safe haven) → diversification
  → Rationale: Negative correlation provides insurance

Regime Switches:
  → Monitor rolling correlation quarterly
  → When corr changes sign, rebalance hedge dynamically
  → Crisis periods: correlation tends toward +1 (everything falls together)
""")

print("\n" + "=" * 80)
```

---

## 6. Challenge Round

1. **Basis Risk Challenge:** You have forward contracts expiring in 3 months, but FX exposure extends to 6 months. How does this create basis risk? How would you hedge the 3-6 month gap?

2. **Carry Trade Paradox:** Interest rate parity suggests forward premiums eliminate arbitrage (2% higher rate ↔ 2% currency depreciation). Yet carry trades (borrow low-yielding currency, invest high-yielding) are highly profitable. Why doesn't interest rate parity hold in practice? What risks justify returns?

3. **Correlation Timing:** Suppose correlation(EAFE, EUR) = -0.4 today. Should you hedge now? What if your investment horizon is 10 years? 1 year? What if hedging costs 1.5% annually?

4. **Partial Hedge Optimization:** Given portfolio weights w = [0.6 EAFE, 0.4 bonds], and knowing that σ_EAFE=18%, σ_FX=12%, ρ(EAFE, FX)=+0.5, calculate the optimal hedge ratio h*. What is the resulting portfolio volatility?

5. **Tail Risk Hedging:** During March 2020 (COVID crash), developed market correlations spiked and currency correlations diverged. JPY strengthened (safe haven), EUR weakened (peripheral risk), AUD fell (commodity crash). A global 60/40 portfolio had long JPY/USD correlation but short AUD/USD. How would a fixed-ratio hedge have performed vs. dynamic rebalancing?

---

## 7. Key References

- **Solnik, B. (1974).** "Why Not Diversify Internationally Rather Than Domestically?" *Financial Analysts Journal* – Foundational work showing international diversification benefits despite currency risk.

- **Frankel, J. & Rose, A. (1995).** "Empirical Research on Nominal Exchange Rates" – Comprehensive review of PPP, interest rate parity, and exchange rate determinants.

- **Black, F. (1989).** "Universal Hedging: Optimizing Currency Risk and Reward for Long-Term Investors" – Argues long-term investors should hedge currency risk to match domestic-currency liabilities.

- **Perold, A.F. & Schulman, E.C. (1988).** "The Free Lunch in Currency Hedging" – Explains why hedging currency risk can improve returns when interest rates exceed equity returns.

- **Froot, K.A., O'Connell, P.G.J., Seasholes, M.S. (2001).** "The Portfolio Flows of International Investors" – Documents currency risk from portfolio perspective and flow dynamics.

- **Federal Reserve: Interest Rate Parity & FX** – https://www.federalreserve.gov/ – Central bank perspective on exchange rate determination.

- **BIS Quarterly Review** – https://www.bis.org – Current analysis of currency markets and hedging practices.

