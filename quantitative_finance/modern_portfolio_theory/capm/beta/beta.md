# Beta (β)

## 1. Concept Skeleton
**Definition:** Systematic risk measure quantifying asset's sensitivity to market movements; slope coefficient from regressing asset returns on market returns  
**Purpose:** Measure non-diversifiable risk, determine expected return via CAPM, portfolio risk management  
**Prerequisites:** Linear regression, covariance, CAPM framework, systematic vs idiosyncratic risk

## 2. Comparative Framing
| Beta Value | β < 0 | 0 < β < 1 | β = 1 | β > 1 |
|------------|-------|-----------|-------|-------|
| **Meaning** | Inverse to market | Defensive | Market-like | Aggressive |
| **Examples** | Gold (sometimes), inverse ETFs | Utilities, consumer staples | Index funds, diversified portfolio | Tech stocks, leveraged ETFs |
| **Risk** | Market hedge | Below-market volatility | Market volatility | Above-market volatility |
| **Bull Market** | Underperform | Underperform | Match market | Outperform |
| **Bear Market** | Outperform (gain) | Outperform (less loss) | Match market | Underperform (bigger loss) |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock returns +20% when market +10% and -15% when market -10%  
β ≈ 1.5 (moves 1.5× market); high systematic risk

**Failure Case:**  
Enron 2001: β=0.6 (defensive utility), but idiosyncratic collapse → -99%; beta didn't capture fraud risk

**Edge Case:**  
Market-neutral hedge fund: β≈0 by design (long/short), but can have high absolute volatility from idiosyncratic bets

## 4. Layer Breakdown
```
Beta Calculation and Interpretation:
├─ Mathematical Definition:
│   ├─ βi = Cov(Ri, Rm) / Var(Rm)
│   ├─ Regression Form: Ri - rf = αi + βi(Rm - rf) + εi
│   ├─ Slope Interpretation: ΔRi / ΔRm expected change
│   └─ Correlation Form: βi = ρi,m × (σi / σm)
├─ Estimation Methods:
│   ├─ Historical Regression:
│   │   ├─ OLS: Regress excess returns on market excess returns
│   │   ├─ Period: Typically 2-5 years of data
│   │   ├─ Frequency: Daily, weekly, or monthly returns
│   │   └─ Standard Error: Measure estimation uncertainty
│   ├─ Market Model: Ri = αi + βi·Rm + εi
│   ├─ Adjusted Beta: (2/3)·βraw + (1/3)·1.0 (Blume adjustment)
│   └─ Fundamental Beta: Based on leverage, industry, size
├─ Portfolio Beta:
│   ├─ Weighted Average: βp = Σ wi·βi
│   ├─ Additivity: Linear combination of betas
│   ├─ Market Portfolio: βmarket = 1.0 by definition
│   └─ Risk-Free Asset: βrf = 0 (no market sensitivity)
├─ Beta Ranges and Interpretation:
│   ├─ β = 0: No systematic risk (risk-free asset, perfect hedge)
│   ├─ 0 < β < 1: Defensive (less volatile than market)
│   ├─ β = 1: Moves with market (index funds)
│   ├─ β > 1: Aggressive (amplifies market moves)
│   ├─ β < 0: Inverse relationship (rare in equities)
│   └─ Typical Ranges: Utilities 0.5-0.8, Tech 1.2-1.8
├─ Time Variation:
│   ├─ Business Cycle: Beta increases in recessions (correlation rise)
│   ├─ Leverage Changes: Higher debt → higher beta
│   ├─ Industry Rotation: Sector betas vary over time
│   └─ Structural Changes: Company transformation alters beta
├─ Limitations:
│   ├─ Historical: Past beta ≠ future beta
│   ├─ Estimation Error: Standard errors often ±0.2-0.3
│   ├─ Non-Stationarity: Beta changes over time
│   ├─ Market Proxy: Different indices give different betas
│   └─ Assumes Linear: Non-linear market relationships ignored
└─ CAPM Application:
    ├─ Expected Return: E[Ri] = rf + βi(E[Rm] - rf)
    ├─ Required Return: Minimum return to compensate risk
    ├─ Security Valuation: Discount future cash flows
    └─ Performance Evaluation: Alpha = Actual - CAPM Expected
```

**Interaction:** Beta × market risk premium determines systematic return component; higher beta = higher required return

## 5. Mini-Project
Calculate and analyze beta for different stocks and portfolios:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

# Download data
tickers = {
    'Defensive': ['PG', 'KO', 'WMT'],  # Consumer staples
    'Market-Like': ['SPY'],  # S&P 500 ETF
    'Aggressive': ['TSLA', 'NVDA', 'AMD'],  # High beta tech
    'Leveraged': ['TQQQ', 'UPRO'],  # 3x leveraged ETFs
}

all_tickers = [t for group in tickers.values() for t in group]
market_ticker = 'SPY'

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Risk-free rate
rf_annual = 0.03
rf_daily = (1 + rf_annual) ** (1/252) - 1

# Calculate excess returns
excess_returns = returns.subtract(rf_daily, axis=0)
market_excess = excess_returns[market_ticker]

def calculate_beta(stock_returns, market_returns, method='ols'):
    """
    Calculate beta using OLS regression
    Returns: beta, alpha, se_beta, t_stat, p_value, r_squared
    """
    # Align data
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ['stock', 'market']
    
    if method == 'ols':
        # Regression method
        X = sm.add_constant(aligned['market'])
        model = sm.OLS(aligned['stock'], X).fit()
        
        beta = model.params[1]
        alpha = model.params[0]
        se_beta = model.bse[1]
        t_stat = model.tvalues[1]
        p_value = model.pvalues[1]
        r_squared = model.rsquared
        
    elif method == 'covariance':
        # Direct calculation
        beta = aligned.cov().loc['stock', 'market'] / aligned['market'].var()
        alpha = aligned['stock'].mean() - beta * aligned['market'].mean()
        se_beta = np.nan
        t_stat = np.nan
        p_value = np.nan
        r_squared = aligned.corr().loc['stock', 'market'] ** 2
    
    return {
        'beta': beta,
        'alpha': alpha * 252,  # Annualize
        'se_beta': se_beta,
        't_statistic': t_stat,
        'p_value': p_value,
        'r_squared': r_squared,
        'n_obs': len(aligned)
    }

def adjusted_beta(raw_beta, adjustment_factor=1/3):
    """
    Blume adjustment: weighted average of raw beta and 1.0
    Theory: betas tend to revert toward 1.0 over time
    """
    return (1 - adjustment_factor) * raw_beta + adjustment_factor * 1.0

# Calculate betas for all stocks
beta_results = {}
for ticker in all_tickers:
    if ticker != market_ticker and ticker in excess_returns.columns:
        results = calculate_beta(excess_returns[ticker], market_excess)
        results['adjusted_beta'] = adjusted_beta(results['beta'])
        beta_results[ticker] = results

# Convert to DataFrame
beta_df = pd.DataFrame(beta_results).T
beta_df = beta_df.sort_values('beta')

print("\n" + "=" * 100)
print("BETA ANALYSIS")
print("=" * 100)
print(beta_df[['beta', 'adjusted_beta', 'se_beta', 'r_squared', 'alpha']].round(4))

# Rolling beta analysis
def rolling_beta(stock_returns, market_returns, window=252):
    """Calculate rolling beta over time"""
    betas = []
    dates = []
    
    for i in range(window, len(stock_returns)):
        window_stock = stock_returns.iloc[i-window:i]
        window_market = market_returns.iloc[i-window:i]
        
        aligned = pd.concat([window_stock, window_market], axis=1).dropna()
        if len(aligned) > 50:
            cov = aligned.cov().iloc[0, 1]
            var = aligned.iloc[:, 1].var()
            beta = cov / var
            
            betas.append(beta)
            dates.append(stock_returns.index[i])
    
    return pd.Series(betas, index=dates)

# Calculate rolling betas for selected stocks
rolling_betas = {}
example_stocks = ['TSLA', 'PG', 'NVDA']
for ticker in example_stocks:
    if ticker in excess_returns.columns:
        rolling_betas[ticker] = rolling_beta(excess_returns[ticker], market_excess, window=252)

# Simulate portfolio betas
def portfolio_beta(weights, betas):
    """Calculate portfolio beta as weighted average"""
    return np.dot(weights, betas)

# Example portfolios
portfolio_compositions = {
    'Conservative': {'PG': 0.4, 'KO': 0.3, 'WMT': 0.3},
    'Balanced': {'PG': 0.25, 'SPY': 0.50, 'TSLA': 0.25},
    'Aggressive': {'TSLA': 0.4, 'NVDA': 0.4, 'AMD': 0.2},
}

portfolio_betas = {}
for port_name, composition in portfolio_compositions.items():
    weights = []
    betas = []
    for ticker, weight in composition.items():
        if ticker in beta_results:
            weights.append(weight)
            betas.append(beta_results[ticker]['beta'])
    
    portfolio_betas[port_name] = {
        'beta': portfolio_beta(np.array(weights), np.array(betas)),
        'composition': composition
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Beta distribution by category
categories = []
beta_values = []
colors_map = {'Defensive': 'green', 'Market-Like': 'gray', 
              'Aggressive': 'red', 'Leveraged': 'darkred'}

for category, ticker_list in tickers.items():
    for ticker in ticker_list:
        if ticker in beta_results:
            categories.append(f"{ticker}\n({category})")
            beta_values.append(beta_results[ticker]['beta'])

x_pos = np.arange(len(categories))
colors = []
for cat in categories:
    for key, color in colors_map.items():
        if key in cat:
            colors.append(color)
            break

bars = axes[0, 0].barh(x_pos, beta_values, color=colors, alpha=0.7)
axes[0, 0].set_yticks(x_pos)
axes[0, 0].set_yticklabels(categories, fontsize=8)
axes[0, 0].axvline(1.0, color='black', linestyle='--', linewidth=2, label='Market (β=1)')
axes[0, 0].axvline(0.0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_xlabel('Beta')
axes[0, 0].set_title('Beta Values by Stock Category')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)

# Add confidence intervals
for i, ticker in enumerate([t.split('\n')[0] for t in categories]):
    if ticker in beta_results and not np.isnan(beta_results[ticker]['se_beta']):
        beta = beta_results[ticker]['beta']
        se = beta_results[ticker]['se_beta']
        axes[0, 0].errorbar(beta, i, xerr=1.96*se, fmt='none', 
                           ecolor='black', capsize=3, alpha=0.5)

# Plot 2: Rolling beta over time
for ticker in example_stocks:
    if ticker in rolling_betas:
        axes[0, 1].plot(rolling_betas[ticker].index, rolling_betas[ticker].values,
                       linewidth=2, label=ticker, alpha=0.8)

axes[0, 1].axhline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Market')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Beta (1-year rolling window)')
axes[0, 1].set_title('Time-Varying Beta')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Beta vs R² scatter
betas_plot = [beta_results[t]['beta'] for t in beta_results.keys()]
r_squareds = [beta_results[t]['r_squared'] for t in beta_results.keys()]
tickers_plot = list(beta_results.keys())

scatter = axes[1, 0].scatter(betas_plot, r_squareds, s=200, alpha=0.6, c=betas_plot,
                             cmap='RdYlGn_r', vmin=0, vmax=2)

for i, ticker in enumerate(tickers_plot):
    axes[1, 0].annotate(ticker, (betas_plot[i], r_squareds[i]),
                       fontsize=8, ha='center', va='bottom')

axes[1, 0].axvline(1.0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Beta')
axes[1, 0].set_ylabel('R² (Systematic Risk %)')
axes[1, 0].set_title('Beta vs Explanatory Power')
axes[1, 0].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Beta Value')

# Plot 4: Portfolio beta composition
port_names = list(portfolio_betas.keys())
port_beta_values = [portfolio_betas[p]['beta'] for p in port_names]

bars = axes[1, 1].bar(range(len(port_names)), port_beta_values, alpha=0.7)
for i, bar in enumerate(bars):
    if port_beta_values[i] < 1:
        bar.set_color('green')
    elif port_beta_values[i] > 1:
        bar.set_color('red')
    else:
        bar.set_color('gray')

axes[1, 1].set_xticks(range(len(port_names)))
axes[1, 1].set_xticklabels(port_names, rotation=45, ha='right')
axes[1, 1].axhline(1.0, color='black', linestyle='--', linewidth=2, label='Market')
axes[1, 1].set_ylabel('Portfolio Beta')
axes[1, 1].set_title('Portfolio Beta (Weighted Average)')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (name, value) in enumerate(zip(port_names, port_beta_values)):
    axes[1, 1].text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Print detailed portfolio analysis
print("\n" + "=" * 100)
print("PORTFOLIO BETA ANALYSIS")
print("=" * 100)

for port_name in port_names:
    port_beta_val = portfolio_betas[port_name]['beta']
    composition = portfolio_betas[port_name]['composition']
    
    print(f"\n{port_name} Portfolio (β = {port_beta_val:.3f}):")
    print("-" * 50)
    
    for ticker, weight in composition.items():
        if ticker in beta_results:
            ticker_beta = beta_results[ticker]['beta']
            contribution = weight * ticker_beta
            print(f"  {ticker:6s}: {weight:>6.1%} × β={ticker_beta:>5.2f} = {contribution:>6.3f}")

# CAPM expected returns
print("\n" + "=" * 100)
print("CAPM EXPECTED RETURNS")
print("=" * 100)

market_return = excess_returns[market_ticker].mean() * 252 + rf_annual
market_premium = market_return - rf_annual

print(f"Risk-Free Rate:      {rf_annual:.2%}")
print(f"Market Return:       {market_return:.2%}")
print(f"Market Risk Premium: {market_premium:.2%}")
print(f"\n{'Ticker':<8} {'Beta':>8} {'CAPM E[R]':>12} {'Actual R':>12} {'Alpha':>12}")
print("-" * 100)

for ticker in beta_results.keys():
    beta = beta_results[ticker]['beta']
    capm_expected = rf_annual + beta * market_premium
    actual_return = (excess_returns[ticker].mean() * 252 + rf_annual) if ticker in excess_returns.columns else np.nan
    alpha = actual_return - capm_expected if not np.isnan(actual_return) else np.nan
    
    print(f"{ticker:<8} {beta:>8.3f} {capm_expected:>11.2%} {actual_return:>11.2%} {alpha:>11.2%}")

# Statistical significance of beta
print("\n" + "=" * 100)
print("BETA STATISTICAL SIGNIFICANCE")
print("=" * 100)
print(f"{'Ticker':<8} {'Beta':>8} {'Std Error':>12} {'t-stat':>10} {'p-value':>10} {'Significant':>12}")
print("-" * 100)

for ticker in beta_results.keys():
    beta = beta_results[ticker]['beta']
    se = beta_results[ticker]['se_beta']
    t_stat = beta_results[ticker]['t_statistic']
    p_val = beta_results[ticker]['p_value']
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "No"
    
    print(f"{ticker:<8} {beta:>8.3f} {se:>12.4f} {t_stat:>10.2f} {p_val:>10.4f} {sig:>12}")

print("\n*** p<0.01, ** p<0.05, * p<0.1")
```

## 6. Challenge Round
When is beta misleading or insufficient?
- Non-linear relationships: Options, structured products have non-constant beta
- Time-varying: Beta changes with leverage, business cycle, company transformation
- Market proxy matters: S&P 500 vs Russell 2000 give different betas
- Crisis periods: Correlations spike, betas unstable, diversification fails
- Idiosyncratic events: High-beta stock can collapse from firm-specific risk

Beta adjustments and alternatives:
- Blume adjustment: Raw beta × 2/3 + 1.0 × 1/3 (mean reversion)
- Levered/unlevered: βlevered = βunlevered × (1 + D/E)
- Downside beta: Only use negative market returns (tail risk)
- Fundamental beta: Based on accounting ratios, industry, size
- Multi-factor: Fama-French adds size, value betas to capture more risk

Common misconceptions:
- "High beta = high risk always": Only systematic risk; could have low total risk if R² high
- "Beta predicts returns": Expected return, not guaranteed; still have alpha component
- "Beta = 1 is average": Market beta, but stock could be volatile from idiosyncratic
- "Beta constant over time": Changes with leverage, business mix, market regime
- "Negative beta = safe asset": Gold sometimes, but can be volatile

## 7. Key References
- [Sharpe, W. (1964) "Capital Asset Prices: A Theory of Market Equilibrium"](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1964.tb02865.x)
- [Black, F. (1972) "Capital Market Equilibrium with Restricted Borrowing"](https://www.jstor.org/stable/2978484)
- [Blume, M. (1971) "On the Assessment of Risk"](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1971.tb00912.x)
- [Investopedia - Beta](https://www.investopedia.com/terms/b/beta.asp)

---
**Status:** Core CAPM risk measure (Sharpe 1964, Nobel 1990) | **Complements:** CAPM, Alpha, Systematic Risk, Security Market Line
