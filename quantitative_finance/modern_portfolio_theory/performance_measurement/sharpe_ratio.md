# Sharpe Ratio

## 1. Concept Skeleton
**Definition:** Risk-adjusted return measure dividing excess return by total volatility  
**Purpose:** Compare portfolios with different risk levels, evaluate return per unit of risk taken  
**Prerequisites:** Expected return, standard deviation, risk-free rate concepts

## 2. Comparative Framing
| Metric | Sharpe Ratio | Treynor Ratio | Sortino Ratio |
|--------|--------------|---------------|---------------|
| **Risk Measure** | Total volatility (σ) | Systematic risk (β) | Downside deviation | 
| **Formula** | (Rp - rf) / σp | (Rp - rf) / βp | (Rp - rf) / σdownside |
| **Use Case** | Total portfolio comparison | Well-diversified portfolios | Asymmetric return preference |
| **Interpretation** | Return per unit total risk | Return per unit market risk | Penalizes only bad volatility |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio A: 12% return, 20% volatility, rf=2% → Sharpe = (12-2)/20 = 0.50  
Portfolio B: 10% return, 15% volatility, rf=2% → Sharpe = (10-2)/15 = 0.53  
Portfolio B superior despite lower absolute return

**Failure Case:**  
Negative excess returns: Portfolio with -5% return, 10% volatility gives Sharpe = -0.7 (less negative better, counterintuitive)

**Edge Case:**  
Leverage effect: 2x leveraged portfolio doubles both return and risk, Sharpe ratio unchanged

## 4. Layer Breakdown
```
Sharpe Ratio Calculation:
├─ Inputs:
│   ├─ Portfolio Return (Rp): Annualized or period return
│   ├─ Risk-Free Rate (rf): T-bill rate, typically 3-month or 10-year
│   └─ Portfolio Volatility (σp): Standard deviation of returns
├─ Computation:
│   ├─ Excess Return: Rp - rf (risk premium)
│   ├─ Risk Adjustment: Divide by σp
│   └─ Result: Sharpe Ratio = (Rp - rf) / σp
├─ Interpretation:
│   ├─ SR > 1: Good risk-adjusted performance
│   ├─ SR > 2: Excellent performance (rare for traditional assets)
│   ├─ SR > 3: Exceptional (hedge fund benchmarks)
│   └─ SR < 0: Loss relative to risk-free rate
└─ Limitations:
    ├─ Assumes normal returns (symmetric distribution)
    ├─ Penalizes upside volatility equally with downside
    ├─ Sensitive to time period chosen
    └─ Non-comparable across different frequencies without annualization
```

**Interaction:** Higher excess return or lower volatility → Higher Sharpe ratio

## 5. Mini-Project
Calculate and compare Sharpe ratios for different portfolio strategies:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Download historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years

tickers = ['SPY', 'AGG', 'GLD', 'QQQ']  # Stocks, Bonds, Gold, Tech
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate returns
returns = data.pct_change().dropna()

# Risk-free rate (approximate, using 3% annual)
rf_annual = 0.03
rf_daily = (1 + rf_annual) ** (1/252) - 1

# Portfolio strategies
portfolios = {
    '100% Stocks': np.array([1.0, 0.0, 0.0, 0.0]),
    '60/40 Stock/Bond': np.array([0.6, 0.4, 0.0, 0.0]),
    'Balanced': np.array([0.4, 0.4, 0.1, 0.1]),
    'Risk Parity': np.array([0.25, 0.5, 0.15, 0.1]),  # Simplified
    '100% Tech': np.array([0.0, 0.0, 0.0, 1.0])
}

def calculate_sharpe_ratio(returns, weights, rf_rate):
    """Calculate annualized Sharpe ratio"""
    portfolio_returns = (returns * weights).sum(axis=1)
    excess_returns = portfolio_returns - rf_rate
    
    # Annualize
    mean_excess = excess_returns.mean() * 252
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    sharpe = mean_excess / volatility if volatility > 0 else 0
    
    return sharpe, mean_excess, volatility, portfolio_returns

# Calculate metrics for each portfolio
results = {}
portfolio_return_series = {}

for name, weights in portfolios.items():
    sharpe, mean_ret, vol, port_rets = calculate_sharpe_ratio(
        returns, weights, rf_daily
    )
    results[name] = {
        'Sharpe Ratio': sharpe,
        'Annual Return': mean_ret,
        'Annual Volatility': vol,
        'Return/Risk': mean_ret / vol if vol > 0 else 0
    }
    portfolio_return_series[name] = (1 + port_rets).cumprod()

# Display results
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('Sharpe Ratio', ascending=False)
print("Portfolio Performance Comparison")
print("=" * 70)
print(results_df.round(4))
print("\n" + "=" * 70)
print(f"Risk-Free Rate (Annual): {rf_annual:.2%}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sharpe Ratio comparison
sharpe_values = results_df['Sharpe Ratio'].sort_values(ascending=True)
colors = ['green' if x > 1 else 'orange' if x > 0.5 else 'red' 
          for x in sharpe_values]
axes[0, 0].barh(range(len(sharpe_values)), sharpe_values, color=colors)
axes[0, 0].set_yticks(range(len(sharpe_values)))
axes[0, 0].set_yticklabels(sharpe_values.index)
axes[0, 0].axvline(1.0, color='black', linestyle='--', linewidth=1, 
                   label='Sharpe = 1.0')
axes[0, 0].set_xlabel('Sharpe Ratio')
axes[0, 0].set_title('Sharpe Ratio Comparison')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Return vs Risk scatter
axes[0, 1].scatter(results_df['Annual Volatility'], 
                   results_df['Annual Return'],
                   s=200, alpha=0.6, c=range(len(results_df)))
for idx, name in enumerate(results_df.index):
    axes[0, 1].annotate(name, 
                       (results_df.loc[name, 'Annual Volatility'],
                        results_df.loc[name, 'Annual Return']),
                       fontsize=9, ha='center')

# Add capital allocation line for best Sharpe
best_portfolio = results_df.iloc[0]
x_range = np.array([0, results_df['Annual Volatility'].max() * 1.2])
y_range = rf_annual + best_portfolio['Sharpe Ratio'] * x_range

axes[0, 1].plot(x_range, y_range, 'r--', linewidth=2, alpha=0.5,
               label=f'CAL (Best Sharpe: {best_portfolio.name})')
axes[0, 1].scatter(0, rf_annual, s=100, color='green', marker='*', 
                  label='Risk-Free Rate')
axes[0, 1].set_xlabel('Annual Volatility')
axes[0, 1].set_ylabel('Annual Return')
axes[0, 1].set_title('Risk-Return Profile')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Cumulative returns
for name, cumulative in portfolio_return_series.items():
    axes[1, 0].plot(cumulative.index, cumulative.values, 
                   label=name, linewidth=2)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Cumulative Growth ($1 initial)')
axes[1, 0].set_title('Portfolio Growth Over Time')
axes[1, 0].legend(loc='best', fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Rolling Sharpe ratio (1-year window)
rolling_window = 252  # 1 year
for name, weights in list(portfolios.items())[:3]:  # Top 3 for clarity
    port_returns = (returns * weights).sum(axis=1)
    excess_returns = port_returns - rf_daily
    
    rolling_sharpe = (excess_returns.rolling(rolling_window).mean() * 252) / \
                     (port_returns.rolling(rolling_window).std() * np.sqrt(252))
    
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values,
                   label=name, linewidth=2)

axes[1, 1].axhline(1.0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Rolling 1-Year Sharpe Ratio')
axes[1, 1].set_title('Time-Varying Sharpe Ratio')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical significance test
print("\n" + "=" * 70)
print("Sharpe Ratio Statistical Significance")
print("=" * 70)
for name, weights in portfolios.items():
    port_returns = (returns * weights).sum(axis=1)
    excess_returns = port_returns - rf_daily
    
    # Standard error of Sharpe ratio (approximation)
    n = len(excess_returns)
    sharpe = results[name]['Sharpe Ratio']
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
    
    # 95% confidence interval
    ci_lower = sharpe - 1.96 * se_sharpe
    ci_upper = sharpe + 1.96 * se_sharpe
    
    print(f"{name:20s} SR={sharpe:.3f}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]")
```

## 6. Challenge Round
When is Sharpe ratio misleading?
- Non-normal returns: Fat tails, skewness ignored (use Sortino or Omega ratio)
- Autocorrelated returns: Smoothed hedge fund returns artificially lower volatility
- Different time periods: 2008 crisis vs 2010s bull market give vastly different Sharpes
- Leverage/derivatives: Options strategies have asymmetric payoffs, volatility not full story
- Negative Sharpe ratios: Ordering inverts (less negative is better, unintuitive)

## 7. Key References
- [Sharpe, W. (1994) "The Sharpe Ratio" Journal of Portfolio Management](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)
- [Investopedia - Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)

---
**Status:** Standard risk-adjusted performance metric | **Complements:** Treynor Ratio, Information Ratio, Sortino Ratio
