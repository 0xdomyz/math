# Jensen's Alpha

## 1. Concept Skeleton
**Definition:** Portfolio excess return above CAPM prediction; measures active management skill after adjusting for systematic risk  
**Purpose:** Isolate manager value-add independent of market timing, quantify alpha generation  
**Prerequisites:** CAPM, beta estimation, regression analysis, risk-adjusted performance

## 2. Comparative Framing
| Metric | Jensen's Alpha | Information Ratio | Treynor Ratio |
|--------|----------------|-------------------|---------------|
| **Output** | Absolute return (%) | Ratio (alpha/TE) | Ratio (excess/β) |
| **Interpretation** | Outperformance magnitude | Consistency of outperformance | Return per systematic risk |
| **Risk Adjustment** | Beta (CAPM) | Tracking error | Beta |
| **Best For** | CAPM attribution | Active manager evaluation | Diversified portfolio ranking |

## 3. Examples + Counterexamples

**Simple Example:**  
Fund return: 15%, Beta: 1.2, Market return: 12%, rf: 3%  
Expected return (CAPM): 3% + 1.2×(12%-3%) = 13.8%  
Jensen's Alpha: 15% - 13.8% = +1.2% (positive alpha, manager added value)

**Failure Case:**  
High beta fund in bull market: 20% return, β=1.8, market +15%  
Expected: 3% + 1.8×(15%-3%) = 24.6%  
Alpha: 20% - 24.6% = -4.6% (negative alpha, underperformed risk-adjusted)

**Edge Case:**  
Market-neutral fund: β≈0, return 5%, market ±10%  
Alpha ≈ 5% - 3% = 2% (CAPM expected ≈ rf since β=0)

## 4. Layer Breakdown
```
Jensen's Alpha Calculation:
├─ CAPM Framework:
│   ├─ Expected Return: E[Ri] = rf + βi(E[Rm] - rf)
│   ├─ Market Risk Premium: E[Rm] - rf
│   └─ Beta: Systematic risk sensitivity to market
├─ Regression Approach:
│   ├─ Model: Rp,t - rf,t = α + β(Rm,t - rf,t) + εt
│   ├─ α (Jensen's Alpha): Regression intercept
│   ├─ β (Beta): Regression slope
│   ├─ εt (Residual): Idiosyncratic component
│   └─ R²: Proportion of variance explained by market
├─ Interpretation:
│   ├─ α > 0: Outperformance (positive abnormal return)
│   ├─ α = 0: Fair compensation for risk taken (CAPM holds)
│   ├─ α < 0: Underperformance (destroyed value)
│   └─ Statistical Significance: t-test on α coefficient
├─ Annualization:
│   ├─ Monthly Data: α_annual = α_monthly × 12
│   ├─ Daily Data: α_annual = α_daily × 252
│   └─ Standard Error: SE_annual = SE_period × √periods
└─ Decomposition:
    ├─ Total Return = rf + β(Rm - rf) + α + ε
    ├─ Risk-Free Return: Compensation for time
    ├─ Systematic Return: Beta-driven market exposure
    ├─ Alpha: Manager skill
    └─ Residual: Unexplained noise
```

**Interaction:** Alpha isolates skill from beta exposure; requires accurate beta estimation

## 5. Mini-Project
Calculate and test statistical significance of Jensen's alpha for mutual funds:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

# Parameters
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Download market and risk-free proxy data
market = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
market_returns = market.pct_change().dropna()

# Use 3-month T-bill rate (approximate with constant for simplicity)
# In practice, download actual T-bill rates
rf_annual = 0.03
rf_monthly = (1 + rf_annual) ** (1/12) - 1

# Sample mutual funds / ETFs
funds = {
    'Berkshire Hathaway': 'BRK-B',
    'ARK Innovation': 'ARKK',
    'Vanguard Value': 'VTV',
    'Invesco QQQ': 'QQQ',
    'SPDR S&P 500': 'SPY'  # Should have alpha ≈ 0
}

def calculate_jensens_alpha(fund_returns, market_returns, rf_rate):
    """
    Calculate Jensen's alpha via regression
    Returns alpha, beta, t-statistic, p-value, R-squared
    """
    # Align dates
    aligned = pd.concat([fund_returns, market_returns], axis=1).dropna()
    aligned.columns = ['fund', 'market']
    
    # Calculate excess returns
    fund_excess = aligned['fund'] - rf_rate
    market_excess = aligned['market'] - rf_rate
    
    # Regression: fund_excess = alpha + beta * market_excess + error
    X = sm.add_constant(market_excess)
    model = sm.OLS(fund_excess, X).fit()
    
    alpha = model.params[0]
    beta = model.params[1]
    t_stat = model.tvalues[0]
    p_value = model.pvalues[0]
    r_squared = model.rsquared
    
    # Standard errors
    alpha_se = model.bse[0]
    beta_se = model.bse[1]
    
    return {
        'alpha': alpha,
        'beta': beta,
        'alpha_se': alpha_se,
        'beta_se': beta_se,
        't_statistic': t_stat,
        'p_value': p_value,
        'r_squared': r_squared,
        'residuals': model.resid
    }

# Analyze funds
results = {}
fund_returns_data = {}

for name, ticker in funds.items():
    try:
        # Download fund data
        fund_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
        fund_returns = fund_data.pct_change().dropna()
        
        # Resample to monthly for more stable estimates
        fund_monthly = fund_data.resample('M').last().pct_change().dropna()
        market_monthly = market.resample('M').last().pct_change().dropna()
        
        # Calculate alpha
        alpha_stats = calculate_jensens_alpha(fund_monthly, market_monthly, rf_monthly)
        
        # Annualize alpha
        alpha_annual = alpha_stats['alpha'] * 12
        alpha_se_annual = alpha_stats['alpha_se'] * np.sqrt(12)
        
        # Calculate average returns
        avg_fund_return = fund_monthly.mean() * 12
        avg_market_return = market_monthly.mean() * 12
        
        # CAPM expected return
        capm_expected = rf_annual + alpha_stats['beta'] * (avg_market_return - rf_annual)
        
        results[name] = {
            'Ticker': ticker,
            'Alpha (Annual)': alpha_annual,
            'Alpha SE (Annual)': alpha_se_annual,
            't-statistic': alpha_stats['t_statistic'],
            'p-value': alpha_stats['p_value'],
            'Beta': alpha_stats['beta'],
            'R-squared': alpha_stats['r_squared'],
            'Actual Return': avg_fund_return,
            'CAPM Expected': capm_expected,
            'Outperformance': avg_fund_return - capm_expected,
            'residuals': alpha_stats['residuals']
        }
        
        fund_returns_data[name] = fund_monthly
        
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Create summary table
summary_cols = ['Ticker', 'Alpha (Annual)', 'Alpha SE (Annual)', 't-statistic', 
                'p-value', 'Beta', 'R-squared', 'Actual Return', 
                'CAPM Expected', 'Outperformance']
summary = pd.DataFrame({k: {col: v[col] for col in summary_cols} 
                       for k, v in results.items()}).T
summary = summary.sort_values('Alpha (Annual)', ascending=False)

print("Jensen's Alpha Analysis")
print("=" * 110)
print(summary.to_string())
print("\n" + "=" * 110)
print("Interpretation:")
print("  Alpha > 0: Outperformance after risk adjustment")
print("  p-value < 0.05: Statistically significant (95% confidence)")
print("  |t-stat| > 2: Roughly significant at 5% level")
print("  R² close to 1: Returns well-explained by market (high systematic risk)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Alpha with confidence intervals
fund_names = summary.index
alphas = summary['Alpha (Annual)']
alpha_ses = summary['Alpha SE (Annual)']
colors = ['green' if p < 0.05 and a > 0 else 'red' if p < 0.05 and a < 0 
          else 'gray' for a, p in zip(alphas, summary['p-value'])]

y_pos = np.arange(len(fund_names))
axes[0, 0].barh(y_pos, alphas, color=colors, alpha=0.6)
axes[0, 0].errorbar(alphas, y_pos, xerr=1.96*alpha_ses, 
                    fmt='none', ecolor='black', capsize=5)
axes[0, 0].set_yticks(y_pos)
axes[0, 0].set_yticklabels(fund_names)
axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Jensen\'s Alpha (Annual %)')
axes[0, 0].set_title('Jensen\'s Alpha with 95% Confidence Intervals')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Security Characteristic Line (SCL) for top fund
top_fund = summary.index[0]
if top_fund in fund_returns_data and top_fund in results:
    fund_rets = fund_returns_data[top_fund]
    market_rets = market.resample('M').last().pct_change().dropna()
    
    aligned = pd.concat([fund_rets, market_rets], axis=1).dropna()
    aligned.columns = ['fund', 'market']
    
    fund_excess = (aligned['fund'] - rf_monthly) * 100  # Convert to %
    market_excess = (aligned['market'] - rf_monthly) * 100
    
    axes[0, 1].scatter(market_excess, fund_excess, alpha=0.6, s=50)
    
    # Regression line
    z = np.polyfit(market_excess, fund_excess, 1)
    p = np.poly1d(z)
    x_line = np.linspace(market_excess.min(), market_excess.max(), 100)
    axes[0, 1].plot(x_line, p(x_line), 'r-', linewidth=2, 
                    label=f'α={results[top_fund]["Alpha (Annual)"]:.2%}, β={results[top_fund]["Beta"]:.2f}')
    
    axes[0, 1].set_xlabel('Market Excess Return (%)')
    axes[0, 1].set_ylabel(f'{top_fund} Excess Return (%)')
    axes[0, 1].set_title(f'Security Characteristic Line: {top_fund}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

# Plot 3: Actual vs CAPM Expected Returns
x_pos = np.arange(len(summary))
width = 0.35

bars1 = axes[1, 0].bar(x_pos - width/2, summary['Actual Return'], 
                       width, label='Actual Return', alpha=0.8)
bars2 = axes[1, 0].bar(x_pos + width/2, summary['CAPM Expected'], 
                       width, label='CAPM Expected', alpha=0.8)

axes[1, 0].set_xlabel('Fund')
axes[1, 0].set_ylabel('Annual Return (%)')
axes[1, 0].set_title('Actual vs CAPM Expected Returns')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(summary.index, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Add outperformance annotations
for i, (actual, expected) in enumerate(zip(summary['Actual Return'], 
                                           summary['CAPM Expected'])):
    diff = actual - expected
    color = 'green' if diff > 0 else 'red'
    axes[1, 0].annotate(f'{diff:+.1%}', xy=(i, max(actual, expected)),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8, color=color, weight='bold')

# Plot 4: Beta vs Alpha scatter
scatter = axes[1, 1].scatter(summary['Beta'], summary['Alpha (Annual)'],
                            s=300, alpha=0.6,
                            c=summary['p-value'], cmap='RdYlGn_r',
                            vmin=0, vmax=0.1)

for idx in summary.index:
    axes[1, 1].annotate(idx.split()[0],  # First word only
                       (summary.loc[idx, 'Beta'],
                        summary.loc[idx, 'Alpha (Annual)']),
                       fontsize=8, ha='center')

axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 1].axvline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Beta (Systematic Risk)')
axes[1, 1].set_ylabel('Jensen\'s Alpha (Annual %)')
axes[1, 1].set_title('Alpha vs Beta (color = p-value)')
axes[1, 1].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='p-value')

plt.tight_layout()
plt.show()

# Statistical tests summary
print("\n" + "=" * 110)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 110)
for name in summary.index:
    alpha = results[name]['Alpha (Annual)']
    t_stat = results[name]['t-statistic']
    p_val = results[name]['p-value']
    
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    conclusion = "Significant" if p_val < 0.05 else "Not Significant"
    
    print(f"{name:20s} α={alpha:+.3%} {sig:3s} (t={t_stat:+.2f}, p={p_val:.4f}) {conclusion}")

print("\n*** p<0.01, ** p<0.05, * p<0.1")
```

## 6. Challenge Round
When is Jensen's alpha appropriate?
- CAPM believers: Evaluating performance within equilibrium framework
- Single-factor attribution: When market is dominant risk factor
- Long-only equity funds: Traditional active management
- Historical performance: Measuring past value-added

Jensen's alpha limitations:
- CAPM assumptions: Fails if CAPM doesn't hold (empirically weak)
- Single factor: Ignores size, value, momentum factors (use multi-factor models)
- Market timing: Confounds stock selection skill with market timing
- Benchmark sensitivity: Different market proxies give different alphas
- Non-normal returns: Assumes symmetric risk, ignores skewness/kurtosis

Better alternatives:
- Fama-French alpha: 3/5-factor models more realistic
- Conditional models: Time-varying betas, macro factors
- Style analysis: Returns-based attribution (Sharpe)
- Omega ratio: Captures all moments of return distribution

## 7. Key References
- [Jensen, M. (1968) "The Performance of Mutual Funds" Journal of Finance](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1968.tb00815.x)
- [Investopedia - Jensen's Alpha](https://www.investopedia.com/terms/j/jensensmeasure.asp)
- [Fama & French (2010) "Luck versus Skill in Mutual Fund Returns"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X10001315)

---
**Status:** CAPM-based performance measure | **Complements:** Information Ratio, Sharpe Ratio, Multi-Factor Models
