import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def fama_french_3factor(asset_returns, factors, rf=0):
    """FF3 regression"""
    excess_returns = asset_returns - rf
    common_dates = excess_returns.index.intersection(factors.index)
    
    y = excess_returns.loc[common_dates]
    X = sm.add_constant(factors.loc[common_dates])
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'beta': results.params['Mkt-RF'],
        'smb': results.params['SMB'],
        'hml': results.params['HML'],
        'alpha_tstat': results.tvalues['const'],
        'rsquared': results.rsquared,
        'results': results
    }

def carhart_4factor(asset_returns, factors, rf=0):
    """Carhart 4-factor regression"""
    excess_returns = asset_returns - rf
    common_dates = excess_returns.index.intersection(factors.index)
    
    y = excess_returns.loc[common_dates]
    X = sm.add_constant(factors.loc[common_dates])
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'beta': results.params['Mkt-RF'],
        'smb': results.params['SMB'],
        'hml': results.params['HML'],
        'wml': results.params['WML'],
        'alpha_tstat': results.tvalues['const'],
        'wml_tstat': results.tvalues['WML'],
        'rsquared': results.rsquared,
        'results': results
    }

# Run regressions
ff3_results = {}
carhart_results = {}

for ticker in tickers.keys():
    asset_ret = returns[ticker]
    
    ff3_results[ticker] = fama_french_3factor(asset_ret, factors_ff3, rf_monthly)
    carhart_results[ticker] = carhart_4factor(asset_ret, factors_carhart, rf_monthly)

# Comparison DataFrame
comparison_data = []
for ticker in tickers.keys():
    ff3 = ff3_results[ticker]
    c4 = carhart_results[ticker]
    
    comparison_data.append({
        'Ticker': ticker,
        'Fund': tickers[ticker],
        'FF3 Alpha (%)': ff3['alpha'] * 12 * 100,
        'FF3 R²': ff3['rsquared'],
        'C4 Alpha (%)': c4['alpha'] * 12 * 100,
        'C4 WML': c4['wml'],
        'WML t-stat': c4['wml_tstat'],
        'C4 R²': c4['rsquared'],
        'R² Improvement': c4['rsquared'] - ff3['rsquared']
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*120)
print("CARHART 4-FACTOR MODEL: Comparison with FF3")
print("="*120)
print(comparison_df.round(4).to_string(index=False))

# Detailed output for momentum fund (MTUM)
print("\n" + "="*120)
print("DETAILED REGRESSION: MTUM (iShares Momentum ETF)")
print("="*120)
print("\nFama-French 3-Factor Model:")
print(ff3_results['MTUM']['results'].summary())

print("\n" + "="*120)
print("Carhart 4-Factor Model:")
print("="*120)
print(carhart_results['MTUM']['results'].summary())

# Momentum factor statistics
print("\n" + "="*120)
print("MOMENTUM FACTOR (WML) STATISTICS")
print("="*120)
wml_stats = pd.DataFrame({
    'Mean (Monthly %)': [wml.mean() * 100],
    'Mean (Annual %)': [wml.mean() * 12 * 100],
    'Std Dev (Annual %)': [wml.std() * np.sqrt(12) * 100],
    'Skewness': [wml.skew()],
    'Kurtosis': [wml.kurtosis()],
    'Sharpe Ratio': [(wml.mean() / wml.std()) * np.sqrt(12)],
    'Min (Monthly %)': [wml.min() * 100],
    'Max (Monthly %)': [wml.max() * 100]
})
print(wml_stats.T.round(3))

# Factor correlations
print("\n" + "="*120)
print("FACTOR CORRELATION MATRIX")
print("="*120)
factor_corr = factors_carhart.corr()
print(factor_corr.round(3))

# Momentum persistence test
print("\n" + "="*120)
print("MOMENTUM PERSISTENCE: Cumulative Returns by Decile")
print("="*120)

# Sort funds by WML loading
wml_loadings = comparison_df.set_index('Ticker')['C4 WML'].sort_values()

print("\nFunds ranked by Momentum Loading (WML):")
for ticker, loading in wml_loadings.items():
    print(f"  {ticker:>6} ({tickers[ticker]:>25}): {loading:>7.3f}")

# Top vs Bottom momentum exposure
high_mom_tickers = wml_loadings.iloc[-3:].index  # Top 3
low_mom_tickers = wml_loadings.iloc[:3].index    # Bottom 3

high_mom_returns = returns[high_mom_tickers].mean(axis=1)
low_mom_returns = returns[low_mom_tickers].mean(axis=1)

high_mom_cum = (1 + high_mom_returns).cumprod()
low_mom_cum = (1 + low_mom_returns).cumprod()

print(f"\nCumulative Returns:")
print(f"  High Momentum Exposure: {(high_mom_cum.iloc[-1] - 1) * 100:>6.1f}%")
print(f"  Low Momentum Exposure:  {(low_mom_cum.iloc[-1] - 1) * 100:>6.1f}%")
print(f"  Difference:             {((high_mom_cum.iloc[-1] - low_mom_cum.iloc[-1])) * 100:>6.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Alpha comparison (FF3 vs Carhart)
x = np.arange(len(comparison_df))
width = 0.35

bars1 = axes[0, 0].bar(x - width/2, comparison_df['FF3 Alpha (%)'], width,
                       label='FF3 Alpha', alpha=0.8)
bars2 = axes[0, 0].bar(x + width/2, comparison_df['C4 Alpha (%)'], width,
                       label='Carhart Alpha', alpha=0.8)

axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(comparison_df['Ticker'], rotation=45, ha='right')
axes[0, 0].set_ylabel('Alpha (% per year)')
axes[0, 0].set_title('Alpha Comparison: FF3 vs Carhart 4-Factor')
axes[0, 0].axhline(0, color='black', linewidth=0.5)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: WML loadings
colors = ['red' if x < 0 else 'green' for x in comparison_df['C4 WML']]
bars = axes[0, 1].barh(comparison_df['Ticker'], comparison_df['C4 WML'], color=colors, alpha=0.7)

axes[0, 1].axvline(0, color='black', linewidth=0.5)
axes[0, 1].set_xlabel('Momentum Loading (WML)')
axes[0, 1].set_title('Momentum Factor Exposure by Fund')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: R² improvement
axes[1, 0].scatter(comparison_df['FF3 R²'], comparison_df['C4 R²'], s=100, alpha=0.7)
for idx, row in comparison_df.iterrows():
    axes[1, 0].annotate(row['Ticker'],
                       (row['FF3 R²'], row['C4 R²']),
                       fontsize=8, ha='right')

# Diagonal line
axes[1, 0].plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3, label='Equal R²')
axes[1, 0].set_xlabel('FF3 R²')
axes[1, 0].set_ylabel('Carhart 4-Factor R²')
axes[1, 0].set_title('Explanatory Power: FF3 vs Carhart')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim([0.5, 1.0])
axes[1, 0].set_ylim([0.5, 1.0])

# Plot 4: Cumulative WML factor performance
wml_cumulative = (1 + wml).cumprod()
axes[1, 1].plot(wml_cumulative.index, wml_cumulative, linewidth=2, label='WML Factor')
axes[1, 1].set_ylabel('Cumulative Return')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_title('Momentum Factor (WML) Performance Over Time')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Highlight drawdowns
drawdown = (wml_cumulative / wml_cumulative.cummax() - 1)
axes[1, 1].fill_between(wml_cumulative.index, 
                        wml_cumulative.values.flatten(), 
                        (wml_cumulative * (1 + drawdown)).values.flatten(),
                        alpha=0.3, color='red', label='Drawdowns')

plt.tight_layout()
plt.show()

# Momentum crash analysis
print("\n" + "="*120)
print("MOMENTUM CRASH ANALYSIS")
print("="*120)

# Worst months for momentum
worst_months = wml.nsmallest(5)
print("\nWorst 5 Months for Momentum Factor:")
for date, ret in worst_months.items():
    print(f"  {date.strftime('%Y-%m')}: {ret*100:>7.2f}%")

# Maximum drawdown
cumulative = (1 + wml).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative / running_max - 1)
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()

print(f"\nMaximum Drawdown: {max_dd*100:.2f}% (on {max_dd_date.strftime('%Y-%m')})")

# Skewness and tail risk
print(f"\nTail Risk Metrics:")
print(f"  Skewness: {wml.skew():.3f} (negative = left tail risk)")
print(f"  Kurtosis: {wml.kurtosis():.3f} (>3 = fat tails)")
print(f"  5th Percentile: {np.percentile(wml, 5)*100:.2f}%")
print(f"  1st Percentile: {np.percentile(wml, 1)*100:.2f}%")

# Key insights
print("\n" + "="*120)
print("KEY INSIGHTS: CARHART FOUR-FACTOR MODEL")
print("="*120)
print("1. Momentum factor (WML) improves R² by 1-3% over FF3 (especially for mutual funds)")
print("2. Positive WML loading indicates trend-following/momentum strategy")
print("3. Many 'hot' funds simply have momentum exposure, not genuine skill (alpha → 0)")
print("4. WML factor has strong premium (~9% annual) but high volatility (~15-20%)")
print("5. Momentum exhibits negative skewness (crash risk): severe losses in reversals")
print("6. Low correlation between momentum and value (HML): diversification benefit")
print("7. Formation period t-12 to t-2 (skip recent month to avoid short-term reversal)")
print("8. Carhart model standard for mutual fund evaluation and performance attribution")

print("\n" + "="*120)
print("FACTOR PREMIUM SUMMARY (Sample Period)")
print("="*120)
factor_premiums = factors_carhart.mean() * 12 * 100
factor_vols = factors_carhart.std() * np.sqrt(12) * 100
factor_sharpe = (factors_carhart.mean() / factors_carhart.std()) * np.sqrt(12)

premium_summary = pd.DataFrame({
    'Premium (%)': factor_premiums,
    'Volatility (%)': factor_vols,
    'Sharpe Ratio': factor_sharpe
})
print(premium_summary.round(3))

print("\nNote: WML typically has highest Sharpe but highest crash risk (negative skew)")