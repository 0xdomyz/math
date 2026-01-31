# Carhart Four-Factor Model

## 1. Concept Skeleton
**Definition:** Extension of Fama-French three-factor model adding momentum (WML) factor; captures trend-following premium  
**Purpose:** Explain momentum anomaly; improve mutual fund performance attribution; decompose active returns  
**Prerequisites:** Fama-French 3F model, momentum effect, time series regression, factor construction

## 2. Comparative Framing
| Model | CAPM | FF3 | Carhart 4F | FF5 | Q-Factor |
|-------|------|-----|-----------|-----|----------|
| **Factors** | Market | + Size, Value | + Momentum | + Profitability, Investment | ROE, Investment, Size |
| **R² (avg)** | ~70% | ~90% | ~92% | ~93% | ~91% |
| **Momentum** | No | No | Yes (WML) | No | Partially (via ROE) |
| **Mutual Funds** | Poor fit | Good fit | Best fit (fund persistence) | Very good | Good |
| **Academic Use** | Standard | Very common | Common (funds) | Increasing | Emerging |

## 3. Examples + Counterexamples

**Momentum Premium:**  
Past winners (top 30% 12-month return) outperform past losers (bottom 30%) by ~9% annually  
WML factor captures this; positive WML loading → momentum/trend-following strategy

**Mutual Fund Persistence:**  
Top-quartile funds (1-year) have 60% chance of remaining top-half next year  
Carhart model explains this as momentum exposure, not manager skill (alpha ≈ 0)

**Reversal Period:**  
Momentum works 3-12 months, reverses at 1-week and 3-5 years  
Skip most recent month in formation (t-2 to t-12), avoid short-term reversal

**Failure Case:**  
2009 crisis: Momentum crashed (-73% in 2009); worst drawdown ever  
WML highly volatile, negative skew (crash risk); can devastate momentum strategies

## 4. Layer Breakdown
```
Carhart Four-Factor Model:
├─ Model Specification:
│   ├─ Ri - rf = αi + βi(Rm - rf) + si·SMB + hi·HML + mi·WML + εi
│   ├─ First 3 factors: Fama-French (Mkt-RF, SMB, HML)
│   ├─ WML (Winners Minus Losers): Momentum factor
│   ├─ mi: Momentum loading (factor sensitivity)
│   ├─ mi > 0: Positive momentum exposure (trend-following)
│   ├─ mi < 0: Negative momentum (contrarian)
│   └─ αi: Four-factor alpha (after adjusting for all 4 factors)
├─ WML Factor Construction (Jegadeesh-Titman methodology):
│   ├─ Formation Period: Use returns from t-12 to t-2 (skip most recent month)
│   │   ├─ Why skip t-1? Avoid short-term reversal (1-month mean reversion)
│   │   ├─ 11-month lookback: Long enough for trend, not too long (>1 year reverses)
│   │   └─ Monthly rebalancing: Update winners/losers each month
│   ├─ Ranking & Portfolio Formation:
│   │   ├─ Sort all stocks by past 11-month return (t-12 to t-2)
│   │   ├─ Top 30%: Winners (high momentum)
│   │   ├─ Bottom 30%: Losers (low/negative momentum)
│   │   ├─ Middle 40%: Ignored (focus on extremes)
│   │   └─ Within each group, can further split by size (small/big)
│   ├─ WML Calculation:
│   │   ├─ WML = Return(Winners) - Return(Losers)
│   │   ├─ Hold for 1 month, then rebalance
│   │   ├─ Value-weighted or equal-weighted portfolio
│   │   └─ Alternative: Use all stocks with formation period weights
│   └─ Data: NYSE/AMEX/NASDAQ stocks, 1926-present (Ken French library)
├─ Economic Rationale for Momentum:
│   ├─ Behavioral Explanations:
│   │   ├─ Underreaction: Investors slow to incorporate news → gradual drift
│   │   ├─ Conservatism Bias: Anchoring on prior beliefs, slow updating
│   │   ├─ Confirmation Bias: Seek information confirming trend
│   │   ├─ Herding: Following others into winners (feedback loop)
│   │   └─ Disposition Effect: Hold losers too long, sell winners too soon
│   ├─ Risk-Based Explanations:
│   │   ├─ Time-varying risk premiums: Winners have higher expected returns
│   │   ├─ Industry concentration: Momentum clusters in sectors
│   │   ├─ Liquidity: Winners more liquid → lower required return (but momentum persists)
│   │   └─ Data mining: Skeptics argue it's spurious (but robust internationally)
│   ├─ Market Microstructure:
│   │   ├─ Information diffusion: News spreads slowly across investors
│   │   ├─ Price pressure: Buying begets more buying (positive feedback)
│   │   └─ Limits to arbitrage: Costly to short losers, constrained capital
│   └─ Debate: Behavioral (overreaction/underreaction) vs Risk-based (incomplete)
├─ Momentum Characteristics:
│   ├─ Premium Magnitude: ~9% annually (1926-2020), robust across periods
│   ├─ Volatility: High (~15-20% annual), higher than size/value factors
│   ├─ Skewness: Strongly negative (-3 to -5); crash risk in reversals
│   ├─ Correlation: Low with value (HML ≈ -0.2), low with SMB
│   ├─ Time-varying: Works best in trending markets; fails in whipsaw/crisis
│   ├─ Horizon: Strongest at 6-12 months; reverses at <1 month and >3 years
│   └─ International: Works globally; consistent across countries
├─ Carhart Model Applications:
│   ├─ Mutual Fund Evaluation:
│   │   ├─ Original purpose (Carhart 1997): Explain fund persistence
│   │   ├─ Hot hands: Top funds have positive WML exposure (momentum traders)
│   │   ├─ True alpha rare: Most "skilled" managers just momentum bets
│   │   └─ 4-factor alpha better benchmark than CAPM or FF3
│   ├─ Hedge Fund Analysis:
│   │   ├─ Many strategies have momentum exposure (trend-following, CTAs)
│   │   ├─ Decompose returns into systematic factors + genuine alpha
│   │   └─ Momentum exposure explains much of hedge fund "alpha"
│   ├─ Portfolio Construction:
│   │   ├─ Target momentum exposure: Positive for trend-following, negative for contrarian
│   │   ├─ Factor timing: Overweight momentum in trending regimes
│   │   └─ Risk management: Monitor WML exposure (crash risk)
│   ├─ Performance Attribution:
│   │   ├─ Separate skill (alpha) from factor exposures (beta, size, value, momentum)
│   │   ├─ Industry standard for mutual fund analysis
│   │   └─ Compensation structure: Pay for alpha, not beta exposure
│   └─ Smart Beta / Factor Investing:
│       ├─ Momentum ETFs: Target positive WML exposure
│       ├─ Multi-factor strategies: Combine value + momentum (low correlation)
│       └─ Risk parity across factors: Equal risk allocation to 4 factors
├─ Empirical Evidence (1926-2020):
│   ├─ WML Premium: ~9% annually (robust, persistent)
│   ├─ R² Improvement: FF3 (90%) → Carhart (92%) for mutual funds
│   ├─ Mutual Fund Alpha: Most funds have insignificant 4-factor alpha
│   ├─ Momentum Crashes: 1932 (-70%), 2009 (-73%); extreme negative skew
│   ├─ Factor Correlations:
│   │   ├─ WML vs Mkt-RF: ≈ -0.1 (slightly negative)
│   │   ├─ WML vs SMB: ≈ 0.0 (uncorrelated)
│   │   ├─ WML vs HML: ≈ -0.2 (negative; value and momentum diverge)
│   │   └─ Diversification benefit: Low correlation among factors
│   └─ International: Momentum premium consistent globally (Europe, Asia, emerging)
├─ Implementation Considerations:
│   ├─ Transaction Costs:
│   │   ├─ Monthly rebalancing: Higher turnover than value strategies
│   │   ├─ Bid-ask spread: 20-50 bps for small/mid-cap stocks
│   │   ├─ Market impact: Large orders move prices (slippage)
│   │   └─ After-costs premium: ~6% (vs 9% gross)
│   ├─ Short Selling:
│   │   ├─ WML requires shorting losers (borrow costs, constraints)
│   │   ├─ Long-only momentum: Winners only (half the portfolio)
│   │   ├─ Performance: Long-only ~60-70% of WML premium
│   │   └─ Regulatory: Some funds can't short (mutual funds)
│   ├─ Capacity:
│   │   ├─ Small-cap momentum: Higher returns but limited capacity
│   │   ├─ Large-cap only: More scalable, lower returns (~6-7%)
│   │   └─ Crowding: As more investors adopt, premiums may compress
│   └─ Risk Management:
│       ├─ Tail risk: Momentum crashes severe (VaR, CVaR underestimate)
│       ├─ Stop-loss: Exit when WML factor declines sharply
│       ├─ Volatility scaling: Reduce exposure in high-vol regimes
│       └─ Diversification: Combine with value (negative correlation helps)
├─ Carhart Model Limitations:
│   ├─ Still Missing Factors: Profitability, investment (captured by FF5)
│   ├─ Momentum Crashes: Model doesn't predict reversals (tail risk)
│   ├─ Time-varying Premiums: Factor loadings stable, but premiums change
│   ├─ Industry Effects: Momentum clusters in industries (missing sector factor?)
│   ├─ Behavioral vs Risk: Doesn't resolve economic source of momentum
│   └─ Short-term Reversal: Model uses t-2 to t-12 (ignores short reversal)
└─ Extensions & Alternatives:
    ├─ FF5 Model: Adds profitability, investment; omits momentum
    ├─ FF6 Model: FF5 + momentum (research ongoing)
    ├─ Q-Factor Model: Hou-Xue-Zhang (profitability, investment, size); competes with FF
    ├─ Betting-Against-Beta: Low-beta stocks outperform (Frazzini-Pedersen)
    └─ Quality Factor: High-quality stocks outperform (profitability + low leverage)
```

**Interaction:** Time series regression adds momentum factor to FF3, capturing trend-following premium

## 5. Mini-Project
Implement Carhart four-factor model with momentum factor construction:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats

# Download asset data
tickers = {
    'SPY': 'S&P 500',
    'IWM': 'Russell 2000',
    'MTUM': 'Momentum ETF',
    'VLUE': 'Value ETF',
    'SIZE': 'Size ETF',
    'ARKK': 'Growth/Momentum Fund',
    'VFINX': 'Vanguard 500 Index',
    'FCNTX': 'Fidelity Contrafund',
    'DODGX': 'Dodge & Cox Stock',
    'PRWCX': 'T. Rowe Price Cap Apprec'
}

end_date = datetime.now()
start_date = datetime(2015, 1, 1)

print("Downloading data...")
data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Construct momentum factor (WML proxy)
# Use universe of stocks to create winners - losers
stock_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
                  'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'DIS', 'BAC',
                  'XOM', 'CVX', 'PFE', 'KO', 'PEP', 'COST', 'WMT', 'MCD', 'CSCO',
                  'IBM', 'INTC', 'ORCL', 'T', 'VZ', 'MRK']

print("Downloading stock universe for momentum factor...")
stock_data = yf.download(stock_universe, start=start_date, end=end_date, progress=False)['Adj Close']
stock_returns = stock_data.pct_change().dropna()

def construct_momentum_factor(returns_data, formation_period=11, skip_month=1):
    """
    Construct WML (Winners Minus Losers) momentum factor
    
    Parameters:
    - returns_data: DataFrame of stock returns
    - formation_period: Lookback period (typically 11 months)
    - skip_month: Skip most recent month (avoid short-term reversal)
    """
    wml_series = pd.Series(index=returns_data.index, dtype=float)
    
    for i in range(formation_period + skip_month, len(returns_data)):
        current_date = returns_data.index[i]
        
        # Formation period: t-12 to t-2 (if skip_month=1, formation_period=11)
        formation_start = i - formation_period - skip_month
        formation_end = i - skip_month
        
        # Calculate cumulative returns during formation period
        formation_returns = (1 + returns_data.iloc[formation_start:formation_end]).prod() - 1
        
        # Rank stocks
        valid_stocks = formation_returns.dropna()
        if len(valid_stocks) < 10:  # Need sufficient stocks
            continue
        
        # Top 30% winners, bottom 30% losers
        n_stocks = len(valid_stocks)
        n_winners = int(n_stocks * 0.3)
        n_losers = int(n_stocks * 0.3)
        
        sorted_stocks = valid_stocks.sort_values(ascending=False)
        winners = sorted_stocks.iloc[:n_winners].index
        losers = sorted_stocks.iloc[-n_losers:].index
        
        # Holding period return (current month)
        if i < len(returns_data):
            winner_return = returns_data.loc[current_date, winners].mean()
            loser_return = returns_data.loc[current_date, losers].mean()
            
            wml_series[current_date] = winner_return - loser_return
    
    return wml_series.dropna()

# Construct WML factor
print("Constructing momentum factor (WML)...")
wml = construct_momentum_factor(stock_returns, formation_period=11, skip_month=1)

# Construct other factors (proxies using ETFs)
rf_monthly = 0.02 / 12  # Risk-free rate approximation

# Market factor
mkt_rf = returns['SPY'] - rf_monthly

# SMB proxy (small - big)
smb = returns['IWM'] - returns['SPY']

# HML proxy (value - growth)
hml = returns['VLUE'] - returns['MTUM']  # Approximation

# Create factor DataFrame
# Align dates (WML may have fewer observations due to formation period)
factors_ff3 = pd.DataFrame({
    'Mkt-RF': mkt_rf,
    'SMB': smb,
    'HML': hml
})

factors_carhart = pd.DataFrame({
    'Mkt-RF': mkt_rf,
    'SMB': smb,
    'HML': hml,
    'WML': wml
})

# Remove NaN
factors_ff3 = factors_ff3.dropna()
factors_carhart = factors_carhart.dropna()

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
```

## 6. Challenge Round
Why add momentum to Fama-French three-factor model?
- Momentum anomaly: Past winners outperform losers by ~9% annually (robust, persistent)
- Mutual fund persistence: Hot funds have positive WML exposure (not skill, just factor)
- R² improvement: Especially significant for actively managed funds (2-3% increase)
- Low correlation: WML uncorrelated with HML (~-0.2); diversification benefit
- Behavioral foundation: Underreaction, herding, confirmation bias drive momentum

Momentum's dark side (why it's controversial):
- Crash risk: Severe reversals (-70% in 1932, -73% in 2009); worst factor drawdowns
- Negative skewness: Fat left tail; standard risk metrics (vol, Sharpe) misleading
- Transaction costs: Monthly rebalancing; high turnover reduces net returns (~3%)
- Short selling: Requires shorting losers (borrow costs, constraints limit implementation)
- Crowding: As strategy becomes popular, premiums may compress (capacity limits)

Why doesn't FF5 include momentum?
- Fama-French perspective: Momentum is behavioral (mispricing), not risk factor
- Profitability + investment capture some momentum-like effects
- Separate papers: FF publish momentum data but don't integrate into main model
- Carhart model serves that role: FF3 + WML when momentum needed
- Academic debate: Risk-based (FF) vs behavioral (momentum) explanations

## 7. Key References
- [Carhart, M.M. (1997) "On Persistence in Mutual Fund Performance"](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb03808.x) - Original 4-factor paper
- [Jegadeesh, N. & Titman, S. (1993) "Returns to Buying Winners and Selling Losers"](https://www.jstor.org/stable/2328882) - Momentum anomaly
- [Daniel, K. & Moskowitz, T.J. (2016) "Momentum Crashes"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X16300800)
- [Kenneth French Data Library - Momentum Factor](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

---
**Status:** Industry standard for mutual fund evaluation | **Complements:** FF3, Momentum Strategies, Performance Attribution
