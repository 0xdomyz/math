# Diversification

## 1. Concept Skeleton
**Definition:** Risk reduction through combining assets with imperfect correlation; portfolio volatility lower than weighted average of individual volatilities  
**Purpose:** Eliminate idiosyncratic risk, improve risk-adjusted returns without sacrificing expected return  
**Prerequisites:** Portfolio variance, correlation coefficient, systematic vs unsystematic risk

## 2. Comparative Framing
| Approach | Naive Diversification | Optimal Diversification | Over-Diversification |
|----------|----------------------|------------------------|----------------------|
| **Method** | Equal weight many assets | Mean-variance optimization | Excessive asset count |
| **Assets Needed** | 20-30 stocks | Depends on correlations | 100+ holdings |
| **Risk Reduction** | ~70% idiosyncratic eliminated | Maximum feasible | Diminishing marginal benefit |
| **Complexity** | Simple (1/n rule) | Requires estimation | High tracking, management costs |

## 3. Examples + Counterexamples

**Simple Example:**  
Two uncorrelated assets (ρ=0): Portfolio with 50-50 allocation has σp = 0.707×σindividual (29.3% risk reduction)

**Failure Case:**  
2008 financial crisis: Asset correlations → 1 during extreme stress, diversification failed when needed most

**Edge Case:**  
Perfect negative correlation (ρ=-1): Can construct zero-variance portfolio, complete risk elimination

## 4. Layer Breakdown
```
Diversification Mechanics:
├─ Mathematical Foundation:
│   ├─ Portfolio Variance: σp² = Σi wi²σi² + Σi Σj≠i wiwjσiσjρij
│   ├─ Two-Asset Case: σp² = w1²σ1² + w2²σ2² + 2w1w2σ1σ2ρ12
│   ├─ Diversification Benefit: σp < Σi wi·σi (when ρ < 1)
│   └─ Maximum Benefit: When correlations are lowest/negative
├─ Risk Decomposition:
│   ├─ Total Risk = Systematic Risk + Idiosyncratic Risk
│   ├─ Systematic (Market): β-driven, non-diversifiable, ~30-50% of total
│   ├─ Idiosyncratic (Specific): Firm-specific events, diversifiable to ~0
│   └─ Well-Diversified: Only systematic risk remains
├─ Asymptotic Results:
│   ├─ Equal-Weighted n Assets: σp² → Average Covariance as n→∞
│   ├─ Formula: σp² = (1/n)·σ̄² + (1 - 1/n)·Cov̄
│   ├─ Large n: First term → 0, second term → Cov̄
│   └─ Minimum Risk: Limited by average pairwise correlation
├─ Practical Thresholds:
│   ├─ 5-10 stocks: ~40-60% idiosyncratic risk eliminated
│   ├─ 20-30 stocks: ~70-80% elimination (diminishing returns)
│   ├─ 100+ stocks: ~90-95% elimination (approaching market portfolio)
│   └─ Optimal: Balance transaction costs vs marginal risk reduction
├─ Correlation Structure:
│   ├─ Low Correlation (<0.3): High diversification benefit
│   ├─ Moderate (0.3-0.7): Typical equity-equity within market
│   ├─ High (>0.7): Limited benefit (e.g., same sector stocks)
│   └─ Negative: Rare but powerful (gold vs stocks sometimes)
└─ Diversification Failure:
    ├─ Crisis Periods: Correlations spike toward 1 (contagion)
    ├─ Concentrated Exposures: Sector/region/factor concentration
    ├─ Hidden Correlations: Seemingly uncorrelated assets share risks
    └─ Structural Breaks: Historical correlations break down
```

**Interaction:** Benefit proportional to (1-ρ); negative correlations provide hedging not just diversification

## 5. Mini-Project
Demonstrate diversification benefits and measure risk reduction empirically:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from itertools import combinations

# Download diverse asset universe
tickers = {
    'US Large': ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM'],
    'US Small': ['ROKU', 'ETSY', 'PINS'],
    'International': ['EFA', 'EEM'],
    'Bonds': ['AGG', 'TLT'],
    'Commodities': ['GLD', 'DBC'],
    'Real Estate': ['VNQ']
}

all_tickers = [t for group in tickers.values() for t in group]

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Annualize statistics
annual_returns = returns.mean() * 252
annual_vols = returns.std() * np.sqrt(252)
corr_matrix = returns.corr()

print("\nAsset Statistics:")
print("=" * 70)
for ticker in all_tickers:
    print(f"{ticker:6s}: Return={annual_returns[ticker]:>6.2%}, Vol={annual_vols[ticker]:>6.2%}")

print("\n\nCorrelation Matrix (selected pairs):")
print("=" * 70)
sample_pairs = [('AAPL', 'MSFT'), ('SPY', 'AGG'), ('GLD', 'SPY'), ('EFA', 'EEM')]
for t1, t2 in sample_pairs:
    if t1 in corr_matrix.columns and t2 in corr_matrix.columns:
        print(f"{t1}-{t2}: {corr_matrix.loc[t1, t2]:.3f}")

# Function to calculate equally-weighted portfolio statistics
def equal_weight_portfolio(returns, tickers_subset):
    """Calculate equal-weight portfolio return and risk"""
    n = len(tickers_subset)
    weights = np.array([1/n] * n)
    
    port_returns = returns[tickers_subset].dot(weights)
    annual_return = port_returns.mean() * 252
    annual_vol = port_returns.std() * np.sqrt(252)
    
    return annual_return, annual_vol, port_returns

# Simulate portfolios with increasing number of stocks
def diversification_curve(returns, tickers, max_assets=15, simulations=100):
    """Show risk reduction as number of assets increases"""
    results = {n: [] for n in range(1, min(max_assets+1, len(tickers)+1))}
    
    np.random.seed(42)
    for n_assets in results.keys():
        for _ in range(simulations):
            # Randomly select n assets
            selected = np.random.choice(tickers, n_assets, replace=False)
            _, vol, _ = equal_weight_portfolio(returns, selected.tolist())
            results[n_assets].append(vol)
    
    # Calculate statistics
    avg_vols = [np.mean(results[n]) for n in sorted(results.keys())]
    std_vols = [np.std(results[n]) for n in sorted(results.keys())]
    min_vols = [np.min(results[n]) for n in sorted(results.keys())]
    max_vols = [np.max(results[n]) for n in sorted(results.keys())]
    
    return sorted(results.keys()), avg_vols, std_vols, min_vols, max_vols

# Get diversification curve for equities only
equity_tickers = [t for key in ['US Large', 'US Small'] for t in tickers[key]]
n_assets, avg_vols, std_vols, min_vols, max_vols = diversification_curve(
    returns, equity_tickers, max_assets=len(equity_tickers), simulations=200
)

# Calculate cross-asset class diversification
portfolios_to_test = {
    'Single Stock (AAPL)': ['AAPL'],
    '5 US Stocks': equity_tickers[:5],
    'All Equities': equity_tickers,
    'Stocks + Bonds': equity_tickers[:5] + tickers['Bonds'],
    'Stocks + Gold': equity_tickers[:5] + ['GLD'],
    'Multi-Asset': equity_tickers[:3] + tickers['Bonds'] + ['GLD', 'VNQ'],
    'Global Diversified': equity_tickers[:3] + tickers['International'] + tickers['Bonds'] + ['GLD']
}

portfolio_stats = {}
for name, ticker_list in portfolios_to_test.items():
    # Filter out any tickers not in our data
    valid_tickers = [t for t in ticker_list if t in returns.columns]
    if valid_tickers:
        ret, vol, _ = equal_weight_portfolio(returns, valid_tickers)
        portfolio_stats[name] = {
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - 0.03) / vol,
            'n_assets': len(valid_tickers)
        }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Diversification curve
axes[0, 0].plot(n_assets, avg_vols, 'b-', linewidth=3, label='Average Volatility')
axes[0, 0].fill_between(n_assets, 
                        np.array(avg_vols) - np.array(std_vols),
                        np.array(avg_vols) + np.array(std_vols),
                        alpha=0.3, label='±1 Std Dev')
axes[0, 0].plot(n_assets, min_vols, 'g--', linewidth=2, alpha=0.7, label='Best Case')
axes[0, 0].plot(n_assets, max_vols, 'r--', linewidth=2, alpha=0.7, label='Worst Case')

# Mark diminishing returns point
axes[0, 0].axvline(20, color='orange', linestyle='--', linewidth=2, alpha=0.5,
                   label='~20 stocks (70-80% reduction)')

axes[0, 0].set_xlabel('Number of Assets')
axes[0, 0].set_ylabel('Portfolio Volatility (Annual)')
axes[0, 0].set_title('Diversification Benefit: Risk Reduction vs Number of Assets')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Risk reduction percentage
single_stock_vol = avg_vols[0]
risk_reduction = [(single_stock_vol - vol) / single_stock_vol * 100 for vol in avg_vols]

axes[0, 1].plot(n_assets, risk_reduction, 'g-', linewidth=3)
axes[0, 1].axhline(70, color='orange', linestyle='--', alpha=0.5, label='70% reduction')
axes[0, 1].axhline(90, color='red', linestyle='--', alpha=0.5, label='90% reduction (limit)')
axes[0, 1].axvline(20, color='orange', linestyle='--', alpha=0.5)

axes[0, 1].set_xlabel('Number of Assets')
axes[0, 1].set_ylabel('Risk Reduction (%)')
axes[0, 1].set_title('Percentage of Idiosyncratic Risk Eliminated')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_ylim(0, 100)

# Plot 3: Cross-asset class comparison
portfolio_names = list(portfolio_stats.keys())
vols = [portfolio_stats[p]['volatility'] for p in portfolio_names]
rets = [portfolio_stats[p]['return'] for p in portfolio_names]

scatter = axes[1, 0].scatter(vols, rets, s=300, alpha=0.6, 
                             c=range(len(portfolio_names)), cmap='viridis')

for i, name in enumerate(portfolio_names):
    axes[1, 0].annotate(name, (vols[i], rets[i]), 
                       fontsize=8, ha='center', va='bottom')

axes[1, 0].set_xlabel('Volatility (Annual)')
axes[1, 0].set_ylabel('Return (Annual)')
axes[1, 0].set_title('Cross-Asset Diversification Benefits')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Correlation heatmap (subset)
important_assets = ['AAPL', 'AGG', 'GLD', 'EFA', 'VNQ']
available_assets = [a for a in important_assets if a in corr_matrix.columns]
subset_corr = corr_matrix.loc[available_assets, available_assets]

im = axes[1, 1].imshow(subset_corr, cmap='RdYlGn_r', vmin=-0.5, vmax=1.0, aspect='auto')
axes[1, 1].set_xticks(range(len(available_assets)))
axes[1, 1].set_yticks(range(len(available_assets)))
axes[1, 1].set_xticklabels(available_assets, rotation=45, ha='right')
axes[1, 1].set_yticklabels(available_assets)

# Add correlation values
for i in range(len(available_assets)):
    for j in range(len(available_assets)):
        text = axes[1, 1].text(j, i, f'{subset_corr.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10)

axes[1, 1].set_title('Correlation Matrix (Lower = More Diversification)')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "=" * 90)
print("DIVERSIFICATION ANALYSIS SUMMARY")
print("=" * 90)

print(f"\nSingle Stock Average Volatility: {avg_vols[0]:.2%}")
print(f"20-Stock Portfolio Volatility:   {avg_vols[min(19, len(avg_vols)-1)]:.2%}")
print(f"Risk Reduction:                  {risk_reduction[min(19, len(risk_reduction)-1)]:.1f}%")

print("\n" + "=" * 90)
print("PORTFOLIO COMPARISON")
print("=" * 90)
print(f"{'Portfolio':<25} {'Assets':>7} {'Return':>10} {'Vol':>10} {'Sharpe':>10}")
print("-" * 90)

for name in portfolio_names:
    stats = portfolio_stats[name]
    print(f"{name:<25} {stats['n_assets']:>7} {stats['return']:>9.2%} "
          f"{stats['volatility']:>9.2%} {stats['sharpe']:>9.3f}")

# Calculate marginal benefit
print("\n" + "=" * 90)
print("MARGINAL DIVERSIFICATION BENEFIT")
print("=" * 90)
print(f"{'From':<10} {'To':<10} {'Vol Reduction':>15} {'% of Total Benefit':>20}")
print("-" * 90)

for i in range(min(10, len(n_assets)-1)):
    if i == 0:
        from_n, to_n = 1, 2
    else:
        from_n, to_n = n_assets[i-1], n_assets[i]
    
    vol_reduction = avg_vols[i] - avg_vols[i+1]
    pct_of_total = (vol_reduction / (avg_vols[0] - avg_vols[-1])) * 100
    
    print(f"{from_n:<10} {to_n:<10} {vol_reduction:>14.4f} {pct_of_total:>19.1f}%")

# Demonstrate correlation impact
print("\n" + "=" * 90)
print("CORRELATION IMPACT ON DIVERSIFICATION")
print("=" * 90)

# Simulate two-asset portfolios with different correlations
rho_values = [-0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
sigma_1 = sigma_2 = 0.20  # 20% volatility each
w = 0.5  # Equal weight

print(f"Two assets with σ1=σ2=20%, equal weighted:")
print(f"{'Correlation':>12} {'Portfolio σ':>15} {'Reduction':>12}")
print("-" * 90)

for rho in rho_values:
    sigma_p = np.sqrt(w**2 * sigma_1**2 + w**2 * sigma_2**2 + 
                     2*w*w*sigma_1*sigma_2*rho)
    reduction = (1 - sigma_p / sigma_1) * 100
    print(f"{rho:>12.1f} {sigma_p:>14.2%} {reduction:>11.1f}%")

print("\nKey Insight: Diversification benefit = f(1-ρ); lower correlation = greater benefit")
```

## 6. Challenge Round
When does diversification fail?
- Systemic crises: All correlations → 1 (2008, COVID crash)
- Hidden factor exposures: "Diversified" portfolio concentrated in single factor
- Liquidity crisis: Cannot exit positions when needed, all assets illiquid
- Structural regime change: Historical correlations break down
- Leverage cascades: Forced selling creates artificial correlation

Diversification misconceptions:
- "Just add more assets": Marginal benefit diminishes rapidly after 20-30 holdings
- "Diversify away all risk": Cannot eliminate systematic/market risk
- "International = diversified": Globalization increased cross-country correlations
- "Alternative assets always diversify": Real estate, commodities can correlate during inflation
- "Diversification guarantees returns": Only reduces volatility, not expected return necessarily

Practical considerations:
- Transaction costs: Each additional holding incurs trading/management costs
- Tracking error: Over-diversification approaches index, why not just hold ETF?
- Correlation instability: Varies over time, especially during stress
- Rebalancing complexity: More holdings = more frequent rebalancing needs

## 7. Key References
- [Statman, M. (1987) "How Many Stocks Make a Diversified Portfolio?"](https://www.jstor.org/stable/4479063)
- [Evans & Archer (1968) "Diversification and the Reduction of Dispersion"](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1968.tb00815.x)
- [Solnik & Roulet (2000) "Dispersion as Cross-Sectional Correlation"](https://www.jstor.org/stable/2676254)
- [Investopedia - Diversification](https://www.investopedia.com/terms/d/diversification.asp)

---
**Status:** Core MPT principle ("free lunch") | **Complements:** Correlation, Systematic Risk, Portfolio Variance
