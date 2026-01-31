import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Estimate forward-looking expected returns; construct risk-return relationship

def estimate_forward_returns(ticker, current_price, div_yield, earnings_growth, pe_ratio=None):
    """
    Estimate forward-looking expected return using dividend + growth approach.
    E[R] ≈ Dividend yield + Growth rate + Multiple expansion (assume 0)
    """
    
    total_return = div_yield + earnings_growth
    
    # Adjust for valuation (simple approach)
    historical_pe = 18  # Historical average
    valuation_adjustment = 0
    
    if pe_ratio:
        if pe_ratio > historical_pe:
            # Valuation above average; lower forward return (multiple compression risk)
            valuation_adjustment = -0.01  # -1% adjustment if elevated
        elif pe_ratio < historical_pe:
            # Valuation below average; higher forward return (multiple expansion)
            valuation_adjustment = 0.01   # +1% adjustment if depressed
    
    forward_return = total_return + valuation_adjustment
    
    return {
        'dividend_yield': div_yield,
        'growth_rate': earnings_growth,
        'valuation_adj': valuation_adjustment,
        'forward_return': forward_return
    }


def historical_expected_returns(tickers, start_date, end_date):
    """Estimate returns using historical averages."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns.mean() * 252  # Annualized


def historical_volatility(tickers, start_date, end_date):
    """Estimate volatility from historical data."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns.std() * np.sqrt(252)  # Annualized


def build_risk_return_frontier(expected_returns, cov_matrix, rf=0.025, num_points=50):
    """
    Construct efficient frontier by varying target return.
    Returns list of (volatility, return, weights) tuples.
    """
    
    n = len(expected_returns)
    frontier = []
    
    # Min and max returns
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    for target in target_returns:
        # Minimize variance subject to return and weight constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target}  # Target return
        ]
        
        bounds = tuple((0, 1) for _ in range(n))  # Long-only constraints
        
        # Initial guess: equal weights
        w0 = np.array([1/n] * n)
        
        # Objective: minimize variance
        def objective(w):
            return np.dot(w, np.dot(cov_matrix, w))
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            vol = np.sqrt(result.fun)
            frontier.append({
                'return': target,
                'volatility': vol,
                'weights': result.x
            })
    
    return pd.DataFrame(frontier)


def compute_expected_return_forward(ticker):
    """
    Forward-looking expected return estimation.
    Uses dividend yield + growth assumptions.
    """
    
    tick = yf.Ticker(ticker)
    info = tick.info
    
    # Get data
    div_yield = info.get('dividendYield', 0) or 0
    pe_ratio = info.get('trailingPE', None)
    peg_ratio = info.get('pegRatio', None)
    
    # Growth assumptions (conservative)
    if peg_ratio and peg_ratio > 0:
        implied_growth = info.get('epsTrailingTwelveMonths', 0) / pe_ratio if pe_ratio else 0.06
        earnings_growth = min(0.15, max(0.03, implied_growth))  # Bound between 3-15%
    else:
        earnings_growth = 0.06  # Default 6% growth
    
    return estimate_forward_returns(ticker, None, div_yield, earnings_growth, pe_ratio)


# Main Analysis
print("=" * 100)
print("EXPECTED RETURN, VARIANCE & RISK-RETURN TRADEOFF")
print("=" * 100)

# 1. Data Collection
print("\n1. PORTFOLIO ASSET SELECTION & RETURN ESTIMATION")
print("-" * 100)

tickers = ['SPY', 'QQQ', 'BND', 'AGG', 'GLD']
names = ['S&P 500', 'Tech (Nasdaq)', 'Bond ETF', 'Aggregate Bonds', 'Gold']

# Historical data
hist_returns = historical_expected_returns(tickers, '2015-01-01', '2024-01-01')
hist_vols = historical_volatility(tickers, '2015-01-01', '2024-01-01')

# Forward-looking estimates (adjusted)
# Based on current yields, growth expectations
forward_returns = pd.Series({
    'SPY': 0.075,      # 7.5% (lower than hist due to valuation)
    'QQQ': 0.085,      # 8.5% (tech premium)
    'BND': 0.045,      # 4.5% (bond yield approximately)
    'AGG': 0.042,      # 4.2% (agg bonds)
    'GLD': 0.030       # 3% (gold; minimal return, hedge value)
})

print("\nAsset Return Estimates:\n")
print(f"{'Asset':<20} {'Historical %':<18} {'Forward-Looking %':<20} {'Volatility %':<15}")
print("-" * 73)

for ticker, name in zip(tickers, names):
    print(f"{name:<20} {hist_returns[ticker]*100:<18.2f} {forward_returns[ticker]*100:<20.2f} "
          f"{hist_vols[ticker]*100:<15.2f}")

# 2. Covariance matrix and correlation
print("\n2. RISK STRUCTURE (COVARIANCE & CORRELATION)")
print("-" * 100)

# Calculate covariance matrix from historical data
data = yf.download(tickers, start='2015-01-01', end='2024-01-01', progress=False)['Adj Close']
returns = data.pct_change().dropna()

cov_matrix = returns.cov() * 252
corr_matrix = returns.corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

print(f"\nCovariance Matrix (Annualized):")
print((cov_matrix * 10000).round(0))  # Show in basis points

# 3. Individual asset risk-return
print("\n3. INDIVIDUAL ASSET RISK-RETURN METRICS")
print("-" * 100)

rf = 0.025

print(f"\n{'Asset':<20} {'Return %':<15} {'Volatility %':<18} {'Sharpe Ratio':<15}")
print("-" * 68)

for ticker, name in zip(tickers, names):
    ret = forward_returns[ticker]
    vol = hist_vols[ticker]
    sharpe = (ret - rf) / vol
    print(f"{name:<20} {ret*100:<15.2f} {vol*100:<18.2f} {sharpe:<15.3f}")

# 4. Build efficient frontier
print("\n4. EFFICIENT FRONTIER CONSTRUCTION")
print("-" * 100)

frontier = build_risk_return_frontier(forward_returns, cov_matrix, rf=rf, num_points=30)

print(f"\nGenerated {len(frontier)} efficient portfolios")
print(f"  Min volatility: {frontier['volatility'].min()*100:.2f}%")
print(f"  Max volatility: {frontier['volatility'].max()*100:.2f}%")
print(f"  Return range: {frontier['return'].min()*100:.2f}% - {frontier['return'].max()*100:.2f}%")

# Find maximum Sharpe portfolio
frontier['sharpe'] = (frontier['return'] - rf) / frontier['volatility']
max_sharpe_idx = frontier['sharpe'].idxmax()
max_sharpe_port = frontier.loc[max_sharpe_idx]

print(f"\nMaximum Sharpe Ratio Portfolio:")
print(f"  Return: {max_sharpe_port['return']*100:.2f}%")
print(f"  Volatility: {max_sharpe_port['volatility']*100:.2f}%")
print(f"  Sharpe Ratio: {max_sharpe_port['sharpe']:.3f}")

# 5. Risk-Return Tradeoff Analysis
print("\n5. RISK-RETURN TRADEOFF QUANTIFICATION")
print("-" * 100)

# Calculate incremental risk-return
frontier_sorted = frontier.sort_values('volatility')
frontier_sorted['return_diff'] = frontier_sorted['return'].diff()
frontier_sorted['vol_diff'] = frontier_sorted['volatility'].diff()
frontier_sorted['marginal_return_per_risk'] = frontier_sorted['return_diff'] / frontier_sorted['vol_diff']

print(f"\nMarginal Risk-Return Trade-off (Slope of frontier):")
print(f"{'Volatility %':<20} {'Return %':<15} {'MR/Risk':<15} {'Interpretation':<30}")
print("-" * 80)

sample_indices = [len(frontier_sorted)//4, len(frontier_sorted)//2, 3*len(frontier_sorted)//4]

for idx in sample_indices:
    if idx < len(frontier_sorted):
        row = frontier_sorted.iloc[idx]
        interp = "Steep" if row['marginal_return_per_risk'] > 2 else ("Moderate" if row['marginal_return_per_risk'] > 0.5 else "Flat")
        print(f"{row['volatility']*100:<20.2f} {row['return']*100:<15.2f} "
              f"{row['marginal_return_per_risk']:<15.3f} {interp:<30}")

# 6. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Efficient Frontier with Assets
ax = axes[0, 0]

# Frontier
ax.plot(frontier['volatility'] * 100, frontier['return'] * 100, 'o-', linewidth=2.5, 
       color='#2ecc71', markersize=4, label='Efficient Frontier', zorder=3)

# Individual assets
for ticker, name in zip(tickers, names):
    ret = forward_returns[ticker]
    vol = hist_vols[ticker]
    ax.scatter(vol * 100, ret * 100, s=200, alpha=0.7, label=name, zorder=4, edgecolors='black', linewidth=1)

# CAL (Capital Allocation Line)
cal_vols = np.linspace(0, frontier['volatility'].max() * 1.3 * 100, 100)
cal_returns = rf * 100 + max_sharpe_port['sharpe'] * cal_vols
ax.plot(cal_vols, cal_returns, 'k--', linewidth=1.5, alpha=0.6, label='CAL', zorder=2)

# Max Sharpe portfolio
ax.scatter(max_sharpe_port['volatility'] * 100, max_sharpe_port['return'] * 100, 
          s=300, marker='*', color='red', label='Max Sharpe', zorder=5, edgecolors='black', linewidth=1)

# Risk-free
ax.scatter(0, rf * 100, s=200, marker='s', color='gray', label='Risk-free', zorder=5, edgecolors='black', linewidth=1)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Efficient Frontier: Risk-Return Tradeoff', fontweight='bold', fontsize=13)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)
ax.set_ylim([0, 10])

# Plot 2: Correlation Heatmap
ax = axes[0, 1]

im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(tickers)))
ax.set_yticks(range(len(tickers)))
ax.set_xticklabels([t for t in tickers], rotation=45, ha='right')
ax.set_yticklabels([t for t in tickers])

# Add correlation values
for i in range(len(tickers)):
    for j in range(len(tickers)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', color='black', fontsize=9)

ax.set_title('Correlation Matrix (Diversification Drivers)', fontweight='bold', fontsize=13)
plt.colorbar(im, ax=ax, label='Correlation')

# Plot 3: Frontier Slope (Marginal Risk-Return)
ax = axes[1, 0]

frontier_sorted_plot = frontier_sorted.dropna()
ax.plot(frontier_sorted_plot['volatility'] * 100, 
       frontier_sorted_plot['marginal_return_per_risk'], 
       'o-', linewidth=2, markersize=6, color='#3498db')

ax.set_xlabel('Portfolio Volatility (%)', fontsize=12)
ax.set_ylabel('Marginal Return per % Risk', fontsize=12)
ax.set_title('Risk-Return Tradeoff Slope (Frontier Curvature)', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 4: Sharpe Ratio along Frontier
ax = axes[1, 1]

sharpes = (frontier['return'] - rf) / frontier['volatility']
ax.plot(frontier['volatility'] * 100, sharpes, 'o-', linewidth=2.5, markersize=6, color='#e74c3c')
ax.axhline(y=max_sharpe_port['sharpe'], color='green', linestyle='--', linewidth=2, alpha=0.7, 
          label=f'Max Sharpe: {max_sharpe_port["sharpe"]:.3f}')

ax.set_xlabel('Portfolio Volatility (%)', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Risk-Adjusted Return along Efficient Frontier', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('expected_return_variance_tradeoff.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: expected_return_variance_tradeoff.png")
plt.show()

# 7. Summary
print("\n" + "=" * 100)
print("KEY INSIGHTS:")
print("=" * 100)
print(f"""
EXPECTED RETURN ESTIMATION:
├─ Historical average unreliable (past ≠ future)
├─ Forward-looking approach: Dividend yield + Growth + Multiple expansion
├─ Current valuation matters: High P/E → Lower forward returns
└─ Consensus: S&P 500 ~7-8% forward (vs 10% historical average)

VARIANCE & RISK MEASUREMENT:
├─ Volatility ~15-18% for stocks (annualized)
├─ Bonds ~3-5% volatility (lower risk)
├─ Gold ~15% (hedge; low correlation to stocks)
└─ Time-varying: Volatility clusters; crisis periods 2-3× normal

RISK-RETURN TRADEOFF (FUNDAMENTAL):
├─ Higher expected return REQUIRES higher volatility (not free lunch)
├─ Efficient frontier shows: Optimal allocation by risk aversion
├─ Sharpe ratio: Return per unit risk; higher is better
├─ Two-fund separation: All investors hold market + risk-free
└─ Allocation depends on λ (risk tolerance), not on individual beliefs

DIVERSIFICATION BENEFIT:
├─ Portfolio volatility < weighted average of individual volatilities
├─ Correlation structure critical: Low ρ → More benefit
├─ Optimal allocation: Not equal-weight; considers risk & return
├─ Stocks-bonds: Correlation 0.3 (low); good diversifier
└─ Gold: Correlation 0 (uncorrelated); small allocation helps

FRONTIER PROPERTIES:
├─ Curved shape: Diminishing marginal return per unit risk
├─ Slope decreases moving right (more conservative portfolios have better Sharpe)
├─ Maximum Sharpe portfolio: Optimal risky asset for all rational investors
├─ CAL (Capital Allocation Line): Linear; connects rf + tangency portfolio
└─ All points above frontier: Impossible (violates optimization)

PRACTICAL IMPLICATIONS:
├─ Use forward-looking return estimates (not historical averages)
├─ Correlations change in stress; plan accordingly
├─ Rebalance toward lower vol in high-return environments (take profits)
├─ Tax-aware: After-tax expected returns differ materially
└─ Accept trade-off: Can't have high return with low risk
""")

print("=" * 100)