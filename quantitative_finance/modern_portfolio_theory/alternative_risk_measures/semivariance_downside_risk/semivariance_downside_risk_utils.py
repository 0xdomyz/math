from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def portfolio_downside_deviation(weights, returns, target=0):
    """
    Calculate portfolio downside deviation
    """
    portfolio_returns = returns @ weights
    dd = downside_deviation(portfolio_returns, target=target)
    return dd

def optimize_mean_semivariance(returns, target_return=None, target_threshold=0):
    """
    Optimize portfolio to minimize downside deviation
    """
    n_assets = returns.shape[1]
    
    # Initial guess
    w0 = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
    ]
    
    if target_return is not None:
        mean_returns = returns.mean()
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ mean_returns - target_return
        })
    
    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        lambda w: portfolio_downside_deviation(w, returns, target_threshold),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

def optimize_mean_variance(returns, target_return=None):
    """
    Traditional mean-variance optimization for comparison
    """
    n_assets = returns.shape[1]
    cov_matrix = returns.cov()
    
    w0 = np.ones(n_assets) / n_assets
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    if target_return is not None:
        mean_returns = returns.mean()
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ mean_returns - target_return
        })
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(
        lambda w: np.sqrt(w @ cov_matrix @ w),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

# Build efficient frontiers
print("\nBuilding efficient frontiers...")
mean_returns = returns.mean() * 252

min_return = mean_returns.min()
max_return = mean_returns.max() * 0.85
target_returns = np.linspace(min_return, max_return, 12)

mv_portfolios = []
msv_portfolios = []

for target_ret in target_returns:
    # Mean-Variance
    mv_weights = optimize_mean_variance(returns, target_ret/252)
    mv_ret = mv_weights @ returns.mean() * 252
    mv_vol = np.sqrt(mv_weights @ returns.cov() @ mv_weights) * np.sqrt(252)
    mv_dd = portfolio_downside_deviation(mv_weights, returns, 0) * np.sqrt(252)
    mv_sortino = (mv_ret - 0.02) / mv_dd if mv_dd > 0 else np.nan
    
    mv_portfolios.append({
        'return': mv_ret,
        'volatility': mv_vol,
        'downside_dev': mv_dd,
        'sortino': mv_sortino,
        'weights': mv_weights
    })
    
    # Mean-Semivariance
    msv_weights = optimize_mean_semivariance(returns, target_ret/252, 0)
    msv_ret = msv_weights @ returns.mean() * 252
    msv_vol = np.sqrt(msv_weights @ returns.cov() @ msv_weights) * np.sqrt(252)
    msv_dd = portfolio_downside_deviation(msv_weights, returns, 0) * np.sqrt(252)
    msv_sortino = (msv_ret - 0.02) / msv_dd if msv_dd > 0 else np.nan
    
    msv_portfolios.append({
        'return': msv_ret,
        'volatility': msv_vol,
        'downside_dev': msv_dd,
        'sortino': msv_sortino,
        'weights': msv_weights
    })

mv_df = pd.DataFrame(mv_portfolios)
msv_df = pd.DataFrame(msv_portfolios)

# Compare specific portfolios
print("\n" + "="*110)
print("PORTFOLIO COMPARISON: Mean-Variance vs Mean-Semivariance (Target Return: 10%)")
print("="*110)

target_idx = np.argmin(np.abs(mv_df['return'] - 0.10))
mv_port = mv_df.iloc[target_idx]
msv_port = msv_df.iloc[target_idx]

comparison = pd.DataFrame({
    'Mean-Variance': [
        mv_port['return'],
        mv_port['volatility'],
        mv_port['downside_dev'],
        mv_port['sortino'],
        (mv_port['return'] - 0.02) / mv_port['volatility']
    ],
    'Mean-Semivariance': [
        msv_port['return'],
        msv_port['volatility'],
        msv_port['downside_dev'],
        msv_port['sortino'],
        (msv_port['return'] - 0.02) / msv_port['volatility']
    ]
}, index=['Return', 'Volatility', 'Downside Dev', 'Sortino', 'Sharpe'])

print(comparison.round(4))

print("\n" + "="*110)
print("WEIGHT ALLOCATION COMPARISON")
print("="*110)
weight_comp = pd.DataFrame({
    'Mean-Variance': mv_port['weights'],
    'Mean-Semivariance': msv_port['weights']
}, index=tickers)
print(weight_comp.round(4))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Efficient Frontier (Return vs Volatility)
asset_returns = returns.mean() * 252
asset_vols = returns.std() * np.sqrt(252)

axes[0, 0].scatter(asset_vols, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 0].annotate(ticker, (asset_vols[i], asset_returns[i]), fontsize=8, ha='right')

axes[0, 0].plot(mv_df['volatility'], mv_df['return'], 'b-o',
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 0].plot(msv_df['volatility'], msv_df['return'], 'r-s',
               label='Mean-Semivariance', alpha=0.7, markersize=4)

axes[0, 0].set_xlabel('Volatility (Annual)')
axes[0, 0].set_ylabel('Expected Return (Annual)')
axes[0, 0].set_title('Efficient Frontiers: Mean-Variance vs Mean-Semivariance')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Return vs Downside Deviation
asset_dds = [downside_deviation(returns[t], 0) * np.sqrt(252) for t in tickers]

axes[0, 1].scatter(asset_dds, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 1].annotate(ticker, (asset_dds[i], asset_returns[i]), fontsize=8, ha='right')

axes[0, 1].plot(mv_df['downside_dev'], mv_df['return'], 'b-o',
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 1].plot(msv_df['downside_dev'], msv_df['return'], 'r-s',
               label='Mean-Semivariance', alpha=0.7, markersize=4)

axes[0, 1].set_xlabel('Downside Deviation (Annual)')
axes[0, 1].set_ylabel('Expected Return (Annual)')
axes[0, 1].set_title('Mean-Downside Risk Frontier')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Sharpe vs Sortino Comparison
sharpes = metrics_df['Sharpe Ratio']
sortinos = metrics_df['Sortino Ratio']

axes[1, 0].scatter(sharpes, sortinos, s=100, alpha=0.7)
for ticker in tickers:
    axes[1, 0].annotate(ticker, 
                       (metrics_df.loc[ticker, 'Sharpe Ratio'],
                        metrics_df.loc[ticker, 'Sortino Ratio']),
                       fontsize=8, ha='right')

# Add diagonal line
max_val = max(sharpes.max(), sortinos.max())
axes[1, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal Line')

axes[1, 0].set_xlabel('Sharpe Ratio')
axes[1, 0].set_ylabel('Sortino Ratio')
axes[1, 0].set_title('Sharpe vs Sortino: Capturing Asymmetric Risk')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Upside vs Downside Capture
upside_caps = metrics_df['Upside Capture'].values
downside_caps = metrics_df['Downside Capture'].values

axes[1, 1].scatter(downside_caps, upside_caps, s=100, alpha=0.7)
for ticker in tickers:
    if ticker != 'SPY':  # SPY is (100, 100) - the benchmark
        axes[1, 1].annotate(ticker,
                           (metrics_df.loc[ticker, 'Downside Capture'],
                            metrics_df.loc[ticker, 'Upside Capture']),
                           fontsize=8, ha='right')

# Add reference lines
axes[1, 1].axhline(100, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].axvline(100, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].plot([0, 150], [0, 150], 'k--', alpha=0.3, label='Equal Capture')

# Ideal quadrant (upper left)
axes[1, 1].fill_between([0, 100], 100, 150, alpha=0.1, color='green', label='Ideal')

axes[1, 1].set_xlabel('Downside Capture (%)')
axes[1, 1].set_ylabel('Upside Capture (%)')
axes[1, 1].set_title('Upside vs Downside Capture Ratios (vs SPY)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim([50, 130])
axes[1, 1].set_ylim([50, 130])

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "="*110)
print("KEY INSIGHTS: SEMI-VARIANCE AND DOWNSIDE RISK")
print("="*110)
print("1. Semi-variance focuses on investor-relevant risk (losses, not gains)")
print("2. Sortino Ratio > Sharpe when returns positively skewed (upside volatility)")
print("3. Mean-semivariance optimization produces different allocations than mean-variance")
print("4. Downside capture < 100% ideal (participate less in market declines)")
print("5. Upside capture > 100% ideal (participate more in market advances)")
print("6. Downside beta measures crash risk specifically (systematic downside)")
print("7. Lower Partial Moments generalize downside risk (different orders)")
print("8. Semi-variance especially useful for asymmetric assets (hedge funds, options)")

# Asymmetry analysis
print("\n" + "="*110)
print("ASYMMETRY ANALYSIS: Variance Decomposition")
print("="*110)
print(f"{'Ticker':>8} {'Variance':>10} {'Downside':>11} {'Upside':>11} {'Skewness':>11} {'Down%':>8}")
print("-"*110)

for ticker in tickers:
    ret = returns[ticker]
    total_var = ret.var() * 252
    down_var = semivariance(ret, ret.mean(), below=True) * 252
    up_var = semivariance(ret, ret.mean(), below=False) * 252
    skew = ret.skew()
    down_pct = down_var / total_var * 100 if total_var > 0 else 0
    
    print(f"{ticker:>8} {total_var:>10.4f} {down_var:>11.4f} {up_var:>11.4f} {skew:>11.3f} {down_pct:>7.1f}%")

print("\nFor symmetric returns: Downside% ≈ 50%")
print("Downside% > 50% indicates negative skewness (fat left tail)")
print("Downside% < 50% indicates positive skewness (fat right tail)")