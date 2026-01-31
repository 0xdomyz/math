import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime

# Efficient frontier construction and visualization

def fetch_market_data(tickers, start_date, end_date):
    """Download return data for multiple assets."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    return returns


def compute_portfolio_stats(weights, returns_data):
    """Calculate return, volatility, Sharpe ratio for portfolio."""
    portfolio_return = np.sum(weights * returns_data.mean()) * 252
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns_data.cov() * 252, weights)))
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe


def negative_sharpe(weights, returns_data, rf=0.025):
    """Objective: Maximize Sharpe (minimize negative Sharpe) for optimization."""
    returns = returns_data.mean() * 252
    cov_matrix = returns_data.cov() * 252
    portfolio_return = np.sum(weights * returns)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    sharpe = (portfolio_return - rf) / portfolio_vol if portfolio_vol > 0 else 0
    return -sharpe


def minimum_variance_portfolio(returns_data):
    """Find minimum variance portfolio (GMV)."""
    n = returns_data.shape[1]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    def variance(w):
        return np.dot(w, np.dot(returns_data.cov() * 252, w))
    
    result = minimize(variance, np.array([1/n]*n), method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x


def efficient_frontier(returns_data, num_points=50):
    """Generate efficient frontier by varying target return."""
    min_ret = returns_data.mean().min() * 252
    max_ret = returns_data.mean().max() * 252
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    frontier = []
    
    for target in target_returns:
        n = returns_data.shape[1]
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(w * returns_data.mean() * 252) - target}
        ]
        bounds = tuple((0, 1) for _ in range(n))
        
        def variance(w):
            return np.dot(w, np.dot(returns_data.cov() * 252, w))
        
        result = minimize(variance, np.array([1/n]*n), method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            vol = np.sqrt(result.fun)
            frontier.append({'return': target, 'volatility': vol})
    
    return pd.DataFrame(frontier)


def optimal_portfolio_by_lambda(returns_data, lambda_val, rf=0.025):
    """Find optimal portfolio for given risk aversion λ."""
    n = returns_data.shape[1]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    def objective(w):
        ret = np.sum(w * returns_data.mean() * 252)
        var = np.dot(w, np.dot(returns_data.cov() * 252, w))
        return -(ret - (lambda_val / 2) * var)
    
    result = minimize(objective, np.array([1/n]*n), method='SLSQP',
                     bounds=bounds, constraints=constraints)
    return result.x


# Main Analysis
print("=" * 100)
print("MODERN PORTFOLIO THEORY (MARKOWITZ) & MEAN-VARIANCE FRAMEWORK")
print("=" * 100)

# 1. Data Collection
print("\n1. MARKET DATA & ASSET SELECTION")
print("-" * 100)

# Portfolio of stocks and sector ETFs
tickers = ['SPY', 'QQQ', 'IWM', 'AGG', 'GLD']  # Large-cap, Tech, Small-cap, Bonds, Gold
names = ['S&P 500', 'Tech (Nasdaq)', 'Small Cap', 'Aggregate Bonds', 'Gold']

returns = fetch_market_data(tickers, '2015-01-01', '2024-01-01')

print(f"\nAssets in portfolio ({len(tickers)} total):")
for i, (tick, name) in enumerate(zip(tickers, names)):
    annual_return = returns[tick].mean() * 252
    annual_vol = returns[tick].std() * np.sqrt(252)
    print(f"  {i+1}. {name:20s} ({tick}): {annual_return*100:6.2f}% return, {annual_vol*100:6.2f}% volatility")

# Correlation matrix
print(f"\nCorrelation Matrix:")
corr = returns.corr()
print(corr.round(3))

# 2. Basic Portfolio Comparisons
print("\n2. PORTFOLIO COMPOSITION COMPARISON")
print("-" * 100)

equal_weight = np.array([1/len(tickers)] * len(tickers))
gmv_weights = minimum_variance_portfolio(returns)

print(f"\nEqual-Weight Portfolio (1/{len(tickers)} each):")
eq_ret, eq_vol, eq_sharpe = compute_portfolio_stats(equal_weight, returns)
print(f"  Return: {eq_ret*100:.2f}%, Volatility: {eq_vol*100:.2f}%, Sharpe: {eq_sharpe:.3f}")

print(f"\nGlobal Minimum Variance (GMV) Portfolio:")
gmv_ret, gmv_vol, gmv_sharpe = compute_portfolio_stats(gmv_weights, returns)
print(f"  Return: {gmv_ret*100:.2f}%, Volatility: {gmv_vol*100:.2f}%, Sharpe: {gmv_sharpe:.3f}")
print(f"  Weights: {', '.join([f'{n}: {w*100:.1f}%' for n, w in zip(names, gmv_weights)])}")

# Diversification benefit
avg_vol = np.mean([returns[t].std() * np.sqrt(252) for t in tickers])
print(f"\nDiversification Benefit:")
print(f"  Average individual volatility: {avg_vol*100:.2f}%")
print(f"  GMV portfolio volatility: {gmv_vol*100:.2f}%")
print(f"  Reduction: {(1 - gmv_vol/avg_vol)*100:.1f}%")

# 3. Efficient Frontier
print("\n3. EFFICIENT FRONTIER")
print("-" * 100)

frontier = efficient_frontier(returns, num_points=40)
print(f"\nFrontier generated ({len(frontier)} portfolios)")
print(f"  Min volatility: {frontier['volatility'].min()*100:.2f}%")
print(f"  Max volatility: {frontier['volatility'].max()*100:.2f}%")
print(f"  Return range: {frontier['return'].min()*100:.2f}% - {frontier['return'].max()*100:.2f}%")

# 4. Optimal Portfolio for Different Risk Aversion Levels
print("\n4. OPTIMAL PORTFOLIOS BY RISK AVERSION")
print("-" * 100)

lambdas = [0.5, 1.0, 2.0, 4.0, 8.0]
optimal_portfolios = {}

print(f"\n{'Lambda':<8} {'Return %':<12} {'Volatility %':<15} {'Sharpe':<10} {'Diversification':<20}")
print("-" * 65)

for lam in lambdas:
    weights = optimal_portfolio_by_lambda(returns, lam)
    optimal_portfolios[lam] = weights
    ret, vol, sharpe = compute_portfolio_stats(weights, returns)
    
    # Diversification measure (Herfindahl index)
    hhi = np.sum(weights ** 2)
    
    print(f"{lam:<8.1f} {ret*100:<12.2f} {vol*100:<15.2f} {sharpe:<10.3f} {hhi:<20.3f}")

# 5. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Efficient Frontier with CAL
ax = axes[0, 0]

ax.plot(frontier['volatility'] * 100, frontier['return'] * 100, 
        label='Efficient Frontier', linewidth=2.5, color='#2ecc71')

# Add individual assets
for tick, name in zip(tickers, names):
    asset_ret = returns[tick].mean() * 252
    asset_vol = returns[tick].std() * np.sqrt(252)
    ax.scatter(asset_vol * 100, asset_ret * 100, s=150, alpha=0.7, label=name)

# Add GMV, Equal-weight, and optimal points
ax.scatter(gmv_vol * 100, gmv_ret * 100, s=200, marker='*', 
          color='red', label='GMV', zorder=5, edgecolors='black', linewidth=1)
ax.scatter(eq_vol * 100, eq_ret * 100, s=200, marker='s', 
          color='purple', label='Equal-Weight', zorder=5, edgecolors='black', linewidth=1)

# CAL from risk-free rate
rf = 0.025
max_sharpe_idx = (frontier['return'] - rf) / frontier['volatility']
max_sharpe_idx = max_sharpe_idx.idxmax()
max_sharpe_vol = frontier.loc[max_sharpe_idx, 'volatility']
max_sharpe_ret = frontier.loc[max_sharpe_idx, 'return']

cal_vols = np.linspace(0, frontier['volatility'].max() * 1.2, 100)
cal_returns = rf + (max_sharpe_ret - rf) / max_sharpe_vol * cal_vols

ax.plot(cal_vols * 100, cal_returns * 100, 'k--', linewidth=1.5, alpha=0.6, label='CAL')
ax.scatter(max_sharpe_vol * 100, max_sharpe_ret * 100, s=200, marker='^',
          color='orange', label='Tangency (Max Sharpe)', zorder=5, edgecolors='black', linewidth=1)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Efficient Frontier & Capital Allocation Line', fontweight='bold', fontsize=13)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)
ax.axhline(y=rf*100, color='gray', linestyle=':', alpha=0.5)

# Plot 2: Optimal Asset Weights by Lambda
ax = axes[0, 1]

lambdas_plot = sorted(optimal_portfolios.keys())
x = np.arange(len(lambdas_plot))
width = 0.15

for i, (name, tick) in enumerate(zip(names, tickers)):
    weights_by_lambda = [optimal_portfolios[lam][i] for lam in lambdas_plot]
    ax.bar(x + i * width, np.array(weights_by_lambda) * 100, width, label=name, alpha=0.8)

ax.set_xlabel('Risk Aversion (λ)', fontsize=12)
ax.set_ylabel('Portfolio Weight (%)', fontsize=12)
ax.set_title('Optimal Asset Allocation by Risk Aversion', fontweight='bold', fontsize=13)
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f'λ={l}' for l in lambdas_plot])
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Risk-Return Tradeoff by Lambda
ax = axes[1, 0]

lambdas_plot = sorted(optimal_portfolios.keys())
rets_by_lambda = []
vols_by_lambda = []
sharpes_by_lambda = []

for lam in lambdas_plot:
    weights = optimal_portfolios[lam]
    ret, vol, sharpe = compute_portfolio_stats(weights, returns)
    rets_by_lambda.append(ret)
    vols_by_lambda.append(vol)
    sharpes_by_lambda.append(sharpe)

ax.plot(vols_by_lambda, np.array(rets_by_lambda) * 100, 'o-', linewidth=2.5, 
       markersize=10, color='#3498db', label='Optimal Portfolios')

for lam, vol, ret in zip(lambdas_plot, vols_by_lambda, rets_by_lambda):
    ax.annotate(f'λ={lam}', (vol, ret*100), textcoords="offset points", 
               xytext=(0,10), ha='center', fontsize=9)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Risk-Return Tradeoff: Optimal Portfolios', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

# Plot 4: Sharpe Ratio by Lambda
ax = axes[1, 1]

ax.plot(lambdas_plot, sharpes_by_lambda, 'o-', linewidth=2.5, markersize=10,
       color='#e74c3c', label='Sharpe Ratio')
ax.axhline(y=max(sharpes_by_lambda), color='g', linestyle='--', alpha=0.5, label='Max Sharpe')

ax.set_xlabel('Risk Aversion (λ)', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Risk-Adjusted Returns by Risk Aversion', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mpt_efficient_frontier.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: mpt_efficient_frontier.png")
plt.show()

# 6. Summary Statistics
print("\n5. KEY INSIGHTS & INTERPRETATION")
print("-" * 100)
print(f"""
MEAN-VARIANCE FRAMEWORK (MARKOWITZ 1952):
├─ Central insight: Diversification reduces portfolio risk below individual asset risk
├─ Mechanism: Correlation < 1 creates covariance benefit (off-diagonal terms negative)
├─ Optimization: Find portfolio maximizing return for risk level (or vice versa)
├─ Key metric: Sharpe ratio = (return - rf) / volatility captures risk-adjusted performance
└─ Two-fund principle: All investors hold same risky portfolio + risk-free in varying proportions

EFFICIENT FRONTIER FINDINGS:
├─ Number of efficient portfolios: Infinite (continuous frontier)
├─ GMV portfolio: Minimum risk; may have low return
├─ Tangency portfolio: Maximum Sharpe ratio; optimal risky asset for all investors
├─ Capital Allocation Line (CAL): Linear combinations of rf + tangency portfolio
└─ Risk-return tradeoff visible: Higher return requires accepting higher volatility

RISK AVERSION IMPACT:
├─ Higher λ (more risk-averse): Portfolio closer to GMV (lower vol, lower return)
├─ Lower λ (risk-seeking): Portfolio closer to highest-return portfolio (higher vol)
├─ Optimal portfolio depends ONLY on λ (not on individual preferences, just risk tolerance)
├─ Same risky asset held by all; only allocation to rf/risky changes
└─ Validates two-fund separation and justifies index investing

DIVERSIFICATION BENEFIT (From GMV Analysis):
├─ Individual asset average volatility: {avg_vol*100:.2f}%
├─ GMV portfolio volatility: {gmv_vol*100:.2f}%
├─ Reduction from diversification: {(1 - gmv_vol/avg_vol)*100:.1f}%
├─ Correlation structure key: Negative correlations provide hedging (best benefit)
└─ Real-world: 20-30 diversified stocks capture most benefit; diminishing returns beyond

PRACTICAL IMPLICATIONS:
├─ Use low-cost index funds (captures diversification, low fees)
├─ Rebalance periodically (quarterly or annually) to maintain target allocation
├─ Choose allocation based on personal risk tolerance (λ)
├─ Tax-loss harvest in taxable accounts to improve after-tax returns
└─ Monitor correlations; they change in crises (diversification can fail)
""")

print("=" * 100)