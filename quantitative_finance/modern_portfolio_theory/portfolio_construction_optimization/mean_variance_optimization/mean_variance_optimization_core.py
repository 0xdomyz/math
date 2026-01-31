import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# Download asset data
tickers = ['SPY', 'TLT', 'GLD', 'EFA', 'VNQ']  # Stocks, Bonds, Gold, Intl, Real Estate
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
returns = data.pct_change().dropna()

# Estimate parameters
mean_returns = returns.mean() * 252  # Annualize
cov_matrix = returns.cov() * 252

print("Expected Annual Returns:")
print(mean_returns.round(4))
print("\nAnnual Covariance Matrix:")
print(cov_matrix.round(6))
print("\nCorrelation Matrix:")
print(returns.corr().round(3))

# Portfolio statistics functions
def portfolio_stats(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    """Negative Sharpe ratio for minimization"""
    p_ret, p_std = portfolio_stats(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

# Optimization constraints and bounds
def constraint_sum(weights):
    """Weights sum to 1"""
    return np.sum(weights) - 1

constraints = ({'type': 'eq', 'fun': constraint_sum})
bounds = tuple((0, 1) for _ in range(len(tickers)))  # Long only

# 1. Minimum Variance Portfolio
def min_variance(mean_returns, cov_matrix):
    """Find minimum variance portfolio"""
    num_assets = len(mean_returns)
    init_weights = np.array([1/num_assets] * num_assets)
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    result = minimize(portfolio_variance, init_weights,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 2. Maximum Sharpe Ratio Portfolio
def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.03):
    """Find portfolio with maximum Sharpe ratio"""
    num_assets = len(mean_returns)
    init_weights = np.array([1/num_assets] * num_assets)
    
    result = minimize(negative_sharpe, init_weights,
                     args=(mean_returns, cov_matrix, risk_free_rate),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 3. Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=50):
    """Generate efficient frontier portfolios"""
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    frontier_volatilities = []
    frontier_returns = []
    frontier_weights = []
    
    for target_return in target_returns:
        # Constraints: sum to 1 and achieve target return
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_stats(w, mean_returns, cov_matrix)[0] - target_return}
        )
        
        num_assets = len(mean_returns)
        init_weights = np.array([1/num_assets] * num_assets)
        
        def portfolio_variance(weights):
            return portfolio_stats(weights, mean_returns, cov_matrix)[1]
        
        result = minimize(portfolio_variance, init_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 500})
        
        if result.success:
            ret, vol = portfolio_stats(result.x, mean_returns, cov_matrix)
            frontier_returns.append(ret)
            frontier_volatilities.append(vol)
            frontier_weights.append(result.x)
    
    return np.array(frontier_returns), np.array(frontier_volatilities), frontier_weights

# Calculate optimal portfolios
mvp_weights = min_variance(mean_returns, cov_matrix)
max_sharpe_weights = max_sharpe_portfolio(mean_returns, cov_matrix)

mvp_return, mvp_vol = portfolio_stats(mvp_weights, mean_returns, cov_matrix)
ms_return, ms_vol = portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix)

print("\n" + "=" * 80)
print("MINIMUM VARIANCE PORTFOLIO")
print("=" * 80)
for ticker, weight in zip(tickers, mvp_weights):
    print(f"{ticker:5s}: {weight:>7.2%}")
print(f"\nExpected Return: {mvp_return:.2%}")
print(f"Volatility:      {mvp_vol:.2%}")
print(f"Sharpe Ratio:    {(mvp_return - 0.03) / mvp_vol:.3f}")

print("\n" + "=" * 80)
print("MAXIMUM SHARPE RATIO PORTFOLIO (Tangency)")
print("=" * 80)
for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"{ticker:5s}: {weight:>7.2%}")
print(f"\nExpected Return: {ms_return:.2%}")
print(f"Volatility:      {ms_vol:.2%}")
print(f"Sharpe Ratio:    {(ms_return - 0.03) / ms_vol:.3f}")

# Generate efficient frontier
ef_returns, ef_volatilities, ef_weights = efficient_frontier(mean_returns, cov_matrix, 50)

# Generate random portfolios for comparison
num_random = 5000
random_returns = []
random_volatilities = []

np.random.seed(42)
for _ in range(num_random):
    weights = np.random.random(len(tickers))
    weights /= weights.sum()
    ret, vol = portfolio_stats(weights, mean_returns, cov_matrix)
    random_returns.append(ret)
    random_volatilities.append(vol)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Efficient Frontier with random portfolios
axes[0, 0].scatter(random_volatilities, random_returns, c='gray', 
                   alpha=0.2, s=10, label='Random Portfolios')
axes[0, 0].plot(ef_volatilities, ef_returns, 'r-', linewidth=3, 
                label='Efficient Frontier')
axes[0, 0].scatter(mvp_vol, mvp_return, c='green', marker='*', s=500, 
                   label='Min Variance', edgecolors='black', linewidths=2)
axes[0, 0].scatter(ms_vol, ms_return, c='gold', marker='*', s=500,
                   label='Max Sharpe', edgecolors='black', linewidths=2)

# Individual assets
for ticker in tickers:
    asset_return = mean_returns[ticker]
    asset_vol = np.sqrt(cov_matrix.loc[ticker, ticker])
    axes[0, 0].scatter(asset_vol, asset_return, s=100, label=ticker)

# Capital Allocation Line
rf_rate = 0.03
x_cal = np.linspace(0, ef_volatilities.max() * 1.1, 100)
sharpe_tangency = (ms_return - rf_rate) / ms_vol
y_cal = rf_rate + sharpe_tangency * x_cal
axes[0, 0].plot(x_cal, y_cal, 'b--', linewidth=2, alpha=0.7, label='CAL')

axes[0, 0].set_xlabel('Volatility (Standard Deviation)')
axes[0, 0].set_ylabel('Expected Return')
axes[0, 0].set_title('Mean-Variance Efficient Frontier')
axes[0, 0].legend(loc='best', fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Weight evolution along efficient frontier
for i, ticker in enumerate(tickers):
    weights_series = [w[i] for w in ef_weights]
    axes[0, 1].plot(ef_returns, weights_series, label=ticker, linewidth=2)

axes[0, 1].axvline(mvp_return, color='green', linestyle='--', alpha=0.5)
axes[0, 1].axvline(ms_return, color='gold', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Target Return')
axes[0, 1].set_ylabel('Portfolio Weight')
axes[0, 1].set_title('Asset Allocation Along Efficient Frontier')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_ylim(-0.05, 1.05)

# Plot 3: Sensitivity to expected return estimates