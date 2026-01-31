# Efficient Frontier

## 1. Concept Skeleton
**Definition:** Set of optimal portfolios offering maximum expected return for each level of risk, or minimum risk for each level of return  
**Purpose:** Visualize risk-return tradeoff, identify dominated portfolios, guide portfolio selection  
**Prerequisites:** Mean-variance optimization, portfolio variance calculation, diversification benefits

## 2. Comparative Framing
| Concept | Efficient Frontier | Capital Market Line | Indifference Curves |
|---------|-------------------|---------------------|---------------------|
| **Definition** | Optimal risky portfolios | Efficient frontier + risk-free asset | Investor utility contours |
| **Dimension** | Risk-return plane | Risk-return plane | Risk-return plane |
| **Shape** | Hyperbola (upper branch) | Straight line (tangent to frontier) | Convex curves |
| **Use** | Feasible optimal set | Leverage/lending decisions | Personal preferences |

## 3. Examples + Counterexamples

**Simple Example:**  
Two assets with ρ=0.5: Frontier is curved, interior portfolios outperform individual assets on risk basis

**Failure Case:**  
Perfect correlation (ρ=1): Efficient frontier collapses to straight line between assets, no diversification benefit

**Edge Case:**  
Negative correlation (ρ=-1): Can achieve zero-variance portfolio, frontier touches vertical axis

## 4. Layer Breakdown
```
Efficient Frontier Construction:
├─ Mathematical Properties:
│   ├─ Equation: σp² = (1/A) + (D/A²)(μp - B/A)²
│   │   where A = 1'Σ⁻¹1, B = 1'Σ⁻¹μ, C = μ'Σ⁻¹μ, D = AC - B²
│   ├─ Shape: Hyperbola in mean-standard deviation space
│   ├─ Global Minimum Variance Portfolio: Vertex at μmvp = B/A, σmvp = 1/√A
│   └─ Efficient Branch: Upper portion (μ ≥ μmvp)
├─ Key Portfolios:
│   ├─ Minimum Variance Portfolio (MVP):
│   │   ├─ Lowest risk point on frontier
│   │   ├─ Weights: wmvp = Σ⁻¹1 / (1'Σ⁻¹1)
│   │   └─ Not necessarily highest Sharpe
│   ├─ Tangency Portfolio:
│   │   ├─ Highest Sharpe ratio (with risk-free asset)
│   │   ├─ Weights: wtan ∝ Σ⁻¹(μ - rf·1)
│   │   └─ Optimal risky portfolio for all investors
│   └─ Target Return Portfolios:
│       ├─ Any specified expected return on efficient branch
│       └─ Solve constrained optimization for that return
├─ Dominated Region:
│   ├─ Below Efficient Frontier: Inferior risk-return
│   ├─ Lower Branch of Hyperbola: Inefficient (higher risk, same return)
│   └─ Interior Random Portfolios: Typically dominated
├─ Two-Fund Separation:
│   ├─ Any efficient portfolio = linear combination of 2 frontier portfolios
│   ├─ With Risk-Free Asset: All efficient = rf + risky (tangency)
│   └─ Practical: Hold market portfolio + adjust risk via cash/leverage
└─ Frontier Shifts:
    ├─ New Asset Added: Frontier moves out (never worse)
    ├─ Correlation Decrease: More curvature, greater benefit
    ├─ Constraints Added: Frontier moves in (more restricted)
    └─ Return Estimates Change: Frontier rotates/shifts
```

**Interaction:** Diversification creates convexity; lower correlations → more pronounced curvature

## 5. Mini-Project
Construct and analyze efficient frontier with various constraints:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# Download data
tickers = ['SPY', 'AGG', 'GLD', 'VNQ', 'DBC']  # Stocks, Bonds, Gold, REIT, Commodities
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

def portfolio_stats(weights, mean_returns, cov_matrix):
    """Portfolio return and volatility"""
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def efficient_frontier(mean_returns, cov_matrix, num_points=100, 
                      short_sales=True, max_weight=1.0):
    """
    Generate efficient frontier
    short_sales: Allow negative weights
    max_weight: Maximum allocation to single asset
    """
    min_ret = mean_returns.min() * 0.8
    max_ret = mean_returns.max() * 1.2
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    frontier_vols = []
    frontier_rets = []
    frontier_weights = []
    
    # Set bounds
    if short_sales:
        bounds = tuple((-1, max_weight) for _ in range(len(mean_returns)))
    else:
        bounds = tuple((0, max_weight) for _ in range(len(mean_returns)))
    
    for target_ret in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_stats(w, mean_returns, cov_matrix)[0] - target_ret}
        )
        
        init_weights = np.array([1/len(mean_returns)] * len(mean_returns))
        
        result = minimize(
            lambda w: portfolio_stats(w, mean_returns, cov_matrix)[1],
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            ret, vol = portfolio_stats(result.x, mean_returns, cov_matrix)
            frontier_rets.append(ret)
            frontier_vols.append(vol)
            frontier_weights.append(result.x)
    
    return np.array(frontier_rets), np.array(frontier_vols), frontier_weights

# Generate multiple efficient frontiers
print("Generating efficient frontiers...")
ef_unconstrained_ret, ef_unconstrained_vol, ef_unc_weights = efficient_frontier(
    mean_returns, cov_matrix, short_sales=True, max_weight=2.0
)

ef_longonly_ret, ef_longonly_vol, ef_lo_weights = efficient_frontier(
    mean_returns, cov_matrix, short_sales=False, max_weight=1.0
)

ef_constrained_ret, ef_constrained_vol, ef_con_weights = efficient_frontier(
    mean_returns, cov_matrix, short_sales=False, max_weight=0.4
)

# Find minimum variance portfolios for each
mvp_idx_unc = np.argmin(ef_unconstrained_vol)
mvp_idx_lo = np.argmin(ef_longonly_vol)
mvp_idx_con = np.argmin(ef_constrained_vol)

# Calculate maximum Sharpe portfolios
rf_rate = 0.03

sharpe_unc = (ef_unconstrained_ret - rf_rate) / ef_unconstrained_vol
sharpe_lo = (ef_longonly_ret - rf_rate) / ef_longonly_vol
sharpe_con = (ef_constrained_ret - rf_rate) / ef_constrained_vol

max_sharpe_idx_unc = np.argmax(sharpe_unc)
max_sharpe_idx_lo = np.argmax(sharpe_lo)
max_sharpe_idx_con = np.argmax(sharpe_con)

# Random portfolios
num_random = 3000
random_rets = []
random_vols = []

np.random.seed(42)
for _ in range(num_random):
    weights = np.random.random(len(tickers))
    weights /= weights.sum()
    ret, vol = portfolio_stats(weights, mean_returns, cov_matrix)
    random_rets.append(ret)
    random_vols.append(vol)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Multiple efficient frontiers
axes[0, 0].scatter(random_vols, random_rets, c='lightgray', 
                   alpha=0.3, s=5, label='Random Portfolios')

axes[0, 0].plot(ef_unconstrained_vol, ef_unconstrained_ret, 
                'b-', linewidth=3, label='Unconstrained (short sales)')
axes[0, 0].plot(ef_longonly_vol, ef_longonly_ret, 
                'g-', linewidth=3, label='Long Only')
axes[0, 0].plot(ef_constrained_vol, ef_constrained_ret, 
                'r-', linewidth=3, label='Max 40% per asset')

# Mark MVPs
axes[0, 0].scatter(ef_unconstrained_vol[mvp_idx_unc], ef_unconstrained_ret[mvp_idx_unc],
                   c='blue', marker='o', s=200, edgecolors='black', linewidths=2)
axes[0, 0].scatter(ef_longonly_vol[mvp_idx_lo], ef_longonly_ret[mvp_idx_lo],
                   c='green', marker='o', s=200, edgecolors='black', linewidths=2)
axes[0, 0].scatter(ef_constrained_vol[mvp_idx_con], ef_constrained_ret[mvp_idx_con],
                   c='red', marker='o', s=200, edgecolors='black', linewidths=2)

# Mark Max Sharpe portfolios
axes[0, 0].scatter(ef_unconstrained_vol[max_sharpe_idx_unc], 
                   ef_unconstrained_ret[max_sharpe_idx_unc],
                   c='blue', marker='*', s=400, edgecolors='black', linewidths=2)
axes[0, 0].scatter(ef_longonly_vol[max_sharpe_idx_lo], 
                   ef_longonly_ret[max_sharpe_idx_lo],
                   c='green', marker='*', s=400, edgecolors='black', linewidths=2)
axes[0, 0].scatter(ef_constrained_vol[max_sharpe_idx_con], 
                   ef_constrained_ret[max_sharpe_idx_con],
                   c='red', marker='*', s=400, edgecolors='black', linewidths=2)

# Individual assets
for ticker in tickers:
    asset_ret = mean_returns[ticker]
    asset_vol = np.sqrt(cov_matrix.loc[ticker, ticker])
    axes[0, 0].scatter(asset_vol, asset_ret, s=100, label=ticker, zorder=5)

axes[0, 0].set_xlabel('Volatility (Standard Deviation)')
axes[0, 0].set_ylabel('Expected Return')
axes[0, 0].set_title('Efficient Frontiers with Different Constraints')
axes[0, 0].legend(loc='best', fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Weights along frontier (long only)
for i, ticker in enumerate(tickers):
    weights = [w[i] for w in ef_lo_weights]
    axes[0, 1].plot(ef_longonly_ret, weights, label=ticker, linewidth=2)

axes[0, 1].axvline(ef_longonly_ret[mvp_idx_lo], color='green', 
                   linestyle='--', alpha=0.5, label='MVP')
axes[0, 1].axvline(ef_longonly_ret[max_sharpe_idx_lo], color='gold',
                   linestyle='--', alpha=0.5, label='Max Sharpe')

axes[0, 1].set_xlabel('Expected Return')
axes[0, 1].set_ylabel('Portfolio Weight')
axes[0, 1].set_title('Asset Allocation Along Efficient Frontier (Long Only)')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_ylim(-0.05, 1.05)

# Plot 3: Capital Allocation Line
tangency_vol = ef_longonly_vol[max_sharpe_idx_lo]
tangency_ret = ef_longonly_ret[max_sharpe_idx_lo]
sharpe_tangency = (tangency_ret - rf_rate) / tangency_vol

x_cal = np.linspace(0, ef_longonly_vol.max() * 1.3, 100)
y_cal = rf_rate + sharpe_tangency * x_cal

axes[1, 0].scatter(random_vols, random_rets, c='lightgray', 
                   alpha=0.3, s=5, label='Random')
axes[1, 0].plot(ef_longonly_vol, ef_longonly_ret, 
                'g-', linewidth=3, label='Efficient Frontier')
axes[1, 0].plot(x_cal, y_cal, 'b--', linewidth=3, label='CAL', alpha=0.7)

axes[1, 0].scatter(0, rf_rate, c='red', marker='D', s=200, 
                   label=f'Risk-Free ({rf_rate:.1%})', zorder=5)
axes[1, 0].scatter(tangency_vol, tangency_ret, c='gold', marker='*', s=500,
                   edgecolors='black', linewidths=2, label='Tangency', zorder=5)

# Show leverage point
leverage_vol = ef_longonly_vol.max() * 1.2
leverage_ret = rf_rate + sharpe_tangency * leverage_vol
axes[1, 0].scatter(leverage_vol, leverage_ret, c='orange', marker='s', s=200,
                   label='Leveraged Portfolio', zorder=5)

axes[1, 0].set_xlabel('Volatility')
axes[1, 0].set_ylabel('Expected Return')
axes[1, 0].set_title('Capital Allocation Line and Tangency Portfolio')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Sharpe ratio evolution
axes[1, 1].plot(ef_longonly_vol, sharpe_lo, 'g-', linewidth=3, label='Long Only')
axes[1, 1].plot(ef_constrained_vol, sharpe_con, 'r-', linewidth=3, 
                label='Constrained (40% max)')

axes[1, 1].axhline(sharpe_lo[max_sharpe_idx_lo], color='green', 
                   linestyle='--', alpha=0.5)
axes[1, 1].axvline(ef_longonly_vol[max_sharpe_idx_lo], color='green',
                   linestyle='--', alpha=0.5)

axes[1, 1].scatter(ef_longonly_vol[max_sharpe_idx_lo], 
                   sharpe_lo[max_sharpe_idx_lo],
                   c='gold', marker='*', s=400, edgecolors='black', linewidths=2)

axes[1, 1].set_xlabel('Portfolio Volatility')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].set_title('Sharpe Ratio Along Efficient Frontier')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Print key portfolios
print("\n" + "=" * 80)
print("MINIMUM VARIANCE PORTFOLIO (Long Only)")
print("=" * 80)
mvp_weights = ef_lo_weights[mvp_idx_lo]
for ticker, weight in zip(tickers, mvp_weights):
    print(f"{ticker:5s}: {weight:>7.2%}")
print(f"\nReturn:     {ef_longonly_ret[mvp_idx_lo]:.2%}")
print(f"Volatility: {ef_longonly_vol[mvp_idx_lo]:.2%}")
print(f"Sharpe:     {sharpe_lo[mvp_idx_lo]:.3f}")

print("\n" + "=" * 80)
print("TANGENCY PORTFOLIO (Maximum Sharpe)")
print("=" * 80)
tan_weights = ef_lo_weights[max_sharpe_idx_lo]
for ticker, weight in zip(tickers, tan_weights):
    print(f"{ticker:5s}: {weight:>7.2%}")
print(f"\nReturn:     {ef_longonly_ret[max_sharpe_idx_lo]:.2%}")
print(f"Volatility: {ef_longonly_vol[max_sharpe_idx_lo]:.2%}")
print(f"Sharpe:     {sharpe_lo[max_sharpe_idx_lo]:.3f}")

# Efficient frontier improvement from diversification
single_asset_sharpes = [(mean_returns[t] - rf_rate) / np.sqrt(cov_matrix.loc[t, t]) 
                        for t in tickers]
best_single_sharpe = max(single_asset_sharpes)

print("\n" + "=" * 80)
print("DIVERSIFICATION BENEFIT")
print("=" * 80)
print(f"Best Single Asset Sharpe:  {best_single_sharpe:.3f}")
print(f"Tangency Portfolio Sharpe: {sharpe_lo[max_sharpe_idx_lo]:.3f}")
print(f"Improvement:               {((sharpe_lo[max_sharpe_idx_lo] / best_single_sharpe) - 1) * 100:.1f}%")
```

## 6. Challenge Round
What causes efficient frontier to shift or change shape?
- Add uncorrelated asset: Frontier expands outward (better opportunities)
- Correlations increase: Frontier flattens (less diversification benefit)
- Short-sale constraints: Frontier shrinks inward
- Concentration limits: Further restricts feasible set
- Estimation error: Frontier unstable, small input changes → large shifts

Limitations and misconceptions:
- Backward-looking: Based on historical estimates, not future reality
- Input sensitivity: Small errors in μ magnified in optimal weights
- Single period: Ignores rebalancing, transaction costs, multi-period goals
- Mean-variance only: Ignores skewness, kurtosis, tail risk
- Practical infeasibility: Some frontier portfolios have extreme leverage/shorts

## 7. Key References
- [Markowitz, H. (1952) "Portfolio Selection"](https://www.jstor.org/stable/2975974)
- [Merton, R. (1972) "An Analytic Derivation of the Efficient Portfolio Frontier"](https://www.jstor.org/stable/2329621)
- [Elton & Gruber (1997) "Modern Portfolio Theory, 1950 to Date"](https://www.sciencedirect.com/science/article/abs/pii/S0378426697000487)
- [Investopedia - Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp)

---
**Status:** Core MPT visualization | **Complements:** Mean-Variance Optimization, Capital Market Line, Two-Fund Separation
