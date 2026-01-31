# Mean-Variance Optimization

## 1. Concept Skeleton
**Definition:** Markowitz framework for constructing optimal portfolios by maximizing return for given risk or minimizing risk for given return  
**Purpose:** Quantitative portfolio selection balancing expected return against variance, foundation of Modern Portfolio Theory  
**Prerequisites:** Expected return, covariance matrix, quadratic optimization, linear algebra

## 2. Comparative Framing
| Approach | Mean-Variance | Equal Weight | Risk Parity | Maximum Sharpe |
|----------|---------------|--------------|-------------|----------------|
| **Objective** | Min variance for target return | Simplicity (1/n) | Equal risk contribution | Max Sharpe ratio |
| **Inputs** | μ, Σ | None | Σ only | μ, Σ, rf |
| **Sensitivity** | High to estimation error | None | Medium | Highest |
| **Use Case** | Optimal allocation | Naive diversification | Risk-balanced | Single optimal portfolio |

## 3. Examples + Counterexamples

**Simple Example:**  
Two assets: E[R₁]=10%, σ₁=15%; E[R₂]=8%, σ₂=10%; ρ=0.3  
Min variance portfolio: w₁≈0.36, w₂≈0.64 (lower volatility than either asset alone)

**Failure Case:**  
Estimation error amplification: Slight change in expected return estimates (11% vs 10%) causes extreme weight shift (w₁: 0.4 → 0.9)

**Edge Case:**  
Perfect positive correlation (ρ=1): No diversification benefit, linear efficient frontier

## 4. Layer Breakdown
```
Mean-Variance Optimization Framework:
├─ Mathematical Formulation:
│   ├─ Objective: Minimize σp² = w'Σw
│   ├─ Subject to:
│   │   ├─ Target Return: w'μ = R*
│   │   ├─ Budget Constraint: w'1 = 1 (fully invested)
│   │   └─ Optional: wi ≥ 0 (no short sales)
│   └─ Lagrangian: L = w'Σw + λ₁(R* - w'μ) + λ₂(1 - w'1)
├─ Solution Methods:
│   ├─ Analytical (Unconstrained):
│   │   ├─ w* = Σ⁻¹(λ₁μ + λ₂1)
│   │   └─ Solve for λ₁, λ₂ using constraints
│   ├─ Quadratic Programming (Constrained):
│   │   ├─ cvxopt, scipy.optimize.minimize
│   │   └─ Handles inequality constraints (wi ≥ 0, wi ≤ max)
│   └─ Numerical Optimization:
│       ├─ Sequential Quadratic Programming (SQP)
│       └─ Interior Point Methods
├─ Key Quantities:
│   ├─ A = 1'Σ⁻¹1 (variance of minimum variance portfolio)
│   ├─ B = 1'Σ⁻¹μ (expected return of MVP scaled)
│   ├─ C = μ'Σ⁻¹μ (weighted expected return)
│   └─ D = AC - B² (discriminant, determines frontier shape)
├─ Efficient Frontier:
│   ├─ Equation: σp² = (1/A) + (D/A²)(μp - B/A)²
│   ├─ Hyperbola in μ-σ space
│   └─ Upper branch only (efficient)
└─ Practical Issues:
    ├─ Estimation Error: μ, Σ estimated with noise
    ├─ Corner Solutions: Extreme weights (99% in one asset)
    ├─ Instability: Small input changes → large weight changes
    └─ Transaction Costs: Optimal theory ignores trading costs
```

**Interaction:** Covariance structure dominates; negative correlations create convex efficient frontier

## 5. Mini-Project
Implement mean-variance optimization and explore sensitivity to inputs:
```python
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
def perturb_returns(mean_returns, perturbation_std=0.02):
    """Add noise to expected returns"""
    noise = np.random.normal(0, perturbation_std, len(mean_returns))
    return mean_returns + noise

num_simulations = 100
simulated_mvp_weights = []
simulated_ms_weights = []

np.random.seed(42)
for _ in range(num_simulations):
    perturbed_returns = perturb_returns(mean_returns)
    
    try:
        mvp_w = min_variance(perturbed_returns, cov_matrix)
        ms_w = max_sharpe_portfolio(perturbed_returns, cov_matrix)
        simulated_mvp_weights.append(mvp_w)
        simulated_ms_weights.append(ms_w)
    except:
        pass

simulated_mvp_weights = np.array(simulated_mvp_weights)
simulated_ms_weights = np.array(simulated_ms_weights)

# Box plot of weight distributions
positions = np.arange(len(tickers))
axes[1, 0].boxplot([simulated_mvp_weights[:, i] for i in range(len(tickers))],
                   positions=positions - 0.2, widths=0.3,
                   patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', alpha=0.7),
                   label='Min Variance')
axes[1, 0].boxplot([simulated_ms_weights[:, i] for i in range(len(tickers))],
                   positions=positions + 0.2, widths=0.3,
                   patch_artist=True,
                   boxprops=dict(facecolor='gold', alpha=0.7),
                   label='Max Sharpe')

axes[1, 0].set_xticks(positions)
axes[1, 0].set_xticklabels(tickers)
axes[1, 0].set_ylabel('Weight')
axes[1, 0].set_title('Weight Sensitivity to Return Estimates (±2% noise)')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Sharpe ratio along efficient frontier
sharpe_ratios = (ef_returns - rf_rate) / ef_volatilities
axes[1, 1].plot(ef_volatilities, sharpe_ratios, 'b-', linewidth=2)
axes[1, 1].axhline((ms_return - rf_rate) / ms_vol, color='gold', 
                   linestyle='--', linewidth=2, label='Maximum Sharpe')
axes[1, 1].axvline(ms_vol, color='gold', linestyle='--', alpha=0.5)
axes[1, 1].scatter(ms_vol, (ms_return - rf_rate) / ms_vol, 
                  c='gold', marker='*', s=300, edgecolors='black', linewidths=2)

axes[1, 1].set_xlabel('Portfolio Volatility')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].set_title('Sharpe Ratio Along Efficient Frontier')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Comparison with naive strategies
equal_weight = np.array([1/len(tickers)] * len(tickers))
eq_return, eq_vol = portfolio_stats(equal_weight, mean_returns, cov_matrix)

print("\n" + "=" * 80)
print("COMPARISON WITH NAIVE STRATEGIES")
print("=" * 80)
print(f"{'Strategy':<25} {'Return':>10} {'Volatility':>12} {'Sharpe':>10}")
print("-" * 80)
print(f"{'Minimum Variance':<25} {mvp_return:>9.2%} {mvp_vol:>11.2%} {(mvp_return-0.03)/mvp_vol:>9.3f}")
print(f"{'Maximum Sharpe':<25} {ms_return:>9.2%} {ms_vol:>11.2%} {(ms_return-0.03)/ms_vol:>9.3f}")
print(f"{'Equal Weight (1/n)':<25} {eq_return:>9.2%} {eq_vol:>11.2%} {(eq_return-0.03)/eq_vol:>9.3f}")

# Individual assets
for ticker in tickers:
    asset_return = mean_returns[ticker]
    asset_vol = np.sqrt(cov_matrix.loc[ticker, ticker])
    asset_sharpe = (asset_return - 0.03) / asset_vol
    print(f"{ticker + ' (100%)':<25} {asset_return:>9.2%} {asset_vol:>11.2%} {asset_sharpe:>9.3f}")
```

## 6. Challenge Round
When does mean-variance optimization fail?
- Estimation error dominance: Noise in μ, Σ → extreme, unstable weights (use shrinkage, resampling)
- Non-normal returns: Fat tails, skewness ignored (use higher moments, CVaR optimization)
- Transaction costs: Frequent rebalancing costly (add turnover penalty to objective)
- Short-sale constraints: Corner solutions, less diversification
- Model misspecification: Real returns don't follow mean-variance assumptions

Practical improvements:
- Black-Litterman: Bayesian approach combining equilibrium with views
- Robust optimization: Min-max, worst-case scenarios
- Resampled efficiency: Michaud's averaging over bootstrapped frontiers
- Shrinkage estimators: James-Stein, Ledoit-Wolf for covariance
- Regularization: L1 (LASSO) for sparse portfolios, L2 (Ridge) for stability

## 7. Key References
- [Markowitz, H. (1952) "Portfolio Selection" Journal of Finance](https://www.jstor.org/stable/2975974)
- [Markowitz, H. (1959) "Portfolio Selection: Efficient Diversification of Investments"](https://www.jstor.org/stable/j.ctt1bh4c8h)
- [Merton, R. (1972) "An Analytic Derivation of the Efficient Portfolio Frontier"](https://www.jstor.org/stable/2329621)
- [DeMiguel et al. (2009) "Optimal Versus Naive Diversification"](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901)

---
**Status:** Foundation of Modern Portfolio Theory (Nobel 1990) | **Complements:** Efficient Frontier, CAPM, Black-Litterman
