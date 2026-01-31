from scipy.optimize import minimize
import numpy as np
import pandas as pd

def optimize_cvar_portfolio(returns, target_return=None, alpha=0.95):
    """
    Find portfolio minimizing CVaR with optional return constraint
    """
    n_assets = returns.shape[1]
    
    # Initial guess (equal weight)
    w0 = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
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
        lambda w: cvar_optimization_objective(w, returns, alpha),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

def mean_variance_optimize(returns, target_return=None):
    """
    Traditional mean-variance optimization for comparison
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    def portfolio_volatility(weights):
        return np.sqrt(weights @ cov_matrix @ weights)
    
    w0 = np.ones(n_assets) / n_assets
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ mean_returns - target_return
        })
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(
        portfolio_volatility,
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

# Compute efficient frontiers
print("\nComputing efficient frontiers...")
mean_returns = returns.mean() * 252  # Annualized
n_points = 15

# Range of target returns
min_return = mean_returns.min()
max_return = mean_returns.max()
target_returns = np.linspace(min_return, max_return * 0.9, n_points)

# Storage
mv_portfolios = []
cvar_portfolios = []

for target_ret in target_returns:
    # Mean-Variance
    mv_weights = mean_variance_optimize(returns, target_ret/252)
    mv_vol = np.sqrt(mv_weights @ returns.cov() @ mv_weights) * np.sqrt(252)
    mv_var, mv_cvar = calculate_var_cvar(returns, mv_weights, 0.95)
    
    mv_portfolios.append({
        'return': target_ret,
        'volatility': mv_vol,
        'var': mv_var * np.sqrt(252),
        'cvar': mv_cvar * np.sqrt(252),
        'weights': mv_weights
    })
    
    # CVaR Optimization
    cvar_weights = optimize_cvar_portfolio(returns, target_ret/252, 0.95)
    cvar_ret = cvar_weights @ returns.mean() * 252
    cvar_vol = np.sqrt(cvar_weights @ returns.cov() @ cvar_weights) * np.sqrt(252)
    cv_var, cv_cvar = calculate_var_cvar(returns, cvar_weights, 0.95)
    
    cvar_portfolios.append({
        'return': cvar_ret,
        'volatility': cvar_vol,
        'var': cv_var * np.sqrt(252),
        'cvar': cv_cvar * np.sqrt(252),
        'weights': cvar_weights
    })

mv_df = pd.DataFrame(mv_portfolios)
cvar_df = pd.DataFrame(cvar_portfolios)

# Individual assets
asset_returns = returns.mean() * 252
asset_vols = returns.std() * np.sqrt(252)
asset_cvars = []
for i in range(len(tickers)):
    weights = np.zeros(len(tickers))
    weights[i] = 1.0
    _, cvar = calculate_var_cvar(returns, weights, 0.95)
    asset_cvars.append(cvar * np.sqrt(252))

# Compare specific portfolios
print("\n" + "="*100)
print("COMPARISON: Mean-Variance vs CVaR Optimization (Target Return: 8%)")
print("="*100)

target_idx = np.argmin(np.abs(mv_df['return'] - 0.08))
mv_port = mv_df.iloc[target_idx]
cvar_port = cvar_df.iloc[target_idx]

comparison = pd.DataFrame({
    'Mean-Variance': [
        mv_port['return'],
        mv_port['volatility'],
        mv_port['var'],
        mv_port['cvar'],
        mv_port['return'] / mv_port['volatility'],
        mv_port['return'] / mv_port['cvar']
    ],
    'CVaR Optimized': [
        cvar_port['return'],
        cvar_port['volatility'],
        cvar_port['var'],
        cvar_port['cvar'],
        cvar_port['return'] / cvar_port['volatility'],
        cvar_port['return'] / cvar_port['cvar']
    ]
}, index=['Annual Return', 'Volatility', '95% VaR', '95% CVaR', 'Sharpe Ratio', 'Return/CVaR'])

print(comparison.round(4))

# Portfolio weights comparison
print("\n" + "="*100)
print("PORTFOLIO WEIGHTS COMPARISON")
print("="*100)
weights_comp = pd.DataFrame({
    'Mean-Variance': mv_port['weights'],
    'CVaR Optimized': cvar_port['weights']
}, index=tickers)
print(weights_comp.round(4))

# CVaR contribution analysis