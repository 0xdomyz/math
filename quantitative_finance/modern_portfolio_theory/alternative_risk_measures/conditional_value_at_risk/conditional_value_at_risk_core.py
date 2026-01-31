import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import norm

# Download data
tickers = ['SPY', 'AGG', 'TLT', 'GLD', 'VNQ', 'EEM', 'DBC']
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

def historical_cvar(returns, weights, alpha=0.95):
    """
    Calculate CVaR using historical method
    """
    portfolio_returns = (returns @ weights)
    var_threshold = -np.percentile(portfolio_returns, (1-alpha)*100)
    
    # Losses exceeding VaR
    tail_losses = -portfolio_returns[portfolio_returns < -var_threshold]
    
    if len(tail_losses) > 0:
        cvar = tail_losses.mean()
    else:
        cvar = var_threshold
    
    return cvar

def parametric_cvar_normal(returns, weights, alpha=0.95):
    """
    Calculate CVaR assuming normal distribution
    """
    portfolio_returns = returns @ weights
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    
    # For normal: CVaR = μ - σ × φ(z_α)/(1-α)
    z_alpha = stats.norm.ppf(alpha)
    phi_z = stats.norm.pdf(z_alpha)
    
    cvar = -(mu - sigma * phi_z / (1 - alpha))
    return cvar

def parametric_cvar_t(returns, weights, alpha=0.95):
    """
    Calculate CVaR assuming t-distribution
    """
    portfolio_returns = returns @ weights
    
    # Fit t-distribution
    params = stats.t.fit(portfolio_returns)
    df, loc, scale = params
    
    # t-distribution CVaR formula
    t_alpha = stats.t.ppf(alpha, df)
    f_t = stats.t.pdf(t_alpha, df)
    
    cvar = -(loc - scale * (df + t_alpha**2) / (df - 1) * f_t / (1 - alpha))
    return cvar

def calculate_var_cvar(returns, weights, alpha=0.95):
    """
    Calculate both VaR and CVaR
    """
    portfolio_returns = returns @ weights
    
    # VaR
    var = -np.percentile(portfolio_returns, (1-alpha)*100)
    
    # CVaR
    cvar = historical_cvar(returns, weights, alpha)
    
    return var, cvar

# CVaR Optimization (Rockafellar-Uryasev formulation)
def cvar_optimization_objective(weights, returns, alpha=0.95):
    """
    Minimize CVaR using historical simulation
    For discrete distribution: CVaR_α = min_t {t + 1/(1-α) * mean([L-t]⁺)}
    """
    portfolio_returns = returns @ weights
    losses = -portfolio_returns
    
    # Optimize over auxiliary variable t (VaR estimate)
    def objective_with_t(t):
        # [L - t]⁺ = max(L - t, 0)
        excess_losses = np.maximum(losses - t, 0)
        cvar = t + np.mean(excess_losses) / (1 - alpha)
        return cvar
    
    # Find optimal t
    result = minimize(objective_with_t, x0=np.percentile(losses, alpha*100), 
                     method='BFGS')
    
    return result.fun
