import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import t, norm
from scipy.stats import norm

# Download portfolio data
tickers = ['SPY', 'AGG', 'GLD', 'EEM']
weights = np.array([0.5, 0.3, 0.1, 0.1])  # Portfolio weights

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Portfolio returns
portfolio_returns = (returns * weights).sum(axis=1)

# Initial portfolio value
initial_value = 1000000  # $1M

def parametric_var(returns, confidence_level=0.95, distribution='normal'):
    """
    Calculate VaR assuming parametric distribution
    """
    mu = returns.mean()
    sigma = returns.std()
    
    if distribution == 'normal':
        z_score = norm.ppf(1 - confidence_level)
        var = -(mu + z_score * sigma)
    elif distribution == 't':
        # Fit t-distribution
        params = t.fit(returns)
        df = params[0]
        loc = params[1]
        scale = params[2]
        t_score = t.ppf(1 - confidence_level, df)
        var = -(loc + t_score * scale)
    
    return var

def historical_var(returns, confidence_level=0.95):
    """
    Calculate VaR using historical simulation
    """
    return -np.percentile(returns, (1 - confidence_level) * 100)

def monte_carlo_var(returns, confidence_level=0.95, n_simulations=10000):
    """
    Calculate VaR using Monte Carlo simulation with t-distribution
    """
    # Fit t-distribution to returns
    params = t.fit(returns)
    df, loc, scale = params
    
    # Generate simulations
    simulated_returns = t.rvs(df, loc=loc, scale=scale, size=n_simulations)
    
    # Calculate VaR
    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return var, simulated_returns

def conditional_var(returns, confidence_level=0.95):
    """
    Calculate CVaR (Expected Shortfall) - average loss beyond VaR
    """
    var_threshold = historical_var(returns, confidence_level)
    # Average of losses exceeding VaR
    tail_losses = returns[returns < -var_threshold]
    if len(tail_losses) > 0:
        cvar = -tail_losses.mean()
    else:
        cvar = var_threshold
    return cvar

# Calculate VaR for different confidence levels and methods
confidence_levels = [0.90, 0.95, 0.99]
var_results = {}

for conf in confidence_levels:
    var_results[conf] = {
        'Parametric (Normal)': parametric_var(portfolio_returns, conf, 'normal'),
        'Parametric (t-dist)': parametric_var(portfolio_returns, conf, 't'),
        'Historical': historical_var(portfolio_returns, conf),
        'Monte Carlo': monte_carlo_var(portfolio_returns, conf)[0],
        'CVaR': conditional_var(portfolio_returns, conf)
    }

# Convert to DataFrame
var_df = pd.DataFrame(var_results).T * 100  # Convert to percentage

print("\n" + "=" * 90)
print("VALUE AT RISK COMPARISON")
print("=" * 90)
print(var_df.round(3))

# Dollar VaR
print("\n" + "=" * 90)
print(f"DOLLAR VaR (Portfolio Value: ${initial_value:,.0f})")
print("=" * 90)
dollar_var_df = var_df * initial_value / 100
print(dollar_var_df.round(0))

# Backtesting