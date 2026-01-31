import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats

# Download data
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'VNQ']
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

def semivariance(returns, target=None, below=True):
    """
    Calculate semi-variance (variance of returns below/above target)
    
    Parameters:
    - returns: Series or array of returns
    - target: Threshold (default: mean)
    - below: If True, variance below target; if False, above target
    """
    if target is None:
        target = returns.mean()
    
    if below:
        # Below target (downside)
        deviations = returns[returns < target] - target
    else:
        # Above target (upside)
        deviations = returns[returns > target] - target
    
    if len(deviations) > 0:
        sv = (deviations ** 2).mean()
    else:
        sv = 0
    
    return sv

def downside_deviation(returns, target=None):
    """
    Calculate downside deviation (square root of semi-variance)
    """
    sv = semivariance(returns, target, below=True)
    return np.sqrt(sv)

def sortino_ratio(returns, target=0, periods=252):
    """
    Calculate Sortino Ratio
    
    Parameters:
    - returns: Daily returns
    - target: Minimum Acceptable Return (MAR)
    - periods: Annualization factor
    """
    excess_returns = returns - target/periods
    mean_excess = excess_returns.mean() * periods
    dd = downside_deviation(returns, target/periods) * np.sqrt(periods)
    
    if dd > 0:
        sortino = mean_excess / dd
    else:
        sortino = np.nan
    
    return sortino

def lower_partial_moment(returns, target=0, order=2):
    """
    Calculate Lower Partial Moment of given order
    
    LPM_n(τ) = E[(max(τ - R, 0))^n]
    """
    shortfalls = np.maximum(target - returns, 0)
    lpm = (shortfalls ** order).mean()
    return lpm

def upside_downside_capture(returns, benchmark_returns, periods=252):
    """
    Calculate upside and downside capture ratios
    """
    # Identify up and down periods for benchmark
    up_periods = benchmark_returns > 0
    down_periods = benchmark_returns < 0
    
    # Average returns in those periods
    asset_up = returns[up_periods].mean() * periods
    asset_down = returns[down_periods].mean() * periods
    bench_up = benchmark_returns[up_periods].mean() * periods
    bench_down = benchmark_returns[down_periods].mean() * periods
    
    # Capture ratios
    upside_capture = (asset_up / bench_up) * 100 if bench_up != 0 else np.nan
    downside_capture = (asset_down / bench_down) * 100 if bench_down != 0 else np.nan
    
    return upside_capture, downside_capture

def downside_beta(returns, market_returns):
    """
    Calculate downside beta (beta conditional on market declines)
    """
    # Only periods when market is down
    market_mean = market_returns.mean()
    down_periods = market_returns < market_mean
    
    if down_periods.sum() > 10:  # Need sufficient observations
        returns_down = returns[down_periods]
        market_down = market_returns[down_periods]
        
        # Covariance and variance in down periods
        cov_down = np.cov(returns_down, market_down)[0, 1]
        var_down = market_down.var()
        
        beta_down = cov_down / var_down if var_down > 0 else np.nan
    else:
        beta_down = np.nan
    
    return beta_down

# Calculate metrics for all assets
metrics = {}

for ticker in tickers:
    ret = returns[ticker]
    
    # Traditional metrics
    annual_return = ret.mean() * 252
    volatility = ret.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / volatility if volatility > 0 else np.nan
    
    # Downside metrics
    sv_mean = semivariance(ret, target=ret.mean())
    sv_zero = semivariance(ret, target=0)
    dd_mean = downside_deviation(ret, target=ret.mean()) * np.sqrt(252)
    dd_zero = downside_deviation(ret, target=0) * np.sqrt(252)
    
    sortino = sortino_ratio(ret, target=0.02, periods=252)
    
    # LPMs
    lpm0 = lower_partial_moment(ret, target=0, order=0)  # Shortfall probability
    lpm1 = lower_partial_moment(ret, target=0, order=1) * 252  # Expected shortfall
    lpm2 = lower_partial_moment(ret, target=0, order=2) * 252  # Semi-variance
    
    # Upside/Downside capture vs SPY
    if ticker != 'SPY':
        up_cap, down_cap = upside_downside_capture(ret, returns['SPY'])
    else:
        up_cap, down_cap = 100, 100
    
    # Downside beta
    if ticker != 'SPY':
        dbeta = downside_beta(ret, returns['SPY'])
    else:
        dbeta = 1.0
    
    metrics[ticker] = {
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Downside Dev (mean)': dd_mean,
        'Downside Dev (0%)': dd_zero,
        'Sortino Ratio': sortino,
        'Shortfall Prob': lpm0,
        'Expected Shortfall': lpm1,
        'Upside Capture': up_cap,
        'Downside Capture': down_cap,
        'Downside Beta': dbeta
    }

metrics_df = pd.DataFrame(metrics).T

print("\n" + "="*110)
print("DOWNSIDE RISK METRICS COMPARISON")
print("="*110)
print(metrics_df.round(4))

# Mean-Semivariance Optimization