import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

# Download data
tickers = {
    'Defensive': ['PG', 'KO', 'WMT'],  # Consumer staples
    'Market-Like': ['SPY'],  # S&P 500 ETF
    'Aggressive': ['TSLA', 'NVDA', 'AMD'],  # High beta tech
    'Leveraged': ['TQQQ', 'UPRO'],  # 3x leveraged ETFs
}

all_tickers = [t for group in tickers.values() for t in group]
market_ticker = 'SPY'

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Risk-free rate
rf_annual = 0.03
rf_daily = (1 + rf_annual) ** (1/252) - 1

# Calculate excess returns
excess_returns = returns.subtract(rf_daily, axis=0)
market_excess = excess_returns[market_ticker]

def calculate_beta(stock_returns, market_returns, method='ols'):
    """
    Calculate beta using OLS regression
    Returns: beta, alpha, se_beta, t_stat, p_value, r_squared
    """
    # Align data
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ['stock', 'market']
    
    if method == 'ols':
        # Regression method
        X = sm.add_constant(aligned['market'])
        model = sm.OLS(aligned['stock'], X).fit()
        
        beta = model.params[1]
        alpha = model.params[0]
        se_beta = model.bse[1]
        t_stat = model.tvalues[1]
        p_value = model.pvalues[1]
        r_squared = model.rsquared
        
    elif method == 'covariance':
        # Direct calculation
        beta = aligned.cov().loc['stock', 'market'] / aligned['market'].var()
        alpha = aligned['stock'].mean() - beta * aligned['market'].mean()
        se_beta = np.nan
        t_stat = np.nan
        p_value = np.nan
        r_squared = aligned.corr().loc['stock', 'market'] ** 2
    
    return {
        'beta': beta,
        'alpha': alpha * 252,  # Annualize
        'se_beta': se_beta,
        't_statistic': t_stat,
        'p_value': p_value,
        'r_squared': r_squared,
        'n_obs': len(aligned)
    }

def adjusted_beta(raw_beta, adjustment_factor=1/3):
    """
    Blume adjustment: weighted average of raw beta and 1.0
    Theory: betas tend to revert toward 1.0 over time
    """
    return (1 - adjustment_factor) * raw_beta + adjustment_factor * 1.0

# Calculate betas for all stocks
beta_results = {}
for ticker in all_tickers:
    if ticker != market_ticker and ticker in excess_returns.columns:
        results = calculate_beta(excess_returns[ticker], market_excess)
        results['adjusted_beta'] = adjusted_beta(results['beta'])
        beta_results[ticker] = results

# Convert to DataFrame
beta_df = pd.DataFrame(beta_results).T
beta_df = beta_df.sort_values('beta')

print("\n" + "=" * 100)
print("BETA ANALYSIS")
print("=" * 100)
print(beta_df[['beta', 'adjusted_beta', 'se_beta', 'r_squared', 'alpha']].round(4))

# Rolling beta analysis
def rolling_beta(stock_returns, market_returns, window=252):
    """Calculate rolling beta over time"""
    betas = []
    dates = []
    
    for i in range(window, len(stock_returns)):
        window_stock = stock_returns.iloc[i-window:i]
        window_market = market_returns.iloc[i-window:i]
        
        aligned = pd.concat([window_stock, window_market], axis=1).dropna()
        if len(aligned) > 50:
            cov = aligned.cov().iloc[0, 1]
            var = aligned.iloc[:, 1].var()
            beta = cov / var
            
            betas.append(beta)
            dates.append(stock_returns.index[i])
    
    return pd.Series(betas, index=dates)

# Calculate rolling betas for selected stocks
rolling_betas = {}
example_stocks = ['TSLA', 'PG', 'NVDA']
for ticker in example_stocks:
    if ticker in excess_returns.columns:
        rolling_betas[ticker] = rolling_beta(excess_returns[ticker], market_excess, window=252)

# Simulate portfolio betas