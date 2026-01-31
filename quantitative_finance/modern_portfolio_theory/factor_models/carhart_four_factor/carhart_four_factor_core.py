import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats

# Download asset data
tickers = {
    'SPY': 'S&P 500',
    'IWM': 'Russell 2000',
    'MTUM': 'Momentum ETF',
    'VLUE': 'Value ETF',
    'SIZE': 'Size ETF',
    'ARKK': 'Growth/Momentum Fund',
    'VFINX': 'Vanguard 500 Index',
    'FCNTX': 'Fidelity Contrafund',
    'DODGX': 'Dodge & Cox Stock',
    'PRWCX': 'T. Rowe Price Cap Apprec'
}

end_date = datetime.now()
start_date = datetime(2015, 1, 1)

print("Downloading data...")
data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Construct momentum factor (WML proxy)
# Use universe of stocks to create winners - losers
stock_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
                  'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'DIS', 'BAC',
                  'XOM', 'CVX', 'PFE', 'KO', 'PEP', 'COST', 'WMT', 'MCD', 'CSCO',
                  'IBM', 'INTC', 'ORCL', 'T', 'VZ', 'MRK']

print("Downloading stock universe for momentum factor...")
stock_data = yf.download(stock_universe, start=start_date, end=end_date, progress=False)['Adj Close']
stock_returns = stock_data.pct_change().dropna()

def construct_momentum_factor(returns_data, formation_period=11, skip_month=1):
    """
    Construct WML (Winners Minus Losers) momentum factor
    
    Parameters:
    - returns_data: DataFrame of stock returns
    - formation_period: Lookback period (typically 11 months)
    - skip_month: Skip most recent month (avoid short-term reversal)
    """
    wml_series = pd.Series(index=returns_data.index, dtype=float)
    
    for i in range(formation_period + skip_month, len(returns_data)):
        current_date = returns_data.index[i]
        
        # Formation period: t-12 to t-2 (if skip_month=1, formation_period=11)
        formation_start = i - formation_period - skip_month
        formation_end = i - skip_month
        
        # Calculate cumulative returns during formation period
        formation_returns = (1 + returns_data.iloc[formation_start:formation_end]).prod() - 1
        
        # Rank stocks
        valid_stocks = formation_returns.dropna()
        if len(valid_stocks) < 10:  # Need sufficient stocks
            continue
        
        # Top 30% winners, bottom 30% losers
        n_stocks = len(valid_stocks)
        n_winners = int(n_stocks * 0.3)
        n_losers = int(n_stocks * 0.3)
        
        sorted_stocks = valid_stocks.sort_values(ascending=False)
        winners = sorted_stocks.iloc[:n_winners].index
        losers = sorted_stocks.iloc[-n_losers:].index
        
        # Holding period return (current month)
        if i < len(returns_data):
            winner_return = returns_data.loc[current_date, winners].mean()
            loser_return = returns_data.loc[current_date, losers].mean()
            
            wml_series[current_date] = winner_return - loser_return
    
    return wml_series.dropna()

# Construct WML factor
print("Constructing momentum factor (WML)...")
wml = construct_momentum_factor(stock_returns, formation_period=11, skip_month=1)

# Construct other factors (proxies using ETFs)
rf_monthly = 0.02 / 12  # Risk-free rate approximation

# Market factor
mkt_rf = returns['SPY'] - rf_monthly

# SMB proxy (small - big)
smb = returns['IWM'] - returns['SPY']

# HML proxy (value - growth)
hml = returns['VLUE'] - returns['MTUM']  # Approximation

# Create factor DataFrame
# Align dates (WML may have fewer observations due to formation period)
factors_ff3 = pd.DataFrame({
    'Mkt-RF': mkt_rf,
    'SMB': smb,
    'HML': hml
})

factors_carhart = pd.DataFrame({
    'Mkt-RF': mkt_rf,
    'SMB': smb,
    'HML': hml,
    'WML': wml
})

# Remove NaN
factors_ff3 = factors_ff3.dropna()
factors_carhart = factors_carhart.dropna()
