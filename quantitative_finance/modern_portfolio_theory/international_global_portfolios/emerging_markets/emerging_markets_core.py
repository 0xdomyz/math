import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Build an EM valuation model to identify buy/sell opportunities and crisis risk

def get_em_data(start_date, end_date):
    """
    Fetch MSCI Emerging Markets data and constituent countries.
    """
    # Main EM indices
    indices = {
        'EEM': 'MSCI Emerging Markets ETF',
        'VWO': 'Vanguard MSCI EM ETF',
        # Individual countries (proxies)
        'ASHR': 'China (Arca)',
        'INDY': 'India (Arca)',
        'EWZ': 'Brazil iShares',
        'ERUS': 'Russia iShares',
        'RSX': 'Russia iShares (alt)',
        'MXEA': 'Mexico iShares',
        'EWW': 'Mexico iShares (alt)',
    }
    
    data = yf.download([k for k in indices.keys() if k not in ['ERUS', 'RSX', 'EWW']], 
                       start=start_date, end=end_date, progress=False)['Adj Close']
    
    returns = data.pct_change().dropna()
    return returns, indices


def calculate_valuation_metrics(prices, start_date, end_date):
    """
    Estimate EM valuation using price-to-book and earnings yield proxies.
    Use rolling correlations as risk indicator.
    """
    # Approximate P/E from historical returns + dividend yield
    # Cannot calculate true P/E from just prices, so use volatility and correlation as proxies
    
    # Calculate rolling 252-day volatility
    rolling_vol = prices.rolling(252).std() * np.sqrt(252)
    
    # Calculate rolling correlation with S&P 500
    if 'SPY' not in prices.columns:
        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
        prices_with_spy = prices.copy()
        prices_with_spy['SPY'] = spy_data
    else:
        prices_with_spy = prices
    
    returns_all = prices_with_spy.pct_change().dropna()
    
    rolling_corr_sp = {}
    for col in ['EEM', 'ASHR', 'INDY', 'EWZ']:
        if col in returns_all.columns:
            rolling_corr_sp[f'{col}_Corr_SPY'] = returns_all[col].rolling(252).corr(returns_all['SPY'])
    
    return rolling_vol, rolling_corr_sp


def EM_crisis_score(rolling_vol, rolling_corr, lookback=252):
    """
    Calculate crisis risk score:
    - High volatility (>25%)
    - High correlation (>0.85) with US stocks
    - Both indicate contagion/crisis risk
    """
    crisis_scores = {}
    
    for col in rolling_vol.columns:
        if col in ['EEM', 'ASHR', 'INDY', 'EWZ']:
            vol = rolling_vol[col]
            
            # Get correlation if available
            corr_col = f'{col}_Corr_SPY'
            if corr_col in rolling_corr:
                corr = rolling_corr[corr_col]
            else:
                corr = 0.75  # Default if not available
            
            # Crisis score: (vol - mean) / std + (corr - mean) / std
            vol_z = (vol - vol.mean()) / vol.std()
            corr_z = (corr - corr.mean()) / (corr.std() if corr.std() > 0 else 1)
            
            crisis_scores[f'{col}_CrisisScore'] = 0.5 * vol_z + 0.5 * corr_z
    
    return pd.DataFrame(crisis_scores)


def valuation_buy_sell_signals(prices, lookback=252):
    """
    Generate buy/sell signals based on:
    1. Price below 200-day MA (oversold)
    2. Price above 200-day MA + high volatility (overbought + risk)
    """
    signals = {}
    
    for col in prices.columns:
        ma_200 = prices[col].rolling(200).mean()
        vol_20 = prices[col].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Signal: -1 = strong sell, 0 = neutral, +1 = strong buy
        signal = np.zeros(len(prices))
        
        signal[prices[col] < 0.9 * ma_200] = 1  # Oversold: buy
        signal[prices[col] > 1.1 * ma_200] = -1  # Overbought: sell
        
        # Adjust by volatility regime
        signal[vol_20 > 0.30] *= 1.5  # High vol increases signal strength
        
        signals[f'{col}_Signal'] = signal
    
    return pd.DataFrame(signals, index=prices.index)

