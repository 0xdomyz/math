import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("TRADING SIGNALS AND STRATEGIES")
print("="*70)


class RSIMomentum(TradingStrategy):
    """RSI-based momentum strategy"""
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, period=14, oversold=30, overbought=70):
        """Buy when oversold, sell when overbought"""
        prices = self.prices['Close']
        
        rsi = self.calculate_rsi(prices, period)
        
        # Signals
        signals = pd.Series(0, index=prices.index)
        
        # Buy when RSI crosses above oversold
        signals[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1
        
        # Sell when RSI crosses below overbought  
        signals[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1
        
        # Forward-fill
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        self.signals = signals
        self.calculate_returns(signals)
        
        return signals
