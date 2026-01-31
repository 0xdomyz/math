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


class BollingerBandsMeanReversion(TradingStrategy):
    """Mean reversion: Bollinger Bands"""
    
    def generate_signals(self, window=20, num_std=2):
        """Buy at lower band, sell at upper band"""
        prices = self.prices['Close']
        
        # Bollinger Bands
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = ma + num_std * std
        lower_band = ma - num_std * std
        
        # Signals
        signals = pd.Series(0, index=prices.index)
        
        # Buy when price touches lower band
        signals[prices < lower_band] = 1
        
        # Sell when price touches upper band
        signals[prices > upper_band] = -1
        
        # Exit when price crosses MA
        signals[prices > ma] = signals[prices > ma].where(signals.shift(1) == 1, 0)
        signals[prices < ma] = signals[prices < ma].where(signals.shift(1) == -1, 0)
        
        # Forward-fill signals (maintain position)
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        self.signals = signals
        self.calculate_returns(signals)
        
        return signals
