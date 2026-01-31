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


class MovingAverageCrossover(TradingStrategy):
    """Trend-following: MA crossover strategy"""
    
    def generate_signals(self, fast=50, slow=200):
        """Golden cross / Death cross"""
        prices = self.prices['Close']
        
        # Moving averages
        ma_fast = prices.rolling(window=fast).mean()
        ma_slow = prices.rolling(window=slow).mean()
        
        # Signals: 1 when fast > slow, -1 when fast < slow
        signals = pd.Series(0, index=prices.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1
        
        self.signals = signals
        self.calculate_returns(signals)
        
        return signals
