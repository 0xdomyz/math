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


class PairsTradingStrategy:
    """Statistical arbitrage: Pairs trading"""
    
    def __init__(self, price_A, price_B, transaction_cost=0.001):
        self.price_A = price_A
        self.price_B = price_B
        self.tc = transaction_cost
        
    def calculate_spread(self, window=60):
        """Calculate spread and z-score"""
        # Hedge ratio from rolling regression
        # Simple version: Use rolling correlation and std ratio
        
        # Normalize prices
        norm_A = self.price_A / self.price_A.iloc[0]
        norm_B = self.price_B / self.price_B.iloc[0]
        
        # Spread (simplified: equal-weighted)
        spread = norm_A - norm_B
        
        # Z-score
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        
        z_score = (spread - spread_mean) / spread_std
        
        return spread, z_score
    
    def generate_signals(self, window=60, entry_z=2.0, exit_z=0.5):
        """
        Entry: |z| > entry_z
        Exit: |z| < exit_z
        """
        spread, z_score = self.calculate_spread(window)
        
        # Signals for the pair
        signals = pd.Series(0, index=z_score.index)
        
        # Long spread (long A, short B) when z < -entry_z
        signals[z_score < -entry_z] = 1
        
        # Short spread (short A, long B) when z > entry_z
        signals[z_score > entry_z] = -1
        
        # Exit when mean-reverts
        signals[abs(z_score) < exit_z] = 0
        
        # Forward-fill positions
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        # Calculate returns
        returns_A = np.log(self.price_A / self.price_A.shift(1))
        returns_B = np.log(self.price_B / self.price_B.shift(1))
        
        # Spread return: Long A, Short B (when signal = 1)
        spread_return = returns_A - returns_B
        
        positions = signals.shift(1).fillna(0)
        strategy_returns = positions * spread_return
        
        # Transaction costs
        position_changes = positions.diff().abs()
        costs = position_changes * self.tc * 2  # Two legs
        
        strategy_returns = strategy_returns - costs
        
        self.returns = strategy_returns.fillna(0)
        self.positions = positions
        self.z_score = z_score
        
        return signals
