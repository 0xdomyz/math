import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("BACKTESTING AND VALIDATION FRAMEWORK")
print("="*70)


class WalkForwardAnalysis:
    """Walk-forward optimization and validation"""
    
    def __init__(self, train_period=504, test_period=126, step=126):
        """
        train_period: Training window in days (504 = 2 years)
        test_period: Test window in days (126 = 6 months)
        step: Step size for rolling (126 = 6 months)
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
    
    def optimize_strategy(self, data, param_grid):
        """
        Find best parameters on training data
        Returns best params and their performance
        """
        best_sharpe = -np.inf
        best_params = None
        
        for params in param_grid:
            # Generate signals with these parameters
            signals = self.generate_signals(data, params)
            
            # Quick performance calc
            returns = signals.shift(1) * np.log(data / data.shift(1))
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return best_params, best_sharpe
    
    def generate_signals(self, prices, params):
        """Generate trading signals based on parameters"""
        # Example: MA crossover with parameters
        fast_period = params.get('fast', 50)
        slow_period = params.get('slow', 200)
        
        ma_fast = prices.rolling(window=fast_period).mean()
        ma_slow = prices.rolling(window=slow_period).mean()
        
        signals = pd.Series(0, index=prices.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1
        
        return signals
    
    def run_walk_forward(self, prices):
        """Execute walk-forward analysis"""
        results = []
        n = len(prices)
        
        start = 0
        while start + self.train_period + self.test_period <= n:
            # Define windows
            train_end = start + self.train_period
            test_end = train_end + self.test_period
            
            train_data = prices.iloc[start:train_end]
            test_data = prices.iloc[train_end:test_end]
            
            # Parameter grid (simplified)
            param_grid = [
                {'fast': 20, 'slow': 50},
                {'fast': 50, 'slow': 200},
                {'fast': 10, 'slow': 30}
            ]
            
            # Optimize on training data
            best_params, train_sharpe = self.optimize_strategy(train_data, param_grid)
            
            # Test on out-of-sample data
            test_signals = self.generate_signals(test_data, best_params)
            test_returns = test_signals.shift(1) * np.log(test_data / test_data.shift(1))
            test_sharpe = (test_returns.mean() * 252) / (test_returns.std() * np.sqrt(252))
            
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'test_returns': test_returns
            })
            
            # Roll forward
            start += self.step
        
        return results
