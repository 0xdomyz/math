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


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, prices, transaction_cost=0.001):
        """
        prices: DataFrame with OHLCV data
        transaction_cost: Round-trip cost as fraction (0.001 = 10 bps)
        """
        self.prices = prices
        self.tc = transaction_cost
        self.signals = None
        self.positions = None
        self.returns = None
        
    def calculate_returns(self, signals):
        """Calculate strategy returns from signals"""
        # Returns: Log returns
        returns = np.log(self.prices['Close'] / self.prices['Close'].shift(1))
        
        # Position: 1 (long), -1 (short), 0 (flat)
        positions = signals.shift(1)  # Trade on next day
        
        # Strategy returns
        strategy_returns = positions * returns
        
        # Transaction costs: Apply when position changes
        position_changes = positions.diff().abs()
        costs = position_changes * self.tc
        
        strategy_returns = strategy_returns - costs
        
        self.returns = strategy_returns.fillna(0)
        self.positions = positions
        
        return strategy_returns
    
    def performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.returns is None:
            return {}
        
        # Remove NaN
        returns = self.returns.dropna()
        
        # Total return
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized metrics (assuming daily data, 252 trading days)
        n_days = len(returns)
        n_years = n_days / 252
        
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming Rf = 0 for simplicity)
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'Num Trades': (self.positions.diff() != 0).sum()
        }
