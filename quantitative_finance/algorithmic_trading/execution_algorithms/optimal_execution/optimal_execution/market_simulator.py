import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("OPTIMAL EXECUTION ALGORITHMS")
print("="*80)


class MarketSimulator:
    """Simulate market dynamics during execution"""
    
    def __init__(self, S0=100, sigma=0.02, dt=1/252/6.5/60):  # 1-minute bars
        self.S0 = S0
        self.sigma = sigma
        self.dt = dt
    
    def generate_path(self, n_steps, seed=None):
        """Generate GBM price path"""
        if seed is not None:
            np.random.seed(seed)
        
        returns = np.random.normal(0, self.sigma * np.sqrt(self.dt), n_steps)
        prices = self.S0 * np.exp(np.cumsum(returns))
        
        return prices
    
    def generate_volume_path(self, n_steps, avg_volume=1000, seasonality=True):
        """Generate intraday volume pattern"""
        t = np.arange(n_steps)
        
        if seasonality:
            # U-shaped volume pattern (high open/close, low midday)
            factor = 1.5 + 0.5 * np.sin(np.pi * t / n_steps)
        else:
            factor = np.ones(n_steps)
        
        volume = avg_volume * factor * np.random.gamma(2, 1, n_steps)
        
        return np.maximum(volume, 10)  # Min 10 shares
