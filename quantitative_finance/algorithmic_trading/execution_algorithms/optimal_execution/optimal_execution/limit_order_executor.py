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


class LimitOrderExecutor(ExecutionAlgorithm):
    """Execution using limit orders at multiple levels"""
    
    def __init__(self, total_shares, T, limit_levels=5, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.limit_levels = limit_levels
    
    def get_execution_path(self, prices):
        """
        Execute via limit orders at different price levels
        Front-load with limit orders, use market orders for remainder
        """
        x_path = np.zeros(self.T)
        executed = 0
        
        for t in range(self.T):
            remaining = self.total_shares - executed
            
            if remaining <= 0:
                break
            
            # Probability of limit order fill decreases with depth
            # Allocate: 20% limit (best bid - 1bp), 30% limit (- 2bp), 30% limit (- 3bp), 20% market
            fill_probs = [0.8, 0.5, 0.2, 0.05, 1.0]
            
            for level, prob in enumerate(fill_probs):
                if level < self.limit_levels:
                    size = remaining * (1 - level * 0.15)
                else:
                    size = remaining
                
                if np.random.random() < prob:
                    x_path[t] += size
                    executed += size
                    break
        
        # Ensure total is hit (force market orders if needed)
        if executed < self.total_shares:
            x_path[self.T - 1] += self.total_shares - executed
        
        return np.minimum(x_path, self.total_shares)
