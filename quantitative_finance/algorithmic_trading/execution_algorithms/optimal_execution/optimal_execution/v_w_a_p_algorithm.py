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


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution"""
    
    def __init__(self, total_shares, T, pov=0.15, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.pov = pov  # Participation of volume
    
    def get_execution_path(self, volumes):
        """Execute proportional to market volume"""
        if volumes is None:
            volumes = np.ones(self.T) * np.mean(volumes or 1000)
        
        # Start with POV-based execution
        x = self.pov * volumes
        
        # Ensure total execution matches target
        remaining = self.total_shares - np.sum(x)
        if remaining > 0:
            # Catch up in later periods
            scale = self.total_shares / np.sum(x)
            x = scale * x
        
        return np.minimum(x, self.total_shares)  # Cap at total shares
