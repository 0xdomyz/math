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


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution"""
    
    def get_execution_path(self, volumes=None):
        """Execute uniformly over time"""
        x = np.ones(self.T) * self.total_shares / self.T
        return x
