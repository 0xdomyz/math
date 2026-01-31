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


class OptimalExecutionAlmgrenChriss(ExecutionAlgorithm):
    """Almgren-Chriss optimal execution (linear impact)"""
    
    def __init__(self, total_shares, T, lambda_risk=1e-6, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.lambda_risk = lambda_risk  # Risk aversion parameter
    
    def get_execution_path(self):
        """
        Optimal piecewise-linear execution path
        Solves: min E[IS] = γX² + τ∫x_t² dt + λ*σ²*∫x_t² dt
        """
        # Almgren-Chriss solution for linear impact
        # x_t = X * sqrt(kappa/T) * sinh(g*t) / sinh(g*T)
        # where g = sqrt(kappa) * sqrt(lambda_risk * sigma² / tau)
        
        # Simplified: use gradient descent optimization
        def objective(x_path):
            """Objective function to minimize"""
            # Ensure sum = total shares
            x_path = np.abs(x_path)  # Ensure positive
            x_path = x_path / np.sum(x_path) * self.total_shares
            
            X_cumsum = np.cumsum(x_path)  # Cumulative executed
            
            # Permanent impact cost: γ × (cumulative executed)²
            permanent_cost = self.gamma * np.sum(X_cumsum**2)
            
            # Temporary impact cost: τ × (sum of squared trades)
            temporary_cost = self.tau * np.sum(x_path**2)
            
            # Timing risk: λ × σ² × (sum of squared trades) × (time to completion)
            timing_risk = self.lambda_risk * (self.sigma**2) * np.sum(x_path**2 * (self.T - np.arange(self.T)))
            
            return permanent_cost + temporary_cost + timing_risk
        
        # Initial guess: uniform (TWAP)
        x0 = np.ones(self.T) * self.total_shares / self.T
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=[(0, self.total_shares) for _ in range(self.T)],
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_shares}
        )
        
        x_opt = result.x
        x_opt = np.maximum(x_opt, 0)  # Ensure non-negative
        x_opt = x_opt / np.sum(x_opt) * self.total_shares  # Re-normalize
        
        return x_opt
