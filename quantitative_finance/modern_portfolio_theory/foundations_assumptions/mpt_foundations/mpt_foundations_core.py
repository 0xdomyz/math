import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import norm, t
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("MODERN PORTFOLIO THEORY: FOUNDATIONS AND ASSUMPTIONS")
print("="*80)

class PortfolioOptimizer:
    """Core portfolio optimization tools"""
    
    def __init__(self, returns_df):
        """
        Initialize with returns DataFrame
        rows: dates, columns: assets
        """
        self.returns = returns_df
        self.n_assets = returns_df.shape[1]
        self.asset_names = returns_df.columns.tolist()
        
        # Compute statistics
        self.mu = returns_df.mean()
        self.sigma = returns_df.std()
        self.cov_matrix = returns_df.cov()
        self.corr_matrix = returns_df.corr()
        self.returns_annual = self.mu * 252
        self.sigma_annual = self.sigma * np.sqrt(252)
    
    def min_variance_portfolio(self, constrain_short_sales=True):
        """Minimum variance portfolio"""
        def objective(w):
            return np.sqrt(w @ self.cov_matrix @ w)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        if constrain_short_sales:
            bounds = Bounds(0, 1)
        else:
            bounds = Bounds(-10, 10)
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets)/self.n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def max_sharpe_portfolio(self, r_f=0.02, constrain_short_sales=True):
        """Maximum Sharpe ratio portfolio"""
        def neg_sharpe(w):
            ret = w @ self.mu * 252
            vol = np.sqrt(w @ self.cov_matrix @ w * 252)
            return -(ret - r_f) / vol
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        if constrain_short_sales:
            bounds = Bounds(0, 1)
        else:
            bounds = Bounds(-10, 10)
        
        result = minimize(
            neg_sharpe,
            x0=np.ones(self.n_assets)/self.n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def efficient_frontier(self, target_returns=None, constrain_short_sales=True, r_f=0.02):
        """Compute efficient frontier"""
        if target_returns is None:
            target_returns = np.linspace(self.returns_annual.min(), self.returns_annual.max(), 50)
        
        frontier_vols = []
        frontier_rets = []
        frontier_weights = []
        
        for target_ret in target_returns:
            def objective(w):
                return np.sqrt(w @ self.cov_matrix @ w)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w @ self.mu * 252 - target_ret}
            ]
            
            if constrain_short_sales:
                bounds = Bounds(0, 1)
            else:
                bounds = Bounds(-10, 10)
            
            try:
                result = minimize(
                    objective,
                    x0=np.ones(self.n_assets)/self.n_assets,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-9}
                )
                
                if result.success:
                    vol = np.sqrt(result.x @ self.cov_matrix @ result.x) * np.sqrt(252)
                    frontier_vols.append(vol)
                    frontier_rets.append(target_ret)
                    frontier_weights.append(result.x)
            except:
                continue
        
        return {
            'returns': frontier_rets,
            'volatilities': frontier_vols,
            'weights': frontier_weights
        }
    
    def portfolio_stats(self, weights):
        """Compute portfolio statistics"""
        ret = weights @ self.mu * 252
        vol = np.sqrt(weights @ self.cov_matrix @ weights * 252)
        sharpe = (ret - 0.02) / vol if vol > 0 else 0
        
        return {
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        }
