import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky
import time
import warnings

# Block 1
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("MONTE CARLO OPTION PRICING")
print("="*60)

class MonteCarloPricer:
    """Monte Carlo option pricing with variance reduction"""
    
    def __init__(self, S0, K, r, T, sigma, option_type='call', n_paths=10000, n_steps=252):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps
    
    def generate_paths(self, antithetic=False):
        """Generate stock price paths using GBM"""
        # Random normal draws
        if antithetic:
            Z = np.random.normal(0, 1, (self.n_paths // 2, self.n_steps))
            Z = np.vstack([Z, -Z])  # Antithetic pairs
        else:
            Z = np.random.normal(0, 1, (self.n_paths, self.n_steps))
        
        # Initialize paths
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate paths
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.r - 0.5*self.sigma**2)*self.dt + 
                self.sigma*np.sqrt(self.dt)*Z[:, t-1]
            )
        
        return paths
    
    def price_european(self, antithetic=False):
        """Price European option"""
        paths = self.generate_paths(antithetic)
        S_T = paths[:, -1]
        
        if self.option_type == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)
        
        # Discount and average
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se, payoffs
    
    def price_asian(self, average_type='arithmetic', antithetic=False):
        """Price Asian option"""
        paths = self.generate_paths(antithetic)
        
        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        if self.option_type == 'call':
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se
    
    def price_barrier(self, barrier, barrier_type='down-and-out', antithetic=False):
        """Price barrier option"""
        paths = self.generate_paths(antithetic)
        S_T = paths[:, -1]
        
        # Check barrier condition
        if barrier_type == 'down-and-out':
            knocked = np.any(paths <= barrier, axis=1)
        elif barrier_type == 'up-and-out':
            knocked = np.any(paths >= barrier, axis=1)
        elif barrier_type == 'down-and-in':
            knocked = ~np.any(paths <= barrier, axis=1)
        else:  # up-and-in
            knocked = ~np.any(paths >= barrier, axis=1)
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)
        
        # Apply barrier condition
        if 'out' in barrier_type:
            payoffs = payoffs * (~knocked)
        else:  # in
            payoffs = payoffs * knocked
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se
    
    def price_with_control_variate(self):
        """Price using European option as control variate"""
        paths = self.generate_paths(antithetic=False)
        S_T = paths[:, -1]
        
        # Target: Asian option
        avg_prices = np.mean(paths, axis=1)
        asian_payoffs = np.maximum(avg_prices - self.K, 0) if self.option_type == 'call' \
                        else np.maximum(self.K - avg_prices, 0)
        
        # Control: European option (known analytical price)
        euro_payoffs = np.maximum(S_T - self.K, 0) if self.option_type == 'call' \
                       else np.maximum(self.K - S_T, 0)
        
        # Analytical European price
        d1 = (np.log(self.S0/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        if self.option_type == 'call':
            euro_analytical = self.S0*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            euro_analytical = self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - self.S0*norm.cdf(-d1)
        
        # Optimal beta
        cov = np.cov(asian_payoffs, euro_payoffs)[0, 1]
        var = np.var(euro_payoffs)
        beta = cov / var if var > 0 else 0
        
        # Control variate estimate
        euro_mc = np.mean(euro_payoffs)
        asian_cv = asian_payoffs - beta * (euro_payoffs - euro_analytical * np.exp(self.r*self.T))
        
        price = np.exp(-self.r*self.T) * np.mean(asian_cv)
        se = np.exp(-self.r*self.T) * np.std(asian_cv) / np.sqrt(self.n_paths)
        
        # Variance reduction factor
        var_ratio = np.var(asian_cv) / np.var(asian_payoffs) if np.var(asian_payoffs) > 0 else 1
        
        return price, se, beta, var_ratio
