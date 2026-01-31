import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.linalg import cholesky
import warnings

# Block 1
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EXOTIC OPTIONS PRICING")
print("="*60)

class ExoticOptionPricer:
    """Pricing engine for exotic options"""
    
    def __init__(self, S0, r, sigma, T):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
    
    def generate_path(self, n_steps):
        """Generate single GBM path"""
        dt = self.T / n_steps
        path = np.zeros(n_steps + 1)
        path[0] = self.S0
        
        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1)
            path[t] = path[t-1] * np.exp(
                (self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
            )
        
        return path
    
    def asian_arithmetic_mc(self, K, n_paths=50000, n_steps=252):
        """Price arithmetic average Asian option via Monte Carlo"""
        payoffs = []
        
        for _ in range(n_paths):
            path = self.generate_path(n_steps)
            avg_price = np.mean(path)
            payoff = max(avg_price - K, 0)
            payoffs.append(payoff)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def asian_geometric_closed_form(self, K):
        """Price geometric average Asian option (closed-form)"""
        # Adjusted parameters for geometric average
        sigma_geo = self.sigma / np.sqrt(3)
        r_geo = 0.5 * (self.r - 0.5*self.sigma**2 + sigma_geo**2)
        
        # Black-Scholes with adjusted parameters
        d1 = (np.log(self.S0/K) + (r_geo + 0.5*sigma_geo**2)*self.T) / (sigma_geo*np.sqrt(self.T))
        d2 = d1 - sigma_geo*np.sqrt(self.T)
        
        price = np.exp(-self.r*self.T) * (self.S0*np.exp(r_geo*self.T)*norm.cdf(d1) - K*norm.cdf(d2))
        
        return price
    
    def barrier_down_out_call(self, K, H, n_paths=50000, n_steps=252):
        """Price down-and-out barrier call"""
        payoffs = []
        
        for _ in range(n_paths):
            path = self.generate_path(n_steps)
            
            # Check if barrier breached
            if np.min(path) > H:  # Not knocked out
                payoff = max(path[-1] - K, 0)
            else:
                payoff = 0
            
            payoffs.append(payoff)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def lookback_floating_call(self, n_paths=50000, n_steps=252):
        """Price floating strike lookback call: S_T - min(S_t)"""
        payoffs = []
        
        for _ in range(n_paths):
            path = self.generate_path(n_steps)
            payoff = path[-1] - np.min(path)  # Always positive
            payoffs.append(payoff)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def digital_call(self, K, cash_payoff=1.0):
        """Price cash-or-nothing digital call"""
        d2 = (np.log(self.S0/K) + (self.r - 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        price = cash_payoff * np.exp(-self.r*self.T) * norm.cdf(d2)
        return price

class MultiAssetExotics:
    """Multi-asset exotic options"""
    
    def __init__(self, S0_list, r, sigma_list, corr_matrix, T):
        self.S0 = np.array(S0_list)
        self.r = r
        self.sigma = np.array(sigma_list)
        self.corr_matrix = corr_matrix
        self.T = T
        self.n_assets = len(S0_list)
        
        # Cholesky decomposition for correlation
        self.chol = cholesky(corr_matrix, lower=True)
    
    def generate_paths(self, n_paths):
        """Generate correlated terminal prices"""
        # Independent normal draws
        Z = np.random.normal(0, 1, (n_paths, self.n_assets))
        
        # Correlate
        Z_corr = Z @ self.chol.T
        
        # Terminal prices
        S_T = self.S0 * np.exp(
            (self.r - 0.5*self.sigma**2)*self.T + 
            self.sigma*np.sqrt(self.T)*Z_corr
        )
        
        return S_T
    
    def basket_option(self, K, weights, n_paths=50000):
        """Price basket call option"""
        S_T = self.generate_paths(n_paths)
        
        # Weighted basket value
        basket_values = S_T @ weights
        payoffs = np.maximum(basket_values - K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def best_of_call(self, K, n_paths=50000):
        """Price best-of (rainbow) call option"""
        S_T = self.generate_paths(n_paths)
        
        # Maximum of all assets
        best_prices = np.max(S_T, axis=1)
        payoffs = np.maximum(best_prices - K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def worst_of_put(self, K, n_paths=50000):
        """Price worst-of put option"""
        S_T = self.generate_paths(n_paths)
        
        # Minimum of all assets
        worst_prices = np.min(S_T, axis=1)
        payoffs = np.maximum(K - worst_prices, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def spread_option(self, K, n_paths=50000):
        """Price spread option on first two assets"""
        S_T = self.generate_paths(n_paths)
        
        # Spread: S1 - S2
        spreads = S_T[:, 0] - S_T[:, 1]
        payoffs = np.maximum(spreads - K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se

# Black-Scholes for vanilla comparison