import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, newton
import time
import warnings

# Block 1
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("IMPLIED VOLATILITY CALCULATION AND ANALYSIS")
print("="*60)

class BlackScholes:
    """Black-Scholes pricing and Greeks"""
    
    @staticmethod
    def d1(S, K, r, T, sigma):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, r, T, sigma):
        return BlackScholes.d1(S, K, r, T, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, r, T, sigma):
        d1 = BlackScholes.d1(S, K, r, T, sigma)
        d2 = BlackScholes.d2(S, K, r, T, sigma)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, r, T, sigma):
        d1 = BlackScholes.d1(S, K, r, T, sigma)
        d2 = BlackScholes.d2(S, K, r, T, sigma)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    @staticmethod
    def vega(S, K, r, T, sigma):
        d1 = BlackScholes.d1(S, K, r, T, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)

class ImpliedVolatility:
    """Implied volatility solvers"""
    
    @staticmethod
    def newton_raphson(S, K, r, T, market_price, option_type='call', 
                       initial_guess=0.2, max_iter=100, tol=1e-6):
        """Newton-Raphson method using Vega"""
        sigma = initial_guess
        
        for i in range(max_iter):
            if option_type == 'call':
                price = BlackScholes.call_price(S, K, r, T, sigma)
            else:
                price = BlackScholes.put_price(S, K, r, T, sigma)
            
            vega = BlackScholes.vega(S, K, r, T, sigma)
            
            diff = price - market_price
            
            if abs(diff) < tol:
                return sigma, i+1  # Converged
            
            if vega < 1e-10:  # Avoid division by very small number
                return np.nan, i+1
            
            sigma = sigma - diff / vega
            
            # Keep sigma positive and reasonable
            sigma = max(0.001, min(sigma, 5.0))
        
        return np.nan, max_iter  # Failed to converge
    
    @staticmethod
    def bisection(S, K, r, T, market_price, option_type='call', 
                  sigma_low=0.001, sigma_high=5.0, tol=1e-6, max_iter=100):
        """Bisection method - robust but slower"""
        
        for i in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            
            if option_type == 'call':
                price = BlackScholes.call_price(S, K, r, T, sigma_mid)
            else:
                price = BlackScholes.put_price(S, K, r, T, sigma_mid)
            
            if abs(price - market_price) < tol:
                return sigma_mid, i+1
            
            if price < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
            
            if sigma_high - sigma_low < tol:
                return sigma_mid, i+1
        
        return np.nan, max_iter
    
    @staticmethod
    def brenner_subrahmanyam(S, K, r, T, market_price):
        """Analytical approximation for ATM short-maturity"""
        return np.sqrt(2 * np.pi / T) * (market_price / S)
    
    @staticmethod
    def check_arbitrage_bounds(S, K, r, T, market_price, option_type='call'):
        """Check if price satisfies no-arbitrage bounds"""
        if option_type == 'call':
            lower_bound = max(S - K * np.exp(-r*T), 0)
            upper_bound = S
        else:
            lower_bound = max(K * np.exp(-r*T) - S, 0)
            upper_bound = K * np.exp(-r*T)
        
        return lower_bound <= market_price <= upper_bound

# Scenario 1: Basic IV calculation
print("\n" + "="*60)
print("SCENARIO 1: Basic Implied Volatility Calculation")
print("="*60)

S, K, r, T = 100, 100, 0.05, 0.25
true_sigma = 0.25

# Generate market price
market_call = BlackScholes.call_price(S, K, r, T, true_sigma)

print(f"\nParameters: S=${S}, K=${K}, r={r:.1%}, T={T}yr")
print(f"True volatility: {true_sigma:.1%}")
print(f"Market call price: ${market_call:.4f}")

# Solve using Newton-Raphson
iv_nr, iter_nr = ImpliedVolatility.newton_raphson(S, K, r, T, market_call, 'call')
print(f"\nNewton-Raphson:")
print(f"  Implied Vol: {iv_nr:.6f} ({iv_nr*100:.4f}%)")
print(f"  Iterations: {iter_nr}")
print(f"  Error: {abs(iv_nr - true_sigma):.8f}")

# Solve using Bisection
iv_bis, iter_bis = ImpliedVolatility.bisection(S, K, r, T, market_call, 'call')
print(f"\nBisection:")
print(f"  Implied Vol: {iv_bis:.6f} ({iv_bis*100:.4f}%)")
print(f"  Iterations: {iter_bis}")
print(f"  Error: {abs(iv_bis - true_sigma):.8f}")

# Analytical approximation
iv_approx = ImpliedVolatility.brenner_subrahmanyam(S, K, r, T, market_call)
print(f"\nBrenner-Subrahmanyam Approximation:")
print(f"  Implied Vol: {iv_approx:.6f} ({iv_approx*100:.4f}%)")
print(f"  Error: {abs(iv_approx - true_sigma):.8f}")

# Scenario 2: Performance comparison
print("\n" + "="*60)
print("SCENARIO 2: Performance Comparison")
print("="*60)

n_trials = 1000
strikes = np.random.uniform(80, 120, n_trials)
times = np.random.uniform(0.1, 2.0, n_trials)

print(f"\nComparing methods on {n_trials} random options:")

# Newton-Raphson timing
start = time.time()
for i in range(n_trials):
    market_price = BlackScholes.call_price(S, strikes[i], r, times[i], 0.25)
    iv, _ = ImpliedVolatility.newton_raphson(S, strikes[i], r, times[i], market_price, 'call')
nr_time = time.time() - start

# Bisection timing
start = time.time()
for i in range(n_trials):
    market_price = BlackScholes.call_price(S, strikes[i], r, times[i], 0.25)
    iv, _ = ImpliedVolatility.bisection(S, strikes[i], r, times[i], market_price, 'call')
bis_time = time.time() - start

print(f"\nNewton-Raphson: {nr_time*1000:.2f} ms ({nr_time/n_trials*1e6:.2f} μs per option)")
print(f"Bisection: {bis_time*1000:.2f} ms ({bis_time/n_trials*1e6:.2f} μs per option)")
print(f"Speedup: {bis_time/nr_time:.1f}x")

# Scenario 3: Volatility smile
print("\n" + "="*60)
print("SCENARIO 3: Volatility Smile Construction")
print("="*60)

S_smile = 100
T_smile = 0.25
r_smile = 0.05

# Strikes from deep OTM put to deep OTM call
strikes_smile = np.linspace(80, 120, 21)

# True volatility smile (stylized equity smile)