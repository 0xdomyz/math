
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def binomial_barrier_call(S0, K, L, r, sigma, T, N, barrier_type='knock_out'):
    """Barrier call option via binomial tree"""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    
    # Prices at terminal nodes
    V = np.zeros(N + 1)
    S = np.zeros(N + 1)
    
    for j in range(N + 1):
        S_temp = S0 * u**j * d**(N - j)
        S[j] = S_temp
        intrinsic = max(S_temp - K, 0)
        
        # Barrier check for terminal nodes
        if barrier_type == 'knock_out':
            if S_temp < L:  # Knockout occurred
                V[j] = 0
            else:
                V[j] = intrinsic
        elif barrier_type == 'knock_in':
            V[j] = intrinsic  # Assume already activated
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S_curr = S0 * u**j * d**(i - j)
            V_hold = (q * V[j + 1] + (1 - q) * V[j]) * np.exp(-r * dt)
            
            if barrier_type == 'knock_out':
                if S_curr < L:
                    V[j] = 0  # Already knocked out
                else:
                    V[j] = V_hold
            elif barrier_type == 'knock_in':
                V[j] = V_hold
    
    return V[0]

def monte_carlo_barrier_call(S0, K, L, r, sigma, T, M=10000, barrier_type='knock_out'):
    """Barrier call via Monte Carlo simulation"""
    dt = T / 252  # Daily steps
    n_steps = int(T / dt)
    
    payoffs = np.zeros(M)
    
    for path in range(M):
        S = S0
        barrier_hit = False
        
        for step in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            S = S * np.exp((r - 0.5*sigma**2) * dt + sigma * dW)
            
            if S < L:
                barrier_hit = True
        
        if barrier_type == 'knock_out':
            if not barrier_hit:
                payoff = max(S - K, 0)
            else:
                payoff = 0
        elif barrier_type == 'knock_in':
            if barrier_hit:
                payoff = max(S - K, 0)
            else:
                payoff = 0
        
        payoffs[path] = payoff * np.exp(-r * T)
    
    return payoffs.mean()

# Parameters
S0, K, L, r, sigma, T = 100, 100, 90, 0.05, 0.2, 1.0

# Compute values
ko_call_tree = binomial_barrier_call(S0, K, L, r, sigma, T, 100, 'knock_out')
ki_call_tree = binomial_barrier_call(S0, K, L, r, sigma, T, 100, 'knock_in')
ko_call_mc = monte_carlo_barrier_call(S0, K, L, r, sigma, T, 50000, 'knock_out')
ki_call_mc = monte_carlo_barrier_call(S0, K, L, r, sigma, T, 50000, 'knock_in')

# Vanilla for comparison
def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

vanilla_call = bs_call(S0, K, r, sigma, T)

print("="*60)
print("BARRIER OPTION VALUATION")
print("="*60)
print(f"S0=${S0}, K=${K}, Barrier L=${L}, T={T}yr, r={r*100:.1f}%, σ={sigma*100:.1f}%")
print("-"*60)
print(f"Vanilla Call: ${vanilla_call:.4f}")
print(f"Down-and-Out Call (Binomial): ${ko_call_tree:.4f}")
print(f"Down-and-In Call (Binomial): ${ki_call_tree:.4f}")
print(f"Down-and-Out Call (MC): ${ko_call_mc:.4f}")
print(f"Down-and-In Call (MC): ${ki_call_mc:.4f}")
print(f"Sum (DI + DO): ${ki_call_tree + ko_call_tree:.4f} (should ≈ Vanilla)")