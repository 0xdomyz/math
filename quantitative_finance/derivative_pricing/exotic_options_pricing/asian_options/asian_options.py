
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

def bs_call(S, K, r, sigma, T):
    """Black-Scholes call"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def asian_geometric_call(S0, K, r, sigma, T):
    """Geometric average Asian call (closed-form)"""
    # Effective volatility: σ_G = σ/√3
    sigma_adj = sigma / np.sqrt(3)
    # Effective drift: adjusted for averaging
    r_adj = 0.5 * (r - sigma**2/6)
    
    # Use BS with adjusted parameters
    d1 = (np.log(S0/K) + (r_adj + 0.5*sigma_adj**2)*T) / (sigma_adj*np.sqrt(T))
    d2 = d1 - sigma_adj*np.sqrt(T)
    
    C_geom = S0*np.exp(-sigma**2*T/6)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return C_geom

def asian_arithmetic_call_mc(S0, K, r, sigma, T, N_paths=50000, N_steps=252):
    """Arithmetic average Asian call (Monte Carlo)"""
    dt = T / N_steps
    
    payoffs = []
    for path in range(N_paths):
        S = S0
        prices = [S0]
        
        for step in range(N_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            S = S * np.exp((r - 0.5*sigma**2)*dt + sigma*dW)
            prices.append(S)
        
        # Arithmetic average
        A = np.mean(prices)
        payoff = max(A - K, 0) * np.exp(-r*T)
        payoffs.append(payoff)
    
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(N_paths)

# Parameters
S0, K, r, sigma, T = 100, 100, 0.05, 0.25, 1.0

# Compute values
vanilla_call = bs_call(S0, K, r, sigma, T)
asian_geom = asian_geometric_call(S0, K, r, sigma, T)
asian_arith_mc, asian_arith_se = asian_arithmetic_call_mc(S0, K, r, sigma, T, 
                                                           N_paths=50000, N_steps=252)

print("="*70)
print("ASIAN OPTION PRICING COMPARISON")
print("="*70)
print(f"S0=${S0}, K=${K}, r={r*100:.1f}%, σ={sigma*100:.1f}%, T={T}yr")
print("-"*70)
print(f"European Vanilla Call: ${vanilla_call:.4f}")
print(f"Asian Geometric (Closed-Form): ${asian_geom:.4f}")
print(f"Asian Arithmetic (Monte Carlo): ${asian_arith_mc:.4f} ± ${asian_arith_se:.4f}")
print(f"\nCost Reduction (Arithmetic vs Vanilla): {100*(1-asian_arith_mc/vanilla_call):.1f}%")

# Spot price sensitivity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

S_range = np.linspace(70, 130, 30)

vanilla_vals = [bs_call(S, K, r, sigma, T) for S in S_range]
asian_geom_vals = [asian_geometric_call(S, K, r, sigma, T) for S in S_range]
asian_arith_vals = []
for S in S_range:
    val, _ = asian_arithmetic_call_mc(S, K, r, sigma, T, N_paths=10000, N_steps=50)
    asian_arith_vals.append(val)

axes[0].plot(S_range, vanilla_vals, 'b-', linewidth=2.5, label='Vanilla')
axes[0].plot(S_range, asian_geom_vals, 'g--', linewidth=2, label='Asian Geometric')
axes[0].plot(S_range, asian_arith_vals, 'r.', markersize=6, label='Asian Arithmetic (MC)')
axes[0].set_title('Option Value vs Spot Price')
axes[0].set_xlabel('Stock Price ($)')
axes[0].set_ylabel('Option Value ($)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Premium reduction
premiums = np.array(vanilla_vals) - np.array(asian_arith_vals)
axes[1].plot(S_range, premiums, 'mo-', linewidth=2, markersize=6)
axes[1].fill_between(S_range, 0, premiums, alpha=0.2)
axes[1].set_title('Asian Premium: (Vanilla - Arithmetic Asian)')
axes[1].set_xlabel('Stock Price ($)')
axes[1].set_ylabel('Premium Reduction ($)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()