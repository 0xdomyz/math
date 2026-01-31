
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes closed-form benchmark
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

# Parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
true_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Black-Scholes Call Price: ${true_price:.4f}")

# Monte Carlo without antithetic
def mc_european_call_standard(S0, K, T, r, sigma, N_paths):
    dt = T / 252  # Daily steps
    n_steps = int(T / dt)
    
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    S = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + 
                               sigma*np.sqrt(dt)*Z, axis=1))
    
    payoffs = np.maximum(S[:, -1] - K, 0)
    price = np.exp(-r*T) * np.mean(payoffs)
    se = np.exp(-r*T) * np.std(payoffs) / np.sqrt(N_paths)
    
    return price, se

# Monte Carlo with antithetic
def mc_european_call_antithetic(S0, K, T, r, sigma, N_paths):
    dt = T / 252
    n_steps = int(T / dt)
    
    # Generate N_paths/2 unique random matrices, then create pairs
    N_unique = N_paths // 2
    Z = np.random.normal(0, 1, (N_unique, n_steps))
    
    # Paths with Z
    S_pos = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + 
                                   sigma*np.sqrt(dt)*Z, axis=1))
    payoff_pos = np.maximum(S_pos[:, -1] - K, 0)
    
    # Paths with -Z (antithetic)
    S_neg = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + 
                                   sigma*np.sqrt(dt)*(-Z), axis=1))
    payoff_neg = np.maximum(S_neg[:, -1] - K, 0)
    
    # Average pairs
    payoffs = (payoff_pos + payoff_neg) / 2
    price = np.exp(-r*T) * np.mean(payoffs)
    se = np.exp(-r*T) * np.std(payoffs) / np.sqrt(N_unique)
    
    return price, se

# Comparison across different sample sizes
sample_sizes = np.array([100, 500, 1000, 5000, 10000, 50000])
standard_prices = []
standard_ses = []
antithetic_prices = []
antithetic_ses = []

np.random.seed(42)
for N in sample_sizes:
    p_std, se_std = mc_european_call_standard(S0, K, T, r, sigma, N)
    standard_prices.append(p_std)
    standard_ses.append(se_std)
    
    np.random.seed(42)
    p_anti, se_anti = mc_european_call_antithetic(S0, K, T, r, sigma, N)
    antithetic_prices.append(p_anti)
    antithetic_ses.append(se_anti)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price convergence
axes[0, 0].plot(sample_sizes, standard_prices, 'o-', label='Standard MC', linewidth=2, markersize=8)
axes[0, 0].plot(sample_sizes, antithetic_prices, 's-', label='Antithetic MC', linewidth=2, markersize=8)
axes[0, 0].axhline(true_price, color='r', linestyle='--', linewidth=2, label='True Price (BS)')
axes[0, 0].set_xscale('log')
axes[0, 0].set_title('Price Convergence')
axes[0, 0].set_xlabel('Number of Paths')
axes[0, 0].set_ylabel('Call Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Standard Error comparison
axes[0, 1].plot(sample_sizes, standard_ses, 'o-', label='Standard MC', linewidth=2, markersize=8)
axes[0, 1].plot(sample_sizes, antithetic_ses, 's-', label='Antithetic MC', linewidth=2, markersize=8)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('Standard Error')
axes[0, 1].set_xlabel('Number of Paths')
axes[0, 1].set_ylabel('Standard Error ($)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Variance reduction ratio
variance_reduction = np.array(standard_ses) / np.array(antithetic_ses)
axes[1, 0].plot(sample_sizes, variance_reduction, 'o-', linewidth=2, markersize=8, color='green')
axes[1, 0].axhline(1.0, color='r', linestyle='--', label='Baseline (1.0)')
axes[1, 0].set_xscale('log')
axes[1, 0].set_title('Variance Reduction Factor')
axes[1, 0].set_xlabel('Number of Paths')
axes[1, 0].set_ylabel('SE(Standard) / SE(Antithetic)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Correlation between paired payoffs
N = 5000
dt = T / 252
n_steps = int(T / dt)
Z = np.random.normal(0, 1, (N, n_steps))

S_pos = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
S_neg = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-Z), axis=1))

payoff_pos = np.maximum(S_pos[:, -1] - K, 0)
payoff_neg = np.maximum(S_neg[:, -1] - K, 0)

correlation = np.corrcoef(payoff_pos, payoff_neg)[0, 1]
axes[1, 1].scatter(payoff_pos, payoff_neg, alpha=0.3, s=10)
axes[1, 1].set_title(f'Payoff Correlation (ρ = {correlation:.3f})')
axes[1, 1].set_xlabel('Payoff(Z)')
axes[1, 1].set_ylabel('Payoff(-Z)')
axes[1, 1].grid(alpha=0.3)

# Add diagonal reference
max_payoff = max(payoff_pos.max(), payoff_neg.max())
axes[1, 1].plot([0, max_payoff], [0, max_payoff], 'r--', alpha=0.5, label='y=x (if identical)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"\nMonte Carlo Results (N=10,000):")
print(f"Standard MC:    ${standard_prices[-2]:.4f} ± ${standard_ses[-2]:.4f}")
print(f"Antithetic MC:  ${antithetic_prices[-2]:.4f} ± ${antithetic_ses[-2]:.4f}")
print(f"Variance Reduction: {(standard_ses[-2]/antithetic_ses[-2]):.2f}x")
print(f"Correlation between pairs: {correlation:.4f}")