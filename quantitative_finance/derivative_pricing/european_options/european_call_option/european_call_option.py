
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes analytical formula for European call
def black_scholes_call(S0, K, T, r, sigma):
    """
    Analytical European call option price.
    
    Parameters:
    - S0: Current stock price
    - K: Strike price
    - T: Time to maturity (years)
    - r: Risk-free rate
    - sigma: Volatility
    
    Returns:
    - call_price: Option value
    - delta: First derivative w.r.t. S
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    
    return call_price, delta

# Monte Carlo European call pricing
def monte_carlo_call(S0, K, T, r, sigma, n_paths, antithetic=False):
    """
    Monte Carlo simulation for European call option.
    
    Parameters:
    - antithetic: If True, use antithetic variance reduction
    
    Returns:
    - call_price: Estimated option value
    - std_error: Standard error of estimate
    - terminal_prices: Array of simulated S_T values
    """
    if antithetic:
        # Generate half paths, use Z and -Z
        n_half = n_paths // 2
        Z = np.random.randn(n_half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = np.random.randn(n_paths)
    
    # GBM: S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z_full
    terminal_prices = S0 * np.exp(drift + diffusion)
    
    # Payoff: max(S_T - K, 0)
    payoffs = np.maximum(terminal_prices - K, 0)
    
    # Discounted expected payoff
    call_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return call_price, std_error, terminal_prices

# Parameters
S0 = 100.0      # Current stock price
K = 105.0       # Strike price (slightly OTM)
T = 1.0         # 1 year to maturity
r = 0.05        # 5% risk-free rate
sigma = 0.20    # 20% volatility

# Analytical Black-Scholes price
bs_price, bs_delta = black_scholes_call(S0, K, T, r, sigma)
print(f"Black-Scholes Price: ${bs_price:.4f}")
print(f"Black-Scholes Delta: {bs_delta:.4f}")

# Monte Carlo convergence analysis
path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
mc_prices = []
mc_errors = []
mc_prices_av = []  # Antithetic variates
mc_errors_av = []

np.random.seed(42)
for n in path_counts:
    # Standard MC
    price, error, _ = monte_carlo_call(S0, K, T, r, sigma, n, antithetic=False)
    mc_prices.append(price)
    mc_errors.append(error)
    
    # Antithetic variates
    price_av, error_av, _ = monte_carlo_call(S0, K, T, r, sigma, n, antithetic=True)
    mc_prices_av.append(price_av)
    mc_errors_av.append(error_av)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Convergence of MC price to BS price
ax = axes[0, 0]
ax.semilogx(path_counts, mc_prices, 'o-', label='Standard MC', linewidth=2)
ax.semilogx(path_counts, mc_prices_av, 's-', label='Antithetic Variates', linewidth=2)
ax.axhline(bs_price, color='red', linestyle='--', label=f'Black-Scholes: ${bs_price:.4f}')
ax.fill_between(path_counts, 
                np.array(mc_prices) - 1.96*np.array(mc_errors),
                np.array(mc_prices) + 1.96*np.array(mc_errors),
                alpha=0.3)
ax.set_xlabel('Number of Paths')
ax.set_ylabel('Option Price ($)')
ax.set_title('European Call Price Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Standard error vs number of paths
ax = axes[0, 1]
ax.loglog(path_counts, mc_errors, 'o-', label='Standard MC', linewidth=2)
ax.loglog(path_counts, mc_errors_av, 's-', label='Antithetic Variates', linewidth=2)
# Theoretical O(1/sqrt(N)) line
theoretical_error = mc_errors[0] * np.sqrt(path_counts[0]) / np.sqrt(path_counts)
ax.loglog(path_counts, theoretical_error, 'k--', label='O(1/√N)', linewidth=1)
ax.set_xlabel('Number of Paths')
ax.set_ylabel('Standard Error ($)')
ax.set_title('Monte Carlo Standard Error')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Distribution of terminal stock prices
np.random.seed(42)
_, _, terminal_prices = monte_carlo_call(S0, K, T, r, sigma, 10000, antithetic=False)
ax = axes[1, 0]
ax.hist(terminal_prices, bins=50, density=True, alpha=0.7, edgecolor='black')
ax.axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.axvline(S0, color='green', linestyle='--', linewidth=2, label=f'Spot S₀=${S0}')
ax.axvline(np.mean(terminal_prices), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean S_T=${np.mean(terminal_prices):.2f}')
ax.set_xlabel('Terminal Stock Price S_T')
ax.set_ylabel('Density')
ax.set_title('Distribution of Terminal Prices (10k paths)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Payoff distribution
payoffs = np.maximum(terminal_prices - K, 0)
ax = axes[1, 1]
ax.hist(payoffs, bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(np.mean(payoffs) * np.exp(-r*T), color='red', linestyle='--', linewidth=2,
           label=f'PV Mean Payoff: ${np.mean(payoffs)*np.exp(-r*T):.4f}')
ax.axvline(bs_price, color='blue', linestyle='--', linewidth=2,
           label=f'BS Price: ${bs_price:.4f}')
ax.set_xlabel('Call Payoff at Maturity')
ax.set_ylabel('Density')
ax.set_title('Distribution of Call Payoffs')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('european_call_monte_carlo.png', dpi=300, bbox_inches='tight')
plt.show()

# Error analysis
print("\n" + "="*60)
print("CONVERGENCE ANALYSIS")
print("="*60)
print(f"{'Paths':<10} {'MC Price':<12} {'Error':<10} {'AV Price':<12} {'AV Error':<10}")
print("-"*60)
for i, n in enumerate(path_counts):
    print(f"{n:<10} ${mc_prices[i]:<11.4f} ${mc_errors[i]:<9.4f} "
          f"${mc_prices_av[i]:<11.4f} ${mc_errors_av[i]:<9.4f}")

# Variance reduction effectiveness
var_reduction = 1 - (np.array(mc_errors_av)**2) / (np.array(mc_errors)**2)
print(f"\nVariance Reduction (Antithetic): {np.mean(var_reduction)*100:.1f}%")