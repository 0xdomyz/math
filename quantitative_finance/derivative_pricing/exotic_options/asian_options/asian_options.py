
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes formula for geometric Asian option
def geometric_asian_call(S0, K, T, r, sigma, n_steps):
    """
    Closed-form price for geometric average Asian call.
    
    Under GBM, geometric average is lognormally distributed.
    """
    # Adjusted parameters for geometric average
    sigma_g = sigma / np.sqrt(3)
    r_g = 0.5 * (r - 0.5 * sigma**2 + sigma**2 / 6)
    
    # Use Black-Scholes with adjusted parameters
    d1 = (np.log(S0 / K) + (r_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)
    
    call_price = S0 * np.exp((r_g - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

# European call (comparison)
def european_call(S0, K, T, r, sigma):
    """Standard Black-Scholes European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo Asian option pricing
def mc_asian_call(S0, K, T, r, sigma, n_paths, n_steps, use_control=False, antithetic=False):
    """
    Monte Carlo pricing for arithmetic average Asian call.
    
    Parameters:
    - use_control: If True, use geometric Asian as control variate
    - antithetic: If True, use antithetic variates
    
    Returns:
    - price: Option value
    - std_error: Standard error
    - arithmetic_avgs: Array of arithmetic averages
    - geometric_avgs: Array of geometric averages (if use_control=True)
    """
    dt = T / n_steps
    
    # Generate random numbers
    if antithetic:
        n_half = n_paths // 2
        Z = np.random.randn(n_half, n_steps)
        Z = np.vstack([Z, -Z])
    else:
        Z = np.random.randn(n_paths, n_steps)
    
    # Initialize paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate paths
    for t in range(n_steps):
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt 
                                              + sigma * np.sqrt(dt) * Z[:, t])
    
    # Compute averages (exclude initial S0, use only S_1, ..., S_n)
    arithmetic_avgs = np.mean(paths[:, 1:], axis=1)
    
    # Payoffs
    arithmetic_payoffs = np.maximum(arithmetic_avgs - K, 0)
    
    if use_control:
        # Geometric average: exp(mean of log prices)
        geometric_avgs = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        geometric_payoffs = np.maximum(geometric_avgs - K, 0)
        
        # Known expected value from closed-form
        geo_expected = geometric_asian_call(S0, K, T, r, sigma, n_steps)
        
        # Control variate adjustment
        # Find optimal beta via covariance
        cov = np.cov(arithmetic_payoffs, geometric_payoffs)[0, 1]
        var_geo = np.var(geometric_payoffs)
        beta = cov / var_geo if var_geo > 0 else 0
        
        # Adjusted payoffs
        adjusted_payoffs = arithmetic_payoffs - beta * (geometric_payoffs - geo_expected / np.exp(-r * T))
        
        price = np.exp(-r * T) * np.mean(adjusted_payoffs)
        std_error = np.exp(-r * T) * np.std(adjusted_payoffs) / np.sqrt(n_paths)
        
        return price, std_error, arithmetic_avgs, geometric_avgs, beta
    else:
        price = np.exp(-r * T) * np.mean(arithmetic_payoffs)
        std_error = np.exp(-r * T) * np.std(arithmetic_payoffs) / np.sqrt(n_paths)
        
        return price, std_error, arithmetic_avgs, None, 0

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.30
n_steps = 252  # Daily monitoring

print("="*80)
print("ASIAN OPTION PRICING")
print("="*80)
print(f"Parameters: S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%")
print(f"Monitoring: {n_steps} observations\n")

# Closed-form geometric Asian
geo_call = geometric_asian_call(S0, K, T, r, sigma, n_steps)
print(f"Geometric Asian Call (Closed-Form): ${geo_call:.6f}")

# European call (upper bound)
euro_call = european_call(S0, K, T, r, sigma)
print(f"European Call (Black-Scholes):     ${euro_call:.6f}")

# Monte Carlo arithmetic Asian
np.random.seed(42)
n_paths = 10000

# Standard MC
price_standard, error_standard, arith_avgs, _, _ = mc_asian_call(
    S0, K, T, r, sigma, n_paths, n_steps, use_control=False, antithetic=False
)
print(f"\nArithmetic Asian Call (Standard MC):    ${price_standard:.6f} ± ${error_standard:.6f}")

# Antithetic variates
price_av, error_av, _, _, _ = mc_asian_call(
    S0, K, T, r, sigma, n_paths, n_steps, use_control=False, antithetic=True
)
print(f"Arithmetic Asian Call (Antithetic):      ${price_av:.6f} ± ${error_av:.6f}")

# Control variate (geometric Asian)
price_cv, error_cv, _, geo_avgs, beta = mc_asian_call(
    S0, K, T, r, sigma, n_paths, n_steps, use_control=True, antithetic=False
)
print(f"Arithmetic Asian Call (Control Variate): ${price_cv:.6f} ± ${error_cv:.6f}")
print(f"  Control Variate β: {beta:.4f}")

# Variance reduction effectiveness
var_reduction_av = 1 - (error_av / error_standard)**2
var_reduction_cv = 1 - (error_cv / error_standard)**2
print(f"\nVariance Reduction:")
print(f"  Antithetic Variates: {var_reduction_av*100:.1f}%")
print(f"  Control Variate: {var_reduction_cv*100:.1f}%")

# Convergence analysis
print("\n" + "="*80)
print("CONVERGENCE ANALYSIS (with Control Variate)")
print("="*80)

path_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000]
prices_cv = []
errors_cv = []

for n in path_counts:
    np.random.seed(42)
    p, e, _, _, _ = mc_asian_call(S0, K, T, r, sigma, n, n_steps, use_control=True)
    prices_cv.append(p)
    errors_cv.append(e)
    print(f"N={n:>6}: ${p:.6f} ± ${e:.6f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample paths with averages
np.random.seed(42)
n_plot = 10
time_grid = np.linspace(0, T, n_steps + 1)

ax = axes[0, 0]
for _ in range(n_plot):
    Z = np.random.randn(n_steps)
    path = np.zeros(n_steps + 1)
    path[0] = S0
    for t in range(n_steps):
        path[t+1] = path[t] * np.exp((r - 0.5 * sigma**2) * (T/n_steps) 
                                      + sigma * np.sqrt(T/n_steps) * Z[t])
    
    ax.plot(time_grid, path, 'b-', alpha=0.4, linewidth=1)
    
    # Running average
    running_avg = np.cumsum(path[1:]) / np.arange(1, n_steps + 1)
    ax.plot(time_grid[1:], running_avg, 'r-', alpha=0.6, linewidth=1.5)

ax.axhline(K, color='green', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Price / Average')
ax.set_title('Sample Paths (Blue) with Running Average (Red)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution of averages (arithmetic vs geometric)
ax = axes[0, 1]
ax.hist(arith_avgs, bins=50, alpha=0.6, color='blue', edgecolor='black', 
        density=True, label='Arithmetic Avg')
if geo_avgs is not None:
    ax.hist(geo_avgs, bins=50, alpha=0.6, color='red', edgecolor='black',
            density=True, label='Geometric Avg')
ax.axvline(K, color='green', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.axvline(np.mean(arith_avgs), color='blue', linestyle=':', linewidth=2, 
           label=f'Mean Arith={np.mean(arith_avgs):.2f}')
if geo_avgs is not None:
    ax.axvline(np.mean(geo_avgs), color='red', linestyle=':', linewidth=2,
               label=f'Mean Geo={np.mean(geo_avgs):.2f}')
ax.set_xlabel('Average Price')
ax.set_ylabel('Density')
ax.set_title('Distribution of Averages (10k paths)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Payoff distribution
arith_payoffs = np.maximum(arith_avgs - K, 0)
ax = axes[0, 2]
ax.hist(arith_payoffs, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(np.mean(arith_payoffs) * np.exp(-r*T), color='red', linestyle='--', 
           linewidth=2, label=f'PV Mean: ${np.mean(arith_payoffs)*np.exp(-r*T):.2f}')
ax.set_xlabel('Asian Call Payoff')
ax.set_ylabel('Frequency')
ax.set_title('Asian Call Payoff Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Convergence with variance reduction
ax = axes[1, 0]
ax.loglog(path_counts, errors_cv, 'go-', linewidth=2, markersize=8, label='Control Variate')
theoretical = errors_cv[0] * np.sqrt(path_counts[0]) / np.sqrt(np.array(path_counts))
ax.loglog(path_counts, theoretical, 'k--', linewidth=2, label='O(1/√N)')
ax.set_xlabel('Number of MC Paths')
ax.set_ylabel('Standard Error ($)')
ax.set_title('Standard Error Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Variance reduction comparison
methods = ['Standard\nMC', 'Antithetic\nVariates', 'Control\nVariate']
errors = [error_standard, error_av, error_cv]
colors = ['blue', 'orange', 'green']

ax = axes[1, 1]
bars = ax.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Standard Error ($)')
ax.set_title(f'Variance Reduction Methods (N={n_paths:,})')
ax.grid(True, axis='y', alpha=0.3)

# Add percentage labels
for i, (bar, error) in enumerate(zip(bars, errors)):
    if i > 0:
        reduction = (1 - error/errors[0]) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'-{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 6: Asian vs European value across strikes
strikes = np.linspace(80, 120, 20)
asian_values = []
european_values = []

for K_i in strikes:
    np.random.seed(42)
    asian_price, _, _, _, _ = mc_asian_call(S0, K_i, T, r, sigma, 5000, n_steps, use_control=True)
    asian_values.append(asian_price)
    european_values.append(european_call(S0, K_i, T, r, sigma))

ax = axes[1, 2]
ax.plot(strikes, european_values, 'b-', linewidth=2, label='European Call')
ax.plot(strikes, asian_values, 'r-', linewidth=2, label='Arithmetic Asian Call')
ax.axvline(S0, color='green', linestyle='--', alpha=0.5, label=f'Spot S₀=${S0}')
ax.set_xlabel('Strike K')
ax.set_ylabel('Option Value ($)')
ax.set_title('Asian vs European Call (Across Strikes)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('asian_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Comparison summary
print("\n" + "="*80)
print("ASIAN vs EUROPEAN COMPARISON")
print("="*80)
print(f"Asian Call:    ${price_cv:.6f}")
print(f"European Call: ${euro_call:.6f}")
print(f"Difference:    ${euro_call - price_cv:.6f} ({(1 - price_cv/euro_call)*100:.1f}% lower)")
print(f"\nAsian options are cheaper due to averaging (reduces volatility impact)")