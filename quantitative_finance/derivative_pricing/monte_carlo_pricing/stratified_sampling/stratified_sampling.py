
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Test function: 2D integral
def test_function_2d(x, y):
    """2D test function: sin(πx) * cos(πy)"""
    return np.sin(np.pi*x) * np.cos(np.pi*y)

# Analytical integral: ∫₀¹ ∫₀¹ sin(πx)cos(πy) dx dy = 0
analytical_integral = 0.0

# Method 1: Standard Monte Carlo (2D)
def standard_mc_2d(N):
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    values = test_function_2d(x, y)
    estimate = np.mean(values)
    se = np.std(values) / np.sqrt(N)
    return estimate, se

# Method 2: Stratified Sampling (2D)
def stratified_sampling_2d(N, n_strata=10):
    """Stratified sampling with n_strata × n_strata strata"""
    n_per_stratum = N // (n_strata**2)
    stratum_size = 1.0 / n_strata
    
    values_all = []
    
    for i in range(n_strata):
        for j in range(n_strata):
            # Bounds of stratum
            x_min, x_max = i*stratum_size, (i+1)*stratum_size
            y_min, y_max = j*stratum_size, (j+1)*stratum_size
            
            # Sample uniformly in stratum
            x = np.random.uniform(x_min, x_max, n_per_stratum)
            y = np.random.uniform(y_min, y_max, n_per_stratum)
            
            values = test_function_2d(x, y)
            values_all.extend(values)
    
    values_all = np.array(values_all)
    estimate = np.mean(values_all)
    se = np.std(values_all) / np.sqrt(len(values_all))
    return estimate, se

# Run comparisons
N_samples = 10000
n_trials = 100

print("=== 2D Integral: ∫₀¹∫₀¹ sin(πx)cos(πy) dx dy ===")
print(f"Analytical result: {analytical_integral}")

# Standard MC
standard_estimates = []
standard_ses = []
for _ in range(n_trials):
    est, se = standard_mc_2d(N_samples)
    standard_estimates.append(est)
    standard_ses.append(se)

# Stratified
stratified_estimates = []
stratified_ses = []
for _ in range(n_trials):
    est, se = stratified_sampling_2d(N_samples, n_strata=10)
    stratified_estimates.append(est)
    stratified_ses.append(se)

print(f"\nStandard MC (100 trials):")
print(f"  Mean estimate: {np.mean(standard_estimates):.6f}")
print(f"  Mean SE: {np.mean(standard_ses):.6f}")
print(f"  Estimate std: {np.std(standard_estimates):.6f}")

print(f"\nStratified Sampling (100 trials, 10×10 strata):")
print(f"  Mean estimate: {np.mean(stratified_estimates):.6f}")
print(f"  Mean SE: {np.mean(stratified_ses):.6f}")
print(f"  Estimate std: {np.std(stratified_estimates):.6f}")

variance_reduction = (1 - (np.mean(stratified_ses)/np.mean(standard_ses))**2) * 100
print(f"\nVariance Reduction: {variance_reduction:.1f}%")

# Application: Option pricing with stratified sampling
print("\n=== EUROPEAN OPTION PRICING ===")

S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

# Black-Scholes benchmark
def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

true_price = bs_call(S0, K, T, r, sigma)

# Standard MC for options
def option_mc_standard(N):
    Z = np.random.normal(0, 1, N)
    S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(S_T - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    se = np.exp(-r*T) * np.std(payoff) / np.sqrt(N)
    return price, se

# Stratified MC for options (stratify by standard normal quantiles)
def option_mc_stratified(N, n_strata=10):
    n_per_stratum = N // n_strata
    payoffs_all = []
    
    # Quantile boundaries for standard normal
    quantiles = np.linspace(0, 1, n_strata+1)
    z_boundaries = norm.ppf(quantiles)
    
    for i in range(n_strata):
        z_min, z_max = z_boundaries[i], z_boundaries[i+1]
        
        # Uniform samples in [0,1] mapped to [z_min, z_max] via inverse CDF
        u = np.random.uniform(0, 1, n_per_stratum)
        z_stratum = norm.ppf(quantiles[i] + u * (quantiles[i+1] - quantiles[i]))
        
        S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z_stratum)
        payoffs = np.maximum(S_T - K, 0)
        payoffs_all.extend(payoffs)
    
    payoffs_all = np.array(payoffs_all)
    price = np.exp(-r*T) * np.mean(payoffs_all)
    se = np.exp(-r*T) * np.std(payoffs_all) / np.sqrt(len(payoffs_all))
    return price, se

# Run option pricing
N = 10000
n_trials = 50

std_prices = []
std_ses = []
strat_prices = []
strat_ses = []

for _ in range(n_trials):
    p_std, se_std = option_mc_standard(N)
    std_prices.append(p_std)
    std_ses.append(se_std)
    
    p_strat, se_strat = option_mc_stratified(N, n_strata=10)
    strat_prices.append(p_strat)
    strat_ses.append(se_strat)

print(f"True Price (BS): ${true_price:.4f}")
print(f"Standard MC: ${np.mean(std_prices):.4f} ± ${np.mean(std_ses):.4f}")
print(f"Stratified: ${np.mean(strat_prices):.4f} ± ${np.mean(strat_ses):.4f}")
print(f"Variance Reduction: {(1 - (np.mean(strat_ses)/np.mean(std_ses))**2)*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: 2D stratified grid visualization
ax = axes[0, 0]
n_strata = 10
stratum_size = 1.0 / n_strata
for i in range(n_strata+1):
    ax.axhline(i*stratum_size, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(i*stratum_size, color='gray', linestyle='--', linewidth=0.5)

# Sample and plot
np.random.seed(42)
for i in range(n_strata):
    for j in range(n_strata):
        x_min, x_max = i*stratum_size, (i+1)*stratum_size
        y_min, y_max = j*stratum_size, (j+1)*stratum_size
        x = np.random.uniform(x_min, x_max, 10)
        y = np.random.uniform(y_min, y_max, 10)
        ax.scatter(x, y, alpha=0.5, s=10)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Stratified Sampling (10×10 grid)')

# Plot 2: Integral estimate convergence (100 trials)
axes[0, 1].plot(range(1, len(standard_estimates)+1), standard_estimates, 'o-', 
               label='Standard MC', alpha=0.7, markersize=4, linewidth=1)
axes[0, 1].plot(range(1, len(stratified_estimates)+1), stratified_estimates, 's-', 
               label='Stratified', alpha=0.7, markersize=4, linewidth=1)
axes[0, 1].axhline(analytical_integral, color='r', linestyle='--', linewidth=2, label='True')
axes[0, 1].set_xlabel('Trial')
axes[0, 1].set_ylabel('Integral Estimate')
axes[0, 1].set_title('Integral Convergence (100 Trials)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Standard error comparison
methods = ['Standard MC', 'Stratified']
ses_mean = [np.mean(standard_ses), np.mean(stratified_ses)]
colors = ['C0', 'C1']

axes[1, 0].bar(methods, ses_mean, color=colors, alpha=0.7, width=0.5)
axes[1, 0].set_ylabel('Mean Standard Error')
axes[1, 0].set_title('SE Comparison (2D Integral)')
axes[1, 0].grid(alpha=0.3, axis='y')

for i, se in enumerate(ses_mean):
    axes[1, 0].text(i, se + 0.001, f'{se:.5f}', ha='center', fontweight='bold')

# Plot 4: Option pricing SE comparison
methods = ['Standard MC', 'Stratified']
ses_option = [np.mean(std_ses), np.mean(strat_ses)]

axes[1, 1].bar(methods, ses_option, color=colors, alpha=0.7, width=0.5)
axes[1, 1].set_ylabel('Mean Standard Error ($)')
axes[1, 1].set_title('SE Comparison (Option Pricing)')
axes[1, 1].grid(alpha=0.3, axis='y')

for i, se in enumerate(ses_option):
    axes[1, 1].text(i, se + 0.0005, f'${se:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()