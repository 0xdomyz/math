
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0, K, T, r, sigma = 100, 130, 0.25, 0.05, 0.2  # Deep OTM call
N_paths = 10000

# Black-Scholes benchmark
def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

true_price = bs_call(S0, K, T, r, sigma)
print(f"True Price (BS): ${true_price:.6f}")

# Path generation function
def generate_paths(N, S0, K, T, r, sigma, shift=0, scale_factor=1):
    """Generate GBM paths with importance sampling parameters"""
    dt = T / 252
    n_steps = int(T / dt)
    
    # Modified drift for importance sampling
    Z = np.random.normal(shift, scale_factor, (N, n_steps))
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    return S

# Method 1: Standard Monte Carlo
print("\n=== STANDARD MC ===")
np.random.seed(42)
S_standard = generate_paths(N_paths, S0, K, T, r, sigma)
payoff_standard = np.maximum(S_standard[:, -1] - K, 0)

price_standard = np.exp(-r*T) * np.mean(payoff_standard)
se_standard = np.exp(-r*T) * np.std(payoff_standard) / np.sqrt(N_paths)
prob_itm_standard = np.mean(S_standard[:, -1] > K)

print(f"Price: ${price_standard:.6f} ± ${se_standard:.6f}")
print(f"P(ITM): {prob_itm_standard:.4f}")

# Method 2: Importance Sampling (shift mean upward)
print("\n=== IMPORTANCE SAMPLING (Shifted Mean) ===")
shift_mu = 0.05  # Shift mean of Z by 0.05 (makes S trend higher)
np.random.seed(42)
S_is = generate_paths(N_paths, S0, K, T, r, sigma, shift=shift_mu)
payoff_is = np.maximum(S_is[:, -1] - K, 0)

# Likelihood ratio: original / importance
# Original: Z ~ N(0, 1)
# Importance: Z ~ N(shift_mu, 1)
# Ratio = exp(-0.5*Z^2) / exp(-0.5*(Z - shift_mu)^2)
#       = exp(shift_mu*Z - 0.5*shift_mu^2)

log_returns_is = (r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*shift_mu + sigma*np.sqrt(T)*np.random.normal(0, 1, N_paths)
S_terminal = S0 * np.exp(log_returns_is)
payoff_is_final = np.maximum(S_terminal - K, 0)

# Compute importance weights
Z_used = (np.log(S_terminal/S0) - (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
log_weight = shift_mu * Z_used - 0.5 * shift_mu**2
weight = np.exp(log_weight)

price_is = np.exp(-r*T) * np.mean(payoff_is_final * weight)
se_is = np.exp(-r*T) * np.std(payoff_is_final * weight) / np.sqrt(N_paths)
prob_itm_is = np.mean(weight[S_terminal > K]) / np.mean(weight)

print(f"Price: ${price_is:.6f} ± ${se_is:.6f}")
print(f"P(ITM): {prob_itm_is:.4f}")
print(f"Weight mean: {np.mean(weight):.4f}, std: {np.std(weight):.4f}")

# Kish effective sample size
kish_ess = (np.sum(weight)**2) / np.sum(weight**2)
print(f"Kish ESS: {kish_ess:.0f} / {N_paths} ({100*kish_ess/N_paths:.1f}%)")

# Variance reduction
var_reduction_percentage = (1 - (se_is/se_standard)**2) * 100
print(f"Variance Reduction: {var_reduction_percentage:.1f}%")

# Monte Carlo simulation comparison across shifted means
shifts = np.linspace(-0.1, 0.15, 15)
prices_is_list = []
ses_is_list = []
ess_ratios = []

np.random.seed(42)
for shift in shifts:
    Z_sample = np.random.normal(shift, 1, N_paths)
    log_S = np.log(S0) + (r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_sample
    S_end = np.exp(log_S)
    payoff_shifted = np.maximum(S_end - K, 0)
    
    # Weight
    log_w = shift * Z_sample - 0.5*shift**2
    w = np.exp(log_w)
    
    # Estimate
    price_shifted = np.exp(-r*T) * np.mean(payoff_shifted * w)
    prices_is_list.append(price_shifted)
    
    # SE
    se_shifted = np.exp(-r*T) * np.std(payoff_shifted * w) / np.sqrt(N_paths)
    ses_is_list.append(se_shifted)
    
    # ESS
    kish = (np.sum(w)**2) / np.sum(w**2)
    ess_ratios.append(kish / N_paths)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price convergence for different shifts
axes[0, 0].plot(shifts, prices_is_list, 'o-', linewidth=2, markersize=8, label='IS Price')
axes[0, 0].axhline(true_price, color='r', linestyle='--', linewidth=2, label='True Price')
opt_shift_idx = np.argmin(ses_is_list)
axes[0, 0].plot(shifts[opt_shift_idx], prices_is_list[opt_shift_idx], 'g*', markersize=20, label='Best Shift')
axes[0, 0].set_xlabel('Mean Shift (μ)')
axes[0, 0].set_ylabel('Call Price ($)')
axes[0, 0].set_title('IS Price vs Importance Shift')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Standard error across shifts
axes[0, 1].plot(shifts, ses_is_list, 'o-', linewidth=2, markersize=8, color='orange')
axes[0, 1].axhline(se_standard, color='r', linestyle='--', linewidth=2, label='Standard MC SE')
axes[0, 1].plot(shifts[opt_shift_idx], ses_is_list[opt_shift_idx], 'g*', markersize=20, label='Best Shift')
axes[0, 1].set_xlabel('Mean Shift (μ)')
axes[0, 1].set_ylabel('Standard Error ($)')
axes[0, 1].set_title('SE vs Importance Shift')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Kish ESS ratio
axes[1, 0].plot(shifts, np.array(ess_ratios)*100, 'o-', linewidth=2, markersize=8, color='green')
axes[1, 0].set_xlabel('Mean Shift (μ)')
axes[1, 0].set_ylabel('Effective Sample Size (%)')
axes[1, 0].set_title('Kish ESS Ratio')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim([0, 105])

# Plot 4: Payoff vs Weight scatter (optimal shift)
Z_opt = np.random.normal(shifts[opt_shift_idx], 1, 5000)
log_S_opt = np.log(S0) + (r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_opt
S_opt = np.exp(log_S_opt)
payoff_opt = np.maximum(S_opt - K, 0)
w_opt = np.exp(shifts[opt_shift_idx]*Z_opt - 0.5*shifts[opt_shift_idx]**2)

axes[1, 1].scatter(payoff_opt[payoff_opt > 0], w_opt[payoff_opt > 0], alpha=0.5, s=10)
axes[1, 1].set_xlabel('Payoff ($)')
axes[1, 1].set_ylabel('Importance Weight')
axes[1, 1].set_title(f'Payoff vs Weight (Shift={shifts[opt_shift_idx]:.3f})')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== SUMMARY ===")
print(f"Best shift: {shifts[opt_shift_idx]:.3f}")
print(f"Best SE: ${ses_is_list[opt_shift_idx]:.6f}")
print(f"Improvement over Standard MC: {(1 - (ses_is_list[opt_shift_idx]/se_standard)**2)*100:.1f}%")