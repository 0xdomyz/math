
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

def bs_call(S, K, r, sigma, T, q=0):
    """European call with dividend yield q"""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def bs_put(S, K, r, sigma, T, q=0):
    """European put with dividend yield q"""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    return put

# Parameters
S_range = np.linspace(60, 140, 100)
K = 100
r = 0.05
sigma = 0.2
T = 1.0

# Scenarios
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Call with dividends (q=0 vs q=0.05)
call_no_div = [bs_call(S, K, r, sigma, T, q=0) for S in S_range]
call_div = [bs_call(S, K, r, sigma, T, q=0.05) for S in S_range]
intrinsic_call = np.maximum(S_range - K, 0)

axes[0, 0].plot(S_range, call_no_div, 'b-', linewidth=2, label='Call (q=0)')
axes[0, 0].plot(S_range, call_div, 'r--', linewidth=2, label='Call (q=5%)')
axes[0, 0].plot(S_range, intrinsic_call, 'k:', linewidth=1, label='Intrinsic')
axes[0, 0].set_title('European Call: Dividend Impact')
axes[0, 0].set_xlabel('Stock Price (S)')
axes[0, 0].set_ylabel('Option Value')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Put option across time to expiry
T_range = np.array([0.1, 0.25, 0.5, 1.0])
colors = plt.cm.viridis(np.linspace(0, 1, len(T_range)))

for T_val, color in zip(T_range, colors):
    put_vals = [bs_put(S, K, r, sigma, T_val) for S in S_range]
    axes[0, 1].plot(S_range, put_vals, color=color, linewidth=2, 
                   label=f'T={T_val}')

intrinsic_put = np.maximum(K - S_range, 0)
axes[0, 1].plot(S_range, intrinsic_put, 'k--', linewidth=1.5, label='Intrinsic')
axes[0, 1].set_title('European Put: Time Decay')
axes[0, 1].set_xlabel('Stock Price (S)')
axes[0, 1].set_ylabel('Option Value')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Time value decomposition (ITM put)
T_vals = np.linspace(0.01, T, 50)
S_ITM = 85  # ITM put

put_vals_T = [bs_put(S_ITM, K, r, sigma, T_val) for T_val in T_vals]
intrinsic_vals = [max(K - S_ITM, 0)] * len(T_vals)
time_value = [pv - intrinsic_vals[i] for i, pv in enumerate(put_vals_T)]

axes[1, 0].fill_between(T_vals, 0, intrinsic_vals, alpha=0.3, label='Intrinsic')
axes[1, 0].fill_between(T_vals, intrinsic_vals, put_vals_T, alpha=0.3, label='Time Value')
axes[1, 0].plot(T_vals, put_vals_T, 'b-', linewidth=2)
axes[1, 0].set_title(f'Put Decomposition (S=${S_ITM}, ITM)')
axes[1, 0].set_xlabel('Time to Expiry (years)')
axes[1, 0].set_ylabel('Option Value ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: American premium estimate (put, high interest rate scenario)
r_range = np.linspace(0, 0.15, 30)
premiums = []
S_val = 90  # ITM

for r_val in r_range:
    eur_put = bs_put(S_val, K, r_val, sigma, T)
    # Rough American approximation (more precise requires binomial)
    intrinsic = max(K - S_val, 0)
    # Premium increases with r (early exercise more valuable)
    premium_estimate = intrinsic - eur_put + 0.01 * (r_val - r)  # heuristic
    premium_estimate = max(premium_estimate, 0)
    premiums.append(premium_estimate)

axes[1, 1].plot(r_range * 100, premiums, 'ro-', linewidth=2, markersize=6)
axes[1, 1].set_title('American Put Premium (Rough Estimate)')
axes[1, 1].set_xlabel('Risk-Free Rate (%)')
axes[1, 1].set_ylabel('American - European ($)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Early Exercise Analysis:")
print(f"European Call (q=0, S=100): ${bs_call(100, K, r, sigma, T, q=0):.3f}")
print(f"European Call (q=5%, S=100): ${bs_call(100, K, r, sigma, T, q=0.05):.3f}")
print(f"European Put (S=90): ${bs_put(90, K, r, sigma, T):.3f}")
print(f"Intrinsic Put (S=90): ${max(K-90, 0):.3f}")