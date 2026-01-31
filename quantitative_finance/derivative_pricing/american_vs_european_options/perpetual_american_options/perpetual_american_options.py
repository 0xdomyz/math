
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def solve_perpetual_put_beta(r, sigma):
    """Solve for β+ (positive root) of characteristic equation"""
    # β² + (2(r-q)/σ² - 1)β - 2r/σ² = 0 (q=0 for perpetual put)
    # β² + (2r/σ² - 1)β - 2r/σ² = 0
    a = 1
    b = 2*r/sigma**2 - 1
    c = -2*r/sigma**2
    
    disc = b**2 - 4*a*c
    beta_plus = (-b + np.sqrt(disc)) / (2*a)
    return beta_plus

def perpetual_put_boundary(K, r, sigma):
    """S* for perpetual American put"""
    beta = solve_perpetual_put_beta(r, sigma)
    # S* = K * β/(β+1)
    S_star = K * beta / (beta + 1)
    return S_star, beta

def perpetual_put_value(S, K, r, sigma):
    """V(S) for perpetual American put"""
    S_star, beta = perpetual_put_boundary(K, r, sigma)
    
    if S <= S_star:
        return K - S  # Exercise region
    else:
        # Hold region: V(S) = (S/S*)^β * (K - S*)
        return ((S/S_star)**beta) * (K - S_star)

# Parameters
K = 100
r_base = 0.05
sigma_base = 0.2
S_range = np.linspace(1, 150, 200)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Put value across S (benchmark case)
r, sigma = r_base, sigma_base
S_star, beta = perpetual_put_boundary(K, r, sigma)
put_vals = [perpetual_put_value(S, K, r, sigma) for S in S_range]
intrinsic = np.maximum(K - S_range, 0)

axes[0, 0].plot(S_range, put_vals, 'b-', linewidth=2.5, label='V(S)')
axes[0, 0].plot(S_range, intrinsic, 'k--', linewidth=1.5, label='Intrinsic')
axes[0, 0].axvline(x=S_star, color='r', linestyle=':', linewidth=2, label=f'S*=${S_star:.2f}')
axes[0, 0].fill_between(S_range[S_range <= S_star], 0, 
                        np.array(put_vals)[S_range <= S_star], 
                        alpha=0.2, color='green', label='Exercise Region')
axes[0, 0].set_title(f'Perpetual Put Value (r={r*100:.1f}%, σ={sigma*100:.1f}%)')
axes[0, 0].set_xlabel('Stock Price S')
axes[0, 0].set_ylabel('Option Value')
axes[0, 0].legend(loc='upper right')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_ylim(0, 50)

# Plot 2: Boundary vs volatility
sigma_range = np.linspace(0.05, 0.50, 50)
boundaries = []
betas = []

for sig in sigma_range:
    S_star, beta = perpetual_put_boundary(K, r_base, sig)
    boundaries.append(S_star)
    betas.append(beta)

axes[0, 1].plot(sigma_range*100, boundaries, 'g-', linewidth=2.5)
axes[0, 1].set_title(f'Perpetual Put Boundary vs Volatility (r={r_base*100:.1f}%)')
axes[0, 1].set_xlabel('Volatility σ (%)')
axes[0, 1].set_ylabel('Exercise Boundary S* ($)')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].fill_between(sigma_range*100, 0, boundaries, alpha=0.2)

# Plot 3: Boundary vs interest rate
r_range = np.linspace(0.01, 0.20, 50)
boundaries_r = []

for r_val in r_range:
    S_star, _ = perpetual_put_boundary(K, r_val, sigma_base)
    boundaries_r.append(S_star)

axes[0, 2].plot(r_range*100, boundaries_r, 'm-', linewidth=2.5)
axes[0, 2].set_title(f'Perpetual Put Boundary vs Rate (σ={sigma_base*100:.1f}%)')
axes[0, 2].set_xlabel('Interest Rate r (%)')
axes[0, 2].set_ylabel('Exercise Boundary S* ($)')
axes[0, 2].grid(alpha=0.3)
axes[0, 2].fill_between(r_range*100, 0, boundaries_r, alpha=0.2, color='purple')

# Plot 4: Beta exponent sensitivity
axes[1, 0].plot(sigma_range*100, betas, 'c-', linewidth=2.5)
axes[1, 0].set_title(f'Beta Exponent vs Volatility (r={r_base*100:.1f}%)')
axes[1, 0].set_xlabel('Volatility σ (%)')
axes[1, 0].set_ylabel('β+ exponent')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='β+=1')
axes[1, 0].legend()

# Plot 5: Value comparison (different r)
r_values = [0.01, 0.05, 0.10]
colors = ['blue', 'green', 'red']

for r_val, color in zip(r_values, colors):
    S_star, beta = perpetual_put_boundary(K, r_val, sigma_base)
    put_vals = [perpetual_put_value(S, K, r_val, sigma_base) for S in S_range]
    axes[1, 1].plot(S_range, put_vals, color=color, linewidth=2, 
                   label=f'r={r_val*100:.1f}%, S*=${S_star:.0f}')

intrinsic = np.maximum(K - S_range, 0)
axes[1, 1].plot(S_range, intrinsic, 'k--', linewidth=1, alpha=0.5)
axes[1, 1].set_title(f'Perpetual Put: Interest Rate Impact (σ={sigma_base*100:.1f}%)')
axes[1, 1].set_xlabel('Stock Price S')
axes[1, 1].set_ylabel('Option Value')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim(0, 50)

# Plot 6: Premium over European (rough comparison)
# European put value from Black-Scholes (for very long T)
from scipy.stats import norm

def european_put_bs(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

T_long = 10  # 10-year approximation
premiums = []
S_test = np.linspace(70, K, 30)

for S_val in S_test:
    perp_val = perpetual_put_value(S_val, K, r_base, sigma_base)
    eur_val = european_put_bs(S_val, K, r_base, sigma_base, T_long)
    premium = perp_val - eur_val
    premiums.append(premium)

axes[1, 2].plot(S_test, premiums, 'o-', linewidth=2, markersize=6, color='orange')
axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1, 2].set_title(f'Perpetual Premium over European (T=10yr)')
axes[1, 2].set_xlabel('Stock Price S')
axes[1, 2].set_ylabel('Premium ($)')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].fill_between(S_test, 0, premiums, alpha=0.2)

plt.tight_layout()
plt.show()

# Summary table
print("Perpetual American Put: Summary Table")
print("="*70)
print(f"Strike K=${K}, Base σ={sigma_base*100:.1f}%, Base r={r_base*100:.1f}%")
print("-"*70)
print(f"{'σ (%)':<10} {'r (%)':<10} {'S* ($)':<12} {'β+ ':<10} {'Premium':<12}")
print("-"*70)

for sigma in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for r in [0.02, 0.05, 0.10]:
        S_star, beta = perpetual_put_boundary(K, r, sigma)
        intrinsic_at_boundary = K - S_star
        print(f"{sigma*100:<10.1f} {r*100:<10.1f} {S_star:<12.2f} {beta:<10.3f} {intrinsic_at_boundary:<12.2f}")