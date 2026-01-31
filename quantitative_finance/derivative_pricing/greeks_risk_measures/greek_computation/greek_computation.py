
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fmin

# Black-Scholes components
def bs_d1(S, K, T, r, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return d1 - sigma*np.sqrt(T)

def bs_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

# Analytical Greeks for European call
def delta_analytical(S, K, T, r, sigma):
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def gamma_analytical(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega_analytical(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1%

def theta_analytical(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) - 
            r * K * np.exp(-r*T) * norm.cdf(d2)) / 365  # Per day

# Finite difference Greeks
def greeks_finite_diff(S, K, T, r, sigma, eps=0.01, greek_type='delta'):
    if greek_type == 'delta':
        V_up = bs_call(S + eps, K, T, r, sigma)
        V_down = bs_call(S - eps, K, T, r, sigma)
        return (V_up - V_down) / (2*eps)
    
    elif greek_type == 'gamma':
        V_up = bs_call(S + eps, K, T, r, sigma)
        V_mid = bs_call(S, K, T, r, sigma)
        V_down = bs_call(S - eps, K, T, r, sigma)
        return (V_up - 2*V_mid + V_down) / eps**2
    
    elif greek_type == 'vega':
        sigma_up = sigma + 0.0001  # 1 basis point
        sigma_down = sigma - 0.0001
        V_up = bs_call(S, K, T, r, sigma_up)
        V_down = bs_call(S, K, T, r, sigma_down)
        return (V_up - V_down) / (2*0.0001) / 100  # Normalize per 1%
    
    elif greek_type == 'theta':
        T_up = T + 1/365
        T_down = max(T - 1/365, 0.001)
        V_up = bs_call(S, K, T_up, r, sigma)
        V_down = bs_call(S, K, T_down, r, sigma)
        return -(V_up - V_down) / (2/365)  # Negative time decay
    
    elif greek_type == 'rho':
        r_up = r + 0.0001
        r_down = r - 0.0001
        V_up = bs_call(S, K, T, r_up, sigma)
        V_down = bs_call(S, K, T, r_down, sigma)
        return (V_up - V_down) / (2*0.0001) / 100  # Per 1%

# Monte Carlo Greeks (European call)
def greeks_mc_pathwise(S, K, T, r, sigma, N_paths=100000, greek_type='delta'):
    np.random.seed(42)
    Z = np.random.normal(0, 1, N_paths)
    S_T = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(S_T - K, 0)
    
    if greek_type == 'delta':
        # dS_T/dS = e^{...} (from chain rule)
        dS_dS0 = np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        dPayoff_dS = (S_T > K).astype(float) * dS_dS0
        delta = np.exp(-r*T) * np.mean(dPayoff_dS)
        se_delta = np.exp(-r*T) * np.std(dPayoff_dS) / np.sqrt(N_paths)
        return delta, se_delta
    
    elif greek_type == 'gamma':
        # Gamma = d²V/dS² (more complex; use finite difference on delta samples)
        eps = 0.01
        delta_up = greeks_mc_pathwise(S + eps, K, T, r, sigma, N_paths, 'delta')[0]
        delta_down = greeks_mc_pathwise(S - eps, K, T, r, sigma, N_paths, 'delta')[0]
        gamma = (delta_up - delta_down) / (2*eps)
        return gamma, 0
    
    elif greek_type == 'vega':
        # dPayoff/dσ ≈ finite difference
        sigma_up = sigma + 0.0001
        sigma_down = sigma - 0.0001
        price_up = np.exp(-r*T) * np.mean(np.maximum(
            S * np.exp((r - 0.5*sigma_up**2)*T + sigma_up*np.sqrt(T)*Z) - K, 0))
        price_down = np.exp(-r*T) * np.mean(np.maximum(
            S * np.exp((r - 0.5*sigma_down**2)*T + sigma_down*np.sqrt(T)*Z) - K, 0))
        vega = (price_up - price_down) / (2*0.0001) / 100
        return vega, 0

# Parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

# Compute Greeks
print("=== EUROPEAN CALL GREEKS COMPUTATION ===\n")

greeks_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
greeks_analytical = {}
greeks_fd = {}
greeks_mc = {}

for greek in greeks_names:
    if greek == 'delta':
        greeks_analytical[greek] = delta_analytical(S0, K, T, r, sigma)
        greeks_fd[greek] = greeks_finite_diff(S0, K, T, r, sigma, greek_type=greek)
        greeks_mc[greek], se_mc = greeks_mc_pathwise(S0, K, T, r, sigma, greek_type=greek)
    elif greek == 'gamma':
        greeks_analytical[greek] = gamma_analytical(S0, K, T, r, sigma)
        greeks_fd[greek] = greeks_finite_diff(S0, K, T, r, sigma, greek_type=greek)
        greeks_mc[greek], se_mc = greeks_mc_pathwise(S0, K, T, r, sigma, greek_type=greek)
    elif greek == 'vega':
        greeks_analytical[greek] = vega_analytical(S0, K, T, r, sigma)
        greeks_fd[greek] = greeks_finite_diff(S0, K, T, r, sigma, greek_type=greek)
        greeks_mc[greek], se_mc = greeks_mc_pathwise(S0, K, T, r, sigma, greek_type=greek)
    elif greek == 'theta':
        greeks_analytical[greek] = theta_analytical(S0, K, T, r, sigma)
        greeks_fd[greek] = greeks_finite_diff(S0, K, T, r, sigma, greek_type=greek)
    elif greek == 'rho':
        greeks_analytical[greek] = 0  # Define if needed
        greeks_fd[greek] = greeks_finite_diff(S0, K, T, r, sigma, greek_type=greek)

print("Greek\t\tAnalytical\tFinite Diff\tMC (pathwise)\tFD Error %\tMC Error %")
print("-" * 80)

for greek in greeks_names:
    if greek in greeks_mc and greeks_mc[greek] is not None:
        anal = greeks_analytical.get(greek, 0)
        fd = greeks_fd.get(greek, 0)
        mc = greeks_mc.get(greek, 0)
        
        fd_error = abs(fd - anal) / abs(anal) * 100 if anal != 0 else 0
        mc_error = abs(mc - anal) / abs(anal) * 100 if anal != 0 else 0
        
        print(f"{greek:15s}\t{anal:8.6f}\t{fd:8.6f}\t{mc:8.6f}\t{fd_error:6.2f}%\t{mc_error:6.2f}%")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot each Greek
spot_range = np.linspace(80, 120, 50)

# Delta
deltas_anal = [delta_analytical(S, K, T, r, sigma) for S in spot_range]
deltas_fd = [greeks_finite_diff(S, K, T, r, sigma, greek_type='delta') for S in spot_range]
axes[0, 0].plot(spot_range, deltas_anal, linewidth=2, label='Analytical', color='blue')
axes[0, 0].plot(spot_range, deltas_fd, 'o-', alpha=0.5, markersize=3, label='Finite Diff', color='orange')
axes[0, 0].set_title('Delta')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Gamma
gammas_anal = [gamma_analytical(S, K, T, r, sigma) for S in spot_range]
gammas_fd = [greeks_finite_diff(S, K, T, r, sigma, greek_type='gamma') for S in spot_range]
axes[0, 1].plot(spot_range, gammas_anal, linewidth=2, label='Analytical', color='blue')
axes[0, 1].plot(spot_range, gammas_fd, 'o-', alpha=0.5, markersize=3, label='Finite Diff', color='orange')
axes[0, 1].set_title('Gamma')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Vega
vegas_anal = [vega_analytical(S, K, T, r, sigma) for S in spot_range]
vegas_fd = [greeks_finite_diff(S, K, T, r, sigma, greek_type='vega') for S in spot_range]
axes[0, 2].plot(spot_range, vegas_anal, linewidth=2, label='Analytical', color='blue')
axes[0, 2].plot(spot_range, vegas_fd, 'o-', alpha=0.5, markersize=3, label='Finite Diff', color='orange')
axes[0, 2].set_title('Vega (per 1% vol)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Theta
thetas_anal = [theta_analytical(S, K, T, r, sigma) for S in spot_range]
thetas_fd = [greeks_finite_diff(S, K, T, r, sigma, greek_type='theta') for S in spot_range]
axes[1, 0].plot(spot_range, thetas_anal, linewidth=2, label='Analytical', color='blue')
axes[1, 0].plot(spot_range, thetas_fd, 'o-', alpha=0.5, markersize=3, label='Finite Diff', color='orange')
axes[1, 0].set_title('Theta (per day)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Rho
rhos_fd = [greeks_finite_diff(S, K, T, r, sigma, greek_type='rho') for S in spot_range]
axes[1, 1].plot(spot_range, rhos_fd, 'o-', markersize=4, label='Finite Diff', color='orange')
axes[1, 1].set_title('Rho (per 1% rate)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Error comparison
eps_values = np.logspace(-5, -1, 20)
delta_errors = []
for eps in eps_values:
    delta_fd_eps = greeks_finite_diff(S0, K, T, r, sigma, eps=eps, greek_type='delta')
    error = abs(delta_fd_eps - greeks_analytical['delta'])
    delta_errors.append(error)

axes[1, 2].loglog(eps_values, delta_errors, 'o-', linewidth=2, label='Delta FD Error')
axes[1, 2].set_xlabel('Step size ε')
axes[1, 2].set_ylabel('Absolute Error')
axes[1, 2].set_title('Finite Difference Error vs Step Size')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()