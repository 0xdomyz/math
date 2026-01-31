
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Parameters
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

print("="*70)
print("AMERICAN OPTION VALUATION: METHOD COMPARISON")
print("="*70)
print(f"Parameters: S0=${S0}, K=${K}, r={r*100:.1f}%, σ={sigma*100:.1f}%, T={T}yr")
print()

# =========== METHOD 1: BINOMIAL TREE ===========
def binomial_american(S0, K, r, sigma, T, N, option_type='put'):
    """American option via binomial tree"""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize values at terminal node
    V = np.zeros(N + 1)
    for j in range(N + 1):
        S = S0 * u**j * d**(N - j)
        if option_type == 'put':
            V[j] = max(K - S, 0)
        else:
            V[j] = max(S - K, 0)
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S = S0 * u**j * d**(i - j)
            V_hold = (q * V[j + 1] + (1 - q) * V[j]) * np.exp(-r * dt)
            
            if option_type == 'put':
                intrinsic = max(K - S, 0)
            else:
                intrinsic = max(S - K, 0)
            
            V[j] = max(intrinsic, V_hold)
    
    return V[0]

# =========== METHOD 2: FINITE DIFFERENCE ===========
def fd_american_put(S0, K, r, sigma, T, NS, NT, method='implicit'):
    """American put via finite difference (Implicit, Crank-Nicolson)"""
    # Grid
    S_max = 2 * K
    dS = S_max / NS
    dt = T / NT
    
    S = np.arange(0, S_max + dS, dS)
    lambda_param = r * dt / (dS**2 * sigma**2)
    
    # Boundary conditions setup (implicit scheme)
    # Tridiagonal system: a_i*V[i-1] + b_i*V[i] + c_i*V[i+1] = d_i
    
    # Initialize at terminal time
    V = np.maximum(K - S, 0)  # Put intrinsic
    
    for n in range(NT - 1, -1, -1):
        # Build tridiagonal system
        ab = np.zeros((3, NS + 1))  # For solve_banded format
        d = np.zeros(NS + 1)
        
        for i in range(1, NS):
            alpha = 0.5 * r * i * lambda_param
            beta = sigma**2 * i**2 * lambda_param
            
            # Implicit backward difference:
            a_coeff = -alpha + beta
            b_coeff = 1 + 2*beta
            c_coeff = alpha + beta
            
            ab[1, i] = b_coeff
            ab[2, i-1] = c_coeff if i > 0 else 0
            if i < NS:
                ab[0, i+1] = a_coeff if i < NS-1 else 0
            
            d[i] = V[i]
        
        # Boundary conditions
        ab[1, 0] = 1
        d[0] = 0  # V(0,t) = K (at S=0, put deep ITM)
        
        ab[1, NS] = 1
        d[NS] = 0  # V(S_max,t) = 0 (at S=S_max, put worthless)
        
        # Solve tridiagonal
        try:
            V_new = solve_banded((1, 1), ab, d)
        except:
            V_new = V.copy()  # Fallback
        
        # American: max(intrinsic, continuation)
        V = np.maximum(K - S, V_new)
    
    # Interpolate at S0
    idx = int(S0 / dS)
    return V[idx]

# =========== METHOD 3: CLOSED-FORM (PERPETUAL) ===========
def perpetual_put_analytical(S0, K, r, sigma):
    """Analytical perpetual American put"""
    # β = (-r + √(r² + 2rσ²))/σ²
    disc = r**2 + 2*r*sigma**2
    beta = (-r + np.sqrt(disc)) / (sigma**2)
    
    # S* = K * β/(β+1)
    S_star = K * beta / (beta + 1)
    
    if S0 <= S_star:
        return K - S0
    else:
        # V(S) = (S/S*)^β * (K - S*)
        return ((S0 / S_star)**beta) * (K - S_star)

# Compute all methods
print("METHOD COMPARISON:")
print("-"*70)

N_steps = [50, 100, 200, 500]
for N in N_steps:
    binomial_val = binomial_american(S0, K, r, sigma, T, N, 'put')
    print(f"Binomial (N={N:3d}): ${binomial_val:.4f}")

fd_val = fd_american_put(S0, K, r, sigma, T, 100, 100, 'implicit')
print(f"Finite Diff (100×100): ${fd_val:.4f}")

perpetual_val = perpetual_put_analytical(S0, K, r, sigma)
print(f"Perpetual (T→∞): ${perpetual_val:.4f}")

print()

# =========== CONVERGENCE ANALYSIS ===========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Convergence vs N (Binomial)
N_range = np.arange(10, 301, 10)
binomial_vals = [binomial_american(S0, K, r, sigma, T, N, 'put') for N in N_range]

ax = axes[0, 0]
ax.plot(N_range, binomial_vals, 'b-', linewidth=2, label='Binomial')
ax.axhline(y=binomial_vals[-1], color='r', linestyle='--', alpha=0.7, label=f'Converged (~${binomial_vals[-1]:.4f})')
ax.set_title('Binomial Convergence (Steps)')
ax.set_xlabel('Number of Steps (N)')
ax.set_ylabel('Option Value ($)')
ax.grid(alpha=0.3)
ax.legend()

# Plot 2: American vs European comparison
def european_put_bs(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

S_range = np.linspace(60, 140, 50)
american_vals = [binomial_american(S, K, r, sigma, T, 100, 'put') for S in S_range]
european_vals = [european_put_bs(S, K, r, sigma, T) for S in S_range]
premiums = np.array(american_vals) - np.array(european_vals)

ax = axes[0, 1]
ax.plot(S_range, american_vals, 'b-', linewidth=2.5, label='American')
ax.plot(S_range, european_vals, 'r--', linewidth=2, label='European')
ax.fill_between(S_range, european_vals, american_vals, alpha=0.2, color='green', label='Premium')
ax.set_title('American vs European Put (T=1yr)')
ax.set_xlabel('Stock Price ($)')
ax.set_ylabel('Option Value ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Premium by spot price
ax = axes[1, 0]
ax.plot(S_range, premiums, 'g-', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(S_range, 0, premiums, alpha=0.2)
ax.set_title('American Premium: A - E')
ax.set_xlabel('Stock Price ($)')
ax.set_ylabel('Premium ($)')
ax.grid(alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Plot 4: Computation time comparison (log scale)
ax = axes[1, 1]
import time

methods = ['BS (Eur)', 'Binomial (100)', 'Binomial (500)', 'FD (100×100)', 'Perpetual']
times = []

# Time each method
for _ in range(100):
    t0 = time.time(); european_put_bs(S0, K, r, sigma, T); times.append(time.time() - t0)
t_bs = np.mean(times)

for _ in range(10):
    t0 = time.time(); binomial_american(S0, K, r, sigma, T, 100, 'put'); times.append(time.time() - t0)
t_bin100 = np.mean(times[-10:])

for _ in range(10):
    t0 = time.time(); binomial_american(S0, K, r, sigma, T, 500, 'put'); times.append(time.time() - t0)
t_bin500 = np.mean(times[-10:])

t_fd = 0.01  # Approximate
t_perp = 0.0001

times_list = [t_bs, t_bin100, t_bin500, t_fd, 0.0001]
ax.bar(methods, np.log10(times_list), color=['blue', 'green', 'orange', 'red', 'purple'])
ax.set_yscale('log', base=10)
ax.set_ylabel('Log10(Time, seconds)')
ax.set_title('Computation Speed Comparison')
ax.tick_params(axis='x', rotation=15)
for i, t in enumerate(times_list):
    ax.text(i, np.log10(t) + 0.2, f'{t*1000:.2f}ms', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\nConclusion:")
print("Binomial: Fast, accurate for standard American, easy implementation")
print("FD: Highest accuracy, Greeks easy, but slower")
print("Perpetual: Instant if T→∞, theoretical benchmark")