
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq
from scipy.stats import norm

def perpetual_put_boundary(K, r, sigma):
    """Compute S* for perpetual American put"""
    # β = (-r + √(r² + 2rσ²))/σ²
    disc = r**2 + 2*r*sigma**2
    beta = (-r + np.sqrt(disc)) / (sigma**2)
    S_star = K * beta / (beta + 1)
    return S_star, beta

def perpetual_put_value(S, K, r, sigma):
    """Value function for perpetual American put"""
    S_star, beta = perpetual_put_boundary(K, r, sigma)
    if S <= S_star:
        return K - S  # Exercise
    else:
        # V(S) = A*(S/S*)^β * (K-S*), solve for A from smooth pasting
        A = (S_star**(1-beta)) * (beta + 1) / beta
        return A * (S**beta) - A*beta*S_star + (K - S_star)

# Parameters
K = 100
r_values = [0.02, 0.05, 0.10]
sigma = 0.2
S_range = np.linspace(1, 150, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Perpetual boundary vs interest rates
ax = axes[0, 0]
boundaries = []
for r in r_values:
    S_star, beta = perpetual_put_boundary(K, r, sigma)
    boundaries.append(S_star)
    ax.axhline(y=S_star, linestyle='--', label=f'r={r*100:.1f}%, S*=${S_star:.2f}')

ax.set_ylim(0, K)
ax.set_xlim(0, 3)
ax.set_title('Perpetual Put Exercise Boundary vs Interest Rate')
ax.set_xlabel('Interest Rate (%)')
ax.set_ylabel('Exercise Boundary S*')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Perpetual put value vs spot
ax = axes[0, 1]
r = 0.05
S_star, beta = perpetual_put_boundary(K, r, sigma)
put_vals = [perpetual_put_value(S, K, r, sigma) for S in S_range]
intrinsic = np.maximum(K - S_range, 0)

ax.plot(S_range, put_vals, 'b-', linewidth=2, label='V(S)')
ax.plot(S_range, intrinsic, 'k--', linewidth=1.5, label='Intrinsic')
ax.axvline(x=S_star, color='r', linestyle=':', linewidth=2, label=f'S*=${S_star:.2f}')
ax.set_title(f'Perpetual Put Value (r={r*100:.1f}%, σ={sigma*100:.1f}%)')
ax.set_xlabel('Stock Price S')
ax.set_ylabel('Option Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Beta sensitivity
sigma_range = np.linspace(0.05, 0.50, 50)
beta_vals = []
for sig in sigma_range:
    _, beta = perpetual_put_boundary(K, r, sig)
    beta_vals.append(beta)

ax = axes[1, 0]
ax.plot(sigma_range*100, beta_vals, 'g-', linewidth=2)
ax.set_title(f'Beta Exponent vs Volatility (r={r*100:.1f}%)')
ax.set_xlabel('Volatility σ (%)')
ax.set_ylabel('β (power in value function)')
ax.grid(alpha=0.3)

# Plot 4: Finite-time put boundary (approximated by binomial)
def american_put_binomial(S0, K, r, sigma, T, N=50, q=0):
    """Simple binomial tree for American put"""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r-q)*dt) - d) / (u - d)
    
    # Initialize tree
    V = np.zeros((N+1, N+1))
    S = np.zeros((N+1, N+1))
    
    # Terminal nodes
    for j in range(N+1):
        S[N, j] = S0 * (u**j) * (d**(N-j))
        V[N, j] = max(K - S[N, j], 0)
    
    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            S[i, j] = S0 * (u**j) * (d**(i-j))
            # European value
            V_hold = (p*V[i+1, j+1] + (1-p)*V[i+1, j]) * np.exp(-r*dt)
            # American: max(intrinsic, hold)
            V[i, j] = max(K - S[i, j], V_hold)
    
    return V[0, 0], S

# Compute boundaries for different T
T_values = [0.5, 1.0, 2.0, 5.0]
ax = axes[1, 1]

time_points = np.linspace(0, T_values[-1], 30)
for T in T_values:
    boundaries_T = []
    S_test = np.linspace(50, 150, 50)
    
    for S_val in S_test:
        val, _ = american_put_binomial(S_val, K, r, sigma, T, N=30)
        intrinsic_val = max(K - S_val, 0)
        # Boundary approximately where value ≈ intrinsic
        if abs(val - intrinsic_val) < 1:
            boundaries_T.append(S_val)
    
    if boundaries_T:
        S_boundary = np.mean(boundaries_T)
        ax.scatter(T, S_boundary, s=100, label=f'T={T}yr')

# Add perpetual as limit
S_perp, _ = perpetual_put_boundary(K, r, sigma)
ax.axhline(y=S_perp, color='k', linestyle='--', linewidth=1.5, 
          label=f'Perpetual: S*=${S_perp:.2f}')
ax.set_title('American Put Exercise Boundary vs Time to Expiry')
ax.set_xlabel('Time to Expiry T (years)')
ax.set_ylabel('Exercise Boundary S*')
ax.set_ylim(60, 100)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Display key results
print("Perpetual American Put Analysis")
print("="*50)
for r_val in r_values:
    S_star, beta = perpetual_put_boundary(K, r_val, sigma)
    print(f"r={r_val*100:.1f}%: S*=${S_star:.2f}, β={beta:.3f}")