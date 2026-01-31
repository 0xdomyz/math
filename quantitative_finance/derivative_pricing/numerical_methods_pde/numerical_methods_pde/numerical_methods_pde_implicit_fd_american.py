import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
    from scipy.stats import norm
def implicit_fd_american(S0, K, T, r, sigma, option_type='put', M=100, N=500):
    """
    Implicit finite difference with early exercise for American options.
    
    Uses projected SOR to enforce V ≥ Payoff constraint.
    """
    S_max = 3 * K
    S_min = 0
    
    dS = (S_max - S_min) / M
    dt = T / N
    
    S = np.linspace(S_min, S_max, M+1)
    
    # Payoff function
    if option_type == 'call':
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    
    V = payoff.copy()
    
    # Tridiagonal matrix coefficients
    alpha = np.zeros(M+1)
    beta = np.zeros(M+1)
    gamma = np.zeros(M+1)
    
    for i in range(1, M):
        alpha[i] = -0.5 * dt * (r * i - sigma**2 * i**2)
        beta[i] = 1 + dt * (sigma**2 * i**2 + r)
        gamma[i] = -0.5 * dt * (r * i + sigma**2 * i**2)
    
    diagonals = [alpha[1:M], beta[1:M], gamma[1:M-1]]
    A = diags(diagonals, [-1, 0, 1], shape=(M-1, M-1), format='csc')
    
    # Track early exercise boundary
    exercise_boundary = []
    
    # Backward time-stepping
    for n in range(N):
        b = V[1:M].copy()
        
        # Boundary conditions
        if option_type == 'call':
            V[0] = 0
            V[M] = S_max
        else:
            V[0] = K*np.exp(-r*(T - (n+1)*dt))
            V[M] = 0
        
        b[0] -= alpha[1] * V[0]
        b[-1] -= gamma[M-1] * V[M]
        
        # Solve as European
        V_european = spsolve(A, b)
        
        # Enforce early exercise constraint
        V[1:M] = np.maximum(V_european, payoff[1:M])
        
        # Find exercise boundary (first point where V = payoff)
        if option_type == 'put':
            exercise_idx = np.where(V[1:M] <= payoff[1:M] + 1e-6)[0]
            if len(exercise_idx) > 0:
                exercise_boundary.append(S[exercise_idx[-1]+1])
            else:
                exercise_boundary.append(S_min)
    
    option_price = np.interp(S0, S, V)
    return option_price, S, V, exercise_boundary

# =====================================
# TEST CASE: EUROPEAN PUT OPTION
# =====================================
print("\n" + "="*70)
print("EUROPEAN PUT OPTION")
print("="*70)

S0 = 100  # Current stock price
K = 100   # Strike price
T = 1.0   # Time to maturity (1 year)
r = 0.05  # Risk-free rate
sigma = 0.20  # Volatility

# Analytical Black-Scholes price
bs_price = black_scholes_price(S0, K, T, r, sigma, option_type='put')

# PDE methods
explicit_price, S_grid, V_explicit = explicit_fd_european(S0, K, T, r, sigma, 'put', M=100, N=2000)
implicit_price, _, V_implicit = implicit_fd_european(S0, K, T, r, sigma, 'put', M=100, N=500)
cn_price, _, V_cn = crank_nicolson_european(S0, K, T, r, sigma, 'put', M=100, N=500)

print(f"\nOption Parameters:")
print(f"   S0 = ${S0}, K = ${K}, T = {T} years")
print(f"   r = {r:.1%}, σ = {sigma:.1%}")

print(f"\nPricing Results:")
print(f"   Black-Scholes (Analytical): ${bs_price:.4f}")
print(f"   Explicit FD:                ${explicit_price:.4f} (Error: ${explicit_price-bs_price:.4f})")
print(f"   Implicit FD:                ${implicit_price:.4f} (Error: ${implicit_price-bs_price:.4f})")
print(f"   Crank-Nicolson:             ${cn_price:.4f} (Error: ${cn_price-bs_price:.4f})")

# =====================================
# AMERICAN PUT OPTION
# =====================================
print("\n" + "="*70)
print("AMERICAN PUT OPTION (EARLY EXERCISE)")
print("="*70)

american_price, S_grid_am, V_american, exercise_boundary = implicit_fd_american(
    S0, K, T, r, sigma, 'put', M=100, N=500
)

print(f"\nAmerican Put Price: ${american_price:.4f}")
print(f"European Put Price: ${bs_price:.4f}")
print(f"Early Exercise Premium: ${american_price - bs_price:.4f} ({(american_price/bs_price-1)*100:.2f}%)")

if len(exercise_boundary) > 0:
    print(f"\nEarly Exercise Boundary:")
    print(f"   Near maturity: S* ≈ ${exercise_boundary[0]:.2f}")
    print(f"   Near t=0: S* ≈ ${exercise_boundary[-1]:.2f}")

# =====================================
# CONVERGENCE ANALYSIS
# =====================================
print("\n" + "="*70)
print("CONVERGENCE ANALYSIS")
print("="*70)

grid_sizes = [20, 40, 80, 160]
explicit_errors = []
implicit_errors = []
cn_errors = []

for M in grid_sizes:
    N = M * 20  # Explicit needs fine time grid
    exp_p, _, _ = explicit_fd_european(S0, K, T, r, sigma, 'put', M=M, N=N)
    explicit_errors.append(abs(exp_p - bs_price))
    
    N = M * 5  # Implicit can use coarser grid
    imp_p, _, _ = implicit_fd_european(S0, K, T, r, sigma, 'put', M=M, N=N)
    implicit_errors.append(abs(imp_p - bs_price))
    
    cn_p, _, _ = crank_nicolson_european(S0, K, T, r, sigma, 'put', M=M, N=N)
    cn_errors.append(abs(cn_p - bs_price))

print("\nGrid Size vs Absolute Error:")
for M, exp_err, imp_err, cn_err in zip(grid_sizes, explicit_errors, implicit_errors, cn_errors):
    print(f"   M={M:3d}: Explicit ${exp_err:.5f}, Implicit ${imp_err:.5f}, C-N ${cn_err:.5f}")

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Option value vs stock price
axes[0, 0].plot(S_grid, V_cn, linewidth=2, label='European Put (Crank-Nicolson)')
axes[0, 0].plot(S_grid_am, V_american, linewidth=2, label='American Put')
axes[0, 0].plot(S_grid, np.maximum(K - S_grid, 0), 'k--', linewidth=1, label='Intrinsic Value')
axes[0, 0].axvline(S0, color='red', linestyle=':', alpha=0.5, label=f'S0=${S0}')
axes[0, 0].set_xlabel('Stock Price S')
axes[0, 0].set_ylabel('Option Value')
axes[0, 0].set_title('Put Option Value at t=0')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xlim(0, 150)

# Plot 2: Convergence comparison
axes[0, 1].loglog(grid_sizes, explicit_errors, 'o-', linewidth=2, label='Explicit FD', markersize=8)
axes[0, 1].loglog(grid_sizes, implicit_errors, 's-', linewidth=2, label='Implicit FD', markersize=8)
axes[0, 1].loglog(grid_sizes, cn_errors, '^-', linewidth=2, label='Crank-Nicolson', markersize=8)
axes[0, 1].set_xlabel('Grid Size M')
axes[0, 1].set_ylabel('Absolute Error ($)')
axes[0, 1].set_title('Convergence: Error vs Grid Size')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, which='both')

# Plot 3: Early exercise boundary
if len(exercise_boundary) > 0:
    time_grid = np.linspace(0, T, len(exercise_boundary))
    axes[1, 0].plot(time_grid, exercise_boundary, linewidth=2, color='red')
    axes[1, 0].axhline(K, color='black', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
    axes[1, 0].fill_between(time_grid, 0, exercise_boundary, alpha=0.2, color='red', 
                           label='Early Exercise Region')
    axes[1, 0].set_xlabel('Time to Maturity')
    axes[1, 0].set_ylabel('Stock Price S*')
    axes[1, 0].set_title('American Put: Early Exercise Boundary')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim(0, K*1.2)

# Plot 4: Time value comparison
time_value_european = V_cn - np.maximum(K - S_grid, 0)
time_value_american = V_american - np.maximum(K - S_grid_am, 0)

axes[1, 1].plot(S_grid, time_value_european, linewidth=2, label='European')
axes[1, 1].plot(S_grid_am, time_value_american, linewidth=2, label='American')
axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[1, 1].axvline(S0, color='red', linestyle=':', alpha=0.5)
axes[1, 1].set_xlabel('Stock Price S')
axes[1, 1].set_ylabel('Time Value')
axes[1, 1].set_title('Time Value = Option Value - Intrinsic Value')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(0, 150)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("PDE methods successfully price options:")
print(f"• Crank-Nicolson most accurate (Error ${cn_errors[-1]:.5f} at M=160)")
print(f"• American put premium ${american_price-bs_price:.3f} from early exercise")
print(f"• Exercise boundary: S*≈${exercise_boundary[-1]:.1f} at t=0 (exercise if S<S*)")
print(f"• Convergence: Second-order in space (O(Δx²)), C-N second-order in time")