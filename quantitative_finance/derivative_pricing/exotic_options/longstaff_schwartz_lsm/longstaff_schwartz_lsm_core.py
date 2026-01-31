import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Block 1

# Black-Scholes European put (benchmark)
def bs_put(S, K, T, r, sigma):
    """European put option price."""
    if T <= 0:
        return np.maximum(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Binomial tree American put (benchmark)
def binomial_american_put(S0, K, T, r, sigma, N):
    """American put via binomial tree (exact as N→∞)."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    prices = np.zeros(N + 1)
    for i in range(N + 1):
        prices[i] = S0 * (u ** (N - i)) * (d ** i)
    
    # Initialize option values at maturity
    values = np.maximum(K - prices, 0)
    
    # Backward induction
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            price = S0 * (u ** (step - i)) * (d ** i)
            hold_value = np.exp(-r * dt) * (p * values[i] + (1 - p) * values[i + 1])
            exercise_value = max(K - price, 0)
            values[i] = max(hold_value, exercise_value)
    
    return values[0]

# Longstaff-Schwartz Algorithm
def longstaff_schwartz_put(S0, K, T, r, sigma, n_paths, n_steps, poly_degree=2):
    """
    LSM algorithm for American put option.
    
    Parameters:
    - poly_degree: Degree of polynomial basis (typically 2-3)
    
    Returns:
    - option_value: LSM estimated price
    - std_error: Standard error
    - exercise_boundary: Early exercise boundary at each time step
    """
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Step 1: Generate paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.randn(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt 
                                              + sigma * np.sqrt(dt) * Z)
    
    # Step 2: Initialize cash flows at maturity
    cash_flows = np.maximum(K - paths[:, -1], 0)
    exercise_times = np.full(n_paths, n_steps)  # Track exercise time for each path
    
    # Store exercise boundary
    exercise_boundary = []
    
    # Step 3: Backward induction
    for t in range(n_steps - 1, 0, -1):
        # Current stock prices
        S_t = paths[:, t]
        
        # Immediate exercise value
        intrinsic = np.maximum(K - S_t, 0)
        
        # Identify ITM paths
        itm = intrinsic > 0
        
        if np.sum(itm) > 0:
            # Regression on ITM paths only
            X = S_t[itm]
            Y = cash_flows[itm] * discount
            
            # Polynomial basis functions: 1, S, S^2, ..., S^poly_degree
            basis = np.column_stack([X**i for i in range(poly_degree + 1)])
            
            # Least-squares regression
            coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
            
            # Predicted continuation value for all ITM paths
            continuation_value = basis @ coeffs
            
            # Decision: exercise if intrinsic > continuation
            exercise = intrinsic[itm] > continuation_value
            
            # Update cash flows
            cash_flows[itm] = np.where(exercise, intrinsic[itm], cash_flows[itm] * discount)
            
            # Update exercise times
            exercise_times[itm] = np.where(exercise, t, exercise_times[itm])
            
            # Record exercise boundary (average S where exercised)
            if np.sum(exercise) > 0:
                exercise_boundary.append((t * dt, np.mean(X[exercise])))
        
        # Discount cash flows for OTM paths (no early exercise)
        cash_flows[~itm] *= discount
    
    # Step 4: Discount to present
    present_values = cash_flows * discount  # One more discount from t=1 to t=0
    
    option_value = np.mean(present_values)
    std_error = np.std(present_values) / np.sqrt(n_paths)
    
    return option_value, std_error, exercise_boundary, exercise_times, paths

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20

print("="*80)
print("AMERICAN PUT OPTION: LONGSTAFF-SCHWARTZ vs BINOMIAL")
print("="*80)
print(f"Parameters: S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%\n")

# European put (lower bound)
euro_put = bs_put(S0, K, T, r, sigma)
print(f"European Put (BS):     ${euro_put:.6f}")

# Binomial tree (benchmark for American)
binom_put = binomial_american_put(S0, K, T, r, sigma, N=500)
print(f"American Put (Binomial N=500): ${binom_put:.6f}")

# LSM with varying parameters
np.random.seed(42)
n_paths = 10000
n_steps = 50

lsm_results = []
for poly_deg in [2, 3, 4]:
    lsm_value, lsm_error, boundary, ex_times, paths = longstaff_schwartz_put(
        S0, K, T, r, sigma, n_paths, n_steps, poly_degree=poly_deg
    )
    lsm_results.append((poly_deg, lsm_value, lsm_error))
    print(f"American Put (LSM poly={poly_deg}):    ${lsm_value:.6f} ± ${lsm_error:.6f}")

print(f"\nEarly Exercise Premium: ${binom_put - euro_put:.6f}")

# Convergence analysis
print("\n" + "="*80)
print("LSM CONVERGENCE ANALYSIS (Polynomial Degree = 2)")
print("="*80)

path_counts = [1000, 2000, 5000, 10000, 20000, 50000]
lsm_prices = []
lsm_errors = []

for n in path_counts:
    np.random.seed(42)
    price, error, _, _, _ = longstaff_schwartz_put(S0, K, T, r, sigma, n, n_steps, poly_degree=2)
    lsm_prices.append(price)
    lsm_errors.append(error)
    print(f"N={n:>6}: ${price:.6f} ± ${error:.6f}  "
          f"[Error vs Binomial: ${abs(price - binom_put):.6f}]")

# Visualization
np.random.seed(42)
lsm_value, lsm_error, boundary, ex_times, paths_vis = longstaff_schwartz_put(
    S0, K, T, r, sigma, 5000, 50, poly_degree=2
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample paths with early exercise
ax = axes[0, 0]
n_plot = 50
for i in range(n_plot):
    time_grid = np.linspace(0, T, n_steps + 1)
    color = 'red' if ex_times[i] < n_steps else 'blue'
    alpha = 0.3 if ex_times[i] < n_steps else 0.1
    ax.plot(time_grid, paths_vis[i, :], color=color, alpha=alpha, linewidth=0.8)
    
    # Mark exercise point
    if ex_times[i] < n_steps:
        ax.scatter(ex_times[i] * T / n_steps, paths_vis[i, ex_times[i]], 
                  color='red', s=20, zorder=5)

ax.axhline(K, color='green', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price S')
ax.set_title(f'Sample Paths (Red=Early Exercise, Blue=Hold to Maturity)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Early exercise boundary
if boundary:
    boundary_times, boundary_prices = zip(*boundary)
    ax = axes[0, 1]
    ax.scatter(boundary_times, boundary_prices, color='red', s=50, alpha=0.6, 
              label='Exercise Points')
    
    # Fit smooth curve to boundary
    if len(boundary_times) > 3:
        z = np.polyfit(boundary_times, boundary_prices, deg=3)
        p = np.poly1d(z)
        t_smooth = np.linspace(min(boundary_times), max(boundary_times), 100)
        ax.plot(t_smooth, p(t_smooth), 'r-', linewidth=2, label='Fitted Boundary')
    
    ax.axhline(K, color='green', linestyle='--', linewidth=2, label=f'Strike K=${K}')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Stock Price S')
    ax.set_title('Early Exercise Boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'No Early Exercise', ha='center', va='center', 
                    transform=axes[0, 1].transAxes)

# Plot 3: Exercise time distribution
ax = axes[0, 2]
exercised_early = ex_times < n_steps
ax.hist(ex_times[exercised_early] * T / n_steps, bins=30, 
        edgecolor='black', alpha=0.7, color='red')
ax.set_xlabel('Exercise Time (years)')
ax.set_ylabel('Frequency')
ax.set_title(f'Early Exercise Times ({np.sum(exercised_early)/len(ex_times)*100:.1f}% exercised)')
ax.grid(True, alpha=0.3)

# Plot 4: LSM convergence vs binomial
ax = axes[1, 0]
ax.semilogx(path_counts, lsm_prices, 'bo-', linewidth=2, markersize=8, label='LSM Price')
ax.axhline(binom_put, color='red', linestyle='--', linewidth=2, label=f'Binomial: ${binom_put:.4f}')
ax.axhline(euro_put, color='green', linestyle='--', linewidth=2, label=f'European: ${euro_put:.4f}')
ax.fill_between(path_counts,
                np.array(lsm_prices) - 1.96*np.array(lsm_errors),
                np.array(lsm_prices) + 1.96*np.array(lsm_errors),
                alpha=0.3, label='95% CI')
ax.set_xlabel('Number of MC Paths')
ax.set_ylabel('Option Price ($)')
ax.set_title('LSM Convergence to Binomial Benchmark')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Polynomial degree comparison
poly_degrees = [deg for deg, _, _ in lsm_results]
poly_prices = [price for _, price, _ in lsm_results]
poly_errors = [error for _, _, error in lsm_results]

ax = axes[1, 1]
ax.bar(poly_degrees, poly_prices, yerr=np.array(poly_errors)*1.96, capsize=10,
       color='purple', alpha=0.7, edgecolor='black')
ax.axhline(binom_put, color='red', linestyle='--', linewidth=2, label=f'Binomial: ${binom_put:.4f}')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Option Price ($)')
ax.set_title('LSM Price vs Basis Function Complexity')
ax.set_xticks(poly_degrees)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Plot 6: Value vs spot price (American premium)
spots = np.linspace(70, 130, 30)
euro_values = [bs_put(S, K, T, r, sigma) for S in spots]
amer_values = []

for S in spots:
    np.random.seed(42)
    price, _, _, _, _ = longstaff_schwartz_put(S, K, T, r, sigma, 5000, 50, poly_degree=2)
    amer_values.append(price)

ax = axes[1, 2]
ax.plot(spots, amer_values, 'r-', linewidth=2, label='American Put (LSM)')
ax.plot(spots, euro_values, 'b--', linewidth=2, label='European Put (BS)')
intrinsic = np.maximum(K - spots, 0)
ax.plot(spots, intrinsic, 'g:', linewidth=2, label='Intrinsic Value')
ax.axvline(K, color='black', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Option Value ($)')
ax.set_title('American vs European Put (Early Exercise Premium)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('longstaff_schwartz_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical analysis
print("\n" + "="*80)
print("EARLY EXERCISE STATISTICS")
print("="*80)
print(f"Total Paths: {n_paths}")
print(f"Early Exercise: {np.sum(exercised_early)} ({np.sum(exercised_early)/n_paths*100:.2f}%)")
print(f"Hold to Maturity: {np.sum(~exercised_early)} ({np.sum(~exercised_early)/n_paths*100:.2f}%)")
if np.sum(exercised_early) > 0:
    print(f"Average Exercise Time: {np.mean(ex_times[exercised_early]) * T / n_steps:.4f} years")
    print(f"Earliest Exercise: {np.min(ex_times[exercised_early]) * T / n_steps:.4f} years")