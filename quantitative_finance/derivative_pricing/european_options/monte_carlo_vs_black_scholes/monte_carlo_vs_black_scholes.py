
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# Black-Scholes formula
def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo European call
def mc_european_call(S0, K, T, r, sigma, n_paths, antithetic=False):
    """Monte Carlo simulation for European call."""
    if antithetic:
        n_half = n_paths // 2
        Z = np.random.randn(n_half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = np.random.randn(n_paths)
    
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z_full
    ST = S0 * np.exp(drift + diffusion)
    
    payoffs = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error

# Monte Carlo Asian call (arithmetic average)
def mc_asian_call(S0, K, T, r, sigma, n_paths, n_steps):
    """
    Asian call: Payoff = max(Average(S) - K, 0).
    No closed-form BS solution; MC required.
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.randn(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt 
                                              + sigma * np.sqrt(dt) * Z)
    
    # Arithmetic average of path
    avg_prices = np.mean(paths, axis=1)
    payoffs = np.maximum(avg_prices - K, 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error

# Monte Carlo barrier call (up-and-out)
def mc_barrier_call(S0, K, B, T, r, sigma, n_paths, n_steps):
    """
    Up-and-out barrier call: Payoff = max(S_T - K, 0) if S_t < B for all t.
    Otherwise payoff = 0 (knocked out).
    """
    dt = T / n_steps
    knocked_out = np.zeros(n_paths, dtype=bool)
    ST = np.ones(n_paths) * S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.randn(n_paths)
        ST = ST * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Check barrier breach
        knocked_out |= (ST >= B)
    
    # Payoff only if not knocked out
    payoffs = np.where(knocked_out, 0, np.maximum(ST - K, 0))
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.25

print("="*80)
print("EUROPEAN CALL: BLACK-SCHOLES vs MONTE CARLO")
print("="*80)

# BS price (exact)
start_time = time.time()
bs_price = bs_call(S0, K, T, r, sigma)
bs_time = time.time() - start_time

print(f"Black-Scholes Price: ${bs_price:.6f}")
print(f"Computation Time: {bs_time*1e6:.2f} microseconds\n")

# MC convergence analysis
path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
mc_prices = []
mc_errors = []
mc_times = []

np.random.seed(42)
for n in path_counts:
    start_time = time.time()
    price, error = mc_european_call(S0, K, T, r, sigma, n, antithetic=True)
    elapsed = time.time() - start_time
    
    mc_prices.append(price)
    mc_errors.append(error)
    mc_times.append(elapsed)
    
    print(f"MC ({n:>6} paths): ${price:.6f} ± ${error:.6f}  "
          f"[Error vs BS: ${abs(price - bs_price):.6f}]  "
          f"Time: {elapsed*1000:.2f}ms")

# Exotic options (no BS closed-form)
print("\n" + "="*80)
print("EXOTIC OPTIONS (Monte Carlo Only)")
print("="*80)

np.random.seed(42)
n_paths_exotic = 50000
n_steps = 252  # Daily monitoring

# Asian call
asian_price, asian_error = mc_asian_call(S0, K, T, r, sigma, n_paths_exotic, n_steps)
print(f"\nAsian Call (Arithmetic Average):")
print(f"  Price: ${asian_price:.6f} ± ${asian_error:.6f}")
print(f"  Paths: {n_paths_exotic:,}, Steps: {n_steps}")

# Barrier call (up-and-out at 120)
barrier = 120.0
barrier_price, barrier_error = mc_barrier_call(S0, K, barrier, T, r, sigma, 
                                                n_paths_exotic, n_steps)
print(f"\nUp-and-Out Barrier Call (Barrier at ${barrier}):")
print(f"  Price: ${barrier_price:.6f} ± ${barrier_error:.6f}")
print(f"  Paths: {n_paths_exotic:,}, Steps: {n_steps}")

# Standard European call for comparison
euro_price, euro_error = mc_european_call(S0, K, T, r, sigma, n_paths_exotic, antithetic=True)
print(f"\nEuropean Call (same # paths):")
print(f"  MC Price: ${euro_price:.6f} ± ${euro_error:.6f}")
print(f"  BS Price: ${bs_price:.6f}")
print(f"  Difference: ${abs(euro_price - bs_price):.6f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: MC convergence to BS
ax = axes[0, 0]
ax.semilogx(path_counts, mc_prices, 'bo-', linewidth=2, markersize=8, label='MC Price')
ax.axhline(bs_price, color='red', linestyle='--', linewidth=2, label=f'BS: ${bs_price:.4f}')
ax.fill_between(path_counts, 
                np.array(mc_prices) - 1.96*np.array(mc_errors),
                np.array(mc_prices) + 1.96*np.array(mc_errors),
                alpha=0.3, label='95% CI')
ax.set_xlabel('Number of MC Paths')
ax.set_ylabel('Option Price ($)')
ax.set_title('MC Convergence to BS (European Call)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Standard error vs paths
ax = axes[0, 1]
ax.loglog(path_counts, mc_errors, 'go-', linewidth=2, markersize=8, label='MC Std Error')
theoretical_line = mc_errors[0] * np.sqrt(path_counts[0]) / np.sqrt(np.array(path_counts))
ax.loglog(path_counts, theoretical_line, 'k--', linewidth=2, label='O(1/√N)')
ax.set_xlabel('Number of MC Paths')
ax.set_ylabel('Standard Error ($)')
ax.set_title('MC Standard Error (O(1/√N) Convergence)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Computation time comparison
ax = axes[0, 2]
ax.semilogx(path_counts, np.array(mc_times) * 1000, 'mo-', linewidth=2, markersize=8, 
            label='MC Time')
ax.axhline(bs_time * 1000, color='red', linestyle='--', linewidth=2, label='BS Time')
ax.set_xlabel('Number of MC Paths')
ax.set_ylabel('Computation Time (ms)')
ax.set_title('Computation Speed: MC vs BS')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Plot 4: Price comparison (European, Asian, Barrier)
option_types = ['European\nCall', 'Asian\nCall', 'Barrier\nCall\n(B=$120)']
prices = [euro_price, asian_price, barrier_price]
errors = [euro_error, asian_error, barrier_error]

ax = axes[1, 0]
bars = ax.bar(option_types, prices, yerr=np.array(errors)*1.96, capsize=10, 
              color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
ax.axhline(bs_price, color='red', linestyle='--', linewidth=2, label=f'BS European: ${bs_price:.2f}')
ax.set_ylabel('Option Price ($)')
ax.set_title(f'Option Price Comparison (N={n_paths_exotic:,} paths)')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Add price labels on bars
for i, (bar, price, error) in enumerate(zip(bars, prices, errors)):
    ax.text(bar.get_x() + bar.get_width()/2, price + error*2, 
            f'${price:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 5: Accuracy vs computation time trade-off
ax = axes[1, 1]
ax.scatter(np.array(mc_times) * 1000, mc_errors, s=150, c=np.log10(path_counts), 
           cmap='viridis', edgecolor='black', linewidth=1.5)
for i, n in enumerate(path_counts):
    ax.annotate(f'{n}', (mc_times[i]*1000, mc_errors[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Computation Time (ms)')
ax.set_ylabel('Standard Error ($)')
ax.set_title('Accuracy vs Speed Trade-off')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('log₁₀(Paths)')

# Plot 6: When to use MC vs BS (decision matrix)
ax = axes[1, 2]
ax.axis('off')

decision_text = """
WHEN TO USE BLACK-SCHOLES:
✓ European vanilla calls/puts
✓ Need instant pricing (microseconds)
✓ Closed-form Greeks required
✓ Single asset, constant volatility
✓ High-frequency trading applications

WHEN TO USE MONTE CARLO:
✓ Path-dependent payoffs (Asian, lookback)
✓ Barrier options (knock-in/out)
✓ Multi-asset options (basket, spread)
✓ Complex models (jumps, stoch-vol)
✓ Exotic/structured products
✓ American options (with LSM)
✓ 4+ correlated assets

COMPUTATIONAL TRADE-OFF:
• BS: O(1) - instant
• MC: O(N) - linear in paths
• Error: MC ~ 1/√N convergence
• For 0.1% accuracy: Need ~1M paths
• Variance reduction: 5-10× speedup
"""

ax.text(0.05, 0.95, decision_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('mc_vs_bs_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical test: Is MC price significantly different from BS?
z_score = (euro_price - bs_price) / euro_error
p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test

print("\n" + "="*80)
print("STATISTICAL COMPARISON")
print("="*80)
print(f"MC Price: ${euro_price:.6f} ± ${euro_error:.6f}")
print(f"BS Price: ${bs_price:.6f}")
print(f"Difference: ${euro_price - bs_price:.6f}")
print(f"Z-Score: {z_score:.3f}")
print(f"P-Value: {p_value:.4f}")
if p_value > 0.05:
    print("✓ MC price not significantly different from BS (p > 0.05)")
else:
    print("✗ MC price significantly different from BS (p ≤ 0.05)")

# Efficiency comparison: Paths needed for target accuracy
target_errors = [0.01, 0.005, 0.001]  # $0.01, $0.005, $0.001
print("\n" + "="*80)
print("PATHS REQUIRED FOR TARGET ACCURACY")
print("="*80)
base_error = mc_errors[-1]  # Error at 100k paths
base_paths = path_counts[-1]

for target_error in target_errors:
    # Error ~ 1/√N → paths ~ (base_error / target_error)²
    required_paths = int(base_paths * (base_error / target_error)**2)
    estimated_time = mc_times[-1] * (required_paths / base_paths)
    print(f"Target Error: ${target_error:.4f}")
    print(f"  Required Paths: {required_paths:,}")
    print(f"  Estimated Time: {estimated_time:.3f} seconds")
    print(f"  vs BS Time: {bs_time*1e6:.2f} microseconds (×{estimated_time/bs_time:.0f})\n")