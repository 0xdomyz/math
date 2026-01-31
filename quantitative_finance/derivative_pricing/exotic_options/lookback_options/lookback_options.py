
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# European call (benchmark)
def european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo lookback option pricing
def mc_lookback_call(S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False):
    """
    Price lookback call option.
    
    Parameters:
    - floating_strike: If True, floating strike (payoff = S_T - min);
                       If False, fixed strike (payoff = max - K)
    
    Returns:
    - price: Option value
    - std_error: Standard error
    - paths: Simulated price paths
    - maxima: Maximum prices per path
    - minima: Minimum prices per path
    """
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    # Generate paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(n_steps):
        Z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt 
                                              + sigma * np.sqrt(dt) * Z)
    
    # Track extrema
    maxima = np.max(paths, axis=1)
    minima = np.min(paths, axis=1)
    terminal = paths[:, -1]
    
    # Compute payoffs
    if floating_strike:
        # Floating strike call: S_T - min
        payoffs = terminal - minima
    else:
        # Fixed strike call: max(max - K, 0)
        payoffs = np.maximum(maxima - K, 0)
    
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error, paths, maxima, minima

def mc_lookback_put(S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False):
    """Price lookback put option."""
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(n_steps):
        Z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt 
                                              + sigma * np.sqrt(dt) * Z)
    
    maxima = np.max(paths, axis=1)
    minima = np.min(paths, axis=1)
    terminal = paths[:, -1]
    
    if floating_strike:
        # Floating strike put: max - S_T
        payoffs = maxima - terminal
    else:
        # Fixed strike put: max(K - min, 0)
        payoffs = np.maximum(K - minima, 0)
    
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error, paths, maxima, minima

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.30
n_steps = 252  # Daily monitoring

print("="*80)
print("LOOKBACK OPTIONS PRICING")
print("="*80)
print(f"S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%\n")

# European benchmark
euro_call = european_call(S0, K, T, r, sigma)
print(f"European Call: ${euro_call:.6f}")

# Lookback options
np.random.seed(42)
n_paths = 50000

# Fixed strike lookback call
fixed_call, fixed_call_err, paths_call, maxima_call, minima_call = mc_lookback_call(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False
)
print(f"\nFixed Strike Lookback Call:    ${fixed_call:.6f} ± ${fixed_call_err:.6f}")
print(f"  Premium over European: ${fixed_call - euro_call:.6f} ({(fixed_call/euro_call - 1)*100:.1f}%)")

# Floating strike lookback call
float_call, float_call_err, _, _, _ = mc_lookback_call(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=True
)
print(f"Floating Strike Lookback Call: ${float_call:.6f} ± ${float_call_err:.6f}")
print(f"  Premium over Fixed: ${float_call - fixed_call:.6f} ({(float_call/fixed_call - 1)*100:.1f}%)")

# Fixed strike lookback put
fixed_put, fixed_put_err, paths_put, maxima_put, minima_put = mc_lookback_put(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False
)
print(f"\nFixed Strike Lookback Put:     ${fixed_put:.6f} ± ${fixed_put_err:.6f}")

# Floating strike lookback put
float_put, float_put_err, _, _, _ = mc_lookback_put(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=True
)
print(f"Floating Strike Lookback Put:  ${float_put:.6f} ± ${float_put_err:.6f}")

# Statistics on extrema
print("\n" + "="*80)
print("EXTREMA STATISTICS")
print("="*80)
print(f"Average Maximum: ${np.mean(maxima_call):.2f}")
print(f"Average Minimum: ${np.mean(minima_call):.2f}")
print(f"Average Range:   ${np.mean(maxima_call - minima_call):.2f}")
print(f"Average Terminal: ${np.mean(paths_call[:, -1]):.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample paths with extrema
ax = axes[0, 0]
n_plot = 20
time_grid = np.linspace(0, T, n_steps + 1)

for i in range(n_plot):
    ax.plot(time_grid, paths_call[i, :], 'b-', alpha=0.4, linewidth=1)
    
    # Mark max and min
    max_idx = np.argmax(paths_call[i, :])
    min_idx = np.argmin(paths_call[i, :])
    ax.scatter(time_grid[max_idx], paths_call[i, max_idx], 
              color='green', s=50, zorder=5, alpha=0.8)
    ax.scatter(time_grid[min_idx], paths_call[i, min_idx],
              color='red', s=50, zorder=5, alpha=0.8)

ax.axhline(K, color='orange', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price S')
ax.set_title('Sample Paths (Green=Max, Red=Min)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution of maxima and minima
ax = axes[0, 1]
ax.hist(maxima_call, bins=50, alpha=0.6, color='green', edgecolor='black',
        density=True, label='Maximum')
ax.hist(minima_call, bins=50, alpha=0.6, color='red', edgecolor='black',
        density=True, label='Minimum')
ax.axvline(S0, color='blue', linestyle='--', linewidth=2, label=f'S₀=${S0}')
ax.axvline(np.mean(maxima_call), color='green', linestyle=':', linewidth=2,
           label=f'Mean Max=${np.mean(maxima_call):.0f}')
ax.axvline(np.mean(minima_call), color='red', linestyle=':', linewidth=2,
           label=f'Mean Min=${np.mean(minima_call):.0f}')
ax.set_xlabel('Price')
ax.set_ylabel('Density')
ax.set_title('Distribution of Extrema')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Payoff distributions
ax = axes[0, 2]
fixed_payoffs = np.maximum(maxima_call - K, 0)
float_payoffs = paths_call[:, -1] - minima_call

ax.hist(fixed_payoffs, bins=50, alpha=0.6, color='blue', edgecolor='black',
        label='Fixed Strike')
ax.hist(float_payoffs, bins=50, alpha=0.6, color='orange', edgecolor='black',
        label='Floating Strike')
ax.set_xlabel('Payoff at Maturity')
ax.set_ylabel('Frequency')
ax.set_title('Lookback Call Payoff Distributions')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Value vs volatility
sigmas = np.linspace(0.10, 0.60, 15)
euro_vals = []
fixed_vals = []
float_vals = []

for sig in sigmas:
    euro_vals.append(european_call(S0, K, T, r, sig))
    
    np.random.seed(42)
    fixed_price, _, _, _, _ = mc_lookback_call(S0, K, T, r, sig, 10000, n_steps, False)
    float_price, _, _, _, _ = mc_lookback_call(S0, K, T, r, sig, 10000, n_steps, True)
    
    fixed_vals.append(fixed_price)
    float_vals.append(float_price)

ax = axes[1, 0]
ax.plot(sigmas * 100, euro_vals, 'g-', linewidth=2, label='European')
ax.plot(sigmas * 100, fixed_vals, 'b-', linewidth=2, label='Fixed Strike Lookback')
ax.plot(sigmas * 100, float_vals, 'r-', linewidth=2, label='Floating Strike Lookback')
ax.set_xlabel('Volatility σ (%)')
ax.set_ylabel('Option Price ($)')
ax.set_title('Lookback Vega: Value vs Volatility')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Value vs spot price
spots = np.linspace(80, 120, 20)
euro_spot = []
fixed_spot = []
float_spot = []

for S in spots:
    euro_spot.append(european_call(S, K, T, r, sigma))
    
    np.random.seed(42)
    fixed_price, _, _, _, _ = mc_lookback_call(S, K, T, r, sigma, 10000, n_steps, False)
    float_price, _, _, _, _ = mc_lookback_call(S, K, T, r, sigma, 10000, n_steps, True)
    
    fixed_spot.append(fixed_price)
    float_spot.append(float_price)

ax = axes[1, 1]
ax.plot(spots, euro_spot, 'g-', linewidth=2, label='European')
ax.plot(spots, fixed_spot, 'b-', linewidth=2, label='Fixed Strike Lookback')
ax.plot(spots, float_spot, 'r-', linewidth=2, label='Floating Strike Lookback')
ax.axvline(K, color='orange', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Option Price ($)')
ax.set_title('Lookback Delta: Value vs Spot')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Comparison bar chart
option_types = ['European\nCall', 'Fixed\nLookback\nCall', 'Floating\nLookback\nCall']
prices = [euro_call, fixed_call, float_call]
colors = ['green', 'blue', 'red']

ax = axes[1, 2]
bars = ax.bar(range(len(option_types)), prices, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(option_types)))
ax.set_xticklabels(option_types)
ax.set_ylabel('Option Price ($)')
ax.set_title('Option Price Comparison')
ax.grid(True, axis='y', alpha=0.3)

for bar, price in zip(bars, prices):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${price:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('lookback_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Range analysis
ranges = maxima_call - minima_call
print("\n" + "="*80)
print("RANGE ANALYSIS (Max - Min)")
print("="*80)
print(f"Mean Range:   ${np.mean(ranges):.2f}")
print(f"Median Range: ${np.median(ranges):.2f}")
print(f"Std Dev:      ${np.std(ranges):.2f}")
print(f"Min Range:    ${np.min(ranges):.2f}")
print(f"Max Range:    ${np.max(ranges):.2f}")

# Hindsight perfection analysis
perfect_buy_low = minima_call
perfect_sell_high = maxima_call
hindsight_profit = perfect_sell_high - perfect_buy_low

print(f"\nHindsight Perfect Trading:")
print(f"  Average Profit: ${np.mean(hindsight_profit):.2f}")
print(f"  This equals Floating Strike Call payoff")