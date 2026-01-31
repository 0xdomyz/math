
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# European call (benchmark)
def european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo barrier option pricing
def mc_barrier_call(S0, K, B, T, r, sigma, n_paths, n_steps, barrier_type='up-and-out', rebate=0):
    """
    barrier_type: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    """
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(n_steps):
        Z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Check barrier breach
    if 'up' in barrier_type:
        breached = np.max(paths, axis=1) >= B
    else:  # down
        breached = np.min(paths, axis=1) <= B
    
    # Compute payoffs
    terminal_payoffs = np.maximum(paths[:, -1] - K, 0)
    
    if 'out' in barrier_type:
        payoffs = np.where(breached, rebate, terminal_payoffs)
    else:  # knock-in
        payoffs = np.where(breached, terminal_payoffs, rebate)
    
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error, paths, breached

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.25
B_up = 120.0
B_down = 80.0
n_steps = 252  # Daily monitoring

print("="*80)
print("BARRIER OPTIONS PRICING")
print("="*80)
print(f"S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%")
print(f"Barriers: Up=${B_up}, Down=${B_down}, Steps={n_steps}\n")

# European benchmark
euro_call = european_call(S0, K, T, r, sigma)
print(f"European Call: ${euro_call:.6f}")

# Barrier options
np.random.seed(42)
n_paths = 50000

# Up-and-out
uo_price, uo_error, uo_paths, uo_breached = mc_barrier_call(
    S0, K, B_up, T, r, sigma, n_paths, n_steps, 'up-and-out'
)
print(f"\nUp-and-Out Call (B=${B_up}):    ${uo_price:.6f} ± ${uo_error:.6f}")
print(f"  Breach Rate: {np.sum(uo_breached)/n_paths*100:.2f}%")

# Up-and-in
ui_price, ui_error, _, ui_breached = mc_barrier_call(
    S0, K, B_up, T, r, sigma, n_paths, n_steps, 'up-and-in'
)
print(f"Up-and-In Call (B=${B_up}):     ${ui_price:.6f} ± ${ui_error:.6f}")
print(f"  Breach Rate: {np.sum(ui_breached)/n_paths*100:.2f}%")

# Verify parity: UO + UI = European
print(f"  Parity Check: UO + UI = ${uo_price + ui_price:.6f} vs Euro ${euro_call:.6f}")

# Down-and-out
do_price, do_error, do_paths, do_breached = mc_barrier_call(
    S0, K, B_down, T, r, sigma, n_paths, n_steps, 'down-and-out'
)
print(f"\nDown-and-Out Call (B=${B_down}):  ${do_price:.6f} ± ${do_error:.6f}")
print(f"  Breach Rate: {np.sum(do_breached)/n_paths*100:.2f}%")

# Down-and-in
di_price, di_error, _, di_breached = mc_barrier_call(
    S0, K, B_down, T, r, sigma, n_paths, n_steps, 'down-and-in'
)
print(f"Down-and-In Call (B=${B_down}):   ${di_price:.6f} ± ${di_error:.6f}")
print(f"  Parity Check: DO + DI = ${do_price + di_price:.6f} vs Euro ${euro_call:.6f}")

# With rebate
uo_rebate_price, _, _, _ = mc_barrier_call(
    S0, K, B_up, T, r, sigma, n_paths, n_steps, 'up-and-out', rebate=5.0
)
print(f"\nUp-and-Out with $5 Rebate: ${uo_rebate_price:.6f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample paths (up-and-out)
ax = axes[0, 0]
n_plot = 30
time_grid = np.linspace(0, T, n_steps + 1)
for i in range(n_plot):
    color = 'red' if uo_breached[i] else 'blue'
    alpha = 0.4 if uo_breached[i] else 0.2
    ax.plot(time_grid, uo_paths[i, :], color=color, alpha=alpha, linewidth=0.8)

ax.axhline(B_up, color='orange', linestyle='--', linewidth=2, label=f'Barrier ${B_up}')
ax.axhline(K, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Strike ${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price S')
ax.set_title('Up-and-Out: Red=Knocked Out, Blue=Survives')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Sample paths (down-and-out)
ax = axes[0, 1]
for i in range(n_plot):
    color = 'red' if do_breached[i] else 'blue'
    alpha = 0.4 if do_breached[i] else 0.2
    ax.plot(time_grid, do_paths[i, :], color=color, alpha=alpha, linewidth=0.8)

ax.axhline(B_down, color='orange', linestyle='--', linewidth=2, label=f'Barrier ${B_down}')
ax.axhline(K, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price S')
ax.set_title('Down-and-Out: Red=Knocked Out, Blue=Survives')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Barrier breach times
breach_times_uo = []
for i in range(n_paths):
    if uo_breached[i]:
        breach_time = np.argmax(uo_paths[i, :] >= B_up) * T / n_steps
        breach_times_uo.append(breach_time)

ax = axes[0, 2]
if breach_times_uo:
    ax.hist(breach_times_uo, bins=30, edgecolor='black', alpha=0.7, color='red')
    ax.set_xlabel('Breach Time (years)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Up-and-Out Breach Times ({len(breach_times_uo)} breaches)')
    ax.grid(True, alpha=0.3)

# Plot 4: Value vs barrier level
barriers_up = np.linspace(110, 150, 15)
prices_uo = []
prices_ui = []

for B in barriers_up:
    np.random.seed(42)
    p_uo, _, _, _ = mc_barrier_call(S0, K, B, T, r, sigma, 10000, n_steps, 'up-and-out')
    p_ui, _, _, _ = mc_barrier_call(S0, K, B, T, r, sigma, 10000, n_steps, 'up-and-in')
    prices_uo.append(p_uo)
    prices_ui.append(p_ui)

ax = axes[1, 0]
ax.plot(barriers_up, prices_uo, 'ro-', linewidth=2, label='Up-and-Out')
ax.plot(barriers_up, prices_ui, 'bo-', linewidth=2, label='Up-and-In')
ax.axhline(euro_call, color='green', linestyle='--', linewidth=2, label='European')
ax.set_xlabel('Barrier Level B')
ax.set_ylabel('Option Price ($)')
ax.set_title('Barrier Option Value vs Barrier Level')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Value vs spot price
spots = np.linspace(80, 120, 20)
prices_uo_spot = []
prices_euro_spot = []

for S in spots:
    np.random.seed(42)
    p_uo, _, _, _ = mc_barrier_call(S, K, B_up, T, r, sigma, 10000, n_steps, 'up-and-out')
    prices_uo_spot.append(p_uo)
    prices_euro_spot.append(european_call(S, K, T, r, sigma))

ax = axes[1, 1]
ax.plot(spots, prices_euro_spot, 'g-', linewidth=2, label='European')
ax.plot(spots, prices_uo_spot, 'r-', linewidth=2, label='Up-and-Out')
ax.axvline(B_up, color='orange', linestyle='--', linewidth=2, alpha=0.5, label=f'Barrier ${B_up}')
ax.axvline(K, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Option Price ($)')
ax.set_title('Option Value vs Spot (Barrier Effects)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Comparison of all barrier types
barrier_types = ['European', 'Up-Out', 'Up-In', 'Down-Out', 'Down-In']
prices = [euro_call, uo_price, ui_price, do_price, di_price]
colors = ['green', 'red', 'blue', 'orange', 'purple']

ax = axes[1, 2]
bars = ax.bar(range(len(barrier_types)), prices, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(barrier_types)))
ax.set_xticklabels(barrier_types, rotation=15, ha='right')
ax.set_ylabel('Option Price ($)')
ax.set_title('Barrier Option Price Comparison')
ax.grid(True, axis='y', alpha=0.3)

for bar, price in zip(bars, prices):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${price:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('barrier_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()