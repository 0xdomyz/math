
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky

# Black-Scholes European call (benchmark)
def european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo rainbow option pricing
def mc_rainbow_option(S0_vec, K, T, r, corr_matrix, vol_vec, n_paths, option_type='best_of_call'):
    """
    Price rainbow option.
    
    Parameters:
    - option_type: 'best_of_call', 'worst_of_call', 'best_of_put', 'worst_of_put'
    
    Returns:
    - price, std_error, terminal_prices, selection_stats
    """
    n_assets = len(S0_vec)
    discount = np.exp(-r * T)
    
    # Cholesky decomposition
    L = cholesky(corr_matrix, lower=True)
    
    # Generate terminal prices
    terminal_prices = np.zeros((n_paths, n_assets))
    
    Z = np.random.randn(n_paths, n_assets)
    X = Z @ L.T  # Correlated normals
    
    for i in range(n_assets):
        terminal_prices[:, i] = S0_vec[i] * np.exp(
            (r - 0.5 * vol_vec[i]**2) * T + vol_vec[i] * np.sqrt(T) * X[:, i]
        )
    
    # Selection
    if option_type == 'best_of_call':
        selected = np.max(terminal_prices, axis=1)
        payoff = np.maximum(selected - K, 0)
    elif option_type == 'worst_of_call':
        selected = np.min(terminal_prices, axis=1)
        payoff = np.maximum(selected - K, 0)
    elif option_type == 'best_of_put':
        selected = np.min(terminal_prices, axis=1)  # min for put
        payoff = np.maximum(K - selected, 0)
    elif option_type == 'worst_of_put':
        selected = np.max(terminal_prices, axis=1)  # max for put
        payoff = np.maximum(K - selected, 0)
    else:
        raise ValueError("Unknown option type")
    
    price = discount * np.mean(payoff)
    std_error = discount * np.std(payoff) / np.sqrt(n_paths)
    
    # Selection statistics (which asset was best/worst)
    if 'best' in option_type:
        winners = np.argmax(terminal_prices, axis=1)
    else:
        winners = np.argmin(terminal_prices, axis=1)
    
    selection_stats = {
        'winners': winners,
        'winner_counts': np.bincount(winners, minlength=n_assets) / n_paths * 100
    }
    
    return price, std_error, terminal_prices, selection_stats, selected

# Parameters
n_assets = 3
S0_vec = np.array([100.0, 100.0, 100.0])
K = 100.0
T = 1.0
r = 0.05
vol_vec = np.array([0.25, 0.30, 0.35])  # Different volatilities

print("="*80)
print("RAINBOW OPTION PRICING")
print("="*80)
print(f"N={n_assets} assets, S₀={S0_vec}, K=${K}, T={T}yr, r={r*100}%")
print(f"Volatilities: {vol_vec*100}%\n")

# Benchmark: Individual European calls
euro_prices = [european_call(S0_vec[i], K, T, r, vol_vec[i]) for i in range(n_assets)]
print("Individual European Calls:")
for i, price in enumerate(euro_prices):
    print(f"  Asset {i+1}: ${price:.6f}")
print(f"  Sum: ${np.sum(euro_prices):.6f} (upper bound for best-of)\n")

# Test correlation impact
np.random.seed(42)
n_paths = 100000

correlations = [0.0, 0.3, 0.6, 0.9]
option_types = ['best_of_call', 'worst_of_call']

results = {opt: {'prices': [], 'errors': []} for opt in option_types}

print("="*80)
print("CORRELATION IMPACT")
print("="*80)

for rho in correlations:
    # Uniform correlation matrix
    corr_matrix = np.ones((n_assets, n_assets)) * rho
    np.fill_diagonal(corr_matrix, 1.0)
    
    print(f"\nCorrelation ρ={rho:.1f}:")
    
    for opt_type in option_types:
        price, error, _, selection_stats, _ = mc_rainbow_option(
            S0_vec, K, T, r, corr_matrix, vol_vec, n_paths, option_type=opt_type
        )
        
        results[opt_type]['prices'].append(price)
        results[opt_type]['errors'].append(error)
        
        print(f"  {opt_type.replace('_', ' ').title()}: ${price:.6f} ± ${error:.6f}")
        if opt_type == 'best_of_call':
            print(f"    Winner distribution: {selection_stats['winner_counts']}")

# Detailed analysis: ρ=0.5
print("\n" + "="*80)
print("DETAILED ANALYSIS (ρ=0.5)")
print("="*80)

corr_matrix = np.array([[1.0, 0.5, 0.5],
                        [0.5, 1.0, 0.5],
                        [0.5, 0.5, 1.0]])

np.random.seed(42)

# Best-of call
best_price, best_error, term_prices, selection, selected_best = mc_rainbow_option(
    S0_vec, K, T, r, corr_matrix, vol_vec, n_paths, option_type='best_of_call'
)

# Worst-of call
worst_price, worst_error, _, _, selected_worst = mc_rainbow_option(
    S0_vec, K, T, r, corr_matrix, vol_vec, n_paths, option_type='worst_of_call'
)

print(f"Best-of Call: ${best_price:.6f} ± ${best_error:.6f}")
print(f"Worst-of Call: ${worst_price:.6f} ± ${worst_error:.6f}")
print(f"\nWinner Selection (Best-of):")
for i, pct in enumerate(selection['winner_counts']):
    print(f"  Asset {i+1} (σ={vol_vec[i]*100}%): {pct:.1f}% of paths")

# Comparison to bounds
basket = np.mean(term_prices, axis=1)
basket_call = np.mean(np.maximum(basket - K, 0)) * np.exp(-r * T)
sum_calls = np.sum(euro_prices)

print(f"\nBounds:")
print(f"  Basket Call (lower): ${basket_call:.6f}")
print(f"  Best-of Call: ${best_price:.6f}")
print(f"  Sum of Calls (upper): ${sum_calls:.6f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Terminal price distributions
ax = axes[0, 0]
colors = ['blue', 'green', 'red']
for i in range(n_assets):
    ax.hist(term_prices[:, i], bins=50, alpha=0.5, color=colors[i],
           label=f'Asset {i+1} (σ={vol_vec[i]*100}%)', density=True)

ax.axvline(K, color='orange', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Terminal Price')
ax.set_ylabel('Density')
ax.set_title('Terminal Price Distributions (ρ=0.5)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Max vs Min distributions
ax = axes[0, 1]
ax.hist(selected_best, bins=60, alpha=0.6, color='green', label='Max (Best-of)', density=True)
ax.hist(selected_worst, bins=60, alpha=0.6, color='red', label='Min (Worst-of)', density=True)
ax.axvline(K, color='orange', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Selected Price (Max/Min)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Max and Min Terminal Prices')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Payoff distributions
ax = axes[0, 2]
payoff_best = np.maximum(selected_best - K, 0)
payoff_worst = np.maximum(selected_worst - K, 0)

ax.hist(payoff_best, bins=60, alpha=0.6, color='green', label='Best-of Call')
ax.hist(payoff_worst, bins=60, alpha=0.6, color='red', label='Worst-of Call')
ax.set_xlabel('Payoff')
ax.set_ylabel('Frequency')
ax.set_title('Payoff Distributions (ρ=0.5)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Price vs correlation
ax = axes[1, 0]
for opt_type, label, color in [('best_of_call', 'Best-of Call', 'green'),
                                ('worst_of_call', 'Worst-of Call', 'red')]:
    prices = results[opt_type]['prices']
    errors = results[opt_type]['errors']
    
    ax.plot(correlations, prices, 'o-', linewidth=2, markersize=8,
           color=color, label=label)
    ax.fill_between(correlations,
                    np.array(prices) - 1.96*np.array(errors),
                    np.array(prices) + 1.96*np.array(errors),
                    alpha=0.2, color=color)

# Add single-asset benchmark (average vol)
avg_vol = np.mean(vol_vec)
single_asset = european_call(S0_vec[0], K, T, r, avg_vol)
ax.axhline(single_asset, color='blue', linestyle='--', linewidth=2,
          label=f'Single Asset (σ={avg_vol*100:.0f}%)')

ax.set_xlabel('Correlation ρ')
ax.set_ylabel('Option Price ($)')
ax.set_title('Rainbow Option Value vs Correlation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Winner selection pie chart
ax = axes[1, 1]
winner_pcts = selection['winner_counts']
labels = [f'Asset {i+1}\n(σ={vol_vec[i]*100}%)' for i in range(n_assets)]
ax.pie(winner_pcts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Best-of Winner Distribution (ρ=0.5)')

# Plot 6: Comparison bar chart
ax = axes[1, 2]
categories = ['Best-of\nCall', 'Worst-of\nCall', 'Single\nAsset', 'Basket\nCall']
prices_compare = [best_price, worst_price, single_asset, basket_call]
colors_compare = ['green', 'red', 'blue', 'purple']

bars = ax.bar(categories, prices_compare, color=colors_compare, alpha=0.7,
             edgecolor='black')
ax.set_ylabel('Option Price ($)')
ax.set_title('Price Comparison (ρ=0.5)')
ax.grid(True, alpha=0.3, axis='y')

# Annotate
for bar, price in zip(bars, prices_compare):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${price:.4f}',
           ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('rainbow_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation sensitivity analysis
print("\n" + "="*80)
print("CORRELATION SENSITIVITY (CEGA)")
print("="*80)

# Numerical derivative ∂V/∂ρ
rho_base = 0.5
drho = 0.01
np.random.seed(42)

corr_base = np.ones((n_assets, n_assets)) * rho_base
np.fill_diagonal(corr_base, 1.0)

corr_up = np.ones((n_assets, n_assets)) * (rho_base + drho)
np.fill_diagonal(corr_up, 1.0)

for opt_type, label in [('best_of_call', 'Best-of Call'), ('worst_of_call', 'Worst-of Call')]:
    price_base, _, _, _, _ = mc_rainbow_option(S0_vec, K, T, r, corr_base, vol_vec, n_paths, opt_type)
    price_up, _, _, _, _ = mc_rainbow_option(S0_vec, K, T, r, corr_up, vol_vec, n_paths, opt_type)
    
    cega = (price_up - price_base) / drho
    
    print(f"{label}:")
    print(f"  V(ρ={rho_base}): ${price_base:.6f}")
    print(f"  V(ρ={rho_base+drho}): ${price_up:.6f}")
    print(f"  Cega (∂V/∂ρ): ${cega:.4f} per 1% change in ρ\n")