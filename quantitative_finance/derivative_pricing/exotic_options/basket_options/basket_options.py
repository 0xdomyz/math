
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky

# European call (single asset benchmark)
def european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo basket option pricing
def mc_basket_call(S0_vec, weights, K, T, r, corr_matrix, vol_vec, n_paths, n_steps):
    """
    Price basket call option.
    
    Parameters:
    - S0_vec: Initial prices [S₁(0), S₂(0), ..., Sₙ(0)]
    - weights: Basket weights [w₁, w₂, ..., wₙ]
    - corr_matrix: N×N correlation matrix
    - vol_vec: Volatilities [σ₁, σ₂, ..., σₙ]
    
    Returns:
    - price: Option value
    - std_error: Standard error
    - basket_paths: Basket value paths
    - asset_paths: Individual asset paths (dict)
    """
    n_assets = len(S0_vec)
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    # Cholesky decomposition of correlation matrix
    L = cholesky(corr_matrix, lower=True)
    
    # Initialize paths
    asset_paths = {i: np.zeros((n_paths, n_steps + 1)) for i in range(n_assets)}
    for i in range(n_assets):
        asset_paths[i][:, 0] = S0_vec[i]
    
    basket_paths = np.zeros((n_paths, n_steps + 1))
    basket_paths[:, 0] = np.dot(weights, S0_vec)
    
    # Generate correlated paths
    for t in range(n_steps):
        # Independent normals
        Z = np.random.randn(n_paths, n_assets)
        
        # Correlated normals via Cholesky: X = Z @ L^T
        X = Z @ L.T
        
        # Update each asset
        for i in range(n_assets):
            asset_paths[i][:, t+1] = asset_paths[i][:, t] * np.exp(
                (r - 0.5 * vol_vec[i]**2) * dt + vol_vec[i] * np.sqrt(dt) * X[:, i]
            )
        
        # Compute basket value
        basket_paths[:, t+1] = sum(weights[i] * asset_paths[i][:, t+1] for i in range(n_assets))
    
    # Payoff
    payoffs = np.maximum(basket_paths[:, -1] - K, 0)
    
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error, basket_paths, asset_paths

# Basket volatility formula
def basket_volatility(weights, vol_vec, corr_matrix):
    """Compute basket volatility: σ_B = √(w^T Σ w)"""
    n = len(weights)
    variance = 0
    for i in range(n):
        for j in range(n):
            variance += weights[i] * weights[j] * vol_vec[i] * vol_vec[j] * corr_matrix[i, j]
    return np.sqrt(variance)

# Parameters
S0_vec = np.array([100.0, 100.0, 100.0])  # 3 assets
weights = np.array([0.4, 0.3, 0.3])  # Portfolio weights
K = 100.0
T = 1.0
r = 0.05
vol_vec = np.array([0.25, 0.30, 0.35])  # Different volatilities

print("="*80)
print("BASKET OPTION PRICING")
print("="*80)
print(f"Assets: {len(S0_vec)}, Weights: {weights}")
print(f"Individual S₀: {S0_vec}, σ: {vol_vec*100}%")
print(f"Strike K=${K}, T={T}yr, r={r*100}%\n")

# Test different correlation scenarios
np.random.seed(42)
n_paths = 50000
n_steps = 100

correlations = [0.0, 0.3, 0.6, 0.9]
basket_prices = []
basket_errors = []
basket_vols = []

for rho in correlations:
    # Correlation matrix (uniform correlation)
    corr_matrix = np.ones((3, 3)) * rho
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Basket volatility
    sigma_basket = basket_volatility(weights, vol_vec, corr_matrix)
    basket_vols.append(sigma_basket)
    
    # Price basket option
    price, error, _, _ = mc_basket_call(S0_vec, weights, K, T, r, corr_matrix, 
                                         vol_vec, n_paths, n_steps)
    basket_prices.append(price)
    basket_errors.append(error)
    
    print(f"Correlation ρ={rho:.1f}:")
    print(f"  Basket Volatility: {sigma_basket*100:.2f}%")
    print(f"  Basket Call Price: ${price:.6f} ± ${error:.6f}")
    
    # Compare to single-asset European with basket vol
    B0 = np.dot(weights, S0_vec)
    euro_equiv = european_call(B0, K, T, r, sigma_basket)
    print(f"  European (basket vol): ${euro_equiv:.6f}")
    print(f"  Difference: ${price - euro_equiv:.6f}\n")

# Detailed analysis with ρ=0.5
print("="*80)
print("DETAILED ANALYSIS (ρ=0.5)")
print("="*80)

corr_matrix = np.array([[1.0, 0.5, 0.5],
                        [0.5, 1.0, 0.5],
                        [0.5, 0.5, 1.0]])

np.random.seed(42)
price, error, basket_paths, asset_paths = mc_basket_call(
    S0_vec, weights, K, T, r, corr_matrix, vol_vec, n_paths, n_steps
)

sigma_basket = basket_volatility(weights, vol_vec, corr_matrix)
print(f"Basket Volatility: {sigma_basket*100:.2f}%")
print(f"Basket Call Price: ${price:.6f} ± ${error:.6f}")

# Terminal statistics
print(f"\nTerminal Statistics:")
print(f"  Basket Mean: ${np.mean(basket_paths[:, -1]):.2f}")
print(f"  Basket Std: ${np.std(basket_paths[:, -1]):.2f}")
for i in range(len(S0_vec)):
    print(f"  Asset {i+1} Mean: ${np.mean(asset_paths[i][:, -1]):.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample asset paths
ax = axes[0, 0]
n_plot = 30
time_grid = np.linspace(0, T, n_steps + 1)
colors = ['blue', 'green', 'red']

for i in range(n_assets):
    for j in range(n_plot):
        ax.plot(time_grid, asset_paths[i][j, :], color=colors[i], alpha=0.3, linewidth=0.8)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Asset Price')
ax.set_title('Individual Asset Paths (Blue, Green, Red)')
ax.grid(True, alpha=0.3)

# Plot 2: Basket value paths
ax = axes[0, 1]
for j in range(n_plot):
    ax.plot(time_grid, basket_paths[j, :], color='purple', alpha=0.4, linewidth=1)

ax.axhline(K, color='orange', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Basket Value')
ax.set_title(f'Basket Paths (ρ=0.5)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Terminal basket distribution
ax = axes[0, 2]
ax.hist(basket_paths[:, -1], bins=60, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.axvline(np.mean(basket_paths[:, -1]), color='blue', linestyle='--', linewidth=2,
           label=f'Mean=${np.mean(basket_paths[:, -1]):.0f}')
ax.set_xlabel('Terminal Basket Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Basket Terminal Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Price vs correlation
ax = axes[1, 0]
ax.plot(correlations, basket_prices, 'bo-', linewidth=2, markersize=8, label='Basket Call')
ax.fill_between(correlations,
                np.array(basket_prices) - 1.96*np.array(basket_errors),
                np.array(basket_prices) + 1.96*np.array(basket_errors),
                alpha=0.3)
ax.set_xlabel('Correlation ρ')
ax.set_ylabel('Option Price ($)')
ax.set_title('Basket Call Price vs Correlation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Basket volatility vs correlation
ax = axes[1, 1]
ax.plot(correlations, np.array(basket_vols) * 100, 'go-', linewidth=2, markersize=8)
ax.set_xlabel('Correlation ρ')
ax.set_ylabel('Basket Volatility (%)')
ax.set_title('Basket Volatility vs Correlation (Diversification Effect)')
ax.grid(True, alpha=0.3)

# Plot 6: Scatter plot of assets at maturity
ax = axes[1, 2]
ax.scatter(asset_paths[0][:1000, -1], asset_paths[1][:1000, -1],
          alpha=0.3, s=10, c='blue', label='Asset 1 vs 2')
ax.set_xlabel('Asset 1 Terminal Price')
ax.set_ylabel('Asset 2 Terminal Price')
ax.set_title('Correlation Structure (ρ=0.5, 1000 samples)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('basket_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Compute empirical correlations
print("\n" + "="*80)
print("EMPIRICAL CORRELATIONS (Terminal Prices)")
print("="*80)
terminal_prices = np.column_stack([asset_paths[i][:, -1] for i in range(n_assets)])
empirical_corr = np.corrcoef(terminal_prices.T)
print("Target Correlation Matrix:")
print(corr_matrix)
print("\nEmpirical Correlation Matrix:")
print(empirical_corr)