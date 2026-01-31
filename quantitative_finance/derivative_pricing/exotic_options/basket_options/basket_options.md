# Basket Options

## 1. Concept Skeleton
**Definition:** Multi-asset options with payoff based on weighted portfolio of underlying assets  
**Purpose:** Diversification exposure; correlation trading; reduce single-stock risk; portfolio hedging  
**Prerequisites:** Multivariate simulation, correlation matrices, Cholesky decomposition, correlated random variables

## 2. Comparative Framing
| Feature | Basket Option | Rainbow Option | Single-Asset European | Index Option |
|---------|---------------|----------------|----------------------|--------------|
| **Underlying** | Weighted portfolio | Best/worst of N | Single stock | Broad index |
| **Payoff** | max(Σw_i S_i - K, 0) | max(max(S_i) - K, 0) | max(S - K, 0) | max(Index - K, 0) |
| **Correlation Impact** | High (diversification) | Moderate | N/A | Implicit in index |
| **Pricing** | Monte Carlo | Monte Carlo | Black-Scholes | Black-Scholes |
| **Complexity** | O(N assets × M paths) | O(N × M) | O(1) | O(1) |

## 3. Examples + Counterexamples

**Simple Example:**  
Basket call on tech stocks: 0.4×AAPL + 0.3×MSFT + 0.3×GOOGL; K=$100 → Payoff = max(weighted sum - 100, 0)

**Failure Case:**  
Perfect correlation (ρ=1): Basket behaves like single asset with blended volatility; loses diversification benefit

**Edge Case:**  
Zero correlation: Maximum diversification → basket volatility = σ/√N → cheaper than single-asset option

## 4. Layer Breakdown
```
Basket Option Pricing Pipeline:
├─ Basket Definition:
│   ├─ Assets: S₁, S₂, ..., Sₙ (N underlyings)
│   ├─ Weights: w₁, w₂, ..., wₙ where Σwᵢ = 1
│   ├─ Basket Value: B_t = Σ wᵢ Sᵢ(t)
│   └─ Payoff: Call = max(B_T - K, 0); Put = max(K - B_T, 0)
├─ Correlation Structure:
│   ├─ Correlation Matrix ρ:
│   │   ├─ ρᵢⱼ = Corr(Sᵢ, Sⱼ) ∈ [-1, 1]
│   │   ├─ ρᵢᵢ = 1 (self-correlation)
│   │   └─ Symmetric: ρᵢⱼ = ρⱼᵢ
│   ├─ Covariance Matrix Σ:
│   │   └─ Σᵢⱼ = ρᵢⱼ σᵢ σⱼ
│   └─ Basket Volatility:
│       └─ σ_basket = √(Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ)
├─ Monte Carlo Simulation:
│   ├─ Correlated Random Variables:
│   │   ├─ Generate Independent Z: Z₁, ..., Zₙ ~ N(0, 1)
│   │   ├─ Cholesky Decomposition: ρ = L L^T (lower triangular L)
│   │   ├─ Correlated Normals: X = L Z → Cov(X) = ρ
│   │   └─ X_i = Σⱼ Lᵢⱼ Zⱼ for i=1..N
│   ├─ Path Generation (for each asset i):
│   │   ├─ S^i_{t+1} = S^i_t exp((rᵢ - σᵢ²/2)Δt + σᵢ√Δt Xᵢ_t)
│   │   ├─ Different drifts: rᵢ (risk-free rates may differ)
│   │   └─ Different vols: σᵢ (each asset has own volatility)
│   ├─ Basket Value at Each Step:
│   │   └─ B_t = Σ wᵢ S^i_t
│   ├─ Terminal Payoff:
│   │   └─ Call: max(B_T - K, 0)
│   └─ Present Value:
│       └─ PV = e^(-rT) × Payoff
├─ Variance Reduction:
│   ├─ Control Variate:
│   │   ├─ Use single asset with similar characteristics
│   │   ├─ Or use geometric basket (has closed-form approximation)
│   │   └─ Correlation typically 0.8-0.95
│   ├─ Antithetic Variates:
│   │   ├─ Z and -Z → Negatively correlated basket values
│   │   └─ Preserves correlation structure (LZ and L(-Z))
│   ├─ Moment Matching:
│   │   └─ Force Σwᵢ S^i_T = Σwᵢ S₀^i e^(rT) (expected value)
│   └─ Stratified Sampling:
│       └─ Stratify on basket terminal value B_T
├─ Correlation Impact:
│   ├─ High Correlation (ρ → 1):
│   │   ├─ Basket behaves like single asset
│   │   ├─ Basket vol → weighted average of individual vols
│   │   └─ Option expensive (no diversification benefit)
│   ├─ Low Correlation (ρ → 0):
│   │   ├─ Maximum diversification
│   │   ├─ Basket vol → σ_avg / √N
│   │   └─ Option cheaper (low basket volatility)
│   └─ Negative Correlation (ρ < 0):
│       ├─ Offsetting movements → very low basket vol
│       └─ Option very cheap (hedge-like behavior)
├─ Greeks:
│   ├─ Deltas: ∂V/∂Sᵢ (one per asset; vector of N deltas)
│   ├─ Gammas: ∂²V/∂Sᵢ² (diagonal) and ∂²V/∂Sᵢ∂Sⱼ (cross-gammas)
│   ├─ Vega: ∂V/∂σᵢ (per-asset vega; changes with weight wᵢ)
│   ├─ Correlation Greeks:
│   │   ├─ Cega: ∂V/∂ρᵢⱼ (sensitivity to correlation changes)
│   │   └─ Important for correlation trading strategies
│   └─ Theta: ∂V/∂t (time decay; similar to single-asset)
└─ Approximations:
    ├─ Moment Matching: Approximate basket with lognormal distribution
    ├─ Curran's Approximation: Condition on geometric average
    ├─ Geometric Basket: Use geometric average (has closed-form)
    └─ Taylor Expansion: Approximate basket dynamics near current value
```

**Interaction:** Generate correlated paths via Cholesky → Compute weighted basket value → Payoff on basket → Discount to present

## 5. Mini-Project
Price basket option on 3-asset portfolio with varying correlations:
```python
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
```

## 6. Challenge Round

**Q1:** Derive basket volatility formula σ_B = √(w^T Σ w). What does it reveal about diversification?  
**A1:** Var(Basket) = Var(Σw_i S_i) = ΣΣ w_i w_j Cov(S_i, S_j) = w^T Σ w where Σ_ij = σ_i σ_j ρ_ij. For equal weights, uncorrelated assets: σ_B = σ_avg/√N → diversification reduces volatility. Perfect correlation: σ_B = Σw_i σ_i (no benefit).

**Q2:** Why is basket option cheaper than sum of individual options (Σ Call_i)?  
**A2:** Jensen's inequality: E[max(Basket - K, 0)] < Σ E[max(S_i - K_i, 0)] for convex payoff. Diversification reduces basket volatility → lower option value. Portfolio of options has independent payoffs; basket has correlated payoffs → less total volatility → cheaper.

**Q3:** Correlation Greeks (Cega): ∂V/∂ρ_ij. Sign and magnitude?  
**A3:** Higher correlation → higher basket vol → higher option value → Cega > 0 for long basket calls. Magnitude: Largest for ρ near 0 (steepest slope); smaller for ρ → 1 (flattens). Used for correlation trading: Bet on correlation changes via basket options.

**Q4:** Cholesky fails if correlation matrix not positive semi-definite. When does this occur?  
**A4:** Inconsistent correlations: e.g., ρ₁₂=0.9, ρ₁₃=0.9, ρ₂₃=-0.9 (contradictory). Matrix must satisfy Σx^T ρ x ≥ 0 for all x. Check: All eigenvalues ≥ 0. If fails, use nearest positive-definite matrix (Higham algorithm) or PCA to reduce dimensions.

**Q5:** Basket on indices (S&P 500, FTSE, Nikkei): Why more expensive than individual index options?  
**A5:** Cross-border correlations typically 0.5-0.7 (not perfect). Basket captures global exposure → higher effective volatility than single index. Time zone differences → asynchronous moves → adds uncertainty. Currency risk if multi-currency basket.

**Q6:** Quanto basket: Assets denominated in different currencies, payoff in single currency. Complexity?  
**A6:** Need 3-way correlation: (Asset 1, Asset 2, FX 1/2). Simulate asset prices in local currency, then convert using FX paths. Compo risk: Correlation between asset and FX. Greeks become multi-dimensional (asset deltas, FX deltas, cross-Greeks).

**Q7:** Compare basket option to spread option max(S₁ - S₂ - K, 0). Which is more complex?  
**A7:** Spread is 2-asset basket with weights [1, -1]. Basket more general (N assets, any weights). Complexity similar: Both need correlated paths. Spread has numerical challenges when S₁ ≈ S₂ (near-zero strike). Basket more stable with diversified weights.

**Q8:** Moment matching for basket: Approximate basket distribution as lognormal. How to match first two moments?  
**A8:** E[B_T] = Σw_i S₀^i e^(rT). Var[ln(B_T)] harder (basket not lognormal). Approximation: Use basket volatility σ_B from formula, treat B as GBM. Pluginto BS formula with (B₀, σ_B, K). Accurate for ATM, high correlation; breaks down for deep OTM/low correlation.

## 7. Key References

**Primary Sources:**
- [Basket Option Wikipedia](https://en.wikipedia.org/wiki/Basket_option) - Overview and correlation impact
- Gentle, J.E. "Random Number Generation and Monte Carlo Methods" (2003) - Cholesky decomposition
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Multi-asset options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Correlated paths (pp. 71-101)
- Curran, M. "Valuing Asian and Basket Options" (1994) - Moment matching approximations

**Thinking Steps:**
1. Define basket weights and correlation matrix (positive semi-definite)
2. Cholesky decomposition: ρ = LL^T for correlated random generation
3. Generate independent normals Z; transform to correlated X = LZ
4. Simulate each asset with GBM using correlated X_i
5. Compute basket value B_t = Σw_i S^i_t at each step
6. Payoff on terminal basket value; discount to present
