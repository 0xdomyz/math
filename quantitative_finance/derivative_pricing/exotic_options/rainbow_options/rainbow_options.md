# Rainbow Options

## 1. Concept Skeleton
**Definition:** Multi-asset options with payoff on best/worst/spread of N underlyings; maximum/minimum selection  
**Purpose:** Portfolio diversification; correlation trading; best-of-best strategies; worst-case hedging  
**Prerequisites:** Multivariate simulation, Cholesky decomposition, order statistics, high-dimensional Monte Carlo

## 2. Comparative Framing
| Feature | Rainbow (Best-of) | Rainbow (Worst-of) | Basket | Spread | Single European |
|---------|-------------------|-------------------|--------|--------|-----------------|
| **Payoff** | max(max(S_i) - K, 0) | max(min(S_i) - K, 0) | max(Σw_i S_i - K, 0) | max(S₁ - S₂ - K, 0) | max(S - K, 0) |
| **Correlation Impact** | Negative (diversification) | Positive (co-movement) | Moderate | High | N/A |
| **Value** | Most expensive | Cheap | Moderate | Correlation-dependent | Baseline |
| **Pricing** | Monte Carlo | Monte Carlo | Monte Carlo | Monte Carlo | Black-Scholes |
| **Complexity** | O(N log N) sorting | O(N log N) sorting | O(N) | O(2) | O(1) |

## 3. Examples + Counterexamples

**Simple Example:**  
Best-of-3 call: Assets A, B, C → Payoff = max(max(S_A, S_B, S_C) - K, 0); investor gets upside of best performer

**Failure Case:**  
Perfect correlation (ρ=1): All assets move identically → rainbow = single-asset option → pays premium for no benefit

**Edge Case:**  
Worst-of put: Pays if ANY asset drops below K → diversification becomes risk (probability N× higher) → expensive hedge

## 4. Layer Breakdown
```
Rainbow Option Pricing Pipeline:
├─ Rainbow Types:
│   ├─ Best-of Call: max(max(S₁, S₂, ..., Sₙ) - K, 0)
│   │   └─ Upside of best performer → most expensive
│   ├─ Worst-of Call: max(min(S₁, S₂, ..., Sₙ) - K, 0)
│   │   └─ Limited by weakest asset → cheaper
│   ├─ Best-of Put: max(K - min(S₁, S₂, ..., Sₙ), 0)
│   │   └─ Pays if ANY asset drops → expensive hedge
│   ├─ Worst-of Put: max(K - max(S₁, S₂, ..., Sₙ), 0)
│   │   └─ Requires ALL assets drop → cheap
│   ├─ Nth-to-Default: Triggered by nth worst performer
│   └─ Rainbow Spread: max(S_best - S_worst, 0)
├─ Multi-Asset Simulation:
│   ├─ Asset Dynamics (each i=1..N):
│   │   ├─ dS_i = r S_i dt + σ_i S_i dW_i
│   │   ├─ Discrete: S^i_{t+1} = S^i_t exp((r - σ_i²/2)dt + σ_i√dt Z_i)
│   │   └─ Different vols σ_i for each asset
│   ├─ Correlation Structure:
│   │   ├─ Correlation Matrix ρ: ρᵢⱼ = Corr(S_i, S_j)
│   │   ├─ Cholesky: ρ = LL^T → Correlated normals X = LZ
│   │   └─ Z ~ N(0, I) independent → X ~ N(0, ρ) correlated
│   ├─ Path Generation:
│   │   ├─ For each time step:
│   │   │   ├─ Generate Z = [Z₁, ..., Zₙ] ~ N(0, I)
│   │   │   ├─ X = LZ (apply Cholesky for correlation)
│   │   │   └─ S^i_{t+1} = S^i_t exp((r - σ_i²/2)dt + σ_i√dt X_i)
│   │   └─ Store terminal prices [S₁_T, ..., Sₙ_T] per path
│   └─ Terminal Selection:
│       ├─ Best: S_max = max(S₁_T, ..., Sₙ_T)
│       ├─ Worst: S_min = min(S₁_T, ..., Sₙ_T)
│       └─ Sort: S_(1) ≤ S_(2) ≤ ... ≤ S_(N) (order statistics)
├─ Payoff Calculation:
│   ├─ Best-of Call: max(S_max - K, 0)
│   ├─ Worst-of Call: max(S_min - K, 0)
│   ├─ Best-of Put: max(K - S_min, 0) (pays if ANY drops)
│   ├─ Worst-of Put: max(K - S_max, 0) (pays if ALL drop)
│   └─ Present Value: V = e^{-rT} E[Payoff]
├─ Correlation Impact:
│   ├─ Best-of Options:
│   │   ├─ Low Correlation (ρ → 0):
│   │   │   ├─ Diversification → high chance one asset performs well
│   │   │   ├─ P(at least one ITM) increases
│   │   │   └─ Option MORE expensive (value increases)
│   │   ├─ High Correlation (ρ → 1):
│   │   │   ├─ Assets move together → no diversification benefit
│   │   │   └─ Option LESS expensive (approaches single-asset)
│   │   └─ Value: V(ρ=0) > V(ρ=0.5) > V(ρ=1) ≈ Single Asset
│   ├─ Worst-of Options:
│   │   ├─ Low Correlation:
│   │   │   ├─ Independent movements → likely one lags
│   │   │   └─ Option LESS expensive (worst asset likely poor)
│   │   ├─ High Correlation:
│   │   │   ├─ Co-movement → if one up, all up
│   │   │   └─ Option MORE expensive (worst not so bad)
│   │   └─ Value: V(ρ=1) > V(ρ=0.5) > V(ρ=0)
│   └─ Opposite Effects: Best-of and worst-of have inverse correlation sensitivity
├─ Order Statistics:
│   ├─ Max Distribution (Best-of):
│   │   ├─ F_max(x) = P(max(S_i) ≤ x) = Π F_i(x) (product of CDFs)
│   │   ├─ Independent case: F_i identical → F_max = F^N
│   │   └─ Tail probability: P(S_max > K) = 1 - Π P(S_i ≤ K)
│   ├─ Min Distribution (Worst-of):
│   │   ├─ F_min(x) = P(min(S_i) ≤ x) = 1 - Π (1 - F_i(x))
│   │   └─ P(S_min > K) = Π P(S_i > K) (all must be above K)
│   └─ Correlation complicates: No closed-form with dependence
├─ Greeks:
│   ├─ Deltas: ∂V/∂S_i (vector of N deltas)
│   │   ├─ Best-of: Highest delta on currently leading asset
│   │   ├─ Worst-of: Highest delta on currently lagging asset
│   │   └─ Discontinuous at crossing points (S_i = S_j)
│   ├─ Cross-Gammas: ∂²V/∂S_i∂S_j
│   │   ├─ Positive for best-of (substitution effect)
│   │   └─ Negative for worst-of (competition effect)
│   ├─ Vega: ∂V/∂σ_i
│   │   ├─ Best-of: High vega (volatility increases upside)
│   │   ├─ Worst-of: Lower vega (volatility helps, but less than single)
│   │   └─ Per-asset vega depends on moneyness
│   ├─ Correlation Greeks (Cega): ∂V/∂ρᵢⱼ
│   │   ├─ Best-of: Negative (lower ρ → more value)
│   │   ├─ Worst-of: Positive (higher ρ → more value)
│   │   └─ Critical for correlation risk management
│   └─ Hedging Challenges:
│       ├─ Switching: Leading asset changes → delta jumps
│       ├─ High dimensionality: N×N Greeks matrix
│       └─ Correlation risk: Hard to hedge (no liquid instruments)
├─ Variance Reduction:
│   ├─ Antithetic Variates:
│   │   ├─ Z and -Z → Negatively correlated max/min
│   │   └─ Preserves Cholesky structure: LZ and L(-Z)
│   ├─ Control Variates:
│   │   ├─ Use basket option (has moment-matching approx)
│   │   ├─ Or use single-asset European with avg volatility
│   │   └─ Correlation 0.6-0.8 typical
│   ├─ Importance Sampling:
│   │   ├─ Shift drift toward region where max > K
│   │   └─ Effective for OTM rainbow options
│   └─ Stratification:
│       └─ Stratify on terminal maximum (best-of) or minimum (worst-of)
├─ Approximations:
│   ├─ Johnson's Bound:
│   │   ├─ Best-of Call ≤ Σ Call_i (sum of individual options)
│   │   ├─ Worst-of Call ≥ max(Basket - K, 0) (basket lower bound)
│   │   └─ Useful for quick checks, not tight
│   ├─ Moment Matching:
│   │   ├─ Approximate max distribution as lognormal
│   │   ├─ Match E[S_max] and Var[S_max]
│   │   └─ Accurate for high correlation
│   ├─ Copula Methods:
│   │   ├─ Model marginals separately from dependence structure
│   │   ├─ Use Gaussian copula (equivalent to Cholesky)
│   │   └─ Or Student-t copula (tail dependence)
│   └─ Kirk's Approximation:
│       └─ For spread options (S₁ - S₂): Approximate as single lognormal
└─ Practical Applications:
    ├─ Altiplano/Himalaya: Sequentially remove best performer (exotic structure)
    ├─ Best-of-Best: Two-level rainbow (e.g., best of 3 regions)
    ├─ Nth-to-Default CDS: Credit derivatives (rainbow on default times)
    ├─ Talent Hedge: Best employee among N candidates
    └─ Natural Resource: Best well/field production
```

**Interaction:** Generate correlated asset paths via Cholesky → Compute max/min at maturity → Apply payoff function → Discount to present

## 5. Mini-Project
Price best-of and worst-of rainbow options with correlation analysis:
```python
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
```

## 6. Challenge Round

**Q1:** Why is best-of call MORE expensive with LOWER correlation?  
**A1:** Low correlation → assets move independently → higher chance at least one performs well → more diversification benefit. High correlation → assets co-move → no diversification → similar to single asset. Math: P(max > K) = 1 - Π P(S_i ≤ K); independent: probabilities multiply → higher probability of max being large.

**Q2:** Worst-of put as portfolio insurance: Why expensive?  
**A2:** Pays if ANY asset drops below K → N independent risks → probability of payout ≈ N × single-asset probability (low correlation). Diversification works AGAINST holder: More assets = more ways to lose. Used for worst-case hedging: Protects against scenario where at least one investment fails.

**Q3:** Derive upper bound: Best-of call ≤ Sum of individual calls. Tight?  
**A3:** E[max(max(S_i) - K, 0)] ≤ E[Σ max(S_i - K, 0)] by max ≤ sum. Not tight: RHS counts multiple payoffs, LHS only largest. Equality only if perfect correlation (all move identically). Gap largest for low correlation (diversification makes best-of much cheaper than sum).

**Q4:** Greeks discontinuity: Delta jumps when leading asset changes. Hedging implications?  
**A4:** At S₁ = S₂ crossing: Δ₁ suddenly drops, Δ₂ jumps up as leader changes. Creates hedging challenge: Frequent rebalancing needed near crossings. Cross-gamma ∂²V/∂S₁∂S₂ is negative spike (substitution effect). Practical: Use smooth approximation or dynamic hedging strategy.

**Q5:** Nth-to-default CDS: Triggered by 2nd default among 5 entities. Pricing vs correlation?  
**A5:** High correlation: Defaults cluster → if one defaults, others likely follow → 2nd default soon → expensive. Low correlation: Independent defaults → 2nd default rare → cheaper. Credit crisis: Correlation often underestimated (assumed 0.3, realized 0.8) → massive mispricing → 2008 losses.

**Q6:** Altiplano: Best-of N assets; each year remove best performer and reset. Complexity?  
**A6:** Path-dependent: Which asset removed depends on history → state space explodes. Need track removed set (2^N states). Pricing: Monte Carlo with careful bookkeeping of active assets. Early exercise: Best performers removed → remaining assets have lower expected growth → option value declines over time.

**Q7:** Compare rainbow to spread option (S₁ - S₂ - K). Which has higher correlation risk?  
**A7:** Spread: max(S₁ - S₂ - K, 0) → payoff directly depends on S₁ - S₂ → Cega very high (correlation dominates value). Rainbow (best-of): Correlation affects max distribution but less directly. Spread: Used explicitly for correlation trading; rainbow: Correlation is secondary to selection benefit. Spread has higher absolute Cega.

**Q8:** Dimension reduction for N=100 rainbow: PCA approach?  
**A8:** PCA: Assets = Σ w_k Factor_k; keep K≪N factors explaining 90% variance. Simulate K factors (uncorrelated), reconstruct N assets. Max selection: Only need factor loadings, not full covariance. Speeds up: O(K) vs O(N²) Cholesky. Loses accuracy if tail dependence important (non-Gaussian copulas).

## 7. Key References

**Primary Sources:**
- [Rainbow Option Wikipedia](https://en.wikipedia.org/wiki/Rainbow_option) - Definitions and structures
- Johnson, N. & Kotz, S. "Continuous Multivariate Distributions" (1972) - Order statistics

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Multi-asset simulation (pp. 101-125)
- Deelstra, G. et al. "Pricing of Basket Options" (2004) - Correlation impact on multi-asset options

**Thinking Steps:**
1. Define rainbow type (best/worst-of, call/put) and correlation structure
2. Cholesky decomposition for correlated asset paths
3. Simulate N correlated terminal prices per path
4. Select max or min depending on option type
5. Apply payoff function to selected price; discount to present
6. Analyze correlation sensitivity (Cega) and Greeks
