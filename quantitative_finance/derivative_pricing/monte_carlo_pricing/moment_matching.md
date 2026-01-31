# Moment Matching

## 1. Concept Skeleton
**Definition:** Variance reduction technique forcing simulated sample moments (mean, variance, skewness) to match theoretical moments analytically  
**Purpose:** Remove systematic bias in finite samples; reduce discrepancy between empirical and true distributions  
**Prerequisites:** Moments (mean, variance, skewness, kurtosis), empirical distributions, bootstrap

## 2. Comparative Framing
| Technique | Moment Matching | Antithetic | Importance | Stratified |
|-----------|-----------------|-----------|-----------|-----------|
| **Variance Reduction** | ~20-40% | ~50% | Problem-specific | ~30-50% |
| **Removes Bias** | Yes | No | Yes (if correct) | No |
| **Affects Mean** | Yes | No | Yes | No |
| **Computational Cost** | Low | Negligible | High | Low |
| **Best For** | Finite N bias correction | Simple payoffs | Rare events | Coverage guarantee |

## 3. Examples + Counterexamples

**Simple Example:**  
N=100 Monte Carlo paths of normal returns: Sample mean ≠ 0, sample variance ≠ 1; moment matching standardizes to (0,1)

**Success Case:**  
Pricing Asian options: Moment matching ensures average price matches theoretical mean → reduced bias in estimated expected payoff

**Limitation Case:**  
Skewed/kurtotic payoffs: Matching first 2 moments misses tail behavior; higher moments needed → complexity increases

**Trade-off Case:**  
Latin Hypercube + Moment Matching: LHS already covers space uniformly; MM improves further ~10%; combined effect better than separate

## 4. Layer Breakdown
```
Moment Matching Process:
├─ Sample Generation:
│   ├─ Generate N samples X₁, ..., Xₙ ~ q (e.g., N(0,1))
│   └─ Compute empirical moments:
│       ├─ X̄ = (1/N) ΣXᵢ (empirical mean)
│       ├─ Ŝ² = (1/N) Σ(Xᵢ - X̄)² (empirical variance)
│       └─ Ĝ₃ = (1/N) Σ(Xᵢ - X̄)³/Ŝ³, Ĝ₄ = ... (skewness, kurtosis)
├─ Theoretical Moments:
│   ├─ μ = theoretical mean
│   ├─ σ² = theoretical variance
│   └─ γ₃, γ₄ = theoretical skewness, kurtosis
├─ Adjustment (Standardization):
│   ├─ Center: Yᵢ = Xᵢ - X̄ + μ
│   ├─ Scale: Yᵢ ← Yᵢ × (σ/Ŝ)
│   ├─ Alternative (more aggressive):
│   │   ├─ Iterative regression: Fit lower-order polynomial to transform X to match moments
│   │   └─ Cornish-Fisher: Match skewness & kurtosis via quantile adjustments
│   └─ Result: Adjusted sample Yᵢ has moments ≈ theoretical
├─ Path Simulation:
│   ├─ Use adjusted samples Yᵢ as random shocks
│   ├─ S(T; Yᵢ) = GBM paths with corrected Gaussian innovations
│   └─ Payoff_i = f(S(T; Yᵢ))
├─ Aggregation:
│   ├─ Price = (1/N) Σ e^{-rT} × Payoff_i
│   └─ Variance: Reduced by eliminating sampling error in moments
└─ Benefits:
    ├─ Empirical distribution matches theoretical
    ├─ Reduced bias for finite N
    ├─ Better tail behavior (if higher moments matched)
    └─ Often combined with antithetic/stratified for stacking effects
```

**Interaction:** Generate paths → compute moments → adjust to match theory → reduced finite-sample bias

## 5. Mini-Project
Implement moment matching for Asian option pricing:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# Black-Scholes
def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
N_paths = 5000
dt = T / 252
n_steps = int(T / dt)

# Theoretical moments of standard normal (shocks)
theo_mean = 0
theo_var = 1
theo_skew = 0
theo_kurtosis = 3  # Excess kurtosis = 0, so kurtosis = 3

# Method 1: Standard Monte Carlo
print("=== STANDARD MONTE CARLO ===")
np.random.seed(42)
prices_standard = []
for trial in range(50):
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    asian_payoff = np.maximum(np.mean(S, axis=1) - K, 0)
    price = np.exp(-r*T) * np.mean(asian_payoff)
    prices_standard.append(price)

# Method 2: Moment Matching (first 2 moments only)
print("=== MOMENT MATCHING (Mean & Variance) ===")
np.random.seed(42)
prices_mm2 = []
for trial in range(50):
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    
    # Adjust each timestep's shocks
    Z_adjusted = np.zeros_like(Z)
    for t in range(n_steps):
        Z_t = Z[:, t]
        # Center
        Z_centered = Z_t - np.mean(Z_t)
        # Scale
        Z_adjusted[:, t] = Z_centered * (np.sqrt(theo_var) / np.std(Z_centered)) + theo_mean
    
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_adjusted
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    asian_payoff = np.maximum(np.mean(S, axis=1) - K, 0)
    price = np.exp(-r*T) * np.mean(asian_payoff)
    prices_mm2.append(price)

# Method 3: Moment Matching (first 4 moments - Cornish-Fisher)
print("=== MOMENT MATCHING (Mean, Variance, Skew, Kurtosis) ===")
np.random.seed(42)
prices_mm4 = []
for trial in range(50):
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    
    Z_adjusted = np.zeros_like(Z)
    for t in range(n_steps):
        Z_t = Z[:, t]
        
        # Standardize to (0,1)
        Z_std = (Z_t - np.mean(Z_t)) / np.std(Z_t)
        
        # Cornish-Fisher adjustment for skewness and kurtosis
        G3 = skew(Z_std)
        G4 = kurtosis(Z_std, fisher=True)  # Excess kurtosis
        
        # Cornish-Fisher transform
        w = Z_std + (G3/6)*Z_std**2 + (G4/24)*(Z_std**3 - 3*Z_std)
        w = w - (G3**2/36)*(2*Z_std**3 - 5*Z_std)
        
        # Scale and center to match theory
        Z_adjusted[:, t] = (w - np.mean(w)) / np.std(w)
    
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_adjusted
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    asian_payoff = np.maximum(np.mean(S, axis=1) - K, 0)
    price = np.exp(-r*T) * np.mean(asian_payoff)
    prices_mm4.append(price)

# Comparison
print(f"\nResults (50 trials, {N_paths} paths each):")
print(f"Standard MC:")
print(f"  Mean: ${np.mean(prices_standard):.6f}")
print(f"  Std Dev: ${np.std(prices_standard):.6f}")

print(f"Moment Matching (2 moments):")
print(f"  Mean: ${np.mean(prices_mm2):.6f}")
print(f"  Std Dev: ${np.std(prices_mm2):.6f}")
print(f"  Variance Reduction: {(1 - (np.std(prices_mm2)/np.std(prices_standard))**2)*100:.1f}%")

print(f"Moment Matching (4 moments):")
print(f"  Mean: ${np.mean(prices_mm4):.6f}")
print(f"  Std Dev: ${np.std(prices_mm4):.6f}")
print(f"  Variance Reduction: {(1 - (np.std(prices_mm4)/np.std(prices_standard))**2)*100:.1f}%")

# Distribution analysis: single trial
print("\n=== DISTRIBUTION ANALYSIS (Single Trial) ===")
np.random.seed(42)
Z = np.random.normal(0, 1, (N_paths, n_steps))
Z_mm2 = np.zeros_like(Z)
Z_mm4 = np.zeros_like(Z)

for t in range(n_steps):
    # MM2
    Z_mm2[:, t] = (Z[:, t] - np.mean(Z[:, t])) / np.std(Z[:, t])
    
    # MM4
    Z_std = (Z[:, t] - np.mean(Z[:, t])) / np.std(Z[:, t])
    G3 = skew(Z_std)
    G4 = kurtosis(Z_std, fisher=True)
    w = Z_std + (G3/6)*Z_std**2 + (G4/24)*(Z_std**3 - 3*Z_std)
    w = w - (G3**2/36)*(2*Z_std**3 - 5*Z_std)
    Z_mm4[:, t] = (w - np.mean(w)) / np.std(w)

print(f"Sample Statistics (first timestep):")
print(f"Standard MC:")
print(f"  Mean: {np.mean(Z[:, 0]):.6f} (theory: 0)")
print(f"  Variance: {np.var(Z[:, 0]):.6f} (theory: 1)")
print(f"  Skewness: {skew(Z[:, 0]):.6f} (theory: 0)")
print(f"  Kurtosis: {kurtosis(Z[:, 0], fisher=True):.6f} (theory: 0)")

print(f"MM2:")
print(f"  Mean: {np.mean(Z_mm2[:, 0]):.6f}")
print(f"  Variance: {np.var(Z_mm2[:, 0]):.6f}")
print(f"  Skewness: {skew(Z_mm2[:, 0]):.6f}")

print(f"MM4:")
print(f"  Mean: {np.mean(Z_mm4[:, 0]):.6f}")
print(f"  Variance: {np.var(Z_mm4[:, 0]):.6f}")
print(f"  Skewness: {skew(Z_mm4[:, 0]):.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price estimates distribution
axes[0, 0].hist(prices_standard, bins=15, alpha=0.6, label='Standard', color='C0', density=True)
axes[0, 0].hist(prices_mm2, bins=15, alpha=0.6, label='MM2', color='C1', density=True)
axes[0, 0].hist(prices_mm4, bins=15, alpha=0.6, label='MM4', color='C2', density=True)
axes[0, 0].set_xlabel('Asian Call Price ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Price Estimates (50 trials)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Convergence
methods = ['Standard', 'MM2', 'MM4']
std_devs = [np.std(prices_standard), np.std(prices_mm2), np.std(prices_mm4)]
colors = ['C0', 'C1', 'C2']

axes[0, 1].bar(methods, std_devs, color=colors, alpha=0.7, width=0.5)
axes[0, 1].set_ylabel('Standard Deviation ($)')
axes[0, 1].set_title('Standard Deviation of Estimates')
axes[0, 1].grid(alpha=0.3, axis='y')

for i, sd in enumerate(std_devs):
    axes[0, 1].text(i, sd + 0.0005, f'${sd:.5f}', ha='center', fontweight='bold')

# Plot 3: Normal QQ plot (standard MC)
from scipy import stats
stats.probplot(Z[:, 0], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Standard MC - Normal Q-Q Plot (first timestep)')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Normal QQ plot (MM4)
stats.probplot(Z_mm4[:, 0], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('MM4 - Normal Q-Q Plot (first timestep)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does moment matching underperform?
- Higher moments unknown: Can't match skewness/kurtosis without estimation; adds complexity
- Iterative methods required: Scaling/centering is order-dependent; convergence not guaranteed
- Multivariate: Matching joint moments in high dimensions exponentially complex
- Non-monotone payoffs: Adjusting shocks changes path structure; may not preserve correlations
- Computational cost vs. benefit: ~10-20% improvement often modest; antithetic/stratified simpler

## 7. Key References
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)
- [Acerbi - Moment Matching Methods](https://ssrn.com/abstract=1107960)
- [Cornish & Fisher - Moments and Cumulants](https://www.jstor.org/stable/2332539)

---
**Status:** Bias reduction technique | **Complements:** Antithetic Variates, Latin Hypercube, Stratified Sampling
