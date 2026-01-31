# Control Variates

## 1. Concept Skeleton
**Definition:** Variance reduction technique using a correlated random variable with known expectation to reduce estimator variance  
**Purpose:** Minimize estimator variance by subtracting off correlated "control" that explains variation; effective when control ρ > 0.5  
**Prerequisites:** Correlation concepts, covariance, linear regression, Monte Carlo basics

## 2. Comparative Framing
| Aspect | Control Variates | Antithetic Variates | Importance Sampling |
|--------|-----------------|-------------------|-------------------|
| **Variance Reduction** | ρ²-dependent; ~80% if ρ ≈ 1 | ~50% always | Problem-specific |
| **Requires Known E[Control]** | Yes, essential | No | No, but alters distribution |
| **Coefficient Optimization** | Regression; β̂ = Cov/Var | Fixed (-1 pair) | Not applicable |
| **Complexity** | Moderate | Trivial | High |
| **Best Use Case** | When close proxy exists | Monotone payoffs | Rare events, discontinuities |

## 3. Examples + Counterexamples

**Simple Example:**  
Option pricing: Use European option price (closed-form) as control for Asian option → strong correlation → ~70% variance reduction

**Strong Case:**  
Geometric Asian option: Control = Arithmetic Asian (easy to price); controls 80% of variance in true price  

**Weak Case:**  
Lookback option with European control: Payoff dependence weak (lookback max is unrelated to European final price) → ρ ≈ 0.2 → minimal benefit

**Failure Case:**  
Using arithmetic average as control for knockout barrier option: If barrier hit, both knockouts together → control provides no variance reduction

## 4. Layer Breakdown
```
Control Variates Process:
├─ Control Selection:
│   ├─ Choose control Y with E[Y] = μ_Y known
│   ├─ Y highly correlated with payoff X (ρ >> 0)
│   └─ Example: Simpler analytical proxy or closed-form benchmark
├─ Coefficient Optimization:
│   ├─ Simulate N paths:
│   │   ├─ Payoff X_i, Control Y_i for each path
│   │   └─ Covariance matrix: Cov(X, Y), Var(Y)
│   ├─ Optimal coefficient: β* = Cov(X,Y) / Var(Y)
│   ├─ Regression: β* ≈ Cov(X̂,Ŷ) / Var(Ŷ)
│   └─ Alternative: Use predetermined β = 1 for simplicity
├─ Adjusted Estimator:
│   ├─ Standard payoff: X̄ = (1/N) ΣXᵢ
│   ├─ Control adjustment: (1/N) Σ(Xᵢ - β(Yᵢ - μ_Y))
│   ├─ Expanded: (1/N) ΣXᵢ - β((1/N) ΣYᵢ - μ_Y)
│   └─ Simplified: X̄ - β(Ȳ - μ_Y)
├─ Variance Reduction:
│   ├─ Original Var(X̄) = σ²_X / N
│   ├─ Controlled Var = (σ²_X(1 - ρ²)) / N
│   ├─ Reduction Factor = (1 - ρ²)
│   └─ Example: ρ = 0.9 → ~19% variance; ρ = 0.99 → ~2% variance
└─ Final Price Estimate:
    ├─ Discounted: e^{-rT} × [X̄ - β(Ȳ - μ_Y)]
    └─ SE: e^{-rT} × σ_controlled / √N
```

**Interaction:** Strong correlation Y→X + known E[Y] + optimized β → most effective variance reduction available

## 5. Mini-Project
Implement control variates for Asian option pricing using European control:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes for European call
def bs_european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

# Parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
N_paths = 10000
dt = T / 252
n_steps = int(T / dt)

# Known European call price (control)
european_price = bs_european_call(S0, K, T, r, sigma)

# Generate paths
np.random.seed(42)
Z = np.random.normal(0, 1, (N_paths, n_steps))

# Log returns for GBM
drift = (r - 0.5*sigma**2) * dt
diffusion = sigma * np.sqrt(dt) * Z
log_returns = drift + diffusion

# Stock prices
S = S0 * np.exp(np.cumsum(log_returns, axis=1))

# Payoffs
asian_payoff = np.maximum(np.mean(S, axis=1) - K, 0)  # Arithmetic average
european_payoff = np.maximum(S[:, -1] - K, 0)  # Final price

# Discount
discount = np.exp(-r * T)

# Method 1: Standard Monte Carlo (no control)
asian_price_standard = discount * np.mean(asian_payoff)
asian_se_standard = discount * np.std(asian_payoff) / np.sqrt(N_paths)

# Method 2: Control Variates (optimized coefficient)
# Compute optimal β
covariance = np.cov(asian_payoff, european_payoff)[0, 1]
variance_european = np.var(european_payoff)
beta_optimal = covariance / variance_european

# Adjusted payoff
adjusted_payoff = asian_payoff - beta_optimal * (european_payoff - european_price)
asian_price_cv = discount * np.mean(adjusted_payoff)
asian_se_cv = discount * np.std(adjusted_payoff) / np.sqrt(N_paths)

# Method 3: Control Variates (fixed β = 1)
adjusted_payoff_fixed = asian_payoff - (european_payoff - european_price)
asian_price_cv_fixed = discount * np.mean(adjusted_payoff_fixed)
asian_se_cv_fixed = discount * np.std(adjusted_payoff_fixed) / np.sqrt(N_paths)

# Correlation analysis
correlation = np.corrcoef(asian_payoff, european_payoff)[0, 1]

# Repeat multiple times to assess stability
results = {'standard': [], 'cv_opt': [], 'cv_fixed': []}
np.random.seed(42)
for trial in range(100):
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    asian_pf = np.maximum(np.mean(S, axis=1) - K, 0)
    european_pf = np.maximum(S[:, -1] - K, 0)
    
    results['standard'].append(discount * np.mean(asian_pf))
    
    cov = np.cov(asian_pf, european_pf)[0, 1]
    var_eu = np.var(european_pf)
    beta_opt = cov / var_eu
    results['cv_opt'].append(discount * np.mean(asian_pf - beta_opt*(european_pf - european_price)))
    results['cv_fixed'].append(discount * np.mean(asian_pf - (european_pf - european_price)))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Payoff scatter
axes[0, 0].scatter(european_payoff, asian_payoff, alpha=0.3, s=20)
axes[0, 0].set_title(f'Correlation between Payoffs (ρ = {correlation:.3f})')
axes[0, 0].set_xlabel('European Payoff ($)')
axes[0, 0].set_ylabel('Asian Payoff ($)')
axes[0, 0].grid(alpha=0.3)

# Add regression line
z = np.polyfit(european_payoff, asian_payoff, 1)
p = np.poly1d(z)
x_line = np.linspace(0, european_payoff.max(), 100)
axes[0, 0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'β = {beta_optimal:.3f}')
axes[0, 0].legend()

# Plot 2: Adjusted vs original payoff
axes[0, 1].scatter(asian_payoff, adjusted_payoff, alpha=0.3, s=20, label='CV Adjusted')
axes[0, 1].scatter(asian_payoff, adjusted_payoff_fixed, alpha=0.3, s=20, label='CV Fixed (β=1)')
axes[0, 1].set_title('Control Variate Adjustment')
axes[0, 1].set_xlabel('Original Asian Payoff ($)')
axes[0, 1].set_ylabel('Adjusted Payoff ($)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Convergence of estimates across trials
axes[1, 0].plot(results['standard'], 'o-', label='Standard MC', linewidth=1, markersize=4, alpha=0.7)
axes[1, 0].plot(results['cv_opt'], 's-', label='CV (Optimized)', linewidth=1, markersize=4, alpha=0.7)
axes[1, 0].plot(results['cv_fixed'], '^-', label='CV (Fixed β)', linewidth=1, markersize=4, alpha=0.7)
axes[1, 0].axhline(asian_price_standard, color='C0', linestyle='--', alpha=0.5)
axes[1, 0].axhline(asian_price_cv, color='C1', linestyle='--', alpha=0.5)
axes[1, 0].axhline(asian_price_cv_fixed, color='C2', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Price Estimates Across 100 Trials')
axes[1, 0].set_xlabel('Trial Number')
axes[1, 0].set_ylabel('Asian Call Price ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Variance comparison
se_values = [np.std(results['standard']), 
             np.std(results['cv_opt']),
             np.std(results['cv_fixed'])]
methods = ['Standard', 'CV (Opt)', 'CV (Fixed)']
colors = ['C0', 'C1', 'C2']

axes[1, 1].bar(methods, se_values, color=colors, alpha=0.7)
axes[1, 1].set_title('Standard Error Across 100 Trials')
axes[1, 1].set_ylabel('Standard Error ($)')
axes[1, 1].grid(alpha=0.3, axis='y')

# Add variance reduction percentages
baseline = se_values[0]
for i, (method, se) in enumerate(zip(methods, se_values)):
    reduction = (1 - se/baseline) * 100
    axes[1, 1].text(i, se + 0.01, f'-{reduction:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Results summary
print(f"Asian Call Option Pricing:")
print(f"European Control Price (known): ${european_price:.4f}")
print(f"\n1. Standard Monte Carlo:")
print(f"   Price: ${asian_price_standard:.4f}")
print(f"   SE: ${asian_se_standard:.4f}")

print(f"\n2. Control Variates (Optimized β = {beta_optimal:.4f}):")
print(f"   Price: ${asian_price_cv:.4f}")
print(f"   SE: ${asian_se_cv:.4f}")
print(f"   Variance Reduction: {(1 - (asian_se_cv/asian_se_standard)**2)*100:.1f}%")

print(f"\n3. Control Variates (Fixed β = 1):")
print(f"   Price: ${asian_price_cv_fixed:.4f}")
print(f"   SE: ${asian_se_cv_fixed:.4f}")
print(f"   Variance Reduction: {(1 - (asian_se_cv_fixed/asian_se_standard)**2)*100:.1f}%")

print(f"\nPayoff Correlation: {correlation:.4f}")
print(f"Theoretical Variance Reduction: {(1 - correlation**2)*100:.1f}%")
```

## 6. Challenge Round
When are controls ineffective?
- Low correlation (ρ < 0.5): Control adds noise; use simpler methods
- Control expectation unknown: Can't apply correction term properly
- High-dimensional problems: Harder to find correlated controls in 10+ dimensions
- Discontinuous payoffs: Lookup barrier versus European control unrelated → ρ ≈ 0
- Multiple controls needed: Requires multivariate regression; can introduce overfitting

## 7. Key References
- [Wikipedia - Control Variates](https://en.wikipedia.org/wiki/Variance_reduction)
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)
- [Broadie & Glasserman - Estimating Security Prices](https://www.jstor.org/stable/1088739)

---
**Status:** Advanced variance reduction | **Complements:** Antithetic Variates, Importance Sampling, Regression Analysis
