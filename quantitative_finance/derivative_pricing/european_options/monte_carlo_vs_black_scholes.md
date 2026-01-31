# Monte Carlo vs Black-Scholes

## 1. Concept Skeleton
**Definition:** Comparison of simulation-based (Monte Carlo) and analytical (Black-Scholes) methods for option pricing  
**Purpose:** Understand computational trade-offs, accuracy, flexibility; select appropriate method for payoff complexity  
**Prerequisites:** Black-Scholes formula, Monte Carlo convergence, law of large numbers, computational complexity

## 2. Comparative Framing
| Aspect | Black-Scholes (Analytical) | Monte Carlo (Simulation) |
|--------|---------------------------|--------------------------|
| **Computation Time** | O(1) - instant | O(N paths) - scales linearly |
| **Accuracy** | Exact (under assumptions) | O(1/√N) convergence; statistical error |
| **Flexibility** | European vanilla only | Any payoff (path-dep, exotic) |
| **Greeks** | Closed-form derivatives | Finite diff, pathwise, LR method |
| **Assumptions** | Constant σ, r; lognormal | Arbitrary dynamics (jumps, stoch-vol) |
| **Multidimensional** | Infeasible (no closed-form) | Linear in dimensions (curse avoided) |

## 3. Examples + Counterexamples

**BS Preferred:**  
European call/put on single stock; need instant quotes for thousands of options; Greeks required for hedging

**MC Preferred:**  
Asian option (path-dependent average); basket option (5+ assets with correlation); barrier option with monitoring

**Failure Case:**  
American option pricing: BS inapplicable (no early exercise); MC requires Longstaff-Schwartz (regression, not simple simulation)

## 4. Layer Breakdown
```
Method Selection Decision Tree:
├─ Option Type:
│   ├─ European Vanilla (call/put):
│   │   ├─ Single Asset, Constant Vol → Black-Scholes (instant, exact)
│   │   └─ Complex Model (jumps, stoch-vol) → Monte Carlo (flexible)
│   ├─ Path-Dependent (Asian, lookback, barrier):
│   │   └─ Monte Carlo (only feasible method for many exotic payoffs)
│   ├─ American / Bermudan:
│   │   ├─ Binomial Tree (discrete time)
│   │   └─ Monte Carlo with LSM (Longstaff-Schwartz)
│   └─ Multi-Asset (basket, spread, rainbow):
│       ├─ 2-3 assets: Analytical approximations possible
│       └─ 4+ assets: Monte Carlo (dimension-independent complexity)
├─ Computational Requirements:
│   ├─ Real-Time Pricing (trading desk):
│   │   ├─ BS: Microseconds for Greeks + price
│   │   └─ MC: Milliseconds (1,000 paths) to seconds (1M paths)
│   ├─ Risk Management (overnight batch):
│   │   └─ MC acceptable: Compute 100k scenarios with variance reduction
│   └─ Model Calibration (iterative):
│       └─ BS preferred: Fast repeated evaluations for optimizer
├─ Accuracy Comparison:
│   ├─ BS Error Sources:
│   │   ├─ Model Misspecification: Real markets have vol smile/skew
│   │   ├─ Parameter Estimation: Historical σ ≠ implied σ
│   │   └─ Continuous Trading: Transaction costs, discrete hedging
│   ├─ MC Error Sources:
│   │   ├─ Statistical Error: SE = σ_payoff / √N → 95% CI = ±1.96 SE
│   │   ├─ Time Discretization: Euler scheme O(Δt) bias
│   │   └─ Random Seed: Different runs give different prices (reproducible with seed)
│   └─ Convergence Speed:
│       ├─ Standard MC: O(N^(-0.5)) - halve error → 4× paths
│       ├─ Quasi-MC (Sobol): O(N^(-1) log^d N) - faster for smooth payoffs
│       └─ Variance Reduction: 2-10× fewer paths for same accuracy
├─ Greeks Computation:
│   ├─ BS Greeks:
│   │   ├─ Closed-Form: Δ = N(d₁), Γ = n(d₁)/(Sσ√T), ν = S√T n(d₁)
│   │   └─ Instant Evaluation: No additional computation
│   ├─ MC Greeks:
│   │   ├─ Finite Difference: Δ ≈ (C(S+ε) - C(S-ε))/(2ε) - requires 2× simulations
│   │   ├─ Pathwise Derivative: Compute ∂Payoff/∂S along each path - efficient
│   │   └─ Likelihood Ratio: Multiply payoff by score function - works for discontinuous payoffs
│   └─ Accuracy: BS exact; MC Greeks have higher variance than price estimates
└─ When to Switch from BS to MC:
    ├─ Payoff Complexity: Path-dependence (Asian, barrier, lookback)
    ├─ Model Complexity: Jump-diffusion, stochastic volatility, local volatility
    ├─ High Dimensions: 5+ correlated assets (BS has no closed-form)
    └─ Custom Payoffs: Structured products, exotic derivatives
```

**Interaction:** Evaluate payoff type → Check model assumptions → Choose method → Implement with error bounds

## 5. Mini-Project
Compare Black-Scholes and Monte Carlo for European options; extend to exotic payoffs where BS fails:
```python
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
```

## 6. Challenge Round

**Q1:** Why does MC have O(1/√N) convergence while quasi-MC achieves O(N^(-1))? What's the catch?  
**A1:** Standard MC: CLT gives SE = σ/√N (random samples). Quasi-MC (Sobol, Halton): Low-discrepancy sequences cover space uniformly → deterministic error bound O(N^(-1)(log N)^d) for smooth integrands. Catch: Requires smooth payoffs (no discontinuities like digital options); high dimensions (d > 10) degrade performance.

**Q2:** For high-dimensional basket options (10 assets), why does MC outperform finite difference?  
**A2:** Finite difference (PDE): Grid size grows as K^d (d dimensions, K points per axis) → curse of dimensionality. MC: Sample paths in d-dimensional space; complexity O(N paths) independent of d. For d > 3, MC far superior.

**Q3:** Greeks via finite difference require multiple MC runs (bump-and-revalue). How does pathwise derivative method fix this?  
**A3:** Pathwise: Compute ∂Payoff/∂S analytically along each path (e.g., ∂max(S_T - K, 0)/∂S = 1_{S_T > K}). Average ∂Payoff/∂S over paths → delta from single simulation. Efficient but fails for discontinuous payoffs (barrier breach).

**Q4:** Compare MC accuracy for European call vs Asian call (same # paths). Which has lower error?  
**A4:** European call: Higher variance (payoff = max(S_T - K, 0) varies widely). Asian call: Lower variance (averaging reduces fluctuations; payoff smoother). Asian SE typically 30-50% lower for same N → faster convergence.

**Q5:** When is binomial tree preferred over both BS and MC?  
**A5:** American options with early exercise: Binomial backward induction optimal (exact as N → ∞). BS inapplicable (no early exercise). MC requires LSM (regression overhead). Tree also good for dividend adjustments, transparent for teaching.

**Q6:** Implement variance reduction for European call. Show antithetic variates halve variance empirically.  
**A6:** Standard MC: Var(Price) = σ²_payoff / N. Antithetic (Z, -Z pairs): Corr(Payoff(Z), Payoff(-Z)) < 0 → Var_AV ≈ Var/2 (if correlation ≈ -1). Empirically: σ²_AV / σ²_standard ≈ 0.45-0.55 for vanilla options.

**Q7:** Why does BS fail for barrier options even if payoff is European (exercise at T only)?  
**A7:** BS assumes terminal payoff depends only on S_T (Markovian). Barrier: Payoff depends on entire path (knocked out if S_t crosses barrier before T) → path-dependent. No closed-form under GBM (some approximations exist). MC handles naturally by monitoring path.

**Q8:** Calibrate BS volatility to market prices. Why does MC NOT replace BS for calibration?  
**A8:** Calibration requires repeated pricing (optimizer calls objective 100+ times). MC per call: 100ms (10k paths). Total: 10+ seconds. BS per call: 0.1ms. Total: 10ms. Speed difference: 1000×. Use BS for calibration; MC only for final pricing with calibrated parameters.

## 7. Key References

**Primary Sources:**
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - Comprehensive MC techniques (Chapters 1-5)
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - BS vs numerical methods (Chapter 21)
- [Monte Carlo in Finance Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance) - Overview

**Technical Details:**
- Boyle, P. "Options: A Monte Carlo Approach" (1977) - Original MC option pricing paper
- L'Ecuyer, P. & Lemieux, C. "Variance Reduction via Lattice Rules" (2000) - Quasi-MC for finance

**Thinking Steps:**
1. Identify payoff type: European vanilla (BS) vs path-dependent (MC)
2. Check model assumptions: Constant vol (BS) vs jumps/stoch-vol (MC)
3. Evaluate dimensions: Single asset (BS fast) vs basket 5+ assets (MC scales)
4. Consider speed requirements: Real-time (BS) vs overnight (MC acceptable)
5. Assess accuracy needs: BS exact under assumptions; MC has statistical error O(1/√N)
6. For exotic payoffs with no closed-form: MC only viable method
