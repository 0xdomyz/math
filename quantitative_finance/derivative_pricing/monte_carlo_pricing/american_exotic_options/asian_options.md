# Asian Options

## 1. Concept Skeleton
**Definition:** Path-dependent options with payoff based on average price over observation period (arithmetic/geometric mean), reducing volatility impact and manipulation risk compared to European spot-price settlement.  
**Purpose:** Hedge average exposure (commodities, FX); reduce settlement manipulation; lower premium than European; corporate hedging (average monthly prices); reduce gamma risk.  
**Prerequisites:** European option pricing, pathwise simulation, law of large numbers, control variate methods, covariance analysis

## 2. Comparative Framing

| Aspect | Arithmetic Asian | Geometric Asian | European Option | Lookback Option |
|--------|-----------------|-----------------|-----------------|-----------------|
| **Payoff Basis** | Arithmetic mean Ā | Geometric mean G | Final spot S_T | Max/Min over path |
| **Closed-Form** | None (MC required) | Yes (simplified BS) | Yes (Black-Scholes) | Partial (some cases) |
| **Premium vs European** | 50-70% (lower vol) | 40-60% (even lower) | 100% (baseline) | 200-300% (higher) |
| **Volatility Sensitivity** | Low (averaging smooths) | Lower (geometric < arithmetic) | High | Very high |
| **Path Dependency** | Strong (all observations) | Strong (all observations) | None | Extreme (max/min) |
| **Use Cases** | Commodity hedging, FX | Theoretical | Speculation | Exotic strategies |
| **MC Efficiency** | Standard | Use geometric control | Standard | Requires monitoring |

## 3. Examples + Counterexamples

**Simple Example: Arithmetic Asian Call**  
Stock starts S₀=$100, observed monthly for T=1yr (12 observations). Path: $100→$105→$110→$95→$98→$105→$108→$112→$115→$110→$108→$120. Arithmetic average: Ā = Σ S_i / 12 = $1,286/12 = $107.17. Strike K=$100. Payoff: max(Ā-K, 0) = max($107.17-$100, 0) = $7.17. European call (same path): max(S_T-K, 0) = max($120-$100, 0) = $20. Asian captures average → lower payoff, lower premium (~50-60% of European).

**Failure Case: Arithmetic Asian with Few Observations**  
Asian call with T=1yr but only 2 observations (start and end). Path: S₀=$100, S_T=$120. Average: Ā = ($100+$120)/2 = $110. Payoff (K=$100): $10. But path could swing $80→$140 mid-year (ignored). "Asian" name misleading → behaves like European with 2-point average. Problem: insufficient observations → doesn't capture true average exposure. Rule: minimum 12 observations/year for true averaging benefit; 52 (weekly) or 252 (daily) for liquid commodities.

**Edge Case: Geometric Asian Closed-Form Pricing**  
Geometric Asian uses G = (Π S_i)^{1/n} instead of Ā = (Σ S_i)/n. Under GBM, geometric mean of lognormals → lognormal → closed-form Black-Scholes-like formula. Parameters adjusted: σ_adj = σ/√3, r_adj = (r + σ²/6)/2. Price G_Asian ≈ 60-70% of European (empirical). Advantage: exact pricing, use as control variate for arithmetic Asian (highly correlated). Disadvantage: geometric mean < arithmetic mean → even lower payoff → less practical.

## 4. Layer Breakdown

```
Asian Option Framework:
├─ Payoff Definitions:
│   ├─ Arithmetic Asian Call: max(Ā - K, 0) where Ā = (1/n)Σ S_{t_i}
│   ├─ Arithmetic Asian Put: max(K - Ā, 0)
│   ├─ Geometric Asian Call: max(G - K, 0) where G = (Π S_{t_i})^{1/n}
│   ├─ Geometric Asian Put: max(K - G, 0)
│   └─ Observation dates: t₁, t₂, ..., t_n (typically equally spaced)
├─ Path Dependency:
│   ├─ European: Only terminal S_T matters → path-independent
│   ├─ Asian: Entire path S_{t_1}, ..., S_{t_n} matters → strongly path-dependent
│   ├─ Early path impact: First half of observations weighted equally to second half
│   └─ Volatility reduction: Averaging n observations → effective vol ≈ σ/√n
├─ Monte Carlo Pricing (Arithmetic Asian):
│   ├─ 1. Simulate M paths under risk-neutral GBM:
│   │   ├─ dS = r·S·dt + σ·S·dW
│   │   ├─ Discretize: S_{i+1} = S_i · exp((r - q - 0.5σ²)dt + σ√dt·Z_i)
│   │   └─ Store: S^{(m)}_{t_1}, S^{(m)}_{t_2}, ..., S^{(m)}_{t_n} for each path m
│   ├─ 2. Compute arithmetic average per path:
│   │   ├─ Ā^{(m)} = (1/n) Σ_{i=1}^n S^{(m)}_{t_i}
│   │   └─ Example: n=12 monthly observations
│   ├─ 3. Compute discounted payoff:
│   │   ├─ Call: V^{(m)} = e^{-rT} × max(Ā^{(m)} - K, 0)
│   │   ├─ Put: V^{(m)} = e^{-rT} × max(K - Ā^{(m)}, 0)
│   │   └─ Discount from T (expiry) back to present
│   ├─ 4. Average over all paths:
│   │   ├─ V_Asian = (1/M) Σ_{m=1}^M V^{(m)}
│   │   ├─ Standard Error: SE = σ_V / √M where σ_V = std(V^{(m)})
│   │   └─ Confidence Interval: V ± 1.96·SE (95% CI)
│   └─ Convergence: O(1/√M) like standard MC (no special acceleration)
├─ Geometric Asian (Closed-Form Benchmark):
│   ├─ Under GBM, geometric mean G ~ Lognormal(μ_G, σ_G²)
│   ├─ Adjusted parameters:
│   │   ├─ σ_G = σ × √((2n+1)/(6(n+1))) ≈ σ/√3 for large n
│   │   ├─ μ_G adjusted similarly (accounts for Jensen's inequality)
│   │   └─ Price via modified Black-Scholes with (S₀, K, T, r, σ_G)
│   ├─ Exact formula: G_call = S₀·e^{(b-r)T}·N(d₁) - K·e^{-rT}·N(d₂)
│   │   ├─ where b = (r - q - σ²/2)/2 + σ²/6
│   │   ├─ σ_adj = σ/√3, d₁, d₂ computed with σ_adj
│   │   └─ Result: Geometric Asian ≈ 60-70% of European (empirical)
│   └─ Use Case: Control variate for arithmetic Asian (Cov(A, G) high)
├─ Variance Reduction via Control Variates:
│   ├─ Idea: Arithmetic Ā and geometric G highly correlated (ρ ≈ 0.95+)
│   ├─ Known: Geometric Asian closed-form price G_exact
│   ├─ Estimate: β = Cov(V_A, V_G) / Var(V_G) ≈ 1 (from pilot simulation)
│   ├─ Control Variate Estimator:
│   │   ├─ V_A^{CV} = V_A + β(G_exact - V_G^{MC})
│   │   ├─ Variance: Var(V_A^{CV}) = Var(V_A)(1 - ρ²)
│   │   └─ Reduction: If ρ=0.95 → Var reduced by 1-0.95²=9.75% of original → SE reduced 10%
│   └─ Implementation: Simulate both Ā and G on same paths, apply CV correction
├─ Asian Option Properties:
│   ├─ Premium Discount vs European:
│   │   ├─ Arithmetic Asian: 50-70% of European (depends on n, T, σ)
│   │   ├─ Geometric Asian: 40-60% of European (Jensen: G < Ā always)
│   │   ├─ Intuition: Averaging reduces effective volatility → lower option value
│   │   └─ Extreme: Daily observations (n=252) → very smooth average → minimal optionality
│   ├─ Volatility Sensitivity (Vega):
│   │   ├─ Asian vega < European vega (factor ~0.5-0.7)
│   │   ├─ Reason: Averaging dampens price swings → less sensitivity to σ
│   │   └─ Practical: Use Asian for vol-insensitive hedging (commodity producers)
│   ├─ Delta & Gamma:
│   │   ├─ Delta: ∂V/∂S₀ ≈ 0.3-0.5 for ATM (vs 0.5 European)
│   │   ├─ Gamma: Lower than European (path averaging → less curvature)
│   │   └─ Hedging: Less frequent rebalancing needed (lower gamma risk)
│   └─ Time Decay (Theta):
│       ├─ Asian theta more uniform (less acceleration near expiry)
│       ├─ Reason: Early observations already locked in → less time-to-maturity impact
│       └─ Late in life (t near T, most obs locked): theta ≈ European theta
└─ Applications:
    ├─ Commodity Hedging (Oil, Gold):
    │   ├─ Producer sells average monthly price (not spot at T)
    │   ├─ Asian put protects average revenue floor
    │   └─ Example: Oil producer hedges $60/bbl average over year (12 months)
    ├─ Foreign Exchange (FX):
    │   ├─ Corporate treasurer hedges average FX rate for monthly payments
    │   ├─ Asian call caps average cost (import hedge)
    │   └─ Example: EUR/USD average rate protection for quarterly expense
    ├─ Index Manipulation Prevention:
    │   ├─ Settlement on average price → harder to manipulate (multiple dates)
    │   ├─ Reduces "pinning" effect (large positions moving expiry spot)
    │   └─ Used in some index-linked products
    └─ Lower Premium Budget:
        ├─ Corporate CFO wants option protection but budget-constrained
        ├─ Asian 50% cheaper than European → affordable hedging
        └─ Trade-off: Protection on average vs. spot (adequate for cash flow hedging)
```

**Interaction:** Path generation → Averaging → Payoff calculation → Control variate adjustment → Asian value

## 5. Mini-Project

Price arithmetic Asian call using MC; implement geometric control variate:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class AsianOption:
    """Monte Carlo pricing for Asian options with control variate"""
    
    def __init__(self, S0, K, T, r, sigma, q=0, n_observations=12, option_type='call'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_observations = n_observations
        self.option_type = option_type
    
    def simulate_paths(self, n_paths, n_steps=252):
        """Generate paths with observation points"""
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for i in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            paths[:, i+1] = paths[:, i] * np.exp(
                (self.r - self.q - 0.5*self.sigma**2)*dt + 
                self.sigma*np.sqrt(dt)*Z
            )
        
        # Extract observation points
        obs_indices = np.linspace(0, n_steps, self.n_observations, dtype=int)
        observations = paths[:, obs_indices]
        
        return paths, observations
    
    def geometric_asian_closed_form(self):
        """Closed-form geometric Asian call/put price"""
        n = self.n_observations
        
        # Adjusted volatility for geometric average
        sigma_adj = self.sigma * np.sqrt((2*n + 1) / (6*(n + 1)))
        
        # Adjusted growth rate
        b = 0.5 * (self.r - self.q - 0.5*self.sigma**2)
        
        # Adjusted forward price
        F = self.S0 * np.exp(b*self.T)
        
        d1 = (np.log(F/self.K) + 0.5*sigma_adj**2*self.T) / (sigma_adj*np.sqrt(self.T))
        d2 = d1 - sigma_adj*np.sqrt(self.T)
        
        if self.option_type == 'call':
            return F*np.exp(-self.r*self.T)*norm.cdf(d1) - \
                   self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:  # put
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - \
                   F*np.exp(-self.r*self.T)*norm.cdf(-d1)
    
    def price_standard_mc(self, n_paths=10000):
        """Standard MC without control variate"""
        paths, observations = self.simulate_paths(n_paths)
        
        # Compute arithmetic average
        arithmetic_avg = np.mean(observations, axis=1)
        
        # Payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(arithmetic_avg - self.K, 0)
        else:  # put
            payoffs = np.maximum(self.K - arithmetic_avg, 0)
        
        # Discount and average
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        std_error = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'payoffs': payoffs,
            'arithmetic_avg': arithmetic_avg
        }
    
    def price_with_control_variate(self, n_paths=10000):
        """MC with geometric Asian as control variate"""
        paths, observations = self.simulate_paths(n_paths)
        
        # Compute arithmetic and geometric averages
        arithmetic_avg = np.mean(observations, axis=1)
        geometric_avg = np.exp(np.mean(np.log(observations), axis=1))
        
        # Arithmetic Asian payoffs
        if self.option_type == 'call':
            payoffs_arith = np.maximum(arithmetic_avg - self.K, 0)
            payoffs_geom = np.maximum(geometric_avg - self.K, 0)
        else:
            payoffs_arith = np.maximum(self.K - arithmetic_avg, 0)
            payoffs_geom = np.maximum(self.K - geometric_avg, 0)
        
        # Geometric Asian closed-form (known exact value)
        geom_exact = self.geometric_asian_closed_form()
        
        # Estimate optimal beta (covariance-based)
        cov = np.cov(payoffs_arith, payoffs_geom)[0, 1]
        var_geom = np.var(payoffs_geom)
        beta = cov / var_geom if var_geom > 0 else 1.0
        
        # Control variate adjustment
        geom_mc = np.mean(payoffs_geom)
        payoffs_cv = payoffs_arith - beta * (payoffs_geom - geom_mc)
        
        # Final price
        price_cv = np.exp(-self.r*self.T) * (np.mean(payoffs_arith) + \
                   beta * (geom_exact - np.exp(-self.r*self.T)*geom_mc))
        std_error_cv = np.exp(-self.r*self.T) * np.std(payoffs_cv) / np.sqrt(n_paths)
        
        # Standard MC for comparison
        price_standard = np.exp(-self.r*self.T) * np.mean(payoffs_arith)
        std_error_standard = np.exp(-self.r*self.T) * np.std(payoffs_arith) / np.sqrt(n_paths)
        
        return {
            'price_cv': price_cv,
            'price_standard': price_standard,
            'std_error_cv': std_error_cv,
            'std_error_standard': std_error_standard,
            'beta': beta,
            'variance_reduction': 1 - (std_error_cv / std_error_standard)**2,
            'geom_exact': geom_exact,
            'arithmetic_avg': arithmetic_avg,
            'geometric_avg': geometric_avg
        }
    
    def european_call_bs(self):
        """Black-Scholes European call for comparison"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        if self.option_type == 'call':
            return self.S0*np.exp(-self.q*self.T)*norm.cdf(d1) - \
                   self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - \
                   self.S0*np.exp(-self.q*self.T)*norm.cdf(-d1)

# Parameters
S0, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
np.random.seed(42)

print("="*80)
print("ASIAN OPTION PRICING (Arithmetic Average with Control Variate)")
print("="*80)

# Price Asian call
asian = AsianOption(S0, K, T, r, sigma, q, n_observations=12, option_type='call')

print(f"\nOption Parameters:")
print(f"  Spot S0:           ${S0:.2f}")
print(f"  Strike K:          ${K:.2f}")
print(f"  Time to Expiry:    {T:.2f} years")
print(f"  Interest Rate:     {r*100:.1f}%")
print(f"  Volatility:        {sigma*100:.1f}%")
print(f"  Dividend Yield:    {q*100:.1f}%")
print(f"  # Observations:    {asian.n_observations} (monthly)")

# Price with control variate
result_cv = asian.price_with_control_variate(n_paths=20000)

print(f"\n{'='*80}")
print("PRICING RESULTS")
print("="*80)
print(f"  Arithmetic Asian (Standard MC):  ${result_cv['price_standard']:.4f} ± ${result_cv['std_error_standard']:.4f}")
print(f"  Arithmetic Asian (Control Var):  ${result_cv['price_cv']:.4f} ± ${result_cv['std_error_cv']:.4f}")
print(f"  Geometric Asian (Closed-Form):   ${result_cv['geom_exact']:.4f}")
print(f"  European Call (Black-Scholes):   ${asian.european_call_bs():.4f}")
print(f"\n  Variance Reduction: {result_cv['variance_reduction']*100:.1f}%")
print(f"  Optimal Beta:       {result_cv['beta']:.3f}")
print(f"\n  Asian/European Ratio: {result_cv['price_cv']/asian.european_call_bs()*100:.1f}%")

# Convergence test
print(f"\n{'='*80}")
print("CONVERGENCE ANALYSIS (Control Variate vs Standard)")
print("="*80)

path_sizes = [1000, 2500, 5000, 10000, 20000, 50000]
prices_standard = []
prices_cv = []
se_standard = []
se_cv = []

for n in path_sizes:
    res = asian.price_with_control_variate(n_paths=n)
    prices_standard.append(res['price_standard'])
    prices_cv.append(res['price_cv'])
    se_standard.append(res['std_error_standard'])
    se_cv.append(res['std_error_cv'])

print(f"\n{'Paths':>8} {'Standard MC':>15} {'Std Error':>12} {'Control Var':>15} {'Std Error':>12} {'Speedup':>10}")
print("-"*80)
for i, n in enumerate(path_sizes):
    speedup = (se_standard[i] / se_cv[i])**2
    print(f"{n:>8}    ${prices_standard[i]:>13.4f}  ${se_standard[i]:>10.4f}    ${prices_cv[i]:>13.4f}  ${se_cv[i]:>10.4f}    {speedup:>9.1f}x")

# Sensitivity analysis
print(f"\n{'='*80}")
print("ASIAN PREMIUM vs NUMBER OF OBSERVATIONS")
print("="*80)

obs_counts = [4, 12, 52, 252]
print(f"\n{'Observations':>15} {'Asian Price':>15} {'European Price':>18} {'Asian/Euro %':>15}")
print("-"*70)
for n_obs in obs_counts:
    asian_temp = AsianOption(S0, K, T, r, sigma, q, n_observations=n_obs, option_type='call')
    res_temp = asian_temp.price_with_control_variate(n_paths=10000)
    euro_temp = asian_temp.european_call_bs()
    print(f"{n_obs:>15}    ${res_temp['price_cv']:>13.4f}    ${euro_temp:>16.4f}    {res_temp['price_cv']/euro_temp*100:>13.1f}%")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Sample paths with average
n_sample = 100
paths_sample, obs_sample = asian.simulate_paths(n_sample, n_steps=252)
time_grid = np.linspace(0, T, 253)
obs_time = np.linspace(0, T, asian.n_observations)

for i in range(min(n_sample, 50)):
    axes[0, 0].plot(time_grid, paths_sample[i, :], alpha=0.2, color='blue', linewidth=0.5)
axes[0, 0].scatter(obs_time, obs_sample[0, :], color='red', s=50, zorder=5, label='Observations')
avg_path = np.mean(obs_sample[0, :])
axes[0, 0].axhline(avg_path, color='green', linestyle='--', linewidth=2, label=f'Average = ${avg_path:.2f}')
axes[0, 0].axhline(K, color='black', linestyle=':', linewidth=2, label='Strike K')
axes[0, 0].set_title('Sample Path with Observations (1 path shown)')
axes[0, 0].set_xlabel('Time (years)')
axes[0, 0].set_ylabel('Stock Price S')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Arithmetic vs Geometric average distribution
arith_avg = result_cv['arithmetic_avg']
geom_avg = result_cv['geometric_avg']

axes[0, 1].hist(arith_avg, bins=50, alpha=0.6, color='blue', label='Arithmetic Avg', edgecolor='black')
axes[0, 1].hist(geom_avg, bins=50, alpha=0.6, color='red', label='Geometric Avg', edgecolor='black')
axes[0, 1].axvline(K, color='black', linestyle='--', linewidth=2, label='Strike K')
axes[0, 1].set_title('Distribution of Averages (Arithmetic vs Geometric)')
axes[0, 1].set_xlabel('Average Price')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Convergence comparison (Standard vs Control Variate)
axes[0, 2].loglog(path_sizes, se_standard, 'bo-', linewidth=2, markersize=8, label='Standard MC')
axes[0, 2].loglog(path_sizes, se_cv, 'rs-', linewidth=2, markersize=8, label='Control Variate')
# Reference line O(1/sqrt(N))
ref_line = se_standard[0] * np.sqrt(path_sizes[0] / np.array(path_sizes))
axes[0, 2].loglog(path_sizes, ref_line, 'k--', linewidth=1.5, alpha=0.5, label='O(1/√N)')
axes[0, 2].set_title('Standard Error Convergence')
axes[0, 2].set_xlabel('Number of Paths')
axes[0, 2].set_ylabel('Standard Error ($)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3, which='both')

# Plot 4: Asian price vs # observations
obs_range = [2, 4, 12, 26, 52, 104, 252]
asian_prices = []
euro_price = asian.european_call_bs()

for n_obs in obs_range:
    asian_temp = AsianOption(S0, K, T, r, sigma, q, n_observations=n_obs)
    res_temp = asian_temp.price_with_control_variate(n_paths=8000)
    asian_prices.append(res_temp['price_cv'])

axes[1, 0].plot(obs_range, asian_prices, 'go-', linewidth=2.5, markersize=8, label='Asian Call')
axes[1, 0].axhline(euro_price, color='blue', linestyle='--', linewidth=2, label='European Call')
axes[1, 0].set_title('Asian Price vs # Observations')
axes[1, 0].set_xlabel('Number of Observations')
axes[1, 0].set_ylabel('Option Price ($)')
axes[1, 0].set_xscale('log')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, which='both')

# Plot 5: Asian vs European across spot prices
spot_range = np.linspace(80, 120, 15)
asian_prices_spot = []
euro_prices_spot = []

for s in spot_range:
    asian_temp = AsianOption(s, K, T, r, sigma, q, n_observations=12)
    res_temp = asian_temp.price_with_control_variate(n_paths=5000)
    asian_prices_spot.append(res_temp['price_cv'])
    euro_prices_spot.append(asian_temp.european_call_bs())

axes[1, 1].plot(spot_range, asian_prices_spot, 'r-', linewidth=2.5, label='Asian Call', marker='o')
axes[1, 1].plot(spot_range, euro_prices_spot, 'b--', linewidth=2.5, label='European Call', marker='s')
axes[1, 1].fill_between(spot_range, asian_prices_spot, euro_prices_spot, 
                       alpha=0.3, color='green', label='Premium Difference')
axes[1, 1].axvline(K, color='black', linestyle=':', alpha=0.5, label='Strike')
axes[1, 1].set_title('Asian vs European Call Value')
axes[1, 1].set_xlabel('Spot Price S')
axes[1, 1].set_ylabel('Option Value ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Variance reduction factor vs beta
beta_range = np.linspace(0, 2, 50)
var_reductions = 1 - (1 - 0.95**2 * beta_range**2)  # Assuming rho=0.95

axes[1, 2].plot(beta_range, var_reductions*100, 'b-', linewidth=2.5)
axes[1, 2].axvline(result_cv['beta'], color='red', linestyle='--', linewidth=2, 
                  label=f'Optimal β={result_cv["beta"]:.2f}')
axes[1, 2].axhline(result_cv['variance_reduction']*100, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.5)
axes[1, 2].set_title('Variance Reduction vs Beta (ρ=0.95)')
axes[1, 2].set_xlabel('Beta Coefficient')
axes[1, 2].set_ylabel('Variance Reduction (%)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('asian_options.png', dpi=100, bbox_inches='tight')
print("\n" + "="*80)
print("Plot saved: asian_options.png")
print("="*80)
```

**Output Interpretation:**
- **Asian premium:** 55-65% of European (depends on # observations)
- **Control variate:** 30-50% variance reduction (3-5× faster convergence)
- **More observations:** Lower Asian value (smoother average → less volatility → cheaper option)

## 6. Challenge Round

**Q1: Why is arithmetic Asian call worth ~55% of European call (same parameters)?**  
A: Averaging reduces effective volatility. European payoff max(S_T-K,0) depends on single terminal spot → full σ volatility. Asian payoff max(Ā-K,0) depends on average → effective vol ≈ σ/√n (n observations). Example: σ=20%, n=12 → σ_eff ≈ 20%/√12 = 5.8%. Lower vol → lower option value. Intuition: Path averaging smooths swings → less extreme payoffs → cheaper premium. Empirical: 12 observations → 50-60% discount, 52 observations → 35-45% discount.

**Q2: Geometric Asian closed-form price is $5.80; arithmetic MC gives $6.50. Why geometric < arithmetic always?**  
A: Jensen's inequality for concave functions: E[ln(X)] < ln(E[X]) → exp(E[ln(X)]) < E[X] → Geometric mean < Arithmetic mean. Example: prices $80, $120 → Arithmetic avg = $100, Geometric avg = √(80×120) = $98. Asian call: max(Avg-K,0) → lower average → lower payoff → cheaper option. Difference: 10-15% typically (geometric 85-90% of arithmetic). Use case: Geometric as control variate → highly correlated (ρ=0.95+) → variance reduction.

**Q3: A commodity hedger uses daily observations (n=252) for Asian call. Why is premium only 30% of European?**  
A: Daily averaging → extreme smoothing. Effective vol: σ_eff ≈ σ/√252 ≈ σ/15.9 → volatility reduced 16×. Example: σ=20% → σ_eff ≈ 1.26% (tiny). Asian value driven by drift (r-q), not volatility. Near-deterministic average → minimal optionality → cheap premium. Trade-off: Perfect hedging (true average exposure) vs. cost (30% of European). Corporate hedgers prefer: lower cost acceptable for average protection (cash flow hedging, not speculation).

**Q4: Control variate using geometric Asian reduces variance 40%. Explain the mechanism.**  
A: V_A^{CV} = V_A - β(V_G^{MC} - V_G^{exact}). Arithmetic payoff V_A and geometric payoff V_G highly correlated (ρ≈0.95) → both rise/fall together. Subtract correlated noise: V_G^{MC} fluctuates around V_G^{exact} → subtracting (V_G^{MC} - V_G^{exact}) removes correlated component of V_A → residual V_A^{CV} has lower variance. Formula: Var(V_A^{CV}) = Var(V_A)(1-ρ²). If ρ=0.95 → Var reduced to (1-0.9025)=9.75% of original → 90% variance reduction → 3× fewer paths for same SE. Practical: Always use geometric control for arithmetic Asian (free variance reduction).

**Q5: Asian put strike K=$100, observed quarterly (n=4). Early observations: $120, $110, $100. Final observation S_T=? for put to finish ITM.**  
A: Current average (3 obs): Ā₃ = ($120+$110+$100)/3 = $110. For ITM at expiry: need Ā₄ < K=$100. Ā₄ = (Ā₃×3 + S_T)/4 = (330 + S_T)/4 < 100 → 330 + S_T < 400 → S_T < $70. Final observation must drop below $70 for put to pay off. Payoff: max(100 - Ā₄, 0). If S_T=$70: Ā₄=$100 → payoff=0 (at strike). If S_T=$50: Ā₄=$95 → payoff=$5. Intuition: Early high observations "anchor" average high → need extreme final drop to push average below strike.

**Q6: Why do commodity producers prefer Asian options over European for hedging annual production?**  
A: Production occurs continuously (monthly/weekly sales) → exposed to average price, not spot at one date. European put hedges S_T (single point) → mismatch with revenue stream. Asian put hedges Ā (average over production period) → perfect match with cash flows. Example: Oil producer sells 100k barrels/month × 12 months → revenue = Ā × 1.2M barrels. Asian put (K=$60/bbl) guarantees floor: min revenue = $60×1.2M = $72M. Cost advantage: Asian 50-60% cheaper → better budget fit. No manipulation risk: average harder to manipulate than single-day settlement.

## 7. Key References

- Kemna & Vorst (1990): "Pricing Method for Options Based on Average Asset Values" — Geometric Asian closed-form
- [Wikipedia: Asian Option](https://en.wikipedia.org/wiki/Asian_option) — Path-dependency, averaging mechanics
- Rogers & Shi (1995): "Value of Asian Option" — Control variate methods, variance reduction
- Hull: *Options, Futures & Derivatives* (Chapter 26) — Exotic options, path-dependent structures

**Status:** ✓ Standalone file. **Complements:** european_option_pricing.md, control_variates.md, lookback_options.md, barrier_options.md
