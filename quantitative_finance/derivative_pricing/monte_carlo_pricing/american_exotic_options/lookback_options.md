# Lookback Options

## 1. Concept Skeleton
**Definition:** Path-dependent options with payoff based on maximum (floating strike call, fixed strike put) or minimum (floating strike put, fixed strike call) price over entire option life, offering guaranteed optimal execution but higher premium.  
**Purpose:** Capture best/worst price retrospectively; eliminate timing risk; hedging hindsight regret; synthetic stop-loss; guaranteed optimal exit; structured product features.  
**Prerequisites:** European option pricing, path simulation, running max/min tracking, order statistics, extreme value distributions

## 2. Comparative Framing

| Aspect | Floating Strike Call | Fixed Strike Lookback Call | European Call | Asian Call |
|--------|---------------------|----------------------------|---------------|------------|
| **Payoff** | S_T - min(S_t) | max(S_t) - K | S_T - K | Ā - K |
| **Strike** | Min price (optimal) | Fixed K | Fixed K | Fixed K |
| **Premium vs European** | 200-300% (very expensive) | 150-200% (expensive) | 100% (baseline) | 50-70% (cheaper) |
| **Path Dependency** | Extreme (track min) | Extreme (track max) | None | Moderate (average) |
| **Hindsight Advantage** | Always optimal strike | Always optimal execution | None | Smoothed |
| **Volatility Sensitivity** | Very high (more vol → wider range) | Very high | High | Low (averaging) |
| **Use Cases** | Guaranteed best entry | Guaranteed best exit | Standard hedging | Cost-effective hedging |

## 3. Examples + Counterexamples

**Simple Example: Floating Strike Lookback Call**  
Stock path over 1 year: $100→$110→$95→$105→$120→$115. Minimum price: min(S_t) = $95. Terminal price: S_T = $115. Payoff: S_T - min = $115 - $95 = $20. Interpretation: Bought at lowest point ($95), sold at final ($115). European call (K=$100): payoff = $115-$100 = $15. Lookback advantage: $20 vs $15 (+33%). Premium: ~$25 (250% of European ~$10). Trade-off: Guaranteed optimal entry vs 2.5× cost.

**Failure Case: Low Volatility Path**  
Floating strike lookback call, stock stays flat $100-$102 entire year. Minimum: $100, Terminal: $101. Payoff: $101-$100 = $1. Premium paid: $25. Loss: -$24. European call (K=$100): payoff $1, premium $5, loss -$4. Lookback magnifies loss in low-vol scenarios (overpaid for optionality not realized). Lesson: Lookback profitable only if realized vol > implied vol significantly (need wide price swings).

**Edge Case: Extreme Swing Then Reversal**  
Fixed strike lookback call: K=$100, stock path $100→$150→$90→$100. Maximum price: max(S_t) = $150. Payoff: max-K = $150-$100 = $50. Terminal: S_T=$100 (back at start). European call: payoff = $0 (ATM at expiry). Lookback: captures $150 peak even though ended flat. This "hindsight perfection" why premium 2-3× European. Investor guaranteed sold at absolute peak regardless of timing.

## 4. Layer Breakdown

```
Lookback Option Framework:
├─ Payoff Types (4 Main Variants):
│   ├─ Floating Strike Lookback Call:
│   │   ├─ Payoff: S_T - min(S_t) for t ∈ [0,T]
│   │   ├─ Interpretation: Buy at historical minimum, sell at expiry
│   │   ├─ Strike: Floating (set to minimum ex-post)
│   │   └─ Always ITM: Payoff ≥ 0 always (S_T ≥ min by definition)
│   ├─ Floating Strike Lookback Put:
│   │   ├─ Payoff: max(S_t) - S_T for t ∈ [0,T]
│   │   ├─ Interpretation: Sell at historical maximum, buy back at expiry
│   │   ├─ Strike: Floating (set to maximum ex-post)
│   │   └─ Always ITM: Payoff ≥ 0 always (max ≥ S_T by definition)
│   ├─ Fixed Strike Lookback Call:
│   │   ├─ Payoff: max(max(S_t) - K, 0) for t ∈ [0,T]
│   │   ├─ Interpretation: Guaranteed execution at best price if ever > K
│   │   ├─ Strike: Fixed K (predetermined)
│   │   └─ Can be OTM: If max(S_t) < K throughout, payoff = 0
│   ├─ Fixed Strike Lookback Put:
│   │   ├─ Payoff: max(K - min(S_t), 0) for t ∈ [0,T]
│   │   ├─ Interpretation: Guaranteed execution at best price if ever < K
│   │   ├─ Strike: Fixed K
│   │   └─ Can be OTM: If min(S_t) > K throughout, payoff = 0
│   └─ Partial Lookback (Relaxed):
│       ├─ Payoff: S_T - λ×min(S_t) - (1-λ)×S_T = λ(S_T - min) for λ ∈ [0,1]
│       ├─ Parameter λ: "Lookback fraction" (λ=1 → full lookback, λ=0 → European)
│       ├─ Premium: Scales with λ (fractional lookback cheaper)
│       └─ Use: Cost-reduction while retaining partial hindsight benefit
├─ Path Dependency Mechanics:
│   ├─ Running Maximum/Minimum:
│   │   ├─ Track M_t = max{S_u : 0 ≤ u ≤ t} (running max)
│   │   ├─ Track m_t = min{S_u : 0 ≤ u ≤ t} (running min)
│   │   ├─ Update: M_{i+1} = max(M_i, S_{i+1}) at each time step
│   │   └─ Terminal values: M_T, m_T used in payoff calculation
│   ├─ Extreme Value Distribution:
│   │   ├─ Under GBM: max(S_t) ~ reflected/absorbed Brownian motion distribution
│   │   ├─ Probability: P(max > H) depends on drift μ, vol σ, barrier H
│   │   └─ Closed-form exists for some cases (Goldman-Sosin-Gatto formula)
│   └─ Continuous vs Discrete Monitoring:
│       ├─ Continuous: min/max over all t ∈ [0,T] (theoretical)
│       ├─ Discrete: min/max over t₁, t₂, ..., t_n (practical)
│       ├─ Difference: Discrete monitoring misses intraday extremes → cheaper by 5-10%
│       └─ MC Implementation: Discrete monitoring default (store running min/max)
├─ Monte Carlo Pricing:
│   ├─ Algorithm:
│   │   ├─ 1. Simulate M paths: S₀, S₁, ..., S_N (N time steps)
│   │   ├─ 2. For each path m, track:
│   │   │   ├─ Running max: M^(m) = max(S₀^(m), S₁^(m), ..., S_N^(m))
│   │   │   └─ Running min: m^(m) = min(S₀^(m), S₁^(m), ..., S_N^(m))
│   │   ├─ 3. Compute payoff:
│   │   │   ├─ Floating call: S_T^(m) - m^(m)
│   │   │   ├─ Floating put: M^(m) - S_T^(m)
│   │   │   ├─ Fixed call: max(M^(m) - K, 0)
│   │   │   └─ Fixed put: max(K - m^(m), 0)
│   │   ├─ 4. Discount: V^(m) = e^{-rT} × payoff^(m)
│   │   └─ 5. Average: V_lookback = mean(V^(m))
│   └─ Convergence: Standard O(1/√M) like other MC methods
├─ Closed-Form Approximations:
│   ├─ Goldman-Sosin-Gatto (1979) Formula:
│   │   ├─ Floating Strike Call: Complex integral involving M₁, M₂ (first two moments of max)
│   │   ├─ E[S_T - min] under GBM → closed-form with modified BS parameters
│   │   ├─ Adjustment: σ_eff accounts for max/min distribution (not standard normal)
│   │   └─ Use: Benchmark for MC validation
│   ├─ Conze-Viswanathan (1991) Formula:
│   │   ├─ Fixed strike lookback options
│   │   ├─ max(max(S_t)-K, 0) → closed-form via barrier option reflection
│   │   └─ Complexity: Requires numerical integration (not fully closed)
│   └─ Practical Use:
│       ├─ Closed-form: Fast pricing, Greeks via finite differences
│       ├─ MC: Flexible for complex paths, multi-asset, discrete monitoring
│       └─ Hybrid: Use closed-form for simple cases, MC for exotics
├─ Greeks & Sensitivities:
│   ├─ Delta (∂V/∂S):
│   │   ├─ Floating call: ∂(S_T - min)/∂S₀ → complex (min depends on S₀)
│   │   ├─ Discontinuous when S₀ = current min or max
│   │   └─ Higher delta than European (path dependency amplifies sensitivity)
│   ├─ Gamma (∂²V/∂S²):
│   │   ├─ Spikes when S near running min/max (path extremes)
│   │   ├─ Higher convexity → more rehedging needed
│   │   └─ Challenge: Dynamic hedging expensive (frequent rebalancing)
│   ├─ Vega (∂V/∂σ):
│   │   ├─ Very high: Lookback value ∝ price range ∝ volatility
│   │   ├─ Floating call: Higher vol → wider min-to-terminal range → higher payoff
│   │   ├─ Vega 2-3× European vega (extreme sensitivity to vol)
│   │   └─ Risk: Implied vol changes → large P&L swings
│   └─ Theta (∂V/∂t):
│       ├─ Negative (time decay) but less severe than European
│       ├─ Reason: Lookback retains value as extremes locked in over time
│       └─ Early in life (extremes volatile), late in life (extremes stable) → theta varies
├─ Premium Structure:
│   ├─ Floating Strike:
│   │   ├─ Always ITM → intrinsic value = expected(S_T - min) > 0
│   │   ├─ Premium ≈ 200-300% of ATM European call (depends on T, σ)
│   │   ├─ Breakdown: 50% intrinsic (E[S_T-min]), 150% time value (vol benefit)
│   │   └─ Example: European $10, Floating lookback $25-30
│   ├─ Fixed Strike:
│   │   ├─ Can be OTM → cheaper than floating (but still > European)
│   │   ├─ Premium ≈ 150-200% of European (depends on moneyness)
│   │   ├─ ATM fixed lookback ≈ 180% of European
│   │   └─ Example: European $10, Fixed lookback $18-20
│   └─ Drivers:
│       ├─ High volatility → wider ranges → higher lookback value
│       ├─ Long maturity → more time to reach extremes → higher value
│       └─ Dividend yield → reduces lookback value (leakage during holding period)
└─ Applications:
    ├─ Guaranteed Best Execution:
    │   ├─ Investor wants to sell at peak (can't time market)
    │   ├─ Buy floating strike put → guaranteed sell at historical max
    │   └─ Trade-off: 2-3× premium vs timing uncertainty eliminated
    ├─ Employee Stock Options (ESO):
    │   ├─ Company offers lookback feature: "Exercise at lowest price last year"
    │   ├─ Retention incentive (extremely valuable to employee)
    │   └─ Cost: Company bears 2-3× higher option expense vs standard ESO
    ├─ Structured Products:
    │   ├─ "Best-of" note: Pays max(S_T, peak over period) - K
    │   ├─ Partial lookback: λ=0.5 → 50% hindsight benefit, 50% cost of full
    │   └─ Retail appeal: "Can't lose" narrative (always get best price)
    └─ Hedging Regret:
        ├─ Trader fears missing rally peak (FOMO)
        ├─ Lookback call guarantees participation in peak
        └─ Psychological value > mathematical value (behavioral finance)
```

**Interaction:** Path generation → Running max/min tracking → Payoff from extremes → Discounting → Lookback value

## 5. Mini-Project

Price floating and fixed strike lookback options; compare to European:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class LookbackOption:
    """Monte Carlo pricing for lookback options"""
    
    def __init__(self, S0, K, T, r, sigma, q=0, lookback_type='floating', 
                 option_type='call', lookback_fraction=1.0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.lookback_type = lookback_type  # 'floating' or 'fixed'
        self.option_type = option_type  # 'call' or 'put'
        self.lookback_fraction = lookback_fraction  # λ ∈ [0,1] for partial lookback
    
    def simulate_paths(self, n_paths, n_steps):
        """Generate GBM paths"""
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for i in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            paths[:, i+1] = paths[:, i] * np.exp(
                (self.r - self.q - 0.5*self.sigma**2)*dt + 
                self.sigma*np.sqrt(dt)*Z
            )
        
        return paths
    
    def compute_payoff(self, paths):
        """Compute lookback payoff"""
        S_T = paths[:, -1]
        running_max = np.max(paths, axis=1)
        running_min = np.min(paths, axis=1)
        
        if self.lookback_type == 'floating':
            if self.option_type == 'call':
                # Floating strike call: S_T - λ×min (partial lookback)
                payoff = S_T - self.lookback_fraction * running_min - \
                        (1 - self.lookback_fraction) * self.S0
            else:  # put
                # Floating strike put: λ×max - S_T
                payoff = self.lookback_fraction * running_max + \
                        (1 - self.lookback_fraction) * self.S0 - S_T
        else:  # fixed strike
            if self.option_type == 'call':
                # Fixed strike lookback call: max(max - K, 0)
                payoff = np.maximum(running_max - self.K, 0)
            else:  # put
                # Fixed strike lookback put: max(K - min, 0)
                payoff = np.maximum(self.K - running_min, 0)
        
        return payoff, running_max, running_min
    
    def price(self, n_paths=20000, n_steps=252):
        """Price lookback option via MC"""
        paths = self.simulate_paths(n_paths, n_steps)
        payoff, running_max, running_min = self.compute_payoff(paths)
        
        # Discount
        price = np.exp(-self.r*self.T) * np.mean(payoff)
        std_error = np.exp(-self.r*self.T) * np.std(payoff) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'paths': paths,
            'payoff': payoff,
            'running_max': running_max,
            'running_min': running_min,
            'avg_range': np.mean(running_max - running_min)
        }
    
    def european_price_bs(self):
        """Black-Scholes European price for comparison"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        if self.option_type == 'call':
            return self.S0*np.exp(-self.q*self.T)*norm.cdf(d1) - \
                   self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:  # put
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - \
                   self.S0*np.exp(-self.q*self.T)*norm.cdf(-d1)

# Parameters
S0, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.25, 0.02
np.random.seed(42)

print("="*80)
print("LOOKBACK OPTION PRICING")
print("="*80)

# Price floating strike lookback call
floating_call = LookbackOption(S0, K, T, r, sigma, q, lookback_type='floating', 
                               option_type='call', lookback_fraction=1.0)
result_floating = floating_call.price(n_paths=20000, n_steps=252)

# Price fixed strike lookback call
fixed_call = LookbackOption(S0, K, T, r, sigma, q, lookback_type='fixed', 
                            option_type='call')
result_fixed = fixed_call.price(n_paths=20000, n_steps=252)

# European benchmark
european_price = floating_call.european_price_bs()

print(f"\nOption Parameters:")
print(f"  Spot S0:         ${S0:.2f}")
print(f"  Strike K:        ${K:.2f}")
print(f"  Time to Expiry:  {T:.2f} years")
print(f"  Interest Rate:   {r*100:.1f}%")
print(f"  Volatility:      {sigma*100:.1f}%")
print(f"  Dividend Yield:  {q*100:.1f}%")

print(f"\n{'='*80}")
print("PRICING RESULTS")
print("="*80)
print(f"  European Call (BS):           ${european_price:.4f}")
print(f"  Floating Strike Lookback Call: ${result_floating['price']:.4f} ± ${result_floating['std_error']:.4f}")
print(f"  Fixed Strike Lookback Call:    ${result_fixed['price']:.4f} ± ${result_fixed['std_error']:.4f}")
print(f"\n  Floating/European Ratio: {result_floating['price']/european_price:.2f}x ({result_floating['price']/european_price*100:.0f}%)")
print(f"  Fixed/European Ratio:    {result_fixed['price']/european_price:.2f}x ({result_fixed['price']/european_price*100:.0f}%)")
print(f"\n  Average Price Range: ${result_floating['avg_range']:.2f}")

# Partial lookback (cost reduction)
print(f"\n{'='*80}")
print("PARTIAL LOOKBACK (Cost Reduction)")
print("="*80)

partial_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"\n{'Lookback %':>12} {'Price':>12} {'vs European':>15} {'Payoff Range':>18}")
print("-"*65)

for lf in partial_fractions:
    partial = LookbackOption(S0, K, T, r, sigma, q, lookback_type='floating', 
                            option_type='call', lookback_fraction=lf)
    res = partial.price(n_paths=10000, n_steps=252)
    ratio = res['price'] / european_price
    print(f"  {lf*100:>10.0f}%    ${res['price']:>10.4f}    {ratio:>13.2f}x    ${np.mean(res['payoff']):>16.2f}")

# Lookback put
print(f"\n{'='*80}")
print("LOOKBACK PUT OPTIONS")
print("="*80)

floating_put = LookbackOption(S0, K, T, r, sigma, q, lookback_type='floating', 
                              option_type='put', lookback_fraction=1.0)
fixed_put = LookbackOption(S0, K, T, r, sigma, q, lookback_type='fixed', 
                           option_type='put')

res_float_put = floating_put.price(n_paths=20000, n_steps=252)
res_fixed_put = fixed_put.price(n_paths=20000, n_steps=252)
euro_put = floating_put.european_price_bs()

print(f"\n  European Put (BS):             ${euro_put:.4f}")
print(f"  Floating Strike Lookback Put:  ${res_float_put['price']:.4f} ± ${res_float_put['std_error']:.4f}")
print(f"  Fixed Strike Lookback Put:     ${res_fixed_put['price']:.4f} ± ${res_fixed_put['std_error']:.4f}")
print(f"\n  Floating/European Ratio: {res_float_put['price']/euro_put:.2f}x")
print(f"  Fixed/European Ratio:    {res_fixed_put['price']/euro_put:.2f}x")

# Volatility sensitivity
print(f"\n{'='*80}")
print("VOLATILITY SENSITIVITY (Vega Analysis)")
print("="*80)

vol_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
print(f"\n{'Volatility':>12} {'Floating Call':>18} {'Fixed Call':>15} {'European Call':>18}")
print("-"*70)

for vol in vol_range:
    float_temp = LookbackOption(S0, K, T, r, vol, q, 'floating', 'call')
    fixed_temp = LookbackOption(S0, K, T, r, vol, q, 'fixed', 'call')
    
    res_float = float_temp.price(n_paths=8000, n_steps=100)
    res_fixed = fixed_temp.price(n_paths=8000, n_steps=100)
    euro = fixed_temp.european_price_bs()
    
    print(f"  {vol*100:>10.0f}%    ${res_float['price']:>16.4f}    ${res_fixed['price']:>13.4f}    ${euro:>16.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Sample paths with max/min
n_sample = 50
sample_paths = result_floating['paths'][:n_sample, :]
time_grid = np.linspace(0, T, 253)

for i in range(n_sample):
    axes[0, 0].plot(time_grid, sample_paths[i, :], alpha=0.3, color='blue', linewidth=0.8)
    # Mark min and max
    min_idx = np.argmin(sample_paths[i, :])
    max_idx = np.argmax(sample_paths[i, :])
    axes[0, 0].scatter(time_grid[min_idx], sample_paths[i, min_idx], 
                      color='red', s=30, alpha=0.6, zorder=5)
    axes[0, 0].scatter(time_grid[max_idx], sample_paths[i, max_idx], 
                      color='green', s=30, alpha=0.6, zorder=5)

axes[0, 0].axhline(K, color='black', linestyle='--', linewidth=2, label='Strike K')
axes[0, 0].set_title('Sample Paths (Red=Min, Green=Max)')
axes[0, 0].set_xlabel('Time (years)')
axes[0, 0].set_ylabel('Stock Price S')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Payoff distribution
axes[0, 1].hist(result_floating['payoff'], bins=50, alpha=0.6, color='blue', 
               edgecolor='black', label='Floating Call Payoff')
axes[0, 1].hist(result_fixed['payoff'], bins=50, alpha=0.6, color='red', 
               edgecolor='black', label='Fixed Call Payoff')
axes[0, 1].axvline(np.mean(result_floating['payoff']), color='blue', 
                  linestyle='--', linewidth=2, label=f'Mean Float=${np.mean(result_floating["payoff"]):.2f}')
axes[0, 1].axvline(np.mean(result_fixed['payoff']), color='red', 
                  linestyle='--', linewidth=2, label=f'Mean Fixed=${np.mean(result_fixed["payoff"]):.2f}')
axes[0, 1].set_title('Lookback Payoff Distributions')
axes[0, 1].set_xlabel('Payoff at Expiry')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price range (max-min) distribution
price_ranges = result_floating['running_max'] - result_floating['running_min']
axes[0, 2].hist(price_ranges, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 2].axvline(np.mean(price_ranges), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean Range=${np.mean(price_ranges):.2f}')
axes[0, 2].set_title('Price Range Distribution (Max - Min)')
axes[0, 2].set_xlabel('Price Range ($)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Partial lookback cost reduction
partial_fracs = np.linspace(0, 1, 11)
prices_partial = []
for lf in partial_fracs:
    partial = LookbackOption(S0, K, T, r, sigma, q, 'floating', 'call', lf)
    res = partial.price(n_paths=5000, n_steps=100)
    prices_partial.append(res['price'])

axes[1, 0].plot(partial_fracs*100, prices_partial, 'bo-', linewidth=2.5, markersize=8)
axes[1, 0].axhline(european_price, color='red', linestyle='--', linewidth=2, 
                  label=f'European (λ=0) = ${european_price:.2f}')
axes[1, 0].set_title('Partial Lookback: Price vs Lookback Fraction λ')
axes[1, 0].set_xlabel('Lookback Fraction λ (%)')
axes[1, 0].set_ylabel('Option Price ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Volatility sensitivity comparison
vol_range_plot = np.linspace(0.10, 0.50, 15)
prices_float_vol = []
prices_fixed_vol = []
prices_euro_vol = []

for vol in vol_range_plot:
    float_temp = LookbackOption(S0, K, T, r, vol, q, 'floating', 'call')
    fixed_temp = LookbackOption(S0, K, T, r, vol, q, 'fixed', 'call')
    
    res_float = float_temp.price(n_paths=5000, n_steps=100)
    res_fixed = fixed_temp.price(n_paths=5000, n_steps=100)
    euro = fixed_temp.european_price_bs()
    
    prices_float_vol.append(res_float['price'])
    prices_fixed_vol.append(res_fixed['price'])
    prices_euro_vol.append(euro)

axes[1, 1].plot(vol_range_plot*100, prices_float_vol, 'b-', linewidth=2.5, 
               marker='o', markersize=6, label='Floating Lookback')
axes[1, 1].plot(vol_range_plot*100, prices_fixed_vol, 'r-', linewidth=2.5, 
               marker='s', markersize=6, label='Fixed Lookback')
axes[1, 1].plot(vol_range_plot*100, prices_euro_vol, 'g--', linewidth=2.5, 
               marker='^', markersize=6, label='European')
axes[1, 1].set_title('Price vs Volatility (Vega)')
axes[1, 1].set_xlabel('Volatility (%)')
axes[1, 1].set_ylabel('Option Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Premium ratios across moneyness
moneyness_range = np.linspace(0.85, 1.15, 15)
ratios_float = []
ratios_fixed = []

for m in moneyness_range:
    s_temp = S0 * m
    float_temp = LookbackOption(s_temp, K, T, r, sigma, q, 'floating', 'call')
    fixed_temp = LookbackOption(s_temp, K, T, r, sigma, q, 'fixed', 'call')
    
    res_float = float_temp.price(n_paths=5000, n_steps=100)
    res_fixed = fixed_temp.price(n_paths=5000, n_steps=100)
    euro = fixed_temp.european_price_bs()
    
    ratios_float.append(res_float['price'] / euro if euro > 0.01 else np.nan)
    ratios_fixed.append(res_fixed['price'] / euro if euro > 0.01 else np.nan)

axes[1, 2].plot(moneyness_range, ratios_float, 'b-', linewidth=2.5, marker='o', 
               markersize=8, label='Floating / European')
axes[1, 2].plot(moneyness_range, ratios_fixed, 'r-', linewidth=2.5, marker='s', 
               markersize=8, label='Fixed / European')
axes[1, 2].axvline(1.0, color='black', linestyle=':', alpha=0.5, label='ATM')
axes[1, 2].axhline(1.0, color='green', linestyle='--', alpha=0.5, label='European Baseline')
axes[1, 2].set_title('Premium Ratio vs Moneyness (S/K)')
axes[1, 2].set_xlabel('Moneyness (S/K)')
axes[1, 2].set_ylabel('Lookback / European Ratio')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lookback_options.png', dpi=100, bbox_inches='tight')
print("\n" + "="*80)
print("Plot saved: lookback_options.png")
print("="*80)
```

**Output Interpretation:**
- **Floating lookback:** 2.2-2.8× European premium (guaranteed optimal strike)
- **Fixed lookback:** 1.6-2.0× European premium (guaranteed optimal execution)
- **Partial lookback (λ=0.5):** 50% cost reduction vs full lookback, still captures half hindsight benefit

## 6. Challenge Round

**Q1: Floating strike lookback call costs $25; European call (K=$100) costs $10. Why 2.5× premium?**  
A: Floating strike = min(S_t) ex-post → always buys at lowest price. European fixed strike K=$100. Expected advantage: E[S_T - min] > E[S_T - K] for fixed K. Example: Path min=$85, terminal S_T=$115 → Floating payoff=$30, European payoff=$15 (2× payoff). But not all paths achieve 2× → average ~1.5× payoff. Premium 2.5× reflects: 1.5× higher expected payoff + convexity (optionality on min) + vol sensitivity (higher vega). Intuition: Guaranteed perfect timing worth 150% premium.

**Q2: Stock stays flat $100±$2 entire year (low vol). Floating lookback call payoff?**  
A: Min price ≈ $98, Terminal ≈ $100. Payoff: $100-$98 = $2. Premium paid: $25. Net loss: -$23. European call (K=$100): payoff ~$0, premium $10, loss -$10. Lookback magnifies loss (overpaid for optionality). Lesson: Lookback profitable only if realized vol >> implied vol (need wide swings to justify 2-3× premium). Low-vol environments: lookback loses big (paid for hindsight, got no price movement to exploit).

**Q3: Fixed strike lookback call (K=$100) vs European call. When does fixed lookback provide no extra value?**  
A: If max(S_t) = S_T (peak at expiry), no hindsight advantage. Fixed lookback payoff: max-K = S_T-K (same as European). Premium wasted. Example: Monotonic path $100→$120 → max=$120 at T → both pay $20. But European cost $10, lookback cost $18 → wasted $8. Lookback adds value only if peak occurs before expiry (mid-path maximum > terminal). Probability: ~50% for random walk → lookback justified if expect mean-reverting paths (peak then fall).

**Q4: Partial lookback (λ=0.5): Payoff = 0.5×(S_T - min) + 0.5×(S_T - S₀). Interpret economically.**  
A: Weighted average of lookback (S_T-min) and European (S_T-S₀). λ=0.5 → 50% hindsight benefit, 50% standard option. Example: min=$85, S_T=$115, S₀=$100. Full lookback: $115-$85=$30. European: $115-$100=$15. Partial: 0.5×$30 + 0.5×$15 = $22.50. Premium: ~60% of full lookback (vs 250% of European for full). Cost savings: 40% vs full lookback, still captures half the hindsight advantage. Corporate use: Budget-constrained hedger accepts partial regret protection at affordable cost.

**Q5: Floating strike put (max-S_T) vs fixed strike put (K-min). Which more expensive?**  
A: Floating more expensive (always ITM). Floating payoff: max-S_T ≥ 0 always (max ≥ S_T by definition). Intrinsic value at inception: E[max-S_T] > 0. Fixed strike put: max(K-min, 0) → can be worthless if min > K (stock never drops below strike). Fixed has OTM scenarios → cheaper. Example: Stock rises entire year ($100→$120) → min=$100. Fixed put (K=$100): payoff=$0. Floating put: max=$120, S_T=$120 → payoff=$0 also (but higher expected value across all paths). Typical: Floating put 2.5-3× European put, Fixed put 1.8-2.2× European put.

**Q6: Discrete monitoring (daily) vs continuous monitoring for lookback. Impact on price?**  
A: Discrete monitoring misses intraday extremes → lower max, higher min → smaller payoff → cheaper option. Example: Intraday spike $100→$130→$105 (close). Daily monitoring: max=$105. Continuous: max=$130. Floating call payoff (continuous) = S_T-min includes intraday extremes → higher value. Difference: 5-10% typically (daily ≈ 92-95% of continuous value). Real markets: Options specified with discrete monitoring (daily close) → practical implementation matches contracts. MC: Discrete monitoring default (252 daily observations standard).

## 7. Key References

- Goldman, Sosin, Gatto (1979): "Path Dependent Options" — Closed-form floating lookback formulas
- Conze & Viswanathan (1991): "Path Dependent Options: Buy at Low, Sell at High" — Fixed strike lookback pricing
- [Wikipedia: Lookback Option](https://en.wikipedia.org/wiki/Lookback_option) — Floating/fixed types, hindsight mechanics
- Hull: *Options, Futures & Derivatives* (Chapter 26) — Exotic options, lookback structures, premium analysis

**Status:** ✓ Standalone file. **Complements:** asian_options.md, barrier_options.md, european_option_pricing.md, path_dependent_options.md
