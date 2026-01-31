# Lookback Options

## 1. Concept Skeleton
**Definition:** Path-dependent options with payoff based on maximum or minimum asset price observed over option's life  
**Purpose:** Hindsight perfection; capture best price; speculation on extreme moves; no regret trading  
**Prerequisites:** Path-dependent payoffs, extreme value statistics, Monte Carlo simulation, min/max tracking

## 2. Comparative Framing
| Feature | Fixed Strike Lookback | Floating Strike Lookback | Asian Option | European Vanilla |
|---------|----------------------|--------------------------|--------------|------------------|
| **Payoff** | max(M - K, 0) or max(K - m, 0) | max(S_T - m, 0) or max(M - S_T, 0) | max(Avg - K, 0) | max(S_T - K, 0) |
| **Strike** | Fixed K | Floating (min or max) | Fixed K | Fixed K |
| **Value** | Very high | Highest | Moderate | Lowest |
| **Vega** | High (extreme sensitivity) | Very high | Low (averaging) | Moderate |
| **Manipulation** | Impossible (max/min) | Impossible | Hard | Easy (spot at T) |

## 3. Examples + Counterexamples

**Simple Example:**  
Fixed strike lookback call: K=$100, Path=[95, 110, 105, 98, 108], Max=110 → Payoff = 110 - 100 = $10

**Failure Case:**  
Single observation: Lookback → European (max of one point = terminal price); loses lookback premium

**Edge Case:**  
Zero volatility: Max = Min = S₀ → Lookback call = max(S₀ - K, 0); floating strike call = 0 (S_T - m = 0)

## 4. Layer Breakdown
```
Lookback Option Classification & Pricing:
├─ Fixed Strike Lookback:
│   ├─ Fixed Strike Call:
│   │   ├─ Payoff: max(M - K, 0) where M = max(S_t) over t ∈ [0, T]
│   │   ├─ Exercise: Optimal hindsight (buy at strike, sell at max)
│   │   ├─ Value: Higher than European (M ≥ S_T always)
│   │   └─ Use Case: Capture upside extremes without timing risk
│   ├─ Fixed Strike Put:
│   │   ├─ Payoff: max(K - m, 0) where m = min(S_t) over t ∈ [0, T]
│   │   ├─ Exercise: Hindsight sell at strike, buy at min
│   │   ├─ Value: Higher than European (m ≤ S_T always)
│   │   └─ Use Case: Capture downside protection at best price
│   └─ Properties:
│       ├─ Always ITM: M > K for call (if ever crosses), m < K for put
│       ├─ No Regret: Guarantees best price over period
│       └─ Premium: Expensive (captures tail value)
├─ Floating Strike Lookback:
│   ├─ Floating Strike Call:
│   │   ├─ Payoff: S_T - m where m = min(S_t) over t ∈ [0, T]
│   │   ├─ Strike: Dynamically set to minimum observed price
│   │   ├─ Exercise: Buy at min, sell at terminal price
│   │   └─ Value: Highest (always positive if S_T > m)
│   ├─ Floating Strike Put:
│   │   ├─ Payoff: M - S_T where M = max(S_t) over t ∈ [0, T]
│   │   ├─ Strike: Dynamically set to maximum observed price
│   │   ├─ Exercise: Sell at max, buy at terminal price
│   │   └─ Value: Highest (always positive if M > S_T)
│   └─ Properties:
│       ├─ Always Positive Payoff: No strike to overcome
│       ├─ Perfect Timing: Captures full range of movement
│       └─ Most Expensive: Maximum optionality
├─ Monte Carlo Pricing:
│   ├─ Path Generation:
│   │   ├─ Fine Time Steps: Daily or finer discretization
│   │   ├─ Euler Scheme: S_{t+1} = S_t exp((r - σ²/2)Δt + σ√Δt Z_t)
│   │   └─ Store All Prices: Need min/max over entire path
│   ├─ Extrema Tracking:
│   │   ├─ Maximum: M = max(S_0, S_1, ..., S_n)
│   │   ├─ Minimum: m = min(S_0, S_1, ..., S_n)
│   │   └─ Running Min/Max: Update at each time step
│   ├─ Payoff Calculation:
│   │   ├─ Fixed Call: max(M - K, 0)
│   │   ├─ Fixed Put: max(K - m, 0)
│   │   ├─ Floating Call: S_T - m (always ≥ 0)
│   │   └─ Floating Put: M - S_T (always ≥ 0)
│   └─ Pricing:
│       ├─ Discount: PV = e^(-rT) × Payoff
│       └─ Average: V = (1/N) Σ PV_i over N paths
├─ Closed-Form Solutions (Limited):
│   ├─ Continuous Monitoring: Available under GBM (Goldman et al.)
│   ├─ Floating Strike: Simpler formulas (no strike parameter)
│   ├─ Fixed Strike: More complex (involves strike position)
│   └─ Discrete Monitoring: No closed-form; use MC or trees
├─ Variance Reduction:
│   ├─ Antithetic Variates: Z and -Z paths give correlated M, m
│   ├─ Control Variate: Use Asian option (correlated but cheaper)
│   ├─ Stratified: Stratify on final price S_T
│   └─ Moment Matching: Force paths to have correct E[M], E[m]
├─ Greeks & Hedging:
│   ├─ Delta: Time-dependent (high early, decreases as M/m established)
│   ├─ Gamma: Positive but decreases over time
│   ├─ Vega: Very high (extreme prices sensitive to volatility)
│   ├─ Theta: Negative (time decay as monitoring period shortens)
│   └─ Hedging: Difficult (path-dependent; delta changes with M/m)
└─ Discrete vs Continuous Monitoring:
    ├─ Continuous: True lookback (all times checked)
    ├─ Discrete: Check at specific times (daily, weekly)
    ├─ Bias: Discrete < Continuous (misses intraperiod extremes)
    └─ Adjustment: Broadie-Glasserman correction for discrete
```

**Interaction:** Generate paths → Track running min/max → Compute payoff on extremes → Discount to present

## 5. Mini-Project
Price fixed and floating strike lookback options and compare to European:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# European call (benchmark)
def european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo lookback option pricing
def mc_lookback_call(S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False):
    """
    Price lookback call option.
    
    Parameters:
    - floating_strike: If True, floating strike (payoff = S_T - min);
                       If False, fixed strike (payoff = max - K)
    
    Returns:
    - price: Option value
    - std_error: Standard error
    - paths: Simulated price paths
    - maxima: Maximum prices per path
    - minima: Minimum prices per path
    """
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    # Generate paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(n_steps):
        Z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt 
                                              + sigma * np.sqrt(dt) * Z)
    
    # Track extrema
    maxima = np.max(paths, axis=1)
    minima = np.min(paths, axis=1)
    terminal = paths[:, -1]
    
    # Compute payoffs
    if floating_strike:
        # Floating strike call: S_T - min
        payoffs = terminal - minima
    else:
        # Fixed strike call: max(max - K, 0)
        payoffs = np.maximum(maxima - K, 0)
    
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error, paths, maxima, minima

def mc_lookback_put(S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False):
    """Price lookback put option."""
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(n_steps):
        Z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt 
                                              + sigma * np.sqrt(dt) * Z)
    
    maxima = np.max(paths, axis=1)
    minima = np.min(paths, axis=1)
    terminal = paths[:, -1]
    
    if floating_strike:
        # Floating strike put: max - S_T
        payoffs = maxima - terminal
    else:
        # Fixed strike put: max(K - min, 0)
        payoffs = np.maximum(K - minima, 0)
    
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error, paths, maxima, minima

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.30
n_steps = 252  # Daily monitoring

print("="*80)
print("LOOKBACK OPTIONS PRICING")
print("="*80)
print(f"S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%\n")

# European benchmark
euro_call = european_call(S0, K, T, r, sigma)
print(f"European Call: ${euro_call:.6f}")

# Lookback options
np.random.seed(42)
n_paths = 50000

# Fixed strike lookback call
fixed_call, fixed_call_err, paths_call, maxima_call, minima_call = mc_lookback_call(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False
)
print(f"\nFixed Strike Lookback Call:    ${fixed_call:.6f} ± ${fixed_call_err:.6f}")
print(f"  Premium over European: ${fixed_call - euro_call:.6f} ({(fixed_call/euro_call - 1)*100:.1f}%)")

# Floating strike lookback call
float_call, float_call_err, _, _, _ = mc_lookback_call(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=True
)
print(f"Floating Strike Lookback Call: ${float_call:.6f} ± ${float_call_err:.6f}")
print(f"  Premium over Fixed: ${float_call - fixed_call:.6f} ({(float_call/fixed_call - 1)*100:.1f}%)")

# Fixed strike lookback put
fixed_put, fixed_put_err, paths_put, maxima_put, minima_put = mc_lookback_put(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=False
)
print(f"\nFixed Strike Lookback Put:     ${fixed_put:.6f} ± ${fixed_put_err:.6f}")

# Floating strike lookback put
float_put, float_put_err, _, _, _ = mc_lookback_put(
    S0, K, T, r, sigma, n_paths, n_steps, floating_strike=True
)
print(f"Floating Strike Lookback Put:  ${float_put:.6f} ± ${float_put_err:.6f}")

# Statistics on extrema
print("\n" + "="*80)
print("EXTREMA STATISTICS")
print("="*80)
print(f"Average Maximum: ${np.mean(maxima_call):.2f}")
print(f"Average Minimum: ${np.mean(minima_call):.2f}")
print(f"Average Range:   ${np.mean(maxima_call - minima_call):.2f}")
print(f"Average Terminal: ${np.mean(paths_call[:, -1]):.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample paths with extrema
ax = axes[0, 0]
n_plot = 20
time_grid = np.linspace(0, T, n_steps + 1)

for i in range(n_plot):
    ax.plot(time_grid, paths_call[i, :], 'b-', alpha=0.4, linewidth=1)
    
    # Mark max and min
    max_idx = np.argmax(paths_call[i, :])
    min_idx = np.argmin(paths_call[i, :])
    ax.scatter(time_grid[max_idx], paths_call[i, max_idx], 
              color='green', s=50, zorder=5, alpha=0.8)
    ax.scatter(time_grid[min_idx], paths_call[i, min_idx],
              color='red', s=50, zorder=5, alpha=0.8)

ax.axhline(K, color='orange', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price S')
ax.set_title('Sample Paths (Green=Max, Red=Min)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution of maxima and minima
ax = axes[0, 1]
ax.hist(maxima_call, bins=50, alpha=0.6, color='green', edgecolor='black',
        density=True, label='Maximum')
ax.hist(minima_call, bins=50, alpha=0.6, color='red', edgecolor='black',
        density=True, label='Minimum')
ax.axvline(S0, color='blue', linestyle='--', linewidth=2, label=f'S₀=${S0}')
ax.axvline(np.mean(maxima_call), color='green', linestyle=':', linewidth=2,
           label=f'Mean Max=${np.mean(maxima_call):.0f}')
ax.axvline(np.mean(minima_call), color='red', linestyle=':', linewidth=2,
           label=f'Mean Min=${np.mean(minima_call):.0f}')
ax.set_xlabel('Price')
ax.set_ylabel('Density')
ax.set_title('Distribution of Extrema')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Payoff distributions
ax = axes[0, 2]
fixed_payoffs = np.maximum(maxima_call - K, 0)
float_payoffs = paths_call[:, -1] - minima_call

ax.hist(fixed_payoffs, bins=50, alpha=0.6, color='blue', edgecolor='black',
        label='Fixed Strike')
ax.hist(float_payoffs, bins=50, alpha=0.6, color='orange', edgecolor='black',
        label='Floating Strike')
ax.set_xlabel('Payoff at Maturity')
ax.set_ylabel('Frequency')
ax.set_title('Lookback Call Payoff Distributions')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Value vs volatility
sigmas = np.linspace(0.10, 0.60, 15)
euro_vals = []
fixed_vals = []
float_vals = []

for sig in sigmas:
    euro_vals.append(european_call(S0, K, T, r, sig))
    
    np.random.seed(42)
    fixed_price, _, _, _, _ = mc_lookback_call(S0, K, T, r, sig, 10000, n_steps, False)
    float_price, _, _, _, _ = mc_lookback_call(S0, K, T, r, sig, 10000, n_steps, True)
    
    fixed_vals.append(fixed_price)
    float_vals.append(float_price)

ax = axes[1, 0]
ax.plot(sigmas * 100, euro_vals, 'g-', linewidth=2, label='European')
ax.plot(sigmas * 100, fixed_vals, 'b-', linewidth=2, label='Fixed Strike Lookback')
ax.plot(sigmas * 100, float_vals, 'r-', linewidth=2, label='Floating Strike Lookback')
ax.set_xlabel('Volatility σ (%)')
ax.set_ylabel('Option Price ($)')
ax.set_title('Lookback Vega: Value vs Volatility')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Value vs spot price
spots = np.linspace(80, 120, 20)
euro_spot = []
fixed_spot = []
float_spot = []

for S in spots:
    euro_spot.append(european_call(S, K, T, r, sigma))
    
    np.random.seed(42)
    fixed_price, _, _, _, _ = mc_lookback_call(S, K, T, r, sigma, 10000, n_steps, False)
    float_price, _, _, _, _ = mc_lookback_call(S, K, T, r, sigma, 10000, n_steps, True)
    
    fixed_spot.append(fixed_price)
    float_spot.append(float_price)

ax = axes[1, 1]
ax.plot(spots, euro_spot, 'g-', linewidth=2, label='European')
ax.plot(spots, fixed_spot, 'b-', linewidth=2, label='Fixed Strike Lookback')
ax.plot(spots, float_spot, 'r-', linewidth=2, label='Floating Strike Lookback')
ax.axvline(K, color='orange', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Option Price ($)')
ax.set_title('Lookback Delta: Value vs Spot')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Comparison bar chart
option_types = ['European\nCall', 'Fixed\nLookback\nCall', 'Floating\nLookback\nCall']
prices = [euro_call, fixed_call, float_call]
colors = ['green', 'blue', 'red']

ax = axes[1, 2]
bars = ax.bar(range(len(option_types)), prices, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(option_types)))
ax.set_xticklabels(option_types)
ax.set_ylabel('Option Price ($)')
ax.set_title('Option Price Comparison')
ax.grid(True, axis='y', alpha=0.3)

for bar, price in zip(bars, prices):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${price:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('lookback_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Range analysis
ranges = maxima_call - minima_call
print("\n" + "="*80)
print("RANGE ANALYSIS (Max - Min)")
print("="*80)
print(f"Mean Range:   ${np.mean(ranges):.2f}")
print(f"Median Range: ${np.median(ranges):.2f}")
print(f"Std Dev:      ${np.std(ranges):.2f}")
print(f"Min Range:    ${np.min(ranges):.2f}")
print(f"Max Range:    ${np.max(ranges):.2f}")

# Hindsight perfection analysis
perfect_buy_low = minima_call
perfect_sell_high = maxima_call
hindsight_profit = perfect_sell_high - perfect_buy_low

print(f"\nHindsight Perfect Trading:")
print(f"  Average Profit: ${np.mean(hindsight_profit):.2f}")
print(f"  This equals Floating Strike Call payoff")
```

## 6. Challenge Round

**Q1:** Why is floating strike lookback always worth more than fixed strike lookback?  
**A1:** Floating strike payoff = S_T - m (call) or M - S_T (put) → always captures full range. Fixed strike = max(M - K, 0) → limited by strike; if M < K, payoff = 0. Floating has no strike barrier → higher optionality → higher value.

**Q2:** Prove lookback call > European call. What's intuition via hindsight?  
**A2:** Fixed lookback: max(M - K, 0) ≥ max(S_T - K, 0) since M ≥ S_T always. Strict inequality unless S_T = M (terminal price is maximum). Hindsight: Lookback exercises at best price; European at terminal only. No-regret premium built into price.

**Q3:** Why does lookback option have extremely high Vega (volatility sensitivity)?  
**A3:** Higher volatility → larger price swings → higher max/min spread → larger payoffs. For floating strike call: Payoff = S_T - m = range. Var(Range) increases with σ² → lookback value highly sensitive to vol. Vega_lookback >> Vega_European.

**Q4:** Discrete monitoring: How does sampling frequency affect lookback value?  
**A4:** Higher frequency (more observations) → closer to continuous → finds true extremes → higher value. Daily vs weekly: Daily captures more extremes → higher max, lower min → larger payoff → higher price. Limit: Continuous monitoring gives upper bound.

**Q5:** Delta of lookback option: How does it change over time?  
**A5:** Early: High delta (max/min not established; sensitive to S movements). Late: Lower delta (extremes already observed; locked in). If current S near established max/min, delta changes (max affects call, min affects put). Time-dependent and path-dependent delta.

**Q6:** Partial lookback: Payoff = max(M - K, 0) but M = max over [t*, T] for t* < T. How to price?  
**A6:** Lookback starts at t*; ignore prices before. MC: Generate full path, compute max only from t* onward. Cheaper than full lookback (less monitoring → smaller max). Used for cheapening premium while keeping lookback feature.

**Q7:** Lookback on portfolio: Max of weighted basket max(w₁M₁ + w₂M₂ - K, 0). Challenge?  
**A7:** Component maximums occur at different times: M₁ at t₁, M₂ at t₂. Weighted sum max(w₁S₁ + w₂S₂) ≠ w₁M₁ + w₂M₂. Must track portfolio value at each time step, find its maximum. More complex; correlation matters less than individual extremes.

**Q8:** Hedging lookback option: Why is it nearly impossible near extrema?  
**A8:** Near established max (for call): Small move up → new max → delta jumps. Small move down → max unchanged → delta near zero. Gamma infinite at max crossing point. Hedging requires continuous rebalancing with infinite frequency → impossible in practice. Transaction costs prohibitive.

## 7. Key References

**Primary Sources:**
- Goldman, M.B., Sosin, H.B., Gatto, M.A. "Path Dependent Options" (1979) - Original lookback pricing
- [Lookback Option Wikipedia](https://en.wikipedia.org/wiki/Lookback_option) - Overview and types
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 27: Lookback Options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Lookback simulation (pp. 392-405)
- Conze, A. & Viswanathan "Path Dependent Options: The Case of Lookback Options" (1991) - Closed-form formulas

**Thinking Steps:**
1. Generate Monte Carlo paths with fine time discretization
2. Track running maximum and minimum along each path
3. Compute payoff: Fixed (max/min vs strike K) or Floating (S_T vs min/max)
4. Discount expected payoff; note always positive for floating strike
5. Higher variance than European due to extreme value dependence
6. Floating strike always more valuable (captures full range, no strike barrier)
