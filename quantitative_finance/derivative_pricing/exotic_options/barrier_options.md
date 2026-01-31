# Barrier Options

## 1. Concept Skeleton
**Definition:** Path-dependent options that activate (knock-in) or extinguish (knock-out) when asset price crosses predetermined barrier  
**Purpose:** Cheaper than vanilla options; custom payoff structures; reduce premium cost for hedging  
**Prerequisites:** Path-dependent payoffs, monitoring frequency, Monte Carlo simulation, rebates

## 2. Comparative Framing
| Feature | Knock-Out | Knock-In | European Vanilla | Double Barrier |
|---------|-----------|----------|------------------|----------------|
| **Activation** | Active until barrier hit | Inactive until barrier hit | Always active | Two barriers |
| **Payoff** | 0 if knocked out | Standard if knocked in | Standard always | Complex conditions |
| **Value** | Cheaper than vanilla | Cheaper than vanilla | Highest | Cheapest |
| **Hedging** | Gamma spikes near barrier | Discontinuous delta | Smooth Greeks | Very complex |
| **Monitoring** | Continuous/discrete | Continuous/discrete | Maturity only | Continuous/discrete |

## 3. Examples + Counterexamples

**Simple Example:**  
Up-and-out call: S₀=$100, K=$100, Barrier=$120; if S never hits $120 → payoff=max(S_T - 100, 0); if hits → payoff=0

**Failure Case:**  
Continuous vs discrete monitoring: Discrete misses intraday breaches → overvalues knock-out (should be cheaper)

**Edge Case:**  
Barrier far from spot (B=$200, S₀=$100): Knock-out ≈ Vanilla (low breach probability); Knock-in ≈ 0

## 4. Layer Breakdown
```
Barrier Option Classification & Pricing:
├─ Barrier Types:
│   ├─ Up-and-Out (UO):
│   │   ├─ Knock-Out Condition: S_t ≥ B for any t ∈ [0, T]
│   │   ├─ Payoff: max(S_T - K, 0) if S_t < B for all t; else 0
│   │   ├─ Use Case: Cap upside exposure; reduce premium
│   │   └─ Value: UO < Vanilla (barrier reduces optionality)
│   ├─ Up-and-In (UI):
│   │   ├─ Knock-In Condition: S_t ≥ B for some t ∈ [0, T]
│   │   ├─ Payoff: max(S_T - K, 0) if barrier touched; else 0
│   │   ├─ Parity: UO + UI = Vanilla (one must pay off)
│   │   └─ Value: UI < Vanilla (requires barrier breach)
│   ├─ Down-and-Out (DO):
│   │   ├─ Knock-Out Condition: S_t ≤ B for any t
│   │   ├─ Payoff: max(S_T - K, 0) if S_t > B for all t; else 0
│   │   ├─ Use Case: Avoid downside scenarios; cheaper puts
│   │   └─ Value: DO < Vanilla
│   ├─ Down-and-In (DI):
│   │   ├─ Knock-In Condition: S_t ≤ B for some t
│   │   ├─ Payoff: max(S_T - K, 0) if barrier touched; else 0
│   │   ├─ Parity: DO + DI = Vanilla
│   │   └─ Value: DI < Vanilla
│   └─ Double Barrier:
│       ├─ Two Barriers: B_lower < S₀ < B_upper
│       ├─ Knock-Out: If S hits either barrier → extinguished
│       └─ Payoff: Standard if stays within corridor
├─ Rebates (Optional):
│   ├─ Knock-Out Rebate: Cash payment R if barrier breached
│   ├─ Knock-In Rebate: Cash if barrier NOT breached
│   ├─ Payment Time: At breach or at maturity
│   └─ Enhances Value: UO with rebate > UO without
├─ Monte Carlo Pricing:
│   ├─ Path Generation: Fine time discretization (daily/hourly)
│   ├─ Barrier Monitoring:
│   │   ├─ Discrete: Check prices at S_{t1}, S_{t2}, ..., S_{tn}
│   │   ├─ Continuous Approximation: Brownian bridge between steps
│   │   └─ Bias: Discrete monitoring overvalues knock-out (misses breaches)
│   ├─ Payoff Logic:
│   │   ├─ Knock-Out: If max(S_path) ≥ B_up or min(S_path) ≤ B_down → Payoff = R (rebate)
│   │   ├─ Knock-In: If barrier touched → Payoff = max(S_T - K, 0); else 0
│   │   └─ Double: If S stays in [B_low, B_high] → Payoff standard
│   └─ Variance: High (many paths have zero payoff → discrete distribution)
├─ Analytical Solutions (Limited):
│   ├─ Continuous Monitoring: Closed-form under GBM (reflection principle)
│   ├─ Conditions: Single barrier, constant vol, no dividends
│   ├─ Formula: Involves reflected Brownian motion probabilities
│   └─ Complex: Double barriers require infinite series
├─ Greeks & Hedging Challenges:
│   ├─ Delta: Discontinuous near barrier (jump when breached)
│   ├─ Gamma: Extremely high near barrier (delta changes rapidly)
│   ├─ Vanna/Volga: Large cross-Greeks (∂²V/∂S∂σ)
│   ├─ Hedging: Difficult near barrier; frequent rebalancing required
│   └─ Vega: Higher for options near barrier (uncertainty in breach timing)
└─ Monitoring Frequency Effect:
    ├─ Continuous: True barrier (all times checked)
    ├─ Daily: Check once per day (misses intraday breaches)
    ├─ Pricing Bias: Discrete < Continuous for knock-out (breaches missed → survives more)
    └─ Adjustment: Broadie-Glasserman-Kou correction for discrete monitoring
```

**Interaction:** Generate paths with fine discretization → Check barrier breach at each step → Apply payoff logic → Discount to present

## 5. Mini-Project
Price barrier options and visualize barrier breach dynamics:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# European call (benchmark)
def european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo barrier option pricing
def mc_barrier_call(S0, K, B, T, r, sigma, n_paths, n_steps, barrier_type='up-and-out', rebate=0):
    """
    barrier_type: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    """
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(n_steps):
        Z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Check barrier breach
    if 'up' in barrier_type:
        breached = np.max(paths, axis=1) >= B
    else:  # down
        breached = np.min(paths, axis=1) <= B
    
    # Compute payoffs
    terminal_payoffs = np.maximum(paths[:, -1] - K, 0)
    
    if 'out' in barrier_type:
        payoffs = np.where(breached, rebate, terminal_payoffs)
    else:  # knock-in
        payoffs = np.where(breached, terminal_payoffs, rebate)
    
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error, paths, breached

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.25
B_up = 120.0
B_down = 80.0
n_steps = 252  # Daily monitoring

print("="*80)
print("BARRIER OPTIONS PRICING")
print("="*80)
print(f"S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%")
print(f"Barriers: Up=${B_up}, Down=${B_down}, Steps={n_steps}\n")

# European benchmark
euro_call = european_call(S0, K, T, r, sigma)
print(f"European Call: ${euro_call:.6f}")

# Barrier options
np.random.seed(42)
n_paths = 50000

# Up-and-out
uo_price, uo_error, uo_paths, uo_breached = mc_barrier_call(
    S0, K, B_up, T, r, sigma, n_paths, n_steps, 'up-and-out'
)
print(f"\nUp-and-Out Call (B=${B_up}):    ${uo_price:.6f} ± ${uo_error:.6f}")
print(f"  Breach Rate: {np.sum(uo_breached)/n_paths*100:.2f}%")

# Up-and-in
ui_price, ui_error, _, ui_breached = mc_barrier_call(
    S0, K, B_up, T, r, sigma, n_paths, n_steps, 'up-and-in'
)
print(f"Up-and-In Call (B=${B_up}):     ${ui_price:.6f} ± ${ui_error:.6f}")
print(f"  Breach Rate: {np.sum(ui_breached)/n_paths*100:.2f}%")

# Verify parity: UO + UI = European
print(f"  Parity Check: UO + UI = ${uo_price + ui_price:.6f} vs Euro ${euro_call:.6f}")

# Down-and-out
do_price, do_error, do_paths, do_breached = mc_barrier_call(
    S0, K, B_down, T, r, sigma, n_paths, n_steps, 'down-and-out'
)
print(f"\nDown-and-Out Call (B=${B_down}):  ${do_price:.6f} ± ${do_error:.6f}")
print(f"  Breach Rate: {np.sum(do_breached)/n_paths*100:.2f}%")

# Down-and-in
di_price, di_error, _, di_breached = mc_barrier_call(
    S0, K, B_down, T, r, sigma, n_paths, n_steps, 'down-and-in'
)
print(f"Down-and-In Call (B=${B_down}):   ${di_price:.6f} ± ${di_error:.6f}")
print(f"  Parity Check: DO + DI = ${do_price + di_price:.6f} vs Euro ${euro_call:.6f}")

# With rebate
uo_rebate_price, _, _, _ = mc_barrier_call(
    S0, K, B_up, T, r, sigma, n_paths, n_steps, 'up-and-out', rebate=5.0
)
print(f"\nUp-and-Out with $5 Rebate: ${uo_rebate_price:.6f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample paths (up-and-out)
ax = axes[0, 0]
n_plot = 30
time_grid = np.linspace(0, T, n_steps + 1)
for i in range(n_plot):
    color = 'red' if uo_breached[i] else 'blue'
    alpha = 0.4 if uo_breached[i] else 0.2
    ax.plot(time_grid, uo_paths[i, :], color=color, alpha=alpha, linewidth=0.8)

ax.axhline(B_up, color='orange', linestyle='--', linewidth=2, label=f'Barrier ${B_up}')
ax.axhline(K, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Strike ${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price S')
ax.set_title('Up-and-Out: Red=Knocked Out, Blue=Survives')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Sample paths (down-and-out)
ax = axes[0, 1]
for i in range(n_plot):
    color = 'red' if do_breached[i] else 'blue'
    alpha = 0.4 if do_breached[i] else 0.2
    ax.plot(time_grid, do_paths[i, :], color=color, alpha=alpha, linewidth=0.8)

ax.axhline(B_down, color='orange', linestyle='--', linewidth=2, label=f'Barrier ${B_down}')
ax.axhline(K, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price S')
ax.set_title('Down-and-Out: Red=Knocked Out, Blue=Survives')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Barrier breach times
breach_times_uo = []
for i in range(n_paths):
    if uo_breached[i]:
        breach_time = np.argmax(uo_paths[i, :] >= B_up) * T / n_steps
        breach_times_uo.append(breach_time)

ax = axes[0, 2]
if breach_times_uo:
    ax.hist(breach_times_uo, bins=30, edgecolor='black', alpha=0.7, color='red')
    ax.set_xlabel('Breach Time (years)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Up-and-Out Breach Times ({len(breach_times_uo)} breaches)')
    ax.grid(True, alpha=0.3)

# Plot 4: Value vs barrier level
barriers_up = np.linspace(110, 150, 15)
prices_uo = []
prices_ui = []

for B in barriers_up:
    np.random.seed(42)
    p_uo, _, _, _ = mc_barrier_call(S0, K, B, T, r, sigma, 10000, n_steps, 'up-and-out')
    p_ui, _, _, _ = mc_barrier_call(S0, K, B, T, r, sigma, 10000, n_steps, 'up-and-in')
    prices_uo.append(p_uo)
    prices_ui.append(p_ui)

ax = axes[1, 0]
ax.plot(barriers_up, prices_uo, 'ro-', linewidth=2, label='Up-and-Out')
ax.plot(barriers_up, prices_ui, 'bo-', linewidth=2, label='Up-and-In')
ax.axhline(euro_call, color='green', linestyle='--', linewidth=2, label='European')
ax.set_xlabel('Barrier Level B')
ax.set_ylabel('Option Price ($)')
ax.set_title('Barrier Option Value vs Barrier Level')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Value vs spot price
spots = np.linspace(80, 120, 20)
prices_uo_spot = []
prices_euro_spot = []

for S in spots:
    np.random.seed(42)
    p_uo, _, _, _ = mc_barrier_call(S, K, B_up, T, r, sigma, 10000, n_steps, 'up-and-out')
    prices_uo_spot.append(p_uo)
    prices_euro_spot.append(european_call(S, K, T, r, sigma))

ax = axes[1, 1]
ax.plot(spots, prices_euro_spot, 'g-', linewidth=2, label='European')
ax.plot(spots, prices_uo_spot, 'r-', linewidth=2, label='Up-and-Out')
ax.axvline(B_up, color='orange', linestyle='--', linewidth=2, alpha=0.5, label=f'Barrier ${B_up}')
ax.axvline(K, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Option Price ($)')
ax.set_title('Option Value vs Spot (Barrier Effects)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Comparison of all barrier types
barrier_types = ['European', 'Up-Out', 'Up-In', 'Down-Out', 'Down-In']
prices = [euro_call, uo_price, ui_price, do_price, di_price]
colors = ['green', 'red', 'blue', 'orange', 'purple']

ax = axes[1, 2]
bars = ax.bar(range(len(barrier_types)), prices, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(barrier_types)))
ax.set_xticklabels(barrier_types, rotation=15, ha='right')
ax.set_ylabel('Option Price ($)')
ax.set_title('Barrier Option Price Comparison')
ax.grid(True, axis='y', alpha=0.3)

for bar, price in zip(bars, prices):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${price:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('barrier_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 6. Challenge Round

**Q1:** Prove knock-out + knock-in = vanilla (parity). What no-arbitrage argument supports this?  
**A1:** At maturity, either barrier touched (knock-in active, knock-out dead) or not (knock-out active, knock-in dead). Exactly one pays vanilla payoff. If UO + UI ≠ Euro, arbitrage: Buy cheap side, sell expensive side, lock risk-free profit.

**Q2:** Why does gamma explode near barrier? Implications for hedging?  
**A2:** Near barrier, small price move determines survival (in/out). Delta jumps from ≈ vanilla to 0 when barrier crossed → ∂Δ/∂S = Γ extremely large. Hedging impossible: Rebalance frequency → ∞ near barrier; transaction costs prohibitive.

**Q3:** Discrete vs continuous monitoring: Derive adjustment factor for knock-out call.  
**A3:** Discrete monitoring misses intraday breaches → survives more often → higher value. Broadie-Glasserman-Kou adjustment: Shift barrier B → B × exp(β σ√(dt)) where β ≈ 0.5826. Makes discrete price closer to continuous.

**Q4:** Double barrier option: What happens when barriers very tight (B_up - B_down → 0)?  
**A4:** Tight corridor → high probability of knock-out → value → 0 as corridor shrinks. Extreme: B_up = B_down = S₀ → instant knock-out → value = rebate only (if any). Used for range trading strategies.

**Q5:** Barrier breach probability: Derive for up-and-out barrier B > S₀ under GBM.  
**A5:** First passage time problem. Probability S_t hits B before T: P = N(d+) + (S₀/B)^(2μ/σ²) N(d-) where μ = r - σ²/2, d± = [ln(B²/S₀²) ± μT] / (σ√T). Reflection principle in Brownian motion theory.

**Q6:** Rebate timing: Payment at breach vs at maturity. Which is more valuable?  
**A6:** Rebate at breach > rebate at maturity (time value of money). Earlier payment → higher present value. Pricing: For breach at time τ, discount as e^(-rτ) R (random τ) vs e^(-rT) R (fixed T). Expected PV higher for immediate rebate.

**Q7:** Barrier option Greeks near barrier: Compute delta for S → B (from below) for up-and-out call.  
**A7:** As S → B⁻: UO call → 0 (about to knock out). Delta = ∂V/∂S → 0 rapidly. Just below barrier: Large negative delta (price drops to zero with small S increase). Gamma extremely negative (delta changes from positive to zero).

**Q8:** Reverse (inside) barrier: Knock-out if S stays INSIDE corridor [B_low, B_high]. When useful?  
**A8:** Opposite of double barrier: Extinguishes if stays in range (no excitement). Rare; used for volatility betting. High vol → likely to breach boundaries → survives → payoff. Low vol → trapped inside → knocks out → zero. Exotic structure for vol traders.

## 7. Key References

**Primary Sources:**
- Rubinstein, M. & Reiner, E. "Breaking Down the Barriers" (1991) - Closed-form barrier formulas
- [Barrier Option Wikipedia](https://en.wikipedia.org/wiki/Barrier_option) - Classification and examples
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 27: Barrier Options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Barrier monitoring (pp. 365-392)
- Broadie, M., Glasserman, P., Kou, S. "Connecting Discrete and Continuous Path-Dependent Options" (1999) - Monitoring bias correction

**Thinking Steps:**
1. Generate paths with fine time discretization (daily or finer)
2. Track maximum and minimum prices along each path
3. Check barrier breach: Compare max/min vs barrier level
4. Apply knock-out logic: Set payoff = rebate if breached
5. Apply knock-in logic: Set payoff = vanilla if breached, else rebate
6. Discount expected payoff; higher variance due to discrete payoff distribution
