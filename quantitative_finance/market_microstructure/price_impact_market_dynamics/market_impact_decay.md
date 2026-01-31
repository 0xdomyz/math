# Market Impact Decay

## 1. Concept Skeleton
**Definition:** Time-dependent relaxation of price impact; permanent component settles while temporary reverts; decay curves and half-lives  
**Purpose:** Understand impact persistence; optimize execution timing; forecast price stabilization; risk management  
**Prerequisites:** Temporary/permanent impact, mean reversion, time scales, decay functions

## 2. Comparative Framing
| Decay Type | Time Scale | Permanence | Driver | Recovery |
|-----------|-----------|-----------|--------|----------|
| **Immediate** | Microseconds | None | Bid-ask bounce | Instant |
| **Fast Decay** | Milliseconds | Temporary | Inventory | Minutes |
| **Medium Decay** | Seconds | Mixed | Learning + inventory | Hours |
| **Slow Decay** | Hours | Permanent | Information | Settles |
| **No Decay** | Days+ | Full | Regime change | Never |

## 3. Examples + Counterexamples

**Fast Decay (Temporary):**  
Illiquid stock: $0.50 spread → large market order → spreads widen to $1.00 → within 30 seconds, other traders step in → spreads revert to $0.60 (temporary bounced back)

**Slow Decay (Permanent):**  
Insider buys 100K shares → price jumps $2 → stays elevated for weeks → new information gradually incorporated → $2 move is permanent (no reversion)

**Mixed Decay:**  
Positive earnings → stock pops $3 immediately → next day up another $1 → reverses $0.50 by next week → $2.50 permanent gain, $0.50 temporary overshoot

**Decay Disruption:**  
Normal decay: Price moves $0.10, reverts to $0.05 permanent within hours → but market crash occurs → $0.05 permanent becomes $0.02 (new regime) → decay interrupted by regime change

## 4. Layer Breakdown
```
Market Impact Decay Framework:
├─ Decay Types and Patterns:
│   ├─ Exponential Decay:
│   │   - Formula: impact(t) = impact(0) × e^(-t/τ)
│   │   - τ (tau): Time constant (half-life)
│   │   - Characteristic: Monotonic smooth decay
│   │   - Property: Never reaches zero (theoretical)
│   │   - Half-life: Time for 50% recovery
│   │   - Typical: τ = 10-100 milliseconds (depends on liquidity)
│   ├─ Power-Law Decay:
│   │   - Formula: impact(t) ∝ t^(-α)
│   │   - α: Exponent (0.5-1.5 typical)
│   │   - Characteristic: Slower tail (fatter)
│   │   - Property: Decays slower than exponential initially
│   │   - Empirical: Often observed in real markets
│   │   - Interpretation: Heavy-tailed processes
│   ├─ Bimodal Decay:
│   │   - Formula: impact = A×e^(-t/τ_1) + B×e^(-t/τ_2)
│   │   - Components: Fast (temporary) + slow (permanent)
│   │   - Fast: Few milliseconds (inventory reversion)
│   │   - Slow: Hours/days (information learning)
│   │   - Separation: Two distinct time scales
│   │   - Interpretation: Matches theory (temp + perm)
│   ├─ Regime-Dependent Decay:
│   │   - Normal: Fast decay (liquid market)
│   │   - Stress: Slow decay (illiquid)
│   │   - Detection: Monitor liquidity indicators
│   │   - Risk: Decay changes unpredictably
│   └─ Non-Monotonic Decay:
│       - Initially: Impact increases briefly
│       - Then: Decays monotonically
│       - Cause: Cascade of stop-losses triggered
│       - Observation: Rare but observed in stress
│
├─ Time Scales:
│   ├─ Microsecond Scale (μs):
│   │   - Duration: 1-100 microseconds
│   │   - Phenomena: Tick data noise, bid-ask bounce
│   │   - Driver: Mechanical order book dynamics
│   │   - Decay: Instantaneous to millisecond
│   │   - Relevance: HFT algorithms
│   ├─ Millisecond Scale (ms):
│   │   - Duration: 1-1000 milliseconds
│   │   - Phenomena: Temporary impact, inventory reversion
│   │   - Driver: MM rebalancing, news dissemination
│   │   - Decay: Exponential with τ ≈ 10-100ms
│   │   - Relevance: Most markets, most traders
│   ├─ Second Scale (s):
│   │   - Duration: 1-60 seconds
│   │   - Phenomena: Order clustering effects
│   │   - Driver: Cascade of related trades
│   │   - Decay: Slower (power-law or mixed)
│   │   - Relevance: Algorithm execution period
│   ├─ Minute Scale (min):
│   │   - Duration: 1 minute - 1 hour
│   │   - Phenomena: Persistent temporary impact
│   │   - Driver: MM learning, algorithmic herding
│   │   - Decay: Slow decay, may not fully recover
│   │   - Relevance: Retail traders, algo completion
│   ├─ Hour/Day Scale (h/d):
│   │   - Duration: 1 hour - 1 day
│   │   - Phenomena: Permanent component isolation
│   │   - Driver: Information revelation, news processing
│   │   - Decay: Very slow or none (becomes permanent)
│   │   - Relevance: Overnight gaps, fundamental news
│   └─ Longer Horizons:
│       - Weeks: Regime switches, fundamental re-evaluation
│       - Months: Mean reversion patterns (different market)
│       - Years: Long-term impact (fully permanent)
│
├─ Empirical Findings:
│   ├─ Stock Markets:
│   │   - Millisecond: 80% recovery in 1 second
│   │   - 10 seconds: 90% recovery
│   │   - 1 minute: 95% recovery (mostly temporary gone)
│   │   - 1 hour: 98% recovery (permanent ~2% remains)
│   │   - Daily: Decay complete (permanent identified)
│   ├─ Market Impact Components:
│   │   - Immediate (bid-ask): Recovers in μs to ms
│   │   - Temporary (inventory): Recovers in ms to seconds
│   │   - Permanent (information): Doesn't recover
│   │   - Half-life estimates:
│   │     - Bid-ask bounce: <1 ms
│   │     - Inventory: 10-100 ms
│   │     - Information: Hours to days
│   ├─ Factors Affecting Decay Rate:
│   │   - Liquidity:
│   │     - High liquidity: Fast decay (short τ)
│   │     - Low liquidity: Slow decay (long τ)
│   │     - Example: Apple 10ms vs microcap 1 second
│   │   - Volatility:
│   │     - High volatility: Slower decay (longer τ)
│   │     - Low volatility: Faster decay (shorter τ)
│   │     - Reason: MM uncertainty increases, quotes widen
│   │   - Order Size:
│   │     - Small orders: Faster decay (deep book)
│   │     - Large orders: Slower decay (consume book)
│   │     - Nonlinear: Decay rate depends on participation
│   │   - Information Content:
│   │     - Informed: Slow decay (permanent)
│   │     - Uninformed: Fast decay (temporary)
│   │     - Challenge: Can't distinguish until after
│   ├─ Asset Class Comparisons:
│   │   - Equities:
│   │     - Large cap: τ ≈ 10-50 ms (fast)
│   │     - Small cap: τ ≈ 50-200 ms (slower)
│   │   - Options:
│   │     - τ ≈ 50-100 ms (similar to equities)
│   │   - Futures:
│   │     - τ ≈ 100-500 ms (slower)
│   │   - FX:
│   │     - τ ≈ 100-1000 ms (very slow, decentralized)
│   ├─ Intraday Patterns:
│   │   - Open: Slower decay (uncertainty after overnight gap)
│   │   - Midday: Faster decay (high liquidity)
│   │   - Earnings: Variable (depends on news)
│   │   - Close: Moderate decay (portfolio adjustments)
│   └─ Seasonal/Cyclical:
│       - Bull market: Faster decay (high participation)
│       - Bear market: Slower decay (reduced participation)
│       - Crisis: Extremely slow (liquidity evaporation)
│
├─ Decay Function Estimation:
│   ├─ Hasbrouck Decomposition Method:
│   │   - Regression: Price on order flow with lags
│   │   - Coefficients: Impulse response function
│   │   - Decay: Trace cumulative response over time
│   │   - Advantage: Econometric rigor
│   │   - Limitation: Assumes linear system
│   ├─ Half-Life Regression:
│   │   - Method: Find time when impact = 50% initial
│   │   - Calculation: Regress price on time
│   │   - τ: Solve: e^(-t/τ) = 0.5 → t = τ × ln(2)
│   │   - Simplicity: Quick calculation
│   │   - Limitation: Assumes exponential (may not hold)
│   ├─ Rolling Window Method:
│   │   - Window: Fixed time after each trade
│   │   - Metric: Percentage price reversion in window
│   │   - Aggregation: Average across many trades
│   │   - Advantage: Non-parametric (no model assumed)
│   │   - Data intensive: Needs large sample
│   ├─ High-Frequency Data Approach:
│   │   - Data: Nanosecond timestamps (FPGA-recorded)
│   │   - Advantage: Fine temporal resolution
│   │   - Cost: Requires specialized data providers
│   │   - Computation: Massive datasets to process
│   └─ Signal Processing Methods:
│       - Fourier transform: Separate frequency components
│       - Wavelet analysis: Time-frequency decomposition
│       - Kalman filter: Optimal state estimation
│       - Advantage: Sophisticated but complex
│
├─ Practical Applications:
│   ├─ Execution Algorithm Design:
│   │   - VWAP: Assumes even participation
│   │   - TWAP: Uniform slicing (ignores decay)
│   │   - Optimal: Account for decay (execute more when spreads tight)
│   │   - Insight: Can predict when decay completes
│   │   - Benefit: Better execution timing
│   ├─ Risk Management:
│   │   - Position risk: Decays over time
│   │   - Unexecuted portion: Faces market risk until filled
│   │   - Model: Use decay curve for risk forecast
│   │   - Decision: Whether to continue or abort
│   ├─ Price Prediction:
│   │   - Observed move: Temporary + permanent components
│   │   - Decay: Separate them over time
│   │   - Forecast: Where price will settle
│   │   - Application: Mean reversion strategies
│   ├─ Liquidity Provision:
│   │   - MM sets quotes accounting for decay
│   │   - If decay slow: Inventory risk high, spread wider
│   │   - If decay fast: Inventory risk low, spread tighter
│   │   - Decision: Quote width = f(decay rate)
│   ├─ Stress Testing:
│   │   - Normal: τ = 50ms (known)
│   │   - Stress: τ = 500ms (estimate)
│   │   - Scenario: If decay 10x slower, what impact on business?
│   │   - Preparation: Plan for regime change
│   └─ Regulatory Compliance:
│       - Best execution: Optimize for decay
│       - Timing: Execute when decay predicted favorable
│       - Documentation: Show decay-based optimization
│
├─ Advanced Decay Topics:
│   ├─ Multi-Asset Decay:
│   │   - Correlated assets: Decay together
│   │   - Lead-lag: One asset leads other (contagion)
│   │   - Cross-decay: Buy one asset affects other's decay
│   │   - Example: S&P 500 sell → all components decay together
│   ├─ Information Decay vs Inventory Decay:
│   │   - Information: Never decays (permanent)
│   │   - Inventory: Fast decay (temporary)
│   │   - Separation: Challenging empirically
│   │   - Method: Use information content measures (PIN, VPIN)
│   ├─ Regime-Switching Decay:
│   │   - Model: Markov chain with different τ per regime
│   │   - Regime 1: Normal (τ = 50ms)
│   │   - Regime 2: Stress (τ = 500ms)
│   │   - Transition: Triggered by liquidity indicators
│   │   - Benefit: Adaptive forecasting
│   ├─ Nonlinear Decay:
│   │   - Assumption: Exponential (linear in log space)
│   │   - Reality: May be nonlinear
│   │   - Model: Power-law or other
│   │   - Challenge: Harder to estimate, forecast
│   └─ Spillover Effects:
│       - One market impacts another (FX → equities)
│       - Decay propagates across markets
│       - Timing: Spillover occurs with lag
│       - Forecasting: More complex, multi-dimensional
│
└─ Challenges and Open Questions:
    ├─ Stationarity:
    │   - Is τ constant? (Evidence: No)
    │   - Changes with market conditions
    │   - Non-stationarity complicates prediction
    │   - Solution: Adaptive/regime-switching models
    ├─ Measurement Precision:
    │   - How fast can we measure? (Data quality)
    │   - Tick data has discreteness issues
    │   - High-frequency data expensive/exclusive
    │   - Trade-off: Cost vs accuracy
    ├─ Causality:
    │   - What causes decay? (Multiple mechanisms)
    │   - Inventory rebalancing? Information learning? Both?
    │   - Difficult to decompose post-hoc
    │   - Implication: Can't predict mechanism-specific
    ├─ Crisis Behavior:
    │   - Does decay continue? (Empirically: changes)
    │   - 2008 crisis: Decay extremely slow (or stops)
    │   - COVID crash: Similar observation
    │   - Unpredictability: Crisis models fail
    └─ Policy Implications:
        - Circuit breakers: Halt if > X% move
        - Outcome: Pauses decay (artificial)
        - Effect: Prevents cascades but postpones discovery
        - Trade-off: Stability vs efficiency (debated)
```

**Interaction:** Order executed → immediate impact from spread → temporary component decays exponentially → permanent component remains → final price reflects information + permanent costs

## 5. Mini-Project
Estimate and forecast market impact decay:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(42)

class ImpactDecayEstimator:
    def __init__(self):
        self.decay_curves = []
        
    def simulate_decay_process(self, initial_impact=0.05, decay_type='exponential', duration=100):
        """Simulate market impact decay"""
        time = np.arange(duration)
        
        if decay_type == 'exponential':
            tau = 10  # Millisecond half-life
            impact = initial_impact * np.exp(-time / tau)
        elif decay_type == 'power_law':
            alpha = 0.5
            impact = initial_impact / (1 + (time / 10) ** alpha)
        elif decay_type == 'bimodal':
            # Fast (temp) + slow (perm)
            tau_fast = 5
            tau_slow = 50
            impact_fast = 0.03 * np.exp(-time / tau_fast)
            impact_slow = 0.02 * np.exp(-time / tau_slow)
            impact = impact_fast + impact_slow
        else:
            impact = initial_impact * np.ones(duration)
        
        return time, impact
    
    def estimate_exponential_decay(self, time, impact):
        """Fit exponential decay model"""
        def exp_model(t, impact0, tau):
            return impact0 * np.exp(-t / tau)
        
        try:
            popt, _ = curve_fit(exp_model, time, impact, p0=[impact[0], 10], maxfev=5000)
            impact_fit = exp_model(time, *popt)
            rmse = np.sqrt(np.mean((impact - impact_fit) ** 2))
            
            impact0, tau = popt
            half_life = tau * np.log(2)
            
            return {
                'impact0': impact0,
                'tau': tau,
                'half_life': half_life,
                'rmse': rmse,
                'fit': impact_fit
            }
        except:
            return None
    
    def estimate_power_law_decay(self, time, impact, exclude_zero=True):
        """Fit power-law decay model"""
        # Avoid log(0)
        if exclude_zero:
            valid_idx = impact > 0.0001
            time_valid = time[valid_idx]
            impact_valid = impact[valid_idx]
        else:
            time_valid = time
            impact_valid = impact
        
        def power_model(t, impact0, alpha):
            return impact0 / (1 + (t / 10) ** alpha)
        
        try:
            popt, _ = curve_fit(power_model, time_valid, impact_valid, p0=[impact[0], 0.5], maxfev=5000)
            impact_fit = power_model(time, *popt)
            rmse = np.sqrt(np.mean((impact - impact_fit) ** 2))
            
            return {
                'impact0': popt[0],
                'alpha': popt[1],
                'rmse': rmse,
                'fit': impact_fit
            }
        except:
            return None

# Scenario 1: Decay type comparison
print("Scenario 1: Impact Decay Types")
print("=" * 80)

estimator = ImpactDecayEstimator()

decay_types = ['exponential', 'power_law', 'bimodal']
for decay_type in decay_types:
    time, impact = estimator.simulate_decay_process(initial_impact=0.05, decay_type=decay_type, duration=100)
    
    # Calculate half-life
    half_idx = np.argmin(np.abs(impact - 0.025))
    half_life = time[half_idx]
    
    print(f"Decay Type: {decay_type:>12}")
    print(f"  Half-life: {half_life:>6.1f} ms")
    print(f"  Remaining at 1 sec: {impact[-1]:>6.4f} ({impact[-1]/impact[0]*100:>5.1f}%)")
    print()

# Scenario 2: Impact persistence across time scales
print("Scenario 2: Impact Persistence by Time Window")
print("=" * 80)

time_windows = [10, 50, 100, 500, 1000]  # ms
initial_impact = 0.05

time_long, impact = estimator.simulate_decay_process(initial_impact=initial_impact, decay_type='exponential', duration=1000)

for window in time_windows:
    if window <= len(impact):
        recovery = 1 - (impact[window-1] / impact[0])
        print(f"Time window: {window:>6} ms | Recovered: {recovery*100:>6.1f}% | Remaining: {impact[window-1]:>7.4f}")

# Scenario 3: Decay rate estimation
print(f"\n\nScenario 3: Decay Rate Estimation")
print("=" * 80)

# Simulate with noise (realistic)
time, true_impact = estimator.simulate_decay_process(initial_impact=0.05, decay_type='exponential', duration=100)
noise = np.random.normal(0, 0.002, len(time))
observed_impact = true_impact + noise
observed_impact = np.maximum(observed_impact, 0)  # Can't go negative

# Estimate
result_exp = estimator.estimate_exponential_decay(time, observed_impact)
result_power = estimator.estimate_power_law_decay(time, observed_impact)

if result_exp:
    print(f"Exponential Model:")
    print(f"  Impact0:   {result_exp['impact0']:.5f}")
    print(f"  Tau (τ):   {result_exp['tau']:.2f} ms")
    print(f"  Half-life: {result_exp['half_life']:.2f} ms")
    print(f"  RMSE:      {result_exp['rmse']:.5f}")

if result_power:
    print(f"\nPower-Law Model:")
    print(f"  Impact0: {result_power['impact0']:.5f}")
    print(f"  Alpha:   {result_power['alpha']:.3f}")
    print(f"  RMSE:    {result_power['rmse']:.5f}")

# Scenario 4: Regime-dependent decay
print(f"\n\nScenario 4: Decay Rate by Market Regime")
print("=" * 80)

regimes = [
    {'name': 'Normal', 'tau': 10, 'liquidity': 'high'},
    {'name': 'Volatile', 'tau': 30, 'liquidity': 'medium'},
    {'name': 'Stressed', 'tau': 100, 'liquidity': 'low'},
    {'name': 'Crisis', 'tau': 500, 'liquidity': 'very low'},
]

for regime in regimes:
    time, impact = estimator.simulate_decay_process(initial_impact=0.05, decay_type='exponential', duration=500)
    
    # Adjust decay for regime
    impact = 0.05 * np.exp(-time / regime['tau'])
    
    # Time to 90% recovery
    recovery_idx = np.argmin(np.abs(impact - 0.005))
    recovery_time = time[recovery_idx]
    
    print(f"Regime: {regime['name']:>10} | τ = {regime['tau']:>4} ms | Liquidity: {regime['liquidity']:>10} | 90% Recovery: {recovery_time:>6.0f} ms")

# Scenario 5: Multi-asset decay correlation
print(f"\n\nScenario 5: Decay Correlation Across Assets")
print("=" * 80)

# Simulate correlated assets
correlation = 0.7
num_assets = 3

# Generate base decay
time = np.arange(0, 100)
base_decay = 0.05 * np.exp(-time / 10)

# Add correlated noise to each asset
decays = []
for asset in range(num_assets):
    decay = base_decay * (1 + 0.1 * np.random.randn(len(time)) * (1 - correlation ** 0.5))
    decay = np.maximum(decay, 0)
    decays.append(decay)

# Calculate correlation
corr_matrix = np.corrcoef(decays)

print(f"Average correlation between assets: {np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]):.3f}")
print(f"(Higher correlation → decay synchronized)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Decay types
time, impact_exp = estimator.simulate_decay_process(0.05, 'exponential', 100)
_, impact_power = estimator.simulate_decay_process(0.05, 'power_law', 100)
_, impact_bi = estimator.simulate_decay_process(0.05, 'bimodal', 100)

axes[0, 0].semilogy(time, impact_exp, linewidth=2, label='Exponential (τ=10ms)')
axes[0, 0].semilogy(time, impact_power, linewidth=2, label='Power-law (α=0.5)')
axes[0, 0].semilogy(time, impact_bi, linewidth=2, label='Bimodal (temp+perm)')
axes[0, 0].axhline(y=0.025, color='r', linestyle='--', alpha=0.5, label='Half-life')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Impact (log scale)')
axes[0, 0].set_title('Scenario 1: Decay Type Comparison')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Persistence by window
windows = np.arange(1, 101)
persistence = []

for window in windows:
    if window <= len(impact):
        pers = impact_exp[window-1] / impact_exp[0]
        persistence.append(pers * 100)

axes[0, 1].plot(windows, persistence, linewidth=2, color='blue')
axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% recovery')
axes[0, 1].axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='90% recovery')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Remaining Impact (%)')
axes[0, 1].set_title('Scenario 2: Impact Persistence')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Decay with noise
axes[1, 0].scatter(time, observed_impact, alpha=0.5, s=20, label='Observed')
if result_exp:
    axes[1, 0].plot(time, result_exp['fit'], linewidth=2, label='Exponential Fit')
if result_power:
    axes[1, 0].plot(time, result_power['fit'], linewidth=2, label='Power-law Fit')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Impact')
axes[1, 0].set_title('Scenario 3: Decay Estimation with Noise')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Regime comparison
regimes_names = [r['name'] for r in regimes]
recovery_times = []

for regime in regimes:
    time_regime, impact_regime = estimator.simulate_decay_process(0.05, 'exponential', 500)
    impact_regime = 0.05 * np.exp(-time_regime / regime['tau'])
    
    recovery_idx = np.argmin(np.abs(impact_regime - 0.005))
    recovery_times.append(time_regime[recovery_idx])

colors_regime = plt.cm.RdYlGn_r(np.linspace(0, 1, len(regimes)))
bars = axes[1, 1].bar(regimes_names, recovery_times, color=colors_regime, alpha=0.7)
axes[1, 1].set_ylabel('Time to 90% Recovery (ms)')
axes[1, 1].set_title('Scenario 4: Decay Speed by Regime')
axes[1, 1].grid(alpha=0.3, axis='y')

for bar, time_val in zip(bars, recovery_times):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.0f}ms', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
if result_exp:
    print(f"Exponential half-life: {result_exp['half_life']:.1f} ms")
    print(f"Time to 99% recovery: {result_exp['half_life'] * np.log(100):.1f} ms")
print(f"Regime difference: 50x slower decay in crisis vs normal")
print(f"Implication: Risk management requires regime-aware models")
```

## 6. Challenge Round
If temporary impact decays predictably (exponential with known τ), why don't traders consistently profit from mean reversion?

- **Information contamination**: Some decay is information (permanent), some is temporary → can't distinguish until after → meanreversion strategy can't separate → may be trading information, not liquidity
- **Stochastic timing**: Although τ is typical, actual recovery time varies → sometimes fast (few ms), sometimes slow (seconds) → uncertainty means can't time entry perfectly
- **Reversal risk**: True reversal happens, but opposite direction → market continues falling after temporary bounce → stop-losses triggered → strategy fails → competition fierce
- **Regime shifts**: Model fitted to normal conditions → crisis arrives → τ changes 10x → strategies underwater → non-stationarity breaks models
- **Costs exceed profits**: Mean reversion spread often <1 bps → transaction costs 1-2 bps → before trade → profits negative → only works for HFTs with zero costs

## 7. Key References
- [Bouchaud et al (2004) - Empirical Properties of Asset Returns](https://arxiv.org/abs/cond-mat/0406224)
- [Hasbrouck (1991) - Measuring Effects of Data Aggregation](https://www.jstor.org/stable/2328955)
- [Gatheral & Schied (2013) - Transient Linear Price Impact](https://arxiv.org/abs/1011.5882)
- [Lillo et al (2003) - Single Curve to Characterize Price Impact](https://arxiv.org/abs/cond-mat/0308191)

---
**Status:** Time-dependent price recovery | **Complements:** Temporary Impact, Mean Reversion, Liquidity Provision
