# Microstructure Noise

## 1. Concept Skeleton
**Definition:** High-frequency price distortions from market friction (bid-ask bounce, discretization, rounding); artifacts unrelated to fundamental value  
**Purpose:** Understand data quality issues; filter noise from true price; estimate volatility consistently; optimize sampling frequency  
**Prerequisites:** Bid-ask spread, tick size, price discretization, efficient prices, volatility estimation

## 2. Comparative Framing
| Noise Type | Time Scale | Source | Impact on Volatility | Mitigation |
|------------|-----------|--------|---------------------|------------|
| **Bid-Ask Bounce** | Trade-to-trade | Spread crossing | Inflates RV 2-10x | Subsampling, mid-quotes |
| **Discretization** | Tick-level | Minimum price increment | Bias in low-price stocks | Larger sampling intervals |
| **Rounding Errors** | Quote-level | Display precision | Negligible (large stocks) | Averaging |
| **Non-Synchronous Trading** | Seconds-Minutes | Stale quotes | Spurious correlations | Lead-lag adjustments |
| **Market Impact** | Milliseconds-Seconds | Order execution | Autocorrelation | TSRV, pre-averaging |

## 3. Examples + Counterexamples

**Bid-Ask Bounce:**  
Stock bid=$100.00, ask=$100.01 → Buy executes at $100.01 → Sell executes at $100.00 → 1bp "return" but no fundamental move → Artificial volatility

**True Price Movement:**  
Stock trades at $100.00 → News: Earnings beat → All quotes shift to $100.50 → True price discovery, not noise

**Tick Size Distortion:**  
Penny stock: $0.50 bid, $0.51 ask → 2% spread → Every trade creates 2% "return" → Dominates fundamental volatility → Noise ratio >90%

**Large-Cap Stability:**  
AAPL: $150.00 bid, $150.01 ask → 0.007% spread → Bid-ask bounce negligible relative to true volatility → Noise ratio <5%

## 4. Layer Breakdown
```
Microstructure Noise Framework:
├─ Types of Microstructure Noise:
│   ├─ Bid-Ask Bounce:
│   │   - Mechanism: Trades alternate between bid and ask
│   │   - Pattern: Negative first-order autocorrelation
│   │   - Magnitude: Half-spread squared (σ²_noise ≈ s²/4)
│   │   - Effect: Realized variance overstated 2-10x
│   │   - Detection: Plot returns ACF (autocorrelation function)
│   │   - Example: $0.01 spread → noise variance ≈ 0.0025 bps²
│   ├─ Discretization (Tick Size):
│   │   - Constraint: Prices round to nearest tick ($0.01)
│   │   - Impact: Small-price stocks affected more (2% tick for $0.50 stock)
│   │   - Rounding bias: Systematic under/overstatement
│   │   - Consequence: Estimated volatility biased upward
│   │   - Mitigation: Use mid-quotes, larger intervals
│   ├─ Price Staleness:
│   │   - Definition: Quotes not updated for extended periods
│   │   - Cause: Low trading activity, market maker inattention
│   │   - Effect: Understatement of volatility (artificial smoothness)
│   │   - Consequence: Spurious autocorrelation
│   │   - Detection: Tick-by-tick timestamp analysis
│   ├─ Non-Synchronous Trading:
│   │   - Cross-sectional: Stocks trade at different times
│   │   - Lead-lag: Large-cap leads small-cap (information diffusion)
│   │   - Spurious correlation: Appears correlated due to timing
│   │   - Hayashi-Yoshida: Estimator for non-synchronous data
│   ├─ Market Impact Noise:
│   │   - Temporary: Price moves beyond fundamental (inventory effect)
│   │   - Autocorrelation: Negative (reversion after trade)
│   │   - Duration: Milliseconds to seconds (decays)
│   │   - Modeling: Separate transient component from permanent
│   └─ Rounding Errors:
│       - Display: Prices shown with limited precision
│       - Database: Storage precision (float vs double)
│       - Accumulation: Errors compound over many observations
│       - Generally minor: <0.01% effect in modern data
│
├─ Statistical Properties:
│   ├─ Autocorrelation Structure:
│   │   - First-order: Negative (ρ₁ < 0) due to bid-ask bounce
│   │   - Higher-order: Near zero (white noise beyond lag 1)
│   │   - Roll model: ρ₁ = -0.25 (theoretical under assumptions)
│   │   - Empirical: ρ₁ ≈ -0.1 to -0.4 (varies by stock, time)
│   ├─ Variance Decomposition:
│   │   - Observed variance: σ²_obs = σ²_true + σ²_noise
│   │   - Signal-to-noise: SNR = σ²_true / σ²_noise
│   │   - High frequency: Noise dominates (SNR < 1)
│   │   - Low frequency: Signal dominates (SNR > 10)
│   │   - Optimal sampling: Balance noise vs information loss
│   ├─ Bias in Realized Variance:
│   │   - Upward bias: RV(Δt) increases as Δt → 0
│   │   - Signature plot: RV vs sampling frequency
│   │   - Minimum: Optimal Δt where bias minimized
│   │   - Rule of thumb: Sample every 5-15 minutes (equities)
│   ├─ Distribution:
│   │   - Noise: Typically symmetric, bounded (bid-ask)
│   │   - True returns: Fat-tailed, skewed (fundamentals)
│   │   - Combined: Mixture distribution (complex)
│   │   - High freq: Noise dominates, appears uniform
│   └─ Time-Varying Noise:
│       - Intraday: Higher at open/close (wider spreads)
│       - Volatility: Noise increases with market volatility
│       - Liquidity: Illiquid periods have more noise
│       - Non-stationary: Noise variance changes over time
│
├─ Measurement & Estimation:
│   ├─ Realized Variance (Naive):
│   │   - Formula: RV = Σ r²ᵢ (sum of squared returns)
│   │   - Problem: Biased upward by microstructure noise
│   │   - Magnitude: Factor of 2-10× overstatement
│   │   - Usage: Avoid at very high frequencies
│   ├─ Subsampling:
│   │   - Method: Sample every K-th observation
│   │   - Benefit: Reduces noise (less bid-ask bounce)
│   │   - Cost: Information loss (fewer observations)
│   │   - Optimal K: Signature plot minimum
│   │   - Example: Sample every 5 minutes instead of 1 second
│   ├─ Sparse Sampling:
│   │   - Multiple grids: Sample at different offsets
│   │   - Average: Combine RV estimates from each grid
│   │   - Benefit: Retains information while reducing noise
│   │   - Zhang (2006): Two-Scales Realized Variance (TSRV)
│   ├─ Two-Scales Realized Variance (TSRV):
│   │   - Slow scale: Low-frequency RV (all data)
│   │   - Fast scale: High-frequency RV (subsampled)
│   │   - Combination: RV_slow - bias correction from RV_fast
│   │   - Consistency: Converges to true IV as n → ∞
│   │   - Popular: Widely used in academic research
│   ├─ Realized Kernel:
│   │   - Bartlett kernel: Weight recent lags higher
│   │   - HAC-type: Heteroskedasticity and autocorrelation consistent
│   │   - Benefit: Optimal rate of convergence
│   │   - Drawback: Computationally intensive
│   ├─ Pre-Averaging:
│   │   - Method: Average returns over short window
│   │   - Then: Compute RV on averaged returns
│   │   - Effect: Smooths out noise, preserves signal
│   │   - Optimal window: Balance bias-variance
│   │   - Jacod et al (2009): Theory and implementation
│   └─ Maximum Likelihood (State-Space):
│       - Model: Latent efficient price + noise
│       - Estimation: Kalman filter, particle filter
│       - Benefit: Separate signal from noise optimally
│       - Cost: Requires parametric model assumptions
│       - Example: Aït-Sahalia & Yu (2009)
│
├─ Impact on Volatility Estimation:
│   ├─ High-Frequency Bias:
│   │   - Phenomenon: RV explodes as sampling → continuous
│   │   - Theory: Should converge to integrated variance
│   │   - Reality: Diverges due to noise
│   │   - Implication: Can't use tick-by-tick naively
│   ├─ Signature Plot:
│   │   - Definition: Plot RV vs sampling frequency
│   │   - Shape: U-shaped or monotone increasing
│   │   - Minimum: Optimal sampling frequency
│   │   - Left of minimum: Information loss dominates
│   │   - Right of minimum: Noise dominates
│   │   - Interpretation: Visual diagnostic for noise
│   ├─ Optimal Sampling Frequency:
│   │   - Theory: Depends on volatility and noise levels
│   │   - Rule: Sample when σ_noise ≈ σ_signal
│   │   - Equities: 5-15 minutes typical
│   │   - FX: 1-5 minutes (less noise, deeper liquidity)
│   │   - Futures: 10-30 seconds (very liquid)
│   ├─ Confidence Intervals:
│   │   - With noise: Standard errors underestimated
│   │   - Correction: Adjust for autocorrelation
│   │   - Bootstrap: Resample with replacement (noise-robust)
│   │   - HAC: Newey-West type standard errors
│   └─ Forecasting:
│       - Noisy RV: Poor predictor of future volatility
│       - Filtered RV: Better predictor (removes noise)
│       - Gain: 10-30% improvement in forecast R²
│       - Application: Option pricing, risk management
│
├─ Data Quality Issues:
│   ├─ Quote Errors (Fat Fingers):
│   │   - Definition: Erroneous quote entry ($100 → $1000)
│   │   - Detection: Outlier filters (>10σ moves)
│   │   - Handling: Remove or replace with interpolation
│   │   - Frequency: Rare but impactful (extreme observations)
│   ├─ Locked/Crossed Markets:
│   │   - Locked: Bid = ask (zero spread)
│   │   - Crossed: Bid > ask (arbitrage, but fleeting)
│   │   - Cause: Latency, routing delays, errors
│   │   - Duration: Milliseconds (quickly corrected)
│   │   - Impact: Distorts spread estimates, creates artificial trades
│   ├─ Stale Quotes:
│   │   - Definition: Quotes not updated despite market moves
│   │   - Effect: Artificial autocorrelation (smoothing)
│   │   - Detection: Long durations without quote changes
│   │   - Mitigation: Filter by time since last update
│   ├─ Missing Data:
│   │   - Gaps: Trading halts, system outages
│   │   - Irregular spacing: Non-equidistant observations
│   │   - Imputation: Forward-fill, interpolation
│   │   - Risk: Introduces additional noise/bias
│   └─ Reporting Delays:
│       - Trade reporting: Executed but reported late
│       - Timestamp: Order time vs execution time vs report time
│       - Impact: Spurious lead-lag relationships
│       - Modern systems: Nanosecond precision reduces this
│
└─ Practical Implications:
    ├─ Algorithm Design:
    │   - Execution algos: Account for noise in price signals
    │   - Risk: React to noise instead of signal → false triggers
    │   - Filter: Use mid-quotes, longer windows
    │   - Testing: Backtest with realistic noise models
    ├─ Risk Management:
    │   - VaR: Overestimated if noise not filtered
    │   - CVaR: Tail risk distorted by extreme noise observations
    │   - Stress testing: Separate noise shocks from fundamental
    ├─ Market Making:
    │   - Quote placement: Noise increases uncertainty
    │   - Inventory: Difficult to assess true position value
    │   - Adverse selection: Hard to detect informed trades in noisy data
    ├─ Regulation:
    │   - Best execution: Noise makes comparison difficult
    │   - Transaction cost analysis: Separate noise from true costs
    │   - Market quality: Measure noise as liquidity indicator
    └─ Academic Research:
        - Inference: Noise biases parameter estimates
        - Hypothesis testing: Inflated test statistics (over-rejection)
        - Model specification: Need noise-robust estimators
        - Data cleaning: Preprocessing critical for valid results
```

**Interaction:** True price evolves smoothly → observed price jumps discretely → trades alternate bid/ask → noise overlays signal → statistical methods extract true volatility

## 5. Mini-Project
Estimate and filter microstructure noise in high-frequency data:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class MicrostructureNoiseSimulator:
    def __init__(self, n_obs=5000, dt=1/252/390):  # 1-minute data
        self.n_obs = n_obs
        self.dt = dt
        
    def simulate_efficient_price(self, sigma_annual=0.20):
        """Simulate true log-price (Brownian motion)"""
        sigma_per_period = sigma_annual * np.sqrt(self.dt)
        log_returns = np.random.normal(0, sigma_per_period, self.n_obs)
        log_price = 100 + np.cumsum(log_returns)
        return log_price
    
    def add_bid_ask_bounce(self, log_price, spread=0.01):
        """Add bid-ask bounce noise"""
        # Trades alternate between bid and ask
        half_spread = spread / 2
        noise = np.random.choice([-half_spread, half_spread], size=len(log_price))
        return log_price + noise
    
    def add_discretization(self, price, tick_size=0.01):
        """Round prices to nearest tick"""
        return np.round(price / tick_size) * tick_size
    
    def calculate_realized_variance(self, prices, sampling_freq=1):
        """Calculate realized variance at given sampling frequency"""
        sampled_prices = prices[::sampling_freq]
        returns = np.diff(sampled_prices)
        rv = np.sum(returns ** 2)
        return rv
    
    def roll_spread_estimator(self, prices):
        """Estimate spread using Roll (1984) model"""
        returns = np.diff(prices)
        cov = np.cov(returns[:-1], returns[1:])[0, 1]
        
        if cov < 0:
            spread_estimate = 2 * np.sqrt(-cov)
        else:
            spread_estimate = 0  # No evidence of bid-ask bounce
        
        return spread_estimate
    
    def two_scales_rv(self, prices, fast_scale=1, slow_scale=10):
        """Two-Scales Realized Variance (Zhang 2006)"""
        # Fast scale (all data)
        rv_fast = self.calculate_realized_variance(prices, fast_scale)
        
        # Slow scale (subsampled)
        rv_slow = self.calculate_realized_variance(prices, slow_scale)
        
        # Bias correction
        n_fast = len(prices) - 1
        n_slow = (len(prices) - 1) // slow_scale
        
        # TSRV estimator
        tsrv = rv_slow - (n_slow / n_fast) * (rv_fast - rv_slow)
        
        return max(0, tsrv)  # Can't be negative

# Scenario 1: Impact of bid-ask bounce on realized variance
print("Scenario 1: Bid-Ask Bounce Impact on Realized Variance")
print("=" * 80)

sim = MicrostructureNoiseSimulator(n_obs=1000)

# Simulate true price
true_log_price = sim.simulate_efficient_price(sigma_annual=0.20)
true_price = np.exp(true_log_price)

# Add noise
spreads = [0.00, 0.01, 0.02, 0.05]  # Different spread levels

results = []
for spread in spreads:
    noisy_price = sim.add_bid_ask_bounce(true_log_price, spread=spread)
    noisy_price = np.exp(noisy_price)
    
    # Calculate RV
    rv = sim.calculate_realized_variance(noisy_price)
    rv_true = sim.calculate_realized_variance(true_price)
    
    # Noise-to-signal ratio
    noise_ratio = (rv - rv_true) / rv_true if rv_true > 0 else np.inf
    
    results.append({
        'spread': spread,
        'rv': rv,
        'rv_true': rv_true,
        'noise_ratio': noise_ratio
    })
    
    print(f"Spread: ${spread:.2f}")
    print(f"  RV (with noise): {rv:.6f}")
    print(f"  RV (true):       {rv_true:.6f}")
    print(f"  Noise inflation: {noise_ratio*100:.1f}%")
    print()

# Scenario 2: Signature plot
print("\nScenario 2: Signature Plot - RV vs Sampling Frequency")
print("=" * 80)

sim2 = MicrostructureNoiseSimulator(n_obs=5000)
true_log_price2 = sim2.simulate_efficient_price(sigma_annual=0.25)
noisy_log_price2 = sim2.add_bid_ask_bounce(true_log_price2, spread=0.02)
noisy_price2 = np.exp(noisy_log_price2)

# Calculate RV at different sampling frequencies
sampling_freqs = [1, 2, 5, 10, 20, 50, 100, 200]
rvs = []

for freq in sampling_freqs:
    rv = sim2.calculate_realized_variance(noisy_price2, sampling_freq=freq)
    rvs.append(rv)
    print(f"Sampling every {freq:>3} observations: RV = {rv:.6f}")

optimal_idx = np.argmin(rvs)
optimal_freq = sampling_freqs[optimal_idx]
print(f"\nOptimal sampling frequency: Every {optimal_freq} observations")

# Scenario 3: Roll spread estimator
print(f"\n\nScenario 3: Roll Spread Estimation")
print("=" * 80)

sim3 = MicrostructureNoiseSimulator(n_obs=1000)
true_spreads = [0.01, 0.02, 0.05, 0.10]

for true_spread in true_spreads:
    log_price = sim3.simulate_efficient_price()
    noisy_log_price = sim3.add_bid_ask_bounce(log_price, spread=true_spread)
    noisy_price = np.exp(noisy_log_price)
    
    estimated_spread = sim3.roll_spread_estimator(noisy_price)
    
    print(f"True spread: ${true_spread:.2f} → Estimated: ${estimated_spread:.4f} (Error: {abs(estimated_spread-true_spread)/true_spread*100:.1f}%)")

# Scenario 4: Two-Scales Realized Variance
print(f"\n\nScenario 4: Two-Scales RV (Noise-Robust Estimator)")
print("=" * 80)

sim4 = MicrostructureNoiseSimulator(n_obs=5000)
true_log_price4 = sim4.simulate_efficient_price(sigma_annual=0.30)
noisy_log_price4 = sim4.add_bid_ask_bounce(true_log_price4, spread=0.03)
noisy_price4 = np.exp(noisy_log_price4)
true_price4 = np.exp(true_log_price4)

# Standard RV (biased)
rv_standard = sim4.calculate_realized_variance(noisy_price4, sampling_freq=1)

# TSRV (noise-robust)
rv_tsrv = sim4.two_scales_rv(noisy_price4, fast_scale=1, slow_scale=10)

# True RV
rv_true = sim4.calculate_realized_variance(true_price4, sampling_freq=1)

print(f"True RV:        {rv_true:.6f}")
print(f"Standard RV:    {rv_standard:.6f} (bias: {(rv_standard-rv_true)/rv_true*100:+.1f}%)")
print(f"TSRV:           {rv_tsrv:.6f} (bias: {(rv_tsrv-rv_true)/rv_true*100:+.1f}%)")

# Scenario 5: Impact of tick size discretization
print(f"\n\nScenario 5: Tick Size Discretization Effects")
print("=" * 80)

sim5 = MicrostructureNoiseSimulator(n_obs=1000)
log_price5 = sim5.simulate_efficient_price()
continuous_price = np.exp(log_price5)

tick_sizes = [0.001, 0.01, 0.05, 0.10]

for tick in tick_sizes:
    discretized_price = sim5.add_discretization(continuous_price, tick_size=tick)
    
    # Calculate information loss
    price_changes = np.sum(np.diff(continuous_price) != 0)
    discrete_changes = np.sum(np.diff(discretized_price) != 0)
    
    info_loss = 1 - (discrete_changes / price_changes)
    
    print(f"Tick size: ${tick:.3f}")
    print(f"  Continuous changes: {price_changes}")
    print(f"  Discrete changes:   {discrete_changes}")
    print(f"  Information loss:   {info_loss*100:.1f}%")
    print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price paths with different noise levels
time_idx = np.arange(500)
axes[0, 0].plot(time_idx, true_price[:500], label='True Price', linewidth=2, alpha=0.7)

for spread in [0.01, 0.05]:
    noisy = sim.add_bid_ask_bounce(true_log_price[:500], spread=spread)
    axes[0, 0].plot(time_idx, np.exp(noisy), label=f'Spread=${spread:.2f}', alpha=0.6, linewidth=1)

axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: True Price vs Noisy Observations')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Signature plot
axes[0, 1].plot(sampling_freqs, rvs, 'o-', linewidth=2, markersize=8, color='blue')
axes[0, 1].axvline(x=optimal_freq, color='r', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_freq})')
axes[0, 1].set_xlabel('Sampling Frequency (every N obs)')
axes[0, 1].set_ylabel('Realized Variance')
axes[0, 1].set_title('Scenario 2: Signature Plot')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xscale('log')

# Plot 3: Roll spread estimation accuracy
true_spreads_plot = [0.01, 0.02, 0.05, 0.10]
estimated_spreads_plot = []

for true_spread in true_spreads_plot:
    log_price_plot = sim3.simulate_efficient_price()
    noisy_log_price_plot = sim3.add_bid_ask_bounce(log_price_plot, spread=true_spread)
    noisy_price_plot = np.exp(noisy_log_price_plot)
    estimated = sim3.roll_spread_estimator(noisy_price_plot)
    estimated_spreads_plot.append(estimated)

axes[1, 0].scatter(true_spreads_plot, estimated_spreads_plot, s=100, alpha=0.7, color='purple')
axes[1, 0].plot([0, 0.10], [0, 0.10], 'r--', linewidth=2, label='Perfect Estimation')
axes[1, 0].set_xlabel('True Spread ($)')
axes[1, 0].set_ylabel('Estimated Spread ($)')
axes[1, 0].set_title('Scenario 3: Roll Spread Estimator Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: RV estimator comparison
estimators = ['True RV', 'Standard RV\n(biased)', 'TSRV\n(robust)']
rv_values = [rv_true, rv_standard, rv_tsrv]
colors_plot = ['green', 'red', 'blue']

bars = axes[1, 1].bar(estimators, rv_values, color=colors_plot, alpha=0.7)
axes[1, 1].axhline(y=rv_true, color='green', linestyle='--', linewidth=2, alpha=0.5, label='True Value')
axes[1, 1].set_ylabel('Realized Variance')
axes[1, 1].set_title('Scenario 4: Volatility Estimator Comparison')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

for bar, val in zip(bars, rv_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.5f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n\nSummary:")
print("=" * 80)
print(f"Microstructure noise inflates RV estimates by 50-500% at high frequencies")
print(f"Optimal sampling: Balance information loss vs noise contamination")
print(f"Roll estimator: Recover spread from negative autocorrelation")
print(f"TSRV: Noise-robust alternative to standard RV (bias correction)")
print(f"Practical implication: Never use tick-by-tick data naively for volatility")
```

## 6. Challenge Round
If microstructure noise is well-documented and estimators exist to correct it, why do practitioners still use contaminated high-frequency data without adjustment?

- **Computational cost**: Noise-robust estimators (TSRV, realized kernel, pre-averaging) require sophisticated implementation → practitioners default to naive RV → legacy systems incompatible
- **Real-time constraints**: Market-making algorithms need instant volatility estimates → no time for complex filtering → trade accuracy for speed → noisy estimates "good enough"
- **Model risk**: Correction methods assume noise structure (e.g., i.i.d., constant variance) → reality violates assumptions → corrected estimates may be worse than naive → practitioners prefer transparency
- **Over-correction danger**: Aggressive filtering removes signal along with noise → underestimate true volatility → mis-price risk → conservatism favors simple methods
- **Regime changes**: Noise characteristics non-stationary (spreads widen in stress) → static correction fails → dynamic adjustment difficult → practitioners avoid complexity

## 7. Key References
- [Aït-Sahalia, Mykland & Zhang (2005) - How Often to Sample a Continuous-Time Process](https://www.jstor.org/stable/1392757)
- [Zhang, Mykland & Aït-Sahalia (2005) - A Tale of Two Time Scales](https://www.jstor.org/stable/3647587)
- [Hansen & Lunde (2006) - Realized Variance and Market Microstructure Noise](https://www.jstor.org/stable/40056847)
- [Barndorff-Nielsen et al (2008) - Designing Realized Kernels](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2008.00894.x)

---
**Status:** High-frequency price distortions from market friction | **Complements:** Bid-Ask Bounce, Realized Volatility, Signature Plot
