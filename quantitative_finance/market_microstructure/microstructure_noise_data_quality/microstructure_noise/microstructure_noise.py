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
