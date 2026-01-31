import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass

@dataclass
class NoiseParameters:
    """Microstructure noise parameters"""
    spread_bps: float = 2.0              # Bid-ask spread in bps
    trade_interval_sec: float = 1.0     # Avg time between trades
    true_volatility_daily: float = 0.02  # True underlying volatility

class BidAskBounceSimulator:
    """Simulate price data with bid-ask bounce"""
    
    def __init__(self, params: NoiseParameters):
        self.params = params
        self.price_mid = 100.0
    
    def generate_price_data(self, n_periods=5000):
        """Generate price series with bid-ask bounce"""
        spread = self.params.spread_bps / 10000
        sigma = self.params.true_volatility_daily / np.sqrt(252)
        
        prices = [self.price_mid]
        mid_prices = [self.price_mid]
        trade_sides = []  # Track bid/ask: +1 for ask, -1 for bid
        
        for t in range(n_periods):
            # True midprice evolution (signal)
            dM = np.random.normal(0, sigma)
            mid_new = self.price_mid * (1 + dM)
            mid_prices.append(mid_new)
            
            # Random trade side (bid or ask)
            side = np.random.choice([-1, 1])  # -1: bid, +1: ask
            trade_sides.append(side)
            
            # Observed price = midprice + spread component
            price_obs = mid_new + side * spread / 2
            prices.append(price_obs)
            
            self.price_mid = mid_new
        
        return np.array(prices), np.array(mid_prices), np.array(trade_sides)
    
    def compute_returns(self, prices):
        """Compute log returns"""
        return np.log(prices[1:] / prices[:-1])
    
    def compute_autocorrelation(self, returns, lag=1):
        """Compute first-order autocorrelation"""
        mean = returns.mean()
        c0 = np.mean((returns - mean) ** 2)
        c1 = np.mean((returns[:-lag] - mean) * (returns[lag:] - mean))
        return c1 / c0 if c0 > 0 else 0
    
    def estimate_realized_volatility(self, prices, interval=1):
        """Estimate realized volatility"""
        # Aggregate prices at specified interval
        prices_agg = prices[::interval]
        returns = np.log(prices_agg[1:] / prices_agg[:-1])
        rv = np.sqrt(np.sum(returns ** 2))
        return rv * np.sqrt(252)  # Annualized
    
    def estimate_roll_spread(self, prices):
        """Estimate bid-ask spread from negative autocorrelation"""
        returns = self.compute_returns(prices)
        cov_lag = np.mean(returns[:-1] * returns[1:])
        
        if cov_lag >= 0:
            return 0  # No bounce signal
        
        spread_est = 2 * np.sqrt(-cov_lag)
        return spread_est

# Run simulation
print("="*80)
print("BID-ASK BOUNCE & MICROSTRUCTURE NOISE")
print("="*80)

params = NoiseParameters(
    spread_bps=2.0,
    trade_interval_sec=1.0,
    true_volatility_daily=0.02
)

simulator = BidAskBounceSimulator(params)

# Generate price data
print("\nGenerating price data with bid-ask bounce...")
prices_observed, prices_mid, trade_sides = simulator.generate_price_data(n_periods=5000)

# Compute returns
returns_observed = simulator.compute_returns(prices_observed)
returns_mid = simulator.compute_returns(prices_mid)

# Analysis
print(f"\nData Summary:")
print(f"  Total observations: {len(prices_observed)}")
print(f"  True spread: {params.spread_bps} bps")
print(f"  True volatility: {params.true_volatility_daily*100:.2f}% daily")

print(f"\nMicrostructure Analysis:")
acf_observed = simulator.compute_autocorrelation(returns_observed)
acf_mid = simulator.compute_autocorrelation(returns_mid)

print(f"  ACF (observed prices): {acf_observed:.4f}")
print(f"  ACF (midprices): {acf_mid:.4f}")
print(f"  Difference: {acf_observed - acf_mid:.4f}")

# Roll estimator
roll_spread = simulator.estimate_roll_spread(prices_observed)
print(f"\nRoll Spread Estimator:")
print(f"  Estimated spread: {roll_spread*10000:.2f} bps")
print(f"  True spread: {params.spread_bps:.2f} bps")
print(f"  Error: {abs(roll_spread*10000 - params.spread_bps):.2f} bps")

# Realized volatility at different frequencies
print(f"\nRealized Volatility by Sampling Frequency:")
frequencies = [1, 5, 10, 30, 60]
rv_observed = []
rv_mid = []

for freq in frequencies:
    rv_obs = simulator.estimate_realized_volatility(prices_observed, interval=freq)
    rv_m = simulator.estimate_realized_volatility(prices_mid, interval=freq)
    rv_observed.append(rv_obs)
    rv_mid.append(rv_m)
    
    print(f"  {freq:3d}-sec interval:")
    print(f"    Observed RV: {rv_obs*100:.3f}%")
    print(f"    Midprice RV: {rv_m*100:.3f}%")
    print(f"    Inflation: {(rv_obs/rv_m - 1)*100:.1f}%")

# Two-Scales Realized Volatility (TSRV)
def compute_tsrv(prices, K=5, N=252):
    """Two-scales realized volatility"""
    # High frequency
    rv_high = np.sum(np.log(prices[1:] / prices[:-1]) ** 2)
    
    # Low frequency (every K observations)
    prices_low = prices[::K]
    rv_low = np.sum(np.log(prices_low[1:] / prices_low[:-1]) ** 2)
    
    # TSRV correction
    n_high = len(prices) - 1
    n_low = len(prices_low) - 1
    
    tsrv = rv_low - (K / n_high) * rv_high
    
    return np.sqrt(tsrv) * np.sqrt(252)  # Annualized

rv_tsrv = compute_tsrv(prices_observed)
rv_true_estimate = simulator.estimate_realized_volatility(prices_mid, interval=60)

print(f"\nNoise-Adjusted Volatility Estimation:")
print(f"  Naive RV (1-sec): {simulator.estimate_realized_volatility(prices_observed, 1)*100:.3f}%")
print(f"  TSRV (2-scale): {rv_tsrv*100:.3f}%")
print(f"  Midprice RV (60-sec): {rv_true_estimate*100:.3f}%")
print(f"  True volatility: {params.true_volatility_daily*100*np.sqrt(252):.3f}%")

# Variance decomposition
var_obs = np.var(returns_observed)
var_mid = np.var(returns_mid)
var_noise = var_obs - var_mid
print(f"\nVariance Decomposition:")
print(f"  Total (observed): {var_obs*10000:.4f} bps²")
print(f"  Signal (midprice): {var_mid*10000:.4f} bps²")
print(f"  Noise: {var_noise*10000:.4f} bps²")
print(f"  Noise fraction: {(var_noise/var_obs)*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Price series comparison
axes[0, 0].plot(prices_observed[:500], alpha=0.7, label='Observed price', linewidth=0.8)
axes[0, 0].plot(prices_mid[:500], alpha=0.7, label='Midprice', linewidth=1)
axes[0, 0].set_title('Price Series: Observed vs Midprice (first 500 periods)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Returns comparison
axes[0, 1].scatter(range(100), returns_observed[:100], alpha=0.5, s=20, label='Observed')
axes[0, 1].scatter(range(100), returns_mid[:100], alpha=0.5, s=20, label='Midprice')
axes[0, 1].set_title('Returns: Observed vs Midprice')
axes[0, 1].set_ylabel('Log Return')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Autocorrelation function
lags = range(1, 31)
acf_obs_lags = [simulator.compute_autocorrelation(returns_observed, lag) for lag in lags]
acf_mid_lags = [simulator.compute_autocorrelation(returns_mid, lag) for lag in lags]

axes[0, 2].plot(lags, acf_obs_lags, marker='o', label='Observed', linewidth=1.5)
axes[0, 2].plot(lags, acf_mid_lags, marker='s', label='Midprice', linewidth=1.5)
axes[0, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 2].set_title('Autocorrelation Function')
axes[0, 2].set_xlabel('Lag')
axes[0, 2].set_ylabel('ACF')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: RV vs Frequency
axes[1, 0].plot(frequencies, np.array(rv_observed)*100, marker='o', label='Observed', linewidth=2)
axes[1, 0].plot(frequencies, np.array(rv_mid)*100, marker='s', label='Midprice', linewidth=2)
axes[1, 0].axhline(params.true_volatility_daily*100*np.sqrt(252), color='red', 
                    linestyle='--', label='True volatility', linewidth=1.5)
axes[1, 0].set_xlabel('Sampling Interval (seconds)')
axes[1, 0].set_ylabel('Realized Volatility (% annual)')
axes[1, 0].set_xscale('log')
axes[1, 0].set_title('RV Volatility Estimate by Sampling Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, which='both')

# Plot 5: Distribution of returns
axes[1, 1].hist(returns_observed*10000, bins=50, alpha=0.6, label='Observed', edgecolor='black')
axes[1, 1].hist(returns_mid*10000, bins=50, alpha=0.6, label='Midprice', edgecolor='black')
axes[1, 1].set_title('Distribution of Returns')
axes[1, 1].set_xlabel('Return (bps)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Cumulative noise impact
noise_cumulative = np.cumsum((returns_observed - returns_mid)**2)
axes[1, 2].plot(noise_cumulative[:1000], linewidth=1)
axes[1, 2].set_title('Cumulative Squared Noise (Return Difference)')
axes[1, 2].set_xlabel('Observation')
axes[1, 2].set_ylabel('Cumulative Noise Variance')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Bid-ask bounce creates strong negative autocorrelation in returns")
print(f"2. Naive RV inflated by {(simulator.estimate_realized_volatility(prices_observed, 1)/simulator.estimate_realized_volatility(prices_mid, 60) - 1)*100:.1f}% at high frequency")
print(f"3. Aggregating to 60+ seconds substantially reduces noise impact")
print(f"4. TSRV provides principled denoising without requiring midprices")
print(f"5. Noise accounts for {(var_noise/var_obs)*100:.1f}% of variance in this simulation")
