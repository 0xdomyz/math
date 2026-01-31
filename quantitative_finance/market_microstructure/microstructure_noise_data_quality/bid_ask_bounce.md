# Bid-Ask Bounce & Microstructure Noise

## 1. Concept Skeleton
**Definition:** Artificial price volatility caused by trades alternating between bid and ask prices; creates spurious variance that disappears when averaged over time, a key component of microstructure noise  
**Purpose:** Distinguish true price discovery (signal) from mechanical bid-ask crossing (noise), correct volatility estimates, identify optimal sampling frequency, prevent overfitting to noise  
**Prerequisites:** Order book mechanics, market microstructure, volatility estimation, time series analysis, stochastic processes

## 2. Comparative Framing
| Component | Bid-Ask Bounce | True Price Move | Noise-Only | Signal+Noise |
|-----------|----------------|-----------------|------------|-------------|
| **Persistence** | Mean-reverting, <1 sec | Permanent, trending | Expires quickly | Mixes both |
| **Source** | Spread crossing | Information arrival | Liquidity provision | Combined |
| **Correlation** | Negative (alternates) | Positive (momentum) | Negative | Positive |
| **Variance Impact** | Inflates (by 0.5-2x) | Legitimate | Inflates RV | Reduces measured signal |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock trading 100.00 bid / 100.01 ask. Buy order: fill at 100.01 (trade price). Sell order: fill at 100.00 (trade price). Recorded as price sequence: 100.00 → 100.01 → 100.00. Appears volatile, but no true move occurred.

**Failure Case:**  
Using tick-by-tick returns to measure volatility without accounting for bounce: σ = 0.50% measured, true σ = 0.30%. Estimate too high by 67%. Risk model overestimates hedging costs.

**Edge Case:**  
Wide-spread illiquid stock (5 bps spread): Bounce dominates. Tight-spread liquid stock (0.5 bps spread): Bounce negligible. Sampling frequency critical for both.

## 4. Layer Breakdown
```
Bid-Ask Bounce & Microstructure Noise:
├─ Theoretical Foundation:
│   ├─ Roll Model (1984):
│   │   ├─ Assumption: Observed prices are noisy versions of true price
│   │   ├─ Model: P_obs(t) = P_true(t) + u(t)
│   │   ├─ u(t): -0.5S with prob 0.5 (bid fill), +0.5S with prob 0.5 (ask fill)
│   │   ├─ Results:
│   │   ├─  Var(ΔP_obs) = Var(ΔP_true) + Var(u)
│   │   ├─  Var(u) ≈ 0.25S² (bounce variance grows with spread)
│   │   └─  Cov(ΔP_obs,t, ΔP_obs,t-1) = -0.25S² (negative autocorr)
│   ├─ Bounce Mechanism:
│   │   ├─ Step 1: Price at bid B = M - S/2
│   │   ├─ Step 2: Buy order fills at ask A = M + S/2
│   │   ├─ Step 3: Sell order fills at bid B = M - S/2
│   │   ├─ Recorded sequence: B → A → B (oscillation)
│   │   ├─ Actual midprice: M → M → M (no move!)
│   │   └─ Variance from oscillation: Spurious
│   ├─ Decomposition:
│   │   ├─ Total variance: σ²_total = σ²_true + σ²_bounce
│   │   ├─ σ²_true: Price discovery component (signal)
│   │   ├─ σ²_bounce: Bid-ask alternation (noise)
│   │   ├─ σ²_bounce ≈ 0.25 × S²
│   │   └─ Implication: Narrow spreads → less bounce variance
│   └─ Stochastic Model:
│       ├─ Midprice follows: dM = μdt + σdW (Brownian motion)
│       ├─ Observed price: P_obs = M + b(t)×S/2
│       ├─ b(t): ±1 indicator (last trade side)
│       ├─ Returns: ΔP_obs = ΔM + (b(t) - b(t-1))×S/2 + measurement error
│       └─ If bid→ask→bid: Return = -S (100% reversal)
├─ Empirical Evidence:
│   ├─ Autocorrelation Analysis:
│   │   ├─ Unfiltered returns: Often negative autocorr (0 to -0.2)
│   │   ├─ Midprice-based returns: Usually positive or zero autocorr
│   │   ├─ Distinction: Bounce creates negative serial correlation
│   │   ├─ Signature: Daily patterns (tight spreads midday → less bounce)
│   │   └─ Measurement: ACF(1) < 0 indicates bounce dominance
│   ├─ Data Example:
│   │   ├─ Liquid large-cap (AAPL): 1-2 bps spread → Bounce minimal
│   │   ├─ Calculated from midprice: σ_mid ≈ 0.30% daily
│   │   ├─ Calculated from trade prices: σ_trade ≈ 0.35% daily
│   │   ├─ Ratio: 1.17x → 17% inflation from bounce
│   │   ├─ Expected from Roll: 0.5S² = 0.5×(0.015)² ≈ 0.000113 ≈ 1.1%/day²
│   │   └─ Empirical: Matches theory well
│   ├─ Intraday Patterns:
│   │   ├─ Open (9:30am): Wide spreads (5-10 bps) → High bounce
│   │   ├─ Mid-day (11am-3pm): Tight spreads (1-2 bps) → Low bounce
│   │   ├─ Close (3:50pm): Widening (3-5 bps) → Moderate bounce
│   │   ├─ After-hours: Very wide (10+ bps) → Extreme bounce
│   │   └─ Pattern: Volatility estimates vary 2-3x intraday (mostly bounce)
│   └─ Cross-Asset:
│       ├─ Equity: Bounce ~30-50% of total variance (liquid large-cap)
│       ├─ Small-cap: Bounce ~50-80% of total variance (illiquid)
│       ├─ Futures: Bounce ~20% of total variance (tight spreads)
│       ├─ Bonds: Bounce ~10% (very wide true spreads)
│       └─ FX: Bounce negligible for majors (<0.1 pips spread)
├─ Bid-Ask Bounce Quantification:
│   ├─ Roll Estimator for Spread:
│   │   ├─ S = 2√(-Cov(ΔP_t, ΔP_t-1))
│   │   ├─ From observed return autocorrelation
│   │   ├─ Negative covariance indicator of bounce
│   │   ├─ Advantage: No need for quote data, only trade prices
│   │   ├─ Limitation: Assumes only source of negative autocorr is bounce
│   │   └─ Reality: Other factors (inventory effects) also create negative autocorr
│   ├─ High-Frequency Bounce:
│   │   ├─ 1-second sampling: Bounce dominates, σ = 0.7%
│   │   ├─ 5-second sampling: Bounce reduces, σ = 0.5%
│   │   ├─ 1-minute sampling: Bounce minimal, σ = 0.3%
│   │   ├─ 5-minute sampling: Mostly signal, σ = 0.25%
│   │   └─ Pattern: σ decreases with aggregation (noise cancellation)
│   ├─ Noise Variance Decomposition:
│   │   ├─ σ²_obs(1-min) = σ²_signal + σ²_microstructure
│   │   ├─ σ²_signal ≈ 0.25% (from 5-min measurement)
│   │   ├─ σ²_microstructure ≈ 0.30% - 0.25% = 0.05%
│   │   ├─ Interpretation: 17% of variance is microstructure noise
│   │   └─ Implication: Daily vol ~0.25%×√252 ≈ 4% vs measured 5% (noise inflation)
│   └─ Bounce Recovery Time:
│       ├─ Half-life: ~1-2 seconds for bounce to revert
│       ├─ Full recovery: ~5-10 seconds (depends on tick size, spread)
│       ├─ Mechanism: Next trade likely reverses last one (mean reversion)
│       └─ Timing: Longer for illiquid assets (fewer trades to offset)
├─ Realized Volatility Under Bounce:
│   ├─ Naive RV (1-min returns):
│   │   ├─ RV_naive = Σ(r_i)² with r_i = log(P_i/P_i-1)
│   │   ├─ Biased upward due to bounce component
│   │   ├─ Typical inflation: 10-50% depending on spread
│   │   └─ Problem: Overestimates true price move volatility
│   ├─ Midprice RV (preferrable):
│   │   ├─ RV_mid = Σ(r_mid,i)² using midprices
│   │   ├─ Removes bid-ask bounce by construction
│   │   ├─ More stable across time-of-day
│   │   ├─ Limitation: Requires quote data (not always available)
│   │   └─ Advantage: Closer to true underlying volatility
│   ├─ Two-Scales RV (TSRV, Zhang 2005):
│   │   ├─ Combines high-frequency and low-frequency sampling
│   │   ├─ Formula: TSRV = RV_5min - (K/N)×RV_1sec
│   │   ├─ First term: Slow scale (removes noise)
│   │   ├─ Second term: Correction for remaining noise
│   │   ├─ K, N: Number of high-freq / low-freq observations
│   │   └─ Result: Nearly unbiased, optimal noise removal
│   ├─ Multi-Scale RV:
│   │   ├─ Generalizes TSRV to multiple timescales
│   │   ├─ Average across many scales for robustness
│   │   ├─ Computationally intensive but very accurate
│   │   └─ Academic standard for RV estimation
│   └─ Pre-Averaging Method:
│       ├─ Aggregate returns before squaring (smooth noise)
│       ├─ PA-RV = Σ(r̄_i)² where r̄ = avg of k returns
│       ├─ k: Pre-averaging window (e.g., 5 returns)
│       ├─ Simple but effective for bias reduction
│       └─ Less rigorous than TSRV but practical
├─ Impact on Applications:
│   ├─ Risk Estimation:
│   │   ├─ Naive RV: σ = 0.35% (inflated)
│   │   ├─ Corrected RV: σ = 0.30% (true)
│   │   ├─ VaR difference: 17% higher for naive method
│   │   ├─ Consequence: Over-hedging, excess costs
│   │   └─ Fix: Use TSRV or midprice-based measures
│   ├─ Portfolio Optimization:
│   │   ├─ Covariance matrix biased if noise ignored
│   │   ├─ Overestimates correlations near zero
│   │   ├─ Efficient frontier suboptimal
│   │   └─ Fix: Clean covariance via RV methods
│   ├─ Market Microstructure Models:
│   │   ├─ High-frequency strategies fail if noise in returns
│   │   ├─ Backtests show inflated Sharpe ratios
│   │   ├─ Live trading disappoints vs backtest
│   │   └─ Fix: Backtest with noise-adjusted returns
│   ├─ Trading Costs:
│   │   ├─ Execution algo assumes σ for impact calc
│   │   ├─ If σ inflated, impact underestimated
│   │   ├─ Orders too aggressive, face higher costs
│   │   └─ Fix: Use true σ in impact models
│   └─ Liquidity Measurement:
│       ├─ Amihud illiquidity depends on return variance
│       ├─ If returns inflated by noise, illiquidity inflated
│       ├─ False signals: "Illiquid" when actually "noisy"
│       └─ Fix: Use noise-corrected volatility
├─ Practical Detection & Mitigation:
│   ├─ Detecting Bounce Dominance:
│   │   ├─ Calculate first-order autocorrelation: ρ₁
│   │   ├─ If ρ₁ < -0.05: Bounce likely significant
│   │   ├─ If ρ₁ > 0: Signal dominates (positive momentum)
│   │   ├─ If ρ₁ ≈ 0: Mixed or minimal bounce
│   │   └─ Rule of thumb: |ρ₁| > spread/price suggests bounce
│   ├─ Sampling Frequency Selection:
│   │   ├─ Too high (1-sec): Bounce dominates, high noise
│   │   ├─ Optimal (5-10 min): Balance signal capture and noise reduction
│   │   ├─ Too low (daily): Lose intraday patterns
│   │   ├─ Recommendation: 5-minute for standard analysis
│   │   └─ Adaptive: Use longer intervals for illiquid assets
│   ├─ Quote-Based Returns:
│   │   ├─ Use midprice (bid+ask)/2 instead of trade price
│   │   ├─ Removes bounce mechanically
│   │   ├─ Trade-off: Loses information about execution side
│   │   └─ Best practice: Use both, compare results
│   ├─ Filtering Methods:
│   │   ├─ Exponential smoothing: Reduce high-freq noise
│   │   ├─ Kalman filter: Optimal Bayesian denoising
│   │   ├─ Moving average: Simple, robust noise reduction
│   │   └─ Threshold: Ignore returns < threshold (micro-noise)
│   └─ Statistical Tests:
│       ├─ Run test: Check for negative autocorr pattern
│       ├─ Variance ratio test: Var(Δ2) / (2×Var(Δ1)) < 2 → noise present
│       ├─ LM test: Specific test for microstructure noise
│       └─ Interpretation: Guides sampling frequency choice
└─ Evolution of Understanding:
    ├─ Pre-2000: Bid-ask bounce largely ignored
    ├─ 2000s: Academic focus on microstructure noise (Aït-Sahalia et al)
    ├─ 2010s: Practitioner adoption of TSRV, denoising methods
    ├─ 2020s: ML approaches for noise detection in high-freq data
    └─ Future: Adaptive sampling based on real-time noise estimates
```

**Interaction:** Trade executes → Price oscillates bid/ask → Bounce creates negative autocorr → Volatility inflates → Aggregate across time → Bounce decays → True signal emerges

## 5. Mini-Project
Decompose microstructure noise and estimate true volatility:
```python
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
```

## 6. Challenge Round
Why does bid-ask bounce create negative autocorrelation?
- **Mechanism**: Trade at ask → next trade likely at bid (mean reversion) → negative correlation
- **Persistence**: Bounce mean-reverts quickly (1-2 seconds), unlike true moves
- **Spread dependence**: Wider spread → larger bounce → stronger negative autocorr
- **Implication**: Negative ACF indicator of microstructure noise presence

How to distinguish bounce from true negative momentum?
- **Timing**: Bounce reverts in <10 seconds; momentum persists longer
- **Spread correlation**: Bounce increases with spread; momentum independent
- **Reversal asymmetry**: Bounce symmetric (bounce back fully); momentum may not
- **Midprice check**: Compute midprice returns; if negative ACF disappears → bounce confirmed

## 7. Key References
- [Roll (1984): A Simple Implicit Measure of the Effective Bid-Ask Spread](https://www.jstor.org/stable/2327617)
- [Aït-Sahalia, Mykland, Zhang (2005): How Often to Sample a Continuous-Time Process](https://www.jstor.org/stable/1392757)
- [Hasbrouck & Saar (2013): Low-Latency Trading](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12022)

---
**Status:** Core noise component in microstructure | **Complements:** Realized Volatility, Data Quality, Signature Plot
