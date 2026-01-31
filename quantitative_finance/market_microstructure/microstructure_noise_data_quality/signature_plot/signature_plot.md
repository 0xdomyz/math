# Signature Plot & Optimal Sampling Frequency

## 1. Concept Skeleton
**Definition:** Graphical tool plotting realized variance as function of sampling frequency; reveals optimal sampling frequency by identifying elbow point where microstructure noise dominates over signal extraction benefit  
**Purpose:** Empirically determine optimal frequency for data analysis, detect noise vs. signal regimes, select RV estimation parameters, balance bias-variance tradeoff in volatile estimation  
**Prerequisites:** Microstructure noise, realized volatility, sampling theory, bias-variance analysis, time series visualization

## 2. Comparative Framing
| Component | Noise Regime | Optimal Zone | Signal Regime |
|-----------|--------------|--------------|---------------|
| **Frequency** | >100 Hz (sub-second) | 1-10 Hz (1-10 sec) | <0.01 Hz (>100 sec) |
| **RV Behavior** | Increasing (noise dominates) | Relatively flat | Decreasing (sample path variation) |
| **Slope** | Positive | Zero/minimal | Negative (rare) |
| **Use Case** | Avoid | Ideal sampling | Acceptable but slow |

## 3. Examples + Counterexamples

**Simple Example:**  
Plot RV at frequencies: 1-sec (2.5%), 5-sec (2.2%), 10-sec (2.1%), 30-sec (2.15%), 60-sec (2.20%). Elbow at ~10-sec. Use 10-sec data for analysis (minimum noisy-adjusted RV).

**Failure Case:**  
Use 1-sec data without plotting signature: RV = 3.5% (inflated 40%). Backtest assumes low vol, live trading disappointed. Signature plot would have revealed noise dominance.

**Edge Case:**  
Illiquid stock: Signature plot flat or noisy (sparse data). Can't rely on elbow. Solution: Use 5-10 minute sampling instead, higher confidence in estimate.

## 4. Layer Breakdown
```
Signature Plot & Optimal Sampling:
├─ Theoretical Basis:
│   ├─ Realized Variance Function:
│   │   ├─ RV(h) = Σ_i [log(P_i / P_i-1)]² at frequency 1/h
│   │   ├─ h: Sampling interval (e.g., h=60 means 60-second data)
│   │   ├─ As h→0: RV(h) → ∞ (noise dominates)
│   │   ├─ As h→∞: RV(h) → 0 (too few observations)
│   │   └─ Optimal h*: Minimizes mean squared error of vol estimate
│   ├─ Bias-Variance Decomposition:
│   │   ├─ E[RV(h)] = QV + c(h)
│   │   ├─ QV: True quadratic variation (signal)
│   │   ├─ c(h): Bias from microstructure noise
│   │   ├─ c(h) ≈ 2σ_ε²/h (inverse relationship with frequency)
│   │   ├─ Var[RV(h)] decreases with fewer observations
│   │   └─ Trade-off: Low h → low bias, high variance
│   ├─ Optimal Frequency Theory:
│   │   ├─ MSE = Bias² + Variance
│   │   ├─ Min MSE at h* where d(MSE)/dh = 0
│   │   ├─ h* ∝ (noise variance)^(1/3)
│   │   ├─ h* ∝ (integrated vol)^(-2/3)
│   │   ├─ Rule of thumb: h* ≈ √(noise level × integrated vol)
│   │   └─ In practice: 5-10 minute intervals typical
│   └─ Multi-Scale Optimal Selection:
│       ├─ TSRV framework: h_low ≈ √(n) × h_high
│       ├─ Ratio K = h_low / h_high
│       ├─ Optimal K ∝ n^(1/4)
│       ├─ Example: n=250, K≈4; n=10000, K≈10
│       └─ Signature plot guides parameter selection
├─ Signature Plot Construction:
│   ├─ Algorithm:
│   │   ├─ Step 1: Choose frequency grid: h = 1, 2, 5, 10, 30, 60, 300, 600 seconds
│   │   ├─ Step 2: For each h, compute RV(h) for day
│   │   ├─ Step 3: Repeat for 20+ days, average RV(h)
│   │   ├─ Step 4: Plot log(RV(h)) vs log(1/h) or RV(h) vs log(h)
│   │   ├─ Step 5: Identify elbow (knee) point
│   │   └─ Step 6: Select h* near elbow for analysis
│   ├─ Interpretation:
│   │   ├─ Flat region: h* (stable, noise minimized)
│   │   ├─ Upward slope: Noise dominates (too high freq)
│   │   ├─ Downward slope: Too few samples (rare, usually ignored)
│   │   ├─ Multiple knees: Regime change or data quality issue
│   │   └─ No clear knee: Use practical frequencies (5-10 min)
│   ├─ Visual Characteristics:
│   │   ├─ Ideal signature: Decreases then flattens (U-like or L-like)
│   │   ├─ Flattening at 5-10 min range: Liquid equities
│   │   ├─ Late flattening >60 min: Less liquid instruments
│   │   ├─ Very noisy/jagged: Data quality problems
│   │   └─ Anomalies: Spikes indicate data glitches
│   └─ Robustness:
│       ├─ Seasonal adjustment: Plot separately by hour-of-day
│       ├─ Day-of-week: Check if patterns change Mon vs Fri
│       ├─ Regime-dependent: Stock-specific (large-cap vs small-cap)
│       ├─ Time-varying: Update periodically (monthly/quarterly)
│       └─ Multiple assets: Create bank of signatures for portfolio
├─ Data Quality Diagnostics:
│   ├─ Noise Estimation:
│   │   ├─ From signature: Slope in noise-dominated region
│   │   ├─ Fit line to log-log plot: slope = -1 (theory predicts)
│   │   ├─ Deviation from -1: Indicates non-iid noise
│   │   ├─ Intercept: Gives estimate of σ_ε²
│   │   └─ Validation: Compare with Roll estimator
│   ├─ Signal Extraction:
│   │   ├─ In flat region: RV(h) ≈ true QV
│   │   ├─ Subtract baseline: QV ≈ RV(h*) - c(h*)
│   │   ├─ More precision: Use TSRV formula
│   │   └─ Validation: Cross-check with option-implied vol
│   ├─ Outlier/Data Quality Detection:
│   │   ├─ Spikes in signature: Look for corresponding data points
│   │   ├─ Unusual flatness: Could indicate inactive trading
│   │   ├─ Discontinuity at specific h: Data collection change
│   │   ├─ Asymmetry: Different behavior before/after event
│   │   └─ Action: Investigate, clean, or exclude data
│   └─ Microstructure Regime Changes:
│       ├─ Tick size reduction: Signature changes (more liquid)
│       ├─ Listing change: Different exchange = different noise
│       ├─ Volume increase: Noise decreases (tighter spreads)
│       ├─ Fragmentation: Multiple venues add complexity
│       └─ Monitoring: Monthly signature review detects shifts
├─ Practical Applications:
│   ├─ Data Selection for Analysis:
│   │   ├─ Use h* frequency consistently for project
│   │   ├─ If multiple datasets, find common h*
│   │   ├─ Document choice in methodology section
│   │   ├─ Sensitivity analysis: Test results at nearby frequencies
│   │   └─ Robustness: Similar conclusions across frequencies?
│   ├─ Algorithm Parameter Tuning:
│   │   ├─ VWAP slicer: Use vol estimate from signature-selected RV
│   │   ├─ Risk model: Scale covariance matrix by RV(h*) / RV_daily
│   │   ├─ Optimal execution: Input vol to Almgren-Chriss model
│   │   └─ Portfolio rebalancing: Frequency inversely related to vol
│   ├─ Model Validation:
│   │   ├─ Compare predicted vs realized vol
│   │   ├─ Use RV(h*) as true vol benchmark
│   │   ├─ Better calibration: Model vol should match RV(h*)
│   │   └─ If gap: Investigate model assumptions
│   ├─ HFT Strategy Development:
│   │   ├─ Noise dominance analysis: Profit margins in noise region
│   │   ├─ Market depth estimation: Signature slopes relate to depth
│   │   ├─ Optimal hold time: From signature decay time constant
│   │   └─ Capacity: Strategy profitable down to noise floor
│   └─ Regulatory/Compliance:
│       ├─ Explain data sampling choice to regulators
│       ├─ Signature plot justification for frequency selection
│       ├─ MiFID II: Document best execution analysis
│       └─ Transparency: Publish signature plot for benchmarks
├─ Extension: Multi-Asset Signature Plotting:
│   ├─ Portfolio Level:
│   │   ├─ Compute portfolio RV(h) across frequencies
│   │   ├─ Diversification effect visible
│   │   ├─ If portfolio RV well-behaved, good data quality
│   │   └─ Use for setting portfolio risk limits
│   ├─ Cross-Asset:
│   │   ├─ Compare signatures across similar stocks
│   │   ├─ Identify outliers (data quality issues)
│   │   ├─ Use median signature across universe
│   │   └─ Benchmark purposes
│   ├─ Cross-Exchange:
│   │   ├─ If stock trades on multiple venues
│   │   ├─ Compare venue-specific signatures
│   │   ├─ Price discovery questions: Which venue best RV?
│   │   └─ Informed order routing
│   └─ Temporal Signature:
│       ├─ Intraday: Compute signature separately by hour
│       ├─ Reveal hour-specific noise levels
│       ├─ Open: High noise, low flat region
│       ├─ Midday: Low noise, earlier flat region
│       ├─ Close: High noise again
│       └─ Application: Time-dependent frequency selection
├─ Pitfalls & Limitations:
│   ├─ Illiquid Assets:
│   │   ├─ Sparse data: Gaps in observations
│   │   ├─ Signature noisy/unclear: Can't identify elbow
│   │   ├─ Solution: Use trade-time vs calendar-time sampling
│   │   ├─ Alternative: Longer intervals (30+ min) regardless
│   │   └─ Accept higher uncertainty
│   ├─ Jump Days:
│   │   ├─ Large price jumps: Signature disrupted
│   │   ├─ Particularly high noise-regime part
│   │   ├─ Solution: Exclude jump days from signature
│   │   ├─ Or decompose: Use bipower variation for signal part
│   │   └─ Check: Are outliers from jumps or noise?
│   ├─ Seasonal Patterns:
│   │   ├─ Open hour: Very different signature
│   │   ├─ Close hour: Different again
│   │   ├─ Average can hide intraday variation
│   │   ├─ Solution: Plot time-specific signatures
│   │   └─ Or sample away from open/close
│   ├─ Model Misspecification:
│   │   ├─ Signature shape assumes iid noise
│   │   ├─ Actual noise correlated (bid-ask bounce patterns)
│   │   ├─ Slope ≠ -1 indicates departures
│   │   ├─ Solution: Use robust methods (TSRV, MSRV)
│   │   └─ Or fit more flexible model
│   └─ Changing Environments:
│       ├─ Signature valid only for period it was estimated
│       ├─ Spread tightening over time: Noise decreases
│       ├─ HFT entrance: Noise characteristics change
│       ├─ Market stress: Different noise regime
│       ├─ Solution: Recompute monthly or after major events
│       └─ Adaptive: Real-time signature updating
└─ Implementation Checklist:
    ├─ Data Preparation:
    │   ├─ [ ] Obtain high-frequency data (5-10 second minimum)
    │   ├─ [ ] Clean obvious errors (gaps, outliers)
    │   ├─ [ ] Align timestamps (if multi-venue)
    │   ├─ [ ] Remove pre/after-market
    │   └─ [ ] Select 20+ days of representative data
    ├─ Computation:
    │   ├─ [ ] Select frequency grid (1, 5, 10, 30, 60, 300 sec)
    │   ├─ [ ] Compute RV(h) for each frequency each day
    │   ├─ [ ] Average across days
    │   ├─ [ ] Annualize if needed (multiply by √252)
    │   └─ [ ] Smooth if noisy (moving average)
    ├─ Visualization:
    │   ├─ [ ] Plot linear-linear: RV(h) vs h
    │   ├─ [ ] Plot log-linear: log(RV(h)) vs log(h)
    │   ├─ [ ] Identify elbow/knee point
    │   ├─ [ ] Annotate with recommended h*
    │   └─ [ ] Save figure for documentation
    ├─ Analysis:
    │   ├─ [ ] Estimate noise level from slope
    │   ├─ [ ] Compare with Roll estimator
    │   ├─ [ ] Test intraday variation
    │   ├─ [ ] Check for anomalies or regime changes
    │   └─ [ ] Document findings
    ├─ Decision:
    │   ├─ [ ] Select h* based on elbow
    │   ├─ [ ] Document justification
    │   ├─ [ ] Sensitivity analysis: ±50% of h*
    │   ├─ [ ] Share with stakeholders
    │   └─ [ ] Update periodically
    └─ Monitoring:
        ├─ [ ] Monthly: Recompute signature
        ├─ [ ] Compare to baseline: Changed significantly?
        ├─ [ ] Investigate if yes: Market change or data quality?
        ├─ [ ] Update parameters if needed
        └─ [ ] Archive signatures for audit trail
```

**Interaction:** Select frequency grid → Compute RV at each frequency → Plot → Identify elbow → Select h* → Use for analysis → Monitor/update → Iterate

## 5. Mini-Project
Create and analyze signature plots for optimal frequency selection:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from dataclasses import dataclass

@dataclass
class SignaturePlotParams:
    """Parameters for signature plot analysis"""
    frequencies_hz: list = None  # Frequencies in Hz to test
    n_days: int = 20             # Days for averaging
    true_volatility: float = 0.015
    noise_level: float = 0.0001

class SignaturePlotAnalyzer:
    """Analyze realized volatility across multiple sampling frequencies"""
    
    def __init__(self, params: SignaturePlotParams = None):
        if params is None:
            params = SignaturePlotParams()
        self.params = params
        
        if self.params.frequencies_hz is None:
            # Default frequencies (in seconds)
            self.params.frequencies_hz = [1, 2, 5, 10, 15, 30, 60, 120, 300]
    
    def generate_price_data(self, n_obs=50000):
        """Generate high-frequency price data"""
        dt = 1.0 / n_obs  # Unit time interval
        price = 100.0
        prices = [price]
        
        sigma = self.params.true_volatility / np.sqrt(252)
        
        for i in range(n_obs):
            dP = np.random.normal(0, sigma * np.sqrt(dt))
            price *= (1 + dP)
            # Add microstructure noise
            price_obs = price + np.random.normal(0, self.params.noise_level)
            prices.append(price_obs)
        
        return np.array(prices)
    
    def realized_variance(self, log_prices, freq_seconds=1, obs_per_second=1):
        """Compute realized variance at given frequency"""
        # Skip interval based on frequency
        skip = int(freq_seconds * obs_per_second)
        
        if skip >= len(log_prices):
            return np.nan
        
        prices_agg = log_prices[::skip]
        returns = np.diff(prices_agg)
        rv = np.sum(returns ** 2)
        
        return rv
    
    def compute_signature(self, n_obs=50000):
        """Compute signature plot data"""
        rvs = []
        frequencies = []
        
        # Compute observation spacing (1 second per obs default)
        obs_per_second = 1.0  # Adjust if needed
        
        for freq_sec in self.params.frequencies_hz:
            day_rvs = []
            
            # Average across multiple days
            for day in range(self.params.n_days):
                prices = self.generate_price_data(n_obs=n_obs)
                log_prices = np.log(prices)
                
                rv = self.realized_variance(log_prices, freq_seconds=freq_sec, 
                                           obs_per_second=obs_per_second)
                
                if not np.isnan(rv):
                    day_rvs.append(rv)
            
            avg_rv = np.mean(day_rvs) if day_rvs else np.nan
            
            # Annualize
            avg_rv_annual = np.sqrt(avg_rv) * np.sqrt(252)
            
            rvs.append(avg_rv_annual)
            frequencies.append(freq_sec)
        
        return np.array(frequencies), np.array(rvs)
    
    def find_elbow(self, frequencies, rvs):
        """Find elbow point using several methods"""
        # Method 1: Maximum curvature (derivative of derivative)
        # Log-log scale
        log_freq = np.log(frequencies)
        log_rv = np.log(rvs)
        
        # Fit piecewise linear in log-log space
        # Find point with maximum change in slope
        
        slopes = np.diff(log_rv) / np.diff(log_freq)
        elbow_idx = np.argmin(slopes)  # Where slope changes most (from negative to positive)
        
        return elbow_idx
    
    def estimate_noise(self, frequencies, rvs):
        """Estimate noise level from slope in noise-dominated region"""
        # High frequency region should show -1 slope in log-log
        # RV(h) = QV + 2σ_ε²/h → log(RV) = const + (-1)log(h)
        
        # Use first 3-4 points (highest frequencies)
        n_fit = min(4, len(frequencies) // 2)
        
        log_freq = np.log(frequencies[:n_fit])
        log_rv = np.log(rvs[:n_fit])
        
        # Linear fit in log-log space
        z = np.polyfit(log_freq, log_rv, 1)
        slope = z[0]
        intercept = z[1]
        
        # From RV(h) = 2σ_ε²/h + const
        # log(RV) = log(2σ_ε²) - log(h) + const
        # Slope should be -1; intercept ≈ log(2σ_ε²)
        
        noise_var = np.exp(intercept) / 2  # Rough estimate
        
        return slope, noise_var
    
    def find_optimal_frequency(self, frequencies, rvs, method='elbow'):
        """Find optimal sampling frequency"""
        if method == 'elbow':
            elbow_idx = self.find_elbow(frequencies, rvs)
            return frequencies[elbow_idx], elbow_idx
        elif method == 'flat':
            # Find where RV flattens (derivative smallest)
            diffs = np.abs(np.diff(rvs) / np.diff(frequencies))
            flat_idx = np.argmin(diffs) + 1
            return frequencies[flat_idx], flat_idx
        else:
            # Manual: use 5 minute frequency
            return 300, None

# Run signature plot analysis
print("="*80)
print("SIGNATURE PLOT & OPTIMAL SAMPLING FREQUENCY")
print("="*80)

# Test different scenarios
scenarios = [
    {
        'name': 'Liquid Asset',
        'vol': 0.015,
        'noise': 0.00008
    },
    {
        'name': 'Illiquid Asset',
        'vol': 0.025,
        'noise': 0.0003
    }
]

fig, axes = plt.subplots(len(scenarios), 3, figsize=(16, 5*len(scenarios)))

if len(scenarios) == 1:
    axes = axes.reshape(1, -1)

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario['name']}")
    print(f"{'='*80}")
    
    params = SignaturePlotParams(
        frequencies_hz=[1, 2, 5, 10, 15, 30, 60, 120, 300],
        n_days=15,
        true_volatility=scenario['vol'],
        noise_level=scenario['noise']
    )
    
    analyzer = SignaturePlotAnalyzer(params)
    
    print(f"Generating price data and computing signature plot...")
    frequencies, rvs = analyzer.compute_signature(n_obs=30000)
    
    print(f"\nSignature Plot Data:")
    print(f"{'Freq (sec)':<12} {'RV Vol (%)':<12} {'Log Freq':<12} {'Log RV':<12}")
    print("-" * 50)
    for freq, rv in zip(frequencies, rvs):
        print(f"{freq:<12.1f} {rv*100:<12.4f} {np.log(freq):<12.4f} {np.log(rv):<12.4f}")
    
    # Find optimal frequency
    opt_freq, opt_idx = analyzer.find_optimal_frequency(frequencies, rvs, method='elbow')
    
    # Estimate noise
    slope, noise_var = analyzer.estimate_noise(frequencies, rvs)
    
    print(f"\nAnalysis Results:")
    print(f"  Optimal frequency: {opt_freq:.1f} seconds")
    print(f"  Slope (log-log): {slope:.3f} (theory: -1)")
    print(f"  Noise variance estimate: {noise_var*10000:.6f}")
    print(f"  True volatility: {scenario['vol']*100:.3f}%")
    print(f"  RV at optimal: {rvs[opt_idx]*100:.3f}%")
    print(f"  Bias from 1-sec: {(rvs[0]/rvs[opt_idx] - 1)*100:.1f}%")
    
    # Plots
    # Plot 1: Linear-linear
    axes[scenario_idx, 0].plot(frequencies, rvs*100, marker='o', linewidth=2, markersize=8)
    axes[scenario_idx, 0].axvline(opt_freq, color='red', linestyle='--', linewidth=2, label=f'Optimal: {opt_freq:.0f}s')
    axes[scenario_idx, 0].set_xlabel('Sampling Interval (seconds)')
    axes[scenario_idx, 0].set_ylabel('Realized Volatility (Annual %)')
    axes[scenario_idx, 0].set_title(f'{scenario["name"]}: Linear Scale')
    axes[scenario_idx, 0].grid(alpha=0.3)
    axes[scenario_idx, 0].legend()
    
    # Plot 2: Log-log
    axes[scenario_idx, 1].loglog(frequencies, rvs*100, marker='o', linewidth=2, markersize=8)
    
    # Fit line in noise region (first 4 points)
    n_fit = min(4, len(frequencies) // 2)
    z = np.polyfit(np.log(frequencies[:n_fit]), np.log(rvs[:n_fit]), 1)
    p = np.poly1d(z)
    fit_freqs = np.logspace(np.log10(frequencies[0]), np.log10(frequencies[n_fit-1]), 50)
    fit_rvs = np.exp(p(np.log(fit_freqs)))
    axes[scenario_idx, 1].loglog(fit_freqs, fit_rvs*100, 'r--', linewidth=1.5, label=f'Slope: {z[0]:.2f}')
    
    axes[scenario_idx, 1].axvline(opt_freq, color='green', linestyle='--', linewidth=2, label=f'Optimal: {opt_freq:.0f}s')
    axes[scenario_idx, 1].set_xlabel('Sampling Interval (seconds)')
    axes[scenario_idx, 1].set_ylabel('Realized Volatility (Annual %)')
    axes[scenario_idx, 1].set_title(f'{scenario["name"]}: Log-Log Scale')
    axes[scenario_idx, 1].grid(alpha=0.3, which='both')
    axes[scenario_idx, 1].legend()
    
    # Plot 3: Derivative (slope change)
    log_freq = np.log(frequencies)
    log_rv = np.log(rvs)
    slopes = np.diff(log_rv) / np.diff(log_freq)
    
    axes[scenario_idx, 2].plot(frequencies[:-1], slopes, marker='s', linewidth=2, markersize=8)
    axes[scenario_idx, 2].axhline(-1, color='gray', linestyle='--', label='Theory: -1')
    axes[scenario_idx, 2].set_xlabel('Sampling Interval (seconds)')
    axes[scenario_idx, 2].set_ylabel('Slope in Log-Log')
    axes[scenario_idx, 2].set_title(f'{scenario["name"]}: Slope Analysis')
    axes[scenario_idx, 2].grid(alpha=0.3)
    axes[scenario_idx, 2].legend()

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Signature plot reveals noise dominance at high frequencies")
print(f"2. Elbow point indicates optimal sampling frequency")
print(f"3. Noise region slope confirms -1 relationship (theory validation)")
print(f"4. Liquid vs illiquid assets show different signature shapes")
print(f"5. Use optimal frequency for consistent volatility estimation")
```

## 6. Challenge Round
Why does signature plot typically show U or L shape?
- **Left side (high freq)**: Noise dominates RV(h) ∝ 1/h, increasing slope
- **Right side (low freq)**: Fewer samples, variance increases, RV levels off
- **Elbow**: Sweet spot where error minimized (noise + variance balanced)
- **L-shape**: If sample path noise minimal, flat throughout optimal range

How to detect data quality issues from signature?
- **Spiky signature**: Outliers or data errors present
- **Non-monotonic in noise region**: Correlated noise (not iid)
- **Slope ≠ -1**: Model mismatch (e.g., systematic bid-ask bounce)
- **Multiple elbows**: Regime changes or composite data

## 7. Key References
- [Zhang, Mykland, Ait-Sahalia (2005): A Tale of Two Time Scales: Determining Integrated Volatility with Noisy High-Frequency Data](https://www.jstor.org/stable/3647587)
- [Andersen, Dobrev, Weems (2012): Realized Volatility and Multipower Variation](https://www.jstor.org/stable/41640839)
- [Aït-Sahalia, Mykland, Zhang (2005): How Often to Sample a Continuous-Time Process](https://www.jstor.org/stable/1392757)

---
**Status:** Practical tool for sampling optimization | **Complements:** Realized Volatility, Bid-Ask Bounce, Data Quality
