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
