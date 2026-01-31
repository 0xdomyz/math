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
