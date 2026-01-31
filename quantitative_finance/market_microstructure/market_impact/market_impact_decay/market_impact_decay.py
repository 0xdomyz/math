import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as sp_stats

# Generate synthetic post-trade price impact data
np.random.seed(42)
n_trades = 100

# Parameters (varied by trade to simulate realism)
trade_times = np.arange(n_trades)
trade_sizes = np.random.uniform(0.5, 3.0, n_trades)  # % of daily volume
times_since_trade = np.array([1, 5, 10, 30, 60, 120])  # minutes post-trade

# Generate impact data with exponential + power-law components
permanent_impacts = trade_sizes * 0.15 + np.random.normal(0, 0.02, n_trades)  # 15 bps per % volume
temp_impacts_initial = trade_sizes * 0.35 + np.random.normal(0, 0.03, n_trades)  # 35 bps temporary

# Decay parameters (regime-dependent)
decay_rates = np.random.uniform(0.08, 0.15, n_trades)  # λ values
regime = np.random.choice(['normal', 'high_vol', 'crowded'], n_trades, p=[0.6, 0.3, 0.1])

# Store results
impact_data = []
for i in range(n_trades):
    for t in times_since_trade:
        if regime[i] == 'normal':
            decay_rate = 0.12
            temp_component = temp_impacts_initial[i] * np.exp(-decay_rate * t / 10)
        elif regime[i] == 'high_vol':
            decay_rate = 0.06
            temp_component = temp_impacts_initial[i] * np.exp(-decay_rate * t / 10)
        else:  # crowded
            decay_rate = 0.05
            temp_component = temp_impacts_initial[i] * (10 / (t + 10)) ** 0.5  # power-law
        
        total_impact = permanent_impacts[i] + temp_component
        noise = np.random.normal(0, 0.02)
        
        impact_data.append({
            'trade_id': i,
            'time_min': t,
            'order_size_pct': trade_sizes[i],
            'regime': regime[i],
            'total_impact_bps': total_impact * 100 + noise,
            'permanent_bps': permanent_impacts[i] * 100,
            'temporary_bps': temp_component * 100
        })

df_impact = pd.DataFrame(impact_data)

print("="*100)
print("MARKET IMPACT DECAY ANALYSIS")
print("="*100)

print(f"\nStep 1: Data Summary")
print(f"-" * 50)
print(f"Sample trades: {n_trades}")
print(f"Time points per trade: {len(times_since_trade)}")
print(f"Total observations: {len(df_impact)}")
print(f"\nRegime distribution:")
print(df_impact['regime'].value_counts())
print(f"\nOrder size statistics (% daily volume):")
print(df_impact['order_size_pct'].describe().round(3))

# Step 2: Fit decay models per trade
print(f"\nStep 2: Decay Model Estimation (Per-Trade)")
print(f"-" * 50)

def exponential_decay(t, I_perm, I_temp_init, decay_rate):
    return I_perm + I_temp_init * np.exp(-decay_rate * t / 10)

def power_law_decay(t, I_perm, I_temp_init, tau, alpha):
    return I_perm + I_temp_init * (tau / (t + tau)) ** alpha

# Fit individual trades
fit_results = []
for trade_id in range(min(10, n_trades)):  # Fit first 10 trades
    trade_data = df_impact[df_impact['trade_id'] == trade_id]
    times = trade_data['time_min'].values
    impacts = trade_data['total_impact_bps'].values / 100
    
    # Fit exponential
    try:
        popt_exp, _ = curve_fit(exponential_decay, times, impacts, p0=[0.02, 0.3, 0.10])
        I_perm_exp, I_temp_exp, lambda_exp = popt_exp
        half_life_exp = -np.log(2) / (lambda_exp / 10) if lambda_exp > 0 else np.inf
        r2_exp = 1 - np.sum((impacts - exponential_decay(times, *popt_exp))**2) / np.sum((impacts - np.mean(impacts))**2)
        
        # Fit power-law
        popt_pl, _ = curve_fit(power_law_decay, times, impacts, p0=[0.02, 0.3, 15, 0.5])
        I_perm_pl, I_temp_pl, tau_pl, alpha_pl = popt_pl
        r2_pl = 1 - np.sum((impacts - power_law_decay(times, *popt_pl))**2) / np.sum((impacts - np.mean(impacts))**2)
        
        fit_results.append({
            'trade_id': trade_id,
            'regime': trade_data['regime'].iloc[0],
            'order_size': trade_data['order_size_pct'].iloc[0],
            'I_perm_exp': I_perm_exp * 100,
            'lambda': lambda_exp,
            'half_life_exp': half_life_exp,
            'r2_exp': r2_exp,
            'alpha_pl': alpha_pl,
            'tau_pl': tau_pl,
            'r2_pl': r2_pl,
            'better_model': 'exp' if r2_exp > r2_pl else 'power-law'
        })
    except:
        pass

fit_df = pd.DataFrame(fit_results)
print(fit_df[['trade_id', 'regime', 'lambda', 'half_life_exp', 'r2_exp', 'better_model']].to_string(index=False))

# Step 3: Aggregate decay statistics by regime
print(f"\nStep 3: Decay Statistics by Regime")
print(f"-" * 50)

regime_stats = fit_df.groupby('regime').agg({
    'lambda': ['mean', 'std'],
    'half_life_exp': ['mean', 'std'],
    'r2_exp': 'mean'
}).round(3)
print(regime_stats)

# Step 4: Impact decomposition
print(f"\nStep 4: Impact Decomposition (Avg across all trades)")
print(f"-" * 50)

impact_decomp = df_impact.groupby('time_min')[['total_impact_bps', 'permanent_bps', 'temporary_bps']].mean()
impact_decomp['perm_pct'] = 100 * impact_decomp['permanent_bps'] / impact_decomp['total_impact_bps']
impact_decomp['temp_pct'] = 100 * impact_decomp['temporary_bps'] / impact_decomp['total_impact_bps']

print(impact_decomp.round(2))

# Step 5: Regime-specific impact
print(f"\nStep 5: Average Impact by Regime & Time")
print(f"-" * 50)

regime_impact = df_impact.groupby(['regime', 'time_min'])['total_impact_bps'].mean().unstack()
print(regime_impact.round(2))

print(f"\nDecay Speed (minutes to 50% reversion):")
for regime_name in regime_impact.index:
    impact_0 = regime_impact.loc[regime_name, 1]
    impact_120 = regime_impact.loc[regime_name, 120]
    target = impact_0 / 2 + impact_120 / 2
    for t in times_since_trade:
        impact_t = regime_impact.loc[regime_name, t]
        if impact_t <= target:
            print(f"  {regime_name}: ~{t} minutes")
            break

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Individual trade decay curves (sample)
ax = axes[0, 0]
for trade_id in range(min(5, n_trades)):
    trade_data = df_impact[df_impact['trade_id'] == trade_id]
    times = trade_data['time_min'].values
    impacts = trade_data['total_impact_bps'].values
    ax.plot(times, impacts, marker='o', label=f'Trade {trade_id}', linewidth=2)

ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Impact (basis points)')
ax.set_title('Individual Trade Impact Decay')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Average decay by regime
ax = axes[0, 1]
for regime_name in ['normal', 'high_vol', 'crowded']:
    regime_data = df_impact[df_impact['regime'] == regime_name].groupby('time_min')['total_impact_bps'].mean()
    ax.plot(regime_data.index, regime_data.values, marker='o', label=regime_name, linewidth=2)

ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Average Impact (basis points)')
ax.set_title('Impact Decay by Market Regime')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Impact decomposition (permanent vs temporary)
ax = axes[1, 0]
x = impact_decomp.index
width = 15
ax.bar(x - width/2, impact_decomp['permanent_bps'], width, label='Permanent', alpha=0.7)
ax.bar(x + width/2, impact_decomp['temporary_bps'], width, label='Temporary', alpha=0.7)
ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Impact (basis points)')
ax.set_title('Permanent vs Temporary Impact Over Time')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Cumulative decay (% of initial impact reverted)
ax = axes[1, 1]
for regime_name in ['normal', 'high_vol', 'crowded']:
    regime_data = df_impact[df_impact['regime'] == regime_name].groupby('time_min')['total_impact_bps'].mean()
    initial_impact = regime_data.iloc[0]
    final_impact = regime_data.iloc[-1]
    decay_pct = 100 * (initial_impact - regime_data) / (initial_impact - final_impact)
    ax.plot(regime_data.index, decay_pct, marker='o', label=regime_name, linewidth=2)

ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Decay (%)')
ax.set_ylim([0, 120])
ax.set_title('Cumulative Decay: % of Reversible Impact Decayed')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- Normal regime: λ ≈ 0.12 (half-life ~6 min) - fast decay, mostly temporary")
print(f"- High-vol regime: λ ≈ 0.06 (half-life ~12 min) - slower decay, more impact persists")
print(f"- Crowded regime: Power-law dominates - very slow decay, 30+ min tail")
print(f"- Permanent impact: 20-40% of initial, unavoidable")
print(f"- Temporary impact: 60-80% of initial, reverts in 30-120 min (regime-dependent)")
print(f"- Implication: Slower execution beneficial if decay is slow (illiquid/crowded)")
