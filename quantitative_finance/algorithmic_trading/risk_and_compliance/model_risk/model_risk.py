import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate synthetic stock returns (momentum + noise)
np.random.seed(42)
n_days = 2000
true_momentum_lookback = 50  # "True" parameter (unknown to optimization)

returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
prices = 100 * np.cumprod(1 + returns)

# Add momentum effect (autocorrelation)
for i in range(true_momentum_lookback, n_days):
    momentum_signal = np.mean(returns[i-true_momentum_lookback:i])
    returns[i] += 0.3 * momentum_signal  # Momentum boosts next-day return

prices = 100 * np.cumprod(1 + returns)

# Strategy: Momentum (buy if past N-day return > 0, else hold cash)
def momentum_strategy(returns, lookback):
    signals = np.zeros(len(returns))
    for i in range(lookback, len(returns)):
        past_return = np.sum(returns[i-lookback:i])
        signals[i] = 1 if past_return > 0 else 0  # 1 = long, 0 = cash
    strategy_returns = signals * returns
    return strategy_returns

# Objective: Maximize Sharpe ratio
def objective(lookback, returns_in_sample):
    lookback = int(lookback[0])
    if lookback < 5 or lookback > 200:
        return 1e6  # Penalty for invalid lookback
    strat_returns = momentum_strategy(returns_in_sample, lookback)
    sharpe = np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252)
    return -sharpe  # Minimize negative Sharpe

# Scenario 1: In-Sample Optimization (OVERFIT)
in_sample_period = returns[:1500]
result = minimize(objective, x0=[50], args=(in_sample_period,), bounds=[(5, 200)], method='L-BFGS-B')
optimal_lookback_overfit = int(result.x[0])
print("=" * 80)
print("SCENARIO 1: IN-SAMPLE OPTIMIZATION (OVERFITTING RISK)")
print("=" * 80)
print(f"Optimal Lookback (In-Sample Fit):   {optimal_lookback_overfit} days")
print(f"True Parameter (Data Generating):    {true_momentum_lookback} days")
print()

# Test on out-of-sample period
out_of_sample_period = returns[1500:]
overfit_returns_IS = momentum_strategy(in_sample_period, optimal_lookback_overfit)
overfit_returns_OOS = momentum_strategy(out_of_sample_period, optimal_lookback_overfit)

sharpe_overfit_IS = np.mean(overfit_returns_IS) / np.std(overfit_returns_IS) * np.sqrt(252)
sharpe_overfit_OOS = np.mean(overfit_returns_OOS) / np.std(overfit_returns_OOS) * np.sqrt(252)

print(f"In-Sample Sharpe Ratio:              {sharpe_overfit_IS:.2f}")
print(f"Out-of-Sample Sharpe Ratio:          {sharpe_overfit_OOS:.2f}")
print(f"Performance Degradation:             {(sharpe_overfit_IS - sharpe_overfit_OOS) / sharpe_overfit_IS * 100:.1f}%")
print()

# Scenario 2: Walk-Forward Analysis (ROBUST)
print("=" * 80)
print("SCENARIO 2: WALK-FORWARD ANALYSIS (ROBUST VALIDATION)")
print("=" * 80)

walk_forward_window = 500  # Optimize on 500 days, test on next 100
test_window = 100
walk_forward_returns = []
walk_forward_lookbacks = []

for start in range(0, len(returns) - walk_forward_window - test_window, test_window):
    train_data = returns[start:start + walk_forward_window]
    test_data = returns[start + walk_forward_window:start + walk_forward_window + test_window]
    
    # Optimize on training window
    result = minimize(objective, x0=[50], args=(train_data,), bounds=[(5, 200)], method='L-BFGS-B')
    optimal_lookback_WF = int(result.x[0])
    walk_forward_lookbacks.append(optimal_lookback_WF)
    
    # Test on next period
    strat_returns_test = momentum_strategy(test_data, optimal_lookback_WF)
    walk_forward_returns.extend(strat_returns_test)

walk_forward_returns = np.array(walk_forward_returns)
sharpe_WF = np.mean(walk_forward_returns) / np.std(walk_forward_returns) * np.sqrt(252)

print(f"Walk-Forward Sharpe Ratio:           {sharpe_WF:.2f}")
print(f"Average Optimal Lookback (WF):       {np.mean(walk_forward_lookbacks):.0f} days (std: {np.std(walk_forward_lookbacks):.0f})")
print(f"Parameter Stability:                 {'STABLE' if np.std(walk_forward_lookbacks) < 20 else 'UNSTABLE'}")
print()

# Comparison
print("=" * 80)
print("COMPARISON: OVERFITTING vs. WALK-FORWARD")
print("=" * 80)
print(f"{'Method':<30} {'In-Sample Sharpe':<20} {'Out-of-Sample Sharpe':<20} {'Degradation':<15}")
print("-" * 80)
print(f"{'In-Sample Optimization':<30} {sharpe_overfit_IS:<20.2f} {sharpe_overfit_OOS:<20.2f} {(sharpe_overfit_IS - sharpe_overfit_OOS):<15.2f}")
print(f"{'Walk-Forward Analysis':<30} {'N/A':<20} {sharpe_WF:<20.2f} {'Adaptive':<15}")
print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Cumulative returns
cum_overfit_OOS = np.cumprod(1 + overfit_returns_OOS) - 1
cum_WF = np.cumprod(1 + walk_forward_returns) - 1
cum_buyhold = np.cumprod(1 + returns[1500:1500+len(cum_overfit_OOS)]) - 1

axes[0].plot(cum_buyhold * 100, label='Buy & Hold', linewidth=1.5, alpha=0.7)
axes[0].plot(cum_overfit_OOS * 100, label=f'Overfit Model (Lookback={optimal_lookback_overfit})', linewidth=1.5)
axes[0].plot(cum_WF[:len(cum_overfit_OOS)] * 100, label='Walk-Forward Model', linewidth=1.5)
axes[0].set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Model Risk: Overfitting vs. Walk-Forward Validation', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)

# Parameter stability
axes[1].plot(walk_forward_lookbacks, marker='o', linestyle='-', linewidth=1.5, markersize=5)
axes[1].axhline(true_momentum_lookback, color='red', linestyle='--', linewidth=2, label=f'True Parameter ({true_momentum_lookback})')
axes[1].axhline(optimal_lookback_overfit, color='orange', linestyle='--', linewidth=2, label=f'Overfit Parameter ({optimal_lookback_overfit})')
axes[1].set_ylabel('Optimal Lookback (days)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Walk-Forward Window', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_risk_overfitting.png', dpi=150)
plt.show()