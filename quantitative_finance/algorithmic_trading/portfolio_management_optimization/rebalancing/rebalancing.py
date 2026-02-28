# %% [markdown]
# # rebalancing
#
# This interactive script follows the quantitative finance topic template.

# %% [markdown]
# ## Section 1 - Overview & Setup
# Define imports, reproducibility settings, and runtime configuration.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
print("Setup complete for topic: rebalancing")

# %% [markdown]
# ## Section 2 - Data Generation
# Generate or load data used by the modeling workflow.

# %%
if 'df' not in globals():
    t = np.arange(500)
    base = np.sin(t / 20.0)
    noise = np.random.normal(0, 0.2, size=len(t))
    df = pd.DataFrame({'t': t, 'signal': base + noise})
print("Data shape:", df.shape)
print(df.head(3))

# %% [markdown]
# ## Section 3 - Model Implementation
# Core topic logic and implementation details.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate asset returns (stocks and bonds, 5 years daily)
np.random.seed(88)
n_days = 252 * 5
stock_returns = np.random.normal(0.0008, 0.015, n_days)  # 20% annual, 15% vol
bond_returns = np.random.normal(0.0003, 0.005, n_days)   # 7.5% annual, 5% vol

# Initial allocation: 60/40
initial_value = 100000
stock_value = initial_value * 0.60
bond_value = initial_value * 0.40
target_stock_weight = 0.60

# Storage for results
results = {
    'date': [],
    'no_rebalance_stock_weight': [],
    'calendar_stock_weight': [],
    'threshold_stock_weight': [],
}

# Tracking variables
no_reb_stock = stock_value
no_reb_bond = bond_value

cal_stock = stock_value
cal_bond = bond_value

thresh_stock = stock_value
thresh_bond = bond_value

threshold = 0.05  # 5% deviation triggers rebalance
rebalance_interval = 63  # Quarterly (252/4)
transaction_cost_bps = 10  # 10 bps per side
day_counter = 0

for day in range(n_days):
    # Update values based on returns
    no_reb_stock *= (1 + stock_returns[day])
    no_reb_bond *= (1 + bond_returns[day])
    
    cal_stock *= (1 + stock_returns[day])
    cal_bond *= (1 + bond_returns[day])
    
    thresh_stock *= (1 + stock_returns[day])
    thresh_bond *= (1 + bond_returns[day])
    
    # Calendar rebalancing (quarterly)
    if day_counter % rebalance_interval == 0 and day > 0:
        cal_total = cal_stock + cal_bond
        cal_target_stock = cal_total * target_stock_weight
        cal_target_bond = cal_total * (1 - target_stock_weight)
        
        # Transaction costs
        turnover = abs(cal_stock - cal_target_stock)
        cost = turnover * (transaction_cost_bps / 10000)
        
        cal_stock = cal_target_stock
        cal_bond = cal_target_bond - cost  # Deduct from bonds
    
    # Threshold rebalancing (check daily)
    thresh_total = thresh_stock + thresh_bond
    thresh_current_weight = thresh_stock / thresh_total
    
    if abs(thresh_current_weight - target_stock_weight) > threshold:
        thresh_target_stock = thresh_total * target_stock_weight
        thresh_target_bond = thresh_total * (1 - target_stock_weight)
        
        # Transaction costs
        turnover = abs(thresh_stock - thresh_target_stock)
        cost = turnover * (transaction_cost_bps / 10000)
        
        thresh_stock = thresh_target_stock
        thresh_bond = thresh_target_bond - cost
    
    # Record weights
    if day % 21 == 0:  # Monthly snapshots
        results['date'].append(day)
        results['no_rebalance_stock_weight'].append(no_reb_stock / (no_reb_stock + no_reb_bond))
        results['calendar_stock_weight'].append(cal_stock / (cal_stock + cal_bond))
        results['threshold_stock_weight'].append(thresh_stock / (thresh_stock + thresh_bond))
    
    day_counter += 1

# Final values
no_reb_total = no_reb_stock + no_reb_bond
cal_total = cal_stock + cal_bond
thresh_total = thresh_stock + thresh_bond

print("=" * 60)
print("REBALANCING STRATEGY COMPARISON (5 Years)")
print("=" * 60)
print(f"Initial Portfolio Value:     ${initial_value:>12,.2f}")
print(f"Target Stock Weight:         {target_stock_weight:>12.0%}")
print(f"Rebalancing Threshold:       {threshold:>12.0%}")
print(f"Transaction Cost:            {transaction_cost_bps:>12} bps")
print()
print(f"No Rebalancing:")
print(f"  Final Value:               ${no_reb_total:>12,.2f}")
print(f"  Final Stock Weight:        {no_reb_stock/no_reb_total:>12.1%}")
print(f"  Total Return:              {(no_reb_total/initial_value - 1)*100:>11.2f}%")
print()
print(f"Calendar Rebalancing (Quarterly):")
print(f"  Final Value:               ${cal_total:>12,.2f}")
print(f"  Final Stock Weight:        {cal_stock/cal_total:>12.1%}")
print(f"  Total Return:              {(cal_total/initial_value - 1)*100:>11.2f}%")
print()
print(f"Threshold Rebalancing (Â±5%):")
print(f"  Final Value:               ${thresh_total:>12,.2f}")
print(f"  Final Stock Weight:        {thresh_stock/thresh_total:>12.1%}")
print(f"  Total Return:              {(thresh_total/initial_value - 1)*100:>11.2f}%")
print("=" * 60)

# Plot weight drift over time
df = pd.DataFrame(results)
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['no_rebalance_stock_weight'], label='No Rebalancing', linewidth=2)
plt.plot(df['date'], df['calendar_stock_weight'], label='Calendar (Quarterly)', linewidth=2, linestyle='--')
plt.plot(df['date'], df['threshold_stock_weight'], label='Threshold (Â±5%)', linewidth=2, linestyle=':')
plt.axhline(target_stock_weight, color='red', linestyle='-', linewidth=1, label='Target (60%)')
plt.axhline(target_stock_weight + threshold, color='gray', linestyle=':', alpha=0.5)
plt.axhline(target_stock_weight - threshold, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Trading Days')
plt.ylabel('Stock Weight')
plt.title('Portfolio Weight Drift: Rebalancing Strategy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 4 - Training & Evaluation
# Compute basic evaluation metrics for demonstration.

# %%
if 'df' in globals() and 'signal' in df.columns:
    eval_mean = float(df['signal'].mean())
    eval_std = float(df['signal'].std())
    print(f"Evaluation summary -> mean: {eval_mean:.4f}, std: {eval_std:.4f}")
else:
    print("Evaluation note: custom topic code executed; add topic-specific metrics as needed.")

# %% [markdown]
# ## Section 5 - Visualization & Interpretation
# Visualize outputs to support interpretation.

# %%
if 'df' in globals() and 'signal' in df.columns:
    ax = df.plot(x='t', y='signal', figsize=(10, 4), title='rebalancing - Signal View')
    ax.set_ylabel('signal')
    plt.tight_layout()
    plt.show()
else:
    print("Visualization note: add topic-specific plots from model outputs.")

# %% [markdown]
# ## Section 6 - Summary & Deployment
#
# Key takeaways:
# - End-to-end workflow scaffold is in place.
# - Replace placeholder evaluation/visualization with production metrics and controls.
# - Validate assumptions under realistic transaction costs, latency, and liquidity constraints.
