import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Simulate market with informed and uninformed traders
n_seconds = 7200  # 2 hours
tick_size = 0.01

# Fundamental value with information events
fundamental_value = 100 * np.ones(n_seconds)
information_events = np.random.poisson(0.002, n_seconds)  # ~14 events
cumulative_info = np.cumsum(information_events * np.random.normal(0, 0.1, n_seconds))
fundamental_value += cumulative_info

# Observable midpoint follows fundamental with lag
midpoint = np.zeros(n_seconds)
midpoint[0] = fundamental_value[0]
adjustment_speed = 0.3  # Speed of price discovery

for t in range(1, n_seconds):
    # Midpoint adjusts toward fundamental value with noise
    midpoint[t] = midpoint[t-1] + adjustment_speed * (fundamental_value[t] - midpoint[t-1]) + \
                  np.random.normal(0, 0.01)

# Bid-ask spread
half_spread = 0.04 + 0.5 * np.abs(np.diff(fundamental_value, prepend=fundamental_value[0]))
bid_prices = midpoint - half_spread
ask_prices = midpoint + half_spread

# Generate trades with different trader types
n_trades = 1000
trade_times = np.sort(np.random.choice(n_seconds - 300, n_trades, replace=False))

# Informed traders trade in direction of fundamental value
prob_informed = 0.25  # 25% of trades are informed
is_informed = np.random.random(n_trades) < prob_informed

trade_directions = np.zeros(n_trades, dtype=int)
for i, t in enumerate(trade_times):
    if is_informed[i]:
        # Informed traders know fundamental value exceeds/falls short of midpoint
        if fundamental_value[t] > midpoint[t]:
            trade_directions[i] = 1  # Buy
        else:
            trade_directions[i] = -1  # Sell
    else:
        # Uninformed traders random direction (liquidity motivated)
        trade_directions[i] = np.random.choice([-1, 1])

# Execute trades
execution_prices = np.where(trade_directions == 1, 
                           ask_prices[trade_times], 
                           bid_prices[trade_times])

midpoint_at_trade = midpoint[trade_times]

# Calculate spreads at multiple time horizons
horizons = [10, 30, 100, 300]  # seconds
horizon_labels = ['10s', '30s', '100s', '5min']

results = {}
for horizon, label in zip(horizons, horizon_labels):
    midpoint_future = midpoint[trade_times + horizon]
    
    # Effective spread
    effective_spreads = 2 * np.abs(execution_prices - midpoint_at_trade)
    
    # Price impact
    price_changes = midpoint_future - midpoint_at_trade
    price_impacts = 2 * price_changes * trade_directions
    
    # Realized spread
    realized_spreads = effective_spreads - price_impacts
    
    results[label] = {
        'effective': effective_spreads,
        'impact': price_impacts,
        'realized': realized_spreads
    }

# Separate by trader type
informed_mask = is_informed
uninformed_mask = ~is_informed

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Spread decomposition by trader type
labels = ['Informed', 'Uninformed']
masks = [informed_mask, uninformed_mask]
colors = ['red', 'blue']

x_pos = np.arange(len(labels))
width = 0.25

for i, (label, mask) in enumerate(zip(labels, masks)):
    effective = results['5min']['effective'][mask].mean()
    impact = results['5min']['impact'][mask].mean()
    realized = results['5min']['realized'][mask].mean()
    
    axes[0, 0].bar(i - width, effective, width, label='Effective' if i==0 else '', 
                   color='lightgray', edgecolor='black')
    axes[0, 0].bar(i, impact, width, label='Price Impact' if i==0 else '', 
                   color='orange', edgecolor='black')
    axes[0, 0].bar(i + width, realized, width, label='Realized' if i==0 else '', 
                   color='lightgreen', edgecolor='black')

axes[0, 0].set_ylabel('Spread ($)')
axes[0, 0].set_title('Spread Components by Trader Type (5min horizon)')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].legend()
axes[0, 0].axhline(0, color='black', linewidth=0.5)
axes[0, 0].grid(alpha=0.3, axis='y')

print("Spread Decomposition by Trader Type (5-minute horizon):")
print(f"\nInformed Traders:")
print(f"  Effective Spread: ${results['5min']['effective'][informed_mask].mean():.4f}")
print(f"  Price Impact: ${results['5min']['impact'][informed_mask].mean():.4f}")
print(f"  Realized Spread: ${results['5min']['realized'][informed_mask].mean():.4f}")
print(f"\nUninformed Traders:")
print(f"  Effective Spread: ${results['5min']['effective'][uninformed_mask].mean():.4f}")
print(f"  Price Impact: ${results['5min']['impact'][uninformed_mask].mean():.4f}")
print(f"  Realized Spread: ${results['5min']['realized'][uninformed_mask].mean():.4f}")

# Plot 2: Realized spread vs price impact scatter
axes[0, 1].scatter(results['5min']['impact'][informed_mask], 
                   results['5min']['realized'][informed_mask],
                   alpha=0.4, s=30, c='red', label='Informed')
axes[0, 1].scatter(results['5min']['impact'][uninformed_mask], 
                   results['5min']['realized'][uninformed_mask],
                   alpha=0.4, s=30, c='blue', label='Uninformed')

# 45-degree line (effective spread constraint)
max_val = max(results['5min']['impact'].max(), results['5min']['realized'].max())
axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='RS = PI')

axes[0, 1].axhline(0, color='black', linewidth=0.5)
axes[0, 1].axvline(0, color='black', linewidth=0.5)
axes[0, 1].set_xlabel('Price Impact ($)')
axes[0, 1].set_ylabel('Realized Spread ($)')
axes[0, 1].set_title('Realized Spread vs Price Impact')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Time horizon analysis
horizon_means_informed = []
horizon_means_uninformed = []
horizon_stds_informed = []
horizon_stds_uninformed = []

for label in horizon_labels:
    horizon_means_informed.append(results[label]['realized'][informed_mask].mean())
    horizon_means_uninformed.append(results[label]['realized'][uninformed_mask].mean())
    horizon_stds_informed.append(results[label]['realized'][informed_mask].std())
    horizon_stds_uninformed.append(results[label]['realized'][uninformed_mask].std())

x_pos = np.arange(len(horizon_labels))
axes[1, 0].plot(x_pos, horizon_means_informed, 'o-', color='red', 
                linewidth=2, markersize=8, label='Informed')
axes[1, 0].plot(x_pos, horizon_means_uninformed, 's-', color='blue', 
                linewidth=2, markersize=8, label='Uninformed')

axes[1, 0].fill_between(x_pos,
                        np.array(horizon_means_informed) - np.array(horizon_stds_informed),
                        np.array(horizon_means_informed) + np.array(horizon_stds_informed),
                        alpha=0.2, color='red')
axes[1, 0].fill_between(x_pos,
                        np.array(horizon_means_uninformed) - np.array(horizon_stds_uninformed),
                        np.array(horizon_means_uninformed) + np.array(horizon_stds_uninformed),
                        alpha=0.2, color='blue')

axes[1, 0].axhline(0, color='black', linewidth=0.5, linestyle='--')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(horizon_labels)
axes[1, 0].set_xlabel('Time Horizon')
axes[1, 0].set_ylabel('Realized Spread ($)')
axes[1, 0].set_title('Realized Spread vs Time Horizon')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

print(f"\nRealized Spread by Time Horizon:")
for label in horizon_labels:
    print(f"{label}: Informed=${results[label]['realized'][informed_mask].mean():.4f}, "
          f"Uninformed=${results[label]['realized'][uninformed_mask].mean():.4f}")

# Plot 4: Cumulative dealer PnL over time
cumulative_pnl = np.cumsum(results['5min']['realized'])
cumulative_pnl_informed = np.cumsum(results['5min']['realized'] * informed_mask)
cumulative_pnl_uninformed = np.cumsum(results['5min']['realized'] * uninformed_mask)

axes[1, 1].plot(cumulative_pnl, linewidth=2, label='Total PnL')
axes[1, 1].plot(cumulative_pnl_informed, linewidth=1.5, alpha=0.7, 
                label='From Informed', color='red')
axes[1, 1].plot(cumulative_pnl_uninformed, linewidth=1.5, alpha=0.7, 
                label='From Uninformed', color='blue')

axes[1, 1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axes[1, 1].set_xlabel('Trade Number')
axes[1, 1].set_ylabel('Cumulative Realized Spread ($)')
axes[1, 1].set_title('Market Maker Cumulative PnL')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

final_pnl = cumulative_pnl[-1]
pnl_from_informed = cumulative_pnl_informed[-1]
pnl_from_uninformed = cumulative_pnl_uninformed[-1]

print(f"\nCumulative Market Maker PnL:")
print(f"Total: ${final_pnl:.2f}")
print(f"From Informed Traders: ${pnl_from_informed:.2f} ({pnl_from_informed/final_pnl*100:.1f}%)")
print(f"From Uninformed Traders: ${pnl_from_uninformed:.2f} ({pnl_from_uninformed/final_pnl*100:.1f}%)")

plt.tight_layout()
plt.show()

# Adverse selection analysis
adverse_selection_cost = results['5min']['impact'].mean()
avg_effective = results['5min']['effective'].mean()
avg_realized = results['5min']['realized'].mean()

print(f"\nAdverse Selection Analysis:")
print(f"Average Effective Spread: ${avg_effective:.4f}")
print(f"Average Price Impact (Adverse Selection): ${adverse_selection_cost:.4f}")
print(f"Average Realized Spread (Dealer Revenue): ${avg_realized:.4f}")
print(f"Adverse Selection %: {(adverse_selection_cost/avg_effective)*100:.1f}%")
print(f"Realized Spread %: {(avg_realized/avg_effective)*100:.1f}%")

# Test if realized spread is significantly different from zero
t_stat, p_value = stats.ttest_1samp(results['5min']['realized'], 0)
print(f"\nStatistical Test (H0: Realized Spread = 0):")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.01:
    print("Reject null: Market makers earn significant positive spreads")
else:
    print("Cannot reject null: No significant dealer profit")
