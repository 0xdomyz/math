import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Glosten-Milgrom model simulation
n_periods = 1000

# True value follows random walk with occasional information events
true_value = np.zeros(n_periods)
true_value[0] = 100

# Information events: private signal arrives
info_event_prob = 0.02
info_events = np.random.random(n_periods) < info_event_prob
info_innovations = np.where(info_events, np.random.normal(0, 1, n_periods), 0)

# Build true value path
for t in range(1, n_periods):
    true_value[t] = true_value[t-1] + info_innovations[t] + np.random.normal(0, 0.05)

# Market maker's belief (lags true value)
belief = np.zeros(n_periods)
belief[0] = true_value[0]

# Trader types
prob_informed = 0.3  # 30% of trades are informed
prob_trade = 0.5  # 50% probability of trade in any period

# Generate trades
trades = np.random.random(n_periods) < prob_trade
is_informed = np.random.random(n_periods) < prob_informed

trade_directions = np.zeros(n_periods)
execution_prices = np.zeros(n_periods)
spreads = np.zeros(n_periods)
adverse_selection_losses = np.zeros(n_periods)

# Initial spread (base + adverse selection premium)
base_spread = 0.1  # Order processing + inventory
current_spread = base_spread

for t in range(n_periods):
    if not trades[t]:
        # No trade this period
        belief[t] = belief[t-1] if t > 0 else belief[0]
        spreads[t] = current_spread
        continue
    
    # Market maker posts bid-ask around belief
    bid = belief[t-1] - current_spread / 2 if t > 0 else belief[0] - current_spread / 2
    ask = belief[t-1] + current_spread / 2 if t > 0 else belief[0] + current_spread / 2
    spreads[t] = current_spread
    
    # Determine trade direction
    if is_informed[t]:
        # Informed traders know true value
        if true_value[t] > ask:
            trade_directions[t] = 1  # Buy (value exceeds ask)
            execution_prices[t] = ask
        elif true_value[t] < bid:
            trade_directions[t] = -1  # Sell (value below bid)
            execution_prices[t] = bid
        else:
            # No profitable trade
            trade_directions[t] = 0
            trades[t] = False
            belief[t] = belief[t-1]
            continue
    else:
        # Uninformed traders random direction (liquidity)
        trade_directions[t] = np.random.choice([-1, 1])
        if trade_directions[t] == 1:
            execution_prices[t] = ask
        else:
            execution_prices[t] = bid
    
    # Calculate adverse selection loss for dealer
    # Dealer sells at ask when true value is higher (informed buy)
    # Dealer buys at bid when true value is lower (informed sell)
    if trade_directions[t] == 1:
        # Dealer sold at ask
        adverse_selection_losses[t] = max(0, true_value[t] - ask)
    else:
        # Dealer bought at bid
        adverse_selection_losses[t] = max(0, bid - true_value[t])
    
    # Bayesian updating of belief
    if t < n_periods - 1:
        # Market maker learns from trade direction
        if trade_directions[t] == 1:
            # Buy order suggests positive information
            belief[t] = belief[t-1] + 0.1 * current_spread
        elif trade_directions[t] == -1:
            # Sell order suggests negative information
            belief[t] = belief[t-1] - 0.1 * current_spread
        
        # Adjust spread based on recent adverse selection
        recent_window = 20
        if t >= recent_window:
            recent_losses = adverse_selection_losses[max(0, t-recent_window):t].mean()
            # Widen spread if experiencing adverse selection
            current_spread = base_spread + 5 * recent_losses
            current_spread = min(current_spread, 2.0)  # Cap at reasonable level

# Calculate PIN (Probability of Informed Trading) - Easley et al
buys = (trade_directions == 1) & trades
sells = (trade_directions == -1) & trades
buy_count = buys.sum()
sell_count = sells.sum()

# Simple PIN approximation
order_imbalance = np.abs(buy_count - sell_count)
total_trades = buy_count + sell_count
PIN_estimate = order_imbalance / total_trades if total_trades > 0 else 0

# Measure price impact (permanent component)
price_impacts = np.zeros(n_periods)
horizon = 10  # Look ahead window

for t in range(n_periods - horizon):
    if trades[t] and trade_directions[t] != 0:
        initial_mid = belief[t]
        future_mid = belief[t + horizon]
        price_impacts[t] = (future_mid - initial_mid) * trade_directions[t]

# Separate informed vs uninformed trades
informed_trades = trades & is_informed
uninformed_trades = trades & ~is_informed

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price dynamics with trades
sample_period = slice(200, 400)
time_axis = np.arange(200, 400)

axes[0, 0].plot(time_axis, true_value[sample_period], 'g-', linewidth=2, label='True Value')
axes[0, 0].plot(time_axis, belief[sample_period], 'b--', linewidth=2, label='Market Maker Belief')

# Mark trades
informed_buys = informed_trades[sample_period] & (trade_directions[sample_period] == 1)
informed_sells = informed_trades[sample_period] & (trade_directions[sample_period] == -1)
uninformed_buys = uninformed_trades[sample_period] & (trade_directions[sample_period] == 1)
uninformed_sells = uninformed_trades[sample_period] & (trade_directions[sample_period] == -1)

axes[0, 0].scatter(time_axis[informed_buys], execution_prices[sample_period][informed_buys],
                   c='darkred', marker='^', s=80, label='Informed Buy', zorder=5)
axes[0, 0].scatter(time_axis[informed_sells], execution_prices[sample_period][informed_sells],
                   c='darkblue', marker='v', s=80, label='Informed Sell', zorder=5)
axes[0, 0].scatter(time_axis[uninformed_buys], execution_prices[sample_period][uninformed_buys],
                   c='lightcoral', marker='^', s=40, alpha=0.5, label='Uninformed Buy')
axes[0, 0].scatter(time_axis[uninformed_sells], execution_prices[sample_period][uninformed_sells],
                   c='lightblue', marker='v', s=40, alpha=0.5, label='Uninformed Sell')

axes[0, 0].set_title('Price Dynamics and Trade Flow')
axes[0, 0].set_xlabel('Time Period')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend(loc='best', fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Spread dynamics
axes[0, 1].plot(spreads, linewidth=1)
axes[0, 1].axhline(base_spread, color='r', linestyle='--', label=f'Base Spread=${base_spread:.2f}')
axes[0, 1].fill_between(range(n_periods), base_spread, spreads, 
                        alpha=0.3, label='Adverse Selection Component')
axes[0, 1].set_title('Dynamic Spread Adjustment')
axes[0, 1].set_xlabel('Time Period')
axes[0, 1].set_ylabel('Bid-Ask Spread ($)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

print("Market Statistics:")
print(f"Total Trades: {trades.sum()}")
print(f"Informed Trades: {informed_trades.sum()} ({informed_trades.sum()/trades.sum()*100:.1f}%)")
print(f"PIN Estimate: {PIN_estimate:.3f}")
print(f"Mean Spread: ${spreads.mean():.4f}")
print(f"Max Spread: ${spreads.max():.4f}")

# Plot 3: Adverse selection losses distribution
losses_informed = adverse_selection_losses[informed_trades]
losses_uninformed = adverse_selection_losses[uninformed_trades]

axes[1, 0].hist(losses_informed[losses_informed > 0], bins=30, alpha=0.6, 
               label=f'Informed (mean=${losses_informed.mean():.4f})', 
               color='red', density=True)
axes[1, 0].hist(losses_uninformed[losses_uninformed > 0], bins=30, alpha=0.6,
               label=f'Uninformed (mean=${losses_uninformed.mean():.4f})',
               color='blue', density=True)
axes[1, 0].set_title('Adverse Selection Loss Distribution')
axes[1, 0].set_xlabel('Loss per Trade ($)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

print(f"\nAdverse Selection Losses:")
print(f"Per Informed Trade: ${losses_informed.mean():.4f}")
print(f"Per Uninformed Trade: ${losses_uninformed.mean():.4f}")
print(f"Total Dealer Loss: ${adverse_selection_losses.sum():.2f}")

# Plot 4: Cumulative dealer PnL
# Revenue from spread
spread_revenue = np.where(trades, spreads / 2, 0)  # Dealer earns half-spread per trade

# Net PnL = spread revenue - adverse selection losses
net_pnl = spread_revenue - adverse_selection_losses
cumulative_pnl = np.cumsum(net_pnl)
cumulative_revenue = np.cumsum(spread_revenue)
cumulative_losses = np.cumsum(adverse_selection_losses)

axes[1, 1].plot(cumulative_revenue, linewidth=2, label='Gross Revenue (Spreads)', color='green')
axes[1, 1].plot(cumulative_losses, linewidth=2, label='Adverse Selection Losses', color='red')
axes[1, 1].plot(cumulative_pnl, linewidth=2.5, label='Net PnL', color='blue')
axes[1, 1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axes[1, 1].set_title('Market Maker Cumulative PnL')
axes[1, 1].set_xlabel('Time Period')
axes[1, 1].set_ylabel('Cumulative $ PnL')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

print(f"\nDealer Economics:")
print(f"Gross Revenue: ${cumulative_revenue[-1]:.2f}")
print(f"Adverse Selection Losses: ${cumulative_losses[-1]:.2f}")
print(f"Net PnL: ${cumulative_pnl[-1]:.2f}")
print(f"Adverse Selection as % of Revenue: {(cumulative_losses[-1]/cumulative_revenue[-1])*100:.1f}%")

plt.tight_layout()
plt.show()

# Statistical tests
print(f"\nStatistical Analysis:")

# Test if price impact differs between informed and uninformed
impact_informed = price_impacts[informed_trades[:-horizon]]
impact_uninformed = price_impacts[uninformed_trades[:-horizon]]

impact_informed_clean = impact_informed[~np.isnan(impact_informed) & (impact_informed != 0)]
impact_uninformed_clean = impact_uninformed[~np.isnan(impact_uninformed) & (impact_uninformed != 0)]

if len(impact_informed_clean) > 0 and len(impact_uninformed_clean) > 0:
    t_stat, p_val = stats.ttest_ind(impact_informed_clean, impact_uninformed_clean)
    print(f"Price Impact - Informed: ${impact_informed_clean.mean():.4f}")
    print(f"Price Impact - Uninformed: ${impact_uninformed_clean.mean():.4f}")
    print(f"t-test p-value: {p_val:.4f}")
    
    if p_val < 0.01:
        print("Reject null: Informed trades have significantly different price impact")

# Measure spread-return correlation (negative correlation indicates adverse selection)
trade_mask = trades & (trade_directions != 0)
if trade_mask.sum() > 10:
    trade_returns = np.diff(execution_prices[trade_mask])
    trade_returns_clean = trade_returns[~np.isnan(trade_returns)]
    
    if len(trade_returns_clean) > 2:
        # Serial correlation in transaction prices (bid-ask bounce)
        autocorr = np.corrcoef(trade_returns_clean[:-1], trade_returns_clean[1:])[0, 1]
        print(f"\nTransaction price autocorrelation: {autocorr:.4f}")
        print("(Negative autocorrelation suggests bid-ask bounce)")

# Calculate effective half-spread decomposition
effective_half_spreads = np.abs(execution_prices - belief) * trades
realized_half_spreads = effective_half_spreads - (price_impacts / 2)

print(f"\nSpread Decomposition (per trade):")
print(f"Effective Half-Spread: ${effective_half_spreads[trades].mean():.4f}")
print(f"Price Impact Component: ${(price_impacts[:-horizon][trades[:-horizon]]/2).mean():.4f}")
print(f"Realized Half-Spread: ${realized_half_spreads[:-horizon][trades[:-horizon]].mean():.4f}")
