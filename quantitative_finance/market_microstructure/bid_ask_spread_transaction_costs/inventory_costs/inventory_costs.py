import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

np.random.seed(42)

# Market parameters
T = 3600  # 1 hour in seconds
dt = 1  # 1 second intervals
n_steps = int(T / dt)

# Price dynamics (Geometric Brownian Motion)
S0 = 100  # Initial price
mu = 0  # No drift (fair game)
sigma = 0.15 / np.sqrt(252 * 6.5 * 3600)  # 15% annual vol, converted to per-second
sigma_daily = 0.15 / np.sqrt(252)

true_price = np.zeros(n_steps)
true_price[0] = S0

for t in range(1, n_steps):
    dW = np.random.normal(0, np.sqrt(dt))
    true_price[t] = true_price[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

# Market maker parameters
gamma = 0.1  # Risk aversion coefficient
k = 1.5  # Order arrival rate parameter
base_spread = 0.05  # Minimum spread

# Market maker state
inventory = np.zeros(n_steps)
cash = np.zeros(n_steps)
wealth = np.zeros(n_steps)
bid_quotes = np.zeros(n_steps)
ask_quotes = np.zeros(n_steps)
reservation_prices = np.zeros(n_steps)

inventory[0] = 0
cash[0] = 0
wealth[0] = cash[0] + inventory[0] * true_price[0]

# Order flow parameters
lambda_buy = 0.1  # Buy arrival rate (per second)
lambda_sell = 0.1  # Sell arrival rate

trade_log = []

for t in range(n_steps):
    # Calculate reservation price (Avellaneda-Stoikov)
    time_remaining = T - t * dt
    reservation_price = true_price[t] - inventory[t] * gamma * sigma**2 * time_remaining
    reservation_prices[t] = reservation_price
    
    # Optimal spread (simplified)
    spread_adjust = gamma * sigma**2 * time_remaining
    optimal_delta = spread_adjust + base_spread
    
    # Inventory-adjusted quotes
    inventory_skew = inventory[t] * 0.02  # Skew quotes based on inventory
    
    bid_quotes[t] = reservation_price - optimal_delta / 2 - inventory_skew
    ask_quotes[t] = reservation_price + optimal_delta / 2 - inventory_skew
    
    # Ensure quotes don't cross
    if ask_quotes[t] <= bid_quotes[t]:
        ask_quotes[t] = bid_quotes[t] + base_spread
    
    # Simulate order arrivals (Poisson process)
    buy_arrival = np.random.random() < lambda_buy * dt
    sell_arrival = np.random.random() < lambda_sell * dt
    
    # Probability of fill depends on quote competitiveness
    # More aggressive quotes (closer to true price) have higher fill probability
    bid_competitiveness = max(0, 1 - 2 * (true_price[t] - bid_quotes[t]) / true_price[t])
    ask_competitiveness = max(0, 1 - 2 * (ask_quotes[t] - true_price[t]) / true_price[t])
    
    # Execute trades
    if buy_arrival and np.random.random() < bid_competitiveness:
        # Someone buys from dealer (dealer sells, inventory decreases)
        trade_price = ask_quotes[t]
        cash_change = trade_price
        inventory_change = -1
        trade_log.append({'time': t, 'side': 'sell', 'price': trade_price, 
                         'inventory_before': inventory[t]})
    elif sell_arrival and np.random.random() < ask_competitiveness:
        # Someone sells to dealer (dealer buys, inventory increases)
        trade_price = bid_quotes[t]
        cash_change = -trade_price
        inventory_change = 1
        trade_log.append({'time': t, 'side': 'buy', 'price': trade_price,
                         'inventory_before': inventory[t]})
    else:
        cash_change = 0
        inventory_change = 0
    
    # Update state
    if t < n_steps - 1:
        inventory[t + 1] = inventory[t] + inventory_change
        cash[t + 1] = cash[t] + cash_change
        wealth[t + 1] = cash[t + 1] + inventory[t + 1] * true_price[t + 1]

# Final liquidation at true price
final_liquidation = inventory[-1] * true_price[-1]
final_wealth = cash[-1] + final_liquidation

# Calculate metrics
pnl = wealth - wealth[0]
inventory_abs_mean = np.abs(inventory).mean()
inventory_std = inventory.std()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price and quotes
sample_period = slice(0, min(600, n_steps))  # First 10 minutes
time_axis = np.arange(len(true_price[sample_period]))

axes[0, 0].plot(time_axis, true_price[sample_period], 'k-', linewidth=2, label='True Price')
axes[0, 0].plot(time_axis, bid_quotes[sample_period], 'b--', linewidth=1, label='Bid Quote', alpha=0.7)
axes[0, 0].plot(time_axis, ask_quotes[sample_period], 'r--', linewidth=1, label='Ask Quote', alpha=0.7)
axes[0, 0].plot(time_axis, reservation_prices[sample_period], 'g:', linewidth=1.5, 
               label='Reservation Price', alpha=0.8)
axes[0, 0].fill_between(time_axis, bid_quotes[sample_period], ask_quotes[sample_period],
                        alpha=0.2, color='gray', label='Bid-Ask Spread')

# Mark trades
for trade in trade_log:
    if trade['time'] < 600:
        color = 'green' if trade['side'] == 'sell' else 'orange'
        marker = 'v' if trade['side'] == 'sell' else '^'
        axes[0, 0].scatter(trade['time'], trade['price'], c=color, marker=marker, 
                          s=50, alpha=0.6, zorder=5)

axes[0, 0].set_title('Market Maker Quotes and Trades')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend(loc='best', fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Inventory evolution
axes[0, 1].plot(inventory, linewidth=2)
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(range(n_steps), 0, inventory, alpha=0.3)
axes[0, 1].set_title(f'Inventory Evolution (mean |inv|={inventory_abs_mean:.2f})')
axes[0, 1].set_xlabel('Time (seconds)')
axes[0, 1].set_ylabel('Inventory (shares)')
axes[0, 1].grid(alpha=0.3)

print("Inventory Statistics:")
print(f"Mean Absolute Inventory: {inventory_abs_mean:.2f} shares")
print(f"Inventory Std Dev: {inventory_std:.2f} shares")
print(f"Max Inventory: {inventory.max():.0f} shares")
print(f"Min Inventory: {inventory.min():.0f} shares")
print(f"Final Inventory: {inventory[-1]:.0f} shares")

# Plot 3: Quote skewing based on inventory
# Show how quotes adjust relative to true price
quote_mid = (bid_quotes + ask_quotes) / 2
quote_skew = quote_mid - true_price

axes[1, 0].scatter(inventory, quote_skew, alpha=0.3, s=10)
# Fit line
z = np.polyfit(inventory, quote_skew, 1)
p = np.poly1d(z)
inv_range = np.linspace(inventory.min(), inventory.max(), 100)
axes[1, 0].plot(inv_range, p(inv_range), 'r--', linewidth=2, 
               label=f'Slope={z[0]:.4f}')
axes[1, 0].axhline(0, color='black', linewidth=0.5)
axes[1, 0].axvline(0, color='black', linewidth=0.5)
axes[1, 0].set_title('Quote Skewing vs Inventory')
axes[1, 0].set_xlabel('Inventory (shares)')
axes[1, 0].set_ylabel('Quote Midpoint - True Price ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

print(f"\nQuote Skewing:")
print(f"Regression slope: {z[0]:.4f} (negative = skew quotes to reduce inventory)")

# Plot 4: Wealth and PnL evolution
axes[1, 1].plot(wealth, linewidth=2, label='Wealth')
axes[1, 1].plot(cash, linewidth=1.5, alpha=0.7, label='Cash')
axes[1, 1].plot(inventory * true_price, linewidth=1.5, alpha=0.7, label='Inventory Value')
axes[1, 1].axhline(wealth[0], color='black', linestyle='--', linewidth=1)
axes[1, 1].set_title(f'Market Maker Wealth Evolution (Final PnL=${final_wealth:.2f})')
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('Value ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

print(f"\nFinancial Performance:")
print(f"Total Trades: {len(trade_log)}")
print(f"Final Cash: ${cash[-1]:.2f}")
print(f"Final Inventory Value: ${final_liquidation:.2f}")
print(f"Final Wealth: ${final_wealth:.2f}")
print(f"Total PnL: ${final_wealth:.2f}")

plt.tight_layout()
plt.show()

# Analyze inventory costs
# Cost of inventory = Variance of inventory value
inventory_value = inventory * true_price
inventory_variance = np.var(inventory_value)
inventory_cost_estimate = gamma * inventory_variance

print(f"\nInventory Cost Analysis:")
print(f"Inventory Value Variance: ${np.sqrt(inventory_variance):.2f} (std)")
print(f"Estimated Inventory Cost: ${inventory_cost_estimate:.2f}")

# Spread decomposition
quoted_spreads = ask_quotes - bid_quotes
mean_spread = quoted_spreads.mean()
base_component = base_spread
inventory_component = mean_spread - base_spread

print(f"\nSpread Decomposition:")
print(f"Mean Quoted Spread: ${mean_spread:.4f}")
print(f"Base Spread (order processing): ${base_component:.4f} ({base_component/mean_spread*100:.1f}%)")
print(f"Inventory Cost Component: ${inventory_component:.4f} ({inventory_component/mean_spread*100:.1f}%)")

# Compare inventory-adjusted vs fixed spread strategy
# Simulate fixed spread (no inventory adjustment)
print(f"\nInventory Management Effectiveness:")
print("Current strategy skews quotes to manage inventory")
print(f"Mean absolute inventory with skewing: {inventory_abs_mean:.2f}")
print("(Lower is better - indicates effective inventory control)")

# Trade profitability analysis
if len(trade_log) > 0:
    trade_times = [t['time'] for t in trade_log]
    trade_profits = []
    
    for i, trade in enumerate(trade_log):
        # Estimate profit as difference from true price
        if trade['side'] == 'sell':
            profit = trade['price'] - true_price[trade['time']]
        else:
            profit = true_price[trade['time']] - trade['price']
        trade_profits.append(profit)
    
    print(f"\nPer-Trade Analysis:")
    print(f"Mean Profit per Trade: ${np.mean(trade_profits):.4f}")
    print(f"Std Dev of Profit: ${np.std(trade_profits):.4f}")
    print(f"Profitable Trades: {(np.array(trade_profits) > 0).sum()}/{len(trade_profits)}")
