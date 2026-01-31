import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

np.random.seed(42)

# Avellaneda-Stoikov Market Making Model
class MarketMaker:
    def __init__(self, gamma, k, sigma, T, tick_size=0.01):
        """
        gamma: risk aversion
        k: order arrival rate parameter
        sigma: volatility
        T: time horizon (seconds)
        """
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = T
        self.tick_size = tick_size
        self.inventory = 0
        self.cash = 0
        self.trade_log = []
        
    def reservation_price(self, S, q, t):
        """Calculate reservation price given position q at time t"""
        time_remaining = self.T - t
        return S - q * self.gamma * (self.sigma ** 2) * time_remaining
    
    def optimal_spread(self, t):
        """Calculate optimal spread"""
        time_remaining = self.T - t
        spread = self.gamma * (self.sigma ** 2) * time_remaining + \
                (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return max(spread, 2 * self.tick_size)  # Minimum 2 ticks
    
    def get_quotes(self, S, q, t):
        """Get bid and ask quotes"""
        r = self.reservation_price(S, q, t)
        delta = self.optimal_spread(t) / 2
        
        # Inventory skewing
        inventory_skew = -q * 0.02  # Lean against inventory
        
        bid = r - delta + inventory_skew
        ask = r + delta + inventory_skew
        
        # Round to tick size
        bid = np.floor(bid / self.tick_size) * self.tick_size
        ask = np.ceil(ask / self.tick_size) * self.tick_size
        
        return bid, ask
    
    def execute_trade(self, side, price):
        """Record trade execution"""
        if side == 'buy':  # MM buys (provides liquidity to seller)
            self.inventory += 1
            self.cash -= price
        else:  # MM sells (provides liquidity to buyer)
            self.inventory -= 1
            self.cash += price
        
        self.trade_log.append({
            'side': side,
            'price': price,
            'inventory': self.inventory,
            'cash': self.cash
        })
    
    def get_wealth(self, S):
        """Calculate mark-to-market wealth"""
        return self.cash + self.inventory * S

# Simulation parameters
T_horizon = 3600  # 1 hour (seconds)
dt = 1  # 1 second intervals
n_steps = int(T_horizon / dt)

# Market parameters
S0 = 100  # Initial price
sigma_annual = 0.20  # 20% annual volatility
sigma_per_second = sigma_annual / np.sqrt(252 * 6.5 * 3600)  # Convert to per-second

# Market maker parameters
gamma = 0.1  # Risk aversion
k = 1.5  # Order arrival intensity
tick_size = 0.01

# Initialize market maker
mm = MarketMaker(gamma=gamma, k=k, sigma=sigma_per_second, T=T_horizon, tick_size=tick_size)

# Price process (Geometric Brownian Motion)
true_price = np.zeros(n_steps)
true_price[0] = S0

for t in range(1, n_steps):
    dW = np.random.normal(0, np.sqrt(dt))
    true_price[t] = true_price[t-1] * np.exp(-0.5 * sigma_per_second**2 * dt + \
                                              sigma_per_second * dW)

# Market making simulation
bid_quotes = np.zeros(n_steps)
ask_quotes = np.zeros(n_steps)
inventory_history = np.zeros(n_steps)
wealth_history = np.zeros(n_steps)
spreads = np.zeros(n_steps)

for t in range(n_steps):
    S = true_price[t]
    q = mm.inventory
    
    # Get optimal quotes
    bid, ask = mm.get_quotes(S, q, t * dt)
    bid_quotes[t] = bid
    ask_quotes[t] = ask
    spreads[t] = ask - bid
    
    # Simulate order arrivals (Poisson process)
    lambda_buy = k * np.exp(-k * (ask - S))  # More likely when ask is closer
    lambda_sell = k * np.exp(-k * (S - bid))  # More likely when bid is closer
    
    # Check for buy order (someone buys from MM, MM sells)
    if np.random.random() < lambda_buy * dt:
        mm.execute_trade('sell', ask)
    
    # Check for sell order (someone sells to MM, MM buys)
    if np.random.random() < lambda_sell * dt:
        mm.execute_trade('buy', bid)
    
    # Record state
    inventory_history[t] = mm.inventory
    wealth_history[t] = mm.get_wealth(S)

# Final liquidation at true price
final_wealth = wealth_history[-1]
pnl = final_wealth

# Analysis
trade_times = [i for i, t in enumerate(mm.trade_log)]
inventory_series = [t['inventory'] for t in mm.trade_log]
pnl_series = [mm.trade_log[i]['cash'] + mm.trade_log[i]['inventory'] * true_price[trade_times[i]] 
              for i in range(len(mm.trade_log))]

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Quotes and trades
sample_period = slice(0, 600)  # First 10 minutes
time_axis = np.arange(600)

axes[0, 0].plot(time_axis, true_price[sample_period], 'k-', linewidth=2, 
               label='True Price', alpha=0.8)
axes[0, 0].plot(time_axis, bid_quotes[sample_period], 'b--', linewidth=1, 
               label='Bid Quote', alpha=0.7)
axes[0, 0].plot(time_axis, ask_quotes[sample_period], 'r--', linewidth=1, 
               label='Ask Quote', alpha=0.7)
axes[0, 0].fill_between(time_axis, bid_quotes[sample_period], 
                        ask_quotes[sample_period], alpha=0.2, color='gray')

# Mark trades
for trade in mm.trade_log:
    t_idx = len([t for t in mm.trade_log if mm.trade_log.index(t) <= mm.trade_log.index(trade)])
    if t_idx < 600:
        color = 'green' if trade['side'] == 'sell' else 'orange'
        marker = 'v' if trade['side'] == 'sell' else '^'
        axes[0, 0].scatter(t_idx, trade['price'], c=color, marker=marker, 
                          s=30, alpha=0.6, zorder=5)

axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Market Maker Quotes and Executions')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

print("Market Making Simulation Results:")
print("=" * 70)
print(f"\nParameters:")
print(f"Risk Aversion (γ): {gamma}")
print(f"Volatility (annualized): {sigma_annual*100:.1f}%")
print(f"Time Horizon: {T_horizon/60:.0f} minutes")
print(f"Tick Size: ${tick_size}")

# Plot 2: Inventory management
axes[0, 1].plot(inventory_history, linewidth=2)
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(range(n_steps), 0, inventory_history, alpha=0.3)
axes[0, 1].set_xlabel('Time (seconds)')
axes[0, 1].set_ylabel('Inventory (shares)')
axes[0, 1].set_title('Inventory Evolution')
axes[0, 1].grid(alpha=0.3)

print(f"\nInventory Management:")
print(f"Total Trades: {len(mm.trade_log)}")
print(f"Buy Trades: {sum(1 for t in mm.trade_log if t['side'] == 'buy')}")
print(f"Sell Trades: {sum(1 for t in mm.trade_log if t['side'] == 'sell')}")
print(f"Mean Absolute Inventory: {np.abs(inventory_history).mean():.2f} shares")
print(f"Max Inventory: {inventory_history.max():.0f} shares")
print(f"Min Inventory: {inventory_history.min():.0f} shares")
print(f"Final Inventory: {mm.inventory:.0f} shares")

# Plot 3: Wealth and PnL
axes[1, 0].plot(wealth_history, linewidth=2, label='Mark-to-Market Wealth')
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('Wealth ($)')
axes[1, 0].set_title(f'Cumulative PnL (Final: ${pnl:.2f})')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

print(f"\nFinancial Performance:")
print(f"Final Cash: ${mm.cash:.2f}")
print(f"Final Inventory Value: ${mm.inventory * true_price[-1]:.2f}")
print(f"Final Wealth: ${final_wealth:.2f}")
print(f"Total PnL: ${pnl:.2f}")

if len(mm.trade_log) > 0:
    pnl_per_trade = pnl / len(mm.trade_log)
    print(f"PnL per Trade: ${pnl_per_trade:.4f}")

# Plot 4: Spread dynamics
axes[1, 1].plot(spreads, linewidth=1.5, alpha=0.7)
axes[1, 1].axhline(spreads.mean(), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: ${spreads.mean():.4f}')
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('Quoted Spread ($)')
axes[1, 1].set_title('Dynamic Spread Adjustment')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

print(f"\nSpread Statistics:")
print(f"Mean Spread: ${spreads.mean():.4f}")
print(f"Min Spread: ${spreads.min():.4f}")
print(f"Max Spread: ${spreads.max():.4f}")
print(f"Spread Std Dev: ${spreads.std():.4f}")

plt.tight_layout()
plt.show()

# Analyze profitability by inventory level
if len(mm.trade_log) > 10:
    # Group trades by inventory bucket
    inventory_buckets = np.array([t['inventory'] for t in mm.trade_log])
    
    print(f"\nProfitability Analysis:")
    
    # Calculate realized profit per trade
    trade_profits = []
    for i in range(1, len(mm.trade_log)):
        prev_wealth = mm.trade_log[i-1]['cash'] + mm.trade_log[i-1]['inventory'] * true_price[trade_times[i-1]]
        curr_wealth = mm.trade_log[i]['cash'] + mm.trade_log[i]['inventory'] * true_price[trade_times[i]]
        profit = curr_wealth - prev_wealth
        trade_profits.append(profit)
    
    if len(trade_profits) > 0:
        print(f"Mean Profit per Trade: ${np.mean(trade_profits):.4f}")
        print(f"Profitable Trades: {(np.array(trade_profits) > 0).sum()}/{len(trade_profits)}")
        print(f"Win Rate: {(np.array(trade_profits) > 0).mean()*100:.1f}%")

# Sharpe ratio
if len(pnl_series) > 1:
    pnl_returns = np.diff(pnl_series)
    if pnl_returns.std() > 0:
        sharpe = (pnl_returns.mean() / pnl_returns.std()) * np.sqrt(252 * 6.5 * 3600 / dt)
        print(f"\nRisk-Adjusted Performance:")
        print(f"Sharpe Ratio (annualized): {sharpe:.2f}")

# Effective spread captured
if len(mm.trade_log) > 0:
    avg_spread_captured = spreads.mean() / 2  # MM earns half-spread
    total_spread_revenue = avg_spread_captured * len(mm.trade_log)
    print(f"\nSpread Revenue Estimate:")
    print(f"Avg Half-Spread: ${avg_spread_captured:.4f}")
    print(f"Total Spread Revenue: ${total_spread_revenue:.2f}")
