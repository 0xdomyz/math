# Liquidity Provision

## 1. Concept Skeleton
**Definition:** Act of posting limit orders to supply immediacy to other traders, earning bid-ask spread while bearing inventory risk  
**Purpose:** Enables market functioning, facilitates price discovery, compensates for adverse selection and inventory costs  
**Prerequisites:** Market making models, order book mechanics, bid-ask spread, inventory management

## 2. Comparative Framing
| Provider Type | Designated MM | HFT Market Maker | Limit Order Trader | Passive Investor |
|---------------|---------------|------------------|-------------------|------------------|
| **Obligation** | Mandatory quotes | No obligation | Opportunistic | Incidental |
| **Speed** | Moderate | Microseconds | Seconds-minutes | Days |
| **Technology** | Professional | Ultra-low latency | Retail/institutional | Basic |
| **Inventory** | Managed actively | Flattened intraday | Tolerates overnight | Long-term hold |

## 3. Examples + Counterexamples

**Active Provision:**  
HFT firm posts 10K shares bid @ $99.99, 10K ask @ $100.01. Captures $0.02 spread when traders cross. Inventory managed continuously

**Passive Provision:**  
Long-term investor enters limit order to buy 1K @ $99.50 (below market). If filled, inadvertently provided liquidity

**Withdrawal:**  
Flash crash: Market makers pull quotes when volatility spikes (VIX >40) → liquidity evaporates, spreads widen 10x

## 4. Layer Breakdown
```
Liquidity Provision Framework:
├─ Market Maker Types:
│   ├─ Designated Market Makers (DMM):
│   │   - NYSE specialists, Nasdaq market makers
│   │   - Regulatory obligations: maintain orderly market
│   │   - Privileges: Information advantage, rebates
│   ├─ High-Frequency Traders (HFT):
│   │   - Electronic market making (60% of US equity volume)
│   │   - No obligations, but competitive quotes
│   │   - Profits from speed, volume, technology
│   ├─ Broker-Dealer Wholesalers:
│   │   - Retail order flow (payment for order flow)
│   │   - Citadel, Virtu, Two Sigma
│   │   - Internalize trades, provide price improvement
│   └─ Limit Order Traders:
│       - Institutional traders, algorithms
│       - Opportunistic liquidity provision
│       - Earn spread incidentally
├─ Economic Incentives:
│   ├─ Revenue Sources:
│   │   - Bid-ask spread capture (half-spread per trade)
│   │   - Maker-taker rebates ($0.0020-$0.0030 per share)
│   │   - Price improvement capture (PFOF revenue)
│   │   - Information advantage (order flow knowledge)
│   ├─ Cost Components:
│   │   - Adverse selection: Trading with informed
│   │   - Inventory risk: Price volatility exposure
│   │   - Order processing: Technology, connectivity
│   │   - Competition: Race to best quotes
│   └─ Break-Even Analysis:
│       - Revenue > Costs for profitable market making
│       - Scale economies favor high-volume players
├─ Optimal Quoting Strategy:
│   ├─ Avellaneda-Stoikov Model:
│   │   - Reservation price: r = S - q × γ × σ² × T
│   │   - Optimal spread: δ = γ σ² T + (2/γ) ln(1 + γ/k)
│   │   - Inventory-adjusted: Skew quotes to reduce position
│   ├─ Gueant-Lehalle-Tapia:
│   │   - Stochastic control framework
│   │   - Optimal bid/ask placement
│   │   - Accounts for order arrival intensity
│   └─ Empirical Strategies:
│       - Penny jumping: Improve quote by 1 tick
│       - Quote shading: Adjust for inventory/flow toxicity
│       - Strategic cancellation: Pull quotes in adverse conditions
├─ Regulatory Framework:
│   ├─ Reg NMS (US): Order protection, access, display rules
│   ├─ MiFID II (Europe): Market making obligations
│   ├─ SEC Market Access Rule: Risk controls
│   └─ FINRA Best Execution: Duty to achieve favorable prices
└─ Market Quality Impact:
    ├─ Benefits: Narrow spreads, deep markets, price discovery
    ├─ Risks: Flash crashes, quote stuffing, phantom liquidity
    ├─ Measurement: Participation rate, quote quality, resilience
    └─ Social Welfare: Reduces transaction costs for investors
```

**Interaction:** Market order arrives → MM quotes → execution → inventory deviation → quote adjustment → spread capture

## 5. Mini-Project
Simulate optimal market making strategy:
```python
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
```

## 6. Challenge Round
Why has HFT market making grown despite lower per-trade profits?
- **Volume compensation**: Spread per trade = $0.001 (down from $0.05 pre-decimalization), but volume 1000x higher → similar total profit
- **Technology scale**: Fixed costs (co-location, data) amortized over billions of trades → cost per trade < $0.0001
- **Maker-taker rebates**: Exchange rebates ($0.002/share) can exceed spread revenue, making negative-spread quotes profitable
- **Speed advantage**: Sub-millisecond latency allows picking off stale quotes, adverse selection avoidance
- **Regulatory support**: Reg NMS order protection ensures HFT quotes get filled (guaranteed execution at NBBO)

## 7. Key References
- [Avellaneda & Stoikov (2008) - High-Frequency Trading in a Limit Order Book](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)
- [Menkveld (2013) - High Frequency Trading and the New Market Makers](https://www.jstor.org/stable/43303831)
- [Biais et al (2015) - Equilibrium High Frequency Trading](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA10486)
- [Cartea et al (2015) - Algorithmic and High-Frequency Trading](https://www.amazon.com/Algorithmic-High-Frequency-Trading-Mathematics-Finance/dp/1107091144)

---
**Status:** Market making strategy | **Complements:** Bid-Ask Spread, Inventory Costs, HFT
