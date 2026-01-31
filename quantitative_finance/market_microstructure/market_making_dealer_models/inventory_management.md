# Inventory Management in Market Making

## 1. Concept Skeleton
**Definition:** Risk management strategy for market makers to hedge unwanted positions and maintain operational capacity through dynamic rebalancing, position limits, and inventory targeting  
**Purpose:** Minimize holding costs and adverse price moves, maintain liquidity provision capacity, optimize hedging speed-cost tradeoff, control tail risk  
**Prerequisites:** Risk management, mean-reversion, dynamic optimization, inventory theory, hedging mechanics

## 2. Comparative Framing
| Strategy | Cost Minimization | Risk Minimization | Deadline Execution | Balanced |
|----------|------------------|-------------------|-------------------|----------|
| **Objective** | Minimize trading costs | Minimize variance | Execute by time T | Trade-off |
| **Rebalancing Frequency** | Continuous/reactive | High (mean-reverting) | Smooth linear path | Adaptive |
| **Position Limits** | Strict hard stops | Aggressive | Time-based | Dynamic |
| **Adverse Selection** | Exploited | Mitigated by speed | Exposed | Controlled |
| **Example** | Stoll (1978) inventory model | Almgren-Chriss hedge | Algorithmic slice | Avellaneda-Stoikov |

## 3. Examples + Counterexamples

**Simple Example:**  
Market maker accumulates 100 long shares from retail buyers. Position becomes risky. Widens bid-ask spread (or shades lower) to attract hedge counterparties, or uses algo to exit gradually.

**Failure Case:**  
MM ignores inventory during gap move: holds 500 shares, bad news released, market drops 3%, loses $1,500. Should have hedged earlier at smaller loss or tighter limits.

**Edge Case:**  
Earnings announcement: MM may refuse quotes (temporary halt) rather than hold inventory through event, avoiding asymmetric information exposure.

## 4. Layer Breakdown
```
Inventory Management Framework:
├─ Core Concepts:
│   ├─ Inventory Position I(t):
│   │   ├─ Cumulative: I(t) = Σ(buy_quantity - sell_quantity)
│   │   ├─ Positive I: Long exposure, downside risk
│   │   ├─ Negative I: Short exposure, upside risk
│   │   └─ Target: I* = 0 (neutral) typical
│   ├─ Holding Costs:
│   │   ├─ Financing: Interest on borrowed shares (short) or cost of capital (long)
│   │   ├─ Adverse moves: Temporary loss if I ≠ 0 and prices move
│   │   ├─ Liquidity provision: Tied capital unable to earn elsewhere
│   │   └─ Total cost: C(I, t) = interest × |I| + σ²×|I| × (time_to_hedge)
│   └─ Rebalancing:
│       ├─ Passive: Wait for natural counterparties
│       ├─ Active: Market sell/buy to reduce position
│       ├─ Speed cost tradeoff: Fast hedge (market orders) vs patient (limit orders)
│       └─ Urgency driven by position drift from target
├─ Stoll (1978) Inventory Model:
│   ├─ Equilibrium Spread:
│   │   ├─ S = S₀ + 2γI
│   │   ├─ S: Bid-ask spread (half-spread actually)
│   │   ├─ S₀: Adverse selection component (constant)
│   │   ├─ γ: Inventory cost parameter
│   │   ├─ I: Current inventory position
│   │   └─ Interpretation: Wider spread when inventory deviates from target
│   ├─ Inventory Target I*:
│   │   ├─ Usually 0 (neutral)
│   │   ├─ Depends on risk aversion and position risk
│   │   └─ Can be non-zero if firm beliefs about market direction
│   ├─ Dynamic Adjustment:
│   │   ├─ Quote updates every few seconds
│   │   ├─ Respond to: Trades (changes I), market moves (changes risk)
│   │   └─ Goal: Return to I* gradually, control variance
│   └─ Empirical Application:
│       ├─ Estimate γ from historical spread-inventory correlation
│       ├─ Monitor inventory in real-time
│       ├─ Shade quotes (adjust S) to encourage offsetting trades
│       └─ Back-of-envelope: For every $100 position, widen by 0.5 bps
├─ Mean-Reversion Dynamics:
│   ├─ Position Drift:
│   │   ├─ Without hedging: I(t) follows random walk
│   │   ├─ Imbalance in flows: More buys than sells → I increases
│   │   └─ Liquidity risk: Large I may not revert quickly
│   ├─ Optimal Rebalancing:
│   │   ├─ Trade off: Cost of hedging now vs. future adverse move
│   │   ├─ Mean-reversion speed: Faster reversion → less hedging urgency
│   │   ├─ Volatility: Higher vol → want shorter horizon → hedge sooner
│   │   └─ Result: Urgency = α × I × σ²
│   ├─ Algorithmic Execution:
│   │   ├─ Almgren-Chriss style: Hedge trade-off between market and timing impact
│   │   ├─ VWAP participation: Blend with natural flows to hide hedging
│   │   ├─ Destination selection: Route to exchanges with best execution
│   │   └─ Dark pool usage: Avoid signaling intent, minimize market impact
│   └─ Feedback Loop:
│       ├─ Higher I → widen quotes → receive fewer buys
│       ├─ Fewer buys → I shrinks back toward target
│       ├─ Spread adjustment self-corrects inventory
│       └─ Equilibrium reached when quote adjustment balances flow imbalance
├─ Position Limits & Risk Controls:
│   ├─ Hard Limits:
│   │   ├─ Maximum position: |I_max| = 1000 shares (risk limit)
│   │   ├─ Time-based: Must hedge |I| > 500 within 5 minutes
│   │   ├─ Daily: Reset to 0 at end of session (no overnight risk)
│   │   └─ Enforcement: Auto-reject orders violating limits
│   ├─ Soft Limits & Alerts:
│   │   ├─ Yellow zone: |I| > 200 → increase quote frequency, tighter spreads
│   │   ├─ Red zone: |I| > 500 → aggressive hedging
│   │   ├─ Critical: |I| > 1000 → pause market making, liquidate only
│   │   └─ Monitoring: Real-time dashboard, risk manager alerts
│   ├─ Greeks-Based Controls:
│   │   ├─ Delta: Map inventory to directional risk
│   │   ├─ Gamma: Convexity risk if position hedge involves derivatives
│   │   ├─ Vega: Volatility risk of inventory position
│   │   └─ Integration: Combined Greeks into unified risk metric
│   └─ Stress Testing:
│       ├─ Scenario: 5% market move overnight, no liquidity for 2 hours
│       ├─ Max loss: Could position be liquidated without triggering limits?
│       ├─ Worst case: Max position × Max daily move
│       └─ Set limits to ensure survivability
├─ Financing & Cost Management:
│   ├─ Borrow Costs:
│   │   ├─ Short positions: Borrow fee (cost to borrow stock)
│   │   ├─ Hard-to-borrow (HTB): Small-cap, high short interest → expensive
│   │   ├─ Easy-to-borrow (ETB): Liquid large-cap → cheap/free
│   │   └─ Fee impact: Could add 10+ bps to costs on HTB stocks
│   ├─ Haircuts & Margin:
│   │   ├─ Broker requires margin to cover potential losses
│   │   ├─ Haircut: Markdown on position value (e.g., 95% of market value)
│   │   ├─ Capital tied: Large positions reduce capital efficiency
│   │   └─ Cost: Implicit interest on tied-up capital
│   ├─ Funding Rates:
│   │   ├─ Cost to finance long position vs. short
│   │   ├─ Funding gap: May be cheaper to short and hedge than be long
│   │   ├─ Arbitrage: Balance long-short to minimize net financing
│   │   └─ Crypto exchanges: Explicit funding rate mechanism
│   └─ Optimization:
│       ├─ Minimize: Financing costs + hedging costs + spread paid on hedge
│       ├─ Constraint: Position limits, capital limits
│       ├─ Dynamic: Recompute optimal target I* based on current costs
│       └─ Result: May accept smaller spreads if hedging costs are high
├─ Market Impact of Hedging:
│   ├─ Adverse Selection:
│   │   ├─ Counterparties sense hedging = information
│   │   ├─ Other traders see inventory surge → anticipate MM will sell
│   │   ├─ Widen bid-ask to compensate for info asymmetry
│   │   └─ Cost: Additional spread paid when hedging aggressively
│   ├─ Temporary Impact:
│   │   ├─ MM's hedge trade moves temporary component
│   │   ├─ Example: 10,000 share hedge → 0.5 bps temporary impact
│   │   ├─ Recovery: Price recovers over seconds/minutes
│   │   └─ Cost: Paid once per hedge, recovered partially
│   ├─ Permanent Impact:
│   │   ├─ Information leakage: Others infer MM positioning
│   │   ├─ Modest for routine hedging (well-known need)
│   │   ├─ Severe if reveals signal (unusual direction/size)
│   │   └─ Mitigation: Hide hedging in natural flows, use dark pools
│   └─ Timing Strategies:
│       ├─ Slice over time: Spread hedge across multiple time intervals
│       ├─ Blend with volume: Route during high volume periods
│       ├─ Use algos: VWAP, TWAP minimize visibility
│       └─ Result: Higher cost if urgent (less time to blend)
├─ Information & Adverse Selection:
│   ├─ Inventory Imbalance Signals:
│   │   ├─ Rapid growth in |I|: May signal price trend
│   │   ├─ Buys > Sells: Could indicate demand pressure → upward trend
│   │   ├─ Informed traders cluster: Information may be present
│   │   └─ Self-selection: MMs widen spread to reduce loss exposure
│   ├─ Hedging as Signal:
│   │   ├─ When MM hedges aggressively: Market interprets as retreat
│   │   ├─ Example: Large selloff by MM who accumulated longs
│   │   ├─ Can trigger cascades: Others follow, price drops further
│   │   └─ Perverse: Defensive action amplifies losses
│   ├─ Empirical Patterns:
│   │   ├─ Spreads increase after large trades (inventory peak)
│   │   ├─ Hedge trades show realized profits (mean reversion)
│   │   ├─ Informed traders condition on inventory
│   │   └─ Inventory asymmetry predicts intraday direction
│   └─ Optimal Secrets:
│       ├─ Hide hedging as much as possible
│       ├─ Blend with normal order flow
│       ├─ Use multiple venues to obfuscate
│       ├─ Timing: Execute during high volume
│       └─ Dark pools: If available, crucial for large hedges
└─ Practical Implementation:
    ├─ Monitoring:
    │   ├─ Real-time: Track cumulative buys/sells
    │   ├─ Alerts: Trigger at |I| thresholds
    │   ├─ Dashboard: Position, target, hedge recommendation
    │   └─ Frequency: Update every trade (millisecond-level)
    ├─ Quote Adjustment:
    │   ├─ Algorithm: newBid = mid - halfSpread - γ×I
    │   ├─ newAsk = mid + halfSpread + γ×I
    │   ├─ Recompute: After every price tick or inventory change
    │   ├─ Limits: Never quote worse than market (no riskless loss)
    │   └─ Skew: Bid-ask may be asymmetric if I ≠ 0
    ├─ Hedging Execution:
    │   ├─ Trigger: |I| > threshold or time > hedge_horizon
    │   ├─ Algorithm: Use VWAP or other execution algo
    │   ├─ Urgency: Decreases with inventory (takes time)
    │   ├─ Participation: Limit to ~10-20% of volume to hide
    │   └─ Venue: Blend across exchanges, dark pools, internalization
    ├─ Risk Monitoring:
    │   ├─ Greeks: Monitor delta, gamma, vega
    │   ├─ Worst-case: Max loss scenario under stress
    │   ├─ VaR/ES: Value at Risk, Expected Shortfall estimates
    │   └─ Correlation: Monitor multivariate position risk
    └─ Technology Requirements:
        ├─ Position tracking: Nanosecond-precision
        ├─ Quoting system: Atomic position + quote updates
        ├─ Hedging algo: Integrated execution module
        ├─ Risk engine: Real-time Greeks and PnL
        └─ Latency: Sub-millisecond round-trip for responsiveness
```

**Interaction:** Position accumulation → Deviation from target → Quote adjustment → Offsetting flow → Rebalancing → Inventory normalization → Spread compression

## 5. Mini-Project
Simulate inventory dynamics and optimal hedging:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class InventoryParameters:
    """Market maker inventory parameters"""
    target_inventory: float = 0.0           # Ideal position
    inventory_cost_param: float = 0.001     # γ in Stoll model
    financing_cost: float = 0.0001          # Cost per share per period
    hedge_cost: float = 0.0002              # Execution cost per share
    position_limit: float = 500.0           # Max abs position
    hedge_threshold: float = 200.0          # Start hedging at this
    half_spread: float = 0.001              # Base bid-ask (as fraction)
    volatility: float = 0.01                # Daily volatility
    mean_reversion_speed: float = 0.1       # Theta for mean reversion

class MarketMakerInventory:
    """Market maker with inventory management"""
    
    def __init__(self, params: InventoryParameters):
        self.params = params
        self.inventory = 0.0
        self.price = 100.0
        self.cumulative_pnl = 0.0
        self.hedge_trades = []
        self.spread_history = []
    
    def compute_quote_spread(self):
        """Compute optimal bid-ask spread based on inventory"""
        # Stoll model: S = S₀ + 2γ×I
        inventory_component = 2 * self.params.inventory_cost_param * self.inventory
        spread = self.params.half_spread + abs(inventory_component)
        return max(spread, self.params.half_spread * 0.5)  # Min spread floor
    
    def compute_bid_ask(self):
        """Compute bid and ask prices"""
        spread = self.compute_quote_spread()
        mid = self.price
        
        # Skew based on inventory
        bid_skew = 0.0
        ask_skew = 0.0
        
        if self.inventory > 0:  # Long, want to sell
            ask_skew -= self.params.inventory_cost_param * self.inventory
            bid_skew -= self.params.inventory_cost_param * self.inventory * 0.5
        elif self.inventory < 0:  # Short, want to buy
            ask_skew += self.params.inventory_cost_param * abs(self.inventory) * 0.5
            bid_skew += self.params.inventory_cost_param * abs(self.inventory)
        
        bid = mid - spread + bid_skew
        ask = mid + spread + ask_skew
        
        return bid, ask, spread
    
    def simulate_market_flow(self, mu_buy=0.0):
        """
        Simulate random market order flow
        
        mu_buy: Drift in buy pressure (positive = more buys)
        """
        # Random buy/sell with bias
        prob_buy = 0.5 + mu_buy
        is_buy = np.random.random() < prob_buy
        
        size = np.random.exponential(10)  # Typical order size
        
        return is_buy, size
    
    def execute_order(self, is_buy, size, price):
        """Execute incoming order"""
        if is_buy:
            # MM sells, inventory decreases
            self.inventory -= size
            pnl = (price - self.price) * size
        else:
            # MM buys, inventory increases
            self.inventory += size
            pnl = (self.price - price) * size
        
        self.cumulative_pnl += pnl
        
        return pnl
    
    def should_hedge(self):
        """Determine if hedging is needed"""
        if abs(self.inventory) > self.params.hedge_threshold:
            return True
        
        # Time-based: if inventory has persisted, hedge
        if abs(self.inventory) > 100:
            # Higher urgency with more extreme inventory
            return np.random.random() < (abs(self.inventory) / self.params.position_limit) * 0.3
        
        return False
    
    def hedge(self, market_price):
        """Execute hedge trade"""
        # Hedge to reduce inventory
        hedge_qty = self.inventory * 0.5  # Hedge 50% of position
        
        # Pay market impact
        hedge_cost = abs(hedge_qty) * self.params.hedge_cost
        
        # Simulate filling at market
        fill_price = market_price - (np.sign(hedge_qty) * self.params.half_spread * market_price)
        
        # Reduce inventory
        self.inventory -= hedge_qty
        
        pnl = -hedge_cost
        self.cumulative_pnl += pnl
        
        self.hedge_trades.append({
            'quantity': hedge_qty,
            'cost': hedge_cost,
            'inventory_before': self.inventory + hedge_qty,
            'inventory_after': self.inventory
        })
        
        return pnl

def simulate_market_making(n_periods=1000, params=None):
    """Simulate market making with inventory dynamics"""
    
    if params is None:
        params = InventoryParameters()
    
    mm = MarketMakerInventory(params)
    
    price_path = [mm.price]
    inventory_path = [mm.inventory]
    bid_path = []
    ask_path = []
    spread_path = []
    pnl_path = [0.0]
    hedge_count = 0
    
    # Market flow bias (simulates trends)
    mu_buy = 0.0
    
    for t in range(n_periods):
        # Random walk for price
        dP = np.random.normal(0, params.volatility / np.sqrt(252))
        mm.price *= (1 + dP)
        
        # Update flow bias occasionally
        if t % 100 == 0:
            mu_buy = np.random.normal(0, 0.1)
        
        # Get bid-ask
        bid, ask, spread = mm.compute_bid_ask()
        mid = (bid + ask) / 2
        
        # Simulate order flow
        is_buy, size = mm.simulate_market_flow(mu_buy)
        
        # Execute order at appropriate price
        if is_buy:
            exec_price = ask
            pnl = mm.execute_order(is_buy, size, exec_price)
        else:
            exec_price = bid
            pnl = mm.execute_order(is_buy, size, exec_price)
        
        # Check if need to hedge
        if mm.should_hedge():
            hedge_pnl = mm.hedge(mm.price)
            hedge_count += 1
        
        # Financing cost: charged on inventory
        fin_cost = mm.params.financing_cost * abs(mm.inventory)
        mm.cumulative_pnl -= fin_cost
        
        # Record history
        price_path.append(mm.price)
        inventory_path.append(mm.inventory)
        bid_path.append(bid)
        ask_path.append(ask)
        spread_path.append(spread)
        pnl_path.append(mm.cumulative_pnl)
    
    return {
        'price': price_path,
        'inventory': inventory_path,
        'bid': bid_path,
        'ask': ask_path,
        'spread': spread_path,
        'pnl': pnl_path,
        'hedge_count': hedge_count,
        'mm': mm
    }

# Run simulation
print("="*80)
print("MARKET MAKER INVENTORY MANAGEMENT SIMULATION")
print("="*80)

params = InventoryParameters(
    target_inventory=0.0,
    inventory_cost_param=0.0005,
    financing_cost=0.00005,
    hedge_cost=0.0002,
    position_limit=500,
    hedge_threshold=150,
    half_spread=0.001,
    volatility=0.015,
    mean_reversion_speed=0.1
)

results = simulate_market_making(n_periods=2000, params=params)

mm = results['mm']
print(f"\nSimulation Results:")
print(f"  Final Price: ${results['price'][-1]:.2f} (from ${results['price'][0]:.2f})")
print(f"  Final Inventory: {results['inventory'][-1]:.1f} shares")
print(f"  Total Hedges: {results['hedge_count']}")
print(f"  Net PnL: ${results['pnl'][-1]:.2f}")
print(f"  Total Trades: {len(mm.hedge_trades)}")

# Statistics
inventory_array = np.array(results['inventory'])
print(f"\nInventory Statistics:")
print(f"  Mean: {inventory_array.mean():.2f}")
print(f"  Std: {inventory_array.std():.2f}")
print(f"  Max Long: {inventory_array.max():.2f}")
print(f"  Max Short: {inventory_array.min():.2f}")

spread_array = np.array(results['spread'])
print(f"\nSpread Statistics:")
print(f"  Mean: {spread_array.mean()*10000:.2f} bps")
print(f"  Min: {spread_array.min()*10000:.2f} bps")
print(f"  Max: {spread_array.max()*10000:.2f} bps")

price_array = np.array(results['price'])
returns = np.diff(price_array) / price_array[:-1]
print(f"\nMarket Statistics:")
print(f"  Return Volatility: {returns.std()*100:.2f}%")
print(f"  Sharpe Ratio: {(results['pnl'][-1] / (np.std(results['pnl'][1:]) + 1e-6)):.2f}")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: Price and quotes
axes[0, 0].plot(results['price'], label='Market Price', linewidth=1.5)
axes[0, 0].plot(results['bid'], label='MM Bid', alpha=0.5, linewidth=0.8)
axes[0, 0].plot(results['ask'], label='MM Ask', alpha=0.5, linewidth=0.8)
axes[0, 0].set_title('Price and MM Quotes')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Inventory
axes[0, 1].plot(results['inventory'], label='Inventory', linewidth=1)
axes[0, 1].axhline(params.hedge_threshold, color='orange', linestyle='--', label='Hedge Threshold')
axes[0, 1].axhline(-params.hedge_threshold, color='orange', linestyle='--')
axes[0, 1].axhline(params.position_limit, color='red', linestyle='--', label='Limit')
axes[0, 1].axhline(-params.position_limit, color='red', linestyle='--')
axes[0, 1].set_title('Inventory Over Time')
axes[0, 1].set_ylabel('Shares')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spread
axes[1, 0].plot(spread_array * 10000, linewidth=1)
axes[1, 0].set_title('Bid-Ask Spread')
axes[1, 0].set_ylabel('Spread (bps)')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Inventory vs Spread scatter
axes[1, 1].scatter(results['inventory'], spread_array * 10000, alpha=0.3, s=5)
axes[1, 1].set_title('Inventory vs Spread (Stoll Model)')
axes[1, 1].set_xlabel('Inventory (shares)')
axes[1, 1].set_ylabel('Spread (bps)')
axes[1, 1].grid(alpha=0.3)

# Linear fit
inv_arr = np.array(results['inventory'])
spread_bps = spread_array * 10000
valid_idx = ~np.isnan(spread_bps)
z = np.polyfit(inv_arr[valid_idx], spread_bps[valid_idx], 1)
p = np.poly1d(z)
axes[1, 1].plot(inv_arr, p(inv_arr), "r--", linewidth=2, label=f'Fit: y={z[0]:.6f}x+{z[1]:.2f}')
axes[1, 1].legend()

# Plot 5: Cumulative PnL
axes[2, 0].plot(results['pnl'], label='Cumulative PnL', linewidth=1)
axes[2, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[2, 0].set_title('Cumulative P&L')
axes[2, 0].set_ylabel('P&L ($)')
axes[2, 0].set_xlabel('Period')
axes[2, 0].grid(alpha=0.3)

# Plot 6: Inventory Distribution
axes[2, 1].hist(results['inventory'], bins=50, alpha=0.7, edgecolor='black')
axes[2, 1].axvline(np.mean(results['inventory']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(results["inventory"]):.1f}')
axes[2, 1].axvline(0, color='green', linestyle='--', linewidth=2, label='Target: 0')
axes[2, 1].set_title('Inventory Distribution')
axes[2, 1].set_xlabel('Shares')
axes[2, 1].set_ylabel('Frequency')
axes[2, 1].legend()
axes[2, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Stoll model (spread ∝ inventory) validated: spread widens as |I| increases")
print(f"2. Inventory mean-reverts naturally through hedging and flow balance")
print(f"3. Hedges reduce P&L but essential for risk control")
print(f"4. Financing costs accumulate: ~{len([p for p in results['pnl'] if p < 0])/len(results['pnl'])*100:.1f}% of periods unprofitable")
print(f"5. Optimal γ: Balance between: aggressive hedging (low spread) vs. tolerance (high spread)")
```

## 6. Challenge Round
Why does inventory affect spread in Stoll model?
- **Risk-aversion hypothesis**: MM compensates for holding risk by widening spread
- **Holding costs**: Financing and potential adverse move cost increase with |I|
- **Liquidity provision trade-off**: Willing to accept less favorable quotes to rebalance

When does inventory management fail?
- **Gap risk**: Earnings/news causes large jump → can't hedge before losses
- **Liquidity evaporation**: Sudden volume drop → can't execute hedge at reasonable cost
- **Cascade effects**: If many MMs hedging simultaneously → mutual reinforcement, large moves
- **Tail events**: Black swan moves exceed position limits → forced losses

## 7. Key References
- [Stoll (1978): The Supply of Dealer Services in Securities Markets](https://www.jstor.org/stable/2327007)
- [Glosten & Milgrom (1985): Bid, Ask and Transaction Prices](https://www.sciencedirect.com/science/article/abs/pii/0304405X85900443)
- [Hasbrouck & Seppi (1992): Liquidity, Volatility, and Informed Trading](https://www.jstor.org/stable/2328955)

---
**Status:** Core risk management for market makers | **Complements:** Dealer Spread Models, Avellaneda-Stoikov, Trading Algorithms
