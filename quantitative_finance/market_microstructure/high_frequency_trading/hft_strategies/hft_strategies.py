import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import time

class HFTMarketMaker:
    """Simulate high-frequency market making strategy"""
    
    def __init__(self, target_spread=0.01, inventory_limit=1000, risk_aversion=0.5):
        self.target_spread = target_spread
        self.inventory_limit = inventory_limit
        self.risk_aversion = risk_aversion
        
        # State
        self.inventory = 0
        self.cash = 0
        self.pnl_history = [0]
        self.trade_count = 0
        
        # Risk management
        self.max_drawdown = 0
        self.peak_pnl = 0
    
    def quote_prices(self, true_value, volatility):
        """
        Determine bid/ask quotes based on inventory and risk
        
        Avellaneda-Stoikov market making model:
        - Widen spread when inventory extreme
        - Skew quotes to reduce inventory
        """
        # Base half-spread
        half_spread = self.target_spread / 2
        
        # Inventory adjustment (skew quotes)
        inventory_skew = self.risk_aversion * self.inventory / self.inventory_limit
        
        # Volatility adjustment (wider in volatile markets)
        vol_adjustment = volatility * 10
        
        # Compute quotes
        bid = true_value - half_spread - inventory_skew - vol_adjustment
        ask = true_value + half_spread - inventory_skew + vol_adjustment
        
        return bid, ask
    
    def execute_trade(self, side, price, size):
        """Record trade execution"""
        if side == 'buy':
            self.inventory += size
            self.cash -= price * size
        else:  # sell
            self.inventory -= size
            self.cash += price * size
        
        self.trade_count += 1
        
        # Update P&L
        self.pnl_history.append(self.cash + self.inventory * price)
        
        # Risk monitoring
        current_pnl = self.pnl_history[-1]
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl
        
        drawdown = self.peak_pnl - current_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def check_inventory_limit(self):
        """Check if inventory exceeds risk limits"""
        return abs(self.inventory) >= self.inventory_limit

class LatencyArbitrageStrategy:
    """Simulate latency arbitrage across venues"""
    
    def __init__(self, latency_advantage_us=100):
        self.latency_advantage_us = latency_advantage_us  # Microseconds
        self.trades = []
        self.successful_arbs = 0
        self.failed_arbs = 0
        self.total_pnl = 0
    
    def detect_arbitrage(self, venue_a_price, venue_b_price, threshold=0.001):
        """Detect price discrepancy across venues"""
        price_diff = abs(venue_a_price - venue_b_price)
        
        if price_diff > threshold:
            return True, venue_a_price, venue_b_price
        
        return False, None, None
    
    def execute_arbitrage(self, fast_price, slow_price, latency_us):
        """
        Try to arbitrage stale quote
        
        Success depends on:
        - Latency advantage
        - Whether other HFTs already arbed
        - Quote refresh rate
        """
        # Simulate race condition
        # If our latency advantage > competitor latency, we win
        competitor_latency = np.random.uniform(50, 200)  # Other HFTs
        
        if latency_us < competitor_latency:
            # Successful arbitrage
            profit = abs(fast_price - slow_price) * 100  # 100 shares
            self.total_pnl += profit
            self.successful_arbs += 1
            
            self.trades.append({
                'success': True,
                'profit': profit,
                'latency_us': latency_us
            })
            
            return True, profit
        else:
            # Quote already updated by competitor
            # We might get adverse fill
            loss = np.random.uniform(0, 0.01) * 100
            self.total_pnl -= loss
            self.failed_arbs += 1
            
            self.trades.append({
                'success': False,
                'profit': -loss,
                'latency_us': latency_us
            })
            
            return False, -loss

def simulate_hft_market_making(n_periods=10000):
    """Simulate HFT market making with realistic order flow"""
    np.random.seed(42)
    
    mm = HFTMarketMaker(target_spread=0.02, inventory_limit=500, risk_aversion=0.3)
    
    # Market state
    true_value = 100.0
    current_volatility = 0.01
    
    history = []
    
    for t in range(n_periods):
        # True value random walk
        true_value += np.random.normal(0, 0.01)
        
        # Volatility clustering
        if np.random.random() < 0.05:
            current_volatility *= np.random.uniform(1.5, 3.0)
        else:
            current_volatility *= 0.99
        
        current_volatility = max(0.005, min(0.05, current_volatility))
        
        # Get quotes
        bid, ask = mm.quote_prices(true_value, current_volatility)
        
        # Simulate order flow
        order_arrival = np.random.random()
        
        if order_arrival < 0.4:  # Market buy order arrives
            if not mm.check_inventory_limit():
                # Fill at our ask
                mm.execute_trade('sell', ask, 10)
        
        elif order_arrival < 0.8:  # Market sell order arrives
            if not mm.check_inventory_limit():
                # Fill at our bid
                mm.execute_trade('buy', bid, 10)
        
        # Occasional large informed order (adverse selection)
        if np.random.random() < 0.01:
            # Price about to move
            direction = np.random.choice([-1, 1])
            true_value += direction * 0.05
            
            # Informed trader hits our quote
            if direction == 1 and not mm.check_inventory_limit():
                # Price going up, they buy from us (we sell)
                mm.execute_trade('sell', ask, 50)
            elif direction == -1 and not mm.check_inventory_limit():
                # Price going down, they sell to us (we buy)
                mm.execute_trade('buy', bid, 50)
        
        # Inventory management: hedge if inventory too large
        if abs(mm.inventory) > mm.inventory_limit * 0.7:
            hedge_size = int(mm.inventory * 0.3)
            if hedge_size != 0:
                side = 'sell' if mm.inventory > 0 else 'buy'
                # Pay spread to reduce inventory
                hedge_price = true_value if side == 'sell' else true_value
                mm.execute_trade(side, hedge_price, abs(hedge_size))
        
        # Record state
        history.append({
            'period': t,
            'true_value': true_value,
            'bid': bid,
            'ask': ask,
            'spread': ask - bid,
            'inventory': mm.inventory,
            'pnl': mm.pnl_history[-1],
            'volatility': current_volatility
        })
    
    return pd.DataFrame(history), mm

def simulate_latency_arbitrage(n_opportunities=1000):
    """Simulate latency arbitrage opportunities"""
    np.random.seed(42)
    
    arb = LatencyArbitrageStrategy(latency_advantage_us=100)
    
    results = []
    
    for i in range(n_opportunities):
        # Simulate price discrepancy event
        venue_a_price = 100 + np.random.normal(0, 0.1)
        
        # Venue B has stale quote with some probability
        if np.random.random() < 0.3:  # 30% of time there's discrepancy
            venue_b_price = venue_a_price + np.random.uniform(-0.01, 0.01)
            
            has_arb, price_a, price_b = arb.detect_arbitrage(venue_a_price, venue_b_price)
            
            if has_arb:
                # Our latency (includes co-location advantage)
                our_latency = np.random.uniform(80, 120)
                
                success, profit = arb.execute_arbitrage(price_a, price_b, our_latency)
                
                results.append({
                    'opportunity': i,
                    'success': success,
                    'profit': profit,
                    'latency_us': our_latency,
                    'price_diff': abs(price_a - price_b)
                })
    
    return pd.DataFrame(results), arb

# Run simulations
print("="*80)
print("HIGH-FREQUENCY TRADING STRATEGY SIMULATION")
print("="*80)

# Simulation 1: Market Making
print("\n" + "="*80)
print("MARKET MAKING STRATEGY")
print("="*80)

df_mm, mm_strategy = simulate_hft_market_making(n_periods=10000)

print(f"\nPerformance Metrics:")
print(f"  Total trades: {mm_strategy.trade_count:,}")
print(f"  Final P&L: ${mm_strategy.pnl_history[-1]:,.2f}")
print(f"  Average P&L per trade: ${mm_strategy.pnl_history[-1]/mm_strategy.trade_count:.4f}")
print(f"  Max drawdown: ${mm_strategy.max_drawdown:,.2f}")
print(f"  Final inventory: {mm_strategy.inventory} shares")
print(f"  Sharpe ratio: {np.mean(np.diff(mm_strategy.pnl_history))/np.std(np.diff(mm_strategy.pnl_history))*np.sqrt(252):.2f}")

# Analyze spread dynamics
avg_spread = df_mm['spread'].mean()
spread_volatility = df_mm['spread'].std()

print(f"\nSpread Statistics:")
print(f"  Average spread: ${avg_spread:.4f}")
print(f"  Spread volatility: ${spread_volatility:.4f}")
print(f"  Min spread: ${df_mm['spread'].min():.4f}")
print(f"  Max spread: ${df_mm['spread'].max():.4f}")

# Simulation 2: Latency Arbitrage
print("\n" + "="*80)
print("LATENCY ARBITRAGE STRATEGY")
print("="*80)

df_arb, arb_strategy = simulate_latency_arbitrage(n_opportunities=1000)

if len(df_arb) > 0:
    print(f"\nArbitrage Statistics:")
    print(f"  Opportunities detected: {len(df_arb)}")
    print(f"  Successful arbitrages: {arb_strategy.successful_arbs}")
    print(f"  Failed arbitrages: {arb_strategy.failed_arbs}")
    print(f"  Success rate: {arb_strategy.successful_arbs/len(df_arb)*100:.1f}%")
    print(f"  Total P&L: ${arb_strategy.total_pnl:.2f}")
    print(f"  Average profit per success: ${df_arb[df_arb['success']]['profit'].mean():.4f}")
    print(f"  Average loss per failure: ${df_arb[~df_arb['success']]['profit'].mean():.4f}")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Market making P&L
axes[0, 0].plot(df_mm['period'], df_mm['pnl'], linewidth=1, alpha=0.8)
axes[0, 0].set_title('Market Making Cumulative P&L')
axes[0, 0].set_xlabel('Period')
axes[0, 0].set_ylabel('P&L ($)')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Inventory over time
axes[0, 1].plot(df_mm['period'], df_mm['inventory'], linewidth=1, alpha=0.8, color='orange')
axes[0, 1].axhline(mm_strategy.inventory_limit, color='red', linestyle='--', 
                   label=f'Limit: ±{mm_strategy.inventory_limit}')
axes[0, 1].axhline(-mm_strategy.inventory_limit, color='red', linestyle='--')
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].set_title('Inventory Management')
axes[0, 1].set_xlabel('Period')
axes[0, 1].set_ylabel('Inventory (shares)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spread dynamics
axes[1, 0].plot(df_mm['period'], df_mm['spread']*10000, linewidth=0.5, alpha=0.7)
axes[1, 0].axhline(avg_spread*10000, color='red', linestyle='--', 
                   label=f'Mean: {avg_spread*10000:.1f} bps')
axes[1, 0].set_title('Quoted Spread Over Time')
axes[1, 0].set_xlabel('Period')
axes[1, 0].set_ylabel('Spread (bps)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Spread vs inventory
axes[1, 1].scatter(df_mm['inventory'], df_mm['spread']*10000, alpha=0.1, s=5)
axes[1, 1].set_title('Spread Adjustment by Inventory')
axes[1, 1].set_xlabel('Inventory (shares)')
axes[1, 1].set_ylabel('Spread (bps)')
axes[1, 1].grid(alpha=0.3)

# Plot 5: Latency arbitrage success rate
if len(df_arb) > 0:
    success_by_latency = df_arb.groupby(pd.cut(df_arb['latency_us'], bins=10))['success'].mean()
    
    axes[2, 0].bar(range(len(success_by_latency)), success_by_latency.values*100, 
                   alpha=0.7, color='green')
    axes[2, 0].set_title('Arbitrage Success Rate by Latency')
    axes[2, 0].set_xlabel('Latency Bin')
    axes[2, 0].set_ylabel('Success Rate (%)')
    axes[2, 0].grid(axis='y', alpha=0.3)

# Plot 6: Arbitrage P&L distribution
if len(df_arb) > 0:
    axes[2, 1].hist(df_arb['profit'], bins=30, alpha=0.7, edgecolor='black')
    axes[2, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[2, 1].set_title('Latency Arbitrage P&L Distribution')
    axes[2, 1].set_xlabel('Profit per Trade ($)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Market making profits from spread capture but faces inventory risk")
print(f"2. Widening spreads during high inventory crucial for risk management")
print(f"3. Adverse selection from informed traders main risk (large losses)")
print(f"4. Latency arbitrage requires sub-millisecond execution advantage")
print(f"5. Success rate drops sharply with even small latency disadvantage")
print(f"6. HFT profitability depends on volume: many small wins, few large losses")
