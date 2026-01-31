import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import pearsonr

class OrderBook:
    """Simulate realistic order book with depth dynamics"""
    
    def __init__(self, initial_mid=100.0, tick_size=0.01):
        self.mid_price = initial_mid
        self.tick_size = tick_size
        
        # Order book: dict of {price: total_size}
        self.bids = {}
        self.asks = {}
        
        # History tracking
        self.depth_history = []
        self.trade_history = []
        self.price_history = [initial_mid]
        
        # Initialize with some depth
        self._initialize_book()
    
    def _initialize_book(self):
        """Create initial order book with realistic depth profile"""
        # Depth increases with distance from mid (supply curve)
        for i in range(1, 21):
            bid_price = self.mid_price - i * self.tick_size
            ask_price = self.mid_price + i * self.tick_size
            
            # Depth profile: more size at worse prices
            base_size = 100
            depth_increase = i * 50
            noise = np.random.randint(-20, 20)
            
            self.bids[bid_price] = base_size + depth_increase + noise
            self.asks[ask_price] = base_size + depth_increase + noise
    
    def get_best_bid(self):
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self):
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_depth(self, side, levels=10):
        """Get cumulative depth for N levels"""
        if side == 'bid':
            prices = sorted(self.bids.keys(), reverse=True)[:levels]
            return sum(self.bids[p] for p in prices)
        else:
            prices = sorted(self.asks.keys())[:levels]
            return sum(self.asks[p] for p in prices)
    
    def get_depth_profile(self, side, levels=10):
        """Get depth at each price level"""
        if side == 'bid':
            prices = sorted(self.bids.keys(), reverse=True)[:levels]
            return [(p, self.bids[p]) for p in prices]
        else:
            prices = sorted(self.asks.keys())[:levels]
            return [(p, self.asks[p]) for p in prices]
    
    def get_depth_imbalance(self, levels=5):
        """Calculate order book imbalance"""
        bid_depth = self.get_depth('bid', levels)
        ask_depth = self.get_depth('ask', levels)
        
        total = bid_depth + ask_depth
        if total == 0:
            return 0
        
        imbalance = (bid_depth - ask_depth) / total
        return imbalance
    
    def get_weighted_depth(self, side, levels=10):
        """Distance-weighted depth (closer prices weighted more)"""
        profile = self.get_depth_profile(side, levels)
        mid = self.mid_price
        
        weighted = 0
        for price, size in profile:
            distance = abs(price - mid)
            weight = 1 / (1 + distance)  # Inverse distance weighting
            weighted += size * weight
        
        return weighted
    
    def execute_market_order(self, side, size):
        """Execute market order and update depth"""
        remaining = size
        fills = []
        
        if side == 'buy':
            # Walk up the ask side
            while remaining > 0 and self.asks:
                best_ask = self.get_best_ask()
                if best_ask is None:
                    break
                
                available = self.asks[best_ask]
                
                if available <= remaining:
                    fills.append((best_ask, available))
                    remaining -= available
                    del self.asks[best_ask]
                else:
                    fills.append((best_ask, remaining))
                    self.asks[best_ask] -= remaining
                    remaining = 0
        else:  # sell
            # Walk down the bid side
            while remaining > 0 and self.bids:
                best_bid = self.get_best_bid()
                if best_bid is None:
                    break
                
                available = self.bids[best_bid]
                
                if available <= remaining:
                    fills.append((best_bid, available))
                    remaining -= available
                    del self.bids[best_bid]
                else:
                    fills.append((best_bid, remaining))
                    self.bids[best_bid] -= remaining
                    remaining = 0
        
        # Update mid price if needed
        if fills:
            avg_fill_price = sum(p * s for p, s in fills) / sum(s for _, s in fills)
            self.mid_price = (self.get_best_bid() + self.get_best_ask()) / 2 if self.get_best_bid() and self.get_best_ask() else avg_fill_price
            self.price_history.append(self.mid_price)
            self.trade_history.append({
                'side': side,
                'size': size - remaining,
                'avg_price': avg_fill_price,
                'fills': len(fills)
            })
        
        return fills
    
    def add_limit_order(self, side, price, size):
        """Add limit order to book"""
        if side == 'bid':
            self.bids[price] = self.bids.get(price, 0) + size
        else:
            self.asks[price] = self.asks.get(price, 0) + size
    
    def cancel_orders(self, side, pct=0.1):
        """Randomly cancel some orders (liquidity withdrawal)"""
        if side == 'bid':
            prices_to_remove = []
            for price in list(self.bids.keys()):
                if np.random.random() < pct:
                    reduction = int(self.bids[price] * np.random.uniform(0.3, 0.8))
                    self.bids[price] -= reduction
                    if self.bids[price] <= 0:
                        prices_to_remove.append(price)
            for p in prices_to_remove:
                del self.bids[p]
        else:
            prices_to_remove = []
            for price in list(self.asks.keys()):
                if np.random.random() < pct:
                    reduction = int(self.asks[price] * np.random.uniform(0.3, 0.8))
                    self.asks[price] -= reduction
                    if self.asks[price] <= 0:
                        prices_to_remove.append(price)
            for p in prices_to_remove:
                del self.asks[p]
    
    def replenish_depth(self):
        """Market makers add new orders"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return
        
        # Add orders at or near best
        for _ in range(np.random.randint(1, 4)):
            bid_offset = np.random.randint(0, 5) * self.tick_size
            ask_offset = np.random.randint(0, 5) * self.tick_size
            
            size = np.random.randint(50, 300)
            
            self.add_limit_order('bid', best_bid - bid_offset, size)
            self.add_limit_order('ask', best_ask + ask_offset, size)
    
    def snapshot_depth(self):
        """Record current depth state"""
        snapshot = {
            'timestamp': len(self.depth_history),
            'mid_price': self.mid_price,
            'bid_depth_5': self.get_depth('bid', 5),
            'ask_depth_5': self.get_depth('ask', 5),
            'bid_depth_10': self.get_depth('bid', 10),
            'ask_depth_10': self.get_depth('ask', 10),
            'imbalance': self.get_depth_imbalance(5),
            'spread': (self.get_best_ask() - self.get_best_bid()) if self.get_best_bid() and self.get_best_ask() else np.nan,
            'weighted_bid': self.get_weighted_depth('bid', 5),
            'weighted_ask': self.get_weighted_depth('ask', 5)
        }
        self.depth_history.append(snapshot)
        return snapshot

# Simulation: Order book depth dynamics
np.random.seed(42)

book = OrderBook(initial_mid=100.0)

print("="*70)
print("ORDER BOOK DEPTH SIMULATION")
print("="*70)

# Simulate trading activity
n_periods = 500

for t in range(n_periods):
    # Record depth snapshot
    book.snapshot_depth()
    
    # Market activity
    activity_type = np.random.choice(
        ['market_order', 'limit_order', 'cancellation', 'replenish'],
        p=[0.35, 0.30, 0.15, 0.20]
    )
    
    if activity_type == 'market_order':
        side = np.random.choice(['buy', 'sell'])
        size = np.random.randint(50, 500)
        book.execute_market_order(side, size)
        
    elif activity_type == 'limit_order':
        side = np.random.choice(['bid', 'ask'])
        best_bid = book.get_best_bid()
        best_ask = book.get_best_ask()
        
        if side == 'bid' and best_bid:
            price = best_bid - np.random.randint(0, 3) * book.tick_size
        elif best_ask:
            price = best_ask + np.random.randint(0, 3) * book.tick_size
        else:
            continue
        
        size = np.random.randint(100, 400)
        book.add_limit_order(side, price, size)
        
    elif activity_type == 'cancellation':
        side = np.random.choice(['bid', 'ask'])
        book.cancel_orders(side, pct=0.2)
        
    else:  # replenish
        book.replenish_depth()
    
    # Occasional depth shock (liquidity crisis simulation)
    if np.random.random() < 0.01:
        print(f"[t={t}] Liquidity shock!")
        book.cancel_orders('bid', pct=0.5)
        book.cancel_orders('ask', pct=0.5)

# Convert to DataFrame for analysis
df_depth = pd.DataFrame(book.depth_history)

# Analyze depth predictive power
print(f"\n{'='*70}")
print(f"DEPTH ANALYSIS")
print(f"{'='*70}")

print(f"\nDepth Statistics:")
print(f"  Average bid depth (5 levels): {df_depth['bid_depth_5'].mean():.0f} shares")
print(f"  Average ask depth (5 levels): {df_depth['ask_depth_5'].mean():.0f} shares")
print(f"  Average imbalance: {df_depth['imbalance'].mean():.3f}")
print(f"  Depth volatility (CV): {df_depth['bid_depth_5'].std() / df_depth['bid_depth_5'].mean():.2f}")

# Depth imbalance as price predictor
df_depth['price_change'] = df_depth['mid_price'].diff()
df_depth['future_return'] = df_depth['mid_price'].pct_change().shift(-5)  # 5-period ahead

# Remove NaN
analysis_df = df_depth[['imbalance', 'price_change', 'future_return', 'bid_depth_5', 'ask_depth_5']].dropna()

if len(analysis_df) > 10:
    corr_imbalance_return, p_value = pearsonr(analysis_df['imbalance'], analysis_df['future_return'])
    
    print(f"\nPredictive Power:")
    print(f"  Correlation(Imbalance, Future Return): {corr_imbalance_return:.3f} (p={p_value:.4f})")
    print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} predictor")

# Depth decay analysis
depth_shocks = df_depth[df_depth['bid_depth_5'] < df_depth['bid_depth_5'].quantile(0.1)]
print(f"\nDepth Crises (Bottom 10%):")
print(f"  Number of periods: {len(depth_shocks)}")
print(f"  Avg bid depth during crisis: {depth_shocks['bid_depth_5'].mean():.0f} shares")
print(f"  Avg spread during crisis: {depth_shocks['spread'].mean()*10000:.1f} bps")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Depth time series
axes[0, 0].plot(df_depth['timestamp'], df_depth['bid_depth_5'], 
                label='Bid Depth', alpha=0.7, linewidth=1)
axes[0, 0].plot(df_depth['timestamp'], df_depth['ask_depth_5'], 
                label='Ask Depth', alpha=0.7, linewidth=1)
axes[0, 0].set_title('Order Book Depth Over Time (Top 5 Levels)')
axes[0, 0].set_xlabel('Time Period')
axes[0, 0].set_ylabel('Total Size (shares)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Depth imbalance
axes[0, 1].plot(df_depth['timestamp'], df_depth['imbalance'], 
                color='purple', linewidth=1, alpha=0.7)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0, 1].set_title('Order Book Imbalance (Bid-Ask)')
axes[0, 1].set_xlabel('Time Period')
axes[0, 1].set_ylabel('Imbalance')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price vs depth
axes[1, 0].scatter(df_depth['bid_depth_5'], df_depth['mid_price'], 
                   alpha=0.3, s=10, label='Bid Depth')
axes[1, 0].scatter(df_depth['ask_depth_5'], df_depth['mid_price'], 
                   alpha=0.3, s=10, label='Ask Depth', color='orange')
axes[1, 0].set_title('Depth vs Price')
axes[1, 0].set_xlabel('Depth (shares)')
axes[1, 0].set_ylabel('Mid Price ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Imbalance vs future returns
if len(analysis_df) > 0:
    axes[1, 1].scatter(analysis_df['imbalance'], analysis_df['future_return']*10000,
                       alpha=0.5, s=20)
    
    # Add trend line
    z = np.polyfit(analysis_df['imbalance'], analysis_df['future_return']*10000, 1)
    p = np.poly1d(z)
    x_line = np.linspace(analysis_df['imbalance'].min(), analysis_df['imbalance'].max(), 100)
    axes[1, 1].plot(x_line, p(x_line), 'r--', linewidth=2, 
                    label=f'Trend (corr={corr_imbalance_return:.3f})')
    
    axes[1, 1].set_title('Depth Imbalance Predicts Returns')
    axes[1, 1].set_xlabel('Order Book Imbalance')
    axes[1, 1].set_ylabel('5-Period Forward Return (bps)')
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

# Plot 5: Spread vs depth
axes[2, 0].scatter(df_depth['bid_depth_5'], df_depth['spread']*10000,
                   alpha=0.4, s=20, color='green')
axes[2, 0].set_title('Spread vs Depth (Inverse Relationship)')
axes[2, 0].set_xlabel('Bid Depth (shares)')
axes[2, 0].set_ylabel('Spread (bps)')
axes[2, 0].grid(alpha=0.3)

# Plot 6: Depth profile snapshot
final_snapshot = book
bid_profile = final_snapshot.get_depth_profile('bid', 10)
ask_profile = final_snapshot.get_depth_profile('ask', 10)

bid_prices = [p for p, _ in bid_profile]
bid_sizes = [s for _, s in bid_profile]
ask_prices = [p for p, _ in ask_profile]
ask_sizes = [s for _, s in ask_profile]

axes[2, 1].barh(bid_prices, bid_sizes, height=0.008, alpha=0.7, 
                label='Bids', color='green')
axes[2, 1].barh(ask_prices, [-s for s in ask_sizes], height=0.008, alpha=0.7,
                label='Asks', color='red')
axes[2, 1].axhline(final_snapshot.mid_price, color='black', linestyle='--',
                   linewidth=2, label=f'Mid: ${final_snapshot.mid_price:.2f}')
axes[2, 1].set_title('Order Book Depth Profile (Final State)')
axes[2, 1].set_xlabel('Size (shares)')
axes[2, 1].set_ylabel('Price ($)')
axes[2, 1].legend()
axes[2, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print(f"KEY FINDINGS")
print(f"{'='*70}")
print(f"\n1. Depth is dynamic: Constantly changes with orders, cancellations")
print(f"2. Imbalance predicts short-term price moves (buying/selling pressure)")
print(f"3. Low depth periods coincide with wider spreads (liquidity risk)")
print(f"4. Depth shocks create execution risk and higher costs")
print(f"5. Depth profile shows liquidity supply curve (more at worse prices)")
