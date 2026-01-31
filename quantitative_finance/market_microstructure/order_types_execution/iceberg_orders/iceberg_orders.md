# Iceberg Orders

## 1. Concept Skeleton
**Definition:** Large limit order with visible portion displayed, remaining quantity hidden from public order book  
**Purpose:** Minimize market impact signaling, execute large size without revealing full intention, reduce adverse selection  
**Prerequisites:** Limit orders, order book transparency, information asymmetry, strategic trading

## 2. Comparative Framing
| Order Type | Iceberg Order | Standard Limit | Dark Pool Order |
|-----------|---------------|----------------|-----------------|
| **Visibility** | Partial (tip shown) | Full display | Zero (hidden venue) |
| **Priority** | Loses time priority on refresh | Full time priority | No displayed priority |
| **Detection** | Observable via repeated fills | Transparent | Undetectable |
| **Use Case** | Large institutional trades | Normal liquidity provision | Block trading |

## 3. Examples + Counterexamples

**Simple Example:**  
Iceberg buy 10,000 shares, display 500: Shows 500, after fill refreshes with another 500, repeats 20 times

**Failure Case:**  
High-frequency traders detect iceberg pattern (repeated 500 lots at $50.00): Front-run by penny-jumping, extract rent

**Edge Case:**  
Ultra-low latency refresh race: Multiple icebergs compete for same price level, first refresh wins time priority

## 4. Layer Breakdown
```
Iceberg Order Mechanics:
├─ Order Submission:
│   ├─ Total Size: Full order quantity (e.g., 10,000 shares)
│   ├─ Display Size: Visible portion (e.g., 500 shares)
│   ├─ Limit Price: Maximum buy / minimum sell price
│   └─ Refresh Logic: Automatic replenishment from hidden reserve
├─ Order Book Representation:
│   ├─ Visible Portion: Displayed in Level 2 market data
│   ├─ Hidden Reserve: Not visible to public, maintained by exchange
│   ├─ Time Priority: Only visible portion gets timestamp priority
│   └─ Detection Risk: Pattern recognition from repeated fills at same price
├─ Execution Cycle:
│   ├─ Phase 1: Visible portion trades (500 shares filled)
│   ├─ Phase 2: Auto-refresh from hidden reserve (new 500 displayed)
│   ├─ Phase 3: New time priority assigned (goes to back of queue at price)
│   └─ Repeat: Continue until hidden reserve exhausted or cancelled
├─ Strategic Considerations:
│   ├─ Display Size Choice: Too small = many refreshes, too large = signal
│   ├─ Price Selection: Passive (at spread) vs aggressive (through spread)
│   ├─ Detection Avoidance: Randomize display size, vary timing
│   └─ Adaptive Tactics: HFTs may front-run detected icebergs
└─ Economic Tradeoffs:
    ├─ Benefit: Reduced market impact vs full display
    ├─ Cost: Time priority loss on each refresh
    ├─ Risk: Detection and exploitation by predatory algorithms
    └─ Alternative: Dark pools avoid detection entirely
```

**Interaction:** Submit with hidden size → Display tip → Fill → Refresh → Repeat → Complete or cancel

## 5. Mini-Project
Simulate iceberg order execution and HFT detection strategies:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

class EnhancedOrderBook:
    """Order book supporting iceberg orders and detection"""
    def __init__(self):
        self.bids = {}  # {price: [(timestamp, order_id, visible_size, hidden_size), ...]}
        self.asks = {}
        self.order_id_counter = 0
        self.timestamp = 0
        self.trade_history = []
        self.iceberg_detection = {}  # Track repeated fills at same price
        
    def add_limit_order(self, side, price, size, hidden_size=0):
        """Add limit or iceberg order"""
        self.order_id_counter += 1
        order_id = self.order_id_counter
        
        order_entry = (self.timestamp, order_id, size, hidden_size)
        
        if side == 'buy':
            if price not in self.bids:
                self.bids[price] = []
            self.bids[price].append(order_entry)
        else:
            if price not in self.asks:
                self.asks[price] = []
            self.asks[price].append(order_entry)
        
        self.timestamp += 1
        
        # Track iceberg characteristics
        if hidden_size > 0:
            self.iceberg_detection[order_id] = {
                'price': price,
                'display_size': size,
                'fills': 0,
                'detected': False
            }
        
        return order_id
    
    def match_market_order(self, side, size):
        """Execute market order and handle iceberg refreshes"""
        remaining = size
        fills = []
        iceberg_refreshes = []
        
        if side == 'buy':
            # Match against asks
            while remaining > 0 and self.asks:
                best_ask = min(self.asks.keys())
                queue = self.asks[best_ask]
                
                i = 0
                while i < len(queue) and remaining > 0:
                    ts, oid, visible_size, hidden_size = queue[i]
                    
                    if visible_size <= remaining:
                        # Full fill of visible portion
                        fills.append((best_ask, visible_size, oid))
                        self.trade_history.append((self.timestamp, best_ask, visible_size, oid))
                        remaining -= visible_size
                        
                        # Check if iceberg needs refresh
                        if hidden_size > 0:
                            # Iceberg refresh: replenish visible from hidden
                            new_display = min(visible_size, hidden_size)
                            new_hidden = hidden_size - new_display
                            
                            # Add refreshed order to back of queue (loses time priority)
                            queue.append((self.timestamp, oid, new_display, new_hidden))
                            iceberg_refreshes.append((oid, best_ask, new_display))
                            
                            # Track detection
                            if oid in self.iceberg_detection:
                                self.iceberg_detection[oid]['fills'] += 1
                                # Detection threshold: 3+ repeated fills of same size
                                if self.iceberg_detection[oid]['fills'] >= 3:
                                    self.iceberg_detection[oid]['detected'] = True
                        
                        queue.pop(i)
                    else:
                        # Partial fill
                        fills.append((best_ask, remaining, oid))
                        self.trade_history.append((self.timestamp, best_ask, remaining, oid))
                        queue[i] = (ts, oid, visible_size - remaining, hidden_size)
                        remaining = 0
                        i += 1
                
                if not queue:
                    del self.asks[best_ask]
        else:
            # Match against bids (symmetric logic)
            while remaining > 0 and self.bids:
                best_bid = max(self.bids.keys())
                queue = self.bids[best_bid]
                
                i = 0
                while i < len(queue) and remaining > 0:
                    ts, oid, visible_size, hidden_size = queue[i]
                    
                    if visible_size <= remaining:
                        fills.append((best_bid, visible_size, oid))
                        self.trade_history.append((self.timestamp, best_bid, visible_size, oid))
                        remaining -= visible_size
                        
                        if hidden_size > 0:
                            new_display = min(visible_size, hidden_size)
                            new_hidden = hidden_size - new_display
                            queue.append((self.timestamp, oid, new_display, new_hidden))
                            iceberg_refreshes.append((oid, best_bid, new_display))
                            
                            if oid in self.iceberg_detection:
                                self.iceberg_detection[oid]['fills'] += 1
                                if self.iceberg_detection[oid]['fills'] >= 3:
                                    self.iceberg_detection[oid]['detected'] = True
                        
                        queue.pop(i)
                    else:
                        fills.append((best_bid, remaining, oid))
                        self.trade_history.append((self.timestamp, best_bid, remaining, oid))
                        queue[i] = (ts, oid, visible_size - remaining, hidden_size)
                        remaining = 0
                        i += 1
                
                if not queue:
                    del self.bids[best_bid]
        
        self.timestamp += 1
        return fills, iceberg_refreshes
    
    def get_visible_depth(self, side, levels=5):
        """Get visible order book depth (excludes hidden iceberg size)"""
        if side == 'buy':
            prices = sorted(self.bids.keys(), reverse=True)[:levels]
            return [(p, sum(q[2] for q in self.bids[p])) for p in prices]
        else:
            prices = sorted(self.asks.keys())[:levels]
            return [(p, sum(q[2] for q in self.asks[p])) for p in prices]
    
    def get_total_depth(self, side, levels=5):
        """Get total depth including hidden reserves"""
        if side == 'buy':
            prices = sorted(self.bids.keys(), reverse=True)[:levels]
            return [(p, sum(q[2] + q[3] for q in self.bids[p])) for p in prices]
        else:
            prices = sorted(self.asks.keys())[:levels]
            return [(p, sum(q[2] + q[3] for q in self.asks[p])) for p in prices]

# Simulation scenarios
np.random.seed(42)

def simulate_iceberg_execution(total_size=10000, display_size=500, 
                              hft_exploitation=False):
    """
    Simulate iceberg order execution:
    - Institutional trader: Large iceberg buy
    - Market activity: Random buy/sell flow
    - HFT detection: Identify and front-run icebergs (if enabled)
    """
    book = EnhancedOrderBook()
    
    # Initialize market around $100
    mid_price = 100.0
    
    # Add background liquidity
    for i in range(10):
        book.add_limit_order('buy', mid_price - 0.01 * (i+1), 
                           np.random.randint(300, 800))
        book.add_limit_order('sell', mid_price + 0.01 * (i+1), 
                           np.random.randint(300, 800))
    
    # Institutional iceberg order on ask side
    iceberg_id = book.add_limit_order('sell', mid_price + 0.01, 
                                     display_size, total_size - display_size)
    
    # Track execution metrics
    fills_record = []
    hft_profits = []
    execution_prices = []
    
    # Simulate trading activity
    for t in range(200):
        # Random market activity (noise traders)
        if np.random.random() < 0.4:
            side = np.random.choice(['buy', 'sell'])
            size = np.random.randint(100, 400)
            fills, refreshes = book.match_market_order(side, size)
            
            # Record iceberg fills
            for price, qty, oid in fills:
                if oid == iceberg_id:
                    execution_prices.append(price)
                    fills_record.append({
                        't': t,
                        'price': price,
                        'qty': qty,
                        'cumulative': sum(f['qty'] for f in fills_record) + qty
                    })
            
            # HFT exploitation: If iceberg detected, front-run
            if hft_exploitation and refreshes:
                for oid, price, _ in refreshes:
                    if oid in book.iceberg_detection and book.iceberg_detection[oid]['detected']:
                        # HFT detected iceberg, place order ahead (penny jump)
                        hft_price = price - 0.01  # One tick better
                        hft_oid = book.add_limit_order('sell', hft_price, 500)
                        
                        # Simulate HFT getting filled
                        if np.random.random() < 0.7:  # 70% fill probability
                            hft_fill_price = hft_price
                            # HFT then covers at worse price (profit = 1 tick)
                            hft_profits.append(0.01 * 500)  # $5 profit
        
        # Add replacement liquidity
        if np.random.random() < 0.3:
            side = np.random.choice(['buy', 'sell'])
            levels = book.get_visible_depth(side, 1)
            if levels:
                best_price = levels[0][0]
                offset = np.random.randint(1, 4) * 0.01
                price = best_price - offset if side == 'buy' else best_price + offset
                book.add_limit_order(side, price, np.random.randint(200, 600))
    
    return fills_record, hft_profits, book

# Run simulations with and without HFT exploitation
print("Scenario 1: Normal Market (No HFT Detection)")
fills_normal, _, book_normal = simulate_iceberg_execution(
    total_size=10000, display_size=500, hft_exploitation=False)

print("\nScenario 2: HFT Exploitation (Detection & Front-Running)")
fills_exploit, hft_profits, book_exploit = simulate_iceberg_execution(
    total_size=10000, display_size=500, hft_exploitation=True)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative execution
if fills_normal:
    df_normal = pd.DataFrame(fills_normal)
    axes[0, 0].step(range(len(df_normal)), df_normal['cumulative'], 
                    where='post', linewidth=2, label='Normal Market')
if fills_exploit:
    df_exploit = pd.DataFrame(fills_exploit)
    axes[0, 0].step(range(len(df_exploit)), df_exploit['cumulative'], 
                    where='post', linewidth=2, label='With HFT Exploitation', alpha=0.7)

axes[0, 0].axhline(10000, color='red', linestyle='--', label='Target Size')
axes[0, 0].set_title('Iceberg Order Execution Progress')
axes[0, 0].set_xlabel('Fill Number')
axes[0, 0].set_ylabel('Cumulative Shares Executed')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Execution prices
if fills_normal:
    axes[0, 1].plot(df_normal['price'], 'o-', alpha=0.6, label='Normal Market')
if fills_exploit:
    axes[0, 1].plot(df_exploit['price'], 's-', alpha=0.6, label='With HFT')

axes[0, 1].set_title('Execution Price Per Fill')
axes[0, 1].set_xlabel('Fill Number')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Fill size histogram (shows regular pattern)
if fills_normal:
    axes[1, 0].hist(df_normal['qty'], bins=20, alpha=0.7, label='Normal Market')
axes[1, 0].axvline(500, color='red', linestyle='--', linewidth=2, 
                   label='Display Size (500)')
axes[1, 0].set_title('Fill Size Distribution (Detection Signal)')
axes[1, 0].set_xlabel('Fill Size (shares)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Plot 4: HFT profit extraction
if hft_profits:
    cumulative_hft_profit = np.cumsum(hft_profits)
    axes[1, 1].plot(cumulative_hft_profit, linewidth=2, color='red')
    axes[1, 1].set_title(f'HFT Profit from Exploitation (Total: ${sum(hft_profits):.0f})')
    axes[1, 1].set_xlabel('Detection Event')
    axes[1, 1].set_ylabel('Cumulative Profit ($)')
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'No HFT Exploitation\nin Normal Scenario', 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*50)
print("EXECUTION SUMMARY")
print("="*50)

if fills_normal:
    avg_price_normal = df_normal['price'].mean()
    num_fills_normal = len(df_normal)
    print(f"\nNormal Market:")
    print(f"  Average execution price: ${avg_price_normal:.4f}")
    print(f"  Number of fills: {num_fills_normal}")
    print(f"  Detection risk: {num_fills_normal} repeated fills observable")

if fills_exploit:
    avg_price_exploit = df_exploit['price'].mean()
    num_fills_exploit = len(df_exploit)
    print(f"\nWith HFT Exploitation:")
    print(f"  Average execution price: ${avg_price_exploit:.4f}")
    print(f"  Number of fills: {num_fills_exploit}")
    print(f"  HFT profits extracted: ${sum(hft_profits):.2f}")
    print(f"  Price deterioration: {(avg_price_exploit - avg_price_normal)*10000:.1f} bps")
```

## 6. Challenge Round
How can institutional traders optimize iceberg usage?
- **Randomize display size**: Avoid predictable patterns (vary 300-700 vs fixed 500)
- **Mix with dark pools**: Split order across lit and dark venues to reduce footprint
- **Dynamic pricing**: Adjust limit price to avoid being sole iceberg at level
- **Time variation**: Spread submissions across time to mask total size
- **Volume participation**: Match display size to market volume (less detectable)
- **Smart order routing**: Route to venues with less HFT activity

When are icebergs inappropriate?
- Ultra-liquid assets (S&P 500 stocks): Full display has minimal impact
- Illiquid stocks: Repeated fills are obvious signal in thin markets
- High urgency: Time priority loss from refreshes delays completion
- Volatile markets: Price may move through limit before full execution

## 7. Key References
- [Moinas: Hidden Limit Orders](https://www.sciencedirect.com/science/article/abs/pii/S0304405X10000048)
- [Bessembinder, Panayides, Venkataraman: Hidden Liquidity Survey](https://www.annualreviews.org/doi/abs/10.1146/annurev-financial-110613-034316)
- [SEC: Large Trader Reporting Rule](https://www.sec.gov/rules/final/2011/34-64976.pdf)

---
**Status:** Advanced institutional execution | **Complements:** Limit Orders, Dark Pools, High-Frequency Trading
