# Effective Spread

## 1. Concept Skeleton
**Definition:** Realized cost of execution measured as twice the absolute difference between trade price and midpoint quote  
**Purpose:** Capture actual transaction costs including hidden costs, benchmark execution quality, compare venues  
**Prerequisites:** Bid-ask spread, midpoint calculation, trade direction classification, transaction cost analysis

## 2. Comparative Framing
| Spread Measure | Quoted Spread | Effective Spread | Realized Spread |
|----------------|---------------|------------------|-----------------|
| **Formula** | Ask - Bid | 2×|Price - Mid| | 2×(Price - Mid_t+τ) |
| **Timing** | Pre-trade | At execution | Post-trade (5 min) |
| **Captures** | Displayed cost | Actual cost | Price reversion |
| **Price Improvement** | No | Yes | Yes + information |

## 3. Examples + Counterexamples

**Simple Example:**  
Quote: $50.00-$50.02 (mid $50.01), buy at $50.015: Effective spread = 2×($50.015-$50.01) = $0.01 (1 cent)

**Failure Case:**  
Price improvement: Buy inside spread at $50.01, quoted spread $0.02 but effective spread = $0.00 (better than NBBO)

**Edge Case:**  
Midpoint trade in dark pool: Effective spread = $0.00, but opportunity cost if price moves unfavorably during delay

## 4. Layer Breakdown
```
Effective Spread Calculation:
├─ Component Identification:
│   ├─ Trade Price (P): Actual execution price
│   ├─ Midpoint (M): (Bid + Ask) / 2 at time of trade
│   ├─ Trade Direction (D): +1 if buy, -1 if sell
│   └─ Time Matching: Align trade to prevailing quote
├─ Formula Variants:
│   ├─ Absolute: ES = 2 × |P - M|
│   ├─ Signed: ES = 2 × D × (P - M)
│   ├─ Relative: ES_rel = ES / M × 10,000 (basis points)
│   └─ Volume-Weighted: VWES = Σ(ES_i × Size_i) / Σ(Size_i)
├─ Trade Classification Methods:
│   ├─ Lee-Ready Algorithm: Quote rule → tick test fallback
│   ├─ Quote Rule: P > M → buy, P < M → sell
│   ├─ Tick Test: If P = M, use price change direction
│   └─ Reverse Trade Algorithm: Match with order book depth
├─ Price Improvement Detection:
│   ├─ Quoted Spread: QS = Ask - Bid
│   ├─ Improvement: If ES < QS/2, execution better than posted
│   ├─ Sources: Hidden orders, midpoint pegs, internalization
│   └─ Measurement: PI = (QS/2 - ES/2) per share
├─ Aggregation Levels:
│   ├─ Trade-Level: Individual execution quality
│   ├─ Order-Level: Multi-fill parent order average
│   ├─ Daily: Time or volume-weighted across all trades
│   └─ Venue Comparison: Rank exchanges by average ES
└─ Interpretation:
    ├─ Low ES: Good liquidity, narrow spreads, price improvement
    ├─ High ES: Poor execution, wide spreads, walked order book
    ├─ ES vs QS: Ratio indicates price improvement prevalence
    └─ Temporal Variation: U-shaped intraday pattern
```

**Interaction:** Trade → Classify direction → Match quote → Calculate ES → Aggregate for TCA

## 5. Mini-Project
Measure effective spreads and analyze price improvement patterns:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TradeAnalyzer:
    """Analyze effective spreads and price improvement"""
    
    @staticmethod
    def classify_trade_direction(trade_price, bid, ask):
        """
        Lee-Ready algorithm for trade classification
        1. Quote rule: Compare to midpoint
        2. Tick test: If at midpoint, use previous price
        """
        midpoint = (bid + ask) / 2
        
        if trade_price > midpoint:
            return 1  # Buy
        elif trade_price < midpoint:
            return -1  # Sell
        else:
            return 0  # At midpoint, need tick test
    
    @staticmethod
    def calculate_effective_spread(trade_price, bid, ask, direction=None):
        """Calculate effective spread in dollars and basis points"""
        midpoint = (bid + ask) / 2
        
        if direction is None:
            direction = TradeAnalyzer.classify_trade_direction(trade_price, bid, ask)
        
        # Signed effective spread
        es_dollar = 2 * direction * (trade_price - midpoint)
        
        # Absolute effective spread
        es_abs = 2 * abs(trade_price - midpoint)
        
        # Relative effective spread (basis points)
        es_bps = (es_abs / midpoint) * 10000
        
        return {
            'es_dollar': es_dollar,
            'es_abs': es_abs,
            'es_bps': es_bps,
            'direction': direction
        }
    
    @staticmethod
    def calculate_price_improvement(trade_price, bid, ask, direction):
        """Calculate price improvement vs quoted spread"""
        quoted_spread = ask - bid
        midpoint = (bid + ask) / 2
        
        if direction == 1:  # Buy
            # Improvement if paid less than ask
            improvement = ask - trade_price
        else:  # Sell
            # Improvement if received more than bid
            improvement = trade_price - bid
        
        improvement_bps = (improvement / midpoint) * 10000
        
        # As fraction of quoted spread
        improvement_pct = (improvement / (quoted_spread / 2)) * 100
        
        return {
            'improvement_dollar': improvement,
            'improvement_bps': improvement_bps,
            'improvement_pct': improvement_pct,
            'quoted_spread': quoted_spread
        }

# Simulation: Generate realistic trade and quote data
np.random.seed(42)

def simulate_trade_data(n_trades=1000, base_price=100, base_spread=0.02):
    """
    Simulate trades with various execution qualities:
    - Market orders walk the book
    - Limit orders get price improvement
    - Midpoint trades in dark pools
    """
    trades = []
    
    # Initialize quotes
    mid_price = base_price
    half_spread = base_spread / 2
    
    for i in range(n_trades):
        # Quote evolution (random walk)
        if np.random.random() < 0.05:
            mid_price += np.random.randn() * 0.05
        
        # Spread varies with volatility and liquidity
        spread_multiplier = np.random.uniform(0.8, 1.5)
        if np.random.random() < 0.02:  # Occasional spike
            spread_multiplier *= np.random.uniform(2, 5)
        
        current_spread = base_spread * spread_multiplier
        current_half_spread = current_spread / 2
        
        bid = mid_price - current_half_spread
        ask = mid_price + current_half_spread
        
        # Determine trade type and execution
        trade_type = np.random.choice(['market_buy', 'market_sell', 'limit_buy', 
                                      'limit_sell', 'dark_buy', 'dark_sell'],
                                     p=[0.25, 0.25, 0.20, 0.20, 0.05, 0.05])
        
        if trade_type == 'market_buy':
            # Market buy: pay ask, possibly walk book
            if np.random.random() < 0.2:  # Walk the book
                trade_price = ask + np.random.uniform(0, 0.02)
            else:
                trade_price = ask
            size = np.random.randint(100, 1000)
            
        elif trade_type == 'market_sell':
            # Market sell: receive bid, possibly walk book
            if np.random.random() < 0.2:
                trade_price = bid - np.random.uniform(0, 0.02)
            else:
                trade_price = bid
            size = np.random.randint(100, 1000)
            
        elif trade_type == 'limit_buy':
            # Limit buy: price improvement inside spread
            trade_price = bid + np.random.uniform(0, current_half_spread * 0.8)
            size = np.random.randint(100, 500)
            
        elif trade_type == 'limit_sell':
            # Limit sell: price improvement inside spread
            trade_price = ask - np.random.uniform(0, current_half_spread * 0.8)
            size = np.random.randint(100, 500)
            
        elif trade_type == 'dark_buy':
            # Dark pool: midpoint execution
            trade_price = mid_price
            size = np.random.randint(500, 2000)
            
        else:  # dark_sell
            trade_price = mid_price
            size = np.random.randint(500, 2000)
        
        trades.append({
            'trade_id': i,
            'timestamp': datetime(2026, 1, 31, 9, 30) + timedelta(seconds=i*10),
            'price': trade_price,
            'size': size,
            'bid': bid,
            'ask': ask,
            'mid': mid_price,
            'trade_type': trade_type
        })
    
    return pd.DataFrame(trades)

# Generate data
df = simulate_trade_data(n_trades=1000)

# Calculate effective spreads and price improvement
results = []

for _, trade in df.iterrows():
    # Classify direction
    direction = TradeAnalyzer.classify_trade_direction(
        trade['price'], trade['bid'], trade['ask'])
    
    # Effective spread
    es = TradeAnalyzer.calculate_effective_spread(
        trade['price'], trade['bid'], trade['ask'], direction)
    
    # Price improvement
    pi = TradeAnalyzer.calculate_price_improvement(
        trade['price'], trade['bid'], trade['ask'], direction)
    
    results.append({
        'trade_id': trade['trade_id'],
        'timestamp': trade['timestamp'],
        'trade_type': trade['trade_type'],
        'direction': direction,
        'es_dollar': es['es_abs'],
        'es_bps': es['es_bps'],
        'quoted_spread': pi['quoted_spread'],
        'price_improvement': pi['improvement_dollar'],
        'pi_bps': pi['improvement_bps'],
        'pi_pct': pi['improvement_pct'],
        'size': trade['size']
    })

df_results = pd.DataFrame(results)

# Analysis
print("="*70)
print("EFFECTIVE SPREAD ANALYSIS")
print("="*70)

# Overall statistics
print(f"\nOverall Execution Quality:")
print(f"  Average effective spread: {df_results['es_bps'].mean():.2f} bps")
print(f"  Average quoted spread: {(df_results['quoted_spread'].mean() / df['mid'].mean() * 10000):.2f} bps")
print(f"  Average price improvement: {df_results['pi_bps'].mean():.2f} bps")
print(f"  Price improvement rate: {(df_results['price_improvement'] > 0).mean()*100:.1f}%")

# By trade type
print(f"\n{'Trade Type':<20} {'Eff Spread (bps)':>18} {'Price Improv (bps)':>20}")
print("-"*70)

for trade_type in df_results['trade_type'].unique():
    subset = df_results[df_results['trade_type'] == trade_type]
    avg_es = subset['es_bps'].mean()
    avg_pi = subset['pi_bps'].mean()
    print(f"{trade_type:<20} {avg_es:>18.2f} {avg_pi:>20.2f}")

# Volume-weighted statistics
vw_es = (df_results['es_bps'] * df_results['size']).sum() / df_results['size'].sum()
vw_pi = (df_results['pi_bps'] * df_results['size']).sum() / df_results['size'].sum()

print(f"\nVolume-Weighted Metrics:")
print(f"  VW Effective Spread: {vw_es:.2f} bps")
print(f"  VW Price Improvement: {vw_pi:.2f} bps")

# Execution quality score
eq_score = (df_results['pi_pct'].mean() / 100) * 100
print(f"\nExecution Quality Score: {eq_score:.1f}/100")
print(f"  (Based on avg price improvement as % of half-spread)")

# Time-series analysis
df_results['hour'] = df_results['timestamp'].dt.hour
hourly_stats = df_results.groupby('hour').agg({
    'es_bps': 'mean',
    'pi_bps': 'mean',
    'quoted_spread': 'mean'
}).reset_index()

print(f"\nIntraday Pattern:")
print(f"{'Hour':>5} {'Eff Spread (bps)':>18} {'Price Improv (bps)':>20}")
print("-"*50)
for _, row in hourly_stats.iterrows():
    print(f"{int(row['hour']):>5} {row['es_bps']:>18.2f} {row['pi_bps']:>20.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Effective spread distribution
axes[0, 0].hist(df_results['es_bps'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(df_results['es_bps'].median(), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {df_results["es_bps"].median():.2f} bps')
axes[0, 0].set_title('Effective Spread Distribution')
axes[0, 0].set_xlabel('Effective Spread (bps)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Effective vs Quoted Spread
quoted_spread_bps = (df_results['quoted_spread'] / df['mid'].mean() * 10000)
axes[0, 1].scatter(quoted_spread_bps, df_results['es_bps'], alpha=0.5, s=20)
axes[0, 1].plot([0, quoted_spread_bps.max()], [0, quoted_spread_bps.max()], 
                'r--', linewidth=2, label='ES = QS')
axes[0, 1].plot([0, quoted_spread_bps.max()], [0, quoted_spread_bps.max()/2], 
                'g--', linewidth=2, label='ES = QS/2')
axes[0, 1].set_title('Effective vs Quoted Spread')
axes[0, 1].set_xlabel('Quoted Spread (bps)')
axes[0, 1].set_ylabel('Effective Spread (bps)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price improvement distribution
axes[0, 2].hist(df_results['pi_bps'], bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='No Improvement')
axes[0, 2].set_title('Price Improvement Distribution')
axes[0, 2].set_xlabel('Price Improvement (bps)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: ES by trade type
trade_types = df_results.groupby('trade_type')['es_bps'].mean().sort_values()
axes[1, 0].barh(range(len(trade_types)), trade_types.values, color='coral', alpha=0.7)
axes[1, 0].set_yticks(range(len(trade_types)))
axes[1, 0].set_yticklabels(trade_types.index)
axes[1, 0].set_title('Avg Effective Spread by Trade Type')
axes[1, 0].set_xlabel('Effective Spread (bps)')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 5: Intraday pattern
axes[1, 1].plot(hourly_stats['hour'], hourly_stats['es_bps'], 
                'o-', linewidth=2, markersize=8, label='Effective Spread')
axes[1, 1].plot(hourly_stats['hour'], hourly_stats['pi_bps'], 
                's-', linewidth=2, markersize=8, label='Price Improvement')
axes[1, 1].set_title('Intraday Execution Quality')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Basis Points')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Price improvement vs trade size
axes[1, 2].scatter(df_results['size'], df_results['pi_bps'], alpha=0.5, s=20)
axes[1, 2].set_title('Price Improvement vs Trade Size')
axes[1, 2].set_xlabel('Trade Size (shares)')
axes[1, 2].set_ylabel('Price Improvement (bps)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# TCA Report
print(f"\n{'='*70}")
print(f"TRANSACTION COST ANALYSIS REPORT")
print(f"{'='*70}")

total_volume = df_results['size'].sum()
total_es_cost = (df_results['es_dollar'] * df_results['size']).sum()
total_pi_savings = (df_results['price_improvement'] * df_results['size']).sum()
net_cost = total_es_cost - total_pi_savings

print(f"\nTotal Statistics:")
print(f"  Total trades: {len(df_results):,}")
print(f"  Total volume: {total_volume:,} shares")
print(f"  Gross spread cost: ${total_es_cost:,.2f}")
print(f"  Price improvement savings: ${total_pi_savings:,.2f}")
print(f"  Net transaction cost: ${net_cost:,.2f}")
print(f"  Cost per share: ${net_cost/total_volume:.4f}")
print(f"  Cost as % of value: {(net_cost/(total_volume * df['mid'].mean()))*100:.3f}%")
```

## 6. Challenge Round
How does effective spread relate to market quality and venue selection?
- **Execution quality ranking**: Lower ES = better venue, but consider fill rates, speed
- **Hidden liquidity**: Dark pools show ES = 0 (midpoint), but delayed execution risk
- **Payment for order flow**: Retail may get price improvement, but routing conflicts
- **Venue competition**: Fragmentation increases price improvement opportunities
- **Time variation**: ES wider at open/close, narrower midday (liquidity U-curve)

What are limitations of effective spread as a metric?
- **Ignores opportunity cost**: Unfilled orders have ES = 0 but infinite opportunity cost
- **No impact component**: Doesn't capture price movement caused by order itself
- **Short-term focus**: Realized spread (5 min later) better for permanent impact
- **Size blind**: Large orders may get good ES but poor overall execution
- **Gaming potential**: Internalizers may optimize ES but worsen total execution

## 7. Key References
- [Bessembinder & Kaufman (1997): Comparison of Trade Execution Costs](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb02744.x)
- [Huang & Stoll (1996): Dealer vs Auction Markets](https://www.jstor.org/stable/2329367)
- [Lee & Ready (1991): Inferring Trade Direction](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1991.tb02683.x)

---
**Status:** Standard TCA metric | **Complements:** Quoted Spread, Realized Spread, Implementation Shortfall
