# Liquidity Definitions

## 1. Concept Skeleton
**Definition:** Market's ability to facilitate large trades with minimal price impact and low transaction costs  
**Purpose:** Characterizes market quality, affects asset pricing, determines trading strategy viability  
**Prerequisites:** Market microstructure basics, bid-ask spread, order book mechanics

## 2. Comparative Framing
| Dimension | Tightness | Depth | Immediacy | Resiliency |
|-----------|-----------|-------|-----------|------------|
| **Measure** | Bid-ask spread | Order book volume | Time to execute | Recovery speed |
| **Indicator** | Cost per share | Shares available | Execution latency | Mean reversion |
| **Good Liquidity** | Narrow spreads | Deep book | Instant fills | Fast recovery |
| **Poor Liquidity** | Wide spreads | Thin book | Delayed fills | Slow recovery |

## 3. Examples + Counterexamples

**High Liquidity:**  
SPY (S&P 500 ETF): Spread=1¢ ($0.01%), depth=100K shares, fills in milliseconds, impact vanishes in seconds

**Low Liquidity:**  
Micro-cap stock: Spread=$0.10 (5%), depth=500 shares, hours between trades, 10% price impact persists

**Edge Case:**  
Flash crash: Appears liquid (tight spreads) until large sell order → liquidity evaporates instantly, spreads widen 100x

## 4. Layer Breakdown
```
Liquidity Dimensions (Kyle 1985):
├─ Tightness:
│   ├─ Definition: Bid-ask spread width
│   ├─ Measures: Quoted spread, effective spread, realized spread
│   ├─ Interpretation: Transaction cost for round-trip trade
│   └─ Drivers: Order processing, inventory, adverse selection
├─ Depth:
│   ├─ Definition: Volume available at best quotes
│   ├─ Measures: Shares at BBO, cumulative depth (5/10 levels)
│   ├─ Interpretation: Capacity to absorb market orders
│   └─ Drivers: Market maker capital, risk tolerance, competition
├─ Immediacy:
│   ├─ Definition: Speed of execution at reasonable cost
│   ├─ Measures: Fill rate, time to execute, order completion
│   ├─ Interpretation: Market responsiveness to trade requests
│   └─ Drivers: Order arrival rate, matching speed, venues
├─ Resiliency:
│   ├─ Definition: Recovery speed after large order
│   ├─ Measures: Autocorrelation decay, price reversion time
│   ├─ Interpretation: Market's ability to absorb shocks
│   └─ Drivers: Information flow, arbitrageur activity, volatility
├─ Market Quality Trade-offs:
│   ├─ Tightness ↔ Depth: Tight spreads may have shallow depth
│   ├─ Immediacy ↔ Cost: Faster execution pays higher spread
│   ├─ Resiliency ↔ Volatility: Volatile markets slower recovery
│   └─ Depth ↔ Resiliency: Deep book doesn't guarantee fast recovery
└─ Comprehensive Measures:
    ├─ Amihud Illiquidity: |Return| / Volume
    ├─ Kyle's Lambda: Price impact per unit volume
    ├─ Market Impact Function: ΔPrice = f(Volume)
    └─ Liquidity Score: Composite of dimensions
```

**Interaction:** Order flow → depth consumed → spread widens → time to replenish → resiliency determines recovery

## 5. Mini-Project
Simulate and measure liquidity dimensions:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Simulate two markets: high liquidity vs low liquidity
n_seconds = 3600  # 1 hour
tick_size = 0.01

# Market parameters
markets = {
    'High Liquidity': {
        'base_spread': 0.02,
        'depth_mean': 5000,
        'order_arrival_rate': 2.0,  # Orders per second
        'replenish_speed': 0.5,  # Fast recovery
        'volatility': 0.0001
    },
    'Low Liquidity': {
        'base_spread': 0.20,
        'depth_mean': 200,
        'order_arrival_rate': 0.1,  # Orders per second
        'replenish_speed': 0.05,  # Slow recovery
        'volatility': 0.0005
    }
}

results = {}

for market_name, params in markets.items():
    # Price process
    true_price = 100 + np.cumsum(np.random.normal(0, params['volatility'], n_seconds))
    
    # Bid-ask spread dynamics
    base_spread = params['base_spread']
    spread = np.ones(n_seconds) * base_spread
    
    # Order book depth (mean-reverting process)
    depth_target = params['depth_mean']
    bid_depth = np.zeros(n_seconds)
    ask_depth = np.zeros(n_seconds)
    bid_depth[0] = depth_target
    ask_depth[0] = depth_target
    
    # Order flow
    order_times = []
    order_sizes = []
    order_directions = []
    order_prices = []
    
    # Generate orders
    for t in range(n_seconds):
        # Check for order arrival (Poisson process)
        n_orders = np.random.poisson(params['order_arrival_rate'])
        
        for _ in range(n_orders):
            order_times.append(t)
            # Order size (lognormal distribution)
            size = int(np.random.lognormal(np.log(100), 0.8))
            order_sizes.append(size)
            # Direction (buy/sell)
            direction = np.random.choice([-1, 1])
            order_directions.append(direction)
    
    order_times = np.array(order_times)
    order_sizes = np.array(order_sizes)
    order_directions = np.array(order_directions)
    
    # Simulate market impact and recovery
    temporary_impact = np.zeros(n_seconds)
    execution_prices = []
    effective_spreads = []
    execution_times = []
    
    for i, (t, size, direction) in enumerate(zip(order_times, order_sizes, order_directions)):
        if t >= n_seconds:
            continue
            
        # Calculate execution price based on depth
        if direction == 1:  # Buy
            available_depth = ask_depth[t] if t < len(ask_depth) else depth_target
            price_impact = base_spread / 2 + (size / available_depth) * 0.1
        else:  # Sell
            available_depth = bid_depth[t] if t < len(bid_depth) else depth_target
            price_impact = base_spread / 2 + (size / available_depth) * 0.1
        
        exec_price = true_price[t] + direction * price_impact
        execution_prices.append(exec_price)
        execution_times.append(t)
        
        # Effective spread
        eff_spread = 2 * abs(exec_price - true_price[t])
        effective_spreads.append(eff_spread)
        
        # Temporary market impact (decays over time)
        impact_magnitude = (size / depth_target) * 0.5
        for future_t in range(t, min(t + 60, n_seconds)):
            decay = np.exp(-params['replenish_speed'] * (future_t - t))
            temporary_impact[future_t] += direction * impact_magnitude * decay
        
        # Depth consumption and recovery
        if t < len(bid_depth) - 1:
            if direction == 1:  # Buy consumes ask depth
                ask_depth[t] = max(0, ask_depth[t] - size)
            else:  # Sell consumes bid depth
                bid_depth[t] = max(0, bid_depth[t] - size)
    
    # Depth replenishment (mean reversion)
    for t in range(1, n_seconds):
        bid_depth[t] = bid_depth[t-1] + params['replenish_speed'] * (depth_target - bid_depth[t-1])
        ask_depth[t] = ask_depth[t-1] + params['replenish_speed'] * (depth_target - ask_depth[t-1])
        
        # Add noise
        bid_depth[t] += np.random.normal(0, depth_target * 0.1)
        ask_depth[t] += np.random.normal(0, depth_target * 0.1)
        bid_depth[t] = max(0, bid_depth[t])
        ask_depth[t] = max(0, ask_depth[t])
    
    # Calculate liquidity metrics
    
    # 1. Tightness: Mean effective spread
    tightness = np.mean(effective_spreads) if effective_spreads else base_spread
    
    # 2. Depth: Mean order book depth
    depth_metric = (bid_depth.mean() + ask_depth.mean()) / 2
    
    # 3. Immediacy: Order arrival rate (trades per second)
    immediacy = len(order_times) / n_seconds
    
    # 4. Resiliency: Autocorrelation decay of temporary impact
    if len(temporary_impact) > 100:
        impact_acf = np.correlate(temporary_impact[:1000], temporary_impact[:1000], mode='full')
        impact_acf = impact_acf[len(impact_acf)//2:]
        impact_acf = impact_acf / impact_acf[0]
        # Half-life: time for impact to decay to 50%
        try:
            half_life = np.where(impact_acf < 0.5)[0][0] if np.any(impact_acf < 0.5) else 100
        except:
            half_life = 100
        resiliency = 1 / (half_life + 1)  # Higher is better
    else:
        resiliency = 0.01
    
    # Amihud illiquidity measure
    price_changes = np.diff(true_price)
    daily_returns = np.abs(price_changes[::3600]) if len(price_changes) >= 3600 else np.abs(price_changes)
    daily_volume = len(order_times) * np.mean(order_sizes) if order_sizes.size > 0 else 1
    amihud = np.mean(daily_returns) / (daily_volume + 1)
    
    results[market_name] = {
        'true_price': true_price,
        'bid_depth': bid_depth,
        'ask_depth': ask_depth,
        'temporary_impact': temporary_impact,
        'order_times': order_times,
        'execution_prices': np.array(execution_prices),
        'effective_spreads': np.array(effective_spreads),
        'tightness': tightness,
        'depth': depth_metric,
        'immediacy': immediacy,
        'resiliency': resiliency,
        'amihud': amihud
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Order book depth comparison
colors = {'High Liquidity': 'blue', 'Low Liquidity': 'red'}
for market_name, color in colors.items():
    data = results[market_name]
    sample = slice(0, 600)  # First 10 minutes
    time_axis = np.arange(600)
    
    axes[0, 0].plot(time_axis, data['bid_depth'][sample], color=color, alpha=0.7, 
                   linewidth=1, label=f'{market_name} (Bid)')
    axes[0, 0].plot(time_axis, data['ask_depth'][sample], color=color, alpha=0.4, 
                   linewidth=1, linestyle='--', label=f'{market_name} (Ask)')

axes[0, 0].set_title('Order Book Depth Over Time')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Depth (shares)')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Liquidity dimensions radar chart
dimensions = ['Tightness', 'Depth', 'Immediacy', 'Resiliency']
high_liq = results['High Liquidity']
low_liq = results['Low Liquidity']

# Normalize metrics to 0-1 scale (higher is better)
high_scores = [
    1 / (1 + high_liq['tightness'] * 100),  # Lower spread is better
    high_liq['depth'] / 10000,  # Normalize depth
    high_liq['immediacy'],  # Already rate
    high_liq['resiliency'] * 10  # Scale resiliency
]
low_scores = [
    1 / (1 + low_liq['tightness'] * 100),
    low_liq['depth'] / 10000,
    low_liq['immediacy'],
    low_liq['resiliency'] * 10
]

# Normalize to same scale
max_vals = np.maximum(high_scores, low_scores)
high_scores_norm = np.array(high_scores) / (np.array(max_vals) + 0.01)
low_scores_norm = np.array(low_scores) / (np.array(max_vals) + 0.01)

x_pos = np.arange(len(dimensions))
width = 0.35

axes[0, 1].bar(x_pos - width/2, high_scores_norm, width, label='High Liquidity', color='blue', alpha=0.7)
axes[0, 1].bar(x_pos + width/2, low_scores_norm, width, label='Low Liquidity', color='red', alpha=0.7)

axes[0, 1].set_ylabel('Normalized Score (Higher = Better)')
axes[0, 1].set_title('Liquidity Dimensions Comparison')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(dimensions, rotation=15, ha='right')
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1.2)
axes[0, 1].grid(alpha=0.3, axis='y')

# Add value labels
for i, (h, l) in enumerate(zip(high_scores_norm, low_scores_norm)):
    axes[0, 1].text(i - width/2, h + 0.05, f'{h:.2f}', ha='center', fontsize=8)
    axes[0, 1].text(i + width/2, l + 0.05, f'{l:.2f}', ha='center', fontsize=8)

print("Liquidity Metrics Comparison:")
print("=" * 70)
for market_name in ['High Liquidity', 'Low Liquidity']:
    data = results[market_name]
    print(f"\n{market_name}:")
    print(f"  Tightness (Avg Effective Spread): ${data['tightness']:.4f}")
    print(f"  Depth (Avg Book Depth): {data['depth']:.0f} shares")
    print(f"  Immediacy (Orders/second): {data['immediacy']:.3f}")
    print(f"  Resiliency (Recovery rate): {data['resiliency']:.4f}")
    print(f"  Amihud Illiquidity: {data['amihud']:.6f}")

# Plot 3: Price impact and recovery (resiliency)
for market_name, color in colors.items():
    data = results[market_name]
    sample = slice(0, 600)
    time_axis = np.arange(600)
    
    axes[1, 0].plot(time_axis, data['temporary_impact'][sample], color=color, 
                   linewidth=1.5, label=market_name, alpha=0.7)

axes[1, 0].axhline(0, color='black', linewidth=0.5, linestyle='--')
axes[1, 0].set_title('Temporary Price Impact and Recovery')
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('Cumulative Temporary Impact ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Effective spread distribution
high_spreads = results['High Liquidity']['effective_spreads']
low_spreads = results['Low Liquidity']['effective_spreads']

if len(high_spreads) > 0 and len(low_spreads) > 0:
    axes[1, 1].hist(high_spreads, bins=50, alpha=0.6, label='High Liquidity', 
                   color='blue', density=True)
    axes[1, 1].hist(low_spreads, bins=50, alpha=0.6, label='Low Liquidity', 
                   color='red', density=True)
    
    axes[1, 1].axvline(high_spreads.mean(), color='blue', linestyle='--', 
                      linewidth=2, label=f'High Mean: ${high_spreads.mean():.4f}')
    axes[1, 1].axvline(low_spreads.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Low Mean: ${low_spreads.mean():.4f}')

axes[1, 1].set_title('Effective Spread Distribution')
axes[1, 1].set_xlabel('Effective Spread ($)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical comparison
if len(high_spreads) > 0 and len(low_spreads) > 0:
    t_stat, p_val = stats.ttest_ind(high_spreads, low_spreads)
    print(f"\nStatistical Test (Effective Spreads):")
    print(f"High Liquidity: ${high_spreads.mean():.4f} ± ${high_spreads.std():.4f}")
    print(f"Low Liquidity: ${low_spreads.mean():.4f} ± ${low_spreads.std():.4f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.6f}")
    if p_val < 0.001:
        print("Result: Significantly different liquidity levels")

# Liquidity score composite
print(f"\nComposite Liquidity Scores (0-1, higher = more liquid):")
for market_name in ['High Liquidity', 'Low Liquidity']:
    data = results[market_name]
    # Weighted composite
    tightness_score = 1 / (1 + data['tightness'] * 100)
    depth_score = min(data['depth'] / 10000, 1)
    immediacy_score = min(data['immediacy'] / 2, 1)
    resiliency_score = min(data['resiliency'] * 10, 1)
    
    composite = 0.3 * tightness_score + 0.3 * depth_score + 0.2 * immediacy_score + 0.2 * resiliency_score
    print(f"{market_name}: {composite:.3f}")
```

## 6. Challenge Round
How do liquidity dimensions interact and why can't one measure capture everything?
- **Tightness-Depth trade-off**: Penny stocks have wide spreads but "deep" book (100 shares at each level still small $). Measures must scale by price and value
- **Flash crashes reveal**: Markets appear liquid (tight spreads) until tested by large order, then depth vanishes instantly (resiliency fails)
- **Immediacy illusion**: HFT quotes update in microseconds (appears immediate) but may be phantom liquidity (canceled before execution)
- **Temporal variation**: Liquidity highest mid-day, lowest at open/close. Single snapshot misleading without time-series context
- **Comprehensive assessment**: Amihud, Roll, Hasbrouck measures combine dimensions but still miss nuances (e.g., quote flickering, hidden liquidity)

## 7. Key References
- [Kyle (1985) - Continuous Auctions and Insider Trading](https://www.jstor.org/stable/1913210)
- [Harris (1990) - Liquidity, Trading Rules, and Electronic Trading Systems](https://www.jstor.org/stable/2962100)
- [Amihud (2002) - Illiquidity and Stock Returns](https://www.sciencedirect.com/science/article/abs/pii/S0304405X01000726)
- [Foucault et al (2013) - Market Liquidity: Theory, Evidence, Policy](https://www.amazon.com/Market-Liquidity-Trading-Regulation-Thierry/dp/0190844469)

---
**Status:** Foundational liquidity framework | **Complements:** Order Book Depth, Liquidity Measures, Market Impact
