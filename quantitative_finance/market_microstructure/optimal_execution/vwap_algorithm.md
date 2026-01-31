# VWAP Algorithm

## 1. Concept Skeleton
**Definition:** Volume-Weighted Average Price algorithm executing orders to match historical intraday volume distribution pattern, minimizing market impact by trading with natural flow  
**Purpose:** Benchmark execution strategy tracking average market price weighted by volume, commonly used for large institutional orders  
**Prerequisites:** Historical volume profiles, order scheduling logic, real-time volume monitoring, trade execution infrastructure

## 2. Comparative Framing
| Algorithm | VWAP | TWAP | POV | Arrival Price |
|-----------|------|------|-----|---------------|
| **Pacing** | Volume-based | Time-based | Real-time volume % | Immediate |
| **Tracking** | Historical pattern | Uniform | Live market | Arrival snapshot |
| **Stealth** | High | Moderate | Low | None |
| **Adaptability** | Pre-determined | None | Reactive | N/A |

## 3. Examples + Counterexamples

**Simple Example:**  
Historical data shows 30% of daily volume trades in first hour. VWAP algo for 10,000 shares executes 3,000 shares in first hour, matching pattern

**Failure Case:**  
News breaks mid-day causing volume spike. VWAP follows historical profile, trades too little during spike, misses better prices, finishes above VWAP benchmark

**Edge Case:**  
Market close auction has 15% of daily volume. VWAP strategy must decide: participate aggressively (impact) or skip (tracking error)

## 4. Layer Breakdown
```
VWAP Algorithm Architecture:
├─ Volume Profile Estimation:
│   ├─ Historical Analysis:
│   │   ├─ Last N days (typically 10-20 days)
│   │   ├─ Exclude outliers (earnings days, expiration)
│   │   ├─ Seasonal adjustments (Monday/Friday patterns)
│   │   └─ Liquidity regime filtering
│   ├─ Intraday Volume Curve:
│   │   ├─ U-shaped pattern (high open/close)
│   │   ├─ Typical: 30% first hour, 25% last hour
│   │   ├─ Lunch hour slowest (11:30-13:30 ET)
│   │   └─ Time buckets (5min, 15min, 30min)
│   ├─ Day-of-Week Effects:
│   │   ├─ Monday: Higher volume (weekend news)
│   │   ├─ Friday: Portfolio rebalancing
│   │   └─ Mid-week: Lower volatility/volume
│   └─ Event Adjustments:
│       ├─ Earnings announcements (higher volume)
│       ├─ FOMC days (elevated activity)
│       ├─ Index rebalance (abnormal flows)
│       └─ Dividend ex-dates
├─ Trade Scheduling:
│   ├─ Target Shares per Interval:
│   │   ├─ Total_shares × Volume_profile[t]
│   │   ├─ Example: 10,000 shares × 8% = 800 shares
│   │   ├─ Rounded to lot sizes
│   │   └─ Carried over if unfilled
│   ├─ Order Slicing:
│   │   ├─ Break interval target into child orders
│   │   ├─ Avoid large prints (>10% of interval volume)
│   │   ├─ Randomization to reduce predictability
│   │   └─ Typical slice: 100-500 shares per order
│   ├─ Participation Rate Limits:
│   │   ├─ Max % of market volume (e.g., 10-20%)
│   │   ├─ Prevents excessive footprint
│   │   ├─ Dynamic adjustment if behind schedule
│   │   └─ Circuit breaker on low liquidity
│   └─ Catch-up Logic:
│   │   ├─ If behind schedule, increase aggression
│   │   ├─ Trade more in next interval
│   │   ├─ Switch from passive to aggressive orders
│   │   └─ Final interval: market orders for remainder
├─ Execution Tactics:
│   ├─ Passive Strategies:
│   │   ├─ Post limit orders at/inside touch
│   │   ├─ Join queue at best bid/ask
│   │   ├─ Patient fill accumulation
│   │   └─ Lower impact but execution risk
│   ├─ Aggressive Strategies:
│   │   ├─ Take liquidity with market orders
│   │   ├─ Cross spread immediately
│   │   ├─ Higher certainty but more cost
│   │   └─ Used when behind schedule
│   ├─ Smart Order Routing:
│   │   ├─ Route to venue with best price
│   │   ├─ Consider maker-taker fees
│   │   ├─ Dark pool access for large slices
│   │   └─ Minimize information leakage
│   └─ Adaptive Tactics:
│       ├─ Monitor queue position
│       ├─ Cancel-replace if quote moves away
│       ├─ Adjust aggressiveness based on progress
│       └─ IOC orders to test liquidity
├─ Real-Time Monitoring:
│   ├─ Volume Tracking:
│   │   ├─ Compare actual vs expected volume
│   │   ├─ Detect anomalies (volume spike/drought)
│   │   ├─ Adjust participation if necessary
│   │   └─ Alert on significant deviation (>30%)
│   ├─ Fill Rate Analysis:
│   │   ├─ Monitor execution progress
│   │   ├─ Passive fill ratio
│   │   ├─ Aggressive fill ratio
│   │   └─ Adjust tactics if lagging
│   ├─ Price Performance:
│   │   ├─ Track intraday VWAP benchmark
│   │   ├─ Compare our average price
│   │   ├─ Slippage attribution (impact vs timing)
│   │   └─ Real-time P&L calculation
│   └─ Risk Controls:
│       ├─ Position limits (don't over-execute)
│       ├─ Price collar (reject bad fills)
│       ├─ Volume spike detection (pause if needed)
│       └─ Manual override capability
├─ VWAP Calculation:
│   ├─ Market VWAP:
│   │   ├─ VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)
│   │   ├─ Sum over all trades in period
│   │   ├─ Typically 9:30-16:00 ET for US equities
│   │   └─ Exchange-provided or calculated from tape
│   ├─ Execution VWAP:
│   │   ├─ Our average fill price
│   │   ├─ Weighted by fill sizes
│   │   ├─ Excludes fees/commissions
│   │   └─ Compared to market VWAP
│   ├─ Performance Metrics:
│   │   ├─ VWAP difference: Execution VWAP - Market VWAP
│   │   ├─ Expressed in bps: (diff / market VWAP) × 10000
│   │   ├─ Negative = better than benchmark
│   │   └─ Positive = worse than benchmark
│   └─ Attribution:
│       ├─ Timing: When we traded vs volume curve
│       ├─ Impact: Price pressure from our orders
│       ├─ Spread: Bid-ask crossing costs
│       └─ Adverse selection: Information content
├─ Variants & Extensions:
│   ├─ TVOL (Target Volume):
│   │   ├─ User-specified volume profile
│   │   ├─ Overrides historical pattern
│   │   ├─ Strategic timing (avoid certain periods)
│   │   └─ Risk management considerations
│   ├─ VWAP+ (Enhanced):
│   │   ├─ Incorporates alpha signals
│   │   ├─ Adjust schedule if directional view
│   │   ├─ Faster execution if bullish (buying)
│   │   └─ Requires risk management overlay
│   ├─ Dark Pool VWAP:
│   │   ├─ Execute in dark venues
│   │   ├─ Reference lit VWAP as benchmark
│   │   ├─ Less market impact
│   │   └─ Execution uncertainty higher
│   └─ Multi-Day VWAP:
│       ├─ Extend horizon beyond single day
│       ├─ Very large orders (>10 days ADV)
│       ├─ Coordination across days
│       └─ Compounding of tracking error risk
└─ Implementation Considerations:
    ├─ Data Requirements:
    │   ├─ Real-time market data feeds
    │   ├─ Historical volume database
    │   ├─ Execution reports (fill confirmations)
    │   └─ Time synchronization (critical for VWAP calc)
    ├─ Infrastructure:
    │   ├─ Low-latency execution system
    │   ├─ Smart order router
    │   ├─ Risk management pre-trade checks
    │   └─ Post-trade TCA (transaction cost analysis)
    ├─ Regulatory Compliance:
    │   ├─ MiFID II best execution
    │   ├─ SEC Rule 606 (order routing disclosure)
    │   ├─ Reg NMS (trade-through protection)
    │   └─ Audit trail for all decisions
    └─ Performance Monitoring:
        ├─ Daily VWAP performance reports
        ├─ Slippage analysis by symbol/strategy
        ├─ Hit rate (% orders beating VWAP)
        └─ Client reporting (quarterly TCA)
```

**Interaction:** Volume profile estimation → Trade scheduling → Order execution → Real-time adjustment → Performance measurement

## 5. Mini-Project
Implement VWAP algorithm with realistic market simulation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

class VolumeProfile:
    """Model historical intraday volume pattern"""
    
    def __init__(self, intervals=78):  # 6.5 hours / 5 min = 78 intervals
        self.intervals = intervals
        
        # Generate realistic U-shaped volume curve
        self.profile = self._generate_profile()
    
    def _generate_profile(self):
        """
        Create U-shaped intraday volume curve
        
        Higher volume at open/close, lower midday
        """
        t = np.linspace(0, 1, self.intervals)
        
        # U-shape: high at 0 and 1, low at 0.5
        base = 0.5 + 1.5 * (2*t - 1)**2
        
        # Add some noise
        noise = np.random.uniform(0.9, 1.1, self.intervals)
        
        profile = base * noise
        
        # Normalize to sum to 1
        profile = profile / profile.sum()
        
        return profile
    
    def get_target_shares(self, total_shares, interval):
        """Get target shares for specific interval"""
        return int(total_shares * self.profile[interval])

class Market:
    """Simulate market with realistic characteristics"""
    
    def __init__(self, S0=100, volatility=0.30, spread=0.01):
        self.S0 = S0
        self.volatility = volatility
        self.spread = spread
        
        # Current state
        self.mid_price = S0
        self.bid = S0 - spread/2
        self.ask = S0 + spread/2
        
        # Volume state
        self.total_volume = 0
        self.interval_volume = 0
        
        # VWAP calculation
        self.vwap_numerator = 0
        self.vwap_denominator = 0
    
    def evolve(self, dt, external_volume=0, our_volume=0, side='buy'):
        """
        Evolve market state
        
        Price moves due to:
        1. Natural volatility
        2. Our market impact
        """
        # Natural price evolution
        shock = self.volatility / np.sqrt(252*78) * np.random.normal()
        self.mid_price += self.mid_price * shock
        
        # Market impact from our trades
        if our_volume > 0:
            # Permanent impact (proportional to volume)
            impact = 0.0001 * our_volume * (1 if side == 'buy' else -1)
            self.mid_price += impact * self.mid_price
        
        # Update quotes
        self.bid = self.mid_price - self.spread/2
        self.ask = self.mid_price + self.spread/2
        
        # Update volume
        self.interval_volume = external_volume + our_volume
        self.total_volume += self.interval_volume
        
        # Update VWAP
        trade_price = self.ask if side == 'buy' else self.bid
        if our_volume > 0:
            self.vwap_numerator += trade_price * our_volume
            self.vwap_denominator += our_volume
        
        # Market VWAP (approximate with mid-price)
        # In reality would track all market trades
        return trade_price
    
    def get_vwap(self):
        """Get current market VWAP"""
        if self.vwap_denominator == 0:
            return self.mid_price
        return self.vwap_numerator / self.vwap_denominator

class VWAPAlgorithm:
    """VWAP execution algorithm"""
    
    def __init__(self, total_shares, side='buy', participation_rate=0.15,
                 max_slice_size=500):
        self.total_shares = total_shares
        self.side = side
        self.participation_rate = participation_rate  # Max % of interval volume
        self.max_slice_size = max_slice_size
        
        # State
        self.shares_executed = 0
        self.shares_remaining = total_shares
        self.execution_log = []
        
        # Performance tracking
        self.total_cost = 0
    
    def execute_interval(self, market: Market, target_shares, interval_idx):
        """
        Execute trades in one time interval
        
        Returns: shares executed, average price
        """
        if self.shares_remaining == 0:
            return 0, 0
        
        # Don't exceed target or remaining
        shares_to_execute = min(target_shares, self.shares_remaining)
        
        # Estimate market volume this interval (random)
        expected_market_volume = np.random.uniform(5000, 20000)
        
        # Respect participation rate limit
        max_allowed = int(expected_market_volume * self.participation_rate)
        shares_to_execute = min(shares_to_execute, max_allowed)
        
        if shares_to_execute == 0:
            return 0, 0
        
        # Break into slices
        num_slices = max(1, shares_to_execute // self.max_slice_size)
        slice_size = shares_to_execute // num_slices
        
        fills = []
        
        for i in range(num_slices):
            # Execute slice
            size = slice_size
            if i == num_slices - 1:
                # Last slice gets remainder
                size = shares_to_execute - sum([f[0] for f in fills])
            
            # Simulate fill (mix of passive and aggressive)
            if np.random.random() < 0.7:
                # Passive fill (at bid for seller, ask for buyer)
                price = market.ask if self.side == 'buy' else market.bid
            else:
                # Aggressive take (cross spread)
                price = market.bid if self.side == 'buy' else market.ask
            
            fills.append((size, price))
            
            # Update market
            market.evolve(dt=1.0/num_slices, external_volume=expected_market_volume/num_slices,
                          our_volume=size, side=self.side)
        
        # Aggregate fills
        total_filled = sum([f[0] for f in fills])
        avg_price = sum([f[0]*f[1] for f in fills]) / total_filled if total_filled > 0 else 0
        
        # Update state
        self.shares_executed += total_filled
        self.shares_remaining -= total_filled
        self.total_cost += avg_price * total_filled
        
        self.execution_log.append({
            'interval': interval_idx,
            'target': target_shares,
            'executed': total_filled,
            'avg_price': avg_price,
            'remaining': self.shares_remaining,
            'market_mid': market.mid_price
        })
        
        return total_filled, avg_price
    
    def get_execution_vwap(self):
        """Calculate our execution VWAP"""
        if self.shares_executed == 0:
            return 0
        return self.total_cost / self.shares_executed

def simulate_vwap_execution(total_shares=10000, intervals=78, side='buy'):
    """Run full VWAP simulation"""
    np.random.seed(42)
    
    # Initialize components
    volume_profile = VolumeProfile(intervals=intervals)
    market = Market(S0=100, volatility=0.30, spread=0.01)
    vwap_algo = VWAPAlgorithm(total_shares=total_shares, side=side, 
                               participation_rate=0.15, max_slice_size=500)
    
    # Track market VWAP
    market_trades = []
    
    # Execute over all intervals
    for i in range(intervals):
        # Get target for this interval
        target = volume_profile.get_target_shares(total_shares, i)
        
        # Execute
        filled, price = vwap_algo.execute_interval(market, target, i)
        
        # Simulate other market activity
        external_volume = np.random.uniform(5000, 20000)
        market_trades.append({
            'interval': i,
            'volume': external_volume + filled,
            'price': market.mid_price
        })
    
    # Calculate market VWAP (simplified: volume-weighted mid-prices)
    df_market = pd.DataFrame(market_trades)
    market_vwap = (df_market['price'] * df_market['volume']).sum() / df_market['volume'].sum()
    
    # Get our execution VWAP
    execution_vwap = vwap_algo.get_execution_vwap()
    
    df_execution = pd.DataFrame(vwap_algo.execution_log)
    
    return df_execution, {
        'execution_vwap': execution_vwap,
        'market_vwap': market_vwap,
        'slippage_bps': (execution_vwap - market_vwap) / market_vwap * 10000,
        'total_executed': vwap_algo.shares_executed,
        'volume_profile': volume_profile.profile,
        'market_prices': df_market
    }

# Run simulation
print("="*80)
print("VWAP ALGORITHM SIMULATION")
print("="*80)

df_exec, metrics = simulate_vwap_execution(total_shares=10000, intervals=78, side='buy')

print(f"\nExecution Summary:")
print(f"  Total shares: {metrics['total_executed']:,}")
print(f"  Execution VWAP: ${metrics['execution_vwap']:.4f}")
print(f"  Market VWAP: ${metrics['market_vwap']:.4f}")
print(f"  Slippage: {metrics['slippage_bps']:.2f} bps")
print(f"  Performance: {'BEAT' if metrics['slippage_bps'] < 0 else 'MISS'} benchmark")

print(f"\nInterval Statistics:")
print(f"  Average target per interval: {df_exec['target'].mean():.0f} shares")
print(f"  Average executed per interval: {df_exec['executed'].mean():.0f} shares")
print(f"  Fill rate: {df_exec['executed'].sum() / df_exec['target'].sum() * 100:.1f}%")

# Analyze by time period
df_exec['period'] = pd.cut(df_exec['interval'], bins=3, labels=['Open', 'Mid', 'Close'])
period_stats = df_exec.groupby('period').agg({
    'executed': 'sum',
    'avg_price': 'mean'
}).round(2)

print(f"\nExecution by Period:")
print(period_stats)

# Compare strategies: VWAP vs TWAP vs Immediate
print("\n" + "="*80)
print("STRATEGY COMPARISON")
print("="*80)

# VWAP (already done)
vwap_slippage = metrics['slippage_bps']

# TWAP simulation (uniform execution)
np.random.seed(42)
market_twap = Market(S0=100, volatility=0.30, spread=0.01)
uniform_shares = metrics['total_executed'] // 78
twap_cost = 0

for i in range(78):
    price = market_twap.ask  # Assume buying
    twap_cost += price * uniform_shares
    market_twap.evolve(dt=1, external_volume=10000, our_volume=uniform_shares, side='buy')

twap_vwap = twap_cost / metrics['total_executed']
twap_slippage = (twap_vwap - metrics['market_vwap']) / metrics['market_vwap'] * 10000

# Immediate execution (all at open)
immediate_price = 100  # Initial price
immediate_slippage = (immediate_price - metrics['market_vwap']) / metrics['market_vwap'] * 10000

comparison = pd.DataFrame({
    'Strategy': ['VWAP', 'TWAP', 'Immediate'],
    'Avg_Price': [metrics['execution_vwap'], twap_vwap, immediate_price],
    'Slippage_bps': [vwap_slippage, twap_slippage, immediate_slippage]
})

print(comparison.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Volume profile
axes[0, 0].bar(range(78), metrics['volume_profile']*100, alpha=0.7)
axes[0, 0].set_title('Historical Volume Profile')
axes[0, 0].set_xlabel('5-min Interval')
axes[0, 0].set_ylabel('% of Daily Volume')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Execution progress
cumulative = df_exec['executed'].cumsum()
cumulative_target = df_exec['target'].cumsum()
axes[0, 1].plot(cumulative, label='Actual', linewidth=2)
axes[0, 1].plot(cumulative_target, label='Target', linestyle='--', linewidth=2)
axes[0, 1].set_title('Execution Progress')
axes[0, 1].set_xlabel('Interval')
axes[0, 1].set_ylabel('Cumulative Shares')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price path
axes[0, 2].plot(metrics['market_prices']['price'], label='Market price', linewidth=1.5)
axes[0, 2].scatter(df_exec['interval'], df_exec['avg_price'], 
                   c='red', s=30, label='Our fills', zorder=5)
axes[0, 2].axhline(metrics['market_vwap'], color='green', linestyle='--', 
                   label=f'Market VWAP: ${metrics["market_vwap"]:.2f}')
axes[0, 2].axhline(metrics['execution_vwap'], color='orange', linestyle='--',
                   label=f'Our VWAP: ${metrics["execution_vwap"]:.2f}')
axes[0, 2].set_title('Price Path & Execution')
axes[0, 2].set_xlabel('Interval')
axes[0, 2].set_ylabel('Price ($)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Execution size per interval
axes[1, 0].bar(df_exec['interval'], df_exec['executed'], alpha=0.7)
axes[1, 0].set_title('Shares Executed per Interval')
axes[1, 0].set_xlabel('Interval')
axes[1, 0].set_ylabel('Shares')
axes[1, 0].grid(alpha=0.3)

# Plot 5: Target vs actual
axes[1, 1].scatter(df_exec['target'], df_exec['executed'], alpha=0.5)
axes[1, 1].plot([0, df_exec['target'].max()], [0, df_exec['target'].max()], 
                'r--', label='Perfect execution')
axes[1, 1].set_title('Target vs Actual Execution')
axes[1, 1].set_xlabel('Target Shares')
axes[1, 1].set_ylabel('Executed Shares')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Strategy comparison
strategies = ['VWAP', 'TWAP', 'Immediate']
slippages = [vwap_slippage, twap_slippage, immediate_slippage]
colors = ['green' if s < 0 else 'red' for s in slippages]

axes[1, 2].barh(strategies, slippages, color=colors, alpha=0.7)
axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 2].set_title('Slippage Comparison')
axes[1, 2].set_xlabel('Slippage (bps)')
axes[1, 2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY FINDINGS")
print(f"{'='*80}")
print(f"\n1. VWAP algorithm tracks historical volume pattern, blends with market flow")
print(f"2. Lower market impact than aggressive (TWAP) or immediate execution")
print(f"3. Performance depends on whether actual volume matches historical profile")
print(f"4. U-shaped volume curve: most execution at open/close")
print(f"5. Participation rate limits prevent excessive footprint (typ. 10-20%)")
print(f"6. VWAP benchmark industry standard for institutional TCA")
```

## 6. Challenge Round
When does VWAP strategy underperform?
- **Volume anomalies**: Actual volume deviates significantly from historical (news events, index rebalance)
- **Directional markets**: Strong trend means early/late execution critical, VWAP neutral timing suboptimal
- **Low liquidity**: Participation rate limits may prevent completion
- **Gaming**: Other traders detect VWAP pattern, front-run execution
- **Close auction**: Large closing auction volume creates benchmark distortion

What are alternatives to standard VWAP?
- **TVOL**: Target specific volume profile, not historical
- **Adaptive VWAP**: Adjust schedule based on realized volume vs expected
- **POV**: Percentage-of-volume, tracks real-time market activity
- **Dark pool VWAP**: Execute in dark venues to reduce information leakage
- **Arrival price**: Don't care about benchmark, minimize cost vs decision time

## 7. Key References
- [Kissell, Glantz (2003): Optimal Trading Strategies](https://www.wiley.com/en-us/Optimal+Trading+Strategies-p-9780814407806)
- [Berkowitz, Logue, Noser (1988): The Total Cost of Transactions](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1988.tb02585.x)
- [Madhavan (2002): VWAP Strategies](https://www.cfapubs.org/doi/pdf/10.2469/faj.v58.n2.2524)

---
**Status:** Industry standard benchmark algorithm | **Complements:** TWAP, POV, Implementation Shortfall, TCA
