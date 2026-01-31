# Time & Sales Data

## 1. Concept Skeleton
**Definition:** Record of executed trades only (no quotes); includes timestamp, price, volume, and trade conditions; subset of tick data  
**Purpose:** Analyze transaction flow; measure trading activity; identify institutional orders; volume-weighted analysis; simpler than full tick data  
**Prerequisites:** Trade execution, market orders, trade reporting, volume analysis, time series basics

## 2. Comparative Framing
| Data Type | Content | Size (per day) | Use Case | Completeness |
|-----------|---------|----------------|----------|--------------|
| **Time & Sales** | Trades only | 10-50 GB | Volume analysis, transactions | Executions only |
| **Tick Data (Full)** | Trades + Quotes | 100-500 GB | Microstructure, spreads | Complete market state |
| **Level 2 Data** | Order book depth | 200+ GB | Liquidity, depth analysis | Top N levels |
| **OHLCV Bars** | Aggregated summary | <1 GB | Price charts, backtesting | Time-aggregated |
| **Quote Data Only** | Bid-ask quotes | 50-200 GB | Spread analysis | No volume info |

## 3. Examples + Counterexamples

**Time & Sales Entry:**  
```
Timestamp: 09:30:15.234567 | Symbol: AAPL | Price: $150.25 | Volume: 500 | Condition: @ (regular)
```

**Not Included (Quotes):**  
```
09:30:15.123456 | AAPL | Bid: $150.24 | Ask: $150.25 | Sizes: 800 x 1200 ← Missing from Time & Sales
```

**Block Trade:**  
```
15:45:32.567890 | MSFT | Price: $325.50 | Volume: 50,000 | Condition: T (late reported, Form-T)
```
Indicates institutional transaction, negotiated off-exchange, reported later

**Retail Trade:**  
```
10:15:22.123456 | TSLA | Price: $180.00 | Volume: 10 | Condition: @ (regular)
```
Small size suggests retail trader, executed at market

## 4. Layer Breakdown
```
Time & Sales Data Framework:
├─ Core Components:
│   ├─ Timestamp:
│   │   - Exchange time: When trade executed
│   │   - Precision: Millisecond to nanosecond
│   │   - Time zone: Typically exchange local or UTC
│   │   - Synchronization: Critical for cross-market analysis
│   │   - Sequence: May include sequence number for ordering
│   ├─ Symbol Identifier:
│   │   - Ticker: AAPL, MSFT, SPY
│   │   - Exchange suffix: AAPL.O (NASDAQ), AAPL.N (NYSE)
│   │   - CUSIP/ISIN: Alternative identifiers
│   │   - Asset class: Equity, option, future, FX
│   ├─ Price:
│   │   - Execution price: Actual transaction price
│   │   - Tick size: Constrained by minimum increment
│   │   - Currency: USD, EUR, etc.
│   │   - Adjustments: May need split/dividend adjustment
│   ├─ Volume:
│   │   - Number of shares: Quantity traded
│   │   - Block indicator: If > threshold (e.g., 10,000)
│   │   - Odd lot: <100 shares (reported separately)
│   │   - Round lot: Multiples of 100 (standard)
│   └─ Trade Conditions:
│       - Regular sale (@): Normal transaction
│       - Late report (T, Z): Form-T (reported after execution)
│       - Cancelled (C, B): Bust/error correction
│       - Opening (O): Opening auction
│       - Closing (6): Closing auction
│       - Intermarket sweep (F): ISO order
│       - Extended hours (U): Pre/post-market
│       - Dozens more: Exchange-specific codes
│
├─ Trade Condition Codes (Selected):
│   ├─ Regular Trades:
│   │   - @: Regular sale (most common)
│   │   - *: Corrected last sale
│   │   - I: Odd lot (< 100 shares)
│   │   - M: Market center official close
│   ├─ Special Conditions:
│   │   - F: Intermarket sweep order (ISO)
│   │   - O: Opening prints (auction)
│   │   - 6: Closing prints (auction)
│   │   - 4: Derivatively priced (e.g., dark pool reference)
│   ├─ Out-of-Sequence:
│   │   - T: Form-T (late reported, within 90 seconds)
│   │   - U: Extended hours trade (pre/post market)
│   │   - Z: Sold (out of sequence, reported late)
│   │   - Implication: May distort intraday analysis
│   ├─ Errors & Cancellations:
│   │   - B: Bunched sold trade (error correction)
│   │   - C: Cash sale (next day settlement)
│   │   - X: Trade cancel (bust by exchange)
│   │   - Handle: Exclude from analysis or flag
│   └─ Price Types:
│       - 1: Stopped stock (guaranteed price)
│       - 2: Average price trade (VWAP)
│       - 3: Cash trade (same-day settlement)
│       - Rare: Most trades are regular (@)
│
├─ Data Sources & Reporting:
│   ├─ Trade Reporting Facilities (TRF):
│   │   - FINRA TRF: Reports off-exchange trades
│   │   - ADF: Alternative Display Facility
│   │   - Timing: Within 10 seconds of execution
│   │   - Coverage: Dark pools, internalization
│   ├─ Exchange Direct Feeds:
│   │   - NYSE TAQ: Time & Sales for NYSE
│   │   - NASDAQ: TotalView includes time & sales
│   │   - Latency: Microseconds to milliseconds
│   │   - Cost: $1,000-$10,000+/month per exchange
│   ├─ Consolidated Tape:
│   │   - CTS (Tape A, B): NYSE, regional exchanges
│   │   - UTP (Tape C): NASDAQ-listed
│   │   - SIP: Securities Information Processor
│   │   - Latency: Higher (aggregation delay)
│   │   - Advantage: Single feed for all venues
│   ├─ Data Vendors:
│   │   - Bloomberg Terminal: Real-time time & sales
│   │   - Refinitiv: Historical tick-by-tick
│   │   - Polygon.io: API-based, retail-friendly
│   │   - Quandl/Nasdaq Data Link: Affordable academic
│   └─ Regulatory Reporting:
│       - CAT: Consolidated Audit Trail (comprehensive)
│       - OATS: Order Audit Trail System (FINRA)
│       - Purpose: Surveillance, enforcement
│       - Access: Restricted to regulators
│
├─ Analysis Techniques:
│   ├─ Volume Profiling:
│   │   - Cumulative volume: Track total shares traded
│   │   - Volume distribution: By price level (volume-at-price)
│   │   - Time-of-day: Intraday volume patterns (U-shaped)
│   │   - VWAP: Volume-weighted average price
│   │   - Use: Identify support/resistance, benchmark execution
│   ├─ Trade Size Analysis:
│   │   - Small trades (<100 shares): Retail
│   │   - Medium (100-1,000): Mixed retail/institutional
│   │   - Large (1,000-10,000): Institutional algos
│   │   - Block (>10,000): Institutional block desks
│   │   - Distribution: Heavily right-skewed (power law)
│   ├─ Trade Direction Classification:
│   │   - Lee-Ready algorithm: Compare to mid-quote (needs quote data)
│   │   - Tick test: Compare to previous trade price
│   │   - Uptick (+): Price higher than previous
│   │   - Downtick (-): Price lower than previous
│   │   - Zero-tick: Same price (use previous direction)
│   │   - Application: Measure buying vs selling pressure
│   ├─ Price Impact Measurement:
│   │   - Immediate: Price change after trade
│   │   - Formula: Impact = (Price_after - Price_before) × Direction
│   │   - Larger trades → larger impact (nonlinear, √ law)
│   │   - Use: Transaction cost analysis, optimal execution
│   ├─ Volume-Weighted Metrics:
│   │   - VWAP: Σ(Price × Volume) / Σ(Volume)
│   │   - Benchmark: Institutional execution quality
│   │   - TWAP: Time-weighted (uniform over time)
│   │   - Difference: VWAP follows volume, TWAP ignores it
│   ├─ Liquidity Indicators:
│   │   - Trade frequency: Trades per minute
│   │   - Average trade size: Larger = more institutional
│   │   - Trade clustering: Bursts of activity (information)
│   │   - Gaps: Periods without trades (illiquidity)
│   ├─ Intraday Patterns:
│   │   - U-shaped volume: High at open, low midday, high at close
│   │   - Volatility: Mirrors volume (high at open/close)
│   │   - Spread: Wider at open/close (uncertainty, imbalance)
│   │   - Application: Optimal execution timing (avoid open/close)
│   └─ Abnormal Activity Detection:
│       - Volume spike: >3σ above average
│       - Price spike: >5% move in <1 minute
│       - Large block: >10,000 shares in single trade
│       - Rapid trades: >100 trades/minute (HFT, news)
│       - Trigger: Alerts for surveillance, trading signals
│
├─ Institutional vs Retail Patterns:
│   ├─ Institutional Signatures:
│   │   - Trade size: 1,000-50,000 shares (large)
│   │   - Frequency: Steady, algorithmic pacing
│   │   - Timing: Avoid open/close (minimize impact)
│   │   - Price: Often passive (limit orders, VWAP)
│   │   - Detection: Clustering of similar-sized trades
│   ├─ Retail Signatures:
│   │   - Trade size: 1-100 shares (small, odd lots)
│   │   - Frequency: Irregular, impulsive
│   │   - Timing: Concentrated at open (morning activity)
│   │   - Price: Market orders (pay spread for immediacy)
│   │   - Behavior: Momentum chasing, news-driven
│   ├─ Algorithmic Signatures:
│   │   - Trade size: Uniform (e.g., 100 shares repeatedly)
│   │   - Frequency: Highly regular (every 30 seconds)
│   │   - VWAP/TWAP: Tracks benchmark closely
│   │   - Detection: Autocorrelation in trade times
│   ├─ Block Trading:
│   │   - Size: >10,000 shares
│   │   - Execution: Often negotiated off-exchange (dark pools)
│   │   - Reporting: Form-T (late reported, condition T)
│   │   - Price: Minimal impact (negotiated, hidden)
│   │   - Implication: May not reflect true supply/demand
│   └─ High-Frequency Trading:
│       - Trade size: 10-100 shares (tiny)
│       - Frequency: Hundreds/thousands per second
│       - Roundtrips: Buy and sell within milliseconds
│       - Price: Typically at bid/ask (market making)
│       - Detection: Extreme frequency, small size
│
├─ Data Quality & Cleaning:
│   ├─ Late Reported Trades:
│   │   - Condition T: Reported within 90 seconds
│   │   - Condition Z: Reported after 90 seconds
│   │   - Problem: Distorts intraday price/volume
│   │   - Solution: Exclude or create separate series
│   ├─ Trade Cancellations (Busts):
│   │   - Condition X, B: Erroneous trade corrected
│   │   - Reason: Fat finger, system error, manipulation
│   │   - Frequency: Rare (<0.01% of trades)
│   │   - Handling: Remove from historical data
│   ├─ Odd Lots (<100 shares):
│   │   - Historically unreported: Pre-2013
│   │   - Now included: SEC rule change
│   │   - Impact: ~20-30% of trade count, ~2% of volume
│   │   - Analysis: Include for completeness
│   ├─ Pre/Post Market:
│   │   - Condition U: Extended hours
│   │   - Hours: 4am-9:30am, 4pm-8pm ET
│   │   - Volume: ~5% of regular session
│   │   - Characteristics: Wider spreads, lower liquidity
│   │   - Handling: Analyze separately or exclude
│   ├─ Price Outliers:
│   │   - Fat fingers: Erroneous price entry
│   │   - Detection: >10σ from recent prices
│   │   - Handling: Remove or cap at reasonable bound
│   │   - Example: $100 stock trades at $1,000 (error)
│   └─ Missing Data:
│       - Gaps: System outages, halts
│       - Duration: Minutes to hours (rare)
│       - Imputation: Not recommended (preserve gaps)
│       - Analysis: Flag periods for exclusion
│
├─ Practical Applications:
│   ├─ Transaction Cost Analysis (TCA):
│   │   - Benchmark: Compare execution to VWAP
│   │   - Slippage: Difference from expected price
│   │   - Timing: When trades executed (spread cost)
│   │   - Venue: Which exchange provided best price
│   ├─ Algorithmic Trading:
│   │   - Signal generation: Volume spikes, imbalances
│   │   - Execution: Monitor own trades vs market
│   │   - Risk management: Real-time P&L from trades
│   │   - Backtesting: Realistic fills using time & sales
│   ├─ Market Surveillance:
│   │   - Manipulation detection: Spoofing, layering
│   │   - Insider trading: Abnormal volume before news
│   │   - Front-running: Pattern of trades ahead of blocks
│   │   - Regulatory: FINRA, SEC use for enforcement
│   ├─ Academic Research:
│   │   - Price discovery: Which venue leads
│   │   - Liquidity measurement: Trade frequency, size
│   │   - Market microstructure: Informed vs uninformed
│   │   - Event studies: Reaction to news, earnings
│   └─ Retail Trading:
│       - Real-time tape: Monitor market activity
│       - Order flow: See institutional buying/selling
│       - Execution quality: Compare own fills to tape
│       - Education: Learn market dynamics visually
│
└─ Limitations & Caveats:
    ├─ No Quote Information:
    │   - Can't see bid-ask spread: Need quote data
    │   - Can't classify trades: Need Lee-Ready (mid-quote)
    │   - Can't measure liquidity: Need depth data
    │   - Workaround: Merge with quote data (full tick)
    ├─ Late Reporting Distortion:
    │   - Form-T trades: Appear minutes later
    │   - Impact: VWAP, volume profiles distorted
    │   - Magnitude: ~10-20% of volume (dark pools)
    │   - Solution: Filter by condition code
    ├─ Exchange Fragmentation:
    │   - Multiple venues: Need consolidated tape
    │   - Latency differences: SIP vs direct feeds
    │   - Data quality: Varies by venue
    │   - Cost: Higher for complete coverage
    ├─ Privacy & Aggregation:
    │   - No participant IDs: Can't identify traders
    │   - Aggregation: Can't see order intent
    │   - Algorithms: Difficult to detect strategies
    │   - Limitation: Less informative than order-level
    └─ Historical Consistency:
        - Rule changes: Odd lots, extended hours reporting
        - Venue changes: Exchanges open/close
        - Corporate actions: Splits, mergers (adjust needed)
        - Comparability: Multi-year studies challenging
```

**Interaction:** Trader submits market order → order routed to exchange → matched against limit orders → execution reported as trade → appears in time & sales feed → analysts observe transaction

## 5. Mini-Project
Analyze time & sales data for trading patterns and execution quality:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

class TimeSalesAnalyzer:
    def __init__(self):
        self.trades = None
        
    def generate_sample_data(self, n_trades=5000, start_time=None):
        """Generate realistic time & sales data"""
        if start_time is None:
            start_time = datetime(2026, 1, 31, 9, 30, 0)
        
        # Timestamps with realistic clustering
        timestamps = []
        current_time = start_time
        
        for _ in range(n_trades):
            # Clustered arrival (Hawkes-like)
            if np.random.random() < 0.1:  # 10% burst
                increment = np.random.exponential(0.1)  # milliseconds
            else:
                increment = np.random.exponential(1.0)  # normal
            
            current_time += timedelta(seconds=increment)
            timestamps.append(current_time)
        
        # Prices (random walk with drift)
        log_returns = np.random.normal(0.0001, 0.002, n_trades)
        prices = 100 * np.exp(np.cumsum(log_returns))
        
        # Volumes (log-normal: many small, few large)
        volumes = np.round(np.exp(np.random.normal(4.6, 1.2, n_trades)))
        volumes = np.clip(volumes, 1, 50000)  # 1 to 50K shares
        
        # Trade conditions (90% regular, 10% special)
        conditions = np.random.choice(
            ['@', 'T', 'O', '6', 'F'], 
            size=n_trades, 
            p=[0.90, 0.05, 0.02, 0.02, 0.01]
        )
        
        # Create DataFrame
        self.trades = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'condition': conditions
        })
        
        return self.trades
    
    def classify_trade_direction(self):
        """Tick test: classify trades as buy/sell based on price change"""
        directions = ['neutral']  # First trade
        
        for i in range(1, len(self.trades)):
            if self.trades.loc[i, 'price'] > self.trades.loc[i-1, 'price']:
                directions.append('buy')
            elif self.trades.loc[i, 'price'] < self.trades.loc[i-1, 'price']:
                directions.append('sell')
            else:
                # Zero tick: use previous direction
                directions.append(directions[-1])
        
        self.trades['direction'] = directions
        return self.trades
    
    def calculate_vwap(self, start_time=None, end_time=None):
        """Calculate Volume-Weighted Average Price"""
        if start_time and end_time:
            mask = (self.trades['timestamp'] >= start_time) & (self.trades['timestamp'] <= end_time)
            trades = self.trades[mask]
        else:
            trades = self.trades
        
        vwap = (trades['price'] * trades['volume']).sum() / trades['volume'].sum()
        return vwap
    
    def categorize_trade_size(self):
        """Categorize trades by size (retail vs institutional)"""
        conditions = [
            (self.trades['volume'] < 100),
            (self.trades['volume'] >= 100) & (self.trades['volume'] < 1000),
            (self.trades['volume'] >= 1000) & (self.trades['volume'] < 10000),
            (self.trades['volume'] >= 10000)
        ]
        
        categories = ['retail', 'small_inst', 'large_inst', 'block']
        self.trades['size_category'] = np.select(conditions, categories, default='unknown')
        
        return self.trades
    
    def detect_volume_spikes(self, window=100, threshold=3):
        """Detect abnormal volume periods"""
        # Rolling statistics
        rolling_mean = self.trades['volume'].rolling(window=window, min_periods=1).mean()
        rolling_std = self.trades['volume'].rolling(window=window, min_periods=1).std()
        
        # Z-score
        z_scores = (self.trades['volume'] - rolling_mean) / (rolling_std + 1e-6)
        
        # Flag spikes
        self.trades['volume_spike'] = np.abs(z_scores) > threshold
        
        return self.trades
    
    def intraday_volume_profile(self, freq='1min'):
        """Aggregate volume by time interval"""
        self.trades.set_index('timestamp', inplace=True)
        profile = self.trades['volume'].resample(freq).sum()
        self.trades.reset_index(inplace=True)
        
        return profile

# Scenario 1: Generate and explore time & sales data
print("Scenario 1: Time & Sales Data Structure")
print("=" * 80)

analyzer = TimeSalesAnalyzer()
trades = analyzer.generate_sample_data(n_trades=5000)
trades = analyzer.classify_trade_direction()
trades = analyzer.categorize_trade_size()

print(f"Total trades: {len(trades):,}")
print(f"Time span: {(trades['timestamp'].iloc[-1] - trades['timestamp'].iloc[0]).total_seconds()/60:.1f} minutes")
print(f"Average price: ${trades['price'].mean():.2f}")
print(f"Price range: ${trades['price'].min():.2f} - ${trades['price'].max():.2f}")
print(f"Total volume: {trades['volume'].sum():,.0f} shares")
print(f"Average trade size: {trades['volume'].mean():.0f} shares")

print(f"\nTrade condition breakdown:")
condition_counts = trades['condition'].value_counts()
for cond, count in condition_counts.items():
    print(f"  {cond}: {count:,} ({count/len(trades)*100:.1f}%)")

print(f"\nFirst 5 trades:")
print(trades.head()[['timestamp', 'price', 'volume', 'condition', 'direction']])

# Scenario 2: Trade size analysis
print(f"\n\nScenario 2: Trade Size Distribution")
print("=" * 80)

size_dist = trades['size_category'].value_counts()
for category, count in size_dist.items():
    total_vol = trades[trades['size_category'] == category]['volume'].sum()
    print(f"{category:>12}: {count:>5} trades ({count/len(trades)*100:>5.1f}%) | Volume: {total_vol:>10,.0f} shares ({total_vol/trades['volume'].sum()*100:>5.1f}%)")

print(f"\nInterpretation:")
print(f"  - Retail trades dominate count but small % of volume")
print(f"  - Block trades rare but significant volume impact")

# Scenario 3: VWAP calculation
print(f"\n\nScenario 3: VWAP Benchmark")
print("=" * 80)

overall_vwap = analyzer.calculate_vwap()
print(f"Overall VWAP: ${overall_vwap:.4f}")

# Morning vs afternoon
morning_end = trades['timestamp'].iloc[0] + timedelta(hours=2)
morning_vwap = analyzer.calculate_vwap(end_time=morning_end)
afternoon_vwap = analyzer.calculate_vwap(start_time=morning_end)

print(f"Morning VWAP:   ${morning_vwap:.4f}")
print(f"Afternoon VWAP: ${afternoon_vwap:.4f}")
print(f"Difference:     ${afternoon_vwap - morning_vwap:+.4f}")

# Scenario 4: Order flow imbalance
print(f"\n\nScenario 4: Order Flow Analysis")
print("=" * 80)

buy_volume = trades[trades['direction'] == 'buy']['volume'].sum()
sell_volume = trades[trades['direction'] == 'sell']['volume'].sum()
total_volume = buy_volume + sell_volume

imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

print(f"Buy volume:  {buy_volume:>10,.0f} shares ({buy_volume/total_volume*100:.1f}%)")
print(f"Sell volume: {sell_volume:>10,.0f} shares ({sell_volume/total_volume*100:.1f}%)")
print(f"Imbalance:   {imbalance:>+10.2%}")

# Price change
price_change = trades['price'].iloc[-1] - trades['price'].iloc[0]
print(f"\nPrice change: ${price_change:+.4f} ({price_change/trades['price'].iloc[0]*100:+.2f}%)")

# Scenario 5: Volume spike detection
print(f"\n\nScenario 5: Abnormal Activity Detection")
print("=" * 80)

trades = analyzer.detect_volume_spikes(window=100, threshold=3)
spikes = trades[trades['volume_spike']]

print(f"Volume spikes detected: {len(spikes)}")

if len(spikes) > 0:
    print(f"Average spike volume: {spikes['volume'].mean():,.0f} shares")
    print(f"Largest spike: {spikes['volume'].max():,.0f} shares")
    
    print(f"\nTop 3 volume spikes:")
    top_spikes = spikes.nlargest(3, 'volume')[['timestamp', 'price', 'volume', 'direction']]
    print(top_spikes.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price and volume over time
time_minutes = [(t - trades['timestamp'].iloc[0]).total_seconds() / 60 for t in trades['timestamp']]

ax1 = axes[0, 0]
ax1.plot(time_minutes, trades['price'], linewidth=1, color='blue', alpha=0.7)
ax1.axhline(y=overall_vwap, color='red', linestyle='--', linewidth=2, label=f'VWAP: ${overall_vwap:.2f}')
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Price ($)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

ax1b = ax1.twinx()
ax1b.bar(time_minutes, trades['volume'], width=0.02, alpha=0.3, color='gray')
ax1b.set_ylabel('Volume (shares)', color='gray')
ax1b.tick_params(axis='y', labelcolor='gray')
ax1.set_title('Scenario 1: Price & Volume Time Series')

# Plot 2: Trade size distribution
size_categories = ['retail', 'small_inst', 'large_inst', 'block']
size_counts = [len(trades[trades['size_category'] == cat]) for cat in size_categories]
size_volumes = [trades[trades['size_category'] == cat]['volume'].sum() for cat in size_categories]

x = np.arange(len(size_categories))
width = 0.35

axes[0, 1].bar(x - width/2, size_counts, width, label='Trade Count', alpha=0.7, color='blue')
ax2b = axes[0, 1].twinx()
ax2b.bar(x + width/2, size_volumes, width, label='Total Volume', alpha=0.7, color='green')

axes[0, 1].set_xlabel('Trade Size Category')
axes[0, 1].set_ylabel('Number of Trades', color='blue')
ax2b.set_ylabel('Total Volume (shares)', color='green')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(size_categories, rotation=45)
axes[0, 1].set_title('Scenario 2: Trade Size Distribution')
axes[0, 1].legend(loc='upper left')
ax2b.legend(loc='upper right')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Cumulative volume (buy vs sell)
buy_trades = trades[trades['direction'] == 'buy'].copy()
sell_trades = trades[trades['direction'] == 'sell'].copy()

buy_times = [(t - trades['timestamp'].iloc[0]).total_seconds() / 60 for t in buy_trades['timestamp']]
sell_times = [(t - trades['timestamp'].iloc[0]).total_seconds() / 60 for t in sell_trades['timestamp']]

cumulative_buy = np.cumsum(buy_trades['volume'].values)
cumulative_sell = np.cumsum(sell_trades['volume'].values)

axes[1, 0].plot(buy_times, cumulative_buy, linewidth=2, color='green', label='Buy Volume')
axes[1, 0].plot(sell_times, cumulative_sell, linewidth=2, color='red', label='Sell Volume')
axes[1, 0].fill_between(buy_times, 0, cumulative_buy, alpha=0.2, color='green')
axes[1, 0].fill_between(sell_times, 0, cumulative_sell, alpha=0.2, color='red')
axes[1, 0].set_xlabel('Time (minutes)')
axes[1, 0].set_ylabel('Cumulative Volume (shares)')
axes[1, 0].set_title('Scenario 4: Cumulative Order Flow')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Intraday volume profile
profile = analyzer.intraday_volume_profile(freq='5min')
profile_times = [(t - trades['timestamp'].iloc[0]).total_seconds() / 60 for t in profile.index]

axes[1, 1].bar(profile_times, profile.values, width=4, alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Time (minutes)')
axes[1, 1].set_ylabel('Volume (5-min bars)')
axes[1, 1].set_title('Scenario 5: Intraday Volume Profile')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n\nSummary:")
print("=" * 80)
print(f"Time & sales captures all executed trades with timestamp, price, volume")
print(f"VWAP provides execution quality benchmark (institutional standard)")
print(f"Trade size analysis distinguishes retail vs institutional activity")
print(f"Order flow imbalance signals directional pressure")
print(f"Volume spike detection identifies abnormal/informed trading periods")
print(f"\nAdvantage over full tick data: Smaller, simpler, sufficient for many analyses")
```

## 6. Challenge Round
If time & sales data omits quotes (bid-ask), how can traders determine if executed trades were buyer- or seller-initiated without quote information?

- **Tick test (sequential)**: Compare trade price to previous trade → uptick = buy, downtick = sell → simple but noisy (zero-ticks ambiguous) → accuracy ~70-80%
- **Trade size heuristic**: Large trades more likely institutional (informed) → direction inferred from subsequent price move → correlation weak, not causal
- **Trade clustering**: Burst of trades in short time → likely single large order split → all same direction → requires pattern recognition, not guaranteed
- **Quote reconstruction**: Merge time & sales with separate quote feed → Lee-Ready algorithm → 85-90% accuracy → but defeats purpose of "no quotes"
- **Fundamental limitation**: Without quotes, directional classification inherently ambiguous → many trades occur at mid-point (dark pools) → impossible to classify → accepts incompleteness

## 7. Key References
- [Lee & Ready (1991) - Inferring Trade Direction from Intraday Data](https://www.jstor.org/stable/2328845)
- [FINRA Trade Reporting Facilities](https://www.finra.org/filing-reporting/trade-reporting-faq)
- [NYSE TAQ Database Guide](https://www.nyse.com/market-data/historical/trades-and-quotes)
- [Ellis, Michaely & O'Hara (2000) - When the Underwriter Is the Market Maker](https://www.jstor.org/stable/222554)

---
**Status:** Executed trades only (no quotes) | **Complements:** Tick Data, Level 2/3 Data, VWAP Algorithms
