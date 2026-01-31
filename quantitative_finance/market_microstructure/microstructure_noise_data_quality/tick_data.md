# Tick Data

## 1. Concept Skeleton
**Definition:** Timestamped record of every trade and quote update in financial markets; microsecond/nanosecond precision; granular order flow  
**Purpose:** High-frequency analysis; market microstructure research; algorithm development; price discovery measurement; liquidity analysis  
**Prerequisites:** Market structure, order types, bid-ask spread, market data feeds, time synchronization

## 2. Comparative Framing
| Data Type | Frequency | Content | Timestamp Precision | Storage Size | Use Case |
|-----------|-----------|---------|-------------------|--------------|----------|
| **Tick Data** | Microseconds | Every trade/quote | Nanoseconds | TB/day | HFT, research |
| **1-Minute Bars** | Minutes | OHLCV aggregated | Seconds | MB/day | Intraday trading |
| **Daily OHLC** | Days | Daily summary | Days | KB/year | Long-term analysis |
| **Level 2 (Depth)** | Microseconds | Top N price levels | Nanoseconds | TB/day | Order book modeling |
| **Time & Sales** | Trades only | Executed trades | Milliseconds | GB/day | Volume analysis |

## 3. Examples + Counterexamples

**Trade Tick:**  
Timestamp: 2026-01-31 09:30:01.234567890 | Symbol: AAPL | Price: $150.23 | Volume: 100 | Conditions: @ (regular trade)

**Quote Tick:**  
Timestamp: 2026-01-31 09:30:01.234567123 | Symbol: AAPL | Bid: $150.22 | Ask: $150.23 | Bid Size: 500 | Ask Size: 800

**High-Frequency Burst:**  
Within 1 millisecond: 47 quote updates (HFT market makers adjusting) → 2 trades → then silence for 50ms → Illustrates clustering

**Aggregated Bar (Not Tick Data):**  
09:30-09:31 | Open: $150.20 | High: $150.30 | Low: $150.18 | Close: $150.23 | Volume: 15,247 → Loses intra-minute dynamics

## 4. Layer Breakdown
```
Tick Data Structure & Analysis:
├─ Data Components:
│   ├─ Trade Ticks:
│   │   - Timestamp: Nanosecond precision (exchange time)
│   │   - Symbol: Ticker identifier (AAPL, MSFT, etc.)
│   │   - Price: Execution price
│   │   - Volume: Number of shares/contracts
│   │   - Trade conditions: Regular (@), late, cancelled, etc.
│   │   - Exchange: Venue identifier (NYSE, NASDAQ, BATS)
│   │   - Sequence number: Monotonic counter for ordering
│   ├─ Quote Ticks:
│   │   - Timestamp: When quote posted
│   │   - Bid price: Best buy price
│   │   - Ask price: Best sell price
│   │   - Bid size: Shares available at bid
│   │   - Ask size: Shares available at ask
│   │   - Quote conditions: Regular, locked, crossed, indicative
│   │   - Market maker ID: Optional (some exchanges)
│   ├─ Metadata:
│   │   - Trading session: Pre-market, regular, after-hours
│   │   - Market status: Open, halted, closed
│   │   - Flags: Short-sale restricted, limit-up/down
│   │   - Data quality: Good, delayed, test
│   └─ Reference Data:
│       - Tick size: Minimum price increment
│       - Lot size: Minimum tradable quantity
│       - Corporate actions: Splits, dividends (adjustments)
│       - Symbology: CUSIP, ISIN, RIC mappings
│
├─ Timestamp Synchronization:
│   ├─ Exchange Timestamps:
│   │   - Matching engine time: When order matched
│   │   - Precision: Nanosecond (modern exchanges)
│   │   - Clock: Atomic clock synchronized (GPS, NTP)
│   │   - Accuracy: ±100 microseconds (regulatory requirement)
│   ├─ Network Timestamps:
│   │   - Gateway time: When message sent from exchange
│   │   - Received time: When data vendor receives
│   │   - Processing time: When vendor processes/enriches
│   │   - Publication time: When client receives
│   ├─ Clock Drift:
│   │   - Problem: Clocks drift over time (microseconds/day)
│   │   - Synchronization: NTP (Network Time Protocol)
│   │   - Precision Time Protocol (PTP): IEEE 1588 (nanosecond accuracy)
│   │   - Importance: Critical for lead-lag analysis, causality
│   ├─ Latency Components:
│   │   - Matching: Order to execution (microseconds)
│   │   - Transmission: Exchange to vendor (milliseconds)
│   │   - Distribution: Vendor to client (milliseconds-seconds)
│   │   - Total: Sum of all components (variable)
│   └─ Time Zones:
│       - UTC: Universal standard (recommended)
│       - Exchange local: EST (NYSE), CET (Euronext)
│       - Conversion: Required for cross-market analysis
│       - Daylight saving: Complicates analysis (beware!)
│
├─ Data Storage & Formats:
│   ├─ Raw Binary:
│   │   - Format: Exchange-specific binary protocols
│   │   - Advantage: Compact, fast to write
│   │   - Disadvantage: Vendor lock-in, complex parsing
│   │   - Examples: ITCH (NASDAQ), PITCH (BATS), CME MDP3
│   ├─ CSV/Text:
│   │   - Format: Human-readable comma-separated
│   │   - Advantage: Universal, easy to inspect
│   │   - Disadvantage: Large file size, slow to parse
│   │   - Usage: Academic research, prototyping
│   ├─ Parquet/Arrow:
│   │   - Format: Columnar binary (Apache projects)
│   │   - Advantage: Compressed, fast queries, schema
│   │   - Usage: Modern analytics, big data pipelines
│   │   - Compression: 10-20x smaller than CSV
│   ├─ HDF5:
│   │   - Format: Hierarchical Data Format
│   │   - Advantage: Random access, nested structure
│   │   - Usage: Scientific computing, time series
│   │   - Drawback: Single-writer bottleneck
│   ├─ Databases:
│   │   - Time-series DB: InfluxDB, TimescaleDB, kdb+
│   │   - Relational: PostgreSQL with TimescaleDB extension
│   │   - NoSQL: Cassandra, MongoDB (less common for ticks)
│   │   - kdb+: Industry standard (finance, very fast)
│   └─ Cloud Storage:
│   │   - S3/GCS/Azure Blob: Object storage for archives
│   │   - Tiering: Hot (recent, fast) → cold (old, slow)
│   │   - Cost: $0.023/GB/month (S3 standard) → $0.004 (glacier)
│
├─ Data Cleaning & Preprocessing:
│   ├─ Outlier Detection:
│   │   - Price spikes: >10σ moves flagged
│   │   - Method: Rolling z-score, Median Absolute Deviation
│   │   - Action: Remove or replace with interpolation
│   │   - Cause: Fat-finger errors, system glitches
│   ├─ Trade Conditions Filtering:
│   │   - Regular trades: Include (primary analysis)
│   │   - Form-T: Late reported (exclude or separate)
│   │   - Cancelled: Remove entirely
│   │   - Adjusted: Include only if after adjustment
│   │   - Example: NYSE condition codes (00-99)
│   ├─ Quote Filtering:
│   │   - Locked/crossed: Flag or remove
│   │   - Zero bid/ask: Invalid, remove
│   │   - Negative spreads: Data error, remove
│   │   - Bid > ask: Fleeting (latency), handle carefully
│   ├─ Duplicate Detection:
│   │   - Definition: Same timestamp + price + volume
│   │   - Cause: Feed redundancy, retransmissions
│   │   - Action: Keep first, discard subsequent
│   │   - Challenge: Distinguish true duplicate from coincidence
│   ├─ Timestamp Ordering:
│   │   - Problem: Messages arrive out-of-order (network)
│   │   - Solution: Sort by sequence number, then timestamp
│   │   - Buffer: Hold messages briefly to allow reordering
│   │   - Critical: Incorrect ordering → wrong causality inference
│   └─ Corporate Action Adjustments:
│       - Splits: Multiply prices by ratio, divide volume
│       - Dividends: Adjust for ex-dividend gap
│       - Mergers: Handle symbol changes, conversions
│       - Importance: Consistent price series for backtesting
│
├─ Common Tick Data Providers:
│   ├─ Exchange Feeds (Direct):
│   │   - NYSE TAQ: Trades and Quotes (North America)
│   │   - NASDAQ TotalView: Full depth-of-book
│   │   - CME Globex: Futures and options
│   │   - Advantage: Lowest latency, authoritative
│   │   - Cost: $10K-$100K+/month per exchange
│   ├─ Consolidated Feeds:
│   │   - CTS/CQS: Consolidated Tape System (US equities)
│   │   - OPRA: Options Price Reporting Authority
│   │   - SIP: Securities Information Processor
│   │   - Latency: Slower than direct feeds (milliseconds)
│   │   - Cost: Lower, but still substantial
│   ├─ Data Vendors:
│   │   - Bloomberg Terminal: Real-time + historical
│   │   - Refinitiv Tick History: Global coverage
│   │   - QuantQuote: Academic pricing
│   │   - Algoseek: Retail-friendly pricing
│   │   - Polygon.io: API-based, modern
│   │   - Cost: $100-$10K/month depending on usage
│   ├─ Academic Sources:
│   │   - WRDS: Wharton Research Data Services (TAQ)
│   │   - LOBSTER: Limit order book reconstruction
│   │   - Free/subsidized: For academic research only
│   │   - Delay: Historical data (real-time restricted)
│   └─ Cryptocurrency:
│       - Coinbase Pro, Binance, Kraken APIs
│       - Often free: Real-time tick data via WebSocket
│       - Caveat: Exchange-specific, no consolidated tape
│
├─ Data Volume Considerations:
│   ├─ Scale:
│   │   - Single stock (liquid): 500K-2M ticks/day
│   │   - S&P 500: ~1 billion ticks/day
│   │   - Full US equities: ~50 billion ticks/day
│   │   - Annual: ~10 TB raw, 1-2 TB compressed
│   ├─ Processing:
│   │   - Real-time: Stream processing (Kafka, Flink)
│   │   - Batch: Nightly aggregation, cleaning
│   │   - Query: Indexed databases (kdb+, TimescaleDB)
│   │   - Bandwidth: 10+ Gbps for real-time feeds
│   ├─ Storage Costs:
│   │   - Raw data: $230/year per TB (S3 standard)
│   │   - Compressed: $23/year per TB (S3 glacier)
│   │   - 5-year archive: $1,150 for 10 TB
│   └─ Computational:
│       - RAM: 64-256 GB for in-memory analysis
│       - CPU: Multi-core for parallel processing
│       - GPU: Optional for certain algorithms (rare)
│       - Cloud: AWS/GCP spot instances ($0.10-0.50/hour)
│
├─ Analysis Techniques:
│   ├─ Trade Classification:
│   │   - Lee-Ready algorithm: Classify buyer- vs seller-initiated
│   │   - Method: Compare trade price to mid-quote
│   │   - Above mid → buyer-initiated, below → seller
│   │   - Application: Order flow toxicity, information content
│   ├─ Price Impact Estimation:
│   │   - Temporary: Price movement that reverts
│   │   - Permanent: Persistent price change (information)
│   │   - Method: Regress price change on signed volume
│   │   - Hasbrouck decomposition: VAR on trades/quotes
│   ├─ Volatility Measurement:
│   │   - Realized variance: Sum of squared returns
│   │   - Signature plot: RV vs sampling frequency
│   │   - Optimal sampling: Minimize microstructure noise
│   │   - TSRV: Two-Scales Realized Variance (noise-robust)
│   ├─ Spread Decomposition:
│   │   - Roll model: Estimate from negative autocorrelation
│   │   - Stoll model: Decompose into adverse selection, inventory, processing
│   │   - High-frequency: Use tick data for precise estimates
│   ├─ Liquidity Metrics:
│   │   - Effective spread: 2 × |price - mid|
│   │   - Realized spread: Accounts for price reversion
│   │   - Depth: Total volume at best bid/ask
│   │   - Resilience: Time to replenish depth after trade
│   ├─ Information Content:
│   │   - PIN: Probability of Informed Trading (Easley et al)
│   │   - VPIN: Volume-Synchronized PIN (high-frequency version)
│   │   - Order flow toxicity: Measure adverse selection
│   └─ Lead-Lag Relationships:
│       - Cross-correlation: Which asset moves first
│       - Granger causality: Statistical causality testing
│       - Application: Identify price discovery leaders
│
└─ Practical Challenges:
    ├─ Data Quality:
    │   - Missing ticks: Network packet loss, system outages
    │   - Erroneous prices: Fat fingers, system errors
    │   - Delayed reporting: Form-T trades (late reported)
    │   - Inconsistent timestamps: Clock drift, time zones
    ├─ Computational Complexity:
    │   - Billions of records: Requires efficient algorithms
    │   - Memory constraints: Can't load all into RAM
    │   - Query speed: Need indexed databases
    │   - Parallelization: Essential for reasonable runtimes
    ├─ Regulatory Constraints:
    │   - Redistribution: Licenses restrict sharing
    │   - Derived data: Rules on publishing analytics
    │   - Privacy: Some exchanges anonymize market makers
    │   - Audit trail: Must retain for compliance (7 years)
    └─ Cost Management:
        - Selective symbols: Don't subscribe to all
        - Tiered storage: Archive old data cheaply
        - Sampling: Use subsampled data for prototyping
        - Compression: 10-20x reduction with minimal loss
```

**Interaction:** Exchange matching engine → generates tick → transmitted to vendors → stored in database → queried by analysts → insights extracted → trading decisions made

## 5. Mini-Project
Process and analyze tick data for microstructure insights:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

class TickDataSimulator:
    def __init__(self, n_ticks=10000, start_price=100.0, spread=0.02):
        self.n_ticks = n_ticks
        self.start_price = start_price
        self.spread = spread
        
    def generate_tick_data(self):
        """Simulate realistic tick data with quotes and trades"""
        # Generate timestamps (microsecond precision)
        start_time = datetime(2026, 1, 31, 9, 30, 0)
        timestamps = [start_time + timedelta(microseconds=i*100 + np.random.randint(-50, 50)) 
                     for i in range(self.n_ticks)]
        
        # Simulate efficient price (random walk)
        price_changes = np.random.normal(0, 0.01, self.n_ticks)
        mid_prices = self.start_price + np.cumsum(price_changes)
        
        # Generate bid-ask quotes
        bids = mid_prices - self.spread / 2
        asks = mid_prices + self.spread / 2
        
        # Generate trade indicators (30% of ticks are trades)
        is_trade = np.random.random(self.n_ticks) < 0.3
        
        # Trade prices (at bid or ask)
        trade_prices = np.where(is_trade,
                               np.where(np.random.random(self.n_ticks) > 0.5, asks, bids),
                               np.nan)
        
        # Trade volumes (100-1000 shares, log-normal)
        volumes = np.where(is_trade, 
                          np.round(np.exp(np.random.normal(5, 1, self.n_ticks))),
                          np.nan)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'bid': bids,
            'ask': asks,
            'mid': mid_prices,
            'trade_price': trade_prices,
            'volume': volumes,
            'is_trade': is_trade,
            'spread': self.spread
        })
        
        return data
    
    def classify_trades(self, data):
        """Lee-Ready algorithm: Classify trades as buy/sell"""
        classifications = []
        
        for idx, row in data.iterrows():
            if row['is_trade']:
                # Compare trade price to mid-quote
                if row['trade_price'] > row['mid']:
                    classifications.append('buy')
                elif row['trade_price'] < row['mid']:
                    classifications.append('sell')
                else:
                    # At mid-point, use tick test (previous trade direction)
                    if len(classifications) > 0:
                        classifications.append(classifications[-1])
                    else:
                        classifications.append('unknown')
            else:
                classifications.append(None)
        
        data['trade_direction'] = classifications
        return data
    
    def calculate_effective_spread(self, data):
        """Effective spread: 2 * |trade_price - mid|"""
        trades = data[data['is_trade']].copy()
        trades['effective_spread'] = 2 * abs(trades['trade_price'] - trades['mid'])
        return trades['effective_spread'].mean()
    
    def calculate_realized_spread(self, data, look_ahead=10):
        """Realized spread: Accounts for price reversion"""
        trades = data[data['is_trade']].copy()
        realized_spreads = []
        
        for idx in trades.index:
            # Current trade
            trade_price = data.loc[idx, 'trade_price']
            mid_now = data.loc[idx, 'mid']
            direction = 1 if data.loc[idx, 'trade_direction'] == 'buy' else -1
            
            # Future mid-price (after look_ahead ticks)
            future_idx = idx + look_ahead
            if future_idx < len(data):
                mid_future = data.loc[future_idx, 'mid']
                
                # Realized spread
                rs = 2 * direction * (trade_price - mid_future)
                realized_spreads.append(rs)
        
        return np.mean(realized_spreads) if realized_spreads else 0
    
    def aggregate_to_bars(self, data, freq='1min'):
        """Aggregate tick data to OHLCV bars"""
        # Use trades only
        trades = data[data['is_trade']].copy()
        trades.set_index('timestamp', inplace=True)
        
        # Resample
        bars = trades['trade_price'].resample(freq).ohlc()
        bars['volume'] = trades['volume'].resample(freq).sum()
        
        return bars
    
    def estimate_volatility(self, data, sampling_freq=100):
        """Calculate realized variance at different frequencies"""
        # Sample every N ticks
        sampled = data.iloc[::sampling_freq].copy()
        returns = sampled['mid'].pct_change().dropna()
        
        # Realized variance
        rv = (returns ** 2).sum()
        
        return rv

# Scenario 1: Generate and examine tick data
print("Scenario 1: Tick Data Generation and Structure")
print("=" * 80)

sim = TickDataSimulator(n_ticks=10000, start_price=100.0, spread=0.02)
ticks = sim.generate_tick_data()

# Classify trades
ticks = sim.classify_trades(ticks)

# Summary statistics
total_ticks = len(ticks)
n_trades = ticks['is_trade'].sum()
n_quotes = total_ticks - n_trades
avg_volume = ticks[ticks['is_trade']]['volume'].mean()
time_span = (ticks['timestamp'].iloc[-1] - ticks['timestamp'].iloc[0]).total_seconds()

print(f"Total ticks: {total_ticks:,}")
print(f"  Trades: {n_trades:,} ({n_trades/total_ticks*100:.1f}%)")
print(f"  Quotes: {n_quotes:,} ({n_quotes/total_ticks*100:.1f}%)")
print(f"Time span: {time_span:.1f} seconds")
print(f"Tick rate: {total_ticks/time_span:.0f} ticks/second")
print(f"Trade rate: {n_trades/time_span:.0f} trades/second")
print(f"Average trade volume: {avg_volume:.0f} shares")

# Show sample ticks
print(f"\nFirst 5 trade ticks:")
print(ticks[ticks['is_trade']].head()[['timestamp', 'bid', 'ask', 'trade_price', 'volume', 'trade_direction']])

# Scenario 2: Spread analysis
print(f"\n\nScenario 2: Spread Decomposition")
print("=" * 80)

effective_spread = sim.calculate_effective_spread(ticks)
realized_spread = sim.calculate_realized_spread(ticks, look_ahead=50)
price_impact = effective_spread - realized_spread

print(f"Effective spread: ${effective_spread:.4f} ({effective_spread/ticks['mid'].mean()*10000:.2f} bps)")
print(f"Realized spread:  ${realized_spread:.4f} ({realized_spread/ticks['mid'].mean()*10000:.2f} bps)")
print(f"Price impact:     ${price_impact:.4f} ({price_impact/ticks['mid'].mean()*10000:.2f} bps)")
print(f"\nInterpretation:")
print(f"  - Effective spread = transaction cost paid by trader")
print(f"  - Realized spread = profit to market maker (after reversion)")
print(f"  - Price impact = information component (permanent)")

# Scenario 3: Order flow analysis
print(f"\n\nScenario 3: Order Flow Imbalance")
print("=" * 80)

trades = ticks[ticks['is_trade']].copy()
buy_volume = trades[trades['trade_direction'] == 'buy']['volume'].sum()
sell_volume = trades[trades['trade_direction'] == 'sell']['volume'].sum()
total_volume = buy_volume + sell_volume

order_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

print(f"Buy volume:  {buy_volume:,.0f} shares")
print(f"Sell volume: {sell_volume:,.0f} shares")
print(f"Total volume: {total_volume:,.0f} shares")
print(f"Order imbalance: {order_imbalance:+.2%}")
print(f"\nInterpretation:")
if abs(order_imbalance) < 0.1:
    print(f"  Balanced order flow (near 50-50 split)")
elif order_imbalance > 0:
    print(f"  Buy pressure dominates → expect price drift upward")
else:
    print(f"  Sell pressure dominates → expect price drift downward")

# Scenario 4: Aggregation to bars
print(f"\n\nScenario 4: Tick-to-Bar Aggregation")
print("=" * 80)

bars_1min = sim.aggregate_to_bars(ticks, freq='1min')
bars_5min = sim.aggregate_to_bars(ticks, freq='5min')

print(f"1-minute bars generated: {len(bars_1min)}")
print(f"5-minute bars generated: {len(bars_5min)}")

print(f"\nFirst 3 bars (1-minute):")
print(bars_1min.head(3))

# Information loss from aggregation
tick_return_vol = ticks['mid'].pct_change().std()
bar_1min_vol = bars_1min['close'].pct_change().std()
info_loss = 1 - (bar_1min_vol / tick_return_vol)

print(f"\nVolatility comparison:")
print(f"  Tick-level: {tick_return_vol:.6f}")
print(f"  1-min bars: {bar_1min_vol:.6f}")
print(f"  Information loss: {info_loss*100:.1f}%")

# Scenario 5: Signature plot (optimal sampling)
print(f"\n\nScenario 5: Optimal Sampling Frequency (Signature Plot)")
print("=" * 80)

sampling_frequencies = [1, 5, 10, 20, 50, 100, 200, 500]
realized_variances = []

for freq in sampling_frequencies:
    rv = sim.estimate_volatility(ticks, sampling_freq=freq)
    realized_variances.append(rv)
    print(f"Sampling every {freq:>3} ticks: RV = {rv:.6f}")

optimal_idx = np.argmin(realized_variances)
optimal_freq = sampling_frequencies[optimal_idx]

print(f"\nOptimal sampling: Every {optimal_freq} ticks")
print(f"  (Minimizes microstructure noise contamination)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price evolution with trades
time_minutes = [(t - ticks['timestamp'].iloc[0]).total_seconds() / 60 for t in ticks['timestamp']]
axes[0, 0].plot(time_minutes, ticks['mid'], linewidth=1, alpha=0.7, color='blue', label='Mid Price')
trades_only = ticks[ticks['is_trade']]
trade_times = [(t - ticks['timestamp'].iloc[0]).total_seconds() / 60 for t in trades_only['timestamp']]
buy_trades = trades_only[trades_only['trade_direction'] == 'buy']
sell_trades = trades_only[trades_only['trade_direction'] == 'sell']
buy_times = [(t - ticks['timestamp'].iloc[0]).total_seconds() / 60 for t in buy_trades['timestamp']]
sell_times = [(t - ticks['timestamp'].iloc[0]).total_seconds() / 60 for t in sell_trades['timestamp']]

axes[0, 0].scatter(buy_times, buy_trades['trade_price'], 
                  color='green', marker='^', s=20, alpha=0.5, label='Buy Trades')
axes[0, 0].scatter(sell_times, sell_trades['trade_price'], 
                  color='red', marker='v', s=20, alpha=0.5, label='Sell Trades')
axes[0, 0].set_xlabel('Time (minutes)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Tick Data - Price & Trades')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Bid-ask spread over time
spread_bps = (ticks['ask'] - ticks['bid']) / ticks['mid'] * 10000
axes[0, 1].plot(time_minutes, spread_bps, linewidth=1, alpha=0.7, color='purple')
axes[0, 1].axhline(y=spread_bps.mean(), color='r', linestyle='--', 
                  label=f'Mean: {spread_bps.mean():.1f} bps')
axes[0, 1].set_xlabel('Time (minutes)')
axes[0, 1].set_ylabel('Spread (basis points)')
axes[0, 1].set_title('Scenario 2: Bid-Ask Spread Dynamics')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Order flow imbalance
# Rolling 100-trade windows
window_size = 100
rolling_imbalance = []
rolling_times = []

for i in range(0, len(trades_only) - window_size, 20):
    window = trades_only.iloc[i:i+window_size]
    buy_vol = window[window['trade_direction'] == 'buy']['volume'].sum()
    sell_vol = window[window['trade_direction'] == 'sell']['volume'].sum()
    total_vol = buy_vol + sell_vol
    
    if total_vol > 0:
        imb = (buy_vol - sell_vol) / total_vol
        rolling_imbalance.append(imb)
        rolling_times.append((window['timestamp'].iloc[-1] - ticks['timestamp'].iloc[0]).total_seconds() / 60)

axes[1, 0].plot(rolling_times, rolling_imbalance, linewidth=2, color='darkgreen')
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].fill_between(rolling_times, 0, rolling_imbalance, 
                        where=np.array(rolling_imbalance) > 0, 
                        color='green', alpha=0.3, label='Buy Pressure')
axes[1, 0].fill_between(rolling_times, 0, rolling_imbalance, 
                        where=np.array(rolling_imbalance) < 0, 
                        color='red', alpha=0.3, label='Sell Pressure')
axes[1, 0].set_xlabel('Time (minutes)')
axes[1, 0].set_ylabel('Order Flow Imbalance')
axes[1, 0].set_title('Scenario 3: Rolling Order Flow Imbalance')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Signature plot
axes[1, 1].plot(sampling_frequencies, realized_variances, 'o-', 
               linewidth=2, markersize=8, color='blue')
axes[1, 1].axvline(x=optimal_freq, color='r', linestyle='--', 
                  alpha=0.7, label=f'Optimal ({optimal_freq})')
axes[1, 1].set_xlabel('Sampling Frequency (ticks)')
axes[1, 1].set_ylabel('Realized Variance')
axes[1, 1].set_title('Scenario 5: Signature Plot')
axes[1, 1].set_xscale('log')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n\nSummary:")
print("=" * 80)
print(f"Tick data provides microsecond-level granularity for microstructure analysis")
print(f"Trade classification (Lee-Ready) identifies buy/sell pressure")
print(f"Spread decomposition separates transaction costs from information")
print(f"Aggregation to bars loses intraday dynamics (information loss)")
print(f"Optimal sampling balances noise reduction vs information retention")
```

## 6. Challenge Round
If tick data provides complete market information, why do researchers still use lower-frequency (daily/weekly) data for many studies?

- **Microstructure noise dominates**: High-frequency returns 80% noise, 20% signal → parameter estimates biased → need sophisticated filters → simpler to use daily (noise-averaged out naturally)
- **Computational burden**: 50 billion ticks/day US equities → requires cluster computing → most researchers lack infrastructure → daily data fits in laptop RAM
- **Statistical power trade-off**: More observations (ticks) but lower SNR → fewer observations (daily) but higher SNR → for many tests, daily optimal → depends on hypothesis
- **Research question mismatch**: Long-term anomalies (momentum, value) operate at monthly scale → tick data irrelevant → introduces spurious patterns → daily/weekly sufficient and cleaner
- **Publication tradition**: 30 years of research using daily data → comparability required → reviewers expect daily baseline → tick-based results supplementary → institutional inertia

## 7. Key References
- [Hasbrouck (2007) - Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading](https://academic.oup.com/book/7502)
- [Lee & Ready (1991) - Inferring Trade Direction from Intraday Data](https://www.jstor.org/stable/2328845)
- [NYSE TAQ Database Documentation](https://www.nyse.com/market-data/historical)
- [NASDAQ TotalView Data Specifications](https://www.nasdaq.com/solutions/nasdaq-totalview)

---
**Status:** Microsecond-precision market data | **Complements:** Time & Sales Data, Level 2/3 Data, Microstructure Noise
