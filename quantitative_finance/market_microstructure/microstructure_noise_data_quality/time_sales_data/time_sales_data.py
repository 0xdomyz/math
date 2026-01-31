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
