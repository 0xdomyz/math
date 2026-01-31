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
