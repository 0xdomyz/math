import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("TRADING SIGNALS AND STRATEGIES")
print("="*70)


class MomentumPortfolio:
    """Cross-sectional momentum across multiple assets"""
    
    def __init__(self, prices_dict, transaction_cost=0.001):
        """
        prices_dict: Dictionary of {asset: price_series}
        """
        self.prices = pd.DataFrame(prices_dict)
        self.tc = transaction_cost
        
    def generate_signals(self, lookback=252, rebalance_freq=21, long_pct=0.3, short_pct=0.3):
        """
        Rank by past returns, long top, short bottom
        lookback: Momentum calculation period (252 = 1 year)
        rebalance_freq: Days between rebalancing (21 = monthly)
        long_pct: Fraction to long (0.3 = top 30%)
        short_pct: Fraction to short (0.3 = bottom 30%)
        """
        # Calculate past returns
        past_returns = self.prices.pct_change(lookback)
        
        # Rank assets
        ranks = past_returns.rank(axis=1, pct=True)
        
        # Signals
        signals = pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)
        
        # Long top performers
        signals[ranks >= (1 - long_pct)] = 1
        
        # Short bottom performers
        signals[ranks <= short_pct] = -1
        
        # Rebalance at specified frequency
        rebalance_dates = range(lookback, len(self.prices), rebalance_freq)
        
        for i in range(len(signals)):
            if i not in rebalance_dates and i > 0:
                signals.iloc[i] = signals.iloc[i-1]
        
        # Calculate returns
        returns = np.log(self.prices / self.prices.shift(1))
        
        # Equal-weight within long and short
        n_long = (signals == 1).sum(axis=1)
        n_short = (signals == -1).sum(axis=1)
        
        # Normalize weights
        weights = signals.copy()
        weights[signals == 1] = weights[signals == 1].div(n_long, axis=0)
        weights[signals == -1] = weights[signals == -1].div(n_short, axis=0)
        
        # Portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        
        # Transaction costs
        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * self.tc
        
        portfolio_returns = portfolio_returns - costs
        
        self.returns = portfolio_returns.fillna(0)
        self.signals = signals
        
        return signals

# Generate synthetic price data for demonstration
def generate_price_data(n_days=1000, trend=0.0001, volatility=0.02, seed=42):
    """Generate synthetic stock price with GBM"""
    np.random.seed(seed)
    
    returns = np.random.normal(trend, volatility, n_days)
    prices = 100 * np.exp(returns.cumsum())
    
    # Create OHLCV dataframe
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, n_days)
    })
    
    df = df.set_index('Date')
    
    return df

# Scenario 1: Moving Average Crossover
print("\n" + "="*70)
print("SCENARIO 1: Moving Average Crossover (Trend Following)")
print("="*70)

# Generate trending market
prices_trend = generate_price_data(n_days=1000, trend=0.0003, volatility=0.015, seed=42)

ma_strategy = MovingAverageCrossover(prices_trend, transaction_cost=0.001)
signals_ma = ma_strategy.generate_signals(fast=50, slow=200)

metrics_ma = ma_strategy.performance_metrics()

print(f"\nMoving Average Crossover (50/200):")
print(f"  Total Return: {metrics_ma['Total Return']:.2%}")
print(f"  CAGR: {metrics_ma['CAGR']:.2%}")
print(f"  Sharpe Ratio: {metrics_ma['Sharpe Ratio']:.3f}")
print(f"  Max Drawdown: {metrics_ma['Max Drawdown']:.2%}")
print(f"  Win Rate: {metrics_ma['Win Rate']:.2%}")
print(f"  Number of Trades: {metrics_ma['Num Trades']:.0f}")

# Buy and hold comparison
bh_returns = np.log(prices_trend['Close'] / prices_trend['Close'].shift(1)).fillna(0)
bh_cumulative = (1 + bh_returns).cumprod()
bh_total_return = bh_cumulative.iloc[-1] - 1

print(f"\nBuy & Hold:")
print(f"  Total Return: {bh_total_return:.2%}")

# Scenario 2: Bollinger Bands Mean Reversion
print("\n" + "="*70)
print("SCENARIO 2: Bollinger Bands Mean Reversion")
print("="*70)

# Generate range-bound market
prices_range = generate_price_data(n_days=1000, trend=0, volatility=0.02, seed=123)

bb_strategy = BollingerBandsMeanReversion(prices_range, transaction_cost=0.001)
signals_bb = bb_strategy.generate_signals(window=20, num_std=2)

metrics_bb = bb_strategy.performance_metrics()

print(f"\nBollinger Bands (20-day, 2σ):")
print(f"  Total Return: {metrics_bb['Total Return']:.2%}")
print(f"  CAGR: {metrics_bb['CAGR']:.2%}")
print(f"  Sharpe Ratio: {metrics_bb['Sharpe Ratio']:.3f}")
print(f"  Max Drawdown: {metrics_bb['Max Drawdown']:.2%}")
print(f"  Win Rate: {metrics_bb['Win Rate']:.2%}")
print(f"  Number of Trades: {metrics_bb['Num Trades']:.0f}")

# Scenario 3: RSI Momentum
print("\n" + "="*70)
print("SCENARIO 3: RSI Momentum Strategy")
print("="*70)

rsi_strategy = RSIMomentum(prices_trend, transaction_cost=0.001)
signals_rsi = rsi_strategy.generate_signals(period=14, oversold=30, overbought=70)

metrics_rsi = rsi_strategy.performance_metrics()

print(f"\nRSI Strategy (14-period, 30/70 thresholds):")
print(f"  Total Return: {metrics_rsi['Total Return']:.2%}")
print(f"  CAGR: {metrics_rsi['CAGR']:.2%}")
print(f"  Sharpe Ratio: {metrics_rsi['Sharpe Ratio']:.3f}")
print(f"  Max Drawdown: {metrics_rsi['Max Drawdown']:.2%}")
print(f"  Win Rate: {metrics_rsi['Win Rate']:.2%}")
print(f"  Number of Trades: {metrics_rsi['Num Trades']:.0f}")

# Scenario 4: Pairs Trading
print("\n" + "="*70)
print("SCENARIO 4: Pairs Trading (Statistical Arbitrage)")
print("="*70)

# Generate cointegrated pair
np.random.seed(42)
n_days = 1000
common_trend = np.cumsum(np.random.normal(0, 0.01, n_days))

price_A = 100 + common_trend + np.cumsum(np.random.normal(0, 0.005, n_days))
price_B = 95 + common_trend + np.cumsum(np.random.normal(0, 0.005, n_days))

price_A = pd.Series(price_A, index=pd.date_range('2020-01-01', periods=n_days))
price_B = pd.Series(price_B, index=pd.date_range('2020-01-01', periods=n_days))

pairs_strategy = PairsTradingStrategy(price_A, price_B, transaction_cost=0.001)
signals_pairs = pairs_strategy.generate_signals(window=60, entry_z=2.0, exit_z=0.5)

# Performance
returns_pairs = pairs_strategy.returns.dropna()
sharpe_pairs = (returns_pairs.mean() * 252) / (returns_pairs.std() * np.sqrt(252))
total_return_pairs = (1 + returns_pairs).cumprod().iloc[-1] - 1

cumulative_pairs = (1 + returns_pairs).cumprod()
running_max_pairs = cumulative_pairs.expanding().max()
drawdown_pairs = (cumulative_pairs - running_max_pairs) / running_max_pairs
max_dd_pairs = drawdown_pairs.min()

print(f"\nPairs Trading (2σ entry, 0.5σ exit):")
print(f"  Total Return: {total_return_pairs:.2%}")
print(f"  Sharpe Ratio: {sharpe_pairs:.3f}")
print(f"  Max Drawdown: {max_dd_pairs:.2%}")
print(f"  Number of Trades: {(signals_pairs.diff() != 0).sum():.0f}")

# Scenario 5: Cross-Sectional Momentum
print("\n" + "="*70)
print("SCENARIO 5: Cross-Sectional Momentum Portfolio")
print("="*70)

# Generate multiple assets
n_assets = 10
prices_dict = {}

for i in range(n_assets):
    trend_i = np.random.uniform(-0.0001, 0.0005)
    vol_i = np.random.uniform(0.015, 0.025)
    prices_dict[f'Asset_{i+1}'] = generate_price_data(
        n_days=1000, trend=trend_i, volatility=vol_i, seed=100+i
    )['Close']

momentum_portfolio = MomentumPortfolio(prices_dict, transaction_cost=0.001)
signals_mom = momentum_portfolio.generate_signals(
    lookback=252, rebalance_freq=21, long_pct=0.3, short_pct=0.3
)

# Performance
returns_mom = momentum_portfolio.returns.dropna()
sharpe_mom = (returns_mom.mean() * 252) / (returns_mom.std() * np.sqrt(252))
total_return_mom = (1 + returns_mom).cumprod().iloc[-1] - 1

cumulative_mom = (1 + returns_mom).cumprod()
running_max_mom = cumulative_mom.expanding().max()
drawdown_mom = (cumulative_mom - running_max_mom) / running_max_mom
max_dd_mom = drawdown_mom.min()

print(f"\nCross-Sectional Momentum (12-month lookback, monthly rebalance):")
print(f"  Total Return: {total_return_mom:.2%}")
print(f"  Sharpe Ratio: {sharpe_mom:.3f}")
print(f"  Max Drawdown: {max_dd_mom:.2%}")
print(f"  Long/Short 30% of universe")

# Scenario 6: Strategy Comparison
print("\n" + "="*70)
print("SCENARIO 6: Strategy Comparison Summary")
print("="*70)

comparison = pd.DataFrame({
    'MA Crossover': [metrics_ma['Total Return'], metrics_ma['Sharpe Ratio'], 
                     metrics_ma['Max Drawdown'], metrics_ma['Win Rate']],
    'Bollinger Bands': [metrics_bb['Total Return'], metrics_bb['Sharpe Ratio'],
                        metrics_bb['Max Drawdown'], metrics_bb['Win Rate']],
    'RSI Momentum': [metrics_rsi['Total Return'], metrics_rsi['Sharpe Ratio'],
                     metrics_rsi['Max Drawdown'], metrics_rsi['Win Rate']],
    'Pairs Trading': [total_return_pairs, sharpe_pairs, max_dd_pairs, np.nan],
    'Cross-Sect Momentum': [total_return_mom, sharpe_mom, max_dd_mom, np.nan]
}, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'])

print("\n", comparison.T.to_string())

print(f"\n Best Sharpe Ratio: {comparison.loc['Sharpe Ratio'].idxmax()}")
print(f"Best Total Return: {comparison.loc['Total Return'].idxmax()}")

# Visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Plot 1: MA Crossover equity curve
ax = axes[0, 0]
cumulative_ma = (1 + ma_strategy.returns).cumprod()
cumulative_bh = (1 + bh_returns).cumprod()

ax.plot(cumulative_ma.index, cumulative_ma.values, 'b-', linewidth=2, label='MA Crossover')
ax.plot(cumulative_bh.index, cumulative_bh.values, 'gray', linewidth=1.5, 
        alpha=0.7, linestyle='--', label='Buy & Hold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.set_title('MA Crossover vs Buy & Hold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: MA Crossover signals
ax = axes[0, 1]
ax.plot(prices_trend.index, prices_trend['Close'], 'k-', linewidth=1.5, alpha=0.7, label='Price')

ma_fast = prices_trend['Close'].rolling(window=50).mean()
ma_slow = prices_trend['Close'].rolling(window=200).mean()

ax.plot(ma_fast.index, ma_fast.values, 'b-', linewidth=1.5, label='MA(50)')
ax.plot(ma_slow.index, ma_slow.values, 'r-', linewidth=1.5, label='MA(200)')

# Mark trades
buy_signals = (signals_ma == 1) & (signals_ma.shift(1) != 1)
sell_signals = (signals_ma == -1) & (signals_ma.shift(1) != -1)

ax.scatter(prices_trend.index[buy_signals], prices_trend['Close'][buy_signals], 
          marker='^', color='green', s=100, label='Buy', zorder=5)
ax.scatter(prices_trend.index[sell_signals], prices_trend['Close'][sell_signals],
          marker='v', color='red', s=100, label='Sell', zorder=5)

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('MA Crossover Signals')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Bollinger Bands
ax = axes[1, 0]
ax.plot(prices_range.index, prices_range['Close'], 'k-', linewidth=1.5, label='Price')

ma_bb = prices_range['Close'].rolling(window=20).mean()
std_bb = prices_range['Close'].rolling(window=20).std()
upper = ma_bb + 2*std_bb
lower = ma_bb - 2*std_bb

ax.plot(ma_bb.index, ma_bb.values, 'b-', linewidth=1, alpha=0.7, label='MA(20)')
ax.plot(upper.index, upper.values, 'r--', linewidth=1, alpha=0.7, label='Upper Band')
ax.plot(lower.index, lower.values, 'g--', linewidth=1, alpha=0.7, label='Lower Band')

ax.fill_between(ma_bb.index, lower.values, upper.values, alpha=0.1, color='gray')

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Bollinger Bands Strategy')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Pairs trading spread
ax = axes[1, 1]
spread, z_score = pairs_strategy.calculate_spread(60)

ax.plot(z_score.index, z_score.values, 'b-', linewidth=1.5, label='Z-Score')
ax.axhline(2, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Entry (+2σ)')
ax.axhline(-2, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Entry (-2σ)')
ax.axhline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(-0.5, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='Exit (±0.5σ)')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

ax.fill_between(z_score.index, -2, 2, alpha=0.1, color='gray')

ax.set_xlabel('Date')
ax.set_ylabel('Z-Score')
ax.set_title('Pairs Trading: Spread Z-Score')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Strategy comparison (equity curves)
ax = axes[2, 0]

cumulative_bb = (1 + bb_strategy.returns).cumprod()
cumulative_rsi = (1 + rsi_strategy.returns).cumprod()

ax.plot(cumulative_ma.index, cumulative_ma.values, linewidth=2, label='MA Crossover')
ax.plot(cumulative_bb.index, cumulative_bb.values, linewidth=2, label='Bollinger Bands')
ax.plot(cumulative_rsi.index, cumulative_rsi.values, linewidth=2, label='RSI')
ax.plot(cumulative_pairs.index, cumulative_pairs.values, linewidth=2, label='Pairs Trading')

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.set_title('Strategy Comparison: Equity Curves')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Drawdowns
ax = axes[2, 1]

dd_ma = (cumulative_ma - cumulative_ma.expanding().max()) / cumulative_ma.expanding().max()
dd_bb = (cumulative_bb - cumulative_bb.expanding().max()) / cumulative_bb.expanding().max()
dd_rsi = (cumulative_rsi - cumulative_rsi.expanding().max()) / cumulative_rsi.expanding().max()

ax.fill_between(dd_ma.index, 0, dd_ma.values*100, alpha=0.5, label='MA Crossover')
ax.fill_between(dd_bb.index, 0, dd_bb.values*100, alpha=0.5, label='Bollinger Bands')
ax.fill_between(dd_rsi.index, 0, dd_rsi.values*100, alpha=0.5, label='RSI')

ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Strategy Drawdowns')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()