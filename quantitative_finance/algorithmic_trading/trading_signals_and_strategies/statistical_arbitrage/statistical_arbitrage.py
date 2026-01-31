import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.stattools import coint, adfuller

# Generate correlated price pairs with occasional breaks
np.random.seed(42)
n_days = 1000

# Base process
common_factor = np.cumsum(np.random.normal(0, 0.01, n_days))
stock_a = 100 * np.exp(common_factor + np.random.normal(0, 0.01, n_days))
stock_b = 50 * np.exp(common_factor + np.random.normal(0, 0.012, n_days))

# Add mispricing events (opportunities)
stock_a[200:210] *= 0.98  # A drops (opportunity: buy A, short B)
stock_b[500:520] *= 1.02  # B rises (opportunity: short B, long A)

dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
df = pd.DataFrame({'A': stock_a, 'B': stock_b}, index=dates)

print("="*100)
print("STATISTICAL ARBITRAGE: PAIRS TRADING PORTFOLIO")
print("="*100)

print(f"\nStep 1: Cointegration Analysis")
print(f"-" * 50)

# Check cointegration
score, p_value, _ = coint(df['A'], df['B'])
print(f"Cointegration test p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"✓ Pairs are cointegrated (stationary spread)")
else:
    print(f"✗ Not cointegrated")

# Regression to find hedge ratio
slope, intercept, r_value, _, _ = linregress(df['B'], df['A'])
print(f"\nHedge ratio (β): {slope:.4f}")
print(f"R-squared: {r_value**2:.4f}")

# Spread calculation
df['spread'] = df['A'] - slope * df['B']
df['mean_spread'] = df['spread'].rolling(20).mean()
df['std_spread'] = df['spread'].rolling(20).std()
df['z_score'] = (df['spread'] - df['mean_spread']) / df['std_spread']

print(f"Mean spread: {df['spread'].mean():.2f}")
print(f"Std dev spread: {df['spread'].std():.2f}")

print(f"\nStep 2: Trading Signals")
print(f"-" * 50)

# Generate signals
entry_threshold = 2.0
df['signal'] = 0
df.loc[df['z_score'] > entry_threshold, 'signal'] = -1  # Short A, long B
df.loc[df['z_score'] < -entry_threshold, 'signal'] = 1  # Long A, short B

n_signals = (df['signal'] != 0).sum()
print(f"Total entry signals: {n_signals}")
print(f"Buy signals (long A): {(df['signal']==1).sum()}")
print(f"Sell signals (short A): {(df['signal']==-1).sum()}")

print(f"\nStep 3: Backtest Performance")
print(f"-" * 50)

position = 0
trades = []

for i in range(1, len(df)):
    if df['signal'].iloc[i] != 0 and position == 0:
        # Enter
        position = df['signal'].iloc[i]
        entry_spread = df['spread'].iloc[i]
        entry_date = df.index[i]
    elif position != 0:
        # Exit if spread reverts or time stop
        if (position == 1 and df['z_score'].iloc[i] > -0.5) or \
           (position == -1 and df['z_score'].iloc[i] < 0.5) or \
           (i - (len(trades) if trades else 0) > 30):
            exit_spread = df['spread'].iloc[i]
            pnl_spread = (exit_spread - entry_spread) * position / entry_spread
            
            # P&L in A and B separately
            pnl_a = (df['A'].iloc[i] / df['A'].iloc[len(trades) if trades else 0] - 1) * position
            pnl_b = (df['B'].iloc[i] / df['B'].iloc[len(trades) if trades else 0] - 1) * (-position * slope)
            pnl_total = pnl_a + pnl_b
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[i],
                'entry_z': df['z_score'].iloc[len(trades)],
                'exit_z': df['z_score'].iloc[i],
                'pnl_pct': pnl_total * 100,
                'duration_days': (df.index[i] - entry_date).days,
            })
            
            position = 0

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"Total trades: {len(trades_df)}")
    print(f"Winning trades: {(trades_df['pnl_pct'] > 0).sum()}")
    print(f"Losing trades: {(trades_df['pnl_pct'] < 0).sum()}")
    print(f"Win rate: {(trades_df['pnl_pct'] > 0).sum() / len(trades_df) * 100:.1f}%")
    print(f"\nAverage P&L: {trades_df['pnl_pct'].mean():.3f}%")
    print(f"Median P&L: {trades_df['pnl_pct'].median():.3f}%")
    print(f"Best trade: {trades_df['pnl_pct'].max():.3f}%")
    print(f"Worst trade: {trades_df['pnl_pct'].min():.3f}%")
    print(f"Avg hold period: {trades_df['duration_days'].mean():.0f} days")
    
    cumul_return = (1 + trades_df['pnl_pct'].sum() / 100) - 1
    print(f"\nCumulative return: {cumul_return*100:.2f}%")
    print(f"Annualized: {(cumul_return / (len(df) / 252))*100:.2f}%")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Price series
ax = axes[0, 0]
ax.plot(df.index, df['A'], label='Stock A', linewidth=1)
ax.plot(df.index, slope * df['B'], label=f'{slope:.3f}×B', linewidth=1, alpha=0.7)
ax.set_title('Price Series & Hedge Ratio')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Spread
ax = axes[0, 1]
ax.plot(df.index, df['spread'], label='Spread', linewidth=1)
ax.plot(df.index, df['mean_spread'], label='Mean', linewidth=2, alpha=0.7)
ax.fill_between(df.index, df['mean_spread'] - 2*df['std_spread'], 
                df['mean_spread'] + 2*df['std_spread'], alpha=0.2, label='±2σ')
ax.set_title('Spread with Bollinger Bands')
ax.set_ylabel('Spread ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Z-score with signals
ax = axes[1, 0]
ax.plot(df.index, df['z_score'], label='Z-score', linewidth=1, color='blue')
ax.axhline(y=2, color='red', linestyle='--', alpha=0.5)
ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
ax.set_title('Trading Signals (Z-score)')
ax.set_ylabel('Z-Score')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Trade returns distribution
if len(trades_df) > 0:
    ax = axes[1, 1]
    ax.hist(trades_df['pnl_pct'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(trades_df['pnl_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {trades_df['pnl_pct'].mean():.3f}%")
    ax.set_title('Trade Returns Distribution')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- Cointegration essential: Ensures spread mean-reverts")
print(f"- Hedge ratio (β): Controls dollar neutrality")
print(f"- Z-score signals: Entry at ±2, exit at ±0.5 (typical)")
print(f"- Win rate typically 50-55% (slightly positive edge)")
print(f"- Diversify: 1 pair risky; 50+ pairs stabilize returns")