import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Synthetic OHLCV data for multiple stocks
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
n_days = len(dates)

# Stock characteristics (realistic)
stocks = {
    'AAPL': {'vol_daily_pct': 1.5, 'volume_M': 60, 'name': 'Apple (liquid)'},
    'MSFT': {'vol_daily_pct': 1.2, 'volume_M': 30, 'name': 'Microsoft (liquid)'},
    'XYZ':  {'vol_daily_pct': 4.0, 'volume_M': 5, 'name': 'Illiquid (small-cap)'},
    'PENNY': {'vol_daily_pct': 8.0, 'volume_M': 0.5, 'name': 'Penny stock (v.illiquid)'},
}

# Generate synthetic OHLCV
data_dict = {}

for ticker, params in stocks.items():
    # Simulate returns (GBM-like)
    returns = np.random.normal(0, params['vol_daily_pct']/100, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Volume with mean reversion
    volume_base = params['volume_M']
    vol_shocks = np.random.normal(1, 0.3, n_days)
    volumes_M = np.maximum(volume_base * vol_shocks, 0.1)  # $M
    
    # OHLC (simplified)
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume_M': volumes_M,
    })
    
    df['close_prev'] = df['close'].shift(1)
    df['return_pct'] = np.abs(df['close'].pct_change() * 100)  # Absolute % change
    df['return_bps'] = df['return_pct'] * 100  # Convert to bps
    
    # Amihud
    df['amihud'] = df['return_bps'] / df['volume_M']
    
    # Handle NaN (first day, zero volume)
    df['amihud'] = df['amihud'].replace([np.inf, -np.inf], np.nan)
    
    # Rolling average (20-day)
    df['amihud_20d'] = df['amihud'].rolling(20, min_periods=5).mean()
    
    data_dict[ticker] = df

# Merge all stocks
all_data = pd.concat([df.assign(ticker=ticker) for ticker, df in data_dict.items()], ignore_index=True)

# Summary statistics
print("AMIHUD SUMMARY STATISTICS\n")
print("=" * 80)

for ticker, df in data_dict.items():
    name = stocks[ticker]['name']
    amihud_mean = df['amihud'].mean()
    amihud_median = df['amihud'].median()
    amihud_std = df['amihud'].std()
    
    print(f"\n{ticker}: {name}")
    print(f"  Mean Amihud:    {amihud_mean:.6f}")
    print(f"  Median Amihud:  {amihud_median:.6f}")
    print(f"  Std Dev:        {amihud_std:.6f}")
    print(f"  Min–Max:        {df['amihud'].min():.6f} – {df['amihud'].max():.6f}")
    print(f"  Avg daily vol:  ${df['volume_M'].mean():.1f}M")

# Correlation analysis
print("\n\nCORRELATION ANALYSIS")
print("=" * 80)

pivot_amihud = all_data.pivot_table(values='amihud_20d', index='date', columns='ticker')
print("\n20-Day Rolling Amihud Correlation:")
print(pivot_amihud.corr().round(3))

# Persistence (AR(1))
print("\n\nAMIHUD PERSISTENCE (Auto-Regressive)")
for ticker, df in data_dict.items():
    amihud_clean = df['amihud'].dropna()
    lag1 = amihud_clean.iloc[:-1].values
    current = amihud_clean.iloc[1:].values
    
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(lag1, current)
    
    print(f"{ticker}: AR(1) coeff = {slope:.3f} (R² = {r_value**2:.3f})")

# Time-of-year pattern
print("\n\nTIME-OF-YEAR PATTERN")
all_data['month'] = pd.to_datetime(all_data['date']).dt.month
monthly_amihud = all_data.groupby(['ticker', 'month'])['amihud'].mean().unstack()
print("\nAverage Amihud by Month:")
print(monthly_amihud.round(6))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Time series of Amihud
ax = axes[0, 0]
for ticker in stocks.keys():
    df = data_dict[ticker]
    ax.plot(df['date'], df['amihud_20d'], label=ticker, linewidth=2, alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Amihud (20-day rolling avg)')
ax.set_title('Liquidity Over Time: Amihud Ratio')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Amihud distribution (boxplot)
ax = axes[0, 1]
amihud_data = [data_dict[ticker]['amihud'].dropna().values for ticker in stocks.keys()]
bp = ax.boxplot(amihud_data, labels=stocks.keys(), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Amihud')
ax.set_title('Amihud Distribution by Stock')
ax.grid(axis='y', alpha=0.3)

# Plot 3: Amihud vs Volume (scatter)
ax = axes[1, 0]
for ticker in stocks.keys():
    df = data_dict[ticker]
    # Remove NaN
    valid_mask = df['amihud'].notna()
    ax.scatter(df.loc[valid_mask, 'volume_M'], df.loc[valid_mask, 'amihud'], 
               alpha=0.5, s=20, label=ticker)
ax.set_xlabel('Daily Volume ($M)')
ax.set_ylabel('Amihud')
ax.set_title('Amihud vs Trading Volume')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
ax.grid(alpha=0.3, which='both')

# Plot 4: Monthly pattern
ax = axes[1, 1]
for ticker in stocks.keys():
    month_amihud = data_dict[ticker].groupby('month')['amihud'].mean()
    ax.plot(month_amihud.index, month_amihud.values, 'o-', label=ticker, linewidth=2, markersize=6)
ax.set_xlabel('Month')
ax.set_ylabel('Average Amihud')
ax.set_title('Seasonal Pattern: Amihud by Month')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.show()

# Practical application: Execution algorithm adjustment
print("\n\nEXECUTION ALGORITHM: DYNAMIC PARTICIPATION BASED ON AMIHUD")
print("=" * 80)
print("\nParticipation Rate (target % of market volume):")
print("  Amihud < 0.0001 (liquid): 30% participation")
print("  Amihud 0.0001–0.001: 20% participation")
print("  Amihud > 0.001 (illiquid): 10% participation")

for ticker, df in data_dict.items():
    recent_amihud = df['amihud_20d'].iloc[-1]  # Most recent 20-day avg
    
    if recent_amihud < 0.0001:
        par_rate = 30
        guidance = "Aggressive execution OK"
    elif recent_amihud < 0.001:
        par_rate = 20
        guidance = "Normal execution"
    else:
        par_rate = 10
        guidance = "Patient execution recommended"
    
    print(f"\n{ticker}: Current 20-d Amihud = {recent_amihud:.6f}")
    print(f"  → Recommended participation: {par_rate}%")
    print(f"  → Strategy: {guidance}")