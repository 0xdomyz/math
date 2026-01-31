# Mean Reversion Strategies

## 1. Concept Skeleton
**Definition:** Trading strategy exploiting tendency of asset prices to revert to historical average after deviations; assumes mean-reverting process; counterintuitive to momentum (assumes prices overshoot equilibrium)  
**Purpose:** Generate alpha from temporary mispricings; profit from price oscillations around fair value; reduce drawdowns during trending markets via contrarian positioning  
**Prerequisites:** Time series analysis (stationarity), mean-reverting processes, technical indicators, volatility normalization, portfolio construction

## 2. Comparative Framing
| Aspect | Pairs Trading | Bollinger Bands | Z-Score | Kalman Filter | Ornstein-Uhlenbeck |
|--------|--------------|------------------|---------|---------------|-------------------|
| **Concept** | Cointegrated spreads | Deviation from bands | Std deviation from mean | Adaptive filtering | Stochastic mean reversion |
| **Signal** | Spread > threshold | Price outside ±2σ | |Z| > 2 | Predicted vs actual | OU mean |
| **Holding Period** | Days to weeks | Days to hours | Hours to days | Adaptive | Weeks to months |
| **Data Requirements** | Two correlated assets | Single asset history | Single asset history | Calibrated model | Pair parameters |
| **False Signal Rate** | Moderate (cointegration breaks) | High in trending markets | Moderate (regime-dependent) | Low (adaptive) | Low (if calibrated) |
| **Profitability** | +1-3% annual | +0.5-1.5% annual | +1-2% annual | +2-4% annual | +1.5-3% annual |
| **Regime Dependency** | High (cointegration breaks) | Extreme (useless in trends) | Moderate | Low (adaptive handles regimes) | Moderate (breaks in crises) |
| **Implementation Complexity** | Moderate | Simple | Simple | Complex (Kalman) | Moderate |

## 3. Examples + Counterexamples

**Bollinger Band Success (2015 Oil Crash):**  
Oil price: $60 (below 2-year low + 2σ). Signal: "Buy oil, too oversold." Subsequent 18 months: Oil recovers to $70+.  
Trade: Buy oil futures, hold 3 months. Profit: +15% (reverting to mean).

**Mean Reversion Failure (2017 Tech Momentum):**  
FAANG stocks (Facebook, Apple, Amazon, Netflix, Google) trade 3σ above 5-year mean. Contrarian signal: "Too expensive, revert downward."  
Reality: Prices accelerate upward for 2 more years (+100% total). Mean reversion trade = -50% loss (caught in momentum squeeze).

**Pairs Trading Success (GLD vs GDX):**  
Gold (GLD) and gold miners (GDX) historically correlated. March 2020: GDX decouples -50% vs GLD -20%. Signal: Spread too wide, revert.  
Trade: Long GDX, short GLD. 3 months later: GDX recovers +80%, GLD +5%. Spread converges. Profit: +40%.

**Pairs Trading Failure (Lehman 2008):**  
Financial sector pairs (JPM vs Lehman, or Bank of America vs Lehman). Correlation: 0.95 for 10 years. September 2008: Lehman collapses, JPM survives.  
Pairs trader: "Spread too wide, revert to normal." Reality: Lehman = $0, JPM = $30. Correlations break in crises. Loss: Entire capital (unleveraged).

**Z-Score Bounce (VIX Spike):**  
VIX = 15 (normal). Then drops to 10 (2σ below mean). Z-score = -2. Signal: "VIX too low, revert upward." Buy VIX calls (bet on vol rise).  
2017: VIX stays <15 for 6 months (volatility regime shift lower). Mean = 10 not 15. Trade loses: -80% (model assumptions break).

**Ornstein-Uhlenbeck Success (GBP/USD):**  
GBP/USD mean-reverting around 1.30 USD/GBP over 20 years. OU model: Half-life ~3 weeks (time to revert 50% of shock).  
When GBP/USD = 1.20 (1σ below): Bet long GBP. Within 3 weeks: 1.28 (converging back). Profit: +2-3%.

## 4. Layer Breakdown
```
Mean Reversion Strategy Architecture:

├─ Signal Generation
│  ├─ Statistical Mean Detection:
│  │   ├─ Rolling mean: Average of last N periods (e.g., 20, 50, 100 days)
│  │   ├─ Exponential moving average (EMA): Recent data weighted higher
│  │   │   EMA(t) = α × Price(t) + (1-α) × EMA(t-1)
│  │   ├─ Kalman filter estimate: Adaptive mean (updates with each price)
│  │   └─ Long-term equilibrium: Fundamental value (from models/economic theory)
│  │
│  ├─ Deviation Quantification:
│  │   ├─ Absolute deviation: Price - Mean
│  │   ├─ Percentage deviation: (Price - Mean) / Mean
│  │   ├─ Standard deviation units (Z-score):
│  │   │   Z = (Price - Mean) / Std Dev
│  │   │   Signal: Z > 2 (overbought) or Z < -2 (oversold)
│  │   ├─ Bollinger Bands: Bands ± 2σ around mean
│  │   │   Buy when Price < Lower Band
│  │   │   Sell when Price > Upper Band
│  │   └─ Distance from equilibrium scaled by volatility
│  │
│  ├─ Stochastic Models:
│  │   ├─ Ornstein-Uhlenbeck process:
│  │   │   dX(t) = κ(μ - X(t))dt + σ dW(t)
│  │   │   - κ: Mean reversion speed (faster = quicker convergence)
│  │   │   - μ: Long-term mean
│  │   │   - σ: Volatility
│  │   │   - Half-life: ln(2)/κ (time to revert 50% of shock)
│  │   │   - Signal: When X(t) < μ-σ, expected return positive
│  │   └─ Vasicek model: Similar to OU, used in interest rates
│  │
│  └─ Pairs Trading Signal:
│      ├─ Spread = Price(A) - β × Price(B)
│      │   β estimated from cointegrating regression
│      ├─ Cointegration test:
│      │   - Johansen test: Check if spread is I(0) (stationary)
│      │   - If I(0): Pairs viable; spread reverts to mean
│      │   - If I(1): Not cointegrated; avoid pairs trade
│      ├─ Signal: When Spread > +2σ, short A/long B (revert)
│      │         When Spread < -2σ, long A/short B (revert)
│      └─ Advantage: Market-neutral (β-hedged), less directional risk
│
├─ Portfolio Construction
│  ├─ Single-Asset Mean Reversion:
│  │   ├─ Entry: Signal triggered (e.g., Z < -2)
│  │   ├─ Position size: Kelly fraction of bankroll
│  │   ├─ Exit: Revert to mean (Close at +0σ) or stop-loss (-3σ)
│  │   ├─ Risk-reward ratio: Win if reverts in N days (expected +2%), risk -5%
│  │   └─ Holding period: Optimized based on OU half-life
│  │
│  ├─ Pairs Trading Portfolio:
│  │   ├─ Long basket of mean-reverting pairs
│  │   ├─ Hedge out market risk (beta-neutral)
│  │   ├─ Sector diversification: Tech pairs, finance pairs, energy pairs
│  │   ├─ Typical construction: 10-30 pairs, equal-risk weighting
│  │   ├─ Turnover: ~100-200% annual (pairs break, need new matches)
│  │   └─ Performance target: +3-8% annual with 5-10% volatility
│  │
│  ├─ Multi-Scale Mean Reversion:
│  │   ├─ Combine signals across timeframes (1h, 4h, 1d, 1w)
│  │   ├─ Stronger signal when multiple timeframes agree
│  │   ├─ Example: 1-day oversold + 1-week oversold = higher confidence
│  │   └─ Reduces false signals (single-timeframe noise)
│  │
│  └─ Risk Controls:
│      ├─ Stop-loss: Exit if deviation exceeds 3σ (tail risk)
│      ├─ Time-based exit: Liquidate if no reversion after N days
│      ├─ Correlation monitoring: Exit pairs if correlation breaks
│      ├─ Regime filters: Disable in strongly trending periods
│      └─ Position sizing: Scale down in high-vol periods
│
├─ Challenges & Failure Modes
│  ├─ Regime Breaks:
│  │   ├─ Financial crisis: Correlations crash to 1 (all risky assets down)
│  │   ├─ Structural changes: Industry disruption changes equilibrium
│  │   ├─ Central bank policy: QE/tightening creates new regimes
│  │   ├─ Crowding: Too many mean-reversion traders → trend reversal
│  │   └─ Detection: Monitor correlation matrices, break detection algorithms
│  │
│  ├─ False Signals in Trends:
│  │   ├─ Asset in uptrend: Dips are buying opportunities, not sell signals
│  │   ├─ Example: TSLA 2020-2021 trending up, "oversold" dips meant +buy
│  │   ├─ Traditional mean reversion: Sell oversold asset = -loss
│  │   ├─ Solution: Add trend filter (ignore signals against strong trend)
│  │   └─ Dual-regime system: Momentum in trends, mean-reversion in ranges
│  │
│  ├─ Cointegration Breaks (Pairs Trading):
│  │   ├─ Pairs cointegrated for 10 years, then break
│  │   ├─ Example: JPM (bank diversified) vs GS (Goldman Sachs, prop-heavy)
│  │   ├─2008: JPM holds up, GS near collapse; spread doesn't revert
│  │   ├─ Lesson: Monitor cointegration continuously
│  │   └─ Response: Dynamically re-estimate pairs, test robustness
│  │
│  ├─ Parameter Estimation Risk:
│  │   ├─ Mean estimated from history; changes during structural breaks
│  │   ├─ Volatility (σ) spikes during crises; Z-score becomes unreliable
│  │   ├─ OU half-life calibrated on old data; may be outdated
│  │   └─ Solution: Use Kalman filter for adaptive estimation
│  │
│  └─ Crash/Black Swan Risk:
│      ├─ Market crashes cause extreme deviations beyond mean-reversion timescale
│      ├─ Example: March 2020, stocks down -35%; mean reversion signal = buy
│      ├─ Doesn't revert for weeks; max drawdown -50%
│      ├─ Tail hedge: Buy puts on highly leveraged strategies
│      └─ Diversification: Don't concentrate mean-reversion capital
│
├─ Refinements & Advanced Techniques
│  ├─ Kalman Filtering:
│  │   ├─ Adaptive mean estimate updating in real-time
│  │   ├─ Advantage: Detects regime changes quickly
│  │   ├─ Less lag than rolling mean
│  │   └─ Reduced false signals compared to static thresholds
│  │
│  ├─ Vasicek Mean Reversion Model:
│  │   ├─ Probability-weighted exit targets
│  │   ├─ More sophisticated than simple Z-score
│  │   ├─ Accounts for mean reversion speed
│  │   └─ Improved position sizing
│  │
│  ├─ Copula-Based Pairs:
│  │   ├─ Non-linear dependency instead of correlation
│  │   ├─ Captures tail dependence
│  │   ├─ Detects breaks earlier (tail events show first)
│  │   └─ More robust to regime changes
│  │
│  └─ Ensemble Mean Reversion:
│      ├─ Combine multiple reversion signals (Bollinger, Z-score, OU, etc.)
│      ├─ Voting system: Trade if 2+ signals agree
│      ├─ Weighted ensemble: Weight by predictive power
│      └─ Reduces false positives
│
└─ Empirical Performance Profile
   ├─ Historical returns (1990-2023): +2-6% annual, varies by asset class
   ├─ Best in: Range-bound markets, low-trending environments
   ├─ Worst in: Strong momentum markets (2017, 2020+), crises
   ├─ Sharpe ratio: 0.5-0.9 (moderate risk-adjusted returns)
   ├─ Max drawdown: -20 to -40% (crashes), but less dramatic than buy-and-hold
   ├─ Win rate: 50-55% (slightly better than random)
   ├─ Profit factor: 1.2-1.5 (average win size > average loss size)
   └─ Trend: Declining performance (more crowded, faster mean reversion dissipation)
```

**Interaction:** Signal detection (Z-score) → Cointegration check (pairs) → Entry signal → Position sizing → Monitoring for regime break → Exit/rebalancing.

## 5. Mini-Project
Backtest mean reversion pairs trading strategy:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller

# Generate two cointegrated price series
np.random.seed(42)
n_days = 1000

# Common factor
common_factor = np.cumsum(np.random.normal(0.0005, 0.01, n_days))

# Two stocks with same drift (cointegrated)
price_A = 100 * np.exp(common_factor + np.random.normal(0, 0.005, n_days))
price_B = 50 * np.exp(common_factor + np.random.normal(0, 0.005, n_days))

dates = pd.date_range('2015-01-01', periods=n_days, freq='D')
df = pd.DataFrame({'A': price_A, 'B': price_B}, index=dates)

print("="*100)
print("MEAN REVERSION PAIRS TRADING BACKTEST")
print("="*100)

# Step 1: Cointegration test
print(f"\nStep 1: Cointegration Analysis")
print(f"-" * 50)

score, p_value, _ = coint(df['A'], df['B'])
print(f"Cointegration test p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"✓ Series are cointegrated (stationary spread)")
else:
    print(f"✗ Series are NOT cointegrated")

# Regression to estimate hedge ratio
from scipy import stats
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(df['B'], df['A'])
hedge_ratio = slope

print(f"\nHedge Ratio (β): {hedge_ratio:.4f}")
print(f"R-squared: {r_value**2:.4f}")

# Step 2: Calculate spread
print(f"\nStep 2: Spread Calculation & Statistics")
print(f"-" * 50)

df['spread'] = df['A'] - hedge_ratio * df['B']
df['mean'] = df['spread'].rolling(window=20).mean()
df['std'] = df['spread'].rolling(window=20).std()
df['z_score'] = (df['spread'] - df['mean']) / df['std']

print(f"Mean spread: {df['spread'].mean():.2f}")
print(f"Std dev spread: {df['spread'].std():.2f}")
print(f"Min Z-score: {df['z_score'].min():.2f}")
print(f"Max Z-score: {df['z_score'].max():.2f}")

# Stationarity test on spread
adf_stat, adf_p, _, _, _, _ = adfuller(df['spread'].dropna())
print(f"\nADF test p-value: {adf_p:.4f}")
if adf_p < 0.05:
    print(f"✓ Spread is stationary (I(0))")
else:
    print(f"✗ Spread is NOT stationary")

# Step 3: Generate trading signals
print(f"\nStep 3: Trading Signals")
print(f"-" * 50)

threshold_entry = 2.0  # Z-score threshold
threshold_exit = 0.0   # Exit at mean

df['signal'] = 0
df.loc[df['z_score'] > threshold_entry, 'signal'] = -1  # Short A, Long B
df.loc[df['z_score'] < -threshold_entry, 'signal'] = 1  # Long A, Short B

print(f"Entry threshold (Z-score): ±{threshold_entry}")
print(f"Number of entry signals: {(df['signal'] != 0).sum()}")

# Step 4: Backtest with position tracking
print(f"\nStep 4: Backtest Execution")
print(f"-" * 50)

position = 0
trades = []
entry_price = None
entry_z = None

for i in range(1, len(df)):
    if position == 0:
        # Check for entry signal
        if df['signal'].iloc[i] != 0:
            position = df['signal'].iloc[i]
            entry_price_A = df['A'].iloc[i]
            entry_price_B = df['B'].iloc[i]
            entry_z = df['z_score'].iloc[i]
            entry_date = df.index[i]
    else:
        # Check for exit signal
        current_z = df['z_score'].iloc[i]
        if (position == 1 and current_z > threshold_exit) or (position == -1 and current_z < -threshold_exit):
            # Exit trade
            exit_price_A = df['A'].iloc[i]
            exit_price_B = df['B'].iloc[i]
            
            # Calculate P&L
            pnl_A = position * (exit_price_A - entry_price_A) / entry_price_A
            pnl_B = -position * hedge_ratio * (exit_price_B - entry_price_B) / entry_price_B
            total_return = (pnl_A + pnl_B) * 100  # %
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[i],
                'entry_z': entry_z,
                'exit_z': current_z,
                'return_pct': total_return,
                'days_held': (df.index[i] - entry_date).days,
            })
            
            position = 0

trades_df = pd.DataFrame(trades)

print(f"Total trades: {len(trades_df)}")
print(f"Winning trades: {(trades_df['return_pct'] > 0).sum()}")
print(f"Losing trades: {(trades_df['return_pct'] < 0).sum()}")
print(f"Win rate: {(trades_df['return_pct'] > 0).sum() / len(trades_df) * 100:.1f}%")
print(f"\nAverage return per trade: {trades_df['return_pct'].mean():.3f}%")
print(f"Median return per trade: {trades_df['return_pct'].median():.3f}%")
print(f"Max return: {trades_df['return_pct'].max():.3f}%")
print(f"Min return: {trades_df['return_pct'].min():.3f}%")
print(f"Std dev: {trades_df['return_pct'].std():.3f}%")
print(f"Average days held: {trades_df['days_held'].mean():.0f}")

# Cumulative return simulation
cumulative_returns = (1 + trades_df['return_pct'] / 100).cumprod()
total_return = cumulative_returns.iloc[-1] - 1

print(f"\nCumulative return: {total_return * 100:.1f}%")
print(f"Annualized return: {(total_return / (len(df) / 252)) * 100:.1f}%")

# VISUALIZATION
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Price series A and B
ax = axes[0, 0]
ax.plot(df.index, df['A'], label='Price A', linewidth=1.5)
ax.plot(df.index, hedge_ratio * df['B'], label=f'{hedge_ratio:.2f} × Price B', linewidth=1.5, alpha=0.7)
ax.set_title('Cointegrated Price Series')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Spread
ax = axes[0, 1]
ax.plot(df.index, df['spread'], label='Spread', linewidth=1)
ax.plot(df.index, df['mean'], label='20-day Mean', linewidth=2, color='orange')
ax.fill_between(df.index, df['mean'] - 2*df['std'], df['mean'] + 2*df['std'], alpha=0.2, color='red')
ax.set_title('Spread with ±2σ Bands')
ax.set_ylabel('Spread ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Z-score with entry signals
ax = axes[1, 0]
ax.plot(df.index, df['z_score'], label='Z-score', linewidth=1, color='blue')
ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry threshold')
ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
entry_long = df[df['signal'] == 1]
entry_short = df[df['signal'] == -1]
ax.scatter(entry_long.index, entry_long['z_score'], color='green', marker='^', s=100, label='Entry Long', zorder=5)
ax.scatter(entry_short.index, entry_short['z_score'], color='red', marker='v', s=100, label='Entry Short', zorder=5)
ax.set_title('Z-Score with Entry Signals')
ax.set_ylabel('Z-Score')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Trade returns distribution
ax = axes[1, 1]
ax.hist(trades_df['return_pct'], bins=30, edgecolor='black', alpha=0.7)
ax.axvline(trades_df['return_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {trades_df["return_pct"].mean():.3f}%')
ax.set_title('Trade Returns Distribution')
ax.set_xlabel('Return (%)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 5: Cumulative returns
ax = axes[2, 0]
ax.plot(range(len(cumulative_returns)), cumulative_returns, marker='o', linewidth=2)
ax.set_title('Cumulative Strategy Returns')
ax.set_xlabel('Trade Number')
ax.set_ylabel('Cumulative Return (Multiple)')
ax.grid(alpha=0.3)

# Plot 6: Holding periods
ax = axes[2, 1]
ax.hist(trades_df['days_held'], bins=20, edgecolor='black', alpha=0.7)
ax.axvline(trades_df['days_held'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {trades_df["days_held"].mean():.0f} days')
ax.set_title('Trade Holding Periods')
ax.set_xlabel('Days Held')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print(f"STRATEGY INSIGHTS")
print(f"="*100)
print(f"- Strategy exploits mean-reverting spreads between cointegrated assets")
print(f"- Best trades: Fast reversion (held <50 days) with +0.5-2% returns")
print(f"- Worst trades: Slow reversion or regime break (held >100 days) with -1-5% losses")
print(f"- Sensitivity: Breaks down if cointegration fails (monitor p-value)")
print(f"- Improvements: Add regime filter, dynamic threshold, portfolio diversification")
```

## 6. Challenge Round
- Test pairs cointegration for 10 industry-matched stocks; rank by strength
- Backtest Bollinger Band strategy vs Z-score strategy on 5 assets
- Calculate OU half-life for mean-reverting asset; design position sizing
- Implement Kalman filter for adaptive mean estimation; compare to rolling mean
- Design pairs portfolio: Select 15 cointegrated pairs, equal-risk weighting, measure drawdowns

## 7. Key References
- [Poterba & Summers, "Mean Reversion in Stock Prices" (1988), JFE](https://www.jstor.org/stable/1913208) — Foundational mean-reversion evidence
- [Gatev, Goetzmann & Rouwenhorst, "Pairs Trading" (2006), JFE](https://www.sciencedirect.com/science/article/pii/S0304405X06000845) — Pairs trading mechanics and profitability
- [Avellaneda & Lee, "Statistical Arbitrage" (2010), Working Paper](https://arxiv.org/abs/1002.3213) — OU modeling and optimal portfolio construction
- [Engle & Granger, "Cointegration & Error Correction" (1987), Econometrica](https://www.jstor.org/stable/1913236) — Theoretical foundation

---
**Status:** Established factor strategy (1980s-present, peak 2000s-2010s, declining) | **Complements:** Momentum Strategies, Statistical Arbitrage, Market Microstructure, Risk Management
