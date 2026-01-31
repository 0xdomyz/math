# Market Depth Imbalance

## 1. Concept Skeleton
**Definition:** Asymmetry between buy-side and sell-side order book depth, signaling directional trading pressure  
**Purpose:** Predicts short-term price movements, informs market making strategies, measures informed flow  
**Prerequisites:** Order book depth, limit order mechanics, price discovery

## 2. Comparative Framing
| Measure | Depth Imbalance | Order Flow Imbalance | Trade Imbalance | Volume Imbalance |
|---------|-----------------|----------------------|-----------------|------------------|
| **Source** | Limit orders (book) | All order submissions | Executed trades only | Trade volume |
| **Timing** | Pre-trade (passive) | Submission time | Post-trade (realized) | Execution |
| **Signal** | Supply/demand pressure | Order aggressiveness | Actual buying/selling | Size-weighted flow |
| **Horizon** | Seconds to minutes | Intraday | Tick-by-tick | Session |

## 3. Examples + Counterexamples

**Positive Imbalance (Bullish):**  
Bid depth=50K shares, Ask depth=20K shares → Imbalance = (50K-20K)/(50K+20K) = +0.43 → Price likely rises

**Negative Imbalance (Bearish):**  
Bid depth=15K shares, Ask depth=45K shares → Imbalance = (15K-45K)/(15K+45K) = -0.50 → Price likely falls

**False Signal:**  
Massive buy-side imbalance from single iceberg order with no intention to execute (strategic quote) → Price doesn't move

## 4. Layer Breakdown
```
Depth Imbalance Framework:
├─ Calculation Methods:
│   ├─ Simple Imbalance: (BidDepth - AskDepth) / TotalDepth
│   ├─ Volume-Weighted: ∑(BidQty × Weight) - ∑(AskQty × Weight)
│   ├─ Price-Weighted: Decay factor for levels away from BBO
│   └─ Multi-Level: Aggregate across n price levels
├─ Cao et al (2009) Specification:
│   ├─ Depth at Best Bid (D^bid): Quantity at best bid
│   ├─ Depth at Best Ask (D^ask): Quantity at best ask
│   ├─ Imbalance: OIB = (D^bid - D^ask) / (D^bid + D^ask)
│   └─ Prediction: E[r_{t+1}] ∝ OIB_t
├─ Predictive Power:
│   ├─ Short-Horizon Returns: Significant correlation (R² ~ 5-15%)
│   ├─ Decay Pattern: Strongest at <1 minute, fades by 30 min
│   ├─ Information Content: Reveals hidden supply/demand
│   └─ Conditional on Volatility: Stronger during calm periods
├─ Information Asymmetry Link:
│   ├─ Informed Traders: Submit limit orders strategically
│   ├─ Inventory Management: Market makers adjust depth
│   ├─ Liquidity Provision: Imbalance reflects MM positioning
│   └─ Toxic Flow: Extreme imbalance signals informed activity
├─ Trading Strategies:
│   ├─ Momentum: Buy when imbalance positive, sell when negative
│   ├─ Mean Reversion: Fade extreme imbalances
│   ├─ Market Making: Lean against imbalance to earn spread
│   └─ Execution Timing: Delay when adverse imbalance
└─ Limitations:
    ├─ Hidden Orders: Iceberg/dark liquidity not captured
    ├─ Cancelled Orders: Strategic quotes inflate imbalance
    ├─ Cross-Venue: Single exchange view incomplete
    └─ Non-Stationarity: Relationship breaks during stress
```

**Interaction:** Imbalance → price pressure → market makers adjust → imbalance mean-reverts → continuous cycle

## 5. Mini-Project
Analyze depth imbalance predictive power:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

np.random.seed(42)

# Simulate order book with strategic traders
n_periods = 2000
tick_size = 0.01

# True fundamental value (random walk)
fundamental = 100 + np.cumsum(np.random.normal(0, 0.02, n_periods))

# Observable midpoint (adjusts toward fundamental with lag)
midpoint = np.zeros(n_periods)
midpoint[0] = 100
adjustment_speed = 0.2

for t in range(1, n_periods):
    midpoint[t] = midpoint[t-1] + adjustment_speed * (fundamental[t] - midpoint[t-1]) + \
                  np.random.normal(0, 0.01)

# Order book depth (influenced by informed traders knowing fundamental)
bid_depth = np.zeros(n_periods)
ask_depth = np.zeros(n_periods)

base_depth = 1000  # Base depth level

for t in range(n_periods):
    # Information asymmetry: traders know if fundamental > midpoint
    informed_signal = fundamental[t] - midpoint[t]
    
    # Informed traders add depth on side of true value
    if informed_signal > 0:  # Fundamental undervalued
        # More aggressive buying (more bid depth)
        bid_depth[t] = base_depth + abs(informed_signal) * 5000 + np.random.normal(0, 200)
        ask_depth[t] = base_depth - abs(informed_signal) * 2000 + np.random.normal(0, 200)
    else:  # Fundamental overvalued
        # More aggressive selling (more ask depth)
        bid_depth[t] = base_depth - abs(informed_signal) * 2000 + np.random.normal(0, 200)
        ask_depth[t] = base_depth + abs(informed_signal) * 5000 + np.random.normal(0, 200)
    
    # Ensure positive depths
    bid_depth[t] = max(100, bid_depth[t])
    ask_depth[t] = max(100, ask_depth[t])

# Calculate depth imbalance
total_depth = bid_depth + ask_depth
depth_imbalance = (bid_depth - ask_depth) / total_depth

# Calculate returns
returns = np.diff(midpoint)
returns_bps = (returns / midpoint[:-1]) * 10000

# Forward-looking returns (what happens after imbalance observed)
forward_horizons = [1, 5, 10, 30, 60]  # periods ahead
forward_returns = {}

for horizon in forward_horizons:
    fwd_ret = np.zeros(n_periods - horizon)
    for t in range(n_periods - horizon):
        fwd_ret[t] = (midpoint[t + horizon] - midpoint[t]) / midpoint[t] * 10000
    forward_returns[horizon] = fwd_ret

# Regression analysis: forward returns ~ imbalance
regression_results = {}

for horizon in forward_horizons:
    fwd_ret = forward_returns[horizon]
    imb = depth_imbalance[:len(fwd_ret)]
    
    # Linear regression
    mask = ~np.isnan(imb) & ~np.isnan(fwd_ret)
    if mask.sum() > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(imb[mask], fwd_ret[mask])
        regression_results[horizon] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value
        }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Depth imbalance and price over time
sample_period = slice(0, 500)
time_axis = np.arange(500)

ax1 = axes[0, 0]
ax2 = ax1.twinx()

ax1.plot(time_axis, depth_imbalance[sample_period], color='purple', linewidth=1.5, 
         label='Depth Imbalance')
ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax1.fill_between(time_axis, 0, depth_imbalance[sample_period], 
                 where=(depth_imbalance[sample_period] > 0), alpha=0.3, color='green', 
                 label='Bid Pressure')
ax1.fill_between(time_axis, 0, depth_imbalance[sample_period], 
                 where=(depth_imbalance[sample_period] < 0), alpha=0.3, color='red', 
                 label='Ask Pressure')

ax2.plot(time_axis, midpoint[sample_period], color='blue', linewidth=2, alpha=0.7, 
         label='Midpoint Price')

ax1.set_xlabel('Time Period')
ax1.set_ylabel('Depth Imbalance', color='purple')
ax2.set_ylabel('Midpoint Price ($)', color='blue')
ax1.set_title('Depth Imbalance vs Price')
ax1.tick_params(axis='y', labelcolor='purple')
ax2.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Scatter plot - imbalance vs forward returns (1-period)
horizon = 1
fwd_ret = forward_returns[horizon]
imb = depth_imbalance[:len(fwd_ret)]

axes[0, 1].scatter(imb, fwd_ret, alpha=0.3, s=10)
axes[0, 1].axhline(0, color='black', linewidth=0.5)
axes[0, 1].axvline(0, color='black', linewidth=0.5)

# Add regression line
mask = ~np.isnan(imb) & ~np.isnan(fwd_ret)
if mask.sum() > 0:
    slope = regression_results[horizon]['slope']
    intercept = regression_results[horizon]['intercept']
    r_sq = regression_results[horizon]['r_squared']
    
    x_fit = np.linspace(imb.min(), imb.max(), 100)
    y_fit = slope * x_fit + intercept
    axes[0, 1].plot(x_fit, y_fit, 'r--', linewidth=2, 
                   label=f'R²={r_sq:.4f}, β={slope:.2f}')

axes[0, 1].set_xlabel('Depth Imbalance (t)')
axes[0, 1].set_ylabel(f'Forward Return (t+{horizon}) [bps]')
axes[0, 1].set_title(f'Imbalance Predictive Power ({horizon}-period ahead)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

print("Depth Imbalance Predictive Power:")
print("=" * 70)
for horizon, results in regression_results.items():
    print(f"\n{horizon}-period ahead forecast:")
    print(f"  Slope (beta): {results['slope']:.4f}")
    print(f"  R-squared: {results['r_squared']:.4f}")
    print(f"  P-value: {results['p_value']:.6f}")
    if results['p_value'] < 0.01:
        print(f"  → Significant predictive power")

# Plot 3: R-squared decay over forecast horizon
horizons_list = list(regression_results.keys())
r_squared_list = [regression_results[h]['r_squared'] for h in horizons_list]
slope_list = [regression_results[h]['slope'] for h in horizons_list]

ax1 = axes[1, 0]
ax2 = ax1.twinx()

ax1.plot(horizons_list, r_squared_list, 'o-', color='blue', linewidth=2, 
         markersize=8, label='R²')
ax2.plot(horizons_list, slope_list, 's-', color='red', linewidth=2, 
         markersize=8, label='Beta')

ax1.set_xlabel('Forecast Horizon (periods)')
ax1.set_ylabel('R-squared', color='blue')
ax2.set_ylabel('Regression Slope (beta)', color='red')
ax1.set_title('Predictive Power Decay')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
ax1.grid(alpha=0.3)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot 4: Trading strategy based on imbalance
# Strategy: Go long when imbalance > threshold, short when < -threshold
threshold = 0.1
positions = np.zeros(n_periods)
positions[depth_imbalance > threshold] = 1  # Long
positions[depth_imbalance < -threshold] = -1  # Short

strategy_returns = positions[:-1] * returns_bps
cumulative_returns = np.cumsum(strategy_returns)

# Buy-and-hold benchmark
buy_hold_returns = np.cumsum(returns_bps)

axes[1, 1].plot(cumulative_returns, linewidth=2, label='Imbalance Strategy')
axes[1, 1].plot(buy_hold_returns, linewidth=2, alpha=0.7, label='Buy & Hold')
axes[1, 1].set_xlabel('Time Period')
axes[1, 1].set_ylabel('Cumulative Return (bps)')
axes[1, 1].set_title(f'Trading Strategy (threshold={threshold})')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Strategy statistics
sharpe_ratio = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(252)
hit_rate = (strategy_returns > 0).mean()

print(f"\nTrading Strategy Performance:")
print(f"Threshold: ±{threshold}")
print(f"Cumulative Return: {cumulative_returns[-1]:.2f} bps")
print(f"Mean Return per Period: {strategy_returns.mean():.4f} bps")
print(f"Volatility: {strategy_returns.std():.4f} bps")
print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
print(f"Hit Rate: {hit_rate*100:.1f}%")
print(f"Trades: Long={np.sum(positions==1)}, Short={np.sum(positions==-1)}")

plt.tight_layout()
plt.show()

# Quantile analysis
imbalance_quantiles = pd.qcut(depth_imbalance[:len(fwd_ret)], q=5, labels=False)
quantile_returns = []

for q in range(5):
    mask = (imbalance_quantiles == q)
    q_ret = fwd_ret[mask].mean()
    quantile_returns.append(q_ret)

print(f"\nReturn by Imbalance Quintile (1-period ahead):")
for q, ret in enumerate(quantile_returns):
    print(f"  Q{q+1} (most {'bearish' if q==0 else 'bullish' if q==4 else 'neutral'}): {ret:.4f} bps")

# Test monotonicity
if len(quantile_returns) == 5:
    spread = quantile_returns[4] - quantile_returns[0]
    print(f"\nQ5-Q1 Spread: {spread:.4f} bps")
    print("(Positive spread confirms imbalance predictive power)")

# Information coefficient
ic = np.corrcoef(imb[mask], fwd_ret[mask])[0, 1]
print(f"\nInformation Coefficient (IC): {ic:.4f}")
print("(Typical good alpha: IC > 0.05)")
```

## 6. Challenge Round
When does depth imbalance fail to predict returns?
- **Strategic quoting**: Spoofing/layering creates fake imbalance without real demand (illegal but happens pre-detection)
- **Iceberg orders**: Large hidden orders on one side not reflected in displayed imbalance
- **Cross-venue fragmentation**: Imbalance on one exchange offset by opposite imbalance elsewhere (need consolidated book)
- **Volatility regimes**: Predictive power breaks down during high volatility (uncertainty dominates supply/demand signals)
- **News events**: Information shocks overwhelm order book signals (imbalance lags fundamental revaluation)

## 7. Key References
- [Cao et al (2009) - Can Price Limits Help When Liquidity Crisis Hits?](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2009.01469.x)
- [Cont et al (2014) - The Price Impact of Order Book Events](https://www.sciencedirect.com/science/article/abs/pii/S0304405X13002675)
- [Chordia & Subrahmanyam (2004) - Order Imbalance and Individual Stock Returns](https://www.sciencedirect.com/science/article/abs/pii/S0304405X04000972)
- [Biais et al (1995) - An Empirical Analysis of the Limit Order Book](https://www.jstor.org/stable/2329299)

---
**Status:** Predictive order book signal | **Complements:** Order Book Depth, Price Discovery, Market Making
