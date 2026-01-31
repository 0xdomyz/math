import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Build momentum strategy and analyze crash risk

def calculate_momentum_scores(returns, lookback_period=126):
    """
    Calculate momentum scores for each asset (past 6 months = 126 trading days).
    """
    momentum = returns.rolling(lookback_period).apply(lambda x: (1 + x).prod() - 1)
    return momentum


def backtest_momentum_strategy(returns, top_n=10, rebalance_freq='Q'):
    """
    Backtest a simple momentum strategy:
    - Hold top N performers from past 6 months
    - Rebalance quarterly
    - Compare to buy-hold
    """
    # Get unique dates for rebalancing
    if rebalance_freq == 'Q':
        rebalance_dates = returns.resample('Q').last().index
    else:  # Monthly
        rebalance_dates = returns.resample('M').last().index
    
    strategy_returns = []
    buy_hold_returns = []
    positions = {}
    
    for t in returns.index:
        # Check if rebalance date
        if len(strategy_returns) > 0 and t not in rebalance_dates:
            # Continue holding current positions
            daily_return = returns.loc[t, list(positions.keys())].sum() / len(positions)
            strategy_returns.append(daily_return)
            buy_hold_returns.append(returns.loc[t].mean())
            continue
        
        # Rebalance: find top momentum performers
        if t not in returns.index[:252]:  # Need 6+ months of data
            strategy_returns.append(0)
            buy_hold_returns.append(returns.loc[t].mean())
            continue
        
        # Get recent momentum (past 6 months from this point)
        recent = returns.loc[:t].tail(126)
        momentum_scores = (1 + recent).prod() - 1
        
        # Select top N
        top_performers = momentum_scores.nlargest(top_n).index.tolist()
        positions = {stock: 1/top_n for stock in top_performers}
        
        # Calculate daily return
        if len(positions) > 0:
            daily_return = returns.loc[t, list(positions.keys())].sum() / len(positions)
            strategy_returns.append(daily_return)
        else:
            strategy_returns.append(0)
        
        buy_hold_returns.append(returns.loc[t].mean())
    
    strategy_returns = pd.Series(strategy_returns, index=returns.index)
    buy_hold_returns = pd.Series(buy_hold_returns, index=returns.index)
    
    return strategy_returns, buy_hold_returns


def calculate_drawdown(returns):
    """
    Calculate drawdown from cumulative returns.
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown


def detect_herd_crowding(returns, threshold_percentile=90):
    """
    Detect when momentum is becoming crowded (tail risk high).
    Crowding = when many stocks have extreme momentum simultaneously.
    """
    momentum = calculate_momentum_scores(returns)
    
    crowding_scores = []
    for t in momentum.index:
        recent_momentum = momentum.loc[t].dropna()
        if len(recent_momentum) == 0:
            crowding_scores.append(0)
            continue
        
        # Crowding = % of stocks in top 20% momentum
        extreme_performers = (recent_momentum > recent_momentum.quantile(0.80)).sum()
        crowding = extreme_performers / len(recent_momentum)
        crowding_scores.append(crowding)
    
    crowding_scores = pd.Series(crowding_scores, index=momentum.index)
    return crowding_scores


# Main Analysis
print("=" * 100)
print("MOMENTUM STRATEGY & HERDING CRASH ANALYSIS")
print("=" * 100)

# Download S&P 500 constituents data (or a subset)
# Use major sectors as proxy for diversified momentum
tickers = ['XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLRE', 'XLU', 'XLF', 'XLE']  # Sector ETFs

returns = yf.download(tickers, start='2015-01-01', end='2024-01-01', progress=False)['Adj Close'].pct_change().dropna()

print(f"\n1. DATA SUMMARY")
print("-" * 100)
print(f"Tickers: {', '.join(tickers)}")
print(f"Time period: 2015-2024 (9 years)")
print(f"Total observations: {len(returns)}")

# 2. Momentum scores analysis
print(f"\n2. MOMENTUM SCORES (Current)")
print("-" * 100)

momentum_scores = calculate_momentum_scores(returns, lookback_period=126)
latest_momentum = momentum_scores.iloc[-1].sort_values(ascending=False)

print(f"Top momentum performers (past 6 months):")
for ticker, score in latest_momentum.head(5).items():
    print(f"  {ticker}: {score:+.2%}")

print(f"\nLowest momentum (potential mean reversion):")
for ticker, score in latest_momentum.tail(3).items():
    print(f"  {ticker}: {score:+.2%}")

# 3. Crowding analysis
print(f"\n3. HERD CROWDING ANALYSIS")
print("-" * 100)

crowding_scores = detect_herd_crowding(returns)
current_crowding = crowding_scores.iloc[-1]
mean_crowding = crowding_scores.mean()

print(f"Current crowding level: {current_crowding:.1%}")
print(f"Historical mean crowding: {mean_crowding:.1%}")
print(f"Crowding assessment: ", end="")
if current_crowding > crowding_scores.quantile(0.75):
    print("HIGH (extreme positioning; crash risk elevated)")
elif current_crowding < crowding_scores.quantile(0.25):
    print("LOW (dispersed; momentum more stable)")
else:
    print("NORMAL")

# Peak crowding periods (when crashes happen)
peak_crowding_dates = crowding_scores.nlargest(3).index
print(f"\nPeak crowding periods (highest crash risk):")
for date in peak_crowding_dates:
    dd = calculate_drawdown(returns.loc[:date].sum(axis=1)).iloc[-1]
    print(f"  {date.date()}: Crowding={crowding_scores.loc[date]:.1%}, Drawdown={dd:.1%}")

# 4. Backtest momentum strategy
print(f"\n4. MOMENTUM STRATEGY BACKTEST")
print("-" * 100)

strategy_rets, buyhold_rets = backtest_momentum_strategy(returns, top_n=3, rebalance_freq='Q')

# Calculate metrics
strategy_cum = (1 + strategy_rets).cumprod()
buyhold_cum = (1 + buyhold_rets).cumprod()

strategy_annual_return = (strategy_cum.iloc[-1] ** (252 / len(strategy_rets)) - 1)
buyhold_annual_return = (buyhold_cum.iloc[-1] ** (252 / len(buyhold_rets)) - 1)

strategy_vol = strategy_rets.std() * np.sqrt(252)
buyhold_vol = buyhold_rets.std() * np.sqrt(252)

strategy_sharpe = strategy_annual_return / strategy_vol if strategy_vol > 0 else 0
buyhold_sharpe = buyhold_annual_return / buyhold_vol if buyhold_vol > 0 else 0

# Drawdowns
strategy_dd = calculate_drawdown(strategy_rets).min()
buyhold_dd = calculate_drawdown(buyhold_rets).min()

print(f"MOMENTUM STRATEGY (Top 3, Quarterly Rebalance):")
print(f"  Annual Return: {strategy_annual_return:.2%}")
print(f"  Annual Volatility: {strategy_vol:.2%}")
print(f"  Sharpe Ratio: {strategy_sharpe:.3f}")
print(f"  Max Drawdown: {strategy_dd:.2%}")

print(f"\nBUY-HOLD (Equal Weight):")
print(f"  Annual Return: {buyhold_annual_return:.2%}")
print(f"  Annual Volatility: {buyhold_vol:.2%}")
print(f"  Sharpe Ratio: {buyhold_sharpe:.3f}")
print(f"  Max Drawdown: {buyhold_dd:.2%}")

print(f"\nMOMENTUM OUTPERFORMANCE:")
print(f"  Return advantage: {(strategy_annual_return - buyhold_annual_return):.2%} p.a.")
print(f"  Risk adjustment: {(strategy_sharpe - buyhold_sharpe):.3f} Sharpe points")
print(f"  Drawdown comparison: {(strategy_dd - buyhold_dd):.2%} (worse if negative)")

# 5. Crash detection
print(f"\n5. MOMENTUM CRASH PERIODS DETECTED")
print("-" * 100)

# Find periods with extreme drawdowns (crashes)
strategy_dd_series = calculate_drawdown(strategy_rets)
crash_periods = strategy_dd_series[strategy_dd_series < -0.10]  # Drawdown > 10%

print(f"Number of >10% drawdowns: {len(crash_periods)}")
print(f"Average crash duration: {len(crash_periods) / len(crash_periods.groupby((crash_periods != crash_periods.shift()).cumsum())):.0f} days")

if len(crash_periods) > 0:
    worst_crash_date = strategy_dd_series.idxmin()
    worst_crash_amount = strategy_dd_series.min()
    print(f"Worst crash: {worst_crash_amount:.1%} on {worst_crash_date.date()}")

# 6. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Momentum scores over time
ax = axes[0, 0]
for ticker in latest_momentum.head(3).index:
    ax.plot(momentum_scores.index, momentum_scores[ticker], label=ticker, linewidth=2, alpha=0.8)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_title('Momentum Scores Over Time (Top 3 Recent)', fontweight='bold')
ax.set_ylabel('6-Month Momentum')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Crowding score
ax = axes[0, 1]
ax.plot(crowding_scores.index, crowding_scores, linewidth=2, color='steelblue')
ax.fill_between(crowding_scores.index, 0, crowding_scores, alpha=0.2)
ax.axhline(crowding_scores.mean(), color='red', linestyle='--', label='Mean')
ax.axhline(crowding_scores.quantile(0.75), color='orange', linestyle='--', label='75th %ile (High Risk)')
ax.set_title('Herd Crowding Score Over Time', fontweight='bold')
ax.set_ylabel('Crowding Level')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Cumulative returns comparison
ax = axes[0, 2]
ax.plot(strategy_cum.index, strategy_cum, label='Momentum Strategy', linewidth=2, color='green')
ax.plot(buyhold_cum.index, buyhold_cum, label='Buy-Hold', linewidth=2, color='blue')
ax.set_title('Cumulative Returns: Momentum vs Buy-Hold', fontweight='bold')
ax.set_ylabel('Cumulative Return (1.0 = start)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Drawdown comparison
ax = axes[1, 0]
strategy_dd_plot = calculate_drawdown(strategy_rets)
buyhold_dd_plot = calculate_drawdown(buyhold_rets)
ax.plot(strategy_dd_plot.index, strategy_dd_plot * 100, label='Momentum Strategy', linewidth=2, color='green', alpha=0.7)
ax.plot(buyhold_dd_plot.index, buyhold_dd_plot * 100, label='Buy-Hold', linewidth=2, color='blue', alpha=0.7)
ax.fill_between(strategy_dd_plot.index, 0, strategy_dd_plot * 100, alpha=0.1, color='green')
ax.set_title('Drawdown Over Time', fontweight='bold')
ax.set_ylabel('Drawdown (%)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Return distribution
ax = axes[1, 1]
ax.hist(strategy_rets * 100, bins=50, alpha=0.6, label='Momentum', color='green', edgecolor='black')
ax.hist(buyhold_rets * 100, bins=50, alpha=0.6, label='Buy-Hold', color='blue', edgecolor='black')
ax.axvline(strategy_rets.mean() * 100, color='green', linestyle='--', linewidth=2, label=f'Momentum Mean: {strategy_rets.mean()*100:.2f}%')
ax.set_title('Daily Return Distribution', fontweight='bold')
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 6: Risk-return scatter
ax = axes[1, 2]
strategies = [
    ('Buy-Hold', buyhold_annual_return, buyhold_vol),
    ('Momentum', strategy_annual_return, strategy_vol),
]
for name, ret, vol in strategies:
    color = 'blue' if 'Buy' in name else 'green'
    ax.scatter(vol * 100, ret * 100, s=400, alpha=0.6, label=name, color=color)
    ax.annotate(name, (vol * 100, ret * 100), fontsize=10, fontweight='bold')

ax.set_xlabel('Annual Volatility (%)')
ax.set_ylabel('Annual Return (%)')
ax.set_title('Risk-Return Profile', fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_herding_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Chart saved: momentum_herding_analysis.png")
plt.show()

# 7. Key insights
print(f"\n6. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 100)
print(f"""
MOMENTUM FINDINGS:
├─ Momentum premium: +{(strategy_annual_return - buyhold_annual_return)*100:.1f}% p.a. (after volatility adjustment)
├─ Volatility trade-off: Momentum vol={strategy_vol:.1%} vs Buy-Hold={buyhold_vol:.1%}
├─ Sharpe ratio: Momentum {strategy_sharpe:.3f} vs Buy-Hold {buyhold_sharpe:.3f}
└─ Drawdown: Momentum {strategy_dd:.1%} vs Buy-Hold {buyhold_dd:.1%}

HERDING & CRASH RISK:
├─ Current crowding: {current_crowding:.1%} (mean={mean_crowding:.1%})
├─ Crowding assessment: {'HIGH' if current_crowding > crowding_scores.quantile(0.75) else 'NORMAL' if current_crowding > crowding_scores.quantile(0.25) else 'LOW'}
├─ Worst crash observed: {strategy_dd_series.min():.1%} drawdown
└─ Implication: Momentum concentrated in bull markets; crashes extreme when herd reverses

TRADING RECOMMENDATIONS:
├─ Pure momentum (100% allocation): Aggressive; capture premium but face crash risk
├─ Hybrid approach (50% momentum, 50% buy-hold): Moderate; reduce volatility while capture premium
├─ Hedged momentum (momentum + tail hedge): Conservative; pay insurance but sleep well
├─ Crowding filter: Reduce momentum when crowding high (above {crowding_scores.quantile(0.75):.1%})
└─ Rebalance frequency: Quarterly works; less frequent (annual) loses faster mean reversion

RISK MANAGEMENT:
├─ Stop loss: Exit momentum if reverses >5% (cut losers early)
├─ Position size: Never >50% portfolio in momentum (tail risk concentrated)
├─ Diversification: Spread across multiple momentum factors (not just trend)
├─ Leverage: Avoid! Momentum crashes destroy leveraged positions
└─ Hedging: Consider VIX calls (pay 1-2% p.a.) to hedge tail
""")

print("=" * 100)