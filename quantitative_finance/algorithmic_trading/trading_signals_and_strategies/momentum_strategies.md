# Momentum Strategies

## 1. Concept Skeleton
**Definition:** Trading strategy exploiting the tendency of assets with strong past returns to continue outperforming; assumes markets under-react to information (price trends persist short-term, reverse long-term); captures continuation via technical analysis, factor-based filtering, or macroeconomic momentum  
**Purpose:** Generate alpha from pricing inefficiencies; capture trend-following profits; diversify away from mean-reversion strategies; exploit slow information diffusion in markets  
**Prerequisites:** Time series analysis (trend detection), factor models, technical indicators, volatility management, portfolio optimization, transaction cost modeling

## 2. Comparative Framing
| Aspect | Price Momentum | Earnings Momentum | Volatility Momentum | Currency Momentum | Commodity Momentum |
|--------|-----------------|-------------------|----------------------|-------------------|-------------------|
| **Signal Source** | Relative returns (last 3-12 mo) | Earnings growth acceleration | Implied volatility trends | FX rate trends | Price trends (crude, gold, etc.) |
| **Holding Period** | 1-12 months | 3-12 months | 1-6 months | 1-6 months | 1-12 months |
| **Mean Reversion Risk** | High after 12mo horizon | Moderate (reversals slower) | Very high (vol mean-reverts) | Low (carry dominates) | Moderate |
| **Data Frequency** | Daily/weekly | Quarterly/annual | Daily | Daily | Daily |
| **Liquidity Needs** | High (liquid equities/futures) | Moderate (stocks only) | Low (futures/options) | Very high (FX spot) | Moderate |
| **Sector Bias** | Tech (high momentum sensitivity) | Tech/Healthcare | Equities (all) | EM currencies (beta) | Energy/Agriculture |
| **Crash Risk** | Severe (forced unwinding) | Moderate | Moderate | Low | High (geopolitical) |
| **Peak Performance** | Growth markets (2013-2020) | Persistent (more stable) | Crisis periods (2011, 2020) | 2008-2012 (carry trade) | 2008 (flight to safety) |

## 3. Examples + Counterexamples

**Successful Momentum Trade (2008 Crisis):**  
Feb 2008: VIX rising (spike from 20 → 40). Momentum signal: "Volatility increasing." Buy VIX calls, sell equity index.  
Oct 2008: VIX explodes to 80+. Profit: $100M position → $500M+ gain. Momentum captured tail.

**Failed Momentum (2009 Flash Crash):**  
May 6, 2009: Market down 5% in 5 minutes. Momentum algorithm: "Sell more! Down=continue down." Forced selling → -9% mini-crash.  
Recovery: Market rebounds within minutes. Algorithm caught in momentum death spiral, liquidation losses.

**Earnings Momentum Success (Tech Boom):**  
2010-2020: Apple earnings beat 4 consecutive quarters (AAPL momentum = +40% YTD). Buy signal each quarter. Earnings momentum worked: Average +8% in next quarter post-beat.  
**Counter:** 2022: Microsoft beat, but guidance weak → -10% next day despite positive momentum. Earnings revisions dominate short-term.

**Currency Momentum Trade:**  
2015-2018: JPY weakens (carry trade, Abenomics stimulus). Momentum: "JPY weak → stay weak." Carry trades leverage to 20x.  
Aug 2016: BOJ surprise tightening → JPY rallies +7% intraday. Leveraged carries liquidate → funding crisis, carry funding evaporates. Stop-loss at +8% → realized.

**Volatility Momentum Paradox:**  
VIX = 15 (low vol), momentum says "stay low." But low volatility regimes are dangerous (complacency). 2017: VIX near 10 for 9 months (all-time low volatility momentum). Feb 2018: VIX spikes to 50 (inverse momentum).

**Commodity Momentum vs Fundamentals:**  
Oil 2014-2015: Price momentum down ($100 → $40, -60% in 1 year). Momentum signal = "keep selling." Production cuts (supply response) take 2 years. Momentum shorted bottom ($37). Rebounded to $60+ (2015 low). Momentum whipsaw.

## 4. Layer Breakdown
```
Momentum Strategy Architecture:

├─ Price Momentum Framework
│  ├─ Signal Generation:
│  │   ├─ Calculation:
│  │   │   ├─ Simple returns: r(t) = ln(P(t) / P(t-τ))
│  │   │   ├─ Lookback window τ (typical 3-12 months, e.g., 252 days = 1 year)
│  │   │   ├─ Momentum score = r(t) - r(benchmark)
│  │   │   ├─ Normalization: Z-score = (r(t) - mean(r)) / std(r)
│  │   │   └─ Signal: Long if Z > 0.5 (top 30%), Short if Z < -0.5 (bottom 30%)
│  │   ├─ Variations:
│  │   │   ├─ 1-month lookback (noise-heavy, captures short-term trends)
│  │   │   ├─ 6-month lookback (balanced, typical)
│  │   │   ├─ 12-month lookback (longer trends, more stable)
│  │   │   ├─ Multi-scale: Blend 3, 6, 12-month for robustness
│  │   │   └─ Relative momentum: Rank vs sector peers, industry, universe
│  │   └─ Interaction with fundamentals:
│  │       ├─ Exclude low-volume, illiquid names (avoid liquidity crashes)
│  │       ├─ Filter by valuation (momentum vs cheap, vs expensive)
│  │       ├─ Adjust for earnings surprises (earnings momentum > price momentum)
│  │       └─ Sector neutralization (avoid sector bet vs stock-specific momentum)
│  │
│  ├─ Portfolio Construction:
│  │   ├─ Long-only:
│  │   │   ├─ Select top N momentum deciles (e.g., top 20% of S&P 500)
│  │   │   ├─ Equal-weight or momentum-weighted
│  │   │   ├─ Concentration risk: Large bets on mega-cap tech (high momentum)
│  │   │   └─ Typical volatility: 12-18% annual
│  │   ├─ Long-short (market-neutral):
│  │   │   ├─ Long top momentum decile, short bottom decile
│  │   │   ├─ Hedge out market beta (momentum doesn't require upmarket)
│  │   │   ├─ Leverage = 2x (1x long, 1x short)
│  │   │   └─ Typical volatility: 6-10% annual (less correlated to market)
│  │   ├─ Risk parity:
│  │   │   ├─ Equalize risk contribution: Bonds momentum + stock momentum + FX momentum
│  │   │   ├─ Less capacity (broad assets), more uncorrelated returns
│  │   │   ├─ Volatility: 5-8% (lower, more stable)
│  │   │   └─ Diversification across asset classes
│  │   └─ Smart beta packaging:
│  │       ├─ Publish factor exposures (smart beta index)
│  │       ├─ Quarterly rebalancing (transaction costs matter)
│  │       ├─ Easy implementation, scales to billions
│  │       └─ Drawback: Predictable rebalancing exploited by high-frequency traders
│  │
│  ├─ Rebalancing & Holding Periods:
│  │   ├─ Daily rebalancing: Active, high transaction costs, drift quickly
│  │   ├─ Weekly rebalancing: Moderate costs, capture short-term continuation
│  │   ├─ Monthly rebalancing: Standard for quant funds, cost-effective
│  │   ├─ Quarterly rebalancing: Low costs, misses within-month momentum
│  │   ├─ Dynamic rebalancing: Rebalance when momentum score changes by threshold
│  │   └─ Volatility regime adjustment: More frequent in low-vol, less in high-vol
│  │
│  ├─ Momentum Decay & Reversal Risk:
│  │   ├─ Time horizon effect:
│  │   │   ├─ 1-month: Strong continuation (+2-3% annual alpha)
│  │   │   ├─ 3-month: Moderate (+1.5-2% alpha)
│  │   │   ├─ 12-month: Weaker, trend exhaustion begins
│  │   │   ├─ 24-36 months: Full reversal (mean-reversion kicks in)
│  │   │   └─ Implication: Exit after 3-12 months, don't hold forever
│  │   ├─ Crowding effect:
│  │   │   ├─ Too many traders on same momentum trade → saturation
│  │   │   ├─ Price inflates faster than justified by fundamentals
│  │   │   ├─ Reversal more severe (forced exits, margin calls)
│  │   │   └─ Monitor: When consensus becomes "everyone knows" → exit
│  │   ├─ Valuation reversion:
│  │   │   ├─ High-momentum stocks often expensive (P/E > 40)
│  │   │   ├─ Reversion triggers: Recession, earnings miss, sector rotation
│  │   │   ├─ Timing: Usually after 12-24 months (not predictable)
│  │   │   └─ Portfolio design: Include valuation filter to reduce tail risk
│  │   └─ Strategies to manage decay:
│  │       ├─ Use trend-following stops (exit when momentum flips)
│  │       ├─ Blend with mean-reversion (reduce crowding sensitivity)
│  │       ├─ Factor cycle management (momentum underperforms late-cycle)
│  │       └─ Cross-asset diversification (different momentum cycles)
│  │
│  └─ Transaction Costs & Slippage:
│      ├─ Turnover: Monthly rebalancing top/bottom 20% = ~50% annual turnover
│      ├─ Bid-ask: Large-cap ($0.01 spread) vs mid-cap ($0.10 spread)
│      ├─ Market impact: Buying $10M + momentum names pushes price up ~0.5%
│      ├─ Optimization: Use execution algorithms (VWAP/TWAP) for patience
│      ├─ Cost reduction: Cap turnover at 100% annual (sacrifice some signal purity)
│      └─ Washout: If costs > alpha, strategy fails
│
├─ Earnings Momentum Framework
│  ├─ Signal Generation:
│  │   ├─ Earnings surprise: Actual EPS - Consensus Estimate
│  │   │   ├─ Positive surprise (beat): +5% signal
│  │   │   ├─ Negative surprise (miss): -5% signal
│  │   │   ├─ Magnitude matters: Beat by $0.10 on $1.00 = 10% upside
│  │   │   └─ Timing: Report date impact (event-driven, not trending)
│  │   ├─ Earnings revision momentum:
│  │   │   ├─ Direction: Analysts raising (bullish) vs lowering (bearish) estimates
│  │   │   ├─ Magnitude: +10% analyst EPS revision → 2-3% stock outperformance
│  │   │   ├─ Breadth: % of analysts revising up vs down (consensus strength)
│  │   │   └─ Forward: Revisions for next quarter/year predict near-term returns
│  │   ├─ Guidance:
│  │   │   ├─ Forward guidance from management ("revenue growth 15-20% next year")
│  │   │   ├─ Positive guidance (raised) → +3-5% stock bounce
│  │   │   ├─ Negative guidance (lowered) → -5-10% stock drop
│  │   │   └─ Often matters MORE than earnings beat itself
│  │   └─ Quality of earnings:
│  │       ├─ Operating earnings (underlying business health)
│  │       ├─ GAAP earnings (accounting-compliant, includes one-offs)
│  │       ├─ Quality score: % of earnings from operations (higher = better)
│  │       └─ Deteriorating quality (one-offs rising) predicts reversal
│  │
│  ├─ Portfolio Application:
│  │   ├─ Post-earnings drift (PED): Stocks continue trending 20 days post-announcement
│  │   │   ├─ Reason: Market under-reacts to earnings initially
│  │   │   ├─ Strategy: Buy 1-2 days post beat (after initial reaction)
│  │   ├─ Pre-earnings announcement drift (PEAD): Anticipatory movement before earnings
│  │   │   ├─ Market prices in expected beat/miss before announcement
│  │   │   ├─ Contrarian: Short expected misses (they're priced in)
│  │   │   ├─ Support: Look for positive revisions despite low expectations
│  │   │   └─ Concept: "Capitulation signal" when estimate suddenly raised before earnings
│  │   ├─ Earnings season portfolio construction:
│  │   │   ├─ Long: Stocks with rising estimates + positive momentum
│  │   │   ├─ Short: Falling estimates + negative momentum
│  │   │   ├─ Timing: Begin 1-2 weeks before earnings date (capture anticipation)
│  │   │   └─ Exit: 3-4 weeks post earnings (capture drift, exit before next cycle)
│  │   └─ Interaction with price momentum:
│  │       ├─ Combined signal stronger than either alone
│  │       ├─ Earnings momentum + price momentum = +2-3% alpha vs +0.8% each alone
│  │       ├─ Diversification: Different news drives different signals
│  │       └─ Avoid: Short earnings misses with strong price momentum (crowded bet)
│  │
│  └─ Earnings Cycle Dynamics:
│      ├─ Q1 (Jan-Mar): Pre-earnings anticipation builds, January effect inflation
│      ├─ Earnings week (3rd-4th week month): Concentrated earnings dates (busy)
│      ├─ Post-earnings (4-5 weeks): Drift continues, revisions adjust
│      ├─ Pre-next-cycle (new guidance): Forward expectations reset
│      ├─ Calendar effects: Tech earnings (late April), financials (mid-Oct)
│      └─ Capacity constraint: As more capital targets earnings anomaly, excess returns compress
│
├─ Volatility Momentum Framework
│  ├─ Vol-of-Vol Concept:
│  │   ├─ Implied vol clustering: When vol is volatile, it tends to stay volatile
│  │   ├─ Indicator: VIX moving average (20-day) vs long-term trend
│  │   ├─ Signal: If vol > 20-day MA, stays high (regime change)
│  │   ├─ Decay pattern: Vol mean-reverts (high vol → drops, low vol → rises)
│  │   └─ Horizon: Vol momentum fades faster than price momentum (2-4 weeks)
│  │
│  ├─ Strategies:
│  │   ├─ Vol term structure trading (curve slope momentum):
│  │   │   ├─ Upward sloping (1M vol < 3M vol): Contango, expect vol rise
│  │   │   ├─ Downward sloping (1M vol > 3M vol): Backwardation, expect vol drop
│  │   │   ├─ Trade: Long front-month (if backwardation expected to reverse)
│  │   │   └─ Gamma decay: Lose money via theta if vol drops (option seller headwind)
│  │   ├─ Vol-of-vol strategies (vol selling when vol is stable):
│  │   │   ├─ VIX < 15 for extended period → "sell vol" (strangle/straddle)
│  │   │   ├─ Vol momentum = "vol stays low" → premium collection works
│  │   │   ├─ Risk: Sudden jump (crisis) → massive loss
│  │   │   └─ Protection: Buy tail hedges (out-of-the-money puts)
│  │   ├─ Momentum vol trading (pairs):
│  │   │   ├─ When momentum stocks high, pair-trade underperformers
│  │   │   ├─ Short correlation between high-momentum vs value = vol play
│  │   │   └─ Vol momentum says correlation stays low (trade persistence)
│  │   └─ Cross-asset vol momentum:
│  │       ├─ Equity vol vs bond vol momentum
│  │       ├─ Rising equity vol = risk-off, typically raises bond vol
│  │       ├─ Divergence = mispricing opportunity
│  │       └─ Vol momentum capturing convergence
│  │
│  ├─ Risks (Vol Momentum is Highest Risk of All Momentum Types):
│  │   ├─ Sudden regime breaks:
│  │   │   ├─ 2017: VIX = 10 all year (complacency)
│  │   │   ├─ Feb 2018: VIX spikes to 50 (14-Sigma move, -20% S&P 1 day)
│  │   │   ├─ XIV (inverse vol ETN) collapses to $0.01 (holders wiped out)
│  │   │   └─ Lesson: Vol momentum can crash faster than realized
│  │   ├─ Volatility cascade:
│  │   │   ├─ High vol → options dealers hedge (sell S&P) → vol rises more (feedback)
│  │   │   ├─ Cascade amplifies moves: 10% initial drop → 20% total drop
│  │   │   ├─ Vol momentum traders caught (momentum breaks down in tail)
│  │   │   └─ Stop-losses trigger en masse → flash crashes
│  │   ├─ Skew risk:
│  │   │   ├─ Skew = vol asymmetry (put vol > call vol in downturns)
│  │   │   ├─ Vol momentum + rising skew = worst case (short vol gets skewed)
│  │   │   └─ Protection: Monitor skew, reduce exposure when skew extreme
│  │   └─ Funding stress:
│  │       ├─ Vol momentum trades use leverage (margins)
│  │       ├─ In crisis, funding dries up (margin increases)
│  │       ├─ Forced liquidations regardless of model prediction
│  │       └─ Systemic risk: Damodaran's "black swan" (Taleb criticism)
│  │
│  └─ Practical Horizon:
│      ├─ Very short-term (days): Vol momentum strong but needs quick execution
│      ├─ Medium-term (weeks): Vol momentum useful, manageable drawdowns
│      ├─ Long-term (months): Vol mean-reverts, momentum decays quickly
│      └─ Recommendation: 2-4 week holding periods for vol momentum
│
├─ Cross-Asset Momentum
│  ├─ Multi-asset momentum portfolio:
│  │   ├─ Assets: US equities, international equities, bonds, commodities, FX
│  │   ├─ Signal: Momentum in each asset independently
│  │   ├─ Portfolio: Equal-risk allocation (1/N risk weight)
│  │   ├─ Rebalancing: Monthly, turnover ~30% (moderate costs)
│  │   ├─ Volatility: 5-8% (diversified, uncorrelated shocks)
│  │   └─ Returns: 3-6% alpha in normal years (vs 0-1% single-asset momentum)
│  │
│  ├─ Momentum factor cycle:
│  │   ├─ Early cycle (post-recession): Value + momentum (growth-hungry)
│  │   ├─ Mid cycle (expansion): Growth + momentum (trends strong)
│  │   ├─ Late cycle (high rates): Momentum fades (reversals begin)
│  │   ├─ Recession: Flight to safety (bonds momentum up, equities down)
│  │   └─ Adapting: Tilt portfolio by cycle (reduce momentum in late cycle)
│  │
│  └─ Implementation:
│      ├─ Hedge fund strategies: "Trend-following" funds run momentum multi-asset
│      ├─ CTAs (Commodity Trading Advisors): Originally momentum, now add other signals
│      ├─ Risk parity funds: Use momentum as component of allocation engine
│      └─ Smart beta: Momentum factor increasingly packaged as ETF
│
└─ Risk & Drawdown Management
   ├─ Momentum Crashes (Common Patterns):
   │   ├─ 2000 (Dot-com): Momentum tech crash (NASDAQ -78% from peak)
   │   ├─ 2008 (Financial Crisis): Momentum reversal, -60% drawdown in 3 months
   │   ├─ 2011 (Sovereign debt): Risk-off reversal, momentum unwind painful
   │   ├─ 2018 (Factor blow-up): Mean-reversion beats momentum for only time in decade
   │   ├─ 2020 (COVID): March drawdown -25%, then recovery +100% (whipsaw)
   │   └─ Lessons: Momentum crashes hard when it does; sizing essential
   │
   ├─ Risk Controls:
   │   ├─ Position sizing: Not leveraged (1x long-short max) prevents cascade
   │   ├─ Stop-losses: Exit if momentum score reverses by 2σ (avoid large reversals)
   │   ├─ Volatility scaling: Reduce size when vol spikes (inverse relationship)
   │   ├─ Sector limits: No more than 30% in one sector (avoid single-name risk)
   │   ├─ Factor diversification: Blend momentum with value, quality (reduce crowding)
   │   └─ Tail hedges: Buy OTM puts quarterly (insurance against crash)
   │
   ├─ Crowding Detection:
   │   ├─ Monitor: % of momentum funds short same names (consensus risk)
   │   ├─ Signal: When >90% funds short bottom decile, reversal likely
   │   ├─ Action: Reduce short exposure, increase longs (contrarian tilt)
   │   ├─ Indicator: Hedge fund positioning data (13F filings, monthly)
   │   └─ Advantage: Early detection of momentum consensus extremes
   │
   └─ Empirical Performance:
       ├─ Historical returns (1970-2023): +4-8% annual excess over benchmark
       ├─ Sharpe ratio: 0.6-1.0 (vs 0.3-0.5 for long-only)
       ├─ Max drawdown: -25 to -50% (crashes when triggers)
       ├─ Autocorrelation to market: -0.2 to +0.1 (mostly uncorrelated)
       ├─ Trend: Performance declining (crowding, lower alpha) since 2010s
       ├─ Factor performance cycles: Momentum strong 2003-2008, 2016-2020, weak 2010-2015, 2021-2023
       └─ Forward outlook: Still viable but lower returns, higher crowding risk expected
```

**Interaction:** Momentum signals (price/earnings/vol/cross-asset) → Portfolio construction (long-short weighting) → Rebalancing (monthly) → Risk controls (stops, vol scaling) → Drawdown management → Performance measurement.

## 5. Mini-Project
Backtest momentum strategy with multi-scale signals:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Simulated price data (would use real OHLC in practice)
np.random.seed(42)
n_days = 2000
dates = pd.date_range('2015-01-01', periods=n_days, freq='D')

# 5 stocks with different momentum profiles
price_data = {}
for stock in ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']:
    # Generate price with trending behavior
    drift = np.random.uniform(0.0002, 0.0008)  # Positive drift (trending)
    volatility = np.random.uniform(0.015, 0.025)
    returns = np.random.normal(drift, volatility, n_days)
    price = 100 * np.exp(np.cumsum(returns))
    price_data[stock] = pd.Series(price, index=dates)

prices_df = pd.DataFrame(price_data)

print("="*100)
print("MOMENTUM STRATEGY BACKTEST: Multi-Scale Signals")
print("="*100)
print(f"\nPrice Data (first 10 days):")
print(prices_df.head(10))

# Step 1: Calculate momentum signals (multiple lookback windows)
print(f"\n" + "="*100)
print("STEP 1: Momentum Signal Calculation")
print("="*100)

lookback_windows = {1: 20, 3: 60, 6: 126, 12: 252}  # months: days
momentum_signals = {}

for window_name, window_days in lookback_windows.items():
    momentum = prices_df.pct_change(window_days)
    momentum_signals[f'{window_name}M'] = momentum
    print(f"\n{window_name}-Month Momentum (lookback {window_days} days):")
    print(f"  Mean: {momentum.mean().mean():.4f}, Std: {momentum.std().mean():.4f}")

# Combined momentum signal (average of normalized scores)
def normalize_signal(signal):
    """Z-score normalization per stock."""
    return signal.rolling(252).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)

combined_signal = pd.DataFrame(index=dates)
for window_name in momentum_signals.keys():
    combined_signal[f'signal_{window_name}'] = normalize_signal(momentum_signals[window_name])

# Final combined signal (average across windows)
combined_signal['combined'] = combined_signal.mean(axis=1)

print(f"\nCombined Signal (average of normalized scores):")
print(f"  Min: {combined_signal['combined'].min():.2f}")
print(f"  Max: {combined_signal['combined'].max():.2f}")
print(f"  Mean: {combined_signal['combined'].mean():.4f}")

# Step 2: Generate trading signals
print(f"\n" + "="*100)
print("STEP 2: Trading Signal Generation")
print("="*100)

threshold = 0.5  # Z-score threshold for long/short
position = combined_signal['combined'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))

print(f"\nSignal Distribution:")
print(f"  Long (position = 1): {(position == 1).sum()} days ({(position == 1).sum() / len(position) * 100:.1f}%)")
print(f"  Neutral (position = 0): {(position == 0).sum()} days ({(position == 0).sum() / len(position) * 100:.1f}%)")
print(f"  Short (position = -1): {(position == -1).sum()} days ({(position == -1).sum() / len(position) * 100:.1f}%)")

# Step 3: Portfolio construction (long-short, equal-weighted)
print(f"\n" + "="*100)
print("STEP 3: Portfolio Construction (Long-Short)")
print("="*100)

# For simplicity: assume equal allocation to top/bottom momentum stocks each day
portfolio_value = []
daily_returns = []

for day in range(1, len(prices_df)):
    # Get current momentum signals
    current_momentum = combined_signal['combined'].iloc[day]
    
    if pd.isna(current_momentum):
        portfolio_value.append(100 if day == 1 else portfolio_value[-1])
        daily_returns.append(0)
        continue
    
    # Get price returns for all stocks
    price_returns = prices_df.pct_change().iloc[day]
    
    # Long top momentum stock, short bottom (simplified: top 1, bottom 1)
    stocks_ranked = current_momentum.nlargest(2).index.tolist()
    
    if len(stocks_ranked) >= 2:
        long_stock = stocks_ranked[0]
        short_stock = stocks_ranked[-1]
        
        # Portfolio: +1 long, -1 short (market-neutral)
        portfolio_return = 0.5 * price_returns[long_stock] - 0.5 * price_returns[short_stock]
        daily_returns.append(portfolio_return)
    else:
        daily_returns.append(0)
    
    # Accumulate portfolio value
    if day == 1:
        portfolio_value.append(100 * (1 + daily_returns[-1]))
    else:
        portfolio_value.append(portfolio_value[-1] * (1 + daily_returns[-1]))

portfolio_series = pd.Series(portfolio_value, index=dates[1:])
daily_returns_series = pd.Series(daily_returns, index=dates[1:])

# Step 4: Performance metrics
print(f"\nStrategy Performance:")
print(f"  Starting Value: $100")
print(f"  Ending Value: ${portfolio_value[-1]:.2f}")
print(f"  Total Return: {(portfolio_value[-1] / 100 - 1) * 100:.2f}%")
print(f"  Annualized Return: {(np.mean(daily_returns) * 252) * 100:.2f}%")
print(f"  Annualized Volatility: {np.std(daily_returns) * np.sqrt(252) * 100:.2f}%")
print(f"  Sharpe Ratio: {(np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)):.2f}")

# Maximum drawdown
cumulative = np.cumprod(1 + daily_returns)
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max
max_drawdown = np.min(drawdown)
print(f"  Maximum Drawdown: {max_drawdown * 100:.2f}%")

# Win rate
win_rate = (np.array(daily_returns) > 0).sum() / len(daily_returns)
print(f"  Win Rate: {win_rate * 100:.1f}%")

# Comparison to buy-and-hold (equal-weight all stocks)
buy_hold_returns = prices_df.pct_change().mean(axis=1)
buy_hold_value = 100 * np.cumprod(1 + buy_hold_returns)

print(f"\nBuy-and-Hold (Equal-Weight Comparison):")
print(f"  Ending Value: ${buy_hold_value.iloc[-1]:.2f}")
print(f"  Total Return: {(buy_hold_value.iloc[-1] / 100 - 1) * 100:.2f}%")
print(f"  Annualized Return: {(np.mean(buy_hold_returns) * 252) * 100:.2f}%")

# Step 5: Momentum decay analysis
print(f"\n" + "="*100)
print("STEP 5: Momentum Decay Analysis")
print("="*100)

# Measure returns in periods following signal generation
decay_analysis = []
for signal_window in ['1M', '3M', '6M', '12M']:
    signal_col = f'signal_{signal_window}'
    
    # Calculate returns in next N periods after high/low momentum signal
    holding_periods = []
    for day in range(100, len(prices_df) - 20):
        if pd.notna(combined_signal[signal_col].iloc[day]):
            signal_val = combined_signal[signal_col].iloc[day]
            if signal_val > 1.0:  # High momentum
                # Calculate next 20-day return
                return_20d = (prices_df.iloc[day + 20] / prices_df.iloc[day] - 1).mean()
                holding_periods.append({'signal': 'strong_long', 'return': return_20d})
            elif signal_val < -1.0:  # Low momentum
                return_20d = (prices_df.iloc[day + 20] / prices_df.iloc[day] - 1).mean()
                holding_periods.append({'signal': 'strong_short', 'return': return_20d})
    
    if holding_periods:
        df_hold = pd.DataFrame(holding_periods)
        long_return = df_hold[df_hold['signal'] == 'strong_long']['return'].mean()
        short_return = df_hold[df_hold['signal'] == 'strong_short']['return'].mean()
        decay_analysis.append({'window': signal_window, 'long_20d_ret': long_return, 'short_20d_ret': short_return})

decay_df = pd.DataFrame(decay_analysis)
print(f"\nMomentum Decay (20-day forward returns after signal):")
print(decay_df.to_string(index=False))

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Strategy vs Buy-and-Hold
ax = axes[0, 0]
ax.plot(dates[1:], portfolio_series, label='Momentum Strategy', linewidth=2)
ax.plot(dates[1:], buy_hold_value.iloc[1:], label='Buy-and-Hold (EW)', linewidth=2, alpha=0.7)
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('Momentum Strategy vs Buy-and-Hold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale('log')

# Plot 2: Daily Returns Distribution
ax = axes[0, 1]
ax.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(daily_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(daily_returns)*100:.3f}%')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Frequency')
ax.set_title('Daily Returns Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Cumulative Drawdown
ax = axes[1, 0]
cumulative_ret = np.cumprod(1 + daily_returns) - 1
running_max_ret = np.maximum.accumulate(cumulative_ret)
drawdown_series = (cumulative_ret - running_max_ret) / (1 + running_max_ret)
ax.fill_between(dates[1:], 0, drawdown_series * 100, alpha=0.5, color='red')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown Over Time')
ax.grid(alpha=0.3)

# Plot 4: Momentum Signal Evolution
ax = axes[1, 1]
ax.plot(dates, combined_signal['combined'], label='Combined Signal', linewidth=1)
ax.axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label=f'Long Threshold ({threshold})')
ax.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5, label=f'Short Threshold (-{threshold})')
ax.fill_between(dates, threshold, 5, alpha=0.2, color='green')
ax.fill_between(dates, -5, -threshold, alpha=0.2, color='red')
ax.set_ylabel('Signal (Z-score)')
ax.set_title('Momentum Signal Over Time (Combined Multi-Scale)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Statistics table
print(f"\n" + "="*100)
print("PERFORMANCE SUMMARY TABLE")
print("="*100)

summary_df = pd.DataFrame({
    'Metric': ['Total Return', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
    'Momentum': [
        f"{(portfolio_value[-1]/100 - 1)*100:.2f}%",
        f"{np.mean(daily_returns)*252*100:.2f}%",
        f"{np.std(daily_returns)*np.sqrt(252)*100:.2f}%",
        f"{(np.mean(daily_returns)*252)/(np.std(daily_returns)*np.sqrt(252)):.2f}",
        f"{max_drawdown*100:.2f}%",
        f"{win_rate*100:.1f}%"
    ],
    'Buy-and-Hold': [
        f"{(buy_hold_value.iloc[-1]/100 - 1)*100:.2f}%",
        f"{np.mean(buy_hold_returns)*252*100:.2f}%",
        f"{np.std(buy_hold_returns)*np.sqrt(252)*100:.2f}%",
        f"{(np.mean(buy_hold_returns)*252)/(np.std(buy_hold_returns)*np.sqrt(252)):.2f}",
        "N/A",
        "N/A"
    ]
})

print(summary_df.to_string(index=False))
```

## 6. Challenge Round
- Design 6-month momentum filter for large-cap tech sector
- Calculate post-earnings drift (PED) alpha for 50-stock portfolio
- Backtest currency momentum strategy (6 FX pairs, 3-month lookback)
- Compare momentum vs value factor returns in different market cycles
- Model crowding risk: When do momentum reversals occur?

## 7. Key References
- [Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers" (1993), JF](https://www.jstor.org/) — Seminal momentum paper
- [Moskowitz, Ooi & Pedersen, "Time Series Momentum" (2012), JFE](https://papers.ssrn.com/) — Cross-asset momentum framework
- [Blitz, Hanauer, Vidojevic & Vliet, "Momentum Everywhere" (2020), Research Affiliates](https://www.researchaffiliates.com/) — Multi-asset decomposition
- [AQR, "Momentum Crashes" (2019), Asness et al.](https://www.aqr.com/) — Momentum reversal mechanics

---
**Status:** Established factor strategy (1990s-present, continuously refined) | **Complements:** Mean Reversion, Factor Models, Technical Analysis, Portfolio Construction
