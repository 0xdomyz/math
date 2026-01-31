# Statistical Arbitrage

## 1. Concept Skeleton
**Definition:** Market-neutral trading strategy exploiting pricing inefficiencies through statistical relationships (cointegration, correlation breakdowns) between securities; buys undervalued, shorts overvalued to capture convergence  
**Purpose:** Generate alpha uncorrelated to market moves (beta-neutral), profit from mean-reversion in relative prices, reduce systematic risk via hedging  
**Prerequisites:** Pairs trading, cointegration analysis, correlation estimation, hedging mechanics, market-neutral portfolio construction

## 2. Comparative Framing
| Strategy | Approach | Correlation Reliance | Market Neutral | Holding Period | Typical Return |
|----------|----------|----------------------|-----------------|-----------------|-----------------|
| **Pairs Trading** | Long X, Short Y (cointegrated) | Cointegration key | Yes (beta-neutral) | Days-weeks | +2-5% annual |
| **Cross-Asset Arb** | Equity vs derivatives (futures, options) | Index correlation | Yes | Minutes-hours | +1-3% annual |
| **ETF Arbitrage** | ETF vs basket of holdings | Component correlation | Yes | Minutes | +0.5-1.5% annual |
| **Index Rebalancing** | Front-run index rebalancing trades | Factor correlation | Partial | Hours-days | +0.5-2% annual |
| **Stat Arb Portfolio** | 50-100 pairs simultaneously | Multiple relationships | Yes (systematic) | Varies | +3-8% annual |
| **Momentum-Reversal Arb** | Exploit mean-reversion across time-scales | Factor correlation | Neutral | Minutes-hours | +1-4% annual |
| **Sector-Rotation Arb** | Long strong sector, short weak | Sector correlation | Partial | Days-weeks | +2-5% annual |

## 3. Examples + Counterexamples

**Stat Arb Success (GLD vs GDX, 2020):**  
GLD (gold ETF) and GDX (gold miners): Historically 0.85 correlation. March 2020 crisis: GDX crashes -50%, GLD -20% (correlation breaks). Signal: Long GDX, short GLD. 3 months later: GDX recovers +80%, GLD +5%. Spread converges. Return: +40% (mark-to-market).

**Stat Arb Failure (Lehman Crisis, 2008):**  
JPM (mega-cap bank) vs Lehman (mid-cap bank): Correlation 0.95 for 10 years. Pairs trader: "Spread too wide, revert." September 2008: Lehman collapses ($0), JPM survives ($30). Correlation breaks, doesn't revert. Loss: Total capital (unleveraged).

**ETF Arb Opportunity (SPY vs Constituents):**  
SPY (S&P 500 ETF) trading at $400, but component stocks indicate fair value $401.  
Arb: Long components (buy the 500 stocks), short SPY (sell ETF).  
Profit: Capture $1 spread minus execution costs (~10-20 bps).  
Execution: Minutes via program trading.

**Cross-Asset Arb (Futures-Spot):**  
ES (E-mini S&P 500 futures) at 4600, SPY at $460 (underlying spot).  
Fair value: ES should = SPY × 100 (no arbitrage).  
Mispricing: ES at 4610 (overpriced futures).  
Arb: Buy SPY, sell ES. Profit: $10 convergence (100x multiplier) minus costs.  
Execution: Milliseconds for HFT; seconds for institutional.

**Momentum Crash (Stat Arb Deleveraging, 2007):**  
Stat arb funds highly leveraged (5-10x) on 100+ pairs.  
Aug 2007: Credit crisis triggers margin calls.  
All stat arb funds deleveraging simultaneously (forced selling).  
Correlations spike to 1.0 (everything correlated in crash).  
Stat arb losses: -20% in weeks (negative alpha in crisis).

**Index Rebalancing Arb (Annual Rebalance):**  
Russell 1000/2000 reconstitution: 100s of stocks reassigned annually (June).  
Before: Small-cap inclusion in R1000 → price run-up (anticipated buying).  
After inclusion: Index funds MUST buy automatically.  
Arb: Buy stocks 2-3 weeks before inclusion (ride run-up), sell at inclusion.  
Return: +2-5% in 2-3 weeks.

## 4. Layer Breakdown
```
Statistical Arbitrage Framework:

├─ Core Strategy: Pairs Trading (Foundation)
│  ├─ Selection:
│  │  ├─ Cointegration test (Johansen test)
│  │  │  ├─ Test if spread is stationary (I(0))
│  │  │  ├─ Null: Each series non-stationary (I(1))
│  │  │  ├─ Alternative: At least one cointegrating relationship
│  │  │  ├─ Regression: Y = α + β×X + ε (where ε is I(0) spread)
│  │  │  ├─ P-value < 0.05: Cointegrated (good for trading)
│  │  │  └─ P-value > 0.05: Not cointegrated (avoid)
│  │  │
│  │  ├─ Correlation Analysis:
│  │  │  ├─ Pearson correlation ≥ 0.7: Candidate pair
│  │  │  ├─ Rolling correlation: Check stability
│  │  │  ├─ Concordance: % of time moving together
│  │  │  └─ Breakdowns: Identify vulnerable pairs
│  │  │
│  │  └─ Screening:
│  │     ├─ Sector pairs: AAPL-MSFT (tech), XOM-CVX (energy)
│  │     ├─ Geographic: US stock - Canadian equivalent
│  │     ├─ Alternative: Stock - futures, stock - option
│  │     └─ Size: Avoid illiquid pairs (widen spreads, impact costs)
│  │
│  ├─ Entry Signal:
│  │  ├─ Zscore(Spread) < -2 (oversold pair-wise):
│  │  │  ├─ Y undervalued, X overvalued
│  │  │  ├─ Long Y, short X (buy low, sell high)
│  │  │  ├─ Example: Bank of America down, JPMorgan flat
│  │  │  ├─ Signal: BAC oversold relative to JPM
│  │  │  └─ Action: Buy BAC, short JPM
│  │  │
│  │  └─ Zscore(Spread) > +2 (oversold opposite direction):
│  │     ├─ Y overvalued, X undervalued
│  │     ├─ Short Y, long X
│  │     └─ Flips signal
│  │
│  ├─ Portfolio Construction:
│  │  ├─ Single pair (simple):
│  │  │  ├─ 1 long, 1 short position
│  │  │  ├─ Beta-neutral (market move cancels)
│  │  │  ├─ Dollar-neutral (notional values equal)
│  │  │  ├─ Advantage: Transparent, low correlation to market
│  │  │  ├─ Disadvantage: High idiosyncratic risk
│  │  │  └─ Usage: Small funds, academic backtests
│  │  │
│  │  ├─ Portfolio of pairs (realistic):
│  │  │  ├─ 10-50 uncorrelated pairs
│  │  │  ├─ Diversification: 1 pair breaks, others compensate
│  │  │  ├─ Correlation matrix: Monitor for breaks
│  │  │  ├─ Equal risk contribution: Size positions inversely to volatility
│  │  │  ├─ Dollar neutral: Σ(Long notional) = Σ(Short notional)
│  │  │  ├─ Beta neutral: Portfolio beta ≈ 0
│  │  │  └─ Typical sharpe: 0.8-1.2 (good risk-adjusted)
│  │  │
│  │  └─ Risk Controls:
│  │     ├─ Position limits: Max 5% in any stock
│  │     ├─ Sector limits: Max 20% in sector
│  │     ├─ Leverage limits: Net 1-2x (highly leveraged funds use more)
│  │     ├─ Correlation monitoring: Alert if breaks >0.1 from estimate
│  │     └─ Daily rebalancing: Keep dollar/beta neutral
│  │
│  ├─ Exit Signal:
│  │  ├─ Spread converges (Zscore → 0):
│  │  │  ├─ Close position (take profit)
│  │  │  ├─ Typical P&L: +0.5-2% per trade
│  │  │  ├─ Trade duration: 5-30 days
│  │  │  └─ Frequency: 50-100+ trades/month in portfolio mode
│  │  │
│  │  ├─ Cointegration breaks (correlation decomposes):
│  │  │  ├─ Stop-loss: Exit if exceeded -5 to -10%
│  │  │  ├─ Time stop: Exit if no convergence after 30 days
│  │  │  ├─ Risk: Spread widened instead of narrowed (wrong direction)
│  │  │  └─ Mitigation: Monitor cointegration dynamically
│  │  │
│  │  └─ Fundamental change:
│  │     ├─ Merger: One company acquired, prices permanently decouple
│  │     ├─ Acquisition: Close pairs trade (deal arbitrage separate)
│  │     └─ Sector shift: Company restructures, comparability broken
│  │
│  └─ Performance Profile:
│     ├─ Historical: 2000-2010s: +3-8% annual, 5-10% vol (excellent)
│     ├─ Crowding: 2010s-2020s: +1-3% annual (AUM explosion dilutes alpha)
│     ├─ Drawdowns: Occasional -10-20% (correlation breaks during stress)
│     ├─ Best in: Range-bound markets (sideways)
│     ├─ Worst in: Trending, crash periods (correlation → 1, spreads widen)
│     └─ Key: Diversification across many pairs reduces crash risk
│
├─ Advanced Strategies:
│  ├─ Cross-Asset Arbitrage:
│  │  ├─ Stock-Futures Arb:
│  │  │  ├─ ES (E-mini S&P 500 futures) vs SPY (ETF)
│  │  │  ├─ Fair value: ES price = (SPY price × multiplier) + carry
│  │  │  ├─ Carry = (Risk-free rate - dividend yield) × time
│  │  │  ├─ Mispricing: Usually <1% (tight)
│  │  │  ├─ Arb: Buy underpriced, sell overpriced
│  │  │  ├─ Profit: Capture basis (if positive) or convergence at expiration
│  │  │  ├─ HFT dominates: Microsecond latency critical
│  │  │  └─ Typical P&L: +0.1-0.5% per trade (small but frequent)
│  │  │
│  │  ├─ ETF-Basket Arb:
│  │  │  ├─ QQQ (NASDAQ-100 ETF) vs component stocks
│  │  │  ├─ QQQ should = weighted average of 100 stocks
│  │  │  ├─ Mispricing: ±0.5% typical (small arbitrage window)
│  │  │  ├─ Mechanics: Buy stocks, short ETF (or vice versa)
│  │  │  ├─ Execution: Program trading (100 stocks simultaneously)
│  │  │  ├─ Advantage: Index funds forces daily creation/redemption flows
│  │  │  └─ Typical: Automated arb funds capture 100+ bps annually
│  │  │
│  │  └─ Option-Stock Arb:
│  │     ├─ Call-put parity: Call - Put = Stock - Strike×discount(t)
│  │     ├─ Mispricing: Call overprice relative to put
│  │     ├─ Arb: Buy stock + put, short call (synthetic short)
│  │     ├─ Profit: Riskless if no mispricing
│  │     ├─ Constraints: Borrowing costs, borrow availability, gaps
│  │     └─ Sophistication: Volatility surface analysis needed
│  │
│  ├─ Momentum-Reversal (Factor Arb):
│  │  ├─ Concept:
│  │  │  ├─ Momentum factor: Winners outperform (12-month lags)
│  │  │  ├─ But: Reversals over shorter periods (daily-weekly)
│  │  │  ├─ Strategy: Short strong winners (reversal bet), long losers
│  │  │  ├─ Or: Long momentum on 5-day, short momentum on 120-day
│  │  │  └─ Arb: Exploit multi-scale mean reversion
│  │  │
│  │  ├─ Execution:
│  │  │  ├─ Screen: Sort stocks by recent return
│  │  │  ├─ Quintiles: Top 5% (strongest), bottom 5% (weakest)
│  │  │  ├─ Trade: Long bottom 5%, short top 5%
│  │  │  ├─ Hold: 5-20 days (reversal window)
│  │  │  └─ Rebalance: Daily or weekly
│  │  │
│  │  └─ Performance:
│  │     ├─ Win rate: ~52-55%
│  │     ├─ Profit factor: 1.2-1.5
│  │     ├─ Drawdown: -15-30% in trends (loses during momentum phases)
│  │     └─ Returns: +1-3% annual (after costs)
│  │
│  └─ Index Rebalancing Arb:
│     ├─ Mechanism:
│     │  ├─ Index reconstitution: Additions/deletions announced
│     │  ├─ Example: A stock added to S&P 500
│     │  ├─ Index funds MUST buy on effective date
│     │  ├─ Before: Demand anticipation (price run-up)
│     │  ├─ After: Index buying demand fulfilled
│     │  ├─ Post: Price may retract or hold
│     │  └─ Arb: Long before announcement, sell at/after index date
│     │
│     ├─ Timing:
│     │  ├─ Russell reconstitution: June/July (annual)
│     │  ├─ S&P changes: Ongoing (weekly/monthly)
│     │  ├─ MSCI: Monthly rebalancing windows
│     │  └─ Front-running: Buy 1-4 weeks before expected date
│     │
│     ├─ Returns:
│     │  ├─ Typical run-up: +2-5% in 2-4 weeks pre-index
│     │  ├─ Post: Sometimes holds (+1-2%), sometimes retracts (-1-2%)
│     │  ├─ Net: +1-3% per trade (variable)
│     │  └─ Prediction: Stock/sector characteristics predict magnitude
│     │
│     └─ Risks:
│        ├─ Crowding: Popular trades (overcrowded → alpha erodes)
│        ├─ False positives: Expected additions don't happen
│        ├─ Volatility: Price can reverse sharply
│        └─ Timing: Early entry can hurt if announcement delayed
│
├─ Challenges & Failure Modes:
│  ├─ Correlation Breakdown (Crisis):
│  │  ├─ Pairs historically correlated 0.90
│  │  ├─ Crisis event: Correlation crashes to 0.60 or lower
│  │  ├─ Spread widens: Instead of converging, diverges
│  │  ├─ Loss: Position moves -20-50% against you
│  │  ├─ Leverage amplifies: -20% becomes -50-100% on leveraged portfolio
│  │  ├─ Historical: 2008 (massive), 2020 COVID (temporary), 2015 China (brief)
│  │  └─ Defense: Diversify across uncorrelated pairs, lower leverage
│  │
│  ├─ Crowding (Alpha Decay):
│  │  ├─ Strategy success attracts capital
│  │  ├─ Year 1: +5% alpha (discovery)
│  │  ├─ Year 5: +2% alpha (crowding, competition)
│  │  ├─ Year 10: +0.5% alpha (saturation, transaction costs)
│  │  ├─ Example: Pairs trading (1990s-2000s excellent, 2010s+ marginal)
│  │  ├─ Mechanism: Easy pairs already traded, leftovers hard to find
│  │  └─ Adaptation: Find new pairs, add ML models, increase frequency
│  │
│  ├─ Execution Risk:
│  │  ├─ Program trading: Must buy 100 stocks simultaneously
│  │  ├─ Slippage: Execute at different prices than ideal
│  │  ├─ Costs: Commissions, market impact, bid-ask spread
│  │  ├─ Example: 0.5% arb opportunity vs 0.4% execution costs = marginal profit
│  │  ├─ Implication: Only HFT/sophisticated algos can profit consistently
│  │  └─ Retail: Too expensive to implement profitably
│  │
│  ├─ Regulatory Risk:
│  │  ├─ Short-selling restrictions (uptick rule)
│  │  ├─ Borrow availability (hard-to-borrow stocks, fees spike)
│  │  ├─ Margin requirements: Spike during stress (forced deleveraging)
│  │  ├─ Circuit breakers: Trading halts disrupt arbitrage windows
│  │  └─ SEC regulations: Evolve, restrict strategies
│  │
│  └─ Data Issues:
│     ├─ Survivorship bias: Exclude bankrupt/delisted stocks
│     ├─ Lookahead bias: Future price in backtest signal
│     ├─ Dividend adjustments: Historical data restated
│     ├─ Splits: Price series discontinuous
│     └─ Gaps: Overnight/weekend gaps limit arbitrage opportunities
│
├─ Implementation Practicalities:
│  ├─ Funding:
│  │  ├─ Leverage: +3-10x typical (use borrowed capital)
│  │  ├─ Collateral: Positions as collateral against margin loans
│  │  ├─ Cost: Borrowing cost reduces alpha (2-3% annual fee)
│  │  ├─ Constraints: Leverage tightens during stress (margin calls)
│  │  └─ Example: +5% alpha at 5x leverage = +25% return on capital
│  │
│  ├─ Operations:
│  │  ├─ Monitoring: Track correlations, spreads in real-time
│  │  ├─ Rebalancing: Daily (keep dollar/beta neutral)
│  │  ├─ Reporting: Risk metrics, attribution, trade execution quality
│  │  ├─ Infrastructure: Robust systems, error handling, disaster recovery
│  │  └─ Staffing: Quants, traders, risk managers, operations
│  │
│  └─ Performance Fees:
│     ├─ Typical structure: 2% management fee + 20% performance fee
│     ├─ Example: +5% return, 5x leverage
│     │  ├─ Gross return: +25% (on capital)
│     │  ├─ Fees: 2% management + 5% performance (20% × 25%) = 7%
│     │  ├─ Net: +18% to investor
│     │  ├─ Returns after fees: +3.6% (18%/5x leverage)
│     │  └─ Barely beats treasury (after leverage/risk considered)
│     └─ Implication: Must find true alpha (>2-3%) to beat benchmarks after costs
│
└─ Modern Challenges (2020s):
   ├─ Algo proliferation: 1000s of arb strategies competing
   ├─ Data efficiency: ML extracts more signal, less alpha available
   ├─ Latency arms race: Microseconds matter, capital-intensive
   ├─ Volatility regimes: COVID, geopolitics, crypto volatility spike
   ├─ Correlation regime: Post-QE world: correlations higher, spreads tighter
   ├─ Deleveraging: Post-2008, leverage constrained (less capital available)
   └─ Trend: Stat arb industry consolidating (scale advantage), alpha declining
```

**Interaction:** Identify cointegrated pairs → Enter on signal → Monitor spread → Rebalance portfolio → Exit on convergence → Repeat.

## 5. Mini-Project
Implement stat arb portfolio with multiple pairs:

```python
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
```

## 6. Challenge Round
- Backtest 10-pair portfolio: Select 10 stock pairs, test cointegration, run portfolio backtest
- Correlation monitoring: Track rolling correlations, identify breakdown events, measure impact
- Regime analysis: Performance in bull/bear/crisis periods; show strategy weakness in trends
- Leverage optimization: Test 2x, 5x, 10x leverage; measure risk-adjusted returns, drawdowns
- Implementation costs: Add realistic commissions, slippage, borrow fees; measure alpha erosion

## 7. Key References
- [Gatev et al (2006), "Pairs Trading: Performance of a Relative-Value Arbitrage Rule," JFE](https://www.sciencedirect.com/science/article/pii/S0304405X06000845) — Foundational pairs trading empirics
- [Vidyamurthy (2004), "Pairs Trading: Quantitative Methods and Analysis," Wiley](https://www.wiley.com/en-us/Pairs+Trading%3A+Quantitative+Methods+and+Analysis-p-9780471460733) — Comprehensive implementation guide
- [Nolte (2016), "Forex Trading using Candlestick and Price Action Techniques," Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119054276) — Statistical arbitrage in FX markets
- [Hasbrouck (2007), "Empirical Market Microstructure," Oxford Press](https://www.jstor.org/stable/j.ctt7s0cc) — Microstructure foundation for relative value

---
**Status:** Mature strategy (1990s-2010s excellent, 2010s+ declining) | **Complements:** Mean Reversion, Correlation Analysis, Market Neutral Investing, Risk Management
