# Backtest Framework

## Concept Skeleton

Backtest framework replicates trading strategy execution on historical data to evaluate performance before live deployment, simulating signal generation, order placement, fills, and position tracking under realistic market conditions. Core challenge: avoid look-ahead bias (using future information), survivorship bias (ignoring delisted securities), and unrealistic assumptions about execution (instant fills at closing prices).

**Core Components:**
- **Data pipeline**: Historical OHLCV (open, high, low, close, volume), corporate actions (splits, dividends), bid-ask spreads
- **Signal generation**: Strategy logic applied bar-by-bar (daily, intraday) with only past data accessible
- **Order execution model**: Simulate market impact, slippage, partial fills, order types (market, limit, stop)
- **Position tracking**: Cash balance, holdings, margin requirements, realized/unrealized PnL
- **Performance analytics**: Return series, drawdowns, turnover, factor exposures

**Why it matters:** Distinguishes profitable strategies from data-mined artifacts; provides risk metrics before capital deployment; identifies hidden implementation costs.

---

## Comparative Framing

| Dimension | **Backtest Framework** | **Paper Trading (Simulated Real-Time)** | **Live Trading** |
|-----------|------------------------|------------------------------------------|------------------|
| **Data source** | Historical snapshots | Real-time feeds (delayed or live) | Real-time market data |
| **Execution model** | Simulated fills (rules-based) | Simulated orders to exchange | Actual exchange execution |
| **Look-ahead risk** | High (must enforce temporal isolation) | Low (data arrives sequentially) | None (natural constraint) |
| **Cost** | Free (historical data) | Free (no capital at risk) | Real transaction costs + slippage |
| **Speed** | Fast (years in minutes) | Real-time (trades as events occur) | Real-time |
| **Overfitting detection** | Walk-forward, out-of-sample testing | Tracks live market conditions | Final validation |

**Key insight:** Backtesting trades speed and flexibility for risk of bias; paper trading bridges gap to live conditions; live trading is ultimate test but costly for experimentation.

---

## Examples & Counterexamples

### Examples of Backtest Framework Usage

1. **Moving Average Crossover Strategy**  
   - Signal: Buy when 50-day SMA crosses above 200-day SMA; sell on reverse crossover  
   - Backtest setup: Daily OHLC data, S&P 500 stocks, 2010–2023  
   - Execution model: Fill at next day's open price, 5 bps slippage, $10 commission per trade  
   - Result: 8% annualized return, 1.2 Sharpe ratio, 15% max drawdown  

2. **Mean Reversion with Bid-Ask Modeling**  
   - Signal: Buy when z-score < -2 (oversold), sell when z-score > 0  
   - Execution: Limit order at bid (buy) or ask (sell); fill only if price reaches limit  
   - Realistic: Models bid-ask spread crossing cost; partial fills if insufficient liquidity  

3. **Multi-Asset Portfolio Rebalancing**  
   - Strategy: 60% equities / 40% bonds, rebalance monthly  
   - Backtest: Track portfolio drift, generate rebalance orders, simulate transaction costs  
   - Metrics: Risk-adjusted return vs buy-and-hold, turnover costs

### Non-Examples (or Edge Cases)

- **Using closing prices for both signal and execution**: Unrealistic (signal generated after market close cannot trade at same close).
- **Ignoring survivorship bias in stock universe**: Backtesting only on current index constituents excludes delisted bankruptcies, inflates returns.
- **No slippage or commission**: Overstates profitability; live trading incurs costs that erode edge.

---

## Layer Breakdown

**Layer 1: Data Integrity & Point-in-Time Correctness**  
Ensure datasets reflect information available at historical decision points. Adjust for stock splits (multiply prices, divide shares), dividends (reduce cash or reinvest), delistings (handle forced liquidation). Use as-of dates for fundamentals (earnings released Q1 2020 available only after announcement date).

**Layer 2: Signal Generation Engine**  
Iterate through time series bar-by-bar (vectorized or event-driven). At each timestamp, compute indicators (SMA, RSI, z-score) using only prior data. Generate orders (buy/sell/hold) based on strategy logic. Store signals in order queue for execution module.

**Layer 3: Execution Simulation**  
Process orders using fill model:  
- **Market orders**: Fill at next bar's open (or midpoint of high-low for intraday)  
- **Limit orders**: Fill if price touches limit; queue if not filled  
- **Slippage**: Add random or fixed percentage (e.g., 5 bps) to simulate market impact  
- **Partial fills**: Limit order fill size proportional to bar volume

**Layer 4: Portfolio Accounting & Performance Calculation**  
Update positions after each trade. Calculate:  
\[
\text{PnL}_t = \text{Position}_{t-1} \times (\text{Price}_t - \text{Price}_{t-1}) - \text{Transaction Costs}
\]  
Accumulate equity curve, compute returns, drawdowns, Sharpe ratio. Track turnover (sum of absolute position changes).

---

## Mini-Project: Simple Backtest Framework in Python

**Goal:** Implement a moving average crossover strategy with transaction costs.

```python
import numpy as np
import pandas as pd

# Simulate price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01))
df = pd.DataFrame({'price': prices, 'date': dates})

# Moving averages
df['sma_50'] = df['price'].rolling(50).mean()
df['sma_200'] = df['price'].rolling(200).mean()

# Signal generation: 1 = long, 0 = flat, -1 = short
df['signal'] = 0
df.loc[df['sma_50'] > df['sma_200'], 'signal'] = 1
df.loc[df['sma_50'] <= df['sma_200'], 'signal'] = 0

# Position changes (detect crossovers)
df['position'] = df['signal'].diff()  # 1 = buy, -1 = sell, 0 = hold

# Execution: fill at next day's price with slippage
slippage_bps = 5  # 5 basis points
commission = 10  # $10 per trade
df['fill_price'] = df['price'].shift(-1)  # Next day's open (simplified)

# Calculate transaction costs
df['slippage_cost'] = 0.0
df.loc[df['position'] != 0, 'slippage_cost'] = (
    df.loc[df['position'] != 0, 'fill_price'] * (slippage_bps / 10000)
)
df['commission_cost'] = 0.0
df.loc[df['position'] != 0, 'commission_cost'] = commission

# Portfolio value (assume 1 share per trade for simplicity)
df['holdings'] = df['signal']  # 1 when long, 0 when flat
df['holdings'] = df['holdings'].fillna(method='ffill').fillna(0)

# Daily PnL
df['daily_pnl'] = df['holdings'].shift(1) * df['price'].diff()
df.loc[df['position'] != 0, 'daily_pnl'] -= (
    df.loc[df['position'] != 0, 'slippage_cost'] + 
    df.loc[df['position'] != 0, 'commission_cost']
)

# Cumulative equity
initial_capital = 10000
df['equity'] = initial_capital + df['daily_pnl'].fillna(0).cumsum()

# Performance metrics
total_return = (df['equity'].iloc[-1] - initial_capital) / initial_capital
sharpe_ratio = df['daily_pnl'].mean() / df['daily_pnl'].std() * np.sqrt(252)
max_drawdown = (df['equity'].cummax() - df['equity']).max() / df['equity'].cummax().max()
num_trades = df['position'].abs().sum()

print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Number of Trades: {int(num_trades)}")
print(f"Final Equity: ${df['equity'].iloc[-1]:,.2f}")
```

**Expected Output (illustrative):**
```
Total Return: 12.34%
Sharpe Ratio: 0.85
Max Drawdown: 8.21%
Number of Trades: 8
Final Equity: $11,234.56
```

**Interpretation:**  
- Positive return but moderate Sharpe ratio suggests strategy has signal but high volatility.  
- Transaction costs (slippage + commission) reduce raw returns; realistic modeling critical.  
- Max drawdown quantifies worst peak-to-trough decline; risk tolerance check.

---

## Challenge Round

1. **Look-Ahead Bias Detection**  
   A backtest uses `df['signal'] = df['price'].rolling(20).mean()` and fills orders at `df['price']` (same bar). What is the problem?

   <details><summary>Hint</summary>Signal calculated using bar's closing price, but execution assumed at same close. In reality, close price not known until after market closes. Fix: Use `df['price'].shift(-1)` for fill price (next bar's open) or calculate signal at prior bar's close.</details>

2. **Survivorship Bias Impact**  
   Backtest on current S&P 500 constituents (2010–2023) shows 15% annual return. Same strategy on historical S&P 500 composition (including delistings) shows 10% return. Explain discrepancy.

   <details><summary>Solution</summary>Current constituents exclude failed companies (bankruptcies, delistings), inflating returns. Historical composition includes survivorship-challenged firms, reducing average performance. Always use point-in-time index membership to avoid bias.</details>

3. **Slippage Modeling**  
   A strategy trades $100k notional per order in a stock with $10M average daily volume. Estimate realistic slippage using square-root market impact model:  
   \[
   \text{Slippage} \approx \sigma \sqrt{\frac{Q}{V}}
   \]  
   where \(\sigma = 1\%\) daily volatility, \(Q = \$100k\), \(V = \$10M\).

   <details><summary>Solution</summary>
   \[
   \text{Slippage} = 0.01 \times \sqrt{\frac{100{,}000}{10{,}000{,}000}} = 0.01 \times 0.1 = 0.1\% = 10 \text{ bps}
   \]  
   Strategy should include 10 bps slippage per side (20 bps round-trip) in backtest execution model.
   </details>

4. **Walk-Forward Robustness**  
   A strategy optimized on 2015–2020 data shows 2.0 Sharpe ratio. Walk-forward test on 2021–2023 shows 0.5 Sharpe ratio. What does this suggest?

   <details><summary>Hint</summary>Likely overfitting to in-sample period. Parameters tuned to historical idiosyncrasies fail to generalize. Solution: Use cross-validation, multiple out-of-sample periods, parameter stability analysis, or simpler models with fewer degrees of freedom.</details>

---

## Key References

- **Backtrader Framework**: Python library for event-driven backtesting ([Backtrader.com](https://www.backtrader.com/))
- **QuantConnect**: Cloud-based backtesting platform with institutional-grade data ([QuantConnect.com](https://www.quantconnect.com/))
- **Zipline**: Open-source Python library developed by Quantopian ([Zipline Docs](https://zipline.ml4trading.io/))
- **Almgren & Chriss (2001)**: Optimal execution and market impact models ([JSTOR](https://www.jstor.org/stable/2645747))

**Further Reading:**  
- *Advances in Financial Machine Learning* by Marcos López de Prado (Chapter on backtesting)  
- *Evidence-Based Technical Analysis* by David Aronson (cognitive biases in backtesting)
