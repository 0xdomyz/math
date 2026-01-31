# Performance Metrics

## Concept Skeleton

Performance metrics quantify trading strategy effectiveness across dimensions of return, risk, consistency, and capital efficiency, enabling comparison between strategies and benchmarks. Key metrics: Sharpe ratio (risk-adjusted return), maximum drawdown (worst peak-to-trough loss), hit rate (win percentage), profit factor (gross profit / gross loss), enabling objective evaluation beyond raw returns.

**Core Components:**
- **Return metrics**: Cumulative return, CAGR (compound annual growth rate), annualized return
- **Risk-adjusted return**: Sharpe ratio, Sortino ratio (downside risk only), Calmar ratio (return / max drawdown)
- **Drawdown analysis**: Maximum drawdown, average drawdown, recovery time
- **Win/loss statistics**: Hit rate, average win/loss, win/loss ratio, expectancy
- **Trading activity**: Turnover, number of trades, holding period, round-trip costs

**Why it matters:** Raw returns alone misleading (high return with high volatility or massive drawdown unacceptable); comprehensive metrics reveal risk-return tradeoffs and strategy fragility.

---

## Comparative Framing

| Metric | **Sharpe Ratio** | **Maximum Drawdown** | **Profit Factor** |
|--------|------------------|----------------------|-------------------|
| **Formula** | \(\frac{\bar{r} - r_f}{\sigma_r} \sqrt{T}\) | \(\max_t (\text{Peak}_t - \text{Equity}_t) / \text{Peak}_t\) | \(\sum \text{Wins} / \sum \|\text{Losses}\|\) |
| **Measures** | Risk-adjusted return | Worst capital loss | Gross profitability |
| **Ideal value** | > 1.0 (good), > 2.0 (excellent) | < 20% (manageable) | > 1.5 (profitable) |
| **Weakness** | Penalizes upside volatility; assumes normal distribution | Ignores frequency and recovery time | Ignores risk and volatility |
| **Use case** | Compare strategies with similar characteristics | Capital preservation; client risk tolerance | Quick profitability check |

**Key insight:** No single metric sufficient; use ensemble to capture return, risk, consistency, and capital efficiency. Sharpe + max drawdown + profit factor provide comprehensive view.

---

## Examples & Counterexamples

### Examples of Performance Metrics in Action

1. **Comparing Two Strategies**  
   - **Strategy A**: 20% annual return, 25% volatility → Sharpe = (0.20 - 0.03) / 0.25 = 0.68  
   - **Strategy B**: 15% annual return, 10% volatility → Sharpe = (0.15 - 0.03) / 0.10 = 1.20  
   - **Conclusion**: Strategy B superior on risk-adjusted basis despite lower raw return  

2. **Drawdown Analysis**  
   - Strategy peaks at $110k, falls to $88k → Max drawdown = (110 - 88) / 110 = 20%  
   - Recovery: 22 trading days to return to $110k peak  
   - **Interpretation**: 20% drawdown manageable for institutional capital; 22-day recovery acceptable  

3. **Hit Rate vs. Win/Loss Ratio**  
   - **Strategy X**: 40% hit rate, avg win $500, avg loss $200 → Expectancy = 0.40 × $500 - 0.60 × $200 = $80 per trade  
   - **Strategy Y**: 60% hit rate, avg win $200, avg loss $250 → Expectancy = 0.60 × $200 - 0.40 × $250 = $20 per trade  
   - **Conclusion**: Strategy X better despite lower hit rate (larger wins offset fewer wins)

### Non-Examples (or Edge Cases)

- **Using only cumulative return**: Ignores volatility and drawdowns; strategy with 50% return but 60% max drawdown may be unacceptable.
- **Sharpe ratio on monthly returns (too few samples)**: Needs at least 30 observations for statistical significance; prefer daily/weekly for 1–2 year backtests.
- **Profit factor without considering frequency**: Profit factor 2.0 from 5 trades is less reliable than profit factor 1.5 from 500 trades.

---

## Layer Breakdown

**Layer 1: Return Calculations**  
Compute returns from equity curve:  
\[
r_t = \frac{\text{Equity}_t - \text{Equity}_{t-1}}{\text{Equity}_{t-1}}
\]  
Cumulative return: \(\prod_{t=1}^{T} (1 + r_t) - 1\)  
CAGR: \(\left( \frac{\text{Final Equity}}{\text{Initial Equity}} \right)^{1/Y} - 1\) where \(Y\) = years  
Annualized return: \(\bar{r} \times T\) (T = 252 for daily, 52 for weekly)

**Layer 2: Risk Metrics**  
Volatility (standard deviation): \(\sigma_r = \sqrt{\frac{1}{N-1} \sum (r_t - \bar{r})^2}\)  
Downside deviation (Sortino): Only include negative returns in variance calculation  
Maximum drawdown: \(\max_{t \in [0, T]} \left( \frac{\text{Peak}_t - \text{Equity}_t}{\text{Peak}_t} \right)\)  
Peak-to-peak duration: Time between all-time highs (measures recovery speed)

**Layer 3: Win/Loss Statistics**  
Hit rate: \(\frac{\text{# Winning Trades}}{\text{# Total Trades}}\)  
Average win: \(\frac{\sum \text{Winning P&L}}{\text{# Winning Trades}}\)  
Average loss: \(\frac{\sum |\text{Losing P&L}|}{\text{# Losing Trades}}\)  
Profit factor: \(\frac{\sum \text{Winning P&L}}{\sum |\text{Losing P&L}|}\)  
Expectancy: \((\text{Hit Rate} \times \text{Avg Win}) - ((1 - \text{Hit Rate}) \times \text{Avg Loss})\)

**Layer 4: Efficiency & Activity**  
Turnover: \(\frac{\sum |\Delta \text{Position}| \times \text{Price}}{2 \times \text{Average AUM}}\) (annualized)  
Round-trip costs: Turnover × (commission + slippage)  
Information ratio: \(\frac{\bar{r} - \bar{r}_{\text{benchmark}}}{\sigma_{r - r_{\text{benchmark}}}}\) (active return per unit tracking error)

---

## Mini-Project: Comprehensive Performance Report

**Goal:** Calculate key metrics for a simulated trading strategy.

```python
import numpy as np
import pandas as pd

# Simulate equity curve
np.random.seed(99)
n_days = 500
daily_returns = np.random.randn(n_days) * 0.015 + 0.0005  # Mean 0.05%, vol 1.5%
equity_curve = 100000 * (1 + daily_returns).cumprod()
df = pd.DataFrame({'equity': equity_curve, 'returns': daily_returns})

# Cumulative return
cumulative_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]

# CAGR
years = n_days / 252
cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0]) ** (1 / years) - 1

# Sharpe ratio (assume 3% risk-free rate)
excess_returns = df['returns'] - 0.03 / 252
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

# Sortino ratio (downside risk only)
downside_returns = df['returns'][df['returns'] < 0]
downside_std = downside_returns.std()
sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252)

# Maximum drawdown
running_max = df['equity'].cummax()
drawdown = (df['equity'] - running_max) / running_max
max_drawdown = drawdown.min()

# Calmar ratio
calmar_ratio = cagr / abs(max_drawdown)

# Simulate individual trades for win/loss stats
np.random.seed(55)
n_trades = 100
trade_pnl = np.random.randn(n_trades) * 500 + 50  # Avg $50, vol $500
winning_trades = trade_pnl[trade_pnl > 0]
losing_trades = trade_pnl[trade_pnl < 0]

hit_rate = len(winning_trades) / n_trades
avg_win = winning_trades.mean()
avg_loss = abs(losing_trades.mean())
profit_factor = winning_trades.sum() / abs(losing_trades.sum())
expectancy = (hit_rate * avg_win) - ((1 - hit_rate) * avg_loss)

# Print performance report
print("=" * 50)
print("PERFORMANCE METRICS REPORT")
print("=" * 50)
print(f"Cumulative Return:        {cumulative_return:>10.2%}")
print(f"CAGR:                     {cagr:>10.2%}")
print(f"Sharpe Ratio:             {sharpe_ratio:>10.2f}")
print(f"Sortino Ratio:            {sortino_ratio:>10.2f}")
print(f"Max Drawdown:             {max_drawdown:>10.2%}")
print(f"Calmar Ratio:             {calmar_ratio:>10.2f}")
print("-" * 50)
print(f"Number of Trades:         {n_trades:>10}")
print(f"Hit Rate:                 {hit_rate:>10.2%}")
print(f"Average Win:              ${avg_win:>9,.2f}")
print(f"Average Loss:             ${avg_loss:>9,.2f}")
print(f"Win/Loss Ratio:           {avg_win/avg_loss:>10.2f}")
print(f"Profit Factor:            {profit_factor:>10.2f}")
print(f"Expectancy per Trade:     ${expectancy:>9,.2f}")
print("=" * 50)
```

**Expected Output (illustrative):**
```
==================================================
PERFORMANCE METRICS REPORT
==================================================
Cumulative Return:             35.67%
CAGR:                          16.34%
Sharpe Ratio:                   1.15
Sortino Ratio:                  1.68
Max Drawdown:                  -12.45%
Calmar Ratio:                   1.31
--------------------------------------------------
Number of Trades:                100
Hit Rate:                      54.00%
Average Win:                  $552.34
Average Loss:                 $485.12
Win/Loss Ratio:                  1.14
Profit Factor:                   1.32
Expectancy per Trade:          $75.31
==================================================
```

**Interpretation:**  
- Sharpe 1.15: Acceptable risk-adjusted return; institutional threshold ~1.0+  
- Max drawdown 12.45%: Manageable; suggests moderate volatility control  
- Profit factor 1.32: Profitable but modest; gross wins 32% higher than gross losses  
- Positive expectancy ($75.31): Strategy has statistical edge

---

## Challenge Round

1. **Sharpe Ratio Limitations**  
   A strategy has symmetric returns: half the time +10%, half -10%. What is Sharpe ratio? Is this a good strategy?

   <details><summary>Solution</summary>
   Mean return: 0.5 × 0.10 + 0.5 × (-0.10) = 0%.  
   Volatility: \(\sqrt{0.5 \times (0.10)^2 + 0.5 \times (-0.10)^2} = 0.10 = 10\%\).  
   Sharpe (assuming 0% risk-free): 0 / 0.10 = 0.  
   **No edge**: Zero expected return despite high volatility. Not investable.
   </details>

2. **Maximum Drawdown vs. Average Drawdown**  
   Strategy A: Max DD 30%, avg DD 5%, few deep drawdowns. Strategy B: Max DD 15%, avg DD 10%, frequent moderate drawdowns. Which is preferable?

   <details><summary>Hint</summary>Depends on risk tolerance. Strategy A: Rare but severe losses (tail risk). Strategy B: Consistent moderate pain. Institutional investors often prefer B (predictable risk). Aggressive investors may tolerate A for higher returns if infrequent. Measure recovery time: If A's 30% DD recovers quickly, may be acceptable.</details>

3. **Hit Rate vs. Profit Factor Trade-Off**  
   Optimize strategy parameters. Option 1: Hit rate 70%, profit factor 1.3. Option 2: Hit rate 45%, profit factor 2.0. Which is better?

   <details><summary>Solution</summary>
   **Expectancy is key**: Assume $100 avg win/loss for simplicity.  
   Option 1: \(E = 0.70 \times 100 - 0.30 \times (100 \times 1.3/0.7) = 70 - 55.7 = \$14.3\).  
   Option 2: Profit factor 2.0 implies wins = 2 × losses. If avg loss $100, avg win $200. \(E = 0.45 \times 200 - 0.55 \times 100 = 90 - 55 = \$35\).  
   **Option 2 superior**: Larger expectancy. High-probability strategies (Option 1) psychologically easier but may have smaller edge.
   </details>

4. **Calmar Ratio vs. Sharpe Ratio**  
   Strategy X: Sharpe 1.5, Calmar 2.0. Strategy Y: Sharpe 1.2, Calmar 2.5. Which is better for risk-averse investor?

   <details><summary>Hint</summary>Calmar ratio (return / max drawdown) emphasizes capital preservation. Strategy Y has lower Sharpe (higher volatility relative to return) but better Calmar (smaller drawdown). Risk-averse investors prioritize downside protection → prefer Strategy Y. Sharpe-focused investors (willing to tolerate volatility) prefer Strategy X.</details>

---

## Key References

- **Sharpe (1994)**: "The Sharpe Ratio" ([Journal of Portfolio Management](https://www.jstor.org/stable/4479935))
- **Sortino & van der Meer (1991)**: "Downside Risk" ([Journal of Portfolio Management](https://www.jstor.org/))
- **Bacon (2008)**: *Practical Portfolio Performance Measurement and Attribution* (comprehensive metrics) ([Wiley](https://www.wiley.com/))
- **Pardo (2008)**: *The Evaluation and Optimization of Trading Strategies* (trade-level metrics) ([Wiley](https://www.wiley.com/))

**Further Reading:**  
- Omega ratio: Probability-weighted ratio of gains vs. losses (addresses non-normal returns)  
- Information ratio for relative performance vs. benchmark  
- Maximum adverse excursion (MAE) / Maximum favorable excursion (MFE) for trade quality analysis
