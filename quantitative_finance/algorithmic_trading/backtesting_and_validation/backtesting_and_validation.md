# Backtesting and Validation

## 1. Concept Skeleton
**Definition:** Simulation of trading strategies on historical data to evaluate performance, robustness, and risk characteristics before live deployment  
**Purpose:** Validate strategy logic; estimate expected returns/risk; detect overfitting; identify parameter sensitivity; prevent costly live trading errors  
**Prerequisites:** Historical data quality, programming, statistics, understanding of biases (look-ahead, survivorship), transaction cost modeling

## 2. Comparative Framing
| Validation Type | In-Sample Testing | Out-of-Sample Testing | Walk-Forward Analysis | Monte Carlo Simulation | Paper Trading |
|-----------------|-------------------|----------------------|----------------------|------------------------|---------------|
| **Data Used** | Training data | Hold-out set | Rolling windows | Resampled returns | Live data (no capital) |
| **Overfitting Risk** | Very high | Moderate | Low | Low | None |
| **Realism** | Low | Moderate | High | Moderate | Highest |
| **Speed** | Fast | Fast | Slow | Slow | Real-time only |
| **Purpose** | Strategy development | Initial validation | Robustness check | Statistical confidence | Final validation |

| Bias Type | Look-Ahead | Survivorship | Data Mining | Overfitting | Psychological |
|-----------|------------|--------------|-------------|-------------|---------------|
| **Cause** | Future data in signals | Delisted stocks excluded | Testing many strategies | Too many parameters | Live vs backtest behavior |
| **Impact on Results** | Inflated returns | +2-3% annual | False discoveries | Degradation live | Worse execution |
| **Detection** | Code review, point-in-time | Check delistings | Multiple testing correction | Out-of-sample decline | Compare paper/live |
| **Prevention** | Strict data discipline | Include delisted data | Pre-register hypothesis | Regularization, CV | Automated execution |

## 3. Examples + Counterexamples

**Simple Example:**  
Backtest MA crossover on S&P 500 (2010-2020). Check if 50/200 MA generates positive returns after transaction costs. Compare to buy-and-hold.

**Perfect Fit:**  
Walk-forward optimization: Train on 2 years, test on 6 months, roll forward. Repeatedly validates on unseen data, prevents parameter overfitting.

**Out-of-Sample Validation:**  
Reserve last 20% of data for final testing. Never optimize on this data. If strategy fails out-of-sample, discard (overfitted to training set).

**Monte Carlo Robustness:**  
Resample actual trades 10,000 times. Calculate 95% confidence interval on Sharpe ratio. If lower bound < 0, strategy lacks statistical significance.

**Poor Fit:**  
Optimize RSI parameters on entire dataset, report best result (RSI=17 gives 50% return!). No out-of-sample testing. Guaranteed to fail live (data mining bias).

**Catastrophic Failure:**  
Backtest shows 100% win rate → Look-ahead bias (used future prices in signals). Or survivorship bias (only tested stocks that survived, ignoring bankruptcies).

## 4. Layer Breakdown
```
Backtesting Framework:

├─ Data Requirements:
│  ├─ Price Data Quality:
│  │   ├─ Adjusted for Corporate Actions:
│  │   │   ├─ Stock splits: Adjust historical prices
│  │   │   ├─ Dividends: Total return calculation
│  │   │   ├─ Rights issues, spin-offs
│  │   │   └─ Mergers & acquisitions
│  │   ├─ Point-in-Time Data:
│  │   │   ├─ Use data available at that moment only
│  │   │   ├─ No revised/restated financials
│  │   │   ├─ Index constituents at each date
│  │   │   └─ Prevents look-ahead bias
│  │   ├─ Survivorship Bias Free:
│  │   │   ├─ Include delisted stocks (bankruptcies)
│  │   │   ├─ Missing: ~2-3% annual return overstatement
│  │   │   ├─ Obtain from providers: CRSP, Bloomberg
│  │   │   └─ Critical for stock selection strategies
│  │   ├─ Data Frequency:
│  │   │   ├─ Daily: Most strategies
│  │   │   ├─ Intraday: HFT, market making
│  │   │   ├─ Minute/tick: Microstructure strategies
│  │   │   └─ Match strategy frequency to data
│  │   └─ Data Cleaning:
│  │       ├─ Outlier detection: Price spikes, bad ticks
│  │       ├─ Missing data: Forward fill, interpolation
│  │       ├─ Volume filters: Minimum liquidity
│  │       └─ Consistency checks: OHLC relationships
│  ├─ Fundamental Data (if used):
│  │   ├─ As-Reported: Not restated
│  │   ├─ Announcement Dates: Actual release times
│  │   ├─ Restatement Tracking: Note corrections
│  │   └─ Timeliness: Data available when?
│  ├─ Alternative Data:
│  │   ├─ News sentiment: Publication timestamps
│  │   ├─ Social media: Real-time scraping
│  │   ├─ Satellite imagery: Date-stamped
│  │   └─ Vendor quality varies widely
│  └─ Benchmark Data:
│      Index levels, risk-free rates, sector returns
│      For performance comparison
├─ Backtesting Engine Architecture:
│  ├─ Event-Driven Design:
│  │   ├─ Time Loop:
│  │   │   For each bar (day/minute):
│  │   │     1. Update market data
│  │   │     2. Calculate indicators
│  │   │     3. Generate signals
│  │   │     4. Execute orders
│  │   │     5. Update portfolio state
│  │   │     6. Record performance
│  │   ├─ Prevents Look-Ahead:
│  │   │   Strictly chronological processing
│  │   │   Signal at t uses data up to t only
│  │   └─ Realistic Execution:
│  │       Orders execute at next bar (delay)
│  │       Cannot use current bar close for entry
│  ├─ Vectorized Approach (alternative):
│  │   ├─ Pandas operations on entire series
│  │   ├─ Fast for simple strategies
│  │   ├─ Must carefully avoid look-ahead
│  │   └─ Harder for complex logic (portfolio)
│  ├─ Portfolio State:
│  │   ├─ Cash balance
│  │   ├─ Holdings: {symbol: shares}
│  │   ├─ Market value: Cash + Σ(price × shares)
│  │   ├─ Open orders
│  │   └─ Historical transactions
│  ├─ Order Management:
│  │   ├─ Order Types:
│  │   │   ├─ Market: Execute at next open/close
│  │   │   ├─ Limit: Fill if price reached
│  │   │   ├─ Stop: Trigger then market
│  │   │   └─ Stop-limit: Trigger then limit
│  │   ├─ Fill Simulation:
│  │   │   ├─ Market: Assume filled at next bar
│  │   │   ├─ Limit: Check if price crosses limit
│  │   │   ├─ Partial fills: Volume-based probability
│  │   │   └─ Slippage model: BPS or % of order size
│  │   └─ Order Rejection:
│  │       Insufficient cash, minimum size, halts
│  └─ Performance Tracking:
│      ├─ Daily P&L
│      ├─ Cumulative returns
│      ├─ Holdings history
│      ├─ Trade log
│      └─ Risk metrics
├─ Transaction Cost Modeling:
│  ├─ Commission:
│  │   ├─ Per-share: $0.005/share (typical)
│  │   ├─ Percentage: 0.1% of notional
│  │   ├─ Flat fee + per-share: $1 + $0.005/share
│  │   └─ Minimum per trade: $1
│  ├─ Slippage:
│  │   ├─ Bid-Ask Spread:
│  │   │   ├─ Half-spread cost per trade
│  │   │   ├─ Liquid stocks: 0.01-0.05%
│  │   │   ├─ Illiquid: 0.5-2%
│  │   │   └─ Can model from order book data
│  │   ├─ Market Impact:
│  │   │   ├─ Temporary: Price moves during execution
│  │   │   ├─ Permanent: Information revelation
│  │   │   ├─ Square-root law: Impact ∝ √(order_size/ADV)
│  │   │   │   ADV = Average daily volume
│  │   │   └─ Almgren-Chriss model: Detailed impact
│  │   └─ Combined Model:
│  │       Total cost = Commission + α×(Order_Size/ADV)^β
│  │       α, β: Calibrated from execution data
│  ├─ Short Selling Costs:
│  │   ├─ Borrow fee: Annual rate (0.3-10%+)
│  │   ├─ Hard-to-borrow stocks: Higher fees
│  │   ├─ Rebate rate: Negative for easy shorts
│  │   └─ Availability: May not be able to short
│  ├─ Margin Interest:
│  │   Cost of leverage (if long > cash)
│  │   Typically broker rate + spread (3-7%)
│  └─ Taxes (if applicable):
│      Short-term vs long-term capital gains
│      Wash sale rules
├─ Common Biases and Pitfalls:
│  ├─ Look-Ahead Bias:
│  │   ├─ Using Future Information:
│  │   │   ├─ Calculate indicator with future data
│  │   │   ├─ Rebalance using close when signal at open
│  │   │   ├─ Use revised financial data
│  │   │   └─ Detection: Unrealistic Sharpe (>3)
│  │   ├─ Prevention:
│  │   │   ├─ Strict data versioning (point-in-time)
│  │   │   ├─ Signal generation before execution
│  │   │   ├─ Execute at next bar open (not current close)
│  │   │   └─ Code review: Check shift() operations
│  │   └─ Example:
│  │       # WRONG: signal = (close > MA).shift(1)
│  │       # RIGHT: position = signal.shift(1)
│  ├─ Survivorship Bias:
│  │   ├─ Cause:
│  │   │   Only test stocks that survived to present
│  │   │   Exclude bankruptcies, delistings
│  │   ├─ Impact:
│  │   │   +2-3% annual return overstatement
│  │   │   Underestimate risk (missing failures)
│  │   ├─ Detection:
│  │   │   Count stocks: Should match historical universe
│  │   │   Check for delisting events
│  │   └─ Fix:
│  │       Use survivorship-bias-free database
│  │       Include delisted stocks with proper handling
│  ├─ Data Mining / Multiple Testing:
│  │   ├─ Problem:
│  │   │   Test 1000 strategies, report best one
│  │   │   p=0.05 → 50 false positives expected
│  │   │   "Torturing data until it confesses"
│  │   ├─ Bonferroni Correction:
│  │   │   Adjusted p-value = p / n_tests
│  │   │   Very conservative
│  │   ├─ Sharpe Ratio Adjustment:
│  │   │   Haircut for multiple tests
│  │   │   Expected max Sharpe ≈ √(2 log n_tests)
│  │   └─ Prevention:
│  │       Pre-register hypothesis (write down before test)
│  │       Use out-of-sample validation
│  │       Economic rationale (not just data mining)
│  ├─ Overfitting:
│  │   ├─ Too Many Parameters:
│  │   │   10 parameters with 5 values each → 10^5 combinations
│  │   │   Will find something that works by chance
│  │   ├─ Symptoms:
│  │   │   ├─ Excellent in-sample, poor out-of-sample
│  │   │   ├─ Complex rules with many conditions
│  │   │   ├─ Parameters at extreme values
│  │   │   └─ High sensitivity to small changes
│  │   ├─ Prevention:
│  │   │   ├─ Limit parameters (< 5 typically)
│  │   │   ├─ Regularization (L1/L2 penalty)
│  │   │   ├─ Cross-validation (walk-forward)
│  │   │   ├─ Parameter stability: Test nearby values
│  │   │   └─ Economic logic: Why should it work?
│  │   └─ Detection:
│  │       Compare in-sample vs out-of-sample Sharpe
│  │       Ratio > 2: Likely overfit
│  ├─ Ignoring Transaction Costs:
│  │   ├─ High-frequency strategies especially sensitive
│  │   ├─ Turnover matters: 500% annual → 5× cost impact
│  │   ├─ Test: Double costs, halve them (robustness)
│  │   └─ Slippage often underestimated in backtest
│  ├─ Small Sample Size:
│  │   ├─ 2-year backtest: ~500 days
│  │   ├─ If strategy trades weekly: ~100 trades
│  │   ├─ Statistical significance low
│  │   └─ Need: 5-10 years data minimum
│  └─ Regime Dependency:
│      Strategy works in bull market only
│      Test across market conditions (2008, 2020)
├─ Validation Techniques:
│  ├─ Train-Test Split:
│  │   ├─ Simple Hold-Out:
│  │   │   ├─ Train: First 70-80% of data
│  │   │   ├─ Test: Last 20-30% of data
│  │   │   ├─ Never optimize on test set
│  │   │   └─ Single test preserves data
│  │   ├─ Multiple Periods:
│  │   │   Alternate train/test periods
│  │   │   Tests robustness across regimes
│  │   └─ Limitations:
│  │       Single test set may not be representative
│  │       Wastes data (test set unused for training)
│  ├─ Walk-Forward Optimization:
│  │   ├─ Process:
│  │   │   1. Train on window (e.g., 2 years)
│  │   │   2. Optimize parameters on training window
│  │   │   3. Test on next period (e.g., 6 months)
│  │   │   4. Roll window forward
│  │   │   5. Repeat until end of data
│  │   │   6. Combine all test periods for total performance
│  │   ├─ Advantages:
│  │   │   ├─ Uses all data (efficient)
│  │   │   ├─ Multiple out-of-sample tests
│  │   │   ├─ Tests parameter stability over time
│  │   │   └─ Realistic: Mimics live re-optimization
│  │   ├─ Parameters:
│  │   │   ├─ Training window: 1-3 years
│  │   │   ├─ Test window: 3-12 months
│  │   │   ├─ Re-optimization frequency: Match test window
│  │   │   └─ Anchored vs rolling: Trade-off data/recency
│  │   └─ Evaluation:
│  │       Aggregate test period results
│  │       Should be consistent with in-sample
│  ├─ Cross-Validation (Time Series):
│  │   ├─ Blocked CV:
│  │   │   Split into sequential blocks
│  │   │   Train on blocks 1-3, test on 4
│  │   │   Train on blocks 2-4, test on 5
│  │   │   Preserves time order
│  │   ├─ Purging:
│  │   │   Remove data near test set from training
│  │   │   Prevents label leakage from autocorrelation
│  │   ├─ Embargo:
│  │   │   Gap between train and test sets
│  │   │   Accounts for order execution delay
│  │   └─ Combinatorial Purged CV:
│  │       Advanced: Multiple paths through data
│  │       Maximizes usage while preserving order
│  ├─ Monte Carlo Simulation:
│  │   ├─ Bootstrap Trades:
│  │   │   ├─ Resample actual trades with replacement
│  │   │   ├─ Generate 10,000 synthetic equity curves
│  │   │   ├─ Calculate percentiles (5th, 50th, 95th)
│  │   │   └─ Confidence intervals on Sharpe, drawdown
│  │   ├─ Return Shuffling:
│  │   │   Randomly permute daily returns
│  │   │   Tests if order matters (should for real edge)
│  │   ├─ Block Bootstrap:
│  │   │   Resample blocks to preserve autocorrelation
│  │   │   More realistic than iid shuffle
│  │   └─ Stress Testing:
│  │       Inject worst historical drawdown
│  │       How would strategy behave?
│  ├─ Sensitivity Analysis:
│  │   ├─ Parameter Sweep:
│  │   │   ├─ Vary each parameter ±20%
│  │   │   ├─ Plot Sharpe vs parameter value
│  │   │   ├─ Robust: Smooth curve, broad maximum
│  │   │   ├─ Overfit: Spiky, single peak
│  │   │   └─ Example: RSI period 10-20, test all
│  │   ├─ Multi-Dimensional:
│  │   │   Heat map: 2 parameters simultaneously
│  │   │   Look for stable regions
│  │   ├─ Transaction Cost Sensitivity:
│  │   │   Double costs: Strategy still profitable?
│  │   │   Critical for high-turnover strategies
│  │   └─ Market Regime:
│  │       Subsample: Bull, bear, high-vol periods
│  │       Consistent performance across all?
│  ├─ Statistical Significance:
│  │   ├─ Sharpe Ratio t-Test:
│  │   │   H0: Sharpe = 0 (no skill)
│  │   │   t = Sharpe × √n, n = num periods
│  │   │   p-value from t-distribution
│  │   ├─ Deflated Sharpe Ratio:
│  │   │   Adjust for multiple testing, non-normality
│  │   │   Bailey & López de Prado (2014)
│  │   ├─ Minimum Track Record Length:
│  │   │   How long to distinguish skill from luck?
│  │   │   Depends on Sharpe, confidence level
│  │   └─ Omega Ratio:
│  │       Probability-weighted gains/losses
│  │       Alternative to Sharpe
│  └─ Paper Trading:
│      ├─ Live market data, simulated execution
│      ├─ Final validation before real capital
│      ├─ Detects: API issues, data feed problems
│      ├─ Duration: 3-6 months minimum
│      └─ Compare to backtest: Similar metrics?
├─ Performance Metrics:
│  ├─ Return Measures:
│  │   ├─ Total Return: (End - Start) / Start
│  │   ├─ CAGR: [(End/Start)^(1/Years) - 1]
│  │   ├─ Arithmetic Mean: Avg(daily returns) × 252
│  │   ├─ Geometric Mean: More accurate for compounding
│  │   └─ Alpha: Excess return vs benchmark (CAPM)
│  ├─ Risk-Adjusted Measures:
│  │   ├─ Sharpe Ratio: (Return - Rf) / σ
│  │   │   ├─ Most common metric
│  │   │   ├─ Assumes normal distribution
│  │   │   ├─ Sharpe > 1: Good, > 2: Excellent, > 3: Suspicious
│  │   │   └─ Annualized: Daily Sharpe × √252
│  │   ├─ Sortino Ratio: (Return - MAR) / Downside_σ
│  │   │   ├─ Only penalizes downside volatility
│  │   │   ├─ MAR = Minimum acceptable return
│  │   │   └─ Better for asymmetric returns
│  │   ├─ Calmar Ratio: CAGR / Max_Drawdown
│  │   │   Focus on tail risk
│  │   ├─ Information Ratio: Alpha / Tracking_Error
│  │   │   Skill relative to benchmark
│  │   └─ Omega Ratio: Gains_above_threshold / Losses_below
│  │       Captures higher moments (skew, kurtosis)
│  ├─ Risk Measures:
│  │   ├─ Volatility: std(returns) × √252
│  │   ├─ Max Drawdown: Largest peak-to-trough decline
│  │   │   Critical metric (can you survive it?)
│  │   ├─ Average Drawdown: Mean of all drawdowns
│  │   ├─ Drawdown Duration: Time to recover peak
│  │   ├─ VaR: Value at Risk (5% quantile)
│  │   ├─ CVaR: Conditional VaR (expected loss beyond VaR)
│  │   ├─ Skewness: Asymmetry of returns
│  │   │   Negative: Fat left tail (bad)
│  │   └─ Kurtosis: Fat tails (excess kurtosis > 3)
│  ├─ Trade Statistics:
│  │   ├─ Win Rate: % profitable trades
│  │   ├─ Avg Win / Avg Loss: Reward-to-risk ratio
│  │   ├─ Expectancy: WinRate×AvgWin - LossRate×AvgLoss
│  │   ├─ Profit Factor: Gross_Profits / Gross_Losses
│  │   ├─ Number of Trades: More = better statistics
│  │   └─ Avg Trade Duration: Holding period
│  ├─ Portfolio Metrics:
│  │   ├─ Turnover: Sum(|trade|) / PortfolioValue (annual)
│  │   │   High turnover (>500%) → Cost-sensitive
│  │   ├─ Leverage: Gross_Exposure / Net_Asset_Value
│  │   │   Long + Short (gross), Long - Short (net)
│  │   ├─ Concentration: Max position size
│  │   │   Herfindahl index for diversification
│  │   └─ Correlation to Market: Beta
│  │       Market-neutral should have β ≈ 0
│  └─ Consistency Metrics:
│      ├─ % Positive Months/Years
│      ├─ Longest Winning/Losing Streak
│      ├─ Rolling Sharpe Ratio (stability)
│      └─ Regime Performance (bull/bear/sideways)
├─ Reporting and Visualization:
│  ├─ Equity Curve:
│  │   ├─ Cumulative returns over time
│  │   ├─ Log scale for long periods
│  │   ├─ Compare to benchmark
│  │   └─ Mark drawdown periods
│  ├─ Drawdown Chart:
│  │   Underwater plot: % off peak
│  │   Shows recovery periods
│  ├─ Rolling Performance:
│  │   ├─ 12-month rolling Sharpe
│  │   ├─ Rolling volatility
│  │   └─ Shows degradation/improvement
│  ├─ Monthly/Annual Returns Table:
│  │   Heat map of returns
│  │   Quickly spot bad periods
│  ├─ Return Distribution:
│  │   Histogram with normal overlay
│  │   Check for fat tails, skewness
│  ├─ Trade Analysis:
│  │   ├─ Win/loss distribution
│  │   ├─ Trade duration histogram
│  │   └─ Profit by entry time, day of week
│  └─ Summary Statistics Table:
│      All metrics in one view
│      Compare strategies side-by-side
└─ Best Practices:
   ├─ Pre-Registration:
   │   Write down hypothesis before testing
   │   Prevents data mining bias
   ├─ Economic Rationale:
   │   Why should strategy work?
   │   Behavioral bias, market inefficiency, risk premium?
   ├─ Start Simple:
   │   Baseline: Buy-and-hold, momentum, mean-reversion
   │   Does complex strategy beat simple one?
   ├─ Incremental Complexity:
   │   Add features one at a time
   │   Does each improve out-of-sample?
   ├─ Long Backtest Period:
   │   Minimum 5 years, preferably 10+
   │   Include crisis: 2008, 2020, etc.
   ├─ Multiple Asset Classes:
   │   Test on stocks, futures, FX
   │   True edge should generalize
   ├─ Conservative Assumptions:
   │   Overestimate costs, underestimate fill rates
   │   Better to be surprised positively live
   ├─ Document Everything:
   │   Code version, data source, parameters
   │   Reproducibility critical
   ├─ Version Control:
   │   Git for code, strategy changes
   │   Track what was tested when
   └─ Continuous Monitoring:
      ├─ Compare live to backtest
      ├─ Track slippage, costs
      ├─ Re-validate periodically
      └─ Adapt to regime changes
```

**Interaction:** Clean data (point-in-time, survivorship-free) → Implement strategy (event-driven) → Backtest with costs → Validate (walk-forward, out-of-sample) → Analyze metrics → Paper trade → Deploy live.

## 5. Mini-Project
Implement comprehensive backtesting framework with validation techniques:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("BACKTESTING AND VALIDATION FRAMEWORK")
print("="*70)

class BacktestEngine:
    """Event-driven backtesting engine"""
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission  # Round-trip as fraction
        self.slippage = slippage  # Each way as fraction
        
        # State
        self.cash = initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_values = []
        self.trades = []
        self.dates = []
        
    def execute_trade(self, symbol, shares, price, date):
        """Execute a trade with transaction costs"""
        if shares == 0:
            return
        
        # Transaction costs
        notional = abs(shares * price)
        commission_cost = notional * self.commission
        slippage_cost = notional * self.slippage
        
        total_cost = commission_cost + slippage_cost
        
        # Update cash (negative for buys, positive for sells)
        self.cash -= shares * price + (total_cost if shares > 0 else -total_cost)
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += shares
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'commission': commission_cost,
            'slippage': slippage_cost,
            'total_cost': total_cost
        })
    
    def get_portfolio_value(self, prices):
        """Calculate total portfolio value"""
        holdings_value = sum(
            self.positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.cash + holdings_value
    
    def record_state(self, date, prices):
        """Record portfolio state for this date"""
        portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_values.append(portfolio_value)
        self.dates.append(date)
    
    def run_backtest(self, signals, prices):
        """
        Run backtest on signal dataframe
        signals: DataFrame with columns = symbols, values = target position (-1, 0, 1)
        prices: DataFrame with columns = symbols, values = prices
        """
        # Ensure aligned
        dates = signals.index.intersection(prices.index)
        
        for date in dates:
            current_prices = prices.loc[date].to_dict()
            target_positions = signals.loc[date].to_dict()
            
            # Calculate required trades
            for symbol in target_positions:
                target_shares = target_positions[symbol]
                current_shares = self.positions.get(symbol, 0)
                
                # Scale by capital allocation (simplified: equal weight)
                if target_shares != 0:
                    position_size = self.initial_capital * 0.1  # 10% per position
                    target_shares = int(position_size / current_prices[symbol]) * target_shares
                
                # Execute trade if needed
                trade_shares = target_shares - current_shares
                if trade_shares != 0 and symbol in current_prices:
                    self.execute_trade(symbol, trade_shares, current_prices[symbol], date)
            
            # Record state
            self.record_state(date, current_prices)
        
        return self.get_results()
    
    def get_results(self):
        """Calculate performance metrics"""
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        
        # Total return
        total_return = (portfolio_series.iloc[-1] / self.initial_capital) - 1
        
        # CAGR
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_dd_length = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_dd_length += 1
            else:
                if current_dd_length > 0:
                    drawdown_periods.append(current_dd_length)
                current_dd_length = 0
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Trade stats
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_dd_duration': max_dd_duration,
            'num_trades': len(self.trades),
            'portfolio_series': portfolio_series,
            'returns': returns,
            'trades_df': trades_df
        }

class WalkForwardAnalysis:
    """Walk-forward optimization and validation"""
    
    def __init__(self, train_period=504, test_period=126, step=126):
        """
        train_period: Training window in days (504 = 2 years)
        test_period: Test window in days (126 = 6 months)
        step: Step size for rolling (126 = 6 months)
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
    
    def optimize_strategy(self, data, param_grid):
        """
        Find best parameters on training data
        Returns best params and their performance
        """
        best_sharpe = -np.inf
        best_params = None
        
        for params in param_grid:
            # Generate signals with these parameters
            signals = self.generate_signals(data, params)
            
            # Quick performance calc
            returns = signals.shift(1) * np.log(data / data.shift(1))
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return best_params, best_sharpe
    
    def generate_signals(self, prices, params):
        """Generate trading signals based on parameters"""
        # Example: MA crossover with parameters
        fast_period = params.get('fast', 50)
        slow_period = params.get('slow', 200)
        
        ma_fast = prices.rolling(window=fast_period).mean()
        ma_slow = prices.rolling(window=slow_period).mean()
        
        signals = pd.Series(0, index=prices.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1
        
        return signals
    
    def run_walk_forward(self, prices):
        """Execute walk-forward analysis"""
        results = []
        n = len(prices)
        
        start = 0
        while start + self.train_period + self.test_period <= n:
            # Define windows
            train_end = start + self.train_period
            test_end = train_end + self.test_period
            
            train_data = prices.iloc[start:train_end]
            test_data = prices.iloc[train_end:test_end]
            
            # Parameter grid (simplified)
            param_grid = [
                {'fast': 20, 'slow': 50},
                {'fast': 50, 'slow': 200},
                {'fast': 10, 'slow': 30}
            ]
            
            # Optimize on training data
            best_params, train_sharpe = self.optimize_strategy(train_data, param_grid)
            
            # Test on out-of-sample data
            test_signals = self.generate_signals(test_data, best_params)
            test_returns = test_signals.shift(1) * np.log(test_data / test_data.shift(1))
            test_sharpe = (test_returns.mean() * 252) / (test_returns.std() * np.sqrt(252))
            
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'test_returns': test_returns
            })
            
            # Roll forward
            start += self.step
        
        return results

class MonteCarloValidator:
    """Monte Carlo simulation for strategy validation"""
    
    def __init__(self, returns, n_simulations=10000):
        self.returns = returns
        self.n_simulations = n_simulations
    
    def bootstrap_trades(self):
        """Bootstrap individual trades"""
        # Assuming we have trade returns (not daily)
        # For daily returns, we'll bootstrap blocks
        results = []
        
        for _ in range(self.n_simulations):
            # Resample with replacement
            simulated_returns = np.random.choice(
                self.returns, size=len(self.returns), replace=True
            )
            
            # Calculate metrics
            cumulative = (1 + simulated_returns).prod() - 1
            sharpe = simulated_returns.mean() / simulated_returns.std() * np.sqrt(252)
            
            # Max drawdown
            cum_series = (1 + pd.Series(simulated_returns)).cumprod()
            running_max = cum_series.expanding().max()
            drawdown = (cum_series - running_max) / running_max
            max_dd = drawdown.min()
            
            results.append({
                'return': cumulative,
                'sharpe': sharpe,
                'max_dd': max_dd
            })
        
        return pd.DataFrame(results)
    
    def get_confidence_intervals(self, confidence=0.95):
        """Calculate confidence intervals"""
        results = self.bootstrap_trades()
        
        alpha = 1 - confidence
        lower = alpha / 2
        upper = 1 - lower
        
        ci = {
            'return': (results['return'].quantile(lower), results['return'].quantile(upper)),
            'sharpe': (results['sharpe'].quantile(lower), results['sharpe'].quantile(upper)),
            'max_dd': (results['max_dd'].quantile(lower), results['max_dd'].quantile(upper))
        }
        
        return ci, results

def detect_overfitting(in_sample_sharpe, out_sample_sharpe):
    """Simple overfitting detection"""
    ratio = in_sample_sharpe / out_sample_sharpe if out_sample_sharpe != 0 else np.inf
    
    if ratio < 1.0:
        return "No overfitting (surprising)"
    elif ratio < 1.5:
        return "Healthy (minimal degradation)"
    elif ratio < 2.0:
        return "Moderate overfitting"
    else:
        return "Severe overfitting"

# Generate synthetic data
def generate_market_data(n_days=1500, n_assets=3, seed=42):
    """Generate synthetic price data"""
    np.random.seed(seed)
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    data = {}
    for i in range(n_assets):
        trend = 0.0003 + np.random.uniform(-0.0002, 0.0002)
        vol = 0.015 + np.random.uniform(0, 0.01)
        
        returns = np.random.normal(trend, vol, n_days)
        prices = 100 * np.exp(returns.cumsum())
        
        data[f'Asset_{i+1}'] = prices
    
    return pd.DataFrame(data, index=dates)

# Scenario 1: Basic Backtest
print("\n" + "="*70)
print("SCENARIO 1: Basic Backtest with Transaction Costs")
print("="*70)

# Generate data
prices_df = generate_market_data(n_days=1000, n_assets=2)

# Simple MA crossover strategy
def generate_ma_signals(prices_df):
    signals = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
    
    for col in prices_df.columns:
        ma_fast = prices_df[col].rolling(window=50).mean()
        ma_slow = prices_df[col].rolling(window=200).mean()
        
        signals[col][ma_fast > ma_slow] = 1
        signals[col][ma_fast < ma_slow] = -1
    
    return signals

signals = generate_ma_signals(prices_df)

# Backtest
engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
results = engine.run_backtest(signals, prices_df)

print(f"\nStrategy Performance:")
print(f"  Total Return: {results['total_return']:.2%}")
print(f"  CAGR: {results['cagr']:.2%}")
print(f"  Volatility: {results['volatility']:.2%}")
print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
print(f"  Max DD Duration: {results['max_dd_duration']:.0f} days")
print(f"  Number of Trades: {results['num_trades']}")

# Buy and hold comparison
bh_value = (prices_df.mean(axis=1) / prices_df.mean(axis=1).iloc[0]) * 100000
bh_return = (bh_value.iloc[-1] / 100000) - 1

print(f"\nBuy & Hold (equal weight):")
print(f"  Total Return: {bh_return:.2%}")

# Scenario 2: Transaction Cost Sensitivity
print("\n" + "="*70)
print("SCENARIO 2: Transaction Cost Sensitivity Analysis")
print("="*70)

cost_levels = [0.0, 0.0005, 0.001, 0.002, 0.005]

print(f"\n{'Cost (bps)':<15} {'Sharpe':<12} {'Total Return':<15}")
print("-" * 42)

for cost in cost_levels:
    engine_cost = BacktestEngine(initial_capital=100000, commission=cost, slippage=cost/2)
    results_cost = engine_cost.run_backtest(signals, prices_df)
    
    print(f"{cost*10000:<15.1f} {results_cost['sharpe_ratio']:<12.3f} {results_cost['total_return']:<15.2%}")

print(f"\nTransaction costs significantly impact high-frequency strategies")

# Scenario 3: In-Sample vs Out-of-Sample
print("\n" + "="*70)
print("SCENARIO 3: In-Sample vs Out-of-Sample Validation")
print("="*70)

# Split data
split_point = int(len(prices_df) * 0.7)

prices_train = prices_df.iloc[:split_point]
prices_test = prices_df.iloc[split_point:]

# Train (in-sample)
signals_train = generate_ma_signals(prices_train)
engine_train = BacktestEngine()
results_train = engine_train.run_backtest(signals_train, prices_train)

# Test (out-of-sample)
signals_test = generate_ma_signals(prices_test)
engine_test = BacktestEngine()
results_test = engine_test.run_backtest(signals_test, prices_test)

print(f"\nIn-Sample (Training, 70% of data):")
print(f"  Sharpe Ratio: {results_train['sharpe_ratio']:.3f}")
print(f"  Total Return: {results_train['total_return']:.2%}")

print(f"\nOut-of-Sample (Testing, 30% of data):")
print(f"  Sharpe Ratio: {results_test['sharpe_ratio']:.3f}")
print(f"  Total Return: {results_test['total_return']:.2%}")

overfitting_assessment = detect_overfitting(
    results_train['sharpe_ratio'], 
    results_test['sharpe_ratio']
)

print(f"\nOverfitting Assessment: {overfitting_assessment}")
print(f"  Ratio (In/Out): {results_train['sharpe_ratio']/results_test['sharpe_ratio'] if results_test['sharpe_ratio'] != 0 else 'inf':.2f}")

# Scenario 4: Walk-Forward Analysis
print("\n" + "="*70)
print("SCENARIO 4: Walk-Forward Analysis")
print("="*70)

# Use single asset for simplicity
prices_single = prices_df['Asset_1']

wfa = WalkForwardAnalysis(train_period=504, test_period=126, step=126)
wf_results = wfa.run_walk_forward(prices_single)

print(f"\nWalk-Forward Windows: {len(wf_results)}")
print(f"\n{'Window':<10} {'Train Sharpe':<15} {'Test Sharpe':<15} {'Best Params':<20}")
print("-" * 60)

for i, result in enumerate(wf_results[:5]):  # Show first 5
    params_str = f"({result['best_params']['fast']}/{result['best_params']['slow']})"
    print(f"{i+1:<10} {result['train_sharpe']:<15.3f} {result['test_sharpe']:<15.3f} {params_str:<20}")

# Aggregate out-of-sample performance
all_test_returns = pd.concat([r['test_returns'] for r in wf_results])
aggregate_sharpe = (all_test_returns.mean() * 252) / (all_test_returns.std() * np.sqrt(252))

print(f"\nAggregate Out-of-Sample Sharpe: {aggregate_sharpe:.3f}")

# Scenario 5: Monte Carlo Validation
print("\n" + "="*70)
print("SCENARIO 5: Monte Carlo Bootstrap Validation")
print("="*70)

# Use strategy returns
strategy_returns = results['returns']

mc_validator = MonteCarloValidator(strategy_returns, n_simulations=1000)
confidence_intervals, mc_results = mc_validator.get_confidence_intervals(confidence=0.95)

print(f"\n95% Confidence Intervals (10,000 bootstraps):")
print(f"  Sharpe Ratio: [{confidence_intervals['sharpe'][0]:.3f}, {confidence_intervals['sharpe'][1]:.3f}]")
print(f"  Total Return: [{confidence_intervals['return'][0]:.2%}, {confidence_intervals['return'][1]:.2%}]")
print(f"  Max Drawdown: [{confidence_intervals['max_dd'][0]:.2%}, {confidence_intervals['max_dd'][1]:.2%}]")

# Check if Sharpe CI includes zero
if confidence_intervals['sharpe'][0] > 0:
    print(f"\n✓ Strategy statistically significant (lower bound > 0)")
else:
    print(f"\n✗ Strategy NOT statistically significant (includes zero)")

# Scenario 6: Look-Ahead Bias Detection
print("\n" + "="*70)
print("SCENARIO 6: Look-Ahead Bias Detection")
print("="*70)

# Correct implementation (no look-ahead)
prices_single = prices_df['Asset_1']
ma_50 = prices_single.rolling(window=50).mean()
ma_200 = prices_single.rolling(window=200).mean()

signals_correct = pd.Series(0, index=prices_single.index)
signals_correct[ma_50 > ma_200] = 1
signals_correct[ma_50 < ma_200] = -1

returns_correct = signals_correct.shift(1) * prices_single.pct_change()
sharpe_correct = (returns_correct.mean() * 252) / (returns_correct.std() * np.sqrt(252))

# Incorrect implementation (look-ahead bias)
signals_wrong = pd.Series(0, index=prices_single.index)
signals_wrong[ma_50 > ma_200] = 1
signals_wrong[ma_50 < ma_200] = -1

# Execute same day (uses current close for entry)
returns_wrong = signals_wrong * prices_single.pct_change()
sharpe_wrong = (returns_wrong.mean() * 252) / (returns_wrong.std() * np.sqrt(252))

print(f"\nCorrect (execute next day):")
print(f"  Sharpe Ratio: {sharpe_correct:.3f}")

print(f"\nWith Look-Ahead Bias (execute same day):")
print(f"  Sharpe Ratio: {sharpe_wrong:.3f}")
print(f"  Artificial boost: {(sharpe_wrong/sharpe_correct - 1)*100:.1f}%")

print(f"\nLook-ahead bias inflates backtested performance!")

# Visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Plot 1: Equity curve with drawdowns
ax = axes[0, 0]
portfolio_series = results['portfolio_series']
cumulative = portfolio_series / portfolio_series.iloc[0]

ax.plot(cumulative.index, cumulative.values, 'b-', linewidth=2, label='Strategy')

# Drawdown shading
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max

ax.fill_between(cumulative.index, cumulative.values, running_max.values, 
                where=(cumulative < running_max), alpha=0.3, color='red', label='Drawdown')

ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (Normalized)')
ax.set_title('Equity Curve with Drawdowns')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Underwater (drawdown) plot
ax = axes[0, 1]
ax.fill_between(drawdown.index, 0, drawdown.values*100, color='red', alpha=0.5)
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title(f'Underwater Plot (Max DD: {results["max_drawdown"]:.2%})')
ax.grid(alpha=0.3)

# Plot 3: Return distribution
ax = axes[1, 0]
returns_clean = results['returns'].dropna()

ax.hist(returns_clean * 100, bins=50, density=True, alpha=0.7, edgecolor='black', color='skyblue')

# Overlay normal distribution
mu, sigma = returns_clean.mean(), returns_clean.std()
x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
ax.plot(x * 100, stats.norm.pdf(x, mu, sigma) / 100, 'r-', linewidth=2, label='Normal')

ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Density')
ax.set_title('Return Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Rolling Sharpe ratio
ax = axes[1, 1]
rolling_window = 252  # 1 year
rolling_sharpe = returns_clean.rolling(window=rolling_window).apply(
    lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252))
)

ax.plot(rolling_sharpe.index, rolling_sharpe.values, 'g-', linewidth=2)
ax.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(results['sharpe_ratio'], color='b', linestyle='--', linewidth=1, 
           alpha=0.5, label=f'Overall: {results["sharpe_ratio"]:.2f}')
ax.set_xlabel('Date')
ax.set_ylabel('Rolling 252-day Sharpe')
ax.set_title('Rolling Sharpe Ratio (Strategy Stability)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Monte Carlo distribution
ax = axes[2, 0]
ax.hist(mc_results['sharpe'], bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(results['sharpe_ratio'], color='r', linestyle='--', linewidth=2, label='Actual')
ax.axvline(confidence_intervals['sharpe'][0], color='g', linestyle=':', linewidth=1.5, 
           label=f"95% CI")
ax.axvline(confidence_intervals['sharpe'][1], color='g', linestyle=':', linewidth=1.5)
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Density')
ax.set_title('Monte Carlo Bootstrap: Sharpe Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Transaction cost impact
ax = axes[2, 1]
costs_plot = [0, 5, 10, 20, 50]
sharpes_plot = []

for cost_bps in costs_plot:
    cost_frac = cost_bps / 10000
    eng = BacktestEngine(initial_capital=100000, commission=cost_frac, slippage=cost_frac/2)
    res = eng.run_backtest(signals, prices_df)
    sharpes_plot.append(res['sharpe_ratio'])

ax.plot(costs_plot, sharpes_plot, 'ro-', linewidth=2, markersize=8)
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax.set_xlabel('Transaction Cost (bps round-trip)')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Transaction Cost Sensitivity')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Survivorship Bias Impact:** Simulate by removing worst 20% performers from backtest universe. Quantify return inflation. How to detect in practice?

2. **Data Mining Correction:** Test 100 random MA combinations, report best. Apply Bonferroni correction to p-value. Calculate haircut to Sharpe ratio.

3. **Combinatorial Purged CV:** Implement De Prado's combinatorial cross-validation with purging and embargo. Compare to simple train-test split.

4. **Regime-Dependent Validation:** Detect bull/bear regimes (HMM), evaluate strategy performance in each. Does strategy work in all regimes?

5. **Optimal Stopping for Walk-Forward:** Determine optimal train/test window sizes. Plot Sharpe vs window length. Trade-off between data and stability?

## 7. Key References
- [Pardo, The Evaluation and Optimization of Trading Strategies (2nd Edition)](https://www.wiley.com/en-us/The+Evaluation+and+Optimization+of+Trading+Strategies%2C+2nd+Edition-p-9780470128015) - comprehensive backtesting methodology
- [Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) - correcting for multiple testing bias
- [López de Prado, Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - cross-validation, overfitting detection, meta-labeling

---
**Status:** Strategy validation methodology | **Complements:** Trading Signals, Risk Management, Execution Algorithms, Statistical Testing, Machine Learning
