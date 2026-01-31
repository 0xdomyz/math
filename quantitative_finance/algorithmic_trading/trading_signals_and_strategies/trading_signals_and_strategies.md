# Trading Signals and Strategies

## 1. Concept Skeleton
**Definition:** Systematic rules combining technical indicators, statistical patterns, and market data to generate buy/sell decisions for algorithmic trading  
**Purpose:** Capture alpha from market inefficiencies; automate execution; eliminate emotional bias; scale trading across instruments  
**Prerequisites:** Time series analysis, probability/statistics, market microstructure, backtesting methodology, risk management

## 2. Comparative Framing
| Strategy Type | Trend Following | Mean Reversion | Statistical Arbitrage | Market Making | Momentum |
|---------------|----------------|----------------|----------------------|---------------|----------|
| **Holding Period** | Days-weeks | Minutes-days | Seconds-days | Seconds-minutes | Days-weeks |
| **Market View** | Directional | Range-bound | Neutral (pairs) | Neutral | Directional |
| **Volatility** | Prefers trending | Prefers stable | Low correlation risk | Benefits from flow | Moderate |
| **Win Rate** | Low (~40%) | High (~60%) | Moderate (~55%) | High (~70%) | Moderate (~50%) |
| **Risk/Reward** | Large wins | Small wins | Small consistent | Spread capture | Medium wins |

| Signal Source | Technical Indicators | Fundamental Data | Order Flow | Machine Learning | Alternative Data |
|---------------|---------------------|------------------|------------|------------------|------------------|
| **Frequency** | High (real-time) | Low (quarterly) | Very high (tick) | Medium | Varies |
| **Latency Sensitivity** | Moderate | Low | Extreme | Moderate | Low-Medium |
| **Noise Level** | High | Moderate | Very high | Depends | High |
| **Alpha Decay** | Fast | Slow | Very fast | Fast | Medium |

## 3. Examples + Counterexamples

**Simple Example:**  
Moving average crossover: Buy when SMA(50) crosses above SMA(200), sell on reverse. Captures trend changes but lags (many false signals in choppy markets).

**Perfect Fit:**  
Pairs trading on cointegrated stocks: Long underperformer, short outperformer when spread exceeds 2σ. Reverts to mean due to fundamental linkage (e.g., Coca-Cola/PepsiCo).

**Momentum Strategy:**  
Buy top decile performers (past 12 months), short bottom decile. Exploits behavioral bias (herding, underreaction). Jegadeesh-Titman effect: ~1% monthly alpha historically.

**Mean Reversion (Intraday):**  
Bollinger Bands: Sell when price hits upper band (2σ above 20-day MA), buy at lower band. Works in range-bound markets, fails in strong trends.

**Statistical Arbitrage:**  
PCA-based factor model: Identify common factors, trade residual mean reversion. Market-neutral by construction. Requires low transaction costs.

**Poor Fit:**  
News sentiment trading without low latency: Alpha decays in milliseconds post-release. Retail traders arrive too late (professional HFT dominates).

## 4. Layer Breakdown
```
Trading Signals & Strategies Framework:

├─ Technical Analysis Signals:
│  ├─ Trend-Following Indicators:
│  │   ├─ Moving Averages:
│  │   │   ├─ Simple MA (SMA): Arithmetic mean over window
│  │   │   ├─ Exponential MA (EMA): Weighted recent data
│  │   │   │   EMA_t = α×Price_t + (1-α)×EMA_{t-1}
│  │   │   │   α = 2/(N+1), N = period
│  │   │   ├─ Crossover Signals:
│  │   │   │   Golden Cross: SMA(50) > SMA(200) → bullish
│  │   │   │   Death Cross: SMA(50) < SMA(200) → bearish
│  │   │   └─ Problems: Lag, whipsaws in sideways markets
│  │   ├─ MACD (Moving Average Convergence Divergence):
│  │   │   ├─ MACD Line: EMA(12) - EMA(26)
│  │   │   ├─ Signal Line: EMA(9) of MACD
│  │   │   ├─ Histogram: MACD - Signal
│  │   │   └─ Signals: MACD crosses signal (bullish/bearish)
│  │   ├─ ADX (Average Directional Index):
│  │   │   ├─ Measures trend strength (not direction)
│  │   │   ├─ ADX > 25: Strong trend
│  │   │   ├─ ADX < 20: Weak trend (choppy)
│  │   │   └─ Use with +DI/-DI for direction
│  │   └─ Parabolic SAR:
│  │       Stop-and-reverse points (trailing stop)
│  │       Good for trending markets
│  ├─ Mean Reversion Indicators:
│  │   ├─ Bollinger Bands:
│  │   │   ├─ Middle: SMA(20)
│  │   │   ├─ Upper: SMA(20) + 2×σ
│  │   │   ├─ Lower: SMA(20) - 2×σ
│  │   │   ├─ Signals:
│  │   │   │   Price touches upper → overbought → sell
│  │   │   │   Price touches lower → oversold → buy
│  │   │   ├─ Band Width: Volatility measure
│  │   │   │   Narrow → low vol → potential breakout
│  │   │   └─ %B: (Price - Lower) / (Upper - Lower)
│  │   ├─ RSI (Relative Strength Index):
│  │   │   ├─ RSI = 100 - 100/(1 + RS)
│  │   │   │   RS = Avg Gain / Avg Loss (14 periods)
│  │   │   ├─ Overbought: RSI > 70 → sell signal
│  │   │   ├─ Oversold: RSI < 30 → buy signal
│  │   │   ├─ Divergence:
│  │   │   │   Price makes new high, RSI doesn't → bearish
│  │   │   │   Price makes new low, RSI doesn't → bullish
│  │   │   └─ Centerline: RSI crosses 50 (momentum shift)
│  │   ├─ Stochastic Oscillator:
│  │   │   ├─ %K = 100×(Close - Low_N) / (High_N - Low_N)
│  │   │   ├─ %D = SMA(%K, 3) (signal line)
│  │   │   ├─ Overbought/oversold: Similar to RSI
│  │   │   └─ Fast vs Slow stochastic (smoothing)
│  │   └─ CCI (Commodity Channel Index):
│  │       Measures deviation from statistical mean
│  │       > +100: Overbought, < -100: Oversold
│  ├─ Volatility Indicators:
│  │   ├─ ATR (Average True Range):
│  │   │   ├─ TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
│  │   │   ├─ ATR = EMA(TR, 14)
│  │   │   └─ Use: Position sizing, stop-loss placement
│  │   ├─ Historical Volatility:
│  │   │   σ = std(log returns) × √252 (annualized)
│  │   ├─ Implied Volatility (from options):
│  │   │   VIX: 30-day implied vol of S&P 500
│  │   │   High IV → uncertain → reduce position size
│  │   └─ Bollinger Band Width:
│  │       BW = (Upper - Lower) / Middle
│  │       Volatility compression → potential breakout
│  └─ Volume Indicators:
│      ├─ Volume-Weighted Average Price (VWAP):
│      │   VWAP = Σ(Price × Volume) / Σ(Volume)
│      │   Intraday benchmark (institutional reference)
│      ├─ On-Balance Volume (OBV):
│      │   OBV_t = OBV_{t-1} + sign(Δ Price) × Volume
│      │   Confirms trends (volume precedes price)
│      ├─ Accumulation/Distribution:
│      │   Measures buying/selling pressure
│      │   A/D = Σ[((Close-Low)-(High-Close))/(High-Low) × Volume]
│      └─ Chaikin Money Flow:
│          Money flow weighted by volume
│          > 0: Buying pressure, < 0: Selling pressure
├─ Quantitative Strategies:
│  ├─ Momentum Strategies:
│  │   ├─ Cross-Sectional Momentum:
│  │   │   ├─ Rank assets by past returns (e.g., 12 months)
│  │   │   ├─ Long top decile, short bottom decile
│  │   │   ├─ Rebalance monthly
│  │   │   ├─ Jegadeesh-Titman (1993): ~1% monthly alpha
│  │   │   └─ Risk: Momentum crashes (2009: -73%)
│  │   ├─ Time-Series Momentum:
│  │   │   ├─ Long if asset > own MA, else short/flat
│  │   │   ├─ Applied to each asset independently
│  │   │   └─ Moskowitz et al. (2012): Works across all assets
│  │   ├─ 52-Week High Strategy:
│  │   │   Buy when price near 52-week high (breakout)
│  │   │   Anchoring bias: Investors anchor to recent highs
│  │   └─ Acceleration (Rate-of-Change):
│  │       ROC = (Price_t / Price_{t-n} - 1) × 100
│  │       Positive acceleration → buy
│  ├─ Mean Reversion:
│  │   ├─ Pairs Trading:
│  │   │   ├─ Cointegration-Based:
│  │   │   │   ├─ Test pair for cointegration (Engle-Granger, Johansen)
│  │   │   │   ├─ Spread = Price_A - β×Price_B (β from regression)
│  │   │   │   ├─ Z-score = (Spread - μ) / σ
│  │   │   │   ├─ Entry: |Z| > 2 (spread exceeds 2σ)
│  │   │   │   ├─ Exit: Z crosses 0 (mean reversion)
│  │   │   │   └─ Stop: |Z| > 4 (cointegration breakdown)
│  │   │   ├─ Correlation-Based (simpler, less robust):
│  │   │   │   Use rolling correlation, trade spread
│  │   │   ├─ Distance Method:
│  │   │   │   Normalize prices, minimize squared distance
│  │   │   └─ Sector Neutral:
│  │   │       Pair within same industry (reduce beta)
│  │   ├─ ETF Arbitrage:
│  │   │   Trade ETF vs basket of constituents
│  │   │   Exploit creation/redemption mechanism
│  │   ├─ Intraday Mean Reversion:
│  │   │   ├─ Opening Range Breakout:
│  │   │   │   Fade moves beyond first 30-min range
│  │   │   ├─ VWAP Reversion:
│  │   │   │   Price deviates >1% from VWAP → revert
│  │   │   └─ End-of-Day Effect:
│  │   │       Institutional window dressing, rebalancing
│  │   └─ Index Reversion:
│  │       Mean revert index vs constituent basket
│  │       Exploit stale pricing, rebalancing flows
│  ├─ Statistical Arbitrage:
│  │   ├─ PCA-Based Stat Arb:
│  │   │   ├─ Decompose returns into principal components
│  │   │   ├─ First few PCs: Common factors (market, sector)
│  │   │   ├─ Residuals: Idiosyncratic (mean-reverting)
│  │   │   ├─ Trade residual deviations (Z-score)
│  │   │   └─ Market-neutral: Orthogonal to factors
│  │   ├─ Cointegration Portfolios:
│  │   │   ├─ Find cointegrated basket (not just pairs)
│  │   │   ├─ Johansen test for multiple assets
│  │   │   └─ Trade portfolio spread mean reversion
│  │   ├─ Market Microstructure:
│  │   │   ├─ Quote Fading: Front-run stale quotes
│  │   │   ├─ Latency Arbitrage: Race to fast exchange
│  │   │   └─ Requires sub-millisecond execution
│  │   └─ Index Arbitrage:
│  │       Futures vs spot index basket
│  │       Exploit mispricing (cost of carry)
│  ├─ Breakout Strategies:
│  │   ├─ Channel Breakout:
│  │   │   ├─ Donchian Channel: N-day high/low
│  │   │   ├─ Entry: Price breaks N-day high → long
│  │   │   │         Price breaks N-day low → short
│  │   │   ├─ Turtle Trading: N=20 (entry), N=10 (exit)
│  │   │   └─ False breakouts: Use volume confirmation
│  │   ├─ Volatility Breakout:
│  │   │   Enter when price moves > X×ATR from open
│  │   │   Expansion from low volatility compression
│  │   ├─ News-Driven Breakout:
│  │   │   Earnings surprise, analyst upgrades
│  │   │   Requires NLP for sentiment analysis
│  │   └─ Pattern Recognition:
│  │       Head-and-shoulders, triangles (dubious edge)
│  └─ Market Making / Liquidity Provision:
│      ├─ Bid-Ask Spread Capture:
│      │   ├─ Post limit orders on both sides
│      │   ├─ Earn half-spread on each round-trip
│      │   ├─ Inventory risk: Adverse selection
│      │   └─ Requires high fill rate, low adverse selection
│      ├─ Inventory Management:
│      │   ├─ Target zero inventory (risk-neutral)
│      │   ├─ Adjust quotes based on position
│      │   │   Long inventory → widen ask, tighten bid
│      │   └─ Almgren-Chriss framework for optimal
│      ├─ Adverse Selection Management:
│      │   ├─ Detect informed traders (toxicity)
│      │   ├─ Widen spreads when volatility spikes
│      │   └─ Use order flow imbalance indicators
│      └─ High-Frequency Market Making:
│          Sub-millisecond latency critical
│          Co-location, FPGA acceleration
├─ Machine Learning Signals:
│  ├─ Supervised Learning:
│  │   ├─ Classification (Direction):
│  │   │   ├─ Logistic Regression:
│  │   │   │   Predict P(Up) from features (indicators)
│  │   │   ├─ Random Forest:
│  │   │   │   Ensemble of decision trees
│  │   │   │   Handles non-linearity, feature importance
│  │   │   ├─ Gradient Boosting (XGBoost):
│  │   │   │   Sequential weak learners
│  │   │   │   Often best for structured data
│  │   │   ├─ Neural Networks:
│  │   │   │   Multi-layer perceptron (MLP)
│  │   │   │   Deep learning for complex patterns
│  │   │   └─ SVM (Support Vector Machine):
│  │   │       Kernel methods for non-linear boundaries
│  │   ├─ Regression (Returns):
│  │   │   ├─ Linear Models: Ridge, Lasso, Elastic Net
│  │   │   ├─ Non-Linear: Kernel regression, MARS
│  │   │   └─ Ensemble: Stacking multiple models
│  │   ├─ Feature Engineering:
│  │   │   ├─ Lagged returns (1-day, 5-day, 20-day)
│  │   │   ├─ Technical indicators (RSI, MACD, etc.)
│  │   │   ├─ Volume features (turnover, VWAP deviation)
│  │   │   ├─ Volatility (historical, GARCH forecast)
│  │   │   ├─ Order book imbalance
│  │   │   ├─ Sentiment (news, social media)
│  │   │   └─ Cross-sectional (rank, Z-score)
│  │   ├─ Overfitting Prevention:
│  │   │   ├─ Cross-validation (walk-forward)
│  │   │   ├─ Regularization (L1, L2 penalty)
│  │   │   ├─ Early stopping (validation loss)
│  │   │   ├─ Feature selection (reduce dimensionality)
│  │   │   └─ Out-of-sample testing (crucial)
│  │   └─ Time Series Considerations:
│  │       ├─ Look-ahead bias: Use only past data
│  │       ├─ Non-stationarity: Retrain frequently
│  │       └─ Autocorrelation: Purging, embargo
│  ├─ Unsupervised Learning:
│  │   ├─ Clustering (Regime Detection):
│  │   │   ├─ K-means: Group market states
│  │   │   ├─ Gaussian Mixture Models (GMM)
│  │   │   └─ Hidden Markov Models (HMM)
│  │   │       Detect bull/bear/sideways regimes
│  │   ├─ Dimensionality Reduction:
│  │   │   ├─ PCA: Extract common factors
│  │   │   ├─ Autoencoders: Non-linear compression
│  │   │   └─ t-SNE, UMAP: Visualization
│  │   └─ Anomaly Detection:
│  │       Detect unusual patterns (flash crashes, etc.)
│  │       One-class SVM, isolation forests
│  ├─ Reinforcement Learning:
│  │   ├─ Q-Learning:
│  │   │   Learn optimal policy (buy/sell/hold)
│  │   │   State: Market features, position
│  │   │   Action: Trade decision
│  │   │   Reward: PnL, Sharpe ratio
│  │   ├─ Deep Q-Networks (DQN):
│  │   │   Neural network approximates Q-function
│  │   ├─ Policy Gradient (A3C, PPO):
│  │   │   Directly optimize trading policy
│  │   └─ Challenges:
│  │       Non-stationary environment, sparse rewards
│  │       Simulation vs live divergence
│  └─ NLP for News/Sentiment:
│      ├─ Sentiment Analysis:
│      │   ├─ Lexicon-Based: Count positive/negative words
│      │   ├─ ML-Based: Train on labeled data
│      │   └─ Transformer Models: BERT, FinBERT
│      ├─ Event Detection:
│      │   Earnings, M&A, regulatory filings
│      │   NER (Named Entity Recognition)
│      └─ Challenges:
│          Speed (alpha decays fast), sarcasm, context
├─ Strategy Construction:
│  ├─ Signal Generation:
│  │   ├─ Single Indicator: Simple, robust, low Sharpe
│  │   ├─ Composite Signals:
│  │   │   ├─ Weighted Average: Combine multiple indicators
│  │   │   ├─ Voting: Majority of signals agree
│  │   │   └─ Ensemble: ML combines raw features
│  │   ├─ Thresholds:
│  │   │   Binary: Cross MA (on/off)
│  │   │   Continuous: Proportional to signal strength
│  │   └─ Filtering:
│  │       Regime filter: Only trade in trending regimes
│  │       Volatility filter: Reduce size in high vol
│  ├─ Position Sizing:
│  │   ├─ Fixed Fractional:
│  │   │   Risk fixed % of capital per trade (e.g., 2%)
│  │   ├─ Kelly Criterion:
│  │   │   f* = (p×b - q) / b
│  │   │   p = win probability, b = win/loss ratio
│  │   │   q = 1-p, f* = fraction of capital
│  │   │   Often use fractional Kelly (0.25×f*)
│  │   ├─ Volatility-Adjusted:
│  │   │   Position ∝ 1/σ (target constant risk)
│  │   │   σ = ATR or historical volatility
│  │   ├─ Risk Parity:
│  │   │   Equal risk contribution per asset
│  │   └─ Optimization-Based:
│  │       Mean-variance, CVaR minimization
│  ├─ Portfolio Construction:
│  │   ├─ Long-Only: Simple, regulatory friendly
│  │   ├─ Long-Short: Market-neutral, double capacity
│  │   │   Dollar-neutral: Long = Short (dollars)
│  │   │   Beta-neutral: Long beta = Short beta
│  │   ├─ Sector Constraints:
│  │   │   Limit exposure to any sector (< 20%)
│  │   └─ Diversification:
│  │       Uncorrelated strategies, multiple timeframes
│  ├─ Risk Management:
│  │   ├─ Stop-Loss:
│  │   │   ├─ Fixed: X% below entry
│  │   │   ├─ Trailing: X% below max profit
│  │   │   ├─ Volatility-Based: 2×ATR
│  │   │   └─ Time-Based: Exit if no progress after N days
│  │   ├─ Take-Profit:
│  │   │   Target multiple of risk (2:1 reward:risk)
│  │   ├─ Maximum Drawdown Limit:
│  │   │   Halt trading if DD > 20% (circuit breaker)
│  │   ├─ Leverage Control:
│  │   │   Gross leverage < 2× (long + short)
│  │   │   Net leverage < 1× (long - short)
│  │   └─ Correlation Risk:
│  │       Monitor correlation between positions
│  │       Reduce if correlation > 0.7
│  └─ Execution:
│      ├─ Order Types:
│      │   ├─ Market: Immediate, high slippage
│      │   ├─ Limit: Price guarantee, no fill guarantee
│      │   ├─ Stop-Limit: Trigger, then limit order
│      │   └─ Iceberg: Hide size, prevent front-running
│      ├─ Execution Algorithms:
│      │   ├─ VWAP: Match volume profile
│      │   ├─ TWAP: Time-weighted (equal slices)
│      │   ├─ POV: Percentage of volume
│      │   └─ Implementation Shortfall: Minimize cost
│      └─ Slippage Management:
│          Trade liquid stocks, avoid market orders
│          Split large orders, use dark pools
├─ Strategy Evaluation:
│  ├─ Performance Metrics:
│  │   ├─ Return Measures:
│  │   │   ├─ Total Return: (End - Start) / Start
│  │   │   ├─ CAGR: [(End/Start)^(1/Years) - 1]
│  │   │   ├─ Alpha: Return above benchmark (risk-adjusted)
│  │   │   └─ Information Ratio: Alpha / Tracking Error
│  │   ├─ Risk Measures:
│  │   │   ├─ Volatility: std(returns) × √252
│  │   │   ├─ Sharpe Ratio: (Mean Return - Rf) / σ
│  │   │   ├─ Sortino Ratio: Return / Downside Deviation
│  │   │   ├─ Calmar Ratio: CAGR / Max Drawdown
│  │   │   └─ VaR, CVaR: Tail risk measures
│  │   ├─ Drawdown:
│  │   │   ├─ Max Drawdown: Peak to trough
│  │   │   ├─ Average Drawdown: Mean of all DDs
│  │   │   └─ Recovery Time: Days to recover peak
│  │   ├─ Win Rate:
│  │   │   % of profitable trades
│  │   │   Expectancy: WinRate×AvgWin - LossRate×AvgLoss
│  │   └─ Transaction Costs:
│  │       Turnover, commissions, slippage impact
│  ├─ Backtesting:
│  │   ├─ Data Quality:
│  │   │   ├─ Survivorship bias: Include delisted stocks
│  │   │   ├─ Adjusted prices: Splits, dividends
│  │   │   └─ Point-in-time: No look-ahead bias
│  │   ├─ Execution Assumptions:
│  │   │   ├─ Slippage: BPS per trade
│  │   │   ├─ Commission: Per share or % notional
│  │   │   ├─ Market impact: Proportional to size
│  │   │   └─ Fill probability: Limit orders may not fill
│  │   ├─ Walk-Forward Optimization:
│  │   │   ├─ Train on in-sample data
│  │   │   ├─ Test on out-of-sample period
│  │   │   ├─ Roll forward, repeat
│  │   │   └─ Avoid overfitting to single period
│  │   └─ Monte Carlo:
│  │       Resample trades to test robustness
│  │       Confidence intervals on Sharpe ratio
│  ├─ Statistical Tests:
│  │   ├─ T-Test: Is Sharpe > 0 significant?
│  │   │   t = Sharpe × √n (n = number of periods)
│  │   ├─ Multiple Testing Correction:
│  │   │   Bonferroni: Adjust p-value for many tests
│  │   │   Prevents data mining (false discoveries)
│  │   └─ Regime Analysis:
│  │       Performance in different market conditions
│  │       Bull, bear, high/low volatility
│  └─ Live Trading Considerations:
│      ├─ Paper Trading: Simulate without capital
│      ├─ Micro-Deployment: Small capital first
│      ├─ Monitoring: Real-time PnL, risk metrics
│      ├─ Decay: Alpha may degrade vs backtest
│      └─ Adaptation: Retrain models, update parameters
└─ Practical Considerations:
   ├─ Data Requirements:
   │   ├─ Historical Prices: EOD, intraday ticks
   │   ├─ Corporate Actions: Splits, dividends, delistings
   │   ├─ Fundamentals: Earnings, financials (if used)
   │   ├─ Alternative Data: Sentiment, satellite, web traffic
   │   └─ Quality Control: Outlier detection, missing data
   ├─ Technology Stack:
   │   ├─ Languages: Python (research), C++ (production)
   │   ├─ Libraries: pandas, NumPy, scikit-learn, backtrader
   │   ├─ Databases: TimescaleDB, InfluxDB (time series)
   │   ├─ Execution: Interactive Brokers API, FIX protocol
   │   └─ Infrastructure: AWS, low-latency networks
   ├─ Common Pitfalls:
   │   ├─ Overfitting: Too many parameters, curve-fitting
   │   ├─ Look-Ahead Bias: Using future data
   │   ├─ Survivorship Bias: Ignoring delisted stocks
   │   ├─ Data Mining: Testing too many strategies
   │   ├─ Ignoring Costs: Unrealistic transaction assumptions
   │   └─ Regime Change: Strategy works in one regime only
   ├─ Regulatory:
   │   ├─ SEC Rules: Pattern day trader (>3 trades/week)
   │   ├─ Market Manipulation: Spoofing, layering illegal
   │   ├─ Short Selling: Locate requirements, uptick rule
   │   └─ Reporting: Large position disclosure (13F)
   └─ Alpha Decay:
      ├─ Crowding: Popular strategies lose edge
      ├─ Arbitrage: Profits attract capital → competes away
      ├─ Market Evolution: Behavior changes, correlations shift
      └─ Adaptation: Continuously research new signals
```

**Interaction:** Generate signal (indicators, ML) → Position sizing (risk management) → Execution (algorithms) → Monitor performance → Adapt/refine strategy.

## 5. Mini-Project
Implement multiple trading strategies with backtesting and performance analysis:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("TRADING SIGNALS AND STRATEGIES")
print("="*70)

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, prices, transaction_cost=0.001):
        """
        prices: DataFrame with OHLCV data
        transaction_cost: Round-trip cost as fraction (0.001 = 10 bps)
        """
        self.prices = prices
        self.tc = transaction_cost
        self.signals = None
        self.positions = None
        self.returns = None
        
    def calculate_returns(self, signals):
        """Calculate strategy returns from signals"""
        # Returns: Log returns
        returns = np.log(self.prices['Close'] / self.prices['Close'].shift(1))
        
        # Position: 1 (long), -1 (short), 0 (flat)
        positions = signals.shift(1)  # Trade on next day
        
        # Strategy returns
        strategy_returns = positions * returns
        
        # Transaction costs: Apply when position changes
        position_changes = positions.diff().abs()
        costs = position_changes * self.tc
        
        strategy_returns = strategy_returns - costs
        
        self.returns = strategy_returns.fillna(0)
        self.positions = positions
        
        return strategy_returns
    
    def performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.returns is None:
            return {}
        
        # Remove NaN
        returns = self.returns.dropna()
        
        # Total return
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized metrics (assuming daily data, 252 trading days)
        n_days = len(returns)
        n_years = n_days / 252
        
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming Rf = 0 for simplicity)
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'Num Trades': (self.positions.diff() != 0).sum()
        }

class MovingAverageCrossover(TradingStrategy):
    """Trend-following: MA crossover strategy"""
    
    def generate_signals(self, fast=50, slow=200):
        """Golden cross / Death cross"""
        prices = self.prices['Close']
        
        # Moving averages
        ma_fast = prices.rolling(window=fast).mean()
        ma_slow = prices.rolling(window=slow).mean()
        
        # Signals: 1 when fast > slow, -1 when fast < slow
        signals = pd.Series(0, index=prices.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1
        
        self.signals = signals
        self.calculate_returns(signals)
        
        return signals

class BollingerBandsMeanReversion(TradingStrategy):
    """Mean reversion: Bollinger Bands"""
    
    def generate_signals(self, window=20, num_std=2):
        """Buy at lower band, sell at upper band"""
        prices = self.prices['Close']
        
        # Bollinger Bands
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = ma + num_std * std
        lower_band = ma - num_std * std
        
        # Signals
        signals = pd.Series(0, index=prices.index)
        
        # Buy when price touches lower band
        signals[prices < lower_band] = 1
        
        # Sell when price touches upper band
        signals[prices > upper_band] = -1
        
        # Exit when price crosses MA
        signals[prices > ma] = signals[prices > ma].where(signals.shift(1) == 1, 0)
        signals[prices < ma] = signals[prices < ma].where(signals.shift(1) == -1, 0)
        
        # Forward-fill signals (maintain position)
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        self.signals = signals
        self.calculate_returns(signals)
        
        return signals

class RSIMomentum(TradingStrategy):
    """RSI-based momentum strategy"""
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, period=14, oversold=30, overbought=70):
        """Buy when oversold, sell when overbought"""
        prices = self.prices['Close']
        
        rsi = self.calculate_rsi(prices, period)
        
        # Signals
        signals = pd.Series(0, index=prices.index)
        
        # Buy when RSI crosses above oversold
        signals[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1
        
        # Sell when RSI crosses below overbought  
        signals[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1
        
        # Forward-fill
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        self.signals = signals
        self.calculate_returns(signals)
        
        return signals

class PairsTradingStrategy:
    """Statistical arbitrage: Pairs trading"""
    
    def __init__(self, price_A, price_B, transaction_cost=0.001):
        self.price_A = price_A
        self.price_B = price_B
        self.tc = transaction_cost
        
    def calculate_spread(self, window=60):
        """Calculate spread and z-score"""
        # Hedge ratio from rolling regression
        # Simple version: Use rolling correlation and std ratio
        
        # Normalize prices
        norm_A = self.price_A / self.price_A.iloc[0]
        norm_B = self.price_B / self.price_B.iloc[0]
        
        # Spread (simplified: equal-weighted)
        spread = norm_A - norm_B
        
        # Z-score
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        
        z_score = (spread - spread_mean) / spread_std
        
        return spread, z_score
    
    def generate_signals(self, window=60, entry_z=2.0, exit_z=0.5):
        """
        Entry: |z| > entry_z
        Exit: |z| < exit_z
        """
        spread, z_score = self.calculate_spread(window)
        
        # Signals for the pair
        signals = pd.Series(0, index=z_score.index)
        
        # Long spread (long A, short B) when z < -entry_z
        signals[z_score < -entry_z] = 1
        
        # Short spread (short A, long B) when z > entry_z
        signals[z_score > entry_z] = -1
        
        # Exit when mean-reverts
        signals[abs(z_score) < exit_z] = 0
        
        # Forward-fill positions
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        # Calculate returns
        returns_A = np.log(self.price_A / self.price_A.shift(1))
        returns_B = np.log(self.price_B / self.price_B.shift(1))
        
        # Spread return: Long A, Short B (when signal = 1)
        spread_return = returns_A - returns_B
        
        positions = signals.shift(1).fillna(0)
        strategy_returns = positions * spread_return
        
        # Transaction costs
        position_changes = positions.diff().abs()
        costs = position_changes * self.tc * 2  # Two legs
        
        strategy_returns = strategy_returns - costs
        
        self.returns = strategy_returns.fillna(0)
        self.positions = positions
        self.z_score = z_score
        
        return signals

class MomentumPortfolio:
    """Cross-sectional momentum across multiple assets"""
    
    def __init__(self, prices_dict, transaction_cost=0.001):
        """
        prices_dict: Dictionary of {asset: price_series}
        """
        self.prices = pd.DataFrame(prices_dict)
        self.tc = transaction_cost
        
    def generate_signals(self, lookback=252, rebalance_freq=21, long_pct=0.3, short_pct=0.3):
        """
        Rank by past returns, long top, short bottom
        lookback: Momentum calculation period (252 = 1 year)
        rebalance_freq: Days between rebalancing (21 = monthly)
        long_pct: Fraction to long (0.3 = top 30%)
        short_pct: Fraction to short (0.3 = bottom 30%)
        """
        # Calculate past returns
        past_returns = self.prices.pct_change(lookback)
        
        # Rank assets
        ranks = past_returns.rank(axis=1, pct=True)
        
        # Signals
        signals = pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)
        
        # Long top performers
        signals[ranks >= (1 - long_pct)] = 1
        
        # Short bottom performers
        signals[ranks <= short_pct] = -1
        
        # Rebalance at specified frequency
        rebalance_dates = range(lookback, len(self.prices), rebalance_freq)
        
        for i in range(len(signals)):
            if i not in rebalance_dates and i > 0:
                signals.iloc[i] = signals.iloc[i-1]
        
        # Calculate returns
        returns = np.log(self.prices / self.prices.shift(1))
        
        # Equal-weight within long and short
        n_long = (signals == 1).sum(axis=1)
        n_short = (signals == -1).sum(axis=1)
        
        # Normalize weights
        weights = signals.copy()
        weights[signals == 1] = weights[signals == 1].div(n_long, axis=0)
        weights[signals == -1] = weights[signals == -1].div(n_short, axis=0)
        
        # Portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        
        # Transaction costs
        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * self.tc
        
        portfolio_returns = portfolio_returns - costs
        
        self.returns = portfolio_returns.fillna(0)
        self.signals = signals
        
        return signals

# Generate synthetic price data for demonstration
def generate_price_data(n_days=1000, trend=0.0001, volatility=0.02, seed=42):
    """Generate synthetic stock price with GBM"""
    np.random.seed(seed)
    
    returns = np.random.normal(trend, volatility, n_days)
    prices = 100 * np.exp(returns.cumsum())
    
    # Create OHLCV dataframe
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, n_days)
    })
    
    df = df.set_index('Date')
    
    return df

# Scenario 1: Moving Average Crossover
print("\n" + "="*70)
print("SCENARIO 1: Moving Average Crossover (Trend Following)")
print("="*70)

# Generate trending market
prices_trend = generate_price_data(n_days=1000, trend=0.0003, volatility=0.015, seed=42)

ma_strategy = MovingAverageCrossover(prices_trend, transaction_cost=0.001)
signals_ma = ma_strategy.generate_signals(fast=50, slow=200)

metrics_ma = ma_strategy.performance_metrics()

print(f"\nMoving Average Crossover (50/200):")
print(f"  Total Return: {metrics_ma['Total Return']:.2%}")
print(f"  CAGR: {metrics_ma['CAGR']:.2%}")
print(f"  Sharpe Ratio: {metrics_ma['Sharpe Ratio']:.3f}")
print(f"  Max Drawdown: {metrics_ma['Max Drawdown']:.2%}")
print(f"  Win Rate: {metrics_ma['Win Rate']:.2%}")
print(f"  Number of Trades: {metrics_ma['Num Trades']:.0f}")

# Buy and hold comparison
bh_returns = np.log(prices_trend['Close'] / prices_trend['Close'].shift(1)).fillna(0)
bh_cumulative = (1 + bh_returns).cumprod()
bh_total_return = bh_cumulative.iloc[-1] - 1

print(f"\nBuy & Hold:")
print(f"  Total Return: {bh_total_return:.2%}")

# Scenario 2: Bollinger Bands Mean Reversion
print("\n" + "="*70)
print("SCENARIO 2: Bollinger Bands Mean Reversion")
print("="*70)

# Generate range-bound market
prices_range = generate_price_data(n_days=1000, trend=0, volatility=0.02, seed=123)

bb_strategy = BollingerBandsMeanReversion(prices_range, transaction_cost=0.001)
signals_bb = bb_strategy.generate_signals(window=20, num_std=2)

metrics_bb = bb_strategy.performance_metrics()

print(f"\nBollinger Bands (20-day, 2σ):")
print(f"  Total Return: {metrics_bb['Total Return']:.2%}")
print(f"  CAGR: {metrics_bb['CAGR']:.2%}")
print(f"  Sharpe Ratio: {metrics_bb['Sharpe Ratio']:.3f}")
print(f"  Max Drawdown: {metrics_bb['Max Drawdown']:.2%}")
print(f"  Win Rate: {metrics_bb['Win Rate']:.2%}")
print(f"  Number of Trades: {metrics_bb['Num Trades']:.0f}")

# Scenario 3: RSI Momentum
print("\n" + "="*70)
print("SCENARIO 3: RSI Momentum Strategy")
print("="*70)

rsi_strategy = RSIMomentum(prices_trend, transaction_cost=0.001)
signals_rsi = rsi_strategy.generate_signals(period=14, oversold=30, overbought=70)

metrics_rsi = rsi_strategy.performance_metrics()

print(f"\nRSI Strategy (14-period, 30/70 thresholds):")
print(f"  Total Return: {metrics_rsi['Total Return']:.2%}")
print(f"  CAGR: {metrics_rsi['CAGR']:.2%}")
print(f"  Sharpe Ratio: {metrics_rsi['Sharpe Ratio']:.3f}")
print(f"  Max Drawdown: {metrics_rsi['Max Drawdown']:.2%}")
print(f"  Win Rate: {metrics_rsi['Win Rate']:.2%}")
print(f"  Number of Trades: {metrics_rsi['Num Trades']:.0f}")

# Scenario 4: Pairs Trading
print("\n" + "="*70)
print("SCENARIO 4: Pairs Trading (Statistical Arbitrage)")
print("="*70)

# Generate cointegrated pair
np.random.seed(42)
n_days = 1000
common_trend = np.cumsum(np.random.normal(0, 0.01, n_days))

price_A = 100 + common_trend + np.cumsum(np.random.normal(0, 0.005, n_days))
price_B = 95 + common_trend + np.cumsum(np.random.normal(0, 0.005, n_days))

price_A = pd.Series(price_A, index=pd.date_range('2020-01-01', periods=n_days))
price_B = pd.Series(price_B, index=pd.date_range('2020-01-01', periods=n_days))

pairs_strategy = PairsTradingStrategy(price_A, price_B, transaction_cost=0.001)
signals_pairs = pairs_strategy.generate_signals(window=60, entry_z=2.0, exit_z=0.5)

# Performance
returns_pairs = pairs_strategy.returns.dropna()
sharpe_pairs = (returns_pairs.mean() * 252) / (returns_pairs.std() * np.sqrt(252))
total_return_pairs = (1 + returns_pairs).cumprod().iloc[-1] - 1

cumulative_pairs = (1 + returns_pairs).cumprod()
running_max_pairs = cumulative_pairs.expanding().max()
drawdown_pairs = (cumulative_pairs - running_max_pairs) / running_max_pairs
max_dd_pairs = drawdown_pairs.min()

print(f"\nPairs Trading (2σ entry, 0.5σ exit):")
print(f"  Total Return: {total_return_pairs:.2%}")
print(f"  Sharpe Ratio: {sharpe_pairs:.3f}")
print(f"  Max Drawdown: {max_dd_pairs:.2%}")
print(f"  Number of Trades: {(signals_pairs.diff() != 0).sum():.0f}")

# Scenario 5: Cross-Sectional Momentum
print("\n" + "="*70)
print("SCENARIO 5: Cross-Sectional Momentum Portfolio")
print("="*70)

# Generate multiple assets
n_assets = 10
prices_dict = {}

for i in range(n_assets):
    trend_i = np.random.uniform(-0.0001, 0.0005)
    vol_i = np.random.uniform(0.015, 0.025)
    prices_dict[f'Asset_{i+1}'] = generate_price_data(
        n_days=1000, trend=trend_i, volatility=vol_i, seed=100+i
    )['Close']

momentum_portfolio = MomentumPortfolio(prices_dict, transaction_cost=0.001)
signals_mom = momentum_portfolio.generate_signals(
    lookback=252, rebalance_freq=21, long_pct=0.3, short_pct=0.3
)

# Performance
returns_mom = momentum_portfolio.returns.dropna()
sharpe_mom = (returns_mom.mean() * 252) / (returns_mom.std() * np.sqrt(252))
total_return_mom = (1 + returns_mom).cumprod().iloc[-1] - 1

cumulative_mom = (1 + returns_mom).cumprod()
running_max_mom = cumulative_mom.expanding().max()
drawdown_mom = (cumulative_mom - running_max_mom) / running_max_mom
max_dd_mom = drawdown_mom.min()

print(f"\nCross-Sectional Momentum (12-month lookback, monthly rebalance):")
print(f"  Total Return: {total_return_mom:.2%}")
print(f"  Sharpe Ratio: {sharpe_mom:.3f}")
print(f"  Max Drawdown: {max_dd_mom:.2%}")
print(f"  Long/Short 30% of universe")

# Scenario 6: Strategy Comparison
print("\n" + "="*70)
print("SCENARIO 6: Strategy Comparison Summary")
print("="*70)

comparison = pd.DataFrame({
    'MA Crossover': [metrics_ma['Total Return'], metrics_ma['Sharpe Ratio'], 
                     metrics_ma['Max Drawdown'], metrics_ma['Win Rate']],
    'Bollinger Bands': [metrics_bb['Total Return'], metrics_bb['Sharpe Ratio'],
                        metrics_bb['Max Drawdown'], metrics_bb['Win Rate']],
    'RSI Momentum': [metrics_rsi['Total Return'], metrics_rsi['Sharpe Ratio'],
                     metrics_rsi['Max Drawdown'], metrics_rsi['Win Rate']],
    'Pairs Trading': [total_return_pairs, sharpe_pairs, max_dd_pairs, np.nan],
    'Cross-Sect Momentum': [total_return_mom, sharpe_mom, max_dd_mom, np.nan]
}, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'])

print("\n", comparison.T.to_string())

print(f"\n Best Sharpe Ratio: {comparison.loc['Sharpe Ratio'].idxmax()}")
print(f"Best Total Return: {comparison.loc['Total Return'].idxmax()}")

# Visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Plot 1: MA Crossover equity curve
ax = axes[0, 0]
cumulative_ma = (1 + ma_strategy.returns).cumprod()
cumulative_bh = (1 + bh_returns).cumprod()

ax.plot(cumulative_ma.index, cumulative_ma.values, 'b-', linewidth=2, label='MA Crossover')
ax.plot(cumulative_bh.index, cumulative_bh.values, 'gray', linewidth=1.5, 
        alpha=0.7, linestyle='--', label='Buy & Hold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.set_title('MA Crossover vs Buy & Hold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: MA Crossover signals
ax = axes[0, 1]
ax.plot(prices_trend.index, prices_trend['Close'], 'k-', linewidth=1.5, alpha=0.7, label='Price')

ma_fast = prices_trend['Close'].rolling(window=50).mean()
ma_slow = prices_trend['Close'].rolling(window=200).mean()

ax.plot(ma_fast.index, ma_fast.values, 'b-', linewidth=1.5, label='MA(50)')
ax.plot(ma_slow.index, ma_slow.values, 'r-', linewidth=1.5, label='MA(200)')

# Mark trades
buy_signals = (signals_ma == 1) & (signals_ma.shift(1) != 1)
sell_signals = (signals_ma == -1) & (signals_ma.shift(1) != -1)

ax.scatter(prices_trend.index[buy_signals], prices_trend['Close'][buy_signals], 
          marker='^', color='green', s=100, label='Buy', zorder=5)
ax.scatter(prices_trend.index[sell_signals], prices_trend['Close'][sell_signals],
          marker='v', color='red', s=100, label='Sell', zorder=5)

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('MA Crossover Signals')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Bollinger Bands
ax = axes[1, 0]
ax.plot(prices_range.index, prices_range['Close'], 'k-', linewidth=1.5, label='Price')

ma_bb = prices_range['Close'].rolling(window=20).mean()
std_bb = prices_range['Close'].rolling(window=20).std()
upper = ma_bb + 2*std_bb
lower = ma_bb - 2*std_bb

ax.plot(ma_bb.index, ma_bb.values, 'b-', linewidth=1, alpha=0.7, label='MA(20)')
ax.plot(upper.index, upper.values, 'r--', linewidth=1, alpha=0.7, label='Upper Band')
ax.plot(lower.index, lower.values, 'g--', linewidth=1, alpha=0.7, label='Lower Band')

ax.fill_between(ma_bb.index, lower.values, upper.values, alpha=0.1, color='gray')

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Bollinger Bands Strategy')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Pairs trading spread
ax = axes[1, 1]
spread, z_score = pairs_strategy.calculate_spread(60)

ax.plot(z_score.index, z_score.values, 'b-', linewidth=1.5, label='Z-Score')
ax.axhline(2, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Entry (+2σ)')
ax.axhline(-2, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Entry (-2σ)')
ax.axhline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(-0.5, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='Exit (±0.5σ)')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

ax.fill_between(z_score.index, -2, 2, alpha=0.1, color='gray')

ax.set_xlabel('Date')
ax.set_ylabel('Z-Score')
ax.set_title('Pairs Trading: Spread Z-Score')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Strategy comparison (equity curves)
ax = axes[2, 0]

cumulative_bb = (1 + bb_strategy.returns).cumprod()
cumulative_rsi = (1 + rsi_strategy.returns).cumprod()

ax.plot(cumulative_ma.index, cumulative_ma.values, linewidth=2, label='MA Crossover')
ax.plot(cumulative_bb.index, cumulative_bb.values, linewidth=2, label='Bollinger Bands')
ax.plot(cumulative_rsi.index, cumulative_rsi.values, linewidth=2, label='RSI')
ax.plot(cumulative_pairs.index, cumulative_pairs.values, linewidth=2, label='Pairs Trading')

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.set_title('Strategy Comparison: Equity Curves')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Drawdowns
ax = axes[2, 1]

dd_ma = (cumulative_ma - cumulative_ma.expanding().max()) / cumulative_ma.expanding().max()
dd_bb = (cumulative_bb - cumulative_bb.expanding().max()) / cumulative_bb.expanding().max()
dd_rsi = (cumulative_rsi - cumulative_rsi.expanding().max()) / cumulative_rsi.expanding().max()

ax.fill_between(dd_ma.index, 0, dd_ma.values*100, alpha=0.5, label='MA Crossover')
ax.fill_between(dd_bb.index, 0, dd_bb.values*100, alpha=0.5, label='Bollinger Bands')
ax.fill_between(dd_rsi.index, 0, dd_rsi.values*100, alpha=0.5, label='RSI')

ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Strategy Drawdowns')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **MACD + RSI Composite:** Combine MACD and RSI into single signal (vote or weighted). Compare Sharpe to individual indicators. Optimal weights?

2. **Pairs Trading with Kalman Filter:** Use Kalman filter for dynamic hedge ratio estimation. Compare to static OLS regression. Does adaptive hedging improve Sharpe?

3. **Machine Learning Signal:** Build RandomForest classifier predicting next-day direction. Features: RSI, MACD, BB %B, ATR, lagged returns. Walk-forward validation. Avoid overfitting.

4. **Volatility-Adjusted Position Sizing:** Implement Kelly criterion for position sizing. Compare fixed fractional vs Kelly vs fractional Kelly (0.25×). Impact on Sharpe and max drawdown?

5. **Regime-Dependent Strategy:** Detect bull/bear regimes (HMM or moving average filter). Switch between trend-following (bull) and mean-reversion (bear). Benchmark vs single strategy.

## 7. Key References
- [Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"](https://www.jstor.org/stable/2328882) - momentum anomaly evidence
- [Chan, Algorithmic Trading: Winning Strategies and Their Rationale](https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146) - practical strategy implementation
- [Pardo, The Evaluation and Optimization of Trading Strategies](https://www.wiley.com/en-us/The+Evaluation+and+Optimization+of+Trading+Strategies%2C+2nd+Edition-p-9780470128015) - backtesting methodology, avoiding pitfalls

---
**Status:** Systematic trading rule generation | **Complements:** Backtesting, Risk Management, Execution Algorithms, Machine Learning, Time Series Analysis
