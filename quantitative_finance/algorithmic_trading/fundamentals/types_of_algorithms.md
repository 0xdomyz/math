# Types of Trading Algorithms: Taxonomy & Classification

## 1. Concept Skeleton
**Definition:** Classification of automated trading strategies by objective: execution (minimize cost), alpha generation (maximize returns), market making (provide liquidity)  
**Purpose:** Understanding algorithm types guides selection for specific trading objectives; each type optimizes different tradeoffs  
**Prerequisites:** Execution costs, market microstructure, alpha models, liquidity provision economics

## 2. Comparative Framing
| Algorithm Type | Primary Goal | Time Horizon | Risk Profile | Typical User |
|----------------|-------------|--------------|--------------|--------------|
| **Execution Algos** | Minimize transaction cost | Minutes-hours | Low (tracking) | Buy-side institutions |
| **Alpha Algos** | Maximize risk-adjusted returns | Hours-days | Medium-high | Hedge funds, prop traders |
| **Market Making** | Earn bid-ask spread | Seconds-minutes | Medium (inventory) | Market makers, HFTs |
| **Arbitrage** | Exploit price discrepancies | Milliseconds-seconds | Low (hedged) | HFT firms |

## 3. Examples + Counterexamples

**Simple Example:**  
Execution algo (VWAP): Minimize slippage on 100K share order | Alpha algo (momentum): Buy stocks with positive 20-day return

**Failure Case:**  
Execution algo tries to generate alpha → Deviates from benchmark → Tracking error explodes → Misses objective

**Edge Case:**  
Market making algo during flash crash → Inventory accumulates → Cannot exit → Massive loss from adverse price move

## 4. Layer Breakdown
```
Trading Algorithm Taxonomy:
├─ I. EXECUTION ALGORITHMS (Cost Minimization):
│   ├─ Benchmark-Driven:
│   │   ├─ VWAP (Volume-Weighted Average Price):
│   │   │   ├─ Objective: Match time-weighted participation
│   │   │   ├─ Schedule: Proportional to volume forecast
│   │   │   ├─ Use Case: Large institutional orders
│   │   │   └─ Risk: Forecast error, predictability
│   │   ├─ TWAP (Time-Weighted Average Price):
│   │   │   ├─ Objective: Uniform temporal distribution
│   │   │   ├─ Schedule: Fixed slices every N minutes
│   │   │   ├─ Use Case: Thin liquidity, unpredictable volume
│   │   │   └─ Risk: Ignores volume, may dominate market
│   │   └─ Closing Price:
│   │       ├─ Objective: Match official close (e.g., NAV)
│   │       ├─ Schedule: Concentrate at 3:50-4:00 PM
│   │       ├─ Use Case: Index tracking, mutual funds
│   │       └─ Risk: Crowded, volatile closing auction
│   ├─ Cost-Driven:
│   │   ├─ Implementation Shortfall (Arrival Price):
│   │   │   ├─ Objective: Min (execution price - decision price)
│   │   │   ├─ Method: Balance urgency vs impact (Almgren-Chriss)
│   │   │   ├─ Use Case: Risk-averse execution
│   │   │   └─ Risk: Requires volatility/impact calibration
│   │   ├─ POV (Percent of Volume):
│   │   │   ├─ Objective: Trade fixed % of market volume
│   │   │   ├─ Method: Real-time volume tracking, adaptive sizing
│   │   │   ├─ Use Case: Blend into flow, low footprint
│   │   │   └─ Risk: Volume spikes, incomplete execution
│   │   └─ Adaptive Execution:
│   │       ├─ Objective: Dynamic adjustment to market state
│   │       ├─ Method: ML models, reinforcement learning
│   │       ├─ Use Case: Complex market conditions
│   │       └─ Risk: Model error, overfitting
│   └─ Venue Selection:
│       ├─ Smart Order Routing (SOR):
│       │   ├─ Scan: All venues for best price + fees
│       │   ├─ Route: Optimal sequence across exchanges
│       │   └─ Rebates: Maker-taker economics
│       └─ Dark Pool Aggregation:
│           ├─ Ping: Multiple dark pools for hidden liquidity
│           ├─ Execute: Large blocks without information leakage
│           └─ Risk: Adverse selection, latency
├─ II. ALPHA GENERATION ALGORITHMS (Return Maximization):
│   ├─ Statistical Arbitrage:
│   │   ├─ Pairs Trading:
│   │   │   ├─ Identify: Cointegrated stock pairs
│   │   │   ├─ Signal: Z-score > threshold → short spread
│   │   │   ├─ Mean Reversion: Profit from spread convergence
│   │   │   └─ Risk: Divergence (co-integration breakdown)
│   │   ├─ Factor Models:
│   │   │   ├─ Signals: Value, momentum, quality, size
│   │   │   ├─ Ranking: Stocks by composite score
│   │   │   ├─ Portfolio: Long top quintile, short bottom
│   │   │   └─ Risk: Factor crowding, regime shifts
│   │   └─ Mean Reversion:
│   │       ├─ Identify: Oversold/overbought (RSI, Bollinger)
│   │       ├─ Entry: Extreme deviations from moving average
│   │       ├─ Exit: Return to mean or time stop
│   │       └─ Risk: Trending markets (no reversion)
│   ├─ Momentum Strategies:
│   │   ├─ Trend Following:
│   │   │   ├─ Signal: Price > moving average → buy
│   │   │   ├─ Holding: Ride trend until reversal signal
│   │   │   ├─ Use Case: Liquid futures, FX
│   │   │   └─ Risk: Whipsaws, choppy markets
│   │   ├─ Breakout:
│   │   │   ├─ Signal: Price crosses resistance → momentum
│   │   │   ├─ Entry: Volume confirmation required
│   │   │   ├─ Stop: Below breakout level
│   │   │   └─ Risk: False breakouts (fakeouts)
│   │   └─ Relative Strength:
│   │       ├─ Rank: Stocks by trailing returns
│   │       ├─ Portfolio: Long winners, short losers
│   │       ├─ Rebalance: Monthly
│   │       └─ Risk: Reversal (winner becomes loser)
│   ├─ Machine Learning:
│   │   ├─ Supervised Learning:
│   │   │   ├─ Features: Price, volume, sentiment, fundamentals
│   │   │   ├─ Target: Next-period return, direction
│   │   │   ├─ Models: Random forest, gradient boosting, LSTM
│   │   │   └─ Risk: Overfitting, non-stationarity
│   │   ├─ Reinforcement Learning:
│   │   │   ├─ Agent: Learns optimal trading policy
│   │   │   ├─ State: Market features + portfolio
│   │   │   ├─ Action: Buy, sell, hold quantities
│   │   │   ├─ Reward: P&L minus transaction costs
│   │   │   └─ Risk: Exploration instability, reward hacking
│   │   └─ Sentiment Analysis:
│   │       ├─ Sources: News, Twitter, SEC filings
│   │       ├─ NLP: Extract sentiment scores
│   │       ├─ Signal: High sentiment → buy, negative → sell
│   │       └─ Risk: False signals, crowding
│   └─ Event-Driven:
│       ├─ Earnings Surprises:
│       │   ├─ Predict: EPS vs consensus
│       │   ├─ Trade: Post-announcement drift
│       │   └─ Risk: Already priced in
│       ├─ Merger Arbitrage:
│       │   ├─ Long: Target stock
│       │   ├─ Short: Acquirer (if stock deal)
│       │   ├─ Profit: Spread convergence at close
│       │   └─ Risk: Deal breaks, regulatory block
│       └─ Macro Announcements:
│           ├─ Events: FOMC, NFP, CPI
│           ├─ Speed: Trade first millisecond
│           └─ Risk: Misinterpretation, volatility
├─ III. MARKET MAKING ALGORITHMS (Liquidity Provision):
│   ├─ Quote Management:
│   │   ├─ Spread Setting:
│   │   │   ├─ Objective: Max profit_per_trade × trade_frequency
│   │   │   ├─ Model: Avellaneda-Stoikov optimal spread
│   │   │   ├─ Factors: Volatility, inventory, adverse selection
│   │   │   └─ Adjustment: Widen in volatility, narrow in competition
│   │   ├─ Inventory Control:
│   │   │   ├─ Target: Zero inventory (risk-neutral)
│   │   │   ├─ Tilt: Skew quotes to reduce inventory
│   │   │   ├─ Example: Long 1000 shares → lower ask, raise bid
│   │   │   └─ Risk: Directional price move
│   │   └─ Adverse Selection Protection:
│   │       ├─ Detect: Informed order flow (volume, speed)
│   │       ├─ Response: Widen spread, reduce size, cancel
│   │       └─ Risk: Lose to HFT predators
│   ├─ High-Frequency Market Making:
│   │   ├─ Latency: Co-location, <1ms response
│   │   ├─ Frequency: Thousands of quotes/second
│   │   ├─ Rebates: Capture maker fees (0.2-0.3 bps)
│   │   └─ Risk: Technology arms race, reg changes
│   └─ Options Market Making:
│       ├─ Greeks: Delta-hedge, manage vega/gamma
│       ├─ Skew: Volatility surface arbitrage
│       └─ Risk: Pin risk, gamma scalping losses
└─ IV. ARBITRAGE ALGORITHMS (Risk-Free Profit):
    ├─ Spatial Arbitrage:
    │   ├─ Cross-Exchange:
    │   │   ├─ Detect: Same asset, different prices
    │   │   ├─ Exploit: Buy low exchange, sell high
    │   │   ├─ Speed: Fastest data feed wins
    │   │   └─ Risk: Execution lag, fees
    │   └─ Triangular (FX):
    │       ├─ Three currencies: USD/EUR, EUR/GBP, GBP/USD
    │       ├─ Circular: If A→B→C→A ≠ 1 → profit
    │       └─ Risk: Extremely short-lived (<100ms)
    ├─ Statistical Arbitrage:
    │   ├─ Index Arbitrage:
    │   │   ├─ Spread: ETF vs basket of components
    │   │   ├─ Trade: Long cheap, short rich
    │   │   ├─ Convergence: Creation/redemption mechanism
    │   │   └─ Risk: Tracking error, dividend timing
    │   └─ Convertible Arbitrage:
    │       ├─ Long: Convertible bond (cheap)
    │       ├─ Short: Underlying stock
    │       ├─ Profit: Bond undervaluation, vol mispricing
    │       └─ Risk: Credit spread widening
    └─ Latency Arbitrage:
        ├─ Speed: Access stale quotes before update
        ├─ Front-Run: Predict large order flow
        ├─ Technology: Microwave networks, FPGAs
        └─ Ethics: Controversial (informed vs predatory)
```

**Interaction:** Algorithm type → Objective function → Market state → Order generation → Execution

## 5. Mini-Project
Implement and compare execution vs alpha algorithms:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ============================================================================
# MARKET ENVIRONMENT
# ============================================================================

class Market:
    """Simulated market with price dynamics and execution costs."""
    
    def __init__(self, n_periods=100, initial_price=100):
        self.n_periods = n_periods
        self.current_period = 0
        
        # True price (GBM with drift)
        drift = 0.0001
        vol = 0.002
        returns = np.random.normal(drift, vol, n_periods)
        self.true_prices = initial_price * np.exp(np.cumsum(returns))
        
        # Observable price (true + noise)
        noise = np.random.normal(0, 0.0005, n_periods)
        self.observed_prices = self.true_prices * (1 + noise)
        
        # Spread and volume
        self.spreads = 0.05 + 0.01 * np.random.rand(n_periods)
        self.volumes = 10000 + 2000 * np.random.randn(n_periods)
    
    def get_execution_price(self, quantity, side='buy'):
        """Execute order with market impact."""
        true_price = self.true_prices[self.current_period]
        spread = self.spreads[self.current_period]
        volume = self.volumes[self.current_period]
        
        # Spread cost
        if side == 'buy':
            price = true_price + spread / 2
        else:
            price = true_price - spread / 2
        
        # Market impact (square root)
        participation = abs(quantity) / volume
        impact_bps = 30 * np.sqrt(participation)
        
        if side == 'buy':
            price *= (1 + impact_bps / 10000)
        else:
            price *= (1 - impact_bps / 10000)
        
        return price
    
    def advance(self):
        self.current_period += 1
    
    def is_done(self):
        return self.current_period >= self.n_periods

# ============================================================================
# EXECUTION ALGORITHM (VWAP)
# ============================================================================

def vwap_execution(market: Market, total_qty=10000, duration=10):
    """Execute using volume-weighted schedule."""
    start_price = market.true_prices[market.current_period]
    
    # Forecast volume (simplified: use actual)
    volume_forecast = []
    start = market.current_period
    for i in range(duration):
        if start + i < market.n_periods:
            volume_forecast.append(market.volumes[start + i])
    
    total_vol = sum(volume_forecast)
    
    execution_prices = []
    quantities = []
    
    for vol in volume_forecast:
        if market.is_done():
            break
        
        slice_qty = int(total_qty * (vol / total_vol))
        exec_price = market.get_execution_price(slice_qty, 'buy')
        
        execution_prices.append(exec_price)
        quantities.append(slice_qty)
        market.advance()
    
    avg_price = np.average(execution_prices, weights=quantities)
    total_executed = sum(quantities)
    slippage = (avg_price - start_price) / start_price * 10000
    
    return {
        'type': 'Execution (VWAP)',
        'avg_price': avg_price,
        'start_price': start_price,
        'slippage_bps': slippage,
        'total_qty': total_executed,
        'periods': len(execution_prices)
    }

# ============================================================================
# ALPHA ALGORITHM (MOMENTUM)
# ============================================================================

def momentum_alpha(market: Market, lookback=20, holding_period=10, capital=100000):
    """Simple momentum strategy: buy if recent return positive."""
    
    # Calculate momentum signal
    if market.current_period < lookback:
        return {
            'type': 'Alpha (Momentum)',
            'pnl': 0,
            'return_pct': 0,
            'trades': 0
        }
    
    # Momentum = return over lookback period
    start_idx = market.current_period - lookback
    momentum_return = (market.observed_prices[market.current_period] / 
                      market.observed_prices[start_idx] - 1)
    
    # Trading signal: buy if momentum > 0
    if momentum_return > 0:
        # Enter long position
        entry_price = market.get_execution_price(1000, 'buy')
        entry_period = market.current_period
        
        # Hold for holding_period
        for _ in range(holding_period):
            market.advance()
            if market.is_done():
                break
        
        # Exit position
        exit_price = market.get_execution_price(1000, 'sell')
        
        # P&L calculation
        pnl = 1000 * (exit_price - entry_price)
        return_pct = (exit_price / entry_price - 1) * 100
        
        return {
            'type': 'Alpha (Momentum)',
            'pnl': pnl,
            'return_pct': return_pct,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trades': 1
        }
    else:
        # No trade
        for _ in range(holding_period):
            market.advance()
            if market.is_done():
                break
        
        return {
            'type': 'Alpha (Momentum)',
            'pnl': 0,
            'return_pct': 0,
            'trades': 0
        }

# ============================================================================
# MARKET MAKING ALGORITHM
# ============================================================================

def market_making(market: Market, duration=50, spread_multiplier=1.5):
    """Simplified market making: post quotes, earn spread."""
    
    total_pnl = 0
    inventory = 0
    trades = 0
    
    for _ in range(duration):
        if market.is_done():
            break
        
        true_price = market.true_prices[market.current_period]
        market_spread = market.spreads[market.current_period]
        
        # Set quotes (wider than market to be profitable)
        our_spread = market_spread * spread_multiplier
        bid = true_price - our_spread / 2
        ask = true_price + our_spread / 2
        
        # Simulate order arrivals (random)
        if np.random.rand() < 0.3:  # 30% chance of buy order
            # We sell at ask
            inventory -= 100
            total_pnl += 100 * ask
            trades += 1
        
        if np.random.rand() < 0.3:  # 30% chance of sell order
            # We buy at bid
            inventory += 100
            total_pnl -= 100 * bid
            trades += 1
        
        market.advance()
    
    # Liquidate remaining inventory at market
    if inventory != 0:
        liquidation_price = market.get_execution_price(abs(inventory), 
                                                       'sell' if inventory > 0 else 'buy')
        if inventory > 0:
            total_pnl += inventory * liquidation_price
        else:
            total_pnl -= abs(inventory) * liquidation_price
    
    return {
        'type': 'Market Making',
        'pnl': total_pnl,
        'trades': trades,
        'final_inventory': 0
    }

# ============================================================================
# SIMULATION & COMPARISON
# ============================================================================

print("="*70)
print("ALGORITHM TYPE COMPARISON")
print("="*70)

n_simulations = 100

vwap_results = []
momentum_results = []
mm_results = []

for sim in range(n_simulations):
    # VWAP execution
    market1 = Market(n_periods=100)
    vwap_result = vwap_execution(market1, total_qty=10000, duration=10)
    vwap_results.append(vwap_result)
    
    # Momentum alpha
    market2 = Market(n_periods=100)
    while not market2.is_done():
        momentum_result = momentum_alpha(market2, lookback=20, holding_period=10)
        if momentum_result['trades'] > 0:
            momentum_results.append(momentum_result)
            break
        elif market2.is_done():
            break
    
    # Market making
    market3 = Market(n_periods=100)
    mm_result = market_making(market3, duration=50, spread_multiplier=1.5)
    mm_results.append(mm_result)

# Aggregate statistics
print("\n1. EXECUTION ALGORITHM (VWAP):")
print(f"   Primary Metric: Slippage (minimize transaction cost)")
vwap_slippages = [r['slippage_bps'] for r in vwap_results]
print(f"   Avg Slippage: {np.mean(vwap_slippages):.2f} ± {np.std(vwap_slippages):.2f} bps")
print(f"   Objective: Track benchmark, minimal deviation")

print("\n2. ALPHA ALGORITHM (Momentum):")
print(f"   Primary Metric: Return (maximize P&L)")
if len(momentum_results) > 0:
    momentum_returns = [r['return_pct'] for r in momentum_results]
    print(f"   Avg Return: {np.mean(momentum_returns):.2f}% ± {np.std(momentum_returns):.2f}%")
    print(f"   Sharpe Ratio: {np.mean(momentum_returns) / np.std(momentum_returns) if np.std(momentum_returns) > 0 else 0:.2f}")
else:
    print(f"   No trades executed")
print(f"   Objective: Generate alpha, outperform market")

print("\n3. MARKET MAKING:")
print(f"   Primary Metric: P&L from spread capture")
mm_pnls = [r['pnl'] for r in mm_results]
mm_trades = [r['trades'] for r in mm_results]
print(f"   Avg P&L: ${np.mean(mm_pnls):.2f} ± ${np.std(mm_pnls):.2f}")
print(f"   Avg Trades: {np.mean(mm_trades):.1f}")
print(f"   Objective: Provide liquidity, earn bid-ask spread")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Execution algo - slippage distribution
ax1 = axes[0, 0]
ax1.hist(vwap_slippages, bins=20, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(np.mean(vwap_slippages), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(vwap_slippages):.2f} bps')
ax1.set_xlabel('Slippage (bps)')
ax1.set_ylabel('Frequency')
ax1.set_title('Execution Algorithm: VWAP Slippage Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Alpha algo - return distribution
ax2 = axes[0, 1]
if len(momentum_results) > 0:
    ax2.hist(momentum_returns, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(momentum_returns), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(momentum_returns):.2f}%')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Alpha Algorithm: Momentum Return Distribution')
    ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Market making - P&L distribution
ax3 = axes[1, 0]
ax3.hist(mm_pnls, bins=20, alpha=0.7, color='orange', edgecolor='black')
ax3.axvline(np.mean(mm_pnls), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: ${np.mean(mm_pnls):.0f}')
ax3.set_xlabel('P&L ($)')
ax3.set_ylabel('Frequency')
ax3.set_title('Market Making: P&L Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Algorithm type comparison (objectives)
ax4 = axes[1, 1]
algo_types = ['Execution\n(VWAP)', 'Alpha\n(Momentum)', 'Market\nMaking']
objectives = ['Min Cost', 'Max Return', 'Earn Spread']
colors = ['blue', 'green', 'orange']

# Create comparison metrics (normalized to 0-100)
metrics = [
    100 - abs(np.mean(vwap_slippages)),  # Lower slippage = better
    50 + np.mean(momentum_returns) * 10 if len(momentum_results) > 0 else 50,  # Scale returns
    50 + np.mean(mm_pnls) / 100  # Scale P&L
]

bars = ax4.bar(algo_types, metrics, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Performance Score (higher = better)')
ax4.set_title('Algorithm Type Performance')
ax4.grid(True, alpha=0.3, axis='y')

# Add objective labels
for i, (bar, obj) in enumerate(zip(bars, objectives)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{obj}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('algorithm_types_comparison.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: algorithm_types_comparison.png")
plt.show()
```

## 6. Challenge Round
When does algorithm type selection go wrong?
- Use execution algo for alpha: VWAP follows benchmark → Zero alpha generation, opportunity cost
- Use alpha algo for execution: Momentum deviates from instruction → Large tracking error, client unhappy
- Market making in one-sided market: Only sells fill, inventory accumulates → Directional loss
- Arbitrage with fees ignored: Profit < transaction costs → Net loss despite correct price prediction
- Wrong time horizon: HFT algo on daily data → No edge, just noise

## 7. Key References
- [Kissell (2011), "Algorithmic Trading Methods"](https://www.wiley.com/en-us/Algorithmic+Trading+Methods-p-9780470643112) - Comprehensive taxonomy
- [Almgren (2003), "Optimal Execution"](https://www.jstor.org/stable/2692547) - Execution algorithms
- [Avellaneda & Stoikov (2008), "Market Making"](https://arxiv.org/abs/0811.3551) - Market making models

---
**Status:** Foundation for algorithm selection | **Complements:** Execution strategies, alpha models, market making | **Guides:** Implementation decisions