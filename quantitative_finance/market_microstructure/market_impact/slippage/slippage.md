# Slippage

## 1. Concept Skeleton
**Definition:** Difference between expected execution price and actual executed price; aggregates all transaction costs including spread, market impact, commissions, and opportunity costs  
**Purpose:** Quantify total execution friction, compare algorithm performance, benchmark against benchmarks (VWAP/TWAP), identify cost-reduction opportunities  
**Prerequisites:** Order execution mechanics, bid-ask spreads, market microstructure, venue fragmentation

## 2. Comparative Framing
| Slippage Type | Source | Magnitude | Control | Trading Impact |
|---------------|--------|-----------|---------|-----------------|
| **Spread Slippage** | Bid-ask spread | 1-50 bps | Venue selection | Fixed, unavoidable |
| **Market Impact** | Order size effect | 5-200 bps | Execution pace | Controllable via speed |
| **Adverse Selection** | Informed traders | 10-100 bps | Order routing | Minimize toxic fill |
| **Timing Slippage** | Delay in execution | 5-100 bps | Algorithm efficiency | Minimize latency |
| **Opportunity Cost** | Missed better prices | 10-200 bps | Order aggressiveness | Risk-return tradeoff |
| **Venue Fragmentation** | Multi-exchange splits | 5-50 bps | SOR quality | Consolidation needed |
| **Commissions** | Broker/exchange fees | 0.5-5 bps | Negotiation | Declining |

## 3. Examples + Counterexamples

**Spread Slippage (Liquid Stock):**  
Apple (AAPL): Bid $150.10, Ask $150.20 (10 cent spread = 7 bps). Buy at ask = 5 bps slippage.

**Market Impact Slippage (Large Order):**  
Execute 100k shares (2% daily volume). Price moves +40 bps during execution. 20 bps permanent + 20 bps temporary = 40 bps total slippage beyond spread.

**Opportunity Cost (Patient Execution):**  
Goal: Buy 50k shares at VWAP. Use passive algo (slow). Market rallies 100 bps during day. Fill only 40k shares (20% miss). Opportunity cost on missed 10k: 100 bps × 10k = significant (regret).

**Adverse Selection (Dark Pool Fill):**  
Submit order to dark pool expecting VWAP. Sophisticated trader accepts → likely informed, worse prices ahead. Slippage: Lose 20+ bps to informed counterparty (selection adverse).

**Timing Slippage (Latency):**  
Quote available at $100.00. System delay 100 msec. By time order reaches exchange, quote moved to $100.05. Latency slippage: 5 bps due to system/network lag.

**Venue Fragmentation (Fragmented Markets):**  
Execute 10k: 5k on exchange A (best price $100.00), 5k on ATS B (lagging at $100.10). Average fill $100.05, midpoint $100.05. Venue mismatch added 5 bps slippage.

**Commission Slippage (Broker Fee Dependent):**  
Discount broker: 0.5 bps ($50 per $1M traded) = 0.5 bps slippage.  
Premium broker: 3 bps ($300 per $1M) = 3 bps slippage. Same execution, different total cost.

## 4. Layer Breakdown
```
Slippage Components & Drivers:

├─ Slippage Decomposition:
│  ├─ Total Slippage:
│  │  ├─ S_total = (Exec_price - Benchmark_price) / Benchmark_price
│  │  ├─ Benchmark typically: VWAP, Arrival price, or midpoint
│  │  ├─ Positive slippage = Adverse (paid more for buy)
│  │  ├─ Negative slippage = Favorable (got better price)
│  │  └─ Typical range: ±50 bps
│  │
│  ├─ Component 1: Spread Slippage
│  │  ├─ Definition: Cost of immediacy from bid-ask spread
│  │  ├─ S_spread = (Ask - Bid) / (2 × Midpoint) ≈ 0.5 × relative_spread
│  │  ├─ Unavoidable: Must cross spread to trade immediately
│  │  ├─ Fixed regardless of order size
│  │  ├─ Controllable: Venue selection (tight spreads), order type (limit orders avoid)
│  │  ├─ Typical ranges:
│  │  │   ├─ Large-cap (AAPL): 1-3 bps
│  │  │   ├─ Mid-cap: 3-10 bps
│  │  │   ├─ Small-cap: 10-50 bps
│  │  │   ├─ OTC/penny stocks: 100+ bps
│  │  ├─ Dynamics:
│  │  │  ├─ Widens during market stress (dealer risk aversion)
│  │  │  ├─ Tightens during high volume (competition)
│  │  │  ├─ Wide at open/close (auction volatility)
│  │  │  └─ Narrow mid-day
│  │  └─ Example: VWAP execution avoiding "touching" spread multiple times
│  │
│  ├─ Component 2: Market Impact Slippage
│  │  ├─ Definition: Price move from executing large order
│  │  ├─ S_impact = (Price_execution - Midpoint_before) / Midpoint
│  │  ├─ Non-linear in order size: Sqrt-law impact
│  │  ├─ S_impact ≈ β × √(Q / V_daily)
│  │  │  ├─ Q = order size
│  │  │  ├─ V_daily = daily volume
│  │  │  ├─ β ≈ 0.0001-0.001 (stock-specific)
│  │  │  └─ Example: 1% of volume → 0.01 impact, 2% → 0.014 impact
│  │  │
│  │  ├─ Components:
│  │  │  ├─ Temporary: Liquidity supplier accommodates (reverts)
│  │  │  ├─ Permanent: Market learns about information (stays)
│  │  │  ├─ Typically: 30-60% permanent, 40-70% temporary
│  │  │  └─ Controllable by execution pace (slow = lower impact)
│  │  │
│  │  └─ Dynamics:
│  │     ├─ High volume days: Lower impact (more liquidity available)
│  │     ├─ Low liquidity periods: Higher impact (fewer counterparties)
│  │     ├─ Stressed markets: Impact function breaks (crisis)
│  │     └─ Crowded strategies: Impact amplified (all traders same side)
│  │
│  ├─ Component 3: Adverse Selection Slippage
│  │  ├─ Definition: Trading with informed counterparties
│  │  ├─ Mechanisms:
│  │  │  ├─ Informed traders execute before news release (know better price)
│  │  │  ├─ Market makers widen spreads (defend against informed)
│  │  │  ├─ PIN (Probability of Informed Trading) high → wider spreads
│  │  │  └─ Uninformed traders face adverse selection cost
│  │  │
│  │  ├─ Identification:
│  │  │  ├─ Large trades at market price tend to lose (informed selling before drop)
│  │  │  ├─ Systematic worse fills → possible informed trading
│  │  │  ├─ VPIN (toxicity) metric identifies stressed periods
│  │  │  └─ High PIN stocks: More adverse selection cost
│  │  │
│  │  ├─ Typical cost: 10-100 bps (varies with information environment)
│  │  └─ Mitigation:
│  │     ├─ Use dark pools (reduce information leakage)
│  │     ├─ Limit order routing (hide order size)
│  │     ├─ Avoid trading near news
│  │     └─ Use execution algorithms (slice orders, reduce visibility)
│  │
│  ├─ Component 4: Timing/Latency Slippage
│  │  ├─ Definition: Cost from system delay/latency
│  │  ├─ Sources:
│  │  │  ├─ Network latency (local to exchange: 1-10 msec)
│  │  │  ├─ Processing delay (system: 5-50 msec)
│  │  │  ├─ Order queue delay (matching engine: 1-5 msec)
│  │  │  ├─ Stale quotes (data lag: 100-500 msec)
│  │  │  └─ Tick-by-tick processing (sequential, not parallel)
│  │  │
│  │  ├─ Magnitude:
│  │  │  ├─ Price move per msec ≈ volatility / √(time_steps)
│  │  │  ├─ Example: Stock with 20% annual vol → ~12 bps per second
│  │  │  ├─ 100 msec latency → ~0.12 bps cost
│  │  │  ├─ Typical 1-5 bps for retail systems, sub-bps for HFT
│  │  │  └─ Large in micro-cap, negligible in large-cap (vol-dependent)
│  │  │
│  │  └─ Mitigation:
│  │     ├─ Co-location (reduce network latency)
│  │     ├─ Direct market access (bypass intermediaries)
│  │     ├─ Faster algorithms (parallel processing)
│  │     └─ Accept latency as cost (less critical in less volatile markets)
│  │
│  ├─ Component 5: Opportunity Cost Slippage
│  │  ├─ Definition: Cost of not achieving full fill at desired price
│  │  ├─ Scenarios:
│  │  │  ├─ Passive execution (low impact): Miss price moves, partial fill
│  │  │  │   ├─ Goal: 50k shares, only fill 40k (80%)
│  │  │  │   ├─ Market rallies 50 bps during day
│  │  │  │   ├─ Cost of missing 10k: 50 bps → = 5 bps on total target
│  │  │  │   ├─ Regret: Should have been more aggressive
│  │  │  │   └─ Tradeoff: Lower impact (5 bps) but miss opportunity (50 bps regret)
│  │  │  │
│  │  │  ├─ Aggressive execution (high impact): Full fill but at worse prices
│  │  │  │   ├─ Goal: 50k shares, fill 100%
│  │  │  │   ├─ Execute quickly but pay +30 bps impact
│  │  │  │   └─ Certainty but higher cost
│  │  │  │
│  │  │  └─ Risk asymmetry: Regret bounded on upside (VWAP/max), unbounded on downside
│  │  │
│  │  ├─ Measurement:
│  │  │  ├─ Full fill benchmark: Completion at end of target period
│  │  │  ├─ Partial fill slippage: (Filled_pct - 0%) × market_move
│  │  │  ├─ Example: 80% filled, market up 100 bps → 80 bps opportunity cost
│  │  │  └─ Only matters if order NOT completed
│  │  │
│  │  └─ Risk management:
│  │     ├─ Set minimum fill thresholds (e.g., >90%)
│  │     ├─ Hard/soft limits (abort if can't fill >X%)
│  │     └─ Tradeoff accepted market impact for certainty
│  │
│  ├─ Component 6: Venue Fragmentation Slippage
│  │  ├─ Definition: Cost from multi-exchange execution split
│  │  ├─ Sources:
│  │  │  ├─ Latency difference: Quote at exchange A, fill at B (A moved higher)
│  │  │  ├─ NBBO violation: Execute at non-best quote (poor SOR)
│  │  │  ├─ Concentration: Not routing to deepest liquidity
│  │  │  └─ Rebate asymmetry: Taker/maker fee structures differ
│  │  │
│  │  ├─ Example:
│  │  │  ├─ Best bid: Exchange A $100.10 (100 shares)
│  │  │  ├─ Secondary bid: Exchange B $100.05 (10k shares)
│  │  │  ├─ Route first 100 to A, rest to B → avg $100.06
│  │  │  ├─ Better SOR: Route all 100 to B at $100.05
│  │  │  └─ SOR slippage: 1 bp per share × 100 shares
│  │  │
│  │  ├─ Regulation (Reg NMS):
│  │  │  ├─ Rule 611 (Trade-Through): Can't execute at worse price if protected quote exists
│  │  │  ├─ Rule 610 (Market Access): Must have reasonable access to exchanges
│  │  │  ├─ Rule 612 (Sub-penny): Prohibits orders finer than $0.01 (except $0-1 range)
│  │  │  └─ Intended: Minimize fragmentation, protect investors
│  │  │
│  │  └─ Best Execution Duty:
│  │     ├─ Brokers must route to best execution (price, size, likelihood)
│  │     ├─ Not just best price (all-in cost, including rebates)
│  │     ├─ Fiduciary responsibility
│  │     └─ Violation → fines, regulatory action
│  │
│  └─ Component 7: Commission/Fee Slippage
│     ├─ Definition: Explicit fees charged by brokers/exchanges
│     ├─ Structures:
│     │  ├─ Flat per-share: $0.001 per share = 0.1 bps on $100 stock
│     │  ├─ Percentage-of-notional: 0.5-3 bps (typical retail)
│     │  ├─ Tiered: Lower fees for higher volumes ($1M+ → 0.1 bps)
│     │  ├─ Maker/taker: Rebates for liquidity provision, fees for removal
│     │  └─ HFT programs: Often sub-0.1 bps (volume + rebates)
│     │
│     ├─ Negotiation Leverage:
│     │  ├─ Retail: Fixed rates (minimal negotiation)
│     │  ├─ Institutional: Volume-based discounts (50% below retail possible)
│     │  ├─ Hedge fund/prop: Custom rates (potentially 0.01-0.05 bps all-in)
│     │  └─ Total cost: Commissions + exchange fees + clearing
│     │
│     └─ Impact:
│        ├─ High-frequency strategies: Commissions critical (volume-dependent)
│        ├─ Long-term investors: Commissions negligible per trade
│        ├─ Total cost per trade: Spread + impact + commissions
│        └─ Can exceed impact cost for small trades
│
├─ Slippage Measurement & Analysis:
│  ├─ Definition Consistency:
│  │  ├─ Benchmark-relative: (Exec_price - Benchmark) / Benchmark
│  │  ├─ Benchmark choice critical (VWAP vs. AP vs. close vs. custom)
│  │  ├─ Different benchmarks → different conclusions
│  │  ├─ Standard: VWAP for algo analysis, AP for TCA
│  │  └─ Disclose benchmark in reporting
│  │
│  ├─ Aggregation Methods:
│  │  ├─ Arithmetic mean: Σ(S_i) / n (simple, equally-weighted)
│  │  ├─ Volume-weighted: Σ(S_i × Q_i) / Σ(Q_i) (size-adjusted)
│  │  ├─ Risk-adjusted: Σ(S_i × weight_i) where weight = function(size, vol)
│  │  └─ Percentile: Median, 90th percentile (robust to outliers)
│  │
│  ├─ Segmentation:
│  │  ├─ By stock: Which names have highest slippage?
│  │  ├─ By size: Do larger orders incur more slippage? (Non-linear)
│  │  ├─ By time: Open/close vs. mid-day slippage
│  │  ├─ By algo: Compare algorithm performance
│  │  ├─ By venue: Exchange A vs. ATS B slippage
│  │  └─ By regime: Normal vs. stress days
│  │
│  ├─ Peer Comparison:
│  │  ├─ Rank traders by average slippage
│  │  ├─ Identify best-in-class (quartile 1)
│  │  ├─ Identify worst-in-class (quartile 4)
│  │  ├─ Adjust for stock/size/time selection (not pure algo skill)
│  │  └─ Account for "lucky" (better macro timing) vs. "skillful"
│  │
│  └─ Attribution:
│     ├─ Break down total slippage into components (if possible)
│     ├─ Assign responsibility: Spread unavoidable, impact partially controllable
│     ├─ Set targets: Spread (minimize via SOR), impact (execute smarter), commissions (negotiate)
│     └─ Continuous improvement cycle
│
├─ Slippage Minimization Strategies:
│  ├─ Reduce Spread Slippage:
│  │  ├─ Use limit orders (avoid spread by providing liquidity)
│  │  ├─ Smart order routing (best exchange)
│  │  ├─ Trade mid-market hours (tighter spreads vs. open/close)
│  │  ├─ Use limit order books aggressively (queue priority)
│  │  └─ Target passive execution during high-liquidity periods
│  │
│  ├─ Reduce Market Impact:
│  │  ├─ Slice orders over time (execution pacing)
│  │  ├─ Use VWAP/TWAP (track volume, even out execution)
│  │  ├─ Adaptive algorithms (adjust pace to market conditions)
│  │  ├─ Synthetic/parent orders (hidden execution)
│  │  ├─ Use dark pools for large blocks (avoid NBBO display)
│  │  └─ Execute off-market hours (lower impact, higher spread)
│  │
│  ├─ Reduce Adverse Selection:
│  │  ├─ Hide order intent (smaller visible size)
│  │  ├─ Execute in dark pools (no information leakage)
│  │  ├─ Avoid trading around news (information risk)
│  │  ├─ Use sophisticated routing (spread risk across venues)
│  │  └─ Monitor for informed trading (PIN, VPIN)
│  │
│  ├─ Reduce Latency Slippage:
│  │  ├─ Co-locate servers at exchange
│  │  ├─ Direct market access (bypass intermediaries)
│  │  ├─ Faster hardware/algorithms
│  │  ├─ Less critical for longer-horizon strategies
│  │  └─ Higher investment for marginal benefit (decreasing returns)
│  │
│  ├─ Manage Opportunity Cost:
│  │  ├─ Balance passive (low impact) vs. aggressive (full fill)
│  │  ├─ Set completion targets (e.g., 90% fill by target time)
│  │  ├─ Accept market-dependent fills (some uncertainty)
│  │  └─ Tradeoff: Lower cost + incomplete vs. higher cost + complete
│  │
│  ├─ Optimize Venue Selection:
│  │  ├─ Smart order routing (SOR) algorithms
│  │  ├─ Multi-venue execution (consolidate best quotes)
│  │  ├─ Evaluate rebates/fees (net cost, not just price)
│  │  ├─ Negotiate rates with exchanges
│  │  └─ Periodically reassess (liquidity migration)
│  │
│  └─ Reduce Commission Slippage:
│     ├─ Volume discounts (negotiate lower rates for higher volume)
│     ├─ Consolidate brokers (fewer, larger relationships)
│     ├─ Use price improvement programs (brokers rebate to clients)
│     ├─ Evaluate all-in cost (rebates may offset higher fees)
│     └─ Shop rates annually (competitive pressure)
│
└─ Technology & Measurement:
   ├─ Real-Time Slippage Monitoring:
   │  ├─ Track VWAP/TWAP continuously
   │  ├─ Compare realized execution vs. benchmark
   │  ├─ Alert on excessive slippage (statistical threshold)
   │  ├─ Drill down to cause (spread widened? impact spike?)
   │  └─ Feedback to traders/algorithms
   │
   ├─ Analytics Platforms:
   │  ├─ Bloomberg AFPT (standard post-trade)
   │  ├─ FactSet Analytics (peer benchmarking)
   │  ├─ ITG Posit (execution analysis)
   │  ├─ Greenlight TCA (specialized)
   │  └─ Custom dashboards (internal systems)
   │
   ├─ Data Requirements:
   │  ├─ Order-level data (submission, execution time/price, size)
   │  ├─ Reference prices (VWAP, TWAP, benchmarks)
   │  ├─ Fee data (commissions, exchange charges, rebates)
   │  ├─ Market data (quotes, trades, volume)
   │  └─ Quality assurance (timestamps, partial fills, corporate actions)
   │
   └─ Interpretation Caution:
      ├─ Survivorship bias (exclude failed orders)
      ├─ Selection bias (choose easiest trades to execute)
      ├─ Market-timing confound (macro moves, not algo skill)
      ├─ Statistical significance (is improvement real or noise?)
      └─ Cost-benefit analysis (improvement worth the complexity?)
```

**Interaction:** Monitor execution → Measure slippage → Decompose components → Identify cost drivers → Adjust strategy → Benchmark against peers → Iterate.

## 5. Mini-Project
Implement comprehensive slippage analysis comparing benchmarks and order types:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# Simulate realistic order execution data with multiple benchmarks
np.random.seed(42)
n_trades = 500

# Generate trade characteristics
trades = pd.DataFrame({
    'trade_id': range(n_trades),
    'stock': np.random.choice(['AAPL', 'MSFT', 'GOOG', 'AMZN'], n_trades),
    'order_size_pct_vol': np.random.lognormal(-1, 1.5, n_trades),  # % of daily volume
    'order_type': np.random.choice(['market', 'limit', 'algo_vwap', 'algo_twap'], n_trades),
    'time_of_day': np.random.choice(['open', 'midday', 'close'], n_trades),
})

# Generate price data (simulated)
trades['arrival_price'] = 100 + np.random.normal(0, 5, n_trades)
trades['vwap'] = trades['arrival_price'] + np.random.normal(0, 0.5, n_trades)
trades['twap'] = trades['arrival_price'] + np.random.normal(0, 0.7, n_trades)

# Generate execution prices (depends on order type and market condition)
spread_by_stock = {'AAPL': 0.02, 'MSFT': 0.03, 'GOOG': 0.04, 'AMZN': 0.025}
impact_by_size = trades['order_size_pct_vol'].apply(lambda x: 0.0002 * np.sqrt(x))

execution_prices = []
for idx, row in trades.iterrows():
    spread = spread_by_stock[row['stock']]
    impact = impact_by_size.iloc[idx]
    
    # Order type effect
    if row['order_type'] == 'market':
        # Market orders: Pay spread + impact
        exec_price = row['arrival_price'] + 0.5 * spread + impact
    elif row['order_type'] == 'limit':
        # Limit orders: May miss (50% fill rate assumed), but get better price
        exec_price = row['arrival_price'] - 0.5 * spread + 0.3 * impact
    elif row['order_type'] == 'algo_vwap':
        # VWAP algo: Tracks VWAP, half spread cost
        exec_price = row['vwap'] + 0.5 * 0.5 * spread + 0.5 * impact
    else:  # algo_twap
        # TWAP algo: Simple time-weighted, pays spread
        exec_price = row['twap'] + 0.7 * 0.5 * spread + 0.7 * impact
    
    execution_prices.append(exec_price)

trades['execution_price'] = execution_prices

# Calculate slippage vs different benchmarks
trades['slippage_vs_arrival'] = (trades['execution_price'] - trades['arrival_price']) / trades['arrival_price']
trades['slippage_vs_vwap'] = (trades['execution_price'] - trades['vwap']) / trades['vwap']
trades['slippage_vs_twap'] = (trades['execution_price'] - trades['twap']) / trades['twap']

# Convert to basis points
trades['slippage_ap_bps'] = trades['slippage_vs_arrival'] * 10000
trades['slippage_vwap_bps'] = trades['slippage_vs_vwap'] * 10000
trades['slippage_twap_bps'] = trades['slippage_vs_twap'] * 10000

# Estimate cost components
trades['spread_cost'] = spread_by_stock[trades['stock'].iloc[0]] / (2 * trades['arrival_price'])
trades['impact_cost'] = impact_by_size
trades['commission_cost'] = 0.0005  # 0.5 bps

print("="*100)
print("SLIPPAGE ANALYSIS")
print("="*100)

print(f"\nStep 1: Summary Statistics")
print(f"-" * 50)
print(f"Total trades: {n_trades}")
print(f"By order type:")
print(trades['order_type'].value_counts())
print(f"\nBy stock:")
print(trades['stock'].value_counts())

print(f"\nStep 2: Slippage Statistics (Basis Points)")
print(f"-" * 50)

slippage_stats = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
    'vs Arrival Price': [
        trades['slippage_ap_bps'].mean(),
        trades['slippage_ap_bps'].median(),
        trades['slippage_ap_bps'].std(),
        trades['slippage_ap_bps'].min(),
        trades['slippage_ap_bps'].max(),
        trades['slippage_ap_bps'].quantile(0.25),
        trades['slippage_ap_bps'].quantile(0.75),
    ],
    'vs VWAP': [
        trades['slippage_vwap_bps'].mean(),
        trades['slippage_vwap_bps'].median(),
        trades['slippage_vwap_bps'].std(),
        trades['slippage_vwap_bps'].min(),
        trades['slippage_vwap_bps'].max(),
        trades['slippage_vwap_bps'].quantile(0.25),
        trades['slippage_vwap_bps'].quantile(0.75),
    ],
    'vs TWAP': [
        trades['slippage_twap_bps'].mean(),
        trades['slippage_twap_bps'].median(),
        trades['slippage_twap_bps'].std(),
        trades['slippage_twap_bps'].min(),
        trades['slippage_twap_bps'].max(),
        trades['slippage_twap_bps'].quantile(0.25),
        trades['slippage_twap_bps'].quantile(0.75),
    ],
})

print(slippage_stats.to_string(index=False))

print(f"\nStep 3: Slippage by Order Type")
print(f"-" * 50)

order_type_slippage = trades.groupby('order_type')[['slippage_ap_bps', 'slippage_vwap_bps']].agg(['mean', 'std', 'count'])
print(order_type_slippage.round(2))

print(f"\nStep 4: Slippage by Stock")
print(f"-" * 50)

stock_slippage = trades.groupby('stock')[['slippage_ap_bps', 'order_size_pct_vol']].agg(['mean', 'std'])
print(stock_slippage.round(2))

print(f"\nStep 5: Slippage by Order Size (Correlation)")
print(f"-" * 50)

corr_size_slippage_ap = trades['order_size_pct_vol'].corr(trades['slippage_ap_bps'])
corr_size_slippage_vwap = trades['order_size_pct_vol'].corr(trades['slippage_vwap_bps'])
print(f"Correlation (order size vs slippage vs AP): {corr_size_slippage_ap:.3f}")
print(f"Correlation (order size vs slippage vs VWAP): {corr_size_slippage_vwap:.3f}")
print(f"Interpretation: {'Strong' if abs(corr_size_slippage_ap) > 0.5 else 'Moderate' if abs(corr_size_slippage_ap) > 0.3 else 'Weak'} relationship")

print(f"\nStep 6: Cost Component Breakdown")
print(f"-" * 50)

avg_spread_cost = trades['spread_cost'].mean() * 10000
avg_impact_cost = trades['impact_cost'].mean() * 10000
avg_commission = 0.5

print(f"Average spread cost: {avg_spread_cost:.2f} bps")
print(f"Average market impact: {avg_impact_cost:.2f} bps")
print(f"Average commission: {avg_commission:.2f} bps")
print(f"Total component estimate: {avg_spread_cost + avg_impact_cost + avg_commission:.2f} bps")
print(f"Actual average slippage (vs AP): {trades['slippage_ap_bps'].mean():.2f} bps")

print(f"\nStep 7: Slippage by Time of Day")
print(f"-" * 50)

tod_slippage = trades.groupby('time_of_day')[['slippage_ap_bps']].agg(['mean', 'std', 'count'])
print(tod_slippage.round(2))

# VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Distribution of slippage
ax = axes[0, 0]
ax.hist(trades['slippage_ap_bps'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(trades['slippage_ap_bps'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {trades['slippage_ap_bps'].mean():.1f} bps")
ax.set_xlabel('Slippage vs Arrival Price (bps)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Slippage')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Slippage by order type
ax = axes[0, 1]
order_type_means = trades.groupby('order_type')['slippage_ap_bps'].mean()
order_type_means.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_ylabel('Average Slippage (bps)')
ax.set_title('Slippage by Order Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='y')

# Plot 3: Slippage vs order size (scatter)
ax = axes[0, 2]
scatter = ax.scatter(trades['order_size_pct_vol'], trades['slippage_ap_bps'], 
                     c=trades['order_type'].astype('category').cat.codes, cmap='tab10', alpha=0.6, s=30)
ax.set_xlabel('Order Size (% of daily volume)')
ax.set_ylabel('Slippage (bps)')
ax.set_title('Slippage vs Order Size')
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Plot 4: Slippage by stock
ax = axes[1, 0]
stock_means = trades.groupby('stock')['slippage_ap_bps'].mean()
stock_means.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_ylabel('Average Slippage (bps)')
ax.set_title('Slippage by Stock')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='y')

# Plot 5: Box plot of slippage by order type
ax = axes[1, 1]
slippage_by_type = [trades[trades['order_type'] == ot]['slippage_ap_bps'].values 
                    for ot in trades['order_type'].unique()]
bp = ax.boxplot(slippage_by_type, labels=trades['order_type'].unique(), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Slippage (bps)')
ax.set_title('Slippage Distribution by Order Type')
ax.grid(alpha=0.3, axis='y')

# Plot 6: Slippage by time of day
ax = axes[1, 2]
tod_means = trades.groupby('time_of_day')['slippage_ap_bps'].mean()
tod_means.plot(kind='bar', ax=ax, color=['green', 'blue', 'red'], edgecolor='black', alpha=0.7)
ax.set_ylabel('Average Slippage (bps)')
ax.set_title('Slippage by Time of Day')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("KEY INSIGHTS")
print(f"="*100)
print(f"- Market orders: Highest slippage ({trades[trades['order_type']=='market']['slippage_ap_bps'].mean():.2f} bps) - pay spread + impact")
print(f"- VWAP/TWAP algos: Lower slippage ({trades[trades['order_type']=='algo_vwap']['slippage_ap_bps'].mean():.2f} bps) - track benchmark")
print(f"- Limit orders: Variable (some miss execution, some get better prices)")
print(f"- Larger orders: Higher slippage (sqrt-law market impact)")
print(f"- Best execution: VWAP benchmark typically optimal")
```

## 6. Challenge Round
- Calculate total cost of ownership: spread + market impact + commissions + opportunity cost for 3 trades
- Build slippage forecast model: Predict slippage based on stock, size, time-of-day, market vol
- Optimize order routing: Given 3 venues with different spreads/rebates, find least-cost execution
- Analyze "slippage drag" over time: Cumulative cost of 100 small trades vs. 10 large trades
- Identify worst-case scenarios: Which scenarios cause >100 bps slippage? Risk management implications

## 7. Key References
- [Bessembinder & Kaufman (1997), "A Comparison of Trade Execution Costs Across the Major U.S. Stock Markets," Journal of Financial Economics](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb02744.x) — Comprehensive slippage measurement framework
- [Hasbrouck & Seppi (2001), "Common Factors in Prices, Order Flows and Liquidity," Journal of Financial Economics](https://www.sciencedirect.com/science/article/pii/S0304405X01000748) — Slippage drivers and decomposition
- [Kissell & Glantz (2003), "Optimal Trading Strategies," AMACOM](https://www.amazon.com/Optimal-Trading-Strategies-Quantitative-Approaches/dp/0814407242) — Practical slippage minimization strategies
- [SEC Regulation NMS](https://www.sec.gov/rules/final/34-51808.pdf) — Regulatory framework for best execution and competition

---
**Status:** Critical execution metric (ubiquitous in trading operations) | **Complements:** Transaction Costs, Market Impact, Execution Algorithms, Venue Selection
