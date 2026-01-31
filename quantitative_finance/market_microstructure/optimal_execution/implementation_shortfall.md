# Implementation Shortfall

## 1. Concept Skeleton
**Definition:** Transaction cost measurement comparing actual execution performance to hypothetical immediate execution at decision price (arrival price), capturing all costs of delayed/partial execution  
**Purpose:** Comprehensive TCA benchmark quantifying opportunity cost from not trading instantly, including market impact, timing risk, and unfilled quantity  
**Prerequisites:** Trade decision timestamp, arrival price, execution fills, market price evolution, delay cost modeling

## 2. Comparative Framing
| Metric | Implementation Shortfall | VWAP | TWAP | Effective Spread |
|--------|-------------------------|------|------|------------------|
| **Benchmark** | Arrival price | Volume-weighted | Time-weighted | Mid-price |
| **Cost Components** | Delay + impact + opportunity | Execution only | Execution only | Spread only |
| **Unrealized** | Included | Excluded | Excluded | Excluded |
| **Decision Focus** | Yes (when to trade) | No | No | No |

## 3. Examples + Counterexamples

**Simple Example:**  
Decide to buy 10,000 shares at $100. Execute 8,000 at avg $100.50, remaining 2,000 unfilled (price now $101). IS = $(0.50×8,000 + 1.00×2,000) = $6,000

**Failure Case:**  
Market moving in favorable direction. Slow VWAP execution captures improving prices, beats VWAP benchmark, but IS shows large opportunity cost from delay

**Edge Case:**  
Order canceled mid-execution (risk management). All unexecuted shares counted as opportunity cost at current market price vs decision price

## 4. Layer Breakdown
```
Implementation Shortfall Framework:
├─ Cost Components:
│   ├─ Decision (Delay) Cost:
│   │   ├─ Time between decision and first execution
│   │   ├─ Price movement during evaluation
│   │   ├─ Cost = ΔPrice × Total_shares
│   │   ├─ Example: Decide at $100, start at $100.10
│   │   └─ Often smallest component (<1 bps)
│   ├─ Execution (Impact) Cost:
│   │   ├─ Realized slippage during trading
│   │   ├─ Difference: Avg_execution_price - Arrival_price
│   │   ├─ Includes: market impact, spread, timing
│   │   ├─ Example: Fill at $100.50 vs arrival $100.10
│   │   └─ Typically largest component (5-20 bps)
│   ├─ Opportunity Cost:
│   │   ├─ Unfilled shares at end of period
│   │   ├─ Cost = (Current_price - Arrival_price) × Unfilled
│   │   ├─ Captures: decision to not complete
│   │   ├─ Example: 2,000 unfilled, price at $101 vs $100
│   │   └─ Can dominate if large unfilled + trending
│   └─ Total Implementation Shortfall:
│       ├─ Sum of all components
│       ├─ Expressed as: Dollars, bps of value, bps of volume
│       ├─ Positive = cost (worse than immediate)
│       ├─ Negative = savings (better than immediate)
│       └─ Benchmark: What if traded all instantly?
├─ Detailed Calculation:
│   ├─ Notation:
│   │   ├─ P₀: Arrival price (decision price)
│   │   ├─ P₁: Price at first execution
│   │   ├─ Pᵢ: Fill price of i-th trade
│   │   ├─ Vᵢ: Volume of i-th trade
│   │   ├─ Pₑ: Price at end of period
│   │   ├─ X: Target order size
│   │   └─ X_filled: Actual filled quantity
│   ├─ Delay Cost:
│   │   ├─ Formula: (P₁ - P₀) × X
│   │   ├─ For buyer: positive if price rose
│   │   ├─ For seller: positive if price fell
│   │   └─ Sign convention: cost always positive
│   ├─ Execution Cost:
│   │   ├─ Formula: Σᵢ(Pᵢ - P₁) × Vᵢ
│   │   ├─ Volume-weighted slippage vs first execution
│   │   ├─ Captures market impact over time
│   │   └─ Includes both permanent and temporary
│   ├─ Opportunity Cost:
│   │   ├─ Formula: (Pₑ - P₁) × (X - X_filled)
│   │   ├─ Unfilled quantity valued at end price
│   │   ├─ Controversial: assumes would have filled at Pₑ
│   │   └─ Alternative: exclude or use benchmark price
│   └─ Total IS (basis points):
│       ├─ IS_bps = (Total_cost / (P₀ × X)) × 10000
│       ├─ Normalized by order value
│       ├─ Comparable across orders/stocks
│       └─ Typical range: -5 to +50 bps
├─ Decomposition & Attribution:
│   ├─ Market Impact:
│   │   ├─ Permanent: Price level shift from trade
│   │   ├─ Temporary: Transient supply/demand pressure
│   │   ├─ Estimated from regression: ΔP ~ Volume
│   │   └─ Controllable by execution strategy
│   ├─ Timing Risk:
│   │   ├─ Volatility-driven price moves
│   │   ├─ Unrelated to our trading
│   │   ├─ Estimated from: σ × √(time) × Z-score
│   │   └─ Unavoidable but manageable via speed
│   ├─ Spread Cost:
│   │   ├─ Bid-ask crossing fees
│   │   ├─ Can separate aggressive vs passive
│   │   ├─ Baseline cost: ~half-spread × shares
│   │   └─ Reducible via patient limit orders
│   └─ Adverse Selection:
│       ├─ Trading against informed counterparties
│       ├─ Price moves against us post-trade
│       ├─ Difficult to measure directly
│       └─ Proxy: Post-trade price reversion analysis
├─ Extensions & Variations:
│   ├─ Arrival Price Benchmark:
│   │   ├─ Price when order received by desk
│   │   ├─ Most common IS definition
│   │   ├─ Separates decision from execution
│   │   └─ Problem: Gaming via delayed order submission
│   ├─ Decision Price Benchmark:
│   │   ├─ Price when PM decides to trade
│   │   ├─ Includes desk delay cost
│   │   ├─ More complete cost picture
│   │   └─ Requires accurate timestamp
│   ├─ Pre-Trade Benchmark:
│   │   ├─ Price before any information leakage
│   │   ├─ Idealized reference point
│   │   ├─ Often unavailable/unobservable
│   │   └─ Theoretical best case
│   └─ Interval VWAP Hybrid:
│       ├─ Use VWAP during execution window
│       ├─ Opportunity cost from VWAP post-window
│       ├─ Reduces sensitivity to single-point arrival
│       └─ More robust to outliers
├─ Statistical Analysis:
│   ├─ IS Distribution:
│   │   ├─ Typically right-skewed (fat right tail)
│   │   ├─ Mean IS: 10-20 bps for large orders
│   │   ├─ Median lower than mean (outliers)
│   │   └─ Standard deviation: 20-50 bps
│   ├─ Hit Rate:
│   │   ├─ % of orders with negative IS (beat arrival)
│   │   ├─ Typical: 40-50% hit rate
│   │   ├─ Higher for smaller orders
│   │   └─ Asymmetry: Small wins, large losses
│   ├─ Regression Analysis:
│   │   ├─ IS ~ f(Order_size, Volatility, Spread, ADV)
│   │   ├─ Control for market conditions
│   │   ├─ Identify systematic biases
│   │   └─ Optimize execution strategy
│   └─ Time Series:
│       ├─ Track IS by month/quarter
│       ├─ Detect performance degradation
│       ├─ Seasonal patterns (rebalance dates)
│       └─ Regulatory reporting (MiFID II)
├─ Practical Implementation:
│   ├─ Data Requirements:
│   │   ├─ High-quality timestamps (microsecond)
│   │   ├─ Arrival price: BBO or mid at decision time
│   │   ├─ All fill reports with prices/sizes
│   │   ├─ Market prices at calculation end
│   │   └─ Order lifecycle events (submit/cancel/fill)
│   ├─ Real-Time Monitoring:
│   │   ├─ Track running IS during execution
│   │   ├─ Alert if exceeding threshold
│   │   ├─ Adaptive strategy adjustment
│   │   └─ Dashboard for traders
│   ├─ Post-Trade Analysis:
│   │   ├─ Automated TCA reports
│   │   ├─ Peer comparison (same stock/day)
│   │   ├─ Historical trends
│   │   └─ Client reporting
│   └─ Regulatory Use:
│       ├─ MiFID II best execution reports
│       ├─ Form ADV disclosures
│       ├─ Internal audit trail
│       └─ Demonstrate duty of care
└─ Controversies & Limitations:
    ├─ Opportunity Cost Debate:
    │   ├─ Should unfilled be included?
    │   ├─ Gaming: Cancel losing orders to reduce IS
    │   ├─ Alternative: Fixed time horizon regardless of fills
    │   └─ Exclude if order intentionally canceled
    ├─ Arrival Price Gaming:
    │   ├─ Delay order submission if price improving
    │   ├─ Submit immediately if deteriorating
    │   ├─ Reduces measured IS artificially
    │   └─ Solution: Decision price benchmark
    ├─ Asymmetry:
    │   ├─ Favorable price moves → negative IS (good)
    │   ├─ Adverse moves → cancel order, small IS
    │   ├─ Biased measure of true cost
    │   └─ Need to adjust for optionality
    └─ Market Conditions:
        ├─ High volatility → high IS unavoidable
        ├─ Flash crash → extreme outliers
        ├─ Low liquidity → completion impossible
        └─ Needs context-adjusted benchmarking
```

**Interaction:** Order decision → Arrival price capture → Execution monitoring → Fill aggregation → Opportunity cost calculation → Performance attribution

## 5. Mini-Project
Calculate implementation shortfall with comprehensive analysis:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    decision_time: float
    arrival_price: float
    
@dataclass
class Fill:
    """Represents an execution fill"""
    time: float
    price: float
    quantity: int

class ImplementationShortfallCalculator:
    """Calculate implementation shortfall with full decomposition"""
    
    def __init__(self, order: Order, fills: List[Fill], end_price: float):
        self.order = order
        self.fills = fills
        self.end_price = end_price
        
        # Sort fills by time
        self.fills.sort(key=lambda x: x.time)
        
        # Calculate metrics
        self._calculate()
    
    def _calculate(self):
        """Compute all IS components"""
        P0 = self.order.arrival_price
        X = self.order.quantity
        side = self.order.side
        
        # Direction multiplier (cost always positive)
        direction = 1 if side == 'buy' else -1
        
        # Total filled quantity
        X_filled = sum(f.quantity for f in self.fills)
        X_unfilled = X - X_filled
        
        if len(self.fills) == 0:
            # No fills - pure opportunity cost
            self.delay_cost = 0
            self.execution_cost = 0
            self.opportunity_cost = direction * (self.end_price - P0) * X
            self.total_cost = self.opportunity_cost
            
            self.avg_fill_price = None
            self.first_fill_price = None
        else:
            # First fill price
            P1 = self.fills[0].price
            
            # Average fill price
            total_value = sum(f.price * f.quantity for f in self.fills)
            self.avg_fill_price = total_value / X_filled
            self.first_fill_price = P1
            
            # 1. Delay cost: Arrival to first fill
            self.delay_cost = direction * (P1 - P0) * X
            
            # 2. Execution cost: Average fill vs first fill
            self.execution_cost = direction * (self.avg_fill_price - P1) * X_filled
            
            # 3. Opportunity cost: Unfilled quantity
            if X_unfilled > 0:
                self.opportunity_cost = direction * (self.end_price - P1) * X_unfilled
            else:
                self.opportunity_cost = 0
            
            # Total IS
            self.total_cost = self.delay_cost + self.execution_cost + self.opportunity_cost
        
        # Express in bps
        order_value = P0 * X
        self.delay_cost_bps = (self.delay_cost / order_value) * 10000
        self.execution_cost_bps = (self.execution_cost / order_value) * 10000
        self.opportunity_cost_bps = (self.opportunity_cost / order_value) * 10000
        self.total_cost_bps = (self.total_cost / order_value) * 10000
        
        # Filled percentage
        self.fill_rate = (X_filled / X) * 100
    
    def summary(self):
        """Return summary dictionary"""
        return {
            'order_size': self.order.quantity,
            'filled': sum(f.quantity for f in self.fills),
            'fill_rate_pct': self.fill_rate,
            'arrival_price': self.order.arrival_price,
            'avg_fill_price': self.avg_fill_price,
            'end_price': self.end_price,
            'delay_cost': self.delay_cost,
            'execution_cost': self.execution_cost,
            'opportunity_cost': self.opportunity_cost,
            'total_cost': self.total_cost,
            'delay_cost_bps': self.delay_cost_bps,
            'execution_cost_bps': self.execution_cost_bps,
            'opportunity_cost_bps': self.opportunity_cost_bps,
            'total_cost_bps': self.total_cost_bps
        }

def simulate_execution_scenarios():
    """Simulate various execution scenarios"""
    np.random.seed(42)
    
    scenarios = []
    
    # Scenario 1: Perfect execution (complete, no delay)
    order1 = Order('AAPL', 'buy', 10000, 0.0, 100.0)
    fills1 = [Fill(0.0, 100.01, 10000)]  # Small spread cost
    end_price1 = 100.05
    
    is1 = ImplementationShortfallCalculator(order1, fills1, end_price1)
    scenarios.append(('Perfect Execution', is1))
    
    # Scenario 2: Gradual execution with impact
    order2 = Order('AAPL', 'buy', 10000, 0.0, 100.0)
    fills2 = [
        Fill(0.1, 100.05, 2000),
        Fill(0.2, 100.10, 2000),
        Fill(0.3, 100.15, 2000),
        Fill(0.4, 100.20, 2000),
        Fill(0.5, 100.25, 2000)
    ]
    end_price2 = 100.30
    
    is2 = ImplementationShortfallCalculator(order2, fills2, end_price2)
    scenarios.append(('Gradual (Full)', is2))
    
    # Scenario 3: Partial fill, adverse price move
    order3 = Order('AAPL', 'buy', 10000, 0.0, 100.0)
    fills3 = [
        Fill(0.1, 100.05, 2000),
        Fill(0.2, 100.10, 2000),
        Fill(0.3, 100.20, 2000)
    ]
    end_price3 = 101.00  # Price ran away
    
    is3 = ImplementationShortfallCalculator(order3, fills3, end_price3)
    scenarios.append(('Partial (Adverse)', is3))
    
    # Scenario 4: Partial fill, favorable price move
    order4 = Order('AAPL', 'sell', 10000, 0.0, 100.0)
    fills4 = [
        Fill(0.1, 99.95, 3000),
        Fill(0.2, 99.90, 3000)
    ]
    end_price4 = 99.50  # Price fell (good for seller)
    
    is4 = ImplementationShortfallCalculator(order4, fills4, end_price4)
    scenarios.append(('Partial (Favorable)', is4))
    
    # Scenario 5: High impact, slow execution
    order5 = Order('AAPL', 'buy', 10000, 0.0, 100.0)
    fills5 = [
        Fill(0.05, 100.10, 1000),  # Delay cost
        Fill(0.1, 100.30, 2000),   # Large impact
        Fill(0.15, 100.50, 2000),
        Fill(0.2, 100.60, 2000),
        Fill(0.25, 100.70, 2000),
        Fill(0.3, 100.80, 1000)
    ]
    end_price5 = 100.90
    
    is5 = ImplementationShortfallCalculator(order5, fills5, end_price5)
    scenarios.append(('High Impact', is5))
    
    # Scenario 6: No fills (canceled order)
    order6 = Order('AAPL', 'buy', 10000, 0.0, 100.0)
    fills6 = []
    end_price6 = 101.50  # Missed opportunity
    
    is6 = ImplementationShortfallCalculator(order6, fills6, end_price6)
    scenarios.append(('Canceled', is6))
    
    return scenarios

# Run simulations
print("="*80)
print("IMPLEMENTATION SHORTFALL ANALYSIS")
print("="*80)

scenarios = simulate_execution_scenarios()

# Summary table
results = []
for name, is_calc in scenarios:
    summary = is_calc.summary()
    summary['scenario'] = name
    results.append(summary)

df_results = pd.DataFrame(results)

print("\nScenario Comparison:")
display_cols = ['scenario', 'fill_rate_pct', 'avg_fill_price', 'end_price', 
                'delay_cost_bps', 'execution_cost_bps', 'opportunity_cost_bps', 'total_cost_bps']
print(df_results[display_cols].to_string(index=False))

print("\n" + "="*80)
print("DETAILED BREAKDOWN")
print("="*80)

for name, is_calc in scenarios[:3]:  # Show first 3 in detail
    print(f"\n{name}:")
    summary = is_calc.summary()
    print(f"  Order: {summary['order_size']:,} shares at ${summary['arrival_price']:.2f}")
    print(f"  Filled: {summary['filled']:,} shares ({summary['fill_rate_pct']:.0f}%)")
    if summary['avg_fill_price']:
        print(f"  Avg fill: ${summary['avg_fill_price']:.2f}")
    print(f"  End price: ${summary['end_price']:.2f}")
    print(f"  Cost Breakdown:")
    print(f"    Delay:       ${summary['delay_cost']:>8,.2f} ({summary['delay_cost_bps']:>6.1f} bps)")
    print(f"    Execution:   ${summary['execution_cost']:>8,.2f} ({summary['execution_cost_bps']:>6.1f} bps)")
    print(f"    Opportunity: ${summary['opportunity_cost']:>8,.2f} ({summary['opportunity_cost_bps']:>6.1f} bps)")
    print(f"    Total IS:    ${summary['total_cost']:>8,.2f} ({summary['total_cost_bps']:>6.1f} bps)")

# Statistical analysis across hypothetical orders
print("\n" + "="*80)
print("PORTFOLIO IS STATISTICS")
print("="*80)

# Simulate 100 orders
np.random.seed(42)
portfolio_is = []

for i in range(100):
    order = Order('STOCK', 'buy', np.random.randint(5000, 20000), 0.0, 100.0)
    
    # Random execution
    n_fills = np.random.randint(3, 10)
    fills = []
    cumulative = 0
    
    for j in range(n_fills):
        if cumulative >= order.quantity:
            break
        
        fill_size = min(np.random.randint(500, 3000), order.quantity - cumulative)
        fill_price = 100.0 + np.random.normal(0.1 * (j+1), 0.05)
        fill_time = j * 0.1
        
        fills.append(Fill(fill_time, fill_price, fill_size))
        cumulative += fill_size
    
    # Random end price
    end_price = 100.0 + np.random.normal(0.3, 0.2)
    
    is_calc = ImplementationShortfallCalculator(order, fills, end_price)
    portfolio_is.append(is_calc.summary())

df_portfolio = pd.DataFrame(portfolio_is)

print(f"\nPortfolio IS Statistics (N={len(df_portfolio)}):")
print(f"  Mean IS: {df_portfolio['total_cost_bps'].mean():.2f} bps")
print(f"  Median IS: {df_portfolio['total_cost_bps'].median():.2f} bps")
print(f"  Std Dev: {df_portfolio['total_cost_bps'].std():.2f} bps")
print(f"  Min IS: {df_portfolio['total_cost_bps'].min():.2f} bps")
print(f"  Max IS: {df_portfolio['total_cost_bps'].max():.2f} bps")
print(f"  Hit rate (IS < 0): {(df_portfolio['total_cost_bps'] < 0).sum()/len(df_portfolio)*100:.1f}%")

print(f"\nComponent Breakdown:")
print(f"  Avg delay: {df_portfolio['delay_cost_bps'].mean():.2f} bps")
print(f"  Avg execution: {df_portfolio['execution_cost_bps'].mean():.2f} bps")
print(f"  Avg opportunity: {df_portfolio['opportunity_cost_bps'].mean():.2f} bps")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: IS by scenario
axes[0, 0].barh(df_results['scenario'], df_results['total_cost_bps'], alpha=0.7)
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title('Total IS by Scenario')
axes[0, 0].set_xlabel('Implementation Shortfall (bps)')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Component breakdown
scenarios_display = df_results['scenario'].values[:5]  # First 5
components = df_results[['delay_cost_bps', 'execution_cost_bps', 'opportunity_cost_bps']].values[:5]

x = np.arange(len(scenarios_display))
width = 0.25

axes[0, 1].bar(x - width, components[:, 0], width, label='Delay', alpha=0.7)
axes[0, 1].bar(x, components[:, 1], width, label='Execution', alpha=0.7)
axes[0, 1].bar(x + width, components[:, 2], width, label='Opportunity', alpha=0.7)

axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(scenarios_display, rotation=45, ha='right')
axes[0, 1].set_title('IS Component Breakdown')
axes[0, 1].set_ylabel('Cost (bps)')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Fill rate vs IS
axes[0, 2].scatter(df_results['fill_rate_pct'], df_results['total_cost_bps'], s=100, alpha=0.6)
for idx, row in df_results.iterrows():
    axes[0, 2].annotate(row['scenario'], (row['fill_rate_pct'], row['total_cost_bps']),
                        fontsize=8, ha='left')
axes[0, 2].set_title('Fill Rate vs Implementation Shortfall')
axes[0, 2].set_xlabel('Fill Rate (%)')
axes[0, 2].set_ylabel('Total IS (bps)')
axes[0, 2].grid(alpha=0.3)

# Plot 4: Portfolio IS distribution
axes[1, 0].hist(df_portfolio['total_cost_bps'], bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero IS')
axes[1, 0].axvline(df_portfolio['total_cost_bps'].mean(), color='green', linestyle='--',
                   linewidth=2, label=f"Mean: {df_portfolio['total_cost_bps'].mean():.1f} bps")
axes[1, 0].set_title('Portfolio IS Distribution')
axes[1, 0].set_xlabel('Implementation Shortfall (bps)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Component correlation
component_cols = ['delay_cost_bps', 'execution_cost_bps', 'opportunity_cost_bps']
corr = df_portfolio[component_cols].corr()

im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(component_cols)))
axes[1, 1].set_yticks(range(len(component_cols)))
axes[1, 1].set_xticklabels(['Delay', 'Execution', 'Opportunity'], rotation=45)
axes[1, 1].set_yticklabels(['Delay', 'Execution', 'Opportunity'])
axes[1, 1].set_title('IS Component Correlation')

for i in range(len(component_cols)):
    for j in range(len(component_cols)):
        axes[1, 1].text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')

plt.colorbar(im, ax=axes[1, 1])

# Plot 6: IS over time (cumulative)
# Simulate time series
time_series_is = []
for t in range(50):
    order = Order('STOCK', 'buy', 10000, t, 100.0)
    
    # Market regime: trending upward
    trend = 0.01 * t
    fills = [Fill(t + 0.1, 100.0 + trend + np.random.normal(0.1, 0.05), 
                  np.random.randint(2000, 4000)) for _ in range(3)]
    end_price = 100.0 + trend + np.random.normal(0.2, 0.1)
    
    is_calc = ImplementationShortfallCalculator(order, fills, end_price)
    time_series_is.append(is_calc.summary()['total_cost_bps'])

cumulative_is = np.cumsum(time_series_is)

axes[1, 2].plot(cumulative_is, linewidth=2)
axes[1, 2].set_title('Cumulative IS Over Time')
axes[1, 2].set_xlabel('Order Number')
axes[1, 2].set_ylabel('Cumulative IS (bps)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. IS captures total cost of delayed/partial execution vs immediate trade")
print(f"2. Opportunity cost dominates when orders unfilled in trending markets")
print(f"3. Execution cost typically largest component for completed orders")
print(f"4. Delay cost usually smallest (<5 bps) with fast order routing")
print(f"5. Portfolio IS distribution right-skewed: many small wins, few large losses")
print(f"6. IS benchmark focuses traders on completion and timing, not just price")
```

## 6. Challenge Round
What are criticisms of implementation shortfall?
- **Opportunity cost controversy**: Including unfilled shares penalizes prudent risk management (canceling losing orders)
- **Gaming incentives**: Traders submit orders late (after price improves) to artificially lower IS
- **Asymmetry**: Negative IS (gains) from favorable moves vs positive IS (losses), creates option-like payoff favoring cancellation
- **Benchmark sensitivity**: Single arrival price point volatile, small timing differences cause large IS swings
- **Incomplete fills**: Should partial fills from risk limits be penalized same as execution failure?

How do practitioners address IS limitations?
- **Decision price**: Use PM decision time, not desk arrival, to prevent gaming
- **Fixed horizon**: Calculate IS at fixed time (e.g., T+30min) regardless of completion status
- **Conditional analysis**: Separate completed vs incomplete orders in reporting
- **Risk-adjusted IS**: Penalize variance not just expected cost (Almgren-Chriss style)
- **Hybrid benchmarks**: Use arrival price for filled portion, VWAP for unfilled

## 7. Key References
- [Perold (1988): Implementation Shortfall Paper](https://www.jstor.org/stable/4479000)
- [Wagner, Edwards (1993): Best Execution](https://www.cfapubs.org/doi/pdf/10.2469/faj.v49.n1.65)
- [Kissell, Malamut (2006): Algorithmic Decision-Making Framework](https://jpm.pm-research.com/content/32/4/41)

---
**Status:** Premier TCA benchmark for institutional trading | **Complements:** Almgren-Chriss, VWAP, Arrival Price, TCA
