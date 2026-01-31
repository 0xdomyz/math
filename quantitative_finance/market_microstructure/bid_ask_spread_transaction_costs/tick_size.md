# Tick Size

## 1. Concept Skeleton
**Definition:** Minimum price increment for quoting securities, constraining the smallest bid-ask spread  
**Purpose:** Determines price grid granularity, affects spread width, liquidity provision incentives, and market quality  
**Prerequisites:** Bid-ask spread, order book mechanics, decimalization history

## 2. Comparative Framing
| Regime | Fractional (pre-2001) | Decimal ($0.01) | Sub-Penny ($0.0001) | Tick Pilot (2016-18) |
|--------|----------------------|-----------------|---------------------|----------------------|
| **Tick Size** | $0.0625 (1/16) | $0.01 (1¢) | $0.0001 | $0.05 (small caps) |
| **Spread Impact** | Wide spreads | 85% reduction | Ultra-tight | Widened for test stocks |
| **Market Maker Profit** | High | Compressed | Minimal | Increased |
| **Quote Updates** | Moderate | High (penny jumping) | Extreme | Lower |

## 3. Examples + Counterexamples

**Simple Example:**  
$100 stock, tick=$0.01 → Minimum spread=1 basis point. $1 stock, tick=$0.01 → Minimum spread=100 basis points

**Binding Constraint:**  
Order processing cost=$0.012, tick=$0.01 → Tick binds, cannot profitably quote tighter (dealer losses on every trade)

**Sub-Penny Exception:**  
Options, currencies, bonds quote in finer increments. FX can quote 5 decimal places ($1.23456/€), spreads <0.1 basis points

## 4. Layer Breakdown
```
Tick Size Economics:
├─ Historical Evolution:
│   ├─ Pre-2001: Fractional pricing (1/16=$0.0625)
│   │   - Wide spreads, high dealer profits
│   │   - Limited price competition
│   ├─ 2001: Decimalization mandated by SEC
│   │   - Tick=$0.01 for stocks >$1
│   │   - Spread compression by 30-40%
│   ├─ 2016-2018: Tick Size Pilot
│   │   - Test Group 1: $0.05 tick (small caps)
│   │   - Control group: $0.01 tick
│   └─ Result: Wider ticks did NOT improve liquidity
├─ Economic Effects:
│   ├─ Spread Constraint: Quoted Spread ≥ Tick Size
│   ├─ Price Improvement: Tick determines minimum improvement
│   ├─ Time Priority: Finer ticks → more penny jumping
│   └─ Market Maker Incentives: Wider ticks → higher profits
├─ Relative Tick Size (Harris 1994):
│   ├─ Definition: Tick Size / Stock Price
│   ├─ Low-Priced Stocks: Tick is larger % (binding constraint)
│   ├─ High-Priced Stocks: Tick is smaller % (not binding)
│   └─ Example: $1 stock (1% tick) vs $100 stock (0.01% tick)
├─ Trade-off:
│   ├─ Larger Ticks:
│   │   ✓ Incentivizes market making (higher profit per trade)
│   │   ✓ Reduces quote flickering, HFT quote stuffing
│   │   ✗ Wider spreads (higher transaction costs for investors)
│   │   ✗ Less price competition
│   └─ Smaller Ticks:
│       ✓ Tighter spreads (better prices for investors)
│       ✓ More precise price discovery
│       ✗ Lower market maker profits (less liquidity provision)
│       ✗ Increased quote traffic (penny jumping wars)
└─ Optimal Tick Size Debate:
    ├─ Academic View: Tick should vary by stock characteristics
    ├─ Practitioner View: Uniform tick simplifies trading
    ├─ Regulatory Experiments: Tick Size Pilot, Canadian experience
    └─ Cross-Asset Variation: Equities ($0.01), options ($0.05), bonds ($0.01-$0.03125)
```

**Interaction:** Tick size sets price grid → constrains spread → affects market maker incentives → determines liquidity

## 5. Mini-Project
Analyze tick size impact on spreads and market quality:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Simulate order books under different tick size regimes
n_stocks = 100
base_prices = np.logspace(np.log10(1), np.log10(500), n_stocks)  # $1 to $500

# Tick size regimes
tick_regimes = {
    'Fractional (pre-2001)': 0.0625,
    'Decimal (2001+)': 0.01,
    'Tick Pilot ($0.05)': 0.05,
    'Sub-Penny': 0.0001
}

# Fundamental spread determinants (independent of tick size)
# Based on: order processing, inventory, adverse selection
base_spread_cost = 0.02  # $0.02 base spread from economic costs
volatility_factor = np.random.uniform(0.5, 2, n_stocks)  # Volatility multiplier
volume_factor = np.random.uniform(0.3, 3, n_stocks)  # Volume effect (higher volume → tighter)

# Economic spread (what spread would be without tick constraints)
economic_spreads = base_spread_cost * volatility_factor / np.sqrt(volume_factor)

results = {}

for regime_name, tick_size in tick_regimes.items():
    # Quoted spread = max(economic spread, tick size)
    quoted_spreads = np.maximum(economic_spreads, tick_size)
    
    # Percentage spreads
    pct_spreads = (quoted_spreads / base_prices) * 100
    
    # Check if tick binds
    tick_binds = (quoted_spreads == tick_size)
    bind_rate = tick_binds.mean()
    
    # Relative tick size
    relative_tick = (tick_size / base_prices) * 100
    
    results[regime_name] = {
        'tick_size': tick_size,
        'quoted_spreads': quoted_spreads,
        'pct_spreads': pct_spreads,
        'tick_binds': tick_binds,
        'bind_rate': bind_rate,
        'relative_tick': relative_tick
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Spread vs price for different tick regimes
colors = {'Fractional (pre-2001)': 'gray', 'Decimal (2001+)': 'blue', 
          'Tick Pilot ($0.05)': 'red', 'Sub-Penny': 'green'}

for regime_name, color in colors.items():
    spreads = results[regime_name]['quoted_spreads']
    axes[0, 0].scatter(base_prices, spreads, alpha=0.4, s=20, 
                      c=color, label=regime_name)

# Plot economic spread for reference
axes[0, 0].scatter(base_prices, economic_spreads, alpha=0.3, s=10, 
                  c='black', label='Economic Spread (no tick constraint)', marker='x')

axes[0, 0].set_xlabel('Stock Price ($)')
axes[0, 0].set_ylabel('Quoted Spread ($)')
axes[0, 0].set_title('Spread vs Price: Tick Size Regimes')
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3, which='both')

print("Tick Size Regime Analysis:")
print("=" * 70)
for regime_name, data in results.items():
    print(f"\n{regime_name} (Tick=${data['tick_size']}):")
    print(f"  Tick Binding Rate: {data['bind_rate']*100:.1f}%")
    print(f"  Mean Quoted Spread: ${data['quoted_spreads'].mean():.4f}")
    print(f"  Mean % Spread: {data['pct_spreads'].mean():.3f}%")
    print(f"  Median % Spread: {np.median(data['pct_spreads']):.3f}%")

# Plot 2: Percentage spread by price bucket
price_bins = [1, 5, 10, 25, 50, 100, 500]
price_labels = ['$1-5', '$5-10', '$10-25', '$25-50', '$50-100', '$100+']

decimal_regime = results['Decimal (2001+)']

bin_spreads = []
for i in range(len(price_bins) - 1):
    mask = (base_prices >= price_bins[i]) & (base_prices < price_bins[i+1])
    bin_spreads.append(decimal_regime['pct_spreads'][mask])

bp = axes[0, 1].boxplot(bin_spreads, labels=price_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

axes[0, 1].set_xlabel('Price Range')
axes[0, 1].set_ylabel('Spread (%)')
axes[0, 1].set_title('Percentage Spread by Price (Decimal Regime)')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3, axis='y')

print(f"\nPercentage Spread by Price Range (Decimal Regime):")
for i, label in enumerate(price_labels):
    if len(bin_spreads[i]) > 0:
        print(f"  {label}: {np.median(bin_spreads[i]):.3f}% median")

# Plot 3: Tick binding analysis
# For low-priced stocks, tick is more likely to bind
decimal_data = results['Decimal (2001+)']
pilot_data = results['Tick Pilot ($0.05)']

# Scatter: relative tick size vs whether tick binds
axes[1, 0].scatter(decimal_data['relative_tick'], decimal_data['tick_binds'], 
                  alpha=0.5, s=50, label='Decimal ($0.01)')
axes[1, 0].scatter(pilot_data['relative_tick'], pilot_data['tick_binds'] + 0.05,  # Offset for visibility
                  alpha=0.5, s=50, label='Tick Pilot ($0.05)', marker='s')

axes[1, 0].set_xlabel('Relative Tick Size (%)')
axes[1, 0].set_ylabel('Tick Binds (1=Yes, 0=No)')
axes[1, 0].set_title('Tick Constraint by Relative Tick Size')
axes[1, 0].set_xscale('log')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Calculate threshold where tick binds
decimal_bind_threshold = decimal_data['relative_tick'][decimal_data['tick_binds']].min()
print(f"\nTick Binding Threshold:")
print(f"Decimal regime: Tick binds when relative size > {decimal_bind_threshold:.3f}%")
print(f"Corresponds to prices below ${0.01/decimal_bind_threshold*100:.2f}")

# Plot 4: Liquidity provision incentives
# Market maker profit = (Spread / 2) - Order Processing Cost
order_processing_cost = 0.008  # $0.008 per trade

regime_names = list(tick_regimes.keys())
mm_profits = []
trade_incentives = []

for regime_name in regime_names:
    spreads = results[regime_name]['quoted_spreads']
    # Market maker earns half the spread
    revenue_per_trade = spreads / 2
    profit_per_trade = revenue_per_trade - order_processing_cost
    
    # Count profitable opportunities
    profitable = (profit_per_trade > 0).sum()
    profitable_pct = profitable / len(profit_per_trade) * 100
    
    mm_profits.append(profit_per_trade.mean())
    trade_incentives.append(profitable_pct)

x_pos = np.arange(len(regime_names))
width = 0.35

ax1 = axes[1, 1]
ax2 = ax1.twinx()

bars1 = ax1.bar(x_pos - width/2, mm_profits, width, label='Avg MM Profit', 
                color='green', alpha=0.7)
bars2 = ax2.bar(x_pos + width/2, trade_incentives, width, label='Profitable %', 
                color='blue', alpha=0.7)

ax1.axhline(0, color='black', linewidth=0.5)
ax1.set_xlabel('Tick Size Regime')
ax1.set_ylabel('Market Maker Profit per Trade ($)', color='green')
ax2.set_ylabel('% Stocks Profitable to Quote (%)', color='blue')
ax1.set_title('Market Making Incentives by Tick Regime')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(regime_names, rotation=20, ha='right', fontsize=8)
ax1.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='blue')

# Add value labels
for i, (profit, pct) in enumerate(zip(mm_profits, trade_incentives)):
    ax1.text(i - width/2, profit + 0.003, f'${profit:.3f}', 
            ha='center', va='bottom', fontsize=8, color='green')
    ax2.text(i + width/2, pct + 2, f'{pct:.0f}%', 
            ha='center', va='bottom', fontsize=8, color='blue')

ax1.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\nMarket Maker Incentives:")
print(f"(Assuming order processing cost = ${order_processing_cost})")
for i, regime_name in enumerate(regime_names):
    print(f"{regime_name}:")
    print(f"  Avg Profit: ${mm_profits[i]:.4f} per trade")
    print(f"  Profitable Stocks: {trade_incentives[i]:.1f}%")

# Analyze Tick Size Pilot impact
print(f"\nTick Size Pilot Analysis:")
decimal_mean = results['Decimal (2001+)']['quoted_spreads'].mean()
pilot_mean = results['Tick Pilot ($0.05)']['quoted_spreads'].mean()
spread_increase = ((pilot_mean - decimal_mean) / decimal_mean) * 100

print(f"Mean spread increase from $0.01 to $0.05 tick: {spread_increase:.1f}%")

decimal_pct = results['Decimal (2001+)']['pct_spreads'].mean()
pilot_pct = results['Tick Pilot ($0.05)']['pct_spreads'].mean()
pct_increase = ((pilot_pct - decimal_pct) / decimal_pct) * 100

print(f"Mean percentage spread increase: {pct_increase:.1f}%")

# Calculate investor cost
# Assume 1 round-trip trade per stock per day
annual_trading_days = 252
investor_cost_decimal = decimal_mean * 2 * annual_trading_days  # Round-trip, annual
investor_cost_pilot = pilot_mean * 2 * annual_trading_days

print(f"\nInvestor Transaction Cost (annual, per stock):")
print(f"Decimal regime: ${investor_cost_decimal:.2f}")
print(f"Tick Pilot regime: ${investor_cost_pilot:.2f}")
print(f"Additional cost: ${investor_cost_pilot - investor_cost_decimal:.2f} ({(investor_cost_pilot/investor_cost_decimal - 1)*100:.1f}%)")

# Statistical test: Did tick pilot significantly widen spreads?
t_stat, p_val = stats.ttest_ind(results['Decimal (2001+)']['quoted_spreads'],
                                 results['Tick Pilot ($0.05)']['quoted_spreads'])
print(f"\nStatistical Test (Decimal vs Tick Pilot):")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")
if p_val < 0.001:
    print("Result: Tick Pilot significantly widened spreads")
```

## 6. Challenge Round
Why did the SEC Tick Size Pilot fail to improve liquidity for small caps?
- **Market maker profits increased** (wider spreads), but **liquidity provision did not** (quote depth unchanged or declined)
- **Investor harm**: Transaction costs increased 10-15% for pilot stocks without offsetting benefits
- **Quote flickering persisted**: HFT firms still updated quotes rapidly despite wider ticks
- **Informed traders**: Wider spreads increased adverse selection (informed traders more profitable, discouraging uninformed flow)
- **Conclusion**: Tick size is not the primary constraint on small-cap liquidity; information asymmetry, low volume, and limited investor interest dominate

## 7. Key References
- [Harris (1994) - Minimum Price Variations and Quotation Sizes](https://www.jstor.org/stable/2962258)
- [SEC Tick Size Pilot Final Report (2018)](https://www.sec.gov/files/Tick%20Size%20Pilot%20Program%20and%20Final%20Assessment.pdf)
- [Angel (1997) - Tick Size, Share Prices, and Stock Splits](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb04811.x)
- [Bessembinder (2003) - Trade Execution Costs and Market Quality After Decimalization](https://www.jstor.org/stable/1262685)

---
**Status:** Price grid constraint | **Complements:** Bid-Ask Spread, Order Processing Costs, Market Quality
