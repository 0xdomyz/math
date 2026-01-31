# Order Processing Costs

## 1. Concept Skeleton
**Definition:** Fixed overhead expenses for clearing, settlement, systems infrastructure, and regulatory compliance per transaction  
**Purpose:** Explains minimum viable spread floor, scale economies in market making, tick size binding constraints  
**Prerequisites:** Market microstructure basics, bid-ask spread components, exchange operations

## 2. Comparative Framing
| Cost Component | Order Processing | Inventory Cost | Adverse Selection | Opportunity Cost |
|----------------|------------------|----------------|-------------------|------------------|
| **Nature** | Fixed per trade | Variable (risk) | Informational | Capital allocation |
| **Scaling** | Constant | Position-dependent | Information-dependent | Volume-dependent |
| **Typical %** | 10-20% of spread | 10-30% | 40-70% | Varies |
| **Reduction Strategy** | Automation, scale | Hedging | Screening | Turnover speed |

## 3. Examples + Counterexamples

**Simple Example:**  
Clearing fee=$0.002, exchange fee=$0.003, system cost=$0.005 → Order processing cost=$0.01 per trade (both sides)

**High-Frequency Trading:**  
100M trades/year, fixed cost=$1M → $0.01/trade. Competitor with 10M trades → $0.10/trade (10x disadvantage)

**Edge Case:**  
Penny stock (price=$0.50), tick size=$0.01 (2% spread). Order processing=$0.008 → 80% of minimum spread is overhead

## 4. Layer Breakdown
```
Order Processing Cost Structure:
├─ Direct Transaction Costs:
│   ├─ Exchange Fees: Per-share or per-trade charges
│   ├─ Clearing & Settlement: DTCC, clearinghouse fees
│   ├─ Regulatory Fees: SEC Section 31, FINRA TAF
│   └─ Connectivity Costs: Market data, colocation, FIX fees
├─ Infrastructure Costs (Amortized):
│   ├─ Trading Systems: Order management, execution algorithms
│   ├─ Risk Management: Pre-trade checks, position monitoring
│   ├─ Market Data: Level 2, historical data, analytics
│   ├─ Compliance: Surveillance, audit trails, reporting
│   └─ Technology: Servers, networks, backup systems
├─ Human Capital Costs:
│   ├─ Traders: Salaries, bonuses (spread across volume)
│   ├─ Developers: System maintenance, algorithm development
│   ├─ Compliance: Legal, regulatory reporting staff
│   └─ Operations: Settlement, reconciliation teams
├─ Scale Economics:
│   ├─ Fixed Cost Amortization: Total Fixed Cost / Volume
│   ├─ Break-Even Volume: Minimum volume for profitability
│   ├─ Marginal Cost Curve: Decreases with volume (up to capacity)
│   └─ Competitive Advantage: High-volume players have lower per-trade costs
└─ Market Structure Impact:
    ├─ Maker-Taker Fees: Rebates offset processing costs (or add to them)
    ├─ Tick Size: Minimum spread must cover processing + profit
    ├─ Penny Pilot: Reduced tick size (2006) compressed spreads
    └─ Fragmentation: Multiple venues increase routing complexity/cost
```

**Interaction:** Trade → routing → execution → clearing → settlement → reconciliation → cost accumulation

## 5. Mini-Project
Analyze order processing costs and scale economies:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

np.random.seed(42)

# Order processing cost components (per trade, dollars)
exchange_fee = 0.003
clearing_fee = 0.002
regulatory_fee = 0.001  # SEC Section 31, FINRA TAF
connectivity_per_trade = 0.001  # Amortized market data, colocation

variable_cost_per_trade = exchange_fee + clearing_fee + regulatory_fee + connectivity_per_trade

# Fixed costs (annual, dollars)
trading_systems = 500_000  # OMS, EMS, FIX engines
risk_management = 200_000  # Pre-trade risk, monitoring
market_data_subscription = 300_000  # Level 2 data, analytics
compliance_systems = 150_000  # Surveillance, reporting
technology_infrastructure = 400_000  # Servers, networks, co-location
human_capital = 1_500_000  # Traders, developers, compliance, ops

total_fixed_costs_annual = (trading_systems + risk_management + market_data_subscription + 
                           compliance_systems + technology_infrastructure + human_capital)

trading_days_per_year = 252

# Analyze different market participants
participants = {
    'Retail Broker': {'annual_volume': 10_000_000, 'avg_trade_size': 100},
    'Mid-Tier MM': {'annual_volume': 100_000_000, 'avg_trade_size': 500},
    'HFT Firm': {'annual_volume': 2_000_000_000, 'avg_trade_size': 200},
    'Bank MM': {'annual_volume': 500_000_000, 'avg_trade_size': 1000}
}

# Calculate costs for each participant
results = {}
for name, params in participants.items():
    annual_volume = params['annual_volume']
    
    # Fixed cost per trade
    fixed_cost_per_trade = total_fixed_costs_annual / annual_volume
    
    # Total order processing cost per trade
    total_cost_per_trade = variable_cost_per_trade + fixed_cost_per_trade
    
    # As percentage of typical trade ($100 × avg_trade_size)
    typical_trade_value = 100 * params['avg_trade_size']
    cost_bps = (total_cost_per_trade / typical_trade_value) * 10000
    
    # Minimum spread needed (2x cost, for both sides)
    min_spread_dollars = 2 * total_cost_per_trade
    min_spread_bps = (min_spread_dollars / 100) * 10000  # Assuming $100 stock
    
    results[name] = {
        'annual_volume': annual_volume,
        'fixed_per_trade': fixed_cost_per_trade,
        'variable_per_trade': variable_cost_per_trade,
        'total_per_trade': total_cost_per_trade,
        'cost_bps': cost_bps,
        'min_spread_dollars': min_spread_dollars,
        'min_spread_bps': min_spread_bps
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Cost breakdown by participant type
names = list(results.keys())
fixed_costs = [results[n]['fixed_per_trade'] for n in names]
variable_costs = [results[n]['variable_per_trade'] for n in names]

x_pos = np.arange(len(names))
width = 0.6

axes[0, 0].bar(x_pos, fixed_costs, width, label='Fixed Costs', color='lightcoral')
axes[0, 0].bar(x_pos, variable_costs, width, bottom=fixed_costs, 
              label='Variable Costs', color='lightblue')

# Add total labels
for i, name in enumerate(names):
    total = results[name]['total_per_trade']
    axes[0, 0].text(i, total + 0.002, f'${total:.4f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[0, 0].set_ylabel('Cost per Trade ($)')
axes[0, 0].set_title('Order Processing Cost Breakdown')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(names, rotation=15, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

print("Order Processing Costs by Participant Type:")
print("=" * 70)
for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Annual Volume: {data['annual_volume']:,} shares")
    print(f"  Fixed Cost per Trade: ${data['fixed_per_trade']:.4f}")
    print(f"  Variable Cost per Trade: ${data['variable_per_trade']:.4f}")
    print(f"  Total Cost per Trade: ${data['total_per_trade']:.4f}")
    print(f"  Cost in Basis Points: {data['cost_bps']:.2f} bps")
    print(f"  Minimum Spread Required: ${data['min_spread_dollars']:.4f} ({data['min_spread_bps']:.2f} bps)")

# Plot 2: Scale economies (cost per trade vs volume)
volumes = np.logspace(6, 10, 100)  # 1M to 10B shares
costs_per_trade = variable_cost_per_trade + total_fixed_costs_annual / volumes

axes[0, 1].semilogx(volumes, costs_per_trade * 1000, linewidth=2)  # Convert to cents
axes[0, 1].axhline(variable_cost_per_trade * 1000, color='r', linestyle='--', 
                  label=f'Variable Cost Floor (${variable_cost_per_trade*1000:.1f}¢)')

# Mark participant positions
for name, data in results.items():
    vol = data['annual_volume']
    cost = data['total_per_trade'] * 1000
    axes[0, 1].scatter(vol, cost, s=100, zorder=5)
    axes[0, 1].annotate(name, (vol, cost), xytext=(10, 5), 
                       textcoords='offset points', fontsize=8)

axes[0, 1].set_xlabel('Annual Volume (shares, log scale)')
axes[0, 1].set_ylabel('Cost per Trade (cents)')
axes[0, 1].set_title('Scale Economics in Order Processing')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Tick size constraints
# Analyze how order processing costs interact with tick size
stock_prices = np.array([1, 5, 10, 25, 50, 100, 200, 500])
tick_size_dollars = 0.01  # Penny tick

# For HFT firm (lowest cost)
hft_cost = results['HFT Firm']['total_per_trade']
hft_min_spread = results['HFT Firm']['min_spread_dollars']

# Tick size as percentage of price
tick_size_pct = (tick_size_dollars / stock_prices) * 100

# Minimum spread needed as percentage
min_spread_pct = (hft_min_spread / stock_prices) * 100

# Check if tick size binds (minimum spread > tick size)
tick_binds = hft_min_spread > tick_size_dollars

axes[1, 0].plot(stock_prices, tick_size_pct, 'b-', linewidth=2, 
               label='Tick Size (1¢)', marker='o')
axes[1, 0].plot(stock_prices, min_spread_pct, 'r--', linewidth=2, 
               label='Min Spread (HFT)', marker='s')
axes[1, 0].axhline(0.1, color='green', linestyle=':', 
                  label='10 bps (typical liquid stock)', linewidth=1)

axes[1, 0].set_xlabel('Stock Price ($)')
axes[1, 0].set_ylabel('Spread (%)')
axes[1, 0].set_title('Tick Size Constraints vs Order Processing Costs')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, which='both')

print(f"\nTick Size Analysis:")
print(f"For HFT firm with ${hft_min_spread:.4f} minimum spread:")
for price in [1, 5, 10, 50, 100]:
    tick_pct = (tick_size_dollars / price) * 10000
    min_pct = (hft_min_spread / price) * 10000
    binds = "BINDS" if hft_min_spread > tick_size_dollars else "not binding"
    print(f"  Price=${price:3d}: Tick={tick_pct:6.1f}bps, Min={min_pct:5.1f}bps [{binds}]")

# Plot 4: Profitability analysis
# Calculate break-even spread for different volumes
daily_volumes = np.array([1000, 5000, 10000, 50000, 100000, 500000, 1000000])
annual_volumes_from_daily = daily_volumes * trading_days_per_year

# Assume capturing half the spread per trade as revenue
# Calculate spread needed to break even (revenue = costs)
break_even_spreads = []
profit_margins = []

typical_stock_price = 100

for annual_vol in annual_volumes_from_daily:
    total_cost = variable_cost_per_trade + total_fixed_costs_annual / annual_vol
    # Break-even spread: revenue per trade = cost per trade
    # Revenue = spread / 2 (dealer earns half-spread)
    break_even_spread = 2 * total_cost
    break_even_spreads.append(break_even_spread)
    
    # Profit margin at 5 bps spread (typical for liquid stock)
    typical_spread_dollars = (5 / 10000) * typical_stock_price  # 5 bps
    revenue_per_trade = typical_spread_dollars / 2
    profit_per_trade = revenue_per_trade - total_cost
    annual_profit = profit_per_trade * annual_vol
    profit_margins.append(annual_profit)

axes[1, 1].bar(range(len(daily_volumes)), np.array(profit_margins) / 1e6, 
              color=['red' if x < 0 else 'green' for x in profit_margins])
axes[1, 1].axhline(0, color='black', linewidth=1)
axes[1, 1].set_xlabel('Daily Volume (shares)')
axes[1, 1].set_ylabel('Annual Profit ($ millions)')
axes[1, 1].set_title('Profitability vs Volume (5 bps spread)')
axes[1, 1].set_xticks(range(len(daily_volumes)))
axes[1, 1].set_xticklabels([f'{int(v/1000)}K' for v in daily_volumes], rotation=45)
axes[1, 1].grid(alpha=0.3, axis='y')

# Add profit values as labels
for i, profit in enumerate(profit_margins):
    label = f'${profit/1e6:.1f}M'
    y_pos = profit / 1e6 + (0.5 if profit > 0 else -0.5)
    axes[1, 1].text(i, y_pos, label, ha='center', va='bottom' if profit > 0 else 'top',
                   fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nBreak-Even Analysis (assuming $100 stock price):")
print("=" * 70)
for i, daily_vol in enumerate(daily_volumes):
    annual_vol = annual_volumes_from_daily[i]
    be_spread = break_even_spreads[i]
    be_spread_bps = (be_spread / typical_stock_price) * 10000
    profit = profit_margins[i]
    
    print(f"Daily Volume: {daily_vol:7,} shares ({annual_vol:12,} annual)")
    print(f"  Break-even Spread: ${be_spread:.4f} ({be_spread_bps:.2f} bps)")
    print(f"  Profit at 5 bps spread: ${profit:,.0f}")
    print()

# Calculate competitive advantage
hft_volume = results['HFT Firm']['annual_volume']
retail_volume = results['Retail Broker']['annual_volume']
hft_cost_advantage = results['Retail Broker']['total_per_trade'] - results['HFT Firm']['total_per_trade']
annual_advantage = hft_cost_advantage * hft_volume

print(f"\nCompetitive Advantage (HFT vs Retail Broker):")
print(f"Cost Advantage: ${hft_cost_advantage:.4f} per trade")
print(f"Annual Value (at HFT volume): ${annual_advantage:,.0f}")
print(f"HFT can profitably quote {hft_cost_advantage:.4f}/{0.01:.4f} = {hft_cost_advantage/0.01:.1f}x tighter")
```

## 6. Challenge Round
Why did decimal pricing (2001) and penny pilot (2006) reshape market structure?
- **Tick size reduction**: $0.0625 → $0.01 (85% reduction) compressed spreads, reduced dealer profits per trade
- **Order processing floor**: Fixed costs became larger % of spread, forcing consolidation (small dealers exited)
- **HFT emergence**: Economies of scale favored high-volume players who could amortize fixed costs over billions of trades
- **Quote competition**: Penny increments enabled penny jumping (queue priority by $0.01), increased quote updates
- **SEC Tick Size Pilot**: 2016-2018 experiment increasing ticks for small caps to study impact on liquidity provision

## 7. Key References
- [Stoll (1989) - Inferring Components of the Bid-Ask Spread](https://www.jstor.org/stable/2352946)
- [Harris (1994) - Minimum Price Variations, Discrete Bid-Ask Spreads, and Quotation Sizes](https://www.jstor.org/stable/2962258)
- [SEC Tick Size Pilot Program](https://www.sec.gov/rules/other/2015/34-74892.pdf)
- [Angel (1997) - Tick Size, Share Prices, and Stock Splits](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb04811.x)

---
**Status:** Fixed cost component of spreads | **Complements:** Bid-Ask Spread, Inventory Costs, Scale Economics
