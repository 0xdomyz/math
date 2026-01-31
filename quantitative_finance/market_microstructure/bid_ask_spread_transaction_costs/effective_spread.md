# Effective Spread

## 1. Concept Skeleton
**Definition:** Twice the absolute difference between execution price and midpoint quote at trade time  
**Purpose:** Measures actual transaction cost paid by traders, accounts for price improvement inside quoted spread  
**Prerequisites:** Bid-ask spread, order book mechanics, trade execution concepts

## 2. Comparative Framing
| Measure | Quoted Spread | Effective Spread | Realized Spread | Price Impact |
|---------|---------------|------------------|-----------------|--------------|
| **Formula** | Ask - Bid | 2×\|Price - Mid\| | Effective - 2×ΔMid | 2×ΔMid×Direction |
| **Timing** | Pre-trade (posted) | At execution | Post-trade (5-30 min) | Post-trade |
| **Captures** | Posted liquidity cost | Actual cost paid | Dealer revenue | Information content |
| **Benchmark** | NBBO width | Execution quality | Market maker profit | Permanent impact |

## 3. Examples + Counterexamples

**Simple Example:**  
Midpoint=$100.00, Buy executes at $100.03 → Effective spread = 2×|100.03-100.00| = $0.06

**Price Improvement Case:**  
Quoted: Bid=$99.95, Ask=$100.05 (spread=$0.10), Buy executes at $100.01 → Effective=$0.02 (80% improvement)

**Edge Case:**  
Trade executes at midpoint ($100.00) → Effective spread = $0.00 (full price improvement, dark pool or midpoint peg)

## 4. Layer Breakdown
```
Effective Spread Measurement:
├─ Components:
│   ├─ Execution Price: Actual trade price from exchange feed
│   ├─ Midpoint Quote: (Bid + Ask) / 2 at execution microsecond
│   ├─ Trade Direction: Buy (+1) or Sell (-1), signed indicator
│   └─ Calculation: 2 × |Execution - Midpoint|
├─ Interpretation:
│   ├─ Zero: Perfect price improvement (midpoint execution)
│   ├─ < Quoted Spread: Price improvement inside spread
│   ├─ = Quoted Spread: Execution at quoted bid/ask
│   └─ > Quoted Spread: Trade walks through book (large order)
├─ Decomposition (Huang-Stoll):
│   ├─ Realized Spread: Dealer immediate revenue component
│   ├─ Price Impact: Information-driven permanent change
│   └─ Identity: Effective = Realized + Price Impact
├─ Time-Weighted Variants:
│   ├─ 1-minute Effective: Uses 1-min post-trade midpoint
│   ├─ 5-minute Effective: Standard industry benchmark
│   └─ 30-minute Effective: Captures slower information revelation
└─ Aggregation Methods:
    ├─ Volume-Weighted: Weights by trade size
    ├─ Time-Weighted: Equal weight per time interval
    └─ Trade-Weighted: Equal weight per transaction
```

**Interaction:** Market order arrives → execution algorithm routes → price improvement applied → effective spread realized

## 5. Mini-Project
Measure effective spreads and price improvement:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Simulate high-frequency order book data
n_seconds = 3600  # 1 hour of data
tick_size = 0.01

# Midpoint price follows random walk with microstructure noise
true_midpoint = 100 + np.cumsum(np.random.normal(0, 0.02, n_seconds))

# Quoted bid-ask spread (varies with volatility)
rolling_vol = np.abs(np.diff(true_midpoint, prepend=true_midpoint[0]))
rolling_vol_smooth = np.convolve(rolling_vol, np.ones(60)/60, mode='same')
quoted_half_spread = 0.03 + 5 * rolling_vol_smooth
quoted_half_spread = np.maximum(quoted_half_spread, tick_size)

bid_prices = true_midpoint - quoted_half_spread
ask_prices = true_midpoint + quoted_half_spread

# Generate trades with realistic microstructure
n_trades = 800
trade_times = np.sort(np.random.choice(n_seconds, n_trades, replace=False))
trade_directions = np.random.choice([-1, 1], n_trades)  # -1=sell, 1=buy
trade_sizes = np.random.lognormal(3, 1, n_trades)  # Size distribution

# Execution model: larger trades have less price improvement
price_improvement_pct = np.zeros(n_trades)
execution_prices = np.zeros(n_trades)
midpoint_at_trade = np.zeros(n_trades)
quoted_spread_at_trade = np.zeros(n_trades)

for i, t in enumerate(trade_times):
    mid = true_midpoint[t]
    half_spread = quoted_half_spread[t]
    midpoint_at_trade[i] = mid
    quoted_spread_at_trade[i] = 2 * half_spread
    
    # Price improvement: smaller trades get more improvement
    # Large trades may walk through the book
    size_percentile = stats.rankdata(trade_sizes)[i] / len(trade_sizes)
    
    if trade_sizes[i] < 100:  # Small retail orders
        # High probability of price improvement
        improvement_prob = 0.7
        if np.random.random() < improvement_prob:
            # Random price improvement between 0% and 80% of half-spread
            improvement = np.random.uniform(0, 0.8) * half_spread
        else:
            improvement = 0
    elif trade_sizes[i] < 500:  # Medium orders
        improvement = np.random.uniform(0, 0.4) * half_spread
    else:  # Large institutional orders
        # May walk through book (negative price improvement)
        improvement = np.random.uniform(-0.2, 0.2) * half_spread
    
    price_improvement_pct[i] = improvement / half_spread if half_spread > 0 else 0
    
    # Execute based on direction
    if trade_directions[i] == 1:  # Buy
        execution_prices[i] = ask_prices[t] - improvement
    else:  # Sell
        execution_prices[i] = bid_prices[t] + improvement

# Calculate effective spread
effective_spreads = 2 * np.abs(execution_prices - midpoint_at_trade)

# Calculate percentage effective spread
pct_effective_spreads = (effective_spreads / midpoint_at_trade) * 100

# Calculate price improvement in dollars
price_improvement_dollars = np.where(
    trade_directions == 1,
    ask_prices[trade_times] - execution_prices,  # Buy: saved vs ask
    execution_prices - bid_prices[trade_times]   # Sell: gained vs bid
)

# Calculate realized spread (5-second forward looking)
realized_spreads = np.zeros(n_trades)
price_impacts = np.zeros(n_trades)

for i, t in enumerate(trade_times):
    future_t = min(t + 5, n_seconds - 1)
    future_mid = true_midpoint[future_t]
    
    # Realized spread: effective spread minus price change
    price_change = (future_mid - midpoint_at_trade[i]) * trade_directions[i]
    price_impacts[i] = 2 * price_change
    realized_spreads[i] = effective_spreads[i] - price_impacts[i]

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Effective vs Quoted Spread
axes[0, 0].scatter(quoted_spread_at_trade, effective_spreads, 
                   alpha=0.3, s=20, c=trade_sizes, cmap='viridis')
axes[0, 0].plot([0, quoted_spread_at_trade.max()], 
               [0, quoted_spread_at_trade.max()], 
               'r--', label='No Price Improvement', linewidth=2)

# Add diagonal lines for different improvement levels
for pct in [0.25, 0.5, 0.75]:
    x = np.linspace(0, quoted_spread_at_trade.max(), 100)
    axes[0, 0].plot(x, pct * x, 'gray', alpha=0.3, linestyle='--', linewidth=1)
    axes[0, 0].text(quoted_spread_at_trade.max() * 0.9, pct * quoted_spread_at_trade.max() * 0.9,
                   f'{int((1-pct)*100)}%', fontsize=8, color='gray')

colorbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
colorbar.set_label('Trade Size')

axes[0, 0].set_title('Effective vs Quoted Spread')
axes[0, 0].set_xlabel('Quoted Spread ($)')
axes[0, 0].set_ylabel('Effective Spread ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

improvement_rate = (effective_spreads < quoted_spread_at_trade).mean() * 100
avg_improvement = price_improvement_dollars.mean()
print(f"Price Improvement Statistics:")
print(f"Trades with price improvement: {improvement_rate:.1f}%")
print(f"Average price improvement: ${avg_improvement:.4f}")

# Plot 2: Price improvement distribution by trade size
size_bins = [0, 100, 500, trade_sizes.max()]
size_labels = ['Small\n(<100)', 'Medium\n(100-500)', 'Large\n(>500)']
improvement_by_size = []

for i in range(len(size_bins) - 1):
    mask = (trade_sizes >= size_bins[i]) & (trade_sizes < size_bins[i+1])
    improvement_by_size.append(price_improvement_dollars[mask])

bp = axes[0, 1].boxplot(improvement_by_size, labels=size_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].set_title('Price Improvement by Trade Size')
axes[0, 1].set_xlabel('Trade Size Category')
axes[0, 1].set_ylabel('Price Improvement ($)')
axes[0, 1].grid(alpha=0.3, axis='y')

print(f"\nPrice Improvement by Size:")
for i, label in enumerate(size_labels):
    print(f"{label.replace(chr(10), ' ')}: ${np.mean(improvement_by_size[i]):.4f}")

# Plot 3: Spread decomposition (Effective = Realized + Impact)
axes[1, 0].scatter(realized_spreads, price_impacts, alpha=0.3, s=20)
axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=0.5)

# Add regression line
mask = ~np.isnan(realized_spreads) & ~np.isnan(price_impacts)
if mask.sum() > 0:
    z = np.polyfit(realized_spreads[mask], price_impacts[mask], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(realized_spreads.min(), realized_spreads.max(), 100)
    axes[1, 0].plot(x_fit, p(x_fit), 'r--', linewidth=2, 
                   label=f'Slope={z[0]:.2f}')

axes[1, 0].set_title('Spread Decomposition: Realized vs Price Impact')
axes[1, 0].set_xlabel('Realized Spread ($)')
axes[1, 0].set_ylabel('Price Impact ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

print(f"\nSpread Decomposition:")
print(f"Mean Effective Spread: ${effective_spreads.mean():.4f}")
print(f"Mean Realized Spread: ${realized_spreads.mean():.4f}")
print(f"Mean Price Impact: ${price_impacts.mean():.4f}")
print(f"Adverse Selection %: {(price_impacts.mean()/effective_spreads.mean())*100:.1f}%")

# Plot 4: Time series of percentage effective spread
time_bins = np.arange(0, n_seconds, 300)  # 5-minute bins
bin_means = []
bin_stds = []

for i in range(len(time_bins) - 1):
    mask = (trade_times >= time_bins[i]) & (trade_times < time_bins[i+1])
    if mask.sum() > 0:
        bin_means.append(pct_effective_spreads[mask].mean())
        bin_stds.append(pct_effective_spreads[mask].std())
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)

bin_centers = (time_bins[:-1] + time_bins[1:]) / 2 / 60  # Convert to minutes

axes[1, 1].plot(bin_centers, bin_means, 'o-', linewidth=2, markersize=6)
axes[1, 1].fill_between(bin_centers, 
                        np.array(bin_means) - np.array(bin_stds),
                        np.array(bin_means) + np.array(bin_stds),
                        alpha=0.3)
axes[1, 1].set_title('Effective Spread Over Time (5-min bins)')
axes[1, 1].set_xlabel('Time (minutes)')
axes[1, 1].set_ylabel('Effective Spread (%)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate execution quality metrics
quoted_spread_bps = (quoted_spread_at_trade / midpoint_at_trade).mean() * 10000
effective_spread_bps = (effective_spreads / midpoint_at_trade).mean() * 10000
price_improvement_bps = ((quoted_spread_at_trade - effective_spreads) / 
                         midpoint_at_trade).mean() * 10000

print(f"\nExecution Quality Metrics (basis points):")
print(f"Quoted Spread: {quoted_spread_bps:.2f} bps")
print(f"Effective Spread: {effective_spread_bps:.2f} bps")
print(f"Price Improvement: {price_improvement_bps:.2f} bps")
print(f"Improvement Rate: {(1 - effective_spread_bps/quoted_spread_bps)*100:.1f}%")

# Analyze effective spread by time of day (U-shaped pattern)
hour_bins = trade_times // 3600
if len(np.unique(hour_bins)) > 1:
    print(f"\nIntraday Pattern:")
    for hour in np.unique(hour_bins):
        mask = hour_bins == hour
        print(f"Period {int(hour)}: ${effective_spreads[mask].mean():.4f} "
              f"({pct_effective_spreads[mask].mean():.3f}%)")
```

## 6. Challenge Round
How does effective spread differ from quoted spread, and why does it matter?
- **Price improvement**: Effective captures actual cost, not posted cost (SEC Rule 606 requires brokers report effective spreads)
- **Execution venue differences**: Dark pools execute at midpoint (0% effective), lit exchanges at NBBO (100% of quoted)
- **Payment for order flow**: Retail brokers route to wholesalers offering price improvement, reducing effective spreads
- **Order size impact**: Small orders get improvement, large orders pay more than quoted (walk the book)
- **Market quality measurement**: Regulators use effective spread to assess market quality and best execution compliance

## 7. Key References
- [Bessembinder & Kaufman (1997) - Comparison of Trade Execution Costs](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb02744.x)
- [Huang & Stoll (1996) - Dealer vs Auction Markets](https://www.jstor.org/stable/2329367)
- [SEC Rule 606 - Order Routing Disclosure](https://www.sec.gov/rules/final/34-43590.htm)
- [FINRA Trade Reporting](https://www.finra.org/filing-reporting/trace)

---
**Status:** Transaction cost analysis benchmark | **Complements:** Bid-Ask Spread, Realized Spread, Implementation Shortfall
