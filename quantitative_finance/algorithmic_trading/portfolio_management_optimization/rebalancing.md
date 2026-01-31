# Rebalancing

## Concept Skeleton

Rebalancing realigns portfolio weights back to target allocations as market movements and cash flows cause drift, balancing the benefits of maintaining strategic asset mix against transaction costs, taxes, and market impact. Approaches range from calendar-based (monthly, quarterly) to threshold-driven (rebalance when allocation deviates >5%), with optimization considering cost-benefit tradeoffs between tracking error reduction and trading expenses.

**Core Components:**
- **Target weights**: Strategic allocation (e.g., 60/40 stocks/bonds) or dynamic optimal weights from periodic re-optimization
- **Drift monitoring**: Current weights vs. target; triggers when deviation exceeds tolerance bands
- **Rebalancing rules**: Calendar (time-based), threshold (deviation-based), or hybrid (calendar with threshold gates)
- **Transaction costs**: Commissions, bid-ask spreads, market impact, taxes (capital gains for taxable accounts)
- **Optimization**: Trade-off between tracking error (cost of not rebalancing) and transaction costs (cost of rebalancing)

**Why it matters:** Prevents unintended risk exposures (winning assets dominate portfolio), enforces discipline (sell high, buy low), but excessive rebalancing erodes returns through costs; optimal frequency is empirical question.

---

## Comparative Framing

| Dimension | **Calendar Rebalancing** | **Threshold Rebalancing** | **Hybrid (Calendar + Threshold)** |
|-----------|--------------------------|---------------------------|------------------------------------|
| **Trigger** | Fixed intervals (monthly, quarterly) | Allocation drift >X% (e.g., 5%) | Check at calendar date, rebalance if threshold breached |
| **Predictability** | High (scheduled trades) | Low (depends on market volatility) | Moderate (bounded by calendar) |
| **Trading frequency** | Constant (e.g., 4×/year) | Variable (more in volatile markets) | Moderate (less than pure calendar) |
| **Transaction costs** | Predictable but may be unnecessary | Minimized (trade only when needed) | Balanced (avoids over-trading) |
| **Tracking error** | Higher between rebalances | Lower (tighter control) | Moderate |
| **Tax efficiency** | Poor (forced trades) | Better (defer gains when drift small) | Better (selective rebalancing) |

**Key insight:** Calendar rebalancing is simple but ignores market conditions; threshold is cost-aware but requires monitoring; hybrid is practical compromise used by most institutional investors.

---

## Examples & Counterexamples

### Examples of Rebalancing

1. **Quarterly Calendar Rebalancing (60/40 Portfolio)**  
   - Target: 60% stocks, 40% bonds  
   - Quarter 1: Stocks rally 10%, bonds flat → Portfolio becomes 63/37  
   - Rebalance: Sell 3% stocks, buy 3% bonds (back to 60/40)  
   - Cost: 6% two-way turnover × (commissions + slippage)  

2. **Threshold Rebalancing with 5% Tolerance Bands**  
   - Target: 60% stocks (±5% bands = 55%–65%)  
   - Scenario A: Stocks drift to 64% → No rebalance (within bands)  
   - Scenario B: Stocks drift to 67% → Rebalance to 60% (breached upper band)  
   - Benefit: Saves transaction costs in Scenario A  

3. **Tax-Loss Harvesting with Rebalancing**  
   - Asset X: Position down 10%, want to rebalance (reduce weight)  
   - Sell at loss to realize capital loss (offsets gains elsewhere)  
   - Replace with correlated Asset Y to maintain exposure (avoid wash sale)  
   - Dual benefit: Rebalance + tax alpha  

4. **Cash Flow Rebalancing**  
   - Portfolio receives $100k contribution  
   - Current allocation: 65% stocks (above 60% target)  
   - Direct new cash to bonds (35% to stocks, 65% to bonds) to restore 60/40 without selling  
   - Zero transaction costs; passive rebalancing via inflows

### Non-Examples (or Edge Cases)

- **Never rebalancing**: Portfolio drifts to 90/10 stocks/bonds after bull market; unintended concentration risk.
- **Daily rebalancing**: Excessive trading costs overwhelm benefits; tracking error minimal at daily frequency.
- **Rebalancing without transaction cost awareness**: Naively trade to exact targets, eroding returns.

---

## Layer Breakdown

**Layer 1: Drift Calculation and Monitoring**  
Current weights: \(w_i^{\text{current}} = \frac{\text{Value}_i}{\sum \text{Value}_j}\).  
Target weights: \(w_i^{\text{target}}\) (from strategic allocation or optimization).  
Tracking error (deviation): \(\text{TE} = \sqrt{\sum (w_i^{\text{current}} - w_i^{\text{target}})^2}\).  
For threshold rule: Rebalance if \(|w_i^{\text{current}} - w_i^{\text{target}}| > \text{Tolerance}\) for any asset \(i\).

**Layer 2: Rebalancing Decision Rule**  
**Calendar-based:**  
IF (Current Date = Rebalance Date) THEN rebalance.  

**Threshold-based:**  
IF (Any \(|w_i^{\text{current}} - w_i^{\text{target}}| > \text{Threshold}\)) THEN rebalance.  

**Hybrid:**  
IF (Rebalance Date AND threshold breached) THEN rebalance.  

**Optimization-based:**  
Solve: Minimize \(\text{TE}^2 + \lambda \cdot \text{Transaction Costs}\) where \(\lambda\) = cost-aversion parameter.

**Layer 3: Trade Execution and Sizing**  
Target trades: \(\Delta w_i = w_i^{\text{target}} - w_i^{\text{current}}\).  
Convert to dollar amounts: \(\Delta \$ = \Delta w_i \times \text{Portfolio Value}\).  
Apply turnover constraints: If \(\sum |\Delta w_i| > \text{Max Turnover}\), scale trades proportionally.  
Execution: Use limit orders or VWAP/TWAP algos to minimize market impact.

**Layer 4: Tax and Cost Considerations**  
**Taxable accounts:**  
- Prefer harvesting losses (sell positions at loss, defer gains)  
- Delay rebalancing if near long-term capital gains threshold (1-year holding period)  

**Transaction costs:**  
- Round small trades (avoid <$1k trades with fixed commissions)  
- Use ETFs for bonds/international (liquid, low-cost vs. individual securities)  
- Net trades across accounts (don't sell in one account, buy same asset in another)

---

## Mini-Project: Threshold Rebalancing Simulation

**Goal:** Compare calendar vs. threshold rebalancing over time.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate asset returns (stocks and bonds, 5 years daily)
np.random.seed(88)
n_days = 252 * 5
stock_returns = np.random.normal(0.0008, 0.015, n_days)  # 20% annual, 15% vol
bond_returns = np.random.normal(0.0003, 0.005, n_days)   # 7.5% annual, 5% vol

# Initial allocation: 60/40
initial_value = 100000
stock_value = initial_value * 0.60
bond_value = initial_value * 0.40
target_stock_weight = 0.60

# Storage for results
results = {
    'date': [],
    'no_rebalance_stock_weight': [],
    'calendar_stock_weight': [],
    'threshold_stock_weight': [],
}

# Tracking variables
no_reb_stock = stock_value
no_reb_bond = bond_value

cal_stock = stock_value
cal_bond = bond_value

thresh_stock = stock_value
thresh_bond = bond_value

threshold = 0.05  # 5% deviation triggers rebalance
rebalance_interval = 63  # Quarterly (252/4)
transaction_cost_bps = 10  # 10 bps per side
day_counter = 0

for day in range(n_days):
    # Update values based on returns
    no_reb_stock *= (1 + stock_returns[day])
    no_reb_bond *= (1 + bond_returns[day])
    
    cal_stock *= (1 + stock_returns[day])
    cal_bond *= (1 + bond_returns[day])
    
    thresh_stock *= (1 + stock_returns[day])
    thresh_bond *= (1 + bond_returns[day])
    
    # Calendar rebalancing (quarterly)
    if day_counter % rebalance_interval == 0 and day > 0:
        cal_total = cal_stock + cal_bond
        cal_target_stock = cal_total * target_stock_weight
        cal_target_bond = cal_total * (1 - target_stock_weight)
        
        # Transaction costs
        turnover = abs(cal_stock - cal_target_stock)
        cost = turnover * (transaction_cost_bps / 10000)
        
        cal_stock = cal_target_stock
        cal_bond = cal_target_bond - cost  # Deduct from bonds
    
    # Threshold rebalancing (check daily)
    thresh_total = thresh_stock + thresh_bond
    thresh_current_weight = thresh_stock / thresh_total
    
    if abs(thresh_current_weight - target_stock_weight) > threshold:
        thresh_target_stock = thresh_total * target_stock_weight
        thresh_target_bond = thresh_total * (1 - target_stock_weight)
        
        # Transaction costs
        turnover = abs(thresh_stock - thresh_target_stock)
        cost = turnover * (transaction_cost_bps / 10000)
        
        thresh_stock = thresh_target_stock
        thresh_bond = thresh_target_bond - cost
    
    # Record weights
    if day % 21 == 0:  # Monthly snapshots
        results['date'].append(day)
        results['no_rebalance_stock_weight'].append(no_reb_stock / (no_reb_stock + no_reb_bond))
        results['calendar_stock_weight'].append(cal_stock / (cal_stock + cal_bond))
        results['threshold_stock_weight'].append(thresh_stock / (thresh_stock + thresh_bond))
    
    day_counter += 1

# Final values
no_reb_total = no_reb_stock + no_reb_bond
cal_total = cal_stock + cal_bond
thresh_total = thresh_stock + thresh_bond

print("=" * 60)
print("REBALANCING STRATEGY COMPARISON (5 Years)")
print("=" * 60)
print(f"Initial Portfolio Value:     ${initial_value:>12,.2f}")
print(f"Target Stock Weight:         {target_stock_weight:>12.0%}")
print(f"Rebalancing Threshold:       {threshold:>12.0%}")
print(f"Transaction Cost:            {transaction_cost_bps:>12} bps")
print()
print(f"No Rebalancing:")
print(f"  Final Value:               ${no_reb_total:>12,.2f}")
print(f"  Final Stock Weight:        {no_reb_stock/no_reb_total:>12.1%}")
print(f"  Total Return:              {(no_reb_total/initial_value - 1)*100:>11.2f}%")
print()
print(f"Calendar Rebalancing (Quarterly):")
print(f"  Final Value:               ${cal_total:>12,.2f}")
print(f"  Final Stock Weight:        {cal_stock/cal_total:>12.1%}")
print(f"  Total Return:              {(cal_total/initial_value - 1)*100:>11.2f}%")
print()
print(f"Threshold Rebalancing (±5%):")
print(f"  Final Value:               ${thresh_total:>12,.2f}")
print(f"  Final Stock Weight:        {thresh_stock/thresh_total:>12.1%}")
print(f"  Total Return:              {(thresh_total/initial_value - 1)*100:>11.2f}%")
print("=" * 60)

# Plot weight drift over time
df = pd.DataFrame(results)
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['no_rebalance_stock_weight'], label='No Rebalancing', linewidth=2)
plt.plot(df['date'], df['calendar_stock_weight'], label='Calendar (Quarterly)', linewidth=2, linestyle='--')
plt.plot(df['date'], df['threshold_stock_weight'], label='Threshold (±5%)', linewidth=2, linestyle=':')
plt.axhline(target_stock_weight, color='red', linestyle='-', linewidth=1, label='Target (60%)')
plt.axhline(target_stock_weight + threshold, color='gray', linestyle=':', alpha=0.5)
plt.axhline(target_stock_weight - threshold, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Trading Days')
plt.ylabel('Stock Weight')
plt.title('Portfolio Weight Drift: Rebalancing Strategy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Expected Output (illustrative):**
```
============================================================
REBALANCING STRATEGY COMPARISON (5 Years)
============================================================
Initial Portfolio Value:         $100,000.00
Target Stock Weight:                      60%
Rebalancing Threshold:                     5%
Transaction Cost:                       10 bps

No Rebalancing:
  Final Value:                 $154,287.35
  Final Stock Weight:                 68.3%
  Total Return:                       54.29%

Calendar Rebalancing (Quarterly):
  Final Value:                 $151,432.78
  Final Stock Weight:                 60.2%
  Total Return:                       51.43%

Threshold Rebalancing (±5%):
  Final Value:                 $152,845.21
  Final Stock Weight:                 61.1%
  Total Return:                       52.85%

============================================================
```

**Interpretation:**  
- No rebalancing: Highest return (stocks outperformed) but unintended risk concentration (68% stocks).  
- Calendar: Disciplined 60/40 maintenance but frequent trades reduce returns.  
- Threshold: Best balance—captures most of stock upside while controlling drift, trades only when needed.

---

## Challenge Round

1. **Rebalancing Premium**  
   Does rebalancing improve returns, or just control risk? Under what conditions does it add alpha?

   <details><summary>Hint</summary>Rebalancing is *volatility harvesting*: forces "buy low, sell high" (contrarian). Adds return when assets mean-revert (negative serial correlation). If trends persist (momentum), rebalancing underperforms (sells winners early, buys losers). Optimal when asset returns are uncorrelated or negatively correlated over rebalancing horizon.</details>

2. **Optimal Rebalancing Frequency**  
   How to determine if quarterly vs. annual rebalancing is better?

   <details><summary>Solution</summary>
   Empirical approach: Backtest both. Metrics: (1) Sharpe ratio (risk-adjusted return), (2) Tracking error vs. target, (3) Total transaction costs. Trade-off: More frequent rebalancing → lower tracking error but higher costs. Optimal frequency depends on: (a) Asset volatility (higher vol → faster drift → more frequent), (b) Transaction costs (higher costs → less frequent), (c) Risk tolerance (lower tolerance → more frequent). Typical: Quarterly for institutional equity, annual for individual 401(k).
   </details>

3. **Cash Flow vs. Security Sales**  
   Portfolio needs $50k to rebalance bonds from 35% to 40% (currently $350k in bonds, target $400k). $50k contribution arrives. How to rebalance?

   <details><summary>Solution</summary>
   **Option A:** Direct $50k to bonds → Zero transaction costs.  
   **Option B:** Sell stocks, buy bonds → Transaction costs on $50k two-way ($100k total turnover).  
   **Optimal:** Option A (cash flow rebalancing). Always use inflows/outflows first before selling existing holdings.
   </details>

4. **Tax-Loss Harvesting During Rebalancing**  
   Portfolio has Asset A (down 15%, need to reduce weight) and Asset B (up 20%, need to reduce weight). Both require $10k sales. Which to sell first?

   <details><summary>Solution</summary>
   Sell Asset A (at loss) to realize capital loss → Tax benefit (offset gains or $3k ordinary income).  
   If also need to reduce Asset B, consider: (1) Wait until long-term gain (>1 year holding), (2) Replace Asset B with similar asset (maintain exposure, defer gain), (3) Donate Asset B (if charitable, avoid capital gains entirely).  
   **Hierarchy:** Harvest losses > Defer gains > Minimize short-term gains.
   </details>

---

## Key References

- **Markowitz & van Dijk (2003)**: "Single-Period Mean-Variance Analysis in a Changing World" ([Financial Analysts Journal](https://www.jstor.org/))
- **Arnott & Lovell (1993)**: "Rebalancing: Why? When? How Often?" ([Journal of Investing](https://www.iijournals.com/))
- **Jaconetti et al. (2010)**: "Best Practices for Portfolio Rebalancing" (Vanguard Research) ([Vanguard](https://www.vanguard.com/))
- **Dichtl et al. (2016)**: "Optimal Rebalancing Frequency" ([Journal of Portfolio Management](https://www.iijournals.com/))

**Further Reading:**  
- Conditional rebalancing (time-varying thresholds based on market volatility)  
- Multi-account rebalancing (taxable + IRA + 401k coordination)  
- Rebalancing with derivatives (futures for tactical shifts, avoid selling equities)
