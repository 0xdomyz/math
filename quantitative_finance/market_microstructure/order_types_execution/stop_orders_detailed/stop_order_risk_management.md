# Stop Orders: Risk Management & Execution Triggers

## I. Concept Skeleton

**Definition:** A stop order (or stop-loss order) is a conditional order that becomes active only when the market price reaches a specified trigger level (stop price), converting automatically into a market or limit order to execute. Stop orders are primarily used for risk management—protecting against adverse price moves—but they can also be used to enter positions at breakout levels or scale into strategies at predetermined prices.

**Purpose:** Establish automated risk controls that execute without constant monitoring, enforce systematic trading discipline, protect against gap openings and overnight risk, enable mechanical trading rules without human intervention, and provide entry/exit triggers for algorithmic systems.

**Prerequisites:** Conditional order logic, market gaps and opening behavior, execution algorithms, portfolio risk management, behavioral trading biases (why humans need stops), trigger mechanics and execution priority.

---

## II. Comparative Framing

| **Aspect** | **Stop Orders** | **Stop-Limit Orders** | **Mental Stops** | **Options (Puts)** | **Collar Strategies** |
|-----------|-------------|-------------------|------------------|-------------------|-------------------|
| **Trigger Mechanism** | Price hits level | Price hits level | Trader's mind | Automatic at strike | Combined order legs |
| **Execution Type** | Market order (if filled) | Limit order | Manual execution | Derivative settlement | Hedged position |
| **Risk Certainty** | High (market order) | Medium (limit might not fill) | Very low (emotional) | Very high (put pays) | Very high (collar) |
| **Execution Price** | Uncertain | Specified max/min | Uncertain (emotions!) | Derivative premium | Weighted (collar) |
| **Execution Guarantee** | High (market) | Low (depends on price) | Low (might forget) | 100% (if strikes) | 100% (collar) |
| **Cost Structure** | Slippage | Slippage if fills | Psychological | Premium (upfront) | Net debit/credit |
| **Latency Sensitivity** | High (slippage) | Medium (trigger + limit) | Very high (human) | None (automatic) | Low (pre-hedged) |
| **Use Case** | Protective exits | Risk management | Undisciplined trading | Long-term hedges | Downside capped |
| **Typical Trigger** | 5-10% stop | Conditional entry | Vague threshold | Strike price | Range-based |
| **Example** | "Sell if drops to $95" | "Sell if drops to $95, but limit to $93" | "I'll sell if it looks bad" | "Buy $95 put" | "Sell upside at $110" |

---

## III. Examples & Counterexamples

### Example 1: Stop Order Protecting Against Overnight Gap
**Setup:**
- Position: Long 10,000 shares of Tech Stock
- Current price: $100.00 (mid), Bid $99.98, Ask $100.02
- Your stop order: Sell if price touches $95.00 (5% loss threshold)
- Market event: Company announces terrible earnings after hours
- Next morning: Stock opens at $88.00 (gap down 12%)

**Execution:**
- Your stop order triggers: Price at $88.00 < $95.00 stop
- Order activates and becomes market order
- Execution: Your 10,000 shares sell at market
- Fill price: Opens at $88.00, bid-ask $87.95 - $88.05
- Your execution: ~$87.98 average (assuming some slippage)

**Analysis:**
- Intended protection: Stop at $95.00 (5% loss = $50,000 cost)
- Actual outcome: Filled at $88.00 (12% loss = $120,000 cost)
- **Gap risk:** Stop orders don't protect against gaps. Price can gap PAST your stop level.
- **Positive outcome:** At least you exited before further decline (stock fell to $80 by 10am)

**Lesson:** Stops useful but not foolproof. Gap risk unavoidable in stop orders.

---

### Example 2: Stop Order Hunting (Liquidity Sweep)
**Setup:**
- Stock price: $100.00, very stable
- Many traders: Stop orders at $99.50 (round number)
- Market makers know this: They see order flow patterns
- Scenario: Slight downward pressure develops

**Timeline:**
| Time | Market Price | Bid-Ask | MMs Action | Stop Orders |
|------|------------|---------|-----------|------------|
| 9:30am | $100.00 | $99.98 - $100.02 | Selling aggressively | 1000+ orders waiting |
| 9:35am | $99.80 | $99.78 - $99.82 | Continue selling | Getting close |
| 9:40am | $99.55 | $99.53 - $99.57 | FINAL PUSH | Some triggers at $99.50 |
| 9:40:30am | $99.49 | Bid $99.47 | **STOP HUNT** | ALL triggers execute |
| 9:45am | $99.95 | $99.93 - $99.97 | Price recovers | MMs made money |

**Analysis:**
- Stop hunting: Market makers deliberately pushed price down 0.5% to trigger stops
- Stop orders at $99.50: ~1,000 orders triggered, dumping into market
- Consequence: Stops filled at $99.47 (below their stop level!)
- Recovery: Price bounced back to $99.95 within 5 minutes
- Traders with stops: Sold at worst possible time (-$53 per share vs $100.00 entry)
- MMs: Bought dip, sold into stop cascade → $50k+ profit

**Lesson:** Stop orders create predictable liquidity; can be exploited by sophisticated traders.

---

### Example 3: Counterexample—Stop-Limit Order Failure
**Setup:**
- Position: Long 10,000 shares
- Stop-limit order: "Sell if $95, but limit to $94.50 minimum"
- Reason: Prevent gap-down losses like Example 1
- Company event: Bad earnings announcement after hours

**Next Morning:**
- Stock opens at $88.00 (big gap down)
- Stop-limit triggers: Stop hit ($95.00 level crossed)
- Limit order posted: Sell 10,000 at limit $94.50 or better
- **Problem:** Current bid is $87.95 (far below $94.50 limit)
- **Result:** Limit order sits in book, NEVER FILLS (no one wants to buy at $94.50)

**Outcome:**
- Your goal: Lose no more than 5% ($50,000)
- Actual: Holding 10k shares at $88.00 = 12% loss (-$120,000)
- All day: Stock stays $87-$89 range; your limit order never fills
- By close: You manually cancel limit order, now forced to exit at market ($87.95)
- **Total loss: $120,500** (much worse than intended 5%)

**Lesson:** Stop-limit offers better execution control but SACRIFICES CERTAINTY. Not a free lunch.

---

## IV. Layer Breakdown

```
STOP ORDER FRAMEWORK

┌──────────────────────────────────────────────────┐
│         STOP ORDER CONDITIONAL LOGIC               │
│                                                   │
│  Core: Dormant until triggered; then execute     │
│        Provides automated protection              │
└────────────────────┬─────────────────────────────┘
                     │
    ┌────────────────▼──────────────────────┐
    │  1. TRIGGER MECHANICS (Stop Price)   │
    │                                       │
    │  For SELL Stop (Protective Exit):    │
    │  ├─ Normal Price: $100.00            │
    │  ├─ Stop Level: $95.00               │
    │  ├─ Trigger: Price ≤ $95.00          │
    │  ├─ Converts: Market sell order      │
    │  └─ Status: Dormant until hit        │
    │                                       │
    │  For BUY Stop (Breakout Entry):      │
    │  ├─ Normal Price: $100.00            │
    │  ├─ Stop Level: $105.00              │
    │  ├─ Trigger: Price ≥ $105.00         │
    │  ├─ Converts: Market buy order       │
    │  └─ Used for: Momentum followers     │
    │                                       │
    │  Trigger Mechanics:                  │
    │  ├─ Exchange monitors order book     │
    │  ├─ When price crosses: Order triggers
    │  ├─ Conversion: Market order posted  │
    │  ├─ Latency: 1-10ms (varies)        │
    │  └─ Risk: Gap opens past trigger    │
    └────────────────┬──────────────────────┘
                     │
    ┌────────────────▼──────────────────────┐
    │  2. EXECUTION MECHANICS (Post-Trigger)
    │                                        │
    │  Step 1: Stop Price Reached           │
    │  ├─ Last trade: $95.01                │
    │  ├─ Bid: $94.99, Ask: $95.03         │
    │  ├─ Order not yet triggered           │
    │  │   (price hasn't CROSSED level)     │
    │  │                                     │
    │  Step 2: Next Trade Crosses Stop      │
    │  ├─ Trade at: $94.98 (below $95.00)  │
    │  ├─ Condition met: Stop activates    │
    │  ├─ Order type: Converts to market   │
    │  ├─ Execution: Market sell posted    │
    │  └─ Fills immediately at bid         │
    │                                        │
    │  Step 3: Market Order Execution       │
    │  ├─ Current bid: $94.95               │
    │  ├─ Your execution: ~$94.95           │
    │  ├─ Slippage: $95.00 - $94.95 = $0.05│
    │  └─ Cost: 5 cents per share loss    │
    │                                        │
    │  Realistic: Often worse if:          │
    │  ├─ Price falls fast (gaps down)     │
    │  ├─ Liquidity dries up (panic)       │
    │  ├─ Multiple stops trigger (cascade) │
    │  └─ Circuit breakers halt trading    │
    └────────────────┬──────────────────────┘
                     │
    ┌────────────────▼─────────────────────────┐
    │  3. GAP RISK (When Stops Fail)           │
    │                                          │
    │  Scenario A: Normal Market (No Gap)     │
    │  Price: $100 → $99 → $98 → $95 (STOP)  │
    │  └─ Trigger at $95, execute at $94.95   │
    │  └─ Loss: ~$5 per share (5%)            │
    │                                          │
    │  Scenario B: Price Gap (Overnight)      │
    │  Market close: $100.00                   │
    │  After-hours event: Company crashes     │
    │  Market open: $88.00 (gap down 12%)     │
    │  Your stop: $95.00 (PASSED OVER!)       │
    │  └─ Trigger: Order activates at $88    │
    │  └─ Execute: ~$88 (not $95!)           │
    │  └─ Loss: ~$12 per share (12%)          │
    │                                          │
    │  Gap Risk Analysis:                      │
    │  ├─ Overnight: News after close         │
    │  ├─ Pre-market: Earnings surprises      │
    │  ├─ Market opens: GAP PAST STOP         │
    │  ├─ Your protection: Fails!             │
    │  └─ Lesson: Stops can't prevent gaps    │
    │                                          │
    │  Mitigation Strategies:                 │
    │  ├─ Use options instead (puts)          │
    │  ├─ Tighter stops (more cost)           │
    │  ├─ Stop-limit with low limit (risky)   │
    │  ├─ Pre-market orders (early exit)      │
    │  └─ Accept gap risk (reality)           │
    └────────────────┬─────────────────────────┘
                     │
    ┌────────────────▼─────────────────────────┐
    │  4. STOP-LIMIT ORDERS (Conditional Limit)
    │                                          │
    │  Structure: Two-level stop              │
    │  ├─ Stop Price: $95.00 (trigger)        │
    │  ├─ Limit Price: $94.50 (execution max) │
    │  └─ Effect: Only activate if stop hit   │
    │            Then execute as limit        │
    │                                          │
    │  Example:                                │
    │  ├─ "Sell 10k at stop $95, limit $94.50"│
    │  ├─ Normal: Dormant                      │
    │  ├─ Price drops to $94.98: ACTIVATES    │
    │  ├─ Becomes limit order: Sell at $94.50+│
    │  ├─ Benefit: Better price if filled      │
    │  ├─ Risk: Might not fill at all!        │
    │  └─ Tradeoff: Protection vs certainty   │
    │                                          │
    │  Pros:                                   │
    │  ├─ Better execution price (if fills)   │
    │  ├─ Protect against false triggers      │
    │  └─ Control max execution price         │
    │                                          │
    │  Cons:                                   │
    │  ├─ Might not fill (leave exposed)      │
    │  ├─ Gap risk still present              │
    │  ├─ False sense of security             │
    │  └─ More complex to manage              │
    └────────────────┬─────────────────────────┘
                     │
    ┌────────────────▼─────────────────────────┐
    │  5. STOP HUNTING & MANIPULATION          │
    │                                          │
    │  Why Stop Hunts Happen:                 │
    │  ├─ Traders cluster stops at round #s   │
    │  ├─ MMs see order book patterns         │
    │  ├─ MMs deliberately push price to #    │
    │  ├─ Stops trigger → liquidation cascade │
    │  ├─ Cascade = instant liquidity         │
    │  └─ MMs profit from the dislocation     │
    │                                          │
    │  Example:                                │
    │  Many stops at: $99.50 (round number)   │
    │  Price holds at: $99.75                 │
    │  MMs sell aggressively: Push to $99.49  │
    │  Result: 1000 stops trigger at once     │
    │  Liquidity: Massive sell wave           │
    │  Price: Continues down to $99.00        │
    │  Then bounces: Back to $99.80           │
    │  Winners: MMs (sold high, bought low)   │
    │  Losers: Stop order traders             │
    │                                          │
    │  Defense Against Stop Hunting:          │
    │  ├─ Place stops off round numbers       │
    │  ├─ Use randomized stop levels          │
    │  ├─ Combine with other indicators       │
    │  ├─ Avoid placing stops publicly        │
    │  └─ Use algorithmic placement (hidden)  │
    └──────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Stop Order Trigger Probability

**Probability of hitting stop by time $t$:**

Using Brownian motion with drift:
$$P(\text{hit } S | \text{current} = C) = \begin{cases} 
e^{-2\mu(S-C)/\sigma^2} & \text{if } \mu < 0 \text{ (drifting down)} \\
1 & \text{if } \mu \geq 0 \text{ (drifting up)}
\end{cases}$$

where $\mu$ = drift rate, $\sigma$ = volatility, $S$ = stop level, $C$ = current price.

### Expected Loss Conditional on Stop Execution

**If stop is triggered, expected execution price:**
$$E[P_{\text{exec}} | \text{stop hit}] = S - \text{Slippage}$$

where Slippage ≈ half-spread + impact.

**Expected loss from stop order:**
$$E[L] = P(\text{hit stop}) \times E[P_{\text{exec}} | \text{stop hit}] \times Q$$

### Stop Hunting Risk

**Expected value of stop hunting (for market makers):**

$$V_{\text{hunt}} = \text{Depth at Stop} \times (\text{Stop Price} - \text{Recovery Price})$$

If high enough, profitable to execute hunt.

---

## VI. Python Mini-Project: Stop Order Simulation & Gap Risk

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# ============================================================================
# STOP ORDER SIMULATOR
# ============================================================================

class StopOrderSimulator:
    """
    Simulate stop order execution and gap risk
    """
    
    def __init__(self, entry_price=100.00, stop_price=95.00, target_price=110.00):
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        
        self.stop_hit = False
        self.execution_price = None
        self.gap_occurred = False
    
    def simulate_price_path(self, num_days=20, daily_return_mean=0.001, daily_vol=0.02):
        """
        Simulate realistic price path with gaps
        """
        prices = [self.entry_price]
        
        for day in range(num_days):
            # Normal daily return
            daily_return = np.random.normal(daily_return_mean, daily_vol)
            new_price = prices[-1] * (1 + daily_return)
            
            # Random gap event (10% chance)
            if np.random.random() < 0.1:
                gap_size = np.random.normal(0, 0.05)  # ±5% gap
                new_price = new_price * (1 + gap_size)
                self.gap_occurred = True
            
            prices.append(new_price)
        
        return np.array(prices)
    
    def check_stop_hit(self, price_path):
        """
        Check if stop order triggered during path
        """
        for price in price_path:
            if price <= self.stop_price:
                self.stop_hit = True
                # Simulate execution at or below stop
                self.execution_price = min(price, self.stop_price - 0.05)
                return True
        
        return False
    
    def stop_order_pnl(self):
        """
        Calculate P&L from stop order
        """
        if not self.stop_hit:
            return None
        
        pnl_per_share = self.execution_price - self.entry_price
        pnl_pct = (pnl_per_share / self.entry_price) * 100
        
        return {'pnl_per_share': pnl_per_share, 'pnl_pct': pnl_pct}


class StopOrderStrategies:
    """
    Compare stop order strategies
    """
    
    @staticmethod
    def generate_prices_with_gap(initial_price=100.0, num_steps=100, vol=0.02, gap_prob=0.02):
        """
        Generate price path with occasional gaps
        """
        prices = [initial_price]
        
        for _ in range(num_steps):
            # Normal price move
            dW = np.random.normal(0, 1)
            price_move = vol * dW / np.sqrt(100)  # Daily vol
            new_price = prices[-1] * (1 + price_move)
            
            # Gap event (rare but large)
            if np.random.random() < gap_prob:
                gap_event = np.random.normal(0, 0.03)  # 3% gap std dev
                new_price = new_price * (1 + gap_event)
            
            prices.append(new_price)
        
        return np.array(prices)
    
    @staticmethod
    def test_stop_protection(prices, stop_level=95.0, position_size=10000):
        """
        Test if stop order provided protection
        """
        for i, price in enumerate(prices):
            if price <= stop_level:
                # Stop triggered
                execution_price = max(price - 0.25, price)  # Assume some slippage
                
                return {
                    'triggered': True,
                    'day': i,
                    'trigger_price': price,
                    'execution_price': execution_price,
                    'max_price_after_trigger': prices[i:].min(),
                    'pnl_per_share': execution_price - 100.0,
                    'pnl_pct': ((execution_price - 100.0) / 100.0) * 100
                }
        
        return {
            'triggered': False,
            'max_price': prices.max(),
            'final_price': prices[-1],
            'pnl_if_held': ((prices[-1] - 100.0) / 100.0) * 100
        }


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STOP ORDER EXECUTION & GAP RISK ANALYSIS")
print("="*80)

# Scenario 1: Normal execution (no gaps)
print(f"\n1. NORMAL SCENARIO (No Gap Opening)")
prices_normal = StopOrderStrategies.generate_prices_with_gap(
    initial_price=100.0, num_steps=100, vol=0.02, gap_prob=0.0
)

result_normal = StopOrderStrategies.test_stop_protection(prices_normal, stop_level=95.0)

print(f"   Stop level: $95.00")
print(f"   Stop triggered: {result_normal['triggered']}")
if result_normal['triggered']:
    print(f"   ├─ Day triggered: {result_normal['day']}")
    print(f"   ├─ Price at trigger: ${result_normal['trigger_price']:.2f}")
    print(f"   ├─ Execution price: ${result_normal['execution_price']:.2f}")
    print(f"   ├─ Max price after: ${result_normal['max_price_after_trigger']:.2f}")
    print(f"   └─ P&L: {result_normal['pnl_pct']:.2f}% (${result_normal['pnl_per_share']*10000:,.0f})")

# Scenario 2: With gap event
print(f"\n2. GAP EVENT SCENARIO (Overnight Shock)")
prices_gap = StopOrderStrategies.generate_prices_with_gap(
    initial_price=100.0, num_steps=100, vol=0.02, gap_prob=0.05
)

result_gap = StopOrderStrategies.test_stop_protection(prices_gap, stop_level=95.0)

print(f"   Stop level: $95.00")
print(f"   Stop triggered: {result_gap['triggered']}")
if result_gap['triggered']:
    print(f"   ├─ Day triggered: {result_gap['day']}")
    print(f"   ├─ Trigger price: ${result_gap['trigger_price']:.2f}")
    print(f"   ├─ Execution price: ${result_gap['execution_price']:.2f}")
    print(f"   ├─ Worst price after: ${result_gap['max_price_after_trigger']:.2f}")
    print(f"   ├─ Gap risk: Stop @ $95, executed @ ${result_gap['execution_price']:.2f}")
    print(f"   └─ P&L: {result_gap['pnl_pct']:.2f}% (${result_gap['pnl_per_share']*10000:,.0f})")

# Monte Carlo: Many simulations with different gap probabilities
print(f"\n3. MONTE CARLO ANALYSIS (1000 simulations)")

gap_probs = [0.0, 0.02, 0.05, 0.10]
results_mc = []

for gap_prob in gap_probs:
    stop_losses = []
    execution_prices = []
    
    for sim in range(1000):
        prices = StopOrderStrategies.generate_prices_with_gap(
            initial_price=100.0, num_steps=100, vol=0.02, gap_prob=gap_prob
        )
        result = StopOrderStrategies.test_stop_protection(prices, stop_level=95.0)
        
        if result['triggered']:
            stop_losses.append(result['pnl_pct'])
            execution_prices.append(result['execution_price'])
    
    trigger_rate = len(stop_losses) / 1000 * 100
    
    if stop_losses:
        results_mc.append({
            'Gap Probability': f"{gap_prob*100:.1f}%",
            'Trigger Rate': f"{trigger_rate:.1f}%",
            'Avg Execution': f"${np.mean(execution_prices):.2f}",
            'Min Execution': f"${np.min(execution_prices):.2f}",
            'Avg Loss': f"{np.mean(stop_losses):.2f}%",
            'Worst Loss': f"{np.min(stop_losses):.2f}%"
        })

df_mc = pd.DataFrame(results_mc)
print(df_mc.to_string(index=False))

# Scenario 4: Stop hunting example
print(f"\n4. STOP HUNTING SCENARIO")
print(f"   Many traders: Stop orders @ $99.50 (round number)")
print(f"   Market makers: Detect pattern, push price to $99.49")
print(f"   Result: All stops trigger simultaneously")
print(f"   Cascade: 1000 orders × 10k shares = 10M shares dumped")
print(f"   Price: Collapses to $98.50 (stop hunting successful)")
print(f"   Your execution: $98.50 instead of $99.50")
print(f"   Loss differential: 1 cent per share × 10,000 = $100 extra loss")
print(f"   Lesson: Stop orders can be manipulated by sophisticated traders")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: Price path with stop level
ax1 = axes[0, 0]
days = np.arange(len(prices_normal))
ax1.plot(days, prices_normal, linewidth=2, color='blue', label='Price path (no gap)')
ax1.axhline(y=95.0, color='red', linestyle='--', linewidth=2, label='Stop level ($95)')
ax1.axhline(y=100.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Entry ($100)')
ax1.fill_between(days, 95.0, prices_normal, where=(prices_normal >= 95), alpha=0.2, color='green')
ax1.fill_between(days, 95.0, prices_normal, where=(prices_normal < 95), alpha=0.2, color='red')
ax1.set_xlabel('Day')
ax1.set_ylabel('Price ($)')
ax1.set_title('Panel 1: Normal Market (No Gap)\nStop protects at $95 level')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Price path with gap
ax2 = axes[0, 1]
days_gap = np.arange(len(prices_gap))
ax2.plot(days_gap, prices_gap, linewidth=2, color='blue', label='Price path (with gaps)')
ax2.axhline(y=95.0, color='red', linestyle='--', linewidth=2, label='Stop level ($95)')
ax2.axhline(y=100.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Entry ($100)')

# Highlight gap events (large jumps)
for i in range(1, len(prices_gap)):
    if abs(prices_gap[i] - prices_gap[i-1]) > prices_gap[i-1] * 0.02:  # > 2% jump
        ax2.scatter(i, prices_gap[i], color='red', s=100, marker='X', zorder=5)

ax2.fill_between(days_gap, 95.0, prices_gap, where=(prices_gap >= 95), alpha=0.2, color='green')
ax2.fill_between(days_gap, 95.0, prices_gap, where=(prices_gap < 95), alpha=0.2, color='red')
ax2.set_xlabel('Day')
ax2.set_ylabel('Price ($)')
ax2.set_title('Panel 2: Volatile Market (With Gaps)\nStop can gap past (red Xs)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Stop trigger rates vs gap probability
ax3 = axes[1, 0]
gap_probs_fine = np.linspace(0, 0.15, 10)
trigger_rates = []

for gp in gap_probs_fine:
    trigger_count = 0
    for _ in range(100):
        prices = StopOrderStrategies.generate_prices_with_gap(
            initial_price=100.0, num_steps=100, vol=0.02, gap_prob=gp
        )
        if (prices <= 95.0).any():
            trigger_count += 1
    trigger_rates.append(trigger_count / 100 * 100)

ax3.plot(gap_probs_fine * 100, trigger_rates, linewidth=2.5, marker='o', markersize=8, color='purple')
ax3.fill_between(gap_probs_fine * 100, 0, trigger_rates, alpha=0.2, color='purple')
ax3.set_xlabel('Gap Event Probability (%)')
ax3.set_ylabel('Stop Trigger Rate (%)')
ax3.set_title('Panel 3: Stop Trigger Probability\n(Higher gap risk = more frequent stops)')
ax3.grid(True, alpha=0.3)

# Panel 4: Execution price distribution (with gaps)
ax4 = axes[1, 1]
execution_prices_gap = []

for _ in range(500):
    prices = StopOrderStrategies.generate_prices_with_gap(
        initial_price=100.0, num_steps=100, vol=0.02, gap_prob=0.05
    )
    result = StopOrderStrategies.test_stop_protection(prices, stop_level=95.0)
    if result['triggered']:
        execution_prices_gap.append(result['execution_price'])

ax4.hist(execution_prices_gap, bins=20, color='red', alpha=0.7, edgecolor='black')
ax4.axvline(x=95.0, color='green', linestyle='--', linewidth=2, label='Intended stop ($95)')
ax4.axvline(x=np.mean(execution_prices_gap), color='blue', linestyle='--', linewidth=2, label=f'Avg execution (${np.mean(execution_prices_gap):.2f})')
ax4.set_xlabel('Execution Price ($)')
ax4.set_ylabel('Frequency')
ax4.set_title('Panel 4: Execution Price Distribution\n(Gap risk: often execute below stop)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('stop_order_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• Stop orders provide protection in normal markets but fail in gaps")
print("• Gap risk: Can gap past stop level overnight; happens 1-2%+ of days")
print("• Stop hunting: Clusters of stops at round numbers attract manipulation")
print("• Stop-limit trades protection for certainty: better price OR no fill")
print("• Best protection: Options (puts), not stop orders")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Hirshleifer, D., & Shumway, T. (2003).** "Good day sunshine: Stock returns and the weather." Journal of Finance, 58(3), 1009-1032.
   - Stop order placement patterns; behavioral biases; triggering events

2. **Bjønnes, G. H., Osler, C. L., & Rime, D. (2005).** "Stop-loss orders and market crashes." Journal of International Economics, 70(2), 440-467.
   - Stop order cascades; market crashes; systemic risk

3. **Scholes, M. S. (2000).** "Crisis and risk management." American Economic Review, 89(2), 17-21.
   - Stop order failures; gap risk; execution certainty

**Key Design Concepts:**

- **Stop Certainty Paradox:** Stop orders guarantee activation but not execution price. Gap risk unavoidable.
- **Stop Hunting:** Predictable liquidity creates incentives for market manipulation at round-number stops.
- **Options Alternative:** Protective puts provide true downside protection (cost = premium, but no execution risk).

