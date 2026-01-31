# Time-in-Force: Order Lifetime & Execution Policies

## I. Concept Skeleton

**Definition:** Time-in-force (TIF) specifies how long an order remains active in the market before it expires or is automatically canceled. Common types include: Day orders (expire at session close), Good-Till-Canceled (active until manually canceled), Immediate-or-Cancel (execute now or cancel remainder), Fill-or-Kill (all-or-nothing), and Good-Till-Date (expire on specific date). TIF directly controls order persistence and fills the gap between passive patience and aggressive urgency.

**Purpose:** Prevent unintended executions (avoid stale limit orders), enforce trade discipline (force decision points), manage overnight risk (prevent gap openings), enable algorithmic automation (trigger expiration logically), and adapt to market sessions (intraday vs overnight).

**Prerequisites:** Order book structure, market sessions (open/close), expiration mechanics, execution algorithms, risk management principles, trade settlement timing.

---

## II. Comparative Framing

| **TIF Type** | **Day Order (DAY)** | **Good-Till-Canceled (GTC)** | **Immediate-or-Cancel (IOC)** | **Fill-or-Kill (FOK)** | **Good-Till-Date (GTD)** |
|-----------|----------|----------|----------|----------|-----------|
| **Duration** | Until market close | Manual cancel only | Immediate (<1ms) | Single attempt | Set date |
| **Expiration** | Auto at 4pm close | Never (persists) | Instant (if partial) | Instant (if any unfilled) | Specified date/time |
| **Partial Fills** | Yes (accumulates) | Yes (accumulates) | Yes (cancel remainder) | No (all or nothing) | Yes (accumulates) |
| **Overnight Risk** | None (expires) | High (persists) | None (never reaches next day) | None (instant) | Depends on GTD date |
| **Use Case** | Intraday trades | Long-term positions | Aggressive fills | Block trades, exact qty | Multi-day strategies |
| **Typical Example** | "Buy 10k at $100, expires 4pm" | "Buy 10k at $100, keep until I cancel" | "Buy up to 10k at $100, execute now" | "Buy exactly 10k at $100, nothing less" | "Buy 10k at $100 by March 15" |
| **Execution Speed** | Slow (waits for match) | Slow (waits indefinitely) | Fast (abandon quickly) | Very fast (no patience) | Slow (waits until date) |
| **Cost** | Low (passive) | Low (passive) | High (market order urgency) | Very high (rigid requirements) | Low (passive) |
| **Fill Probability** | Medium | High (eventually fills) | Low (if partial rejected) | Very low (unless exact qty) | Medium |
| **Information Leakage** | Visible (until close) | Very high (persistent) | Very low (instant) | Low (instant) | High (persistent) |
| **Practical Advantage** | Safety (resets daily) | Convenience (no resubmit) | Efficiency (waste minimized) | Certainty (no partial liability) | Flexibility (multi-day plan) |
| **Practical Risk** | Must resubmit each day | Stale order (market moved) | Rejection if partial unfillable | Impossibly rigid | Complexity (track date) |

---

## III. Examples & Counterexamples

### Example 1: Day Order vs Persistent Order - Forgotten Limit Trade
**Setup:**
- Monday morning: You post limit buy order for 5,000 shares at $100.00
- Order type: Good-Till-Canceled (GTC) — intended to be passive, multi-day
- Actual activity: Price never hits $100.00 on Monday (stock at $101-$105 range)
- Your monitoring: Forget about the order (busy with other trades)

**Timeline:**
| Time | Market Price | Your Order | Status |
|------|------------|-----------|--------|
| Mon 9:30am | $102.50 | GTC $100 posted | Sitting in book |
| Mon 4:00pm | $104.00 | Still active | Market close, order persists |
| Tue 9:30am | $98.50 | **FILLS** | You didn't see it! |
| Tue 10:00am | $97.50 | Holding long 5k | Price fell after fill |

**Problem:**
- You forgot about the order
- Price fell to $98.50; order filled you at $100.00
- You bought at worst time (price fell 1.5% afterward)
- You're now holding unwanted position (holding loss)

**Alternative with Day Order:**
- Post limit at $100.00 with Day expiration
- Monday 4:00pm: Order expires (didn't fill)
- Tuesday: Order gone; clean slate; no accidental fills
- You must repost Tuesday if still interested (active discipline)

**Lesson:** GTC convenient but risky (stale orders). Day orders safer (forced review).

---

### Example 2: IOC (Immediate-or-Cancel) - Aggressive Partial Execution
**Setup:**
- Scenario: You need 50,000 shares urgently
- Order: "Buy 50k IOC at $100.00 or better"
- Market: Bid $99.95, Ask $100.02; only 30,000 available at best prices
- Order type: Immediate-or-Cancel (execute now or cancel what can't fill)

**Execution:**
| Level | Available | Price | Your IOC Order |
|--------|-----------|-------|---------|
| Level 1 | 25,000 @ $99.95 | Bid | Fill 25,000 |
| Level 2 | 5,000 @ $99.97 | Bid | Fill 5,000 |
| Remaining Need | 20,000 | - | Can't fill at $100, below! |
| **Action** | - | - | **CANCEL REMAINDER** |

**Result:**
- Filled: 30,000 shares (60% of order)
- Unfilled: 20,000 cancelled (NOT placed in book)
- Your final qty: 30,000 shares (not the 50k you wanted)
- Advantage: No hanging limit order (clean exit)
- Disadvantage: Partial fill (needed 50k, got 30k)

**Alternative with Day Order:**
- Post limit 50k at $100.00 Day
- Fills 30,000 immediately
- Remaining 20,000 sits in book
- Problem: You're committed to 20k more if price hits $100 (might not want it)

**Lesson:** IOC for aggressive fills (abandons if can't fill completely in that instant).

---

### Example 3: FOK (Fill-or-Kill) - Block Trade
**Setup:**
- Large institutional order: Buy 100,000 shares as single block
- Reason: Must keep weighted average price tight (quantity matters)
- Order: "Buy exactly 100k FOK at $100.00 or better"
- Market: Only 80,000 available at best prices

**Execution:**
- Your FOK order posts
- Exchange attempts to fill 100,000
- Can only get 80,000
- Status: **ORDER REJECTED / KILLED** (not partially filled)
- Result: 0 shares executed (all-or-nothing failed)

**Why FOK is Used:**
- Portfolio balancing: Need exact size or none (weighting)
- Pair trades: Must get both legs simultaneously
- Arbitrage: Both sides must work or unwind both

**Alternative with IOC:**
- IOC posts: Gets 80,000
- Remainder cancelled
- You have 80,000 but needed 100,000 (incomplete trade)
- Leaves you exposed (only half-hedged)

**Lesson:** FOK rigid but needed for structured trades; IOC more flexible (accepts partial).

---

## IV. Layer Breakdown

```
TIME-IN-FORCE FRAMEWORK

┌──────────────────────────────────────────────────┐
│        TIME-IN-FORCE ORDER LIFECYCLE              │
│                                                   │
│  Core: When does an order expire/cancel?         │
│        TIF defines survival duration              │
└────────────────────┬─────────────────────────────┘
                     │
    ┌────────────────▼─────────────────────────┐
    │  1. DAY ORDERS (Most Common)              │
    │                                           │
    │  Specification:                           │
    │  ├─ Type: Day (auto-expires at close)   │
    │  ├─ Valid Time: Market open → market close
    │  ├─ Auto-expires: 4:00pm ET (or close)  │
    │  └─ Status at close: CANCELLED           │
    │                                           │
    │  Execution Timeline:                      │
    │  9:30am: Post limit order $100 (Day)     │
    │  │       Qty: 5,000 shares               │
    │  │       Status: Active in book          │
    │  │                                        │
    │  12:30pm: Order hasn't filled            │
    │  │        Still waiting                  │
    │  │        Status: Still active           │
    │  │                                        │
    │  2:45pm: Price moves up, misses your fill│
    │  │       Status: Still active (hours left)
    │  │                                        │
    │  4:00pm: MARKET CLOSE                     │
    │  │       Your unfilled limit → CANCELLED │
    │  │       Status: No longer active        │
    │  │       Action: Must repost tomorrow    │
    │                                           │
    │  Advantages:                              │
    │  ├─ Auto-cancellation (no stale orders)  │
    │  ├─ Forces daily review                  │
    │  ├─ No overnight gap risk                │
    │  ├─ Clean slate each day                 │
    │  └─ Reduces information leakage          │
    │                                           │
    │  Disadvantages:                           │
    │  ├─ Must resubmit if not filled          │
    │  ├─ Multi-day trades require daily repost│
    │  ├─ Inconvenient for patient entry/exit  │
    │  └─ Market remembers you (repeated reposts)
    └────────────────┬─────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  2. GOOD-TILL-CANCELED (GTC)              │
    │                                            │
    │  Specification:                            │
    │  ├─ Type: GTC (persist indefinitely)      │
    │  ├─ Valid Time: Post time → manual cancel │
    │  ├─ Duration: Days, weeks, months(!!)     │
    │  ├─ Expiration: Only by trader action     │
    │  └─ Risk: Stale order persists            │
    │                                            │
    │  Execution Timeline:                       │
    │  Monday 10am: Post limit $100 (GTC)       │
    │  │           Qty: 5,000 shares            │
    │  │           Status: Active, in book      │
    │  │                                         │
    │  Monday 4pm: Market close                 │
    │  │          Your order: STILL ACTIVE      │
    │  │          Persists through night        │
    │  │          No expiration!                │
    │  │                                         │
    │  Tuesday 9:30am: Market opens             │
    │  │              Order still there!        │
    │  │              Status: Active            │
    │  │              Problem: You forgot it?   │
    │  │                                         │
    │  Tuesday 10:00am: Price hits $100         │
    │  │                **ORDER FILLS**          │
    │  │                You bought 5k           │
    │  │                (Maybe you didn't want!)│
    │  │                                         │
    │  Wednesday: You discover fill             │
    │  │          "When did I buy these?"       │
    │  │          Position value: ??? (moved)   │
    │  │          Status: Underwater(-1%)       │
    │  │                                         │
    │  Advantages:                               │
    │  ├─ No resubmit needed (convenient)       │
    │  ├─ Long-term entry strategy possible     │
    │  ├─ Patient execution (price comes to you)│
    │  └─ Less visible (one order, not daily)   │
    │                                            │
    │  Disadvantages:                            │
    │  ├─ Forget about it (mental hazard)       │
    │  ├─ Stale orders (market moved)           │
    │  ├─ Accidental fills (unexpected)         │
    │  ├─ Overnight gap risk (open next day)    │
    │  └─ Administrative burden (track list)    │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  3. IMMEDIATE-OR-CANCEL (IOC)             │
    │                                            │
    │  Specification:                            │
    │  ├─ Type: IOC (aggressive limit)          │
    │  ├─ Execution: All-or-partial allowed     │
    │  ├─ Unfilled: Cancelled immediately       │
    │  ├─ Duration: <1 millisecond              │
    │  └─ Result: No standing order in book     │
    │                                            │
    │  Execution Mechanism:                      │
    │  10:00:00.000 Post: "Buy 50k IOC @ $100"  │
    │  │                                         │
    │  10:00:00.001 Matching engine:            │
    │  │  ├─ Check bid side for 50k available   │
    │  │  ├─ Find: 30,000 @ $99.98-$99.99      │
    │  │  ├─ Fill: 30,000                       │
    │  │  └─ Remaining: 20,000 needed           │
    │  │                                         │
    │  10:00:00.002 Decision:                   │
    │  │  ├─ Can fill all 50k? NO              │
    │  │  ├─ IOC says: Cancel remainder        │
    │  │  ├─ Action: Kill 20,000 unsold        │
    │  │  └─ Result: Only 30k filled           │
    │  │                                         │
    │  10:00:00.003 Status:                     │
    │  │  No standing order (all cancelled)     │
    │  │  No book presence (immediate exit)     │
    │  │  Final qty: 30,000 shares              │
    │                                            │
    │  Advantages:                               │
    │  ├─ No hanging orders (clean)             │
    │  ├─ Limit on urgency (execute now/abandon)│
    │  ├─ No overnight risk (instant decision)  │
    │  └─ Aggressive limit (captures momentum)  │
    │                                            │
    │  Disadvantages:                            │
    │  ├─ Partial fills (might not complete)    │
    │  ├─ No persistence (one-shot execution)   │
    │  ├─ Rejection risk (if any can't fill)    │
    │  └─ More complex trade management         │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  4. FILL-OR-KILL (FOK)                    │
    │                                            │
    │  Specification:                            │
    │  ├─ Type: FOK (all-or-nothing)            │
    │  ├─ Execution: Full qty or rejected       │
    │  ├─ Partial Fills: NOT allowed            │
    │  ├─ Duration: <1 millisecond              │
    │  └─ Result: 100% filled or 0% filled      │
    │                                            │
    │  Execution Mechanism:                      │
    │  Post: "Buy exactly 100k FOK @ $100"      │
    │  │                                         │
    │  Matching:                                 │
    │  ├─ Check: Can I fill 100,000?            │
    │  ├─ Available: Only 80,000 @ best price   │
    │  ├─ Result: NO (can't fill 100%)          │
    │  └─ Action: **REJECT ORDER** (FOK rule)   │
    │                                            │
    │  Status: 0 shares filled (all-or-nothing)│
    │  Consequence: Trade failed completely    │
    │                                            │
    │  vs IOC equivalent:                        │
    │  Post: "Buy up to 100k IOC @ $100"        │
    │  Matching: 80,000 filled, 20k cancelled   │
    │  Status: 80,000 filled (partial OK)       │
    │                                            │
    │  Use Case (FOK needed):                    │
    │  ├─ Block trades (must be size)           │
    │  ├─ Paired trades (both sides or none)    │
    │  ├─ Arbitrage (risk/reward depends on qty)│
    │  └─ Portfolio rebalancing (exact weights) │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  5. GOOD-TILL-DATE (GTD)                  │
    │                                            │
    │  Specification:                            │
    │  ├─ Type: GTD (expire on date)            │
    │  ├─ Valid Time: Post → GTD date           │
    │  ├─ Duration: Days, weeks, months         │
    │  ├─ Auto-expires: At specified GTD time   │
    │  └─ Compromise: GTC discipline + DAY safety
    │                                            │
    │  Example:                                  │
    │  Today (March 1): Post limit $100 (GTD)   │
    │  GTD Date: March 15 (2 weeks out)         │
    │  │                                         │
    │  March 1-14: Order active in book         │
    │  │           Persists across all days     │
    │  │           Multi-day patience strategy  │
    │  │                                         │
    │  March 14 11:59pm: Last moment alive      │
    │  │                 Status: Still active   │
    │  │                                         │
    │  March 15 12:00am: GTD expires            │
    │  │                 Automatic cancellation│
    │  │                 No more standing order│
    │  │                                         │
    │  Benefits:                                 │
    │  ├─ Long-lived (not day-limited)          │
    │  ├─ Auto-expiring (not forever like GTC)  │
    │  ├─ Bounded (won't persist indefinitely)  │
    │  ├─ Disciplined (forces deadline)         │
    │  └─ Flexible (chose your date)            │
    └──────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Order Expiration Probability

**Day order:** Probability of being active at time $t$:
$$P(\text{active at } t) = \begin{cases} 1 & \text{if } t < T_{\text{close}} \\ 0 & \text{if } t \geq T_{\text{close}} \end{cases}$$

**GTC order:** Probability of being active at time $t$:
$$P(\text{active at } t) = 1 \text{ (until manually canceled or filled)}$$

### IOC Partial Fill Probability

**Given order size $Q$ and available liquidity $L$:**
$$P(\text{fill all IOC}) = \mathbb{1}[L \geq Q]$$

$$P(\text{partial fill IOC}) = \mathbb{1}[L < Q] \times \frac{L}{Q}$$

$$P(\text{no fill IOC}) = 1 - P(\text{partial}) - P(\text{full})$$

### FOK Rejection Probability

$$P(\text{FOK rejected}) = P(L < Q)$$

where $L$ = available liquidity at the limit price.

---

## VI. Python Mini-Project: TIF Comparison & Lifecycle Management

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

# ============================================================================
# TIME-IN-FORCE SIMULATOR
# ============================================================================

class OrderLifecycle:
    """
    Simulate order execution under different TIF rules
    """
    
    def __init__(self, order_qty=5000, limit_price=100.00):
        self.order_qty = order_qty
        self.limit_price = limit_price
        self.tif = None
        self.filled_qty = 0
        self.fill_price = None
        self.execution_day = None
        self.status = 'pending'
    
    def simulate_day_order(self, prices, days_elapsed):
        """
        Day order: expires at market close
        """
        market_close_hour = 16  # 4pm
        
        for day in days_elapsed:
            hour_of_day = day % 1 * 24
            
            if hour_of_day > market_close_hour:
                # Market closed, order expired
                self.status = 'expired_day_close'
                break
            
            # Check if price hit during day
            if prices[int(day)] <= self.limit_price:
                self.filled_qty = self.order_qty
                self.fill_price = min(prices[int(day)], self.limit_price)
                self.execution_day = int(day)
                self.status = 'filled'
                break
        
        if self.status == 'pending':
            self.status = 'expired_day_close'
        
        return self
    
    def simulate_gtc_order(self, prices):
        """
        GTC order: persists until canceled or filled
        """
        for day, price in enumerate(prices):
            if price <= self.limit_price:
                self.filled_qty = self.order_qty
                self.fill_price = min(price, self.limit_price)
                self.execution_day = day
                self.status = 'filled'
                return self
        
        # Never filled
        self.status = 'gtc_unfilled'
        return self
    
    def simulate_ioc_order(self, available_qty_at_limit):
        """
        IOC order: fill what's available, cancel remainder immediately
        """
        if available_qty_at_limit >= self.order_qty:
            # Full fill possible
            self.filled_qty = self.order_qty
            self.fill_price = self.limit_price
            self.status = 'filled_ioc_full'
        elif available_qty_at_limit > 0:
            # Partial fill, then cancel
            self.filled_qty = available_qty_at_limit
            self.fill_price = self.limit_price
            self.status = 'filled_ioc_partial'
        else:
            # No fill, cancel all
            self.filled_qty = 0
            self.status = 'rejected_ioc_no_fill'
        
        return self
    
    def simulate_fok_order(self, available_qty_at_limit):
        """
        FOK order: all-or-nothing
        """
        if available_qty_at_limit >= self.order_qty:
            self.filled_qty = self.order_qty
            self.fill_price = self.limit_price
            self.status = 'filled_fok'
        else:
            self.filled_qty = 0
            self.status = 'rejected_fok_insufficient_qty'
        
        return self


class TIFComparison:
    """
    Compare different TIF strategies across market scenarios
    """
    
    def __init__(self, order_qty=5000, limit_price=100.00):
        self.order_qty = order_qty
        self.limit_price = limit_price
    
    def generate_price_path(self, num_days=10, daily_vol=0.02):
        """Generate price path"""
        prices = np.random.normal(100, 0.5, num_days)
        prices = np.cumsum(prices) / num_days + 99
        return prices
    
    def run_comparison(self, prices, available_qty_at_limit=3000):
        """Run all TIF strategies on same price path"""
        results = {}
        
        # Day order (simulates multiple days, trying each day)
        day_orders_filled = 0
        for day in range(len(prices)):
            order_day = OrderLifecycle(self.order_qty, self.limit_price)
            order_day.simulate_day_order(prices, [day])
            if order_day.status == 'filled':
                day_orders_filled += 1
                results['day_filled'] = (order_day.fill_price, order_day.filled_qty)
                break
        
        if 'day_filled' not in results:
            results['day_filled'] = (None, 0)  # Never filled in window
        
        # GTC order (single order, entire period)
        order_gtc = OrderLifecycle(self.order_qty, self.limit_price)
        order_gtc.simulate_gtc_order(prices)
        results['gtc'] = (order_gtc.fill_price, order_gtc.filled_qty, order_gtc.execution_day if order_gtc.filled_qty > 0 else None)
        
        # IOC order (one-shot aggressive)
        order_ioc = OrderLifecycle(self.order_qty, self.limit_price)
        order_ioc.simulate_ioc_order(available_qty_at_limit)
        results['ioc'] = (order_ioc.fill_price, order_ioc.filled_qty)
        
        # FOK order (all-or-nothing)
        order_fok = OrderLifecycle(self.order_qty, self.limit_price)
        order_fok.simulate_fok_order(available_qty_at_limit)
        results['fok'] = (order_fok.fill_price, order_fok.filled_qty)
        
        return results


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TIME-IN-FORCE ORDER COMPARISON")
print("="*80)

# Setup
order_qty = 5000
limit_price = 100.00

# Scenario 1: Price doesn't reach limit
print(f"\n1. PRICE STAYS ABOVE LIMIT (no fill scenario)")
prices_no_hit = np.linspace(100.5, 102.0, 10)  # Price rises, never hits $100

comp = TIFComparison(order_qty, limit_price)
results_no_hit = comp.run_comparison(prices_no_hit, available_qty_at_limit=3000)

print(f"   Price range: ${prices_no_hit.min():.2f} - ${prices_no_hit.max():.2f}")
print(f"   Limit: $100.00")
print(f"\n   DAY Order:")
print(f"   ├─ Status: {results_no_hit['day_filled'][1] > 0 and 'Filled' or 'Expired'}")
print(f"   ├─ Filled qty: {results_no_hit['day_filled'][1]}")
print(f"   └─ Remaining: Repost next day needed")

print(f"\n   GTC Order:")
if results_no_hit['gtc'][1] > 0:
    print(f"   ├─ Status: Filled on day {results_no_hit['gtc'][2]}")
    print(f"   ├─ Filled qty: {results_no_hit['gtc'][1]}")
    print(f"   └─ Avg price: ${results_no_hit['gtc'][0]:.2f}")
else:
    print(f"   ├─ Status: Still active (unfilled)")
    print(f"   ├─ Filled qty: 0")
    print(f"   └─ Risk: Persists indefinitely (forgotten order)")

print(f"\n   IOC Order (one attempt):")
print(f"   ├─ Status: {results_no_hit['ioc'][1] > 0 and 'Partial fill' or 'No fill'}")
print(f"   ├─ Filled qty: {results_no_hit['ioc'][1]}")
print(f"   └─ Unfilled: Cancelled immediately")

print(f"\n   FOK Order (all-or-nothing):")
print(f"   ├─ Status: {'Filled' if results_no_hit['fok'][1] > 0 else 'Rejected'}")
print(f"   ├─ Filled qty: {results_no_hit['fok'][1]}")
print(f"   └─ Reason: {results_no_hit['fok'][1] == 0 and 'Insufficient liquidity' or 'Order too large'}")

# Scenario 2: Price hits limit during period
print(f"\n2. PRICE HITS LIMIT (execution scenario)")
prices_with_hit = np.array([101.0, 100.5, 99.8, 100.2, 101.0, 100.8, 99.9, 100.1, 100.3, 100.5])

results_with_hit = comp.run_comparison(prices_with_hit, available_qty_at_limit=3000)

print(f"   Price range: ${prices_with_hit.min():.2f} - ${prices_with_hit.max():.2f}")
print(f"   Limit: $100.00")
print(f"   Liquidity available: 3,000 shares")

print(f"\n   DAY Order (daily repost):")
print(f"   ├─ Status: {results_with_hit['day_filled'][1] > 0 and 'Filled' or 'Not filled in window'}")
print(f"   ├─ Filled qty: {results_with_hit['day_filled'][1]}")
print(f"   └─ Advantage: Daily reset (stale order impossible)")

print(f"\n   GTC Order:")
if results_with_hit['gtc'][1] > 0:
    print(f"   ├─ Status: Filled on day {results_with_hit['gtc'][2]}")
    print(f"   ├─ Filled qty: {results_with_hit['gtc'][1]}")
    print(f"   ├─ Fill price: ${results_with_hit['gtc'][0]:.2f}")
    print(f"   └─ Advantage: Single order, multi-day patience")
else:
    print(f"   ├─ Status: Unfilled (price didn't hit)")
    print(f"   └─ Status: Still active (never expires)")

print(f"\n   IOC Order:")
print(f"   ├─ Status: {'Partial fill' if results_with_hit['ioc'][1] > 0 else 'No fill'}")
print(f"   ├─ Filled qty: {results_with_hit['ioc'][1]} (out of 5k requested)")
print(f"   └─ Remaining: Cancelled immediately")

print(f"\n   FOK Order:")
print(f"   ├─ Status: {'Filled!' if results_with_hit['fok'][1] > 0 else 'Rejected'}")
print(f"   ├─ Filled qty: {results_with_hit['fok'][1]}")
print(f"   └─ Reason: {'Insufficient liquidity (only 3k available, need 5k)' if results_with_hit['fok'][1] == 0 else 'All filled'}")

# Scenario 3: Monte Carlo - which TIF works best?
print(f"\n3. MONTE CARLO ANALYSIS (100 simulations, different scenarios)")

tif_stats = {
    'day': {'filled_count': 0, 'total_qty': 0, 'avg_price': []},
    'gtc': {'filled_count': 0, 'total_qty': 0, 'avg_price': []},
    'ioc': {'filled_count': 0, 'total_qty': 0, 'avg_price': []},
    'fok': {'filled_count': 0, 'total_qty': 0, 'avg_price': []}
}

for sim in range(100):
    prices = np.random.normal(100, 1, 20)  # Random walk
    available = np.random.randint(2000, 4500)  # Varying liquidity
    
    comp = TIFComparison(5000, 100.00)
    results = comp.run_comparison(prices, available_qty_at_limit=available)
    
    # DAY
    if results['day_filled'][1] > 0:
        tif_stats['day']['filled_count'] += 1
        tif_stats['day']['total_qty'] += results['day_filled'][1]
        tif_stats['day']['avg_price'].append(results['day_filled'][0])
    
    # GTC
    if results['gtc'][1] > 0:
        tif_stats['gtc']['filled_count'] += 1
        tif_stats['gtc']['total_qty'] += results['gtc'][1]
        tif_stats['gtc']['avg_price'].append(results['gtc'][0])
    
    # IOC
    if results['ioc'][1] > 0:
        tif_stats['ioc']['filled_count'] += 1
        tif_stats['ioc']['total_qty'] += results['ioc'][1]
        tif_stats['ioc']['avg_price'].append(results['ioc'][0])
    
    # FOK
    if results['fok'][1] > 0:
        tif_stats['fok']['filled_count'] += 1
        tif_stats['fok']['total_qty'] += results['fok'][1]
        tif_stats['fok']['avg_price'].append(results['fok'][0])

print(f"\n   Statistic              DAY      GTC      IOC      FOK")
print(f"   ─────────────────────────────────────────────────────")
print(f"   Fill rate:         {tif_stats['day']['filled_count']/100*100:>5.0f}%   {tif_stats['gtc']['filled_count']/100*100:>5.0f}%   {tif_stats['ioc']['filled_count']/100*100:>5.0f}%   {tif_stats['fok']['filled_count']/100*100:>5.0f}%")
print(f"   Avg qty filled:    {tif_stats['day']['total_qty']/100:>5.0f}   {tif_stats['gtc']['total_qty']/100:>5.0f}   {tif_stats['ioc']['total_qty']/100:>5.0f}   {tif_stats['fok']['total_qty']/100:>5.0f}")
if tif_stats['day']['avg_price']:
    print(f"   Avg exec price:   ${np.mean(tif_stats['day']['avg_price']):>6.2f}  ${np.mean(tif_stats['gtc']['avg_price']):>6.2f}  ${np.mean(tif_stats['ioc']['avg_price']):>6.2f}  ${np.mean(tif_stats['fok']['avg_price']):>6.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: Price path with limit level (no hit)
ax1 = axes[0, 0]
days1 = np.arange(len(prices_no_hit))
ax1.plot(days1, prices_no_hit, linewidth=2.5, marker='o', markersize=6, color='blue', label='Price')
ax1.axhline(y=100.0, color='red', linestyle='--', linewidth=2, label='Limit $100')
ax1.fill_between(days1, 100.0, prices_no_hit, alpha=0.2, color='blue')
ax1.set_xlabel('Day')
ax1.set_ylabel('Price ($)')
ax1.set_title('Panel 1: No Fill Scenario\n(Price stays above limit)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Price path with hit
ax2 = axes[0, 1]
days2 = np.arange(len(prices_with_hit))
ax2.plot(days2, prices_with_hit, linewidth=2.5, marker='o', markersize=6, color='blue', label='Price')
ax2.axhline(y=100.0, color='red', linestyle='--', linewidth=2, label='Limit $100')
ax2.fill_between(days2, 100.0, prices_with_hit, where=(prices_with_hit <= 100), alpha=0.3, color='red', label='Fill opportunity')
ax2.scatter([2, 6, 8], [99.8, 99.9, 99.95], color='red', s=100, marker='*', zorder=5, label='Fill candidates')
ax2.set_xlabel('Day')
ax2.set_ylabel('Price ($)')
ax2.set_title('Panel 2: Execution Scenario\n(Price crosses limit, multiple times)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: TIF fill rates comparison
ax3 = axes[1, 0]
tif_types = ['DAY', 'GTC', 'IOC', 'FOK']
fill_rates = [
    tif_stats['day']['filled_count'] / 100 * 100,
    tif_stats['gtc']['filled_count'] / 100 * 100,
    tif_stats['ioc']['filled_count'] / 100 * 100,
    tif_stats['fok']['filled_count'] / 100 * 100
]
colors_tif = ['green', 'blue', 'orange', 'red']

bars = ax3.bar(tif_types, fill_rates, color=colors_tif, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Fill Rate (%)')
ax3.set_title('Panel 3: Fill Rate Comparison\n(Monte Carlo 100 sims)')
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, rate in zip(bars, fill_rates):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{rate:.0f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 4: TIF characteristics matrix
ax4 = axes[1, 1]
ax4.axis('off')

characteristics = [
    ['Aspect', 'DAY', 'GTC', 'IOC', 'FOK'],
    ['Persistence', 'Expires', 'Forever', '1ms', '1ms'],
    ['Partial Fills', 'Yes', 'Yes', 'Yes', 'No'],
    ['Fill Prob', 'Medium', 'High', 'Low', 'Very Low'],
    ['Info Leakage', 'Daily', 'High', 'Minimal', 'Minimal'],
    ['Overnight Risk', 'None', 'High', 'None', 'None'],
    ['Best For', 'Intraday', 'Patient', 'Aggressive', 'Structured']
]

table = ax4.table(cellText=characteristics, cellLoc='center', loc='center',
                  colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color header row
for i in range(len(characteristics[0])):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows alternately
for i in range(1, len(characteristics)):
    for j in range(len(characteristics[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F5E9')
        else:
            table[(i, j)].set_facecolor('#F5F5F5')

ax4.set_title('Panel 4: TIF Characteristics & Use Cases', fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('time_in_force_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• DAY orders: Safe (reset daily) but inconvenient (must repost)")
print("• GTC orders: Convenient (persist) but risky (forgotten fills)")
print("• IOC orders: Aggressive (fill-what-you-can) but might leave gaps")
print("• FOK orders: Rigid (all-or-nothing) but needed for structured trades")
print("• Best practice: Use DAY + IOC combo (daily discipline + aggressive efficiency)")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Foucault, T., Kadan, O., & Kandel, E. (2005).** "Limit order book as a market for liquidity." Review of Financial Studies, 18(4), 1171-1217.
   - Order persistence strategies; GTC vs Day order dynamics

2. **Biais, B., Hillion, P., & Spatt, C. (1995).** "An empirical analysis of the limit order book and the order flow in the Paris Bourse." Journal of Finance, 50(5), 1655-1689.
   - TIF choice patterns; execution strategies

**Key Design Concepts:**

- **Expiration Discipline:** TIF enforces order review cycles; daily resets prevent stale positions.
- **Execution vs Persistence Trade-off:** IOC/FOK sacrifices persistence for speed; GTC trades convenience for risk.
- **Overnight Gap Protection:** Day orders eliminate overnight risk automatically; GTC requires manual management.

