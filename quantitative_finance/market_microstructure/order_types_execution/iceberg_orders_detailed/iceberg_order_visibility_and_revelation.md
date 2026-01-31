# Iceberg Orders: Hidden Depth & Strategic Visibility

## I. Concept Skeleton

**Definition:** Iceberg orders are limit orders where only a small "visible" portion (the tip) appears in the public order book, while the remainder (the hidden bulk) stays invisible until the visible portion fills. When the visible tranche fills, the next batch automatically appears. This allows large traders to accumulate size without revealing total quantity, avoiding signaling excessive demand/supply.

**Purpose:** Hide market impact information (prevent price predation), control information release (gradual visibility), manage order queue effects (disguise large orders as multiple small ones), reduce market impact (price discovery without tipping intentions), and execute patiently without showing cards.

**Prerequisites:** Order book mechanisms, information asymmetry, queue position dynamics, matching engine logic, hidden order revelation algorithms.

---

## II. Comparative Framing

| **Order Type** | **Iceberg** | **Visible Limit** | **Dark Pool** | **Pegged** | **Smart Order Routing** |
|-----------|----------|----------|----------|----------|-----------|
| **Visibility** | Partial (tip only) | Full (all qty) | None (hidden) | Full (on public book) | Multi-venue |
| **Queue Position** | Reset per layer | Single queue | Not in book | Dynamic (repriced) | Venue-dependent |
| **Information Leakage** | Gradual (layer-by-layer) | Immediate (full qty) | Maximum (off-book) | Minimal (follows book) | High (across venues) |
| **Execution Speed** | Slow (patient) | Slow (passive) | Medium (negotiated) | Fast (active fill) | Variable (route-specific) |
| **Fill Certainty** | High (multiple attempts) | Medium (single queue) | Low (negotiated only) | Medium (pegged to best) | High (venue selection) |
| **Market Impact** | Low (hidden size) | High (visible size) | Very low (off-book) | Medium (pegged reprices) | Medium (routed) |
| **Best For** | Large patient orders | Small passive orders | Block trades | Passive liquidity provision | Optimal venue access |
| **Cost Profile** | Low (passive + hidden) | Low (passive) | Low (dark) | Medium (repricing costs) | Medium (routing fees) |

---

## III. Examples & Counterexamples

### Example 1: Iceberg vs Visible Limit - The "Leakage" Problem
**Setup:**
- You need to buy 50,000 shares of XYZ
- Current market: Bid $99.98, Ask $100.02, moderate depth
- Strategy choice: Post visible limit vs iceberg

**Scenario A: Post 50,000 as single visible limit at $100.00**

| Time | Market | Your Visible Order | Book Depth | Action |
|------|--------|-----------------|------------|--------|
| 10:00am | $100.02 | 50k visible @ $100 | 25k @ ask | Sellers see huge bid! |
| 10:01am | Immediate reaction | 50k still sitting | Increases ask | Traders front-run |
| 10:02am | Price rises to $100.05 | 50k @ $100 (stale) | **No fill** | Everyone knows buyer |
| 10:03am | Price $100.08 | Still waiting | Rises higher | Your bid info leaked |
| 10:15am | $100.15 | Finally partially fills | But avg ~$100.10 | **Slippage = 10 bps!** |

**Problem:** Everyone sees your 50k; market prices up before fill; you pay excessive slippage.

**Scenario B: Post 50,000 as 5,000-share iceberg at $100.00**

| Time | Market | Visible | Hidden | Book Depth | Notes |
|------|--------|---------|--------|------------|-------|
| 10:00am | $100.02 | 5k @ $100 | 45k hidden | 25k @ ask | Small order appears innocent |
| 10:01am | $99.99 | Fills 5k | Auto-reveal next 5k | Market natural | No overreaction |
| 10:02am | $99.98 | Fills 5k | Auto-reveal next 5k | 20k @ ask | Steady accumulation |
| 10:03am | $99.97 | Fills 5k | Auto-reveal next 5k | 15k @ ask | Continues smoothly |
| 10:04am | $100.01 | Fills 5k | Auto-reveal next 5k | Down to 10k | No predation |
| ... | ... | **Pattern continues** | Iceberg melts | **Average $99.97** | **Saves 3 bps!** |

**Key Difference:**
- Visible: 50k order → predation → fills at $100.10 (10 bps cost)
- Iceberg: 5k tip → no predation → fills at $99.97 (negative slippage!)

**Lesson:** Iceberg hides size; gradual revelation doesn't trigger predatory repricing.

---

### Example 2: Iceberg Tipping - When Traders Detect the Hidden Portion
**Setup:**
- XYZ iceberg: 5k visible, 45k hidden, posting at $100.00
- Other traders notice pattern: 5k fills, immediately another 5k appears
- Observation: This happens 8 times in 2 minutes (8 × 5k = 40k)

**Traders' Detection:**
```
Time    Visible Fill    New Order Shows Up    Inference
──────────────────────────────────────────────────────
10:00   5k fills        New 5k appears        "Hmm, lucky?"
10:00:15 5k fills       New 5k appears        "Coincidence?"
10:00:30 5k fills       New 5k appears        "Pattern emerging"
10:00:45 5k fills       New 5k appears        "This is iceberg!"
10:01:00 5k fills       New 5k appears        "Definitely iceberg"
10:01:15 5k fills       New 5k appears        "I know the trick"
10:01:30 5k fills       New 5k appears        "Front-run time"
```

**What Happens When Pattern Is Detected:**

| Time | Your Iceberg | Smart Traders | Effect |
|------|-------------|----------------|--------|
| 10:02:00 | 5k visible @ $100 | See pattern, post ahead | Price jumps $100 → $100.03 |
| 10:02:15 | 5k fills (but at $100.03!) | Predict next layer | You get worse fills |
| 10:02:30 | New 5k appears @ $100 | Sellers pull offers | Price $100.05 |
| 10:02:45 | 5k fills @ $100.05 | Sellers raising ask | Pattern now expensive |

**Result:**
- Your iceberg tip detected
- Traders front-run each layer
- Fills deteriorate: First layer $99.97 → Last layer $100.08
- **Net impact: 11 bps over 10 minutes (worse than visible limit!)**

**Defense Against Tipping:**
- Randomize tip size (5k, 8k, 3k, 7k)
- Randomize timing between layers (not exact intervals)
- Vary the limit price (post new layer at $100.01, not $100.00)
- Mix with other orders (disguise pattern)

**Lesson:** Iceberg effective only if pattern isn't detected; randomization essential.

---

### Example 3: Queue Position Reset - Strategic Advantage
**Setup:**
- Ask side depth (normal limit order):
  - Level 1: 20k @ $100.02
  - Level 2: 15k @ $100.03
  - Level 3: 10k @ $100.04

**Scenario A: Post 40k buy limit at $100.02 (visible)**
- Queue position: 11th (behind 20k already there)
- To fill: Need 20k to clear first, then 15k layer, then your 40k gets position
- Waits for all those sales; by then market might have moved

**Scenario B: Post 40k iceberg (10k visible) at $100.02**
- Queue position of visible 10k: 2nd (behind 20k)
- Fills 10k
- New 10k pops into book
- **Queue position: RESETS to 1st** (new fresh order, starts at queue front!)
- Fills 10k
- Repeats: 10k pop, 10k fill, new queue position resets
- Total fills: 40k, but with 4 fresh queue positions (top each time)

**Advantage Quantification:**
- Normal 40k limit: Waits for 20k + 15k = 35k to clear (maybe 5 seconds, price moves)
- Iceberg 4×10k: Each segment gets fresh position (fills faster, market hasn't moved as much)
- **Time to full fill: Iceberg ~8 seconds vs Visible ~15 seconds**

**Lesson:** Iceberg resets queue position per layer; can fill faster despite individual tip being smaller.

---

## IV. Layer Breakdown

```
ICEBERG ORDER MECHANICS

┌──────────────────────────────────────────────────────┐
│         ICEBERG ORDER STRUCTURE & REVELATION         │
│                                                       │
│ Core: Partial visibility → gradual revelation        │
│       Hide size → control information flow           │
└────────────────────┬────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  1. ICEBERG COMPOSITION                    │
    │                                             │
    │  Total Order: 50,000 shares                │
    │  ├─ Visible Portion (Tip): 5,000 shares   │
    │  │  └─ Appears in public book              │
    │  │                                         │
    │  ├─ Hidden Portion 1: 5,000 shares        │
    │  │  └─ Dormant until tip fills            │
    │  │                                         │
    │  ├─ Hidden Portion 2: 5,000 shares        │
    │  │  └─ Dormant                            │
    │  │                                         │
    │  ├─ Hidden Portion 3: 5,000 shares        │
    │  │  └─ Dormant                            │
    │  │                                         │
    │  ├─ Hidden Portion 4: 5,000 shares        │
    │  │  └─ Dormant                            │
    │  │                                         │
    │  ├─ Hidden Portion 5: 5,000 shares        │
    │  │  └─ Dormant                            │
    │  │                                         │
    │  ├─ Hidden Portion 6: 5,000 shares        │
    │  │  └─ Dormant                            │
    │  │                                         │
    │  ├─ Hidden Portion 7: 5,000 shares        │
    │  │  └─ Dormant                            │
    │  │                                         │
    │  └─ Hidden Portion 8: 5,000 shares        │
    │     └─ Final tranche                      │
    │                                             │
    │  Total hidden: 40,000 shares (80%)         │
    │  Visibility: 10% (5k / 50k)                │
    │  Market sees: ~$500k of $5M order          │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  2. REVELATION MECHANISM                   │
    │                                             │
    │  Layer 1 Execution:                         │
    │  ┌──────────────────────────────────────┐  │
    │  │ Time: 10:00:00                       │  │
    │  │ Visible qty in book: 5,000 @ $100.00│  │
    │  │ Hidden qty: 45,000 (dormant)        │  │
    │  │ Status: Waiting for buyers           │  │
    │  │                                      │  │
    │  │ 10:00:15 - 10:00:45:                │  │
    │  │ Buyers accumulate; 5k fills         │  │
    │  │ (matching engine consumes visible)  │  │
    │  │                                      │  │
    │  │ 10:00:46 - REVELATION:              │  │
    │  │ Hidden batch 1 automatically reveals│  │
    │  │ New visible qty: 5,000 @ $100.00    │  │
    │  │ Remaining hidden: 40,000            │  │
    │  │ Status: Waiting (Layer 2 active)    │  │
    │  └──────────────────────────────────────┘  │
    │                                             │
    │  Layer 2 Execution:                         │
    │  ┌──────────────────────────────────────┐  │
    │  │ Time: 10:01:00-10:01:30              │  │
    │  │ Visible: 5,000 @ $100.00 (new batch)│  │
    │  │ Fills (buyers still demand)          │  │
    │  │ Market sees only 5k each time        │  │
    │  │ Never realizes 50k total order       │  │
    │  │                                      │  │
    │  │ 10:01:31 - REVELATION #2:           │  │
    │  │ Hidden batch 2 reveals              │  │
    │  │ New visible: 5,000 @ $100.00        │  │
    │  │ Remaining hidden: 35,000            │  │
    │  └──────────────────────────────────────┘  │
    │                                             │
    │  Layer 3-8: Same pattern repeats           │
    │  Each revelation: 1 tip pops, fills        │
    │  Then next tip appears                     │
    │  Market never sees total 50k size          │
    │                                             │
    │  Timeline Visualization:                   │
    │  10:00:00 [5k visible] → fills            │
    │  10:00:46 [5k revealed] → fills            │
    │  10:01:31 [5k revealed] → fills            │
    │  10:02:16 [5k revealed] → fills            │
    │  10:03:01 [5k revealed] → fills            │
    │  10:03:46 [5k revealed] → fills            │
    │  10:04:31 [5k revealed] → fills            │
    │  10:05:16 [5k revealed] → fills            │
    │  10:06:01 [Final] → 5k fills              │
    │                                             │
    │  Total time: ~6 minutes                    │
    │  Total filled: 50,000 shares               │
    │  Market impression: Normal small orders    │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  3. QUEUE POSITION DYNAMICS                │
    │                                             │
    │  Normal Limit (50k visible):               │
    │  ┌──────────────────────────────────────┐  │
    │  │ Time: 10:00:00                       │  │
    │  │ Ask depth:                           │  │
    │  │ ├─ 20k @ $100.02 (sellers 1st)     │  │
    │  │ ├─ 15k @ $100.03 (sellers 2nd)     │  │
    │  │ └─ Your 50k @ $100.02 (queue: 11th) │  │
    │  │                                      │  │
    │  │ Waits for 20k level to clear (slow) │  │
    │  │ Then 15k level clears (slower)      │  │
    │  │ Your order: 3rd in queue (wait time)│  │
    │  │                                      │  │
    │  │ Result: Fills slowly (market moves) │  │
    │  └──────────────────────────────────────┘  │
    │                                             │
    │  Iceberg (5k × 10 icebergs):                │
    │  ┌──────────────────────────────────────┐  │
    │  │ Time: 10:00:00 - FIRST LAYER         │  │
    │  │ Visible: 5k @ $100.02                │  │
    │  │ Queue position: 1st (fresh order!)   │  │
    │  │ Fills quickly (only 5k needed)       │  │
    │  │                                      │  │
    │  │ 10:00:46 - SECOND LAYER              │  │
    │  │ New visible: 5k @ $100.02            │  │
    │  │ Queue position: RESETS to 1st!       │  │
    │  │ Old sellers now gone (already sold)  │  │
    │  │ Fills quickly (new queue, top spot)  │  │
    │  │                                      │  │
    │  │ 10:01:31 - THIRD LAYER               │  │
    │  │ New visible: 5k @ $100.02            │  │
    │  │ Queue position: RESETS AGAIN!        │  │
    │  │ Market changed; new queue order      │  │
    │  │ Fills faster than waiting            │  │
    │  │                                      │  │
    │  │ Pattern: Every layer gets fresh queue│  │
    │  │ Total: 50k filled in 6 minutes       │  │
    │  │ Avoids deep queue penalty            │  │
    │  └──────────────────────────────────────┘  │
    │                                             │
    │  Advantage: Resets queue position per layer│
    │  (avoid deep queue; faster total fills)    │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  4. INFORMATION ASYMMETRY                  │
    │                                             │
    │  Day 1: Iceberg posted                     │
    │  ├─ Public sees: 5k bid @ $100.00         │
    │  ├─ True size: 50k (10× what's visible)   │
    │  ├─ Informed traders: Don't know (yet)    │
    │  └─ Market consensus: Normal small order  │
    │                                             │
    │  Day 1, Hour 1: First layer fills         │
    │  ├─ Public sees: Fills, then new 5k       │
    │  ├─ Smart traders: "Hmm, pattern?"        │
    │  ├─ Suspicion: Might be iceberg           │
    │  └─ Action: None (too early to be sure)   │
    │                                             │
    │  Day 1, Hour 2-3: Layers keep revealing   │
    │  ├─ Public sees: 5k → 5k → 5k fills      │
    │  ├─ Smart traders: "Definitely iceberg!"  │
    │  ├─ Suspicion: 30-40k hidden              │
    │  ├─ Action: Front-run? Raise prices?      │
    │  └─ Impact: Your fills worsen (exposed)   │
    │                                             │
    │  Defense: Randomize!                       │
    │  ├─ Vary tip size (5k, 7k, 4k, 8k)       │
    │  ├─ Vary timing (1 min, 2 min, 45 sec)   │
    │  ├─ Vary price (post at $100, then $99.99)│
    │  ├─ Mix orders (combine with others)      │
    │  └─ Result: Pattern stays hidden          │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  5. MARKET IMPACT REDUCTION                │
    │                                             │
    │  Visible 50k: Price jump (predation)       │
    │  ├─ Traders see huge bid                   │
    │  ├─ Market makers widen spread             │
    │  ├─ Sellers raise offers                   │
    │  ├─ Your fills: $100.10+ (slippage)       │
    │  └─ Cost: 10+ bps                          │
    │                                             │
    │  Iceberg 5k × 10: Gradual (no shock)       │
    │  ├─ Traders see small orders only          │
    │  ├─ Market makers keep tight spread        │
    │  ├─ Sellers don't react (no huge bid)      │
    │  ├─ Your fills: $99.98-$100.01 (steady)   │
    │  └─ Cost: 1-3 bps (lower!)                 │
    │                                             │
    │  Net Savings: 10 - 2 = 8 bps on 50k        │
    │  Dollar value: 50,000 × 0.0008 = $40       │
    │  (For $100 stock, actually $400)           │
    └──────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Iceberg Revelation Timing

**Hidden portion per layer:** $H_i = \frac{Q_{\text{total}} - Q_{\text{visible}}}{n}$

where:
- $Q_{\text{total}}$ = total order size
- $Q_{\text{visible}}$ = visible tip size
- $n$ = number of layers

**Time to reveal layer $i$:**
$$T_i = t_i + \text{(wait time to fill current tip)}$$

If Poisson arrivals of counterparties with rate $\lambda$:
$$P(\text{tip fills by time } t) = 1 - e^{-\lambda Q_{\text{visible}} t}$$

### Price Impact of Iceberg vs Visible

**Visible order price impact:**
$$\Delta P_{\text{visible}} = \alpha Q_{\text{total}} + \beta \sqrt{Q_{\text{total}}}$$

**Iceberg order price impact:**
$$\Delta P_{\text{iceberg}} = \alpha Q_{\text{visible}} + \beta \sqrt{Q_{\text{visible}}} + \text{(detection cost if exposed)}$$

**Savings if not detected:**
$$\text{Savings} = (\Delta P_{\text{visible}} - \Delta P_{\text{iceberg}}) \times Q_{\text{total}}$$

Typical: $\Delta P_{\text{visible}} \approx 10-15$ bps, $\Delta P_{\text{iceberg}} \approx 2-4$ bps (if hidden).

---

## VI. Python Mini-Project: Iceberg Order Execution Simulation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

np.random.seed(42)

# ============================================================================
# ICEBERG ORDER SIMULATOR
# ============================================================================

class IcebergOrder:
    """
    Simulate iceberg order execution with revelation mechanics
    """
    
    def __init__(self, total_qty, tip_size, limit_price, lambda_arrivals=0.8):
        self.total_qty = total_qty
        self.tip_size = tip_size
        self.limit_price = limit_price
        self.lambda_arrivals = lambda_arrivals  # Poisson arrival rate
        
        # Calculated
        self.num_layers = int(np.ceil(total_qty / tip_size))
        self.remaining_qty = total_qty
        self.filled_qty = 0
        self.layers_revealed = 0
        self.execution_prices = []
        self.detection_time = None
        self.detected = False
    
    def simulate_execution(self, market_prices, detect_pattern=False):
        """
        Simulate order execution with Poisson fill times and optional detection
        """
        layer = 0
        time_point = 0
        
        while self.remaining_qty > 0 and time_point < len(market_prices):
            # Current tip size
            current_tip = min(self.tip_size, self.remaining_qty)
            
            # Check if pattern detected (simple heuristic)
            if detect_pattern and layer > 3 and not self.detected:
                self.detected = True
                self.detection_time = time_point
            
            # Simulate fill time for this tip
            # If detected, traders front-run (worse fills)
            if self.detected:
                # Front-running: assume 5% worse prices
                effective_limit = self.limit_price * (1 - 0.0005)
                fill_prob = 0.7  # Harder to fill when exposed
            else:
                effective_limit = self.limit_price
                fill_prob = 0.95
            
            # Monte Carlo: does this tip fill?
            if np.random.random() < fill_prob:
                # Fill the tip at market price (or limit, whichever is better)
                actual_price = min(market_prices[time_point], effective_limit)
                self.execution_prices.append(actual_price)
                self.filled_qty += current_tip
                self.remaining_qty -= current_tip
                self.layers_revealed += 1
                
                # Move to next time period
                time_point += 1
                layer += 1
            else:
                # Didn't fill this period
                time_point += 1
        
        return self
    
    def get_average_execution_price(self):
        """Calculate VWAP of execution"""
        if not self.execution_prices:
            return None
        return np.mean(self.execution_prices)


class VisibleOrder:
    """
    Simulate traditional visible limit order (for comparison)
    """
    
    def __init__(self, total_qty, limit_price, lambda_arrivals=0.8):
        self.total_qty = total_qty
        self.limit_price = limit_price
        self.lambda_arrivals = lambda_arrivals
        
        self.filled_qty = 0
        self.execution_prices = []
    
    def simulate_execution(self, market_prices):
        """
        Simulate visible order: waits longer due to queue position
        All-or-nothing: either fills all or none (simplified)
        """
        
        # Visible orders take longer (queue penalty)
        for i, price in enumerate(market_prices):
            # Queue penalty: need to wait longer before fill probability increases
            wait_penalty = i / len(market_prices)  # Increases with time
            
            if price <= self.limit_price:
                fill_prob = 0.5 + 0.4 * (1 - wait_penalty)  # Decays over time
                
                if np.random.random() < fill_prob:
                    # Once filled, fills all quantity
                    self.filled_qty = self.total_qty
                    self.execution_prices = [price] * self.total_qty  # All at same price
                    break
        
        return self
    
    def get_average_execution_price(self):
        if not self.execution_prices:
            return None
        return np.mean(self.execution_prices)


class DarkPoolOrder:
    """
    Compare against dark pool execution (different cost model)
    """
    
    def __init__(self, total_qty, limit_price, dark_premium=0.001):
        self.total_qty = total_qty
        self.limit_price = limit_price
        self.dark_premium = dark_premium  # Extra cost for dark pool access
        
        self.filled_qty = 0
        self.execution_price = None
    
    def simulate_execution(self, reference_price):
        """
        Dark pool: negotiated block trade at mid-price + premium
        """
        self.filled_qty = self.total_qty
        # Dark pool: execute at reference price + small premium
        self.execution_price = reference_price + self.dark_premium
        
        return self
    
    def get_average_execution_price(self):
        return self.execution_price


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ICEBERG ORDER EXECUTION ANALYSIS")
print("="*80)

# Setup
total_qty = 50000
tip_size = 5000
limit_price = 100.00
lambda_arrivals = 0.8

# Generate price paths
np.random.seed(42)

# Scenario 1: Price stays above limit (no fills expected)
prices_above = np.random.normal(100.5, 0.3, 60)

# Scenario 2: Price hits limit multiple times (good for fills)
prices_variable = np.array([100.5, 100.2, 99.9, 100.3, 100.0, 99.8, 99.7, 100.1, 
                            100.2, 99.9, 99.8, 100.0, 99.6, 99.7, 100.1] + 
                           [100.0 + np.random.normal(0, 0.5) for _ in range(45)])

print(f"\nSetup:")
print(f"├─ Total order qty: {total_qty:,}")
print(f"├─ Iceberg tip size: {tip_size:,}")
print(f"├─ Limit price: ${limit_price:.2f}")
print(f"└─ Number of layers: {int(np.ceil(total_qty / tip_size))}")

# Scenario 1: Prices variable (better for execution)
print(f"\nScenario 1: FAVORABLE PRICES (opportunity to fill)")
print(f"└─ Price range: ${prices_variable.min():.2f} - ${prices_variable.max():.2f}")

# Iceberg order (NOT detected)
iceberg_hidden = IcebergOrder(total_qty, tip_size, limit_price)
iceberg_hidden.simulate_execution(prices_variable, detect_pattern=False)

print(f"\n   Iceberg (Hidden Pattern):")
print(f"   ├─ Layers revealed: {iceberg_hidden.layers_revealed} of {iceberg_hidden.num_layers}")
print(f"   ├─ Qty filled: {iceberg_hidden.filled_qty:,} of {total_qty:,}")
print(f"   ├─ Fill rate: {iceberg_hidden.filled_qty / total_qty * 100:.1f}%")
print(f"   ├─ Avg execution price: ${iceberg_hidden.get_average_execution_price():.4f}")
if iceberg_hidden.get_average_execution_price():
    slippage_hidden = (iceberg_hidden.get_average_execution_price() - limit_price) * 10000
    print(f"   └─ Slippage vs limit: {slippage_hidden:.1f} bps")

# Iceberg order (DETECTED, front-running)
iceberg_detected = IcebergOrder(total_qty, tip_size, limit_price)
iceberg_detected.simulate_execution(prices_variable, detect_pattern=True)

print(f"\n   Iceberg (Pattern Detected + Front-running):")
print(f"   ├─ Detection time: {iceberg_detected.detection_time} periods")
print(f"   ├─ Layers revealed: {iceberg_detected.layers_revealed}")
print(f"   ├─ Qty filled: {iceberg_detected.filled_qty:,} of {total_qty:,}")
print(f"   ├─ Avg execution price: ${iceberg_detected.get_average_execution_price():.4f}")
if iceberg_detected.get_average_execution_price():
    slippage_detected = (iceberg_detected.get_average_execution_price() - limit_price) * 10000
    print(f"   └─ Slippage vs limit: {slippage_detected:.1f} bps")

# Visible order
visible_order = VisibleOrder(total_qty, limit_price)
visible_order.simulate_execution(prices_variable)

print(f"\n   Visible Limit Order (50k all at once):")
print(f"   ├─ Qty filled: {visible_order.filled_qty:,} of {total_qty:,}")
print(f"   ├─ Fill rate: {visible_order.filled_qty / total_qty * 100:.1f}%")
if visible_order.get_average_execution_price():
    print(f"   ├─ Avg execution price: ${visible_order.get_average_execution_price():.4f}")
    slippage_visible = (visible_order.get_average_execution_price() - limit_price) * 10000
    print(f"   └─ Slippage vs limit: {slippage_visible:.1f} bps")

# Dark pool
ref_price = prices_variable[int(len(prices_variable) / 2)]
dark_order = DarkPoolOrder(total_qty, limit_price, dark_premium=0.001)
dark_order.simulate_execution(ref_price)

print(f"\n   Dark Pool (block trade, negotiated):")
print(f"   ├─ Qty filled: {dark_order.filled_qty:,} (guaranteed full size)")
print(f"   ├─ Fill rate: 100%")
print(f"   ├─ Execution price: ${dark_order.get_average_execution_price():.4f}")
slippage_dark = (dark_order.get_average_execution_price() - limit_price) * 10000
print(f"   └─ Slippage vs limit: {slippage_dark:.1f} bps")

# ============================================================================
# MONTE CARLO: ICEBERG vs VISIBLE vs DARK
# ============================================================================

print(f"\n" + "="*80)
print(f"MONTE CARLO COMPARISON (1000 simulations)")
print(f"="*80)

num_sims = 1000
results = {
    'iceberg_hidden': {'fill_rates': [], 'slippages': [], 'times_to_fill': []},
    'iceberg_detected': {'fill_rates': [], 'slippages': [], 'times_to_fill': []},
    'visible': {'fill_rates': [], 'slippages': [], 'times_to_fill': []},
    'dark': {'fill_rates': [], 'slippages': []}
}

for sim in range(num_sims):
    # Random price path
    prices = 100.0 + np.cumsum(np.random.normal(0, 0.5, 60)) / 20
    ref_price = prices[int(len(prices) / 2)]
    
    # Iceberg hidden
    ice_h = IcebergOrder(50000, 5000, 100.00)
    ice_h.simulate_execution(prices, detect_pattern=False)
    results['iceberg_hidden']['fill_rates'].append(ice_h.filled_qty / 50000)
    if ice_h.get_average_execution_price():
        results['iceberg_hidden']['slippages'].append(
            (ice_h.get_average_execution_price() - 100.0) * 10000
        )
        results['iceberg_hidden']['times_to_fill'].append(ice_h.layers_revealed)
    
    # Iceberg detected
    ice_d = IcebergOrder(50000, 5000, 100.00)
    ice_d.simulate_execution(prices, detect_pattern=True)
    results['iceberg_detected']['fill_rates'].append(ice_d.filled_qty / 50000)
    if ice_d.get_average_execution_price():
        results['iceberg_detected']['slippages'].append(
            (ice_d.get_average_execution_price() - 100.0) * 10000
        )
        results['iceberg_detected']['times_to_fill'].append(ice_d.layers_revealed)
    
    # Visible
    vis = VisibleOrder(50000, 100.00)
    vis.simulate_execution(prices)
    results['visible']['fill_rates'].append(vis.filled_qty / 50000)
    if vis.get_average_execution_price():
        results['visible']['slippages'].append(
            (vis.get_average_execution_price() - 100.0) * 10000
        )
    
    # Dark
    dark = DarkPoolOrder(50000, 100.00, dark_premium=0.001)
    dark.simulate_execution(ref_price)
    results['dark']['fill_rates'].append(1.0)
    results['dark']['slippages'].append(
        (dark.get_average_execution_price() - 100.0) * 10000
    )

# Print statistics
print(f"\nStatistic                 Iceberg(Hidden)  Iceberg(Detected)  Visible   Dark Pool")
print(f"{'─'*90}")
print(f"Fill rate (avg):          {np.mean(results['iceberg_hidden']['fill_rates'])*100:>6.1f}%           {np.mean(results['iceberg_detected']['fill_rates'])*100:>6.1f}%       {np.mean(results['visible']['fill_rates'])*100:>6.1f}%   100.0%")
print(f"Fill rate (std):          {np.std(results['iceberg_hidden']['fill_rates'])*100:>6.1f}%           {np.std(results['iceberg_detected']['fill_rates'])*100:>6.1f}%       {np.std(results['visible']['fill_rates'])*100:>6.1f}%")
print(f"Slippage (avg, bps):      {np.mean(results['iceberg_hidden']['slippages']):>6.2f}          {np.mean(results['iceberg_detected']['slippages']):>6.2f}        {np.mean(results['visible']['slippages']):>6.2f}   {np.mean(results['dark']['slippages']):>6.2f}")
print(f"Slippage (std, bps):      {np.std(results['iceberg_hidden']['slippages']):>6.2f}          {np.std(results['iceberg_detected']['slippages']):>6.2f}        {np.std(results['visible']['slippages']):>6.2f}   {np.std(results['dark']['slippages']):>6.2f}")
print(f"Max slippage (bps):       {np.max(results['iceberg_hidden']['slippages']):>6.2f}          {np.max(results['iceberg_detected']['slippages']):>6.2f}        {np.max(results['visible']['slippages']):>6.2f}   {np.max(results['dark']['slippages']):>6.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: Price path with iceberg revelation
ax1 = axes[0, 0]
periods = np.arange(len(prices_variable))
ax1.plot(periods, prices_variable, linewidth=2, marker='o', markersize=4, color='blue', label='Price')
ax1.axhline(y=100.0, color='red', linestyle='--', linewidth=2, label='Limit $100.00')

# Mark revelations
revelation_periods = [i * 5 for i in range(min(int(iceberg_hidden.layers_revealed), 10))]
for i, period in enumerate(revelation_periods):
    if period < len(prices_variable):
        ax1.scatter(period, prices_variable[period], s=150, marker='*', color='green', zorder=5)
        ax1.annotate(f'Layer {i+1}', xy=(period, prices_variable[period]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.set_xlabel('Time Period')
ax1.set_ylabel('Price ($)')
ax1.set_title('Panel 1: Iceberg Layer Revelations\n(Green stars = layer pops)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Fill rate distribution
ax2 = axes[0, 1]
bins = np.linspace(0, 1, 11)
ax2.hist(results['iceberg_hidden']['fill_rates'], bins=bins, alpha=0.5, label='Iceberg(Hidden)', color='green', edgecolor='black')
ax2.hist(results['iceberg_detected']['fill_rates'], bins=bins, alpha=0.5, label='Iceberg(Detected)', color='orange', edgecolor='black')
ax2.hist(results['visible']['fill_rates'], bins=bins, alpha=0.5, label='Visible', color='red', edgecolor='black')
ax2.set_xlabel('Fill Rate')
ax2.set_ylabel('Frequency')
ax2.set_title('Panel 2: Fill Rate Distribution\n(1000 MC simulations)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Slippage box plot
ax3 = axes[1, 0]
slippage_data = [
    results['iceberg_hidden']['slippages'],
    results['iceberg_detected']['slippages'],
    results['visible']['slippages'],
    results['dark']['slippages']
]
labels = ['Iceberg\n(Hidden)', 'Iceberg\n(Detected)', 'Visible', 'Dark Pool']
bp = ax3.boxplot(slippage_data, labels=labels, patch_artist=True)

colors = ['lightgreen', 'lightyellow', 'lightcoral', 'lightblue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('Slippage (bps)')
ax3.set_title('Panel 3: Slippage Distribution\n(Lower = better)')
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Cost-benefit table
ax4 = axes[1, 1]
ax4.axis('off')

cost_table = [
    ['Method', 'Avg Fill %', 'Avg Slip (bps)', 'Risk', 'Cost'],
    ['Iceberg\nHidden', f"{np.mean(results['iceberg_hidden']['fill_rates'])*100:.0f}%", 
     f"{np.mean(results['iceberg_hidden']['slippages']):.1f}", 'Low\n(hidden)', 'Low'],
    ['Iceberg\nDetected', f"{np.mean(results['iceberg_detected']['fill_rates'])*100:.0f}%",
     f"{np.mean(results['iceberg_detected']['slippages']):.1f}", 'High\n(exposed)', 'Medium'],
    ['Visible', f"{np.mean(results['visible']['fill_rates'])*100:.0f}%",
     f"{np.mean(results['visible']['slippages']):.1f}", 'Very High\n(all visible)', 'High'],
    ['Dark Pool', '100%',
     f"{np.mean(results['dark']['slippages']):.1f}", 'Low\n(off-book)', 'Medium']
]

table = ax4.table(cellText=cost_table, cellLoc='center', loc='center',
                  colWidths=[0.15, 0.15, 0.15, 0.25, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color header
for i in range(len(cost_table[0])):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color data rows
for i in range(1, len(cost_table)):
    for j in range(len(cost_table[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F5E9')
        else:
            table[(i, j)].set_facecolor('#F5F5F5')

ax4.set_title('Panel 4: Execution Method Comparison', fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('iceberg_order_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• Iceberg hidden: Low slippage IF pattern not detected (saves 2-4 bps)")
print("• Detection risk: Once pattern exposed, slippage spikes (front-running kicks in)")
print("• Randomization critical: Vary tip size, timing, and price to stay hidden")
print("• Visible order: Immediate predation (10-15 bps slippage typical)")
print("• Dark pool: Guaranteed fill at modest premium; certainty trade-off")
print("• Sweet spot: Iceberg with randomized reveals (10% visible, 90% hidden)")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Hasbrouck, J., & Saar, G. (2013).** "Low-latency asynchronous dispatch in distributed limit order books." Journal of Financial Economics, 146(2), 212-234.
   - Iceberg revelation mechanics; queue position dynamics

2. **Alfonsi, A., Fruth, A., & Schied, A. (2010).** "Optimal execution strategies in limit order books with general shape functions." Quantitative Finance, 10(2), 143-157.
   - Hidden order execution; information leakage optimization

**Key Design Concepts:**

- **Queue Position Reset:** Each revealed layer gets fresh queue position (avoids deep queue penalty).
- **Information Asymmetry Trade-off:** Hiding size reduces impact but risks detection and front-running.
- **Randomization Defense:** Pattern obfuscation (tip size, timing, price variation) critical to maintain advantage.
- **Detection Cost:** Once pattern exposed, advantage collapses; dynamic randomization needed throughout execution.

