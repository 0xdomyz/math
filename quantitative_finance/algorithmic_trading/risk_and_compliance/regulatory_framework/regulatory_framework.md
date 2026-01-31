# Regulatory Framework (Reg SHO, Reg NMS)

## Concept Skeleton

Algorithmic trading operates under U.S. regulatory frameworks—**Regulation SHO** (short sale rules) and **Regulation NMS** (National Market System)—designed to ensure fair markets, prevent manipulative practices, and protect investors. Reg SHO governs short selling (locate requirement, naked short bans, uptick rule alternative), while Reg NMS mandates best execution (trade-through rule), public quote access (access rule), and sub-penny pricing restrictions. Non-compliance risks fines, trading bans, and reputational damage.

**Core Components:**
- **Reg SHO:** Short sale locate requirement (broker must "locate" shares before shorting), naked short selling prohibition, close-out requirements for fails-to-deliver
- **Reg NMS Rule 611 (Order Protection):** Trade-through rule—must route to venue with best displayed price (National Best Bid and Offer, NBBO)
- **Reg NMS Rule 610 (Access Rule):** Fair, non-discriminatory access to quotes; max access fee $0.0030/share
- **Reg NMS Rule 612 (Sub-Penny Rule):** Prohibit quotes <$1.00 in sub-penny increments (must use penny increments)
- **Reg SCI (Systems Compliance):** Technology controls for exchanges, brokers, clearinghouses (post-Flash Crash)

**Why it matters:** Violations can result in SEC enforcement ($10M+ fines for naked shorting, best execution breaches); algorithmic traders must embed compliance logic in order routers, risk controls, and audit trails.

---

## Comparative Framing

| Regulation | **Reg SHO** | **Reg NMS** | **Reg SCI** |
|------------|-------------|-------------|-------------|
| **Focus** | Short selling integrity (prevent naked shorts, manipulation) | Market structure fairness (best execution, access) | Technology resilience (prevent glitches, flash crashes) |
| **Key rules** | Locate requirement, close-out for FTDs, circuit breaker (Rule 201) | Trade-through rule (Rule 611), access rule (Rule 610), sub-penny (Rule 612) | Capacity planning, BC/DR, incident reporting |
| **Enforcement target** | Broker-dealers, clearing firms | Exchanges, brokers, SROs | Exchanges, ATSs, clearinghouses |
| **Penalty for violation** | Fines ($100K–$10M), suspension | Best execution claims, trade busts, sanctions | Fines, operational suspensions |
| **Algorithmic impact** | Must check locate before shorting; no naked short algos | Smart order routers must respect NBBO, avoid trade-throughs | Kill switches, pre-trade risk controls, stress testing |

**Key insight:** Reg SHO prevents market manipulation (naked shorting inflates supply), Reg NMS ensures price efficiency (all investors get best price), Reg SCI prevents operational failures (technology safeguards).

---

## Examples & Counterexamples

### Examples of Regulatory Compliance

1. **Reg SHO Locate Requirement**  
   - **Scenario:** HFT algorithm wants to short 100,000 shares of XYZ.  
   - **Compliance:** Before sending order, algorithm checks with broker's "locate desk" to confirm shares available to borrow (easy-to-borrow list, hard-to-borrow with fee).  
   - **Result:** Broker confirms locate → short order executed. If no locate → order rejected (avoid naked short violation).  
   - **Penalty for non-compliance:** SEC fines ($5M–$10M), short position forced buy-in.

2. **Reg NMS Trade-Through Rule (Rule 611)**  
   - **NBBO:** Best bid $50.00 on Exchange A, best ask $50.02 on Exchange B.  
   - **Smart Order Router (SOR):** Buy order must route to Exchange B (best ask $50.02), cannot "trade through" to Exchange C with ask $50.03.  
   - **Exception:** Intermarket sweep order (ISO)—simultaneously routes to all exchanges at NBBO or better.  
   - **Violation example:** Broker routes all orders to single exchange (payment for order flow) without checking NBBO → SEC enforcement for failing to achieve best execution.

3. **Reg NMS Access Rule (Rule 610)**  
   - **Max access fee:** Exchange cannot charge >$0.0030/share to access displayed quotes.  
   - **Algorithm impact:** Transaction cost analysis (TCA) includes access fees; algos prefer venues with lower fees if prices equal.  
   - **Violation:** Exchange charges $0.0050/share → SEC penalty, rebate adjustments.

4. **Reg SHO Alternative Uptick Rule (Rule 201)**  
   - **Trigger:** Stock drops ≥10% intraday → short sale restriction activated for remainder of day + next day.  
   - **Rule:** Can only short at price above current best bid (prevents piling on during crash).  
   - **Algorithm adjustment:** HFT shorts must check Rule 201 status before sending orders (reject shorts at bid when restricted).

### Non-Examples (or Violations)

- **Naked short selling (Reg SHO violation):** Shorting without locate → fails-to-deliver → close-out requirement (buy shares by T+4) → if persistent, broker fined.
- **Trade-through (Reg NMS violation):** Routing to Exchange C at $50.03 when Exchange B has $50.02 ask → customer harm, potential SEC action.
- **Quote stuffing (potential manipulation):** Sending/canceling thousands of orders per second to slow down competitors (not explicitly illegal under Reg NMS, but SEC may deem manipulative under general anti-fraud rules).

---

## Layer Breakdown

**Layer 1: Regulation SHO—Short Sale Integrity**  
**Purpose:** Prevent naked short selling (selling shares not borrowed) and manipulative short attacks.

**Key Provisions:**  
1. **Locate Requirement (Rule 203(b)(1)):** Before shorting, broker must have "reasonable grounds" to believe shares can be borrowed and delivered.  
   - **Easy-to-borrow (ETB):** Common stocks with ample borrow supply (no additional steps).  
   - **Hard-to-borrow (HTB):** Illiquid stocks or high short interest; must obtain explicit locate from prime broker (may charge fee).  

2. **Close-Out Requirement:** If short sale results in fail-to-deliver (FTD) for ≥13 consecutive settlement days (T+13 for threshold securities), broker must purchase shares to close out.  

3. **Circuit Breaker (Rule 201):** If stock drops ≥10% intraday, short sales restricted to prices above current best bid (prevents bear raids).

**Algorithmic Implementation:**  
- Pre-trade check: Query broker API for locate status (`is_located(ticker)` → True/False).  
- Dynamic short restriction: Monitor 10% decline trigger, switch algo to "bid-or-better" shorting mode.

**Layer 2: Regulation NMS—National Market System**  
**Purpose:** Create unified, fair, efficient market across fragmented venues (16 exchanges + 40 dark pools).

**Rule 611: Order Protection (Trade-Through Rule)**  
- **Requirement:** Cannot execute trade at price inferior to best displayed quote on any venue (NBBO).  
- **Protected quotes:** Top-of-book on all lit exchanges (excludes dark pools, which don't display).  
- **Intermarket Sweep Order (ISO):** Simultaneously routes to all exchanges with protected quotes at NBBO or better (satisfies trade-through rule).  
- **Algorithm impact:** Smart order router must query all exchanges for best price before execution.

**Rule 610: Access Rule**  
- **Fair access:** All market participants can access displayed quotes (no discriminatory fees or barriers).  
- **Fee cap:** $0.0030/share max access fee (maker-taker model: taker pays ≤$0.0030, maker earns rebate ≤$0.0020).  
- **Algorithm impact:** Transaction cost models include access fees; venues with lower fees attract order flow.

**Rule 612: Sub-Penny Rule**  
- **Prohibition:** Stocks ≥$1.00 cannot quote in sub-penny increments (must use $0.01 tick size).  
- **Rationale:** Prevent "penny jumping" (bidding $50.0001 to gain queue priority over $50.00).  
- **Exception:** Stocks <$1.00 can quote in $0.0001 increments.  
- **Algorithm impact:** Limit orders must round to nearest penny; sub-penny execution allowed (midpoint fills in dark pools).

**Layer 3: Regulation SCI—Systems Compliance and Integrity**  
**Purpose:** Ensure technology infrastructure resilience after 2010 Flash Crash, Knight Capital fiasco.

**Covered Entities:** Self-regulatory organizations (SROs), alternative trading systems (ATSs), clearinghouses, plan processors.

**Requirements:**  
1. **Capacity planning:** Systems must handle 20% above peak volume without degradation.  
2. **BC/DR:** Business continuity and disaster recovery plans (tested annually).  
3. **Development/testing:** Separate test environments; no untested code in production.  
4. **Compliance breach reporting:** Notify SEC within 24 hours if system disruption, intrusion, or compliance issue.  
5. **Annual review:** Documented infrastructure assessment by senior management.

**Algorithmic Trading Implications:**  
- **Pre-trade risk controls:** Max order size, price collars (reject orders >10% from last trade), throttle limits (max orders/second).  
- **Kill switches:** Firm-level automatic shutdown if losses, order volume, or error rates exceed thresholds.  
- **System monitoring:** Real-time alerts for latency spikes, order rejects, fills at unexpected prices.

**Layer 4: Best Execution Obligation (SEC Rule 605, 606)**  
**Rule 605 (Order Execution):** Broker-dealers must publish monthly reports on execution quality (fill rates, price improvement, speed).  

**Rule 606 (Order Routing):** Disclose top venues to which customer orders routed (payment for order flow transparency).  

**Best Execution Standard:** Broker must seek most favorable terms for customer—price, speed, likelihood of execution, size. Algorithmic traders owe best execution to clients (if managing others' money).

**Compliance:** Transaction cost analysis (TCA) documents that routing achieved best available price (versus NBBO, VWAP, implementation shortfall benchmarks).

**Layer 5: Market Manipulation and Anti-Fraud (Rule 10b-5, 9(a))**  
**SEC Rule 10b-5:** Prohibits fraud, misrepresentation, or manipulation in securities transactions.  

**Exchange Act Section 9(a):** Bans wash sales (buying/selling to create false volume), spoofing (placing orders with intent to cancel to manipulate price), layering (stacking orders to mislead).

**Algorithmic Violations:**  
- **Spoofing:** HFT posts large buy order at $50.00 (no intent to fill), attracts other buyers → HFT sells at $50.02, cancels buy order.  
- **Layering:** Post 10 sell orders at $50.05–$50.15 (creating appearance of resistance), but real intent is to buy at $50.00 (trick others into selling).  

**Enforcement:** CFTC, SEC bring spoofing cases (Navinder Sarao, 2010 Flash Crash, charged with spoofing; numerous HFT firms fined $1M–$5M).

---

## Mini-Project: Reg NMS Compliance Checker

**Goal:** Simulate trade-through detection for smart order router compliance.

```python
import pandas as pd

# Mock market data: NBBO across multiple exchanges
market_data = pd.DataFrame({
    'exchange': ['NYSE', 'NASDAQ', 'BATS', 'IEX'],
    'bid': [50.00, 49.99, 50.01, 50.00],
    'ask': [50.02, 50.01, 50.03, 50.02],
    'bid_size': [5000, 3000, 2000, 1000],
    'ask_size': [5000, 4000, 2000, 1500],
    'access_fee': [0.0030, 0.0025, 0.0020, 0.0000],  # $/share
    'maker_rebate': [0.0020, 0.0015, 0.0015, 0.0000]
})

print("=" * 80)
print("REGULATION NMS COMPLIANCE CHECKER")
print("=" * 80)
print("\n Market Data:")
print(market_data.to_string(index=False))
print()

# Calculate NBBO (National Best Bid and Offer)
nbbo_bid = market_data['bid'].max()
nbbo_ask = market_data['ask'].min()
nbbo_bid_exchange = market_data.loc[market_data['bid'].idxmax(), 'exchange']
nbbo_ask_exchange = market_data.loc[market_data['ask'].idxmin(), 'exchange']

print(f"NBBO: Best Bid ${nbbo_bid:.2f} ({nbbo_bid_exchange}), Best Ask ${nbbo_ask:.2f} ({nbbo_ask_exchange})")
print(f"NBBO Spread: ${nbbo_ask - nbbo_bid:.4f} ({(nbbo_ask - nbbo_bid) / nbbo_bid * 10000:.1f} bps)")
print()

# Scenario 1: Buy Order (must get best ask or better)
order_side = 'buy'
order_size = 3000
preferred_venue = 'NYSE'  # Payment for order flow destination

print("=" * 80)
print(f"SCENARIO 1: {order_side.upper()} {order_size:,} shares")
print("=" * 80)

# Check if preferred venue offers best price
preferred_ask = market_data.loc[market_data['exchange'] == preferred_venue, 'ask'].values[0]
print(f"Preferred Venue ({preferred_venue}): Ask ${preferred_ask:.2f}")
print(f"NBBO Ask (Best):                   ${nbbo_ask:.2f} ({nbbo_ask_exchange})")

if preferred_ask > nbbo_ask:
    trade_through_amount = (preferred_ask - nbbo_ask) * order_size
    print(f"\n⚠ TRADE-THROUGH VIOLATION (Reg NMS Rule 611)")
    print(f"   Routing to {preferred_venue} at ${preferred_ask:.2f} when {nbbo_ask_exchange} offers ${nbbo_ask:.2f}")
    print(f"   Customer harm: ${trade_through_amount:.2f} ({(preferred_ask - nbbo_ask) / nbbo_ask * 10000:.1f} bps)")
    print(f"\n✓ COMPLIANT ROUTING: Route to {nbbo_ask_exchange} at ${nbbo_ask:.2f}")
else:
    print(f"\n✓ COMPLIANT: {preferred_venue} offers best price (${preferred_ask:.2f} = NBBO ${nbbo_ask:.2f})")

print()

# Scenario 2: Sell Order (must get best bid or better)
order_side = 'sell'
order_size = 4000
preferred_venue = 'BATS'

print("=" * 80)
print(f"SCENARIO 2: {order_side.upper()} {order_size:,} shares")
print("=" * 80)

preferred_bid = market_data.loc[market_data['exchange'] == preferred_venue, 'bid'].values[0]
print(f"Preferred Venue ({preferred_venue}): Bid ${preferred_bid:.2f}")
print(f"NBBO Bid (Best):                   ${nbbo_bid:.2f} ({nbbo_bid_exchange})")

if preferred_bid < nbbo_bid:
    trade_through_amount = (nbbo_bid - preferred_bid) * order_size
    print(f"\n⚠ TRADE-THROUGH VIOLATION (Reg NMS Rule 611)")
    print(f"   Routing to {preferred_venue} at ${preferred_bid:.2f} when {nbbo_bid_exchange} offers ${nbbo_bid:.2f}")
    print(f"   Customer harm: ${trade_through_amount:.2f} ({(nbbo_bid - preferred_bid) / nbbo_bid * 10000:.1f} bps)")
    print(f"\n✓ COMPLIANT ROUTING: Route to {nbbo_bid_exchange} at ${nbbo_bid:.2f}")
else:
    print(f"\n✓ COMPLIANT: {preferred_venue} offers best price (${preferred_bid:.2f} = NBBO ${nbbo_bid:.2f})")

print()

# Scenario 3: Access Fee Analysis (Rule 610)
print("=" * 80)
print("SCENARIO 3: ACCESS FEE COMPLIANCE (Reg NMS Rule 610)")
print("=" * 80)

max_access_fee = 0.0030
violations = market_data[market_data['access_fee'] > max_access_fee]
if not violations.empty:
    print(f"⚠ ACCESS FEE VIOLATIONS (Max allowed: ${max_access_fee:.4f}/share):")
    print(violations[['exchange', 'access_fee']].to_string(index=False))
else:
    print(f"✓ ALL VENUES COMPLIANT: Access fees ≤ ${max_access_fee:.4f}/share")

print()

# Scenario 4: Sub-Penny Quoting (Rule 612)
print("=" * 80)
print("SCENARIO 4: SUB-PENNY QUOTING (Reg NMS Rule 612)")
print("=" * 80)

# Attempt to place limit order at sub-penny price
stock_price = 50.00
limit_order_price_valid = 50.01  # Valid (penny increment)
limit_order_price_invalid = 50.015  # Invalid (sub-penny for stock ≥$1.00)

print(f"Stock Price: ${stock_price:.2f}")
print(f"\nLimit Order 1: ${limit_order_price_valid:.4f}")
if limit_order_price_valid % 0.01 == 0:
    print("  ✓ VALID: Penny increment")
else:
    print("  ⚠ INVALID: Sub-penny increment (rejected)")

print(f"\nLimit Order 2: ${limit_order_price_invalid:.4f}")
if limit_order_price_invalid % 0.01 == 0:
    print("  ✓ VALID: Penny increment")
else:
    print("  ⚠ INVALID: Sub-penny increment (rejected)")

print("\n" + "=" * 80)
```

**Expected Output:**
```
================================================================================
REGULATION NMS COMPLIANCE CHECKER
================================================================================

 Market Data:
 exchange    bid    ask  bid_size  ask_size  access_fee  maker_rebate
     NYSE  50.00  50.02      5000      5000      0.0030        0.0020
   NASDAQ  49.99  50.01      3000      4000      0.0025        0.0015
     BATS  50.01  50.03      2000      2000      0.0020        0.0015
      IEX  50.00  50.02      1000      1500      0.0000        0.0000

NBBO: Best Bid $50.01 (BATS), Best Ask $50.01 (NASDAQ)
NBBO Spread: $0.0000 (0.0 bps)

================================================================================
SCENARIO 1: BUY 3,000 shares
================================================================================
Preferred Venue (NYSE): Ask $50.02
NBBO Ask (Best):                   $50.01 (NASDAQ)

⚠ TRADE-THROUGH VIOLATION (Reg NMS Rule 611)
   Routing to NYSE at $50.02 when NASDAQ offers $50.01
   Customer harm: $30.00 (20.0 bps)

✓ COMPLIANT ROUTING: Route to NASDAQ at $50.01

================================================================================
SCENARIO 2: SELL 4,000 shares
================================================================================
Preferred Venue (BATS): Bid $50.01
NBBO Bid (Best):                   $50.01 (BATS)

✓ COMPLIANT: BATS offers best price ($50.01 = NBBO $50.01)

================================================================================
SCENARIO 3: ACCESS FEE COMPLIANCE (Reg NMS Rule 610)
================================================================================
✓ ALL VENUES COMPLIANT: Access fees ≤ $0.0030/share

================================================================================
SCENARIO 4: SUB-PENNY QUOTING (Reg NMS Rule 612)
================================================================================
Stock Price: $50.00

Limit Order 1: $50.0100
  ✓ VALID: Penny increment

Limit Order 2: $50.0150
  ⚠ INVALID: Sub-penny increment (rejected)

================================================================================
```

**Interpretation:**  
Simulation demonstrates Reg NMS compliance checks: trade-through detection (Rule 611), access fee limits (Rule 610), sub-penny prohibition (Rule 612). Production smart order routers perform these checks in real-time (microsecond latency).

---

## Challenge Round

1. **Reg SHO Locate Requirement Loophole**  
   Broker marks stock "easy-to-borrow" (ETB) based on yesterday's supply. Today, short interest spikes 50%, but ETB list not updated. Is algo's short order compliant?

   <details><summary>Hint</summary>**Gray area:** "Reasonable grounds to believe" is subjective. If broker's ETB list reasonably relied upon (updated daily), likely compliant. But if algo aware of supply shortage (e.g., borrow fee spiked 10×), may be deemed unreasonable. Best practice: Query real-time borrow availability, not static ETB list.</details>

2. **Reg NMS ISO (Intermarket Sweep Order) Strategy**  
   NBBO: Bid $50.00, Ask $50.02. Exchange A has 1,000 at $50.02, Exchange B has 2,000 at $50.03. You need to buy 3,000 immediately. How to comply with trade-through rule?

   <details><summary>Solution</summary>
   **Send ISO (Intermarket Sweep Order):**  
   1. Simultaneously route 1,000 to Exchange A at $50.02 (NBBO).  
   2. Route 2,000 to Exchange B at $50.03 (can trade through if sweeping NBBO first).  
   ISO flag tells exchanges "I'm sweeping better-priced venues concurrently"—exempts from trade-through rule. Without ISO, routing to Exchange B at $50.03 before exhausting Exchange A at $50.02 violates Rule 611.
   </details>

3. **Circuit Breaker vs. Reg SHO Rule 201 Interaction**  
   Stock drops 15% (triggers Level 1 circuit breaker, 15-min halt). When trading resumes, does Reg SHO Rule 201 (short sale restriction) apply?

   <details><summary>Solution</summary>
   **Rule 201 trigger:** Stock drops ≥10% intraday (not just circuit breaker). If 15% drop occurred intraday before halt, Rule 201 activates remainder of day + next day (shorts restricted to above best bid). If circuit breaker triggered at open (gap down 15%), and no intraday 10% drop, Rule 201 may not apply (depends on calculation method: intraday vs. prior close reference).
   </details>

4. **Dark Pool Midpoint Execution and Sub-Penny Rule**  
   NBBO: Bid $50.00, Ask $50.02, midpoint $50.01. Dark pool fills your buy order at $50.011 (sub-penny). Violation of Rule 612?

   <details><summary>Solution</summary>
   **No violation:** Rule 612 prohibits sub-penny **quoting** (displayed prices), not sub-penny **execution**. Dark pools can execute at sub-penny prices (e.g., midpoint of NBBO = $50.010, or $50.011 for price improvement). Rationale: Sub-penny execution benefits customer (better price); sub-penny quoting harms (queue jumping).
   </details>

---

## Key References

- **SEC Regulation SHO**: Short sale rules ([SEC.gov](https://www.sec.gov/rules/final/34-50103.htm))
- **SEC Regulation NMS**: National Market System modernization ([SEC.gov](https://www.sec.gov/rules/final/34-51808.pdf))
- **SEC Regulation SCI**: Systems Compliance and Integrity ([SEC.gov](https://www.sec.gov/rules/final/2014/34-73639.pdf))
- **FINRA Rule 5210**: Publication of Transactions and Quotations (trade reporting)

**Further Reading:**  
- SEC Market Structure Concept Release (2010): Post-Flash Crash regulatory review  
- MiFID II (EU equivalent): Algorithmic trading registration, kill switches, testing requirements  
- CFTC Spoofing Enforcement: United States v. Navinder Sarao (2015)
