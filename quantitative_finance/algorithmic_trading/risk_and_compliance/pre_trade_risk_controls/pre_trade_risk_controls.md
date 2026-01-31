# Pre-Trade Risk Controls: Position Limits & Order Validation

## 1. Concept Skeleton
**Definition:** Automated checks before order submission to prevent erroneous trades, fat-fingers, limit violations, and excessive risk  
**Purpose:** Stop Knight Capital disasters before they start; enforce position/notional limits; validate order parameters; enable kill switches  
**Prerequisites:** Order management systems, risk aggregation, real-time monitoring, limit frameworks, approval workflows

## 2. Comparative Framing
| Control Type | Pre-Trade Controls | Post-Trade Controls | Intraday Monitoring | End-of-Day Reconciliation |
|--------------|-------------------|---------------------|---------------------|--------------------------|
| **Timing** | Before order submitted | After fills received | Continuous (real-time) | After market close |
| **Purpose** | Prevent errors | Detect discrepancies | Track P&L, positions | Verify accuracy |
| **Examples** | Size limits, price checks | Fill vs expected, T-cost | Real-time P&L alerts | NAV calculation |
| **Latency** | <1ms (critical path) | Seconds to minutes | Seconds | Hours |
| **Failure mode** | False reject (opportunity cost) | Losses realized | Alert fatigue | Errors discovered late |

## 3. Examples + Counterexamples

**Simple Example:**  
Order: Sell 10,000 shares XYZ @ limit $50.00; pre-trade checks: (1) Position: Long 15,000 shares ✓ (2) Price: Last trade $50.05; limit $50.00 within 5% ✓ (3) Notional: $500k < $1M limit ✓ → Order approved; submitted to market

**Failure Case:**  
Fat-finger: Trader intends 1,000 shares; types 1,000,000; pre-trade check disabled ("VIP bypass"); order submits; $50M notional (100× limit); market impact 10%; loss $5M in 30 seconds; could have been prevented by simple size check

**Edge Case:**  
Aggregation error: Firm has 5 algos trading XYZ; each submits 50k shares (within individual 100k limit); but aggregate 250k exceeds firm-wide 200k limit; pre-trade checks not aggregating across algos (siloed risk); violation undetected until end-of-day

## 4. Layer Breakdown
```
Pre-Trade Risk Controls Structure:
├─ Order Validation (Syntax & Semantics):
│   ├─ Symbol Validation:
│   │   ├─ Check: Is symbol in approved universe?
│   │   │   ├─ Whitelist: 5,000 stocks (S&P 500, Russell 2000, approved intl)
│   │   │   ├─ Reject: ABCD (unknown symbol; typo?)
│   │   │   ├─ Reject: TWTRQ (bankrupt entity; likely error vs TWTR)
│   │   │   └─ Override: Risk manager can add symbol to whitelist (rare OTC names)
│   │   ├─ Corporate actions check:
│   │   │   ├─ Stock split today? Adjust quantity/price expectations
│   │   │   ├─ Halted? Reject (cannot trade)
│   │   │   └─ Delisted? Reject (off-market)
│   │   └─ Asset class validation:
│   │       ├─ Equity order to equity router ✓
│   │       ├─ Futures order to futures router ✓
│   │       └─ Equity order to futures router ✗ (reject; misrouting)
│   ├─ Quantity Validation:
│   │   ├─ Minimum: >0 shares (cannot submit zero or negative)
│   │   ├─ Minimum economic: >$100 notional (avoid dust trades; exchange fees >profit)
│   │   ├─ Maximum per order: <1M shares (typical cap; varies by liquidity)
│   │   │   ├─ Large-cap (AAPL): 1M shares = 0.1% ADV (reasonable)
│   │   │   ├─ Small-cap (10k ADV): 1M shares = 100× ADV (clearly error; reject)
│   │   │   └─ Dynamic: Adjust limit based on ADV (e.g., max 10% ADV)
│   │   ├─ Round lots: Some exchanges prefer 100-share increments
│   │   │   ├─ Order 10,523 shares → OK (odd lot accepted)
│   │   │   ├─ But may split: 10,500 (round) + 23 (odd)
│   │   │   └─ Odd lots: Higher rejection risk; lower priority
│   │   └─ Sanity check: Compare to recent orders
│   │       ├─ Historical: Last 100 orders average 5,000 shares (std dev 2,000)
│   │       ├─ Current: 500,000 shares (250× mean; 247 std devs)
│   │       └─ Alert: "Unusually large order; confirm?" (fat-finger detection)
│   ├─ Price Validation:
│   │   ├─ Last trade reference:
│   │   │   ├─ Last: $50.05
│   │   │   ├─ Limit order: $45.00 (buy) → 10% below last; flag as aggressive
│   │   │   ├─ Limit order: $75.00 (sell) → 50% above last; likely fat-finger; REJECT
│   │   │   └─ Threshold: ±5% for standard; ±20% for volatile stocks (dynamic)
│   │   ├─ NBBO reference:
│   │   │   ├─ NBBO: Bid $50.00, Ask $50.01
│   │   │   ├─ Buy limit $55.00 → 10% above ask; paying up excessively; flag
│   │   │   ├─ Sell limit $45.00 → 10% below bid; giving away; flag
│   │   │   └─ Market orders: Check if NBBO reasonable (not stub quote $0.01)
│   │   ├─ Collar (min/max price):
│   │   │   ├─ Firm policy: No orders <$0.10 or >$10,000/share
│   │   │   ├─ Protects: Stub quotes ($0.01); erroneous ($100,000 AAPL)
│   │   │   └─ Exception: High-priced stocks (BRK.A $500k/share; allow if whitelisted)
│   │   ├─ Decimal places:
│   │   │   ├─ US stocks: Max 2 decimals ($50.12 ✓; $50.123 ✗)
│   │   │   ├─ Sub-penny rule: >$1 stocks must be penny increments
│   │   │   └─ Crypto: 8 decimals typical (BTC $50,123.45678901)
│   │   └─ Stub quote detection:
│   │       ├─ Price <50% of last trade; likely stale quote
│   │       ├─ Example: Last $50; bid shows $10 → stub quote (liquidity withdrawn)
│   │       └─ Action: Reject market order; require limit (protect from bad fill)
│   ├─ Notional Validation:
│   │   ├─ Calculate: Quantity × Price (if limit) or Quantity × Last (if market)
│   │   ├─ Firm limit: $10M per order (typical; varies by capital)
│   │   │   ├─ Order: 100k shares @ $50 = $5M ✓
│   │   │   ├─ Order: 1M shares @ $50 = $50M ✗ (exceeds limit; reject or split)
│   │   │   └─ Dynamic: Adjust limit by trader tier (junior $1M; senior $10M; PM $50M)
│   │   ├─ Concentration: Max 20% of firm capital in single name
│   │   │   ├─ Firm capital: $100M
│   │   │   ├─ Existing XYZ position: $15M
│   │   │   ├─ New order: $10M → Total $25M (25% of capital; exceeds 20%; reject)
│   │   │   └─ Alternative: Approve $5M order (brings to $20M = 20% limit)
│   │   └─ Liquidity check: Notional vs average daily notional traded
│   │       ├─ Stock: $500M ADV
│   │       ├─ Order: $50M (10% ADV; reasonable)
│   │       ├─ If order $250M (50% ADV): Flag as high-impact; require split
│   │       └─ Guideline: Single order <5% ADV (minimize market impact)
│   └─ Order Type Validation:
│       ├─ Allowed types: Market, Limit, Stop, Stop-Limit, MOC (market-on-close)
│       ├─ Disallowed: Certain exotic types if not supported by exchange
│       ├─ Stop orders: Stop price must be away from current market
│       │   ├─ Stop-loss sell: Stop <current price (e.g., $48 stop when trading $50)
│       │   ├─ Stop-loss buy: Stop >current price (e.g., $52 stop when trading $50)
│       │   └─ If backwards: Reject (e.g., stop-loss sell $52 when trading $50; triggers immediately; likely error)
│       └─ Time-in-force (TIF) validation:
│           ├─ DAY: Cancels at 4 PM ✓
│           ├─ GTC (good-till-cancel): Persists until filled or manually canceled
│           │   ├─ Risk: Forgotten GTC orders linger; may fill weeks later (stale intent)
│           │   ├─ Control: Auto-cancel GTCs >5 days old
│           │   └─ Exception: Long-term limit orders (investor buy $100 AAPL; willing to wait)
│           └─ IOC/FOK: Immediate-or-cancel / Fill-or-kill; check if exchange supports
├─ Position Limits:
│   ├─ Gross Position Limits:
│   │   ├─ Definition: Absolute value of position (ignores long/short)
│   │   │   ├─ Long 50k XYZ + Short 30k ABC → Gross = 80k
│   │   │   ├─ Firm limit: 100k shares per name; 80k ✓
│   │   │   └─ Purpose: Limit concentration risk (single name blows up)
│   │   ├─ Hierarchy:
│   │   │   ├─ Per security: Max 100k shares (e.g., XYZ ≤100k)
│   │   │   ├─ Per sector: Max 500k shares (e.g., Technology ≤500k aggregate)
│   │   │   ├─ Per asset class: Max 2M shares (e.g., US Equities ≤2M)
│   │   │   └─ Firm-wide: Max $500M gross notional (all positions)
│   │   ├─ Pre-trade check:
│   │   │   ├─ Current position: Long 90k shares XYZ
│   │   │   ├─ New order: Buy 20k shares
│   │   │   ├─ Projected: 110k shares (exceeds 100k limit; 10k overage)
│   │   │   └─ Action: Reject full order OR Approve 10k only (brings to 100k limit)
│   │   └─ Real-time aggregation:
│   │       ├─ Must aggregate across: All algos, all accounts, all venues
│   │       ├─ Example: Algo A holds 60k; Algo B holds 30k; Algo C wants 20k
│   │       │   ├─ Total: 60k + 30k + 20k = 110k (exceeds 100k)
│   │       │   └─ Reject Algo C order (or reduce to 10k)
│   │       └─ Latency: Aggregation <10ms (pre-trade critical path)
│   ├─ Net Position Limits:
│   │   ├─ Definition: Long - Short (directional exposure)
│   │   │   ├─ Long 80k XYZ, Short 30k XYZ → Net +50k (long bias)
│   │   │   ├─ Firm limit: Max ±50k shares net per name
│   │   │   └─ Purpose: Control directional risk (market moves against)
│   │   ├─ Market-neutral strategies:
│   │   │   ├─ Goal: Net ~0 (hedge out market exposure)
│   │   │   ├─ Tolerance: ±10k shares net (tight limit)
│   │   │   ├─ Example: Long 100k SPY, Short 95k SPY futures
│   │   │   │   ├─ Net: +5k SPY equivalent (5% imbalance)
│   │   │   │   └─ Within ±10k tolerance ✓
│   │   │   └─ Rebalance: If net drifts to ±9k, adjust (keep market-neutral)
│   │   └─ Directional strategies:
│   │       ├─ Allowed net exposure: ±200k shares
│   │       ├─ Example: Net +180k (bullish tilt); within limit
│   │       └─ If market drops: Net +180k shares × -3% = -5,400 shares × $50 = -$270k loss
│   ├─ Notional Limits:
│   │   ├─ Gross notional: Sum of |Long notional| + |Short notional|
│   │   │   ├─ Long $80M + Short $60M → Gross $140M
│   │   │   ├─ Firm capital: $100M → Leverage = 140M/100M = 1.4× (reasonable)
│   │   │   └─ Max leverage: 3× (gross notional ≤ $300M)
│   │   ├─ Net notional: Long - Short (directional $)
│   │   │   ├─ Long $80M - Short $60M = Net +$20M (long bias)
│   │   │   ├─ Firm limit: Net ≤ $50M (50% of capital)
│   │   │   └─ Within limit ✓
│   │   └─ Sector notional:
│   │       ├─ Technology: Max $100M gross (concentration limit)
│   │       ├─ Current: $90M
│   │       ├─ New order: $15M AAPL buy → Total $105M (exceeds; reject or reduce)
│   │       └─ Purpose: Diversify (avoid single sector wipeout)
│   ├─ Greek Limits (Options/Derivatives):
│   │   ├─ Delta: Directional exposure ($1 move in underlying)
│   │   │   ├─ Limit: Max ±$500k delta (= 10k shares of $50 stock)
│   │   │   ├─ Current: +$400k delta
│   │   │   ├─ New order: +$150k delta → Total +$550k (exceeds; reject)
│   │   │   └─ Action: Approve $100k delta order (brings to $500k limit)
│   │   ├─ Gamma: Convexity (delta changes as market moves)
│   │   │   ├─ Limit: Max 10k gamma (delta changes $10k per $1 move)
│   │   │   └─ High gamma: Risky (flash crash delta explodes; forced hedging)
│   │   ├─ Vega: Volatility exposure ($1 VIX move)
│   │   │   ├─ Limit: Max ±$1M vega
│   │   │   └─ Example: Short straddles have negative vega (lose if vol rises)
│   │   └─ Theta: Time decay ($ per day)
│   │       ├─ Typical: Theta-positive strategies (collect premium; time decay profits)
│   │       └─ Limit: Max -$50k theta (avoid excessive decay loss if holding long options)
│   └─ Limit Monitoring & Breaches:
│       ├─ Real-time: Check limits every order (<1ms latency)
│       ├─ Dashboard: Display utilization
│       │   ├─ Example: "Position Limit: 80k / 100k (80% utilized)"
│       │   ├─ Color-code: Green <70%; Yellow 70-90%; Red >90%
│       │   └─ Alert: Email risk manager if >90% (approaching limit)
│       ├─ Breach handling:
│       │   ├─ Hard limit: Order rejected; trader notified
│       │   ├─ Soft limit: Order flags; requires approval (risk manager override)
│       │   └─ Emergency: If breach occurs (e.g., market move pushed position over), force reduction
│       └─ End-of-day reconciliation:
│           ├─ Verify positions match limits (no intraday breaches lingered)
│           ├─ Investigate: If limit breached, root cause analysis
│           └─ Reporting: Daily report to CRO (Chief Risk Officer)
├─ Fat-Finger Prevention:
│   ├─ Size Anomaly Detection:
│   │   ├─ Statistical: Order size >3 std devs from trader's historical mean
│   │   │   ├─ Historical: Mean 5,000 shares; std dev 2,000
│   │   │   ├─ Threshold: 5k + 3×2k = 11k shares
│   │   │   ├─ Order: 50,000 shares (2.5× threshold) → FLAG
│   │   │   └─ Prompt: "Order size unusually large. Confirm: [Yes] [No]"
│   │   ├─ Absolute thresholds:
│   │   │   ├─ Quantity: >100k shares (regardless of history)
│   │   │   ├─ Notional: >$5M
│   │   │   └─ Orders exceeding: Require second approval (two-factor)
│   │   └─ Comparison to ADV:
│   │       ├─ Stock ADV: 500k shares
│   │       ├─ Order: 250k shares (50% ADV; extremely aggressive)
│   │       └─ Alert: "Order is 50% of daily volume; high market impact expected. Proceed?"
│   ├─ Price Anomaly Detection:
│   │   ├─ Limit price check:
│   │   │   ├─ Last trade: $50.00
│   │   │   ├─ Buy limit: $75.00 (50% above; paying way up)
│   │   │   ├─ Likely: Intended $50.00; typed $75 by mistake (fat-finger)
│   │   │   └─ Action: Reject; display "Price 50% above market; verify"
│   │   ├─ Decimal error:
│   │   │   ├─ Intended: $5.00; Typed: $50.00 (missed decimal)
│   │   │   ├─ Detection: Price 10× historical range
│   │   │   └─ Alert: "Price unusual; did you mean $5.00?"
│   │   └─ Currency confusion:
│   │       ├─ Stock trades £10 (GBP); trader types $10 (USD)
│   │       ├─ Conversion: $10 = £8 (20% error)
│   │       └─ System: Force currency selection; convert automatically
│   ├─ Duplicate Order Prevention:
│   │   ├─ Check: Identical order within 1 second
│   │   │   ├─ Order 1: Buy 10k XYZ @ $50.00 (9:30:00.000)
│   │   │   ├─ Order 2: Buy 10k XYZ @ $50.00 (9:30:00.500; 500ms later)
│   │   │   └─ Likely: Trader double-clicked "Submit"; intended 1 order
│   │   ├─ Action: Block duplicate; notify "Order already submitted"
│   │   ├─ Exception: If different accounts or intentional (trader confirms "Submit 2 orders")
│   │   └─ Timing window: 1-5 seconds (balance false positives vs duplicates)
│   ├─ Symbol Confusion:
│   │   ├─ Example: "TWTR" (Twitter) vs "TWTRQ" (bankrupt entity)
│   │   ├─ Detection: Similar tickers; one halted/delisted
│   │   └─ Alert: "Did you mean TWTR (active) instead of TWTRQ (delisted)?"
│   └─ Confirmation Workflow:
│       ├─ Order >$5M: Require explicit confirmation
│       │   ├─ Popup: "Order: 100k shares XYZ @ $50.00 = $5M. Confirm?"
│       │   ├─ Buttons: [Cancel] [Confirm]
│       │   └─ Timeout: 30 seconds; auto-cancel if no response (prevent accidental lingering)
│       ├─ Two-factor for large orders:
│       │   ├─ Trader submits order
│       │   ├─ Risk manager receives notification; must approve within 2 minutes
│       │   ├─ If approved: Order submits
│       │   └─ If denied: Order canceled; reason logged
│       └─ Audit trail:
│           ├─ Log: Order, timestamp, who submitted, who approved, reason
│           ├─ Retention: 7 years (regulatory requirement)
│           └─ Review: Monthly audit of large orders (verify controls working)
├─ Kill Switches:
│   ├─ Manual Kill Switch:
│   │   ├─ Physical button: Red button on trading desk (emergency stop)
│   │   │   ├─ Action: Cancel ALL orders; stop new submissions
│   │   │   ├─ Scope: Firm-wide or per-trader (configurable)
│   │   │   └─ Recovery: Requires risk officer to re-enable (cannot resume automatically)
│   │   ├─ Software kill switch: GUI button in OMS (Order Management System)
│   │   │   ├─ Click: Immediate halt
│   │   │   ├─ Confirmation: "Are you sure? This will cancel all orders."
│   │   │   └─ Latency: <1 second (cancel messages broadcast to all venues)
│   │   └─ Hotkey: Keyboard shortcut (e.g., Ctrl+Alt+K); faster than mouse
│   ├─ Automated Kill Switch:
│   │   ├─ Loss threshold:
│   │   │   ├─ Trigger: If P&L loss >$500k in 10 minutes
│   │   │   ├─ Action: Auto-halt; notify risk manager; page CEO
│   │   │   └─ Example: Knight Capital would have saved $430M (halted after $10M loss)
│   │   ├─ Order rate threshold:
│   │   │   ├─ Trigger: >1,000 orders/second (vs normal 10/sec)
│   │   │   ├─ Action: Auto-halt; assume algo loop error
│   │   │   └─ Example: Algo stuck in loop; submitting duplicate orders
│   │   ├─ Position threshold:
│   │   │   ├─ Trigger: Gross position >2× limit (e.g., 200k when limit 100k)
│   │   │   ├─ Action: Auto-halt; flag compliance
│   │   │   └─ Cause: Fills arrived faster than expected; exceeded limit intraday
│   │   ├─ Volatility threshold:
│   │   │   ├─ Trigger: Realized vol >3× historical (market chaos)
│   │   │   ├─ Action: Pause execution (wait for clarity)
│   │   │   └─ Resume: Manual (after risk review)
│   │   └─ Market data loss:
│   │       ├─ Trigger: No quote updates >5 seconds (feed failure)
│   │       ├─ Action: Auto-halt (trading on stale data dangerous)
│   │       └─ Failover: Switch to backup data feed; resume if feed restored
│   ├─ Partial Kill Switches:
│   │   ├─ Per-symbol: Halt XYZ only (if issue isolated)
│   │   ├─ Per-algo: Halt Algo A; allow Algo B,C to continue
│   │   ├─ Per-exchange: Halt NYSE routing; use NASDAQ (if exchange connectivity issue)
│   │   └─ Gradual: Reduce order rate to 10% normal (throttle vs full stop)
│   └─ Post-Kill Switch:
│       ├─ Forensics: Analyze what triggered (logs, order history, P&L)
│       ├─ Root cause: Was it algo bug, market event, or human error?
│       ├─ Corrective action: Fix bug, adjust limits, improve controls
│       ├─ Re-enable: Risk officer approval required (cannot auto-resume)
│       └─ Reporting: Notify regulators if material (Reg SCI event)
├─ Approval Workflows:
│   ├─ Tiered Approval:
│   │   ├─ Tier 1 (Junior trader): Max $1M per order; no approval needed
│   │   ├─ Tier 2 (Senior trader): Max $10M per order; risk manager approval if >$5M
│   │   ├─ Tier 3 (Portfolio manager): Max $50M per order; CRO approval if >$25M
│   │   └─ Tier 4 (CIO): Unlimited; but board notification if >$100M
│   │   ├─ Real-time: Approval request sent via Slack/email; 2-minute SLA
│   │   └─ Escalation: If no response in 2 min, order auto-canceled (cannot linger)
│   ├─ After-Hours Approval:
│   │   ├─ Market hours: Standard approval workflow
│   │   ├─ After-hours (4:00-8:00 PM): Elevated approval (CRO must sign off)
│   │   │   ├─ Rationale: Lower liquidity; higher risk; less oversight
│   │   │   └─ Exception: Emergency hedging (market crashes; must respond)
│   │   └─ Overnight (8 PM-9:30 AM): No trading unless CEO approves (rare)
│   ├─ New Strategy Approval:
│   │   ├─ Requirement: Before deploying new algo, must pass review:
│   │   │   ├─ Compliance: Legal review (Reg SHO, Reg NMS compliant?)
│   │   │   ├─ Risk: Stress testing (flash crash scenario; how does algo respond?)
│   │   │   ├─ Technology: Code review, unit tests, integration tests
│   │   │   └─ Operations: Runbook (how to monitor, restart, troubleshoot)
│   │   ├─ Pilot phase: Deploy at 10% scale; monitor 1 week; if OK, scale to 100%
│   │   └─ Sign-off: CRO + CTO + Compliance Officer (three-party approval)
│   └─ Limit Override:
│       ├─ Scenario: Trader needs to exceed limit (unusual opportunity)
│       ├─ Request: Submit override request with justification
│       ├─ Approval: Risk manager reviews; can approve temporary limit increase
│       │   ├─ Example: Normal limit 100k shares; approved 150k for today only
│       │   └─ Expiration: Limit reverts to 100k at 4 PM (temporary)
│       └─ Audit: All overrides logged; reviewed quarterly (are overrides abused?)
└─ Regulatory Compliance (Reg SCI, FINRA):
    ├─ Order Audit Trail (OATS/CAT):
    │   ├─ Requirement: Record every order event (submission, modification, cancellation, fill)
    │   ├─ Data: Timestamp (μs precision), symbol, quantity, price, account, trader ID
    │   ├─ Retention: 7 years (SEC requirement)
    │   └─ Consolidated Audit Trail (CAT): Industry-wide system; launched 2020
    ├─ Best Execution (Rule 606):
    │   ├─ Broker duty: Seek best execution for client orders
    │   ├─ Pre-trade: Route to venue with best NBBO; document decision
    │   ├─ Quarterly report: Disclose routing practices, payment for order flow
    │   └─ Verification: Transaction cost analysis (TCA); compare fills to benchmarks
    ├─ Market Access Rule (Rule 15c3-5):
    │   ├─ Requirement: Broker-dealers must have pre-trade risk controls
    │   ├─ Controls mandated:
    │   │   ├─ Position limits (gross, net, sector)
    │   │   ├─ Notional limits (per order, per day)
    │   │   ├─ Duplicate order prevention
    │   │   └─ Erroneous order checks (fat-finger)
    │   ├─ Testing: Annual review; ensure controls functional
    │   ├─ Documentation: Policies & procedures; updated annually
    │   └─ Penalties: $1M-$10M fines if controls inadequate (Knight Capital violated)
    ├─ FINRA Rule 3110 (Supervision):
    │   ├─ Requirement: Supervise trading activity; detect violations
    │   ├─ Surveillance: Real-time monitoring for:
    │   │   ├─ Layering/spoofing (manipulative orders)
    │   │   ├─ Wash trades (self-matching)
    │   │   ├─ Marking the close (manipulating closing price)
    │   │   └─ Front-running (trading ahead of client)
    │   ├─ Alerts: Automated flags; reviewed by compliance officer
    │   └─ Escalation: Suspicious activity reported to FINRA (SAR - Suspicious Activity Report)
    └─ Reg SCI (Systems Compliance & Integrity):
        ├─ Kill switch requirement: Must halt trading <5 seconds
        ├─ Capacity testing: Quarterly stress tests (handle peak + 20%)
        ├─ Incident reporting: Notify SEC within 1 hour of "SCI event"
        └─ Change management: Test all code changes; rollback capability
```

**Key Insight:** Pre-trade controls are first line of defense; catch 99% of errors before submission; but add 1-5ms latency (trade-off: safety vs speed)

## 5. Mini-Project
Simulate pre-trade control system with various violation scenarios:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Pre-trade control system
class PreTradeControlSystem:
    def __init__(self):
        # Position limits
        self.position_limit_per_stock = 100_000  # shares
        self.notional_limit_per_order = 5_000_000  # $5M
        self.sector_limit = {'Technology': 100_000_000, 'Finance': 80_000_000}  # $100M, $80M
        
        # Current state
        self.positions = {'AAPL': 80_000, 'MSFT': 50_000, 'GOOGL': 60_000}  # shares
        self.sector_exposure = {'Technology': 75_000_000, 'Finance': 40_000_000}  # $
        
        # Fat-finger thresholds
        self.max_quantity = 500_000  # shares per order
        self.max_price_deviation = 0.10  # 10% from last trade
        
        # Order history (for duplicate detection)
        self.recent_orders = []
        
        # Audit log
        self.audit_log = []
    
    def validate_order(self, order):
        """Run all pre-trade checks"""
        checks = {
            'symbol_check': self.check_symbol(order),
            'quantity_check': self.check_quantity(order),
            'price_check': self.check_price(order),
            'notional_check': self.check_notional(order),
            'position_limit_check': self.check_position_limit(order),
            'sector_limit_check': self.check_sector_limit(order),
            'duplicate_check': self.check_duplicate(order),
            'fat_finger_check': self.check_fat_finger(order)
        }
        
        # Aggregate results
        passed = all(checks.values())
        failed_checks = [k for k, v in checks.items() if not v]
        
        # Log
        self.audit_log.append({
            'timestamp': datetime.now(),
            'order': order,
            'passed': passed,
            'failed_checks': failed_checks
        })
        
        return passed, failed_checks
    
    def check_symbol(self, order):
        """Validate symbol exists and is tradeable"""
        approved_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        return order['symbol'] in approved_symbols
    
    def check_quantity(self, order):
        """Validate quantity within bounds"""
        qty = order['quantity']
        return 0 < qty <= self.max_quantity
    
    def check_price(self, order):
        """Validate price not too far from market"""
        if order['order_type'] == 'MARKET':
            return True  # Market orders don't have price limit
        
        last_price = order['last_trade']
        limit_price = order['limit_price']
        
        if order['side'] == 'BUY':
            # Buy limit should not be way above market (overpaying)
            max_price = last_price * (1 + self.max_price_deviation)
            return limit_price <= max_price
        else:  # SELL
            # Sell limit should not be way below market (giving away)
            min_price = last_price * (1 - self.max_price_deviation)
            return limit_price >= min_price
    
    def check_notional(self, order):
        """Validate order notional within limit"""
        if order['order_type'] == 'MARKET':
            price = order['last_trade']
        else:
            price = order['limit_price']
        
        notional = order['quantity'] * price
        return notional <= self.notional_limit_per_order
    
    def check_position_limit(self, order):
        """Validate position after order would not exceed limit"""
        symbol = order['symbol']
        current_position = self.positions.get(symbol, 0)
        
        if order['side'] == 'BUY':
            projected_position = current_position + order['quantity']
        else:
            projected_position = current_position - order['quantity']
        
        return abs(projected_position) <= self.position_limit_per_stock
    
    def check_sector_limit(self, order):
        """Validate sector exposure after order"""
        sector = order.get('sector', 'Technology')  # Default
        
        if order['order_type'] == 'MARKET':
            price = order['last_trade']
        else:
            price = order['limit_price']
        
        order_notional = order['quantity'] * price
        
        if order['side'] == 'BUY':
            projected_exposure = self.sector_exposure.get(sector, 0) + order_notional
        else:
            projected_exposure = self.sector_exposure.get(sector, 0) - order_notional
        
        sector_limit = self.sector_limit.get(sector, 1e12)  # Default unlimited
        return abs(projected_exposure) <= sector_limit
    
    def check_duplicate(self, order):
        """Check for duplicate orders in last 5 seconds"""
        now = datetime.now()
        for recent in self.recent_orders:
            time_diff = (now - recent['timestamp']).total_seconds()
            
            if time_diff < 5:  # Within 5 seconds
                # Check if same symbol, side, quantity
                if (recent['symbol'] == order['symbol'] and
                    recent['side'] == order['side'] and
                    recent['quantity'] == order['quantity']):
                    return False  # Duplicate detected
        
        # Add to recent orders
        self.recent_orders.append({
            'timestamp': now,
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity']
        })
        
        # Cleanup old orders (>1 minute)
        self.recent_orders = [o for o in self.recent_orders 
                             if (now - o['timestamp']).total_seconds() < 60]
        
        return True  # Not a duplicate
    
    def check_fat_finger(self, order):
        """Detect potential fat-finger errors"""
        # Check 1: Quantity ends in many zeros (e.g., 1,000,000 vs intended 10,000)
        qty = order['quantity']
        if qty >= 100_000 and qty % 100_000 == 0:
            # Likely fat-finger (e.g., 1,000,000; 500,000)
            return False
        
        # Check 2: Price is round number AND far from market (e.g., $100 when trading $50)
        if order['order_type'] != 'MARKET':
            limit = order['limit_price']
            last = order['last_trade']
            
            # Round number (e.g., $50.00, $100.00)
            if limit == int(limit) and abs(limit - last) / last > 0.20:
                return False  # 20% away AND round; suspicious
        
        return True
    
    def execute_order(self, order):
        """Execute order (update positions)"""
        symbol = order['symbol']
        quantity = order['quantity']
        
        if order['side'] == 'BUY':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Update sector exposure (simplified)
        sector = order.get('sector', 'Technology')
        price = order.get('limit_price', order['last_trade'])
        notional = quantity * price
        
        if order['side'] == 'BUY':
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + notional
        else:
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) - notional

# Generate test orders (mix of valid and violations)
def generate_test_orders():
    orders = [
        # Order 1: Valid order
        {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 10_000,
            'order_type': 'LIMIT',
            'limit_price': 180.0,
            'last_trade': 178.0,
            'sector': 'Technology'
        },
        # Order 2: Position limit violation
        {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 50_000,  # Current 80k + 50k = 130k (exceeds 100k limit)
            'order_type': 'LIMIT',
            'limit_price': 180.0,
            'last_trade': 178.0,
            'sector': 'Technology'
        },
        # Order 3: Fat-finger (quantity)
        {
            'symbol': 'MSFT',
            'side': 'BUY',
            'quantity': 1_000_000,  # Suspicious (many zeros)
            'order_type': 'MARKET',
            'last_trade': 350.0,
            'sector': 'Technology'
        },
        # Order 4: Price deviation violation
        {
            'symbol': 'GOOGL',
            'side': 'SELL',
            'quantity': 5_000,
            'order_type': 'LIMIT',
            'limit_price': 120.0,  # Last $150; selling $120 = 20% below (likely error)
            'last_trade': 150.0,
            'sector': 'Technology'
        },
        # Order 5: Notional limit violation
        {
            'symbol': 'AMZN',
            'side': 'BUY',
            'quantity': 40_000,
            'order_type': 'LIMIT',
            'limit_price': 180.0,  # 40k × $180 = $7.2M (exceeds $5M limit)
            'last_trade': 178.0,
            'sector': 'Technology'
        },
        # Order 6: Unknown symbol
        {
            'symbol': 'ABCD',  # Not in approved list
            'side': 'BUY',
            'quantity': 1_000,
            'order_type': 'MARKET',
            'last_trade': 50.0,
            'sector': 'Technology'
        },
        # Order 7: Valid order (different stock)
        {
            'symbol': 'TSLA',
            'side': 'SELL',
            'quantity': 5_000,
            'order_type': 'LIMIT',
            'limit_price': 240.0,
            'last_trade': 242.0,
            'sector': 'Technology'
        }
    ]
    return orders

# Run simulation
system = PreTradeControlSystem()
orders = generate_test_orders()

results = []
for i, order in enumerate(orders, 1):
    passed, failed_checks = system.validate_order(order)
    
    results.append({
        'Order #': i,
        'Symbol': order['symbol'],
        'Side': order['side'],
        'Quantity': f"{order['quantity']:,}",
        'Type': order['order_type'],
        'Price': f"${order.get('limit_price', order['last_trade']):.2f}",
        'Passed': '✓' if passed else '✗',
        'Violations': ', '.join(failed_checks) if failed_checks else 'None'
    })
    
    if passed:
        # Execute order (update positions)
        system.execute_order(order)

df_results = pd.DataFrame(results)

print("="*100)
print("Pre-Trade Control System: Order Validation Results")
print("="*100)
print(df_results.to_string(index=False))

# Statistics
total_orders = len(orders)
passed_orders = sum(1 for r in results if r['Passed'] == '✓')
rejected_orders = total_orders - passed_orders

print(f"\n{'='*100}")
print("Summary Statistics")
print(f"{'='*100}")
print(f"Total orders submitted: {total_orders}")
print(f"Orders passed: {passed_orders} ({passed_orders/total_orders*100:.0f}%)")
print(f"Orders rejected: {rejected_orders} ({rejected_orders/total_orders*100:.0f}%)")
print(f"\nRejection reasons breakdown:")
for check_type in ['position_limit_check', 'fat_finger_check', 'price_check', 
                   'notional_check', 'symbol_check']:
    count = sum(1 for r in results if check_type in r['Violations'])
    if count > 0:
        print(f"  - {check_type.replace('_', ' ').title()}: {count}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Pass vs Reject
labels = ['Passed', 'Rejected']
sizes = [passed_orders, rejected_orders]
colors = ['green', 'red']
explode = (0.1, 0)

axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
            shadow=True, startangle=90)
axes[0].set_title('Order Approval Rate')

# Chart 2: Rejection reasons
rejection_reasons = {}
for r in results:
    if r['Passed'] == '✗':
        for violation in r['Violations'].split(', '):
            rejection_reasons[violation] = rejection_reasons.get(violation, 0) + 1

if rejection_reasons:
    reasons_sorted = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)
    reasons_labels = [r[0].replace('_check', '').replace('_', ' ').title() for r in reasons_sorted]
    reasons_counts = [r[1] for r in reasons_sorted]
    
    axes[1].barh(reasons_labels, reasons_counts, color='orange')
    axes[1].set_xlabel('Count')
    axes[1].set_title('Rejection Reasons')
    axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('pre_trade_controls.png', dpi=300, bbox_inches='tight')
plt.show()

# Final positions
print(f"\n{'='*100}")
print("Updated Positions (After Approved Orders)")
print(f"{'='*100}")
for symbol, position in sorted(system.positions.items()):
    utilization = abs(position) / system.position_limit_per_stock * 100
    status = 'OK' if utilization < 90 else 'WARNING'
    print(f"{symbol}: {position:>10,} shares ({utilization:>5.1f}% of limit) [{status}]")

print(f"\nSector Exposure:")
for sector, exposure in sorted(system.sector_exposure.items()):
    limit = system.sector_limit[sector]
    utilization = abs(exposure) / limit * 100
    status = 'OK' if utilization < 90 else 'WARNING'
    print(f"{sector}: ${exposure:>12,.0f} ({utilization:>5.1f}% of ${limit:,.0f}) [{status}]")
```

## 6. Challenge Round
When pre-trade controls create unintended problems:
- **False positives**: Aggressive order (50% ADV) rejected as "fat-finger"; but trader intended (unusual opportunity); missed alpha; opportunity cost $500k (control too strict)
- **Latency cost**: Pre-trade checks add 5ms; in HFT, 5ms means stale prices; order arrives after queue filled; zero fills; control defeats purpose (speed vs safety)
- **Aggregation lag**: Position aggregated across algos every 100ms; Algo A submits order based on stale position (50ms old); Algo B already increased position; total exceeds limit (race condition)
- **Override culture**: Senior traders routinely override controls ("VIP bypass"); controls exist but unused; Knight Capital-like disaster still possible (compliance theater)
- **Regulatory burden**: Market Access Rule requires controls; implementation costs $5M + $1M/year maintenance; small firms exit market (consolidation); reduced competition
- **Kill switch hesitation**: Auto-kill at $100k loss; but normal volatility can cause $80k drawdowns; if kill triggered daily (false alarms), traders disable it (boy who cried wolf)

## 7. Key References
- [SEC Market Access Rule 15c3-5 (2010)](https://www.sec.gov/rules/final/2010/34-63241.pdf) - Pre-trade control requirements
- [FINRA Regulatory Notice 15-09](https://www.finra.org/rules-guidance/notices/15-09) - Supervision of algorithmic trading
- [FIA Principal Traders Group: Best Practices (2016)](https://www.fia.org/ptg) - Industry guidelines for pre-trade risk controls

---
**Status:** Pre-Trade | **Complements:** Order Management Systems, Risk Aggregation, Kill Switches, Reg SCI Compliance, Fat-Finger Prevention
