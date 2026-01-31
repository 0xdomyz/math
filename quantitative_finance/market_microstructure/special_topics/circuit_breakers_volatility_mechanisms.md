# Circuit Breakers & Volatility Mechanisms

## 1. Concept Skeleton
**Definition:** Market-wide and single-stock circuit breakers are automatic trading halts triggered by price movements exceeding defined thresholds; designed to prevent cascading crashes and allow time for information absorption  
**Purpose:** Provide circuit-breaker cooling-off periods during extreme volatility, prevent predatory algorithms exploiting panic-driven dislocations, maintain market integrity and public confidence  
**Prerequisites:** Market structure, systemic risk, information cascades, HFT dominance, market crashes

## 2. Comparative Framing
| Mechanism | Trigger | Duration | Scope | Goal |
|-----------|---------|----------|-------|------|
| **Market-Wide Halt (Rule 80B)** | S&P 500 ±7%, ±13%, ±20% | 15 min / EOD | All trading | Systemic protection |
| **Single-Stock Halt (Rule 10b-1)** | Stock price move >10% in 5 min | 5 minutes | Individual stock | Predatory protection |
| **Volatility Auction** | IV >2σ hist | 5 minute auction | Stock-level | Price discovery |
| **Trading Halt (Disclosure)** | Company news pending | Duration varies | Stock-level | Info asymmetry |
| **Clearing Member Default** | Default events | Gradual (hours) | Clearing house | Counterparty risk |

## 3. Examples + Counterexamples

**Simple Example:**  
S&P 500 index down 7% at 10:15 AM. Rule 80B Level 1 triggered → 15-minute halt → All trading frozen → 10:30 AM, trading resumes → Market digests information, algorithms recalibrate → Recovery possible without panic cascade.

**Failure Case (May 6, 2010 Flash Crash):**  
Single-stock halts triggered cascadingly (1000s of halts) but not coordinated with index halt. Trading resumed piecemeal, causing fresh dislocations. Trade-throughs proliferated. 20,000+ trades reversed. Lesson: Need coordinated, market-wide circuit breaker (now in place).

**Edge Case (March 16, 2020 COVID Crash):**  
S&P 500 down 12% in opening minutes. Triggered Level 2 (13%) halt immediately. Unusual: Halt not at 7% (Level 1) because previous day recovery. Triggered halt at 13% just before close. Debate: Should thresholds be percentage or absolute? Current: Percentage-based.

## 4. Layer Breakdown
```
Circuit Breaker & Volatility Mechanisms Framework:
├─ Historical Evolution:
│   ├─ Pre-1987: No market-wide circuit breakers
│   │   ├─ October 19, 1987 (Black Monday): 22% down in single day
│   │   ├─ Trading never halted, cascade continued
│   │   ├─ Outcome: Panic selling, margin calls, bankruptcies
│   │   ├─ Lesson: Market-wide coordination critical
│   │   └─ Response: Brady Commission recommendations (1988)
│   ├─ 1988-2007: Gradual Implementation:
│   │   ├─ Rule 80B (1992): Market-wide halts at 7%, 13%, 20% thresholds
│   │   ├─ Thresholds: Percentage of S&P 500 closing price
│   │   ├─ Duration: 30 minutes initially, later 15 minutes
│   │   ├─ Scope: NYSE-led, but NASDAQ coordinates
│   │   ├─ Effectiveness: Few triggers (market did not crash >7% often)
│   │   └─ Complacency: Believed rules sufficient
│   ├─ 2008-2010: Post-Crisis Tightening:
│   │   ├─ Financial Crisis (2008): 20%+ declines, halts triggered
│   │   ├─ Single-stock halts: Introduced Rule 10b-1 (2009)
│   │   ├─ Coordination: All SROs must coordinate halts
│   │   ├─ May 6, 2010 (Flash Crash):
│   │   │   ├─ 7% drop in 5 minutes, triggered Rule 80B
│   │   │   ├─ But individual stocks fell 70% in seconds
│   │   │   ├─ Massive dislocations (Procter & Gamble: $60 → $42 → $60)
│   │   │   ├─ 20,000+ trades reversed afterward
│   │   │   ├─ Root cause: Algorithmic panic, liquidity vacuum
│   │   │   └─ Exposure: Circuit breakers insufficient without speed limits
│   │   └─ Response: Expanded Rule 10b-1 single-stock halts
│   ├─ 2012-Present: Continuous Evolution:
│   │   ├─ Rule 10b-1 Refined: 5-minute halts for 10%+ moves
│   │   ├─ Tick Pilot (2016-2018): Tested sub-millisecond tick widening
│   │   ├─ Volatility Auctions: NYSE/NASDAQ introduced
│   │   ├─ International Coordination: EMEA, APAC adopt similar rules
│   │   ├─ Technology: Automated halt processing (millisecond-level)
│   │   └─ Debate: Thresholds need adjustment for market changes?
│   └─ Outlook:
│       ├─ Proposals: Dynamic thresholds (adjust for volatility)
│       ├─ Challenges: Cross-asset correlation (stocks move with crypto, bonds)
│       ├─ Technology: AI-driven early warning systems
│       └─ International: Regulatory arbitrage (halt in one market, trade in another)
├─ Market-Wide Circuit Breakers (Rule 80B):
│   ├─ Trigger Mechanisms:
│   │   ├─ Level 1 (7% decline):
│   │   │   ├─ Measurement: S&P 500 index vs previous market close
│   │   │   ├─ Time Window: Intraday (any time market open)
│   │   │   ├─ Action: 15-minute trading halt
│   │   │   ├─ Exception: After 3:25 PM, halt trading for day close
│   │   │   ├─ Purpose: Cool-off, information absorption
│   │   │   └─ Data: ~20 triggers since 2008 (rare)
│   │   ├─ Level 2 (13% decline):
│   │   │   ├─ Measurement: S&P 500 down 13% from prior close
│   │   │   ├─ Action: 15-minute halt (same as Level 1)
│   │   │   ├─ Purpose: Escalated panic prevention
│   │   │   ├─ Historical: Triggered 3 times (2008, 2011, 2020)
│   │   │   └─ Pattern: Often follows Level 1 (same event)
│   │   └─ Level 3 (20% decline):
│       ├─ Measurement: S&P 500 down 20% from prior close
│       ├─ Action: Market closes for the day (no partial recovery trading)
│       ├─ Purpose: Maximum circuit-breaker enforcement
│       ├─ Psychological: 20% = bear market threshold (severe downturn)
│       ├─ Historical: Triggered once (March 16, 2020 COVID crash)
│       └─ Debate: 20% too high? Should intervene earlier?
│   ├─ Implementation Details:
│   │   ├─ Responsibility:
│   │   │   ├─ NYSE: Primary exchange monitors S&P 500 in real-time
│   │   │   ├─ Decision: NYSE Chief Regulation Officer declares halt
│   │   │   ├─ Communication: Notification via regulatory system (milliseconds)
│   │   │   ├─ Coordination: All SROs (NASDAQ, CBOE, regional) halt simultaneously
│   │   │   └─ Timing: Halt takes effect immediately (no trading after threshold)
│   │   ├─ S&P 500 Calculation:
│   │   │   ├─ Index Definition: 500 large-cap US stocks
│   │   │   ├─ Weighting: Market-cap weighted
│   │   │   ├─ Data: Real-time trade data from all venues
│   │   │   ├─ Lag: <100ms from trade to index update
│   │   │   ├─ Accuracy: Essential (market closure depends on it)
│   │   │   └─ Redundancy: Multiple feeds, audit trails
│   │   ├─ Halt Procedures:
│   │   │   ├─ Announcement: "CIRCUIT BREAKER - TRADING HALT"
│   │   │   ├─ Duration: Exactly 15 minutes (or day close if after 3:25 PM)
│   │   │   ├─ Order Handling: Orders accepted but not executed during halt
│   │   │   ├─ Queue Preservation: FIFO order book reset, new book at restart
│   │   │   ├─ Market Maker Obligations: Resume quoting at restart
│   │   │   └─ Information: News/company announcements can occur during halt
│   │   └─ Trading Resumption:
│       ├─ Process: Opening auction (similar to market open)
│       ├─ Price Discovery: Orders match at market-clearing price
│       ├─ Volatility: Often sharp movements resumption (pent-up orders)
│       ├─ Risk: Potential for fresh dislocations if information negative
│       └─ Monitoring: Enhanced surveillance during resumption
│   ├─ Effectiveness & Criticisms:
│   │   ├─ Strengths:
│   │   │   ├─ Cascade Prevention: Halts interrupt selling cascade
│   │   │   ├─ Psychological: Forces pause, discourages panic
│   │   │   ├─ Information: Companies can issue statements during halt
│   │   │   ├─ Algorithm Reset: HFT models recalibrate with new data
│   │   │   └─ Evidence: Few market crashes >20% since implementation
│   │   ├─ Weaknesses:
│   │   │   ├─ Blunt Tool: Fixed thresholds not responsive to conditions
│   │   │   ├─ May 6, 2010: Halted market-wide (S&P 500 7%), but individual stocks crashed 70%+
│   │   │   ├─ Latency: Individual stock halts take ~30 seconds to implement
│   │   │   ├─ Dislocations: In that 30 seconds, massive damage can occur
│   │   │   ├─ Manipulation: Theoretically exploitable (coordinate selling then reverse)
│   │   │   └─ Threshold Debate: 7%, 13%, 20% optimal?
│   │   ├─ Empirical Evidence:
│   │   │   ├─ Study (Lehmann & Modest, 1994): Halts may increase volatility (time-compression)
│   │   │   ├─ Counter: Prevents cascades (less net volatility)
│   │   │   ├─ International: Other countries use different thresholds (some lower)
│   │   │   ├─ Outcome: Mixed evidence on effectiveness
│   │   │   └─ General: Rules seem to have prevented worst-case crashes
│   │   └─ Proposed Improvements:
│       ├─ Dynamic Thresholds: Adjust 7%/13%/20% based on realized volatility
│       ├─ Intraday Volatility Auctions: More frequent trading halts
│       ├─ Speed Limits: Restrict HFT order flow during volatility (controversial)
│       ├─ Single-Stock Halts: Faster implementation (currently ~30 sec lag)
│       └─ Cross-Market Coordination: Crypto, forex must also participate
├─ Single-Stock Circuit Breakers (Rule 10b-1):
│   ├─ Trigger Mechanism:
│   │   ├─ Condition: Individual stock price move >10% within 5 minutes
│   │   ├─ Application: All stocks (NYSE, NASDAQ, regional exchanges)
│   │   ├─ Types of Moves:
│   │   │   ├─ Up 10%+: Halt triggered (positive scenarios too)
│   │   │   ├─ Down 10%+: Halt triggered (typical negative case)
│   │   │   └─ Net change: Absolute move magnitude |Δ|>10%
│   │   ├─ Time Window: 5-minute rolling window
│   │   ├─ Implementation: Halt immediately when threshold crossed
│   │   ├─ Duration: 5 minutes (fixed, no market-wide coordination needed)
│   │   └─ Frequency: Hundreds of halts per year (especially during volatile markets)
│   ├─ Mechanics:
│   │   ├─ Detection:
│   │   │   ├─ Venue SRO monitors all trades in real-time
│   │   │   ├─ Calculation: Reference price (typically previous close or opening price)
│   │   ├─ Decision: Automatic in most cases (no discretion)
│   │   ├─ Announcement: "TRADING HALT - VOLATILITY" broadcast
│   │   ├─ Market Participants: All venues must halt simultaneously
│   │   └─ Precision: Rule processing must be <1 second (regulatory requirement)
│   ├─ Operational Details:
│   │   ├─ Reference Price Selection:
│   │   │   ├─ During Market Open (9:30-16:00): Previous close or calculated
│   │   │   ├─ After-Hours: Previous session's close
│   │   ├─ Halt Processing:
│   │   │   ├─ All trading stops immediately
│   │   │   ├─ Orders continue to accumulate (not executed)
│   │   │   ├─ Order book queued (FIFO preserved)
│   │   │   └─ 5 minutes: Counted from halt initiation
│   │   ├─ Resumption:
│   │   │   ├─ Automatic: After exactly 5 minutes
│   │   │   ├─ Process: Auction matching (similar to market open)
│   │   │   ├─ Opening Price: Supply/demand matching (may differ significantly)
│   │   │   └─ Volatility: Often sharp move at restart (pent-up orders)
│   │   └─ Repeat Halts:
│       ├─ If price moves >10% again within same session: Halt again
│       ├─ Multiple halts: Common in highly volatile stocks (e.g., pandemic stocks)
│       ├─ Cumulative effect: Stock may be halt-locked for extended periods
│       └─ Extreme: Some stocks halted 5+ times in single day
│   ├─ Exceptions & Variations:
│   │   ├─ Options Expiration: Halts may extend during options expiry (edge cases)
│   │   ├─ Earnings Releases: Company announcements may pause halts
│   │   ├─ Pre-Market/After-Hours: Different rules (less frequent halts)
│   │   ├─ Illiquid Stocks: Halts more common (lower volume → bigger %moves)
│   │   └─ Penny Stocks: Stricter handling (regulatory concerns)
│   ├─ Effectiveness:
│   │   ├─ Goals:
│   │   │   ├─ Prevent panic in individual stocks
│   │   │   ├─ Reduce predatory algorithmic trading
│   │   ├─ Outcomes:
│   │   │   ├─ Reduces extreme moves (evidence: fewer >20% single-stock crashes)
│   │   │   ├─ Cooling-off effect: Forces review of company news
│   │   │   └─ Issue: Still doesn't prevent cascades (all stocks moving together)
│   │   └─ Criticism:
│       ├─ May trap retail investors (can't exit during halt)
│       ├─ Illusion of protection (pause doesn't change fundamentals)
│       └─ Potential manipulation (deliberate price moves then reverse)
├─ Volatility Auctions (Exchange-Specific):
│   ├─ NYSE Volatility Halt Auction (V-Halt):
│   │   ├─ Trigger: Stock price move >5% in <1 second (sub-second precision)
│   │   ├─ Duration: 5-minute auction (similar to Rule 10b-1)
│   │   ├─ Purpose: More granular than 10% threshold (catch smaller dislocations)
│   │   ├─ Mechanism: Orders accumulate during halt, matched at equilibrium price
│   │   ├─ Communication: Real-time data feed updates
│   │   └─ Coordination: Other venues must halt simultaneously (regulatory requirement)
│   ├─ NASDAQ Volatility Cross:
│   │   ├─ Similar: 5-10% thresholds, exchange-specific
│   │   ├─ Additional: Includes options cross coordination
│   │   ├─ Innovation: Batch auction process more sophisticated
│   │   └─ Participation: All participants see clearing price pre-auction
│   ├─ Other Venues:
│   │   ├─ CBOE: Similar but adapted for options (underlying stock halts trigger options halts)
│   │   ├─ Regional Exchanges: Generally follow NYSE/NASDAQ lead
│   │   └─ Purpose: Consistency across market
│   └─ Debate:
│       ├─ Pro: Earlier intervention (5% vs 10%) prevents cascades
│       ├─ Con: Too frequent halts (disrupt trading for minor moves)
│       ├─ Evidence: Mixed (some studies show increased volatility post-halt)
│       └─ Optimization: Research ongoing on optimal thresholds
├─ Information Cascades & Algorithms During Volatility:
│   ├─ Behavioral Dynamics:
│   │   ├─ Information Cascade:
│   │   │   ├─ First Shock: One large sell order (e.g., fund redemption)
│   │   │   ├─ Response: Algorithms see selling, initiate sells
│   │   │   ├─ Feedback: Price drops, triggers more selling
│   │   │   ├─ Amplification: No new information, just cascade
│   │   │   ├─ Extent: Can drop 10-20% in minutes (no fundamental change)
│   │   │   └─ Recovery: When algorithms realize buying opportunity, reversal
│   │   ├─ Liquidity Evaporation:
│   │   │   ├─ Normal: Market makers quote 2-way (buy/sell)
│   │   │   ├─ Stress: Market makers withdraw quotes (inventory risk)
│   │   │   ├─ Result: Bid-ask spreads widen (5 cents → $1+)
│   │   │   ├─ Problem: Harder to execute at any price
│   │   │   └─ Circuit breaker effect: Gives market makers time to recalibrate
│   │   ├─ Predatory Algorithm Behavior:
│   │   │   ├─ Advantage: Price falls → Algorithms spot opportunity
│   │   │   ├─ Strategy: "Rebate arbitrage" while others panicking
│   │   │   ├─ Tactics: Momentary orders to infer direction, rapid cancellation
│   │   │   ├─ Profit: Capture fractions of cents on millions of orders
│   │   │   └─ Effect: Amplifies cascades (appears to confirm selling)
│   │   └─ Circuit Breaker's Role:
│       ├─ Halt: Stops cascade temporarily
│       ├─ Information: News released, context provided
│       ├─ Reset: Market makers' algorithms repriced
│       ├─ Recovery: Often sharp (pent-up orders, bargain-hunting)
│       └─ Outcome: Usually stabilizes within 15-30 minutes post-restart
│   ├─ Model of Cascade (Theoretical):
│   │   ├─ Price Process: S(t) = S(0) + μt + σ dW(t) + cascade term
│   │   ├─ Cascade Term: -λ × N(t) where N(t) = cumulative sell orders
│   │   ├─ Feedback: λ > 0 implies selling begets more selling
│   │   ├─ Trigger: If μ < 0 and λ large → runaway selling
│   │   ├─ Circuit Breaker: λ temporarily = 0 during halt
│   │   └─ Recovery: Usually strong when λ reactivates (bargain-hunting)
│   └─ Empirical Evidence:
│       ├─ Study: Fuller et al. (2011) on May 6, 2010 flash crash
│       │   ├─ Finding: Cascade detected (order flow imbalance)
│       │   ├─ Magnitude: 70% price move in 5 minutes (no fundamentals)
│       │   ├─ Recovery: 99% reversed within hours
│       │   └─ Implication: Circuit breakers insufficient (needed speed limits too)
│       ├─ Post-2010 Analysis:
│       │   ├─ Data: Halts reduce magnitude of cascades
│       │   ├─ Conclusion: 5-15 minute halt seems effective
│       │   └─ Open Question: Optimal halt duration?
│       └─ Cross-Asset Contagion:
│           ├─ Observation: Stock cascades spread to ETFs, futures, bonds
│           ├─ Challenge: Different halt rules across assets (less coordination)
│           ├─ Debate: Should all assets halt in unison?
│           └─ Reality: Regulatory arbitrage (trade in less-halted asset)
└─ Policy & Regulation Evolution:
    ├─ Current Rules (2024):
    │   ├─ Rule 80B: Maintained as-is (7%, 13%, 20%)
    │   ├─ Rule 10b-1: Refined (faster implementation, better coordination)
    │   ├─ SCI Requirements: Enhanced system reliability mandates
    │   ├─ Surveillance: Real-time anomaly detection required
    │   └─ Testing: Annual stress tests for halt procedures
    ├─ Proposed Changes (Under Review):
    │   ├─ Dynamic Thresholds:
    │   │   ├─ Rationale: 7% threshold set in 1992 (market dynamics changed)
    │   │   ├─ Proposal: Adjust thresholds based on 20-day realized vol
    │   │   ├─ Example: High-vol period → 10% threshold; low-vol → 5%
    │   │   ├─ Pro: More responsive, earlier intervention
    │   │   ├─ Con: Complexity, potential for gaming
    │   │   └─ Status: SEC studying (no finalized proposal)
    ├─ International Coordination:
    │   ├─ EMEA (Europe): ESMA rule similar (5% single stock, 20% market)
    │   ├─ APAC (Asia): Varying standards (Tokyo, Hong Kong, Shanghai)
    │   ├─ Challenge: Overnight arbitrage (US market closed, Europe opens)
    │   └─ Future: Likely convergence toward harmonized rules
    ├─ Technology & Speed:
    │   ├─ Latency Reduction: Goal <1 second from breach to halt
    │   ├─ Current: Achieved for most halts
    │   ├─ Future: Sub-second halts (competitive advantage in arms race)
    │   └─ Risk: Faster halts = more frequent halts = trading disruptions
    └─ Debate on Effectiveness:
        ├─ Academic Split:
        │   ├─ Pro-circuit-breaker: Prevents worst-case crashes
        │   ├─ Anti-circuit-breaker: May amplify volatility (compressed time)
        │   ├─ Evidence: Mixed (depends on model assumptions)
        │   └─ Consensus: Some halts needed, but optimal design unclear
        ├─ Practical Issues:
        │   ├─ Retail trapped: Can't exit during halt (anxiety)
        │   ├─ Institutional exploits: Can position ahead of restart
        │   ├─ Algos gaming: Deliberately trigger halt, profit on restart
        │   └─ Monitoring: SEC/FINRA watch for manipulation attempts
        └─ Future Direction:
            ├─ Likely: Gradual tightening (more halts, not fewer)
            ├─ Possible: Coupling halts with forced position limits
            ├─ Controversial: Speed caps (universal maximum order rate)
            └─ Uncertain: Crypto/decentralized exchange integration
```

**Interaction:** Market stress detected → Threshold comparison → Halt decision → Communication to all venues → Trading frozen → Information absorbed → 5-15 min pause → Restart auction → Price discovery → Recovery

## 5. Mini-Project
Simulate circuit breaker triggers and cascade dynamics:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from collections import deque

class HaltLevel(Enum):
    NONE = 0
    SINGLE_STOCK = 5  # 5 minutes
    LEVEL_1 = 15      # 7% market move
    LEVEL_2 = 15      # 13% market move
    LEVEL_3 = 1440    # 20% market move (EOD)

@dataclass
class CircuitBreakerParams:
    """Parameters for circuit breaker simulation"""
    market_halt_7pct: float = 0.07
    market_halt_13pct: float = 0.13
    market_halt_20pct: float = 0.20
    single_stock_10pct: float = 0.10
    cascade_lambda: float = 0.15  # Feedback strength
    liquidity_evaporation_rate: float = 0.3  # How much liquidity disappears

class MarketSimulator:
    """Simulate circuit breaker triggers and market cascades"""
    
    def __init__(self, initial_price: float, params: CircuitBreakerParams):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.params = params
        
        self.time_steps = []
        self.prices = []
        self.halts = []
        self.cascade_intensity = []
        self.spreads = []
    
    def simulate_shock(self, magnitude: float, duration_steps: int = 100):
        """
        Simulate market shock and cascade dynamics
        
        magnitude: initial shock severity (-0.05 = -5%)
        duration_steps: how many time steps shock persists
        """
        
        halt_active = False
        halt_end_time = 0
        market_open_price = self.current_price
        cumulative_shock = 0
        
        for t in range(duration_steps):
            # Check if halt is active
            if halt_active and t >= halt_end_time:
                halt_active = False
                # Restart: pent-up demand/supply creates sharp move
                self.current_price *= 1.02  # 2% recovery bounce
            
            if not halt_active:
                # Initial shock
                if t < 10:
                    shock_term = magnitude / 10  # Distribute shock over 10 steps
                    cumulative_shock += shock_term
                else:
                    shock_term = 0
                
                # Cascade feedback: selling begets more selling
                cascade_term = -self.params.cascade_lambda * cumulative_shock
                
                # Liquidity: as panic sets in, spreads widen
                liquidity_multiplier = 1 + self.params.liquidity_evaporation_rate * abs(cumulative_shock)
                
                # Price move
                price_move = shock_term + cascade_term
                self.current_price *= (1 + price_move)
                
                # Spread widening
                bid_ask_spread = 0.01 * liquidity_multiplier
                
                # Check for halts
                market_move = (self.current_price - market_open_price) / market_open_price
                
                # Market-wide halts (Rule 80B)
                if abs(market_move) > self.params.market_halt_20pct:
                    halt_active = True
                    halt_end_time = t + int(HaltLevel.LEVEL_3.value)  # EOD
                    self.halts.append({'time': t, 'level': 'EOD (20%)', 'price': self.current_price})
                
                elif abs(market_move) > self.params.market_halt_13pct:
                    halt_active = True
                    halt_end_time = t + HaltLevel.LEVEL_2.value
                    self.halts.append({'time': t, 'level': 'Level 2 (13%)', 'price': self.current_price})
                
                elif abs(market_move) > self.params.market_halt_7pct:
                    halt_active = True
                    halt_end_time = t + HaltLevel.LEVEL_1.value
                    self.halts.append({'time': t, 'level': 'Level 1 (7%)', 'price': self.current_price})
                
                # Record
                self.time_steps.append(t)
                self.prices.append(self.current_price)
                self.cascade_intensity.append(abs(cumulative_shock))
                self.spreads.append(bid_ask_spread)
            else:
                # During halt: accumulate orders but no execution
                self.time_steps.append(t)
                self.prices.append(self.current_price)  # Price frozen
                self.cascade_intensity.append(abs(cumulative_shock))
                self.spreads.append(0.1)  # Spread huge during halt (no quotes)
    
    def analyze_cascade(self):
        """Analyze cascade severity and halt effectiveness"""
        if not self.prices:
            return None
        
        df = pd.DataFrame({
            'time': self.time_steps,
            'price': self.prices,
            'cascade_intensity': self.cascade_intensity,
            'spread_bps': [s * 10000 for s in self.spreads]
        })
        
        max_decline = (min(self.prices) - self.initial_price) / self.initial_price * 100
        cascade_severity = max(self.cascade_intensity)
        
        analysis = {
            'max_decline_pct': max_decline,
            'cascade_severity': cascade_severity,
            'num_halts': len(self.halts),
            'halts': self.halts,
            'final_price': self.current_price,
            'recovery_pct': (self.current_price - min(self.prices)) / min(self.prices) * 100
        }
        
        return df, analysis

# Run simulations
print("="*80)
print("CIRCUIT BREAKER CASCADE SIMULATOR")
print("="*80)

params = CircuitBreakerParams(
    cascade_lambda=0.15,
    liquidity_evaporation_rate=0.3
)

scenarios = [
    {'name': 'Small Shock (3%)', 'magnitude': -0.03},
    {'name': 'Medium Shock (7%)', 'magnitude': -0.07},
    {'name': 'Large Shock (15%)', 'magnitude': -0.15},
    {'name': 'Extreme Shock (25%)', 'magnitude': -0.25}
]

all_results = {}

for scenario in scenarios:
    print(f"\n{scenario['name']}:")
    print("-" * 40)
    
    sim = MarketSimulator(100.0, params)
    sim.simulate_shock(scenario['magnitude'], duration_steps=150)
    
    df, analysis = sim.analyze_cascade()
    all_results[scenario['name']] = (df, analysis)
    
    print(f"  Max decline: {analysis['max_decline_pct']:.2f}%")
    print(f"  Cascade severity: {analysis['cascade_severity']:.3f}")
    print(f"  Number of halts: {analysis['num_halts']}")
    print(f"  Final price: ${analysis['final_price']:.2f}")
    print(f"  Recovery: {analysis['recovery_pct']:.2f}% from low")
    
    if analysis['halts']:
        print(f"  Halts triggered:")
        for halt in analysis['halts']:
            print(f"    T={halt['time']}: {halt['level']} @ ${halt['price']:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, (scenario_name, (df, analysis)) in enumerate(all_results.items()):
    ax = axes[idx]
    
    # Plot price evolution
    ax.plot(df['time'], df['price'], 'b-', linewidth=2, label='Price', alpha=0.8)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.5, label='Initial')
    
    # Mark halts
    for halt in analysis['halts']:
        ax.axvline(halt['time'], color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(halt['time'], ax.get_ylim()[1] * 0.95, halt['level'], 
               rotation=90, fontsize=8, ha='right')
    
    # Fill halt regions
    for halt in analysis['halts']:
        ax.axvspan(halt['time'], halt['time'] + 15, alpha=0.2, color='red')
    
    ax.set_title(f"{scenario_name}\nDecline: {analysis['max_decline_pct']:.1f}%, Halts: {analysis['num_halts']}")
    ax.set_xlabel('Time (steps)')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([85, 105])

plt.tight_layout()
plt.show()

# Comparative analysis
print(f"\n{'='*80}")
print("CIRCUIT BREAKER EFFECTIVENESS ANALYSIS")
print(f"{'='*80}")

print(f"\nShock Magnitude vs Halt Triggers:")
print(f"{'Scenario':<25} {'Decline %':<15} {'Halts':<10} {'Recovery %':<15}")
print("-" * 65)

for scenario_name, (df, analysis) in all_results.items():
    print(f"{scenario_name:<25} {analysis['max_decline_pct']:>7.1f}% {analysis['num_halts']:>15} {analysis['recovery_pct']:>12.1f}%")

print(f"\nKey Insights:")
print(f"1. Small shocks (<3%): No halts, cascade contained by spreads widening")
print(f"2. Medium shocks (7%): Trigger Level 1 halt (Rule 80B 7%), allows recovery")
print(f"3. Large shocks (15%): Multiple halts, pent-up demand creates recovery bounce")
print(f"4. Extreme shocks (25%): Cascades hard, halts may be insufficient")
print(f"\nCircuit Breaker Tradeoff:")
print(f"- Pro: Prevents cascades (halts interrupt feedback loop)")
print(f"- Con: Pent-up orders create sharp recoveries (whipsaw risk)")
print(f"- Optimal: Balance between cascade prevention and trading disruption")
```

## 6. Challenge Round
Why trigger circuit breakers at fixed percentages (7%, 13%, 20%) rather than dynamic thresholds?
- **Historical**: Set in 1992 based on data then; market structure changed
- **Simplicity**: Fixed rules easier to communicate and implement
- **Fairness**: Same threshold for all stocks (not dependent on individual volatility)
- **Debate**: Dynamic thresholds proposed but not adopted (complexity/gaming risk)

Could circuit breakers themselves create market crashes by preventing normal price discovery?
- **Theory**: Halts compress information into restart price (sharper move)
- **Evidence**: Mixed (some studies show increased volatility, others show prevention)
- **Mechanism**: Depends on information timing (release during halt vs after)
- **Practice**: Combination of halts + mandatory disclosure seems effective

## 7. Key References
- [SEC Rule 80B: Market-Wide Circuit Breakers](https://www.sec.gov/rules/final/2012/34-66814.pdf)
- [SEC Rule 10b-1: Coordinated Volatility Halts (2009)](https://www.sec.gov/rules/final/209-5778.pdf)
- [Flash Crash Report: SEC-CFTC (May 6, 2010)](https://www.sec.gov/news/studies/2010/marketevents-report.pdf)
- [Lehmann & Modest (1994): Cascade Dynamics & Halts](https://www.jstor.org/stable/2953701)

---
**Status:** Systemic risk control mechanisms | **Complements:** Market Fragmentation, Information Cascades, Algorithmic Trading
