# Exchanges and Trading Venues

## 1. Concept Skeleton
**Definition:** Central marketplaces where buyers and sellers meet; standardized contracts; transparent order books; regulatory oversight  
**Purpose:** Price discovery via centralized matching; liquidity aggregation; transparent execution; regulatory protection  
**Prerequisites:** Order book mechanics, liquidity, market structure, regulation

## 2. Comparative Framing
| Venue | Structure | Transparency | Participants | Regulation |
|-------|-----------|--------------|--------------|-----------|
| **Exchange** | Centralized | Fully transparent | All types | Heavy (SEC) |
| **Dark Pool** | Private | Hidden orders | Institutional | Moderate (ATS) |
| **ECN/ATS** | Electronic | Partial | All types | Moderate (ATS) |
| **OTC** | Bilateral | None (private) | Sophisticated | Light (Dodd-Frank) |

## 3. Examples + Counterexamples

**Exchange Success:**  
NYSE trades Apple → 500M shares/day → tight spread $0.01 → transparent NBBO → retail can trade fairly → confidence in price discovery

**Exchange Failure:**  
Small illiquid stock → 10K shares/day on exchange → spread $0.10 → better to find counterparty OTC → centralization doesn't help if volume too small

**Venue Fragmentation Success:**  
Order flow splits across 10 venues → spreads compress $0.05 → 0.01 as competition increases → retail gets better prices → innovation drives improvement

**Venue Fragmentation Failure:**  
10 venues means checking 10 order books → HFT costs explode (connectivity, colocation) → small venues become uneconomical → consolidation inevitable

## 4. Layer Breakdown
```
Exchange Ecosystem:
├─ Market Structure:
│   ├─ Centralized Order Book:
│   │   - Single source of truth
│   │   - All orders visible (full transparency)
│   │   - Matching engine: Central computer matches
│   │   - Price discovery: Real-time visible prices
│   │   - Example: NYSE Euronext matching engine processes 10,000 trades/sec
│   ├─ Exchange Rules:
│   │   - Order types: Limit, market, stop, etc.
│   │   - Trading hours: 9:30 AM - 4:00 PM EST (stocks)
│   │   - Halts: Trading paused if volatility extreme
│   │   - Circuit breakers: Market-wide pause if S&P 500 drops 20%
│   ├─ Member Requirements:
│   │   - Registration: Must register with SEC
│   │   - Capital: Minimum net capital requirements
│   │   - Compliance: Risk management and surveillance
│   │   - Surveillance: Monitor for manipulation and abuse
│   ├─ Settlement:
│   │   - T+2: Settlement on trade date + 2 days
│   │   - DTCC clearinghouse: Central clearinghouse
│   │   - Netting: Offsetting positions
│   │   - Guarantees: Clearinghouse guarantees trades
│   └─ Governance:
│       - Exchanges operate as utilities
│       - Owned by: Members or public companies
│       - Boards: Regulate member conduct
│       - Self-regulatory: Enforce rules, not government
│
├─ Major U.S. Stock Exchanges:
│   ├─ NYSE (New York Stock Exchange):
│   │   - Largest: ~$21 trillion market cap listed
│   │   - Prestige: Blue-chip stocks
│   │   - Market maker: DMM (Designated Market Maker)
│   │   - Volume: ~3.2 billion shares/day
│   │   - Auction: Opening and closing auctions
│   │   - History: Founded 1792, oldest exchange
│   ├─ NASDAQ:
│   │   - Tech focus: Tech companies prefer NASDAQ
│   │   - Size: ~$9 trillion market cap
│   │   - Market maker: Multiple market makers per stock
│   │   - Volume: ~2.5 billion shares/day
│   │   - Fully electronic: No physical floor
│   │   - Growth: Fastest growing in late 1990s
│   ├─ CBOE (Chicago Board Options Exchange):
│   │   - Derivatives: Options trading
│   │   - Size: ~2 billion options/day
│   │   - VIX: Implied volatility index (fear gauge)
│   │   - Regulation: Options Industry Council (OIC)
│   │   - History: Founded 1973 (first options exchange)
│   ├─ CME (Chicago Mercantile Exchange):
│   │   - Futures: Futures contracts
│   │   - Commodities: Energy, metals, agriculture
│   │   - Currencies: FX futures (big volume)
│   │   - Index futures: S&P 500, NASDAQ futures
│   │   - Volume: Trillions in notional/day
│   ├─ FINRA (Financial Industry Regulatory Authority):
│   │   - OTC bonds and equities
│   │   - Over-the-counter marketplace
│   │   - FINRA Trade Reporting (FIN-trade)
│   │   - Market surveillance
│   │   - Not exchange, but primary regulator
│   ├─ Regional Exchanges (smaller):
│   │   - Bats Global Markets (now part of CBOE)
│   │   - Investors Exchange (IEX)
│   │   - NYSE American (formerly AMEX)
│   │   - EDGX, EDGA (NYSE owned)
│   │   - Trend: Consolidation reducing number of exchanges
│   └─ International Exchanges:
│       - LSE (London): $3.5 trillion
│       - Tokyo: $5 trillion
│       - Shanghai: $5.5 trillion (growing fast)
│       - HK, Mumbai, São Paulo: Growing markets
│
├─ Market Fragmentation (Post-Reg NMS):
│   ├─ Regulation NMS (National Market System):
│   │   - SEC rule: Best execution across venues
│   │   - NBBO (National Best Bid Offer): Best price
│   │   - Quote rule: Venues must display best prices
│   │   - Order protection: Orders can't trade through
│   │   - Effective: 2005 onwards
│   ├─ Effects:
│   │   - Multiple venues: 13+ stock trading venues exist
│   │   - Spread compression: Wider spreads before → narrower now
│   │   - Complexity: Brokers must check all venues
│   │   - Technology: Arms race for speed
│   │   - Cost: Higher technology costs for compliance
│   ├─ Venue Competition:
│   │   - Price competition: Lower fees to attract volume
│   │   - Rebates: Pay traders to bring orders (toxic rebates)
│   │   - Queue positioning: Faster infrastructure
│   │   - Routing: Smart order routing complexity
│   │   - Example: BATS/EDGX reduced fees, gained volume
│   ├─ Market Quality Effects:
│   │   - Bid-ask spreads: From 5-10 cents to 1 cent average
│   │   - Transaction costs: Fell ~50% post-Reg NMS
│   │   - Retail benefit: Better execution than pre-2005
│   │   - Volatility: Slightly higher intraday volatility
│   │   - Liquidity: Fragmented but deeper overall
│   └─ HFT Implications:
│       - Technology cost: HFTs invest $millions in colocation
│       - Profitability: Decimalization + fragmentation enables HFT
│       - Speed: Incentive to be faster than others
│       - Arms race: Each innovation requires response
│       - Controversy: Helps consumers (tighter spreads) but creates inequality
│
├─ Listing Standards and Delisting:
│   ├─ NYSE Listing Requirements:
│   │   - Shareholders: 2,000+ shareholders
│   │   - Shares: 1.1M publicly held shares
│   │   - Market cap: $100M+ global market cap
│   │   - Price: $4+ per share
│   │   - Profitability: Usually positive earnings
│   │   - Dividend: $1M+ annual dividend (alternative: earnings)
│   ├─ NASDAQ Listing Requirements:
│   │   - Lower bar: Easier to list than NYSE
│   │   - Shareholders: 1,250+ shareholders
│   │   - Shares: 1.1M publicly held
│   │   - Market cap: $45M minimum
│   │   - Price: $4+ per share (some exceptions)
│   │   - Profit: Not always required
│   ├─ Delisting Triggers:
│   │   - Price: Below $1/share for 30+ days
│   │   - Volume: Insufficient trading volume
│   │   - Shareholders: Falls below threshold
│   │   - Compliance: Regulatory violations
│   │   - Bankruptcy: Filing for Chapter 11
│   ├─ Delistings and Failures:
│   │   - Bankruptcy: 30-40 annual delistings
│   │   - Failures: Companies merge or acquired
│   │   - Pink Sheets: Move to OTC if delisted
│   │   - Recovery: Can relist if conditions met
│   └─ Strategic Listing Choice:
│       - Prestige: NYSE more prestigious than NASDAQ
│       - Sectors: Tech favors NASDAQ; Finance favors NYSE
│       - Costs: NYSE higher; NASDAQ lower
│       - Investor base: NYSE attracts institutions
│       - Visibility: NYSE more visible (brand value)
│
├─ Trading Rules and Halts:
│   ├─ Trading Halts:
│   │   - Sec filing halt: 20 minutes while news released
│   │   - Volatility halt: 5 minute halt if moved 10%+ (single stock)
│   │   - Circuit breaker: 15 minute halt if S&P down 7% (market wide)
│   │   - Tier 2: 15 minute halt if down 13% (market wide)
│   │   - Tier 3: 15 minute halt if down 20% (market wide, then close for day)
│   ├─ Opening Auction:
│   │   - Before 9:30 AM: Orders accumulate
│   │   - 9:30 AM opening: Single price matches all at once
│   │   - Purpose: Discover opening price, minimize gaps
│   │   - Volume: Often large execution at open
│   │   - Example: Millions of shares execute at 9:30 AM sharp
│   ├─ Closing Auction:
│   │   - 3:50 PM: Final 10 minutes trading begins
│   │   - 4:00 PM: Closing auction matches orders
│   │   - Close: Official close price is auction price
│   │   - Importance: Benchmarks, index rebalancing
│   │   - Volume: Huge volume as funds rebalance
│   └─ Order Types and Restrictions:
│       - Limit orders: Price protection
│       - Market orders: Speed vs protection tradeoff
│       - Iceberg: Hidden order size, shows in tranches
│       - Stop orders: Conditional market orders
│       - Time restrictions: Good-till-canceled, fill-or-kill, etc.
│       - Uptick rule (partial): Short-sell restrictions
│
├─ Venue Economics:
│   ├─ Revenue Models:
│   │   - Trading fees: Per-share or per-contract fees
│   │   - Listing fees: Annual fee per listed stock
│   │   - Data: Market data fees (expensive)
│   │   - Technology: Colocation and connectivity fees
│   │   - Clearing: Settlement and clearing fees
│   ├─ Revenue per Trade:
│   │   - Average: $0.0001 to $0.001 per share
│   │   - Example: 1B shares × $0.0005 = $500K revenue/day
│   │   - Volume-dependent: Lower volumes reduce revenue
│   │   - Competition: Fees declining over time
│   ├─ Market Data Revenue:
│   │   - Massive: $8+ billion/year globally
│   │   - Expensive: Data feeds cost $1,000-10,000/month
│   │   - Profitable: Highest margin business
│   │   - Proprietary: Each exchange controls own data
│   ├─ Profitability:
│   │   - ICE/CME/CBOE: $1B+ annual profits
│   │   - Competitive pressures: Margins declining
│   │   - Consolidation: Mergers create efficiency
│   │   - Global: Exchanges diversifying internationally
│   └─ Consolidation Trend:
│       - ICE acquired NYSE/EUREX
│       - CME acquired CBOT/NYMEX
│       - CBOE acquired BATS/EDGX
│       - Trend: Industry consolidating into 3-4 players
│
├─ Regulatory Framework:
│   ├─ Securities Act of 1933:
│   │   - Registration: Securities must register
│   │   - Disclosure: Issuers must disclose info
│   │   - Fraud: Anti-fraud provisions
│   │   - IPO: Rules for initial public offering
│   ├─ Securities Exchange Act of 1934:
│   │   - Exchanges: Must register with SEC
│   │   - Self-regulation: Exchanges regulate members
│   │   - Insider trading: Section 10(b), Rule 10b-5
│   │   - Manipulation: Prohibited
│   │   - Reporting: Insiders must report trades
│   ├─ Reg NMS (2005):
│   │   - Fragmentation: National market system
│   │   - Best execution: Trade-through protection
│   │   - NBBO: National best bid/offer
│   │   - Locked markets: Can't have better bid with worse ask
│   │   - Technology: Increased complexity/cost
│   ├─ Dodd-Frank (2010):
│   │   - Volcker Rule: Banks' prop trading limits
│   │   - Swap rules: Derivatives regulation
│   │   - Clearing: Central clearing of swaps
│   │   - Transparency: Trade reporting requirements
│   ├─ SEC Oversight:
│   │   - Rule 10b-5: Fraud enforcement
│   │   - Circuit breaker: Volatility limits
│   │   - Naked short selling: Fails-to-deliver limits
│   │   - Flash crashes: Increased monitoring
│   ├─ Self-Regulatory Organizations (SROs):
│   │   - FINRA: Regulates OTC markets
│   │   - Exchange surveillance: Each exchange monitors
│   │   - Member discipline: Can fine/suspend
│   │   - Appeals: Members can appeal discipline
│   └─ International Regulation:
│       - EU: MiFID, MiFID II (stricter)
│       - UK: FCA regulation
│       - Asia: Various national regulators
│       - Trend: Coordinating globally
│
├─ Market Microstructure Effects:
│   ├─ Liquidity:
│   │   - Centralized: Order book visible → easier to assess
│   │   - Aggregation: All participants in one place
│   │   - Depth: More liquidity at each price level
│   │   - Resilience: Spreads widen but recover fast
│   ├─ Price Discovery:
│   │   - Visible: Transparent prices discoverable
│   │   - Continuous: Real-time price updates
│   │   - Efficient: Information incorporated fast
│   │   - Fundamental: Prices track fundamentals well
│   ├─ Volatility:
│   │   - Liquidity provision: Easier to enter/exit
│   │   - Trade impact: Less price impact on large trades
│   │   - Calm periods: Tight spreads, low volatility
│   │   - Stress periods: Spreads widen, cascades possible
│   ├─ Market Power:
│   │   - Monopoly: If only one venue possible
│   │   - Abuse: Can charge high fees
│   │   - Regulation: Restrains abuse
│   │   - Competition: Multiple exchanges limit power
│   └─ Systemic Risk:
│       - Concentration: Risk if exchange fails
│       - Cyber: Attacks could halt trading
│       - Operational: Technology failures
│       - Contagion: Failures propagate to others
│       - Lessons: Post-2008 crisis reforms
│
├─ Trading Venue Evolution:
│   ├─ Floor Trading (pre-2000):
│   │   - Physical floor: Traders shouting bids/offers
│   │   - Specialists: One trader per stock
│   │   - Auction: Continuous auction mechanism
│   │   - Inefficiency: Humans can't process fast
│   │   - Cost: High fees, wide spreads
│   ├─ Electronic Centralization (2000-2010):
│   │   - Computers: Replace humans
│   │   - NASDAQ: Fully electronic leader
│   │   - NYSE: Transitioned from floor
│   │   - Speed: 10x faster order processing
│   │   - Efficiency: Spreads fell dramatically
│   ├─ Fragmentation (2005+):
│   │   - Reg NMS: Allow multiple venues
│   │   - HFT: Arise due to multiple venues
│   │   - Complexity: Need routing algorithms
│   │   - Competition: Fees decrease
│   │   - Specialization: Venues differentiate
│   └─ Future Trends:
│       - Blockchain: Could decentralize trading (still experimental)
│       - Global: Continued consolidation globally
│       - Automation: More algo trading
│       - Retail: Broker-less trading platforms
│       - Regulation: Likely stricter rules
│
└─ Strategic Venue Considerations:
    ├─ Listing Decisions:
    │   - Prestige: NYSE vs NASDAQ choice
    │   - Cost: Different fee structures
    │   - Investor base: NYSE attracts institutions
    │   - Visibility: NYSE brand value
    │   - Global: Dual listing opportunities
    ├─ Trading Venues:
    │   - Execution: Where to send orders
    │   - Smart routing: Check all venues for best price
    │   - Speed: Latency differences between venues
    │   - Cost: Rebates and fee structures vary
    │   - Liquidity: Volume varies by stock/venue
    ├─ Regulatory Compliance:
    │   - Venue monitoring: Each venue reports trades
    │   - Record keeping: Maintain audit trail
    │   - Best execution: Document routing decisions
    │   - Insider trading: Monitor for violations
    │   - Market manipulation: Detect and prevent
    └─ Financial Impact:
        - Listing costs: $1M+ for NYSE, $500K+ for NASDAQ
        - Annual fees: $100K - $1M+ annually
        - Trading fees: $0.0001-0.001/share impact
        - Data costs: $1,000-10,000/month
        - Total cost: Material for strategy profitability
```

**Interaction:** Trader sends order to exchange → executed against best price on book → cleared by clearinghouse → settled T+2

## 5. Mini-Project
Simulate multi-venue market fragmentation effects on spreads and execution quality:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class MultiVenueSimulator:
    def __init__(self, num_venues=3):
        self.num_venues = num_venues
        self.venues = [{'bid': 100.0, 'ask': 100.01, 'bid_vol': 100000, 'ask_vol': 100000}
                      for _ in range(num_venues)]
        self.trades = []
        self.nbbo_history = []
        
    def get_nbbo(self):
        """Calculate National Best Bid Offer across venues"""
        best_bid = max(v['bid'] for v in self.venues)
        best_ask = min(v['ask'] for v in self.venues)
        return best_bid, best_ask
    
    def process_order(self, buy_side=True):
        """Process order with best execution routing"""
        # Add random noise to each venue (different prices)
        for i, venue in enumerate(self.venues):
            noise = np.random.normal(0, 0.002)
            venue['bid'] += noise
            venue['ask'] += noise
            # Ensure bid < ask
            if venue['bid'] >= venue['ask']:
                venue['ask'] = venue['bid'] + 0.005
        
        # Update volumes (random walk)
        for venue in self.venues:
            venue['bid_vol'] += np.random.randint(-10000, 10000)
            venue['ask_vol'] += np.random.randint(-10000, 10000)
            venue['bid_vol'] = max(10000, venue['bid_vol'])
            venue['ask_vol'] = max(10000, venue['ask_vol'])
        
        best_bid, best_ask = self.get_nbbo()
        spread = best_ask - best_bid
        
        if buy_side:
            # Best ask execution
            execution_price = best_ask
            execution_venue = min(range(self.num_venues), key=lambda i: self.venues[i]['ask'])
        else:
            # Best bid execution
            execution_price = best_bid
            execution_venue = max(range(self.num_venues), key=lambda i: self.venues[i]['bid'])
        
        self.trades.append({
            'price': execution_price,
            'side': 'buy' if buy_side else 'sell',
            'spread': spread,
            'venue': execution_venue
        })
        self.nbbo_history.append((best_bid, best_ask, spread))
        
        return execution_price, spread

# Scenario 1: Single venue vs Multi-venue
print("Scenario 1: Single Venue vs Multi-Venue Competition")
print("=" * 80)

# Single venue (monopoly)
single_sim = MultiVenueSimulator(num_venues=1)
single_spreads = []
for _ in range(100):
    _, spread = single_sim.process_order()
    single_spreads.append(spread)

# Multi-venue (competition)
multi_sim = MultiVenueSimulator(num_venues=5)
multi_spreads = []
for _ in range(100):
    _, spread = multi_sim.process_order()
    multi_spreads.append(spread)

print(f"Single Venue (Monopoly):")
print(f"  Average Spread: ${np.mean(single_spreads):.4f}")
print(f"  Median Spread:  ${np.median(single_spreads):.4f}")
print(f"  Std Dev:        ${np.std(single_spreads):.4f}")
print(f"  Min:            ${np.min(single_spreads):.4f}")
print(f"  Max:            ${np.max(single_spreads):.4f}")

print(f"\nMulti-Venue (5 Venues):")
print(f"  Average Spread: ${np.mean(multi_spreads):.4f}")
print(f"  Median Spread:  ${np.median(multi_spreads):.4f}")
print(f"  Std Dev:        ${np.std(multi_spreads):.4f}")
print(f"  Min:            ${np.min(multi_spreads):.4f}")
print(f"  Max:            ${np.max(multi_spreads):.4f}")

print(f"\nSpread Compression:")
print(f"  Reduction: {(1 - np.mean(multi_spreads)/np.mean(single_spreads))*100:.1f}%")
print(f"  $ Savings per trade (100 shares): ${(np.mean(single_spreads) - np.mean(multi_spreads))*100:,.2f}")

# Scenario 2: Venue fragmentation effects
print(f"\n\nScenario 2: Varying Degree of Fragmentation")
print("=" * 80)

fragmentation_levels = [1, 2, 3, 5, 10, 20]
avg_spreads_by_fragmentation = []
spread_variance_by_fragmentation = []

for num_venues in fragmentation_levels:
    sim = MultiVenueSimulator(num_venues=num_venues)
    spreads = []
    
    for _ in range(50):
        _, spread = sim.process_order()
        spreads.append(spread)
    
    avg_spreads_by_fragmentation.append(np.mean(spreads))
    spread_variance_by_fragmentation.append(np.std(spreads))
    
    print(f"{num_venues:>2} Venues: Avg Spread ${np.mean(spreads):.4f}, Std Dev ${np.std(spreads):.4f}")

# Scenario 3: Venue migration (all orders go to best venue)
print(f"\n\nScenario 3: Smart Order Routing (All to Best Venue)")
print("=" * 80)

# Fixed cost for switching venues
switching_cost = 0.0005

multi_sim = MultiVenueSimulator(num_venues=5)
smart_routed_prices = []
naive_prices = []

for _ in range(100):
    # All go to best venue (smart routing)
    for i, venue in enumerate(multi_sim.venues):
        noise = np.random.normal(0, 0.002)
        venue['bid'] += noise
        venue['ask'] += noise
        if venue['bid'] >= venue['ask']:
            venue['ask'] = venue['bid'] + 0.005
    
    best_bid, best_ask = multi_sim.get_nbbo()
    spread = best_ask - best_bid
    
    best_ask_venue = min(range(5), key=lambda i: multi_sim.venues[i]['ask'])
    smart_routed = multi_sim.venues[best_ask_venue]['ask']
    
    # Naive: just pick first venue
    naive = multi_sim.venues[0]['ask']
    
    smart_routed_prices.append(smart_routed + switching_cost)  # Add switching cost
    naive_prices.append(naive)

print(f"Smart Routing (Route to Best Ask):")
print(f"  Average Price: ${np.mean(smart_routed_prices):.4f}")
print(f"  Total Cost: ${np.sum(smart_routed_prices):.2f}")

print(f"\nNaive Routing (First Venue):")
print(f"  Average Price: ${np.mean(naive_prices):.4f}")
print(f"  Total Cost: ${np.sum(naive_prices):.2f}")

print(f"\nSmart Routing Savings:")
print(f"  Per Trade: ${np.mean(naive_prices) - np.mean(smart_routed_prices):.4f}")
print(f"  Total (100 trades): ${np.sum(naive_prices) - np.sum(smart_routed_prices):.2f}")

# Scenario 4: Order flow distribution by venue
print(f"\n\nScenario 4: Market Share by Venue (100 Random Executions)")
print("=" * 80)

multi_sim = MultiVenueSimulator(num_venues=3)
venue_volume = [0, 0, 0]

for _ in range(100):
    _, spread = multi_sim.process_order()
    venue = multi_sim.trades[-1]['venue']
    venue_volume[venue] += 1

for i, vol in enumerate(venue_volume):
    print(f"Venue {i}: {vol:>3} shares ({vol}%)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Spread distribution (single vs multi)
axes[0, 0].hist(single_spreads, bins=20, alpha=0.5, label='Single Venue', color='red')
axes[0, 0].hist(multi_spreads, bins=20, alpha=0.5, label='5 Venues', color='green')
axes[0, 0].set_xlabel('Spread ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Scenario 1: Spread Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Fragmentation effect on spreads
axes[0, 1].plot(fragmentation_levels, avg_spreads_by_fragmentation, 'o-', linewidth=2, markersize=8)
axes[0, 1].fill_between(fragmentation_levels, 
                        np.array(avg_spreads_by_fragmentation) - np.array(spread_variance_by_fragmentation),
                        np.array(avg_spreads_by_fragmentation) + np.array(spread_variance_by_fragmentation),
                        alpha=0.3)
axes[0, 1].set_xlabel('Number of Venues')
axes[0, 1].set_ylabel('Average Spread ($)')
axes[0, 1].set_title('Scenario 2: Fragmentation Effect')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Smart routing savings
routing_types = ['Smart\nRouting', 'Naive\nRouting']
total_costs = [np.sum(smart_routed_prices), np.sum(naive_prices)]
colors_routing = ['green', 'red']

bars = axes[1, 0].bar(routing_types, total_costs, color=colors_routing, alpha=0.7)
axes[1, 0].set_ylabel('Total Execution Cost ($)')
axes[1, 0].set_title('Scenario 3: Smart Routing Benefit (100 trades)')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, cost in zip(bars, total_costs):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Market share distribution
colors_venues = ['#1f77b4', '#ff7f0e', '#2ca02c']
wedges, texts, autotexts = axes[1, 1].pie(venue_volume, labels=[f'Venue {i}' for i in range(3)],
                                           autopct='%1.1f%%', colors=colors_venues, startangle=90)
axes[1, 1].set_title('Scenario 4: Market Share Distribution')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Spread Reduction (5 venues): {(1 - np.mean(multi_spreads)/np.mean(single_spreads))*100:.1f}%")
print(f"Optimal fragmentation: 5-10 venues (law of diminishing returns)")
print(f"Smart routing value: ${(np.sum(naive_prices) - np.sum(smart_routed_prices)):.2f}/100 trades")
print(f"Switching cost impact: Can negate benefits if too high")
```

## 6. Challenge Round
Why did regulators introduce Reg NMS fragmentation when it increased technology costs and complexity for market participants?

- **Retail benefit**: Spread compression from 5-10 cents to 1 cent on average → retail investors got far better execution → democratic value despite infrastructure burden
- **Innovation incentive**: Fragmentation forced competition → exchanges compete on speed/fees → innovation in infrastructure → benefits eventually pass to all
- **Predatory prevention**: Locked markets (better bid with worse ask) prevented manipulation → Trade-through rule stopped predatory pricing → fairness improved
- **Unforeseen costs**: Reg NMS didn't anticipate HFT arms race (colocation, latency races) → technology costs exploded → technology industry gained, others lost
- **Current debate**: Is 1-cent spread worth $billions in HFT infrastructure? Some say yes (retail), some say no (it's just extraction) → trade-off unresolved

## 7. Key References
- [Reg NMS Overview (SEC)](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000100640&type=&dateb=&owner=exclude&count=40)
- [Bessembinder (2003) - Trade Execution Quality on Centralized Exchanges](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=365941)
- [Hasbrouck & Saar (2013) - Low-latency Trading](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695460)
- [Harris (2013) - Roasting Broken Eggs: Competing Venues After Reg NMS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2260841)

---
**Status:** Central venues with regulatory oversight | **Complements:** Fragmentation, Price Discovery, Regulation, Market Quality
