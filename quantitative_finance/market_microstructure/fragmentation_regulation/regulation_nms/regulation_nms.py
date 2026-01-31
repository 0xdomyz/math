import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

class ExecutionVenue(Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CBOE = "CBOE"
    DARKPOOL = "Dark Pool"

@dataclass
class Quote:
    venue: ExecutionVenue
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: float

@dataclass
class RoutingParams:
    """Parameters for order routing under Reg NMS"""
    maker_taker_rebate: float = 0.0001  # $0.0001 per share rebate
    access_fee: float = 0.00003         # Access fee to other venues
    maker_fee: float = 0.0003           # Posting fee
    taker_fee: float = 0.0005           # Taking fee
    compliance_window_ms: float = 500.0 # Reg NMS compliance window

class RegulationNMSRouter:
    """Order routing system compliant with Regulation NMS"""
    
    def __init__(self, params: RoutingParams):
        self.params = params
        self.routes = []
        self.trades = []
    
    def compute_nbbo(self, quotes: list):
        """
        Compute National Best Bid & Offer (NBBO)
        
        Protected quotations: must be firm, genuine, accessible
        """
        # Filter valid quotes (>= 100 shares minimum)
        valid_quotes = [q for q in quotes if q.bid_size >= 100 and q.ask_size >= 100]
        
        if not valid_quotes:
            return None, None
        
        # Best bid: highest bid among valid quotes
        best_bid_q = max(valid_quotes, key=lambda q: q.bid)
        best_bid = best_bid_q.bid
        best_bid_venue = best_bid_q.venue
        
        # Best ask: lowest ask among valid quotes
        best_ask_q = min(valid_quotes, key=lambda q: q.ask)
        best_ask = best_ask_q.ask
        best_ask_venue = best_ask_q.venue
        
        return (best_bid, best_bid_venue), (best_ask, best_ask_venue)
    
    def check_trade_through(self, execution_price, nbbo_bid, nbbo_ask, side):
        """
        Rule 611: Trade-Through Prohibition
        
        Cannot execute at worse price if better price available
        """
        if side == 'buy':
            # Buying: ask side. Cannot pay more than best ask
            is_trade_through = execution_price > nbbo_ask + 0.0001  # 1 bp tolerance
        else:  # sell
            # Selling: bid side. Cannot receive less than best bid
            is_trade_through = execution_price < nbbo_bid - 0.0001
        
        return is_trade_through
    
    def route_order(self, order_side, order_size, order_type, quotes, market_time):
        """
        Route order under Regulation NMS compliance
        
        order_side: 'buy' or 'sell'
        order_size: number of shares
        order_type: 'market' or 'limit'
        quotes: list of current quotes from all venues
        """
        nbbo_bid_info, nbbo_ask_info = self.compute_nbbo(quotes)
        
        if nbbo_bid_info is None:
            return None  # No valid quotes
        
        best_bid, best_bid_venue = nbbo_bid_info
        best_ask, best_ask_venue = nbbo_ask_info
        spread = best_ask - best_bid
        
        routing_decision = {
            'timestamp': market_time,
            'side': order_side,
            'size': order_size,
            'nbbo_bid': best_bid,
            'nbbo_ask': best_ask,
            'spread_bps': spread * 10000,
            'routes': []
        }
        
        if order_type == 'market':
            if order_side == 'buy':
                # Buy market: must execute at or better than best ask (Rule 611)
                venue = best_ask_venue
                price = best_ask
                execution_desc = f"Routed to {venue.value} at best ask ${price:.4f}"
                
                routing_decision['routes'].append({
                    'venue': venue.value,
                    'price': price,
                    'size': min(order_size, 1000),  # Initial fill
                    'rationale': 'Best ask, Rule 611 compliant'
                })
            
            else:  # sell market
                # Sell market: must execute at or better than best bid (Rule 611)
                venue = best_bid_venue
                price = best_bid
                execution_desc = f"Routed to {venue.value} at best bid ${price:.4f}"
                
                routing_decision['routes'].append({
                    'venue': venue.value,
                    'price': price,
                    'size': min(order_size, 1000),
                    'rationale': 'Best bid, Rule 611 compliant'
                })
        
        elif order_type == 'limit':
            # Limit order: can route to best venue or post
            if order_side == 'buy':
                limit_price = best_ask - 0.01  # Post 1 cent below ask
                
                # Check if price is reasonable (not trade-through)
                if limit_price < best_bid - 0.01:
                    # Post on NASDAQ
                    routing_decision['routes'].append({
                        'venue': 'NASDAQ',
                        'price': limit_price,
                        'size': order_size,
                        'rationale': 'Limit price, no trade-through risk'
                    })
                else:
                    # Route to best ask
                    routing_decision['routes'].append({
                        'venue': best_ask_venue.value,
                        'price': best_ask,
                        'size': min(order_size, 1000),
                        'rationale': 'Immediate execution at best ask'
                    })
        
        self.routes.append(routing_decision)
        return routing_decision
    
    def simulate_trading_day(self, n_orders=500, venues=None):
        """Simulate a trading day with order routing"""
        
        if venues is None:
            venues = [ExecutionVenue.NYSE, ExecutionVenue.NASDAQ, 
                     ExecutionVenue.CBOE, ExecutionVenue.DARKPOOL]
        
        # Simulate quotes evolution
        base_price = 100.0
        
        for t in range(n_orders):
            # Generate quote updates
            quotes = []
            
            for venue in venues:
                # Venues quote around mid
                mid = base_price + np.random.normal(0, 0.02)
                
                bid = mid - np.random.uniform(0.005, 0.015)
                ask = mid + np.random.uniform(0.005, 0.015)
                
                # Dark pool: wider spread, less size
                if venue == ExecutionVenue.DARKPOOL:
                    bid_size = np.random.randint(500, 2000)
                    ask_size = np.random.randint(500, 2000)
                else:
                    bid_size = np.random.randint(1000, 5000)
                    ask_size = np.random.randint(1000, 5000)
                
                quotes.append(Quote(
                    venue=venue,
                    bid=bid,
                    ask=ask,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    timestamp=t
                ))
            
            # Generate order
            order_side = np.random.choice(['buy', 'sell'])
            order_size = np.random.randint(100, 5000)
            order_type = np.random.choice(['market', 'limit'], p=[0.6, 0.4])
            
            # Route order
            route_result = self.route_order(order_side, order_size, order_type, quotes, t)
            
            # Update base price
            base_price = (quotes[0].bid + quotes[0].ask) / 2
        
        return pd.DataFrame(self.routes)

# Run simulation
print("="*80)
print("REGULATION NMS ORDER ROUTING SIMULATOR")
print("="*80)

params = RoutingParams(
    maker_taker_rebate=0.0001,
    access_fee=0.00003,
    maker_fee=0.0003,
    taker_fee=0.0005,
    compliance_window_ms=500.0
)

router = RegulationNMSRouter(params)

print("\nSimulating trading day with Reg NMS routing...")
routes_df = router.simulate_trading_day(n_orders=500)

print(f"\nRouting Summary (first 10 orders):")
print(routes_df.head(10).to_string())

print(f"\nStatistics:")
print(f"  Total orders routed: {len(routes_df)}")
print(f"  Average spread: {routes_df['spread_bps'].mean():.2f} bps")
print(f"  Spread std dev: {routes_df['spread_bps'].std():.2f} bps")

# Venue distribution
venue_counts = {}
for routes in routes_df['routes']:
    if routes:
        venue = routes[0]['venue']
        venue_counts[venue] = venue_counts.get(venue, 0) + 1

print(f"\nVenue Distribution:")
for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(routes_df) * 100
    print(f"  {venue}: {count} orders ({pct:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Spread over time
axes[0, 0].plot(routes_df['spread_bps'], linewidth=1, alpha=0.7)
axes[0, 0].axhline(routes_df['spread_bps'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].set_title('Bid-Ask Spread Over Time')
axes[0, 0].set_xlabel('Order #')
axes[0, 0].set_ylabel('Spread (bps)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Spread distribution
axes[0, 1].hist(routes_df['spread_bps'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribution of Bid-Ask Spreads')
axes[0, 1].set_xlabel('Spread (bps)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Buy vs Sell orders
buy_sell_counts = routes_df['side'].value_counts()
axes[1, 0].bar(buy_sell_counts.index, buy_sell_counts.values, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Buy vs Sell Orders')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Venue distribution pie
if venue_counts:
    venues = list(venue_counts.keys())
    counts = list(venue_counts.values())
    axes[1, 1].pie(counts, labels=venues, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Order Routing by Venue')

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Reg NMS (Rule 611) ensures NBBO protection: no trade-throughs allowed")
print(f"2. Order routing must check best price across all venues")
print(f"3. Venue fragmentation enabled by Rule 610 (access mandate)")
print(f"4. Maker-taker rebates influence routing decisions (conflict)")
print(f"5. Sub-penny rule maintains minimum 1-cent tick size")
print(f"6. Compliance window (500ms) governs routing latency requirements")
