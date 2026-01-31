import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import heapq

@dataclass
class Venue:
    """Trading venue characteristics"""
    name: str
    latency_ms: float  # Round-trip latency in milliseconds
    maker_fee: float  # Negative = rebate
    taker_fee: float  # Per-share fee
    best_bid: float
    best_ask: float
    bid_size: int
    ask_size: int
    hft_presence: float  # 0-1: Probability of latency arbitrage
    
    def get_spread(self):
        return self.best_ask - self.best_bid
    
    def get_midpoint(self):
        return (self.best_bid + self.best_ask) / 2

class SmartOrderRouter:
    """Simulate smart order routing logic"""
    def __init__(self, venues: List[Venue], dark_pools: List[Venue] = None):
        self.venues = venues
        self.dark_pools = dark_pools or []
        self.execution_log = []
        
    def get_nbbo(self):
        """Calculate National Best Bid and Offer"""
        best_bid = max(v.best_bid for v in self.venues)
        best_ask = min(v.best_ask for v in self.venues)
        return best_bid, best_ask
    
    def route_market_buy(self, size: int, strategy='price_time') -> List[Tuple]:
        """
        Route market buy order across venues
        Strategies: 'price_time', 'dark_first', 'fee_aware', 'latency_aware'
        """
        remaining = size
        fills = []
        
        # Strategy 1: Check dark pools first (midpoint execution)
        if strategy in ['dark_first', 'smart'] and self.dark_pools:
            for dp in self.dark_pools:
                if remaining <= 0:
                    break
                    
                # Dark pool fills at midpoint if liquidity available
                available = min(dp.ask_size, remaining)
                if available > 0:
                    fill_price = dp.get_midpoint()
                    fills.append({
                        'venue': dp.name,
                        'price': fill_price,
                        'size': available,
                        'fee': 0,  # Dark pools typically no fees
                        'latency': dp.latency_ms
                    })
                    remaining -= available
        
        # Route to lit venues
        if remaining > 0:
            # Rank venues by strategy
            if strategy == 'price_time':
                # Best price, then lowest latency
                ranked = sorted(self.venues, 
                              key=lambda v: (v.best_ask, v.latency_ms))
            elif strategy == 'fee_aware':
                # Minimize total cost (price + taker fee)
                ranked = sorted(self.venues,
                              key=lambda v: (v.best_ask + v.taker_fee, v.latency_ms))
            elif strategy == 'latency_aware':
                # Avoid high-latency (likely stale quotes)
                ranked = sorted(self.venues,
                              key=lambda v: (v.best_ask, v.hft_presence, v.latency_ms))
            else:  # 'smart' - multi-factor
                # Custom scoring: price, latency, HFT risk, fees
                ranked = sorted(self.venues,
                              key=lambda v: (
                                  v.best_ask * 1000 +  # Price (most important)
                                  v.taker_fee * 1000 +  # Fees
                                  v.hft_presence * 10 +  # HFT risk
                                  v.latency_ms * 0.1  # Latency
                              ))
            
            # Execute on ranked venues
            for venue in ranked:
                if remaining <= 0:
                    break
                
                # Simulate latency arbitrage risk
                if np.random.random() < venue.hft_presence:
                    # HFT arbed away liquidity, price moved up
                    actual_price = venue.best_ask + 0.01
                    available = max(0, venue.ask_size // 2)  # Reduced liquidity
                else:
                    actual_price = venue.best_ask
                    available = venue.ask_size
                
                fill_size = min(available, remaining)
                
                if fill_size > 0:
                    fills.append({
                        'venue': venue.name,
                        'price': actual_price,
                        'size': fill_size,
                        'fee': venue.taker_fee * fill_size,
                        'latency': venue.latency_ms
                    })
                    remaining -= fill_size
        
        self.execution_log.append({
            'strategy': strategy,
            'fills': fills,
            'unfilled': remaining
        })
        
        return fills
    
    def calculate_execution_quality(self, fills: List[dict], nbbo_ask: float) -> dict:
        """Calculate execution quality metrics"""
        if not fills:
            return {
                'avg_price': None,
                'total_cost': 0,
                'total_fees': 0,
                'slippage_bps': None,
                'fill_rate': 0
            }
        
        total_shares = sum(f['size'] for f in fills)
        total_cost = sum(f['price'] * f['size'] for f in fills)
        total_fees = sum(f['fee'] for f in fills)
        avg_price = total_cost / total_shares if total_shares > 0 else 0
        
        slippage_bps = ((avg_price - nbbo_ask) / nbbo_ask * 10000) if total_shares > 0 else 0
        
        return {
            'avg_price': avg_price,
            'total_cost': total_cost + total_fees,
            'total_fees': total_fees,
            'slippage_bps': slippage_bps,
            'fill_rate': total_shares / (total_shares + fills[0].get('unfilled', 0))
        }

# Create realistic market structure
np.random.seed(42)

# Define venues (fragmented market)
venues = [
    Venue('NYSE', latency_ms=0.3, maker_fee=-0.0020, taker_fee=0.0030,
          best_bid=100.00, best_ask=100.01, bid_size=1000, ask_size=1200, hft_presence=0.2),
    Venue('NASDAQ', latency_ms=0.25, maker_fee=-0.0020, taker_fee=0.0030,
          best_bid=100.00, best_ask=100.01, bid_size=800, ask_size=1500, hft_presence=0.3),
    Venue('BATS', latency_ms=0.4, maker_fee=-0.0015, taker_fee=0.0025,
          best_bid=99.99, best_ask=100.01, bid_size=600, ask_size=900, hft_presence=0.4),
    Venue('EDGX', latency_ms=0.35, maker_fee=-0.0010, taker_fee=0.0020,
          best_bid=100.00, best_ask=100.02, bid_size=500, ask_size=700, hft_presence=0.25),
    Venue('IEX', latency_ms=0.35, maker_fee=0.0000, taker_fee=0.0009,
          best_bid=100.00, best_ask=100.01, bid_size=400, ask_size=800, hft_presence=0.05),
]

dark_pools = [
    Venue('SIGMA_DARK', latency_ms=0.5, maker_fee=0, taker_fee=0,
          best_bid=100.00, best_ask=100.01, bid_size=500, ask_size=500, hft_presence=0.0),
    Venue('CROSSFINDER', latency_ms=0.6, maker_fee=0, taker_fee=0,
          best_bid=100.00, best_ask=100.01, bid_size=300, ask_size=400, hft_presence=0.0),
]

# Initialize router
router = SmartOrderRouter(venues, dark_pools)

# Test different routing strategies
order_size = 3000
strategies = ['price_time', 'dark_first', 'fee_aware', 'latency_aware', 'smart']

print("SMART ORDER ROUTING COMPARISON")
print("="*60)
print(f"Order: BUY {order_size} shares\n")

# Get NBBO
best_bid, best_ask = router.get_nbbo()
print(f"NBBO: ${best_bid:.2f} - ${best_ask:.2f}")
print(f"NBBO Spread: {(best_ask - best_bid)*10000:.1f} bps\n")

results = {}

for strategy in strategies:
    # Reset venues (simplified - real markets are dynamic)
    router = SmartOrderRouter(venues.copy(), dark_pools.copy())
    
    fills = router.route_market_buy(order_size, strategy=strategy)
    quality = router.calculate_execution_quality(fills, best_ask)
    
    results[strategy] = {
        'fills': fills,
        'quality': quality
    }
    
    print(f"\n{strategy.upper().replace('_', ' ')} Strategy:")
    print(f"  Average price: ${quality['avg_price']:.4f}")
    print(f"  Total cost: ${quality['total_cost']:.2f}")
    print(f"  Fees: ${quality['total_fees']:.2f}")
    print(f"  Slippage: {quality['slippage_bps']:.2f} bps")
    print(f"  Venues used: {len(fills)}")
    
    for fill in fills:
        print(f"    {fill['venue']:<15} {fill['size']:>5} @ ${fill['price']:.4f} "
              f"(fee: ${fill['fee']:.2f}, latency: {fill['latency']:.2f}ms)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Execution price comparison
strategy_names = list(results.keys())
avg_prices = [results[s]['quality']['avg_price'] for s in strategy_names]

axes[0, 0].bar(strategy_names, avg_prices, alpha=0.7, color='steelblue')
axes[0, 0].axhline(best_ask, color='red', linestyle='--', linewidth=2, label=f'NBBO Ask: ${best_ask:.2f}')
axes[0, 0].set_title('Average Execution Price by Strategy')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Total cost (price + fees)
total_costs = [results[s]['quality']['total_cost'] for s in strategy_names]

axes[0, 1].bar(strategy_names, total_costs, alpha=0.7, color='coral')
axes[0, 1].set_title('Total Execution Cost (Price + Fees)')
axes[0, 1].set_ylabel('Total Cost ($)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Slippage in basis points
slippages = [results[s]['quality']['slippage_bps'] for s in strategy_names]

axes[1, 0].bar(strategy_names, slippages, alpha=0.7, color='green')
axes[1, 0].axhline(0, color='black', linewidth=1)
axes[1, 0].set_title('Slippage vs NBBO Ask')
axes[1, 0].set_ylabel('Slippage (bps)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Venue distribution (for 'smart' strategy)
smart_fills = results['smart']['fills']
venue_sizes = {}
for fill in smart_fills:
    venue_sizes[fill['venue']] = venue_sizes.get(fill['venue'], 0) + fill['size']

axes[1, 1].pie(venue_sizes.values(), labels=venue_sizes.keys(), autopct='%1.1f%%',
               startangle=90)
axes[1, 1].set_title('Venue Distribution (SMART Strategy)')

plt.tight_layout()
plt.show()

# Cost savings analysis
print("\n" + "="*60)
print("COST SAVINGS ANALYSIS")
print("="*60)

baseline = results['price_time']['quality']['total_cost']
for strategy in strategies[1:]:
    cost = results[strategy]['quality']['total_cost']
    savings = baseline - cost
    savings_bps = (savings / baseline) * 10000
    print(f"{strategy.replace('_', ' ').title():>20}: ${savings:>8.2f} ({savings_bps:>6.2f} bps)")
