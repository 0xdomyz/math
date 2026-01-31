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
