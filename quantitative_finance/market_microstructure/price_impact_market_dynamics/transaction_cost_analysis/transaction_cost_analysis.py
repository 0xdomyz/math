import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

class TransactionCostAnalyzer:
    def __init__(self):
        self.trades = []
        self.benchmarks = []
        
    def generate_market_volume_profile(self, num_periods=390, u_shape=True):
        """Generate realistic intraday volume pattern"""
        if u_shape:
            # U-shaped volume: high at open/close, low midday
            periods = np.arange(1, num_periods + 1)
            volume = 100 + 50 * (np.sin(periods / num_periods * np.pi) ** 2)
        else:
            # Flat volume
            volume = np.ones(num_periods) * 100
        
        return volume / volume.sum()  # Normalize to probabilities
    
    def simulate_execution_and_market(self, order_size, execution_strategy='vwap'):
        """Simulate order execution with realistic market"""
        num_periods = 60  # 1 hour = 60 minutes
        volume_profile = self.generate_market_volume_profile(num_periods)
        
        # Market prices (random walk)
        prices = [100.0]
        for _ in range(num_periods):
            move = np.random.normal(0, 0.02)
            prices.append(prices[-1] + move)
        
        # Market volumes (proportional to profile)
        total_market_volume = order_size * 5  # Market volume 5x order
        market_volumes = (volume_profile * total_market_volume).astype(int)
        
        # Execution strategy determines filling
        if execution_strategy == 'vwap':
            execution_sizes = (volume_profile * order_size).astype(int)
        elif execution_strategy == 'twap':
            execution_sizes = np.ones(num_periods) * (order_size / num_periods)
        elif execution_strategy == 'front_loaded':
            execution_sizes = np.where(np.arange(num_periods) < 20,
                                      order_size / 20, 0).astype(int)
        elif execution_strategy == 'passive':
            execution_sizes = np.minimum((volume_profile * order_size * 0.5).astype(int),
                                        order_size // num_periods).astype(int)
        else:
            execution_sizes = (volume_profile * order_size).astype(int)
        
        # Calculate benchmarks
        arrival_price = prices[0]
        vwap = np.sum(prices[:-1] * market_volumes) / np.sum(market_volumes)
        twap = np.mean(prices[:-1])
        execution_price = np.sum(prices[:-1] * execution_sizes) / np.sum(execution_sizes) if np.sum(execution_sizes) > 0 else prices[-1]
        final_price = prices[-1]
        
        return {
            'arrival_price': arrival_price,
            'vwap': vwap,
            'twap': twap,
            'execution_price': execution_price,
            'final_price': final_price,
            'execution_sizes': execution_sizes,
            'market_volumes': market_volumes,
            'prices': prices,
            'order_size': order_size
        }
    
    def calculate_tca_metrics(self, execution_data, benchmark='vwap'):
        """Calculate comprehensive TCA metrics"""
        arrival = execution_data['arrival_price']
        executed = execution_data['execution_price']
        vwap = execution_data['vwap']
        twap = execution_data['twap']
        final = execution_data['final_price']
        order_size = execution_data['order_size']
        
        metrics = {
            'benchmark': benchmark,
            'execution_price': executed,
            'arrival_price': arrival,
            'vwap': vwap,
            'twap': twap,
            'final_price': final,
        }
        
        if benchmark == 'vwap':
            benchmark_price = vwap
        elif benchmark == 'twap':
            benchmark_price = twap
        elif benchmark == 'arrival':
            benchmark_price = arrival
        else:
            benchmark_price = vwap
        
        # Cost metrics
        cost_vs_benchmark = (executed - benchmark_price) * order_size
        cost_bps = (executed / benchmark_price - 1) * 10000  # in basis points
        
        # Implementation shortfall (vs arrival)
        impl_shortfall = (executed - arrival) * order_size
        impl_shortfall_bps = (executed / arrival - 1) * 10000
        
        # Timing cost (opportunity cost)
        timing_cost = (final - arrival) * order_size
        timing_cost_bps = (final / arrival - 1) * 10000
        
        metrics.update({
            'cost_vs_benchmark': cost_vs_benchmark,
            'cost_bps': cost_bps,
            'implementation_shortfall': impl_shortfall,
            'impl_shortfall_bps': impl_shortfall_bps,
            'timing_cost': timing_cost,
            'timing_cost_bps': timing_cost_bps,
            'participation_rate': np.sum(execution_data['execution_sizes']) / np.sum(execution_data['market_volumes']) * 100,
        })
        
        return metrics

# Scenario 1: Different execution strategies comparison
print("Scenario 1: Execution Strategy Comparison")
print("=" * 80)

analyzer = TransactionCostAnalyzer()
strategies = ['vwap', 'twap', 'front_loaded', 'passive']
results = {}

for strategy in strategies:
    exec_data = analyzer.simulate_execution_and_market(order_size=100000, execution_strategy=strategy)
    metrics = analyzer.calculate_tca_metrics(exec_data, benchmark='vwap')
    results[strategy] = metrics
    
    print(f"Strategy: {strategy:>15}")
    print(f"  Execution Price: ${metrics['execution_price']:.4f}")
    print(f"  VWAP:            ${metrics['vwap']:.4f}")
    print(f"  Cost vs VWAP:    {metrics['cost_bps']:>8.2f} bps (${metrics['cost_vs_benchmark']:>12,.0f})")
    print(f"  Impl Shortfall:  {metrics['impl_shortfall_bps']:>8.2f} bps (${metrics['implementation_shortfall']:>12,.0f})")
    print(f"  Participation:   {metrics['participation_rate']:>8.1f}% of volume")
    print()

# Scenario 2: Market impact vs order size
print("Scenario 2: Market Impact vs Order Size")
print("=" * 80)

order_sizes = [10000, 50000, 100000, 250000, 500000]
impact_results = []

for size in order_sizes:
    exec_data = analyzer.simulate_execution_and_market(order_size=size, execution_strategy='vwap')
    metrics = analyzer.calculate_tca_metrics(exec_data, benchmark='vwap')
    impact_results.append(metrics['cost_bps'])
    
    participation = size / (size * 5) * 100  # Rough participation rate
    print(f"Order Size: {size:>10,} | Impact: {metrics['cost_bps']:>8.2f} bps | Participation: {participation:>6.1f}%")

# Scenario 3: Arrival price vs VWAP vs final price analysis
print(f"\n\nScenario 3: Price Dynamics Analysis")
print("=" * 80)

exec_data = analyzer.simulate_execution_and_market(order_size=100000, execution_strategy='vwap')
metrics = analyzer.calculate_tca_metrics(exec_data, benchmark='vwap')

print(f"Arrival Price:     ${metrics['arrival_price']:.4f}")
print(f"Execution Price:   ${metrics['execution_price']:.4f}")
print(f"VWAP:              ${metrics['vwap']:.4f}")
print(f"Final Price:       ${metrics['final_price']:.4f}")
print(f"\nCost Decomposition:")
print(f"  Execution vs Arrival: {metrics['impl_shortfall_bps']:>8.2f} bps")
print(f"  Market move (timing): {metrics['timing_cost_bps']:>8.2f} bps")
print(f"  Execution vs VWAP:    {metrics['cost_bps']:>8.2f} bps")

# Scenario 4: Broker/Venue ranking
print(f"\n\nScenario 4: Broker Performance Ranking")
print("=" * 80)

# Simulate multiple executions by different "brokers"
brokers = ['Broker A', 'Broker B', 'Broker C', 'Broker D']
num_executions = 20
broker_results = {broker: [] for broker in brokers}

for broker in brokers:
    for _ in range(num_executions):
        exec_data = analyzer.simulate_execution_and_market(order_size=100000, execution_strategy='vwap')
        # Add broker-specific noise
        noise = np.random.normal(0, 0.5 if 'A' in broker else 1.0 if 'B' in broker else 0.3)
        exec_data['execution_price'] += noise / 10000
        
        metrics = analyzer.calculate_tca_metrics(exec_data, benchmark='vwap')
        broker_results[broker].append(metrics['cost_bps'])

# Aggregate and rank
print(f"{'Broker':>15} | {'Avg Cost (bps)':>15} | {'Std Dev':>10} | Rank")
print("-" * 60)

broker_avgs = {broker: np.mean(results) for broker, results in broker_results.items()}
for rank, (broker, avg) in enumerate(sorted(broker_avgs.items(), key=lambda x: x[1]), 1):
    std_dev = np.std(broker_results[broker])
    print(f"{broker:>15} | {avg:>15.2f} | {std_dev:>10.2f} | #{rank}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Strategy comparison
strategy_names = list(results.keys())
strategy_costs = [results[s]['cost_bps'] for s in strategy_names]
colors_strat = plt.cm.viridis(np.linspace(0, 1, len(strategy_names)))

bars = axes[0, 0].bar(strategy_names, strategy_costs, color=colors_strat, alpha=0.7)
axes[0, 0].set_ylabel('Cost vs VWAP (bps)')
axes[0, 0].set_title('Scenario 1: Execution Strategy Comparison')
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 0].grid(alpha=0.3, axis='y')

for bar, cost in zip(bars, strategy_costs):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{cost:.1f}', ha='center', va='bottom' if cost > 0 else 'top', fontweight='bold')

# Plot 2: Impact vs order size
axes[0, 1].plot(order_sizes, impact_results, 'o-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Order Size (shares)')
axes[0, 1].set_ylabel('Market Impact (bps)')
axes[0, 1].set_title('Scenario 2: Impact Scaling')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price levels during execution
prices = exec_data['prices']
periods = np.arange(len(prices))

axes[1, 0].plot(periods, prices, linewidth=2, label='Market Price')
axes[1, 0].axhline(y=metrics['arrival_price'], color='g', linestyle='--', label='Arrival Price', alpha=0.7)
axes[1, 0].axhline(y=metrics['vwap'], color='b', linestyle='--', label='VWAP', alpha=0.7)
axes[1, 0].axhline(y=metrics['execution_price'], color='r', linestyle='--', label='Execution Price', alpha=0.7)
axes[1, 0].fill_between(periods, metrics['arrival_price'], metrics['execution_price'], alpha=0.2)
axes[1, 0].set_xlabel('Period')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].set_title('Scenario 3: Execution vs Market')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Broker rankings
broker_names_sorted = sorted(broker_avgs.keys(), key=lambda x: broker_avgs[x])
broker_costs_sorted = [broker_avgs[b] for b in broker_names_sorted]
colors_brokers = plt.cm.RdYlGn_r(np.linspace(0, 1, len(broker_names_sorted)))

bars = axes[1, 1].barh(broker_names_sorted, broker_costs_sorted, color=colors_brokers, alpha=0.7)
axes[1, 1].set_xlabel('Average Cost (bps)')
axes[1, 1].set_title('Scenario 4: Broker Performance Ranking')
axes[1, 1].grid(alpha=0.3, axis='x')

for bar, cost in zip(bars, broker_costs_sorted):
    width = bar.get_width()
    axes[1, 1].text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{cost:.1f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Best strategy: {min(results, key=lambda x: results[x]['cost_bps'])}")
print(f"Worst strategy: {max(results, key=lambda x: results[x]['cost_bps'])}")
print(f"Range: {max(impact_results) - min(impact_results):.1f} bps (√ law supports smaller orders)")
print(f"Benchmark choice matters: Different benchmarks show different results")
