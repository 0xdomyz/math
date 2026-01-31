import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("OPTIMAL EXECUTION ALGORITHMS")
print("="*80)


class ExecutionCostCalculator:
    """Calculate execution costs given an execution path"""
    
    def __init__(self, prices, volumes, tau=0.001, gamma=0.0001):
        self.prices = prices
        self.volumes = volumes
        self.tau = tau
        self.gamma = gamma
        self.S0 = prices[0]
    
    def calculate_execution_cost(self, x_path):
        """
        Calculate total execution cost (Implementation Shortfall)
        
        IS = Σ[x_t * (temporary_impact + permanent_impact)]
        """
        n = len(x_path)
        costs = np.zeros(n)
        
        X_cumsum = np.cumsum(x_path)  # Cumulative executed
        
        for t in range(n):
            # Temporary impact: recovers after execution
            temp_impact = self.tau * x_path[t]
            
            # Permanent impact: reflects in all future prices
            perm_impact = self.gamma * X_cumsum[t]
            
            # Total impact cost at time t
            impact_cost = temp_impact + perm_impact
            
            # Execution price = market price - impact
            execution_price = self.prices[t] - impact_cost
            
            # Cost for shares executed at time t
            costs[t] = x_path[t] * (self.prices[t] - execution_price)
        
        total_cost = np.sum(costs)
        cost_basis_points = (total_cost / (self.S0 * self.total_shares)) * 10000 if self.total_shares > 0 else 0
        
        return {
            'total_cost': total_cost,
            'cost_bps': cost_basis_points,
            'avg_execution_price': (self.S0 * self.total_shares - total_cost) / self.total_shares if self.total_shares > 0 else self.S0,
            'costs_by_period': costs
        }

# Simulation setup
total_shares = 100000  # 100k shares
total_days = 5
periods_per_day = 6.5 * 60  # 1-minute bars
T = total_days  # 5 periods for simplicity

# Generate market data
simulator = MarketSimulator(S0=100, sigma=0.015, dt=1/252)
prices = simulator.generate_path(n_steps=T, seed=42)
volumes = simulator.generate_volume_path(n_steps=T, avg_volume=total_shares / T, seasonality=True)

print(f"\nInitial price: ${prices[0]:.2f}")
print(f"Final price: ${prices[-1]:.2f}")
print(f"Total shares to execute: {total_shares:,}")
print(f"Average daily volume: {np.mean(volumes):.0f} shares")

# Impact parameters (calibrated to typical market)
impact_params = {
    'tau': 0.0001,      # Temporary impact: 1 bp per 1% of daily volume
    'gamma': 0.00001,   # Permanent impact: 0.1 bp per 1% of daily volume
    'sigma': 0.015      # Daily volatility
}

# Test different algorithms
algorithms = {
    'TWAP': TWAPAlgorithm(total_shares, T, impact_params),
    'VWAP': VWAPAlgorithm(total_shares, T, pov=0.3, impact_params=impact_params),
    'Almgren-Chriss (λ=1e-6)': OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=1e-6, impact_params=impact_params),
    'Almgren-Chriss (λ=1e-5)': OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=1e-5, impact_params=impact_params),
    'Almgren-Chriss (λ=1e-4)': OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=1e-4, impact_params=impact_params),
}

print("\n" + "="*80)
print("EXECUTION ALGORITHM COMPARISON")
print("="*80)

results = {}

for name, algo in algorithms.items():
    if 'VWAP' in name:
        x_path = algo.get_execution_path(volumes)
    else:
        x_path = algo.get_execution_path()
    
    # Normalize to ensure we execute all shares
    x_path = x_path / np.sum(x_path) * total_shares
    
    # Calculate costs
    calculator = ExecutionCostCalculator(prices, volumes, tau=impact_params['tau'], gamma=impact_params['gamma'])
    calculator.total_shares = total_shares
    cost_result = calculator.calculate_execution_cost(x_path)
    
    results[name] = {
        'x_path': x_path,
        'cost_result': cost_result,
        'total_executed': np.sum(x_path)
    }
    
    print(f"\n{name}:")
    print(f"  Execution path (shares/period): {x_path}")
    print(f"  Total execution cost: ${cost_result['total_cost']:,.2f}")
    print(f"  Cost (basis points): {cost_result['cost_bps']:.2f} bps")
    print(f"  Average execution price: ${cost_result['avg_execution_price']:.4f}")
    print(f"  vs Arrival price difference: ${(prices[0] - cost_result['avg_execution_price']):.4f}")

# Compare to benchmarks
print("\n" + "="*80)
print("BENCHMARK COMPARISON")
print("="*80)

# VWAP benchmark
vwap = np.average(prices, weights=volumes)
twap = np.mean(prices)
arrival_price = prices[0]

print(f"\nVWAP: ${vwap:.4f}")
print(f"TWAP: ${twap:.4f}")
print(f"Arrival Price: ${arrival_price:.4f}")

# Compare algorithm execution prices
print(f"\n{'Algorithm':<30} {'Execution Price':<15} {'vs VWAP (bps)':<15} {'vs Arrival (bps)':<15}")
print("-" * 75)

for name, result in results.items():
    exec_price = result['cost_result']['avg_execution_price']
    vs_vwap = (exec_price - vwap) / vwap * 10000
    vs_arrival = (exec_price - arrival_price) / arrival_price * 10000
    
    print(f"{name:<30} ${exec_price:<14.4f} {vs_vwap:<14.2f} {vs_arrival:<14.2f}")

# Sensitivity analysis: varying risk aversion
print("\n" + "="*80)
print("SENSITIVITY: RISK AVERSION PARAMETER (λ)")
print("="*80)

lambda_values = np.logspace(-8, -3, 10)
sensitivity_results = []

for lambda_risk in lambda_values:
    algo = OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=lambda_risk, impact_params=impact_params)
    x_path = algo.get_execution_path()
    x_path = x_path / np.sum(x_path) * total_shares
    
    calculator = ExecutionCostCalculator(prices, volumes, tau=impact_params['tau'], gamma=impact_params['gamma'])
    calculator.total_shares = total_shares
    cost_result = calculator.calculate_execution_cost(x_path)
    
    sensitivity_results.append({
        'lambda': lambda_risk,
        'cost_bps': cost_result['cost_bps'],
        'max_execution_size': np.max(x_path),
        'avg_execution_size': np.mean(x_path)
    })

df_sensitivity = pd.DataFrame(sensitivity_results)
print(f"\n{'λ':<15} {'Cost (bps)':<15} {'Max Size':<15} {'Avg Size':<15}")
print("-" * 60)
for _, row in df_sensitivity.iterrows():
    print(f"{row['lambda']:<15.2e} {row['cost_bps']:<15.2f} {row['max_execution_size']:<15,.0f} {row['avg_execution_size']:<15,.0f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Price path
ax = axes[0, 0]
ax.plot(prices, 'b-', linewidth=2, label='Market Price')
ax.axhline(vwap, color='g', linestyle='--', linewidth=1.5, label=f'VWAP: ${vwap:.2f}')
ax.axhline(twap, color='r', linestyle=':', linewidth=1.5, label=f'TWAP: ${twap:.2f}')
ax.set_title('Price Path During Execution')
ax.set_xlabel('Period')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Volume path
ax = axes[0, 1]
ax.bar(range(len(volumes)), volumes, alpha=0.6, color='blue')
ax.set_title('Market Volume Pattern')
ax.set_xlabel('Period')
ax.set_ylabel('Volume (shares)')
ax.grid(alpha=0.3)

# Plot 3: Execution paths comparison
ax = axes[0, 2]
for name, result in results.items():
    ax.plot(result['x_path'], marker='o', label=name, linewidth=1.5)
ax.set_title('Execution Paths by Algorithm')
ax.set_xlabel('Period')
ax.set_ylabel('Execution Size (shares)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Cumulative execution
ax = axes[1, 0]
for name, result in results.items():
    cumsum = np.cumsum(result['x_path'])
    ax.plot(cumsum, marker='s', label=name, linewidth=1.5)
ax.axhline(total_shares, color='k', linestyle='--', alpha=0.5, label='Target')
ax.set_title('Cumulative Execution')
ax.set_xlabel('Period')
ax.set_ylabel('Cumulative Shares')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 5: Cost comparison (bar chart)
ax = axes[1, 1]
costs_bps = [results[name]['cost_result']['cost_bps'] for name in results.keys()]
colors = ['green' if c < np.mean(costs_bps) else 'red' for c in costs_bps]
ax.bar(range(len(results)), costs_bps, color=colors, alpha=0.7)
ax.set_xticks(range(len(results)))
ax.set_xticklabels(list(results.keys()), rotation=45, ha='right', fontsize=8)
ax.set_title('Execution Cost Comparison')
ax.set_ylabel('Cost (basis points)')
ax.axhline(np.mean(costs_bps), color='k', linestyle='--', alpha=0.5, label='Mean')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Sensitivity to λ
ax = axes[1, 2]
ax.semilogx(df_sensitivity['lambda'], df_sensitivity['cost_bps'], marker='o', linewidth=2, markersize=6)
ax.set_title('Sensitivity: Execution Cost vs Risk Aversion (λ)')
ax.set_xlabel('Risk Aversion Parameter (λ)')
ax.set_ylabel('Cost (basis points)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*80)
print("EXECUTION SUMMARY STATISTICS")
print("="*80)

print(f"\nCost Statistics (basis points):")
costs = [r['cost_result']['cost_bps'] for r in results.values()]
print(f"  Mean cost: {np.mean(costs):.2f} bps")
print(f"  Min cost: {np.min(costs):.2f} bps (best algorithm)")
print(f"  Max cost: {np.max(costs):.2f} bps (worst algorithm)")
print(f"  Std dev: {np.std(costs):.2f} bps")
print(f"  Spread: {np.max(costs) - np.min(costs):.2f} bps")

best_algo = min(results.items(), key=lambda x: x[1]['cost_result']['cost_bps'])
print(f"\nBest performing algorithm: {best_algo[0]}")
print(f"  Cost: {best_algo[1]['cost_result']['cost_bps']:.2f} bps")
print(f"  Savings vs worst: {np.max(costs) - best_algo[1]['cost_result']['cost_bps']:.2f} bps")

# Trade-off: speed vs cost
print(f"\nExecution Speed Trade-offs:")
for name, result in results.items():
    x_path = result['x_path']
    # Measure concentration (lower = more spread out)
    concentration = np.max(x_path) / np.mean(x_path)
    cost_bps = result['cost_result']['cost_bps']
    print(f"  {name}: Concentration={concentration:.2f}, Cost={cost_bps:.2f} bps")