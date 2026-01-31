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

class MarketSimulator:
    """Simulate market dynamics during execution"""
    
    def __init__(self, S0=100, sigma=0.02, dt=1/252/6.5/60):  # 1-minute bars
        self.S0 = S0
        self.sigma = sigma
        self.dt = dt
    
    def generate_path(self, n_steps, seed=None):
        """Generate GBM price path"""
        if seed is not None:
            np.random.seed(seed)
        
        returns = np.random.normal(0, self.sigma * np.sqrt(self.dt), n_steps)
        prices = self.S0 * np.exp(np.cumsum(returns))
        
        return prices
    
    def generate_volume_path(self, n_steps, avg_volume=1000, seasonality=True):
        """Generate intraday volume pattern"""
        t = np.arange(n_steps)
        
        if seasonality:
            # U-shaped volume pattern (high open/close, low midday)
            factor = 1.5 + 0.5 * np.sin(np.pi * t / n_steps)
        else:
            factor = np.ones(n_steps)
        
        volume = avg_volume * factor * np.random.gamma(2, 1, n_steps)
        
        return np.maximum(volume, 10)  # Min 10 shares

class ExecutionAlgorithm:
    """Base class for execution algorithms"""
    
    def __init__(self, total_shares, T, impact_params=None):
        self.total_shares = total_shares
        self.T = T  # Total periods
        self.tau = impact_params.get('tau', 0.001) if impact_params else 0.001  # Temporary impact
        self.gamma = impact_params.get('gamma', 0.0001) if impact_params else 0.0001  # Permanent impact
        self.sigma = impact_params.get('sigma', 0.02) if impact_params else 0.02  # Volatility
    
    def get_execution_path(self, volumes=None):
        """Return execution quantities at each period"""
        raise NotImplementedError

class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution"""
    
    def get_execution_path(self, volumes=None):
        """Execute uniformly over time"""
        x = np.ones(self.T) * self.total_shares / self.T
        return x

class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution"""
    
    def __init__(self, total_shares, T, pov=0.15, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.pov = pov  # Participation of volume
    
    def get_execution_path(self, volumes):
        """Execute proportional to market volume"""
        if volumes is None:
            volumes = np.ones(self.T) * np.mean(volumes or 1000)
        
        # Start with POV-based execution
        x = self.pov * volumes
        
        # Ensure total execution matches target
        remaining = self.total_shares - np.sum(x)
        if remaining > 0:
            # Catch up in later periods
            scale = self.total_shares / np.sum(x)
            x = scale * x
        
        return np.minimum(x, self.total_shares)  # Cap at total shares

class OptimalExecutionAlmgrenChriss(ExecutionAlgorithm):
    """Almgren-Chriss optimal execution (linear impact)"""
    
    def __init__(self, total_shares, T, lambda_risk=1e-6, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.lambda_risk = lambda_risk  # Risk aversion parameter
    
    def get_execution_path(self):
        """
        Optimal piecewise-linear execution path
        Solves: min E[IS] = γX² + τ∫x_t² dt + λ*σ²*∫x_t² dt
        """
        # Almgren-Chriss solution for linear impact
        # x_t = X * sqrt(kappa/T) * sinh(g*t) / sinh(g*T)
        # where g = sqrt(kappa) * sqrt(lambda_risk * sigma² / tau)
        
        # Simplified: use gradient descent optimization
        def objective(x_path):
            """Objective function to minimize"""
            # Ensure sum = total shares
            x_path = np.abs(x_path)  # Ensure positive
            x_path = x_path / np.sum(x_path) * self.total_shares
            
            X_cumsum = np.cumsum(x_path)  # Cumulative executed
            
            # Permanent impact cost: γ × (cumulative executed)²
            permanent_cost = self.gamma * np.sum(X_cumsum**2)
            
            # Temporary impact cost: τ × (sum of squared trades)
            temporary_cost = self.tau * np.sum(x_path**2)
            
            # Timing risk: λ × σ² × (sum of squared trades) × (time to completion)
            timing_risk = self.lambda_risk * (self.sigma**2) * np.sum(x_path**2 * (self.T - np.arange(self.T)))
            
            return permanent_cost + temporary_cost + timing_risk
        
        # Initial guess: uniform (TWAP)
        x0 = np.ones(self.T) * self.total_shares / self.T
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=[(0, self.total_shares) for _ in range(self.T)],
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_shares}
        )
        
        x_opt = result.x
        x_opt = np.maximum(x_opt, 0)  # Ensure non-negative
        x_opt = x_opt / np.sum(x_opt) * self.total_shares  # Re-normalize
        
        return x_opt

class LimitOrderExecutor(ExecutionAlgorithm):
    """Execution using limit orders at multiple levels"""
    
    def __init__(self, total_shares, T, limit_levels=5, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.limit_levels = limit_levels
    
    def get_execution_path(self, prices):
        """
        Execute via limit orders at different price levels
        Front-load with limit orders, use market orders for remainder
        """
        x_path = np.zeros(self.T)
        executed = 0
        
        for t in range(self.T):
            remaining = self.total_shares - executed
            
            if remaining <= 0:
                break
            
            # Probability of limit order fill decreases with depth
            # Allocate: 20% limit (best bid - 1bp), 30% limit (- 2bp), 30% limit (- 3bp), 20% market
            fill_probs = [0.8, 0.5, 0.2, 0.05, 1.0]
            
            for level, prob in enumerate(fill_probs):
                if level < self.limit_levels:
                    size = remaining * (1 - level * 0.15)
                else:
                    size = remaining
                
                if np.random.random() < prob:
                    x_path[t] += size
                    executed += size
                    break
        
        # Ensure total is hit (force market orders if needed)
        if executed < self.total_shares:
            x_path[self.T - 1] += self.total_shares - executed
        
        return np.minimum(x_path, self.total_shares)

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