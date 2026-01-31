import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class MarketParams:
    """Market microstructure parameters"""
    sigma: float  # Volatility (annualized)
    gamma: float  # Permanent impact coefficient
    eta: float    # Temporary impact coefficient
    spread: float # Bid-ask spread
    
@dataclass
class ExecutionParams:
    """Execution problem parameters"""
    X: int        # Total shares to trade
    T: float      # Time horizon (days)
    lam: float    # Risk aversion parameter
    N: int        # Number of time intervals

class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution model"""
    
    def __init__(self, market: MarketParams, execution: ExecutionParams):
        self.market = market
        self.execution = execution
        
        # Derived parameters
        self.dt = execution.T / execution.N  # Time step
        self.kappa = np.sqrt(execution.lam * market.sigma**2 / market.eta)
        
    def optimal_trajectory_continuous(self):
        """
        Compute optimal trajectory using continuous-time solution
        
        x(t) = X * sinh(κ(T-t)) / sinh(κT)
        """
        times = np.linspace(0, self.execution.T, self.execution.N + 1)
        
        X = self.execution.X
        T = self.execution.T
        kappa = self.kappa
        
        # Inventory remaining at each time
        inventory = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)
        
        # Trade sizes (differences)
        trades = -np.diff(inventory)
        
        return times, inventory, trades
    
    def optimal_trajectory_discrete(self):
        """
        Compute optimal trajectory using discrete-time quadratic program
        
        More accurate for discrete execution
        """
        X = self.execution.X
        N = self.execution.N
        T = self.execution.T
        dt = self.dt
        
        sigma = self.market.sigma / np.sqrt(252)  # Daily volatility
        gamma = self.market.gamma
        eta = self.market.eta
        lam = self.execution.lam
        
        # Build quadratic program matrices
        # Minimize: (1/2) * v^T Q v + c^T v
        # subject to: sum(v) = X
        
        # Q matrix (captures temporary impact and risk)
        Q = np.zeros((N, N))
        
        for i in range(N):
            # Temporary impact cost
            Q[i, i] += 2 * eta
            
            # Risk penalty (variance of unexecuted inventory)
            for j in range(i, N):
                # Inventory variance contribution
                Q[i, j] += 2 * lam * sigma**2 * dt
                Q[j, i] = Q[i, j]
        
        # Linear term (permanent impact)
        c = np.ones(N) * gamma
        
        # Solve using closed-form solution for equality constraint
        # v* = (1/sum(Q^-1)) * Q^-1 * 1
        Q_inv = np.linalg.inv(Q)
        ones = np.ones(N)
        
        v_star = (X / (ones @ Q_inv @ ones)) * Q_inv @ ones
        
        # Construct inventory trajectory
        inventory = np.zeros(N + 1)
        inventory[0] = X
        
        for i in range(N):
            inventory[i + 1] = inventory[i] - v_star[i]
        
        times = np.linspace(0, T, N + 1)
        
        return times, inventory, v_star
    
    def expected_cost(self, trades):
        """Calculate expected execution cost"""
        X = self.execution.X
        
        # Permanent impact cost
        total_executed = np.cumsum(trades)
        permanent_cost = self.market.gamma * np.sum(trades * total_executed)
        
        # Temporary impact cost
        temporary_cost = self.market.eta * np.sum(trades**2)
        
        # Spread cost
        spread_cost = (self.market.spread / 2) * np.sum(trades)
        
        total_cost = permanent_cost + temporary_cost + spread_cost
        
        return {
            'total': total_cost,
            'permanent': permanent_cost,
            'temporary': temporary_cost,
            'spread': spread_cost
        }
    
    def risk_measure(self, inventory):
        """Calculate timing risk (variance of execution)"""
        dt = self.dt
        sigma = self.market.sigma / np.sqrt(252)  # Daily volatility
        
        # Variance from holding unexecuted inventory
        # Each period contributes: σ² × inventory² × dt
        variance = np.sum(inventory[:-1]**2 * sigma**2 * dt)
        
        return variance

def simulate_execution(model: AlmgrenChrissModel, trajectory_type='continuous', 
                       price_process='realistic'):
    """
    Simulate actual execution following optimal trajectory
    
    Includes realistic price dynamics and market impact
    """
    np.random.seed(42)
    
    # Get optimal trajectory
    if trajectory_type == 'continuous':
        times, inventory, trades = model.optimal_trajectory_continuous()
    else:
        times, inventory, trades = model.optimal_trajectory_discrete()
    
    X = model.execution.X
    N = model.execution.N
    T = model.execution.T
    dt = model.dt
    
    sigma = model.market.sigma / np.sqrt(252)  # Daily volatility
    gamma = model.market.gamma
    eta = model.market.eta
    spread = model.market.spread
    
    # Simulate price path
    S0 = 100.0
    prices = [S0]
    mid_prices = [S0]  # Mid-price without our impact
    
    # Execution record
    execution_log = []
    cumulative_volume = 0
    
    for i in range(N):
        # Natural price evolution (without our trades)
        drift = 0  # Assume no drift
        shock = sigma * np.sqrt(dt) * np.random.normal()
        natural_price = mid_prices[-1] + drift * dt + shock * mid_prices[-1]
        
        # Our trade
        v = trades[i]
        
        # Permanent impact (moves mid-price)
        permanent_impact = gamma * v
        
        # Temporary impact (we pay more/receive less)
        temporary_impact = eta * v
        
        # Execution price (for seller: mid - permanent - temporary - spread/2)
        # Assume we're selling
        execution_price = natural_price - permanent_impact - temporary_impact - spread/2
        
        # Mid-price after permanent impact
        new_mid_price = natural_price - permanent_impact
        
        mid_prices.append(new_mid_price)
        prices.append(execution_price)
        
        cumulative_volume += v
        
        execution_log.append({
            'time': times[i],
            'trade_size': v,
            'cumulative_volume': cumulative_volume,
            'mid_price': natural_price,
            'execution_price': execution_price,
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact,
            'inventory_remaining': inventory[i+1]
        })
    
    df_execution = pd.DataFrame(execution_log)
    
    # Calculate realized metrics
    arrival_price = S0
    avg_execution_price = np.average(df_execution['execution_price'], 
                                      weights=df_execution['trade_size'])
    
    implementation_shortfall = (arrival_price - avg_execution_price) * X
    
    return df_execution, {
        'arrival_price': arrival_price,
        'avg_execution_price': avg_execution_price,
        'implementation_shortfall': implementation_shortfall,
        'total_volume': cumulative_volume,
        'mid_prices': mid_prices,
        'execution_prices': prices
    }

# Example 1: Baseline execution
print("="*80)
print("ALMGREN-CHRISS OPTIMAL EXECUTION MODEL")
print("="*80)

# Market parameters
market = MarketParams(
    sigma=0.30,      # 30% annualized volatility
    gamma=0.001,     # Permanent impact: 0.1% per 1000 shares
    eta=0.01,        # Temporary impact: 1% per 1000 shares
    spread=0.01      # 1% bid-ask spread
)

# Execution problem: liquidate 10,000 shares over 1 day
execution = ExecutionParams(
    X=10000,
    T=1.0,        # 1 day
    lam=1e-6,     # Risk aversion
    N=20          # 20 time intervals
)

# Create model
model = AlmgrenChrissModel(market, execution)

print(f"\nModel Parameters:")
print(f"  Total shares: {execution.X:,}")
print(f"  Time horizon: {execution.T} days")
print(f"  Volatility: {market.sigma*100:.1f}%")
print(f"  Permanent impact: {market.gamma*100:.3f}% per share")
print(f"  Temporary impact: {market.eta*100:.2f}% per share")
print(f"  Risk aversion: {execution.lam:.2e}")
print(f"  Urgency parameter κ: {model.kappa:.4f}")

# Compute optimal trajectories
times_cont, inv_cont, trades_cont = model.optimal_trajectory_continuous()
times_disc, inv_disc, trades_disc = model.optimal_trajectory_discrete()

print(f"\nOptimal Execution Schedule (Continuous):")
print(f"  First trade: {trades_cont[0]:.0f} shares ({trades_cont[0]/execution.X*100:.1f}%)")
print(f"  Last trade: {trades_cont[-1]:.0f} shares ({trades_cont[-1]/execution.X*100:.1f}%)")
print(f"  Average trade: {np.mean(trades_cont):.0f} shares")

# Expected costs
costs_cont = model.expected_cost(trades_cont)
risk_cont = model.risk_measure(inv_cont)

print(f"\nExpected Costs (Continuous Trajectory):")
print(f"  Permanent impact: ${costs_cont['permanent']:,.2f}")
print(f"  Temporary impact: ${costs_cont['temporary']:,.2f}")
print(f"  Spread cost: ${costs_cont['spread']:,.2f}")
print(f"  Total expected cost: ${costs_cont['total']:,.2f}")
print(f"  Timing risk (variance): {risk_cont:.4f}")
print(f"  Risk-adjusted cost: ${costs_cont['total'] + execution.lam * risk_cont:,.2f}")

# Simulate execution
df_exec, metrics = simulate_execution(model, trajectory_type='discrete', price_process='realistic')

print(f"\nRealized Execution Metrics:")
print(f"  Arrival price: ${metrics['arrival_price']:.2f}")
print(f"  Average execution price: ${metrics['avg_execution_price']:.2f}")
print(f"  Implementation shortfall: ${metrics['implementation_shortfall']:,.2f}")
print(f"  Shortfall (bps): {metrics['implementation_shortfall']/(metrics['arrival_price']*execution.X)*10000:.1f} bps")

# Compare different risk aversion levels
print("\n" + "="*80)
print("RISK AVERSION SENSITIVITY ANALYSIS")
print("="*80)

risk_aversions = [1e-7, 1e-6, 1e-5, 1e-4]
results = []

for lam in risk_aversions:
    exec_params = ExecutionParams(X=10000, T=1.0, lam=lam, N=20)
    model_temp = AlmgrenChrissModel(market, exec_params)
    
    times, inv, trades = model_temp.optimal_trajectory_discrete()
    costs = model_temp.expected_cost(trades)
    risk = model_temp.risk_measure(inv)
    
    results.append({
        'lambda': lam,
        'kappa': model_temp.kappa,
        'first_trade_pct': trades[0] / execution.X * 100,
        'expected_cost': costs['total'],
        'risk_variance': risk,
        'risk_adjusted_cost': costs['total'] + lam * risk
    })

df_sensitivity = pd.DataFrame(results)

print("\nRisk Aversion Impact:")
print(df_sensitivity.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Optimal inventory trajectory
axes[0, 0].plot(times_cont, inv_cont, 'b-', label='Continuous', linewidth=2)
axes[0, 0].plot(times_disc, inv_disc, 'r--', label='Discrete', linewidth=2)
axes[0, 0].set_title('Optimal Inventory Trajectory')
axes[0, 0].set_xlabel('Time (days)')
axes[0, 0].set_ylabel('Shares Remaining')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Optimal trade schedule
axes[0, 1].bar(times_cont[:-1], trades_cont, width=model.dt*0.8, alpha=0.7)
axes[0, 1].set_title('Optimal Trade Schedule')
axes[0, 1].set_xlabel('Time (days)')
axes[0, 1].set_ylabel('Trade Size (shares)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Risk aversion sensitivity
axes[0, 2].plot(df_sensitivity['lambda'], df_sensitivity['first_trade_pct'], 
                marker='o', linewidth=2)
axes[0, 2].set_xscale('log')
axes[0, 2].set_title('Risk Aversion vs Urgency')
axes[0, 2].set_xlabel('Risk Aversion λ (log scale)')
axes[0, 2].set_ylabel('First Trade (%)')
axes[0, 2].grid(alpha=0.3)

# Plot 4: Simulated execution
axes[1, 0].plot(df_exec['time'], df_exec['mid_price'], 'b-', 
                label='Mid-price', linewidth=2)
axes[1, 0].scatter(df_exec['time'], df_exec['execution_price'], 
                   c='red', s=50, label='Execution', zorder=5)
axes[1, 0].axhline(metrics['arrival_price'], color='green', linestyle='--', 
                   label='Arrival price')
axes[1, 0].set_title('Price Path During Execution')
axes[1, 0].set_xlabel('Time (days)')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Market impact decomposition
impact_data = df_exec[['permanent_impact', 'temporary_impact']].values
axes[1, 1].bar(df_exec['time'], df_exec['permanent_impact'], 
               width=model.dt*0.8, label='Permanent', alpha=0.7)
axes[1, 1].bar(df_exec['time'], df_exec['temporary_impact'], 
               width=model.dt*0.8, label='Temporary', alpha=0.7, bottom=df_exec['permanent_impact'])
axes[1, 1].set_title('Market Impact Decomposition')
axes[1, 1].set_xlabel('Time (days)')
axes[1, 1].set_ylabel('Impact ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Cost vs risk tradeoff
axes[1, 2].scatter(df_sensitivity['risk_variance'], df_sensitivity['expected_cost'], 
                   s=200, c=np.log10(df_sensitivity['lambda']), cmap='viridis', alpha=0.7)
for idx, row in df_sensitivity.iterrows():
    axes[1, 2].annotate(f"λ={row['lambda']:.0e}", 
                        (row['risk_variance'], row['expected_cost']), 
                        fontsize=8, ha='left')
axes[1, 2].set_title('Cost vs Risk Tradeoff')
axes[1, 2].set_xlabel('Timing Risk (Variance)')
axes[1, 2].set_ylabel('Expected Cost ($)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Optimal trajectory front-loads execution (first trade largest)")
print(f"2. Higher risk aversion → faster execution → lower timing risk")
print(f"3. Higher temporary impact → slower execution → lower transaction costs")
print(f"4. Trade-off quantified by urgency parameter κ = √(λσ²/η)")
print(f"5. Implementation shortfall captures realized vs arrival price cost")
print(f"6. Model provides benchmark for evaluating execution performance")
