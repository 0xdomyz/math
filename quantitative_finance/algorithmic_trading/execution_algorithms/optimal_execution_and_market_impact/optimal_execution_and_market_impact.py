# Simplified market impact + opportunity cost optimization
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def total_execution_cost(N, X, sigma, gamma_temp, gamma_perm, lambda_risk):
    """
    Total cost as function of # execution slices N
    N: number of slices
    X: total order size
    sigma: volatility
    gamma_temp: temporary impact coefficient
    gamma_perm: permanent impact coefficient
    lambda_risk: risk aversion parameter
    """
    dt = 1 / N  # time interval
    
    # Market impact cost: linear approximation
    v = X / N  # size per slice
    permanent_impact = gamma_perm * (X / 2)  # average cumulative impact
    temporary_impact = gamma_temp * v * N  # sum of all slices
    market_impact_cost = permanent_impact + temporary_impact
    
    # Opportunity cost: variance of residual risk
    # Approx: variance accumulates quadratically in time
    opportunity_cost_var = 0.5 * sigma**2 * (1/N)  # simplified
    opportunity_cost = lambda_risk * opportunity_cost_var
    
    # Total
    total_cost = market_impact_cost + opportunity_cost
    return total_cost

# Parameters
X = 100000  # total order size
sigma = 0.0200  # 2% volatility
gamma_temp = 0.0001  # temporary impact (per share)
gamma_perm = 0.00005  # permanent impact
lambda_risk = 1000  # risk aversion

N_range = np.arange(1, 1001)  # 1 to 1000 slices
costs = [total_execution_cost(N, X, sigma, gamma_temp, gamma_perm, lambda_risk) 
         for N in N_range]

# Find optimal
N_optimal = N_range[np.argmin(costs)]
cost_optimal = min(costs)

print("="*60)
print("OPTIMAL EXECUTION ANALYSIS")
print("="*60)
print(f"Order Size: {X:,} shares")
print(f"Volatility: {sigma*100:.1f}%")
print(f"Optimal # Slices: {N_optimal}")
print(f"Optimal Time per Slice: {1/N_optimal * 252 * 6.5:.1f} seconds (assuming 1 = 1 day/252 of 6.5 hrs)")
print(f"Minimum Total Cost: ${cost_optimal:.2f} ({cost_optimal/X*10000:.2f} bps)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Cost components
market_impact_costs = []
opportunity_costs = []

for N in N_range:
    v = X / N
    permanent_impact = gamma_perm * (X / 2)
    temporary_impact = gamma_temp * v * N
    mi_cost = permanent_impact + temporary_impact
    market_impact_costs.append(mi_cost)
    
    opp_var = 0.5 * sigma**2 * (1/N)
    opp_cost = lambda_risk * opp_var
    opportunity_costs.append(opp_cost)

ax = axes[0]
ax.plot(N_range, np.array(market_impact_costs)/X*10000, 'b-', linewidth=2, 
       label='Market Impact Cost (bps)')
ax.plot(N_range, np.array(opportunity_costs)/X*10000, 'r-', linewidth=2, 
       label='Opportunity Cost (bps)')
ax.plot(N_range, np.array(costs)/X*10000, 'g-', linewidth=2.5, 
       label='Total Cost (bps)')
ax.axvline(x=N_optimal, color='purple', linestyle='--', linewidth=1.5, 
          label=f'Optimal N={N_optimal}')
ax.set_xlabel('Number of Slices (N)')
ax.set_ylabel('Cost (bps)')
ax.set_title('Execution Cost Components')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)

# Plot 2: Total cost vs execution pace
ax = axes[1]
ax.plot(N_range, np.array(costs)/X*10000, 'b-', linewidth=2.5)
ax.scatter([N_optimal], [cost_optimal/X*10000], color='red', s=200, zorder=5, 
          label=f'Optimal (N={N_optimal})')
ax.axhline(y=cost_optimal/X*10000, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Slices (N) - Execution Speed')
ax.set_ylabel('Total Cost (bps)')
ax.set_title('Total Execution Cost Curve (U-Shaped)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)

plt.tight_layout()
plt.show()