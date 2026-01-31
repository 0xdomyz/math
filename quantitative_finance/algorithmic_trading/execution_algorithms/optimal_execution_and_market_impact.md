# Optimal Execution & Market Impact

## 1. Concept Skeleton
**Definition:** Framework for minimizing total transaction costs by balancing market impact (costs of urgency) against opportunity cost (risk of delay); solves optimal trade-off between speed and cost  
**Purpose:** Theoretically determine optimal execution schedule; minimize price impact from large orders; quantify timing vs market impact tradeoff  
**Prerequisites:** Market impact models, stochastic control, transaction costs, Almgren-Chriss theory, optimization algorithms

## 2. Comparative Framing
| Approach | Patient | Moderate | Urgent | Emergency |
|----------|---------|----------|--------|-----------|
| **Execution Horizon** | Full day+ | Several hours | Minutes | Seconds |
| **Market Impact** | Low (gradual) | Moderate | High (visible) | Extreme (market order) |
| **Opportunity Cost** | High (price moves) | Moderate | Low (quick) | None (instant) |
| **Total Cost** | U-shaped optimal | Medium | Medium | High |
| **Benchmark** | VWAP, TWAP | Arrival price | Implementation shortfall | Market |
| **Algorithm** | Patient VWAP | POV, IS | TWAP + aggressive | Market order |

## 3. Examples + Counterexamples

**Simple Tradeoff:**  
Sell 100k shares, market=100,1% volatility. Patient execution over day: avg price $99.98 (miss upside 2bps from timing). Urgent 1-hour execution: price $99.90 (market impact 10bps). Patient better if σ small; urgent if stock hot.

**Optimal Curve:**  
Total cost = Market Impact + Opportunity Cost (convex function). Minimize at intermediate execution time (few hours typically). Patience (far left): opportunity cost dominates. Urgency (far right): market impact dominates.

**Liquidity Cascade:**  
100k order in deep pool (50k at bid). First 50k fills easily; next 50k: market impact explodes as liquidity depletes. Slower execution better (rebuild liquidity between fills).

**News Event:**  
Merger announce; need to reduce exposure immediately. Opportunity cost of waiting >> market impact of urgency. Market order optimal even at cost.

**Parameter Sensitivity:**  
Volatility σ ↑ → opportunity cost ↑ → execute faster. Spread γ ↑ → market impact ↑ → execute slower. Careful calibration critical.

## 4. Layer Breakdown
```
Optimal Execution Framework:

├─ Cost Components:
│  ├─ Market Impact Cost:
│  │   Temporary impact: Bid-ask spread, dealer inventory cost
│  │   Permanent impact: Information revelation, price adjustment
│  │   Linear model: Impact = α + β × (order size / market volume)
│  │   Power law: Impact ∝ (order size)^λ, λ ≈ 0.5-1.5
│  ├─ Opportunity Cost:
│  │   Risk of price movement: Volatility × Time × Position
│  │   Uncovered period risk: dP = σ dW √(dt)
│  │   Expected loss: E[|dP|] ≈ σ √(T / N) per slice
│  ├─ Timing Cost:
│  │   Delay in execution: Gap between decision and fill
│  │   Adverse selection: Information leakage penalty
│  │   Market inertia: Latency in price adjustment
│  ├─ Total Cost Function:
│  │   TC = Market_Impact_Cost(N, v) + Opportunity_Cost(T, σ, N)
│  │   TC = f(execution_pace, market_conditions, volatility)
│  └─ Minimization:
│      ∂TC/∂N = 0 ⟹ optimal pace N*
│      TC_min = minimum total transaction cost
├─ Almgren-Chriss Model (Canonical):
│  ├─ Setup:
│  │   Execute order X over time horizon T
│  │   Split into N slices at times 0, Δt, 2Δt, ..., T
│  │   Δt = T / N (equal time intervals)
│  ├─ Temporary Impact:
│  │   Price = Mid ± γ × (order_size)
│  │   Dealer spread capture: γ per share
│  │   Paid on each slice
│  ├─ Permanent Impact:
│  │   Price shift: ξ × cumulative_volume
│  │   Linear: price moves linearly with order flow
│  │   Persistent across time
│  ├─ Stochastic Execution:
│  │   Residual risk: Σᵢ ε̂ᵢ from unexecuted portion
│  │   Variance accumulates over execution window
│  ├─ Objective:
│  │   Minimize: E[TC] = Market_impact + λ × Var[residual]
│  │   λ = risk aversion parameter (rate per unit variance)
│  ├─ Solutions:
│  │   Linear execution: x̂ᵢ ∝ linear ramp
│  │   Exponential execution: x̂ᵢ ∝ exp(decay factor)
│  │   Optimal blend: depends on λ, T, σ, γ
│  └─ Key Result:
│      Aggressive strategy (fast): High variance, low market impact
│      Patient strategy (slow): Low variance, high market impact
│      Optimal balances both
├─ Market Impact Models:
│  ├─ Linear (Simplest):
│  │   Impact_i = α_temp × v_i + α_perm × Σⱼ≤ᵢ v_j
│  │   Easy to implement, may underestimate large orders
│  ├─ Concave (Realistic):
│  │   Impact ∝ √(v_i) temporary (liquidity absorbed)
│  │   Impact ∝ v_i permanent (info content)
│  ├─ Power Law:
│  │   Impact = c × (v_i / V_market)^λ
│  │   λ ≈ 0.5 empirically; economies of scale
│  ├─ Calibration:
│  │   Estimate from recent trades (size × price move)
│  │   Regress: ΔP = α + β × (ΔVolume / Total_Vol) + ε
│  │   Slope β ≈ permanent impact coefficient
│  └─ Seasonality & State-Dependence:
│      Market conditions affect impact (vol, spread, liquidity)
│      Time-of-day effects (high impact at open/close)
│      Stock-specific (liquid mega-cap vs illiquid micro-cap)
├─ Opportunity Cost Modeling:
│  ├─ Geometric Brownian Motion:
│  │   dP = μ dt + σ dW
│  │   P_T = P_0 × exp((μ-σ²/2)T + σ√T Z)
│  ├─ Potential Upside/Downside:
│  │   If selling: Fear of price drop → execute fast
│  │   If buying: Fear of price rise → execute fast
│  ├─ Expected Value Lost (Sell Side):
│  │   E[Loss] ≈ 0.5 × σ² × (Time_remaining)
│  │   Quadratic in time → strong incentive to finish
│  ├─ Risk-Adjusted Cost:
│  │   λ × Variance captures risk aversion
│  │   λ = utility of money / utility of return variance
│  │   Higher λ → lower risk tolerance → faster execution
│  └─ Multi-Period Formulation:
│      Variance = 0.5 × σ² × (Σᵢ (N-i) × Δt²)
│      Recursive optimization across time periods
├─ Execution Schedules:
│  ├─ Uniform (TWAP):
│  │   x̂ᵢ = X / N constant
│  │   Simplest, often suboptimal
│  ├─ Linear:
│  │   x̂ᵢ = X × i / (N × (N+1) / 2) ramp up
│  │   More aggressive early (front-load)
│  ├─ Exponential:
│  │   x̂ᵢ = X × (1 - e^(-κᵢ)) / (1 - e^(-κN))
│  │   Smooth transition; parameter κ controls pace
│  ├─ Derived Optimal:
│  │   From Almgren-Chriss: piecewise linear/exponential
│  │   Depends on risk aversion λ
│  │   λ→0: Fast (aggressive)
│  │   λ→∞: Slow (patient)
│  └─ Practical Variants:
│      Limited market impact: Cap allocation per time slice
│      POV adaptive: Match % of observed market volume
│      Event-triggered: React to volume/volatility spikes
├─ Parameter Estimation:
│  ├─ Volatility σ:
│  │   Historical (rolling window): σ_hist
│  │   Implied (from options): σ_impl
│  │   Intraday estimate: Higher frequency
│  ├─ Temporary Impact γ:
│  │   Bid-ask spread: γ ≈ spread / 2
│  │   Regress small trade price move on volume
│  ├─ Permanent Impact ξ:
│  │   5-minute price move vs trade volume
│  │   Filter for information events
│  ├─ Risk Aversion λ:
│  │   From firm's policy (target execution horizon)
│  │   Calibrate to historical typical execution profiles
│  └─ Market Volume V:
│      Exchange data (VWAP volume)
│      Estimate from time-of-day patterns
├─ Practical Considerations:
│  ├─ Partial Execution:
│  │   Actual fills may deviate from schedule
│  │   Rebalance remaining allocation dynamically
│  ├─ Multiple Venues:
│  │   Distribute across exchanges
│  │   Minimize signal to any single venue
│  ├─ Information Leakage:
│  │   Brokers, algos can deduce order size
│  │   Adaptive opponents may front-run
│  ├─ Market Conditions:
│  │   If volatility spikes → accelerate execution
│  │   If liquidity dries up → defer
│  └─ Regulatory:
│      Documentation of execution strategy
│      Fair execution requirements (Reg FD)
│      Audit trail retention
└─ Extensions:
   ├─ Multi-Asset Execution:
   │   Correlated orders across basket
   │   Coordination across venues
   ├─ Stochastic Volatility:
   │   Impact varies with vol regime
   │   Adapt execution pace
   ├─ Dynamic Programming:
   │   Backward induction for multi-period
   │   Hamilton-Jacobi-Bellman equation
   └─ Reinforcement Learning:
       Train agent to learn optimal policy
       Data-driven rather than model-based
```

**Interaction:** Market impact (cost of speed) ↔ Opportunity cost (cost of delay) → U-shaped total cost curve → optimal execution in middle.

## 5. Mini-Project (See VWAP file for Monte Carlo code demonstrating execution simulation)

```python
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
```

## 6. Challenge Round
- Derive Almgren-Chriss solution for linear execution schedule
- How does volatility affect optimal execution pace?
- Design execution strategy for gaps (market closed overnight)
- Estimate market impact coefficients from order book data
- Compare execution costs for different risk aversion levels

## 7. Key References
- [Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2001)](https://www.jstor.org/stable/2645747) — Foundational theory
- [Almgren, "Optimal Execution with Nonlinear Impact Functions" (2003)](https://www.jstor.org/stable/2692547) — Extensions
- [Konishi, "Optimal Slice of a Block Trade" (2002)](https://www.sciencedirect.com/science/article/pii/S0165410102000932)

---
**Status:** Theoretical foundation for execution | **Complements:** VWAP, TWAP, Market Microstructure, Transaction Costs
