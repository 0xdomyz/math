# Hedging Strategy

## 1. Concept Skeleton
**Definition:** Dynamic portfolio rebalancing strategy using Greeks to neutralize directional and other risks; maintaining delta-neutral (or target delta) position while monitoring higher-order Greeks  
**Purpose:** Isolate specific risks; reduce or eliminate exposure to spot, volatility, time decay; profit from realized vs. implied volatility differential  
**Prerequisites:** Greeks (Delta, Gamma, Vega, Theta), portfolio management, rebalancing mechanics, transaction costs

## 2. Comparative Framing
| Strategy | Delta-Neutral | Gamma-Neutral | Vega-Neutral | Theta-Neutral |
|----------|---------------|---------------|-------------|--------------|
| **What's Hedged** | Directional spot | Convexity (non-linearity) | Volatility changes | Time decay |
| **Typical Position** | Long option + short delta shares | Long + short gamma instruments | Long + short vega | Long + short theta |
| **Profit Source** | Theta decay (if short vega) | Realized vol > implied | Volatility contraction | Daily decay collection |
| **Loss Source** | Vega exposure | Realized vol < implied | Realized vol spike | Gamma cost |
| **Rehedging Frequency** | Daily/continuous | Continuous | Daily/weekly | Triggered |
| **Transaction Costs** | Moderate | High | Low | Low |

## 3. Examples + Counterexamples

**Simple Example:**  
Long 1 call with Δ = 0.6: Short 0.6 shares to delta-hedge; spot move ↑$1 → call ↑$0.60 - hedge loss $0.60 = $0 (approximately)

**Practical Case:**  
Market maker: Sells volatility (short straddle, negative vega); delta-hedges continuously; profits if realized vol < implied vol; loses if volatility spikes

**Gamma Trap:**  
Delta-hedged long call in falling spot market: Rehedge repeatedly at higher spot levels (buy high, sell low); gamma loss amplifies; total loss = realized vol gain - theta decay

**Calendar Spread Hedging:**  
Buy long-dated call, sell short-dated call: Delta-hedge the short call; long calendar theta; expires in 3mo; roll forward to repeat

## 4. Layer Breakdown
```
Hedging Strategy Framework:
├─ Objective Definition:
│   ├─ Primary risk to eliminate: Delta (directional), Vega (volatility), Gamma (convexity)
│   ├─ Secondary risks: Theta (time decay), higher-order Greeks
│   ├─ Constraints:
│   │   ├─ Budget: Cash for hedges
│   │   ├─ Liquidity: Available instruments
│   │   ├─ Costs: Transaction fees, bid-ask spreads
│   │   └─ Regulatory: Capital requirements, position limits
├─ Delta Hedging (Most Common):
│   ├─ Setup:
│   │   ├─ Initial: Long option, short Δ × N shares (N = option contracts)
│   │   ├─ Δ = ∂V/∂S; negative for puts, positive for calls
│   │   └─ Result: Portfolio ≈ market-neutral (directional insensitive)
│   ├─ Rebalancing:
│   │   ├─ Frequency: Continuous (ideal) vs. discrete (practical)
│   │   ├─ Trigger: Fixed delta threshold (e.g., |Δ| > 0.05) or time interval
│   │   ├─ Rebalance cost: Proportional to γ × (ΔS)² (gamma P&L from spot move)
│   │   └─ Optimal frequency: Balances hedging cost vs. drift risk
│   ├─ P&L Analysis:
│   │   ├─ Spot P&L: ~0 (delta-hedged)
│   │   ├─ Vega P&L: Σ νᵢ × Δσ (volatility exposure remains)
│   │   ├─ Theta P&L: Σ θᵢ / 365 × days (daily time decay)
│   │   ├─ Gamma P&L: Σ γᵢ / 2 × (ΔSᵢ)² (convexity profit/loss)
│   │   └─ Total: θ + γ/2 × (realized vol)² - ν × (implied vol changes)
├─ Vega Hedging:
│   ├─ Goal: Eliminate volatility exposure
│   ├─ Instruments:
│   │   ├─ Opposite-signed vega options (calls/puts)
│   │   ├─ Variance swaps (direct vol hedging)
│   │   ├─ VIX futures (implied vol index)
│   │   └─ Volatility spreads (calendar, diagonal)
│   ├─ Implementation:
│   │   ├─ Compute portfolio vega: Σ νᵢ × Qᵢ (Greeks × quantities)
│   │   ├─ Hedge instrument vega: νₕ
│   │   ├─ Hedge ratio: |Vega_portfolio| / νₕ
│   │   ├─ Short/Long: Opposite sign to portfolio vega
│   │   └─ Monitor: Rebalance if vega drifts beyond threshold
├─ Gamma Management:
│   ├─ Strategy Choices:
│   │   ├─ Long gamma: Long options; profit from moves; pay vega/theta
│   │   ├─ Short gamma: Sell options; collect premium; lose on large moves
│   │   └─ Gamma-neutral: Use spreads (long near-dated, short far-dated)
│   ├─ Risk Profile:
│   │   ├─ Delta-hedged gamma position: P&L = θ + γ/2 × (ΔS)²
│   │   ├─ Long gamma expected profit: ≈ γ / 2 × (realized vol)² × T (annualized)
│   │   ├─ Breakeven vol: Where gamma P&L = theta decay cost
│   │   └─ Exposure: Daily gamma × spot move² → cumulative P&L
├─ Theta Harvesting:
│   ├─ Strategy:
│   │   ├─ Sell short-dated options (high theta decay)
│   │   ├─ Delta-hedge to neutralize spot exposure
│   │   ├─ Collect daily theta as time passes
│   │   ├─ Roll position forward weekly/monthly
│   │   └─ Repeat to scale portfolio
│   ├─ Dynamics:
│   │   ├─ Daily P&L: +θ/365 (decay benefit) - γ/2 × (ΔS)² (if moves occur)
│   │   ├─ Breakeven: Realized vol must be < implied vol (theta benefit > gamma loss)
│   │   ├─ Margin requirement: theta strategies need buffer for adverse moves
│   │   └─ Gamma drag: High gamma near expiry accelerates cost
├─ Practical Considerations:
│   ├─ Costs:
│   │   ├─ Bid-ask spread: Entry + exit cost per rehedge
│   │   ├─ Commissions: Fixed or proportional
│   │   ├─ Slippage: Execution risk; actual price vs. quoted
│   │   └─ Market impact: Large hedges move prices
│   ├─ Frequency Optimization:
│   │   ├─ Daily: Standard (most liquid times)
│   │   ├─ Weekly: Lower cost; more drift risk
│   │   ├─ On-demand: Triggered by threshold breach
│   │   └─ Optimal: Minimizes hedging cost + drift cost
│   ├─ Discretionary Overrides:
│   │   ├─ Vol forecasts: If expecting vol spike, reduce hedging
│   │   ├─ Spot forecasts: If expecting directional move, adjust delta target
│   │   ├─ Liquidity: Reduce rehedging if market stress
│   │   └─ Risk limits: Never exceed position limits despite Greeks
└─ Greeks Portfolio View:
    ├─ Position Greeks: Sum of all Greeks across book
    ├─ Greek limits: Max allowed delta, gamma, vega, theta
    ├─ Rebalancing: Coordinate across all positions
    ├─ Monitoring dashboard: Real-time Greeks with thresholds
    └─ Risk committee: Escalate if Greeks breach limits
```

**Interaction:** Compute portfolio Greeks → identify dominant risks → execute hedges → rebalance on schedule/threshold → repeat

## 5. Mini-Project
Implement delta-hedging strategy with gamma-theta tradeoff analysis:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes components
def bs_d1(S, K, T, r, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return d1 - sigma*np.sqrt(T)

def bs_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def delta_bs(S, K, T, r, sigma):
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def gamma_bs(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def theta_bs(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) - 
            r * K * np.exp(-r*T) * norm.cdf(d2)) / 365  # Per day

# Parameters
S0, K, T0, r, sigma_implied = 100, 100, 1, 0.05, 0.20

# Scenario: Two realized volatility paths
print("=== DELTA HEDGING SIMULATION ===\n")

# Scenario 1: High realized volatility
np.random.seed(42)
realized_vol_high = 0.30
daily_returns_high = np.random.normal(0, realized_vol_high/np.sqrt(252), 252)
S_path_high = np.array([S0])
for ret in daily_returns_high:
    S_path_high = np.append(S_path_high, S_path_high[-1] * (1 + ret))

# Scenario 2: Low realized volatility
realized_vol_low = 0.10
daily_returns_low = np.random.normal(0, realized_vol_low/np.sqrt(252), 252)
S_path_low = np.array([S0])
for ret in daily_returns_low:
    S_path_low = np.append(S_path_low, S_path_low[-1] * (1 + ret))

# Delta hedging simulation function
def simulate_delta_hedging(S_path, K, T0, r, sigma_implied, rehedge_frequency=1):
    """
    Simulate delta-hedged long call position
    rehedge_frequency: days between rehedges
    """
    T_remaining = np.linspace(T0, 0.001, len(S_path))
    
    # Initial position
    delta_initial = delta_bs(S0, K, T0, r, sigma_implied)
    call_price_initial = bs_call(S0, K, T0, r, sigma_implied)
    
    # Tracking arrays
    deltas = []
    gammas = []
    thetas = []
    call_values = []
    hedge_shares = []
    cumulative_gamma_pnl = 0
    cumulative_theta_pnl = 0
    cumulative_rehedge_cost = 0
    total_pnls = []
    
    for i, (S, T_rem) in enumerate(zip(S_path, T_remaining)):
        if T_rem > 0:
            delta = delta_bs(S, K, T_rem, r, sigma_implied)
            gamma = gamma_bs(S, K, T_rem, r, sigma_implied)
            theta = theta_bs(S, K, T_rem, r, sigma_implied)
            call_value = bs_call(S, K, T_rem, r, sigma_implied)
        else:
            delta = 1.0 if S > K else 0.0
            gamma = 0.0
            theta = 0.0
            call_value = max(S - K, 0)
        
        deltas.append(delta)
        gammas.append(gamma)
        thetas.append(theta)
        call_values.append(call_value)
        
        # Rehedge logic
        if i % rehedge_frequency == 0:
            hedge_shares.append(delta)
            if i > 0:
                # Rehedge cost: buy/sell shares at market
                share_price_old = S_path[i-1]
                share_price_new = S
                rehedge_cost = (delta - hedge_shares[-2]) * share_price_old
                cumulative_rehedge_cost += rehedge_cost
        else:
            hedge_shares.append(hedge_shares[-1])
        
        # P&L components
        if i > 0:
            dS = S - S_path[i-1]
            
            # Gamma P&L: benefit from |moves|, cost from hedging
            gamma_pnl = gammas[i-1] / 2 * dS**2
            cumulative_gamma_pnl += gamma_pnl
            
            # Theta P&L: daily decay benefit
            theta_pnl = thetas[i-1]
            cumulative_theta_pnl += theta_pnl
            
            # Call + Hedge P&L
            call_pnl = call_values[i] - call_values[i-1]
            hedge_pnl = -hedge_shares[i-1] * dS
            total_pnl = call_pnl + hedge_pnl
            
            total_pnls.append(total_pnl)
        else:
            total_pnls.append(0)
    
    return {
        'deltas': deltas,
        'gammas': gammas,
        'thetas': thetas,
        'call_values': call_values,
        'cumulative_gamma_pnl': cumulative_gamma_pnl,
        'cumulative_theta_pnl': cumulative_theta_pnl,
        'cumulative_rehedge_cost': cumulative_rehedge_cost,
        'total_pnls': np.cumsum(total_pnls),
        'final_call_value': call_values[-1],
        'final_total_pnl': sum(total_pnls)
    }

# Run scenarios
result_high = simulate_delta_hedging(S_path_high, K, T0, r, sigma_implied, rehedge_frequency=1)
result_low = simulate_delta_hedging(S_path_low, K, T0, r, sigma_implied, rehedge_frequency=1)

print("SCENARIO 1: High Realized Volatility ({:.1%})".format(realized_vol_high))
print(f"  Final spot: ${S_path_high[-1]:.2f}")
print(f"  Call value at expiry: ${result_high['final_call_value']:.2f}")
print(f"  Gamma P&L: ${result_high['cumulative_gamma_pnl']:.2f}")
print(f"  Theta P&L: ${result_high['cumulative_theta_pnl']:.2f}")
print(f"  Rehedge cost: ${result_high['cumulative_rehedge_cost']:.2f}")
print(f"  Total P&L: ${result_high['final_total_pnl']:.2f}")

print(f"\nSCENARIO 2: Low Realized Volatility ({:.1%})".format(realized_vol_low))
print(f"  Final spot: ${S_path_low[-1]:.2f}")
print(f"  Call value at expiry: ${result_low['final_call_value']:.2f}")
print(f"  Gamma P&L: ${result_low['cumulative_gamma_pnl']:.2f}")
print(f"  Theta P&L: ${result_low['cumulative_theta_pnl']:.2f}")
print(f"  Rehedge cost: ${result_low['cumulative_rehedge_cost']:.2f}")
print(f"  Total P&L: ${result_low['final_total_pnl']:.2f}")

print(f"\nBREAKEVEN ANALYSIS:")
implied_vol = sigma_implied
print(f"Implied vol: {implied_vol:.1%}")
print(f"High real vol: {realized_vol_high:.1%} → P&L positive if gamma > theta decay")
print(f"Low real vol: {realized_vol_low:.1%} → P&L positive from theta decay")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Spot paths
axes[0, 0].plot(S_path_high, label='High Vol Path', linewidth=2, alpha=0.7)
axes[0, 0].plot(S_path_low, label='Low Vol Path', linewidth=2, alpha=0.7)
axes[0, 0].axhline(K, color='r', linestyle='--', alpha=0.5, label='Strike')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Spot Price ($)')
axes[0, 0].set_title('Spot Price Paths')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Delta evolution
axes[0, 1].plot(result_high['deltas'], label='High Vol Delta', linewidth=1, alpha=0.7)
axes[0, 1].plot(result_low['deltas'], label='Low Vol Delta', linewidth=1, alpha=0.7)
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Delta')
axes[0, 1].set_title('Delta Evolution')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Gamma evolution
axes[0, 2].plot(result_high['gammas'], label='High Vol', linewidth=1, alpha=0.7)
axes[0, 2].plot(result_low['gammas'], label='Low Vol', linewidth=1, alpha=0.7)
axes[0, 2].set_xlabel('Day')
axes[0, 2].set_ylabel('Gamma')
axes[0, 2].set_title('Gamma Evolution')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Cumulative P&L
days = np.arange(len(result_high['total_pnls']))
axes[1, 0].plot(days, result_high['total_pnls'], label='High Vol', linewidth=2)
axes[1, 0].plot(days, result_low['total_pnls'], label='Low Vol', linewidth=2)
axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Cumulative P&L ($)')
axes[1, 0].set_title('Delta-Hedged P&L')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: P&L components comparison
categories = ['Gamma', 'Theta', 'Rehedge Cost', 'Total']
high_values = [result_high['cumulative_gamma_pnl'], 
               result_high['cumulative_theta_pnl'],
               result_high['cumulative_rehedge_cost'],
               result_high['final_total_pnl']]
low_values = [result_low['cumulative_gamma_pnl'],
              result_low['cumulative_theta_pnl'],
              result_low['cumulative_rehedge_cost'],
              result_low['final_total_pnl']]

x = np.arange(len(categories))
width = 0.35

axes[1, 1].bar(x - width/2, high_values, width, label='High Vol', alpha=0.7)
axes[1, 1].bar(x + width/2, low_values, width, label='Low Vol', alpha=0.7)
axes[1, 1].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].set_ylabel('P&L ($)')
axes[1, 1].set_title('P&L Components')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Realized vs Implied Volatility
print(f"\nActual realized vols: High = {np.std(np.log(S_path_high[1:]/S_path_high[:-1]))*np.sqrt(252):.1%}, Low = {np.std(np.log(S_path_low[1:]/S_path_low[:-1]))*np.sqrt(252):.1%}")

vols_range = np.linspace(0.05, 0.50, 100)
pnl_high_vs_vol = []
pnl_low_vs_vol = []

for vol in vols_range:
    # Estimate P&L if realized vol matches
    # P&L ≈ Vega × (realized vol - implied vol) + Gamma P&L
    vega_estimate = 0.4  # Approximate ATM vega
    
    pnl_high = result_high['cumulative_gamma_pnl'] - vega_estimate * 100 * (vol - sigma_implied)
    pnl_low = result_low['cumulative_gamma_pnl'] - vega_estimate * 100 * (vol - sigma_implied)
    
    pnl_high_vs_vol.append(pnl_high)
    pnl_low_vs_vol.append(pnl_low)

axes[1, 2].plot(vols_range, pnl_high_vs_vol, label='High Vol Scenario', linewidth=2)
axes[1, 2].plot(vols_range, pnl_low_vs_vol, label='Low Vol Scenario', linewidth=2)
axes[1, 2].axvline(sigma_implied, color='r', linestyle='--', alpha=0.5, label='Implied Vol')
axes[1, 2].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1, 2].set_xlabel('Realized Volatility')
axes[1, 2].set_ylabel('Estimated P&L ($)')
axes[1, 2].set_title('Delta-Hedged P&L vs Realized Vol')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does hedging fail?
- Model risk: Greeks assume BS model; reality has jumps, stochastic vol, correlations
- Execution delays: Can't hedge instantly; gap risk overnight or during market stress
- Liquidity crunch: Can't execute hedges due to illiquidity; forced to hold unhedged
- Correlation breakdowns: Assumed hedges become imperfect (e.g., basis risk)
- Tail events: Greeks linear approximations; massive moves break assumptions

## 7. Key References
- [Hull - Options, Futures & Derivatives (Chapters 19-20)](https://www-2.rotman.utoronto.ca/~hull)
- [Taleb - Dynamic Hedging (Complete)](https://www.paulwilmott.com)
- [Natenberg - Option Volatility & Pricing (Chapters 15-16)](https://www.amazon.com/Option-Volatility-Pricing-Advanced-Strategies/dp/1557784124)

---
**Status:** Core portfolio management technique | **Complements:** Greeks Framework, Risk Management, Portfolio Greeks
