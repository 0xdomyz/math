
# Block 1
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