
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes components
def bs_d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return d1 - sigma*np.sqrt(T)

def bs_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def bs_put(S, K, T, r, sigma):
    call = bs_call(S, K, T, r, sigma)
    put = call - S + K*np.exp(-r*T)
    return put

def delta_call(S, K, T, r, sigma):
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def delta_put(S, K, T, r, sigma):
    return delta_call(S, K, T, r, sigma) - 1

# Numerical delta (finite difference)
def delta_numerical(S, K, T, r, sigma, option_type='call', eps=0.01):
    if option_type == 'call':
        V_up = bs_call(S + eps, K, T, r, sigma)
        V_down = bs_call(S - eps, K, T, r, sigma)
    else:
        V_up = bs_put(S + eps, K, T, r, sigma)
        V_down = bs_put(S - eps, K, T, r, sigma)
    return (V_up - V_down) / (2*eps)

# Parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

# Compute delta across spot prices
spot_prices = np.linspace(50, 150, 100)
delta_calls = [delta_call(S, K, T, r, sigma) for S in spot_prices]
delta_puts = [delta_put(S, K, T, r, sigma) for S in spot_prices]
delta_calls_num = [delta_numerical(S, K, T, r, sigma, 'call') for S in spot_prices]

# Compute option prices
call_prices = [bs_call(S, K, T, r, sigma) for S in spot_prices]
put_prices = [bs_put(S, K, T, r, sigma) for S in spot_prices]

# Delta hedging simulation
print("=== DELTA HEDGING SIMULATION ===")
np.random.seed(42)
S_path = np.array([100])
spot_moves = np.random.normal(0.02, 0.02, 252)  # Daily returns
for move in spot_moves:
    S_path = np.append(S_path, S_path[-1] * (1 + move))

T_remaining = np.linspace(T, 0, len(S_path))
deltas = []
call_values = []
hedge_pnl = []
call_pnl = []
cumulative_hedge_cost = 0
option_position = 1  # Long 1 call

for i, (S, T_remain) in enumerate(zip(S_path, T_remaining)):
    if T_remain > 0:
        delta = delta_call(S, K, T_remain, r, sigma)
        call_value = bs_call(S, K, T_remain, r, sigma)
    else:
        delta = max(1, 0) if S > K else 0
        call_value = max(S - K, 0)
    
    deltas.append(delta)
    call_values.append(call_value)
    
    # Hedge: short delta shares
    if i > 0:
        # P&L from hedging position (short delta shares)
        hedge_pnl_daily = -deltas[i-1] * (S_path[i] - S_path[i-1])
        hedge_pnl.append(hedge_pnl_daily)
        
        # P&L from call option
        call_pnl_daily = call_values[i] - call_values[i-1]
        call_pnl.append(call_pnl_daily)
        
        cumulative_hedge_cost += hedge_pnl_daily

print(f"Call option value at expiry: ${call_values[-1]:.2f}")
print(f"Cumulative P&L from hedging: ${sum(hedge_pnl):.2f}")
print(f"Total P&L (call - hedge): ${call_values[-1] - call_values[0] + sum(hedge_pnl):.2f}")
print(f"Final spot price: ${S_path[-1]:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Delta vs Spot
axes[0, 0].plot(spot_prices, delta_calls, linewidth=2, label='Call Delta (analytical)')
axes[0, 0].plot(spot_prices, delta_calls_num, 'o-', alpha=0.3, markersize=3, label='Call Delta (numerical)')
axes[0, 0].plot(spot_prices, delta_puts, linewidth=2, label='Put Delta')
axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[0, 0].axhline(0.5, color='r', linestyle=':', alpha=0.5, label='ATM (0.5)')
axes[0, 0].axvline(K, color='r', linestyle=':', alpha=0.5)
axes[0, 0].set_xlabel('Spot Price ($)')
axes[0, 0].set_ylabel('Delta')
axes[0, 0].set_title('Delta across Spot Prices (T=1yr, K=100)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Option Price vs Spot
axes[0, 1].plot(spot_prices, call_prices, linewidth=2, label='Call Value')
axes[0, 1].plot(spot_prices, put_prices, linewidth=2, label='Put Value')
axes[0, 1].axvline(K, color='r', linestyle='--', alpha=0.5, label='Strike')
axes[0, 1].set_xlabel('Spot Price ($)')
axes[0, 1].set_ylabel('Option Value ($)')
axes[0, 1].set_title('Option Value vs Spot Price')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spot path and cumulative delta hedge
axes[1, 0].plot(S_path, linewidth=2, label='Spot Price')
axes[1, 0].fill_between(range(len(S_path)), S_path, alpha=0.2)
ax_twin = axes[1, 0].twinx()
ax_twin.plot(deltas, color='orange', linewidth=2, label='Delta', alpha=0.7)
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Spot Price ($)', color='C0')
ax_twin.set_ylabel('Delta', color='orange')
axes[1, 0].set_title('Spot Path and Delta Over Time')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend(loc='upper left')
ax_twin.legend(loc='upper right')

# Plot 4: Cumulative P&L
cumulative_call_pnl = np.cumsum(call_pnl)
cumulative_hedge_pnl = np.cumsum(hedge_pnl)
total_pnl = cumulative_call_pnl + cumulative_hedge_pnl

axes[1, 1].plot(cumulative_call_pnl, label='Call P&L', linewidth=2)
axes[1, 1].plot(cumulative_hedge_pnl, label='Hedge P&L', linewidth=2)
axes[1, 1].plot(total_pnl, label='Total (call + hedge)', linewidth=2, linestyle='--')
axes[1, 1].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Cumulative P&L ($)')
axes[1, 1].set_title('Delta Hedging P&L')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nDelta hedging isolation: Total P&L standard error ~ Gamma × (ΔS)² / 2")