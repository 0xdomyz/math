import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import pandas as pd
import warnings

# Block 1
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("OPTION PRICING BASICS")
print("="*60)

# Parameters
S = 100  # Current stock price
K = 100  # Strike price
r = 0.05  # Risk-free rate
T = 1.0  # Time to expiry (1 year)
sigma = 0.2  # Volatility

print(f"\nMarket Parameters:")
print(f"  Stock Price (S): ${S}")
print(f"  Strike Price (K): ${K}")
print(f"  Risk-free Rate (r): {r:.2%}")
print(f"  Time to Expiry (T): {T} year(s)")
print(f"  Volatility (σ): {sigma:.1%}")

# === Scenario 1: Black-Scholes Pricing ===
print("\n" + "="*60)
print("SCENARIO 1: European Option Pricing (Black-Scholes)")
print("="*60)

# Black-Scholes formulas
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

N_d1 = norm.cdf(d1)
N_d2 = norm.cdf(d2)
N_minus_d1 = norm.cdf(-d1)
N_minus_d2 = norm.cdf(-d2)

C_BS = S*N_d1 - K*np.exp(-r*T)*N_d2
P_BS = K*np.exp(-r*T)*N_minus_d2 - S*N_minus_d1

print(f"\nBlack-Scholes Values (European):")
print(f"  Call Value: ${C_BS:.2f}")
print(f"  Put Value: ${P_BS:.2f}")
print(f"  Put-Call Parity Check: C - P = {C_BS - P_BS:.4f}")
print(f"  Theoretical: S - Ke^(-rT) = {S - K*np.exp(-r*T):.4f}")

# Intrinsic and Time Values
intrinsic_call = max(S - K, 0)
intrinsic_put = max(K - S, 0)
time_value_call = C_BS - intrinsic_call
time_value_put = P_BS - intrinsic_put

print(f"\nValue Components:")
print(f"  Call: Intrinsic=${intrinsic_call:.2f} + Time Value=${time_value_call:.2f} = ${C_BS:.2f}")
print(f"  Put: Intrinsic=${intrinsic_put:.2f} + Time Value=${time_value_put:.2f} = ${P_BS:.2f}")

# === Scenario 2: No-Arbitrage Bounds ===
print("\n" + "="*60)
print("SCENARIO 2: No-Arbitrage Bounds")
print("="*60)

# European Call bounds
C_lower = max(S - K*np.exp(-r*T), 0)
C_upper = S

# European Put bounds
P_lower = max(K*np.exp(-r*T) - S, 0)
P_upper = K*np.exp(-r*T)

print(f"\nEuropean Call Bounds:")
print(f"  Lower: max(S - Ke^(-rT), 0) = ${C_lower:.2f}")
print(f"  Upper: S = ${C_upper:.2f}")
print(f"  BS Value: ${C_BS:.2f} (within bounds? {C_lower <= C_BS <= C_upper})")

print(f"\nEuropean Put Bounds:")
print(f"  Lower: max(Ke^(-rT) - S, 0) = ${P_lower:.2f}")
print(f"  Upper: Ke^(-rT) = ${P_upper:.2f}")
print(f"  BS Value: ${P_BS:.2f} (within bounds? {P_lower <= P_BS <= P_upper})")

# American lower bounds (with dividends considered as 0)
C_american_lower = max(S - K, 0)
P_american_lower = max(K - S, 0)

print(f"\nAmerican Option Lower Bounds (no dividends):")
print(f"  Call: max(S - K, 0) = ${C_american_lower:.2f}")
print(f"  Put: max(K - S, 0) = ${P_american_lower:.2f}")

# === Scenario 3: Greeks Sensitivity ===
print("\n" + "="*60)
print("SCENARIO 3: Greeks & Sensitivities")
print("="*60)

# Delta
delta_call = N_d1
delta_put = -N_minus_d1

# Gamma (common for both call and put)
gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Theta (per day)
theta_call = -S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*N_d2
theta_put = -S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*N_minus_d2
theta_call_daily = theta_call / 365
theta_put_daily = theta_put / 365

# Vega (per 1% change in volatility)
vega = S * norm.pdf(d1) * np.sqrt(T) / 100

# Rho (per 1% change in rate)
rho_call = K * T * np.exp(-r*T) * N_d2 / 100
rho_put = -K * T * np.exp(-r*T) * N_minus_d2 / 100

print(f"\nGreeks (European Options):")
print(f"  Delta Call: {delta_call:.4f}")
print(f"  Delta Put: {delta_put:.4f}")
print(f"  Gamma (both): {gamma:.6f}")
print(f"  Theta Call (daily): ${theta_call_daily:.4f}")
print(f"  Theta Put (daily): ${theta_put_daily:.4f}")
print(f"  Vega (per 1% vol change): ${vega:.4f}")
print(f"  Rho Call (per 1% rate change): ${rho_call:.4f}")
print(f"  Rho Put (per 1% rate change): ${rho_put:.4f}")

# === Scenario 4: Sensitivity Analysis ===
print("\n" + "="*60)
print("SCENARIO 4: Impact of Parameters on Option Value")
print("="*60)

# Impact of spot price
spot_range = np.linspace(80, 120, 41)
call_values = []
put_values = []

for s in spot_range:
    d1_temp = (np.log(s/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2_temp = d1_temp - sigma*np.sqrt(T)
    c = s*norm.cdf(d1_temp) - K*np.exp(-r*T)*norm.cdf(d2_temp)
    p = K*np.exp(-r*T)*norm.cdf(-d2_temp) - s*norm.cdf(-d1_temp)
    call_values.append(c)
    put_values.append(p)

# Impact of volatility
vol_range = np.linspace(0.05, 0.5, 20)
call_vol = []
put_vol = []

for sig in vol_range:
    d1_temp = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2_temp = d1_temp - sig*np.sqrt(T)
    c = S*norm.cdf(d1_temp) - K*np.exp(-r*T)*norm.cdf(d2_temp)
    p = K*np.exp(-r*T)*norm.cdf(-d2_temp) - S*norm.cdf(-d1_temp)
    call_vol.append(c)
    put_vol.append(p)

# Impact of time to expiry
time_range = np.linspace(0.01, 1.5, 30)
call_time = []
put_time = []

for t in time_range:
    d1_temp = (np.log(S/K) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2_temp = d1_temp - sigma*np.sqrt(t)
    c = S*norm.cdf(d1_temp) - K*np.exp(-r*t)*norm.cdf(d2_temp)
    p = K*np.exp(-r*t)*norm.cdf(-d2_temp) - S*norm.cdf(-d1_temp)
    call_time.append(c)
    put_time.append(p)

print(f"\nSensitivity Summary:")
print(f"  If S increases by $1 → Call increases by ${delta_call:.2f}, Put changes by ${delta_put:.2f}")
print(f"  If σ increases by 1% → Call increases by ${vega/100:.2f}, Put increases by ${vega/100:.2f}")
print(f"  If r increases by 1% → Call changes by ${rho_call/100:.2f}, Put changes by ${rho_put/100:.2f}")
print(f"  1 day of decay → Call loses ${abs(theta_call_daily):.2f}, Put loses ${abs(theta_put_daily):.2f}")

# === Scenario 5: Put-Call Parity Arbitrage ===
print("\n" + "="*60)
print("SCENARIO 5: Put-Call Parity & Arbitrage Detection")
print("="*60)

# Suppose we observe market prices (intentionally mispriced)
C_market = 10.5  # Slightly overpriced
P_market = 4.2   # Slightly underpriced

# Theoretical relationship
parity_diff = C_market - P_market - (S - K*np.exp(-r*T))

print(f"\nMarket Prices (hypothetical):")
print(f"  Call: ${C_market:.2f}")
print(f"  Put: ${P_market:.2f}")
print(f"\nParity Check:")
print(f"  C - P: {C_market - P_market:.4f}")
print(f"  S - Ke^(-rT): {S - K*np.exp(-r*T):.4f}")
print(f"  Difference: {parity_diff:.4f} {'(Arbitrage!)' if abs(parity_diff) > 0.01 else '(Fair)'}")

if parity_diff > 0.01:
    print(f"\n  Arbitrage Strategy (Reversal):")
    print(f"    1. Short stock at ${S}")
    print(f"    2. Short put at ${P_market} (pay ${P_market})")
    print(f"    3. Buy call at ${C_market} (pay ${C_market})")
    print(f"    Net cash: ${S - P_market - C_market:.2f} invested at r={r}")
    print(f"    At expiry: Payoff = max(S_T - K, 0) - max(K - S_T, 0) = S_T - K (from long call + short put)")
    print(f"    Offset by short stock at K")
    print(f"    Risk-free profit: ${-(S - P_market - C_market) * (np.exp(r*T) - 1):.2f}")

# === Visualization ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Payoff diagrams
ax = axes[0, 0]
S_range_payoff = np.linspace(60, 140, 200)
call_payoff = np.maximum(S_range_payoff - K, 0)
put_payoff = np.maximum(K - S_range_payoff, 0)
ax.plot(S_range_payoff, call_payoff, 'b-', linewidth=2.5, label='Call Payoff')
ax.plot(S_range_payoff, put_payoff, 'r-', linewidth=2.5, label='Put Payoff')
ax.axvline(K, color='k', linestyle='--', alpha=0.5, label=f'Strike=${K}')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Stock Price at Expiry')
ax.set_ylabel('Payoff')
ax.set_title('Option Payoff Diagrams')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Option value vs Stock price
ax = axes[0, 1]
ax.plot(spot_range, call_values, 'b-', linewidth=2.5, label='Call Value')
ax.plot(spot_range, put_values, 'r-', linewidth=2.5, label='Put Value')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.axhline(C_BS, color='b', linestyle=':', alpha=0.5)
ax.axhline(P_BS, color='r', linestyle=':', alpha=0.5)
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Option Value')
ax.set_title('Option Values vs Stock Price')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Option value vs Volatility
ax = axes[0, 2]
ax.plot(vol_range*100, call_vol, 'b-', linewidth=2.5, label='Call Value', marker='o')
ax.plot(vol_range*100, put_vol, 'r-', linewidth=2.5, label='Put Value', marker='s')
ax.axvline(sigma*100, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Volatility (σ) [%]')
ax.set_ylabel('Option Value')
ax.set_title('Option Values vs Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Option value vs Time to expiry
ax = axes[1, 0]
ax.plot(time_range, call_time, 'b-', linewidth=2.5, label='Call Value', marker='o')
ax.plot(time_range, put_time, 'r-', linewidth=2.5, label='Put Value', marker='s')
ax.axvline(T, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Time to Expiry (Years)')
ax.set_ylabel('Option Value')
ax.set_title('Option Values vs Time to Expiry')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Greeks for Call vs Stock Price
ax = axes[1, 1]
deltas = []
gammas = []
for s in spot_range:
    d1_temp = (np.log(s/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1_temp)
    gamma = norm.pdf(d1_temp) / (s * sigma * np.sqrt(T))
    deltas.append(delta)
    gammas.append(gamma)

ax_twin = ax.twinx()
ax.plot(spot_range, deltas, 'b-', linewidth=2.5, label='Delta')
ax_twin.plot(spot_range, gammas, 'g-', linewidth=2.5, label='Gamma')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Delta', color='b')
ax_twin.set_ylabel('Gamma', color='g')
ax.set_title('Greeks vs Stock Price')
ax.legend(loc='upper left')
ax_twin.legend(loc='upper right')
ax.grid(alpha=0.3)

# Plot 6: Greeks table
ax = axes[1, 2]
ax.axis('off')
greeks_data = [
    ['Greek', 'Call', 'Put'],
    ['Delta', f'{delta_call:.4f}', f'{delta_put:.4f}'],
    ['Gamma', f'{gamma:.6f}', f'{gamma:.6f}'],
    ['Theta (daily)', f'${theta_call_daily:.2f}', f'${theta_put_daily:.2f}'],
    ['Vega', f'${vega:.2f}', f'${vega:.2f}'],
    ['Rho', f'${rho_call:.2f}', f'${rho_put:.2f}'],
]

table = ax.table(cellText=greeks_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Greeks Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.show()