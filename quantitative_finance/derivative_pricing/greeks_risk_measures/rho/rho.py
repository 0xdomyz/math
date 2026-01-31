
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

def bs_put(S, K, T, r, sigma):
    call = bs_call(S, K, T, r, sigma)
    put = call - S + K*np.exp(-r*T)
    return put

def rho_call_bs(S, K, T, r, sigma):
    """Rho per 1% rate change for call"""
    d2 = bs_d2(S, K, T, r, sigma)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    return rho

def rho_put_bs(S, K, T, r, sigma):
    """Rho per 1% rate change for put"""
    d2 = bs_d2(S, K, T, r, sigma)
    rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)
    return rho

def delta_bs(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def gamma_bs(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Parameters
S0, K, T, sigma = 100, 100, 5, 0.2  # Long-dated for rho effect
rates = np.array([0.01, 0.02, 0.05, 0.10])  # 1%, 2%, 5%, 10%

# 1. Rho across spot prices
print("=== RHO ANALYSIS ===")
print("\nCall Rho (per 1% rate) across spot prices and rates:")

spot_prices = np.linspace(80, 120, 10)
r_base = 0.05

print(f"\nAt rate r = {r_base:.2%}:")
print("Spot\tRho (call)\tRho (put)")
for S in spot_prices:
    rho_c = rho_call_bs(S, K, T, r_base, sigma)
    rho_p = rho_put_bs(S, K, T, r_base, sigma)
    print(f"{S:5.0f}\t${rho_c:7.2f}\t\t${rho_p:7.2f}")

# 2. Rho vs rate level
print("\n\nCall Rho vs Interest Rate:")
spot_prices_range = np.linspace(80, 120, 100)
rhos_call = [rho_call_bs(S, K, T, r_base, sigma) for S in spot_prices_range]
rhos_put = [rho_put_bs(S, K, T, r_base, sigma) for S in spot_prices_range]

# 3. Rho vs maturity
T_values = np.array([0.25, 0.5, 1, 2, 5, 10])
rhos_by_maturity = [rho_call_bs(S0, K, T, r_base, sigma) for T in T_values]

# 4. Rate scenario impact analysis
print("\n=== RATE SCENARIO ANALYSIS (5-year ATM call) ===")
print("Rate scenario\tCall Price\tRho Estimate\tActual Diff")

r_current = 0.05
call_price_current = bs_call(S0, K, T, r_current, sigma)
rho_current = rho_call_bs(S0, K, T, r_current, sigma)

rate_scenarios = np.array([0.02, 0.03, 0.05, 0.07, 0.08])

for r_new in rate_scenarios:
    call_price_new = bs_call(S0, K, T, r_new, sigma)
    rate_change = (r_new - r_current) * 100  # in percentage points
    
    rho_estimate = call_price_current + rho_current * (r_new - r_current) * 0.01
    actual_diff = call_price_new - call_price_current
    error = abs(rho_estimate - call_price_new)
    
    print(f"{r_new:.2%}\t\t${call_price_new:.4f}\t\t${rho_estimate:.4f}\t\t${actual_diff:.4f}")

# 5. Portfolio rho analysis
print("\n=== PORTFOLIO RHO ===")
# Portfolio: Long 100 calls + Short 150 puts (synthetic long)
long_calls = 100
short_puts = 150

portfolio_rho = long_calls * rho_call_bs(S0, K, T, r_base, sigma) + \
                short_puts * rho_put_bs(S0, K, T, r_base, sigma)

print(f"Portfolio: {long_calls} long calls + {short_puts} short puts")
print(f"Total portfolio rho: ${portfolio_rho:.2f} (per 1% rate change)")
print(f"Daily rate change impact: ~${portfolio_rho/252:.4f} (assuming 1% annual move)")

# 6. Correlation with other Greeks
print("\n=== RHOS CORRELATION WITH OTHER GREEKS ===")
delta_current = delta_bs(S0, K, T, r_base, sigma)
gamma_current = gamma_bs(S0, K, T, r_base, sigma)

print(f"Delta: {delta_current:.4f}")
print(f"Gamma: {gamma_current:.6f}")
print(f"Rho (call): ${rho_current:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Rho vs Spot
axes[0, 0].plot(spot_prices_range, rhos_call, linewidth=2, label='Call Rho', color='blue')
axes[0, 0].plot(spot_prices_range, rhos_put, linewidth=2, label='Put Rho', color='red')
axes[0, 0].axvline(K, color='k', linestyle='--', alpha=0.5)
axes[0, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[0, 0].set_xlabel('Spot Price ($)')
axes[0, 0].set_ylabel('Rho (per 1% rate change)')
axes[0, 0].set_title(f'Rho vs Spot Price (T={T}yr)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Rho vs Maturity
axes[0, 1].plot(T_values, rhos_by_maturity, 'o-', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Time to Expiry (years)')
axes[0, 1].set_ylabel('Rho (per 1% rate change)')
axes[0, 1].set_title(f'Rho vs Maturity (S={S0}, K={K}, ATM)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Call price across rate scenarios
call_prices_rates = [bs_call(S0, K, T, r, sigma) for r in rate_scenarios]
put_prices_rates = [bs_put(S0, K, T, r, sigma) for r in rate_scenarios]

axes[1, 0].plot(rate_scenarios*100, call_prices_rates, 'o-', linewidth=2, 
               markersize=8, label='Call Price', color='blue')
axes[1, 0].plot(rate_scenarios*100, put_prices_rates, 's-', linewidth=2, 
               markersize=8, label='Put Price', color='red')
axes[1, 0].axvline(r_base*100, color='k', linestyle='--', alpha=0.5, label='Current Rate')
axes[1, 0].set_xlabel('Interest Rate (%)')
axes[1, 0].set_ylabel('Option Price ($)')
axes[1, 0].set_title('Option Prices vs Interest Rate')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Greeks comparison (all Greeks at ATM)
spot_range = np.linspace(80, 120, 100)
deltas = [delta_bs(S, K, T, r_base, sigma) for S in spot_range]
gammas = [gamma_bs(S, K, T, r_base, sigma) for S in spot_range]
rhos = [rho_call_bs(S, K, T, r_base, sigma) for S in spot_range]

ax1 = axes[1, 1]
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))

line1 = ax1.plot(spot_range, deltas, linewidth=2, label='Delta', color='blue')
line2 = ax2.plot(spot_range, [g*100 for g in gammas], linewidth=2, label='Gamma (×100)', color='green')
line3 = ax3.plot(spot_range, rhos, linewidth=2, label='Rho', color='red')

ax1.set_xlabel('Spot Price ($)')
ax1.set_ylabel('Delta', color='blue')
ax2.set_ylabel('Gamma (×100)', color='green')
ax3.set_ylabel('Rho', color='red')
ax1.set_title('All Greeks vs Spot Price')

ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='green')
ax3.tick_params(axis='y', labelcolor='red')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity elasticity
print("\n=== RHO ELASTICITY (% change in price per % change in rate) ===")
rate_elasticity = (rho_current / call_price_current) * 100
print(f"Call price elasticity to rates: {rate_elasticity:.2f}%")
print(f"Interpretation: 1% rate change → ~{rate_elasticity:.2f}% call price change")