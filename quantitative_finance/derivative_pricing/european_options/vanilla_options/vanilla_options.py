
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes formulas
def bs_call(S, K, T, r, sigma):
    """European call option price."""
    if T <= 0:
        return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    """European put option price."""
    if T <= 0:
        return np.maximum(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Option delta."""
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

# Classify moneyness
def moneyness(S, K, option_type='call'):
    """Classify option as ITM, ATM, or OTM."""
    ratio = S / K
    if abs(ratio - 1.0) < 0.02:  # Within 2%
        return 'ATM'
    elif option_type == 'call':
        return 'ITM' if S > K else 'OTM'
    else:  # put
        return 'ITM' if S < K else 'OTM'

# Parameters
S0 = 100.0
K_options = [90, 95, 100, 105, 110]  # Strike ladder
T = 0.5  # 6 months
r = 0.05
sigma = 0.25

print("="*80)
print("VANILLA OPTIONS: STRIKE LADDER ANALYSIS")
print("="*80)
print(f"Spot: S₀ = ${S0}, Time to Expiry: T = {T*12:.0f} months")
print(f"Volatility: σ = {sigma*100}%, Risk-free Rate: r = {r*100}%\n")

print(f"{'Strike':<8} {'Moneyness':<10} {'Call Price':<12} {'Put Price':<12} "
      f"{'Call Δ':<10} {'Put Δ':<10}")
print("-"*80)

for K in K_options:
    call_price = bs_call(S0, K, T, r, sigma)
    put_price = bs_put(S0, K, T, r, sigma)
    call_delta = bs_delta(S0, K, T, r, sigma, 'call')
    put_delta = bs_delta(S0, K, T, r, sigma, 'put')
    money = moneyness(S0, K, 'call')
    
    print(f"${K:<7.0f} {money:<10} ${call_price:<11.4f} ${put_price:<11.4f} "
          f"{call_delta:<9.4f} {put_delta:<9.4f}")

# Intrinsic vs time value
print("\n" + "="*80)
print("INTRINSIC vs TIME VALUE (Call @ K=$100)")
print("="*80)

K = 100
spots = np.array([80, 90, 100, 110, 120])
times = [1.0, 0.5, 0.25, 0.0]

for S in spots:
    print(f"\nSpot S = ${S} (Moneyness: {moneyness(S, K, 'call')})")
    print(f"{'Time (years)':<15} {'Call Price':<12} {'Intrinsic':<12} {'Time Value':<12}")
    print("-"*55)
    
    for t in times:
        call_price = bs_call(S, K, t, r, sigma)
        intrinsic = max(S - K, 0)
        time_value = call_price - intrinsic
        
        print(f"{t:<15.2f} ${call_price:<11.4f} ${intrinsic:<11.4f} ${time_value:<11.4f}")

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Call and put payoffs at maturity
S_range = np.linspace(50, 150, 200)
K = 100

call_payoff = np.maximum(S_range - K, 0)
put_payoff = np.maximum(K - S_range, 0)

ax = axes[0, 0]
ax.plot(S_range, call_payoff, 'b-', linewidth=2, label='Call Payoff')
ax.plot(S_range, put_payoff, 'r-', linewidth=2, label='Put Payoff')
ax.axvline(K, color='black', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Payoff ($)')
ax.set_title('Vanilla Option Payoffs (Intrinsic Value)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Option prices before expiry (T=0.5yr)
call_prices = [bs_call(S, K, T, r, sigma) for S in S_range]
put_prices = [bs_put(S, K, T, r, sigma) for S in S_range]

ax = axes[0, 1]
ax.plot(S_range, call_prices, 'b-', linewidth=2, label=f'Call (T={T}yr)')
ax.plot(S_range, put_prices, 'r-', linewidth=2, label=f'Put (T={T}yr)')
ax.plot(S_range, call_payoff, 'b--', linewidth=1, alpha=0.5, label='Call Intrinsic')
ax.plot(S_range, put_payoff, 'r--', linewidth=1, alpha=0.5, label='Put Intrinsic')
ax.axvline(K, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Option Price ($)')
ax.set_title('Option Prices (Before Expiry)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Time value evolution
times_plot = np.linspace(0, 1, 50)
S_samples = [80, 90, 100, 110, 120]
colors = plt.cm.viridis(np.linspace(0, 1, len(S_samples)))

ax = axes[0, 2]
for S, color in zip(S_samples, colors):
    time_values = []
    for t in times_plot:
        call_price = bs_call(S, K, t, r, sigma)
        intrinsic = max(S - K, 0)
        time_values.append(call_price - intrinsic)
    ax.plot(times_plot, time_values, linewidth=2, color=color, label=f'S=${S}')
ax.set_xlabel('Time to Expiry (years)')
ax.set_ylabel('Time Value ($)')
ax.set_title('Time Value Decay (Call @ K=$100)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Moneyness heatmap (strike vs spot)
strikes_heat = np.linspace(70, 130, 30)
spots_heat = np.linspace(70, 130, 30)
K_grid, S_grid = np.meshgrid(strikes_heat, spots_heat)
moneyness_grid = (S_grid - K_grid) / K_grid  # (S-K)/K

ax = axes[1, 0]
contour = ax.contourf(K_grid, S_grid, moneyness_grid * 100, levels=20, cmap='RdYlGn')
ax.plot([70, 130], [70, 130], 'k--', linewidth=2, label='ATM Line (S=K)')
ax.set_xlabel('Strike K')
ax.set_ylabel('Spot S')
ax.set_title('Moneyness Heatmap: (S-K)/K (%)')
plt.colorbar(contour, ax=ax, label='Moneyness (%)')
ax.legend()

# Plot 5: Delta profiles
deltas_call = [bs_delta(S, K, T, r, sigma, 'call') for S in S_range]
deltas_put = [bs_delta(S, K, T, r, sigma, 'put') for S in S_range]

ax = axes[1, 1]
ax.plot(S_range, deltas_call, 'b-', linewidth=2, label='Call Delta')
ax.plot(S_range, deltas_put, 'r-', linewidth=2, label='Put Delta')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(0.5, color='b', linestyle='--', alpha=0.3)
ax.axhline(-0.5, color='r', linestyle='--', alpha=0.3)
ax.axvline(K, color='black', linestyle='--', alpha=0.5, label='ATM')

# Mark moneyness regions
ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], 50, K, alpha=0.1, color='red', label='OTM Call')
ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], K, 150, alpha=0.1, color='green', label='ITM Call')

ax.set_xlabel('Spot Price S')
ax.set_ylabel('Delta')
ax.set_title('Delta vs Moneyness (K=$100)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Covered call strategy (Long stock + Short call)
stock_pnl = S_range - S0
call_premium = bs_call(S0, K, T, r, sigma)
short_call_pnl = call_premium - np.maximum(S_range - K, 0)  # Premium collected - payoff
covered_call_pnl = stock_pnl + short_call_pnl

ax = axes[1, 2]
ax.plot(S_range, stock_pnl, 'g--', linewidth=2, label='Long Stock Only')
ax.plot(S_range, covered_call_pnl, 'b-', linewidth=2, label='Covered Call')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(S0, color='purple', linestyle='--', alpha=0.5, label=f'Entry S₀=${S0}')
ax.axvline(K, color='orange', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Profit/Loss ($)')
ax.set_title(f'Covered Call Strategy (Collect ${call_premium:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 7: Protective put strategy (Long stock + Long put)
put_premium = bs_put(S0, K, T, r, sigma)
long_put_pnl = np.maximum(K - S_range, 0) - put_premium  # Payoff - premium paid
protective_put_pnl = stock_pnl + long_put_pnl

ax = axes[2, 0]
ax.plot(S_range, stock_pnl, 'g--', linewidth=2, label='Long Stock Only')
ax.plot(S_range, protective_put_pnl, 'r-', linewidth=2, label='Protective Put')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(K - S0 - put_premium, color='red', linestyle='--', linewidth=1.5,
           label=f'Floor=${K - S0 - put_premium:.2f}')
ax.axvline(S0, color='purple', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Profit/Loss ($)')
ax.set_title(f'Protective Put Strategy (Pay ${put_premium:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 8: Long call P&L diagram
call_cost = bs_call(S0, K, T, r, sigma)
long_call_pnl = np.maximum(S_range - K, 0) - call_cost

ax = axes[2, 1]
ax.plot(S_range, long_call_pnl, 'b-', linewidth=2, label='Long Call')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(-call_cost, color='red', linestyle='--', alpha=0.5, label=f'Max Loss=${call_cost:.2f}')
ax.axvline(K, color='green', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.axvline(K + call_cost, color='orange', linestyle='--', alpha=0.5, 
           label=f'Breakeven=${K + call_cost:.2f}')
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Profit/Loss ($)')
ax.set_title('Long Call P&L (Bullish Speculation)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 9: Long put P&L diagram
put_cost = bs_put(S0, K, T, r, sigma)
long_put_pnl = np.maximum(K - S_range, 0) - put_cost

ax = axes[2, 2]
ax.plot(S_range, long_put_pnl, 'r-', linewidth=2, label='Long Put')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(-put_cost, color='blue', linestyle='--', alpha=0.5, label=f'Max Loss=${put_cost:.2f}')
ax.axvline(K, color='green', linestyle='--', alpha=0.5, label=f'Strike K=${K}')
ax.axvline(K - put_cost, color='orange', linestyle='--', alpha=0.5, 
           label=f'Breakeven=${K - put_cost:.2f}')
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Profit/Loss ($)')
ax.set_title('Long Put P&L (Bearish Speculation)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vanilla_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Comparison of strategies
print("\n" + "="*80)
print("STRATEGY COMPARISON @ MATURITY (Various Stock Prices)")
print("="*80)

final_prices = [70, 85, 100, 115, 130]
print(f"\n{'S_T':<8} {'Stock':<10} {'Long Call':<12} {'Long Put':<12} "
      f"{'Cov. Call':<12} {'Prot. Put':<12}")
print("-"*80)

for S_T in final_prices:
    stock_pnl = S_T - S0
    call_pnl = max(S_T - K, 0) - call_cost
    put_pnl = max(K - S_T, 0) - put_cost
    cov_call = (S_T - S0) + (call_premium - max(S_T - K, 0))
    prot_put = (S_T - S0) + (max(K - S_T, 0) - put_premium)
    
    print(f"${S_T:<7.0f} ${stock_pnl:<9.2f} ${call_pnl:<11.2f} ${put_pnl:<11.2f} "
          f"${cov_call:<11.2f} ${prot_put:<11.2f}")