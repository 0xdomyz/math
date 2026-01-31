
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes analytical formulas
def black_scholes_call(S0, K, T, r, sigma):
    """European call option price (Black-Scholes)."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S0, K, T, r, sigma):
    """
    European put option price (Black-Scholes).
    
    Returns:
    - put_price: Option value
    - delta: First derivative w.r.t. S (negative for puts)
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1  # Negative for puts
    
    return put_price, delta

# Monte Carlo put pricing
def monte_carlo_put(S0, K, T, r, sigma, n_paths, antithetic=False):
    """
    Monte Carlo simulation for European put option.
    
    Returns:
    - put_price: Estimated option value
    - std_error: Standard error of estimate
    - terminal_prices: Array of simulated S_T values
    - payoffs: Array of put payoffs
    """
    if antithetic:
        n_half = n_paths // 2
        Z = np.random.randn(n_half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = np.random.randn(n_paths)
    
    # GBM terminal prices
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z_full
    terminal_prices = S0 * np.exp(drift + diffusion)
    
    # Put payoff: max(K - S_T, 0)
    payoffs = np.maximum(K - terminal_prices, 0)
    
    # Discounted expected payoff
    put_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return put_price, std_error, terminal_prices, payoffs

# Parameters
S0 = 100.0      # Current stock price
K = 100.0       # Strike price (ATM)
T = 1.0         # 1 year to maturity
r = 0.05        # 5% risk-free rate
sigma = 0.25    # 25% volatility

# Analytical Black-Scholes prices
bs_put, bs_delta_put = black_scholes_put(S0, K, T, r, sigma)
bs_call = black_scholes_call(S0, K, T, r, sigma)

print("="*60)
print("BLACK-SCHOLES PRICES")
print("="*60)
print(f"Put Price:  ${bs_put:.4f}")
print(f"Call Price: ${bs_call:.4f}")
print(f"Put Delta:  {bs_delta_put:.4f}")

# Verify put-call parity: C - P = S0 - K*exp(-rT)
parity_lhs = bs_call - bs_put
parity_rhs = S0 - K * np.exp(-r * T)
print(f"\nPut-Call Parity Check:")
print(f"  C - P = ${parity_lhs:.4f}")
print(f"  S₀ - Ke^(-rT) = ${parity_rhs:.4f}")
print(f"  Difference: ${abs(parity_lhs - parity_rhs):.6f}")

# Monte Carlo convergence analysis
np.random.seed(42)
n_paths = 100000

mc_put, mc_error, terminal_prices, put_payoffs = monte_carlo_put(
    S0, K, T, r, sigma, n_paths, antithetic=True
)

print(f"\nMonte Carlo Put Price (N={n_paths}): ${mc_put:.4f} ± ${1.96*mc_error:.4f}")
print(f"Difference from BS: ${abs(mc_put - bs_put):.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Put price surface (strike vs spot)
spots = np.linspace(50, 150, 50)
strikes = [80, 90, 100, 110, 120]
ax = axes[0, 0]
for K_i in strikes:
    put_prices = [black_scholes_put(S, K_i, T, r, sigma)[0] for S in spots]
    ax.plot(spots, put_prices, label=f'K=${K_i}', linewidth=2)
ax.axvline(S0, color='black', linestyle='--', alpha=0.5, label=f'Current S₀=${S0}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Put Option Price ($)')
ax.set_title('European Put Value vs Spot (T=1yr)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Put payoff diagram at maturity
ax = axes[0, 1]
S_range = np.linspace(50, 150, 100)
put_payoff = np.maximum(K - S_range, 0)
profit = put_payoff - bs_put  # P&L including premium paid

ax.plot(S_range, put_payoff, 'b-', linewidth=2, label='Payoff at Maturity')
ax.plot(S_range, profit, 'r-', linewidth=2, label='Profit (net premium)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(K, color='green', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.axvline(K - bs_put, color='orange', linestyle='--', linewidth=1.5, 
           label=f'Breakeven=${K - bs_put:.2f}')
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Put Value ($)')
ax.set_title('Put Payoff Diagram')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Distribution of terminal prices
ax = axes[0, 2]
ax.hist(terminal_prices, bins=60, density=True, alpha=0.7, edgecolor='black')
ax.axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.axvline(S0, color='green', linestyle='--', linewidth=2, label=f'Spot S₀=${S0}')
ax.axvline(np.mean(terminal_prices), color='blue', linestyle='--', linewidth=2,
           label=f'E[S_T]=${np.mean(terminal_prices):.2f}')
# Expected value under risk-neutral measure
expected_ST = S0 * np.exp(r * T)
ax.axvline(expected_ST, color='purple', linestyle=':', linewidth=2,
           label=f'S₀e^(rT)=${expected_ST:.2f}')
ax.set_xlabel('Terminal Stock Price S_T')
ax.set_ylabel('Density')
ax.set_title(f'Distribution of S_T ({n_paths:,} paths)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Put payoff distribution
ax = axes[1, 0]
ax.hist(put_payoffs, bins=60, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(np.mean(put_payoffs) * np.exp(-r*T), color='red', linestyle='--', linewidth=2,
           label=f'PV Mean: ${np.mean(put_payoffs)*np.exp(-r*T):.4f}')
ax.axvline(bs_put, color='blue', linestyle='--', linewidth=2,
           label=f'BS Price: ${bs_put:.4f}')
ax.set_xlabel('Put Payoff at Maturity')
ax.set_ylabel('Density')
ax.set_title('Distribution of Put Payoffs')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Protective put strategy (long stock + long put)
ax = axes[1, 1]
stock_profit = S_range - S0
protective_put_payoff = S_range + np.maximum(K - S_range, 0)
protective_put_profit = protective_put_payoff - S0 - bs_put

ax.plot(S_range, stock_profit, 'g--', linewidth=2, label='Long Stock Only')
ax.plot(S_range, protective_put_profit, 'b-', linewidth=2, label='Protective Put')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(K - S0 - bs_put, color='red', linestyle='--', linewidth=1.5,
           label=f'Floor=${K - S0 - bs_put:.2f}')
ax.axvline(S0, color='purple', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Profit ($)')
ax.set_title('Protective Put Strategy (Insurance)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Put delta profile
ax = axes[1, 2]
deltas = [black_scholes_put(S, K, T, r, sigma)[1] for S in spots]
ax.plot(spots, deltas, 'b-', linewidth=2)
ax.axhline(-0.5, color='red', linestyle='--', alpha=0.5, label='Δ = -0.5 (ATM)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(S0, color='green', linestyle='--', alpha=0.5, label=f'S₀=${S0}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Delta (∂P/∂S)')
ax.set_title('Put Delta Profile')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.05, 0.05)

plt.tight_layout()
plt.savefig('european_put_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Put-call parity verification with MC
np.random.seed(42)
mc_call_price = []
mc_put_price = []

for _ in range(100):  # 100 independent MC runs
    Z = np.random.randn(10000)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)
    
    call_payoffs = np.maximum(ST - K, 0)
    put_payoffs = np.maximum(K - ST, 0)
    
    mc_call_price.append(np.exp(-r * T) * np.mean(call_payoffs))
    mc_put_price.append(np.exp(-r * T) * np.mean(put_payoffs))

mc_parity_diff = np.array(mc_call_price) - np.array(mc_put_price) - (S0 - K * np.exp(-r * T))

print("\n" + "="*60)
print("PUT-CALL PARITY VERIFICATION (100 MC runs)")
print("="*60)
print(f"Mean C - P - (S₀ - Ke^(-rT)): ${np.mean(mc_parity_diff):.6f}")
print(f"Std Dev of Parity Error: ${np.std(mc_parity_diff):.6f}")
print(f"Max Absolute Error: ${np.max(np.abs(mc_parity_diff)):.6f}")