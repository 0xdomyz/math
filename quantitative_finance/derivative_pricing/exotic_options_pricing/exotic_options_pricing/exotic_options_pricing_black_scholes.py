import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.linalg import cholesky
import warnings
def black_scholes(S, K, r, T, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Scenario 1: Asian options
print("\n" + "="*60)
print("SCENARIO 1: Asian Options (Path-Dependent)")
print("="*60)

S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20

pricer = ExoticOptionPricer(S0, r, sigma, T)

# Arithmetic Asian
price_arith, se_arith = pricer.asian_arithmetic_mc(K, n_paths=50000)

# Geometric Asian (closed-form)
price_geom = pricer.asian_geometric_closed_form(K)

# Vanilla for comparison
vanilla_call = black_scholes(S0, K, r, T, sigma, 'call')

print(f"\nParameters: S=${S0}, K=${K}, r={r:.1%}, T={T}yr, σ={sigma:.1%}")

print(f"\nArithmetic Average Asian Call:")
print(f"  Price: ${price_arith:.4f} ± ${se_arith:.4f}")
print(f"  Discount vs Vanilla: {(vanilla_call - price_arith)/vanilla_call*100:.1f}%")

print(f"\nGeometric Average Asian Call:")
print(f"  Price: ${price_geom:.4f} (closed-form)")
print(f"  Discount vs Vanilla: {(vanilla_call - price_geom)/vanilla_call*100:.1f}%")

print(f"\nVanilla European Call: ${vanilla_call:.4f}")
print(f"\nAsian options cheaper due to reduced volatility from averaging")

# Scenario 2: Barrier options
print("\n" + "="*60)
print("SCENARIO 2: Barrier Options")
print("="*60)

barriers = [85, 90, 95]

print(f"\nDown-and-Out Call Options (K=${K}):")
print(f"{'Barrier':<12} {'Price':<12} {'Vanilla':<12} {'Discount %':<12}")
print("-" * 48)

for H in barriers:
    price_barrier, se_barrier = pricer.barrier_down_out_call(K, H, n_paths=50000)
    discount_pct = (vanilla_call - price_barrier) / vanilla_call * 100
    
    print(f"${H:<11} ${price_barrier:<11.4f} ${vanilla_call:<11.4f} {discount_pct:<11.1f}%")

print(f"\nLower barrier → higher knock-out probability → cheaper option")

# In-out parity check
H_test = 90
price_out, _ = pricer.barrier_down_out_call(K, H_test, n_paths=50000)

# Simulate down-and-in
payoffs_in = []
for _ in range(50000):
    path = pricer.generate_path(252)
    if np.min(path) <= H_test:  # Knocked in
        payoff = max(path[-1] - K, 0)
    else:
        payoff = 0
    payoffs_in.append(payoff)

price_in = np.exp(-r*T) * np.mean(payoffs_in)

print(f"\nIn-Out Parity Check (Barrier=${H_test}):")
print(f"  Down-and-Out: ${price_out:.4f}")
print(f"  Down-and-In: ${price_in:.4f}")
print(f"  Sum: ${price_out + price_in:.4f}")
print(f"  Vanilla: ${vanilla_call:.4f}")
print(f"  Difference: ${abs(price_out + price_in - vanilla_call):.4f}")

# Scenario 3: Lookback options
print("\n" + "="*60)
print("SCENARIO 3: Lookback Options")
print("="*60)

price_lookback, se_lookback = pricer.lookback_floating_call(n_paths=50000)

print(f"\nFloating Strike Lookback Call:")
print(f"  Payoff: S_T - min(S_t)")
print(f"  Price: ${price_lookback:.4f} ± ${se_lookback:.4f}")
print(f"  Vanilla Call: ${vanilla_call:.4f}")
print(f"  Premium: ${price_lookback - vanilla_call:.4f} ({(price_lookback/vanilla_call - 1)*100:.1f}%)")

print(f"\nLookback guarantees best execution → always ITM → expensive")

# Scenario 4: Digital options
print("\n" + "="*60)
print("SCENARIO 4: Digital (Binary) Options")
print("="*60)

strikes_digital = np.linspace(90, 110, 9)
cash_payoff = 10.0

print(f"\nCash-or-Nothing Digital Call (pays ${cash_payoff} if ITM):")
print(f"{'Strike':<12} {'Digital Price':<15} {'Risk-Neutral Prob':<20}")
print("-" * 47)

for K_dig in strikes_digital:
    price_dig = pricer.digital_call(K_dig, cash_payoff)
    prob = price_dig / (cash_payoff * np.exp(-r*T))
    
    if K_dig in [90, 100, 110]:
        print(f"${K_dig:<11} ${price_dig:<14.4f} {prob*100:<19.2f}%")

print(f"\nDigital price = Discounted probability × Cash payoff")

# Scenario 5: Multi-asset options
print("\n" + "="*60)
print("SCENARIO 5: Multi-Asset (Rainbow) Options")
print("="*60)

S0_multi = [100, 100, 100]
sigma_multi = [0.20, 0.25, 0.30]
corr_matrix = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])
weights = np.array([0.4, 0.3, 0.3])

multi_pricer = MultiAssetExotics(S0_multi, r, sigma_multi, corr_matrix, T)

# Basket
price_basket, se_basket = multi_pricer.basket_option(K, weights, n_paths=50000)

# Best-of
price_best, se_best = multi_pricer.best_of_call(K, n_paths=50000)

# Worst-of
price_worst, se_worst = multi_pricer.worst_of_put(K, n_paths=50000)

# Spread
price_spread, se_spread = multi_pricer.spread_option(0, n_paths=50000)

# Individual calls for comparison
vanilla_calls_sum = sum([black_scholes(S0_multi[i], K, r, T, sigma_multi[i], 'call') 
                         for i in range(3)])

print(f"\n3-Asset Options (S=[{S0_multi[0]}, {S0_multi[1]}, {S0_multi[2]}]):")
print(f"Correlations: ρ₁₂={corr_matrix[0,1]:.1f}, ρ₁₃={corr_matrix[0,2]:.1f}, ρ₂₃={corr_matrix[1,2]:.1f}")

print(f"\nBasket Call (weights={weights}):")
print(f"  Price: ${price_basket:.4f} ± ${se_basket:.4f}")
print(f"  Sum of individual calls: ${vanilla_calls_sum:.4f}")
print(f"  Diversification benefit: ${vanilla_calls_sum - price_basket:.4f}")

print(f"\nBest-of Call (max of 3 assets):")
print(f"  Price: ${price_best:.4f} ± ${se_best:.4f}")
print(f"  Premium over single: {price_best/vanilla_call - 1:.1%}")

print(f"\nWorst-of Put (min of 3 assets):")
print(f"  Price: ${price_worst:.4f} ± ${se_worst:.4f}")

print(f"\nSpread Option (S₁ - S₂):")
print(f"  Price: ${price_spread:.4f} ± ${se_spread:.4f}")

# Scenario 6: Correlation impact on basket
print("\n" + "="*60)
print("SCENARIO 6: Correlation Impact on Basket Options")
print("="*60)

correlations_test = [0.0, 0.3, 0.6, 0.9]

print(f"\nBasket Call Sensitivity to Correlation:")
print(f"{'Correlation':<15} {'Price':<12} {'vs ρ=0':<15}")
print("-" * 42)

prices_by_corr = []

for rho in correlations_test:
    # Uniform correlation matrix
    corr_test = np.eye(3) + (1 - np.eye(3)) * rho
    
    pricer_corr = MultiAssetExotics(S0_multi, r, sigma_multi, corr_test, T)
    price_corr, _ = pricer_corr.basket_option(K, weights, n_paths=30000)
    prices_by_corr.append(price_corr)
    
    if rho == 0.0:
        base_price = price_corr
        diff_str = "baseline"
    else:
        diff_str = f"+${price_corr - base_price:.4f}"
    
    print(f"ρ={rho:<13.1f} ${price_corr:<11.4f} {diff_str:<15}")

print(f"\nHigher correlation → less diversification → higher basket value")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Sample paths with Asian averaging
ax = axes[0, 0]
n_sample_paths = 10
n_steps_viz = 252

for _ in range(n_sample_paths):
    path = pricer.generate_path(n_steps_viz)
    times = np.linspace(0, T, n_steps_viz + 1)
    ax.plot(times, path, 'b-', alpha=0.5, linewidth=1)
    
    # Show average
    avg = np.mean(path)
    ax.axhline(avg, color='r', linestyle='--', alpha=0.3, linewidth=1)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Sample Paths (Asian Average in Red)')
ax.grid(alpha=0.3)

# Plot 2: Barrier knock-out illustration
ax = axes[0, 1]
np.random.seed(123)

for i in range(15):
    path = pricer.generate_path(n_steps_viz)
    times = np.linspace(0, T, n_steps_viz + 1)
    
    barrier = 90
    knocked = np.min(path) <= barrier
    color = 'red' if knocked else 'green'
    alpha = 0.3 if knocked else 0.7
    
    ax.plot(times, path, color=color, alpha=alpha, linewidth=1.5)

ax.axhline(barrier, color='black', linestyle='--', linewidth=2, label=f'Barrier ${barrier}')
ax.axhline(K, color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Strike ${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Down-and-Out Paths (Red=Knocked Out)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Asian vs Vanilla price comparison
ax = axes[0, 2]
strikes_range = np.linspace(85, 115, 15)
asian_prices = []
vanilla_prices = []

for K_test in strikes_range:
    pricer_test = ExoticOptionPricer(S0, r, sigma, T)
    p_asian, _ = pricer_test.asian_arithmetic_mc(K_test, n_paths=10000, n_steps=100)
    p_vanilla = black_scholes(S0, K_test, r, T, sigma, 'call')
    asian_prices.append(p_asian)
    vanilla_prices.append(p_vanilla)

ax.plot(strikes_range, vanilla_prices, 'b-', linewidth=2.5, marker='o', label='Vanilla Call')
ax.plot(strikes_range, asian_prices, 'r-', linewidth=2.5, marker='s', label='Asian Call')
ax.axvline(S0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Strike')
ax.set_ylabel('Option Price')
ax.set_title('Asian vs Vanilla Call Prices')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Digital call delta profile
ax = axes[1, 0]
spots_digital = np.linspace(80, 120, 100)
digital_prices = []

for S_test in spots_digital:
    pricer_dig = ExoticOptionPricer(S_test, r, sigma, T)
    p_dig = pricer_dig.digital_call(K, cash_payoff=1.0)
    digital_prices.append(p_dig)

ax.plot(spots_digital, digital_prices, 'purple', linewidth=2.5)
ax.axvline(K, color='r', linestyle='--', linewidth=2, label=f'Strike ${K}')
ax.set_xlabel('Spot Price')
ax.set_ylabel('Digital Call Price')
ax.set_title('Digital Option: Discontinuous Payoff')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Lookback payoff distribution
ax = axes[1, 1]
lookback_payoffs = []

for _ in range(5000):
    path = pricer.generate_path(252)
    payoff = path[-1] - np.min(path)
    lookback_payoffs.append(payoff)

ax.hist(lookback_payoffs, bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(np.mean(lookback_payoffs), color='r', linestyle='--', linewidth=2, 
           label=f'Mean: ${np.mean(lookback_payoffs):.2f}')
ax.set_xlabel('Payoff (S_T - min)')
ax.set_ylabel('Density')
ax.set_title('Lookback Option Payoff Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Basket price vs correlation
ax = axes[1, 2]
ax.plot(correlations_test, prices_by_corr, 'go-', linewidth=2.5, markersize=10)
ax.set_xlabel('Correlation')
ax.set_ylabel('Basket Option Price')
ax.set_title('Basket Call Price vs Correlation')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()