import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import warnings
    from scipy.special import comb

# Block 1
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("BINOMIAL TREE MODEL IMPLEMENTATION")
print("="*60)

class BinomialTree:
    """Binomial tree option pricing"""
    
    def __init__(self, S0, K, r, T, sigma, n, option_type='call', exercise='european', q=0):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.n = n
        self.option_type = option_type
        self.exercise = exercise
        self.q = q  # Dividend yield
        
        # Time step
        self.dt = T / n
        
        # Up and down factors (Cox-Ross-Rubinstein)
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        
        # Risk-neutral probability
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)
        
        # Discount factor
        self.discount = np.exp(-r * self.dt)
        
        # Validate probability
        if not (0 < self.p < 1):
            raise ValueError(f"Invalid risk-neutral probability p={self.p:.4f}")
    
    def build_stock_tree(self):
        """Build stock price tree"""
        tree = np.zeros((self.n + 1, self.n + 1))
        
        for i in range(self.n + 1):
            for j in range(i + 1):
                tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        
        return tree
    
    def price(self):
        """Price option using backward induction"""
        # Build stock tree
        stock_tree = self.build_stock_tree()
        
        # Initialize option value tree
        option_tree = np.zeros((self.n + 1, self.n + 1))
        
        # Terminal payoff at expiry
        for j in range(self.n + 1):
            if self.option_type == 'call':
                option_tree[j, self.n] = max(stock_tree[j, self.n] - self.K, 0)
            else:  # put
                option_tree[j, self.n] = max(self.K - stock_tree[j, self.n], 0)
        
        # Backward induction
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value (discounted expected value)
                continuation = self.discount * (
                    self.p * option_tree[j, i + 1] + 
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )
                
                if self.exercise == 'american':
                    # Early exercise value
                    if self.option_type == 'call':
                        intrinsic = max(stock_tree[j, i] - self.K, 0)
                    else:
                        intrinsic = max(self.K - stock_tree[j, i], 0)
                    
                    # Take maximum
                    option_tree[j, i] = max(continuation, intrinsic)
                else:
                    option_tree[j, i] = continuation
        
        return option_tree[0, 0], option_tree, stock_tree
    
    def greeks(self):
        """Calculate Greeks using finite differences"""
        price, option_tree, stock_tree = self.price()
        
        # Delta: (V_up - V_down) / (S_up - S_down)
        if self.n >= 1:
            delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])
        else:
            delta = np.nan
        
        # Gamma: Second derivative
        if self.n >= 2:
            delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (stock_tree[0, 2] - stock_tree[1, 2])
            delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (stock_tree[1, 2] - stock_tree[2, 2])
            gamma = (delta_up - delta_down) / ((stock_tree[0, 1] - stock_tree[1, 1]))
        else:
            gamma = np.nan
        
        # Theta: (V_t1 - V_t0) / dt
        if self.n >= 1:
            theta = (option_tree[0, 1] - option_tree[0, 0]) / self.dt
        else:
            theta = np.nan
        
        return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta}

# Black-Scholes for comparison
def black_scholes(S, K, r, T, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Scenario 1: Basic pricing
print("\n" + "="*60)
print("SCENARIO 1: European Option Pricing")
print("="*60)

S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.2
n = 100

bt_call = BinomialTree(S0, K, r, T, sigma, n, 'call', 'european')
bt_put = BinomialTree(S0, K, r, T, sigma, n, 'put', 'european')

call_price, _, _ = bt_call.price()
put_price, _, _ = bt_put.price()

bs_call = black_scholes(S0, K, r, T, sigma, 'call')
bs_put = black_scholes(S0, K, r, T, sigma, 'put')

print(f"\nParameters: S=${S0}, K=${K}, r={r:.1%}, T={T}yr, σ={sigma:.1%}, n={n}")
print(f"\nEuropean Call:")
print(f"  Binomial: ${call_price:.4f}")
print(f"  Black-Scholes: ${bs_call:.4f}")
print(f"  Difference: ${abs(call_price - bs_call):.4f}")

print(f"\nEuropean Put:")
print(f"  Binomial: ${put_price:.4f}")
print(f"  Black-Scholes: ${bs_put:.4f}")
print(f"  Difference: ${abs(put_price - bs_put):.4f}")

# Scenario 2: American options
print("\n" + "="*60)
print("SCENARIO 2: American vs European Put")
print("="*60)

bt_euro_put = BinomialTree(S0, K, r, T, sigma, n, 'put', 'european')
bt_amer_put = BinomialTree(S0, K, r, T, sigma, n, 'put', 'american')

euro_put, _, _ = bt_euro_put.price()
amer_put, _, _ = bt_amer_put.price()

early_exercise_premium = amer_put - euro_put

print(f"\nPut Option Comparison (K=${K}):")
print(f"  European Put: ${euro_put:.4f}")
print(f"  American Put: ${amer_put:.4f}")
print(f"  Early Exercise Premium: ${early_exercise_premium:.4f}")
print(f"  Premium as %: {early_exercise_premium/euro_put*100:.2f}%")

# Scenario 3: Convergence to Black-Scholes
print("\n" + "="*60)
print("SCENARIO 3: Convergence Analysis")
print("="*60)

steps = [5, 10, 20, 50, 100, 200, 500]
convergence_data = []

print(f"\n{'Steps':<8} {'Call Price':<12} {'BS Error':<12} {'Time (ms)':<12}")
print("-" * 44)

for n in steps:
    start = time.time()
    bt = BinomialTree(S0, K, r, T, sigma, n, 'call', 'european')
    price, _, _ = bt.price()
    elapsed = (time.time() - start) * 1000
    
    error = abs(price - bs_call)
    convergence_data.append((n, price, error, elapsed))
    
    print(f"{n:<8} ${price:<11.4f} ${error:<11.6f} {elapsed:<11.2f}")

# Scenario 4: Greeks calculation
print("\n" + "="*60)
print("SCENARIO 4: Greeks from Binomial Tree")
print("="*60)

n_greeks = 100
bt_greeks = BinomialTree(S0, K, r, T, sigma, n_greeks, 'call', 'european')
greeks = bt_greeks.greeks()

print(f"\nGreeks for European Call (n={n_greeks}):")
print(f"  Price: ${greeks['price']:.4f}")
print(f"  Delta: {greeks['delta']:.4f}")
print(f"  Gamma: {greeks['gamma']:.6f}")
print(f"  Theta: ${greeks['theta']:.4f}")

# Compare with Black-Scholes Greeks
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
bs_delta = norm.cdf(d1)
bs_gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))

print(f"\nBlack-Scholes Greeks:")
print(f"  Delta: {bs_delta:.4f}")
print(f"  Gamma: {bs_gamma:.6f}")

# Scenario 5: Visualize tree
print("\n" + "="*60)
print("SCENARIO 5: Small Tree Visualization")
print("="*60)

n_small = 4
bt_small = BinomialTree(S0, K, r, T, sigma, n_small, 'put', 'american')
price_small, option_tree, stock_tree = bt_small.price()

print(f"\nStock Price Tree (n={n_small}):")
for j in range(n_small + 1):
    print(f"  Step {j}: ", end='')
    for i in range(j + 1):
        print(f"${stock_tree[i, j]:6.2f}", end='  ')
    print()

print(f"\nOption Value Tree (American Put, K=${K}):")
for j in range(n_small + 1):
    print(f"  Step {j}: ", end='')
    for i in range(j + 1):
        print(f"${option_tree[i, j]:6.2f}", end='  ')
    print()

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Convergence of binomial to BS
ax = axes[0, 0]
steps_conv = [d[0] for d in convergence_data]
prices_conv = [d[1] for d in convergence_data]
ax.plot(steps_conv, prices_conv, 'bo-', linewidth=2, markersize=8, label='Binomial')
ax.axhline(bs_call, color='r', linestyle='--', linewidth=2, label='Black-Scholes')
ax.set_xlabel('Number of Steps')
ax.set_ylabel('Call Price')
ax.set_title('Convergence to Black-Scholes')
ax.set_xscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Error vs steps
ax = axes[0, 1]
errors_conv = [d[2] for d in convergence_data]
ax.plot(steps_conv, errors_conv, 'ro-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Steps')
ax.set_ylabel('Absolute Error')
ax.set_title('Pricing Error vs Steps')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3)

# Plot 3: Computation time vs steps
ax = axes[0, 2]
times_conv = [d[3] for d in convergence_data]
ax.plot(steps_conv, times_conv, 'go-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Steps')
ax.set_ylabel('Time (ms)')
ax.set_title('Computation Time vs Steps')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3)

# Plot 4: American vs European value
ax = axes[1, 0]
spot_range = np.linspace(70, 130, 20)
euro_values = []
amer_values = []

for s in spot_range:
    bt_e = BinomialTree(s, K, r, T, sigma, 50, 'put', 'european')
    bt_a = BinomialTree(s, K, r, T, sigma, 50, 'put', 'american')
    euro_values.append(bt_e.price()[0])
    amer_values.append(bt_a.price()[0])

ax.plot(spot_range, euro_values, 'b-', linewidth=2.5, label='European Put')
ax.plot(spot_range, amer_values, 'r-', linewidth=2.5, label='American Put')
ax.plot(spot_range, np.maximum(K - spot_range, 0), 'k--', alpha=0.5, label='Intrinsic')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Option Value')
ax.set_title('American vs European Put Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Tree structure visualization
ax = axes[1, 1]
n_viz = 5
bt_viz = BinomialTree(S0, K, r, T, sigma, n_viz, 'call', 'european')
_, _, stock_tree_viz = bt_viz.price()

for i in range(n_viz + 1):
    for j in range(i + 1):
        x = i
        y = stock_tree_viz[j, i]
        ax.plot(x, y, 'bo', markersize=8)
        
        # Draw lines to next nodes
        if i < n_viz:
            ax.plot([x, x+1], [y, stock_tree_viz[j, i+1]], 'b-', alpha=0.5)
            ax.plot([x, x+1], [y, stock_tree_viz[j+1, i+1]], 'b-', alpha=0.5)

ax.set_xlabel('Time Step')
ax.set_ylabel('Stock Price')
ax.set_title(f'Binomial Tree Structure (n={n_viz})')
ax.grid(alpha=0.3)

# Plot 6: Risk-neutral probability distribution at expiry
ax = axes[1, 2]
n_dist = 50
prices_expiry = []
probabilities = []

for j in range(n_dist + 1):
    price = S0 * (bt_small.u ** (n_dist - j)) * (bt_small.d ** j)
    # Binomial probability
    prob = comb(n_dist, j) * (bt_small.p ** (n_dist - j)) * ((1 - bt_small.p) ** j)
    prices_expiry.append(price)
    probabilities.append(prob)

ax.bar(prices_expiry, probabilities, width=3, alpha=0.7, edgecolor='black')
ax.axvline(S0 * np.exp(r*T), color='r', linestyle='--', linewidth=2, label='Forward Price')
ax.set_xlabel('Stock Price at Expiry')
ax.set_ylabel('Risk-Neutral Probability')
ax.set_title(f'Terminal Distribution (n={n_dist})')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()