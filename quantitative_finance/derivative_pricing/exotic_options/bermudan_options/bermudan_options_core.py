import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Block 1

# Black-Scholes European call
def european_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Binomial tree for American option (benchmark)
def binomial_american_call(S0, K, T, r, sigma, N):
    """American call via binomial tree (N steps)."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Terminal payoffs
    ST = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    V = np.maximum(ST - K, 0)
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        S = np.array([S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
        V = discount * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        intrinsic = np.maximum(S - K, 0)
        V = np.maximum(V, intrinsic)  # Early exercise
    
    return V[0]

# Longstaff-Schwartz for Bermudan option
def lsm_bermudan_call(S0, K, T, r, sigma, exercise_dates, n_paths, degree=2):
    """
    Price Bermudan call using Longstaff-Schwartz.
    
    Parameters:
    - exercise_dates: Array of exercise times [t₁, t₂, ..., tₙ]
    
    Returns:
    - price: Option value
    - std_error: Standard error
    - exercise_info: Dict with exercise boundary info
    """
    n_dates = len(exercise_dates)
    dt_vec = np.diff(np.concatenate(([0], exercise_dates)))
    
    # Generate paths (forward simulation)
    paths = np.zeros((n_paths, n_dates))
    S = S0 * np.ones(n_paths)
    
    for i in range(n_dates):
        dt = dt_vec[i]
        Z = np.random.randn(n_paths)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        paths[:, i] = S
    
    # Initialize cash flows at maturity
    cash_flows = np.maximum(paths[:, -1] - K, 0)
    exercise_indicators = np.zeros((n_paths, n_dates), dtype=bool)
    exercise_indicators[:, -1] = cash_flows > 0
    
    # Backward induction
    for i in range(n_dates - 2, -1, -1):
        dt_forward = exercise_dates[i + 1] - exercise_dates[i]
        discount = np.exp(-r * dt_forward)
        
        # Discount future cash flows
        cash_flows = discount * cash_flows
        
        # Current stock prices
        S_current = paths[:, i]
        
        # Intrinsic value
        intrinsic = np.maximum(S_current - K, 0)
        
        # Regression on ITM paths only
        itm_mask = intrinsic > 0
        
        if np.sum(itm_mask) > 10:  # Need enough ITM paths
            X_itm = S_current[itm_mask].reshape(-1, 1)
            Y_itm = cash_flows[itm_mask]
            
            # Polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_itm)
            
            # Regression
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_poly, Y_itm)
            
            # Continuation value
            X_all_poly = poly.transform(S_current.reshape(-1, 1))
            continuation = reg.predict(X_all_poly)
        else:
            continuation = cash_flows  # Not enough ITM, just continue
        
        # Exercise decision
        exercise = (intrinsic > continuation) & (intrinsic > 0)
        exercise_indicators[:, i] = exercise
        
        # Update cash flows
        cash_flows[exercise] = intrinsic[exercise]
    
    # Discount to t=0
    price = np.mean(np.exp(-r * exercise_dates[0]) * cash_flows)
    std_error = np.std(np.exp(-r * exercise_dates[0]) * cash_flows) / np.sqrt(n_paths)
    
    # Exercise boundary approximation
    exercise_boundary = []
    for i in range(n_dates):
        exercised_paths = exercise_indicators[:, i]
        if np.any(exercised_paths):
            boundary = np.percentile(paths[exercised_paths, i], 50)  # Median
            exercise_boundary.append(boundary)
        else:
            exercise_boundary.append(np.nan)
    
    exercise_info = {
        'boundary': exercise_boundary,
        'exercise_dates': exercise_dates,
        'paths': paths,
        'indicators': exercise_indicators
    }
    
    return price, std_error, exercise_info

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.30

print("="*80)
print("BERMUDAN OPTION PRICING")
print("="*80)
print(f"S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%\n")

# Benchmark prices
euro_price = european_call(S0, K, T, r, sigma)
american_price = binomial_american_call(S0, K, T, r, sigma, N=500)

print(f"European Call: ${euro_price:.6f}")
print(f"American Call: ${american_price:.6f} (binomial 500 steps)")
print(f"Early Exercise Premium: ${american_price - euro_price:.6f}\n")

# Test different exercise frequencies
np.random.seed(42)
n_paths = 50000

frequencies = {
    'Quarterly (N=4)': np.linspace(T/4, T, 4),
    'Monthly (N=12)': np.linspace(T/12, T, 12),
    'Weekly (N=52)': np.linspace(T/52, T, 52),
    'Daily (N=252)': np.linspace(T/252, T, 252)
}

print("="*80)
print("EXERCISE FREQUENCY IMPACT")
print("="*80)

bermudan_prices = {}
bermudan_errors = {}

for label, dates in frequencies.items():
    price, error, _ = lsm_bermudan_call(S0, K, T, r, sigma, dates, n_paths, degree=2)
    bermudan_prices[label] = price
    bermudan_errors[label] = error
    
    pct_of_premium = (price - euro_price) / (american_price - euro_price) * 100
    
    print(f"{label}:")
    print(f"  Price: ${price:.6f} ± ${error:.6f}")
    print(f"  % of Early Exercise Premium: {pct_of_premium:.1f}%\n")

# Detailed analysis: Quarterly exercise
print("="*80)
print("DETAILED ANALYSIS (Quarterly Exercise)")
print("="*80)

np.random.seed(42)
exercise_dates = np.array([0.25, 0.5, 0.75, 1.0])
price, error, exercise_info = lsm_bermudan_call(S0, K, T, r, sigma, exercise_dates, n_paths, degree=3)

print(f"Bermudan Call Price: ${price:.6f} ± ${error:.6f}")
print(f"\nExercise Boundary (median of exercised paths):")
for i, (t, B) in enumerate(zip(exercise_dates, exercise_info['boundary'])):
    if not np.isnan(B):
        print(f"  t={t:.2f}yr: S ≥ ${B:.2f}")
    else:
        print(f"  t={t:.2f}yr: No early exercise")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Sample paths with exercise points
ax = axes[0, 0]
n_plot = 30
paths = exercise_info['paths']
indicators = exercise_info['indicators']

for m in range(n_plot):
    ax.plot(exercise_dates, paths[m, :], 'b-', alpha=0.3, linewidth=1)
    
    # Mark exercise points
    exercised = indicators[m, :]
    if np.any(exercised):
        ex_idx = np.where(exercised)[0][0]  # First exercise
        ax.plot(exercise_dates[ex_idx], paths[m, ex_idx], 'ro', markersize=6)

ax.axhline(K, color='orange', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Sample Paths with Exercise Points (red dots)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Exercise boundary
ax = axes[0, 1]
boundary = exercise_info['boundary']
valid_boundary = [B for B in boundary if not np.isnan(B)]
valid_dates = [t for t, B in zip(exercise_dates, boundary) if not np.isnan(B)]

if len(valid_boundary) > 0:
    ax.plot(valid_dates, valid_boundary, 'go-', linewidth=2, markersize=8, label='Exercise Boundary')
ax.axhline(K, color='red', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Exercise Boundary S*(t)')
ax.set_title('Optimal Exercise Boundary (median of exercised paths)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Exercise frequency histogram
ax = axes[0, 2]
exercise_time_idx = np.argmax(indicators, axis=1)
exercise_times = exercise_dates[exercise_time_idx]

# Only count paths that actually exercised early
early_exercise_mask = np.any(indicators[:, :-1], axis=1)
if np.sum(early_exercise_mask) > 0:
    ax.hist(exercise_times[early_exercise_mask], bins=len(exercise_dates),
           alpha=0.7, color='purple', edgecolor='black')

ax.set_xlabel('Exercise Time (years)')
ax.set_ylabel('Number of Paths')
ax.set_title('Distribution of Exercise Times')
ax.grid(True, alpha=0.3)

# Plot 4: Value vs exercise frequency
ax = axes[1, 0]
frequencies_list = [4, 12, 52, 252]
freq_prices = [bermudan_prices[label] for label in bermudan_prices.keys()]

ax.plot(frequencies_list, freq_prices, 'bo-', linewidth=2, markersize=8, label='Bermudan')
ax.axhline(euro_price, color='green', linestyle='--', linewidth=2, label='European')
ax.axhline(american_price, color='red', linestyle='--', linewidth=2, label='American (binomial)')
ax.set_xlabel('Number of Exercise Dates')
ax.set_ylabel('Option Price ($)')
ax.set_title('Bermudan Price vs Exercise Frequency')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Plot 5: Premium captured vs frequency
ax = axes[1, 1]
total_premium = american_price - euro_price
pct_captured = [(p - euro_price) / total_premium * 100 for p in freq_prices]

ax.plot(frequencies_list, pct_captured, 'go-', linewidth=2, markersize=8)
ax.axhline(100, color='red', linestyle='--', linewidth=2, label='American (100%)')
ax.set_xlabel('Number of Exercise Dates')
ax.set_ylabel('% of Early Exercise Premium Captured')
ax.set_title('Convergence to American Option Value')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Plot 6: Comparison bar chart
ax = axes[1, 2]
labels = ['European', 'Quarterly\n(N=4)', 'Monthly\n(N=12)', 'Weekly\n(N=52)', 'American\n(binomial)']
prices = [euro_price] + freq_prices[:-1] + [american_price]
colors = ['green', 'blue', 'blue', 'blue', 'red']

bars = ax.bar(labels, prices, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Option Price ($)')
ax.set_title('Price Hierarchy')
ax.grid(True, alpha=0.3, axis='y')

# Annotate values
for bar, price in zip(bars, prices):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${price:.4f}',
           ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('bermudan_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Convergence analysis
print("\n" + "="*80)
print("CONVERGENCE TO AMERICAN OPTION")
print("="*80)
print(f"European: ${euro_price:.6f} (0.00% of premium)")
for label, price in bermudan_prices.items():
    pct = (price - euro_price) / (american_price - euro_price) * 100
    print(f"{label}: ${price:.6f} ({pct:.1f}% of premium)")
print(f"American: ${american_price:.6f} (100.00% of premium)")