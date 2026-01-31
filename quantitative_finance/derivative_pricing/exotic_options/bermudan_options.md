# Bermudan Options

## 1. Concept Skeleton
**Definition:** Options exercisable on specific discrete dates between European (one date) and American (continuous)  
**Purpose:** Practical early exercise approximation; computational efficiency; match contract specifications (quarterly exercise)  
**Prerequisites:** Dynamic programming, regression-based methods, Longstaff-Schwartz algorithm, optimal stopping theory

## 2. Comparative Framing
| Feature | Bermudan | American | European | Basket |
|---------|----------|----------|----------|--------|
| **Exercise Dates** | N discrete dates | Continuous (any time) | One date (maturity) | One date |
| **Payoff** | max(S - K, 0) at exercise | max(S - K, 0) any time | max(S_T - K, 0) | max(Basket - K, 0) |
| **Pricing** | Longstaff-Schwartz | Binomial/FD/LSM | Black-Scholes | Monte Carlo |
| **Complexity** | O(N dates × M paths) | O(√ε) grid density | O(1) | O(N assets) |
| **Value** | European ≤ Bermudan ≤ American | Always | Baseline | Correlation-dependent |

## 3. Examples + Counterexamples

**Simple Example:**  
Bermudan call on stock with quarterly exercise: T=1yr, exercise dates at t=0.25, 0.5, 0.75, 1.0 → 4 opportunities

**Failure Case:**  
Exercise dates after ex-dividend dates: Early exercise valuable for American, but Bermudan can't exercise → loses dividend capture value → cheaper

**Edge Case:**  
High exercise frequency (N → ∞): Bermudan → American limit; computational cost explodes without benefit

## 4. Layer Breakdown
```
Bermudan Option Pricing Pipeline:
├─ Exercise Schedule:
│   ├─ Dates: t₁ < t₂ < ... < tₙ = T (N exercise opportunities)
│   ├─ European: N=1 (only T); American: N=∞ (continuous)
│   ├─ Practical: N=4-12 (monthly/quarterly)
│   └─ Δt = tᵢ₊₁ - tᵢ (interval between exercise dates)
├─ Optimal Exercise Policy:
│   ├─ At each date tᵢ: Compare intrinsic vs continuation
│   ├─ Intrinsic Value IV(tᵢ): max(S_tᵢ - K, 0) (immediate payoff)
│   ├─ Continuation Value CV(tᵢ): E[V(tᵢ₊₁) | S_tᵢ] (hold value)
│   └─ Exercise if IV(tᵢ) > CV(tᵢ) else continue
├─ Longstaff-Schwartz for Bermudan:
│   ├─ Forward Simulation:
│   │   ├─ Generate M paths of asset prices
│   │   ├─ S^m_tᵢ for m=1..M paths, i=1..N dates
│   │   └─ Store entire M×N matrix of prices
│   ├─ Backward Induction (from tₙ to t₁):
│   │   ├─ Initialize at maturity tₙ:
│   │   │   └─ CF^m = max(S^m_tₙ - K, 0) for all m
│   │   ├─ For each earlier date tᵢ (i = N-1 down to 1):
│   │   │   ├─ Filter ITM paths: I = {m : S^m_tᵢ > K}
│   │   │   ├─ Regression (on ITM paths only):
│   │   │   │   ├─ X = [1, S^m_tᵢ, (S^m_tᵢ)², ...] (basis functions)
│   │   │   │   ├─ Y = e^(-r(tᵢ₊₁-tᵢ)) CF^m (discounted future CF)
│   │   │   │   └─ β̂ = (X^T X)^{-1} X^T Y (OLS)
│   │   │   ├─ Continuation Value:
│   │   │   │   └─ CV^m = X^m β̂ (fitted value from regression)
│   │   │   ├─ Exercise Decision:
│   │   │   │   ├─ IV^m = max(S^m_tᵢ - K, 0)
│   │   │   │   └─ If IV^m > CV^m: Exercise now
│   │   │   ├─ Update Cash Flows:
│   │   │   │   ├─ If exercise: CF^m ← IV^m
│   │   │   │   └─ If continue: Keep CF^m = future payoff
│   │   │   └─ Record exercise indicator: Exercise^m_tᵢ ∈ {0,1}
│   │   └─ Discount back: CF^m ← e^(-r(tᵢ-tᵢ₋₁)) CF^m
│   └─ Final Value: V₀ = (1/M) Σ e^(-rt₁) CF^m
├─ Pricing Algorithms:
│   ├─ Monte Carlo + LSM:
│   │   ├─ Pros: Handles high dimensions, path-dependent features
│   │   ├─ Cons: Regression error, needs many paths (M ≥ 50k)
│   │   └─ Convergence: Slow (O(1/√M)), biased (low bias ~1%)
│   ├─ Binomial Trees:
│   │   ├─ Natural for discrete exercise dates
│   │   ├─ Backward induction: V(tᵢ, S) = max(IV, e^(-rΔt) E[V(tᵢ₊₁)])
│   │   └─ Accurate but exponential in dimensions (curse of dimensionality)
│   ├─ Finite Difference Methods:
│   │   ├─ PDE approach: ∂V/∂t + LV = 0 with free boundary
│   │   ├─ Exercise boundary: B(tᵢ) where IV = CV
│   │   └─ Grid-based: Accurate for low dimensions
│   └─ Dynamic Programming:
│       └─ Value iteration on exercise dates backward
├─ Exercise Boundary:
│   ├─ Optimal Boundary B(t): Stock price where IV = CV
│   ├─ Exercise Region: {S : S > B(t)} (for calls)
│   ├─ Properties:
│   │   ├─ B(T) = K (at maturity, always exercise if ITM)
│   │   ├─ B(t) increasing in t (more likely to exercise later)
│   │   └─ B(t) ≥ K always (never exercise deep OTM)
│   └─ Approximation: Fit polynomial to regression boundary
├─ Convergence & Bias:
│   ├─ Upward Bias: Using same paths for exercise decision & valuation
│   ├─ Mitigation:
│   │   ├─ Fresh paths: Generate new paths for final pricing
│   │   ├─ Cross-validation: Split into training & test sets
│   │   └─ Dual approach: Primal (LSM) gives lower bound, dual gives upper bound
│   ├─ Path Requirements: M ≥ 50k for stable regression
│   └─ Exercise Dates: More dates → better American approximation but slower
├─ Greeks:
│   ├─ Delta: ∂V/∂S (via pathwise derivatives or finite differences)
│   ├─ Gamma: ∂²V/∂S² (unstable near exercise boundary)
│   ├─ Vega: ∂V/∂σ (higher than European, lower than American)
│   ├─ Theta: ∂V/∂T (discontinuous at exercise dates)
│   └─ Rho: ∂V/∂r (affects both drift and discounting)
└─ Practical Considerations:
    ├─ Exercise Frequency:
    │   ├─ Monthly (N=12): Good balance between value and cost
    │   ├─ Quarterly (N=4): Common for equity options
    │   └─ Daily (N=252): Approaches American but computationally expensive
    ├─ Typical Applications:
    │   ├─ Swaptions: Exercise on swap payment dates
    │   ├─ Callable Bonds: Redemption on coupon dates
    │   ├─ Employee Stock Options: Vesting schedule
    │   └─ Real Options: Project go/no-go decisions at milestones
    └─ Value Hierarchy: V_European ≤ V_Bermudan ≤ V_American
        └─ Difference: V_American - V_Bermudan ~1-5% for typical parameters
```

**Interaction:** Generate forward paths → Backward regression at each exercise date → Compare IV vs CV → Update cash flows if early exercise optimal → Discount to present

## 5. Mini-Project
Price Bermudan call with varying exercise frequency and compare to American/European:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
```

## 6. Challenge Round

**Q1:** Why is Bermudan cheaper than American? Quantify the difference.  
**A1:** American allows exercise anytime → more flexibility → higher value. Bermudan only exercises at discrete dates → may miss optimal exercise timing. Difference: Typically 1-5% of option value. For quarterly exercise (N=4), captures 80-90% of early exercise premium vs American. Converges as N → ∞.

**Q2:** When does Bermudan ≈ European (no early exercise)?  
**A2:** Non-dividend paying stock: Early exercise of call suboptimal (time value > intrinsic always). Deep OTM: Continuation value > intrinsic at all exercise dates. Very short maturity: Little time value to sacrifice → no incentive to wait, but also little gain from early exercise.

**Q3:** Exercise boundary behavior: Why B(t) increasing in t for calls?  
**A3:** Near maturity: Low time value → exercise more readily → lower boundary. Far from maturity: High time value → hold longer → higher boundary needed to exercise. B(T) = K (at maturity, exercise all ITM). Mathematically: ∂B/∂t > 0 from optimal stopping theory.

**Q4:** Regression degree choice in LSM: Why not always use high degree?  
**A4:** Low degree (1-2): Underfitting, poor CV approximation → overexercise → underpriced. High degree (5+): Overfitting, noisy CV → spurious exercise decisions → instability. Optimal: degree=2-3 for single asset (balance bias-variance tradeoff). More assets → higher degree may help.

**Q5:** Dual method for upper bound: How does it work?  
**A5:** LSM gives lower bound (suboptimal policy). Dual: Use any exercise policy π to compute upper bound via martingale stopping. V ≤ E[e^{-rt} max(S_t - K, 0) - M_t] where M_t is martingale penalizing suboptimality. Tight bounds: Upper - Lower < 1% → confidence in price.

**Q6:** Bermudan swaption: Exercise on swap payment dates. Why Bermudan not American?  
**A6:** Swap starts on exercise date → only sensible to exercise on payment dates (quarterly/semi-annually). Exercising between payments captures no additional value. Contract specification: Exercise rights only on coupon dates. Computational: Bermudan much faster than American for multi-factor interest rate models.

**Q7:** Compare Bermudan vs American for dividends. Which matters more?  
**A7:** Dividends increase early exercise value (stock drops on ex-date). American: Exercise just before ex-dividend if dividend > time value. Bermudan: Only if ex-date coincides with exercise date → may miss optimal timing. Difference significant (5-10%) if large dividend between exercise dates.

**Q8:** Path-dependent Bermudan (e.g., Asian payoff at exercise): Complexity increase?  
**A8:** Need path state variable (e.g., running average) in regression. Basis functions: f(S, A) where A = average-to-date. Higher dimensions → more paths needed (M ≥ 100k). Curse of dimensionality: Regression accuracy degrades. Alternatives: Factor models, dimension reduction (PCA).

## 7. Key References

**Primary Sources:**
- [Bermudan Option Wikipedia](https://en.wikipedia.org/wiki/Bermudan_option) - Definition and applications
- Longstaff, F.A. & Schwartz, E.S. "Valuing American Options by Simulation" (2001) - LSM algorithm

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - Bermudan pricing (pp. 449-472)
- Andersen, L. & Broadie, M. "Primal-Dual Simulation Algorithm" (2004) - Dual upper bounds

**Thinking Steps:**
1. Define exercise schedule (quarterly, monthly, etc.)
2. Generate forward paths for all stock prices at exercise dates
3. Backward induction: At each date, regress continuation value on ITM paths
4. Compare intrinsic vs continuation → exercise if intrinsic > continuation
5. Update cash flows and discount back to present
