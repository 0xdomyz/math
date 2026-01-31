# Implied Volatility

## 1. Concept Skeleton
**Definition:** Volatility parameter making Black-Scholes price equal observed market option price; inverts pricing formula to extract market's volatility expectation  
**Purpose:** Convert option prices to volatility units for comparison across strikes/maturities; reveal supply/demand imbalances; identify mispricing opportunities  
**Prerequisites:** Black-Scholes model, option Greeks, numerical root-finding (Newton-Raphson), volatility concepts, arbitrage bounds

## 2. Comparative Framing
| Method | Newton-Raphson | Bisection | Analytical Approx | Direct Formula |
|--------|----------------|-----------|-------------------|----------------|
| **Speed** | Fast (quadratic) | Slow (linear) | Very fast | Instant |
| **Robustness** | Can fail (bad guess) | Always converges | Less accurate | Limited range |
| **Complexity** | Requires Vega | Simple bracketing | Complex math | Only approximation |
| **Accuracy** | Exact | Exact | ~0.01 vol error | ~0.1 vol error |

| Phenomenon | Volatility Smile | Term Structure | Sticky Strike | Sticky Delta |
|------------|------------------|----------------|---------------|--------------|
| **Pattern** | U-shaped IV vs K | IV vs maturity | IV fixed at K | IV fixed at Δ |
| **Cause** | Fat tails, jumps | Mean reversion | Supply/demand | Hedging flows |
| **Impact** | Skew risk | Calendar spreads | Post-move IV | Rehedging IV |
| **Model** | Local vol | Stochastic vol | Market convention | Trader behavior |

## 3. Examples + Counterexamples

**Simple Example:**  
Market call price $5, BS with σ=20% gives $4.80. Increase σ to 21.5% → BS price=$5.00. Implied vol=21.5%.

**Perfect Fit:**  
ATM options with high liquidity: IV converges rapidly with Newton-Raphson (3-4 iterations). Vega large and stable, excellent numerical behavior.

**Volatility Smile:**  
Equity index: OTM puts (K=90) have IV=25%, ATM (K=100) has IV=20%, OTM calls (K=110) have IV=22%. Smile reflects crash risk (left tail fat).

**Volatility Term Structure:**  
Front-month IV=30% (earnings event), 3-month IV=22% (mean reversion), 1-year IV=20% (long-run average). Term structure flattens after event.

**Deep OTM Failure:**  
Far OTM option (Δ=0.01) with price=$0.02: Vega≈0, Newton-Raphson unstable. Bisection more reliable but slow. Analytical bounds needed.

**Poor Fit:**  
American options: IV solver uses European BS but American worth more due to early exercise → solved IV too high, doesn't represent true volatility expectation.

## 4. Layer Breakdown
```
Implied Volatility Framework:

├─ Mathematical Foundation:
│  ├─ Inverse Problem: Given C_market, find σ such that:
│  │   BS(S, K, r, T, σ) = C_market
│  ├─ Non-closed form: No analytical solution for σ
│  ├─ Monotonicity: ∂C/∂σ > 0 (Vega always positive)
│  │   → Unique solution exists if C_market in valid range
│  ├─ Bounds: Check arbitrage bounds first:
│  │   ├─ Lower: C ≥ max(S - K e^(-rT), 0)
│  │   ├─ Upper: C ≤ S
│  │   └─ Invalid prices → No valid IV exists
│  └─ Domain: σ ∈ (0, ∞), practically σ ∈ [0.01, 5.0]
├─ Numerical Methods:
│  ├─ Newton-Raphson (Standard Approach):
│  │   ├─ Iteration: σ_{n+1} = σ_n - (BS(σ_n) - C_market) / Vega(σ_n)
│  │   ├─ Convergence: Quadratic (doubles digits each iteration)
│  │   ├─ Initial guess: Critical for success
│  │   │   ├─ Simple: σ_0 = 0.20 (20%)
│  │   │   ├─ Better: √(2π/T) × C / S (Brenner-Subrahmanyam)
│  │   │   └─ Adjacent strike IV (for interpolation)
│  │   ├─ Stopping criterion: |σ_{n+1} - σ_n| < ε (e.g., 1e-6)
│  │   ├─ Max iterations: Typically 10-20 sufficient
│  │   ├─ Advantages: Fast convergence, industry standard
│  │   └─ Disadvantages: Requires Vega, can diverge if bad guess
│  ├─ Bisection Method (Robust Fallback):
│  │   ├─ Bracket: [σ_low, σ_high] where BS(σ_low) < C < BS(σ_high)
│  │   ├─ Iteration: σ_mid = (σ_low + σ_high) / 2
│  │   │   If BS(σ_mid) < C: σ_low = σ_mid
│  │   │   If BS(σ_mid) > C: σ_high = σ_mid
│  │   ├─ Convergence: Linear (halves interval each step)
│  │   ├─ Advantages: Always converges, no derivatives needed
│  │   └─ Disadvantages: Slower than Newton-Raphson
│  ├─ Analytical Approximations:
│  │   ├─ Brenner-Subrahmanyam (ATM, short maturity):
│  │   │   σ ≈ √(2π/T) × (C/S)
│  │   ├─ Corrado-Miller (improved accuracy):
│  │   │   Includes higher-order terms for better fit
│  │   ├─ Use case: Fast initial guess or rough estimate
│  │   └─ Error: ~0.01-0.1 in vol units
│  └─ Hybrid Approach:
│      ├─ Start with analytical guess
│      ├─ Newton-Raphson for 3-5 iterations
│      └─ Fall back to bisection if diverges
├─ Volatility Surface:
│  ├─ Definition: IV(K, T) across all strikes and maturities
│  ├─ Dimensions:
│  │   ├─ Strike axis: Moneyness (K/S or K/F)
│  │   ├─ Maturity axis: Time to expiry T
│  │   └─ IV value: Height of surface
│  ├─ Smile/Skew:
│  │   ├─ Equity: Negative skew (put IV > call IV)
│  │   │   → Crash protection premium
│  │   ├─ FX: Symmetric smile (straddle more expensive)
│  │   │   → Currency can move either direction
│  │   ├─ Commodities: Varies by market structure
│  │   └─ Causes: Jump risk, leverage effect, supply/demand
│  ├─ Term Structure:
│  │   ├─ Upward sloping: Mean reversion expected
│  │   ├─ Downward sloping: Event risk (earnings, elections)
│  │   ├─ Humped: Near-term event, long-term reversion
│  │   └─ Drivers: Supply/demand, hedging flows, calendar effects
│  ├─ Interpolation/Extrapolation:
│  │   ├─ Strike interpolation: Cubic spline, SABR, SVI
│  │   ├─ Time interpolation: Variance interpolation (linear in σ²T)
│  │   ├─ Arbitrage-free constraints: Butterfly, calendar spreads
│  │   └─ Extrapolation: Flatten wings, avoid negative densities
│  └─ Surface Dynamics:
│      ├─ Sticky strike: IV stays at strike level (convention)
│      ├─ Sticky delta: IV moves with option's delta (hedger view)
│      ├─ Sticky moneyness: IV at K/S (hybrid)
│      └─ Reality: Combination depending on market conditions
├─ Applications:
│  ├─ Option Pricing:
│  │   ├─ Quote in vol terms: "25-delta put at 22 vol"
│  │   ├─ Trader language: More intuitive than dollar prices
│  │   └─ Standardization: Compare across strikes/underlyings
│  ├─ Arbitrage Detection:
│  │   ├─ Butterfly arbitrage: Check IV convexity
│  │   │   (IV_K1 + IV_K3) / 2 should ≥ IV_K2
│  │   ├─ Calendar arbitrage: Check variance increasing in time
│  │   │   σ₁²T₁ ≤ σ₂²T₂ for T₁ < T₂
│  │   └─ Put-call parity violations: IV_call ≠ IV_put at same K
│  ├─ Relative Value Trading:
│  │   ├─ Rich/cheap analysis: Compare IV to historical levels
│  │   ├─ Cross-strike dispersion: Buy low IV, sell high IV
│  │   ├─ Term structure trades: Calendar spreads
│  │   └─ Vol surface arbitrage: Complex multi-leg strategies
│  ├─ Risk Management:
│  │   ├─ Vega bucketing: By strike/maturity
│  │   ├─ Volatility Greeks: Vanna (∂Δ/∂σ), Volga (∂ν/∂σ)
│  │   ├─ Scenario analysis: Parallel shift, twist, skew change
│  │   └─ VaR/Expected Shortfall: Using IV for mark-to-market
│  └─ Model Calibration:
│      ├─ Extract parameters: Fit local vol, stochastic vol models
│      ├─ Objective: Minimize (Model_IV - Market_IV)²
│      ├─ Weights: By vega, liquidity, bid-ask spread
│      └─ Regularization: Smooth parameter evolution
├─ Volatility Indices (VIX):
│  ├─ VIX Calculation:
│  │   ├─ Model-free: Uses strip of OTM options
│  │   ├─ Formula: σ² = (2/T) Σ (ΔK/K²) e^(rT) Q(K)
│  │   │   where Q(K) = option mid-price
│  │   ├─ Weights: Inverse square of strike
│  │   └─ Result: 30-day expected volatility
│  ├─ Interpretation:
│  │   ├─ VIX = 20: Market expects ~20% annual vol
│  │   ├─ VIX = 40: Crisis levels (2008, 2020)
│  │   └─ Term structure: VIX vs VXV (3-month)
│  ├─ Trading:
│  │   ├─ VIX futures: Cash-settled on VIX level
│  │   ├─ VIX options: European, expire to VIX future
│  │   └─ ETFs: VXX, UVXY (roll futures, contango drag)
│  └─ "Vol of vol": Volatility of implied volatility itself
├─ Advanced Topics:
│  ├─ Implied Volatility of Implied Volatility:
│  │   Options on VIX → Second-order vol expectations
│  ├─ Correlation Surface:
│  │   Implied correlation from multi-asset options
│  ├─ Dividends:
│  │   Adjust for known dividends in IV calculation
│  │   Use dividend-adjusted forward price F = S e^((r-q)T)
│  ├─ American Options:
│  │   Approximate: Use binomial tree for American IV
│  │   Faster: Barone-Adesi-Whaley approximation
│  └─ Model Risk:
│      IV assumes BS framework → Errors if reality differs
└─ Practical Considerations:
   ├─ Market Data Quality:
   │   ├─ Stale quotes: Use bid-ask midpoint carefully
   │   ├─ Illiquid options: Wide spreads → noisy IV
   │   ├─ Pinning: IV collapses near expiry at popular strikes
   │   └─ Early exercise: Use American pricing for puts
   ├─ Numerical Stability:
   │   ├─ Near expiry: T→0 causes numerical issues
   │   ├─ Deep OTM: Vega→0, Newton-Raphson fails
   │   ├─ Extreme strikes: Check bounds before solving
   │   └─ Error handling: Return NaN or error code gracefully
   ├─ Performance:
   │   ├─ Vectorization: Solve entire surface in parallel
   │   ├─ Caching: Store IV, recalculate only on price update
   │   ├─ Approximations: Use for real-time quotes
   │   └─ GPU acceleration: For large-scale calibration
   └─ Conventions:
      ├─ Quote convention: Vol as %, e.g., "22 vol" = 22%
      ├─ Day count: Actual/365 or Actual/360
      ├─ Business days: Trading days vs calendar days
      └─ Settlement: T+1 or T+2 affects forward price
```

**Interaction:** Market price → IV solver (Newton-Raphson) → Volatility surface → Trading signals; IV surface feeds back into pricing exotic options and risk management.

## 5. Mini-Project
Implement implied volatility solver with multiple methods:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, newton
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("IMPLIED VOLATILITY CALCULATION AND ANALYSIS")
print("="*60)

class BlackScholes:
    """Black-Scholes pricing and Greeks"""
    
    @staticmethod
    def d1(S, K, r, T, sigma):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, r, T, sigma):
        return BlackScholes.d1(S, K, r, T, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, r, T, sigma):
        d1 = BlackScholes.d1(S, K, r, T, sigma)
        d2 = BlackScholes.d2(S, K, r, T, sigma)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, r, T, sigma):
        d1 = BlackScholes.d1(S, K, r, T, sigma)
        d2 = BlackScholes.d2(S, K, r, T, sigma)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    @staticmethod
    def vega(S, K, r, T, sigma):
        d1 = BlackScholes.d1(S, K, r, T, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)

class ImpliedVolatility:
    """Implied volatility solvers"""
    
    @staticmethod
    def newton_raphson(S, K, r, T, market_price, option_type='call', 
                       initial_guess=0.2, max_iter=100, tol=1e-6):
        """Newton-Raphson method using Vega"""
        sigma = initial_guess
        
        for i in range(max_iter):
            if option_type == 'call':
                price = BlackScholes.call_price(S, K, r, T, sigma)
            else:
                price = BlackScholes.put_price(S, K, r, T, sigma)
            
            vega = BlackScholes.vega(S, K, r, T, sigma)
            
            diff = price - market_price
            
            if abs(diff) < tol:
                return sigma, i+1  # Converged
            
            if vega < 1e-10:  # Avoid division by very small number
                return np.nan, i+1
            
            sigma = sigma - diff / vega
            
            # Keep sigma positive and reasonable
            sigma = max(0.001, min(sigma, 5.0))
        
        return np.nan, max_iter  # Failed to converge
    
    @staticmethod
    def bisection(S, K, r, T, market_price, option_type='call', 
                  sigma_low=0.001, sigma_high=5.0, tol=1e-6, max_iter=100):
        """Bisection method - robust but slower"""
        
        for i in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            
            if option_type == 'call':
                price = BlackScholes.call_price(S, K, r, T, sigma_mid)
            else:
                price = BlackScholes.put_price(S, K, r, T, sigma_mid)
            
            if abs(price - market_price) < tol:
                return sigma_mid, i+1
            
            if price < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
            
            if sigma_high - sigma_low < tol:
                return sigma_mid, i+1
        
        return np.nan, max_iter
    
    @staticmethod
    def brenner_subrahmanyam(S, K, r, T, market_price):
        """Analytical approximation for ATM short-maturity"""
        return np.sqrt(2 * np.pi / T) * (market_price / S)
    
    @staticmethod
    def check_arbitrage_bounds(S, K, r, T, market_price, option_type='call'):
        """Check if price satisfies no-arbitrage bounds"""
        if option_type == 'call':
            lower_bound = max(S - K * np.exp(-r*T), 0)
            upper_bound = S
        else:
            lower_bound = max(K * np.exp(-r*T) - S, 0)
            upper_bound = K * np.exp(-r*T)
        
        return lower_bound <= market_price <= upper_bound

# Scenario 1: Basic IV calculation
print("\n" + "="*60)
print("SCENARIO 1: Basic Implied Volatility Calculation")
print("="*60)

S, K, r, T = 100, 100, 0.05, 0.25
true_sigma = 0.25

# Generate market price
market_call = BlackScholes.call_price(S, K, r, T, true_sigma)

print(f"\nParameters: S=${S}, K=${K}, r={r:.1%}, T={T}yr")
print(f"True volatility: {true_sigma:.1%}")
print(f"Market call price: ${market_call:.4f}")

# Solve using Newton-Raphson
iv_nr, iter_nr = ImpliedVolatility.newton_raphson(S, K, r, T, market_call, 'call')
print(f"\nNewton-Raphson:")
print(f"  Implied Vol: {iv_nr:.6f} ({iv_nr*100:.4f}%)")
print(f"  Iterations: {iter_nr}")
print(f"  Error: {abs(iv_nr - true_sigma):.8f}")

# Solve using Bisection
iv_bis, iter_bis = ImpliedVolatility.bisection(S, K, r, T, market_call, 'call')
print(f"\nBisection:")
print(f"  Implied Vol: {iv_bis:.6f} ({iv_bis*100:.4f}%)")
print(f"  Iterations: {iter_bis}")
print(f"  Error: {abs(iv_bis - true_sigma):.8f}")

# Analytical approximation
iv_approx = ImpliedVolatility.brenner_subrahmanyam(S, K, r, T, market_call)
print(f"\nBrenner-Subrahmanyam Approximation:")
print(f"  Implied Vol: {iv_approx:.6f} ({iv_approx*100:.4f}%)")
print(f"  Error: {abs(iv_approx - true_sigma):.8f}")

# Scenario 2: Performance comparison
print("\n" + "="*60)
print("SCENARIO 2: Performance Comparison")
print("="*60)

n_trials = 1000
strikes = np.random.uniform(80, 120, n_trials)
times = np.random.uniform(0.1, 2.0, n_trials)

print(f"\nComparing methods on {n_trials} random options:")

# Newton-Raphson timing
start = time.time()
for i in range(n_trials):
    market_price = BlackScholes.call_price(S, strikes[i], r, times[i], 0.25)
    iv, _ = ImpliedVolatility.newton_raphson(S, strikes[i], r, times[i], market_price, 'call')
nr_time = time.time() - start

# Bisection timing
start = time.time()
for i in range(n_trials):
    market_price = BlackScholes.call_price(S, strikes[i], r, times[i], 0.25)
    iv, _ = ImpliedVolatility.bisection(S, strikes[i], r, times[i], market_price, 'call')
bis_time = time.time() - start

print(f"\nNewton-Raphson: {nr_time*1000:.2f} ms ({nr_time/n_trials*1e6:.2f} μs per option)")
print(f"Bisection: {bis_time*1000:.2f} ms ({bis_time/n_trials*1e6:.2f} μs per option)")
print(f"Speedup: {bis_time/nr_time:.1f}x")

# Scenario 3: Volatility smile
print("\n" + "="*60)
print("SCENARIO 3: Volatility Smile Construction")
print("="*60)

S_smile = 100
T_smile = 0.25
r_smile = 0.05

# Strikes from deep OTM put to deep OTM call
strikes_smile = np.linspace(80, 120, 21)

# True volatility smile (stylized equity smile)
def true_vol_smile(K, S, T):
    """Stylized volatility smile - downward sloping"""
    moneyness = np.log(K/S)
    base_vol = 0.20
    skew = -0.15  # Negative skew for equities
    curvature = 0.05
    return base_vol + skew * moneyness + curvature * moneyness**2

implied_vols = []
deltas = []

print(f"\nConstructing smile (S=${S_smile}, T={T_smile}yr):")
print(f"{'Strike':<10} {'Market Price':<15} {'Impl Vol':<12} {'Delta':<10}")
print("-" * 47)

for K_smile in strikes_smile:
    true_vol = true_vol_smile(K_smile, S_smile, T_smile)
    market_price = BlackScholes.call_price(S_smile, K_smile, r_smile, T_smile, true_vol)
    
    iv, _ = ImpliedVolatility.newton_raphson(S_smile, K_smile, r_smile, T_smile, 
                                             market_price, 'call', initial_guess=0.2)
    
    # Calculate delta
    d1 = BlackScholes.d1(S_smile, K_smile, r_smile, T_smile, iv)
    delta = norm.cdf(d1)
    
    implied_vols.append(iv)
    deltas.append(delta)
    
    if K_smile in [80, 90, 100, 110, 120]:
        print(f"${K_smile:<9} ${market_price:<14.4f} {iv*100:<11.2f}% {delta:<10.4f}")

# Scenario 4: Term structure
print("\n" + "="*60)
print("SCENARIO 4: Volatility Term Structure")
print("="*60)

K_atm = 100
maturities = np.array([1/12, 3/12, 6/12, 1, 2])  # 1m, 3m, 6m, 1y, 2y

# Stylized term structure (downward sloping - event risk)
def term_structure_vol(T):
    """Term structure - mean reverting"""
    short_vol = 0.30  # High near-term vol (event)
    long_vol = 0.18   # Long-term mean reversion
    decay = 2.0
    return long_vol + (short_vol - long_vol) * np.exp(-decay * T)

term_vols = []

print(f"\nATM Volatility Term Structure (K=${K_atm}):")
print(f"{'Maturity':<15} {'Market Price':<15} {'Impl Vol':<12} {'Ann. Variance':<15}")
print("-" * 57)

for T_term in maturities:
    true_vol = term_structure_vol(T_term)
    market_price = BlackScholes.call_price(S, K_atm, r, T_term, true_vol)
    
    iv, _ = ImpliedVolatility.newton_raphson(S, K_atm, r, T_term, market_price, 'call')
    variance = iv**2 * T_term
    
    term_vols.append(iv)
    
    maturity_label = f"{T_term*12:.0f} months" if T_term < 1 else f"{T_term:.1f} years"
    print(f"{maturity_label:<15} ${market_price:<14.4f} {iv*100:<11.2f}% {variance:<15.4f}")

# Scenario 5: Error cases
print("\n" + "="*60)
print("SCENARIO 5: Error Handling and Edge Cases")
print("="*60)

test_cases = [
    ("Valid ATM call", 100, 100, 10.0, 'call', True),
    ("Deep ITM call", 100, 80, 21.0, 'call', True),
    ("Deep OTM call", 100, 130, 0.1, 'call', True),
    ("Price too high", 100, 100, 105.0, 'call', False),  # Violates upper bound
    ("Price too low", 100, 100, -1.0, 'call', False),    # Negative price
    ("Near expiry", 100, 100, 5.0, 'call', True),
]

print(f"\n{'Case':<20} {'Valid Bounds':<15} {'IV (NR)':<15} {'IV (Bisect)':<15}")
print("-" * 65)

T_test = 0.25

for case_name, S_test, K_test, price, opt_type, should_work in test_cases:
    # Check bounds
    bounds_ok = ImpliedVolatility.check_arbitrage_bounds(S_test, K_test, r, T_test, price, opt_type)
    
    if bounds_ok:
        iv_nr, _ = ImpliedVolatility.newton_raphson(S_test, K_test, r, T_test, price, opt_type)
        iv_bis, _ = ImpliedVolatility.bisection(S_test, K_test, r, T_test, price, opt_type)
        
        iv_nr_str = f"{iv_nr*100:.2f}%" if not np.isnan(iv_nr) else "FAILED"
        iv_bis_str = f"{iv_bis*100:.2f}%" if not np.isnan(iv_bis) else "FAILED"
    else:
        iv_nr_str = "N/A"
        iv_bis_str = "N/A"
    
    print(f"{case_name:<20} {'✓' if bounds_ok else '✗':<14} {iv_nr_str:<15} {iv_bis_str:<15}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Volatility smile
ax = axes[0, 0]
ax.plot(strikes_smile, np.array(implied_vols)*100, 'bo-', linewidth=2.5, markersize=8)
ax.axvline(S_smile, color='r', linestyle='--', alpha=0.5, label='ATM')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility Smile (Equity Style)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Smile by delta
ax = axes[0, 1]
ax.plot(deltas, np.array(implied_vols)*100, 'go-', linewidth=2.5, markersize=8)
ax.set_xlabel('Delta')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility by Delta')
ax.grid(alpha=0.3)

# Plot 3: Term structure
ax = axes[0, 2]
ax.plot(maturities*12, np.array(term_vols)*100, 'ro-', linewidth=2.5, markersize=10)
ax.set_xlabel('Maturity (months)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('ATM Volatility Term Structure')
ax.grid(alpha=0.3)

# Plot 4: Convergence comparison
ax = axes[1, 0]
iterations_nr = []
iterations_bis = []
test_strikes = np.linspace(85, 115, 15)

for K_test in test_strikes:
    market_price = BlackScholes.call_price(S, K_test, r, T, 0.25)
    _, iter_n = ImpliedVolatility.newton_raphson(S, K_test, r, T, market_price, 'call')
    _, iter_b = ImpliedVolatility.bisection(S, K_test, r, T, market_price, 'call')
    iterations_nr.append(iter_n)
    iterations_bis.append(iter_b)

ax.plot(test_strikes, iterations_nr, 'b-', linewidth=2.5, marker='o', label='Newton-Raphson')
ax.plot(test_strikes, iterations_bis, 'r-', linewidth=2.5, marker='s', label='Bisection')
ax.set_xlabel('Strike')
ax.set_ylabel('Iterations to Converge')
ax.set_title('Convergence Speed Comparison')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: IV surface (strike vs maturity)
ax = axes[1, 1]
strikes_surf = np.linspace(85, 115, 20)
maturities_surf = np.linspace(0.1, 2, 20)
IV_surface = np.zeros((len(maturities_surf), len(strikes_surf)))

for i, T_surf in enumerate(maturities_surf):
    for j, K_surf in enumerate(strikes_surf):
        true_vol = true_vol_smile(K_surf, S, T_surf) * term_structure_vol(T_surf) / 0.20
        market_price = BlackScholes.call_price(S, K_surf, r, T_surf, true_vol)
        iv, _ = ImpliedVolatility.newton_raphson(S, K_surf, r, T_surf, market_price, 'call')
        IV_surface[i, j] = iv * 100 if not np.isnan(iv) else 20

im = ax.contourf(strikes_surf, maturities_surf*12, IV_surface, levels=15, cmap='viridis')
ax.set_xlabel('Strike')
ax.set_ylabel('Maturity (months)')
ax.set_title('Implied Volatility Surface')
plt.colorbar(im, ax=ax, label='IV (%)')

# Plot 6: Vega profile (showing why Newton-Raphson works)
ax = axes[1, 2]
strikes_vega = np.linspace(70, 130, 50)
vegas = []

for K_vega in strikes_vega:
    vega = BlackScholes.vega(S, K_vega, r, T, 0.25)
    vegas.append(vega)

ax.plot(strikes_vega, vegas, 'purple', linewidth=2.5)
ax.axvline(S, color='r', linestyle='--', alpha=0.5, label='ATM (max Vega)')
ax.set_xlabel('Strike')
ax.set_ylabel('Vega')
ax.set_title('Vega Profile (Why NR Works)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **SVI Parameterization:** Implement Stochastic Volatility Inspired (SVI) model for smile fitting. Ensure no calendar arbitrage. How does it compare to cubic spline?

2. **Jump to Default:** Include credit spread in IV calculation for single-name equity options. How does default risk affect OTM put IVs?

3. **Dividend Impact:** Implement IV solver with discrete dividends (ex-dates within option life). How does dividend affect smile near ex-date?

4. **American IV:** Adapt solver for American options using binomial tree. Compare American vs European IV for ITM puts. When does difference exceed 1 vol point?

5. **VIX Replication:** Implement model-free variance calculation using strip of OTM options. Compare to ATM implied vol. Why do they differ?

## 7. Key References
- [Black & Scholes (1973) - Original Pricing Formula](https://www.jstor.org/stable/1831029)
- [Brenner & Subrahmanyam (1988) - Analytical IV Approximation](https://www.sciencedirect.com/science/article/abs/pii/0378426694900721)
- [Gatheral, The Volatility Surface (Chapter 2-3)](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792529)
- [CBOE VIX White Paper - Model-Free Variance](https://www.cboe.com/micro/vix/vixwhite.pdf)

---
**Status:** Market standard for option quoting | **Complements:** Black-Scholes Model, Greeks, Volatility Surface, Option Trading Strategies
