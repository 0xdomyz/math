# Black-Scholes Closed Form

## 1. Concept Skeleton
**Definition:** Analytical solution for European option prices under geometric Brownian motion with constant volatility  
**Purpose:** Benchmark for option pricing; Greeks computation; market-implied volatility extraction; arbitrage-free valuation  
**Prerequisites:** Stochastic calculus (Ito's lemma), risk-neutral pricing, lognormal distribution, partial differential equations

## 2. Comparative Framing
| Method | Black-Scholes | Binomial Tree | Monte Carlo | Finite Difference |
|--------|---------------|---------------|-------------|-------------------|
| **Computation** | O(1) instant | O(N steps) | O(M paths) | O(N time × K space) |
| **Accuracy** | Exact (under assumptions) | Converges to BS | O(1/√M) error | Discretization error |
| **Flexibility** | European only | American feasible | Exotic payoffs | PDEs, boundaries |
| **Greeks** | Analytical formulas | Finite difference | Pathwise/LR | Implicit in grid |

## 3. Examples + Counterexamples

**Simple Example:**  
S₀=$100, K=$100, σ=20%, r=5%, T=1yr → Call=$10.45; d₁=0.35, d₂=0.15; N(d₁)=0.637

**Failure Case:**  
American put with dividends: BS undervalues (ignores early exercise); use binomial tree or Longstaff-Schwartz

**Edge Case:**  
T → 0: Call → max(S₀ - K, 0), Put → max(K - S₀, 0); d₁, d₂ → ±∞; option converges to intrinsic value

## 4. Layer Breakdown
```
Black-Scholes Derivation & Implementation:
├─ Assumptions:
│   ├─ Frictionless Market: No transaction costs, continuous trading
│   ├─ Constant Parameters: σ, r constant over [0, T]
│   ├─ Lognormal Prices: dS = μS dt + σS dW → S_T ~ lognormal
│   ├─ No Dividends: (Extension: replace S₀ → S₀e^(-qT) for dividend yield q)
│   └─ No Arbitrage: Self-financing replicating portfolio
├─ Risk-Neutral Pricing:
│   ├─ Option Value: V(S, t) = e^(-r(T-t)) E^Q[Payoff(S_T) | S_t = S]
│   ├─ Risk-Neutral Drift: Replace μ → r in asset dynamics
│   └─ Terminal Distribution: ln(S_T) ~ N(ln(S₀) + (r - σ²/2)T, σ²T)
├─ Call Option Formula:
│   ├─ Payoff: C(S_T) = max(S_T - K, 0)
│   ├─ d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
│   ├─ d₂ = d₁ - σ√T
│   └─ Call Price: C = S₀N(d₁) - Ke^(-rT)N(d₂)
├─ Put Option Formula:
│   ├─ Payoff: P(S_T) = max(K - S_T, 0)
│   └─ Put Price: P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
│   └─ Alternative (via parity): P = C - S₀ + Ke^(-rT)
├─ Greeks (Analytical Derivatives):
│   ├─ Delta: Δ_call = N(d₁), Δ_put = N(d₁) - 1
│   ├─ Gamma: Γ = n(d₁) / (S₀σ√T) (same for call/put)
│   ├─ Vega: ν = S₀√T n(d₁) (same for call/put)
│   ├─ Theta: θ_call = -S₀n(d₁)σ/(2√T) - rKe^(-rT)N(d₂)
│   ├─ Rho: ρ_call = KTe^(-rT)N(d₂), ρ_put = -KTe^(-rT)N(-d₂)
│   └─ Note: n(x) = φ(x) = (1/√(2π))e^(-x²/2) is standard normal PDF
├─ Implied Volatility:
│   ├─ Inverse Problem: Given market price C_market, find σ_imp
│   ├─ Method: Newton-Raphson iteration using Vega
│   └─ Iteration: σ_{n+1} = σ_n - (C_BS(σ_n) - C_market) / Vega(σ_n)
└─ Numerical Considerations:
    ├─ Extreme Strikes: d₁, d₂ → large |values| → N(d) near 0 or 1
    ├─ Short Expiry: σ√T → 0 → instability; use intrinsic value
    └─ Volatility Bounds: σ_imp unstable for deep OTM options
```

**Interaction:** Asset price S → d₁, d₂ computation → Cumulative normals → Option price

## 5. Mini-Project
Implement Black-Scholes formulas, compute Greeks, and extract implied volatility:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# Black-Scholes formulas
def bs_d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 parameters."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call option price."""
    if T <= 0:
        return np.maximum(S - K, 0)
    
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def bs_put(S, K, T, r, sigma):
    """Black-Scholes European put option price."""
    if T <= 0:
        return np.maximum(K - S, 0)
    
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Greeks
def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Compute all Greeks for European option.
    
    Parameters:
    - option_type: 'call' or 'put'
    
    Returns:
    - dict with delta, gamma, vega, theta, rho
    """
    if T <= 0:
        # At expiry, Greeks undefined or trivial
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    
    # Standard normal PDF
    n_d1 = norm.pdf(d1)
    
    # Greeks (common to both call and put)
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * n_d1 / 100  # Divide by 100 for 1% change
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (- S * n_d1 * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # Per day
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change
    else:  # put
        delta = norm.cdf(d1) - 1
        theta = (- S * n_d1 * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365  # Per day
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

# Implied volatility
def implied_volatility(option_price, S, K, T, r, option_type='call', tol=1e-6):
    """
    Compute implied volatility using Brent's method.
    
    Returns:
    - sigma_implied or None if no solution found
    """
    if T <= 0:
        return None
    
    # Objective function: BS_price(sigma) - market_price = 0
    def objective(sigma):
        if option_type == 'call':
            return bs_call(S, K, T, r, sigma) - option_price
        else:
            return bs_put(S, K, T, r, sigma) - option_price
    
    try:
        # Search in range [0.01, 5.0] (1% to 500% volatility)
        sigma_imp = brentq(objective, 0.01, 5.0, xtol=tol)
        return sigma_imp
    except ValueError:
        # No solution in range (e.g., option mispriced)
        return None

# Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.25

# Compute option prices
call_price = bs_call(S0, K, T, r, sigma)
put_price = bs_put(S0, K, T, r, sigma)

print("="*70)
print("BLACK-SCHOLES OPTION PRICES")
print("="*70)
print(f"Parameters: S₀=${S0}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%")
print(f"\nCall Price: ${call_price:.4f}")
print(f"Put Price:  ${put_price:.4f}")

# Verify put-call parity
parity_lhs = call_price - put_price
parity_rhs = S0 - K * np.exp(-r * T)
print(f"\nPut-Call Parity Check:")
print(f"  C - P = ${parity_lhs:.6f}")
print(f"  S₀ - Ke^(-rT) = ${parity_rhs:.6f}")
print(f"  Difference: ${abs(parity_lhs - parity_rhs):.8f}")

# Compute Greeks
call_greeks = bs_greeks(S0, K, T, r, sigma, 'call')
put_greeks = bs_greeks(S0, K, T, r, sigma, 'put')

print("\n" + "="*70)
print("GREEKS (Call)")
print("="*70)
for greek, value in call_greeks.items():
    print(f"{greek.capitalize():8s}: {value:.6f}")

print("\n" + "="*70)
print("GREEKS (Put)")
print("="*70)
for greek, value in put_greeks.items():
    print(f"{greek.capitalize():8s}: {value:.6f}")

# Implied volatility test
market_call_price = call_price  # Use BS price as "market" price
sigma_implied = implied_volatility(market_call_price, S0, K, T, r, 'call')
print(f"\n" + "="*70)
print("IMPLIED VOLATILITY")
print("="*70)
print(f"Input Volatility: {sigma*100:.2f}%")
print(f"Implied Volatility: {sigma_implied*100:.2f}%")
print(f"Difference: {abs(sigma - sigma_implied)*100:.6f}%")

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Option prices vs spot
spots = np.linspace(50, 150, 100)
call_prices = [bs_call(S, K, T, r, sigma) for S in spots]
put_prices = [bs_put(S, K, T, r, sigma) for S in spots]
intrinsic_call = np.maximum(spots - K, 0)
intrinsic_put = np.maximum(K - spots, 0)

ax = axes[0, 0]
ax.plot(spots, call_prices, 'b-', linewidth=2, label='Call Price')
ax.plot(spots, put_prices, 'r-', linewidth=2, label='Put Price')
ax.plot(spots, intrinsic_call, 'b--', linewidth=1, alpha=0.5, label='Call Intrinsic')
ax.plot(spots, intrinsic_put, 'r--', linewidth=1, alpha=0.5, label='Put Intrinsic')
ax.axvline(K, color='black', linestyle='--', alpha=0.5)
ax.axvline(S0, color='green', linestyle='--', alpha=0.5, label=f'S₀=${S0}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Option Price ($)')
ax.set_title('BS Option Prices vs Spot')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Option prices vs volatility
sigmas = np.linspace(0.05, 1.0, 100)
call_vol = [bs_call(S0, K, T, r, s) for s in sigmas]
put_vol = [bs_put(S0, K, T, r, s) for s in sigmas]

ax = axes[0, 1]
ax.plot(sigmas * 100, call_vol, 'b-', linewidth=2, label='Call')
ax.plot(sigmas * 100, put_vol, 'r-', linewidth=2, label='Put')
ax.axvline(sigma * 100, color='green', linestyle='--', alpha=0.5, label=f'σ={sigma*100}%')
ax.set_xlabel('Volatility σ (%)')
ax.set_ylabel('Option Price ($)')
ax.set_title('Option Prices vs Volatility (Vega)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Option prices vs time to maturity
times = np.linspace(0.01, 2.0, 100)
call_time = [bs_call(S0, K, t, r, sigma) for t in times]
put_time = [bs_put(S0, K, t, r, sigma) for t in times]

ax = axes[0, 2]
ax.plot(times, call_time, 'b-', linewidth=2, label='Call')
ax.plot(times, put_time, 'r-', linewidth=2, label='Put')
ax.axvline(T, color='green', linestyle='--', alpha=0.5, label=f'T={T}yr')
ax.axhline(max(S0 - K, 0), color='b', linestyle='--', alpha=0.3, label='Call Intrinsic')
ax.axhline(max(K - S0, 0), color='r', linestyle='--', alpha=0.3, label='Put Intrinsic')
ax.set_xlabel('Time to Maturity T (years)')
ax.set_ylabel('Option Price ($)')
ax.set_title('Option Prices vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Delta vs spot
deltas_call = [bs_greeks(S, K, T, r, sigma, 'call')['delta'] for S in spots]
deltas_put = [bs_greeks(S, K, T, r, sigma, 'put')['delta'] for S in spots]

ax = axes[1, 0]
ax.plot(spots, deltas_call, 'b-', linewidth=2, label='Call Delta')
ax.plot(spots, deltas_put, 'r-', linewidth=2, label='Put Delta')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(0.5, color='b', linestyle='--', alpha=0.3)
ax.axhline(-0.5, color='r', linestyle='--', alpha=0.3)
ax.axvline(K, color='black', linestyle='--', alpha=0.5, label=f'K=${K}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Delta (∂V/∂S)')
ax.set_title('Delta Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Gamma vs spot
gammas = [bs_greeks(S, K, T, r, sigma)['gamma'] for S in spots]

ax = axes[1, 1]
ax.plot(spots, gammas, 'g-', linewidth=2)
ax.axvline(K, color='red', linestyle='--', alpha=0.5, label=f'K=${K} (max Γ)')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Gamma (∂²V/∂S²)')
ax.set_title('Gamma Profile (same for call/put)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Vega vs spot
vegas = [bs_greeks(S, K, T, r, sigma)['vega'] for S in spots]

ax = axes[1, 2]
ax.plot(spots, vegas, 'purple', linewidth=2)
ax.axvline(K, color='red', linestyle='--', alpha=0.5, label=f'K=${K} (max ν)')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Vega (∂V/∂σ per 1%)')
ax.set_title('Vega Profile (same for call/put)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 7: Theta vs time to maturity
times_theta = np.linspace(0.01, 2.0, 100)
theta_call = [bs_greeks(S0, K, t, r, sigma, 'call')['theta'] for t in times_theta]
theta_put = [bs_greeks(S0, K, t, r, sigma, 'put')['theta'] for t in times_theta]

ax = axes[2, 0]
ax.plot(times_theta, theta_call, 'b-', linewidth=2, label='Call Theta')
ax.plot(times_theta, theta_put, 'r-', linewidth=2, label='Put Theta')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time to Maturity T (years)')
ax.set_ylabel('Theta (∂V/∂t per day)')
ax.set_title('Theta (Time Decay)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 8: Implied volatility surface (Strike vs Time)
strikes = np.linspace(70, 130, 30)
maturities = np.linspace(0.1, 2.0, 20)
K_grid, T_grid = np.meshgrid(strikes, maturities)
iv_surface = np.zeros_like(K_grid)

for i in range(len(maturities)):
    for j in range(len(strikes)):
        # Generate "market" price with constant vol
        market_price = bs_call(S0, strikes[j], maturities[i], r, sigma)
        iv = implied_volatility(market_price, S0, strikes[j], maturities[i], r, 'call')
        iv_surface[i, j] = iv if iv else np.nan

ax = axes[2, 1]
contour = ax.contourf(K_grid, T_grid, iv_surface * 100, levels=20, cmap='viridis')
ax.axvline(S0, color='red', linestyle='--', linewidth=2, label=f'S₀=${S0}')
ax.set_xlabel('Strike K')
ax.set_ylabel('Time to Maturity T (years)')
ax.set_title('Implied Volatility Surface (flat at input σ)')
plt.colorbar(contour, ax=ax, label='Implied Vol (%)')
ax.legend()

# Plot 9: d1 and d2 vs spot
d1_values = []
d2_values = []
for S in spots:
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    d1_values.append(d1)
    d2_values.append(d2)

ax = axes[2, 2]
ax.plot(spots, d1_values, 'b-', linewidth=2, label='d₁')
ax.plot(spots, d2_values, 'r-', linewidth=2, label='d₂')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(K, color='green', linestyle='--', alpha=0.5, label=f'K=${K}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('d₁, d₂')
ax.set_title('d₁ and d₂ Parameters')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('black_scholes_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Greeks numerical verification (finite difference)
eps = 0.01  # 1 cent for delta/gamma
call_up = bs_call(S0 + eps, K, T, r, sigma)
call_down = bs_call(S0 - eps, K, T, r, sigma)
delta_numerical = (call_up - call_down) / (2 * eps)
gamma_numerical = (call_up - 2 * call_price + call_down) / (eps**2)

print("\n" + "="*70)
print("GREEKS VERIFICATION (Finite Difference)")
print("="*70)
print(f"Delta (analytical): {call_greeks['delta']:.6f}")
print(f"Delta (numerical):  {delta_numerical:.6f}")
print(f"Gamma (analytical): {call_greeks['gamma']:.6f}")
print(f"Gamma (numerical):  {gamma_numerical:.6f}")
```

## 6. Challenge Round

**Q1:** Derive BS call formula from E^Q[max(S_T - K, 0)] where S_T ~ lognormal. Show N(d₁) term arises.  
**A1:** E[max(S_T - K, 0)] = ∫ₖ^∞ (S - K) f(S) dS where f is lognormal. Split: ∫ₖ^∞ S f(S) dS - K ∫ₖ^∞ f(S) dS. First integral = S₀e^(rT) P(S_T > K | shifted dist) = S₀e^(rT) N(d₁). Second = KN(d₂). Discount: C = e^(-rT)[S₀e^(rT)N(d₁) - KN(d₂)] = S₀N(d₁) - Ke^(-rT)N(d₂).

**Q2:** Why is d₁ - d₂ = σ√T? What does this separation represent?  
**A2:** d₁ = [ln(S/K) + (r + σ²/2)T] / σ√T; d₂ = [ln(S/K) + (r - σ²/2)T] / σ√T. Difference: d₁ - d₂ = σ²T / σ√T = σ√T. Represents volatility drag over time; d₁ relates to stock numeraire, d₂ to cash numeraire.

**Q3:** Interpret N(d₁) and N(d₂). What probabilities do they represent?  
**A3:** N(d₂) = risk-neutral probability option expires ITM (S_T > K). N(d₁) = delta; also probability ITM under stock numeraire (measure change). Both are cumulative probabilities under shifted lognormal distributions.

**Q4:** BS assumes constant volatility. How does volatility smile/skew violate this?  
**A4:** Market IVs vary by strike (smile) and time (term structure). OTM puts have higher IV (skew); tail risk priced. BS inapplicable as single σ; local volatility or stochastic vol models required (Heston, SABR).

**Q5:** Derive Black-Scholes PDE from hedging argument (no stochastic calculus).  
**A5:** Portfolio Π = V - ΔS replicates option. Instantaneous return must equal risk-free rate (no arbitrage): dΠ = r Π dt. Expand dV via Ito: dV = (∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S²)dt. Substitute: ∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² - rV = 0.

**Q6:** Greeks satisfy ∂C/∂T = rKe^(-rT)N(d₂) - S₀n(d₁)σ/(2√T). Verify this equals theta formula.  
**A6:** Theta = -∂C/∂T (convention: time decay). Differentiate C = S₀N(d₁) - Ke^(-rT)N(d₂). Use ∂N(d₁)/∂T = n(d₁)∂d₁/∂T, ∂d₁/∂T = -σ/(2√T) - [ln(S/K) + (r + σ²/2)T]/(2T^(3/2)σ). Simplify (algebra intensive) → θ = -S₀n(d₁)σ/(2√T) - rKe^(-rT)N(d₂).

**Q7:** Implement Newton-Raphson for implied volatility. Why use Vega as denominator?  
**A7:** Newton-Raphson: σ_{n+1} = σ_n - f(σ_n)/f'(σ_n) where f(σ) = C_BS(σ) - C_market. Derivative f'(σ) = ∂C/∂σ = Vega. Converges quadratically (2-3 iterations typical) if initial guess reasonable (e.g., σ₀ = 0.3).

**Q8:** BS call delta = N(d₁). For ATM option (S = K), what is delta and why?  
**A8:** ATM: d₁ = (r + σ²/2)T / σ√T ≈ σ√T/2 (if r small). For T = 1, σ = 0.2: d₁ ≈ 0.1 → N(d₁) ≈ 0.54. Not exactly 0.5 due to drift term (r + σ²/2)T; symmetric only if r = -σ²/2 (rare).

## 7. Key References

**Primary Sources:**
- Black, F. & Scholes, M. "The Pricing of Options and Corporate Liabilities" (1973) - Original paper
- [Black-Scholes Model Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) - Comprehensive overview
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 15: BS Model

**Technical Details:**
- Shreve, S. *Stochastic Calculus for Finance II* (2004) - Rigorous derivation (pp. 215-280)
- Wilmott, P. *Paul Wilmott on Quantitative Finance* (2006) - PDE approach (Vol 1, pp. 89-134)

**Thinking Steps:**
1. Define risk-neutral measure: drift μ → r in asset dynamics
2. Terminal price distribution: ln(S_T) ~ N(ln(S₀) + (r - σ²/2)T, σ²T)
3. Compute E^Q[max(S_T - K, 0)] via lognormal integrals
4. Recognize ∫ S f(S) dS = S₀e^(rT) N(d₁) (shifted mean)
5. Discount to present: C = S₀N(d₁) - Ke^(-rT)N(d₂)
6. Differentiate analytically for Greeks (delta, gamma, vega, theta, rho)
