# Early Exercise Feature

## 1. Concept Skeleton
**Definition:** American options' right to exercise at any time before or at expiration, versus European options' fixed expiration-only exercise  
**Purpose:** Captures flexibility value in real financial markets; explains American-European price premium; determines optimal stopping problem  
**Prerequisites:** Option pricing basics, martingale theory, optimal stopping, dynamic programming, American option valuation

## 2. Comparative Framing
| Aspect | American | European | Impact |
|--------|----------|----------|--------|
| **Exercise Timing** | Any time T ≤ t ≤ T_exp | Only at T_exp | Higher flexibility → higher value |
| **Optimal Exercise** | May exercise early | Never optimal early | Depends on dividends, rate paths |
| **Call (non-div)** | Never early | Same | Price equality |
| **Put (no div)** | Often early (ITM, high r) | Never early | American premium can be significant |
| **Callable Bonds** | Issuer exercises | Fixed maturity | Reduces bondholder value |
| **Swing Options** | Multiple exercise dates | One date | Energy derivatives |

## 3. Examples + Counterexamples

**Simple Example:**  
Put option: S=$90, K=$100, r=5%, σ=20%, T=6 months. European value ~$10.25. American allows early exercise, capturing intrinsic $10 now + optionality if S rises → American premium ~$0.50-$1.

**Call on Dividend Stock:**  
Stock pays 5% annual dividend. Before ex-dividend date: ITM call holder faces dividend loss. Early exercise captures intrinsic, justifies American > European.

**Deep ITM Put:**  
S=$10, K=$100, r=10%, σ=5%, T=1 year. Intrinsic = $90. Holding European ties up capital earning only r; exercising gives $90 now, invested at r. American captures this → significant premium.

**When Not Early:**  
OTM options (calls with high S, puts with low S) rarely exercise early; European price ≈ American.

**Deep OTM:**  
S=$150, K=$100 (call), small σ, short T. American = European ≈ $0 (both worthless).

## 4. Layer Breakdown
```
Early Exercise Feature:

├─ Exercise Right:
│  ├─ Fundamental difference:
│  │   American: max_{t ∈ [0,T]} E[e^(-rt) Payoff(t)]
│  │   European: E[e^(-rT) Payoff(T)]
│  ├─ Optimal stopping problem:
│  │   Find τ* maximizing expected discounted payoff
│  │   τ* = argmax_τ E[e^(-rτ) Payoff(τ)]
│  └─ Exercise boundary:
│      S(t) ≥ S*(t) ⟹ Exercise now (early stopping region)
│      S(t) < S*(t) ⟹ Hold (continuation region)
├─ Value Decomposition:
│  ├─ American option value:
│  │   V_A = max(Intrinsic, Option_Value_if_Hold)
│  ├─ European option value:
│  │   V_E = E^Q[e^(-rT) Payoff(T)]
│  ├─ American Premium:
│  │   Premium = V_A - V_E ≥ 0 (always)
│  │   V_A ≥ Intrinsic (American)
│  │   V_E can be < Intrinsic (European)
│  └─ Intuition: Extra optionality is valuable, free to hold
├─ Call Options (Non-Dividend):
│  ├─ Dividend-free assumption:
│  │   No benefit to capturing dividends early
│  │   Holding → investment return at r
│  ├─ Optimal policy:
│  │   Never exercise early! 
│  │   V_A(call) = V_E(call)
│  ├─ Intuition:
│  │   Intrinsic S-K < Option value (chance to rise)
│  │   Discounting reduces future payoff
│  │   But keeping option open better
│  └─ Exception: If r → ∞, premium→0, can become equal
├─ Call Options (With Dividends):
│  ├─ Continuous dividend yield q:
│  │   ex-dividend date approaching → S drops by dividend
│  ├─ Optimal exercise:
│  │   Just before ex-dividend: capture intrinsic
│  │   S - K vs (S - div) - K
│  ├─ Exercise boundary S*(t):
│  │   High q → early exercise more valuable
│  │   Low q → more like European
│  └─ Premium mechanism:
│      Capture dividend, avoid post-ex drop, arbitrage opportunity
├─ Put Options:
│  ├─ Intrinsic always available:
│  │   ITM put: Intrinsic = K - S > 0
│  ├─ Why exercise early:
│  │   High interest rate r: $K now > e^(-r*Δt)*K later
│  │   Stock volatility σ: Exercise locks in K-S immediately
│  │   Volatility risk: Avoid risk of S rising further
│  ├─ Exercise boundary S*_put(t):
│  │   S ≤ S* → Exercise now
│  │   S > S* → Hold (option value > intrinsic)
│  │   S* typically 70-90% of strike (depends on r, σ, T)
│  └─ Empirical: American puts often 5-20% premium over European
├─ Optimal Stopping Boundary:
│  ├─ Free boundary problem:
│  │   In continuation region: LV = rV (no drift, risk-neutral)
│  │   On boundary: V = Intrinsic (exercise)
│  ├─ Numerical solution:
│  │   Find S*(t) such that:
│  │   ├─ V(S*(t), t) = S*(t) - K (call) or K - S*(t) (put)
│  │   ├─ Smooth pasting (C² continuity): ∂V/∂S continuous
│  │   └─ Iterative refinement (binomial, finite difference)
│  ├─ Monotonicity:
│  │   Put boundary S*_put(t) increases with time (more intrinsic)
│  │   Call boundary S*_call(t) more complex (dividend vs rate)
│  └─ Limiting behavior:
│      S*(∞) → stable level (perpetual option)
│      S*(0) → boundary determined by payoff
├─ Time Value & Intrinsic:
│  ├─ European always:
│  │   Value ≥ Intrinsic or ≤ Intrinsic (depending on option type)
│  │   But discounting can make V_E < Intrinsic
│  ├─ American:
│  │   Always V_A ≥ Intrinsic (exercise if V < Intrinsic)
│  │   Time value = V_A - Intrinsic ≥ 0
│  ├─ Decay near expiry:
│  │   T→0: V_A → Intrinsic (option must be exercised or expire)
│  │   V_E → Intrinsic (for ITM); European premium vanishes
│  └─ Impact: American → more time value retained
└─ Extensions:
   ├─ Perpetual options (T→∞):
   │   S*(t)→S* (constant), can solve analytically
   ├─ Bermudan options:
   │   Exercise at discrete dates only
   │   Premium between European & American
   ├─ Path-dependent:
   │   Lookback: Exercise value depends on max/min S(t)
   │   Barrier: Exercise blocked if S crosses level
   └─ Multi-asset:
       Exchange options, basket options with early exercise
```

**Interaction:** Early exercise value = Difference in optimal stopping regimes; maximized when rates high/volatility low/dividends present.

## 5. Mini-Project
Analyze early exercise premium across parameters:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

def bs_call(S, K, r, sigma, T, q=0):
    """European call with dividend yield q"""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def bs_put(S, K, r, sigma, T, q=0):
    """European put with dividend yield q"""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    return put

# Parameters
S_range = np.linspace(60, 140, 100)
K = 100
r = 0.05
sigma = 0.2
T = 1.0

# Scenarios
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Call with dividends (q=0 vs q=0.05)
call_no_div = [bs_call(S, K, r, sigma, T, q=0) for S in S_range]
call_div = [bs_call(S, K, r, sigma, T, q=0.05) for S in S_range]
intrinsic_call = np.maximum(S_range - K, 0)

axes[0, 0].plot(S_range, call_no_div, 'b-', linewidth=2, label='Call (q=0)')
axes[0, 0].plot(S_range, call_div, 'r--', linewidth=2, label='Call (q=5%)')
axes[0, 0].plot(S_range, intrinsic_call, 'k:', linewidth=1, label='Intrinsic')
axes[0, 0].set_title('European Call: Dividend Impact')
axes[0, 0].set_xlabel('Stock Price (S)')
axes[0, 0].set_ylabel('Option Value')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Put option across time to expiry
T_range = np.array([0.1, 0.25, 0.5, 1.0])
colors = plt.cm.viridis(np.linspace(0, 1, len(T_range)))

for T_val, color in zip(T_range, colors):
    put_vals = [bs_put(S, K, r, sigma, T_val) for S in S_range]
    axes[0, 1].plot(S_range, put_vals, color=color, linewidth=2, 
                   label=f'T={T_val}')

intrinsic_put = np.maximum(K - S_range, 0)
axes[0, 1].plot(S_range, intrinsic_put, 'k--', linewidth=1.5, label='Intrinsic')
axes[0, 1].set_title('European Put: Time Decay')
axes[0, 1].set_xlabel('Stock Price (S)')
axes[0, 1].set_ylabel('Option Value')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Time value decomposition (ITM put)
T_vals = np.linspace(0.01, T, 50)
S_ITM = 85  # ITM put

put_vals_T = [bs_put(S_ITM, K, r, sigma, T_val) for T_val in T_vals]
intrinsic_vals = [max(K - S_ITM, 0)] * len(T_vals)
time_value = [pv - intrinsic_vals[i] for i, pv in enumerate(put_vals_T)]

axes[1, 0].fill_between(T_vals, 0, intrinsic_vals, alpha=0.3, label='Intrinsic')
axes[1, 0].fill_between(T_vals, intrinsic_vals, put_vals_T, alpha=0.3, label='Time Value')
axes[1, 0].plot(T_vals, put_vals_T, 'b-', linewidth=2)
axes[1, 0].set_title(f'Put Decomposition (S=${S_ITM}, ITM)')
axes[1, 0].set_xlabel('Time to Expiry (years)')
axes[1, 0].set_ylabel('Option Value ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: American premium estimate (put, high interest rate scenario)
r_range = np.linspace(0, 0.15, 30)
premiums = []
S_val = 90  # ITM

for r_val in r_range:
    eur_put = bs_put(S_val, K, r_val, sigma, T)
    # Rough American approximation (more precise requires binomial)
    intrinsic = max(K - S_val, 0)
    # Premium increases with r (early exercise more valuable)
    premium_estimate = intrinsic - eur_put + 0.01 * (r_val - r)  # heuristic
    premium_estimate = max(premium_estimate, 0)
    premiums.append(premium_estimate)

axes[1, 1].plot(r_range * 100, premiums, 'ro-', linewidth=2, markersize=6)
axes[1, 1].set_title('American Put Premium (Rough Estimate)')
axes[1, 1].set_xlabel('Risk-Free Rate (%)')
axes[1, 1].set_ylabel('American - European ($)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Early Exercise Analysis:")
print(f"European Call (q=0, S=100): ${bs_call(100, K, r, sigma, T, q=0):.3f}")
print(f"European Call (q=5%, S=100): ${bs_call(100, K, r, sigma, T, q=0.05):.3f}")
print(f"European Put (S=90): ${bs_put(90, K, r, sigma, T):.3f}")
print(f"Intrinsic Put (S=90): ${max(K-90, 0):.3f}")
```

## 6. Challenge Round
- When is early exercise optimal for calls with dividends? (Just before ex-dividend)
- Why doesn't American call (non-div) command premium? (No benefit to early exercise)
- How does volatility affect early exercise propensity? (High vol delays exercise)
- What's the perpetual put exercise boundary? (Constant S* independent of time)

## 7. Key References
- [Merton, "Theory of Rational Option Pricing" (1973)](https://doi.org/10.1016/B978-0-44-403970-2.50005-8) — American option framework
- [Hull, Options, Futures, and Derivatives (Chapter 8)](https://www.wiley.com/en-us/Options%2C+Futures%2C+and+Other+Derivatives%2C+11th+Edition-p-9781119259503) — Early exercise analysis
- [Binomial Trees for American Options](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

---
**Status:** Fundamental for American vs European distinction | **Complements:** Binomial Trees, Perpetual Options, Optimal Stopping Theory
