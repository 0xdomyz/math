# Black-Scholes Model

## 1. Concept Skeleton
**Definition:** Closed-form mathematical model for pricing European options on non-dividend-paying stocks; assumes log-normal stock price distribution, constant volatility, no arbitrage, continuous trading  
**Purpose:** Foundation for derivatives pricing; enables rapid option valuation; basis for implied volatility calculations; practical benchmark despite unrealistic assumptions  
**Prerequisites:** Option pricing basics, stochastic calculus, log-normal distribution, risk-neutral valuation, no-arbitrage principle

## 2. Comparative Framing
| Model | Black-Scholes | Binomial | Monte Carlo | Numerical PDE |
|-------|---------------|----------|-------------|---------------|
| **Type** | Closed-form | Tree-based | Simulation | Grid-based |
| **Speed** | Instant | Fast (recombines) | Slow (many paths) | Moderate |
| **Accuracy** | Good (standard assumptions) | Exact (for grid) | Improves with paths | Accurate (fine grid) |
| **Dividends** | Extension available | Easy to add | Straightforward | Easy to add |
| **American Options** | N/A | Natural fit | Approximate needed | Natural fit |
| **Exotics** | Limited | Limited | Excellent | Good |
| **Intuition** | Medium | High | Medium | Low |
| **Computation** | Analytical | Numerical recurrence | Sampling | Numerical solver |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock $100, strike $100, r=5%, T=1 year, σ=20%. BS formula gives C≈$10.45, P≈$5.57. Matches market prices well for liquid, vanilla options.

**Excellent Fit:**  
Short-dated ATM options on large-cap stocks: Continuous trading, low transaction costs, relatively stable volatility. BS very accurate.

**Poor Fit:**  
Very volatile stocks (σ changes), illiquid markets (bid-ask spreads large), short-dated options deep OTM (jumps matter, assumes continuous paths), or when large discrete dividends occur.

**American Options:**  
BS doesn't handle early exercise. Must use binomial, Monte Carlo, or numerical PDE. For calls on non-dividend stocks, BS bound is tight (American=European).

**Extension - With Dividends:**  
Modify to S*e^(-q*T) where q is continuous dividend yield. Reduces call value, increases put value.

## 4. Layer Breakdown
```
Black-Scholes Framework:

├─ Assumptions (Critical):
│  ├─ Stock price follows geometric Brownian motion:
│  │   dS = μS dt + σS dW
│  │   (constant volatility σ, drift μ)
│  ├─ Continuous trading (no bid-ask spreads)
│  ├─ No arbitrage (can replicate option payoff)
│  ├─ Risk-free rate r constant
│  ├─ No dividends (or constant dividend yield)
│  ├─ Frictionless market (no taxes, costs)
│  ├─ No restrictions on short-selling
│  ├─ Log-normal distribution for S_T
│  └─ European exercise only
├─ Mathematical Derivation:
│  ├─ Setup: Option value C(S,t) dependent on spot, time
│  ├─ Replicating portfolio: Hold Δ shares + bond
│  ├─ No-arbitrage condition: Portfolio return = r
│  ├─ Ito's Lemma applied to C(S,t)
│  ├─ Results in PDE:
│  │   ∂C/∂t + rS(∂C/∂S) + (1/2)σ²S²(∂²C/∂S²) = rC
│  ├─ Boundary conditions:
│  │   C(S,T) = max(S-K, 0) [call payoff at expiry]
│  │   C(0,t) = 0 [worthless if S→0]
│  │   C(S,t) ≈ S as S→∞ [behaves like stock]
│  └─ Solution: Closed-form formulas
├─ Black-Scholes Formulas:
│  ├─ Call Price:
│  │   C = S₀ N(d₁) - K e^(-rT) N(d₂)
│  ├─ Put Price:
│  │   P = K e^(-rT) N(-d₂) - S₀ N(-d₁)
│  ├─ Where:
│  │   d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
│  │   d₂ = d₁ - σ√T
│  ├─ Components:
│  │   S₀: Current spot price
│  │   K: Strike price
│  │   r: Risk-free rate
│  │   T: Time to expiry
│  │   σ: Volatility (annualized)
│  │   N(.): Standard normal CDF
│  └─ Put-Call Parity: C - P = S₀ - K e^(-rT)
├─ Greeks Derivation:
│  ├─ Delta (∂C/∂S):
│  │   Δ_call = N(d₁) ∈ [0, 1]
│  │   Δ_put = -N(-d₁) ∈ [-1, 0]
│  │   Hedging interpretation: Buy Δ shares per short call
│  ├─ Gamma (∂²C/∂S²):
│  │   Γ = N'(d₁) / (S σ√T) > 0
│  │   Peaks near ATM, highest near expiry
│  │   Hedging cost: Must rebalance as S moves
│  ├─ Theta (∂C/∂t):
│  │   Θ_call = -S N'(d₁) σ/(2√T) - r K e^(-rT) N(d₂)
│  │   Negative for long calls (time decay)
│  │   Θ_put = -S N'(d₁) σ/(2√T) + r K e^(-rT) N(-d₂)
│  ├─ Vega (∂C/∂σ):
│  │   ν = S N'(d₁) √T > 0 (same for calls and puts)
│  │   Peak ATM, decreases near expiry or deep in/out
│  └─ Rho (∂C/∂r):
│      ρ_call = K T e^(-rT) N(d₂)
│      ρ_put = -K T e^(-rT) N(-d₂)
├─ Implied Volatility:
│  ├─ Invert BS formula: Given market price → find σ
│  ├─ No closed form; use Newton-Raphson
│  ├─ Volatility smile/skew: IV varies by strike
│  ├─ Term structure: IV varies by expiry
│  └─ Used for comparisons, quoting, risk assessment
├─ Violations & Reality Checks:
│  ├─ Constant volatility assumption violated:
│  │   ├─ Volatility changes over time (stochastic vol)
│  │   ├─ Different strikes imply different vols (smile)
│  │   └─ Solution: Heston model, SABR, etc.
│  ├─ Jump risk:
│  │   ├─ Stock prices jump (gap events, earnings)
│  │   ├─ Log-normal doesn't account for gaps
│  │   └─ Solution: Jump-diffusion models
│  ├─ Discrete rebalancing costs:
│  │   ├─ Hedging only possible at discrete times
│  │   ├─ Gamma risk accumulates
│  │   └─ Transaction costs eat into profits
│  └─ Liquidity/bid-ask spreads:
│      ├─ Pricing models assume perfect markets
│      ├─ Real bid-ask affects profitable hedging
│      └─ Adjustment: Add transaction cost factor
└─ Extensions:
   ├─ With dividends (yield q):
   │   C = S₀ e^(-qT) N(d₁) - K e^(-rT) N(d₂)
   │   d₁ = [ln(S₀/K) + (r-q+σ²/2)T] / (σ√T)
   ├─ With foreign exchange (similar form)
   ├─ With futures (different drift term)
   └─ Approximations for small moves (delta/gamma terms)
```

**Interaction:** BS price sensitive to all five parameters (S, K, r, T, σ); Greeks measure each sensitivity; Vega largest uncertainty (volatility hardest to estimate).

## 5. Mini-Project
Implement Black-Scholes, compute Greeks, analyze sensitivity:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("BLACK-SCHOLES MODEL IMPLEMENTATION")
print("="*60)

class BlackScholes:
    """Black-Scholes European option pricing"""
    
    def __init__(self, S, K, r, T, sigma, q=0):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.q = q  # Dividend yield
        
        # Calculate d1, d2
        self.d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
        
    def call_price(self):
        return (self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2))
    
    def put_price(self):
        return (self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - 
                self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1))
    
    def delta_call(self):
        return np.exp(-self.q*self.T) * norm.cdf(self.d1)
    
    def delta_put(self):
        return -np.exp(-self.q*self.T) * norm.cdf(-self.d1)
    
    def gamma(self):
        return (np.exp(-self.q*self.T) * norm.pdf(self.d1) / 
                (self.S * self.sigma * np.sqrt(self.T)))
    
    def vega(self):
        return self.S * np.exp(-self.q*self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100
    
    def theta_call(self):
        term1 = -self.S * np.exp(-self.q*self.T) * norm.pdf(self.d1) * self.sigma / (2*np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)
        term3 = self.q * self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1)
        return (term1 + term2 + term3) / 365
    
    def theta_put(self):
        term1 = -self.S * np.exp(-self.q*self.T) * norm.pdf(self.d1) * self.sigma / (2*np.sqrt(self.T))
        term2 = self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2)
        term3 = -self.q * self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1)
        return (term1 + term2 + term3) / 365
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(self.d2) / 100
    
    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(-self.d2) / 100
    
    def implied_volatility(self, market_price, option_type='call'):
        """Find IV using Newton-Raphson"""
        def objective(sigma):
            bs = BlackScholes(self.S, self.K, self.r, self.T, sigma, self.q)
            if option_type == 'call':
                return bs.call_price() - market_price
            else:
                return bs.put_price() - market_price
        
        def vega_func(sigma):
            bs = BlackScholes(self.S, self.K, self.r, self.T, sigma, self.q)
            return bs.vega() * 100  # Revert from /100 to get sensitivity
        
        try:
            iv = brentq(objective, 0.001, 2.0)
            return iv
        except:
            return np.nan

# Scenario 1: Basic pricing
print("\n" + "="*60)
print("SCENARIO 1: Basic Black-Scholes Pricing")
print("="*60)

S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.2

bs = BlackScholes(S, K, r, T, sigma)
call = bs.call_price()
put = bs.put_price()

print(f"\nParameters:")
print(f"  S = ${S}, K = ${K}, r = {r:.1%}, T = {T} year, σ = {sigma:.1%}")
print(f"\nOption Prices:")
print(f"  Call: ${call:.2f}")
print(f"  Put: ${put:.2f}")
print(f"  Put-Call Parity: C - P = {call - put:.4f}, S - Ke^(-rT) = {S - K*np.exp(-r*T):.4f}")

# Scenario 2: Greeks calculation
print("\n" + "="*60)
print("SCENARIO 2: Greeks for ATM Option")
print("="*60)

delta_c = bs.delta_call()
delta_p = bs.delta_put()
gamma = bs.gamma()
vega = bs.vega()
theta_c = bs.theta_call()
theta_p = bs.theta_put()
rho_c = bs.rho_call()
rho_p = bs.rho_put()

print(f"\nCall Greeks:")
print(f"  Δ = {delta_c:.4f} (hedge by selling {delta_c:.4f} shares per call)")
print(f"  Γ = {gamma:.6f} (delta changes by {gamma:.6f} per $1 move)")
print(f"  Θ (daily) = ${theta_c:.4f} (loses ${abs(theta_c):.4f}/day)")
print(f"  ν (per 1% vol) = ${vega:.4f}")
print(f"  ρ (per 1% rate) = ${rho_c:.4f}")

print(f"\nPut Greeks:")
print(f"  Δ = {delta_p:.4f}")
print(f"  Γ = {gamma:.6f} (same as call)")
print(f"  Θ (daily) = ${theta_p:.4f}")
print(f"  ν = ${vega:.4f} (same as call)")
print(f"  ρ = ${rho_p:.4f}")

# Scenario 3: Sensitivity analysis
print("\n" + "="*60)
print("SCENARIO 3: Sensitivity Analysis")
print("="*60)

# Change each parameter by 1 unit and measure impact
S_bump = S + 1
r_bump = r + 0.01
T_bump = T - 1/365  # 1 day passes
sigma_bump = sigma + 0.01

bs_bump_s = BlackScholes(S_bump, K, r, T, sigma)
bs_bump_r = BlackScholes(S, K, r_bump, T, sigma)
bs_bump_t = BlackScholes(S, K, r, T_bump, sigma)
bs_bump_vol = BlackScholes(S, K, r, T, sigma_bump)

call_delta_approx = delta_c * 1
call_rho_approx = rho_c * 1
call_theta_approx = theta_c * 1
call_vega_approx = vega * 1

print(f"\nApprox vs Actual Change in Call Price:")
print(f"  S +$1:")
print(f"    Approximate (Delta): ${call_delta_approx:.4f}")
print(f"    Actual: ${bs_bump_s.call_price() - call:.4f}")
print(f"  r +1%:")
print(f"    Approximate (Rho): ${call_rho_approx:.4f}")
print(f"    Actual: ${bs_bump_r.call_price() - call:.4f}")
print(f"  T -1 day:")
print(f"    Approximate (Theta): ${call_theta_approx:.4f}")
print(f"    Actual: ${bs_bump_t.call_price() - call:.4f}")
print(f"  σ +1%:")
print(f"    Approximate (Vega): ${call_vega_approx:.4f}")
print(f"    Actual: ${bs_bump_vol.call_price() - call:.4f}")

# Scenario 4: Greeks surface across spot prices
print("\n" + "="*60)
print("SCENARIO 4: Greeks Across Moneyness")
print("="*60)

spot_range = np.linspace(80, 120, 9)
print(f"\n{'Spot':<8} {'Delta':<10} {'Gamma':<12} {'Theta/day':<12} {'Vega':<10}")
print("-" * 52)
for s in spot_range:
    bs_temp = BlackScholes(s, K, r, T, sigma)
    print(f"${s:>6.0f}  {bs_temp.delta_call():>8.4f}  {bs_temp.gamma():>10.6f}  ${bs_temp.theta_call():>10.4f}  ${bs_temp.vega():>8.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Option prices vs spot
ax = axes[0, 0]
spot_range_plot = np.linspace(70, 130, 100)
calls = [BlackScholes(s, K, r, T, sigma).call_price() for s in spot_range_plot]
puts = [BlackScholes(s, K, r, T, sigma).put_price() for s in spot_range_plot]
intrinsic_call = np.maximum(spot_range_plot - K, 0)
intrinsic_put = np.maximum(K - spot_range_plot, 0)

ax.plot(spot_range_plot, calls, 'b-', linewidth=2.5, label='Call Value')
ax.plot(spot_range_plot, puts, 'r-', linewidth=2.5, label='Put Value')
ax.plot(spot_range_plot, intrinsic_call, 'b--', alpha=0.5, label='Call Intrinsic')
ax.plot(spot_range_plot, intrinsic_put, 'r--', alpha=0.5, label='Put Intrinsic')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Option Value')
ax.set_title('BS Option Prices vs Spot')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Greeks vs spot
ax = axes[0, 1]
deltas = [BlackScholes(s, K, r, T, sigma).delta_call() for s in spot_range_plot]
gammas = [BlackScholes(s, K, r, T, sigma).gamma() for s in spot_range_plot]

ax_twin = ax.twinx()
ax.plot(spot_range_plot, deltas, 'b-', linewidth=2.5, label='Delta')
ax_twin.plot(spot_range_plot, gammas, 'g-', linewidth=2.5, label='Gamma')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Delta', color='b')
ax_twin.set_ylabel('Gamma', color='g')
ax.set_title('Delta & Gamma vs Spot')
ax.legend(loc='upper left')
ax_twin.legend(loc='upper right')
ax.grid(alpha=0.3)

# Plot 3: Theta decay
ax = axes[0, 2]
time_range = np.linspace(T, 0.01, 50)
theta_decay = [BlackScholes(S, K, r, t, sigma).call_price() for t in time_range]
ax.plot(time_range, theta_decay, 'b-', linewidth=2.5)
ax.fill_between(time_range, theta_decay, alpha=0.3)
ax.set_xlabel('Time to Expiry (years)')
ax.set_ylabel('Call Value')
ax.set_title('Theta Decay (ATM Call)')
ax.grid(alpha=0.3)

# Plot 4: Volatility sensitivity
ax = axes[1, 0]
vol_range = np.linspace(0.05, 0.5, 50)
calls_vol = [BlackScholes(S, K, r, T, v).call_price() for v in vol_range]
puts_vol = [BlackScholes(S, K, r, T, v).put_price() for v in vol_range]
ax.plot(vol_range*100, calls_vol, 'b-', linewidth=2.5, label='Call')
ax.plot(vol_range*100, puts_vol, 'r-', linewidth=2.5, label='Put')
ax.axvline(sigma*100, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Option Value')
ax.set_title('Option Value vs Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Greeks surface (Delta vs spot and time)
ax = axes[1, 1]
spot_fine = np.linspace(70, 130, 30)
time_fine = np.linspace(T, 0.1, 30)
delta_surface = np.zeros((len(time_fine), len(spot_fine)))

for i, t in enumerate(time_fine):
    for j, s in enumerate(spot_fine):
        delta_surface[i, j] = BlackScholes(s, K, r, t, sigma).delta_call()

contour = ax.contourf(spot_fine, time_fine, delta_surface, levels=15, cmap='RdYlGn')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_title('Delta Surface')
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Delta')

# Plot 6: Put-Call Parity verification
ax = axes[1, 2]
spot_parity = np.linspace(80, 120, 50)
parity_lhs = []
parity_rhs = []

for s in spot_parity:
    bs_temp = BlackScholes(s, K, r, T, sigma)
    lhs = bs_temp.call_price() - bs_temp.put_price()
    rhs = s - K*np.exp(-r*T)
    parity_lhs.append(lhs)
    parity_rhs.append(rhs)

ax.plot(spot_parity, parity_lhs, 'b-', linewidth=2.5, label='C - P')
ax.plot(spot_parity, parity_rhs, 'r--', linewidth=2.5, label='S - Ke^(-rT)')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Value Difference')
ax.set_title('Put-Call Parity Verification')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Implied Volatility:** Market call price $11. Calculate IV using Newton-Raphson. What if IV fluctuates?

2. **Model Violations:** Compare BS price to actual market data (stock with jump risk, high volatility). Explore which Greeks most affected.

3. **Discretization Impact:** Implement discrete delta-hedging over time. Compare P&L to theoretical BS (continuous hedging).

4. **Dividend Yields:** Add continuous dividend q. How does it affect delta, theta? Special cases: deep ITM/OTM.

5. **Extreme Moves:** Sample from actual log-returns (likely fatter tails than normal). Price options; compare to BS (should overprice/underprice depending on tail).

## 7. Key References
- [Black, Scholes, Merton (1973) - Original Paper](https://www.jstor.org/stable/3003143)
- [Hull, Options, Futures, and Other Derivatives (Chapter 15)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000006649)
- [Stochastic Calculus for Finance (Shreve)](https://www.springer.com/series/3401)
- [Greeks Explained (Fisher, 2000)](https://www.jstor.org/stable/2676670)

---
**Status:** Foundation model in derivatives pricing | **Complements:** Option Pricing Basics, Implied Volatility, Greeks, Monte Carlo Pricing
