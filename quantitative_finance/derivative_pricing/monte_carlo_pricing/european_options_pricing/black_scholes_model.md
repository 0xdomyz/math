# Black-Scholes Model

## 1. Concept Skeleton
**Definition:** Closed-form mathematical model for pricing European options under continuous-time stochastic asset dynamics, assuming log-normal price distribution and frictionless markets.  
**Purpose:** Benchmark option valuation, calibrate volatility, compute hedge ratios (Greeks), foundation for advanced pricing models, market conventions  
**Prerequisites:** Stochastic differential equations, Itô's lemma, risk-neutral valuation, geometric Brownian motion, option payoff structures

## 2. Comparative Framing

| Model | Black-Scholes | Binomial Tree | Monte Carlo | Local Volatility |
|-------|---------------|---------------|-------------|------------------|
| **Closed-Form** | Yes | No | No | No |
| **Time Complexity** | O(1) | O(n²) for American | O(n × paths) | O(space grid²) |
| **Exotic Payoffs** | No | Yes (complex) | Yes | Limited |
| **Early Exercise** | No | Yes | Yes (LSM) | No |
| **Calibration** | Easy (σ match) | Medium | Hard | Very hard |
| **Smile/Skew** | None assumed | Can apply | Can apply | Natural fit |
| **Industry Use** | Ubiquitous | Rarely now | Common | Advanced books |

## 3. Examples + Counterexamples

**Simple Example:**  
ATM European call, S=100, K=100, T=1yr, σ=20%, r=5%:
- d₁ = [ln(1) + (0.05 + 0.02)(1)] / (0.20) ≈ 0.349
- d₂ = 0.349 - 0.20 ≈ 0.149
- N(d₁) ≈ 0.636, N(d₂) ≈ 0.559
- C = 100×0.636 - 100×e^{-0.05}×0.559 ≈ $10.45

**Failure Case:**  
Merton (1973): Assumes perfect markets, no transaction costs, continuous trading. Real market: bid-ask spreads, discrete rebalancing, jumps during crises. On March 2020 (COVID): realized vol >> implied vol; stop-losses trigger waterfall. BS hedges failed (gamma losses, liquidity crisis). Model risk quantified: loss = realized variance pick-up + gamma loss + jump gap loss.

**Edge Case:**  
Very short maturity T → 0: d₁, d₂ → ±∞ depending on S vs K. Call → max(S - K, 0) (intrinsic), put → max(K - S, 0). Theta effect dominates: per-day value decay accelerates. BS breaks at T=0 (singularity); use binomial at final node or numerical methods.

## 4. Layer Breakdown

```
Black-Scholes Valuation Framework:
├─ Asset Dynamics (Assumptions):
│   ├─ dS = μS dt + σS dW (geometric Brownian motion)
│   ├─ μ, σ constant (no mean reversion, jumps, stochastic vol)
│   ├─ Frictionless market (no spreads, slippage, taxes)
│   ├─ Continuous trading, no gaps
│   └─ No dividends (extended: q term added)
├─ Derivative Pricing PDE:
│   ├─ ∂C/∂t + 0.5σ²S² ∂²C/∂S² + r S ∂C/∂S - rC = 0
│   ├─ Derivation: Itô's lemma + delta hedge + no-arbitrage
│   ├─ Boundary: C(0, t) = 0, C(S→∞, t) → S - Ke^{-r(T-t)}
│   ├─ Terminal: C(S_T, T) = max(S_T - K, 0)
│   └─ Solution: Closed-form via Fourier transform, Hermite expansion
├─ Closed-Form Solution:
│   ├─ d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
│   ├─ d₂ = d₁ - σ√T
│   ├─ C = S e^{-qT} N(d₁) - K e^{-rT} N(d₂)
│   ├─ P = K e^{-rT} N(-d₂) - S e^{-qT} N(-d₁)
│   └─ N(·) = standard normal CDF
├─ Greeks (Sensitivities):
│   ├─ Δ = ∂C/∂S = e^{-qT} N(d₁)
│   ├─ Γ = ∂²C/∂S² = e^{-qT} n(d₁) / (S σ√T)
│   ├─ ν = ∂C/∂σ = S e^{-qT} n(d₁) √T
│   ├─ θ = -∂C/∂t (complex formula; two competing effects)
│   └─ ρ = ∂C/∂r = K T e^{-rT} N(d₂)
├─ Risk-Neutral Valuation:
│   ├─ Change measure from real (μ) to risk-neutral (r)
│   ├─ Expected payoff: E^Q[max(S_T - K, 0)]
│   ├─ Discount at risk-free rate r
│   └─ μ disappears; only S₀, K, T, σ, r, q matter
└─ Calibration:
    ├─ σ (volatility) from market prices via inverse optimization
    ├─ Objective: |BS_price(σ) - market_price| → minimize
    ├─ Method: Newton-Raphson (fast, stable)
    └─ Output: Implied vol (market's forecast of future volatility)
```

**Interaction:** As S increases, d₁ increases → N(d₁) increases → delta increases → Gamma effect dominates near K.

## 5. Mini-Project

Implement Black-Scholes pricing, Greeks, implied volatility calibration, and compare to Monte Carlo:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fminbound
from scipy.special import erf

class BlackScholes:
    def __init__(self, S, K, T, r, sigma, q=0, option_type='call'):
        self.S = S  # Spot price
        self.K = K  # Strike
        self.T = T  # Time to maturity
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.q = q  # Dividend yield
        self.option_type = option_type
    
    def d1_d2(self):
        """Compute d1 and d2"""
        d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        return d1, d2
    
    def price(self):
        """BS price"""
        d1, d2 = self.d1_d2()
        if self.option_type == 'call':
            return self.S*np.exp(-self.q*self.T)*norm.cdf(d1) - \
                   self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:  # put
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - \
                   self.S*np.exp(-self.q*self.T)*norm.cdf(-d1)
    
    def delta(self):
        """Delta: ∂V/∂S"""
        d1, _ = self.d1_d2()
        if self.option_type == 'call':
            return np.exp(-self.q*self.T)*norm.cdf(d1)
        else:
            return -np.exp(-self.q*self.T)*norm.cdf(-d1)
    
    def gamma(self):
        """Gamma: ∂²V/∂S²"""
        d1, _ = self.d1_d2()
        return np.exp(-self.q*self.T)*norm.pdf(d1) / (self.S*self.sigma*np.sqrt(self.T))
    
    def vega(self):
        """Vega: ∂V/∂σ (per 1% change)"""
        d1, _ = self.d1_d2()
        return self.S*np.exp(-self.q*self.T)*norm.pdf(d1)*np.sqrt(self.T) / 100
    
    def theta(self):
        """Theta: ∂V/∂t (per day)"""
        d1, d2 = self.d1_d2()
        if self.option_type == 'call':
            theta_annual = -self.S*np.exp(-self.q*self.T)*norm.pdf(d1)*self.sigma/(2*np.sqrt(self.T)) + \
                          self.q*self.S*np.exp(-self.q*self.T)*norm.cdf(d1) - \
                          self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            theta_annual = -self.S*np.exp(-self.q*self.T)*norm.pdf(d1)*self.sigma/(2*np.sqrt(self.T)) - \
                          self.q*self.S*np.exp(-self.q*self.T)*norm.cdf(-d1) + \
                          self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-d2)
        return theta_annual / 365
    
    def rho(self):
        """Rho: ∂V/∂r (per 1% change)"""
        _, d2 = self.d1_d2()
        if self.option_type == 'call':
            return self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(d2) / 100
        else:
            return -self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(-d2) / 100
    
    def implied_vol(self, market_price, tol=1e-6, max_iter=100):
        """Back out implied volatility from market price using Newton-Raphson"""
        sigma_guess = 0.3
        for _ in range(max_iter):
            bs_temp = BlackScholes(self.S, self.K, self.T, self.r, sigma_guess, self.q, self.option_type)
            bs_price = bs_temp.price()
            bs_vega = bs_temp.vega()
            
            if bs_vega < 1e-10:
                break
            
            sigma_new = sigma_guess - (bs_price - market_price) / bs_vega
            
            if abs(sigma_new - sigma_guess) < tol:
                return sigma_new
            
            sigma_guess = max(sigma_new, 0.001)  # Prevent negative vol
        
        return sigma_guess

# Parameters
S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
call = BlackScholes(S, K, T, r, sigma, q, 'call')
put = BlackScholes(S, K, T, r, sigma, q, 'put')

# Compute prices and Greeks
call_price = call.price()
put_price = put.price()
call_delta = call.delta()
call_gamma = call.gamma()
call_vega = call.vega()
call_theta = call.theta()
call_rho = call.rho()

print(f"{'='*60}")
print(f"Black-Scholes Option Pricing")
print(f"{'='*60}")
print(f"Call Price: ${call_price:.4f}")
print(f"Put Price:  ${put_price:.4f}")
print(f"Put-Call Parity Check: C - P = {call_price - put_price:.4f}")
print(f"Expected: S e^(-qT) - K e^(-rT) = {S*np.exp(-q*T) - K*np.exp(-r*T):.4f}")
print(f"\nCall Greeks:")
print(f"  Delta: {call_delta:.4f}")
print(f"  Gamma: {call_gamma:.6f}")
print(f"  Vega:  ${call_vega:.4f} (per 1% vol change)")
print(f"  Theta: ${call_theta:.4f} (per day)")
print(f"  Rho:   ${call_rho:.4f} (per 1% rate change)")

# Implied volatility calibration
market_price = call_price * 1.02  # Assume market price 2% higher
iv = call.implied_vol(market_price)
print(f"\nImplied Volatility:")
print(f"  Market Price: ${market_price:.4f}")
print(f"  Implied Vol: {iv*100:.2f}% (True: {sigma*100:.2f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Call price across spot prices
spot_range = np.linspace(80, 120, 100)
call_prices = [BlackScholes(S_temp, K, T, r, sigma, q, 'call').price() for S_temp in spot_range]
call_deltas = [BlackScholes(S_temp, K, T, r, sigma, q, 'call').delta() for S_temp in spot_range]

axes[0, 0].plot(spot_range, call_prices, 'b-', linewidth=2, label='Call Price')
axes[0, 0].plot(spot_range, np.maximum(spot_range - K*np.exp(-r*T), 0), 'g--', 
                linewidth=1.5, label='Intrinsic')
axes[0, 0].axvline(S, color='r', linestyle=':', alpha=0.7)
axes[0, 0].set_title('Call Price vs Spot')
axes[0, 0].set_xlabel('Spot Price S')
axes[0, 0].set_ylabel('Call Value')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Delta across spot prices
axes[0, 1].plot(spot_range, call_deltas, 'b-', linewidth=2)
axes[0, 1].axhline(0.5, color='g', linestyle='--', alpha=0.5, label='Δ=0.5 (ATM)')
axes[0, 1].axvline(S, color='r', linestyle=':', alpha=0.7)
axes[0, 1].set_title('Call Delta vs Spot')
axes[0, 1].set_xlabel('Spot Price S')
axes[0, 1].set_ylabel('Delta')
axes[0, 1].set_ylim(-0.05, 1.05)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Greeks across spot prices
call_gammas = [BlackScholes(S_temp, K, T, r, sigma, q, 'call').gamma() for S_temp in spot_range]
call_vegas = [BlackScholes(S_temp, K, T, r, sigma, q, 'call').vega() for S_temp in spot_range]
call_thetas = [BlackScholes(S_temp, K, T, r, sigma, q, 'call').theta() for S_temp in spot_range]

ax_gamma = axes[0, 2]
ax_vega = ax_gamma.twinx()
ax_theta = ax_gamma.twinx()
ax_theta.spines['right'].set_position(('outward', 60))

p1, = ax_gamma.plot(spot_range, call_gammas, 'b-', linewidth=2, label='Gamma')
p2, = ax_vega.plot(spot_range, call_vegas, 'g-', linewidth=2, label='Vega')
p3, = ax_theta.plot(spot_range, call_thetas, 'r-', linewidth=2, label='Theta (daily)')

ax_gamma.set_xlabel('Spot Price S')
ax_gamma.set_ylabel('Gamma', color='b')
ax_vega.set_ylabel('Vega', color='g')
ax_theta.set_ylabel('Theta', color='r')
ax_gamma.tick_params(axis='y', labelcolor='b')
ax_vega.tick_params(axis='y', labelcolor='g')
ax_theta.tick_params(axis='y', labelcolor='r')
ax_gamma.set_title('Gamma, Vega, Theta vs Spot')
ax_gamma.grid(alpha=0.3)

# Plot 4: Volatility smile (assuming local volatility effect)
strikes = np.linspace(80, 120, 20)
implied_vols = []
for K_temp in strikes:
    call_temp = BlackScholes(S, K_temp, T, r, 0.20, q, 'call')
    market_price_smile = call_temp.price() * (1 + 0.05*((K_temp/S - 1)**2))  # Add smile
    iv_smile = call_temp.implied_vol(market_price_smile)
    implied_vols.append(iv_smile)

axes[1, 0].plot(strikes/S, implied_vols, 'o-', linewidth=2, markersize=6)
axes[1, 0].axvline(1, color='r', linestyle=':', alpha=0.7, label='ATM')
axes[1, 0].set_title('Volatility Smile (Synthetic)')
axes[1, 0].set_xlabel('Moneyness (K/S)')
axes[1, 0].set_ylabel('Implied Volatility')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Price across time to maturity
times = np.linspace(T, 0.01, 50)
call_prices_time = [BlackScholes(S, K, t, r, sigma, q, 'call').price() for t in times]

axes[1, 1].plot(times, call_prices_time, 'b-', linewidth=2)
axes[1, 1].fill_between(times, np.maximum(S*np.exp(-q*times) - K*np.exp(-r*times), 0), 
                        call_prices_time, alpha=0.2)
axes[1, 1].set_title('Call Price vs Time to Maturity')
axes[1, 1].set_xlabel('Time to Maturity (years)')
axes[1, 1].set_ylabel('Call Value')
axes[1, 1].invert_xaxis()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Price across volatility
vols = np.linspace(0.01, 0.6, 50)
call_prices_vol = [BlackScholes(S, K, T, r, v, q, 'call').price() for v in vols]

axes[1, 2].plot(vols*100, call_prices_vol, 'b-', linewidth=2)
axes[1, 2].axvline(sigma*100, color='r', linestyle=':', alpha=0.7, label=f'Current σ')
axes[1, 2].set_title('Call Price vs Volatility')
axes[1, 2].set_xlabel('Volatility (%)')
axes[1, 2].set_ylabel('Call Value')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('black_scholes_analysis.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: black_scholes_analysis.png")
```

**Output Interpretation:**
- **Price Profile:** Curves match boundary conditions (S→0 gives 0, S→∞ gives linear).
- **Delta:** Smooth sigmoid from 0 to 1; ATM ≈ 0.5. Hedge ratio for replication.
- **Gamma:** Peaks ATM; largest convexity where uncertainty highest. Critical for rehedging.

## 6. Challenge Round

**Q1: Derive the Black-Scholes PDE from Itô's lemma and delta hedging. Why does drift μ cancel?**  
A: Start with dS = μS dt + σS dW. Consider option V(S, t). By Itô: dV = (∂V/∂t + μS ∂V/∂S + 0.5σ²S² ∂²V/∂S²) dt + σS ∂V/∂S dW. Construct delta-neutral portfolio: Π = V - Δ×S where Δ = ∂V/∂S. Then dΠ = (∂V/∂t + 0.5σ²S² ∂²V/∂S²) dt (Brownian term cancels). No-arbitrage: dΠ/Π = r dt (bond return). Therefore: ∂V/∂t + 0.5σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0. Drift μ disappears because we hedge it away; only σ and r remain (hence "risk-neutral").

**Q2: What happens to Black-Scholes when volatility σ → 0? When σ → ∞?**  
A: As σ → 0: d1, d2 → ln(S/K) + (r-q)T / 0⁺ → ±∞ (for S ≠ K). Call → max(S e^{-qT} - K e^{-rT}, 0) (deterministic payoff). Intuitively: no randomness, only forward contract. As σ → ∞: d1, d2 → ±∞; N(d1) → 1, N(d2) → 0 for any S. Call → S e^{-qT} (stock worth spot price discounted by q; strike K becomes negligible). Put → K e^{-rT} (strike worth full present value). Reason: infinite volatility makes all states equally likely; call owner captures S, put owner captures K.

**Q3: Explain the relationship between BS Gamma and realized P&L for a delta-hedged position.**  
A: Gamma is ∂²V/∂S². For small move ΔS: realized P&L ≈ Gamma × (ΔS)²/2. If Gamma > 0 (long option), you profit from large moves (buy low, rebalance up; sell high, rebalance down). If Gamma < 0 (short option), you lose. Over time: long P&L ≈ Gamma × [realized variance - implied variance]. This is how traders profit from volatility: sell high IV calls/puts, realize lower σ.

**Q4: Black-Scholes assumes log-normal prices. Why not normal prices? What breaks if prices can be negative?**  
A: Under BS, S_T ~ LogNormal, so S_T > 0 always (stock can't be negative). If prices Normal: S_T ∈ (-∞, ∞), creating arbitrage (short stock infinitely, pocket premium). Moreover: call payoff = max(S_T - K, 0) requires S_T ≥ 0 for meaningful exercise. LogNormal ensures no arbitrage on short side. For rates (can be negative): use Vasicek/Hull-White (Normal distribution under Q measure) or shifted-lognormal (adds shift parameter to ensure positivity floor).

**Q5: How do dividends affect the BS formula? Why does q reduce call value but increase put value?**  
A: With dividend yield q, forward price F = S e^{(r-q)T}. Dividend reduces future S_T (stockholder gets q×S paid out, leaving (1-q)×S to appreciate). Call payoff max(S_T - K, 0) becomes max((1-q factor effect) - K, 0) → lower. Put payoff max(K - S_T, 0) becomes higher (S_T smaller → deeper ITM more likely). Mathematically: q term reduces d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T) → N(d1) decreases for call, increases for put via N(-d1).

**Q6: What is the relationship between BS vega and the volatility smile? How do you price an ATM option if market prices show a skew?**  
A: BS assumes flat volatility (vega = ∂C/∂σ treats σ as constant). Real markets show volatility smile/skew: IV depends on strike K. OTM puts have higher IV (tail risk premium; want to pay more for downside insurance). If smile present: interpolate IV(K) from market prices, then use BS with local σ(K). More sophisticated: use Heston (stochastic volatility) or local volatility models that fit entire surface. For pricing ATM: use surface interpolation (e.g., SABR model) to extract IV(K=S), then BS. Arbitrage-free calibration ensures smooth vol surface.

## 7. Key References

- [Wikipedia: Black–Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) — Formula, Greeks, derivation, assumptions
- [Wikipedia: Itô's Lemma](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma) — Stochastic calculus, chain rule for SDEs
- Hull: *Options, Futures & Derivatives* — Chapters 13-15: BS derivation, Greeks, implied vol
- Paul Wilmott: *Introduces Quantitative Finance* — Intuitive PDE setup, replication hedging strategy

**Status:** ✓ Standalone file. **Complements:** european_call_option.md, european_put_option.md, monte_carlo_vs_black_scholes.md, greeks_delta_hedging.md
