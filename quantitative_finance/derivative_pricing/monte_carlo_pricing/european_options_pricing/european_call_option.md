# European Call Option

## 1. Concept Skeleton
**Definition:** A financial derivative giving the holder the right, but not the obligation, to purchase an underlying asset at a predetermined strike price K on a fixed expiration date T.  
**Purpose:** Hedge downside risk, speculate on price increases, implement spread strategies, synthetic replication of forward contracts  
**Prerequisites:** Options basics, payoff diagrams, risk-neutral valuation, continuous-time stochastic processes

## 2. Comparative Framing

| Aspect | European Call | American Call | Barrier Call | Digital Call |
|--------|---------------|---------------|--------------|--------------|
| **Exercise** | At maturity T only | Any time ≤ T | At T if barrier not hit | At T if S_T > K |
| **Payoff** | max(S_T - K, 0) | Optimal early exercise | max(S_T - K, 0) × I(barrier) | 1 × I(S_T > K) |
| **Value** | Lower | Higher (early exercise option) | Lower (barrier constraint) | Binary: discrete |
| **Computation** | Black-Scholes, MC | Binomial, LSM | MC with monitoring | Closed-form for BS |

## 3. Examples + Counterexamples

**Simple Example:**  
European call with K=100, S₀=100, T=1yr, σ=20%, r=5%:  
- If S_T = 110: payoff = 10 (profit after discounting vs cost of premium)
- If S_T = 95: payoff = 0 (loss limited to premium paid)

**Failure Case:**  
Assuming constant volatility when market exhibits volatility smile: Underprices OTM calls, overprices ATM. Real option prices show skew → use Heston or local vol models.

**Edge Case:**  
Dividend payments before T: reduces call value (forward price drops when ex-dividend occurs). Standard BS assumes no dividends; adjust S₀ → S₀ exp(-qT) where q = dividend yield.

## 4. Layer Breakdown

```
European Call Option Framework:
├─ Specification:
│   ├─ Underlying Asset S (stock, commodity, currency)
│   ├─ Strike Price K (contractual exercise price)
│   ├─ Maturity T (fixed expiration date)
│   ├─ Intrinsic Value: max(S_T - K, 0)
│   └─ Time Value: Call Price - Intrinsic Value
├─ Valuation Methods:
│   ├─ Black-Scholes (closed-form, GBM assumption)
│   ├─ Binomial Tree (discrete time, flexible)
│   ├─ Monte Carlo (multivariate, exotic payoffs)
│   └─ Numerical PDE (finite difference)
├─ Pricing Drivers:
│   ├─ Moneyness: S/K (in/at/out-of-money)
│   ├─ Time to Maturity: τ = T - t
│   ├─ Volatility σ (realized + implied)
│   ├─ Interest Rate r (drift)
│   └─ Dividend Yield q (carry)
├─ Greeks (Sensitivities):
│   ├─ Delta Δ = ∂C/∂S (hedge ratio; 0 < Δ < 1)
│   ├─ Gamma Γ = ∂²C/∂S² (convexity; always > 0)
│   ├─ Vega ν = ∂C/∂σ (volatility risk; > 0)
│   ├─ Theta θ = -∂C/∂t (time decay; typically < 0)
│   └─ Rho ρ = ∂C/∂r (rate sensitivity; > 0)
└─ Payoff Diagram:
    └─ Profit/Loss vs S_T: max(0, S_T - K) - C₀
        ├─ Max Profit: Unlimited (S_T → ∞)
        ├─ Max Loss: Premium paid C₀
        └─ Break-even: S_T = K + C₀
```

**Interaction:** Call value increases with S, σ, T, r; decreases with K, q. Delta hedge requires rebalancing as S and σ change.

## 5. Mini-Project

Price a European call using Black-Scholes and Monte Carlo; visualize convergence:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve

class EuropeanCall:
    def __init__(self, S0, K, T, r, sigma, q=0):
        self.S0 = S0  # Current price
        self.K = K    # Strike
        self.T = T    # Time to maturity (years)
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
        self.q = q    # Dividend yield
    
    def black_scholes(self):
        """Closed-form Black-Scholes call price"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        call = self.S0 * np.exp(-self.q*self.T) * norm.cdf(d1) - \
               self.K * np.exp(-self.r*self.T) * norm.cdf(d2)
        return call
    
    def greek_delta(self):
        """Delta: ∂C/∂S"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        return np.exp(-self.q*self.T) * norm.cdf(d1)
    
    def greek_gamma(self):
        """Gamma: ∂²C/∂S²"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        return np.exp(-self.q*self.T) * norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
    
    def greek_vega(self):
        """Vega: ∂C/∂σ (per 1% change)"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        return self.S0 * np.exp(-self.q*self.T) * norm.pdf(d1) * np.sqrt(self.T) / 100
    
    def monte_carlo(self, n_paths=100000, n_steps=252):
        """Price using Monte Carlo simulation"""
        dt = self.T / n_steps
        np.random.seed(42)
        
        # Generate correlated Brownian paths
        Z = np.random.randn(n_paths, n_steps)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(n_steps):
            paths[:, t+1] = paths[:, t] * np.exp(
                (self.r - self.q - 0.5*self.sigma**2)*dt + 
                self.sigma*np.sqrt(dt)*Z[:, t]
            )
        
        payoff = np.maximum(paths[:, -1] - self.K, 0)
        call_price = np.exp(-self.r*self.T) * np.mean(payoff)
        std_error = np.exp(-self.r*self.T) * np.std(payoff) / np.sqrt(n_paths)
        
        return call_price, std_error, paths
    
    def convergence_test(self, n_paths_range=None):
        """Test MC convergence as N increases"""
        if n_paths_range is None:
            n_paths_range = np.logspace(2, 6, 20).astype(int)
        
        bs_price = self.black_scholes()
        mc_prices = []
        std_errors = []
        
        for n_paths in n_paths_range:
            mc_price, std_error, _ = self.monte_carlo(n_paths=n_paths, n_steps=100)
            mc_prices.append(mc_price)
            std_errors.append(std_error)
        
        return n_paths_range, mc_prices, std_errors, bs_price

# Parameters
call = EuropeanCall(S0=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02)

# Compute Black-Scholes price
bs_price = call.black_scholes()
print(f"Black-Scholes Call Price: ${bs_price:.2f}")

# Compute Greeks
delta = call.greek_delta()
gamma = call.greek_gamma()
vega = call.greek_vega()
print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: ${vega:.4f}")

# Monte Carlo pricing
mc_price, std_err, paths = call.monte_carlo(n_paths=100000, n_steps=252)
print(f"Monte Carlo Price: ${mc_price:.2f} ± ${1.96*std_err:.2f} (95% CI)")

# Convergence test
n_paths_range, mc_prices, std_errors, bs_price = call.convergence_test()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sample price paths
axes[0, 0].plot(paths[:100, :].T, alpha=0.3, linewidth=0.5)
axes[0, 0].axhline(call.K, color='r', linestyle='--', label='Strike K', linewidth=2)
axes[0, 0].set_title('100 Sample Price Paths (MC)')
axes[0, 0].set_xlabel('Time Steps')
axes[0, 0].set_ylabel('Stock Price S')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Call value vs spot price
spot_range = np.linspace(80, 120, 50)
call_values = []
for S in spot_range:
    call_temp = EuropeanCall(S, call.K, call.T, call.r, call.sigma, call.q)
    call_values.append(call_temp.black_scholes())

axes[0, 1].plot(spot_range, call_values, 'b-', linewidth=2, label='Call Value')
axes[0, 1].plot(spot_range, np.maximum(spot_range - call.K, 0), 'g--', 
                linewidth=2, label='Intrinsic Value')
axes[0, 1].axvline(call.S0, color='r', linestyle=':', alpha=0.7, label='Current S')
axes[0, 1].set_title('Call Price vs Spot Price')
axes[0, 1].set_xlabel('Spot Price S')
axes[0, 1].set_ylabel('Call Value')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: MC convergence
axes[1, 0].loglog(n_paths_range, mc_prices, 'o-', label='MC Price', linewidth=2, markersize=6)
axes[1, 0].axhline(bs_price, color='r', linestyle='--', label='BS Price', linewidth=2)
axes[1, 0].fill_between(n_paths_range, 
                        np.array(mc_prices) - 1.96*np.array(std_errors),
                        np.array(mc_prices) + 1.96*np.array(std_errors),
                        alpha=0.2, color='blue', label='95% CI')
axes[1, 0].set_title('Monte Carlo Convergence')
axes[1, 0].set_xlabel('Number of Paths (log scale)')
axes[1, 0].set_ylabel('Call Price (log scale)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Greeks across moneyness
moneyness = spot_range / call.K
deltas, gammas, vegas = [], [], []
for S in spot_range:
    call_temp = EuropeanCall(S, call.K, call.T, call.r, call.sigma, call.q)
    deltas.append(call_temp.greek_delta())
    gammas.append(call_temp.greek_gamma())
    vegas.append(call_temp.greek_vega())

ax4a = axes[1, 1]
ax4b = ax4a.twinx()
ax4c = ax4a.twinx()
ax4c.spines['right'].set_position(('outward', 60))

p1, = ax4a.plot(moneyness, deltas, 'b-', linewidth=2, label='Delta')
p2, = ax4b.plot(moneyness, gammas, 'g-', linewidth=2, label='Gamma')
p3, = ax4c.plot(moneyness, vegas, 'r-', linewidth=2, label='Vega')

ax4a.set_xlabel('Moneyness (S/K)')
ax4a.set_ylabel('Delta', color='b')
ax4b.set_ylabel('Gamma', color='g')
ax4c.set_ylabel('Vega (per 1%)', color='r')
ax4a.tick_params(axis='y', labelcolor='b')
ax4b.tick_params(axis='y', labelcolor='g')
ax4c.tick_params(axis='y', labelcolor='r')
ax4a.set_title('Greeks vs Moneyness')
ax4a.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('european_call_analysis.png', dpi=100, bbox_inches='tight')
print("Plot saved: european_call_analysis.png")
```

**Output Interpretation:**
- **Path Diagram:** Shows median and spread of S_T outcomes; intrinsic payoff captured by max(S_T - K, 0)
- **Convergence:** MC prices cluster around BS as N increases; 95% CI shrinks as O(1/√N)
- **Greeks:** Delta approaches 1 as S >> K; Gamma peaks ATM (highest sensitivity); Vega linear in time value

## 6. Challenge Round

**Q1: Why is the call value always at least as large as max(S - Ke^{-rT}, 0)?**  
A: This is the lower bound from arbitrage-free pricing. If C < max(S - Ke^{-rT}, 0), buy call and sell stock, invest K at rate r. At maturity: receive S_T from short stock, pay max(S_T - K, 0) to exercise, withdraw Ke^{rT}. For S_T ≥ K: profit = Ke^{rT} - K > 0 (arbitrage). For S_T < K: profit = S_T + Ke^{rT} - S_T = Ke^{rT} - K > 0 (locked-in arbitrage).

**Q2: How does increasing volatility σ affect a European call? Why is it counterintuitive?**  
A: Higher σ increases call value (vega > 0). This is because the call holder benefits from large upside moves but is protected on downside by the strike (payoff floor at 0). The asymmetry favors optionality. With σ=10%, tight distribution around S; with σ=40%, 5% tail chance of 2x move upside vs only 0 on downside at K.

**Q3: A European call on a dividend-paying stock is worth less than on non-dividend stock (same S, K, T, r, σ). Explain why.**  
A: Dividends reduce the stock price on ex-dividend dates. When dividend D is paid: S_t → S_t - D. This shrinks the forward price F = S_0 e^{(r-q)T}, making future S_T smaller on average. Since call payoff = max(S_T - K, 0), lower expected S_T → lower call value. Mathematically: q in BS reduces d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T).

**Q4: Why does Black-Scholes assume a European (not American) option for closed-form solution?**  
A: American options allow early exercise, introducing an optimal stopping problem. For European: value path-independent (only S_T matters). For American: must compare hold vs exercise at every node → dynamic programming required. Closed-form exists only for special cases (e.g., perpetual American call on non-dividend stock: C = S iff S ≥ K* where K* solves: K* - K = (K* - K)^{σ²/(2r)}). General case requires binomial tree or finite difference.

**Q5: If realized volatility σ_realized << implied volatility σ_IV, what happens to the written (short) call?**  
A: Short call P&L = C_premium_received - C_current_market. If σ drops, BS price drops (vega effect), so C_current < C_premium. Profit = realized variance pick-up. However, if S moves significantly in either direction, delta losses dominate (gamma loss). Net P&L depends on realized moves vs vega gain. Specifically: P&L ≈ Vega × (σ_realized² - σ_IV²) × T - Gamma × (ΔS)² (realized P&L formula).

**Q6: How would you price a European call with time-varying interest rates r(t)?**  
A: Replace discount factor e^{-rT} with discount bond price P(0, T) = exp(-∫₀ᵀ r(t) dt). Under instantaneous rate models (Vasicek, CIR), forward price F = S_0 exp(∫₀ᵀ [r(t) - q] dt). For MC: use stochastic rate process (e.g., Vasicek) correlated with equity; simulate (S_t, r_t) pairs, discount each path using realized r(t). BS no longer applies; use MC or PDE.

## 7. Key References

- [Wikipedia: Call Option](https://en.wikipedia.org/wiki/Call_option) — Definition, payoff diagram, valuation basics
- [Wikipedia: Black–Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) — Closed-form formula, Greeks, assumptions
- [Hull: Options, Futures & Derivatives](https://www-2.rotman.utoronto.ca/~hull) — Comprehensive derivatives textbook; Chapter 13: Option pricing
- Paul Wilmott: *Introduces Quantitative Finance* — Intuitive derivation of BS PDE, Greeks interpretation

**Status:** ✓ Standalone file. **Complements:** european_put_option.md, black_scholes_model.md, monte_carlo_vs_black_scholes.md
