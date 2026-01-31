# European Put Option

## 1. Concept Skeleton
**Definition:** A financial derivative giving the holder the right, but not the obligation, to sell an underlying asset at a predetermined strike price K on a fixed expiration date T.  
**Purpose:** Hedge downside risk (portfolio insurance), speculate on price declines, implement protective/synthetic strategies, shorting without borrowing  
**Prerequisites:** Call-put parity, intrinsic vs time value, payoff functions, risk-neutral valuation

## 2. Comparative Framing

| Aspect | European Put | European Call | American Put | Barrier Put |
|--------|--------------|---------------|--------------|-------------|
| **Payoff at T** | max(K - S_T, 0) | max(S_T - K, 0) | Optimal early exercise | max(K - S_T, 0) × I(barrier) |
| **Max Profit** | K - premium | Unlimited - premium | K - premium (higher) | K - premium (lower) |
| **Intrinsic Value** | max(K - S_T, 0) | max(S_T - K, 0) | Same; often exercised early | Limited by barrier |
| **Time Decay (θ)** | Typically positive | Typically negative | More positive (early ex) | Varies |
| **Dividend Impact** | Increases value | Decreases value | Increases early ex incentive | Model-dependent |

## 3. Examples + Counterexamples

**Simple Example:**  
European put with K=100, S₀=100, T=1yr, σ=20%, r=5%:  
- If S_T = 90: payoff = 10 (profit: exercise, sell at 100)
- If S_T = 110: payoff = 0 (expire worthless, limit loss to premium)

**Failure Case:**  
Hedging portfolio with out-of-the-money puts (K < S) on significant price drops. If market crashes 30% overnight, put is suddenly ITM but illiquid → cannot sell at fair value, hedging fails due to liquidity gap.

**Edge Case:**  
Zero-dividend-paying bond (cash) put: As T → 0, European put value → max(K - S_T, 0) but American put can be exercised immediately for max(K - S_0, 0). European value < American value significantly near maturity when S near K.

## 4. Layer Breakdown

```
European Put Option Framework:
├─ Specification:
│   ├─ Underlying Asset S (stock, index, commodity)
│   ├─ Strike Price K (exercise price, floor on stock value)
│   ├─ Maturity T (fixed expiration date)
│   ├─ Intrinsic Value: max(K - S_T, 0)
│   └─ Time Value: Put Price - Intrinsic Value
├─ Put-Call Parity:
│   ├─ Relationship: C - P = S e^{-qT} - K e^{-rT}
│   ├─ Arbitrage: If violated, buy cheap leg, sell expensive
│   ├─ Static hedge: Long stock + long put = synthetic call
│   └─ Used for: European options valuation cross-check
├─ Valuation Methods:
│   ├─ Black-Scholes (closed-form, GBM)
│   ├─ Binomial Tree (discrete, early exercise)
│   ├─ Monte Carlo (flexible payoff definition)
│   └─ Finite Difference PDE (numerical)
├─ Pricing Drivers:
│   ├─ Spot Price S (lower S → higher put value)
│   ├─ Strike K (higher K → higher put value)
│   ├─ Time to Maturity T (longer T → usually higher value)
│   ├─ Volatility σ (higher σ → higher put value, vega > 0)
│   ├─ Interest Rate r (higher r → lower put value, rho < 0)
│   └─ Dividend Yield q (higher q → higher put value)
├─ Greeks (Sensitivities):
│   ├─ Delta Δ = ∂P/∂S (always < 0 for put, hedge -1 share)
│   ├─ Gamma Γ = ∂²P/∂S² (always > 0, same as call)
│   ├─ Vega ν = ∂P/∂σ (> 0, like call; benefit from volatility)
│   ├─ Theta θ = -∂P/∂t (often > 0 for European put)
│   └─ Rho ρ = ∂P/∂r (< 0, opposite to call)
└─ Portfolio Insurance:
    ├─ Strategy: Long stock + long put on stock
    ├─ Payoff: max(S_T, K) (floor at K, unlimited upside)
    ├─ Cost: S_0 + P_0 (stock + put premium)
    └─ Net: Protected portfolio, insurance cost paid upfront
```

**Interaction:** Put value increases with K and σ, decreases with S and r. Time decay reversed (θ often positive) due to insurance value. Optimal early exercise for American puts when deep ITM + high rates.

## 5. Mini-Project

Price a European put using Black-Scholes and Monte Carlo; compare put-call parity:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class EuropeanPut:
    def __init__(self, S0, K, T, r, sigma, q=0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def black_scholes(self):
        """Closed-form Black-Scholes put price"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        put = self.K * np.exp(-self.r*self.T) * norm.cdf(-d2) - \
              self.S0 * np.exp(-self.q*self.T) * norm.cdf(-d1)
        return put
    
    def put_call_parity(self, call_price=None):
        """Verify put-call parity: C - P = S e^{-qT} - K e^{-rT}"""
        if call_price is None:
            # Compute call from put using parity
            put_price = self.black_scholes()
            call_price = put_price + self.S0 * np.exp(-self.q*self.T) - \
                        self.K * np.exp(-self.r*self.T)
            return call_price
        else:
            # Verify parity
            put_price = self.black_scholes()
            lhs = call_price - put_price
            rhs = self.S0 * np.exp(-self.q*self.T) - self.K * np.exp(-self.r*self.T)
            parity_error = abs(lhs - rhs)
            return parity_error
    
    def greek_delta(self):
        """Delta: ∂P/∂S (always < 0 for put)"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        return -np.exp(-self.q*self.T) * norm.cdf(-d1)
    
    def greek_gamma(self):
        """Gamma: ∂²P/∂S² (same as call)"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        return np.exp(-self.q*self.T) * norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
    
    def greek_theta(self):
        """Theta: ∂P/∂t (often > 0 for European put)"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        theta = -self.S0 * np.exp(-self.q*self.T) * norm.pdf(d1) * self.sigma / (2*np.sqrt(self.T)) + \
                self.q * self.S0 * np.exp(-self.q*self.T) * norm.cdf(-d1) - \
                self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-d2)
        return theta / 365  # Per day
    
    def monte_carlo(self, n_paths=100000, n_steps=252):
        """Price using Monte Carlo simulation"""
        dt = self.T / n_steps
        np.random.seed(42)
        
        Z = np.random.randn(n_paths, n_steps)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(n_steps):
            paths[:, t+1] = paths[:, t] * np.exp(
                (self.r - self.q - 0.5*self.sigma**2)*dt + 
                self.sigma*np.sqrt(dt)*Z[:, t]
            )
        
        payoff = np.maximum(self.K - paths[:, -1], 0)
        put_price = np.exp(-self.r*self.T) * np.mean(payoff)
        std_error = np.exp(-self.r*self.T) * np.std(payoff) / np.sqrt(n_paths)
        
        return put_price, std_error, paths

# Parameters
put = EuropeanPut(S0=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02)

# Compute Black-Scholes price
bs_put_price = put.black_scholes()
print(f"Black-Scholes Put Price: ${bs_put_price:.2f}")

# Verify put-call parity
bs_call_price = put.put_call_parity()
print(f"Black-Scholes Call Price (from parity): ${bs_call_price:.2f}")

# Check parity error
from scipy.stats import norm
d1 = (np.log(put.S0/put.K) + (put.r - put.q + 0.5*put.sigma**2)*put.T) / \
     (put.sigma * np.sqrt(put.T))
d2 = d1 - put.sigma * np.sqrt(put.T)
call_direct = put.S0 * np.exp(-put.q*put.T) * norm.cdf(d1) - \
              put.K * np.exp(-put.r*put.T) * norm.cdf(d2)
parity_error = put.put_call_parity(call_direct)
print(f"Put-Call Parity Error: ${parity_error:.6f}")

# Compute Greeks
delta = put.greek_delta()
gamma = put.greek_gamma()
theta_daily = put.greek_theta()
print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Theta: ${theta_daily:.4f}/day")

# Monte Carlo pricing
mc_put_price, std_err, paths = put.monte_carlo(n_paths=100000, n_steps=252)
print(f"Monte Carlo Put Price: ${mc_put_price:.2f} ± ${1.96*std_err:.2f} (95% CI)")

# Portfolio Insurance Analysis
portfolio_value = put.S0 + bs_put_price
floor_value = put.K
print(f"\nPortfolio Insurance (Long Stock + Long Put):")
print(f"Current Portfolio Value: ${portfolio_value:.2f}")
print(f"Protected Floor: ${floor_value:.2f}")
print(f"Insurance Cost: ${bs_put_price:.2f} ({100*bs_put_price/put.S0:.1f}% of stock price)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Put value vs spot price
spot_range = np.linspace(80, 120, 50)
put_values = []
call_values = []
for S in spot_range:
    put_temp = EuropeanPut(S, put.K, put.T, put.r, put.sigma, put.q)
    put_values.append(put_temp.black_scholes())
    call_values.append(put_temp.put_call_parity())

axes[0, 0].plot(spot_range, put_values, 'b-', linewidth=2, label='Put Value')
axes[0, 0].plot(spot_range, np.maximum(put.K - spot_range, 0), 'g--', 
                linewidth=2, label='Intrinsic Value')
axes[0, 0].axvline(put.S0, color='r', linestyle=':', alpha=0.7, label='Current S')
axes[0, 0].set_title('Put Price vs Spot Price')
axes[0, 0].set_xlabel('Spot Price S')
axes[0, 0].set_ylabel('Put Value')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Put-Call Parity
axes[0, 1].plot(spot_range, call_values, 'b-', linewidth=2, label='Call (from parity)')
axes[0, 1].plot(spot_range, put_values, 'g-', linewidth=2, label='Put')
axes[0, 1].plot(spot_range, spot_range * np.exp(-put.q*put.T) - 
                put.K * np.exp(-put.r*put.T), 'r--', linewidth=2, 
                label='C - P = S e^{-qT} - K e^{-rT}')
axes[0, 1].set_title('Put-Call Parity Relationship')
axes[0, 1].set_xlabel('Spot Price S')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Portfolio Insurance Payoff
axes[1, 0].plot(spot_range, spot_range, 'g-', linewidth=2, label='Long Stock')
axes[1, 0].plot(spot_range, put_values, 'b-', linewidth=2, label='Long Put')
axes[1, 0].plot(spot_range, spot_range + np.array(put_values) - bs_put_price, 
                'r-', linewidth=2.5, label='Protected Portfolio')
axes[1, 0].axhline(put.K, color='k', linestyle='--', alpha=0.5, label='Floor (K)')
axes[1, 0].set_title('Portfolio Insurance: Long Stock + Long Put')
axes[1, 0].set_xlabel('Stock Price at Maturity S_T')
axes[1, 0].set_ylabel('Portfolio Value')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Put Greeks across moneyness
moneyness = spot_range / put.K
deltas, gammas, thetas = [], [], []
for S in spot_range:
    put_temp = EuropeanPut(S, put.K, put.T, put.r, put.sigma, put.q)
    deltas.append(put_temp.greek_delta())
    gammas.append(put_temp.greek_gamma())
    thetas.append(put_temp.greek_theta())

ax4a = axes[1, 1]
ax4b = ax4a.twinx()
ax4c = ax4a.twinx()
ax4c.spines['right'].set_position(('outward', 60))

p1, = ax4a.plot(moneyness, deltas, 'b-', linewidth=2, label='Delta')
p2, = ax4b.plot(moneyness, gammas, 'g-', linewidth=2, label='Gamma')
p3, = ax4c.plot(moneyness, thetas, 'r-', linewidth=2, label='Theta (daily)')

ax4a.set_xlabel('Moneyness (S/K)')
ax4a.set_ylabel('Delta', color='b')
ax4b.set_ylabel('Gamma', color='g')
ax4c.set_ylabel('Theta', color='r')
ax4a.tick_params(axis='y', labelcolor='b')
ax4b.tick_params(axis='y', labelcolor='g')
ax4c.tick_params(axis='y', labelcolor='r')
ax4a.set_title('Put Greeks vs Moneyness')
ax4a.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('european_put_analysis.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: european_put_analysis.png")
```

**Output Interpretation:**
- **Payoff Diagram:** Shows protected floor at K; downside capped, upside unlimited
- **Parity Check:** Call - Put = S e^{-qT} - K e^{-rT} verified; useful for arbitrage detection
- **Theta:** Often positive (time decay favors holder), especially for OTM puts

## 6. Challenge Round

**Q1: Why is the European put value always at least max(Ke^{-rT} - S, 0)?**  
A: Lower bound from arbitrage. If P < max(Ke^{-rT} - S, 0), buy put and stock, borrow K at rate r. At maturity: exercise put (sell at K), repay Ke^{rT}. If S_T ≥ K: receive K, pay Ke^{rT}, loss = Ke^{rT} - K < 0 (violates if P near 0). If S_T < K: receive K, pay S_T on open market, net = K - S_T (captured by put payoff). Arbitrage condition forces P ≥ max(Ke^{-rT} - S, 0).

**Q2: A European put on a dividend-paying stock is worth MORE than on a non-dividend stock. Why?**  
A: Dividend payments reduce S_t on ex-dates, lowering expected S_T → higher put payoff max(K - S_T, 0). Moreover, investors holding stock receive dividends (opportunity lost if short for hedging). Put gains value since it compensates for dividend downside. Mathematically: q increases d1 in negative direction, reducing N(-d1) term, increasing put value.

**Q3: What is the relationship between put delta and the probability of exercise (S_T < K)?**  
A: Put delta Δ_put ≈ -e^{-qT} N(-d1) ≈ -P(exercise), where N(-d1) ≈ P(ITM under risk-neutral measure). More precisely: Δ_put = -N(-d1) measures sensitivity, not raw probability (which is ≈ N(-d2) for European option). Delta represents the hedging quantity: short Δ_put shares to delta-hedge (e.g., Δ_put = -0.3 means short 0.3 shares per put).

**Q4: A portfolio manager insures a $1M stock portfolio with puts. How does time decay affect the protection?**  
A: Near expiry, unexercised OTM puts approach zero value → protection evaporates. If market rallied (stock up 10%), insurance put expires worthless. Manager must "roll forward": sell expiring puts, buy new puts further out. Cost = lost time value + new premium. If market fell (stock down 5%), put is ITM, exercise to realize protection, then re-insure. Time decay creates rolling cost (insurance drag on returns) quantified by theta.

**Q5: How do you price a put on a zero-coupon bond (guarantee of minimum sale price)?**  
A: Treat bond as underlying asset S (price moves based on interest rates). If bond has floor K (put strike), payoff at T = max(K - B_T, 0), where B_T is bond value at maturity. Key difference: bond price inversely correlates with rates; BS model with σ estimated from bond price history (not rate volatility directly). For interest rate puts (caps/floors on rates), use rate models (Vasicek, CIR) not Black-Scholes.

**Q6: Explain why American puts are often exercised early, but American calls rarely are.**  
A: For puts: early exercise captures the strike price K immediately (earning r × time value on K). If stock crashes, put is deep ITM, interest earned on K dominates time decay. For calls: early exercise forgoes dividends + time value optionality; investor owns stock (receives dividends but delayed capital gains). Only exercised if dividend yield q very high (early exercise captures upcoming dividend), rare in practice. American put value = European put value + early exercise premium (typically 0-15%).

## 7. Key References

- [Wikipedia: Put Option](https://en.wikipedia.org/wiki/Put_option) — Definition, payoff, protective strategies
- [Wikipedia: Put-Call Parity](https://en.wikipedia.org/wiki/Put%E2%80%93call_parity) — Arbitrage-free relationship, replication
- [Hull: Options, Futures & Derivatives](https://www-2.rotman.utoronto.ca/~hull) — Chapter 13-14: Puts, portfolio insurance
- Paul Wilmott: *Introduces Quantitative Finance* — Greeks computation, hedging dynamics

**Status:** ✓ Standalone file. **Complements:** european_call_option.md, put_call_parity.md, portfolio_insurance.md
