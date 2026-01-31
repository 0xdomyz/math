# European Put Option

## 1. Concept Skeleton
**Definition:** Contract granting right (not obligation) to sell underlying asset at strike price K on expiration date T  
**Purpose:** Profit from downward price movements with limited downside (premium); protective hedge for long stock; speculation  
**Prerequisites:** Risk-neutral pricing, discounting, put-call parity, Geometric Brownian Motion

## 2. Comparative Framing
| Feature | European Put | American Put | Binary Put | Protective Put |
|---------|--------------|--------------|------------|----------------|
| **Exercise** | Maturity only | Anytime ≤ T | Maturity only | Strategy (long stock + put) |
| **Payoff** | max(K - S_T, 0) | max(K - S_t, 0) | 1 if S_T < K | S_T + max(K - S_T, 0) |
| **Pricing** | Closed-form (BS) | Numerical (LSM) | Closed-form | Sum of stock + put |
| **Value** | Lower than American | Higher (early exercise) | Fixed payout | Minimum K at maturity |

## 3. Examples + Counterexamples

**Simple Example:**  
S₀ = $100, K = $95, σ = 20%, r = 5%, T = 1yr → BS put ≈ $3.71; if S_T = $85, payoff = $10

**Failure Case:**  
American put on high-dividend stock: European formula undervalues; early exercise optimal when dividend > time value

**Edge Case:**  
Deep ITM put (K = $200, S₀ = $50): Put ~ K - S₀e^(rT); payoff certain; minimal volatility sensitivity (Vega ≈ 0)

## 4. Layer Breakdown
```
European Put Pricing Pipeline:
├─ Model Setup:
│   ├─ Asset Dynamics: dS = rS dt + σS dW (risk-neutral GBM)
│   ├─ Parameters: S₀ (spot), K (strike), T (maturity), σ (vol), r (risk-free rate)
│   └─ Payoff Asymmetry: Put protects downside; max(K - S_T, 0)
├─ Monte Carlo Simulation:
│   ├─ Path Generation: S_T = S₀ exp((r - σ²/2)T + σ√T Z_i) for Z_i ~ N(0,1)
│   ├─ Payoff Computation: P_i = max(K - S_T^(i), 0) for i = 1...N
│   ├─ Discounting: Present value = e^(-rT) × mean(P_i)
│   └─ Standard Error: SE = std(P_i) / √N → 95% CI = Price ± 1.96 SE
├─ Black-Scholes Formula:
│   ├─ d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
│   ├─ d₂ = d₁ - σ√T
│   └─ Put Price: P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
├─ Put-Call Parity:
│   ├─ Relationship: C - P = S₀ - Ke^(-rT) (no arbitrage)
│   ├─ Synthetic Put: P = C - S₀ + Ke^(-rT)
│   └─ Arbitrage Detection: If violated, buy cheap side, sell expensive side
├─ Greeks (Sensitivities):
│   ├─ Delta (Δ): N(d₁) - 1 ∈ [-1, 0]; negative hedge ratio
│   ├─ Gamma (Γ): n(d₁) / (S₀σ√T); same as call (convexity)
│   ├─ Vega (ν): S₀√T n(d₁); same as call (positive)
│   ├─ Theta (θ): Often positive for ITM puts (carry arbitrage)
│   └─ Rho (ρ): -KTe^(-rT)N(-d₂); negative (inverse rate sensitivity)
└─ Convergence Analysis:
    ├─ Error ~ O(1/√N) for standard MC
    ├─ Put-call symmetry: Put variance ≈ Call variance for ATM
    └─ Antithetic variates: Correlation Corr(P(Z), P(-Z)) < 0
```

**Interaction:** GBM paths → Terminal prices S_T → Put payoff → Discount to present → Verify parity

## 5. Mini-Project
Price European put using Monte Carlo, verify put-call parity, and analyze protective put strategy:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes analytical formulas
def black_scholes_call(S0, K, T, r, sigma):
    """European call option price (Black-Scholes)."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S0, K, T, r, sigma):
    """
    European put option price (Black-Scholes).
    
    Returns:
    - put_price: Option value
    - delta: First derivative w.r.t. S (negative for puts)
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1  # Negative for puts
    
    return put_price, delta

# Monte Carlo put pricing
def monte_carlo_put(S0, K, T, r, sigma, n_paths, antithetic=False):
    """
    Monte Carlo simulation for European put option.
    
    Returns:
    - put_price: Estimated option value
    - std_error: Standard error of estimate
    - terminal_prices: Array of simulated S_T values
    - payoffs: Array of put payoffs
    """
    if antithetic:
        n_half = n_paths // 2
        Z = np.random.randn(n_half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = np.random.randn(n_paths)
    
    # GBM terminal prices
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z_full
    terminal_prices = S0 * np.exp(drift + diffusion)
    
    # Put payoff: max(K - S_T, 0)
    payoffs = np.maximum(K - terminal_prices, 0)
    
    # Discounted expected payoff
    put_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return put_price, std_error, terminal_prices, payoffs

# Parameters
S0 = 100.0      # Current stock price
K = 100.0       # Strike price (ATM)
T = 1.0         # 1 year to maturity
r = 0.05        # 5% risk-free rate
sigma = 0.25    # 25% volatility

# Analytical Black-Scholes prices
bs_put, bs_delta_put = black_scholes_put(S0, K, T, r, sigma)
bs_call = black_scholes_call(S0, K, T, r, sigma)

print("="*60)
print("BLACK-SCHOLES PRICES")
print("="*60)
print(f"Put Price:  ${bs_put:.4f}")
print(f"Call Price: ${bs_call:.4f}")
print(f"Put Delta:  {bs_delta_put:.4f}")

# Verify put-call parity: C - P = S0 - K*exp(-rT)
parity_lhs = bs_call - bs_put
parity_rhs = S0 - K * np.exp(-r * T)
print(f"\nPut-Call Parity Check:")
print(f"  C - P = ${parity_lhs:.4f}")
print(f"  S₀ - Ke^(-rT) = ${parity_rhs:.4f}")
print(f"  Difference: ${abs(parity_lhs - parity_rhs):.6f}")

# Monte Carlo convergence analysis
np.random.seed(42)
n_paths = 100000

mc_put, mc_error, terminal_prices, put_payoffs = monte_carlo_put(
    S0, K, T, r, sigma, n_paths, antithetic=True
)

print(f"\nMonte Carlo Put Price (N={n_paths}): ${mc_put:.4f} ± ${1.96*mc_error:.4f}")
print(f"Difference from BS: ${abs(mc_put - bs_put):.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Put price surface (strike vs spot)
spots = np.linspace(50, 150, 50)
strikes = [80, 90, 100, 110, 120]
ax = axes[0, 0]
for K_i in strikes:
    put_prices = [black_scholes_put(S, K_i, T, r, sigma)[0] for S in spots]
    ax.plot(spots, put_prices, label=f'K=${K_i}', linewidth=2)
ax.axvline(S0, color='black', linestyle='--', alpha=0.5, label=f'Current S₀=${S0}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Put Option Price ($)')
ax.set_title('European Put Value vs Spot (T=1yr)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Put payoff diagram at maturity
ax = axes[0, 1]
S_range = np.linspace(50, 150, 100)
put_payoff = np.maximum(K - S_range, 0)
profit = put_payoff - bs_put  # P&L including premium paid

ax.plot(S_range, put_payoff, 'b-', linewidth=2, label='Payoff at Maturity')
ax.plot(S_range, profit, 'r-', linewidth=2, label='Profit (net premium)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(K, color='green', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.axvline(K - bs_put, color='orange', linestyle='--', linewidth=1.5, 
           label=f'Breakeven=${K - bs_put:.2f}')
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Put Value ($)')
ax.set_title('Put Payoff Diagram')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Distribution of terminal prices
ax = axes[0, 2]
ax.hist(terminal_prices, bins=60, density=True, alpha=0.7, edgecolor='black')
ax.axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike K=${K}')
ax.axvline(S0, color='green', linestyle='--', linewidth=2, label=f'Spot S₀=${S0}')
ax.axvline(np.mean(terminal_prices), color='blue', linestyle='--', linewidth=2,
           label=f'E[S_T]=${np.mean(terminal_prices):.2f}')
# Expected value under risk-neutral measure
expected_ST = S0 * np.exp(r * T)
ax.axvline(expected_ST, color='purple', linestyle=':', linewidth=2,
           label=f'S₀e^(rT)=${expected_ST:.2f}')
ax.set_xlabel('Terminal Stock Price S_T')
ax.set_ylabel('Density')
ax.set_title(f'Distribution of S_T ({n_paths:,} paths)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Put payoff distribution
ax = axes[1, 0]
ax.hist(put_payoffs, bins=60, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(np.mean(put_payoffs) * np.exp(-r*T), color='red', linestyle='--', linewidth=2,
           label=f'PV Mean: ${np.mean(put_payoffs)*np.exp(-r*T):.4f}')
ax.axvline(bs_put, color='blue', linestyle='--', linewidth=2,
           label=f'BS Price: ${bs_put:.4f}')
ax.set_xlabel('Put Payoff at Maturity')
ax.set_ylabel('Density')
ax.set_title('Distribution of Put Payoffs')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Protective put strategy (long stock + long put)
ax = axes[1, 1]
stock_profit = S_range - S0
protective_put_payoff = S_range + np.maximum(K - S_range, 0)
protective_put_profit = protective_put_payoff - S0 - bs_put

ax.plot(S_range, stock_profit, 'g--', linewidth=2, label='Long Stock Only')
ax.plot(S_range, protective_put_profit, 'b-', linewidth=2, label='Protective Put')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(K - S0 - bs_put, color='red', linestyle='--', linewidth=1.5,
           label=f'Floor=${K - S0 - bs_put:.2f}')
ax.axvline(S0, color='purple', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price at Maturity S_T')
ax.set_ylabel('Profit ($)')
ax.set_title('Protective Put Strategy (Insurance)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Put delta profile
ax = axes[1, 2]
deltas = [black_scholes_put(S, K, T, r, sigma)[1] for S in spots]
ax.plot(spots, deltas, 'b-', linewidth=2)
ax.axhline(-0.5, color='red', linestyle='--', alpha=0.5, label='Δ = -0.5 (ATM)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(S0, color='green', linestyle='--', alpha=0.5, label=f'S₀=${S0}')
ax.set_xlabel('Spot Price S')
ax.set_ylabel('Delta (∂P/∂S)')
ax.set_title('Put Delta Profile')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.05, 0.05)

plt.tight_layout()
plt.savefig('european_put_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Put-call parity verification with MC
np.random.seed(42)
mc_call_price = []
mc_put_price = []

for _ in range(100):  # 100 independent MC runs
    Z = np.random.randn(10000)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)
    
    call_payoffs = np.maximum(ST - K, 0)
    put_payoffs = np.maximum(K - ST, 0)
    
    mc_call_price.append(np.exp(-r * T) * np.mean(call_payoffs))
    mc_put_price.append(np.exp(-r * T) * np.mean(put_payoffs))

mc_parity_diff = np.array(mc_call_price) - np.array(mc_put_price) - (S0 - K * np.exp(-r * T))

print("\n" + "="*60)
print("PUT-CALL PARITY VERIFICATION (100 MC runs)")
print("="*60)
print(f"Mean C - P - (S₀ - Ke^(-rT)): ${np.mean(mc_parity_diff):.6f}")
print(f"Std Dev of Parity Error: ${np.std(mc_parity_diff):.6f}")
print(f"Max Absolute Error: ${np.max(np.abs(mc_parity_diff)):.6f}")
```

## 6. Challenge Round

**Q1:** Prove put-call parity: C - P = S₀ - Ke^(-rT). What arbitrage exists if violated?  
**A1:** Consider two portfolios at T: (A) Long call + Ke^(-rT) cash; (B) Long stock + Long put. Both worth max(S_T, K). By no-arbitrage, equal at t=0: C + Ke^(-rT) = S₀ + P. If C - P > S₀ - Ke^(-rT), sell (C + cash), buy (S + P), lock profit.

**Q2:** Why is American put worth more than European put, but American call (no dividends) equals European call?  
**A2:** Put: Early exercise can capture intrinsic value K when stock crashes (time value of money favors K today vs K at T). Call: Early exercise forfeits time value; never optimal without dividends (deferred payment of K preferable).

**Q3:** Derive put price from call via put-call parity. What does this imply about Greeks?  
**A3:** P = C - S₀ + Ke^(-rT). Differentiate: Δ_put = Δ_call - 1, Γ_put = Γ_call, ν_put = ν_call, θ_put = θ_call + rKe^(-rT), ρ_put = ρ_call - KTe^(-rT). Gamma/Vega identical; Delta shifted by -1; Theta/Rho differ by parity terms.

**Q4:** Protective put vs collar: Compare downside protection and cost.  
**A4:** Protective put: Long stock + Long put (floor at K); costs premium P. Collar: Long stock + Long put at K₁ + Short call at K₂ (K₂ > K₁); downside protected, upside capped, lower net cost (call premium offsets put).

**Q5:** For deep ITM put (S₀ << K), BS price → K - S₀e^(rT). Explain why Vega → 0.  
**A5:** When S₀ << K, exercise almost certain; payoff ≈ K - S_T with tiny probability of S_T > K. Volatility doesn't affect outcome (put expires ITM); Vega = S₀√T n(d₁) ≈ 0 as n(d₁) → 0.

**Q6:** Implement delta hedging for short put position. How does P&L differ from short call hedge?  
**A6:** Short put: Δ = N(d₁) - 1 ∈ [-1, 0]; hedge by shorting |Δ| shares (negative delta). Downside risk limited (worst case pay K - 0 = K). Rebalancing buys shares as price falls (buy low); gamma gains offset theta decay.

**Q7:** Why do puts have negative rho (ρ_put < 0) while calls have positive rho?  
**A7:** Higher rates increase forward price S₀e^(rT) → calls more likely ITM (ρ_call > 0). For puts, higher rates decrease PV of strike Ke^(-rT) → puts less valuable (ρ_put = ρ_call - KTe^(-rT) < 0).

**Q8:** Simulate put price under stochastic volatility (Heston). How does it differ from BS?  
**A8:** Heston: dν_t = κ(θ - ν_t)dt + ξ√ν_t dW_2; leverage effect (ρ_{W1,W2} < 0) → skew. OTM puts more expensive than BS (crash risk); volatility smile emerges. MC required (no closed-form for European put).

## 7. Key References

**Primary Sources:**
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) - Put option pricing formulas
- [Put-Call Parity](https://en.wikipedia.org/wiki/Put%E2%80%93call_parity) - No-arbitrage relationship
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 10: Put Option Properties

**Technical Details:**
- Cox, J.C. & Rubinstein, M. *Options Markets* (1985) - Early exercise boundaries (pp. 156-189)
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - Put pricing variance (pp. 201-218)

**Thinking Steps:**
1. Define put payoff max(K - S_T, 0) under risk-neutral measure
2. Simulate GBM terminal prices; compute put payoffs
3. Discount expected payoff to present value
4. Verify put-call parity C - P = S₀ - Ke^(-rT) with MC prices
5. Compare BS analytical solution to MC estimate (convergence check)
6. Analyze protective put strategy: minimum portfolio value = K at maturity
