# European Put Option

## Concept Skeleton
**Definition:** Contract granting right (not obligation) to sell underlying asset at strike price K on expiration date T  
**Purpose:** Profit from downward price movements with limited downside (premium); protective hedge for long stock; speculation  
**Prerequisites:** Risk-neutral pricing, discounting, put-call parity, Geometric Brownian Motion

## Comparative Framing
| Feature | European Put | American Put | Binary Put | Protective Put |
|---------|--------------|--------------|------------|----------------|
| **Exercise** | Maturity only | Anytime ≤ T | Maturity only | Strategy (long stock + put) |
| **Payoff** | max(K - S_T, 0) | max(K - S_t, 0) | 1 if S_T < K | S_T + max(K - S_T, 0) |
| **Pricing** | Closed-form (BS) | Numerical (LSM) | Closed-form | Sum of stock + put |
| **Value** | Lower than American | Higher (early exercise) | Fixed payout | Minimum K at maturity |

## Examples + Counterexamples
**Simple Example:**  
S₀ = $100, K = $95, σ = 20%, r = 5%, T = 1yr → BS put ≈ $3.71; if S_T = $85, payoff = $10

**Failure Case:**  
American put on high-dividend stock: European formula undervalues; early exercise optimal when dividend > time value

**Edge Case:**  
Deep ITM put (K = $200, S₀ = $50): Put ~ K - S₀e^(rT); payoff certain; minimal volatility sensitivity (Vega ≈ 0)

## Layer Breakdown
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

## Challenge Round
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

## Key References
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
