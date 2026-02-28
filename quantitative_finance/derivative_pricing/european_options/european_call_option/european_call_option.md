# European Call Option

## Concept Skeleton
**Definition:** Contract granting right (not obligation) to buy underlying asset at strike price K on expiration date T  
**Purpose:** Profit from upward price movements with limited downside (premium paid); hedging long positions; speculation  
**Prerequisites:** Risk-neutral pricing, discounting, Geometric Brownian Motion, payoff functions

## Comparative Framing
| Feature | European Call | American Call | Digital Call | Asian Call |
|---------|---------------|---------------|--------------|------------|
| **Exercise** | Maturity only | Anytime ≤ T | Maturity only | Maturity only |
| **Payoff** | max(S_T - K, 0) | max(S_t - K, 0) | 1 if S_T > K | max(Avg(S) - K, 0) |
| **Pricing** | Closed-form (BS) | Numerical | Closed-form | Monte Carlo |
| **Value** | Lower bound for American | Higher (early exercise premium) | Fixed payout | Path-dependent |

## Examples + Counterexamples
**Simple Example:**  
S₀ = $100, K = $105, σ = 20%, r = 5%, T = 1yr → BS price ≈ $8.02; if S_T = $115, payoff = $10

**Failure Case:**  
Dividends paid before expiry: Standard BS overvalues; adjust S₀ → S₀e^(-qT) where q = dividend yield

**Edge Case:**  
Deep OTM call (K = $200, S₀ = $100): Small probability, high gamma; MC converges slowly → importance sampling

## Layer Breakdown
```
European Call Pricing Pipeline:
├─ Model Setup:
│   ├─ Asset Dynamics: dS = rS dt + σS dW (risk-neutral GBM)
│   ├─ Parameters: S₀ (spot), K (strike), T (maturity), σ (vol), r (risk-free rate)
│   └─ Risk-Neutral Measure: Drift μ → r (no arbitrage)
├─ Monte Carlo Simulation:
│   ├─ Path Generation: S_T = S₀ exp((r - σ²/2)T + σ√T Z_i) for Z_i ~ N(0,1)
│   ├─ Payoff Computation: C_i = max(S_T^(i) - K, 0) for i = 1...N
│   ├─ Discounting: Present value = e^(-rT) × mean(C_i)
│   └─ Standard Error: SE = std(C_i) / √N → 95% CI = Price ± 1.96 SE
├─ Black-Scholes Benchmark:
│   ├─ d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
│   ├─ d₂ = d₁ - σ√T
│   └─ Call Price: C = S₀N(d₁) - Ke^(-rT)N(d₂)
├─ Greeks (Sensitivities):
│   ├─ Delta (Δ): N(d₁) ∈ [0, 1]; hedge ratio
│   ├─ Gamma (Γ): n(d₁) / (S₀σ√T); convexity
│   ├─ Vega (ν): S₀√T n(d₁); volatility sensitivity
│   ├─ Theta (θ): Time decay (usually negative)
│   └─ Rho (ρ): KTe^(-rT)N(d₂); rate sensitivity
└─ Convergence Analysis:
    ├─ Error ~ O(1/√N) for standard MC
    ├─ Quasi-MC (Sobol): O((ln N)^d / N)
    └─ Variance reduction: Control variates, antithetic paths
```

**Interaction:** GBM paths → Terminal prices S_T → Payoff function → Discount to present → Aggregate

## Challenge Round
**Q1:** Why does risk-neutral pricing replace drift μ with r in asset dynamics?  
**A1:** No-arbitrage: Portfolio of option + hedge replicates risk-free bond; expected return must be r else arbitrage exists.

**Q2:** For deep OTM calls (K >> S₀), MC converges slowly. Why and how to fix?  
**A2:** Few paths contribute non-zero payoff → high variance. Use importance sampling: shift distribution toward ITM region, reweight samples.

**Q3:** Compare MC computational cost to BS for European call. When is MC justified?  
**A3:** MC: O(N paths × M time steps); BS: O(1). MC justified for path-dependent payoffs (Asians, barriers) or non-lognormal models (jumps, stochastic vol).

**Q4:** Derive Black-Scholes formula intuitively without stochastic calculus.  
**A4:** Under risk-neutral measure, S_T ~ lognormal. Call price = e^(-rT) E[max(S_T - K, 0)] = e^(-rT) ∫ₖ^∞ (S - K) f(S) dS where f is lognormal PDF. Evaluate integral → BS formula.

**Q5:** How does call price change with volatility σ? Explain via payoff convexity.  
**A5:** Higher σ increases probability mass in tails; max(S_T - K, 0) is convex → Jensen's inequality: E[max(S_T - K, 0)] increases with σ. Vega always positive for calls.

**Q6:** Implement delta hedging: Replicate call by holding Δ shares. What is P&L at maturity?  
**A6:** Dynamic hedge: Hold Δ(t) = N(d₁(t)) shares, rebalance continuously. Under BS assumptions, P&L = 0 (perfect replication). With transaction costs or discrete rebalancing → tracking error.

**Q7:** Why is call price bounded below by intrinsic value (S₀ - Ke^(-rT))?  
**A7:** Otherwise arbitrage: Buy call for C, short stock for S₀, invest K at rate r. At maturity, exercise call (pay K, get S_T), close short (deliver S_T), profit = S₀ - C - K. If C < S₀ - Ke^(-rT), profit > 0.

**Q8:** Simulate jump-diffusion (Merton model) and price call. How does it differ from GBM?  
**A8:** Add Poisson jumps: dS = rS dt + σS dW + S J dN where J ~ log-normal jump size, N ~ Poisson. Jumps increase tail risk → higher OTM call prices (fat tails). No closed-form; MC required.

## Key References
**Primary Sources:**
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) - Foundational option pricing theory
- [Monte Carlo Methods in Finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance) - Simulation techniques
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 13: Pricing European Options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - Variance reduction (pp. 185-243)
- Shreve, S. *Stochastic Calculus for Finance II* (2004) - Risk-neutral pricing (pp. 215-256)

**Thinking Steps:**
1. Define risk-neutral dynamics (drift → r) for no-arbitrage
2. Simulate terminal prices S_T under GBM with lognormal increments
3. Compute payoffs max(S_T - K, 0) for each path
4. Discount expected payoff at rate r to present value
5. Quantify convergence via standard error O(1/√N)
6. Benchmark against Black-Scholes closed-form solution
