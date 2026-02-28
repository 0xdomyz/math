# Black-Scholes Closed Form

## Concept Skeleton
**Definition:** Analytical solution for European option prices under geometric Brownian motion with constant volatility  
**Purpose:** Benchmark for option pricing; Greeks computation; market-implied volatility extraction; arbitrage-free valuation  
**Prerequisites:** Stochastic calculus (Ito's lemma), risk-neutral pricing, lognormal distribution, partial differential equations

## Comparative Framing
| Method | Black-Scholes | Binomial Tree | Monte Carlo | Finite Difference |
|--------|---------------|---------------|-------------|-------------------|
| **Computation** | O(1) instant | O(N steps) | O(M paths) | O(N time × K space) |
| **Accuracy** | Exact (under assumptions) | Converges to BS | O(1/√M) error | Discretization error |
| **Flexibility** | European only | American feasible | Exotic payoffs | PDEs, boundaries |
| **Greeks** | Analytical formulas | Finite difference | Pathwise/LR | Implicit in grid |

## Examples + Counterexamples
**Simple Example:**  
S₀=$100, K=$100, σ=20%, r=5%, T=1yr → Call=$10.45; d₁=0.35, d₂=0.15; N(d₁)=0.637

**Failure Case:**  
American put with dividends: BS undervalues (ignores early exercise); use binomial tree or Longstaff-Schwartz

**Edge Case:**  
T → 0: Call → max(S₀ - K, 0), Put → max(K - S₀, 0); d₁, d₂ → ±∞; option converges to intrinsic value

## Layer Breakdown
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

## Challenge Round
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

## Key References
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
