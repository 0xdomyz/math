# Basket Options

## Concept Skeleton
**Definition:** Multi-asset options with payoff based on weighted portfolio of underlying assets  
**Purpose:** Diversification exposure; correlation trading; reduce single-stock risk; portfolio hedging  
**Prerequisites:** Multivariate simulation, correlation matrices, Cholesky decomposition, correlated random variables

## Comparative Framing
| Feature | Basket Option | Rainbow Option | Single-Asset European | Index Option |
|---------|---------------|----------------|----------------------|--------------|
| **Underlying** | Weighted portfolio | Best/worst of N | Single stock | Broad index |
| **Payoff** | max(Σw_i S_i - K, 0) | max(max(S_i) - K, 0) | max(S - K, 0) | max(Index - K, 0) |
| **Correlation Impact** | High (diversification) | Moderate | N/A | Implicit in index |
| **Pricing** | Monte Carlo | Monte Carlo | Black-Scholes | Black-Scholes |
| **Complexity** | O(N assets × M paths) | O(N × M) | O(1) | O(1) |

## Examples + Counterexamples
**Simple Example:**  
Basket call on tech stocks: 0.4×AAPL + 0.3×MSFT + 0.3×GOOGL; K=$100 → Payoff = max(weighted sum - 100, 0)

**Failure Case:**  
Perfect correlation (ρ=1): Basket behaves like single asset with blended volatility; loses diversification benefit

**Edge Case:**  
Zero correlation: Maximum diversification → basket volatility = σ/√N → cheaper than single-asset option

## Layer Breakdown
```
Basket Option Pricing Pipeline:
├─ Basket Definition:
│   ├─ Assets: S₁, S₂, ..., Sₙ (N underlyings)
│   ├─ Weights: w₁, w₂, ..., wₙ where Σwᵢ = 1
│   ├─ Basket Value: B_t = Σ wᵢ Sᵢ(t)
│   └─ Payoff: Call = max(B_T - K, 0); Put = max(K - B_T, 0)
├─ Correlation Structure:
│   ├─ Correlation Matrix ρ:
│   │   ├─ ρᵢⱼ = Corr(Sᵢ, Sⱼ) ∈ [-1, 1]
│   │   ├─ ρᵢᵢ = 1 (self-correlation)
│   │   └─ Symmetric: ρᵢⱼ = ρⱼᵢ
│   ├─ Covariance Matrix Σ:
│   │   └─ Σᵢⱼ = ρᵢⱼ σᵢ σⱼ
│   └─ Basket Volatility:
│       └─ σ_basket = √(Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ)
├─ Monte Carlo Simulation:
│   ├─ Correlated Random Variables:
│   │   ├─ Generate Independent Z: Z₁, ..., Zₙ ~ N(0, 1)
│   │   ├─ Cholesky Decomposition: ρ = L L^T (lower triangular L)
│   │   ├─ Correlated Normals: X = L Z → Cov(X) = ρ
│   │   └─ X_i = Σⱼ Lᵢⱼ Zⱼ for i=1..N
│   ├─ Path Generation (for each asset i):
│   │   ├─ S^i_{t+1} = S^i_t exp((rᵢ - σᵢ²/2)Δt + σᵢ√Δt Xᵢ_t)
│   │   ├─ Different drifts: rᵢ (risk-free rates may differ)
│   │   └─ Different vols: σᵢ (each asset has own volatility)
│   ├─ Basket Value at Each Step:
│   │   └─ B_t = Σ wᵢ S^i_t
│   ├─ Terminal Payoff:
│   │   └─ Call: max(B_T - K, 0)
│   └─ Present Value:
│       └─ PV = e^(-rT) × Payoff
├─ Variance Reduction:
│   ├─ Control Variate:
│   │   ├─ Use single asset with similar characteristics
│   │   ├─ Or use geometric basket (has closed-form approximation)
│   │   └─ Correlation typically 0.8-0.95
│   ├─ Antithetic Variates:
│   │   ├─ Z and -Z → Negatively correlated basket values
│   │   └─ Preserves correlation structure (LZ and L(-Z))
│   ├─ Moment Matching:
│   │   └─ Force Σwᵢ S^i_T = Σwᵢ S₀^i e^(rT) (expected value)
│   └─ Stratified Sampling:
│       └─ Stratify on basket terminal value B_T
├─ Correlation Impact:
│   ├─ High Correlation (ρ → 1):
│   │   ├─ Basket behaves like single asset
│   │   ├─ Basket vol → weighted average of individual vols
│   │   └─ Option expensive (no diversification benefit)
│   ├─ Low Correlation (ρ → 0):
│   │   ├─ Maximum diversification
│   │   ├─ Basket vol → σ_avg / √N
│   │   └─ Option cheaper (low basket volatility)
│   └─ Negative Correlation (ρ < 0):
│       ├─ Offsetting movements → very low basket vol
│       └─ Option very cheap (hedge-like behavior)
├─ Greeks:
│   ├─ Deltas: ∂V/∂Sᵢ (one per asset; vector of N deltas)
│   ├─ Gammas: ∂²V/∂Sᵢ² (diagonal) and ∂²V/∂Sᵢ∂Sⱼ (cross-gammas)
│   ├─ Vega: ∂V/∂σᵢ (per-asset vega; changes with weight wᵢ)
│   ├─ Correlation Greeks:
│   │   ├─ Cega: ∂V/∂ρᵢⱼ (sensitivity to correlation changes)
│   │   └─ Important for correlation trading strategies
│   └─ Theta: ∂V/∂t (time decay; similar to single-asset)
└─ Approximations:
    ├─ Moment Matching: Approximate basket with lognormal distribution
    ├─ Curran's Approximation: Condition on geometric average
    ├─ Geometric Basket: Use geometric average (has closed-form)
    └─ Taylor Expansion: Approximate basket dynamics near current value
```

**Interaction:** Generate correlated paths via Cholesky → Compute weighted basket value → Payoff on basket → Discount to present

## Challenge Round
**Q1:** Derive basket volatility formula σ_B = √(w^T Σ w). What does it reveal about diversification?  
**A1:** Var(Basket) = Var(Σw_i S_i) = ΣΣ w_i w_j Cov(S_i, S_j) = w^T Σ w where Σ_ij = σ_i σ_j ρ_ij. For equal weights, uncorrelated assets: σ_B = σ_avg/√N → diversification reduces volatility. Perfect correlation: σ_B = Σw_i σ_i (no benefit).

**Q2:** Why is basket option cheaper than sum of individual options (Σ Call_i)?  
**A2:** Jensen's inequality: E[max(Basket - K, 0)] < Σ E[max(S_i - K_i, 0)] for convex payoff. Diversification reduces basket volatility → lower option value. Portfolio of options has independent payoffs; basket has correlated payoffs → less total volatility → cheaper.

**Q3:** Correlation Greeks (Cega): ∂V/∂ρ_ij. Sign and magnitude?  
**A3:** Higher correlation → higher basket vol → higher option value → Cega > 0 for long basket calls. Magnitude: Largest for ρ near 0 (steepest slope); smaller for ρ → 1 (flattens). Used for correlation trading: Bet on correlation changes via basket options.

**Q4:** Cholesky fails if correlation matrix not positive semi-definite. When does this occur?  
**A4:** Inconsistent correlations: e.g., ρ₁₂=0.9, ρ₁₃=0.9, ρ₂₃=-0.9 (contradictory). Matrix must satisfy Σx^T ρ x ≥ 0 for all x. Check: All eigenvalues ≥ 0. If fails, use nearest positive-definite matrix (Higham algorithm) or PCA to reduce dimensions.

**Q5:** Basket on indices (S&P 500, FTSE, Nikkei): Why more expensive than individual index options?  
**A5:** Cross-border correlations typically 0.5-0.7 (not perfect). Basket captures global exposure → higher effective volatility than single index. Time zone differences → asynchronous moves → adds uncertainty. Currency risk if multi-currency basket.

**Q6:** Quanto basket: Assets denominated in different currencies, payoff in single currency. Complexity?  
**A6:** Need 3-way correlation: (Asset 1, Asset 2, FX 1/2). Simulate asset prices in local currency, then convert using FX paths. Compo risk: Correlation between asset and FX. Greeks become multi-dimensional (asset deltas, FX deltas, cross-Greeks).

**Q7:** Compare basket option to spread option max(S₁ - S₂ - K, 0). Which is more complex?  
**A7:** Spread is 2-asset basket with weights [1, -1]. Basket more general (N assets, any weights). Complexity similar: Both need correlated paths. Spread has numerical challenges when S₁ ≈ S₂ (near-zero strike). Basket more stable with diversified weights.

**Q8:** Moment matching for basket: Approximate basket distribution as lognormal. How to match first two moments?  
**A8:** E[B_T] = Σw_i S₀^i e^(rT). Var[ln(B_T)] harder (basket not lognormal). Approximation: Use basket volatility σ_B from formula, treat B as GBM. Pluginto BS formula with (B₀, σ_B, K). Accurate for ATM, high correlation; breaks down for deep OTM/low correlation.

## Key References
**Primary Sources:**
- [Basket Option Wikipedia](https://en.wikipedia.org/wiki/Basket_option) - Overview and correlation impact
- Gentle, J.E. "Random Number Generation and Monte Carlo Methods" (2003) - Cholesky decomposition
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Multi-asset options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Correlated paths (pp. 71-101)
- Curran, M. "Valuing Asian and Basket Options" (1994) - Moment matching approximations

**Thinking Steps:**
1. Define basket weights and correlation matrix (positive semi-definite)
2. Cholesky decomposition: ρ = LL^T for correlated random generation
3. Generate independent normals Z; transform to correlated X = LZ
4. Simulate each asset with GBM using correlated X_i
5. Compute basket value B_t = Σw_i S^i_t at each step
6. Payoff on terminal basket value; discount to present
