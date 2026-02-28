# Bermudan Options

## Concept Skeleton
**Definition:** Options exercisable on specific discrete dates between European (one date) and American (continuous)  
**Purpose:** Practical early exercise approximation; computational efficiency; match contract specifications (quarterly exercise)  
**Prerequisites:** Dynamic programming, regression-based methods, Longstaff-Schwartz algorithm, optimal stopping theory

## Comparative Framing
| Feature | Bermudan | American | European | Basket |
|---------|----------|----------|----------|--------|
| **Exercise Dates** | N discrete dates | Continuous (any time) | One date (maturity) | One date |
| **Payoff** | max(S - K, 0) at exercise | max(S - K, 0) any time | max(S_T - K, 0) | max(Basket - K, 0) |
| **Pricing** | Longstaff-Schwartz | Binomial/FD/LSM | Black-Scholes | Monte Carlo |
| **Complexity** | O(N dates × M paths) | O(√ε) grid density | O(1) | O(N assets) |
| **Value** | European ≤ Bermudan ≤ American | Always | Baseline | Correlation-dependent |

## Examples + Counterexamples
**Simple Example:**  
Bermudan call on stock with quarterly exercise: T=1yr, exercise dates at t=0.25, 0.5, 0.75, 1.0 → 4 opportunities

**Failure Case:**  
Exercise dates after ex-dividend dates: Early exercise valuable for American, but Bermudan can't exercise → loses dividend capture value → cheaper

**Edge Case:**  
High exercise frequency (N → ∞): Bermudan → American limit; computational cost explodes without benefit

## Layer Breakdown
```
Bermudan Option Pricing Pipeline:
├─ Exercise Schedule:
│   ├─ Dates: t₁ < t₂ < ... < tₙ = T (N exercise opportunities)
│   ├─ European: N=1 (only T); American: N=∞ (continuous)
│   ├─ Practical: N=4-12 (monthly/quarterly)
│   └─ Δt = tᵢ₊₁ - tᵢ (interval between exercise dates)
├─ Optimal Exercise Policy:
│   ├─ At each date tᵢ: Compare intrinsic vs continuation
│   ├─ Intrinsic Value IV(tᵢ): max(S_tᵢ - K, 0) (immediate payoff)
│   ├─ Continuation Value CV(tᵢ): E[V(tᵢ₊₁) | S_tᵢ] (hold value)
│   └─ Exercise if IV(tᵢ) > CV(tᵢ) else continue
├─ Longstaff-Schwartz for Bermudan:
│   ├─ Forward Simulation:
│   │   ├─ Generate M paths of asset prices
│   │   ├─ S^m_tᵢ for m=1..M paths, i=1..N dates
│   │   └─ Store entire M×N matrix of prices
│   ├─ Backward Induction (from tₙ to t₁):
│   │   ├─ Initialize at maturity tₙ:
│   │   │   └─ CF^m = max(S^m_tₙ - K, 0) for all m
│   │   ├─ For each earlier date tᵢ (i = N-1 down to 1):
│   │   │   ├─ Filter ITM paths: I = {m : S^m_tᵢ > K}
│   │   │   ├─ Regression (on ITM paths only):
│   │   │   │   ├─ X = [1, S^m_tᵢ, (S^m_tᵢ)², ...] (basis functions)
│   │   │   │   ├─ Y = e^(-r(tᵢ₊₁-tᵢ)) CF^m (discounted future CF)
│   │   │   │   └─ β̂ = (X^T X)^{-1} X^T Y (OLS)
│   │   │   ├─ Continuation Value:
│   │   │   │   └─ CV^m = X^m β̂ (fitted value from regression)
│   │   │   ├─ Exercise Decision:
│   │   │   │   ├─ IV^m = max(S^m_tᵢ - K, 0)
│   │   │   │   └─ If IV^m > CV^m: Exercise now
│   │   │   ├─ Update Cash Flows:
│   │   │   │   ├─ If exercise: CF^m ← IV^m
│   │   │   │   └─ If continue: Keep CF^m = future payoff
│   │   │   └─ Record exercise indicator: Exercise^m_tᵢ ∈ {0,1}
│   │   └─ Discount back: CF^m ← e^(-r(tᵢ-tᵢ₋₁)) CF^m
│   └─ Final Value: V₀ = (1/M) Σ e^(-rt₁) CF^m
├─ Pricing Algorithms:
│   ├─ Monte Carlo + LSM:
│   │   ├─ Pros: Handles high dimensions, path-dependent features
│   │   ├─ Cons: Regression error, needs many paths (M ≥ 50k)
│   │   └─ Convergence: Slow (O(1/√M)), biased (low bias ~1%)
│   ├─ Binomial Trees:
│   │   ├─ Natural for discrete exercise dates
│   │   ├─ Backward induction: V(tᵢ, S) = max(IV, e^(-rΔt) E[V(tᵢ₊₁)])
│   │   └─ Accurate but exponential in dimensions (curse of dimensionality)
│   ├─ Finite Difference Methods:
│   │   ├─ PDE approach: ∂V/∂t + LV = 0 with free boundary
│   │   ├─ Exercise boundary: B(tᵢ) where IV = CV
│   │   └─ Grid-based: Accurate for low dimensions
│   └─ Dynamic Programming:
│       └─ Value iteration on exercise dates backward
├─ Exercise Boundary:
│   ├─ Optimal Boundary B(t): Stock price where IV = CV
│   ├─ Exercise Region: {S : S > B(t)} (for calls)
│   ├─ Properties:
│   │   ├─ B(T) = K (at maturity, always exercise if ITM)
│   │   ├─ B(t) increasing in t (more likely to exercise later)
│   │   └─ B(t) ≥ K always (never exercise deep OTM)
│   └─ Approximation: Fit polynomial to regression boundary
├─ Convergence & Bias:
│   ├─ Upward Bias: Using same paths for exercise decision & valuation
│   ├─ Mitigation:
│   │   ├─ Fresh paths: Generate new paths for final pricing
│   │   ├─ Cross-validation: Split into training & test sets
│   │   └─ Dual approach: Primal (LSM) gives lower bound, dual gives upper bound
│   ├─ Path Requirements: M ≥ 50k for stable regression
│   └─ Exercise Dates: More dates → better American approximation but slower
├─ Greeks:
│   ├─ Delta: ∂V/∂S (via pathwise derivatives or finite differences)
│   ├─ Gamma: ∂²V/∂S² (unstable near exercise boundary)
│   ├─ Vega: ∂V/∂σ (higher than European, lower than American)
│   ├─ Theta: ∂V/∂T (discontinuous at exercise dates)
│   └─ Rho: ∂V/∂r (affects both drift and discounting)
└─ Practical Considerations:
    ├─ Exercise Frequency:
    │   ├─ Monthly (N=12): Good balance between value and cost
    │   ├─ Quarterly (N=4): Common for equity options
    │   └─ Daily (N=252): Approaches American but computationally expensive
    ├─ Typical Applications:
    │   ├─ Swaptions: Exercise on swap payment dates
    │   ├─ Callable Bonds: Redemption on coupon dates
    │   ├─ Employee Stock Options: Vesting schedule
    │   └─ Real Options: Project go/no-go decisions at milestones
    └─ Value Hierarchy: V_European ≤ V_Bermudan ≤ V_American
        └─ Difference: V_American - V_Bermudan ~1-5% for typical parameters
```

**Interaction:** Generate forward paths → Backward regression at each exercise date → Compare IV vs CV → Update cash flows if early exercise optimal → Discount to present

## Challenge Round
**Q1:** Why is Bermudan cheaper than American? Quantify the difference.  
**A1:** American allows exercise anytime → more flexibility → higher value. Bermudan only exercises at discrete dates → may miss optimal exercise timing. Difference: Typically 1-5% of option value. For quarterly exercise (N=4), captures 80-90% of early exercise premium vs American. Converges as N → ∞.

**Q2:** When does Bermudan ≈ European (no early exercise)?  
**A2:** Non-dividend paying stock: Early exercise of call suboptimal (time value > intrinsic always). Deep OTM: Continuation value > intrinsic at all exercise dates. Very short maturity: Little time value to sacrifice → no incentive to wait, but also little gain from early exercise.

**Q3:** Exercise boundary behavior: Why B(t) increasing in t for calls?  
**A3:** Near maturity: Low time value → exercise more readily → lower boundary. Far from maturity: High time value → hold longer → higher boundary needed to exercise. B(T) = K (at maturity, exercise all ITM). Mathematically: ∂B/∂t > 0 from optimal stopping theory.

**Q4:** Regression degree choice in LSM: Why not always use high degree?  
**A4:** Low degree (1-2): Underfitting, poor CV approximation → overexercise → underpriced. High degree (5+): Overfitting, noisy CV → spurious exercise decisions → instability. Optimal: degree=2-3 for single asset (balance bias-variance tradeoff). More assets → higher degree may help.

**Q5:** Dual method for upper bound: How does it work?  
**A5:** LSM gives lower bound (suboptimal policy). Dual: Use any exercise policy π to compute upper bound via martingale stopping. V ≤ E[e^{-rt} max(S_t - K, 0) - M_t] where M_t is martingale penalizing suboptimality. Tight bounds: Upper - Lower < 1% → confidence in price.

**Q6:** Bermudan swaption: Exercise on swap payment dates. Why Bermudan not American?  
**A6:** Swap starts on exercise date → only sensible to exercise on payment dates (quarterly/semi-annually). Exercising between payments captures no additional value. Contract specification: Exercise rights only on coupon dates. Computational: Bermudan much faster than American for multi-factor interest rate models.

**Q7:** Compare Bermudan vs American for dividends. Which matters more?  
**A7:** Dividends increase early exercise value (stock drops on ex-date). American: Exercise just before ex-dividend if dividend > time value. Bermudan: Only if ex-date coincides with exercise date → may miss optimal timing. Difference significant (5-10%) if large dividend between exercise dates.

**Q8:** Path-dependent Bermudan (e.g., Asian payoff at exercise): Complexity increase?  
**A8:** Need path state variable (e.g., running average) in regression. Basis functions: f(S, A) where A = average-to-date. Higher dimensions → more paths needed (M ≥ 100k). Curse of dimensionality: Regression accuracy degrades. Alternatives: Factor models, dimension reduction (PCA).

## Key References
**Primary Sources:**
- [Bermudan Option Wikipedia](https://en.wikipedia.org/wiki/Bermudan_option) - Definition and applications
- Longstaff, F.A. & Schwartz, E.S. "Valuing American Options by Simulation" (2001) - LSM algorithm

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - Bermudan pricing (pp. 449-472)
- Andersen, L. & Broadie, M. "Primal-Dual Simulation Algorithm" (2004) - Dual upper bounds

**Thinking Steps:**
1. Define exercise schedule (quarterly, monthly, etc.)
2. Generate forward paths for all stock prices at exercise dates
3. Backward induction: At each date, regress continuation value on ITM paths
4. Compare intrinsic vs continuation → exercise if intrinsic > continuation
5. Update cash flows and discount back to present
