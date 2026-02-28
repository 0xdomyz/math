# Longstaff-Schwartz Algorithm (LSM)

## Concept Skeleton
**Definition:** Least-squares Monte Carlo method for pricing American options via regression on in-the-money paths  
**Purpose:** Estimate continuation value; compare with immediate exercise; backward induction for optimal stopping  
**Prerequisites:** American option early exercise, regression analysis, dynamic programming, Monte Carlo simulation

## Comparative Framing
| Method | LSM (Monte Carlo) | Binomial Tree | Finite Difference | Analytical (BS) |
|--------|-------------------|---------------|-------------------|-----------------|
| **American Options** | Yes (LSM regression) | Yes (backward induction) | Yes (free boundary PDE) | No (European only) |
| **Computation** | O(N paths × M steps) | O(N² steps) | O(N time × K space) | O(1) |
| **Dimensions** | Scalable (5+ assets) | 1-2 assets max | 1-3 assets | 1 asset |
| **Path Dependence** | Handles exotic features | Limited | Limited | None |
| **Early Exercise** | Approximates via regression | Exact at nodes | Exact on grid | N/A |

## Examples + Counterexamples
**Simple Example:**  
American put S₀=$100, K=$100; at step 3, S=95 → Exercise=$5; Continuation (regressed)=$3 → Exercise now

**Failure Case:**  
Deep OTM paths: Regression on few ITM points → unstable estimates; need importance sampling for rare events

**Edge Case:**  
T → 0: American → European (no time for early exercise); LSM continuation value → intrinsic value

## Layer Breakdown
```
Longstaff-Schwartz Algorithm Pipeline:
├─ Step 1: Forward Path Simulation:
│   ├─ Generate N Monte Carlo paths: S^(i)_0, S^(i)_1, ..., S^(i)_M for i=1..N
│   ├─ Time discretization: t_j = j × T/M for j=0..M
│   ├─ Euler/Milstein scheme: S_{j+1} = S_j exp((r - σ²/2)Δt + σ√Δt Z_j)
│   └─ Store all paths in memory: N × (M+1) matrix
├─ Step 2: Initialize at Maturity T:
│   ├─ Cash Flow at T: CF^(i)_M = Payoff(S^(i)_M) for all paths i
│   ├─ American Put: CF_M = max(K - S_M, 0)
│   ├─ American Call: CF_M = max(S_M - K, 0)
│   └─ Continuation Value: CV_M = 0 (no future value)
├─ Step 3: Backward Induction (for j=M-1 down to 0):
│   ├─ For Each Time Step j:
│   │   ├─ Identify In-The-Money Paths:
│   │   │   ├─ Put: ITM_j = {i : K - S^(i)_j > 0}
│   │   │   └─ Call: ITM_j = {i : S^(i)_j - K > 0}
│   │   ├─ Immediate Exercise Value:
│   │   │   └─ IV^(i)_j = Payoff(S^(i)_j) for i ∈ ITM_j
│   │   ├─ Continuation Value (via Regression):
│   │   │   ├─ Discounted Future Cash Flow: Y^(i) = e^(-rΔt) CF^(i)_{j+1}
│   │   │   ├─ Basis Functions: X^(i) = [1, S^(i)_j, (S^(i)_j)², (S^(i)_j)³, ...]
│   │   │   ├─ Least-Squares Regression: Y = Xβ + ε → β̂ = (X'X)^(-1)X'Y
│   │   │   └─ Predicted Continuation: CV^(i)_j = X^(i)β̂
│   │   ├─ Optimal Decision:
│   │   │   ├─ If IV^(i)_j > CV^(i)_j: Exercise now → CF^(i)_j = IV^(i)_j
│   │   │   └─ If IV^(i)_j ≤ CV^(i)_j: Hold → CF^(i)_j = e^(-rΔt) CF^(i)_{j+1}
│   │   └─ Update Cash Flows: CF^(i)_j for all paths
│   └─ Continue to j=j-1
├─ Step 4: Price Estimation:
│   ├─ Discount Cash Flows to t=0: PV^(i) = e^(-r t_τ(i)) CF^(i)_{τ(i)}
│   │   where τ(i) = exercise time for path i
│   ├─ Average Across Paths: V_0 = (1/N) Σ PV^(i)
│   └─ Standard Error: SE = std(PV^(i)) / √N
├─ Step 5: Basis Function Selection:
│   ├─ Polynomial: 1, S, S², S³ (typical order 2-4)
│   ├─ Laguerre: L_0(S), L_1(S), L_2(S) (orthogonal, stable)
│   ├─ Weighted: S^k × e^(-S/K) for k=0,1,2,... (emphasis near strike)
│   └─ Cross-Product (Multi-Asset): S_1, S_2, S_1 S_2, S_1², S_2²
└─ Key Considerations:
    ├─ ITM Filter: Only regress on ITM paths (avoid noise from OTM)
    ├─ Overfitting: High polynomial order → overfits → biased low prices
    ├─ Underfitting: Too few basis → misses nonlinearity → biased high prices
    ├─ Path Reuse: Same paths for regression and valuation (slight bias)
    └─ Convergence: Need N >> M for stable regression (rule: N ≥ 50M)
```

**Interaction:** Simulate paths forward → Regress backward → Compare exercise vs continuation → Update cash flows → Discount to present

## Challenge Round
**Q1:** Why regress only on ITM paths? What happens if OTM paths included?  
**A1:** OTM paths have zero immediate exercise value and near-zero continuation value → add noise to regression without information. Including OTM biases continuation estimates downward (many zeros), leading to spurious early exercise decisions. ITM filter focuses regression on relevant decision boundary.

**Q2:** Prove LSM uses same paths for regression and valuation introduces bias. Is it low or high bias?  
**A2:** Low bias (prices slightly below true value). Regression fits training data (same paths used for valuation) → overfits → overestimates continuation → exercises less often than optimal → misses some early exercise opportunities → underprices American premium. Bias typically < 1% for N >> M.

**Q3:** Compare polynomial vs Laguerre basis functions. When is each preferred?  
**A3:** Polynomial (1, S, S²): Simple, unstable for high degree (multicollinearity). Laguerre (L_k(S)): Orthogonal, numerically stable, weighted toward ITM region. Use Laguerre for stability; polynomial sufficient for low degree (2-3). Weighted polynomials S^k e^(-S/K) also effective.

**Q4:** LSM for multi-asset American options: How to construct basis for basket (S₁, S₂)?  
**A4:** Cross-product basis: 1, S₁, S₂, S₁², S₂², S₁S₂, ... Include interaction terms S₁S₂ for correlation effects. For d assets, polynomial degree p → O(p^d) terms (curse of dimensionality). Use sparse basis or neural networks for high dimensions.

**Q5:** Why does American call on non-dividend stock have zero early exercise premium (equals European)?  
**A5:** Early exercise call: Receive S - K today. Wait to T: Keep optionality + earn interest on K. Time value of strike payment (Ke^(rT) - K > 0) always exceeds early exercise benefit. LSM regression confirms: continuation value > intrinsic for all ITM paths.

**Q6:** Implement upper bound for American option via perfect foresight (dual approach). How to use with LSM?  
**A6:** Dual: V ≤ sup_τ E[e^(-rτ) Payoff(τ)]. Use LSM exercise policy as martingale in dual formulation → upper bound. True value ∈ [LSM lower bound, Dual upper bound]. Andersen-Broadie algorithm: Simulate nested paths to tighten bounds.

**Q7:** Compare LSM computational cost to binomial tree for American option. When does LSM dominate?  
**A7:** Binomial: O(N²) for N steps (recombining tree). LSM: O(M paths × K steps × p² basis). For 1D: Binomial faster (N=500 vs M=10k paths). For multi-asset (d > 2): Binomial O(N^(2d)) explodes; LSM stays O(M × K) → LSM dominates.

**Q8:** Early exercise boundary for American put: How does it change with volatility σ?  
**A8:** Higher σ → deeper boundary (exercise at lower S). Intuition: High vol increases option time value (more upside potential) → prefer holding → exercise only when very deep ITM. Low vol → shallow boundary (exercise sooner). LSM boundary visualized in plot 2.

## Key References
**Primary Sources:**
- Longstaff, F. & Schwartz, E. "Valuing American Options by Simulation" (2001) - [Original paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=191649)
- Glasserman, P. *Monte Carlo Methods in Financial Engineering* (2004) - LSM implementation (pp. 449-485)

**Technical Details:**
- Clément, E., Lamberton, D., Protter, P. "An Analysis of Least-Squares Regression" (2002) - Convergence theory
- Andersen, L. & Broadie, M. "Primal-Dual Algorithm for American Options" (2004) - Dual bounds

**Thinking Steps:**
1. Simulate forward paths using GBM (store all time steps)
2. Initialize terminal cash flows: Payoff(S_M)
3. Backward induction: For each time j from M-1 to 1
4. Filter ITM paths where immediate exercise > 0
5. Regress discounted future cash flows on current stock price (polynomial basis)
6. Compare intrinsic value vs continuation value (regression prediction)
7. Exercise if intrinsic > continuation; else hold
8. Update cash flows and discount to present
