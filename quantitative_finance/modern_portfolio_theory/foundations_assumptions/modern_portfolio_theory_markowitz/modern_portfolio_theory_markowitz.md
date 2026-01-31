# Modern Portfolio Theory (Markowitz) & Mean-Variance Framework

## 1. Concept Skeleton
**Definition:** Modern Portfolio Theory (MPT): Mathematical framework for constructing portfolios that maximize expected return for given risk level (or minimize risk for target return). Core principle: diversification reduces risk; optimal portfolio balances return maximization against volatility minimization using mean-variance optimization.  
**Purpose:** Solve portfolio construction problem: How much to allocate to each asset? Provides mathematical foundation for all modern finance (CAPM, factor models, asset allocation). Revolutionized investing from stock-picking to systematic portfolio construction.  
**Prerequisites:** Expected value and variance (statistics), basic optimization (calculus/linear algebra), correlation concept

---

## 2. Comparative Framing

| Aspect | MPT (Markowitz) | Traditional Stock-Picking | Single-Asset Investing | Risk Parity | Black-Litterman |
|--------|-----------------|--------------------------|------------------------|------------|-----------------|
| **Decision Variable** | Portfolio weights (w1, w2, ..., wn) | Individual stock selection | All in one security | Equal risk contribution | Market weights + views |
| **Objective** | Maximize return for risk level | Beat market via picks | Maximize absolute return | Equal diversification | Balanced equilibrium + opinions |
| **Risk Measure** | Variance σ² (or std dev σ) | Volatility of picks | Single asset volatility | Risk contribution per asset | Variance with constraints |
| **Diversification** | Explicit; minimize correlation drag | Ignored (concentrates risk) | None (single asset) | Automatic via equal weighting | Market-driven (implicit) |
| **Covariance Use** | Central; ρij reduces portfolio risk | Ignored or estimated poorly | N/A | Equal risk shares | Market-based + adjustments |
| **Return Forecast** | Historical averages E[Ri] | Analyst predictions | Single return expectation | Historical multi-asset | Market-implied + adjustments |
| **Optimization** | Quadratic program; analytical solution | Qualitative (experience-based) | Trivial (1 asset) | Mechanical (equal weighting) | Bayesian optimization |
| **Constraints** | Long-only, sector, concentration | Self-imposed limits | None possible | Sector/geographic | Confidence intervals |
| **Output** | Efficient frontier; optimal portfolio | Selected stocks | Single allocation | Equal-weighted basket | Refined market portfolio |
| **Complexity** | Medium (matrix algebra, solver) | Low (stock research) | Trivial | Very low (equal weights) | High (view elicitation) |

**Key Insight:** MPT introduced mathematics and diversification; stock-picking ignores correlation; risk parity avoids forecasting; Black-Litterman bridges objective data with subjective expertise.

---

## 3. Examples + Counterexamples

**Example 1: Two-Asset Portfolio (Simple Diversification Benefit)**

Investor has $100k; chooses between:
- Asset A: E[RA] = 10%, σA = 15%, allocation wA
- Asset B: E[RB] = 5%, σB = 10%, allocation wB
- Correlation: ρAB = 0.3

**All-in Asset A (no diversification):**
- Return: 10%
- Risk: 15%
- Sharpe ratio: (10% - 2.5%) / 15% = 0.5

**50/50 Portfolio (diversification):**
- Return: 0.5 × 10% + 0.5 × 5% = 7.5%
- Variance: (0.5)² × (15%)² + (0.5)² × (10%)² + 2 × 0.5 × 0.5 × 0.3 × 15% × 10%
  - = 0.025 × 0.0225 + 0.025 × 0.01 + 0.015 × 0.03
  - = 0.0005625 + 0.00025 + 0.00045 = 0.0012625
  - σ = √0.0012625 = 3.55% (NOT 12.5% average!)
- Sharpe ratio: (7.5% - 2.5%) / 3.55% = 1.41

**Key observation:** Risk drops from 15% to 3.55% (76% reduction!) while return only drops 25%. This is the **"free lunch" of diversification** (Markowitz key insight).

**Optimal Portfolio (via quadratic programming):**
- Maximize: E[Rp] - (λ/2) σp² for some λ
- Result: ~70% Asset A, 30% Asset B (different from 50/50!)
- Return: 8.5%, Risk: ~4.2%, Sharpe: 1.43 (even better)

**Implication:** Equal weighting doesn't minimize risk; correlation matters; optimal weights are precise (need covariance matrix).

---

**Example 2: Three Risky Assets + Risk-Free**

Market:
- Treasury bills: rf = 2.5% (risk-free, σ = 0)
- Stock portfolio: E[Rm] = 10%, σm = 15%, β = 1
- Bond fund: E[RB] = 4%, σB = 5%, β = -0.5 (inverse stocks)

Naive allocation (1/3 each):
- Return: (1/3) × 2.5% + (1/3) × 10% + (1/3) × 4% = 5.5%
- Risk: Includes correlation; compute covariance matrix
- Likely suboptimal (why 1/3? no reason)

MPT Optimal (solve for λ = 2):
- Result: 20% bills + 60% stocks + 20% bonds
- Return: 0.2 × 2.5% + 0.6 × 10% + 0.2 × 4% = 7.3%
- Risk: Lower than equal-weight (benefits from bonds offsetting stock volatility)
- Insight: Bonds reduce total portfolio risk (negative correlation to stocks)

**Implication:** MPT automatically finds diversification benefits; naive allocations miss risk reduction.

---

**Example 3: Large Portfolio (100+ Stocks)**

Problem: Estimate covariance matrix for 500 stocks → 500 × 500 matrix = 125,000 parameters (mostly unknown!)

MPT challenge: Each parameter estimated with error; errors compound.

Solution 1: Single-index model (β-based):
- Covariance: σij = βi βj σm² + σε,i σε,j (if i ≠ j independent)
- Reduces parameters: 500 parameters (β, σε) vs 125,000
- Trade-off: Less accurate but more stable

Solution 2: PCA/Factor model:
- Extract 10 principal components
- Covariance: Σ = F Λ F' (low-rank approximation)
- Reduces parameters further; works well empirically

**Implication:** MPT becomes practical only with dimension reduction (factors, indices).

---

**Example 4: COUNTEREXAMPLE — Non-Normal Returns**

MPT assumes returns are normal. Reality: Stock returns have fat tails.

October 1987 Black Monday:
- S&P 500: -22% in one day (6+ sigma event under normality!)
- MPT prediction: Probability ~ 10^-10 (essentially zero)
- Reality: Happened (and will again)

Implication of fat tails:
- Variance insufficient to capture risk (only captures typical variation)
- Optimal portfolio via variance minimization may have tail risk
- Alternative: Use CVaR or higher moments
- **Fix:** Extended MPT (consider skewness, kurtosis)

---

**Example 5: COUNTEREXAMPLE — Costs & Constraints**

Theory: Optimal portfolio composition static (hold, rebalance infrequently).

Reality: Transaction costs change optimal policy.

Scenario: Quarterly rebalancing of 100-stock portfolio.
- Average bid-ask spread: 0.05%
- Commissions: 0.01%
- Estimated market impact: 0.02%
- Total cost per trade: ~0.08%
- Trades per quarter (rebalance): 50 stocks traded
- Cost: 50 × 0.08% = 4% per quarter = 16% annual drag!

**Implication:** Optimal portfolio must balance expected return against rebalancing costs; narrow trading bands (rebalance only if >5% drift) often better than mechanical quarterly rebalancing.

---

## 4. Layer Breakdown

```
Modern Portfolio Theory & Mean-Variance Framework Architecture:

├─ Historical Context & Motivation:
│   ├─ Pre-MPT (1920s-1950s):
│   │   ├─ Investing: Stock-picking dominated; diversification ignored
│   │   ├─ Problem: How to evaluate portfolio? (Diversification benefits unclear)
│   │   ├─ Return emphasis: Focus on expected return; ignore risk
│   │   ├─ Risk measurement: Informal (volatility understood, not modeled)
│   │   └─ Result: High-risk, concentrated portfolios; crashes severe (1929)
│   │
│   ├─ Markowitz's Insight (1952):
│   │   ├─ Key contribution: Formalize diversification mathematically
│   │   ├─ Risk measurement: Variance σ² captures portfolio risk
│   │   ├─ Key formula: σp² = Σ wi² σi² + Σ Σ wi wj σi σj ρij
│   │   │   └─ Covariance term ρij shows diversification benefit
│   │   ├─ Optimization: Find weights maximizing return for risk level
│   │   ├─ Revolutionary: Risk and return both matter; trade-off is optimal choice
│   │   └─ Result: Nobel Prize 1990 (Markowitz, Sharpe, Miller)
│   │
│   ├─ Evolution (1960s-2000s):
│   │   ├─ CAPM (Sharpe, 1964): Equilibrium pricing; what returns are "fair"?
│   │   ├─ Factor models (Fama-French, 1992): Multiple risks, multiple premiums
│   │   ├─ Behavioral critique (Kahneman-Tversky, 1979): Humans not mean-variance
│   │   ├─ Implementation (Black-Litterman, 1992): Combine MPT with views
│   │   └─ Extensions: Dynamic models, CVaR, robust optimization
│   │
│   └─ Current state (2020s):
│       ├─ MPT central to finance theory and practice
│       ├─ Limitations recognized (tails, costs, behavior)
│       ├─ Extensions adopted (CVaR, robust optimization, multi-period)
│       └─ Index investing validates core MPT principle (diversification)
│
├─ The Mean-Variance Framework (Core Mathematics):
│   ├─ Expected Return (Portfolio Level):
│   │   │ Definition: E[Rp] = Σ wi E[Ri]
│   │   │ (weighted average of individual returns)
│   │   │
│   │   ├─ Properties:
│   │   │   ├─ Linear in weights: Doubling all wi doubles return
│   │   │   ├─ Bounded by assets: Min = min(E[Ri]), Max = max(E[Ri])
│   │   │   ├─ Diversification doesn't change expected return
│   │   │   │   └─ (Linear combination; no "magic")
│   │   │   └─ Only allocation weights matter (individual μ and correlations don't)
│   │   │
│   │   ├─ Examples:
│   │   │   ├─ Two assets: E[Rp] = w1 × 8% + w2 × 5%
│   │   │   │   └─ 60/40 mix: E[Rp] = 0.6 × 8% + 0.4 × 5% = 6.8%
│   │   │   └─ Three assets: E[Rp] = w1 μ1 + w2 μ2 + w3 μ3
│   │   │
│   │   └─ Key insight: Return is predictable (linear); risk is the complex part
│   │
│   ├─ Portfolio Variance (Risk):
│   │   │ Definition: σp² = Σ wi² σi² + Σ Σ (i≠j) wi wj σi σj ρij
│   │   │
│   │   ├─ Decomposition:
│   │   │   ├─ First term: Σ wi² σi² (variance of individual assets, weighted by wi²)
│   │   │   ├─ Second term: Σ Σ (i≠j) wi wj σi σj ρij (covariance effects, weighted)
│   │   │   └─ Together: Total portfolio variance
│   │   │
│   │   ├─ Two-asset special case:
│   │   │   σp² = w1² σ1² + w2² σ2² + 2 w1 w2 σ1 σ2 ρ12
│   │   │   └─ (a + b)² form; familiar structure
│   │   │
│   │   ├─ Key observation (Diversification Magic):
│   │   │   ├─ Equal-weight portfolio (w1 = w2 = 0.5):
│   │   │   │   σp² = 0.25 σ1² + 0.25 σ2² + 2 × 0.25 × σ1 σ2 ρ12
│   │   │   │        = 0.25 (σ1² + σ2²) + 0.5 σ1 σ2 ρ12
│   │   │   │
│   │   │   ├─ If ρ12 = 1 (perfect correlation):
│   │   │   │   σp = 0.5 (σ1 + σ2) = average of individual vols (no benefit)
│   │   │   │
│   │   │   ├─ If ρ12 = 0 (no correlation):
│   │   │   │   σp = 0.5 √(σ1² + σ2²) < 0.5 (σ1 + σ2) (BENEFIT!)
│   │   │   │
│   │   │   └─ If ρ12 = -1 (perfect negative):
│   │   │       σp = 0.5 |σ1 - σ2| (maximum benefit; potential cancellation)
│   │   │
│   │   ├─ Correlation Impact on Risk:
│   │   │   ├─ Portfolio risk depends on correlations (not just individual risks)
│   │   │   ├─ Lower correlation → More diversification benefit
│   │   │   ├─ Negative correlation → Best case (hedging)
│   │   │   └─ High correlation → Minimal benefit; risk nearly additive
│   │   │
│   │   └─ General property: Portfolio variance < (average individual variance)
│   │       └─ Covariance terms reduce total; this is core of MPT
│   │
│   ├─ Covariance Matrix Σ:
│   │   │ Definition: n × n symmetric matrix; element (i,j) = σi σj ρij
│   │   │
│   │   ├─ Structure (for 3 assets):
│   │   │   ⎛ σ1²        σ12      σ13    ⎞
│   │   │   Σ = ⎜ σ12       σ2²      σ23   ⎟
│   │   │   ⎝ σ13        σ23      σ3²    ⎠
│   │   │
│   │   │   Where σij = σi σj ρij (or computed from return data)
│   │   │
│   │   ├─ Properties:
│   │   │   ├─ Symmetric: Σ = Σ'
│   │   │   ├─ Positive semi-definite: w' Σ w ≥ 0 for all w
│   │   │   ├─ Diagonal entries: Variances σi²
│   │   │   ├─ Off-diagonal: Covariances σij
│   │   │   └─ Diagonal = 1 normalized form (correlation matrix)
│   │   │
│   │   └─ Estimation:
│   │       ├─ Historical: Σ = Cov(R1, R2, ..., Rn) from past returns
│   │       ├─ Forward-looking: Forecast covariances (harder; assumptions-heavy)
│   │       ├─ Shrinkage: Ledoit-Wolf (compromise between estimated & benchmark)
│   │       └─ Factor model: Σ = F Λ F' (low-rank approximation)
│   │
│   ├─ Portfolio Variance Equation (Matrix Form):
│   │   │ σp² = w' Σ w
│   │   │
│   │   │ Where: w = column vector of weights
│   │   │        Σ = covariance matrix
│   │   │
│   │   ├─ Expands to: σp² = Σ Σ wi wj Σij
│   │   ├─ Intuition: Each pair (i,j) weighted by wi wj; diagonal = var, off-diag = cov
│   │   └─ Computational: Use matrix multiplication (efficient for large n)
│   │
│   └─ Correlation vs Covariance (Key Distinction):
│       ├─ Covariance: σij = σi σj ρij (includes scale of individual risks)
│       ├─ Correlation: ρij ∈ [-1, 1] (standardized; scale-independent)
│       ├─ Portfolio variance uses covariance (not correlation alone)
│       ├─ Example: ρAB = 0 (uncorrelated)
│       │   ├─ If σA = 20%, σB = 5%: σAB = 20% × 5% × 0 = 0
│       │   └─ If σA = 5%, σB = 20%: σAB = 5% × 20% × 0 = 0 (same, by symmetry)
│       └─ Portfolio risk depends on both: correlation AND individual volatilities
│
├─ The Optimization Problem (Central to MPT):
│   ├─ Standard Formulation:
│   │   │ max E[Rp] - (λ/2) σp²
│   │   │ s.t. Σ wi = 1, wi ≥ 0 (no short selling)
│   │   │
│   │   │ Where λ = risk aversion parameter
│   │   │       E[Rp] = Σ wi E[Ri] (return objective)
│   │   │       σp² = w' Σ w (risk penalty)
│   │   │
│   │   ├─ Interpretation:
│   │   │   ├─ Maximize return; penalize risk by λ/2
│   │   │   ├─ Higher λ → More risk-averse → Lower portfolio volatility
│   │   │   ├─ λ → 0 → Ignore risk; maximize return regardless
│   │   │   ├─ λ → ∞ → Ignore return; minimize risk
│   │   │   └─ Optimal λ depends on investor preferences
│   │   │
│   │   └─ Constraint: w weights must sum to 1 (fully invested)
│   │
│   ├─ Alternative Formulations:
│   │   ├─ (1) Minimize risk for target return:
│   │   │   min σp²
│   │   │   s.t. E[Rp] = R_target, Σ wi = 1
│   │   │   └─ Equivalent to changing λ
│   │   │
│   │   ├─ (2) Maximize return for target risk:
│   │   │   max E[Rp]
│   │   │   s.t. σp² ≤ σ_target, Σ wi = 1
│   │   │   └─ Different form; same concept
│   │   │
│   │   ├─ (3) Maximize Sharpe ratio (often preferred):
│   │   │   max (E[Rp] - rf) / σp
│   │   │   s.t. Σ wi = 1
│   │   │   └─ Return per unit risk; determines market portfolio
│   │   │
│   │   └─ (4) With constraints (practical):
│   │       ├─ wi ≥ 0 (no short selling)
│   │       ├─ wi ≤ 0.3 (concentration limit)
│   │       ├─ Σ w_sector ≤ 0.5 (sector limit)
│   │       └─ All feasible with quadratic programming
│   │
│   ├─ Solution Methods:
│   │   ├─ Analytical (2-3 assets):
│   │   │   ├─ Write Lagrangian
│   │   │   ├─ Take derivatives; set to zero
│   │   │   ├─ Solve linear system: ∇ = λ ∇constraint
│   │   │   └─ Result: Closed-form weights
│   │   │
│   │   ├─ Numerical (many assets):
│   │   │   ├─ Quadratic programming (QP) solver
│   │   │   ├─ Scipy.optimize, cvxpy, MATLAB
│   │   │   ├─ Iterative: gradient descent, interior point methods
│   │   │   └─ Convergence: Guaranteed for convex (variance minimization)
│   │   │
│   │   └─ Computational complexity:
│   │       ├─ Covariance matrix: O(n²) to store, O(n³) to invert
│   │       ├─ Large portfolios (1000+ assets): Slow without dimension reduction
│   │       └─ Solution: Factor models (e.g., Fama-French) reduce dimensions
│   │
│   └─ Sensitivity to Inputs:
│       ├─ Small change in μ → Large change in weights (input sensitivity)
│       ├─ Example: Increase E[Ri] by 0.5% → wi might swing 10%+
│       ├─ Problem: Estimation error in μ, Σ propagates to weight errors
│       ├─ Solution: Shrinkage (Ledoit-Wolf), resampling, robust optimization
│       └─ Implication: MPT weights fragile to input estimates
│
├─ Efficient Frontier & Risk-Return Geometry:
│   ├─ Definition:
│   │   │ Set of portfolios offering maximum expected return for each risk level
│   │   │ (or minimum risk for each return level)
│   │   │
│   │   └─ Shape: Curve in mean-std space; parabolic when starting from risk-free
│   │
│   ├─ Construction:
│   │   ├─ For each target return R:
│   │   │   ├─ Solve: min σp² s.t. E[Rp] = R, Σ wi = 1
│   │   │   ├─ Record (σp*, R) as point on frontier
│   │   ├─ Vary R from minimum to maximum
│   │   ├─ Connect points; smooth curve = efficient frontier
│   │   └─ Interior portfolios: Not on frontier (same return but more risk)
│   │
│   ├─ Properties:
│   │   ├─ Downward-sloping in std-dev space: Higher return → Higher risk
│   │   ├─ Upper envelope: No portfolio better than frontier (Pareto optimal)
│   │   ├─ Curvature: Depends on correlation structure
│   │   │   ├─ Low correlations → Curved frontier (large diversification benefit)
│   │   │   └─ High correlations → Nearly linear (little benefit)
│   │   ├─ Left endpoint: Global minimum variance portfolio (GMV)
│   │   │   └─ Lowest possible risk; may have negative return
│   │   └─ Right endpoint: Highest-return portfolio (often one asset)
│   │
│   ├─ Capital Allocation Line (CAL):
│   │   │ Linear line tangent to efficient frontier at one point
│   │   │ E[Rp] = rf + Sharpe × σp
│   │   │
│   │   ├─ Interpretation:
│   │   │   ├─ Y-intercept: rf (return on risk-free asset)
│   │   │   ├─ Slope: Sharpe ratio = (E[Rm] - rf) / σm (reward per risk)
│   │   │   ├─ Tangency point: Optimal risky portfolio (market portfolio)
│   │   │   ├─ Points above frontier: Impossible (violates optimization)
│   │   │   ├─ Points on CAL: Achievable (mix rf + optimal risky portfolio)
│   │   │   └─ Points below CAL: Suboptimal (available but worse)
│   │   │
│   │   └─ Why linear? Because combining risk-free (σ=0) with risky (σ>0) creates linear relationship
│   │
│   ├─ Two-Fund Separation Theorem:
│   │   │ All investors hold same risky portfolio (tangency point)
│   │   │ + risk-free asset in varying proportions
│   │   │
│   │   ├─ Implication:
│   │   │   ├─ No stock-picking needed; all hold market portfolio
│   │   │   ├─ Only decision: How much risk to take (allocation to risky vs rf)
│   │   │   ├─ Differs by risk aversion λ; risky portion same for all
│   │   │   └─ Validates index investing strategy
│   │   │
│   │   └─ Requires homogeneous expectations (all investors same beliefs)
│   │
│   └─ Indifference Curves & Optimal Portfolio:
│       │ Investor's preference curves (constant utility)
│       │ U = E[Rp] - (λ/2) σp² = constant
│       │
│       ├─ Shape: Upward-sloping (higher return offsets higher risk)
│       ├─ Tangent with CAL: Optimal portfolio
│       │   └─ Where investor's preferences meet market opportunity
│       ├─ Higher λ (risk-averse): Steeper curves; prefer less risk
│       │   └─ Indifference curves tighter; less return for same risk increase
│       ├─ Lower λ (risk-seeking): Flatter curves; accept more risk
│       │   └─ Indifference curves spread out; willing to take volatility
│       └─ Tangency condition: Slope of indifference curve = Slope of CAL
│           └─ MRS = (∂U/∂R) / (∂U/∂σ) = Sharpe ratio
│
├─ Practical Considerations & Constraints:
│   ├─ Transaction Costs:
│   │   ├─ Bid-ask spreads: 0.01% - 0.1% per trade
│   │   ├─ Commissions: Often eliminated (brokers zero-commission)
│   │   ├─ Market impact: Larger orders move price; cost ~0.1% - 1%
│   │   ├─ Impact on MPT: Rebalancing less frequent; wider bands
│   │   └─ Total annual cost: 0.5% - 2% if rebalancing quarterly
│   │
│   ├─ Taxes:
│   │   ├─ Capital gains tax: Higher for short-term (> ordinary income)
│   │   ├─ Tax-loss harvesting: Sell losers to offset gains
│   │   ├─ Tax deferral: 401k/IRA accounts ideal (no tax drag)
│   │   ├─ After-tax MPT: Adjust returns for tax rate; different weights
│   │   └─ Implication: Taxable vs tax-deferred accounts need different strategies
│   │
│   ├─ Leverage & Borrowing Constraints:
│   │   ├─ Margin limits: Typically 50% (can borrow up to amount invested)
│   │   ├─ Borrowing cost: > rf (credit spread 1-3%)
│   │   ├─ Margin calls: Forced liquidation if equity drops below threshold
│   │   └─ Implication: Can't achieve theoretical optimal if requires excessive leverage
│   │
│   ├─ Short-Selling Restrictions:
│   │   ├─ Some assets: Restricted (hard to borrow); costly to short
│   │   ├─ Some funds: Prohibited by charter (mutual fund, ETF limitations)
│   │   ├─ Constraint: wi ≥ 0 (long-only constraint)
│   │   └─ Effect: Shifts optimal portfolio; may increase risk
│   │
│   ├─ Liquidity Constraints:
│   │   ├─ Illiquid assets: Hard to trade; may not be includable
│   │   ├─ Real estate, private equity: Often excluded from MPT
│   │   └─ Practical: Focus on tradeable assets only
│   │
│   └─ Estimation Error & Robustness:
│       ├─ Input sensitivity: Small errors in μ, Σ → large weight errors
│       ├─ Concentration: Optimal portfolio often concentrated (few assets)
│       ├─ Fragility: Historical μ often poor predictor of future
│       └─ Solutions: Shrinkage, resampling, robust optimization, Black-Litterman
│
└─ MPT in Practice:
    ├─ Implementation:
    │   ├─ Collect data: Historical returns (250+ days for daily; 5+ years for monthly)
    │   ├─ Estimate μ, Σ: Use sample means and covariances
    │   ├─ Set constraints: Long-only (wi ≥ 0), sector limits, etc.
    │   ├─ Solve: Quadratic programming solver
    │   └─ Rebalance: Quarterly or when drift >5%
    │
    ├─ Benefits:
    │   ├─ Systematic: Objective, repeatable, auditable
    │   ├─ Diversification: Automatic; exploits correlations
    │   ├─ Scalable: Works for 10 assets or 1000+
    │   └─ Empirically supported: 85% of active managers underperform after fees
    │
    ├─ Limitations:
    │   ├─ Input sensitivity: Garbage in, garbage out (errors in μ, Σ)
    │   ├─ Non-normal tails: Variance inadequate for fat-tail risk
    │   ├─ Homogeneity: Assumes all investors same beliefs (violates reality)
    │   ├─ Static: Single-period; ignores multi-period rebalancing dynamics
    │   └─ Costs: Transaction costs, taxes erode theoretical gains
    │
    └─ Extensions & Alternatives:
        ├─ Black-Litterman: Incorporate subjective views; reduce input sensitivity
        ├─ Robust optimization: Handle uncertainty in parameters
        ├─ Multi-factor: Use factors (Fama-French) instead of individual assets
        ├─ CVaR optimization: Downside risk instead of variance
        ├─ Dynamic MPT: Multi-period optimization with rebalancing
        └─ Behavioral portfolio theory: Account for mental accounting, constraints
```

**Mathematical Formulas:**

Portfolio expected return:
$$E[R_p] = \sum_{i=1}^{n} w_i E[R_i]$$

Portfolio variance:
$$\sigma_p^2 = \sum_{i=1}^{n} w_i^2 \sigma_i^2 + \sum_{i \neq j} w_i w_j \sigma_i \sigma_j \rho_{ij} = w^T \Sigma w$$

Mean-variance optimization:
$$\max_{w} E[R_p] - \frac{\lambda}{2} \sigma_p^2 \quad \text{s.t.} \quad \sum w_i = 1, \, w_i \geq 0$$

---

## 5. Mini-Project: Build Efficient Frontier & Optimal Portfolio

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime

# Efficient frontier construction and visualization

def fetch_market_data(tickers, start_date, end_date):
    """Download return data for multiple assets."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    return returns


def compute_portfolio_stats(weights, returns_data):
    """Calculate return, volatility, Sharpe ratio for portfolio."""
    portfolio_return = np.sum(weights * returns_data.mean()) * 252
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns_data.cov() * 252, weights)))
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    return portfolio_return, portfolio_vol, sharpe


def negative_sharpe(weights, returns_data, rf=0.025):
    """Objective: Maximize Sharpe (minimize negative Sharpe) for optimization."""
    returns = returns_data.mean() * 252
    cov_matrix = returns_data.cov() * 252
    portfolio_return = np.sum(weights * returns)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    sharpe = (portfolio_return - rf) / portfolio_vol if portfolio_vol > 0 else 0
    return -sharpe


def minimum_variance_portfolio(returns_data):
    """Find minimum variance portfolio (GMV)."""
    n = returns_data.shape[1]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    def variance(w):
        return np.dot(w, np.dot(returns_data.cov() * 252, w))
    
    result = minimize(variance, np.array([1/n]*n), method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x


def efficient_frontier(returns_data, num_points=50):
    """Generate efficient frontier by varying target return."""
    min_ret = returns_data.mean().min() * 252
    max_ret = returns_data.mean().max() * 252
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    frontier = []
    
    for target in target_returns:
        n = returns_data.shape[1]
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(w * returns_data.mean() * 252) - target}
        ]
        bounds = tuple((0, 1) for _ in range(n))
        
        def variance(w):
            return np.dot(w, np.dot(returns_data.cov() * 252, w))
        
        result = minimize(variance, np.array([1/n]*n), method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            vol = np.sqrt(result.fun)
            frontier.append({'return': target, 'volatility': vol})
    
    return pd.DataFrame(frontier)


def optimal_portfolio_by_lambda(returns_data, lambda_val, rf=0.025):
    """Find optimal portfolio for given risk aversion λ."""
    n = returns_data.shape[1]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    def objective(w):
        ret = np.sum(w * returns_data.mean() * 252)
        var = np.dot(w, np.dot(returns_data.cov() * 252, w))
        return -(ret - (lambda_val / 2) * var)
    
    result = minimize(objective, np.array([1/n]*n), method='SLSQP',
                     bounds=bounds, constraints=constraints)
    return result.x


# Main Analysis
print("=" * 100)
print("MODERN PORTFOLIO THEORY (MARKOWITZ) & MEAN-VARIANCE FRAMEWORK")
print("=" * 100)

# 1. Data Collection
print("\n1. MARKET DATA & ASSET SELECTION")
print("-" * 100)

# Portfolio of stocks and sector ETFs
tickers = ['SPY', 'QQQ', 'IWM', 'AGG', 'GLD']  # Large-cap, Tech, Small-cap, Bonds, Gold
names = ['S&P 500', 'Tech (Nasdaq)', 'Small Cap', 'Aggregate Bonds', 'Gold']

returns = fetch_market_data(tickers, '2015-01-01', '2024-01-01')

print(f"\nAssets in portfolio ({len(tickers)} total):")
for i, (tick, name) in enumerate(zip(tickers, names)):
    annual_return = returns[tick].mean() * 252
    annual_vol = returns[tick].std() * np.sqrt(252)
    print(f"  {i+1}. {name:20s} ({tick}): {annual_return*100:6.2f}% return, {annual_vol*100:6.2f}% volatility")

# Correlation matrix
print(f"\nCorrelation Matrix:")
corr = returns.corr()
print(corr.round(3))

# 2. Basic Portfolio Comparisons
print("\n2. PORTFOLIO COMPOSITION COMPARISON")
print("-" * 100)

equal_weight = np.array([1/len(tickers)] * len(tickers))
gmv_weights = minimum_variance_portfolio(returns)

print(f"\nEqual-Weight Portfolio (1/{len(tickers)} each):")
eq_ret, eq_vol, eq_sharpe = compute_portfolio_stats(equal_weight, returns)
print(f"  Return: {eq_ret*100:.2f}%, Volatility: {eq_vol*100:.2f}%, Sharpe: {eq_sharpe:.3f}")

print(f"\nGlobal Minimum Variance (GMV) Portfolio:")
gmv_ret, gmv_vol, gmv_sharpe = compute_portfolio_stats(gmv_weights, returns)
print(f"  Return: {gmv_ret*100:.2f}%, Volatility: {gmv_vol*100:.2f}%, Sharpe: {gmv_sharpe:.3f}")
print(f"  Weights: {', '.join([f'{n}: {w*100:.1f}%' for n, w in zip(names, gmv_weights)])}")

# Diversification benefit
avg_vol = np.mean([returns[t].std() * np.sqrt(252) for t in tickers])
print(f"\nDiversification Benefit:")
print(f"  Average individual volatility: {avg_vol*100:.2f}%")
print(f"  GMV portfolio volatility: {gmv_vol*100:.2f}%")
print(f"  Reduction: {(1 - gmv_vol/avg_vol)*100:.1f}%")

# 3. Efficient Frontier
print("\n3. EFFICIENT FRONTIER")
print("-" * 100)

frontier = efficient_frontier(returns, num_points=40)
print(f"\nFrontier generated ({len(frontier)} portfolios)")
print(f"  Min volatility: {frontier['volatility'].min()*100:.2f}%")
print(f"  Max volatility: {frontier['volatility'].max()*100:.2f}%")
print(f"  Return range: {frontier['return'].min()*100:.2f}% - {frontier['return'].max()*100:.2f}%")

# 4. Optimal Portfolio for Different Risk Aversion Levels
print("\n4. OPTIMAL PORTFOLIOS BY RISK AVERSION")
print("-" * 100)

lambdas = [0.5, 1.0, 2.0, 4.0, 8.0]
optimal_portfolios = {}

print(f"\n{'Lambda':<8} {'Return %':<12} {'Volatility %':<15} {'Sharpe':<10} {'Diversification':<20}")
print("-" * 65)

for lam in lambdas:
    weights = optimal_portfolio_by_lambda(returns, lam)
    optimal_portfolios[lam] = weights
    ret, vol, sharpe = compute_portfolio_stats(weights, returns)
    
    # Diversification measure (Herfindahl index)
    hhi = np.sum(weights ** 2)
    
    print(f"{lam:<8.1f} {ret*100:<12.2f} {vol*100:<15.2f} {sharpe:<10.3f} {hhi:<20.3f}")

# 5. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Efficient Frontier with CAL
ax = axes[0, 0]

ax.plot(frontier['volatility'] * 100, frontier['return'] * 100, 
        label='Efficient Frontier', linewidth=2.5, color='#2ecc71')

# Add individual assets
for tick, name in zip(tickers, names):
    asset_ret = returns[tick].mean() * 252
    asset_vol = returns[tick].std() * np.sqrt(252)
    ax.scatter(asset_vol * 100, asset_ret * 100, s=150, alpha=0.7, label=name)

# Add GMV, Equal-weight, and optimal points
ax.scatter(gmv_vol * 100, gmv_ret * 100, s=200, marker='*', 
          color='red', label='GMV', zorder=5, edgecolors='black', linewidth=1)
ax.scatter(eq_vol * 100, eq_ret * 100, s=200, marker='s', 
          color='purple', label='Equal-Weight', zorder=5, edgecolors='black', linewidth=1)

# CAL from risk-free rate
rf = 0.025
max_sharpe_idx = (frontier['return'] - rf) / frontier['volatility']
max_sharpe_idx = max_sharpe_idx.idxmax()
max_sharpe_vol = frontier.loc[max_sharpe_idx, 'volatility']
max_sharpe_ret = frontier.loc[max_sharpe_idx, 'return']

cal_vols = np.linspace(0, frontier['volatility'].max() * 1.2, 100)
cal_returns = rf + (max_sharpe_ret - rf) / max_sharpe_vol * cal_vols

ax.plot(cal_vols * 100, cal_returns * 100, 'k--', linewidth=1.5, alpha=0.6, label='CAL')
ax.scatter(max_sharpe_vol * 100, max_sharpe_ret * 100, s=200, marker='^',
          color='orange', label='Tangency (Max Sharpe)', zorder=5, edgecolors='black', linewidth=1)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Efficient Frontier & Capital Allocation Line', fontweight='bold', fontsize=13)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)
ax.axhline(y=rf*100, color='gray', linestyle=':', alpha=0.5)

# Plot 2: Optimal Asset Weights by Lambda
ax = axes[0, 1]

lambdas_plot = sorted(optimal_portfolios.keys())
x = np.arange(len(lambdas_plot))
width = 0.15

for i, (name, tick) in enumerate(zip(names, tickers)):
    weights_by_lambda = [optimal_portfolios[lam][i] for lam in lambdas_plot]
    ax.bar(x + i * width, np.array(weights_by_lambda) * 100, width, label=name, alpha=0.8)

ax.set_xlabel('Risk Aversion (λ)', fontsize=12)
ax.set_ylabel('Portfolio Weight (%)', fontsize=12)
ax.set_title('Optimal Asset Allocation by Risk Aversion', fontweight='bold', fontsize=13)
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f'λ={l}' for l in lambdas_plot])
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Risk-Return Tradeoff by Lambda
ax = axes[1, 0]

lambdas_plot = sorted(optimal_portfolios.keys())
rets_by_lambda = []
vols_by_lambda = []
sharpes_by_lambda = []

for lam in lambdas_plot:
    weights = optimal_portfolios[lam]
    ret, vol, sharpe = compute_portfolio_stats(weights, returns)
    rets_by_lambda.append(ret)
    vols_by_lambda.append(vol)
    sharpes_by_lambda.append(sharpe)

ax.plot(vols_by_lambda, np.array(rets_by_lambda) * 100, 'o-', linewidth=2.5, 
       markersize=10, color='#3498db', label='Optimal Portfolios')

for lam, vol, ret in zip(lambdas_plot, vols_by_lambda, rets_by_lambda):
    ax.annotate(f'λ={lam}', (vol, ret*100), textcoords="offset points", 
               xytext=(0,10), ha='center', fontsize=9)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Risk-Return Tradeoff: Optimal Portfolios', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

# Plot 4: Sharpe Ratio by Lambda
ax = axes[1, 1]

ax.plot(lambdas_plot, sharpes_by_lambda, 'o-', linewidth=2.5, markersize=10,
       color='#e74c3c', label='Sharpe Ratio')
ax.axhline(y=max(sharpes_by_lambda), color='g', linestyle='--', alpha=0.5, label='Max Sharpe')

ax.set_xlabel('Risk Aversion (λ)', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Risk-Adjusted Returns by Risk Aversion', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mpt_efficient_frontier.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: mpt_efficient_frontier.png")
plt.show()

# 6. Summary Statistics
print("\n5. KEY INSIGHTS & INTERPRETATION")
print("-" * 100)
print(f"""
MEAN-VARIANCE FRAMEWORK (MARKOWITZ 1952):
├─ Central insight: Diversification reduces portfolio risk below individual asset risk
├─ Mechanism: Correlation < 1 creates covariance benefit (off-diagonal terms negative)
├─ Optimization: Find portfolio maximizing return for risk level (or vice versa)
├─ Key metric: Sharpe ratio = (return - rf) / volatility captures risk-adjusted performance
└─ Two-fund principle: All investors hold same risky portfolio + risk-free in varying proportions

EFFICIENT FRONTIER FINDINGS:
├─ Number of efficient portfolios: Infinite (continuous frontier)
├─ GMV portfolio: Minimum risk; may have low return
├─ Tangency portfolio: Maximum Sharpe ratio; optimal risky asset for all investors
├─ Capital Allocation Line (CAL): Linear combinations of rf + tangency portfolio
└─ Risk-return tradeoff visible: Higher return requires accepting higher volatility

RISK AVERSION IMPACT:
├─ Higher λ (more risk-averse): Portfolio closer to GMV (lower vol, lower return)
├─ Lower λ (risk-seeking): Portfolio closer to highest-return portfolio (higher vol)
├─ Optimal portfolio depends ONLY on λ (not on individual preferences, just risk tolerance)
├─ Same risky asset held by all; only allocation to rf/risky changes
└─ Validates two-fund separation and justifies index investing

DIVERSIFICATION BENEFIT (From GMV Analysis):
├─ Individual asset average volatility: {avg_vol*100:.2f}%
├─ GMV portfolio volatility: {gmv_vol*100:.2f}%
├─ Reduction from diversification: {(1 - gmv_vol/avg_vol)*100:.1f}%
├─ Correlation structure key: Negative correlations provide hedging (best benefit)
└─ Real-world: 20-30 diversified stocks capture most benefit; diminishing returns beyond

PRACTICAL IMPLICATIONS:
├─ Use low-cost index funds (captures diversification, low fees)
├─ Rebalance periodically (quarterly or annually) to maintain target allocation
├─ Choose allocation based on personal risk tolerance (λ)
├─ Tax-loss harvest in taxable accounts to improve after-tax returns
└─ Monitor correlations; they change in crises (diversification can fail)
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Correlation Assumption:** If correlation between stocks changes from 0.5 to 0.8 during market stress, how does GMV portfolio change? What does this imply for diversification?

2. **Leverage vs No Leverage:** Conservative investor (λ=8) wants 8% return. Can achieve via: (a) 100% stocks (gives 10% return) + hold cash (not invested), (b) borrow at 3% to buy more stocks, or (c) add bonds. Which is better? Why?

3. **Input Sensitivity Problem:** Small change in expected return (μ increases 0.5%) causes optimal weights to swing 20%+. How would you fix this (hint: shrinkage, resampling)?

4. **Non-Normal Returns:** S&P 500 fat tails mean variance underestimates risk (1987 crash was 6-sigma event under normality). Should replace variance with CVaR or other tail risk measure? Trade-offs?

5. **Multi-Period Rebalancing:** If rebalancing quarterly costs 0.4% in transaction costs, is it worth doing? Compare: (a) quarterly mechanical rebalancing, (b) threshold-based (only rebalance if drift >10%), (c) no rebalancing. Which maximizes returns after costs?

---

## 7. Key References

- **Markowitz, H.M. (1952).** "Portfolio Selection" *Journal of Finance* – Foundational paper; mean-variance optimization framework.

- **Sharpe, W.F. (1964).** "Capital Asset Prices: A Theory of Market Equilibrium" – CAPM derivation; Nobel Prize 1990.

- **Tobin, J. (1958).** "Liquidity Preference as Behavior Towards Risk" – Two-fund separation theorem; capital allocation line.

- **Bodie, Z., Kane, A., & Marcus, A.J. (2017).** "Investments" (11th ed.) – Comprehensive textbook; MPT practical implementation; constraints.

- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** "Optimal Versus Naive Diversification" – Empirical comparison; naive diversification often competitive.

- **Ledoit, O. & Wolf, M. (2004).** "Honey, I Shrunk the Sample Covariance Matrix" – Addressing estimation error; shrinkage estimator.

- **Michaud, R.O. (1998).** "Efficient Asset Management" – Resampled frontier; robustness to input errors.

