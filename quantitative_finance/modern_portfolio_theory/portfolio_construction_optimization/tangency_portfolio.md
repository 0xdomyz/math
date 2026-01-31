# Tangency Portfolio

## 1. Concept Skeleton
**Definition:** Portfolio on efficient frontier with maximum Sharpe ratio when combined with risk-free asset; optimal risky portfolio for all investors  
**Purpose:** Identify single optimal risky asset allocation; foundation for Capital Market Line and two-fund separation theorem  
**Prerequisites:** Efficient frontier, Sharpe ratio, risk-free asset, capital allocation line

## 2. Comparative Framing
| Portfolio | Tangency | Minimum Variance | Market Portfolio | Equal Weight |
|-----------|----------|------------------|------------------|--------------|
| **Criterion** | Max Sharpe ratio | Min volatility | Aggregate holdings | Simplicity (1/n) |
| **Risk Level** | Variable | Lowest on frontier | β=1 by definition | Medium |
| **Optimality** | Unique optimal risky | Unique min risk | Equilibrium (CAPM) | Naive baseline |
| **Use** | All investors + rf | Risk-averse focus | Index investing | Benchmark |

## 3. Examples + Counterexamples

**Simple Example:**  
Two assets: A (12% return, 20% vol), B (8% return, 10% vol), ρ=0.2, rf=3%  
Tangency portfolio: ~60% A, ~40% B; Sharpe = 0.52 (better than either asset alone)

**Failure Case:**  
All assets have returns below risk-free rate: Tangency becomes negative weights (short risky, hold cash)

**Edge Case:**  
Single risky asset: Tangency portfolio is 100% that asset (no diversification needed)

## 4. Layer Breakdown
```
Tangency Portfolio Construction:
├─ Mathematical Derivation:
│   ├─ Objective: Maximize Sharpe = (Rp - rf) / σp
│   ├─ Equivalent: Maximize (w'μ - rf) / √(w'Σw)
│   ├─ First-order condition: Σ⁻¹(μ - rf·1) ∝ w*
│   └─ Normalized Solution: w* = Σ⁻¹(μ - rf·1) / [1'Σ⁻¹(μ - rf·1)]
├─ Geometric Interpretation:
│   ├─ Capital Allocation Line: rf + (Rtan - rf)/σtan × σp
│   ├─ Tangent Point: Line from rf touches efficient frontier
│   ├─ Unique: Only one tangency point (convex frontier)
│   └─ Dominates: All other risky portfolios when rf available
├─ Two-Fund Separation Theorem:
│   ├─ All investors hold same risky portfolio (tangency)
│   ├─ Risk tolerance: Adjust via rf allocation (lending/borrowing)
│   ├─ Conservative: More in rf, less in tangency
│   └─ Aggressive: Borrow at rf, leverage tangency
├─ Portfolio Statistics:
│   ├─ Expected Return: Rtan = w*'μ
│   ├─ Volatility: σtan = √(w*'Σw*)
│   ├─ Sharpe Ratio: SRtan = (Rtan - rf) / σtan
│   └─ Comparison: SRtan ≥ SR(any other risky portfolio)
├─ Practical Considerations:
│   ├─ Estimation Risk: Small changes in μ → large weight changes
│   ├─ Constraint Impact: Long-only → lower Sharpe than unconstrained
│   ├─ Transaction Costs: Optimal theory ignores rebalancing costs
│   └─ Leverage Limits: Borrowing at rf may be infeasible
└─ Relationship to CAPM:
    ├─ Equilibrium: Tangency = Market Portfolio