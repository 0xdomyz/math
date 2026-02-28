# European Options Pricing

## Concept Skeleton
- **Definition**: European Options Pricing is a derivative pricing topic that translates market assumptions into a valuation and risk view for a specific contract class.
- **Purpose**: Used to support pricing, hedging, model governance, and scenario analysis under realistic market constraints.
- **Prerequisites**: Time value of money, stochastic processes, no-arbitrage intuition, and numerical methods used across derivative_pricing topics.

## Comparative Framing
| Method | Complexity | Interpretability | Speed | Accuracy | Use Case |
|---|---|---|---|---|---|
| Closed-form benchmark | Low | High | Very high | High under assumptions | Fast baseline pricing |
| Lattice / tree | Medium | Medium | Medium | Good for early exercise | American-style features |
| Monte Carlo simulation | Medium-High | Medium | Medium-Low | Flexible, convergence-based | Path-dependent payoffs |
| PDE / finite difference | High | Medium | Medium | High with stable grids | Boundary-value formulations |

## Examples + Counterexamples
- **Simple Example**: Price a 1-year call with spot = 100, strike = 100, rate = 3%, volatility = 20%; compare two numerical methods and report absolute error.
- **Realistic Failure Case**: A model calibrated in calm periods underprices during volatility regime shifts.
- **Edge Case**: Near-expiry options where Greeks become unstable and discretization error dominates.
- **Technical Counterexample**: Using historical drift in risk-neutral valuation instead of risk-free drift.

## Layer Breakdown
### Phase 1 - Problem Framing
- Specify contract terms, market data timestamp, and target outputs (price + risk sensitivities).
- Validate data quality and consistency before calibration.

`	ext
Problem Framing
 Contract specification
    Payoff type
    Exercise style
 Market inputs
    Spot / forward
    Rates / carry
    Implied volatility data
 Output targets
     Fair value
     Risk metrics (Greeks)
`

### Phase 2 - Model + Computation
- Select model family consistent with product features and computational budget.
- Solve with a method matched to payoff path dependence and exercise logic.

`	ext
Model + Computation
 Model selection
    Diffusion assumptions
    Jump / stochastic volatility extensions
 Numerical engine
    Closed-form / transform
    Tree / PDE
    Monte Carlo
 Core formulas
     Discounting: PV = E_Q[Payoff] e^{-rT}
     Convergence checks
`

### Phase 3 - Validation + Deployment
- Backtest against market quotes and benchmark models.
- Monitor model drift, stability, and hedge performance in production.

`	ext
Validation + Deployment
 Validation
    Benchmark comparison
    Sensitivity sanity checks
    Stress scenarios
 Governance
    Assumption documentation
    Model limitations
 Production
     Runtime controls
     Exception handling
`

**Key Dependencies**: Clean market data, calibration quality, stable numerical settings, and alignment between product features and model assumptions.

## Challenge Round
- Mis-specified boundary conditions create unstable valuations.
- Sparse or noisy implied volatility surfaces distort calibration.
- Discrete hedging error dominates in jumpy or illiquid markets.
- Overfitting to one date weakens out-of-sample robustness.

## Key References
1. John C. Hull, *Options, Futures, and Other Derivatives* (latest edition)  standard derivatives framework.
2. Paul Wilmott, *Paul Wilmott on Quantitative Finance*  practical modeling and implementation trade-offs.
3. Steven E. Shreve, *Stochastic Calculus for Finance II*  rigorous risk-neutral pricing foundations.
4. Darrell Duffie, *Dynamic Asset Pricing Theory*  equilibrium and arbitrage-based valuation perspective.
5. Emanuel Derman & Iraj Kani papers on implied trees/smile  volatility structure intuition.
