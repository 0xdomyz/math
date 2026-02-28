# Volatility Surface and Skew

## Concept Skeleton
**Definition:** Three-dimensional structure of implied volatility varying across strike and maturity; skew is asymmetric smile pattern reflecting non-lognormal returns and crash risk  
**Purpose:** Model market's volatility expectations across all option strikes/maturities; price exotic options consistently; capture tail risk and jump dynamics  
**Prerequisites:** Implied volatility calculation, option pricing, probability distributions, arbitrage-free constraints, local/stochastic volatility models

## Comparative Framing
| Feature | Volatility Smile | Volatility Skew | Flat Vol Surface | Term Structure |
|---------|------------------|-----------------|------------------|----------------|
| **Shape** | U-shaped (symmetric) | Downward sloping | Constant across K | IV varies by T |
| **Market** | FX, commodities | Equity indices | Theory (BS) | All markets |
| **Cause** | Fat tails both sides | Leverage, crashes | Perfect model | Mean reversion |
| **Risk** | Straddle expensive | OTM puts pricey | No skew risk | Calendar spreads |

| Model | Local Volatility | Stochastic Volatility | Implied Vol | Jump-Diffusion |
|-------|------------------|----------------------|-------------|----------------|
| **Calibration** | Fit surface exactly | Approximate fit | Direct from market | Add jump terms |
| **Dynamics** | Deterministic σ(S,t) | Random volatility | Static snapshot | Discrete jumps |
| **Smile Dynamics** | Sticky strike | Sticky delta | No model | Mixed behavior |
| **Complexity** | Moderate | High | None (data) | Moderate |

## Examples + Counterexamples
**Simple Example:**  
SPX options: 95% strike IV=22%, 100% ATM IV=18%, 105% strike IV=17%. Negative skew reflects left-tail crash fear premium.

**Perfect Fit:**  
Black Monday aftermath: OTM put IVs spike to 40% while ATM=25%. Market prices insurance against crashes → persistent downward skew pattern.

**Volatility Smile (FX):**  
EUR/USD: 90% strike IV=12%, 100% ATM IV=10%, 110% strike IV=12%. Symmetric smile reflects currency can spike either direction (risk-reversal patterns).

**Term Structure Interaction:**  
Front-month ATM IV=30% (earnings), 3-month IV=20%, 1-year IV=18%. Combine with strike skew → full 3D surface shows event risk fading over time.

**Arbitrage Violation:**  
Calendar spread: σ₁²T₁ > σ₂²T₂ with T₁ > T₂ → negative variance increment → arbitrage. Surface must respect increasing total variance.

**Poor Fit:**  
Using flat volatility for barrier options: Market skew means OTM barriers hit more frequently than BS predicts → significant mispricing (~10-30%).

## Layer Breakdown
```
Volatility Surface Framework:

├─ Market Observation:
│  ├─ Raw Data: Option prices across strikes and maturities
│  ├─ Implied Vol Calculation: Invert BS for each (K, T) pair
│  ├─ Moneyness Measures:
│  │   ├─ Absolute: K (strike level)
│  │   ├─ Relative: K/S or K/F (moneyness ratio)
│  │   ├─ Log-moneyness: ln(K/S) or ln(K/F)
│  │   └─ Delta: Option's hedge ratio (standardized)
│  ├─ Data Quality Issues:
│  │   ├─ Illiquid strikes: Wide bid-ask, stale quotes
│  │   ├─ Missing points: Sparse data away from ATM
│  │   ├─ Outliers: Fat-finger trades, illiquidity
│  │   └─ Asynchronous quotes: Time stamps differ
│  └─ Preprocessing:
│      ├─ Filter by bid-ask spread < threshold
│      ├─ Remove arbitrage violations
│      ├─ Interpolate missing points
│      └─ Smooth outliers
├─ Volatility Patterns:
│  ├─ Volatility Smile (Symmetric):
│  │   ├─ Shape: U-shaped, minimum at ATM
│  │   ├─ Markets: FX, commodities, rates
│  │   ├─ Interpretation: Fat tails (kurtosis > 3)
│  │   │   Both large up and down moves more likely than BS
│  │   ├─ Cause: Jump risk, stochastic volatility
│  │   └─ Trading: Straddles expensive, butterflies cheap
│  ├─ Volatility Skew (Asymmetric):
│  │   ├─ Negative skew (equity):
│  │   │   ├─ OTM put IV > ATM > OTM call IV
│  │   │   ├─ Downward sloping to the right
│  │   │   ├─ Crash protection premium
│  │   │   └─ Leverage effect: Falling S → higher σ
│  │   ├─ Positive skew (rare):
│  │   │   ├─ OTM call IV > ATM
│  │   │   ├─ Upside tail risk
│  │   │   └─ Example: Takeover targets
│  │   └─ Quantification:
│  │       ├─ Skew = IV(90% strike) - IV(110% strike)
│  │       ├─ Risk reversal: IV(25Δ put) - IV(25Δ call)
│  │       └─ Slope: ∂IV/∂K (per strike unit)
│  ├─ Volatility Term Structure:
│  │   ├─ Upward sloping: σ(T₁) < σ(T₂) for T₁ < T₂
│  │   │   Mean reversion: Low vol expected to rise
│  │   ├─ Downward sloping: σ(T₁) > σ(T₂)
│  │   │   Event risk: Near-term uncertainty, long-term calm
│  │   ├─ Humped: Peak at intermediate maturity
│  │   │   Specific event (earnings, election) in near future
│  │   └─ VIX term structure:
│  │       VIX, VIX3M, VIX6M quotes show market's vol expectations
│  └─ Full Surface (3D):
│      IV = IV(K, T) varies across both dimensions
│      Combines smile/skew (strike) with term structure (time)
├─ Parametric Models (Smile Interpolation):
│  ├─ SVI (Stochastic Volatility Inspired):
│  │   ├─ Formula: σ²(k) = a + b[ρ(k-m) + √((k-m)² + ξ²)]
│  │   │   where k = ln(K/F), 5 parameters (a,b,ρ,m,ξ)
│  │   ├─ Advantages: Flexible, no arbitrage with constraints
│  │   ├─ Calibration: Minimize (Model_IV - Market_IV)²
│  │   ├─ Constraints: Ensure no butterfly arbitrage
│  │   │   ∂²σ²/∂k² ≥ -2 (density stays positive)
│  │   └─ Extensions: SSVI (surface SVI) for term structure
│  ├─ SABR (Stochastic Alpha Beta Rho):
│  │   ├─ Model: dF = α F^β dW₁, dα = ν α dW₂, Cov(dW₁,dW₂) = ρ dt
│  │   ├─ Approximation: Analytical formula for IV(K)
│  │   │   σ_SABR(K) = function of (α, β, ρ, ν, F, K)
│  │   ├─ Parameters:
│  │   │   ├─ α: ATM volatility level
│  │   │   ├─ β: Backbone (0=normal, 1=lognormal)
│  │   │   ├─ ρ: Correlation (skew direction)
│  │   │   └─ ν: Vol-of-vol (smile curvature)
│  │   ├─ Market standard: FX, rates (swaptions)
│  │   └─ Limitations: Approximation breaks for extreme strikes
│  ├─ Polynomial Fits:
│  │   ├─ Quadratic: σ(k) = a + bk + ck²
│  │   ├─ Simple but inflexible
│  │   └─ Risk: Can violate arbitrage away from fit points
│  └─ Cubic Splines:
│      ├─ Piecewise polynomials with smooth joins
│      ├─ Advantages: Flexible, smooth
│      ├─ Disadvantages: No arbitrage guarantee
│      └─ Need additional constraints (monotone, convex)
├─ Arbitrage-Free Constraints:
│  ├─ Static Arbitrage:
│  │   ├─ Call prices: C(K₁) ≥ C(K₂) for K₁ < K₂ (monotone)
│  │   ├─ Convexity: ∂²C/∂K² ≥ 0
│  │   │   Equivalent to: Risk-neutral density ≥ 0
│  │   ├─ Butterfly spread: C(K-δ) - 2C(K) + C(K+δ) ≥ 0
│  │   └─ In vol terms: Complex constraint on ∂²σ²/∂k²
│  ├─ Calendar Arbitrage:
│  │   ├─ Total variance increasing: σ₁²T₁ ≤ σ₂²T₂ for T₁ < T₂
│  │   ├─ Forward variance positive:
│  │   │   σ²_fwd = (σ₂²T₂ - σ₁²T₁) / (T₂ - T₁) ≥ 0
│  │   └─ Equivalently: ∂(σ²T)/∂T ≥ 0
│  ├─ Call Spread Arbitrage:
│  │   (C(K₁) - C(K₂))/(K₂ - K₁) should be in [0, 1]
│  ├─ Put-Call Parity:
│  │   C - P = F e^(-rT) - K e^(-rT)
│  │   Ensures call IV = put IV at same strike
│  └─ Detection:
│      ├─ Numerical checks on fitted surface
│      ├─ Perturb surface, check arbitrage appears
│      └─ Use optimization constraints during calibration
├─ Surface Dynamics (How Surface Evolves):
│  ├─ Sticky Strike:
│  │   ├─ IV stays at strike level K
│  │   ├─ If spot moves, IV(K) unchanged
│  │   ├─ Used for: P&L attribution, scenario analysis
│  │   └─ Observed: Short-term moves, post-event
│  ├─ Sticky Delta:
│  │   ├─ IV stays at delta level
│  │   ├─ If spot moves, IV moves with option's new delta
│  │   ├─ Used for: Hedging, vega bucketing
│  │   └─ Observed: Medium-term, normal market conditions
│  ├─ Sticky Moneyness:
│  │   ├─ IV stays at K/S ratio
│  │   ├─ Hybrid between strike and delta
│  │   └─ Observed: Long-term, structural changes
│  ├─ Reality: Combination of all three
│  │   ├─ Short-term: More sticky strike
│  │   ├─ Medium-term: Mix of delta and strike
│  │   └─ Shocks: Can reset entire surface
│  └─ Vanna-Volga:
│      Cross-sensitivity: ∂Δ/∂σ captures surface dynamics
│      Important for hedging skew risk
├─ Local Volatility Model:
│  ├─ Dupire's Formula:
│  │   σ_local²(K,T) = [∂C/∂T + rK∂C/∂K] / [½K²∂²C/∂K²]
│  │   Extract local vol from option prices
│  ├─ Properties:
│  │   ├─ Fits any arbitrage-free surface exactly
│  │   ├─ Deterministic: σ = σ(S,t)
│  │   ├─ Forward smile: Generated by spot moves and local vol
│  │   └─ Implementation: Forward PDE or Monte Carlo
│  ├─ Limitations:
│  │   ├─ Sticky strike dynamics (unrealistic)
│  │   ├─ Forward smile too flat
│  │   ├─ Poor for exotics with vol exposure
│  │   └─ Calibration instability (numerical derivatives)
│  └─ Uses:
│      Barrier options, lookbacks, any path-dependent
│      Better than flat vol, worse than stochastic vol
├─ Stochastic Volatility Models:
│  ├─ Heston Model:
│  │   ├─ Dynamics:
│  │   │   dS = μS dt + √v S dW₁
│  │   │   dv = κ(θ - v)dt + ξ√v dW₂
│  │   │   Cov(dW₁, dW₂) = ρ dt
│  │   ├─ Parameters:
│  │   │   ├─ v₀: Initial variance
│  │   │   ├─ θ: Long-run variance (mean reversion level)
│  │   │   ├─ κ: Mean reversion speed
│  │   │   ├─ ξ: Vol-of-vol
│  │   │   └─ ρ: Spot-vol correlation (skew)
│  │   ├─ Calibration: Fit to option prices across strikes/maturities
│  │   ├─ Smile: Negative ρ creates skew, ξ creates curvature
│  │   └─ Forward smile: More realistic than local vol
│  ├─ SABR Model:
│  │   Already described above
│  │   Used directly for quoting (FX markets)
│  └─ Advantages:
│      ├─ Captures smile dynamics better
│      ├─ Vega risk more realistic
│      └─ Better for exotic options with vol exposure
├─ Market Conventions:
│  ├─ Quoting by Delta:
│  │   ├─ "25-delta put" refers to put with Δ=-0.25
│  │   ├─ Standardized across strikes/spots
│  │   ├─ Common: 10Δ, 25Δ, 50Δ (ATM)
│  │   └─ Risk-reversal: 25Δ call - 25Δ put (skew measure)
│  ├─ Butterfly (Smile Curvature):
│  │   Butterfly = (25Δ call + 25Δ put)/2 - 50Δ straddle
│  │   Measures smile width/convexity
│  ├─ ATM Definition:
│  │   ├─ ATM strike: K = S (spot)
│  │   ├─ ATM forward: K = F = S e^(rT) (forward)
│  │   ├─ ATM delta: K where Δ = 0.5 (delta-neutral)
│  │   └─ Market convention varies (FX vs equity)
│  └─ Variance Swap Strike:
│      Fair variance = ∫ IV(K)² × weight(K) dK
│      Model-free measure of expected variance
├─ Practical Applications:
│  ├─ Exotic Option Pricing:
│  │   ├─ Use calibrated surface, not flat vol
│  │   ├─ Local vol or stochastic vol model
│  │   └─ Critical for barriers, digitals, lookbacks
│  ├─ Risk Management:
│  │   ├─ Vega by strike: Greeks at each vol point
│  │   ├─ Skew risk: Exposure to skew steepening
│  │   ├─ Surface risk: Parallel shift vs twist vs skew change
│  │   └─ Scenario analysis: Shock different surface regions
│  ├─ Trading Strategies:
│  │   ├─ Skew trades: Buy low IV strikes, sell high IV
│  │   ├─ Calendar spreads: Term structure steepening/flattening
│  │   ├─ Butterfly spreads: Smile width expansion/contraction
│  │   └─ Dispersion: Trade realized vs implied correlation
│  ├─ Model Validation:
│  │   ├─ Mark-to-market: Reprice portfolio with new surface
│  │   ├─ P&L explain: Decompose into spot, vol, skew changes
│  │   └─ Backtesting: Historical surface accuracy
│  └─ Hedging:
│      Dynamic hedging considers surface moves, not just ATM vol
└─ Advanced Topics:
   ├─ Jump-Diffusion Models:
   │   Add discrete jumps to capture gap risk
   │   Merton, Kou models
   ├─ Rough Volatility:
   │   Fractional Brownian motion (H < 0.5)
   │   Better fits to high-frequency vol dynamics
   ├─ Smile Extrapolation:
   │   Far OTM/ITM wings behavior
   │   Power-law tails, exponential decay
   ├─ Multi-Asset Surfaces:
   │   Correlation surface: Implied correlations
   │   Basket options require vol surface + correlation
   └─ Machine Learning:
      Neural networks to interpolate/extrapolate surface
      Enforce arbitrage via constraints or loss function
```

**Interaction:** Market prices → IV extraction → Surface construction → Arbitrage checks → Model calibration → Exotic pricing; skew reflects non-BS dynamics and must be modeled consistently.

## Challenge Round
1. **SSVI Calibration:** Implement Surface SVI (SSVI) for full surface parameterization. Ensure calendar arbitrage-free across all maturities. How many parameters needed?

2. **Local Vol Extraction:** Use Dupire's formula to extract local volatility σ_local(K,T) from implied vol surface. Compare forward smile to market. Why does it flatten?

3. **Sticky Delta Simulation:** Simulate spot move +10%. Update surface using sticky delta rule. Recalculate portfolio Greeks. How much does vega P&L differ from sticky strike?

4. **Arbitrage Detection:** Create surface with deliberate butterfly violation. Write algorithm to detect and fix (minimal perturbation). Use quadratic programming?

5. **Variance Swap:** Price variance swap using strip of options across strikes. Compare to ATM vol. Why is variance swap strike higher than ATM²?

## Key References
- [Gatheral, The Volatility Surface (Chapters 3-5)](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792529)
- [Dupire (1994) - Pricing with a Smile](https://www.sciencedirect.com/science/article/abs/pii/0165188994900201)
- [Hagan et al (2002) - SABR Model](https://www.researchgate.net/publication/235622441_Managing_Smile_Risk)
- [Gatheral & Jacquier (2014) - Arbitrage-Free SVI](https://arxiv.org/abs/1204.0646)

---
**Status:** Core market microstructure | **Complements:** Implied Volatility, Local Volatility, Stochastic Volatility, Greeks, Exotic Options
