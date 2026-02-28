# Implied Volatility

## Concept Skeleton
**Definition:** Volatility parameter making Black-Scholes price equal observed market option price; inverts pricing formula to extract market's volatility expectation  
**Purpose:** Convert option prices to volatility units for comparison across strikes/maturities; reveal supply/demand imbalances; identify mispricing opportunities  
**Prerequisites:** Black-Scholes model, option Greeks, numerical root-finding (Newton-Raphson), volatility concepts, arbitrage bounds

## Comparative Framing
| Method | Newton-Raphson | Bisection | Analytical Approx | Direct Formula |
|--------|----------------|-----------|-------------------|----------------|
| **Speed** | Fast (quadratic) | Slow (linear) | Very fast | Instant |
| **Robustness** | Can fail (bad guess) | Always converges | Less accurate | Limited range |
| **Complexity** | Requires Vega | Simple bracketing | Complex math | Only approximation |
| **Accuracy** | Exact | Exact | ~0.01 vol error | ~0.1 vol error |

| Phenomenon | Volatility Smile | Term Structure | Sticky Strike | Sticky Delta |
|------------|------------------|----------------|---------------|--------------|
| **Pattern** | U-shaped IV vs K | IV vs maturity | IV fixed at K | IV fixed at Δ |
| **Cause** | Fat tails, jumps | Mean reversion | Supply/demand | Hedging flows |
| **Impact** | Skew risk | Calendar spreads | Post-move IV | Rehedging IV |
| **Model** | Local vol | Stochastic vol | Market convention | Trader behavior |

## Examples + Counterexamples
**Simple Example:**  
Market call price $5, BS with σ=20% gives $4.80. Increase σ to 21.5% → BS price=$5.00. Implied vol=21.5%.

**Perfect Fit:**  
ATM options with high liquidity: IV converges rapidly with Newton-Raphson (3-4 iterations). Vega large and stable, excellent numerical behavior.

**Volatility Smile:**  
Equity index: OTM puts (K=90) have IV=25%, ATM (K=100) has IV=20%, OTM calls (K=110) have IV=22%. Smile reflects crash risk (left tail fat).

**Volatility Term Structure:**  
Front-month IV=30% (earnings event), 3-month IV=22% (mean reversion), 1-year IV=20% (long-run average). Term structure flattens after event.

**Deep OTM Failure:**  
Far OTM option (Δ=0.01) with price=$0.02: Vega≈0, Newton-Raphson unstable. Bisection more reliable but slow. Analytical bounds needed.

**Poor Fit:**  
American options: IV solver uses European BS but American worth more due to early exercise → solved IV too high, doesn't represent true volatility expectation.

## Layer Breakdown
```
Implied Volatility Framework:

├─ Mathematical Foundation:
│  ├─ Inverse Problem: Given C_market, find σ such that:
│  │   BS(S, K, r, T, σ) = C_market
│  ├─ Non-closed form: No analytical solution for σ
│  ├─ Monotonicity: ∂C/∂σ > 0 (Vega always positive)
│  │   → Unique solution exists if C_market in valid range
│  ├─ Bounds: Check arbitrage bounds first:
│  │   ├─ Lower: C ≥ max(S - K e^(-rT), 0)
│  │   ├─ Upper: C ≤ S
│  │   └─ Invalid prices → No valid IV exists
│  └─ Domain: σ ∈ (0, ∞), practically σ ∈ [0.01, 5.0]
├─ Numerical Methods:
│  ├─ Newton-Raphson (Standard Approach):
│  │   ├─ Iteration: σ_{n+1} = σ_n - (BS(σ_n) - C_market) / Vega(σ_n)
│  │   ├─ Convergence: Quadratic (doubles digits each iteration)
│  │   ├─ Initial guess: Critical for success
│  │   │   ├─ Simple: σ_0 = 0.20 (20%)
│  │   │   ├─ Better: √(2π/T) × C / S (Brenner-Subrahmanyam)
│  │   │   └─ Adjacent strike IV (for interpolation)
│  │   ├─ Stopping criterion: |σ_{n+1} - σ_n| < ε (e.g., 1e-6)
│  │   ├─ Max iterations: Typically 10-20 sufficient
│  │   ├─ Advantages: Fast convergence, industry standard
│  │   └─ Disadvantages: Requires Vega, can diverge if bad guess
│  ├─ Bisection Method (Robust Fallback):
│  │   ├─ Bracket: [σ_low, σ_high] where BS(σ_low) < C < BS(σ_high)
│  │   ├─ Iteration: σ_mid = (σ_low + σ_high) / 2
│  │   │   If BS(σ_mid) < C: σ_low = σ_mid
│  │   │   If BS(σ_mid) > C: σ_high = σ_mid
│  │   ├─ Convergence: Linear (halves interval each step)
│  │   ├─ Advantages: Always converges, no derivatives needed
│  │   └─ Disadvantages: Slower than Newton-Raphson
│  ├─ Analytical Approximations:
│  │   ├─ Brenner-Subrahmanyam (ATM, short maturity):
│  │   │   σ ≈ √(2π/T) × (C/S)
│  │   ├─ Corrado-Miller (improved accuracy):
│  │   │   Includes higher-order terms for better fit
│  │   ├─ Use case: Fast initial guess or rough estimate
│  │   └─ Error: ~0.01-0.1 in vol units
│  └─ Hybrid Approach:
│      ├─ Start with analytical guess
│      ├─ Newton-Raphson for 3-5 iterations
│      └─ Fall back to bisection if diverges
├─ Volatility Surface:
│  ├─ Definition: IV(K, T) across all strikes and maturities
│  ├─ Dimensions:
│  │   ├─ Strike axis: Moneyness (K/S or K/F)
│  │   ├─ Maturity axis: Time to expiry T
│  │   └─ IV value: Height of surface
│  ├─ Smile/Skew:
│  │   ├─ Equity: Negative skew (put IV > call IV)
│  │   │   → Crash protection premium
│  │   ├─ FX: Symmetric smile (straddle more expensive)
│  │   │   → Currency can move either direction
│  │   ├─ Commodities: Varies by market structure
│  │   └─ Causes: Jump risk, leverage effect, supply/demand
│  ├─ Term Structure:
│  │   ├─ Upward sloping: Mean reversion expected
│  │   ├─ Downward sloping: Event risk (earnings, elections)
│  │   ├─ Humped: Near-term event, long-term reversion
│  │   └─ Drivers: Supply/demand, hedging flows, calendar effects
│  ├─ Interpolation/Extrapolation:
│  │   ├─ Strike interpolation: Cubic spline, SABR, SVI
│  │   ├─ Time interpolation: Variance interpolation (linear in σ²T)
│  │   ├─ Arbitrage-free constraints: Butterfly, calendar spreads
│  │   └─ Extrapolation: Flatten wings, avoid negative densities
│  └─ Surface Dynamics:
│      ├─ Sticky strike: IV stays at strike level (convention)
│      ├─ Sticky delta: IV moves with option's delta (hedger view)
│      ├─ Sticky moneyness: IV at K/S (hybrid)
│      └─ Reality: Combination depending on market conditions
├─ Applications:
│  ├─ Option Pricing:
│  │   ├─ Quote in vol terms: "25-delta put at 22 vol"
│  │   ├─ Trader language: More intuitive than dollar prices
│  │   └─ Standardization: Compare across strikes/underlyings
│  ├─ Arbitrage Detection:
│  │   ├─ Butterfly arbitrage: Check IV convexity
│  │   │   (IV_K1 + IV_K3) / 2 should ≥ IV_K2
│  │   ├─ Calendar arbitrage: Check variance increasing in time
│  │   │   σ₁²T₁ ≤ σ₂²T₂ for T₁ < T₂
│  │   └─ Put-call parity violations: IV_call ≠ IV_put at same K
│  ├─ Relative Value Trading:
│  │   ├─ Rich/cheap analysis: Compare IV to historical levels
│  │   ├─ Cross-strike dispersion: Buy low IV, sell high IV
│  │   ├─ Term structure trades: Calendar spreads
│  │   └─ Vol surface arbitrage: Complex multi-leg strategies
│  ├─ Risk Management:
│  │   ├─ Vega bucketing: By strike/maturity
│  │   ├─ Volatility Greeks: Vanna (∂Δ/∂σ), Volga (∂ν/∂σ)
│  │   ├─ Scenario analysis: Parallel shift, twist, skew change
│  │   └─ VaR/Expected Shortfall: Using IV for mark-to-market
│  └─ Model Calibration:
│      ├─ Extract parameters: Fit local vol, stochastic vol models
│      ├─ Objective: Minimize (Model_IV - Market_IV)²
│      ├─ Weights: By vega, liquidity, bid-ask spread
│      └─ Regularization: Smooth parameter evolution
├─ Volatility Indices (VIX):
│  ├─ VIX Calculation:
│  │   ├─ Model-free: Uses strip of OTM options
│  │   ├─ Formula: σ² = (2/T) Σ (ΔK/K²) e^(rT) Q(K)
│  │   │   where Q(K) = option mid-price
│  │   ├─ Weights: Inverse square of strike
│  │   └─ Result: 30-day expected volatility
│  ├─ Interpretation:
│  │   ├─ VIX = 20: Market expects ~20% annual vol
│  │   ├─ VIX = 40: Crisis levels (2008, 2020)
│  │   └─ Term structure: VIX vs VXV (3-month)
│  ├─ Trading:
│  │   ├─ VIX futures: Cash-settled on VIX level
│  │   ├─ VIX options: European, expire to VIX future
│  │   └─ ETFs: VXX, UVXY (roll futures, contango drag)
│  └─ "Vol of vol": Volatility of implied volatility itself
├─ Advanced Topics:
│  ├─ Implied Volatility of Implied Volatility:
│  │   Options on VIX → Second-order vol expectations
│  ├─ Correlation Surface:
│  │   Implied correlation from multi-asset options
│  ├─ Dividends:
│  │   Adjust for known dividends in IV calculation
│  │   Use dividend-adjusted forward price F = S e^((r-q)T)
│  ├─ American Options:
│  │   Approximate: Use binomial tree for American IV
│  │   Faster: Barone-Adesi-Whaley approximation
│  └─ Model Risk:
│      IV assumes BS framework → Errors if reality differs
└─ Practical Considerations:
   ├─ Market Data Quality:
   │   ├─ Stale quotes: Use bid-ask midpoint carefully
   │   ├─ Illiquid options: Wide spreads → noisy IV
   │   ├─ Pinning: IV collapses near expiry at popular strikes
   │   └─ Early exercise: Use American pricing for puts
   ├─ Numerical Stability:
   │   ├─ Near expiry: T→0 causes numerical issues
   │   ├─ Deep OTM: Vega→0, Newton-Raphson fails
   │   ├─ Extreme strikes: Check bounds before solving
   │   └─ Error handling: Return NaN or error code gracefully
   ├─ Performance:
   │   ├─ Vectorization: Solve entire surface in parallel
   │   ├─ Caching: Store IV, recalculate only on price update
   │   ├─ Approximations: Use for real-time quotes
   │   └─ GPU acceleration: For large-scale calibration
   └─ Conventions:
      ├─ Quote convention: Vol as %, e.g., "22 vol" = 22%
      ├─ Day count: Actual/365 or Actual/360
      ├─ Business days: Trading days vs calendar days
      └─ Settlement: T+1 or T+2 affects forward price
```

**Interaction:** Market price → IV solver (Newton-Raphson) → Volatility surface → Trading signals; IV surface feeds back into pricing exotic options and risk management.

## Challenge Round
1. **SVI Parameterization:** Implement Stochastic Volatility Inspired (SVI) model for smile fitting. Ensure no calendar arbitrage. How does it compare to cubic spline?

2. **Jump to Default:** Include credit spread in IV calculation for single-name equity options. How does default risk affect OTM put IVs?

3. **Dividend Impact:** Implement IV solver with discrete dividends (ex-dates within option life). How does dividend affect smile near ex-date?

4. **American IV:** Adapt solver for American options using binomial tree. Compare American vs European IV for ITM puts. When does difference exceed 1 vol point?

5. **VIX Replication:** Implement model-free variance calculation using strip of OTM options. Compare to ATM implied vol. Why do they differ?

## Key References
- [Black & Scholes (1973) - Original Pricing Formula](https://www.jstor.org/stable/1831029)
- [Brenner & Subrahmanyam (1988) - Analytical IV Approximation](https://www.sciencedirect.com/science/article/abs/pii/0378426694900721)
- [Gatheral, The Volatility Surface (Chapter 2-3)](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792529)
- [CBOE VIX White Paper - Model-Free Variance](https://www.cboe.com/micro/vix/vixwhite.pdf)

---
**Status:** Market standard for option quoting | **Complements:** Black-Scholes Model, Greeks, Volatility Surface, Option Trading Strategies
