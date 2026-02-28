# Barrier Options

## Concept Skeleton
**Definition:** Path-dependent options that activate (knock-in) or extinguish (knock-out) when asset price crosses predetermined barrier  
**Purpose:** Cheaper than vanilla options; custom payoff structures; reduce premium cost for hedging  
**Prerequisites:** Path-dependent payoffs, monitoring frequency, Monte Carlo simulation, rebates

## Comparative Framing
| Feature | Knock-Out | Knock-In | European Vanilla | Double Barrier |
|---------|-----------|----------|------------------|----------------|
| **Activation** | Active until barrier hit | Inactive until barrier hit | Always active | Two barriers |
| **Payoff** | 0 if knocked out | Standard if knocked in | Standard always | Complex conditions |
| **Value** | Cheaper than vanilla | Cheaper than vanilla | Highest | Cheapest |
| **Hedging** | Gamma spikes near barrier | Discontinuous delta | Smooth Greeks | Very complex |
| **Monitoring** | Continuous/discrete | Continuous/discrete | Maturity only | Continuous/discrete |

## Examples + Counterexamples
**Simple Example:**  
Up-and-out call: S₀=$100, K=$100, Barrier=$120; if S never hits $120 → payoff=max(S_T - 100, 0); if hits → payoff=0

**Failure Case:**  
Continuous vs discrete monitoring: Discrete misses intraday breaches → overvalues knock-out (should be cheaper)

**Edge Case:**  
Barrier far from spot (B=$200, S₀=$100): Knock-out ≈ Vanilla (low breach probability); Knock-in ≈ 0

## Layer Breakdown
```
Barrier Option Classification & Pricing:
├─ Barrier Types:
│   ├─ Up-and-Out (UO):
│   │   ├─ Knock-Out Condition: S_t ≥ B for any t ∈ [0, T]
│   │   ├─ Payoff: max(S_T - K, 0) if S_t < B for all t; else 0
│   │   ├─ Use Case: Cap upside exposure; reduce premium
│   │   └─ Value: UO < Vanilla (barrier reduces optionality)
│   ├─ Up-and-In (UI):
│   │   ├─ Knock-In Condition: S_t ≥ B for some t ∈ [0, T]
│   │   ├─ Payoff: max(S_T - K, 0) if barrier touched; else 0
│   │   ├─ Parity: UO + UI = Vanilla (one must pay off)
│   │   └─ Value: UI < Vanilla (requires barrier breach)
│   ├─ Down-and-Out (DO):
│   │   ├─ Knock-Out Condition: S_t ≤ B for any t
│   │   ├─ Payoff: max(S_T - K, 0) if S_t > B for all t; else 0
│   │   ├─ Use Case: Avoid downside scenarios; cheaper puts
│   │   └─ Value: DO < Vanilla
│   ├─ Down-and-In (DI):
│   │   ├─ Knock-In Condition: S_t ≤ B for some t
│   │   ├─ Payoff: max(S_T - K, 0) if barrier touched; else 0
│   │   ├─ Parity: DO + DI = Vanilla
│   │   └─ Value: DI < Vanilla
│   └─ Double Barrier:
│       ├─ Two Barriers: B_lower < S₀ < B_upper
│       ├─ Knock-Out: If S hits either barrier → extinguished
│       └─ Payoff: Standard if stays within corridor
├─ Rebates (Optional):
│   ├─ Knock-Out Rebate: Cash payment R if barrier breached
│   ├─ Knock-In Rebate: Cash if barrier NOT breached
│   ├─ Payment Time: At breach or at maturity
│   └─ Enhances Value: UO with rebate > UO without
├─ Monte Carlo Pricing:
│   ├─ Path Generation: Fine time discretization (daily/hourly)
│   ├─ Barrier Monitoring:
│   │   ├─ Discrete: Check prices at S_{t1}, S_{t2}, ..., S_{tn}
│   │   ├─ Continuous Approximation: Brownian bridge between steps
│   │   └─ Bias: Discrete monitoring overvalues knock-out (misses breaches)
│   ├─ Payoff Logic:
│   │   ├─ Knock-Out: If max(S_path) ≥ B_up or min(S_path) ≤ B_down → Payoff = R (rebate)
│   │   ├─ Knock-In: If barrier touched → Payoff = max(S_T - K, 0); else 0
│   │   └─ Double: If S stays in [B_low, B_high] → Payoff standard
│   └─ Variance: High (many paths have zero payoff → discrete distribution)
├─ Analytical Solutions (Limited):
│   ├─ Continuous Monitoring: Closed-form under GBM (reflection principle)
│   ├─ Conditions: Single barrier, constant vol, no dividends
│   ├─ Formula: Involves reflected Brownian motion probabilities
│   └─ Complex: Double barriers require infinite series
├─ Greeks & Hedging Challenges:
│   ├─ Delta: Discontinuous near barrier (jump when breached)
│   ├─ Gamma: Extremely high near barrier (delta changes rapidly)
│   ├─ Vanna/Volga: Large cross-Greeks (∂²V/∂S∂σ)
│   ├─ Hedging: Difficult near barrier; frequent rebalancing required
│   └─ Vega: Higher for options near barrier (uncertainty in breach timing)
└─ Monitoring Frequency Effect:
    ├─ Continuous: True barrier (all times checked)
    ├─ Daily: Check once per day (misses intraday breaches)
    ├─ Pricing Bias: Discrete < Continuous for knock-out (breaches missed → survives more)
    └─ Adjustment: Broadie-Glasserman-Kou correction for discrete monitoring
```

**Interaction:** Generate paths with fine discretization → Check barrier breach at each step → Apply payoff logic → Discount to present

## Challenge Round
**Q1:** Prove knock-out + knock-in = vanilla (parity). What no-arbitrage argument supports this?  
**A1:** At maturity, either barrier touched (knock-in active, knock-out dead) or not (knock-out active, knock-in dead). Exactly one pays vanilla payoff. If UO + UI ≠ Euro, arbitrage: Buy cheap side, sell expensive side, lock risk-free profit.

**Q2:** Why does gamma explode near barrier? Implications for hedging?  
**A2:** Near barrier, small price move determines survival (in/out). Delta jumps from ≈ vanilla to 0 when barrier crossed → ∂Δ/∂S = Γ extremely large. Hedging impossible: Rebalance frequency → ∞ near barrier; transaction costs prohibitive.

**Q3:** Discrete vs continuous monitoring: Derive adjustment factor for knock-out call.  
**A3:** Discrete monitoring misses intraday breaches → survives more often → higher value. Broadie-Glasserman-Kou adjustment: Shift barrier B → B × exp(β σ√(dt)) where β ≈ 0.5826. Makes discrete price closer to continuous.

**Q4:** Double barrier option: What happens when barriers very tight (B_up - B_down → 0)?  
**A4:** Tight corridor → high probability of knock-out → value → 0 as corridor shrinks. Extreme: B_up = B_down = S₀ → instant knock-out → value = rebate only (if any). Used for range trading strategies.

**Q5:** Barrier breach probability: Derive for up-and-out barrier B > S₀ under GBM.  
**A5:** First passage time problem. Probability S_t hits B before T: P = N(d+) + (S₀/B)^(2μ/σ²) N(d-) where μ = r - σ²/2, d± = [ln(B²/S₀²) ± μT] / (σ√T). Reflection principle in Brownian motion theory.

**Q6:** Rebate timing: Payment at breach vs at maturity. Which is more valuable?  
**A6:** Rebate at breach > rebate at maturity (time value of money). Earlier payment → higher present value. Pricing: For breach at time τ, discount as e^(-rτ) R (random τ) vs e^(-rT) R (fixed T). Expected PV higher for immediate rebate.

**Q7:** Barrier option Greeks near barrier: Compute delta for S → B (from below) for up-and-out call.  
**A7:** As S → B⁻: UO call → 0 (about to knock out). Delta = ∂V/∂S → 0 rapidly. Just below barrier: Large negative delta (price drops to zero with small S increase). Gamma extremely negative (delta changes from positive to zero).

**Q8:** Reverse (inside) barrier: Knock-out if S stays INSIDE corridor [B_low, B_high]. When useful?  
**A8:** Opposite of double barrier: Extinguishes if stays in range (no excitement). Rare; used for volatility betting. High vol → likely to breach boundaries → survives → payoff. Low vol → trapped inside → knocks out → zero. Exotic structure for vol traders.

## Key References
**Primary Sources:**
- Rubinstein, M. & Reiner, E. "Breaking Down the Barriers" (1991) - Closed-form barrier formulas
- [Barrier Option Wikipedia](https://en.wikipedia.org/wiki/Barrier_option) - Classification and examples
- Hull, J.C. *Options, Futures, and Other Derivatives* (2021) - Chapter 27: Barrier Options

**Technical Details:**
- Glasserman, P. *Monte Carlo Methods* (2004) - Barrier monitoring (pp. 365-392)
- Broadie, M., Glasserman, P., Kou, S. "Connecting Discrete and Continuous Path-Dependent Options" (1999) - Monitoring bias correction

**Thinking Steps:**
1. Generate paths with fine time discretization (daily or finer)
2. Track maximum and minimum prices along each path
3. Check barrier breach: Compare max/min vs barrier level
4. Apply knock-out logic: Set payoff = rebate if breached
5. Apply knock-in logic: Set payoff = vanilla if breached, else rebate
6. Discount expected payoff; higher variance due to discrete payoff distribution
