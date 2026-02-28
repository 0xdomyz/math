# Barrier Options

## Concept Skeleton
**Definition:** Options that activate (knock-in) or terminate (knock-out) if the underlying asset price touches/crosses a specified barrier level during the option's life  
**Purpose:** Reduce option cost by removing unfavorable scenarios; create conditioned exposure; structure risk in specific price regions  
**Prerequisites:** Vanilla option pricing, reflection principle, boundary conditions, binomial/MC methods, path-dependent valuation

## Comparative Framing
| Feature | Knock-Out | Knock-In | Vanilla | Rebate |
|---------|-----------|----------|---------|--------|
| **Activation** | Born active | Triggered if L crossed | Always active | Upon termination |
| **Cost vs Vanilla** | Cheaper | More expensive | Baseline | Reduces credit |
| **Best Use** | Sell (premium) | Buy (cheap leverage) | Neutral | Both (payoff adjust) |
| **Buyer Profit** | S far from L | S hits L | S far from K | Fixed | contingent |
| **Seller Profit** | S stays far from L | S never hits L | S near K/expiry | Fixed contingent |
| **Greeks Complex** | Yes (kink at L) | Yes (kink at L) | Smooth | Yes (discontinuous) |

## Examples + Counterexamples
**Down-and-Out Call Success:**  
S=$110, K=$100, barrier L=$90, r=5%, σ=20%, T=1yr. If S stays >$90, standard call payoff. If S touches $90, worthless instantly. Trader expects consolidation around current level → sells at discount vs vanilla.

**Up-and-In Put Win:**  
S=$95, K=$100, barrier U=$105, r=5%, T=1yr. Put worthless now ($5 OTM). But if S rallies to $105, put activates → sudden downside protection. Cheap bet on rally then reversal.

**Knock-Out Failure Case:**  
S=$99, K=$100 (call), barrier L=$98. Market tanks to $97.50 → option knocked out just before possible recovery. Barrier hit, no more payoff. Pure volatility trader loses.

**Rebate Payment:**  
Down-and-out with $2 rebate. If barrier hit, receive $2 instead of $0. Partially compensates for loss; reduces negative surprise, not quite like vanilla.

**Discrete Monitoring Issue:**  
Barrier monitored daily (not continuously). If S dips below L intraday but recovers by next observation, option survives. Continuous barrier ≠ discrete; discrete cheaper.

## Layer Breakdown
```
Barrier Options Framework:

├─ Barrier Types:
│  ├─ Down-and-Out:
│  │   ├─ Definition: Option worthless if S ≤ L
│  │   ├─ Payoff: max(S-K, 0) if S_min > L, else 0 (call)
│  │   ├─ Use: Seller wants premium, expects S > L
│  │   └─ Greeks: Delta & gamma have discontinuity at L
│  ├─ Up-and-Out:
│  │   ├─ Definition: Option worthless if S ≥ U
│  │   ├─ Use: Sell protection against large rallies
│  │   └─ Pricing: Higher payoff probability (U far from S0)
│  ├─ Down-and-In:
│  │   ├─ Definition: Option activates if S ≤ L (European at expiry)
│  │   ├─ Payoff: max(S-K, 0) if ever S ≤ L, else 0 (call)
│  │   ├─ Use: Leverage bet; cheap long exposure if rarely triggered
│  │   └─ Greeks: Large vega (gamma) if far from barrier
│  └─ Up-and-In:
│      ├─ Definition: Option activates if S ≥ U
│      ├─ Use: Participate if rally confirmed
│      └─ Example: Up-and-in put (rally then fall)
├─ Knock-Out Options Details:
│  ├─ Payoff at expiry:
│  │   V(S,T) = Payoff_vanilla(S) if S ∈ valid region
│  │         = Rebate (if any) if barrier crossed
│  ├─ Path condition (down-and-out):
│  │   S_min > L throughout [0,T] ⟹ retain payoff
│  │   S_min ≤ L ⟹ option ends, receive rebate R
│  ├─ Reflection principle (semi-analytical):
│  │   P(S ≤ L) by reflecting path at L
│  │   P(down-out) = P(vanilla) - P(reflected)
│  ├─ Rebate mechanics:
│  │   Rebate R: Fixed compensation if knocked out
│  │   R=0 (typical): Full loss; R>0: Partial protection
│  │   Paid at: Knockout date (discrete) or discounted if T
│  └─ Early termination:
│      ├─ Barrier touched at time τ < T
│      ├─ Option worthless (or worth rebate) instantly
│      ├─ No further path-dependency after barrier
│      └─ Greeks "explode" near barrier (numerical challenge)
├─ Knock-In Options Details:
│  ├─ Payoff at expiry (after activation):
│  │   If activated: V = Payoff_vanilla(S)
│  │   If never activated: V = 0 (or rebate)
│  ├─ Path condition (down-and-in):
│  │   S ≤ L at some τ ∈ [0,T] ⟹ option "born", exercised at T
│  │   Equivalence: Down-and-in call = Vanilla - Down-and-out
│  ├─ Pricing via Complementarity:
│  │   V_in = V_vanilla - V_out (same K, T, r, σ)
│  │   Elegant decomposition
│  ├─ Greeks pre-activation:
│  │   Delta ≈ 0 if S far from L (option worthless w.p. high)
│  │   Vega large: Vol raises P(S ≤ L)
│  │   Gamma can be high/negative near L
│  └─ Activation value:
│      ├─ If S = L exactly, option "born"
│      ├─ Payoff ladder: Closer to L → higher activation probability
│      └─ Knock-in value peaks when S close to L
├─ Valuation Methods:
│  ├─ Analytical (Partial):
│  │   ├─ Reflection principle: Down-out = Vanilla - Reflected
│  │   ├─ Closed form for perpetual barriers
│  │   ├─ Numerical integral for finite T
│  │   └─ Complexity: Requires error function, special functions
│  ├─ Lattice/Binomial:
│  │   ├─ Tree with barrier tracking
│  │   ├─ Nodes crossing L → payoff = 0 (or rebate)
│  │   ├─ Nodes below L → exclude (down-out)
│  │   ├─ Nodes above L (for up-out) → exclude
│  │   ├─ Convergence: Slower if barrier crosses many lattice lines
│  │   └─ Rebate: Attach to termination branch
│  ├─ Finite Difference:
│  │   ├─ Grid with absorbing BC at L
│  │   ├─ V(S=L,t) = Rebate R (or 0)
│  │   ├─ Solution domain: S ∈ (L, S_max)
│  │   ├─ Backward induction with constraint
│  │   └─ Greeks: Differentiate solution w.r.t S, σ, r
│  ├─ Monte Carlo:
│  │   ├─ Simulate path S(t)
│  │   ├─ Track if S crosses barrier L
│  │   ├─ If crossed: Payoff = Rebate
│  │   ├─ If not crossed: Payoff = Vanilla(S_T)
│  │   ├─ Average over paths
│  │   └─ Barrier crossing: Brownian bridge for accuracy
│  └─ Brownian Bridge Refinement (MC):
│      ├─ Check if path crosses L between timesteps
│      ├─ P(cross | S_n, S_{n+1}) = exp(-2LnL_{n+1}/(σ²Δt))
│      ├─ Exact for GBM crossing probability
│      └─ More accurate than naive timestep check
├─ Continuous vs Discrete Monitoring:
│  ├─ Continuous (classical theory):
│  │   ├─ S_min = min{S(t) : t ∈ [0,T]}
│  │   ├─ Barrier checked continuously
│  │   ├─ Mathematical: Exact, elegant solutions
│  │   ├─ Market: Standard assumption
│  │   └─ Price: Lower (harder to avoid barrier)
│  ├─ Discrete (practical):
│  │   ├─ S checked at specific dates (daily, weekly)
│  │   ├─ Intraday barrier crossings missed
│  │   ├─ Probability: Higher (easier to avoid barrier)
│  │   ├─ Price: Higher than continuous
│  │   └─ Adjustment: 10-20% premium for discrete
│  └─ Sticky Barriers:
│      ├─ Barrier level S* adjusts with time
│      ├─ Example: L(t) = L_0 × (1 + r(T-t))
│      ├─ Tracks interest accrual
│      └─ Used: Exotic structures, CMS-linked notes
├─ Greeks & Hedging:
│  ├─ Delta:
│  │   ├─ Smooth away from L
│  │   ├─ Discontinuity at L (kink)
│  │   ├─ Spike near L for knock-ins
│  │   ├─ Seller faces gamma risk at L
│  │   └─ Hedge: Rebalance portfolio dynamically
│  ├─ Gamma:
│  │   ├─ Positive away from L (convex)
│  │   ├─ Negative/explosive at L
│  │   ├─ Barrier event = realized big loss/gain
│  │   └─ Monitor closely: Gamma trap
│  ├─ Vega:
│  │   ├─ Higher vol → larger P(hit barrier)
│  │   ├─ Vega sign depends on position:
│  │   │   Long knock-out: Negative vega (vol hurts)
│  │   │   Long knock-in: Positive vega (vol helps)
│  │   ├─ Vega can be 2-3x vanilla at low spot
│  │   └─ Volga (vega-gamma): Important for risk
│  ├─ Theta:
│  │   ├─ Decay reduces option value normally
│  │   ├─ But theta reduces probability of barrier hit
│  │   ├─ Competing effects near L
│  │   └─ Near expiry: Theta → sign depends on barrier proximity
│  └─ Rho:
│      ├─ Interest rate impact via drift
│      ├─ High r: Upward drift reduces barrier hit likelihood
│      ├─ Low r: Downward drift increases hit likelihood
│      └─ Rho complex, rarely hedged directly
└─ Practical Considerations:
   ├─ Barrier Observation Window:
   │   ├─ Start date: When barrier monitoring begins
   │   ├─ End date: Usually option maturity T
   │   ├─ Delayed start: Barrier inactive first weeks (caps cost)
   │   └─ Early end: Barrier deactivated before expiry
   ├─ Rebate Specifications:
   │   ├─ No rebate (R=0): Total loss if knocked out
   │   ├─ Fixed rebate: R$ paid if barrier crossed
   │   ├─ Floating rebate: R = f(barrier_touch_level, time)
   │   └─ Payment timing: At trigger vs at maturity (discounting)
   ├─ Barrier Precision:
   │   ├─ Single barrier: L or U level
   │   ├─ Double barrier: Both L and U (knocked out if either)
   │   ├─ Moving barrier: L(t) changes over time
   │   └─ Stochastic barrier: Linked to another asset
   └─ Documentation Risk:
       ├─ "Continuous" vs "Discrete" observation gaps
       ├─ Rounding conventions (e.g., L=$95 or L=$94.999?)
       ├─ Barrier reached = touches or exceeds?
       └─ Critical for precise valuation, disputes
```

**Interaction:** Barrier presence → path-dependent probability → cheaper cost (knock-out) or higher leverage (knock-in) → gamma/vega risk → careful Greeks monitoring.

## Challenge Round
- Prove: Down-in + Down-out = Vanilla (decomposition)
- Derive reflection principle for down-out (cumulative normal terms)
- Why does discrete barrier increase option value? (Lower barrier hit probability)
- How does barrier gamma change sign at L?
- Design barrier for 20% cost reduction vs vanilla

## Key References
- [Reiner & Rubinstein, "Breaking Down the Barriers" (1991)](https://www.jstor.org/stable/2352569) — Closed-form barrier formulas
- [Merton, "Option Pricing When Underlying Stock Returns Jump" (1976)](https://doi.org/10.1086/260637) — Barrier jump risk
- [Haug, Handbook of Exotic Options](https://www.wiley.com/en-us/Complete+Guide+to+Option+Pricing+Formulas%2C+2nd+Edition-p-9780071389976)

---
**Status:** Essential exotic structure | **Complements:** Knock-In/Knock-Out, Rebates, Discrete Monitoring
