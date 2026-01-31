# Barrier Options

## 1. Concept Skeleton
**Definition:** Path-dependent options activated (knock-in) or extinguished (knock-out) when underlying asset price crosses predetermined barrier level H during option life, offering cheaper hedging but discontinuous payoff risk.  
**Purpose:** Reduce premium cost (50-70% of vanilla); express directional views; manage knock-out risk; hedge structured products; exploit volatility smile; corporate cost-effective hedging.  
**Prerequisites:** European option pricing, path simulation, continuous monitoring, reflection principle, rebate valuation, first-passage time distributions

## 2. Comparative Framing

| Aspect | Down-and-Out Call | Up-and-Out Put | Knock-In Barrier | European Option | Double Barrier |
|--------|-------------------|----------------|------------------|-----------------|----------------|
| **Barrier Location** | H < S₀ (below spot) | H > S₀ (above spot) | Activates at H | No barrier | Two barriers (H_L, H_U) |
| **Payoff if Hit** | Worthless (knocked out) | Worthless | Activated | Always active | Complex (hit either) |
| **Premium vs European** | 40-70% (cheaper) | 40-70% (cheaper) | 30-60% (cheaper initially) | 100% (baseline) | 20-50% (cheapest) |
| **Risk** | Discontinuous (knock-out) | Discontinuous | Initially worthless | Continuous | Double discontinuity |
| **Monitoring** | Continuous (harder) | Continuous | Continuous | None | Continuous (both) |
| **Use Case** | Bull view, cost reduction | Bear view, cost reduction | Contrarian bet | Standard hedging | Range-bound strategies |

## 3. Examples + Counterexamples

**Simple Example: Down-and-Out Call**  
S₀=$100, K=$100, H=$90 (barrier below strike), T=1yr. Stock path: $100→$105→$110→$95→$98→$105... never touches $90. At expiry: S_T=$120. Payoff: max($120-$100,0) = $20 (like European call). Premium: ~$6 (vs $10 European) — 40% cheaper. If path had touched $90 at any point → knocked out → payoff=$0 (even if S_T=$120). Trade-off: 40% cost savings vs knock-out risk.

**Failure Case: Knock-Out Just Before Expiry**  
Down-and-out call: S₀=$100, K=$95, H=$92, T=1yr. Stock path: stable $105-$110 for 11 months (call deep ITM, value ~$15). Day 364: Flash crash → S drops to $91 (touches barrier) → knocked out → value instantly $0. Next day: S recovers to $115. At expiry T: S_T=$118. European call payoff: $23. Barrier call payoff: $0 (knocked out Day 364). Loss: -$23. Lesson: Barrier options have discontinuous risk; timing of barrier touch matters critically.

**Edge Case: Barrier Very Close to Strike (H≈K)**  
Up-and-out call: S₀=$100, K=$100, H=$101 (barrier 1% above ATM). Tiny room before knock-out. Any small upward move → knocked out. Effective payoff: max(S_T-K,0) if S never > $101 (nearly impossible if stock rises). Premium: ~$1-2 (90% discount vs European $10). Use case: Bet on stock staying flat/down; extremely cheap lottery ticket. Risk: Almost certain knock-out if bullish.

## 4. Layer Breakdown

```
Barrier Option Framework:
├─ Barrier Types (8 Standard Variants):
│   ├─ Down-and-Out (DO):
│   │   ├─ Call (DOC): Knocked out if S touches H < S₀; payoff max(S_T-K,0) if never touched H
│   │   ├─ Put (DOP): Knocked out if S touches H < S₀; payoff max(K-S_T,0) if never touched H
│   │   └─ Use: Bullish strategy with protection against large drops
│   ├─ Down-and-In (DI):
│   │   ├─ Call (DIC): Activated only if S touches H < S₀; worthless unless barrier hit
│   │   ├─ Put (DIP): Activated only if S touches H < S₀
│   │   └─ Use: Contrarian bet (expect initial drop then recovery)
│   ├─ Up-and-Out (UO):
│   │   ├─ Call (UOC): Knocked out if S touches H > S₀; payoff max(S_T-K,0) if never touched H
│   │   ├─ Put (UOP): Knocked out if S touches H > S₀; payoff max(K-S_T,0) if never touched H
│   │   └─ Use: Bearish strategy with cap on upside (corporate hedging)
│   ├─ Up-and-In (UI):
│   │   ├─ Call (UIC): Activated only if S touches H > S₀
│   │   ├─ Put (UIP): Activated only if S touches H > S₀
│   │   └─ Use: Bet on breakout above resistance level
│   └─ Parity Relationships:
│       ├─ European Call = DOC + DIC (knock-out + knock-in = vanilla)
│       ├─ European Put = UOP + UIP
│       └─ Pricing: Price one type, infer other via parity
├─ Continuous vs Discrete Monitoring:
│   ├─ Continuous Monitoring (theoretical):
│   │   ├─ Barrier checked at every instant t ∈ [0,T]
│   │   ├─ Mathematical: min(S_t) for t ∈ [0,T] compared to H
│   │   ├─ Closed-form pricing available (reflection principle)
│   │   └─ Reality: Impossible; approximation only
│   ├─ Discrete Monitoring (practical):
│   │   ├─ Barrier checked at specific times: t₁, t₂, ..., t_n (e.g., daily close)
│   │   ├─ Cheaper premium: Less frequent monitoring → lower knock-out probability
│   │   ├─ Example: Daily monitoring vs continuous → 5-10% cheaper
│   │   └─ MC pricing: Check barrier only at discrete observation points
│   └─ Monitoring Frequency Impact:
│       ├─ Continuous → Most expensive (highest knock-out risk)
│       ├─ Daily (252 obs/year) → Standard (95-98% of continuous value)
│       ├─ Weekly (52 obs/year) → Cheaper (85-90% of continuous)
│       └─ Monthly (12 obs/year) → Cheapest (70-80% of continuous)
├─ Monte Carlo Pricing with Barrier Monitoring:
│   ├─ Algorithm:
│   │   ├─ 1. Simulate M paths under risk-neutral GBM
│   │   ├─ 2. For each path, check if barrier H ever touched:
│   │   │   ├─ Continuous approx: min(S_t) < H (down barrier) or max(S_t) > H (up barrier)
│   │   │   └─ Discrete: Check S at monitoring times t₁, ..., t_n
│   │   ├─ 3. Determine option status:
│   │   │   ├─ Knock-out: If barrier touched → payoff = rebate R (often 0)
│   │   │   └─ Knock-in: If barrier touched → payoff = max(S_T-K,0); else 0
│   │   ├─ 4. Compute discounted payoff: V^(m) = e^{-rT} × payoff^(m)
│   │   └─ 5. Average: V_barrier = mean(V^(m))
│   ├─ Brownian Bridge Refinement (Continuous Monitoring):
│   │   ├─ Problem: Discrete time steps miss intraday barrier touches
│   │   ├─ Solution: Use Brownian bridge to estimate P(barrier touched between t_i and t_{i+1})
│   │   ├─ Formula: P(min(S_t) < H | S_{t_i}, S_{t_{i+1}}) = exp(-2·ln(S_{t_i}/H)·ln(S_{t_{i+1}}/H) / (σ²·dt))
│   │   ├─ Apply: For each path, if discrete check misses barrier, sample Brownian bridge
│   │   └─ Accuracy: Captures intraday knock-outs → closer to continuous pricing
│   └─ Rebate Valuation:
│       ├─ Rebate R: Payment if knocked out (compensation for early termination)
│       ├─ Timing: At knock-out time τ (when barrier touched) or at expiry T
│       ├─ Pricing: Discount rebate e^{-rτ} for immediate rebate, e^{-rT} for end-of-period
│       └─ Example: DOC with R=$5 → if knocked out → receive $5 (reduces downside)
├─ Closed-Form Approximations (Reflection Principle):
│   ├─ Down-and-Out Call (continuous monitoring):
│   │   ├─ DOC = European Call - Correction Term
│   │   ├─ Correction = (H/S₀)^{2λ} × [European Call(H²/S₀, K) - ...]
│   │   ├─ λ = (r - q + σ²/2) / σ² (drift parameter)
│   │   └─ Intuition: Reflect paths that cross barrier → subtract reflected value
│   ├─ Boundary Conditions:
│   │   ├─ If H ≥ K (down-and-out call, barrier above strike): DOC ≈ European call (barrier irrelevant if OTM)
│   │   ├─ If H < K (barrier below strike): DOC < European (knock-out risk material)
│   │   └─ As H → 0: DOC → European call (barrier infinitely far away)
│   └─ Limitations:
│       ├─ Closed-form assumes continuous monitoring (not realistic)
│       ├─ Discrete monitoring: Use MC or adjust closed-form with correction factor
│       └─ Complex barriers (time-dependent H_t): No closed-form → MC required
├─ Greeks for Barrier Options:
│   ├─ Delta (∂V/∂S):
│   │   ├─ Discontinuous at barrier: Delta jumps when S near H
│   │   ├─ Example: DOC with S=H+ε → Delta ≈ 0 (imminent knock-out)
│   │   └─ Hedging challenge: Frequent rebalancing near barrier
│   ├─ Gamma (∂²V/∂S²):
│   │   ├─ Spikes near barrier (extreme convexity)
│   │   ├─ Positive gamma away from H, large negative gamma at H (discontinuity)
│   │   └─ Risk: Gamma explosion → hedging costs high near barrier
│   ├─ Vega (∂V/∂σ):
│   │   ├─ Complex: Higher vol → more likely to hit barrier (bad for knock-out, good for knock-in)
│   │   ├─ Down-and-out: Negative vega (higher vol → higher knock-out risk)
│   │   ├─ Down-and-in: Positive vega (higher vol → higher activation chance)
│   │   └─ Non-monotonic near barrier
│   └─ Barrier Greeks Computation:
│       ├─ Finite Differences: ΔV ≈ (V(S+ε) - V(S-ε)) / (2ε)
│       ├─ Pathwise Derivative: For smooth payoffs (requires careful treatment at barrier)
│       └─ Likelihood Ratio Method: For discontinuous payoffs (barrier touch)
└─ Applications & Use Cases:
    ├─ Cost Reduction Hedging (Corporates):
    │   ├─ Company hedges FX exposure with down-and-out put (40% cheaper)
    │   ├─ Accept knock-out risk if extreme move unlikely
    │   └─ Example: Hedge EUR/USD with H=1.05, K=1.10 (put knocked out if EUR spikes)
    ├─ Structured Products:
    │   ├─ "Autocallable" notes: Pay coupon if stock stays in range (double barrier)
    │   ├─ If breached → early termination or rebate
    │   └─ Popular in Asia (retail structured products)
    ├─ Volatility Trading:
    │   ├─ Sell down-and-out options (collect premium, hope barrier not touched)
    │   ├─ Buy knock-in options (cheap lottery ticket on extreme moves)
    │   └─ Exploit vol smile: Barriers OTM often mispriced
    └─ Regulatory Capital (Banks):
        ├─ Barrier options reduce notional exposure (knocked out → exposure vanishes)
        ├─ Lower capital requirements vs vanilla options
        └─ Risk: Discontinuous P&L, model risk at barrier
```

**Interaction:** Path generation → Barrier monitoring → Knock-out/in status → Payoff determination → Discounting

## 5. Mini-Project

Implement down-and-out call with continuous monitoring approximation (Brownian bridge):

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class BarrierOption:
    """Monte Carlo pricing for barrier options with Brownian bridge"""
    
    def __init__(self, S0, K, H, T, r, sigma, q=0, barrier_type='down-and-out', 
                 option_type='call', rebate=0):
        self.S0 = S0
        self.K = K
        self.H = H
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.barrier_type = barrier_type  # 'down-and-out', 'up-and-out', 'down-and-in', 'up-and-in'
        self.option_type = option_type  # 'call' or 'put'
        self.rebate = rebate
    
    def simulate_paths(self, n_paths, n_steps):
        """Generate GBM paths"""
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for i in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            paths[:, i+1] = paths[:, i] * np.exp(
                (self.r - self.q - 0.5*self.sigma**2)*dt + 
                self.sigma*np.sqrt(dt)*Z
            )
        
        return paths
    
    def check_barrier_discrete(self, paths):
        """Check barrier touch with discrete monitoring"""
        if 'down' in self.barrier_type:
            # Check if any point touches barrier from above
            touched = np.any(paths <= self.H, axis=1)
        else:  # up barrier
            # Check if any point touches barrier from below
            touched = np.any(paths >= self.H, axis=1)
        
        return touched
    
    def check_barrier_brownian_bridge(self, paths):
        """Check barrier with Brownian bridge correction (continuous approximation)"""
        n_paths, n_steps_plus_1 = paths.shape
        dt = self.T / (n_steps_plus_1 - 1)
        touched = np.zeros(n_paths, dtype=bool)
        
        for i in range(n_steps_plus_1 - 1):
            S_start = paths[:, i]
            S_end = paths[:, i+1]
            
            if 'down' in self.barrier_type:
                # Down barrier: check if minimum between steps < H
                # Discrete check
                discrete_touch = (S_start <= self.H) | (S_end <= self.H)
                
                # Brownian bridge: P(min(S_t) < H | S_start, S_end)
                # If both above barrier, check if crossed intraday
                above_barrier = (S_start > self.H) & (S_end > self.H)
                if np.any(above_barrier):
                    log_ratio = np.log(S_start[above_barrier] / self.H) * \
                               np.log(S_end[above_barrier] / self.H)
                    prob_cross = np.exp(-2 * log_ratio / (self.sigma**2 * dt))
                    # Random draw: did path cross intraday?
                    cross_intraday = np.random.random(np.sum(above_barrier)) < prob_cross
                    
                    # Update touched array
                    indices = np.where(above_barrier)[0]
                    touched[indices[cross_intraday]] = True
                
                touched |= discrete_touch
                
            else:  # up barrier
                discrete_touch = (S_start >= self.H) | (S_end >= self.H)
                
                # Brownian bridge for up barrier
                below_barrier = (S_start < self.H) & (S_end < self.H)
                if np.any(below_barrier):
                    log_ratio = np.log(self.H / S_start[below_barrier]) * \
                               np.log(self.H / S_end[below_barrier])
                    prob_cross = np.exp(-2 * log_ratio / (self.sigma**2 * dt))
                    cross_intraday = np.random.random(np.sum(below_barrier)) < prob_cross
                    
                    indices = np.where(below_barrier)[0]
                    touched[indices[cross_intraday]] = True
                
                touched |= discrete_touch
        
        return touched
    
    def payoff(self, S_T):
        """Terminal payoff (call or put)"""
        if self.option_type == 'call':
            return np.maximum(S_T - self.K, 0)
        else:  # put
            return np.maximum(self.K - S_T, 0)
    
    def price(self, n_paths=20000, n_steps=100, use_brownian_bridge=True):
        """Price barrier option"""
        paths = self.simulate_paths(n_paths, n_steps)
        
        # Check barrier
        if use_brownian_bridge:
            barrier_touched = self.check_barrier_brownian_bridge(paths)
        else:
            barrier_touched = self.check_barrier_discrete(paths)
        
        # Terminal payoff
        terminal_payoff = self.payoff(paths[:, -1])
        
        # Apply barrier logic
        if 'out' in self.barrier_type:
            # Knock-out: payoff only if barrier NOT touched
            final_payoff = np.where(barrier_touched, self.rebate, terminal_payoff)
        else:  # knock-in
            # Knock-in: payoff only if barrier touched
            final_payoff = np.where(barrier_touched, terminal_payoff, self.rebate)
        
        # Discount
        price = np.exp(-self.r*self.T) * np.mean(final_payoff)
        std_error = np.exp(-self.r*self.T) * np.std(final_payoff) / np.sqrt(n_paths)
        
        # Statistics
        knock_out_pct = np.mean(barrier_touched) * 100 if 'out' in self.barrier_type else \
                       (100 - np.mean(barrier_touched) * 100)
        
        return {
            'price': price,
            'std_error': std_error,
            'barrier_touched_pct': np.mean(barrier_touched) * 100,
            'knock_out_pct': knock_out_pct,
            'paths': paths,
            'barrier_touched': barrier_touched,
            'final_payoff': final_payoff
        }
    
    def european_price_bs(self):
        """Black-Scholes European price (benchmark)"""
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        if self.option_type == 'call':
            return self.S0*np.exp(-self.q*self.T)*norm.cdf(d1) - \
                   self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:  # put
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - \
                   self.S0*np.exp(-self.q*self.T)*norm.cdf(-d1)

# Parameters
S0, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.25, 0.02
H_down = 85  # Down barrier (15% below spot)
H_up = 115   # Up barrier (15% above spot)
np.random.seed(42)

print("="*80)
print("BARRIER OPTION PRICING (Down-and-Out Call)")
print("="*80)

# Price down-and-out call
doc = BarrierOption(S0, K, H_down, T, r, sigma, q, barrier_type='down-and-out', 
                    option_type='call', rebate=0)

result_discrete = doc.price(n_paths=20000, n_steps=100, use_brownian_bridge=False)
result_continuous = doc.price(n_paths=20000, n_steps=100, use_brownian_bridge=True)
european_price = doc.european_price_bs()

print(f"\nOption Parameters:")
print(f"  Spot S0:             ${S0:.2f}")
print(f"  Strike K:            ${K:.2f}")
print(f"  Barrier H (down):    ${H_down:.2f}")
print(f"  Time to Expiry:      {T:.2f} years")
print(f"  Interest Rate:       {r*100:.1f}%")
print(f"  Volatility:          {sigma*100:.1f}%")
print(f"  Dividend Yield:      {q*100:.1f}%")

print(f"\n{'='*80}")
print("PRICING RESULTS")
print("="*80)
print(f"  European Call (BS):              ${european_price:.4f}")
print(f"  Down-and-Out Call (Discrete):    ${result_discrete['price']:.4f} ± ${result_discrete['std_error']:.4f}")
print(f"  Down-and-Out Call (Brownian Br): ${result_continuous['price']:.4f} ± ${result_continuous['std_error']:.4f}")
print(f"\n  Discrete Monitoring Discount:    {(1-result_discrete['price']/european_price)*100:.1f}%")
print(f"  Continuous Monitoring Discount:  {(1-result_continuous['price']/european_price)*100:.1f}%")
print(f"\n  Barrier Touch Probability:")
print(f"    Discrete (100 steps):   {result_discrete['barrier_touched_pct']:.1f}%")
print(f"    Continuous (Brownian):  {result_continuous['barrier_touched_pct']:.1f}%")

# Barrier sensitivity
print(f"\n{'='*80}")
print("BARRIER LEVEL SENSITIVITY")
print("="*80)

barrier_levels = [75, 80, 85, 90, 95]
print(f"\n{'Barrier H':>12} {'DO Call Price':>18} {'Discount vs Euro':>20} {'Touch Prob %':>15}")
print("-"*70)

for H in barrier_levels:
    doc_temp = BarrierOption(S0, K, H, T, r, sigma, q, 'down-and-out', 'call')
    res = doc_temp.price(n_paths=10000, n_steps=100, use_brownian_bridge=True)
    discount = (1 - res['price']/european_price) * 100
    print(f"  ${H:>10.0f}    ${res['price']:>16.4f}    {discount:>18.1f}%    {res['barrier_touched_pct']:>13.1f}%")

# Compare all barrier types
print(f"\n{'='*80}")
print("ALL BARRIER TYPES (K=$100, H_down=$85, H_up=$115)")
print("="*80)

barrier_configs = [
    ('down-and-out', 'call', H_down),
    ('down-and-in', 'call', H_down),
    ('up-and-out', 'call', H_up),
    ('up-and-in', 'call', H_up),
    ('down-and-out', 'put', H_down),
    ('up-and-out', 'put', H_up),
]

print(f"\n{'Type':>25} {'Price':>12} {'vs European':>15} {'Touch %':>12}")
print("-"*70)

for btype, otype, H in barrier_configs:
    opt = BarrierOption(S0, K, H, T, r, sigma, q, btype, otype)
    res = opt.price(n_paths=8000, n_steps=100, use_brownian_bridge=True)
    euro_price = opt.european_price_bs()
    pct = res['price'] / euro_price * 100
    print(f"  {btype:>15} {otype:>8}    ${res['price']:>10.4f}    {pct:>13.1f}%    {res['barrier_touched_pct']:>10.1f}%")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Sample paths (knocked out vs survived)
n_sample = 100
result_viz = doc.price(n_paths=n_sample, n_steps=252, use_brownian_bridge=False)
paths_viz = result_viz['paths']
knocked_out = result_viz['barrier_touched']
time_grid = np.linspace(0, T, 253)

for i in range(n_sample):
    color = 'red' if knocked_out[i] else 'blue'
    alpha = 0.6 if knocked_out[i] else 0.2
    linewidth = 1.0 if knocked_out[i] else 0.5
    axes[0, 0].plot(time_grid, paths_viz[i, :], color=color, alpha=alpha, linewidth=linewidth)

axes[0, 0].axhline(H_down, color='black', linestyle='--', linewidth=2.5, label=f'Barrier H=${H_down}')
axes[0, 0].axhline(K, color='green', linestyle=':', linewidth=2, label=f'Strike K=${K}')
axes[0, 0].set_title('Down-and-Out Call Paths (Red=Knocked Out, Blue=Survived)')
axes[0, 0].set_xlabel('Time (years)')
axes[0, 0].set_ylabel('Stock Price S')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_ylim(70, 130)

# Plot 2: Payoff distribution
terminal_prices = result_continuous['paths'][:, -1]
payoffs = result_continuous['final_payoff']

axes[0, 1].scatter(terminal_prices, payoffs, alpha=0.3, s=5, c=result_continuous['barrier_touched'], 
                  cmap='coolwarm')
axes[0, 1].axvline(K, color='green', linestyle='--', linewidth=2, label='Strike K')
axes[0, 1].set_title('Terminal Price vs Payoff (Red=Knocked Out)')
axes[0, 1].set_xlabel('Terminal Stock Price S_T')
axes[0, 1].set_ylabel('Payoff at Expiry')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Barrier level sensitivity
barrier_range = np.linspace(70, 98, 15)
prices_doc = []
prices_dic = []

for H in barrier_range:
    doc_temp = BarrierOption(S0, K, H, T, r, sigma, q, 'down-and-out', 'call')
    dic_temp = BarrierOption(S0, K, H, T, r, sigma, q, 'down-and-in', 'call')
    
    res_doc = doc_temp.price(n_paths=5000, n_steps=50, use_brownian_bridge=True)
    res_dic = dic_temp.price(n_paths=5000, n_steps=50, use_brownian_bridge=True)
    
    prices_doc.append(res_doc['price'])
    prices_dic.append(res_dic['price'])

axes[0, 2].plot(barrier_range, prices_doc, 'b-', linewidth=2.5, marker='o', label='Down-and-Out Call')
axes[0, 2].plot(barrier_range, prices_dic, 'r-', linewidth=2.5, marker='s', label='Down-and-In Call')
axes[0, 2].axhline(european_price, color='green', linestyle='--', linewidth=2, label='European Call')
axes[0, 2].axvline(S0, color='black', linestyle=':', alpha=0.5, label='Spot S0')
axes[0, 2].set_title('Barrier Level Sensitivity')
axes[0, 2].set_xlabel('Barrier Level H')
axes[0, 2].set_ylabel('Option Price ($)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Monitoring frequency impact
frequencies = [12, 52, 100, 252, 500]
prices_by_freq = []
se_by_freq = []

for freq in frequencies:
    res = doc.price(n_paths=8000, n_steps=freq, use_brownian_bridge=False)
    prices_by_freq.append(res['price'])
    se_by_freq.append(res['std_error'])

axes[1, 0].errorbar(frequencies, prices_by_freq, yerr=se_by_freq, fmt='o-', 
                   linewidth=2, markersize=8, capsize=5, label='Discrete Monitoring')
axes[1, 0].axhline(result_continuous['price'], color='red', linestyle='--', 
                  linewidth=2, label='Continuous (Brownian Bridge)')
axes[1, 0].set_title('Monitoring Frequency Impact on Price')
axes[1, 0].set_xlabel('Observations per Year')
axes[1, 0].set_ylabel('Down-and-Out Call Price ($)')
axes[1, 0].set_xscale('log')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, which='both')

# Plot 5: DO Call price vs spot price
spot_range = np.linspace(86, 115, 20)
doc_prices_spot = []
euro_prices_spot = []

for s in spot_range:
    doc_temp = BarrierOption(s, K, H_down, T, r, sigma, q, 'down-and-out', 'call')
    res_doc = doc_temp.price(n_paths=5000, n_steps=100, use_brownian_bridge=True)
    doc_prices_spot.append(res_doc['price'])
    euro_prices_spot.append(doc_temp.european_price_bs())

axes[1, 1].plot(spot_range, doc_prices_spot, 'r-', linewidth=2.5, marker='o', label='Down-and-Out Call')
axes[1, 1].plot(spot_range, euro_prices_spot, 'b--', linewidth=2.5, marker='s', label='European Call')
axes[1, 1].axvline(H_down, color='black', linestyle='--', linewidth=2, label=f'Barrier H=${H_down}')
axes[1, 1].axvline(K, color='green', linestyle=':', linewidth=2, label=f'Strike K=${K}')
axes[1, 1].set_title('Option Value vs Spot Price')
axes[1, 1].set_xlabel('Spot Price S')
axes[1, 1].set_ylabel('Option Value ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Parity check (DO + DI = European)
spot_range_parity = np.linspace(85, 115, 15)
doc_prices = []
dic_prices = []
euro_prices = []

for s in spot_range_parity:
    doc_temp = BarrierOption(s, K, H_down, T, r, sigma, q, 'down-and-out', 'call')
    dic_temp = BarrierOption(s, K, H_down, T, r, sigma, q, 'down-and-in', 'call')
    
    res_doc = doc_temp.price(n_paths=8000, n_steps=100, use_brownian_bridge=True)
    res_dic = dic_temp.price(n_paths=8000, n_steps=100, use_brownian_bridge=True)
    
    doc_prices.append(res_doc['price'])
    dic_prices.append(res_dic['price'])
    euro_prices.append(doc_temp.european_price_bs())

sum_prices = np.array(doc_prices) + np.array(dic_prices)

axes[1, 2].plot(spot_range_parity, euro_prices, 'g-', linewidth=2.5, marker='o', label='European Call')
axes[1, 2].plot(spot_range_parity, sum_prices, 'ro', linewidth=2, markersize=8, 
               alpha=0.6, label='DO + DI (MC)')
axes[1, 2].set_title('Barrier Parity: European = DO + DI')
axes[1, 2].set_xlabel('Spot Price S')
axes[1, 2].set_ylabel('Option Value ($)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('barrier_options.png', dpi=100, bbox_inches='tight')
print("\n" + "="*80)
print("Plot saved: barrier_options.png")
print("="*80)
```

**Output Interpretation:**
- **Down-and-out premium:** 40-60% of European (depends on barrier distance)
- **Brownian bridge:** Captures 5-10% more knock-outs (continuous approximation)
- **Parity holds:** DO + DI ≈ European (within MC error)

## 6. Challenge Round

**Q1: Down-and-out call (S₀=$100, K=$100, H=$85) costs $6. European call costs $10. Why 40% discount?**  
A: Knock-out probability ≈ 20-25% (barrier 15% below spot → material risk of touching in 1 year with σ=25%). When knocked out, payoff=$0 (vs European max(S_T-K,0) > 0). Expected value: 0.75×(European payoff) → price ≈ 0.75×$10 = $7.50 if linear, but option convexity → actually $6 (40% discount). Intuition: Deep OTM barrier touch → lose all time value + intrinsic value → substantial discount.

**Q2: Discrete monitoring (daily) prices barrier option at $6.20; continuous (Brownian bridge) at $5.80. Why cheaper with continuous?**  
A: Continuous monitoring checks barrier at every instant → higher probability of knock-out. Daily monitoring checks only at close → misses intraday touches. Example: Stock opens $100, drops to $84 intraday (touches barrier), closes $95. Daily: barrier NOT touched (close > H=$85). Continuous: barrier touched → knocked out. More knock-outs → lower option value. Difference: 5-10% typically (daily ≈ 95% of continuous value).

**Q3: Up-and-out put (S₀=$100, K=$100, H=$110) vs European put. Which costs less and why?**  
A: Up-and-out put cheaper (50-60% of European). Knock-out if stock rises above $110 → lose put value. Put profits when S drops (S < K) → stock rising to H=$110 bad for put anyway (already losing value). Knock-out accelerates loss. Example: S rises $100→$115 → European put worth ~$0 (OTM), Up-and-out put knocked out at $110 → also $0. But if S rises then falls ($100→$115→$95 at expiry) → European put pays $5, Up-and-out pays $0 (already knocked out). Trade-off: 40-50% cheaper premium for accepting path-dependent risk.

**Q4: Brownian bridge formula: P(min < H | S_start, S_end) = exp(-2 ln(S_start/H) ln(S_end/H) / (σ²dt)). Explain intuition.**  
A: Reflection principle: Path crossing barrier from above → reflected path crosses downward. Probability proportional to (distance to barrier)² for both endpoints. Log terms measure distance: ln(S/H) = # std deviations away. Product of log terms → joint likelihood. If both S_start and S_end far above H → product large → exp(-large) ≈ 0 (unlikely to cross). If either near H → product small → exp(-small) ≈ 1 (likely to cross). Formula captures geometric Brownian motion's continuous-time crossing probability.

**Q5: Trader hedges down-and-out call delta. Stock drops toward barrier H. What happens to delta and hedging cost?**  
A: Delta discontinuous at barrier. Far from H: delta ≈ European delta (0.5 for ATM). Near H: delta → 0 (imminent knock-out → option worthless). Hedging: As S approaches H, delta collapses → must sell hedge shares (delta-hedging sells high as stock rises, buys low as stock falls). But if stock bounces away from H → delta jumps back → must rebuy shares (buy high). Gamma spike at barrier → frequent rebalancing → high transaction costs. Worst case: Stock oscillates around H → delta whipsaws → hemorrhaging costs. Solution: Widen rehedge tolerance near barrier or accept delta exposure.

**Q6: Bank sells structured note: "Pays 10% coupon if stock stays between $90-$110; else pays 0%." How model this with barriers?**  
A: Double barrier option (down-and-out at $90, up-and-out at $110). Payoff: $10 coupon if neither barrier touched; $0 if either touched. Equivalent: Sell zero-coupon bond ($100) + long double-knock-out digital call (pays $10 if survives). Pricing: MC with two barrier checks. Knock-out probability ≈ 40-60% (depends on vol, time). Fair coupon ≈ r/(1-P_knockout) → if P_knockout=50%, fair coupon ≈ 10% (matches risk-free rate 5% × 2 adjustment). Bank profits from: (1) selling above fair value, (2) hedging costs lower than implied, (3) correlation/vol surface arbitrage. Risk: Both barriers near spot → high gamma → expensive hedging.

## 7. Key References

- Rubinstein & Reiner (1991): "Unscrambling the Binary Code" — Closed-form barrier formulas, reflection principle
- [Wikipedia: Barrier Option](https://en.wikipedia.org/wiki/Barrier_option) — Types, monitoring, applications
- Boyle & Lau (1994): "Bumping Up Against the Barrier" — Brownian bridge correction, continuous monitoring
- Hull: *Options, Futures & Derivatives* (Chapter 26) — Exotic options, barrier structures

**Status:** ✓ Standalone file. **Complements:** asian_options.md, lookback_options.md, european_option_pricing.md, path_dependent_options.md
