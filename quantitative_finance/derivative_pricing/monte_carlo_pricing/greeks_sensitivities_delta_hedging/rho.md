# Rho

## 1. Concept Skeleton
**Definition:** The first derivative of option price with respect to interest rates (ρ = ∂C/∂r), measuring price sensitivity to changes in the risk-free rate affecting discount factors and forward prices.  
**Purpose:** Quantify interest rate exposure, manage rate risk in option portfolios, hedge against rate changes in fixed-income derivatives, assess impact of rate regime shifts on derivatives, model cross-asset (equity-rate) correlations  
**Prerequisites:** Discount factors, forward prices, interest rate models, option pricing under constant rates, portfolio duration concepts

## 2. Comparative Framing

| Aspect | Call Rho | Put Rho | Long Bond Future | Short Bond Future | Rate Swap |
|--------|----------|---------|------------------|-------------------|-----------|
| **Rho Sign** | ρ > 0 (positive) | ρ < 0 (negative) | ρ < 0 (inverse) | ρ > 0 | ρ ≈ 0 (neutral) |
| **Rate Increase** | Gain | Loss | Loss | Gain | Minimal impact |
| **Rate Decrease** | Loss | Gain | Gain | Loss | Minimal impact |
| **Impact Magnitude** | Moderate (% of price) | Moderate | Large (duration effect) | Large | Minimal |
| **Trading Frequency** | Rarely hedge | Rarely hedge | Constantly hedged | Constantly hedged | Core hedging tool |
| **Portfolio Importance** | Low (equity focus) | Low (equity focus) | High (rate-sensitive) | High | Essential |
| **Typical Value** | $0.10-$0.50 | -$0.10-$0.50 | Large negative/positive | Large | Near-zero |

## 3. Examples + Counterexamples

**Simple Example: Call Rho**  
Long call: K=100, T=1yr, S=100, σ=20%, r=5%. BS call price ≈ $10.45. If r increases to 6% (+1%): call price ≈ $10.70 (gain ~$0.25). Rho ≈ $0.25. Interpretation: each 1% rate increase adds ~$0.25 to call value. Intuition: higher r increases forward price F = S e^{rT} → call payoff expected value increases.

**Failure Case: Ignoring Rho in Structured Products**  
Bank sells structured note: "equity-linked return" with embedded short call on SPX (S&P 500). Bank hedges: long SPX, short call delta-hedged. Rho ignored. Rate shock: Fed raises rates 1% → all equity calls rise in value (bank's short call loses). Rate shock magnitude: large portfolio rho exposure (10,000 short calls × $0.25 rho ≈ $2.5M loss). Bank underestimated rate risk; profit margin erased by rate move (unhedged rho).

**Edge Case: Very Short/Long Maturity**  
T → 0 (expiry tomorrow): rho → 0 (discount factor nearly 1). Change in r has minimal impact on final payoff. T → ∞ (far future): rho increases. But far-dated options are illiquid; practical rho risk concentrated in liquid tenors (6M-2Y). Near-maturity options: rho negligible. Long-dated options: rho material but often hedged separately (via bonds/swaps).

## 4. Layer Breakdown

```
Rho Framework:
├─ Definition & Interpretation:
│   ├─ Rho ρ = ∂C/∂r (per 1% change in rate)
│   ├─ For calls: ρ > 0 (rates up → calls up)
│   ├─ For puts: ρ < 0 (rates up → puts down)
│   ├─ Units: dollars per 1% rate change
│   └─ Magnitude:
│       ├─ Modest for short-dated (T < 6M): rho ≈ $0.05-0.10 per 1%
│       ├─ Significant for long-dated (T=2-5Y): rho ≈ $0.50-2.00 per 1%
│       └─ Very large for perpetual/far-dated options (used in structured products)
├─ Mechanisms:
│   ├─ Discount Factor Effect:
│   │   ├─ Option value = E^Q[discounted payoff] = e^{-rT} E^Q[payoff at T]
│   │   ├─ Higher r → smaller e^{-rT} → option value decreases
│   │   ├─ BUT forward price F = S e^{rT} increases (dominates for calls)
│   │   └─ Net: for calls (long forward), F increase dominates discount effect
│   ├─ Forward Price Effect:
│   │   ├─ Call payoff depends on S_T vs K
│   │   ├─ Higher r → higher expected S_T (increased drift)
│   │   ├─ Call beneficiary: more likely ITM, larger payoff
│   │   └─ Put suffers: less likely ITM, smaller payoff
│   ├─ Black-Scholes Rho Formula:
│   │   ├─ Call Rho: ρ_C = K T e^{-rT} N(d2)
│   │   ├─ Put Rho: ρ_P = -K T e^{-rT} N(-d2)
│   │   ├─ Where d2 = [ln(S/K) + (r-q+σ²/2)T]/(σ√T) - σ√T
│   │   └─ Properties:
│   │       ├─ Proportional to K (strike): higher strike = higher rho
│   │       ├─ Proportional to T (maturity): longer → higher rho
│   │       ├─ Rho peak at ATM (like other Greeks)
│   │       └─ Rho increases with N(d2), which increases with r (feedback)
│   └─ Interest Rate Term Structure:
│       ├─ Rho for short-dated: uses 1-week rate r_1W
│       ├─ Rho for long-dated: uses 2-year rate r_2Y (term-dependent)
│       ├─ Discrepancy: if curve inverted (r_1W > r_2Y), rho affects different maturities differently
│       └─ Portfolio rho: aggregate across curve (parallel shift assumption or bucketed)
├─ Rho P&L:
│   ├─ Simple: ΔC ≈ ρ × Δr
│   ├─ Example: long 100 1-year ATM calls, ρ = $0.30/call
│   │   ├─ Total rho: 100 × $0.30 = $30 per 1% rate move
│   │   ├─ If rates rise 0.5%: P&L = $30 × 0.5 = +$15 (gains)
│   │   └─ If rates fall 0.5%: P&L = $30 × (-0.5) = -$15 (losses)
│   ├─ Rho vs Realized Rate Path:
│   │   ├─ Rho hedges parallel shifts, not curve changes
│   │   ├─ If short-end rises, long-end falls (twist): rho mismatches
│   │   └─ Advanced: key rate duration bucketing (per maturity segment)
│   └─ Portfolio Rho:
│       ├─ Ρ_portfolio = Σ ρᵢ × qᵢ (sum across all positions)
│       ├─ Positive: net long rate risk (benefit if rates rise)
│       ├─ Negative: net short rate risk (benefit if rates fall)
│       └─ Monitoring: daily; stress test ±25bp, ±100bp moves
├─ Practical Rho Trading & Hedging:
│   ├─ Equity Derivatives (Rho Secondary):
│   │   ├─ Rho typically low (equity traders ignore)
│   │   ├─ Unless: long-dated structures, high-strike OTM calls
│   │   └─ Hedge: via bond futures or rate swaps if material
│   ├─ Interest Rate Derivatives (Rho Primary):
│   │   ├─ Swaptions (options on swaps): large rho
│   │   ├─ Caps/Floors: rho material; hedge via short/long bonds
│   │   └─ Bond options: rho dominates; core risk metric
│   ├─ Cross-Asset Hedging:
│   │   ├─ Long equity call + short 5Y interest rate cap
│   │   ├─ If rates spike: call loses (negative rho effect), cap loses (loses premium)
│   │   ├─ Correlation break: both lose simultaneously
│   │   └─ Hedge: opposite direction in bonds/swaps to offset
│   ├─ Rate Scenario Analysis:
│   │   ├─ Parallel shift: rates up 1% uniformly
│   │   ├─ Steepening: short rates up 0.5%, long rates up 1.5%
│   │   ├─ Flattening: short rates up 1.5%, long rates up 0.5%
│   │   └─ Twist: short down, long up (most complex)
│   └─ Limits & Monitoring:
│       ├─ Rho limits per desk (dollar exposure to rate moves)
│       ├─ Stress: parallel 1% move across all maturities
│       ├─ Key rate duration: sensitivity per tenor bucket (1Y, 5Y, 10Y, 30Y)
│       └─ Daily P&L tracking: rho contribution separate
└─ Rho in Different Market Environments:
    ├─ Low-rate environment (r near 0%):
    │   ├─ Rho impact magnified (small rate % moves = large rho loss)
    │   ├─ Example: rate from 0.50% to 1.50% (+1%): large rho impact
    │   └─ 2022 example: rates from 0% to 3% rapid → hedges costly
    ├─ High-rate environment (r > 5%):
    │   ├─ Rho impact percentage-wise less critical
    │   ├─ Absolute dollar impact still material for large notional
    │   └─ Less hedging urgency (rates already embed risk premium)
    └─ Rate volatility:
        ├─ Increasing vol of rates increases option value (vega of interest rates)
        ├─ Rho captures level; separate vol-of-rates captures uncertainty
        └─ Cross-gamma: rates and equity correlated negatively (crisis → both move against longs)
```

**Interaction:** Rho increases with maturity T and strike K; dominates for long-dated and far-OTM options; less material for short-dated equity derivatives.

## 5. Mini-Project

Analyze rho exposure; simulate rate moves; hedge rho with bond positions:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class RhoAnalysis:
    """Analyze rho (interest rate sensitivity)"""
    
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def bs_call_price(self, S, t, r=None):
        if r is None:
            r = self.r
        tau = self.T - t
        if tau <= 0:
            return max(S - self.K, 0)
        d1 = (np.log(S/self.K) + (r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        return S*np.exp(-self.q*tau)*norm.cdf(d1) - \
               self.K*np.exp(-r*tau)*norm.cdf(d2)
    
    def bs_rho_call(self, S, t):
        """Call rho, per 1% rate change"""
        tau = self.T - t
        if tau <= 0:
            return 0.0
        d2 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau)) - self.sigma*np.sqrt(tau)
        return self.K * tau * np.exp(-self.r*tau) * norm.cdf(d2) / 100
    
    def bs_rho_put(self, S, t):
        """Put rho, per 1% rate change"""
        tau = self.T - t
        if tau <= 0:
            return 0.0
        d2 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau)) - self.sigma*np.sqrt(tau)
        return -self.K * tau * np.exp(-self.r*tau) * norm.cdf(-d2) / 100

# Parameters
S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
analyzer = RhoAnalysis(S, K, T, r, sigma, q)

print("="*70)
print("RHO ANALYSIS")
print("="*70)

# Current rho at different tenors
print(f"\nCurrent Rho (per 1% rate change):")
print(f"  Call (ATM): ${analyzer.bs_rho_call(S, 0):.4f}")
print(f"  Put (ATM): ${analyzer.bs_rho_put(S, 0):.4f}")

# Rho across tenors
print(f"\n{'Maturity (Years)':>15} {'Call Rho':>15} {'Put Rho':>15}")
print("-"*45)
for tenor in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
    analyzer_tenor = RhoAnalysis(S, K, tenor, r, sigma, q)
    print(f"{tenor:>15.2f}    ${analyzer_tenor.bs_rho_call(S, 0):>13.4f}    ${analyzer_tenor.bs_rho_put(S, 0):>13.4f}")

# Simulate rate scenario impact
print(f"\n{'='*70}")
print("RATE SCENARIO ANALYSIS (1Y ATM Calls)")
print("="*70)

print(f"\n{'Rate Scenario':>20} {'Old Call Price':>20} {'New Call Price':>20} {'P&L Change':>20}")
print("-"*80)

base_price = analyzer.bs_call_price(S, 0)
rho = analyzer.bs_rho_call(S, 0)

for rate_scenario in [0.03, 0.04, 0.05, 0.06, 0.07]:
    new_price = analyzer.bs_call_price(S, 0, r=rate_scenario)
    pnl_change = new_price - base_price
    rho_approx = rho * (rate_scenario - r) * 100
    print(f"{rate_scenario*100:>18.1f}%    ${base_price:>18.4f}    ${new_price:>18.4f}    ${pnl_change:>18.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Rho profile across spot prices
spot_range = np.linspace(80, 120, 100)
rhos_call = [analyzer.bs_rho_call(s, 0) for s in spot_range]
rhos_put = [analyzer.bs_rho_put(s, 0) for s in spot_range]

axes[0, 0].plot(spot_range, rhos_call, 'b-', linewidth=2, label='Call Rho')
axes[0, 0].plot(spot_range, rhos_put, 'r-', linewidth=2, label='Put Rho')
axes[0, 0].axvline(S, color='g', linestyle=':', alpha=0.7, label='ATM')
axes[0, 0].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[0, 0].set_title('Rho Profile Across Spot (1Y)')
axes[0, 0].set_xlabel('Spot Price S')
axes[0, 0].set_ylabel('Rho ($ per 1% rate)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Rho across tenors
tenors = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
rhos_tenor_call = []
rhos_tenor_put = []
for tenor in tenors:
    analyzer_tenor = RhoAnalysis(S, K, tenor, r, sigma, q)
    rhos_tenor_call.append(analyzer_tenor.bs_rho_call(S, 0))
    rhos_tenor_put.append(analyzer_tenor.bs_rho_put(S, 0))

axes[0, 1].plot(tenors, rhos_tenor_call, 'bo-', linewidth=2, markersize=8, label='Call Rho')
axes[0, 1].plot(tenors, rhos_tenor_put, 'rs-', linewidth=2, markersize=8, label='Put Rho')
axes[0, 1].set_title('Rho Term Structure (ATM)')
axes[0, 1].set_xlabel('Time to Maturity (years)')
axes[0, 1].set_ylabel('Rho ($ per 1% rate)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Call price sensitivity to rate changes
rate_range = np.linspace(0.01, 0.10, 100)
call_prices_by_rate = [analyzer.bs_call_price(S, 0, r=r_temp) for r_temp in rate_range]

axes[0, 2].plot(rate_range*100, call_prices_by_rate, 'b-', linewidth=2)
axes[0, 2].axvline(r*100, color='r', linestyle=':', alpha=0.7, label='Current Rate')
axes[0, 2].axhline(base_price, color='r', linestyle=':', alpha=0.7)
axes[0, 2].set_title('Call Price vs Interest Rate')
axes[0, 2].set_xlabel('Interest Rate (%)')
axes[0, 2].set_ylabel('Call Price')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Rho vs other Greeks (composition of risk)
greeks_labels = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
from scipy.stats import norm
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
delta = np.exp(-q*T)*norm.cdf(d1)
gamma = np.exp(-q*T)*norm.pdf(d1) / (S*sigma*np.sqrt(T)) * 100  # Scale for visibility
vega = S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T) / 100
theta = -S*np.exp(-q*T)*norm.pdf(d1)*sigma/(2*np.sqrt(T)) / 365
rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100

greeks_values = [delta, gamma, vega, theta, rho]
colors_greeks = ['blue', 'green', 'red', 'orange', 'purple']

axes[1, 0].barh(greeks_labels, greeks_values, color=colors_greeks, alpha=0.7)
axes[1, 0].set_title('Greeks Magnitude Comparison (1Y ATM Call)')
axes[1, 0].set_xlabel('Value (different units)')
axes[1, 0].grid(alpha=0.3)

# Plot 5: P&L from rate moves (100 contracts)
rate_moves = np.linspace(-0.02, 0.02, 50)
pnl_rho = 100 * rho * rate_moves * 100  # 100 contracts, scaled

axes[1, 1].plot(rate_moves*100, pnl_rho, 'b-', linewidth=2)
axes[1, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 1].axvline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 1].fill_between(rate_moves*100, 0, pnl_rho, where=pnl_rho>=0, alpha=0.2, color='green', label='Profit')
axes[1, 1].fill_between(rate_moves*100, 0, pnl_rho, where=pnl_rho<0, alpha=0.2, color='red', label='Loss')
axes[1, 1].set_title('Portfolio P&L from Rate Moves (100 1Y Calls)')
axes[1, 1].set_xlabel('Rate Move (%)')
axes[1, 1].set_ylabel('P&L ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Combined Greeks P&L: delta move + rate move scenario
spot_moves = np.linspace(-5, 5, 50)
spot_pnl_matrix = []
for sm in spot_moves:
    pnl_spot = 100 * sm * delta
    pnl_rate = 100 * rho * 0.01 * 100  # +1% rate move
    spot_pnl_matrix.append(pnl_spot + pnl_rate)

axes[1, 2].plot(spot_moves, spot_pnl_matrix, 'b-', linewidth=2, label='Delta + Rho')
axes[1, 2].plot(spot_moves, 100*spot_moves*delta, 'g--', linewidth=2, label='Delta only', alpha=0.7)
axes[1, 2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 2].set_title('Combined P&L: Spot Move + Rate Up 1%')
axes[1, 2].set_xlabel('Spot Price Move ($)')
axes[1, 2].set_ylabel('P&L ($)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rho_analysis.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: rho_analysis.png")
```

**Output Interpretation:**
- **Rho Profile:** Positive for calls, negative for puts; material for long-dated (rho scales with T)
- **Term Structure:** Rho increases with maturity; 10Y rho >> 3M rho
- **P&L Sensitivity:** 100bp rate move → $0.30 × 100 = $30 per 100 contracts (material for large books)

## 6. Challenge Round

**Q1: Why is call rho positive (rates up → call value up) while put rho is negative?**  
A: Rho = ∂C/∂r = K T e^{-rT} N(d2). Two effects: (1) discount factor e^{-rT} decreases with r (option value decreases), (2) forward price F = S e^{rT} increases with r. For calls: higher r → higher expected S_T (increased drift in risk-neutral measure) → call more likely ITM → payoff increases → dominates discount effect. For puts: opposite; higher r → less likely ITM → payoff decreases. Net: call rho positive, put rho negative.

**Q2: An equity options trader ignores rho for short-dated options (T < 1M). Is this reasonable?**  
A: Yes, reasonable for short-dated. Rho ∝ T; 1-month call rho ≈ $0.01 (minimal), 1-year call rho ≈ $0.30 (material). 1M option expires before meaningful rate move typically occurs (unless FOMC surprise). However: if rates volatile (e.g., Fed announcement tomorrow), even small rho matters. Rule of thumb: ignore rho if T < 1M and rate vol low; hedge if T > 6M or rate vol high.

**Q3: Portfolio rho is -$500. How would a trader hedge this exposure?**  
A: Net short rate risk: rates falling hurt portfolio. Hedge: buy long-duration bonds or go long interest rate futures. Example: 10-year Treasury futures contract (notional ≈ $100k per contract, duration ≈ 8 years). Duration P&L ≈ -8% × (rate move) per $100k. Target: rates down 1% → bond gain ≈ $8,000. Offset: rho loss $500 × 1% = $500 loss on options. Net: $8,000 - $500 = $7,500 (slight over-hedge). Adjust position size to fine-tune.

**Q4: Why is rho more important for swaptions (options on interest rate swaps) than for equity options?**  
A: Swaption payoff depends on swap rate (long-term rate dynamics). Rho captures sensitivity to discount rates, which directly affect swap valuation. Equity option payoff depends on stock price (equity dynamics); rates enter only via discount factor and forward price (secondary effects). For swaptions: rho is first-order risk (core sensitivity); for equity options: rho is second-order (theta, gamma, vega dominate). Example: 1Y equity call rho ≈ $0.25 vs. 1Y swaption rho ≈ $50 (200× larger on same notional).

**Q5: Rates fall from 5% to 4% (-1%). Long equity calls gain from rho. Does the underlying stock also move?**  
A: Not directly specified; but empirically: rates fall → economy often stimulated → equity demand rises → stock rises (correlation typically positive). For option: (1) rho effect: rates -1% → call rho P&L ≈ -$0.30 × (-1%) = +$0.003 (gain from rho), (2) delta effect: stock rises 2% → call delta $0.50 × $2 = $1 (large gain from spot). Combined: call gains much more from spot move than from rho. Rho secondary. Cross-gamma risk: if rates-stock correlation negative (crisis scenario): rates spike → equity crash → both rho and delta losses (compounded).

**Q6: Explain the relationship between rho, bond futures prices, and hedging efficiency.**  
A: Bond futures price = (notional bond value) × (1 + duration × rate change). If rho negative (short rate risk), hedge by going long bond futures. Efficiency depends on: (1) duration match (bond duration ≈ rate sensitivity of portfolio), (2) correlation (rates and portfolio move together), (3) basis risk (futures-cash basis diverges). Practical: 10Y Treasury futures used for long-dated equity swaptions (duration ≈ 8 years). Position sizing: target delta-neutral in rate terms. Example: portfolio rho = $100/bp, bond future duration ≈ 8 years, contract value $100k → buy 1.25 contracts to hedge (100bp × $100/bp = $100bp exposure → need $100bp gain in bonds → $100k × 0.0125 = $1.25M adjusted notional).

## 7. Key References

- [Wikipedia: Rho (Finance)](https://en.wikipedia.org/wiki/Rho_(finance)) — Definition, interest rate sensitivity
- [Wikipedia: Greeks (Finance)](https://en.wikipedia.org/wiki/Greeks_(finance)) — Rho in context
- Hull: *Options, Futures & Derivatives* (Chapter 19) — Greeks, rho profile, interest rate hedging
- Tuckman: *Fixed Income Securities* (Chapter 8) — Duration, interest rate risk, hedging

**Status:** ✓ Standalone file. **Complements:** delta.md, gamma.md, vega.md, theta.md, delta_hedging_strategies.md, greeks_interactions.md
