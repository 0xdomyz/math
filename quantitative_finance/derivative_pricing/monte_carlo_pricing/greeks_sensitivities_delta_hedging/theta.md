# Theta

## 1. Concept Skeleton
**Definition:** The first derivative of option price with respect to time (θ = ∂C/∂t or -∂C/∂τ where τ is time-to-maturity), measuring the rate of decay of option value as expiration approaches.  
**Purpose:** Quantify time decay effects, optimize holding periods for options, profit from time decay without market moves, measure daily P&L from calendar effects, balance theta income vs gamma loss in hedged portfolios  
**Prerequisites:** Option pricing, time value concepts, extrinsic value decay, option strategies, portfolio rebalancing

## 2. Comparative Framing

| Aspect | Long Call/Put | Short Call/Put | Long Straddle | Short Straddle | Out-of-Money | At-the-Money | In-the-Money |
|--------|---------------|----------------|---------------|----------------|--------------|--------------|--------------|
| **Theta** | θ < 0 (decay) | θ > 0 (benefit) | θ < 0 | θ > 0 | Lower \|θ\| | Peak \|θ\| | Lower \|θ\| |
| **Time Value** | Decreases daily | Benefits daily | Both decay | Benefits | Mostly time value | All time value | Some intrinsic |
| **Profit Motive** | Directional bet | Collect decay | Vol bet | Vol collection | Quick expiry | Extended hold | Dividend capture |
| **Rebalancing Need** | Frequent (gamma) | Frequent (gamma) | High (gamma swing) | High | Low | High (ATM sensitivity) | Low |
| **Breakeven Move** | Large | Small (premium) | Large (collect decay) | Small | Likely out-of-money | 50/50 chance | Likely in-money |

## 3. Examples + Counterexamples

**Simple Example: Theta Decay**  
Long call: K=100, S=100, T=1 day, σ=20%, r=5%, q=0. BS price ≈ $0.60 (mostly time value). Next day (T → 0): price → max(S - K, 0) = $0 if S ≤ K (entire $0.60 premium lost to theta). Even if S stays at 100, option worthless; theta = -$0.60/day. Loss crystallized purely from time passage.

**Failure Case: Long Straddle with Theta Drag**  
Buy straddle (call + put): K=100, T=1month, premium = $5. Plan: "market will move 15%, I profit from vol." But theta = -$0.05/day (drag). Over 20 days at no move: theta loss = $1. Market must move enough to overcome theta decay + rehedging costs. If move is delayed until day 25, time value even lower → need 20%+ move to profit (was 15% needed if done day-1).

**Edge Case: Theta Inversion**  
OTM call deep: S=80, K=100, T=1day. Theta minimal (option likely worthless, small decay). ITM call deep: S=120, K=100, T=1day. Theta also small (option is stock-like, time value minimal). Peak theta: ATM with ~30 days left → balance between rapid expiry and meaningful time value.

## 4. Layer Breakdown

```
Theta Framework:
├─ Definition & Interpretation:
│   ├─ Theta θ = -∂V/∂τ where τ = T - t (time to maturity)
│   ├─ Often quoted as θ = ∂V/∂t (positive for time moving forward)
│   ├─ Units: dollars per day (or per year, then divide by 365)
│   ├─ Intuition:
│   │   ├─ Long option: θ < 0 (option decays, lose money daily)
│   │   ├─ Short option: θ > 0 (collect decay, gain daily)
│   │   └─ Zero-move benchmark: profit/loss from time alone
│   ├─ Theta vs Intrinsic Value:
│   │   ├─ Time value = option price - intrinsic value
│   │   ├─ Theta ≈ daily time value decay (approximately)
│   │   └─ OTM options: 100% time value, all theta decay
│   └─ Theta Profile:
│       ├─ OTM: low absolute theta (small time value)
│       ├─ ATM: high absolute theta (maximum time value, rapid decay)
│       ├─ ITM: moderate theta (diminishing time value)
│       └─ Accelerates near expiry (√T effect amplifies theta)
├─ Black-Scholes Theta (Call):
│   ├─ θ_call = -S σ e^{-qT} n(d1) / (2√T) + r K e^{-rT} N(d2) - q S e^{-qT} N(d1)
│   ├─ Three terms:
│   │   ├─ First term: time decay of volatility (always negative)
│   │   ├─ Second term: discount rate effect (positive, future payment discounted)
│   │   └─ Third term: dividend drag (negative if q > 0)
│   ├─ Put Theta (different sign on second/third terms):
│   │   ├─ θ_put = -S σ e^{-qT} n(d1) / (2√T) - r K e^{-rT} N(-d2) + q S e^{-qT} N(-d1)
│   │   └─ Often positive (put values increase as T decreases in certain scenarios)
│   └─ Properties:
│       ├─ Theta ∝ -1/√T (accelerates as T → 0)
│       ├─ Peak at ATM; wings (OTM/ITM) have lower absolute theta
│       ├─ Near expiry (T → 0): theta → -∞ for OTM options
│       └─ Deep ITM/OTM: theta → 0 (option intrinsic, no time decay)
├─ Theta P&L:
│   ├─ Daily P&L from theta: θ × 1 day (simplified)
│   ├─ Long option holding: lose θ daily (if theta negative)
│   ├─ Short option position: gain θ daily
│   ├─ Portfolio theta:
│   │   ├─ Θ_portfolio = Σ θᵢ × qᵢ (sum across all positions)
│   │   ├─ Positive: collect decay daily
│   │   └─ Negative: bleed from decay daily
│   ├─ Theta vs Other Greeks:
│   │   ├─ Delta-hedged: Δ ≈ 0, but Θ, Γ still active
│   │   ├─ P&L from no-move scenario: primarily theta + gamma loss
│   │   ├─ Theta gain offset by gamma loss in rehedging
│   │   └─ Profitable if theta > gamma loss (time decay > rehedging costs)
│   └─ Theta in Strategies:
│       ├─ Calendar spread (long long-dated, short short-dated): long theta decay slope
│       ├─ Theta decay in spreads: max profit as time passes, spreads tighten
│       ├─ Theta harvesting: collect decay without directional moves (covered calls)
│       └─ Theta acceleration: near expiry, decay rate increases dramatically
├─ Practical Theta Trading:
│   ├─ Covered Call Strategy:
│   │   ├─ Own stock, sell call: collect theta from call decay
│   │   ├─ Stock move up: cap profit at call strike; theta gain partially offset
│   │   ├─ Stock move down: theta gain may not offset stock loss (bearish scenario)
│   │   └─ Scenario best: sideways market; collect premium, stock stable
│   ├─ Theta in Calendar Spreads:
│   │   ├─ Long near-term, short far-term same strike (butterfly variation)
│   │   ├─ Net theta: profit from near-term decay faster than far-term
│   │   ├─ Requires frequent rebalancing to maintain theta benefit
│   │   └─ Concentrated risk: max loss if price gaps far from strike
│   ├─ Theta Monitoring:
│   │   ├─ Track Θ_portfolio daily
│   │   ├─ Stress: assume 1 day passes, all else held (market impact removed)
│   │   ├─ Target: positive Θ for income, negative for upside bets
│   │   └─ Limits: cap daily theta collection (risk/reward alignment)
│   └─ Theta Decay Curve:
│       ├─ Non-linear acceleration: decay accelerates near expiry
│       ├─ 60 days out: θ = -$0.01/day
│       ├─ 10 days out: θ = -$0.10/day (10× faster)
│       ├─ 1 day out: θ = -$0.50+/day (50× faster) for ATM options
│       └─ Implication: timing of holding period critical for theta harvesting
└─ Theta vs Market Reality:
    ├─ Black-Scholes assumes no trading costs, continuous rebalancing
    ├─ Real theta eroded by bid-ask spread (especially near expiry)
    ├─ Gamma loss can exceed theta gain on volatile days
    ├─ Theta only collected if position held to expiry or close
    └─ Exit before expiry: theta benefit incomplete if forced to sell early
```

**Interaction:** Theta benefits short positions (collect decay); conflicts with gamma (rehedging losses outweigh theta on large moves); interacts with dividends and rates.

## 5. Mini-Project

Analyze theta decay; simulate covered call strategy; compare theta profiles:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class ThetaAnalysis:
    """Analyze theta (time decay) effects"""
    
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def bs_call_price(self, S, t):
        tau = self.T - t
        if tau <= 0:
            return max(S - self.K, 0)
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        return S*np.exp(-self.q*tau)*norm.cdf(d1) - \
               self.K*np.exp(-self.r*tau)*norm.cdf(d2)
    
    def bs_theta_call(self, S, t):
        """Call theta, per day"""
        tau = self.T - t
        if tau <= 0:
            return 0.0
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        theta_annual = -S*np.exp(-self.q*tau)*norm.pdf(d1)*self.sigma/(2*np.sqrt(tau)) + \
                      self.q*S*np.exp(-self.q*tau)*norm.cdf(d1) - \
                      self.r*self.K*np.exp(-self.r*tau)*norm.cdf(d2)
        return theta_annual / 365
    
    def bs_theta_put(self, S, t):
        """Put theta, per day"""
        tau = self.T - t
        if tau <= 0:
            return 0.0
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        theta_annual = -S*np.exp(-self.q*tau)*norm.pdf(d1)*self.sigma/(2*np.sqrt(tau)) - \
                      self.q*S*np.exp(-self.q*tau)*norm.cdf(-d1) + \
                      self.r*self.K*np.exp(-self.r*tau)*norm.cdf(-d2)
        return theta_annual / 365

# Parameters
S, K, T, r, sigma, q = 100, 100, 0.25, 0.05, 0.20, 0.02
analyzer = ThetaAnalysis(S, K, T, r, sigma, q)

print("="*70)
print("THETA ANALYSIS")
print("="*70)

# Current theta at ATM
theta_call_atm = analyzer.bs_theta_call(S, 0)
theta_put_atm = analyzer.bs_theta_put(S, 0)
print(f"\nCurrent Theta (ATM, per day):")
print(f"  Call: ${theta_call_atm:.4f}")
print(f"  Put: ${theta_put_atm:.4f}")
print(f"  Straddle: ${theta_call_atm + theta_put_atm:.4f}")

# Theta decay profile over time
times_decay = np.linspace(T, 0.01, 50)
thetas_call = [analyzer.bs_theta_call(S, T-t) for t in times_decay]
thetas_put = [analyzer.bs_theta_put(S, T-t) for t in times_decay]
days_left = times_decay * 365

# Theta acceleration table
print(f"\n{'Days to Expiry':>15} {'Call Theta':>15} {'Put Theta':>15} {'Straddle Theta':>15}")
print("-"*60)
for days, t in zip([90, 30, 10, 5, 1], [0.25, 30/365, 10/365, 5/365, 1/365]):
    theta_c = analyzer.bs_theta_call(S, T-t)
    theta_p = analyzer.bs_theta_put(S, T-t)
    print(f"{days:>15}    ${theta_c:>13.4f}    ${theta_p:>13.4f}    ${theta_c + theta_p:>13.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Call option price across time (no move)
times_hold = np.linspace(0, T, 90)
call_prices = [analyzer.bs_call_price(S, t) for t in times_hold]

axes[0, 0].plot(times_hold*365, call_prices, 'b-', linewidth=2)
axes[0, 0].fill_between(times_hold*365, 0, call_prices, alpha=0.2)
axes[0, 0].set_title('Call Price Decay (No Stock Move)')
axes[0, 0].set_xlabel('Days (holding period)')
axes[0, 0].set_ylabel('Call Value')
axes[0, 0].invert_xaxis()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Theta profile across spot prices (ATM peak)
spot_range = np.linspace(80, 120, 100)
thetas_spot = [analyzer.bs_theta_call(s, 0) for s in spot_range]

axes[0, 1].plot(spot_range, thetas_spot, 'b-', linewidth=2)
axes[0, 1].axvline(S, color='r', linestyle=':', alpha=0.7, label='ATM')
axes[0, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[0, 1].fill_between(spot_range, 0, thetas_spot, where=np.array(thetas_spot)<0, alpha=0.2, color='red', label='Negative theta (long loses)')
axes[0, 1].set_title('Call Theta Profile (Across Spot)')
axes[0, 1].set_xlabel('Spot Price S')
axes[0, 1].set_ylabel('Theta ($/day)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Theta acceleration (non-linear time decay)
axes[0, 2].plot(days_left, thetas_call, 'b-', linewidth=2, label='Call Theta')
axes[0, 2].plot(days_left, thetas_put, 'r-', linewidth=2, label='Put Theta')
axes[0, 2].set_title('Theta Acceleration Near Expiry (ATM)')
axes[0, 2].set_xlabel('Days to Expiry')
axes[0, 2].set_ylabel('Theta ($/day)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Covered call strategy P&L
spot_final_range = np.linspace(80, 120, 100)
call_payoff = np.maximum(spot_final_range - K, 0)
stock_pnl = spot_final_range - S
covered_call_pnl = stock_pnl - (analyzer.bs_call_price(S, 0) - call_payoff)

axes[1, 0].plot(spot_final_range, stock_pnl, 'g--', linewidth=2, label='Long Stock')
axes[1, 0].plot(spot_final_range, covered_call_pnl, 'b-', linewidth=2, label='Covered Call')
axes[1, 0].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 0].fill_between(spot_final_range, stock_pnl, covered_call_pnl, alpha=0.2, label='Theta Gain / Cap')
axes[1, 0].set_title('Covered Call Strategy (Stock + Short Call)')
axes[1, 0].set_xlabel('Stock Price at Expiry')
axes[1, 0].set_ylabel('P&L ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Long straddle theta drag
times_hold = np.linspace(0, T, 90)
straddle_theta_drag = []
for t in times_hold:
    theta_drift = (analyzer.bs_theta_call(S, t) + analyzer.bs_theta_put(S, t)) * (T - t) * 365
    straddle_theta_drag.append(theta_drift)

axes[1, 1].plot(times_hold*365, straddle_theta_drag, 'r-', linewidth=2)
axes[1, 1].fill_between(times_hold*365, 0, straddle_theta_drag, alpha=0.2, color='red')
axes[1, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 1].set_title('Long Straddle Theta Drag')
axes[1, 1].set_xlabel('Holding Period (days)')
axes[1, 1].set_ylabel('Cumulative Theta Loss ($)')
axes[1, 1].invert_xaxis()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Theta term structure (across expirations)
tenors = np.array([7, 14, 30, 60, 90, 180, 365]) / 365
thetas_tenor = []
for tenor in tenors:
    theta_temp = ThetaAnalysis(S, K, tenor, r, sigma, q)
    thetas_tenor.append(theta_temp.bs_theta_call(S, 0))

axes[1, 2].plot(tenors*365, thetas_tenor, 'bo-', linewidth=2, markersize=8)
axes[1, 2].set_title('Theta Term Structure (ATM)')
axes[1, 2].set_xlabel('Days to Expiry')
axes[1, 2].set_ylabel('Theta ($/day)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('theta_analysis.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: theta_analysis.png")
```

**Output Interpretation:**
- **Price Decay:** Option value decreases as T → 0 (concave curve, acceleration near end)
- **Theta Peak:** ATM has maximum theta; wings negligible
- **Acceleration:** Daily decay accelerates non-linearly as expiry approaches

## 6. Challenge Round

**Q1: Call theta is typically negative, but put theta can be positive. Why does this asymmetry exist?**  
A: Call theta formula: θ_C = -S σ e^{-qT} n(d1) / (2√T) + r K e^{-rT} N(d2) - q S e^{-qT} N(d1). First term (vol decay) always negative. Second term (discount rate benefit): positive for calls (K paid in future, discounted benefit increases as T shrinks). Third term (dividend drag): negative if q > 0. For ATM calls, vol decay dominates; for OTM calls, discount rate effect negligible. For puts, all three terms often balance; put theta often positive (especially for OTM puts) because discount rate effect benefits payers (sellers receive K now in put-put-forward).

**Q2: A trader buys a call at θ = -$0.05/day and holds for 30 days with no stock move. What is the loss?**  
A: Simple approximation: loss ≈ -$0.05 × 30 = -$1.50 per share. Example: buying 100 call contracts (10,000 shares notional): loss ≈ $15,000 from pure time decay. Reality: theta is non-linear (accelerates near end), so actual loss > simple calculation. More precise: cumulative = Σ θᵢ over path, not linear. If held all 30 days until expiry ATM: further loss near end as theta acceleration kicks in. Profit breakeven: stock must move enough to create intrinsic value > theta loss. Theta creates urgency for traders holding long options; encourages closing position before full decay.

**Q3: A short straddle position has positive theta. Why are traders not wealthy collecting theta?**  
A: Theta is collected only if market stays near strike; the moment market moves, gamma loss exceeds theta gain. Example: short straddle, collect $0.05/day theta. Stock moves $5 → gamma loss ≈ 0.5 × Γ × $25 (where Γ ≈ 0.02) ≈ $0.25 (5 days of theta collected in one move). Plus: rehedging costs, bid-ask spreads, vega losses if vol spikes during moves. Traders must believe: realized vol < implied vol, so theta gain accumulates faster than gamma loss realizes. If wrong, theta collection becomes theta trap.

**Q4: Why does theta acceleration near expiry create "gamma explosion"?**  
A: Gamma Γ ∝ 1/√T and Theta θ ∝ -1/√T. As T → 0, both explode. Practical: at expiry, delta jumps discontinuously (from 0 to 1 for ITM, stays 0 for OTM). Theta acceleration means option value drops steeply on final days. Gamma explosion means delta super-sensitive to tiny spot moves (delta can swing 0.50 in $1 move on expiry day, vs. delta swing 0.01 in $1 move a month prior). This creates: (1) hedging difficulty (can't rehedge enough, too expensive), (2) gap risk (overnight move, can't execute hedge), (3) liquidity crisis (bid-ask explodes on expiry).

**Q5: How do dividends affect theta? Why does a stock paying high dividend have different call/put theta?**  
A: Call theta includes dividend drag term: -q S e^{-qT} N(d1) < 0. Higher q → more negative call theta. Intuition: dividend reduces forward price → call owner loses expected future value. Put theta benefits from dividend (S drops ex-dividend, ITM more likely) → put theta less negative or even positive with high q. Practical: high-dividend stocks (q=5%) have much faster call decay than low-dividend stocks; investors prefer puts (not calls) in high-q environments.

**Q6: A calendar spread (long 1-month, short 1-week, same strike) is long theta. Explain the P&L.**  
A: Short 1-week theta ≈ -$0.10/day (decays rapidly). Long 1-month theta ≈ -$0.03/day (slower decay). Net theta: short ≈ -$0.07/day (initially short, profile inverts as weeks pass). But 1-week expires first → short leg falls off, leaving long 1-month naked (now long theta if market moves). P&L: (1) first week: collect |−$0.07|=+$0.07/day theta (positive), (2) post-week-1: rebalance or close short leg, recalibrate. Max profit: same strike, both legs expire simultaneously (rare). Gamma risk: asymmetric; long-dated leg has low gamma, short-dated has high gamma → rehedging needed for directional moves.

## 7. Key References

- [Wikipedia: Theta (Finance)](https://en.wikipedia.org/wiki/Theta_(finance)) — Time decay definition, formulas
- [Wikipedia: Greeks (Finance)](https://en.wikipedia.org/wiki/Greeks_(finance)) — Theta in context
- Hull: *Options, Futures & Derivatives* (Chapter 19) — Greeks, theta profile, strategies
- Paul Wilmott: *Introduces Quantitative Finance* — Theta trading, time decay P&L

**Status:** ✓ Standalone file. **Complements:** delta.md, gamma.md, vega.md, rho.md, delta_hedging_strategies.md, greeks_interactions.md
