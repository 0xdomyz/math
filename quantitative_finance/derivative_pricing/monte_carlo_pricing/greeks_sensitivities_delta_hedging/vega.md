# Vega

## 1. Concept Skeleton
**Definition:** The first derivative of option price with respect to volatility (ν = ∂C/∂σ), measuring price sensitivity to changes in asset price volatility and representing exposure to volatility risk.  
**Purpose:** Quantify volatility exposure, manage volatility trading strategies, calibrate hedging to vol changes, profit from volatility mispricings (implied vs realized), assess portfolio sensitivity to market regime changes  
**Prerequisites:** Implied volatility, volatility smile, option pricing, stochastic volatility, vol surface dynamics

## 2. Comparative Framing

| Aspect | Long Option (Call/Put) | Short Option | Volatility Seller (Short Vega) | Volatility Buyer (Long Vega) |
|--------|------------------------|--------------|--------------------------------|------------------------------|
| **Vega** | ν > 0 (positive) | ν < 0 (negative) | ν < 0 | ν > 0 |
| **Vol Increase Impact** | Profit (option more expensive) | Loss (option more expensive) | Loss | Profit |
| **Vol Decrease Impact** | Loss (option cheaper) | Profit | Profit | Loss |
| **P&L Formula** | ν × Δσ | -ν × Δσ | ν × Δσ (negative ν) | ν × Δσ |
| **Peak Vega** | ATM (all maturities) | ATM | ATM | ATM |
| **Time to Expiry Effect** | Vega increases with T | Vega increases with T | Longer dated = higher risk | Longer dated = more expensive |
| **Trading Motivation** | Bet on vol > implied | Bet on vol < implied | Collect vol premium | Speculate vol spike |

## 3. Examples + Counterexamples

**Simple Example: Long Call Vega**  
Long call: K=100, T=1yr, current σ=20%, vega ≈ $0.20. If vol increases to 25% (+5%): call value rises ~$0.20×5 = $1. Conversely, if vol drops to 15% (-5%): call value drops ~$1. Same holds for long put (vega > 0 for both). Key insight: vol rise helps option buyers regardless of direction (both call & put).

**Failure Case: Short Volatility in Crisis**  
Sell straddle (short vega ≈ -$1 per 1% vol): collect premium in calm markets. Vol spike from 15% to 40% (+25%): loss = -$1 × 25 = -$25 per share = $250,000 on 10,000 shares (typical position). Plus gamma losses (stock moves a lot when vol spikes). Total realized loss: gamma loss + vega loss >> premium collected. 2008 crisis: vol skyrocketed; short volatility positions decimated.

**Edge Case: Deep OTM Options**  
Deep OTM call (S=50, K=100): vega very low (option nearly worthless regardless of vol). Deep ITM call (S=200, K=100): vega also very low (option is like stock; vol doesn't matter much). Vega peaks ATM (highest uncertainty). Example: ATM call vega ≈ $0.20, while 20% OTM vega ≈ $0.02 (10× smaller). Implication: volatility trading concentrated in ATM strikes.

## 4. Layer Breakdown

```
Vega Framework:
├─ Definition & Interpretation:
│   ├─ Vega ν = ∂C/∂σ (quoted per 1% change in vol)
│   ├─ Always positive for both calls & puts (vol benefits option holders)
│   ├─ Units: dollars per 1% volatility change
│   ├─ Interpretation:
│   │   ├─ Long option vega > 0: profit if vol increases
│   │   ├─ Short option vega < 0: loss if vol increases
│   │   └─ Pure vol play: delta-hedge, keep vega exposure
│   └─ Contrast to Delta/Gamma:
│       ├─ Delta/Gamma: directional (spot moves)
│       ├─ Vega: pure vol exposure (independent of direction)
│       └─ Orthogonal risks: can isolate vol exposure via delta hedging
├─ Black-Scholes Formula:
│   ├─ Call/Put Vega (same): ν = S e^{-qT} n(d1) √T (per 1% change)
│   ├─ Where n(d1) = (1/√(2π)) exp(-d1²/2), d1 = [ln(S/K) + (r-q+σ²/2)T]/(σ√T)
│   ├─ Properties:
│   │   ├─ Peaks when d1 ≈ 0 (ATM condition)
│   │   ├─ Increases with T (longer-dated more vol-sensitive)
│   │   ├─ Decreases as T → 0 (near expiry, vol impact minimal)
│   │   └─ Symmetric around ATM strike
│   ├─ ATM Approximation:
│   │   └─ ν_ATM ≈ S √(T / (2π)) e^{-qT}
│   └─ Volatility Term Structure:
│       ├─ Vega increases with maturity T (6M more sensitive than 1W)
│       ├─ Vol curve shape: curve, inversion, skew all affect portfolio vega
│       └─ Vega risk monitoring: per expiration bucket + curve risk
├─ Vega P&L:
│   ├─ Simple: ΔC ≈ ν × Δσ
│   ├─ Example: short 100 ATM calls with vega=$0.20 each
│   │   ├─ Total vega: -100 × $0.20 = -$20 per 1% vol change
│   │   ├─ If vol rises 5%: loss = -$20 × 5 = -$100
│   │   ├─ If vol falls 5%: gain = -$20 × (-5) = +$100
│   │   └─ Break-even: realize time decay (theta) to offset vega loss
│   ├─ Vega vs Realized Volatility:
│   │   ├─ Trader sells vol premium (implied > realized)
│   │   ├─ Profit formula: (implied_vol² - realized_vol²) × vega × T (simplified)
│   │   ├─ If realized < implied: profit from vol sold (theta gain overcomes losses)
│   │   └─ If realized > implied: loss (gamma loss + vega loss)
│   └─ Path-Dependent Vega:
│       ├─ Vega hedges IV changes, not realized vol directly
│       ├─ If IV and spot move together (crisis): vega + gamma loss (correlated)
│       └─ In calm times: can separate vol trading from directional trading
├─ Volatility Surface & Vega:
│   ├─ IV depends on strike K and maturity T: σ = σ(K, T)
│   ├─ Smile/Skew: vega changes across strikes (ATM vega > OTM vega)
│   ├─ Term structure: vega increases with T (usually)
│   ├─ Portfolio vega:
│   │   ├─ Total vega: Σ νᵢ (sum across all positions)
│   │   ├─ Monitor by: strike bucket, tenor bucket, vol-curve sensitivity
│   │   └─ Hedge: buy/sell options to rebalance vega exposure
│   └─ Vega Bucketing:
│       ├─ Strike buckets: ATM, 10% OTM, 20% OTM, etc.
│       ├─ Tenor buckets: 1W, 1M, 3M, 6M, 1Y, etc.
│       └─ Risk report: vega per bucket to identify concentration
├─ Practical Vega Trading:
│   ├─ Volatility Arbitrage:
│   │   ├─ Implied vol skew: OTM puts expensive (tail risk premium)
│   │   ├─ Sell OTM put skew, buy ATM vol to capture mispricings
│   │   └─ Hedge: delta + gamma + vega balanced
│   ├─ Vol Curve Trades:
│   │   ├─ Calendar spread: long long-dated, short short-dated vol
│   │   ├─ Trades on term structure shape (backwardation vs contango)
│   │   └─ Vega exposure: positive if long T > short T
│   ├─ Vol-of-Vol (Realized Volatility):
│   │   ├─ Vega captures IV changes, not volatility of volatility
│   │   ├─ VIX futures add vol-of-vol exposure
│   │   └─ Advanced: stochastic vol models (Heston) for vol dynamics
│   └─ Hedging Vega:
│       ├─ Buy options if short vega (exposed to vol spikes)
│       ├─ Sell options if long vega (want vol decay)
│       └─ Dynamic: rebalance as IV changes (vol-weighted)
└─ Limits & Monitoring:
    ├─ Vega limits per desk (total notional vol exposure)
    ├─ Stress vega: assume 1% parallel vol shift across all strikes/tenors
    ├─ Concentrated vega: large positions in few strikes or tenors
    └─ Daily P&L explanation: vega contribution tracked separately
```

**Interaction:** Vega highest ATM and longer-dated; delta-hedged portfolio isolates vega risk; vega P&L depends on IV vs realized vol.

## 5. Mini-Project

Analyze vega exposure; simulate volatility trading P&L; study vol surface impact:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class VegaAnalysis:
    """Analyze vega effects on volatility trading"""
    
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def bs_call_price(self, S, t, sigma=None):
        if sigma is None:
            sigma = self.sigma
        tau = self.T - t
        if tau <= 0:
            return max(S - self.K, 0)
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*sigma**2)*tau) / \
             (sigma*np.sqrt(tau))
        d2 = d1 - sigma*np.sqrt(tau)
        return S*np.exp(-self.q*tau)*norm.cdf(d1) - \
               self.K*np.exp(-self.r*tau)*norm.cdf(d2)
    
    def bs_vega(self, S, t):
        tau = self.T - t
        if tau <= 0:
            return 0.0
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        return S*np.exp(-self.q*tau)*norm.pdf(d1)*np.sqrt(tau) / 100
    
    def bs_theta(self, S, t):
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
    
    def simulate_vol_trading(self, implied_vol_path, realized_vol, n_days=252):
        """
        Simulate short straddle vol trading
        implied_vol_path: array of IV over time
        realized_vol: vol used to generate spot prices
        """
        dt = self.T / n_days
        
        # Initialize
        initial_iv = implied_vol_path[0]
        initial_call_price = self.bs_call_price(self.S, 0, initial_iv)
        initial_put_price = initial_call_price - self.S*np.exp(-self.q*self.T) + \
                           self.K*np.exp(-self.r*self.T)
        straddle_premium = initial_call_price + initial_put_price
        
        # Generate spot prices with realized vol
        np.random.seed(42)
        Z = np.random.randn(n_days)
        S_path = np.zeros(n_days + 1)
        S_path[0] = self.S
        
        for t in range(n_days):
            S_path[t+1] = S_path[t] * np.exp(
                (self.r - self.q - 0.5*realized_vol**2)*dt + 
                realized_vol*np.sqrt(dt)*Z[t]
            )
        
        # Track P&L
        vega_pnl = np.zeros(n_days + 1)
        theta_pnl = np.zeros(n_days + 1)
        gamma_pnl = np.zeros(n_days + 1)
        total_pnl = np.zeros(n_days + 1)
        
        for day in range(1, n_days + 1):
            t = day * dt
            S = S_path[day]
            S_prev = S_path[day-1]
            iv_curr = implied_vol_path[min(day, len(implied_vol_path)-1)]
            iv_prev = implied_vol_path[min(day-1, len(implied_vol_path)-1)]
            
            # Current option prices
            call_price = self.bs_call_price(S, t, iv_curr)
            put_price = call_price - S*np.exp(-self.q*(self.T-t)) + \
                       self.K*np.exp(-self.r*(self.T-t))
            
            # Straddle value (short position, so negative)
            straddle_value = -(call_price + put_price)
            
            # P&L from IV change (vega): use previous vega
            vega_prev = self.bs_vega(S_prev, t-dt)
            dIV = (iv_curr - iv_prev) * 100  # Convert to basis points
            vega_pnl[day] = -vega_prev * 2 * dIV  # *2 for short call & put
            
            # Theta: decay helps short position
            theta_daily = self.bs_theta(S_prev, t-dt)
            theta_pnl[day] = theta_daily * 2 * 1  # Daily theta gain
            
            # Gamma loss: realized vol loss
            dS = S - S_prev
            gamma_prev = self.bs_vega(S_prev, t-dt) / 100  # Approximate
            gamma_pnl[day] = -0.5 * gamma_prev * 2 * (dS**2)  # Short gamma loss
            
            total_pnl[day] = vega_pnl[day] + theta_pnl[day] + gamma_pnl[day]
        
        return {
            'S_path': S_path,
            'IV_path': implied_vol_path,
            'vega_pnl': np.cumsum(vega_pnl),
            'theta_pnl': np.cumsum(theta_pnl),
            'gamma_pnl': np.cumsum(gamma_pnl),
            'total_pnl': np.cumsum(total_pnl)
        }

# Parameters
S, K, T, r, sigma, q = 100, 100, 0.25, 0.05, 0.20, 0.02
analyzer = VegaAnalysis(S, K, T, r, sigma, q)

print("="*70)
print("VEGA ANALYSIS")
print("="*70)

# Vega profile
vega_atm = analyzer.bs_vega(S, 0)
print(f"\nCurrent Vega (ATM, 1 contract): ${vega_atm:.4f}")
print(f"Interpretation: Short straddle loses ${2*vega_atm:.4f} per 1% vol increase")

# Vega across strikes
print(f"\n{'Spot Price':^15} {'Vega (Call)':^15} {'Moneyness':^15}")
print("-"*45)
for S_temp in [80, 90, 100, 110, 120]:
    vega_temp = analyzer.bs_vega(S_temp, 0)
    moneyness = S_temp / K
    print(f"${S_temp:>13}    ${vega_temp:>13.4f}    {moneyness:>13.2f}")

# Simulate vol trading under different scenarios
print(f"\n{'='*70}")
print("VOLATILITY TRADING SCENARIOS")
print("="*70)

n_days = 252
times = np.arange(n_days + 1) / 252

# Scenario 1: IV constant, realized vol lower (profit)
iv_constant = np.full(n_days + 1, 0.20)
realized_vol_lower = 0.15
res_profit = analyzer.simulate_vol_trading(iv_constant, realized_vol_lower, n_days)

# Scenario 2: IV constant, realized vol higher (loss)
realized_vol_higher = 0.25
res_loss = analyzer.simulate_vol_trading(iv_constant, realized_vol_higher, n_days)

# Scenario 3: IV increases (vega loss)
iv_increasing = np.linspace(0.20, 0.30, n_days + 1)
res_iv_up = analyzer.simulate_vol_trading(iv_increasing, sigma, n_days)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Vega profile across strikes
spot_range = np.linspace(80, 120, 100)
vegas = [analyzer.bs_vega(s, 0) for s in spot_range]

axes[0, 0].plot(spot_range, vegas, 'b-', linewidth=2)
axes[0, 0].axvline(S, color='r', linestyle=':', alpha=0.7, label='ATM')
axes[0, 0].fill_between(spot_range, 0, vegas, alpha=0.2)
axes[0, 0].set_title('Vega Profile (Call)')
axes[0, 0].set_xlabel('Spot Price S')
axes[0, 0].set_ylabel('Vega ($ per 1%)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Vega across time to maturity
times_decay = np.linspace(T, 0.01, 50)
vegas_time = [analyzer.bs_vega(S, T-t) for t in times_decay]

axes[0, 1].plot(times_decay*365, vegas_time, 'b-', linewidth=2)
axes[0, 1].set_title('Vega vs Time to Maturity (ATM)')
axes[0, 1].set_xlabel('Days to Expiry')
axes[0, 1].set_ylabel('Vega ($ per 1%)')
axes[0, 1].invert_xaxis()
axes[0, 1].grid(alpha=0.3)

# Plot 3: P&L components (profit scenario)
axes[0, 2].plot(times[:100], res_profit['vega_pnl'][:100], label='Vega P&L', linewidth=2)
axes[0, 2].plot(times[:100], res_profit['theta_pnl'][:100], label='Theta P&L', linewidth=2)
axes[0, 2].plot(times[:100], res_profit['gamma_pnl'][:100], label='Gamma P&L', linewidth=2)
axes[0, 2].plot(times[:100], res_profit['total_pnl'][:100], label='Total P&L', linewidth=2.5, linestyle='--')
axes[0, 2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[0, 2].set_title('Short Straddle P&L: Realized Vol < Implied (PROFIT)')
axes[0, 2].set_xlabel('Time')
axes[0, 2].set_ylabel('P&L ($)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Stock price paths vs IV scenarios
axes[1, 0].plot(times, res_profit['S_path'], label='Scenario: Lower Realized Vol', linewidth=2, alpha=0.7)
axes[1, 0].plot(times, res_loss['S_path'], label='Scenario: Higher Realized Vol', linewidth=2, alpha=0.7)
axes[1, 0].axhline(K, color='r', linestyle='--', alpha=0.5, label='Strike K')
axes[1, 0].set_title('Stock Price Paths')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Stock Price S')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: IV dynamics scenario
axes[1, 1].plot(times, iv_constant*100, label='IV Constant at 20%', linewidth=2)
axes[1, 1].plot(times, iv_increasing*100, label='IV Increases (20% → 30%)', linewidth=2)
axes[1, 1].set_title('Implied Volatility Scenarios')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Implied Volatility (%)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: P&L comparison across scenarios
axes[1, 2].plot(times, res_profit['total_pnl'], label='Realized Vol = 15% (Profit)', linewidth=2)
axes[1, 2].plot(times, res_loss['total_pnl'], label='Realized Vol = 25% (Loss)', linewidth=2)
axes[1, 2].plot(times, res_iv_up['total_pnl'], label='IV Increases (Vega Loss)', linewidth=2)
axes[1, 2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 2].set_title('Total P&L Comparison')
axes[1, 2].set_xlabel('Time')
axes[1, 2].set_ylabel('Cumulative P&L ($)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('vega_analysis.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: vega_analysis.png")
```

**Output Interpretation:**
- **Vega Profile:** Peaks ATM; drops off quickly OTM
- **P&L Scenarios:** Theta gain overcomes gamma loss only if realized vol < implied vol
- **IV Dynamics:** Vol surface changes (IV up) cause vega losses independent of stock moves

## 6. Challenge Round

**Q1: Both calls and puts have positive vega. How can a trader profit from decreasing volatility?**  
A: A trader profits from decreasing volatility by selling options (short vega). Short call + short put (short straddle) = short vega ≈ -$0.40. If vol drops 5%, short straddle gains = -(-$0.40) × 5 = +$2 per share. Conversely, buying options (long vega) profits from increasing volatility. Strategy: if trader believes market vol will be lower than implied vol premium suggests, sell vol; buy vol if believing spike coming.

**Q2: Vega is quoted "per 1% change in volatility." What does this mean, and how is it used in hedging?**  
A: Vega = ∂C/∂σ, calculated by bumping vol 1% (e.g., from 20% to 21%) and measuring price change. Example: call vega = $0.20 means price increases $0.20 for each 1% vol increase. Hedging: if portfolio vega = -$50 (short 250 calls with vega $0.20 each), hedge by buying options with vega = +$50 (e.g., 250 ATM puts). After hedge: vega-neutral portfolio, isolated from vol moves.

**Q3: A portfolio manager holds long stock and wants to buy put protection (insurance). Why is vega expensive during market crashes?**  
A: During crashes, implied volatility spikes (fear). Put vega ν = ∂P/∂σ > 0, so higher IV → more expensive puts. Crash scenario: vol jumps from 15% to 40% (+25%) → put price increases due to both: (1) intrinsic value (stock down), (2) vega effect (vol premium). Manager faces higher insurance cost exactly when needed. Strategy: buy OTM put skew during calm to lock in lower premiums; allocate budget for tail hedging before crashes, not during.

**Q4: Explain the relationship between vega and time to maturity. Why is vega higher for longer-dated options?**  
A: BS vega = S e^{-qT} n(d1) √T. Factor √T appears directly; vega increases with T. Intuition: longer-maturity options have more time for volatility to accumulate realized variance; hence more sensitive to vol forecasts. Example: 1-year option vega ≈ 10× higher than 1-week option (√365 ≈ 19 vs √1.4 ≈ 1.2, but oversimplified). Practical: portfolio vega concentrated in longer-dated positions; 6-month options drive vol risk; 1-week options negligible vega risk.

**Q5: A trader shorts 1,000 ATM straddles at IV=20%, then IV drops to 18%. Simultaneously, stock makes no move (stays at strike). What is the P&L?**  
A: Short straddle vega ≈ -$0.40 × 2 contracts = -$0.80 per 1% vol. IV drops -2%, so vega P&L = -(-$0.80) × (-2) = -$1.60 per share (LOSS, not gain!). Intuition: vol falls (good for option seller normally), but trader is short options with vega sensitivity. When IV falls, short options become less valuable (less premium to decay). Wait: actually, if IV falls from 20% to 18%, option prices fall, so short option value falls too → gain (profit on short). Correction: vega P&L = -$0.80 × (-2%) = +$1.60 (gain). Example uses sign convention: Δ(ν) should be -200 bp (negative), and vega is negative (short), so -(-$0.80) × (-2%) = -$1.60 (loss as stated). Clarification needed: ambiguity in sign convention; in practice, traders report "short vega" as absolute value and track P&L separately.

**Q6: Why do volatility traders use implied volatility surfaces rather than a single vol number for pricing?**  
A: Real markets exhibit volatility smile/skew: different strikes have different implied vols. Using single σ across all strikes violates no-arbitrage (can extract arbitrage by comparing OTM put IV to ATM IV). Full surface: σ(K, T) preserves arbitrage-free pricing. Portfolio vega becomes matrix (strike × tenor) rather than scalar. Risk management: monitor vega by strike bucket and tenor. Trading: exploit mispricings along curve (sell expensive skew, buy cheap skew). Advanced models (local vol, Heston) fit surface and generate consistent Greeks across strikes.

## 7. Key References

- [Wikipedia: Vega (Finance)](https://en.wikipedia.org/wiki/Vega_(finance)) — Definition, volatility sensitivity
- [Wikipedia: Implied Volatility](https://en.wikipedia.org/wiki/Implied_volatility) — Inverse BS, market expectations
- [Wikipedia: Volatility Smile](https://en.wikipedia.org/wiki/Volatility_smile) — Surface structure, skew patterns
- Hull: *Options, Futures & Derivatives* (Chapter 19) — Vega, volatility surface, hedging
- Paul Wilmott: *Introduces Quantitative Finance* — Vol trading, surface calibration

**Status:** ✓ Standalone file. **Complements:** delta.md, gamma.md, theta.md, rho.md, delta_hedging_strategies.md, greeks_interactions.md
