# Vanilla Options

## 1. Concept Skeleton
**Definition:** Standard European and American call and put options with simple linear payoff functions, representing the most liquid and widely-traded derivative instruments in global markets.  
**Purpose:** Hedge price risk, speculate on directional moves, implement spread strategies, foundational building blocks for structured products, primary benchmarks for model validation  
**Prerequisites:** Option payoff functions, moneyness concepts, call-put parity, exercise styles, market conventions

## 2. Comparative Framing

| Characteristic | European Call | European Put | American Call | American Put |
|---|---|---|---|---|
| **Exercise Timing** | At T only | At T only | Any time ≤ T | Any time ≤ T |
| **Payoff** | max(S_T - K, 0) | max(K - S_T, 0) | Same as European | Same as European |
| **Value** | Lower | Lower | ≥ European | > European |
| **Pricing** | Black-Scholes | Black-Scholes | Binomial, LSM | Binomial, LSM |
| **Early Exercise** | N/A | N/A | Rare (divs high) | Common (deep ITM) |
| **Liquidity** | Very high (exchanges) | Very high (exchanges) | Lower (OTC) | Lower (OTC) |
| **Bid-Ask Spread** | $0.01-0.05 | $0.01-0.05 | $0.05-0.20 | $0.05-0.20 |
| **Market Size** | Trillions | Trillions | Billions | Billions |

## 3. Examples + Counterexamples

**Simple Example: Long Call Strategy**  
Buy call: K=100, premium C=$5. Scenarios at maturity:
- S_T = 90: Payoff = 0, Loss = -$5 (premium)
- S_T = 105: Payoff = 5, P&L = 0 (breakeven)
- S_T = 120: Payoff = 20, P&L = +$15 (profit)

**Failure Case: Selling Naked Calls**  
Sell call: K=100, receive $5 premium. Obligated to deliver stock at 100 even if S_T > 150. Unlimited loss potential. Downside: margin requirement, forced liquidation on gap moves. Practical: covered calls (own stock) reduce risk but cap upside.

**Edge Case: At-the-Money (ATM) Options**  
S = K: Both call and put have maximum time value (peak theta decay). ATM options are most sensitive to volatility changes (gamma highest here). Traders use ATM options for volatility plays; selling ATM strangles (sell call + put) profits if realized vol < implied vol.

## 4. Layer Breakdown

```
Vanilla Options Framework:
├─ Market Structure:
│   ├─ Organized Exchanges (listed):
│   │   ├─ Equity options (CBOE, Eurex): standardized contracts
│   │   ├─ Strike intervals: $0.50-$2.50 (ATM), wider OTM
│   │   ├─ Expiration: monthly (3rd Fri), weekly, daily
│   │   ├─ High liquidity, tight spreads, exchange guarantee
│   │   └─ Regulation: SEC oversight, margin requirements
│   ├─ Over-the-Counter (OTC):
│   │   ├─ Bilateral agreements, customizable K and T
│   │   ├─ Lower liquidity, wider spreads ($0.10-$1.00+)
│   │   ├─ Counterparty risk (mitigated by collateral/CCPs)
│   │   └─ Larger notional size ($1M-$100M+ typical)
│   └─ Implied Volatility:
│       ├─ Extracted from market prices via BS inverse
│       ├─ Volatility smile/skew: IV varies by K
│       ├─ Term structure: IV varies by T (curve shape)
│       └─ Market expectation of future price uncertainty
├─ Payoff Structures:
│   ├─ Call Payoff: max(S_T - K, 0)
│   │   ├─ Intrinsic: max(S_T - K, 0) at T
│   │   ├─ Time Value: C_t - Intrinsic (decays as T → 0)
│   │   └─ P&L per unit: Payoff - Premium paid
│   ├─ Put Payoff: max(K - S_T, 0)
│   │   ├─ Intrinsic: max(K - S_T, 0) at T
│   │   ├─ Time Value: P_t - Intrinsic
│   │   └─ P&L per unit: Payoff - Premium paid
│   └─ Combinations:
│       ├─ Call spread (buy call K1, sell K2): capped profit
│       ├─ Put spread: capped profit, limited loss
│       ├─ Straddle (buy call + put at K): long volatility
│       ├─ Strangle (buy OTM call + OTM put): lower cost
│       └─ Collar (buy OTM put, sell OTM call): hedge cost
├─ Moneyness & Regimes:
│   ├─ In-the-Money (ITM):
│   │   ├─ Call: S > K (intrinsic value positive)
│   │   ├─ Put: S < K (intrinsic value positive)
│   │   └─ Behavior: approaching stock-like (call delta → 1)
│   ├─ At-the-Money (ATM):
│   │   ├─ S ≈ K (highest uncertainty, max theta decay)
│   │   ├─ Delta ≈ 0.5 (call), ≈ -0.5 (put)
│   │   └─ Gamma, Vega peak (maximum sensitivity)
│   ├─ Out-of-the-Money (OTM):
│   │   ├─ Call: S < K (no intrinsic, pure time value)
│   │   ├─ Put: S > K (no intrinsic, pure time value)
│   │   └─ Behavior: approaching worthless (delta → 0)
│   └─ Deep ITM/OTM: High gamma risk (sudden delta shifts)
└─ Market Mechanics:
    ├─ Bid-Ask Spread: $0.01-0.05 (equity, ATM); wider OTM
    ├─ Implied Vol Surface: K × T grid of IV values
    ├─ Greeks Impact: delta-hedging, gamma P&L, vega rebalancing
    ├─ Settlement: T+0 (cash) or T+1 (physical delivery)
    └─ Margin: Portfolio margin, SPAN (for spreads)
```

**Interaction:** Vanilla option value driven by (S, K, T, σ, r, q); Greeks describe sensitivities; market prices determine implied vol surface.

## 5. Mini-Project

Analyze vanilla options across strikes and strategies; price, Greeks, portfolio construction:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fminbound

class VanillaOption:
    """European vanilla call/put pricing and Greeks"""
    
    def __init__(self, S, K, T, r, sigma, q=0, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type
    
    def d1_d2(self):
        d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        return d1, d2
    
    def price(self):
        d1, d2 = self.d1_d2()
        if self.option_type == 'call':
            return self.S*np.exp(-self.q*self.T)*norm.cdf(d1) - \
                   self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - \
                   self.S*np.exp(-self.q*self.T)*norm.cdf(-d1)
    
    def intrinsic(self):
        """Intrinsic value (exercise immediately)"""
        if self.option_type == 'call':
            return max(self.S - self.K, 0)
        else:
            return max(self.K - self.S, 0)
    
    def time_value(self):
        """Time value = option price - intrinsic value"""
        return self.price() - self.intrinsic()
    
    def delta(self):
        d1, _ = self.d1_d2()
        if self.option_type == 'call':
            return np.exp(-self.q*self.T)*norm.cdf(d1)
        else:
            return -np.exp(-self.q*self.T)*norm.cdf(-d1)
    
    def gamma(self):
        d1, _ = self.d1_d2()
        return np.exp(-self.q*self.T)*norm.pdf(d1) / (self.S*self.sigma*np.sqrt(self.T))
    
    def vega(self):
        d1, _ = self.d1_d2()
        return self.S*np.exp(-self.q*self.T)*norm.pdf(d1)*np.sqrt(self.T) / 100
    
    def theta(self):
        d1, d2 = self.d1_d2()
        if self.option_type == 'call':
            theta_annual = -self.S*np.exp(-self.q*self.T)*norm.pdf(d1)*self.sigma/(2*np.sqrt(self.T)) + \
                          self.q*self.S*np.exp(-self.q*self.T)*norm.cdf(d1) - \
                          self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            theta_annual = -self.S*np.exp(-self.q*self.T)*norm.pdf(d1)*self.sigma/(2*np.sqrt(self.T)) - \
                          self.q*self.S*np.exp(-self.q*self.T)*norm.cdf(-d1) + \
                          self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-d2)
        return theta_annual / 365
    
    def moneyness_description(self):
        """Describe moneyness status"""
        ratio = self.S / self.K
        if ratio > 1.05:
            return "Deep ITM"
        elif ratio > 0.95:
            return "ITM"
        elif ratio > 1.00:
            return "Slightly ITM"
        elif ratio < 0.95:
            return "OTM"
        elif ratio < 0.95:
            return "Deep OTM"
        else:
            return "ATM"

class VanillaOptionPortfolio:
    """Portfolio of vanilla options with combined Greeks"""
    
    def __init__(self):
        self.positions = []  # List of (option, quantity) tuples
    
    def add_position(self, option, quantity=1):
        """Add option position"""
        self.positions.append((option, quantity))
    
    def portfolio_price(self):
        """Total portfolio value"""
        return sum(opt.price() * qty for opt, qty in self.positions)
    
    def portfolio_delta(self):
        """Portfolio delta"""
        return sum(opt.delta() * qty for opt, qty in self.positions)
    
    def portfolio_gamma(self):
        """Portfolio gamma"""
        return sum(opt.gamma() * qty for opt, qty in self.positions)
    
    def portfolio_vega(self):
        """Portfolio vega (per 1% vol change)"""
        return sum(opt.vega() * qty for opt, qty in self.positions)
    
    def portfolio_theta(self):
        """Portfolio theta (per day)"""
        return sum(opt.theta() * qty for opt, qty in self.positions)

# Parameters
S, K, T, r, sigma, q = 100, 100, 0.25, 0.05, 0.20, 0.02

# Create vanilla options
call = VanillaOption(S, K, T, r, sigma, q, 'call')
put = VanillaOption(S, K, T, r, sigma, q, 'put')

print("="*70)
print("VANILLA OPTIONS PRICING & ANALYSIS")
print("="*70)
print(f"\nParameters: S=${S}, K=${K}, T={T} yrs, r={r*100:.1f}%, σ={sigma*100:.1f}%, q={q*100:.1f}%\n")

# Individual option analysis
print(f"{'EUROPEAN CALL':^35} {'EUROPEAN PUT':^35}")
print("-"*70)
print(f"Price:         ${call.price():>8.2f}      Price:         ${put.price():>8.2f}")
print(f"Intrinsic:     ${call.intrinsic():>8.2f}      Intrinsic:     ${put.intrinsic():>8.2f}")
print(f"Time Value:    ${call.time_value():>8.2f}      Time Value:    ${put.time_value():>8.2f}")
print(f"\nGreeks:")
print(f"Delta:         {call.delta():>8.4f}      Delta:         {put.delta():>8.4f}")
print(f"Gamma:         {call.gamma():>8.6f}      Gamma:         {put.gamma():>8.6f}")
print(f"Vega:          ${call.vega():>7.4f}      Vega:          ${put.vega():>7.4f}")
print(f"Theta:         ${call.theta():>7.4f}/day  Theta:         ${put.theta():>7.4f}/day")
print(f"\nMoneyness: {call.moneyness_description()}")

# Strategy: Long Call Spread (buy K1, sell K2)
print(f"\n{'='*70}")
print("STRATEGY: CALL SPREAD (Buy K=95, Sell K=105)")
print("="*70)

call_long = VanillaOption(S, 95, T, r, sigma, q, 'call')
call_short = VanillaOption(S, 105, T, r, sigma, q, 'call')
spread_cost = call_long.price() - call_short.price()
spread_pnl_at_expiry = lambda S_T: max(S_T - 95, 0) - max(S_T - 105, 0) - spread_cost

print(f"Cost: ${spread_cost:.2f}")
print(f"Max Profit: ${10 - spread_cost:.2f} (if S_T ≥ 105)")
print(f"Max Loss: ${spread_cost:.2f} (if S_T ≤ 95)")
print(f"Breakeven: S_T = {95 + spread_cost:.2f}")

# Strategy: Long Straddle (buy call + put)
print(f"\n{'='*70}")
print("STRATEGY: LONG STRADDLE (Buy Call K=100 + Put K=100)")
print("="*70)

straddle_cost = call.price() + put.price()
straddle_pnl = lambda S_T: max(S_T - 100, 0) + max(100 - S_T, 0) - straddle_cost

print(f"Cost: ${straddle_cost:.2f}")
print(f"Max Profit: Unlimited (both up and down)")
print(f"Max Loss: ${straddle_cost:.2f} (if S_T = 100 at expiry)")
print(f"Breakeven: S_T = {100 - straddle_cost:.2f} or {100 + straddle_cost:.2f}")
print(f"Profit if S moves > ${straddle_cost:.2f} in either direction")

straddle_portfolio = VanillaOptionPortfolio()
straddle_portfolio.add_position(call, 1)
straddle_portfolio.add_position(put, 1)
print(f"\nStraddle Greeks:")
print(f"  Delta: {straddle_portfolio.portfolio_delta():.4f} (delta-neutral)")
print(f"  Gamma: {straddle_portfolio.portfolio_gamma():.6f} (long gamma, profit on move)")
print(f"  Vega:  ${straddle_portfolio.portfolio_vega():.4f} (long volatility)")
print(f"  Theta: ${straddle_portfolio.portfolio_theta():.4f}/day (time decay)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Call and Put payoffs
spot_range = np.linspace(80, 120, 100)
call_payoffs = [max(s - K, 0) - call.price() for s in spot_range]
put_payoffs = [max(K - s, 0) - put.price() for s in spot_range]

axes[0, 0].plot(spot_range, call_payoffs, 'b-', linewidth=2, label='Long Call')
axes[0, 0].plot(spot_range, put_payoffs, 'r-', linewidth=2, label='Long Put')
axes[0, 0].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[0, 0].axvline(S, color='g', linestyle=':', alpha=0.7, label='Current S')
axes[0, 0].set_xlabel('Stock Price at Expiry S_T')
axes[0, 0].set_ylabel('P&L')
axes[0, 0].set_title('Vanilla Option Payoff Diagrams')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Call Spread payoff
call_spread_payoffs = [spread_pnl_at_expiry(s) for s in spot_range]
axes[0, 1].plot(spot_range, call_spread_payoffs, 'g-', linewidth=2.5, label='Call Spread (95/105)')
axes[0, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[0, 1].axvline(S, color='r', linestyle=':', alpha=0.7)
axes[0, 1].fill_between(spot_range, 0, call_spread_payoffs, 
                        where=np.array(call_spread_payoffs) > 0, alpha=0.3, color='green', label='Profit')
axes[0, 1].fill_between(spot_range, 0, call_spread_payoffs, 
                        where=np.array(call_spread_payoffs) < 0, alpha=0.3, color='red', label='Loss')
axes[0, 1].set_xlabel('Stock Price at Expiry S_T')
axes[0, 1].set_ylabel('P&L')
axes[0, 1].set_title('Call Spread (95-105) P&L')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Straddle payoff
straddle_payoffs = [straddle_pnl(s) for s in spot_range]
axes[0, 2].plot(spot_range, straddle_payoffs, 'purple', linewidth=2.5, label='Long Straddle')
axes[0, 2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[0, 2].axvline(S, color='r', linestyle=':', alpha=0.7, label='Current S')
axes[0, 2].fill_between(spot_range, 0, straddle_payoffs, 
                        where=np.array(straddle_payoffs) > 0, alpha=0.3, color='green')
axes[0, 2].fill_between(spot_range, 0, straddle_payoffs, 
                        where=np.array(straddle_payoffs) < 0, alpha=0.3, color='red')
axes[0, 2].set_xlabel('Stock Price at Expiry S_T')
axes[0, 2].set_ylabel('P&L')
axes[0, 2].set_title(f'Long Straddle (ATM) P&L')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Greeks across strikes
strikes = np.linspace(80, 120, 50)
deltas_call, gammas, vegas = [], [], []
for K_temp in strikes:
    opt = VanillaOption(S, K_temp, T, r, sigma, q, 'call')
    deltas_call.append(opt.delta())
    gammas.append(opt.gamma())
    vegas.append(opt.vega())

axes[1, 0].plot(strikes, deltas_call, 'b-', linewidth=2, label='Delta')
axes[1, 0].axhline(0.5, color='g', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Strike K')
axes[1, 0].set_ylabel('Delta')
axes[1, 0].set_title('Call Delta vs Strike')
axes[1, 0].grid(alpha=0.3)

# Plot 5: Gamma vs strike
axes[1, 1].plot(strikes, gammas, 'g-', linewidth=2)
axes[1, 1].axvline(S, color='r', linestyle=':', alpha=0.7, label='ATM')
axes[1, 1].set_xlabel('Strike K')
axes[1, 1].set_ylabel('Gamma')
axes[1, 1].set_title('Gamma vs Strike (Peaks ATM)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Implied volatility smile (synthetic)
strikes_smile = np.linspace(85, 115, 30)
ivs = []
for K_temp in strikes_smile:
    moneyness = S / K_temp
    # Synthetic smile: volatility increases away from ATM
    iv_smile = sigma * (1 + 0.3 * ((moneyness - 1)**2))
    ivs.append(iv_smile)

axes[1, 2].plot(strikes_smile/S, np.array(ivs)*100, 'o-', linewidth=2, markersize=5)
axes[1, 2].axvline(1, color='r', linestyle=':', alpha=0.7, label='ATM')
axes[1, 2].set_xlabel('Moneyness (K/S)')
axes[1, 2].set_ylabel('Implied Volatility (%)')
axes[1, 2].set_title('Volatility Smile (Synthetic)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('vanilla_options_analysis.png', dpi=100, bbox_inches='tight')
print("\n\nPlot saved: vanilla_options_analysis.png")
```

**Output Interpretation:**
- **Payoff Diagrams:** Show profit/loss profiles; illustrate call spread capped upside vs straddle unlimited potential
- **Greeks:** Call delta increases with strikes; gamma peaks ATM (highest sensitivity)
- **Smile:** IV varies by K; important for portfolio pricing accuracy

## 6. Challenge Round

**Q1: Why are vanilla equity options the most liquid financial instruments after bonds and FX?**  
A: (1) **Standardized contracts**: listed exchanges (CBOE) guarantee settlement, enable clearing. (2) **High leverage**: small premium controls large notional (attract speculators). (3) **Two-sided market**: hedgers (downside insurance) meet speculators (directional bets); supply/demand balanced. (4) **Foundation for complex products**: structured notes, exotics all built from vanilla vanilla pricing models. (5) **Risk management essential**: every equity trader hedges with options. Volume: ~100M contracts/day on CBOE alone (notional $100B+).

**Q2: A stock pays a dividend next month. How does this affect call and put values? Why do some American calls get exercised just before ex-dividend?**  
A: Dividend reduces S_t on ex-date (stock drops by ~dividend amount). Call value decreases (forward price lower). Put value increases (stock cheaper → deeper ITM possible). American call exercise: just before ex-dividend, if ITM call is deep, holder may exercise to receive stock + dividend. Cost: exercise call (pay K), own stock for dividend (receive d), sell stock tomorrow. Gain: dividend + intrinsic value. European call cannot capture this → American call worth more (early exercise premium).

**Q3: Explain the Put-Call Parity relationship for vanilla options: C - P = S e^{-qT} - K e^{-rT}. How do traders exploit parity violations?**  
A: Parity is an arbitrage-free relationship: owning a call is equivalent to owning stock + long put + financing (risk-neutral setup). If C - P > S e^{-qT} - K e^{-rT}: overpriced call or underpriced put → Buy put, sell call, short stock, invest K at rate r → locked-in arbitrage (risk-free profit). If C - P < RHS: buy call, sell put, buy stock → hedge profit. Market makers continuously rebalance to maintain parity (eliminates mispricings quickly); arbitrage opportunities are ephemeral (microseconds).

**Q4: A portfolio manager holds stock worth $10M. How can vanilla options reduce downside risk while preserving upside? What is the cost?**  
A: **Protective Put Strategy**: Buy puts with K = current stock price (hedge floor). At expiry: if stock drops, put protects at K; if stock rises, profit captured. Cost: put premium (typically 2-5% of stock value/year). **Collar Strategy** (cheaper): Buy OTM put (K slightly below S), sell OTM call (K above S). Net cost: reduced or zero. Tradeoff: profit capped above strike. **Outcome**: downside capped at cost, upside capped at call strike. Used by executives with concentrated holdings, index funds protecting against crash risk.

**Q5: Volatility smile/skew in equity options causes which strikes to be overpriced and which underpriced relative to Black-Scholes?**  
A: **Equity skew pattern**: OTM puts (downside) have higher IV than ATM, which higher than OTM calls. Reason: market prices tail risk (crash protection) more expensive after 2008 crisis. **Pricing impact**: BS (assumes flat vol) underprices OTM puts (actual IV higher), overprices OTM calls (actual IV lower). **P&L**: Short put + long call (call spread) loses money if real vol skew present (put deeper in red than call in green). Traders must calibrate to local vol surface, not BS single σ.

**Q6: Why are American vanilla options rarely exercised before expiry, but American vanilla puts sometimes are?**  
A: **American calls**: Early exercise only if dividend benefit > time value cost. Most stocks pay low dividends; time value too high. Exception: deep ITM calls on high-dividend stocks (10%+ yield). **American puts**: Early exercise rational when (a) deep ITM (intrinsic >> time value), (b) high interest rates (early receipt of K @ compound benefit), (c) low time value remaining (close to expiry). Example: put K=100, S=60, r=10%, T=10 days. Immediate exercise: receive $100 (invest @ 10% → $100 * e^{0.1*10/365} ≈ $100.27). Waiting: risk stock rises (less than $100 payoff). Arbitrage-free premium boundary: American put value ≥ max(K - S, BS_european_put).

## 7. Key References

- [Wikipedia: Vanilla Option](https://en.wikipedia.org/wiki/Vanilla_option) — Definition, types, examples, market structure
- [CBOE: Options Education](https://www.cboe.com/trading-learn/options-education/) — Market data, strategies, tools
- [Wikipedia: Call Option](https://en.wikipedia.org/wiki/Call_option) & [Put Option](https://en.wikipedia.org/wiki/Put_option) — Definitions, payoffs, valuation
- Hull: *Options, Futures & Derivatives* (Chapters 7-10) — Vanilla option strategies, pricing basics, early exercise
- Paul Wilmott: *Introduces Quantitative Finance* — Greeks, market conventions, hedging strategies

**Status:** ✓ Standalone file. **Complements:** european_call_option.md, european_put_option.md, black_scholes_model.md, option_strategies.md
