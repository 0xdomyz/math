# Vega (ν)

## 1. Concept Skeleton
**Definition:** First-order partial derivative of option price with respect to volatility; measures sensitivity to changes in price volatility (σ)  
**Purpose:** Quantify volatility exposure; hedge against implied volatility fluctuations; differentiate vol risk from directional delta risk  
**Prerequisites:** Volatility concepts, partial derivatives, implied volatility, stochastic volatility

## 2. Comparative Framing
| Aspect | Vega | Delta | Gamma | Theta |
|--------|------|-------|-------|-------|
| **Measure** | ∂V/∂σ (volatility) | ∂V/∂S (spot) | ∂²V/∂S² | ∂V/∂T |
| **Call Sign** | Always > 0 | 0 to +1 | Always > 0 | Usually < 0 |
| **Put Sign** | Always > 0 | -1 to 0 | Always > 0 | Mixed |
| **Highest at** | ATM | N/A | ATM | Short expiry |
| **Risk Exposure** | Volatility (exogenous) | Directional (spot) | Nonlinearity | Decay |
| **Hedging Vehicle** | VIX, variance swaps | Underlying | Options | Calendar spreads |

## 3. Examples + Counterexamples

**Simple Example:**  
ATM call: Vega ≈ 0.4; if implied vol ↑ 1%, call value ↑ $0.40; long vega benefit from vol spike

**Practical Case:**  
Strangle seller (sold OTM call + put): Short vega; collects premium if realized vol < implied vol; profits from vol contraction

**Counterintuitive Case:**  
Very short-dated OTM option: Vega ≈ 0 (little time for volatility benefit); theta dominates; vol changes have minimal impact

**Edge Case:**  
Variance swap vs option: Variance swap vega linear in realized vol; option vega nonlinear (changes with spot); different hedging mechanics

## 4. Layer Breakdown
```
Vega Concept & Role:
├─ Mathematical Definition:
│   ├─ Vega: ν = ∂V/∂σ (notation: sometimes normalized per 1% vol change)
│   ├─ Per 1% vol (0.01): νₚₜ = ν / 100 (practical convention)
│   ├─ Interpretation: Option value changes by ν dollars per 1% volatility increase
│   └─ Scaling: Vega depends on option type, strike, moneyness, time
├─ Black-Scholes Formula:
│   ├─ Vega = S × N'(d1) × √T (same for calls and puts)
│   │   where N'(d1) = (1/√(2π)) × exp(-d1²/2)
│   ├─ Properties:
│   │   ├─ Always positive (both calls and puts benefit from vol ↑)
│   │   ├─ Maximum at ATM (S ≈ K)
│   │   ├─ Increases with time to expiry (√T term)
│   │   └─ Symmetric: calls and puts have identical vega
├─ Vega Dynamics:
│   ├─ Increases with:
│   │   ├─ Longer maturity (√T grows)
│   │   ├─ Spot near strike (ATM concentration)
│   │   └─ Lower risk-free rate (subtle)
│   ├─ Decreases with:
│   │   ├─ Spot away from strike (OTM/ITM decay)
│   │   ├─ Short maturity (expiration → vega → 0)
│   │   └─ Lower implied volatility (convexity effect in BS)
├─ Volatility Exposure Layers:
│   ├─ Implied Volatility: Market-quoted (tradeable via options)
│   ├─ Realized Volatility: Actual historical/forecast moves
│   ├─ Volatility Smile/Skew: IV varies by strike (affects delta-hedged vega)
│   ├─ Term Structure: IV varies by maturity (calendar vega)
│   └─ Cross-vega: Multi-asset vega (basket correlation risk)
├─ Hedging Vega Exposure:
│   ├─ Long vega: Hold long options (calls/puts); profit from vol ↑
│   ├─ Short vega: Sell options; profit if realized vol < implied vol
│   ├─ Vega hedge: Buy/sell options or variance swaps to offset exposure
│   ├─ Vega scaling: Divide total vega by single option vega to get hedge ratio
│   └─ Monitoring: Track portfolio vega by strike, maturity, asset
├─ Relationship to Other Greeks:
│   ├─ Vega-Delta coupling: Vega hedges often change delta; requires rebalancing
│   ├─ Vega-Theta tradeoff: Long vega → long gamma → positive theta decay cost
│   ├─ Vega skew: OTM puts have higher vega per vol unit due to smile
│   └─ Vega term: Calendar spread; long-dated options higher vega
```

**Interaction:** Vol uncertainty → vega exposure quantifies risk → hedge with vol derivatives → rebalance as spot/vol change

## 5. Mini-Project
Implement vega computation and volatility hedging:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes components
def bs_d1(S, K, T, r, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def vega_bs(S, K, T, r, sigma):
    """Vega per 1% change in volatility"""
    d1 = bs_d1(S, K, T, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Normalized per 1%
    return vega

def bs_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma*np.sqrt(T)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def delta_bs(S, K, T, r, sigma):
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def gamma_bs(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Parameters
S0, K, T, r = 100, 100, 1, 0.05
sigma_implied = 0.20

# 1. Vega across spot prices
spot_prices = np.linspace(80, 120, 100)
vegas = [vega_bs(S, K, T, r, sigma_implied) for S in spot_prices]

# 2. Vega across volatilities
vols = np.linspace(0.05, 0.5, 100)
vegas_vol = [vega_bs(S0, K, T, r, v) for v in vols]

# 3. Vega across time
times = np.linspace(T, 0.01, 100)
vegas_time = [vega_bs(S0, K, t, r, sigma_implied) for t in times]

# 4. Volatility scenario analysis
print("=== VEGA & VOLATILITY HEDGING ===")

# Portfolio: Long 100 calls (long vega)
position_size = 100
position_vega = position_size * vega_bs(S0, K, T, r, sigma_implied)

print(f"\nPortfolio: {position_size} long calls")
print(f"Single call vega (per 1% vol): ${vega_bs(S0, K, T, r, sigma_implied):.4f}")
print(f"Total portfolio vega: ${position_vega:.2f}")

# Scenarios for implied volatility
vol_scenarios = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
spot_scenarios = np.array([90, 95, 100, 105, 110])

print(f"\nProfit/Loss from volatility change (holding spot constant at ${S0}):")
for vol_new in vol_scenarios:
    if vol_new != sigma_implied:
        call_price_new = bs_call(S0, K, T, r, vol_new)
        call_price_old = bs_call(S0, K, T, r, sigma_implied)
        pnl = position_size * (call_price_new - call_price_old)
        vol_change_pct = (vol_new - sigma_implied) * 100
        print(f"  Vol {vol_new:.1%} (Δ{vol_change_pct:+.0f}%): ${pnl:+.2f}")

# 5. Volatility hedging
print(f"\n=== VEGA HEDGING ===")

# Hedge with short-dated option (higher gamma, lower vega per unit)
K_hedge = 105  # OTM put for hedge
T_hedge = 0.25  # 3 months
vega_hedge_single = vega_bs(S0, K_hedge, T_hedge, r, sigma_implied)

# Hedge ratio to make portfolio vega-neutral
hedge_ratio = position_vega / vega_hedge_single
print(f"\nHedge instrument: Short-dated OTM put (K={K_hedge}, T={T_hedge}yr)")
print(f"Hedge vega (single): ${vega_hedge_single:.4f}")
print(f"Hedge ratio: Short {hedge_ratio:.0f} puts to neutralize vega")

# New portfolio vega after hedge
portfolio_vega_hedged = position_vega - hedge_ratio * vega_hedge_single
print(f"Portfolio vega after hedge: ${portfolio_vega_hedged:.2f} (should be ~0)")

# 6. Vega scenario P&L after hedging
print(f"\nP&L with vega hedge (across volatility scenarios):")
for vol_new in vol_scenarios:
    call_pnl = position_size * (bs_call(S0, K, T, r, vol_new) - 
                                bs_call(S0, K, T, r, sigma_implied))
    put_pnl = -hedge_ratio * (bs_call(S0, K_hedge, T_hedge, r, vol_new) -
                              bs_call(S0, K_hedge, T_hedge, r, sigma_implied))
    
    # Note: using call price for put via put-call parity for simplicity
    put_price_new = (bs_call(S0, K_hedge, T_hedge, r, vol_new) - S0 + 
                     K_hedge * np.exp(-r*T_hedge))
    put_price_old = (bs_call(S0, K_hedge, T_hedge, r, sigma_implied) - S0 + 
                     K_hedge * np.exp(-r*T_hedge))
    put_pnl = -hedge_ratio * (put_price_new - put_price_old)
    
    total_pnl = call_pnl + put_pnl
    print(f"  Vol {vol_new:.1%}: Call ${call_pnl:+.2f}, Hedge ${put_pnl:+.2f}, Total ${total_pnl:+.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Vega vs Spot
axes[0, 0].plot(spot_prices, vegas, linewidth=2, color='blue')
axes[0, 0].axvline(K, color='r', linestyle='--', alpha=0.5, label='Strike')
axes[0, 0].set_xlabel('Spot Price ($)')
axes[0, 0].set_ylabel('Vega (per 1% vol)')
axes[0, 0].set_title('Vega across Spot Prices')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Vega vs Volatility
axes[0, 1].plot(vols, vegas_vol, linewidth=2, color='green')
axes[0, 1].axvline(sigma_implied, color='r', linestyle='--', alpha=0.5, label='Current σ')
axes[0, 1].set_xlabel('Volatility')
axes[0, 1].set_ylabel('Vega (per 1% vol)')
axes[0, 1].set_title('Vega vs Volatility Level')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Vega vs Time
axes[0, 2].plot(times, vegas_time, linewidth=2, color='purple')
axes[0, 2].set_xlabel('Time to Expiry (years)')
axes[0, 2].set_ylabel('Vega (per 1% vol)')
axes[0, 2].set_title('Vega vs Time to Expiry')
axes[0, 2].grid(alpha=0.3)

# Plot 4: P&L from vol change (long call)
vol_changes = (vol_scenarios - sigma_implied) * 100
pnls = []
for vol_new in vol_scenarios:
    call_pnl = position_size * (bs_call(S0, K, T, r, vol_new) - 
                                bs_call(S0, K, T, r, sigma_implied))
    pnls.append(call_pnl)

axes[1, 0].bar(vol_changes, pnls, color=['red' if p < 0 else 'green' for p in pnls], alpha=0.7)
axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].set_xlabel('Volatility Change (percentage points)')
axes[1, 0].set_ylabel('P&L ($)')
axes[1, 0].set_title('Long Call P&L from Volatility Change')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 5: Volatility surface (Spot vs Vol)
spot_mesh = np.linspace(80, 120, 40)
vol_mesh = np.linspace(0.10, 0.40, 40)
vega_surface = np.zeros((len(vol_mesh), len(spot_mesh)))

for i, v in enumerate(vol_mesh):
    for j, s in enumerate(spot_mesh):
        vega_surface[i, j] = vega_bs(s, K, T, r, v)

im = axes[1, 1].contourf(spot_mesh, vol_mesh, vega_surface, levels=20, cmap='viridis')
axes[1, 1].set_xlabel('Spot Price ($)')
axes[1, 1].set_ylabel('Volatility')
axes[1, 1].set_title('Vega Surface')
plt.colorbar(im, ax=axes[1, 1])

# Plot 6: Portfolio Greeks comparison
strike_range = np.linspace(90, 110, 20)
deltas_range = [delta_bs(s, K, T, r, sigma_implied) for s in strike_range]
gammas_range = [gamma_bs(s, K, T, r, sigma_implied) for s in strike_range]
vegas_range = [vega_bs(s, K, T, r, sigma_implied) for s in strike_range]

ax_twin = axes[1, 2]
ax_twin2 = ax_twin.twinx()
ax_twin3 = ax_twin.twinx()
ax_twin3.spines['right'].set_position(('outward', 60))

line1 = ax_twin.plot(strike_range, deltas_range, linewidth=2, label='Delta', color='blue')
line2 = ax_twin2.plot(strike_range, gammas_range, linewidth=2, label='Gamma', color='green')
line3 = ax_twin3.plot(strike_range, vegas_range, linewidth=2, label='Vega', color='red')

ax_twin.set_xlabel('Spot Price ($)')
ax_twin.set_ylabel('Delta', color='blue')
ax_twin2.set_ylabel('Gamma', color='green')
ax_twin3.set_ylabel('Vega', color='red')
ax_twin.set_title('Greeks Comparison')
ax_twin.tick_params(axis='y', labelcolor='blue')
ax_twin2.tick_params(axis='y', labelcolor='green')
ax_twin3.tick_params(axis='y', labelcolor='red')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax_twin.legend(lines, labels, loc='upper left')
ax_twin.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is vega exposure misleading?
- Volatility smile: Vega aggregates across strikes; skew effects not captured (need spot-vega separately)
- Stochastic vol: Implied vol changes with spot (volvol); adds correlation risk beyond vega
- Term structure: Long-dated vega sensitive to different vol factors; not additive across maturities
- Realized vs implied: Vega hedges implied vol risk; doesn't protect if realized vol diverges from path-dependent realized vol
- Gamma-vega tradeoff: Long vol position = long gamma + long vega; requires monitoring both

## 7. Key References
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Taleb - Dynamic Hedging (Chapter 7)](https://www.paulwilmott.com)
- [Wilmott - Volatility Smile (Chapter 4)](https://www.paulwilmott.com)

---
**Status:** Primary volatility risk metric | **Complements:** Gamma, Theta, Volatility Surface, Options Greeks
