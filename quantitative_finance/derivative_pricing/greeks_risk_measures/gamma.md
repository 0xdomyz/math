# Gamma (Γ)

## 1. Concept Skeleton
**Definition:** Second-order partial derivative of option price with respect to underlying asset price; measures the rate of change of delta itself  
**Purpose:** Quantify delta convexity/nonlinearity; long gamma benefits from large moves (profitable rehedging); monitor delta hedging residual risk  
**Prerequisites:** Second derivatives, delta concepts, convexity, risk management

## 2. Comparative Framing
| Aspect | Gamma | Delta | Vega | Theta |
|--------|-------|-------|------|-------|
| **Order** | 2nd derivative (∂²V/∂S²) | 1st derivative | 1st derivative | 1st derivative |
| **Interpretation** | Delta sensitivity to spot | Directional hedge | Volatility exposure | Time decay |
| **Call Sign** | Always > 0 | 0 to +1 | Always > 0 | Usually < 0 |
| **Put Sign** | Always > 0 | -1 to 0 | Always > 0 | Usually > 0 |
| **High Gamma Location** | At-the-money (ATM) | N/A | ATM | Short maturity |
| **Risk Type** | Convexity (rehedge cost) | Directional | Volatility | Time |

## 3. Examples + Counterexamples

**Simple Example:**  
ATM call: Gamma highest (~0.02 per $1 spot move); delta changes by ~0.02 when spot moves $1

**Practical Case:**  
Short straddle (sold call + put): Negative gamma; loses money on large moves in either direction; "long volatility" risk

**Long Gamma Benefit:**  
Bought butterfly spread: Long gamma; benefits from realized volatility being higher than implied; profits from rehedging gains

**Counterintuitive Case:**  
Digital option at strike: Gamma → ∞ as expiry approaches; discontinuous delta; infinitely convex payoff

## 4. Layer Breakdown
```
Gamma Concept & Dynamics:
├─ Mathematical Definition:
│   ├─ Gamma: Γ = ∂²V/∂S² = ∂Δ/∂S
│   ├─ Taylor expansion: ΔV ≈ Δ × ΔS + (Γ/2) × (ΔS)²
│   ├─ Interpretation: For $1 move, delta changes by ~Γ
│   └─ Quadratic term: Γ/2 × (ΔS)² = rehedging profit/loss
├─ Black-Scholes Formula:
│   ├─ Gamma = N'(d1) / (S × σ × √T)
│   │   where N'(d1) = (1/√(2π)) × exp(-d1²/2) [normal PDF]
│   ├─ Properties:
│   │   ├─ Same for calls and puts
│   │   ├─ Always positive (convex payoff)
│   │   ├─ Maximum at ATM (S ≈ K)
│   │   └─ Highest for short maturity (small √T in denominator)
├─ Dynamics Across Parameters:
│   ├─ Increases with:
│   │   ├─ Lower spot (moves away from ATM → ATM → increases Γ)
│   │   ├─ Shorter time to expiry (↓√T → ↑Γ)
│   │   └─ Lower volatility (↓σ → ↑N'(d1) concentration)
│   ├─ Decreases with:
│   │   ├─ Higher spot (moves away from ATM)
│   │   ├─ Longer time to expiry
│   │   └─ Higher volatility (payoff spread over wider range)
├─ Rehedging P&L:
│   ├─ Realized P&L from gamma: Σ (Γᵢ/2) × (ΔSᵢ)²
│   ├─ Long gamma → profit if realized vol > implied vol
│   ├─ Short gamma → loss if realized vol > implied vol
│   ├─ Breakeven realized vol: Where gamma P&L = theta decay
│   └─ Volatility smile impact: OTM gamma skew; asymmetric hedging
├─ Practical Risk Management:
│   ├─ Gamma limit: Max acceptable Γ for portfolio
│   ├─ Rebalancing frequency: High gamma → frequent hedging
│   ├─ Cost of hedging: Proportional to Γ × vol² × time
│   └─ Hedging optimization: Balance gamma risk vs. transaction costs
└─ Relationship to Other Greeks:
    ├─ Γ-Θ relationship: θ ≈ -rK×N(-d2) - (Γ/2)×S²×σ² [theta decay offsets gamma gains for ATM]
    ├─ Γ-Vega linkage: Short Γ ⇔ need to be short vega (convexity cost)
    └─ Γ-Delta cycle: Δ changes → Γ determines rate → rehedging needed
```

**Interaction:** Delta constantly changes (Gamma) → requires rehedging → costs money if realized vol < implied vol

## 5. Mini-Project
Implement gamma computation and rehedging P&L analysis:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes components
def bs_d1(S, K, T, r, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def gamma_bs(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def delta_bs(S, K, T, r, sigma):
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def bs_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma*np.sqrt(T)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

# Parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

# Gamma across spot prices
spot_prices = np.linspace(80, 120, 100)
gammas = [gamma_bs(S, K, T, r, sigma) for S in spot_prices]
deltas = [delta_bs(S, K, T, r, sigma) for S in spot_prices]

# Gamma vs time
times = np.linspace(T, 0.01, 50)
gammas_time = [gamma_bs(S0, K, t, r, sigma) for t in times]

# Gamma vs volatility
vols = np.linspace(0.05, 0.5, 50)
gammas_vol = [gamma_bs(S0, K, T, r, v) for v in vols]

# Rehedging P&L simulation
print("=== GAMMA & REHEDGING P&L ANALYSIS ===")
np.random.seed(42)

# Scenario 1: High realized volatility
spot_moves_high = np.random.normal(0, 0.03, 252)  # High volatility: 3% daily
S_path_high = np.array([S0])
for move in spot_moves_high:
    S_path_high = np.append(S_path_high, S_path_high[-1] * (1 + move))

# Scenario 2: Low realized volatility
spot_moves_low = np.random.normal(0, 0.005, 252)  # Low volatility: 0.5% daily
S_path_low = np.array([S0])
for move in spot_moves_low:
    S_path_low = np.append(S_path_low, S_path_low[-1] * (1 + move))

# Compute realized volatility
realized_vol_high = np.std(np.log(S_path_high[1:] / S_path_high[:-1])) * np.sqrt(252)
realized_vol_low = np.std(np.log(S_path_low[1:] / S_path_low[:-1])) * np.sqrt(252)

print(f"Implied Volatility: {sigma:.2%}")
print(f"Realized Vol (High Scenario): {realized_vol_high:.2%}")
print(f"Realized Vol (Low Scenario): {realized_vol_low:.2%}")

# Rehedging P&L calculation
def calculate_rehedging_pnl(S_path, K, T, r, sigma, option_type='call'):
    T_remaining = np.linspace(T, 0, len(S_path))
    deltas = []
    gammas = []
    rehedge_pnl = []
    
    for i, (S, T_rem) in enumerate(zip(S_path, T_remaining)):
        if T_rem > 0:
            delta = delta_bs(S, K, T_rem, r, sigma)
            gamma = gamma_bs(S, K, T_rem, r, sigma)
        else:
            delta = 1.0 if S > K else 0.0
            gamma = 0.0
        
        deltas.append(delta)
        gammas.append(gamma)
        
        # Rehedging P&L: Γ/2 × (ΔS)²
        if i > 0:
            dS = S_path[i] - S_path[i-1]
            pnl_gamma = gammas[i-1] / 2 * dS**2
            rehedge_pnl.append(pnl_gamma)
    
    return np.array(deltas), np.array(gammas), np.cumsum(rehedge_pnl)

deltas_high, gammas_high, rehedge_pnl_high = calculate_rehedging_pnl(
    S_path_high, K, T, r, sigma)
deltas_low, gammas_low, rehedge_pnl_low = calculate_rehedging_pnl(
    S_path_low, K, T, r, sigma)

print(f"\nRehedging P&L (1000 contracts):")
print(f"High Vol Scenario: ${rehedge_pnl_high[-1]*1000:.2f}")
print(f"Low Vol Scenario: ${rehedge_pnl_low[-1]*1000:.2f}")

# Long gamma vs short gamma
# Long call + short delta hedge = long gamma
# Short call + long delta hedge = short gamma

print(f"\nGamma P&L Analysis:")
print(f"  Long 1000 calls (long gamma):")
print(f"    High vol: +${rehedge_pnl_high[-1]*1000:.2f} (profit)")
print(f"    Low vol: +${rehedge_pnl_low[-1]*1000:.2f} (profit)")
print(f"  Short 1000 calls (short gamma):")
print(f"    High vol: -${rehedge_pnl_high[-1]*1000:.2f} (loss)")
print(f"    Low vol: -${rehedge_pnl_low[-1]*1000:.2f} (loss)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Gamma vs Spot
axes[0, 0].plot(spot_prices, gammas, linewidth=2, color='green')
axes[0, 0].axvline(K, color='r', linestyle='--', alpha=0.5, label='Strike')
axes[0, 0].set_xlabel('Spot Price ($)')
axes[0, 0].set_ylabel('Gamma')
axes[0, 0].set_title('Gamma across Spot Prices')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Gamma vs Time
axes[0, 1].plot(times, gammas_time, linewidth=2, color='purple')
axes[0, 1].set_xlabel('Time to Expiry (years)')
axes[0, 1].set_ylabel('Gamma')
axes[0, 1].set_title('Gamma vs Time to Expiry (ATM)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Gamma vs Volatility
axes[0, 2].plot(vols, gammas_vol, linewidth=2, color='brown')
axes[0, 2].axvline(sigma, color='r', linestyle='--', alpha=0.5, label='Current σ')
axes[0, 2].set_xlabel('Volatility')
axes[0, 2].set_ylabel('Gamma')
axes[0, 2].set_title('Gamma vs Volatility (ATM, T=1yr)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Spot paths
axes[1, 0].plot(S_path_high, label='High Vol Path', linewidth=1.5, alpha=0.7)
axes[1, 0].plot(S_path_low, label='Low Vol Path', linewidth=1.5, alpha=0.7)
axes[1, 0].axhline(K, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Spot Price ($)')
axes[1, 0].set_title('Spot Price Paths')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Gamma over time (high vol path)
T_remaining_high = np.linspace(T, 0, len(S_path_high))
axes[1, 1].plot(gammas_high, linewidth=2, color='green', label='High Vol')
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Gamma')
axes[1, 1].set_title('Gamma Evolution (High Vol Path)')
axes[1, 1].grid(alpha=0.3)

# Plot 6: Rehedging P&L
rehedge_pnl_high_scaled = rehedge_pnl_high * 1000
rehedge_pnl_low_scaled = rehedge_pnl_low * 1000

axes[1, 2].plot(rehedge_pnl_high_scaled, linewidth=2, label='High Vol', color='red')
axes[1, 2].plot(rehedge_pnl_low_scaled, linewidth=2, label='Low Vol', color='blue')
axes[1, 2].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1, 2].set_xlabel('Day')
axes[1, 2].set_ylabel('Cumulative Rehedging P&L ($)')
axes[1, 2].set_title('Long Gamma P&L (1000 contracts)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Gamma-Theta tradeoff at ATM
print("\n=== GAMMA-THETA TRADEOFF ===")
theta_bs = lambda S, K, T, r, sigma: (
    -S * norm.pdf(bs_d1(S, K, T, r, sigma)) * sigma / (2 * np.sqrt(T)) -
    r * K * np.exp(-r*T) * norm.cdf(bs_d1(S, K, T, r, sigma) - sigma*np.sqrt(T))
)

atm_gamma = gamma_bs(K, K, T, r, sigma)
atm_theta = theta_bs(K, K, T, r, sigma)

print(f"ATM Gamma: {atm_gamma:.6f}")
print(f"ATM Theta (daily): {atm_theta/365:.6f}")
print(f"Gamma-Theta ratio: {atm_gamma / (atm_theta/365):.2f}")
print(f"\nInterpretation: For every $1² of realized moves,")
print(f"  Gamma P&L ≈ {atm_gamma/2:.6f}; Theta decay ≈ {atm_theta/365:.6f}/day")
```

## 6. Challenge Round
When is gamma management critical?
- Short volatility portfolios: Negative gamma exposes to large moves; requires active hedging
- Near expiry: ATM gamma explodes; delta changes rapidly; rehedging becomes expensive
- Vega selling: Need offsetting long gamma to limit volatility losses on large moves
- Transaction costs: Frequent rehedging due to gamma costs money; optimal rehedge interval minimizes total cost
- Tail risk: Gamma doesn't protect against gaps; overnight moves bypass delta hedge

## 7. Key References
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Wilmott - Volatility Smile (Chapter 6)](https://www.paulwilmott.com)
- [Natenberg - Option Volatility & Pricing (Chapter 10)](https://www.amazon.com/Option-Volatility-Pricing-Advanced-Strategies/dp/1557784124)

---
**Status:** Secondary option Greek | **Complements:** Delta, Vega, Hedging Strategy, Greeks Portfolio
