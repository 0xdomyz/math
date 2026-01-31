# Option Pricing Basics

## 1. Concept Skeleton
**Definition:** Foundation for valuing financial derivatives; establishes framework connecting spot price, strike price, time value, and volatility to determine fair value of options  
**Purpose:** Understand intrinsic vs time value; establish pricing bounds (no-arbitrage); introduce concepts used in advanced models; practical valuation framework  
**Prerequisites:** Financial derivatives, option types (call/put), payoff diagrams, no-arbitrage principle, interest rates, spot-forward relationship

## 2. Comparative Framing
| Concept | American Option | European Option | Forward Contract | Futures |
|---------|-----------------|-----------------|-----------------|---------|
| **Exercise** | Any time to expiry | Only at expiry | Obligation at maturity | Daily settlement |
| **Early Exercise** | Possible | Not possible | N/A | N/A |
| **Value** | ≥ European | Baseline | Linear in spot | Marked-to-market |
| **Pricing** | No simple formula | Closed forms available | F = S × e^(rT) | Similar to forward |
| **Holder Optionality** | Yes | Limited | None | Limited |
| **Typical Use** | Stock options | Easier to analyze theoretically | Hedging | Speculation |
| **Complexity** | Higher | Moderate | Simple | Moderate |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock at $100, call option with strike $105, 1 year to expiry. Intrinsic value = max(100-105, 0) = $0. Time value encodes probability stock rises above $105.

**Arbitrage Bound:**  
Call value C must satisfy: max(S-K, 0) ≤ C ≤ S. If C > S, arbitrage: buy stock, sell call. If C < max(S-K, 0), arbitrage: buy call, immediate exercise if positive intrinsic.

**Put-Call Parity:**  
C - P = S - Ke^(-rT). If violated, arbitrage. Example: S=$50, K=$50, r=5%, T=1, C=$3, P=$1. Check: 3-1=2 vs 50-50e^(-0.05)≈2.44. Mispriced; arbitrage opportunity.

**Time Decay:**  
At-the-money option loses value as expiration approaches (theta decay), all else equal. Far out-of-the-money: minimal time value, negligible decay.

**Early Exercise Value:**  
American call on non-dividend stock: never optimal to exercise early (maintain flexibility). American call with dividends: may exercise just before ex-dividend date.

## 4. Layer Breakdown
```
Option Pricing Foundations:

├─ Payoff Structures:
│  ├─ Call Option Payoff:
│  │   ├─ At expiry: max(S_T - K, 0)
│  │   ├─ Value components: Intrinsic + Time Value
│  │   ├─ Intrinsic: max(S - K, 0) (immediate exercise value)
│  │   └─ Time Value: Option Price - Intrinsic
│  ├─ Put Option Payoff:
│  │   ├─ At expiry: max(K - S_T, 0)
│  │   ├─ Intrinsic: max(K - S, 0)
│  │   └─ Time value (similar decay as calls)
│  └─ Portfolio Combinations:
│      ├─ Straddle: Buy call + put (bet on volatility)
│      ├─ Strangle: OTM call + OTM put (cheaper straddle)
│      ├─ Spread: Long call + short call (limited profit/loss)
│      └─ Collar: Long stock + long put + short call (downside protection)
├─ No-Arbitrage Bounds:
│  ├─ Call Bounds:
│  │   ├─ Lower: max(S - Ke^(-rT), 0) ≤ C
│  │   ├─ Upper: C ≤ S (must not exceed stock value)
│  │   └─ For American: C_American ≥ max(S - K, 0)
│  ├─ Put Bounds:
│  │   ├─ Lower: max(Ke^(-rT) - S, 0) ≤ P
│  │   ├─ Upper: P ≤ Ke^(-rT)
│  │   └─ For American: P_American ≥ max(K - S, 0)
│  └─ Arbitrage Strategies:
│      ├─ Conversion: Long stock + long put + short call
│      │   (Creates synthetic risk-free bond)
│      ├─ Reversal: Short stock + short put + long call
│      └─ Box Spread: Call spread + put spread (synthesizes bond)
├─ Put-Call Parity (European):
│  ├─ C - P = S - Ke^(-rT)
│  ├─ Derivation: Construct two portfolios with same payoff
│  │   ├─ Portfolio A: Long call + cash Ke^(-rT)
│  │   ├─ Portfolio B: Long stock + long put
│  │   └─ Both worth max(S_T, K) at expiry
│  ├─ Rearrangement: P = C - S + Ke^(-rT)
│  ├─ Implications: Call and put are related; can't price independently
│  └─ American Exception: C_Am - P_Am ≤ S - Ke^(-rT) (inequality, not equality)
├─ Greeks & Sensitivities:
│  ├─ Delta (∂C/∂S): Change in option price per $1 stock move
│  │   ├─ Range: 0 to 1 for calls, -1 to 0 for puts
│  │   ├─ Interpretation: Equivalent shares of stock
│  │   └─ Hedging: Delta-neutral portfolio
│  ├─ Gamma (∂²C/∂S²): Delta sensitivity to stock price
│  │   ├─ Highest at-the-money
│  │   ├─ Increases near expiry
│  │   └─ Risk for hedgers: Delta becomes inaccurate
│  ├─ Theta (∂C/∂t): Time decay (usually negative for buyer)
│  │   ├─ Long calls/puts: Lose time value
│  │   ├─ Near expiry: Steep decay
│  │   └─ Short gamma = earn theta
│  ├─ Vega (∂C/∂σ): Volatility sensitivity
│  │   ├─ Long options: Positive vega (profit from vol increases)
│  │   ├─ At-the-money: Highest vega
│  │   └─ Volatility traders main concern
│  └─ Rho (∂C/∂r): Interest rate sensitivity
│      ├─ Weaker effect than other Greeks
│      ├─ Long options: Positive rho for calls, negative for puts
│      └─ More important for bonds/long-dated options
├─ Factors Affecting Option Value:
│  ├─ Stock Price (S): ↑S → ↑Call, ↓Put
│  ├─ Strike Price (K): ↑K → ↓Call, ↑Put
│  ├─ Time to Expiry (T): Usually ↑T → ↑Option value (more time = more optionality)
│  │   Exception: Deep in-the-money put value might decrease
│  ├─ Volatility (σ): ↑σ → ↑Call value, ↑Put value
│  │   (Both benefit from increased uncertainty)
│  ├─ Risk-free Rate (r): ↑r → ↑Call, ↓Put
│  │   (Higher discount rate affects forward price)
│  ├─ Dividends (D): ↑D → ↓Call, ↑Put
│  │   (Reduces expected stock price at expiry)
│  └─ Early Exercise (American): Adds value due to flexibility
└─ Early Exercise Decisions:
   ├─ American Call (no dividends): Never optimal early
   │   (Intrinsic max(S-K,0) < option value due to time value)
   ├─ American Call (with dividends): May exercise before ex-date
   ├─ American Put: Always has early exercise value
   │   (Can lock in intrinsic max(K-S, 0))
   ├─ Trigger: Exercise if S - D*e^(rT) > C_continuation
   └─ Binomial model easily captures this optionality
```

**Interaction:** Spot price determines intrinsic value; volatility and time drive time value; interest rates affect discounting and forward pricing.

## 5. Mini-Project
Analyze option valuations, bounds, and arbitrage:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("OPTION PRICING BASICS")
print("="*60)

# Parameters
S = 100  # Current stock price
K = 100  # Strike price
r = 0.05  # Risk-free rate
T = 1.0  # Time to expiry (1 year)
sigma = 0.2  # Volatility

print(f"\nMarket Parameters:")
print(f"  Stock Price (S): ${S}")
print(f"  Strike Price (K): ${K}")
print(f"  Risk-free Rate (r): {r:.2%}")
print(f"  Time to Expiry (T): {T} year(s)")
print(f"  Volatility (σ): {sigma:.1%}")

# === Scenario 1: Black-Scholes Pricing ===
print("\n" + "="*60)
print("SCENARIO 1: European Option Pricing (Black-Scholes)")
print("="*60)

# Black-Scholes formulas
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

N_d1 = norm.cdf(d1)
N_d2 = norm.cdf(d2)
N_minus_d1 = norm.cdf(-d1)
N_minus_d2 = norm.cdf(-d2)

C_BS = S*N_d1 - K*np.exp(-r*T)*N_d2
P_BS = K*np.exp(-r*T)*N_minus_d2 - S*N_minus_d1

print(f"\nBlack-Scholes Values (European):")
print(f"  Call Value: ${C_BS:.2f}")
print(f"  Put Value: ${P_BS:.2f}")
print(f"  Put-Call Parity Check: C - P = {C_BS - P_BS:.4f}")
print(f"  Theoretical: S - Ke^(-rT) = {S - K*np.exp(-r*T):.4f}")

# Intrinsic and Time Values
intrinsic_call = max(S - K, 0)
intrinsic_put = max(K - S, 0)
time_value_call = C_BS - intrinsic_call
time_value_put = P_BS - intrinsic_put

print(f"\nValue Components:")
print(f"  Call: Intrinsic=${intrinsic_call:.2f} + Time Value=${time_value_call:.2f} = ${C_BS:.2f}")
print(f"  Put: Intrinsic=${intrinsic_put:.2f} + Time Value=${time_value_put:.2f} = ${P_BS:.2f}")

# === Scenario 2: No-Arbitrage Bounds ===
print("\n" + "="*60)
print("SCENARIO 2: No-Arbitrage Bounds")
print("="*60)

# European Call bounds
C_lower = max(S - K*np.exp(-r*T), 0)
C_upper = S

# European Put bounds
P_lower = max(K*np.exp(-r*T) - S, 0)
P_upper = K*np.exp(-r*T)

print(f"\nEuropean Call Bounds:")
print(f"  Lower: max(S - Ke^(-rT), 0) = ${C_lower:.2f}")
print(f"  Upper: S = ${C_upper:.2f}")
print(f"  BS Value: ${C_BS:.2f} (within bounds? {C_lower <= C_BS <= C_upper})")

print(f"\nEuropean Put Bounds:")
print(f"  Lower: max(Ke^(-rT) - S, 0) = ${P_lower:.2f}")
print(f"  Upper: Ke^(-rT) = ${P_upper:.2f}")
print(f"  BS Value: ${P_BS:.2f} (within bounds? {P_lower <= P_BS <= P_upper})")

# American lower bounds (with dividends considered as 0)
C_american_lower = max(S - K, 0)
P_american_lower = max(K - S, 0)

print(f"\nAmerican Option Lower Bounds (no dividends):")
print(f"  Call: max(S - K, 0) = ${C_american_lower:.2f}")
print(f"  Put: max(K - S, 0) = ${P_american_lower:.2f}")

# === Scenario 3: Greeks Sensitivity ===
print("\n" + "="*60)
print("SCENARIO 3: Greeks & Sensitivities")
print("="*60)

# Delta
delta_call = N_d1
delta_put = -N_minus_d1

# Gamma (common for both call and put)
gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Theta (per day)
theta_call = -S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*N_d2
theta_put = -S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*N_minus_d2
theta_call_daily = theta_call / 365
theta_put_daily = theta_put / 365

# Vega (per 1% change in volatility)
vega = S * norm.pdf(d1) * np.sqrt(T) / 100

# Rho (per 1% change in rate)
rho_call = K * T * np.exp(-r*T) * N_d2 / 100
rho_put = -K * T * np.exp(-r*T) * N_minus_d2 / 100

print(f"\nGreeks (European Options):")
print(f"  Delta Call: {delta_call:.4f}")
print(f"  Delta Put: {delta_put:.4f}")
print(f"  Gamma (both): {gamma:.6f}")
print(f"  Theta Call (daily): ${theta_call_daily:.4f}")
print(f"  Theta Put (daily): ${theta_put_daily:.4f}")
print(f"  Vega (per 1% vol change): ${vega:.4f}")
print(f"  Rho Call (per 1% rate change): ${rho_call:.4f}")
print(f"  Rho Put (per 1% rate change): ${rho_put:.4f}")

# === Scenario 4: Sensitivity Analysis ===
print("\n" + "="*60)
print("SCENARIO 4: Impact of Parameters on Option Value")
print("="*60)

# Impact of spot price
spot_range = np.linspace(80, 120, 41)
call_values = []
put_values = []

for s in spot_range:
    d1_temp = (np.log(s/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2_temp = d1_temp - sigma*np.sqrt(T)
    c = s*norm.cdf(d1_temp) - K*np.exp(-r*T)*norm.cdf(d2_temp)
    p = K*np.exp(-r*T)*norm.cdf(-d2_temp) - s*norm.cdf(-d1_temp)
    call_values.append(c)
    put_values.append(p)

# Impact of volatility
vol_range = np.linspace(0.05, 0.5, 20)
call_vol = []
put_vol = []

for sig in vol_range:
    d1_temp = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2_temp = d1_temp - sig*np.sqrt(T)
    c = S*norm.cdf(d1_temp) - K*np.exp(-r*T)*norm.cdf(d2_temp)
    p = K*np.exp(-r*T)*norm.cdf(-d2_temp) - S*norm.cdf(-d1_temp)
    call_vol.append(c)
    put_vol.append(p)

# Impact of time to expiry
time_range = np.linspace(0.01, 1.5, 30)
call_time = []
put_time = []

for t in time_range:
    d1_temp = (np.log(S/K) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2_temp = d1_temp - sigma*np.sqrt(t)
    c = S*norm.cdf(d1_temp) - K*np.exp(-r*t)*norm.cdf(d2_temp)
    p = K*np.exp(-r*t)*norm.cdf(-d2_temp) - S*norm.cdf(-d1_temp)
    call_time.append(c)
    put_time.append(p)

print(f"\nSensitivity Summary:")
print(f"  If S increases by $1 → Call increases by ${delta_call:.2f}, Put changes by ${delta_put:.2f}")
print(f"  If σ increases by 1% → Call increases by ${vega/100:.2f}, Put increases by ${vega/100:.2f}")
print(f"  If r increases by 1% → Call changes by ${rho_call/100:.2f}, Put changes by ${rho_put/100:.2f}")
print(f"  1 day of decay → Call loses ${abs(theta_call_daily):.2f}, Put loses ${abs(theta_put_daily):.2f}")

# === Scenario 5: Put-Call Parity Arbitrage ===
print("\n" + "="*60)
print("SCENARIO 5: Put-Call Parity & Arbitrage Detection")
print("="*60)

# Suppose we observe market prices (intentionally mispriced)
C_market = 10.5  # Slightly overpriced
P_market = 4.2   # Slightly underpriced

# Theoretical relationship
parity_diff = C_market - P_market - (S - K*np.exp(-r*T))

print(f"\nMarket Prices (hypothetical):")
print(f"  Call: ${C_market:.2f}")
print(f"  Put: ${P_market:.2f}")
print(f"\nParity Check:")
print(f"  C - P: {C_market - P_market:.4f}")
print(f"  S - Ke^(-rT): {S - K*np.exp(-r*T):.4f}")
print(f"  Difference: {parity_diff:.4f} {'(Arbitrage!)' if abs(parity_diff) > 0.01 else '(Fair)'}")

if parity_diff > 0.01:
    print(f"\n  Arbitrage Strategy (Reversal):")
    print(f"    1. Short stock at ${S}")
    print(f"    2. Short put at ${P_market} (pay ${P_market})")
    print(f"    3. Buy call at ${C_market} (pay ${C_market})")
    print(f"    Net cash: ${S - P_market - C_market:.2f} invested at r={r}")
    print(f"    At expiry: Payoff = max(S_T - K, 0) - max(K - S_T, 0) = S_T - K (from long call + short put)")
    print(f"    Offset by short stock at K")
    print(f"    Risk-free profit: ${-(S - P_market - C_market) * (np.exp(r*T) - 1):.2f}")

# === Visualization ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Payoff diagrams
ax = axes[0, 0]
S_range_payoff = np.linspace(60, 140, 200)
call_payoff = np.maximum(S_range_payoff - K, 0)
put_payoff = np.maximum(K - S_range_payoff, 0)
ax.plot(S_range_payoff, call_payoff, 'b-', linewidth=2.5, label='Call Payoff')
ax.plot(S_range_payoff, put_payoff, 'r-', linewidth=2.5, label='Put Payoff')
ax.axvline(K, color='k', linestyle='--', alpha=0.5, label=f'Strike=${K}')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Stock Price at Expiry')
ax.set_ylabel('Payoff')
ax.set_title('Option Payoff Diagrams')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Option value vs Stock price
ax = axes[0, 1]
ax.plot(spot_range, call_values, 'b-', linewidth=2.5, label='Call Value')
ax.plot(spot_range, put_values, 'r-', linewidth=2.5, label='Put Value')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.axhline(C_BS, color='b', linestyle=':', alpha=0.5)
ax.axhline(P_BS, color='r', linestyle=':', alpha=0.5)
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Option Value')
ax.set_title('Option Values vs Stock Price')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Option value vs Volatility
ax = axes[0, 2]
ax.plot(vol_range*100, call_vol, 'b-', linewidth=2.5, label='Call Value', marker='o')
ax.plot(vol_range*100, put_vol, 'r-', linewidth=2.5, label='Put Value', marker='s')
ax.axvline(sigma*100, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Volatility (σ) [%]')
ax.set_ylabel('Option Value')
ax.set_title('Option Values vs Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Option value vs Time to expiry
ax = axes[1, 0]
ax.plot(time_range, call_time, 'b-', linewidth=2.5, label='Call Value', marker='o')
ax.plot(time_range, put_time, 'r-', linewidth=2.5, label='Put Value', marker='s')
ax.axvline(T, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Time to Expiry (Years)')
ax.set_ylabel('Option Value')
ax.set_title('Option Values vs Time to Expiry')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Greeks for Call vs Stock Price
ax = axes[1, 1]
deltas = []
gammas = []
for s in spot_range:
    d1_temp = (np.log(s/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1_temp)
    gamma = norm.pdf(d1_temp) / (s * sigma * np.sqrt(T))
    deltas.append(delta)
    gammas.append(gamma)

ax_twin = ax.twinx()
ax.plot(spot_range, deltas, 'b-', linewidth=2.5, label='Delta')
ax_twin.plot(spot_range, gammas, 'g-', linewidth=2.5, label='Gamma')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Delta', color='b')
ax_twin.set_ylabel('Gamma', color='g')
ax.set_title('Greeks vs Stock Price')
ax.legend(loc='upper left')
ax_twin.legend(loc='upper right')
ax.grid(alpha=0.3)

# Plot 6: Greeks table
ax = axes[1, 2]
ax.axis('off')
greeks_data = [
    ['Greek', 'Call', 'Put'],
    ['Delta', f'{delta_call:.4f}', f'{delta_put:.4f}'],
    ['Gamma', f'{gamma:.6f}', f'{gamma:.6f}'],
    ['Theta (daily)', f'${theta_call_daily:.2f}', f'${theta_put_daily:.2f}'],
    ['Vega', f'${vega:.2f}', f'${vega:.2f}'],
    ['Rho', f'${rho_call:.2f}', f'${rho_put:.2f}'],
]

table = ax.table(cellText=greeks_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Greeks Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Early Exercise:** American call on dividend-paying stock with D paid just before expiry. When is early exercise optimal? Derive condition.

2. **Put-Call Parity Violations:** Find or construct real market prices violating parity. Execute arbitrage; calculate risk-free profit.

3. **Straddle Pricing:** Buy call + put at same strike. How does value vary with volatility? What's breakeven on moves?

4. **Volatility Surface:** Different strikes/expirations have different implied volatilities. Plot surface; identify "smile" or "skew."

5. **Approximations:** For small T, use Taylor expansion of Black-Scholes. First-order approximation: C ≈ max(S-K,0) + (time value term). Compare to exact.

## 7. Key References
- [Hull, Options, Futures, and Other Derivatives (Chapter 9-10)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000006649)
- [Black-Scholes Model (Wikipedia)](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Put-Call Parity (Investopedia)](https://www.investopedia.com/terms/p/putcallparity.asp)
- [Option Greeks Explained](https://www.investopedia.com/terms/g/greeks.asp)

---
**Status:** Foundation derivative pricing | **Complements:** Black-Scholes Model, Binomial Trees, Implied Volatility, Risk Measures
