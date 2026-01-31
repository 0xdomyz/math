# Theta (θ)

## 1. Concept Skeleton
**Definition:** First-order partial derivative of option price with respect to time; measures the rate of change in option value as time passes (time decay)  
**Purpose:** Quantify temporal decay of option value; monitor erosion of premium; identify time decay profit/loss in positions  
**Prerequisites:** Partial derivatives, option pricing, time value, decay concepts

## 2. Comparative Framing
| Aspect | Theta | Delta | Gamma | Vega |
|--------|-------|-------|-------|------|
| **Measure** | ∂V/∂T (time) | ∂V/∂S (spot) | ∂²V/∂S² | ∂V/∂σ (vol) |
| **Call Sign** | Usually < 0 (decay) | 0 to +1 | Always > 0 | Always > 0 |
| **Put Sign** | Usually > 0 (wait value) | -1 to 0 | Always > 0 | Always > 0 |
| **Most Negative** | OTM call, short maturity | N/A | ATM, short maturity | N/A |
| **Hedging Vehicle** | Calendar spread | Underlying | Options | Vol derivatives |
| **Time Dependency** | Accelerates as T → 0 | N/A | Accelerates as T → 0 | Linear in √T |

## 3. Examples + Counterexamples

**Simple Example:**  
OTM call 1 day to expiry, worth $0.01: Theta ≈ -$0.01/day; expires worthless tomorrow

**Practical Case:**  
ATM straddle seller: Short theta benefits from daily time decay; collects premium as both calls and puts lose value

**Counterintuitive Case:**  
Deep ITM put: Positive theta; benefits from time decay (wait value decreases, intrinsic value + decay benefit = erosion of discount)

**Edge Case:**  
Dividend-paying stock: Call theta can be positive pre-dividend (dividend yield benefit > decay); changes sign

## 4. Layer Breakdown
```
Theta Concept & Dynamics:
├─ Mathematical Definition:
│   ├─ Theta: θ = ∂V/∂T (sometimes denoted as -∂V/∂t to show decay)
│   ├─ Daily theta: θ/365 (annualized theta divided by trading days)
│   ├─ Interpretation: Option value changes by θ dollars per year (or θ/365 per day)
│   └─ Conventionally: Theta usually negative for long calls (decay erodes value)
├─ Black-Scholes Formula:
│   ├─ Call theta = -S×N'(d1)×σ/(2√T) - r×K×e^{-rT}×N(d2)
│   ├─ Put theta = -S×N'(d1)×σ/(2√T) + r×K×e^{-rT}×N(-d2)
│   ├─ Where N'(d1) = normal PDF
│   ├─ Properties:
│   │   ├─ Usually negative for calls (time decay)
│   │   ├─ Usually positive for ITM puts (intrinsic value dominates)
│   │   ├─ Accelerates near expiry (√T → 0)
│   │   └─ Zero for ATM options at very long maturity (time effect minimal)
├─ Gamma-Theta Relationship:
│   ├─ Theta ≈ -γ×S²×σ²/2 - r×V (approximation; gamma decay + interest cost)
│   ├─ For ATM option: γ effect dominates; theta ≈ -γ×S²×σ²/2
│   ├─ Long gamma → negative theta (pay for convexity)
│   ├─ Short gamma → positive theta (collect for risk)
│   └─ Breakeven: At what realized vol does gamma P&L offset theta loss?
├─ Time Dependencies:
│   ├─ Acceleration: |θ| increases as T → 0 (√T in denominator)
│   ├─ Scaling: √T term → theta ∝ 1/√T near expiry
│   ├─ ATM vs OTM: ATM theta most negative (highest time value at risk)
│   ├─ ITM options: Decay slower than OTM (less time value)
│   └─ Deep ITM: Theta → intrinsic value (little decay left)
├─ Hedging Theta:
│   ├─ Long theta strategies: Sell premium (short straddles, call spreads)
│   ├─ Calendar spreads: Buy long-dated, sell short-dated → harvest theta differential
│   ├─ Cost: Long theta = short gamma (negative convexity)
│   ├─ Monitoring: Daily theta tracking; rebalance if position theta drifts
│   └─ Greeks change: Theta constantly changes with spot, vol, time
└─ Practical Metrics:
    ├─ Theta decay curve: Shows θ vs time; accelerates exponentially
    ├─ Dollar theta: Actual daily decay in position value
    ├─ Percentage theta: θ as % of option price (decay rate)
    ├─ Theta per gamma: Efficiency of theta collection relative to gamma risk
    └─ Rolling strategies: Extend theta by rolling positions forward
```

**Interaction:** Time passes → option loses time value → theta decay → selling premium profitable if vol stays low

## 5. Mini-Project
Implement theta computation and calendar spread analysis:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes components
def bs_d1(S, K, T, r, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return d1 - sigma*np.sqrt(T)

def bs_call(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def bs_put(S, K, T, r, sigma):
    call = bs_call(S, K, T, r, sigma)
    put = call - S + K*np.exp(-r*T)
    return put

def theta_call_bs(S, K, T, r, sigma):
    """Theta per year for call"""
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) - 
             r * K * np.exp(-r*T) * norm.cdf(d2))
    return theta

def theta_put_bs(S, K, T, r, sigma):
    """Theta per year for put"""
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) + 
             r * K * np.exp(-r*T) * norm.cdf(-d2))
    return theta

def gamma_bs(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Parameters
S0, K, r, sigma = 100, 100, 0.05, 0.2

# 1. Theta across time (for various spot prices)
print("=== THETA ANALYSIS ===")
T_values = np.array([365, 90, 30, 7, 1]) / 365  # Days to years
print("\nCall Theta (per day) across time to expiry:")
print("Days\tS=90\t\tS=100\t\tS=110")
print("-" * 50)
for T in T_values:
    theta_90 = theta_call_bs(90, K, T, r, sigma) / 365
    theta_100 = theta_call_bs(100, K, T, r, sigma) / 365
    theta_110 = theta_call_bs(110, K, T, r, sigma) / 365
    days = T * 365
    print(f"{days:3.0f}\t${theta_90:.4f}\t\t${theta_100:.4f}\t\t${theta_110:.4f}")

# 2. Theta vs spot for different expirations
spot_prices = np.linspace(80, 120, 100)
T_scenarios = [1, 0.25, 0.083, 0.027]  # 1yr, 3mo, 1mo, 1 week

# 3. Calendar spread analysis
print("\n=== CALENDAR SPREAD (BUY LONG, SELL SHORT) ===")

# Buy long-dated call (T=1yr), sell short-dated call (T=3mo)
T_long = 1.0
T_short = 0.25

# At initiation
call_long_price = bs_call(S0, K, T_long, r, sigma)
call_short_price = bs_call(S0, K, T_short, r, sigma)
calendar_cost = call_long_price - call_short_price

print(f"\nInitial Setup:")
print(f"  Buy 1yr call: ${call_long_price:.2f}")
print(f"  Sell 3mo call: ${call_short_price:.2f}")
print(f"  Net cost: ${calendar_cost:.2f}")

# Theta of calendar spread
theta_long = theta_call_bs(S0, K, T_long, r, sigma)
theta_short = theta_call_bs(S0, K, T_short, r, sigma)
calendar_theta = theta_long - theta_short

print(f"\nTheta Analysis:")
print(f"  Long call theta (per year): ${theta_long:.2f}")
print(f"  Short call theta (per year): ${theta_short:.2f}")
print(f"  Calendar spread theta: ${calendar_theta:.2f}")
print(f"  Daily theta decay: ${calendar_theta/365:.4f}")

# Simulate calendar spread P&L over time (assuming spot stays at S0)
time_steps = np.linspace(T_short, 0.01, 50)
calendar_values = []
spot_constant = S0

for t_elapsed in np.linspace(0, T_short, 50):
    T_long_rem = T_long - t_elapsed
    T_short_rem = T_short - t_elapsed
    
    call_long_rem = bs_call(spot_constant, K, T_long_rem, r, sigma) if T_long_rem > 0 else max(spot_constant - K, 0)
    call_short_rem = bs_call(spot_constant, K, T_short_rem, r, sigma) if T_short_rem > 0 else max(spot_constant - K, 0)
    
    spread_value = call_long_rem - call_short_rem
    calendar_values.append(spread_value)

calendar_pnl = np.array(calendar_values) - calendar_cost

# Theta-Gamma tradeoff
print("\n=== THETA-GAMMA TRADEOFF ===")
print("Daily theta gain offset by gamma loss at different realized volatilities:")

time_periods = np.array([1, 7, 30])  # Days

for days in time_periods:
    T_rem = max(T_short - days/365, 0.01)
    
    # Theta gain: daily theta decay
    daily_theta = calendar_theta / 365
    
    # Gamma loss: from spot move
    gamma_long = gamma_bs(S0, K, T_long - days/365, r, sigma)
    gamma_short = gamma_bs(S0, K, T_short - days/365, r, sigma)
    calendar_gamma = gamma_long - gamma_short
    
    # Breakeven spot move where gamma loss = theta gain
    if calendar_gamma != 0:
        breakeven_move = np.sqrt(2 * abs(daily_theta) / calendar_gamma)
    else:
        breakeven_move = np.inf
    
    print(f"  Day {days}: Theta ${daily_theta:.4f}, Gamma {calendar_gamma:.6f}, " +
          f"Breakeven move ${breakeven_move:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Theta vs Spot (multiple expirations)
for T in T_scenarios:
    thetas = [theta_call_bs(S, K, T, r, sigma)/365 for S in spot_prices]
    label = f'T={T*365:.0f}d'
    axes[0, 0].plot(spot_prices, thetas, linewidth=2, label=label)

axes[0, 0].axvline(K, color='r', linestyle='--', alpha=0.5)
axes[0, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[0, 0].set_xlabel('Spot Price ($)')
axes[0, 0].set_ylabel('Theta (per day)')
axes[0, 0].set_title('Call Theta vs Spot (different expirations)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Theta acceleration near expiry
T_all = np.linspace(1, 0.01, 200)
theta_atm = [theta_call_bs(K, K, t, r, sigma)/365 for t in T_all]

axes[0, 1].plot(T_all*365, theta_atm, linewidth=2, color='red')
axes[0, 1].fill_between(T_all*365, theta_atm, alpha=0.2, color='red')
axes[0, 1].set_xlabel('Days to Expiry')
axes[0, 1].set_ylabel('Theta (per day)')
axes[0, 1].set_title('ATM Call Theta Acceleration')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Calendar spread P&L
t_elapsed = np.linspace(0, T_short*365, len(calendar_pnl))
axes[1, 0].plot(t_elapsed, calendar_pnl, linewidth=2, color='green')
axes[1, 0].fill_between(t_elapsed, calendar_pnl, alpha=0.2, color='green')
axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].set_xlabel('Days Elapsed')
axes[1, 0].set_ylabel('Calendar Spread P&L ($)')
axes[1, 0].set_title('Calendar Spread P&L (spot constant)')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Theta vs Gamma scatter
gammas_plot = []
thetas_plot = []
spots_plot = np.linspace(80, 120, 40)

for S in spots_plot:
    gamma = gamma_bs(S, K, T_short, r, sigma)
    theta = theta_call_bs(S, K, T_short, r, sigma) / 365
    gammas_plot.append(gamma)
    thetas_plot.append(theta)

scatter = axes[1, 1].scatter(gammas_plot, thetas_plot, c=spots_plot, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='k')
axes[1, 1].set_xlabel('Gamma')
axes[1, 1].set_ylabel('Theta (per day)')
axes[1, 1].set_title('Theta-Gamma Tradeoff (3mo call)')
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Spot Price ($)')

# Add breakeven line (approximate)
gamma_range = np.linspace(min(gammas_plot), max(gammas_plot), 100)
theta_breakeven = -gamma_range * S0**2 * sigma**2 / 2 / 365
axes[1, 1].plot(gamma_range, theta_breakeven, 'r--', linewidth=2, alpha=0.5, label='Breakeven')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does theta analysis fail?
- Dividend drops: Ex-dividend date → sudden price drop; call theta can spike negative
- Earnings announcements: Volatility jumps break theta model (BS assumes constant vol)
- Interest rate changes: Theta = f(r); rate changes affect call theta calculation
- Non-calendar spreads: Rolling positions changes moneyness; theta is path-dependent
- Bid-ask spread: Theta profit can be erased by transaction costs on frequent rehedging

## 7. Key References
- [Hull - Options, Futures & Derivatives (Chapter 19)](https://www-2.rotman.utoronto.ca/~hull)
- [Natenberg - Option Volatility & Pricing (Chapter 12)](https://www.amazon.com/Option-Volatility-Pricing-Advanced-Strategies/dp/1557784124)
- [Wilmott - Quantitative Finance (Chapter 7)](https://www.paulwilmott.com)

---
**Status:** Time decay metric | **Complements:** Gamma, Vega, Calendar Spreads, Greeks Dynamics
