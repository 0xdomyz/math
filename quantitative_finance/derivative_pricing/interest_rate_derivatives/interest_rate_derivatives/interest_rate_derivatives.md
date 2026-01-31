# Interest Rate Derivatives

## 1. Concept Skeleton
**Definition:** Financial contracts whose value depends on future interest rates, including caps, floors, swaptions, and bonds, priced under term structure models  
**Purpose:** Hedge interest rate risk, speculate on rate movements, manage duration exposure in portfolios  
**Prerequisites:** Term structure models, bond math, Black formula for caps/floors, LIBOR market model

## 2. Comparative Framing
| Product | Cap | Floor | Swaption | Interest Rate Swap |
|---------|-----|-------|----------|-------------------|
| **Payoff** | Max(L-K,0) per period | Max(K-L,0) per period | Option to enter swap | Fixed vs floating exchange |
| **Use Case** | Protect against rising rates | Protect against falling rates | Hedge future swap entry | Convert fixed/floating exposure |
| **Pricing Model** | Black '76 formula (caplet sum) | Black '76 formula (floorlet sum) | Black formula or LMM | Discount cash flows to PV |
| **Volatility Input** | Cap volatility surface | Floor volatility surface | Swaption vol matrix | N/A (deterministic) |

## 3. Examples + Counterexamples

**Simple Example:**  
3-year cap on 3M LIBOR, strike 3%, notional $10M → quarterly payoffs Max(LIBOR_t - 3%, 0) × 0.25 × $10M, priced as sum of 12 caplets

**Failure Case:**  
Using Black-Scholes for swaptions with constant volatility → ignores term structure dynamics, volatility smile → SABR or LMM required for accuracy

**Edge Case:**  
Negative interest rates (EUR, JPY 2015-2020): Black formula breaks down (assumes lognormal rates) → shifted lognormal or normal model needed

## 4. Layer Breakdown
```
Interest Rate Derivatives:
├─ Foundational Products:
│   ├─ Interest Rate Swap (IRS):
│   │   ├─ Fixed Leg: ∑ K × τᵢ × DF(tᵢ) (K = fixed rate)
│   │   ├─ Floating Leg: ∑ L(tᵢ₋₁,tᵢ) × τᵢ × DF(tᵢ) (L = LIBOR/SOFR)
│   │   ├─ Swap Value: V_swap = V_float - V_fixed (receiver pays fixed)
│   │   ├─ Par Swap Rate: K such that V_swap = 0 at inception
│   │   └─ Use: Convert floating exposure to fixed (or vice versa)
│   ├─ Forward Rate Agreement (FRA):
│   │   ├─ Payoff: (L - K) × τ × N × DF(T) (settled at T)
│   │   ├─ Forward Rate: F(t,T,T+τ) = [DF(T)/DF(T+τ) - 1]/τ
│   │   └─ Use: Lock in future borrowing/lending rate
│   └─ Zero-Coupon Bond:
│       ├─ Price: P(t,T) = DF(T) = e^(-r(T-t)(T-t)) (continuous)
│       └─ Yield: y = -ln(P)/T (zero rate)
├─ Caps and Floors:
│   ├─ Interest Rate Cap:
│   │   ├─ Definition: Portfolio of caplets, each pays Max(L-K,0) × τ × N
│   │   ├─ Caplet Pricing (Black '76):
│   │   │   Caplet(T) = N × τ × DF(T) × [F × Φ(d₁) - K × Φ(d₂)]
│   │   │   where F = forward LIBOR, d₁ = [ln(F/K) + ½σ²T]/(σ√T), d₂ = d₁ - σ√T
│   │   ├─ Cap Value: Sum of all caplet values
│   │   ├─ Cap Volatility Surface: σ_cap(K,T) varies by strike and maturity
│   │   └─ Use: Protection against rising rates (borrower buys cap)
│   ├─ Interest Rate Floor:
│   │   ├─ Floorlet: Max(K-L,0) × τ × N per period
│   │   ├─ Pricing: Similar to caplet, Φ(-d₂) and Φ(-d₁) in formula
│   │   └─ Use: Protection against falling rates (lender buys floor)
│   └─ Collar:
│       ├─ Long Cap + Short Floor (or vice versa)
│       ├─ Zero-Cost Collar: Choose strikes so cap premium = floor premium
│       └─ Limits interest rate exposure to [K_floor, K_cap] range
├─ Swaptions:
│   ├─ Definition: Option to enter interest rate swap at future date
│   │   ├─ Payer Swaption: Right to pay fixed (receive floating)
│   │   ├─ Receiver Swaption: Right to receive fixed (pay floating)
│   │   └─ European vs Bermudan: Single vs multiple exercise dates
│   ├─ Pricing (Black Formula):
│   │   V_payer = A × DF(T₀) × [S × Φ(d₁) - K × Φ(d₂)]
│   │   where S = forward swap rate, K = strike, A = annuity factor
│   │   A = ∑ τᵢ × DF(tᵢ) (PV of $1 per period)
│   ├─ Swaption Volatility Matrix:
│   │   ├─ Rows: Option expiry (1Y, 2Y, 5Y, 10Y, ...)
│   │   ├─ Columns: Swap tenor (1Y, 5Y, 10Y, 30Y, ...)
│   │   └─ Notation: 2Y5Y swaption = option expiring in 2Y on 5Y swap
│   ├─ Use Cases:
│   │   ├─ Hedge callable bonds (issuer has option to prepay)
│   │   ├─ Monetize rate views without immediate swap commitment
│   │   └─ Portfolio immunization strategies
│   └─ Advanced Pricing: LIBOR Market Model for Bermudan swaptions
├─ Term Structure Models:
│   ├─ Short-Rate Models:
│   │   ├─ Vasicek: dr = a(b-r)dt + σdW (mean-reverting, Gaussian)
│   │   ├─ Cox-Ingersoll-Ross (CIR): dr = a(b-r)dt + σ√r dW (non-negative)
│   │   ├─ Hull-White: dr = [θ(t) - ar]dt + σdW (time-dependent mean reversion)
│   │   └─ Calibration: Fit to current yield curve, then price derivatives
│   ├─ LIBOR Market Model (LMM):
│   │   ├─ Model forward LIBOR rates directly: dLᵢ/Lᵢ = σᵢdWᵢ
│   │   ├─ Advantages: Market-consistent (match cap/floor prices), lognormal rates
│   │   ├─ Calibration: Match cap volatilities across strikes and maturities
│   │   └─ Simulation: Monte Carlo for exotic derivatives, Bermudan swaptions
│   └─ Heath-Jarrow-Morton (HJM):
│       ├─ Model forward rate curve evolution: df(t,T) = α(t,T)dt + σ(t,T)dW
│       ├─ No-arbitrage drift: α(t,T) = σ(t,T)∫ᵗᵀ σ(t,s)ds
│       └─ Flexible but high-dimensional (infinite-dimensional system)
├─ Convexity Adjustments:
│   ├─ Timing Mismatch: Payment at T₂, fixing at T₁ (T₁ < T₂)
│   │   ├─ Adjustment: E^T₂[L(T₁,T₂)] ≠ F(0,T₁,T₂) (forward rate)
│   │   └─ Formula: Adjusted forward = F × (1 + ½σ²T₁τ/(1+Fτ))
│   ├─ Constant Maturity Swap (CMS):
│   │   ├─ Floating leg pays swap rate (not LIBOR)
│   │   └─ Requires convexity adjustment due to nonlinearity
│   └─ In-Arrears Swaps: Fixing and payment same date → convexity correction
└─ Risk Management:
    ├─ Duration: ∂P/∂y sensitivity to parallel yield shift
    ├─ Key Rate Duration: Sensitivity to specific maturity buckets
    ├─ DV01 (Dollar Value of 1bp): Change in value for 1bp rate change
    └─ Vega: Sensitivity to volatility changes (caps, swaptions)
```

**Interaction:** Construct yield curve → Calculate forward rates → Price caplets/swaptions using Black '76 → Sum to get cap/swaption value

## 5. Mini-Project
Price interest rate derivatives (caps, floors, swaptions) with market data:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# =====================================
# YIELD CURVE CONSTRUCTION
# =====================================
print("="*70)
print("INTEREST RATE DERIVATIVES PRICING")
print("="*70)

# Market data: Zero rates (continuously compounded)
market_data = {
    'Maturity': [0.25, 0.5, 1, 2, 3, 5, 7, 10],
    'ZeroRate': [0.025, 0.028, 0.030, 0.032, 0.034, 0.036, 0.037, 0.038]
}
yield_curve = pd.DataFrame(market_data)

# Interpolate yield curve
zero_rate_interp = interp1d(yield_curve['Maturity'], yield_curve['ZeroRate'], 
                             kind='cubic', fill_value='extrapolate')

def discount_factor(T):
    """Calculate discount factor DF(T) = e^(-r*T)."""
    r = zero_rate_interp(T)
    return np.exp(-r * T)

def forward_rate(T1, T2):
    """
    Calculate forward rate F(T1,T2) from discount factors.
    F = [DF(T1)/DF(T2) - 1] / (T2-T1)
    """
    df1 = discount_factor(T1)
    df2 = discount_factor(T2)
    tau = T2 - T1
    forward = (df1 / df2 - 1) / tau
    return forward

print("\nYield Curve (Zero Rates):")
print(yield_curve.to_string(index=False))

# Calculate forward rates
print("\nForward Rates:")
for i in range(len(yield_curve) - 1):
    T1 = yield_curve.loc[i, 'Maturity']
    T2 = yield_curve.loc[i+1, 'Maturity']
    fwd = forward_rate(T1, T2)
    print(f"   F({T1:.2f}, {T2:.2f}) = {fwd:.4%}")

# =====================================
# CAP PRICING (BLACK '76 FORMULA)
# =====================================
print("\n" + "="*70)
print("INTEREST RATE CAP PRICING")
print("="*70)

def black_76_caplet(F, K, T, sigma, tau, N, df):
    """
    Price a single caplet using Black '76 formula.
    
    Caplet = N × τ × DF(T) × [F × Φ(d₁) - K × Φ(d₂)]
    
    Parameters:
    - F: Forward rate
    - K: Strike rate
    - T: Time to caplet expiry (years)
    - sigma: Volatility
    - tau: Period length (e.g., 0.25 for quarterly)
    - N: Notional
    - df: Discount factor DF(T)
    """
    if T <= 0:
        # Expired caplet
        return max(F - K, 0) * tau * N * df
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    caplet_value = N * tau * df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return caplet_value

def price_cap(K, maturity, sigma, N=1e6, freq=4):
    """
    Price an interest rate cap as portfolio of caplets.
    
    Parameters:
    - K: Strike rate (cap rate)
    - maturity: Cap maturity in years
    - sigma: Flat volatility assumption
    - N: Notional
    - freq: Payment frequency (4 = quarterly)
    """
    tau = 1.0 / freq  # Period length
    n_periods = int(maturity * freq)
    
    cap_value = 0
    caplet_details = []
    
    for i in range(1, n_periods + 1):
        T_start = (i - 1) * tau
        T_end = i * tau
        T_fixing = T_start  # Fixing at start of period
        
        # Forward rate for this period
        F = forward_rate(T_start, T_end)
        
        # Discount factor to payment date
        df = discount_factor(T_end)
        
        # Price caplet
        caplet_val = black_76_caplet(F, K, T_fixing, sigma, tau, N, df)
        cap_value += caplet_val
        
        caplet_details.append({
            'Period': i,
            'Fixing': T_fixing,
            'Payment': T_end,
            'Forward_Rate': F,
            'Caplet_Value': caplet_val
        })
    
    return cap_value, pd.DataFrame(caplet_details)

# Example: 3-year cap
K_cap = 0.035  # 3.5% strike
maturity_cap = 3.0
sigma_cap = 0.20  # 20% volatility
N = 10e6  # $10 million notional

cap_value, caplet_df = price_cap(K_cap, maturity_cap, sigma_cap, N)

print(f"\nCap Parameters:")
print(f"   Strike: {K_cap:.2%}")
print(f"   Maturity: {maturity_cap} years")
print(f"   Volatility: {sigma_cap:.1%}")
print(f"   Notional: ${N/1e6:.1f}M")
print(f"   Frequency: Quarterly")

print(f"\nCap Value: ${cap_value:,.2f}")
print(f"Cap Premium (bps of notional): {cap_value/N * 10000:.1f} bps")

print("\nCaplet Breakdown (first 4 periods):")
print(caplet_df.head(4).to_string(index=False))

# =====================================
# FLOOR PRICING
# =====================================
print("\n" + "="*70)
print("INTEREST RATE FLOOR PRICING")
print("="*70)

def black_76_floorlet(F, K, T, sigma, tau, N, df):
    """Price a single floorlet using Black '76 formula."""
    if T <= 0:
        return max(K - F, 0) * tau * N * df
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    floorlet_value = N * tau * df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return floorlet_value

def price_floor(K, maturity, sigma, N=1e6, freq=4):
    """Price an interest rate floor as portfolio of floorlets."""
    tau = 1.0 / freq
    n_periods = int(maturity * freq)
    
    floor_value = 0
    
    for i in range(1, n_periods + 1):
        T_start = (i - 1) * tau
        T_end = i * tau
        T_fixing = T_start
        
        F = forward_rate(T_start, T_end)
        df = discount_factor(T_end)
        
        floorlet_val = black_76_floorlet(F, K, T_fixing, sigma, tau, N, df)
        floor_value += floorlet_val
    
    return floor_value

K_floor = 0.025  # 2.5% strike
floor_value = price_floor(K_floor, maturity_cap, sigma_cap, N)

print(f"\nFloor Parameters:")
print(f"   Strike: {K_floor:.2%}")
print(f"   Maturity: {maturity_cap} years")
print(f"   Volatility: {sigma_cap:.1%}")

print(f"\nFloor Value: ${floor_value:,.2f}")
print(f"Floor Premium (bps): {floor_value/N * 10000:.1f} bps")

# Put-Call Parity for Caps and Floors
# Cap - Floor = Swap (fixed vs floating)
swap_value_implied = cap_value - floor_value
print(f"\nPut-Call Parity Check:")
print(f"   Cap - Floor = ${swap_value_implied:,.2f}")

# =====================================
# SWAPTION PRICING
# =====================================
print("\n" + "="*70)
print("SWAPTION PRICING")
print("="*70)

def swap_rate(T_start, T_end, freq=2):
    """
    Calculate par swap rate (semi-annual payments).
    
    Swap Rate S = [DF(T_start) - DF(T_end)] / Annuity
    where Annuity = Σ τᵢ × DF(tᵢ)
    """
    tau = 1.0 / freq
    n_periods = int((T_end - T_start) * freq)
    
    # Calculate annuity (PV of $1 per period)
    annuity = 0
    for i in range(1, n_periods + 1):
        t = T_start + i * tau
        annuity += tau * discount_factor(t)
    
    # Par swap rate
    df_start = discount_factor(T_start)
    df_end = discount_factor(T_end)
    S = (df_start - df_end) / annuity
    
    return S, annuity

def price_swaption(T_expiry, swap_tenor, K, sigma, N=1e6, option_type='payer'):
    """
    Price European swaption using Black formula.
    
    Parameters:
    - T_expiry: Option expiry (years)
    - swap_tenor: Length of underlying swap (years)
    - K: Strike (fixed rate)
    - sigma: Swaption volatility
    - option_type: 'payer' (pay fixed) or 'receiver' (receive fixed)
    """
    T_start = T_expiry
    T_end = T_expiry + swap_tenor
    
    # Forward swap rate and annuity
    S, annuity = swap_rate(T_start, T_end)
    
    # Black formula for swaption
    if T_expiry <= 0:
        intrinsic = max((S - K) if option_type == 'payer' else (K - S), 0)
        return intrinsic * annuity * N
    
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T_expiry) / (sigma * np.sqrt(T_expiry))
    d2 = d1 - sigma * np.sqrt(T_expiry)
    
    if option_type == 'payer':
        swaption_value = N * annuity * (S * norm.cdf(d1) - K * norm.cdf(d2))
    else:  # receiver
        swaption_value = N * annuity * (K * norm.cdf(-d2) - S * norm.cdf(-d1))
    
    return swaption_value

# Example: 2Y5Y payer swaption (option expires in 2Y on 5Y swap)
T_expiry_sw = 2.0
swap_tenor = 5.0
K_swaption = 0.035  # 3.5% strike
sigma_swaption = 0.25  # 25% swaption volatility

swaption_value = price_swaption(T_expiry_sw, swap_tenor, K_swaption, sigma_swaption, N)

# Calculate forward swap rate
S_forward, annuity = swap_rate(T_expiry_sw, T_expiry_sw + swap_tenor)

print(f"\nSwaption Parameters:")
print(f"   Expiry: {T_expiry_sw} years")
print(f"   Swap Tenor: {swap_tenor} years")
print(f"   Strike: {K_swaption:.2%}")
print(f"   Volatility: {sigma_swaption:.1%}")
print(f"   Notional: ${N/1e6:.1f}M")

print(f"\nForward Swap Rate: {S_forward:.4%}")
print(f"Annuity Factor: {annuity:.4f}")
print(f"\n2Y5Y Payer Swaption Value: ${swaption_value:,.2f}")
print(f"Swaption Premium (bps): {swaption_value/N * 10000:.1f} bps")

# Receiver swaption
receiver_value = price_swaption(T_expiry_sw, swap_tenor, K_swaption, sigma_swaption, N, 'receiver')
print(f"\n2Y5Y Receiver Swaption Value: ${receiver_value:,.2f}")

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Yield Curve and Forward Rates
T_plot = np.linspace(0.1, 10, 100)
zero_rates_plot = [zero_rate_interp(t) * 100 for t in T_plot]

axes[0, 0].plot(T_plot, zero_rates_plot, linewidth=2, label='Zero Rates')
axes[0, 0].scatter(yield_curve['Maturity'], yield_curve['ZeroRate']*100, 
                   s=100, c='red', zorder=5, label='Market Points')

# Plot forward rates
T_fwd = np.linspace(0.1, 9.5, 50)
fwd_rates = [forward_rate(t, t+0.5) * 100 for t in T_fwd]
axes[0, 0].plot(T_fwd, fwd_rates, '--', linewidth=2, label='6M Forward Rates')

axes[0, 0].set_xlabel('Maturity (years)')
axes[0, 0].set_ylabel('Rate (%)')
axes[0, 0].set_title('Yield Curve and Forward Rates')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Caplet Values
axes[0, 1].bar(caplet_df['Period'], caplet_df['Caplet_Value']/1000, 
               alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Caplet Period')
axes[0, 1].set_ylabel('Caplet Value ($000s)')
axes[0, 1].set_title(f'Caplet Values (Strike {K_cap:.1%}, Total ${cap_value/1e6:.2f}M)')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Cap/Floor value vs Strike
strikes = np.linspace(0.02, 0.05, 20)
cap_values = [price_cap(k, maturity_cap, sigma_cap, N)[0]/1e6 for k in strikes]
floor_values = [price_floor(k, maturity_cap, sigma_cap, N)/1e6 for k in strikes]

axes[1, 0].plot(strikes*100, cap_values, linewidth=2, label='Cap', marker='o')
axes[1, 0].plot(strikes*100, floor_values, linewidth=2, label='Floor', marker='s')
axes[1, 0].axvline(K_cap*100, color='red', linestyle='--', alpha=0.5, label=f'Strike {K_cap:.1%}')
axes[1, 0].set_xlabel('Strike Rate (%)')
axes[1, 0].set_ylabel('Option Value ($M)')
axes[1, 0].set_title('Cap and Floor Values vs Strike')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Swaption value vs volatility
vols = np.linspace(0.10, 0.50, 30)
payer_values = [price_swaption(T_expiry_sw, swap_tenor, K_swaption, v, N)/1e6 for v in vols]
receiver_values = [price_swaption(T_expiry_sw, swap_tenor, K_swaption, v, N, 'receiver')/1e6 for v in vols]

axes[1, 1].plot(vols*100, payer_values, linewidth=2, label='Payer Swaption', marker='o')
axes[1, 1].plot(vols*100, receiver_values, linewidth=2, label='Receiver Swaption', marker='s')
axes[1, 1].axvline(sigma_swaption*100, color='red', linestyle='--', alpha=0.5, 
                   label=f'σ={sigma_swaption:.0%}')
axes[1, 1].set_xlabel('Volatility (%)')
axes[1, 1].set_ylabel('Swaption Value ($M)')
axes[1, 1].set_title(f'2Y{int(swap_tenor)}Y Swaption Vega')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Interest rate derivatives pricing complete:")
print(f"• Cap (strike {K_cap:.1%}): ${cap_value/1e6:.3f}M ({cap_value/N*10000:.0f} bps)")
print(f"• Floor (strike {K_floor:.1%}): ${floor_value/1e6:.3f}M ({floor_value/N*10000:.0f} bps)")
print(f"• 2Y5Y Payer Swaption: ${swaption_value/1e6:.3f}M ({swaption_value/N*10000:.0f} bps)")
print(f"• Forward swap rate: {S_forward:.3%} (strike {K_swaption:.1%}, {'ITM' if S_forward>K_swaption else 'OTM'})")
print(f"• Models: Black '76 for caps/floors, Black formula for swaptions")
```

## 6. Challenge Round
When does Black '76 formula fail for interest rate derivatives?
- **Negative rates:** Lognormal assumption breaks (ln(negative) undefined) → shifted lognormal or normal (Bachelier) model required
- **Volatility smile:** Flat vol assumption inaccurate, especially for deep OTM/ITM → SABR model captures smile dynamics
- **Bermudan swaptions:** Multiple exercise dates require Monte Carlo + optimal stopping (Longstaff-Schwartz) or tree methods
- **Correlation structure:** Multi-factor products (e.g., CMS spread options) need full LIBOR Market Model with factor correlations
- **Long-dated products:** Term structure of volatility matters, constant vol inadequate → time-dependent volatility σ(t,T)

Modern approaches: SABR for smile, LMM for complex path-dependent payoffs, Hull-White for callable bonds needing analytical tractability.

## 7. Key References
- [Brigo & Mercurio (2006) Interest Rate Models – Theory and Practice](https://link.springer.com/book/10.1007/978-3-540-34604-3) - Comprehensive LIBOR Market Model
- [Rebonato (2004) Volatility and Correlation in Interest Rate Derivatives](https://www.wiley.com/en-us/Volatility+and+Correlation%3A+The+Perfect+Hedger+and+the+Fox%2C+2nd+Edition-p-9780470091395) - Swaption volatility surface modeling
- [Hull (2018) Options, Futures, and Other Derivatives, Ch. 32](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) - Interest rate derivatives overview
- [Andersen & Piterbarg (2010) Interest Rate Modeling](https://www.atlanticfinancial.com/interest-rate-modeling-volumes-1-2-and-3-lech-grzelak-cornelis-oosterlee) - Advanced topics including CVA for rates

---
**Status:** Core fixed income derivative class | **Complements:** Black-Scholes, Monte Carlo, Term Structure Models
