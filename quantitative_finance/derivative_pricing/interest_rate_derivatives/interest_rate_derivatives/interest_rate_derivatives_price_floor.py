import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
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