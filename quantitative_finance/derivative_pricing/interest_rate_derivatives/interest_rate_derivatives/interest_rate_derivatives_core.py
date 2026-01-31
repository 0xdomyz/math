import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# Block 1

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
