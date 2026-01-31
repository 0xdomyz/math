import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.integrate import quad

# Block 1

# =====================================
# HESTON MODEL CALIBRATION
# =====================================
print("="*70)
print("HESTON MODEL CALIBRATION TO MARKET OPTIONS")
print("="*70)

# =====================================
# HESTON PRICING via Characteristic Function
# =====================================
def heston_characteristic_function(u, S0, v0, kappa, theta, sigma_v, rho, r, T):
    """
    Heston characteristic function φ(u) for log(S_T).
    Used in Fourier inversion for option pricing.
    """
    # Parameters
    lam = 0  # Risk premium (often set to 0)
    d = np.sqrt((rho * sigma_v * u * 1j - kappa)**2 + sigma_v**2 * (u * 1j + u**2))
    g = (kappa - rho * sigma_v * u * 1j - d) / (kappa - rho * sigma_v * u * 1j + d)
    
    # Characteristic exponents
    C = r * u * 1j * T + (kappa * theta / sigma_v**2) * \
        ((kappa - rho * sigma_v * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    
    D = ((kappa - rho * sigma_v * u * 1j - d) / sigma_v**2) * \
        ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    phi = np.exp(C + D * v0 + 1j * u * np.log(S0))
    return phi

def heston_call_price(S0, K, v0, kappa, theta, sigma_v, rho, r, T):
    """
    Price European call option under Heston model using Fourier inversion.
    
    C = S0 * P1 - K * exp(-rT) * P2
    where P1, P2 are probabilities computed via characteristic function.
    """
    # P1: Probability under stock measure
    def integrand_P1(u):
        phi = heston_characteristic_function(u - 1j, S0, v0, kappa, theta, sigma_v, rho, r, T)
        numerator = np.exp(-1j * u * np.log(K)) * phi
        denominator = 1j * u * S0
        return np.real(numerator / denominator)
    
    P1 = 0.5 + (1 / np.pi) * quad(integrand_P1, 0, 100, limit=100)[0]
    
    # P2: Probability under money market measure
    def integrand_P2(u):
        phi = heston_characteristic_function(u, S0, v0, kappa, theta, sigma_v, rho, r, T)
        numerator = np.exp(-1j * u * np.log(K)) * phi
        denominator = 1j * u
        return np.real(numerator / denominator)
    
    P2 = 0.5 + (1 / np.pi) * quad(integrand_P2, 0, 100, limit=100)[0]
    
    # Call price
    call_price = S0 * P1 - K * np.exp(-r * T) * P2
    return max(call_price, 0)  # Ensure non-negative

# =====================================
# MARKET DATA (Synthetic)
# =====================================
# Simulate "market" data with known Heston parameters
np.random.seed(42)

S0_true = 100.0
r_true = 0.05
v0_true = 0.04       # Initial variance (σ₀² = 20%²)
kappa_true = 2.0     # Mean reversion speed
theta_true = 0.04    # Long-term variance
sigma_v_true = 0.3   # Vol-of-vol
rho_true = -0.7      # Negative correlation (leverage effect)

# Generate market option grid
strikes = np.array([80, 90, 95, 100, 105, 110, 120])
maturities = np.array([0.25, 0.5, 1.0])

market_data = []
for T in maturities:
    for K in strikes:
        true_price = heston_call_price(S0_true, K, v0_true, kappa_true, theta_true, 
                                        sigma_v_true, rho_true, r_true, T)
        # Add small noise to simulate market imperfections
        market_price = true_price + np.random.normal(0, 0.05)
        
        # Calculate vega for weighting (approximate)
        vega_weight = S0_true * np.sqrt(T) * norm.pdf(norm.ppf(0.5))  # Simplified
        
        market_data.append({
            'Strike': K,
            'Maturity': T,
            'Market_Price': max(market_price, 0.01),
            'True_Price': true_price,
            'Vega_Weight': vega_weight
        })

market_df = pd.DataFrame(market_data)

print("\nMarket Option Data (Sample):")
print(market_df.head(10).to_string(index=False))
print(f"\nTotal instruments: {len(market_df)}")

# =====================================
# CALIBRATION OBJECTIVE FUNCTION
# =====================================
def calibration_objective(params, market_df, S0, r, weight_type='uniform'):
    """
    Calibration loss function: Sum of squared pricing errors.
    
    Parameters to calibrate: [v0, kappa, theta, sigma_v, rho]
    """
    v0, kappa, theta, sigma_v, rho = params
    
    # Feller condition check
    if 2 * kappa * theta < sigma_v**2:
        return 1e10  # Penalty for violating Feller condition
    
    total_error = 0
    for idx, row in market_df.iterrows():
        K = row['Strike']
        T = row['Maturity']
        market_price = row['Market_Price']
        
        try:
            model_price = heston_call_price(S0, K, v0, kappa, theta, sigma_v, rho, r, T)
            error = (model_price - market_price)**2
            
            # Apply weights
            if weight_type == 'vega':
                weight = row['Vega_Weight']
            else:
                weight = 1.0
            
            total_error += weight * error
        except:
            return 1e10  # Return large penalty for numerical failures
    
    return total_error

# =====================================
# CALIBRATION (Global Optimization)
# =====================================
print("\n" + "="*70)
print("CALIBRATION: Differential Evolution (Global Optimizer)")
print("="*70)

# Parameter bounds [v0, kappa, theta, sigma_v, rho]
bounds = [
    (0.01, 0.5),    # v0: Initial variance
    (0.1, 5.0),     # kappa: Mean reversion speed
    (0.01, 0.5),    # theta: Long-term variance
    (0.05, 1.0),    # sigma_v: Vol-of-vol
    (-0.99, -0.01)  # rho: Negative correlation (equity typical)
]

print("\nTrue Parameters (used to generate market data):")
print(f"   v0     = {v0_true:.4f}")
print(f"   kappa  = {kappa_true:.4f}")
print(f"   theta  = {theta_true:.4f}")
print(f"   sigma_v= {sigma_v_true:.4f}")
print(f"   rho    = {rho_true:.4f}")

print("\nCalibrating... (this may take 30-60 seconds)")

result = differential_evolution(
    calibration_objective,
    bounds,
    args=(market_df, S0_true, r_true, 'uniform'),
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    tol=1e-6,
    seed=42,
    disp=False
)

v0_cal, kappa_cal, theta_cal, sigma_v_cal, rho_cal = result.x

print("\nCalibrated Parameters:")
print(f"   v0     = {v0_cal:.4f}  (true: {v0_true:.4f}, error: {abs(v0_cal-v0_true):.4f})")
print(f"   kappa  = {kappa_cal:.4f}  (true: {kappa_true:.4f}, error: {abs(kappa_cal-kappa_true):.4f})")
print(f"   theta  = {theta_cal:.4f}  (true: {theta_true:.4f}, error: {abs(theta_cal-theta_true):.4f})")
print(f"   sigma_v= {sigma_v_cal:.4f}  (true: {sigma_v_true:.4f}, error: {abs(sigma_v_cal-sigma_v_true):.4f})")
print(f"   rho    = {rho_cal:.4f}  (true: {rho_true:.4f}, error: {abs(rho_cal-rho_true):.4f})")

print(f"\nFeller Condition Check: 2κθ = {2*kappa_cal*theta_cal:.4f} vs σ_v² = {sigma_v_cal**2:.4f}")
print(f"   {'✓ PASS' if 2*kappa_cal*theta_cal >= sigma_v_cal**2 else '✗ FAIL'}")

# =====================================
# VALIDATION: Model vs Market Prices
# =====================================
print("\n" + "="*70)
print("CALIBRATION FIT ANALYSIS")
print("="*70)

market_df['Model_Price'] = market_df.apply(
    lambda row: heston_call_price(S0_true, row['Strike'], v0_cal, kappa_cal, theta_cal,
                                   sigma_v_cal, rho_cal, r_true, row['Maturity']),
    axis=1
)

market_df['Pricing_Error'] = market_df['Model_Price'] - market_df['Market_Price']
market_df['Relative_Error'] = (market_df['Pricing_Error'] / market_df['Market_Price']) * 100

print("\nPricing Errors (Sample):")
print(market_df[['Strike', 'Maturity', 'Market_Price', 'Model_Price', 'Pricing_Error']].head(10).to_string(index=False))

rmse = np.sqrt(np.mean(market_df['Pricing_Error']**2))
mae = np.mean(np.abs(market_df['Pricing_Error']))
max_error = np.max(np.abs(market_df['Pricing_Error']))

print(f"\nCalibration Metrics:")
print(f"   RMSE:        ${rmse:.4f}")
print(f"   MAE:         ${mae:.4f}")
print(f"   Max Error:   ${max_error:.4f}")
print(f"   Mean Rel Error: {np.mean(np.abs(market_df['Relative_Error'])):.2f}%")

# =====================================
# IMPLIED VOLATILITY SURFACE
# =====================================