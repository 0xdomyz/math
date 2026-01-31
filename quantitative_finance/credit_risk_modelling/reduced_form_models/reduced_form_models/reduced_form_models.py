# Auto-extracted from markdown file
# Source: reduced_form_models.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(42)

# Market data: Credit spreads for different maturities
print("=== Reduced-Form PD from Market Spreads ===")

spreads_market = pd.DataFrame({
    'Maturity': [0.5, 1, 2, 3, 5, 7, 10],
    'CDS_Spread_bps': [80, 120, 160, 200, 250, 280, 300],
    'Bond_Spread_bps': [100, 140, 180, 220, 280, 310, 330]
})

print("Market CDS and Bond Spreads:")
print(spreads_market)

# Assumptions
coupon = 0.04  # 4% coupon
face_value = 100
recovery_rate = 0.40
risk_free_rate = 0.03  # 3% risk-free rate

# CDS-based PD extraction
# Simplified: Annual CDS spread ≈ λ × LGD (annualized)
# More precise: Solve bond pricing equation

def extract_hazard_rate_from_cds(cds_bps, recovery):
    """
    Simplified extraction: λ = CDS_spread / LGD
    More accurate requires iterative solving bond pricing equation
    """
    lgd = 1 - recovery
    hazard_rate = (cds_bps / 10000) / lgd
    return hazard_rate

spreads_market['LGD'] = 1 - recovery_rate
spreads_market['Hazard_Rate'] = spreads_market['CDS_Spread_bps'].apply(
    lambda x: extract_hazard_rate_from_cds(x, recovery_rate)
)

# Convert hazard rate to survival probability and PD
spreads_market['Survival_Prob'] = np.exp(-spreads_market['Hazard_Rate'] * spreads_market['Maturity'])
spreads_market['PD_Cumulative'] = 1 - spreads_market['Survival_Prob']

print("\n=== PD Extraction ===")
print(spreads_market[['Maturity', 'CDS_Spread_bps', 'Hazard_Rate', 'PD_Cumulative']])

# Forward PD (annual default rate)
spreads_market['Forward_PD'] = spreads_market['Hazard_Rate']  # Approximate

print("\n=== Forward Default Rates ===")
for idx, row in spreads_market.iterrows():
    print(f"Year {row['Maturity']:.1f}: Forward PD = {row['Forward_PD']:.2%}, Cumulative PD = {row['PD_Cumulative']:.2%}")

# Stochastic intensity model: Vasicek framework
# λ(t) = α + β × Y(t)
# where Y(t) follows OU process

print("\n=== Stochastic Intensity Model ===")

# Fit intensity curve to market data
maturity_range = np.linspace(0.5, 10, 50)

# Fit polynomial to hazard rates
from np.polynomial import Polynomial
p = Polynomial.fit(spreads_market['Maturity'], spreads_market['Hazard_Rate'], 2)
hazard_fitted = p(maturity_range)

# Interpolate for pricing
hazard_interp = interp1d(spreads_market['Maturity'], spreads_market['Hazard_Rate'], 
                        kind='cubic', fill_value='extrapolate')

# Bond price calculation with fitted hazard rates
def bond_price_reduced_form(maturity, coupon, face, hazard_rate_func, rf_rate, recovery):
    """
    Bond price = E[PV of expected cash flows]
    Includes credit loss when default occurs
    """
    # Simple case: constant hazard rate over period
    hazard_avg = np.mean([hazard_rate_func(t) for t in np.linspace(0, maturity, 10)])
    survival = np.exp(-hazard_avg * maturity)
    
    # Bond value = coupon annuity × survival + recovery × default loss
    bond_value = 0
    discount_factors = [(1 + rf_rate) ** -t for t in range(1, int(maturity) + 1)]
    
    # Simplification: approximate as single lump sum
    pv_coupons = coupon * face * (1 - np.exp(-rf_rate * maturity)) / rf_rate
    pv_principal = face * np.exp(-rf_rate * maturity)
    
    bond_value = (pv_coupons + pv_principal) * survival + recovery * face * (1 - survival) * np.exp(-rf_rate * maturity)
    
    return bond_value

# Simulate paths of hazard rate (Vasicek-type)
dt = 0.01
n_steps = int(10 / dt)
n_paths = 1000

# Vasicek: dλ = κ(θ - λ)dt + σ dW
kappa = 0.3  # Mean reversion speed
theta = 0.15  # Long-run mean hazard rate
sigma = 0.05  # Volatility of hazard rate

lambda_paths = np.zeros((n_paths, n_steps + 1))
lambda_paths[:, 0] = spreads_market['Hazard_Rate'].iloc[0]

np.random.seed(42)
for step in range(n_steps):
    dW = np.random.normal(0, np.sqrt(dt), n_paths)
    lambda_paths[:, step+1] = (lambda_paths[:, step] + 
                              kappa * (theta - lambda_paths[:, step]) * dt +
                              sigma * dW)
    lambda_paths[:, step+1] = np.maximum(lambda_paths[:, step+1], 0)  # No negative rates

# Calculate survival probabilities from simulated paths
time_axis = np.arange(n_steps + 1) * dt
survival_paths = np.exp(-np.cumsum(lambda_paths * dt, axis=1))

# Default probability at each time
default_prob_simulated = 1 - survival_paths.mean(axis=0)

print("\nStochastic Intensity Simulation:")
print(f"Initial hazard rate: {lambda_paths[0, 0]:.2%}")
print(f"Mean long-run hazard: {lambda_paths[:, -1].mean():.2%}")
print(f"Std of final hazard rate: {lambda_paths[:, -1].std():.2%}")

# Correlation analysis: Spreads across maturities
print("\n=== Term Structure of Credit Risk ===")
print("Maturity | CDS Spread | Bond Spread | Spread Difference")
print("-" * 55)
for idx, row in spreads_market.iterrows():
    diff = row['Bond_Spread_bps'] - row['CDS_Spread_bps']
    print(f"{row['Maturity']:6.1f}Y  | {row['CDS_Spread_bps']:9d} | {row['Bond_Spread_bps']:10d} | {diff:17d}")

print("\nAsset swap spread (Bond - CDS) reflects liquidity/basis risk")

# Multi-name correlation: CDS index
print("\n=== CDS Index Modeling ===")
n_names = 125  # Standard CDX/iTraxx size
individual_spreads = np.random.normal(150, 80, n_names)
individual_spreads = np.maximum(individual_spreads, 10)  # Floor at 10 bps

index_spread = np.mean(individual_spreads)
index_std = np.std(individual_spreads)

print(f"Number of names: {n_names}")
print(f"Index spread: {index_spread:.0f} bps")
print(f"Std of individual spreads: {index_std:.0f} bps")
print(f"Implied correlation: {(index_spread / np.mean(individual_spreads)):.2f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Term structure of CDS spreads
ax1 = axes[0, 0]
ax1.plot(spreads_market['Maturity'], spreads_market['CDS_Spread_bps'], 'o-', 
        linewidth=2, markersize=8, label='CDS Spreads')
ax1.plot(spreads_market['Maturity'], spreads_market['Bond_Spread_bps'], 's-', 
        linewidth=2, markersize=8, label='Bond Spreads')
ax1.fill_between(spreads_market['Maturity'], spreads_market['CDS_Spread_bps'], 
                 spreads_market['Bond_Spread_bps'], alpha=0.2)
ax1.set_xlabel('Maturity (Years)')
ax1.set_ylabel('Spread (bps)')
ax1.set_title('Term Structure of Credit Spreads')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative PD by maturity
ax2 = axes[0, 1]
ax2.plot(spreads_market['Maturity'], spreads_market['PD_Cumulative']*100, 'o-', 
        linewidth=2, markersize=8)
ax2.fill_between(spreads_market['Maturity'], 0, spreads_market['PD_Cumulative']*100, alpha=0.2)
ax2.set_xlabel('Maturity (Years)')
ax2.set_ylabel('Cumulative PD (%)')
ax2.set_title('Cumulative Probability of Default\n(Backed out from CDS spreads)')
ax2.grid(True, alpha=0.3)

# Plot 3: Hazard rate term structure
ax3 = axes[0, 2]
ax3.plot(spreads_market['Maturity'], spreads_market['Hazard_Rate']*100, 'o-', 
        linewidth=2, markersize=8, label='Market-implied')
ax3.plot(maturity_range, hazard_fitted*100, '--', linewidth=2, label='Fitted curve')
ax3.set_xlabel('Maturity (Years)')
ax3.set_ylabel('Hazard Rate (%)')
ax3.set_title('Hazard Rate Term Structure')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Stochastic intensity paths
ax4 = axes[1, 0]
for i in range(min(100, n_paths)):
    ax4.plot(time_axis, lambda_paths[i, :]*100, alpha=0.1, color='blue')
ax4.plot(time_axis, lambda_paths.mean(axis=0)*100, 'r-', linewidth=2, label='Mean path')
ax4.set_xlabel('Time (Years)')
ax4.set_ylabel('Hazard Rate (%)')
ax4.set_title('Stochastic Intensity Paths\n(Vasicek model)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Survival probability paths
ax5 = axes[1, 1]
for i in range(min(100, n_paths)):
    ax5.plot(time_axis, survival_paths[i, :], alpha=0.1, color='green')
ax5.plot(time_axis, survival_paths.mean(axis=0), 'r-', linewidth=2, label='Expected survival')
ax5.set_xlabel('Time (Years)')
ax5.set_ylabel('Survival Probability')
ax5.set_title('Simulated Survival Paths')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Individual name spreads distribution
ax6 = axes[1, 2]
ax6.hist(individual_spreads, bins=20, edgecolor='black', alpha=0.7)
ax6.axvline(index_spread, color='r', linestyle='--', linewidth=2, label=f'Index = {index_spread:.0f} bps')
ax6.axvline(np.median(individual_spreads), color='orange', linestyle='--', linewidth=2, 
           label=f'Median = {np.median(individual_spreads):.0f} bps')
ax6.set_xlabel('CDS Spread (bps)')
ax6.set_ylabel('Number of Names')
ax6.set_title(f'CDS Index Composition\n(n={n_names} names)')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n=== Reduced-Form Summary ===")
print(f"Approach: Market prices → Hazard rate → PD")
print(f"Advantage: Forward-looking, based on real transactions")
print(f"Limitation: Sensitive to market liquidity, bid-ask spreads")

