# Auto-extracted from markdown file
# Source: mortality_tables.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

# 1. SIMULATE INSURANCE CLAIMS DATA
np.random.seed(42)

# Generate realistic insurance experience
ages = np.arange(0, 101)
base_qx = 0.0001 + 0.00008 * np.exp(0.07 * ages)  # Gompertz base
base_qx = np.minimum(base_qx, 0.99999)

# Simulate claims
sample_size_per_age = 500  # 500 policies per age
deaths_per_age = np.random.binomial(sample_size_per_age, base_qx)
exposures = sample_size_per_age * np.ones(101)

# Create claims dataset
claims_data = pd.DataFrame({
    'Age': ages,
    'Deaths': deaths_per_age,
    'Exposures': exposures,
    'qx_empirical': deaths_per_age / exposures
})

print("SAMPLE CLAIMS DATA:")
print(claims_data[claims_data['Age'].isin([20, 40, 60, 80])].to_string(index=False))
print()

# 2. CRUDE (UNGRADUATED) MORTALITY RATES
qx_crude = claims_data['qx_empirical'].values

print("CRUDE RATES (high volatility):")
for age in [20, 40, 60, 80]:
    print(f"Age {age}: qx = {qx_crude[age]:.5f}")
print()

# 3. PARAMETRIC GRADUATION: Gompertz-Makeham
def gompertz_makeham(x, params):
    A, B, C = params
    return np.minimum(A + B * np.exp(C * x), 0.99999)

def log_likelihood_gm(params, ages_data, deaths, exposures):
    """Negative log-likelihood for Gompertz-Makeham"""
    A, B, C = params
    
    if A < 0 or B < 0 or C < 0 or A + B > 1:
        return 1e10
    
    qx = gompertz_makeham(ages_data, params)
    
    # Binomial log-likelihood
    ll = np.sum(deaths * np.log(qx + 1e-10) + 
                (exposures - deaths) * np.log(1 - qx + 1e-10))
    return -ll

# Fit on ages 20-95
mask_fit = (ages >= 20) & (ages <= 95)
ages_fit = ages[mask_fit]
deaths_fit = deaths_per_age[mask_fit]
exposures_fit = exposures[mask_fit]

p0 = [0.0001, 0.00008, 0.07]
result = minimize(log_likelihood_gm, p0,
                 args=(ages_fit, deaths_fit, exposures_fit),
                 method='Nelder-Mead')

params_gm = result.x
qx_parametric = gompertz_makeham(ages, params_gm)

print("PARAMETRIC (GOMPERTZ-MAKEHAM) FITTING:")
print(f"Parameters: A={params_gm[0]:.6f}, B={params_gm[1]:.6f}, C={params_gm[2]:.5f}")
print(f"Log-Likelihood: {-result.fun:.2f}")
print()

# 4. WHITTAKER-HENDERSON GRADUATION
# Minimize: SSE + λ * smoothness_penalty
def whittaker_henderson(qx_crude, exposures, lambda_smooth=1.0):
    """Whittaker-Henderson graduation"""
    n = len(qx_crude)
    
    # Log transformation for stability
    log_qx = np.log(np.maximum(qx_crude, 1e-6))
    
    # Difference matrix (2nd differences for smoothness)
    D2 = np.zeros((n-2, n))
    for i in range(n-2):
        D2[i, i] = 1
        D2[i, i+1] = -2
        D2[i, i+2] = 1
    
    def objective(log_qx_smooth):
        # Weighted sum of squared errors (more weight to high exposure)
        sse = np.sum(exposures * (log_qx - log_qx_smooth)**2)
        
        # Smoothness penalty (2nd differences)
        penalty = lambda_smooth * np.sum((D2 @ log_qx_smooth)**2)
        
        return sse + penalty
    
    result = minimize(objective, log_qx, method='BFGS')
    qx_smooth = np.exp(result.x)
    
    return qx_smooth

qx_wh = whittaker_henderson(qx_crude, exposures, lambda_smooth=100)

print("WHITTAKER-HENDERSON GRADUATION (λ=100):")
for age in [20, 40, 60, 80]:
    print(f"Age {age}: crude={qx_crude[age]:.5f}, WH={qx_wh[age]:.5f}")
print()

# 5. SPLINE GRADUATION
from scipy.interpolate import UnivariateSpline

# Use log scale for stability
log_qx_crude = np.log(np.maximum(qx_crude, 1e-6))

# Fit spline with smoothing factor
# Smoothing factor s inversely related to lambda
spline = UnivariateSpline(ages, log_qx_crude, s=50, k=3)
log_qx_spline = spline(ages)
qx_spline = np.exp(log_qx_spline)
qx_spline = np.minimum(np.maximum(qx_spline, 0), 0.99999)

# 6. GOODNESS-OF-FIT TEST
def chi_square_test(observed, expected, exposures):
    """Chi-square test for goodness of fit"""
    standardized_residuals = (observed - expected * exposures) / np.sqrt(np.maximum(expected * exposures * (1 - expected), 1))
    chi2_stat = np.sum(standardized_residuals**2)
    return chi2_stat

chi2_parametric = chi_square_test(deaths_per_age, qx_parametric, 1)
chi2_wh = chi_square_test(deaths_per_age, qx_wh, 1)
chi2_spline = chi_square_test(deaths_per_age, qx_spline, 1)

print("GOODNESS-OF-FIT COMPARISON:")
print(f"Parametric (Gompertz-Makeham): χ² = {chi2_parametric:.1f}")
print(f"Whittaker-Henderson:            χ² = {chi2_wh:.1f}")
print(f"Spline:                         χ² = {chi2_spline:.1f}")
print()

# 7. MORTALITY IMPROVEMENT FACTORS
# Compare current table to historical (simulate older experience)
historical_qx = gompertz_makeham(ages, [0.0002, 0.0001, 0.07])

improvement_factors = historical_qx / np.maximum(qx_parametric, 1e-6)
improvement_factors = np.minimum(np.maximum(improvement_factors, 0.5), 2.0)

print("MORTALITY IMPROVEMENT FACTORS (Historical / Current):")
for age in [20, 40, 60, 80]:
    print(f"Age {age}: {improvement_factors[age]:.3f}x")
print()

# 8. BUILD LIFE TABLE FROM GRADUATED RATES
def build_life_table(qx_table, radix=100000):
    """Build complete life table from qx"""
    n = len(qx_table)
    lx = np.zeros(n)
    dx = np.zeros(n)
    Lx = np.zeros(n)
    Tx = np.zeros(n)
    ex = np.zeros(n)
    
    lx[0] = radix
    
    for x in range(n-1):
        dx[x] = lx[x] * qx_table[x]
        lx[x+1] = lx[x] - dx[x]
        Lx[x] = (lx[x] + lx[x+1]) / 2
    
    Lx[n-1] = 0
    Tx[n-1] = Lx[n-1]
    for x in range(n-2, -1, -1):
        Tx[x] = Lx[x] + Tx[x+1]
    
    ex = np.where(lx > 0, Tx / lx, 0)
    
    return pd.DataFrame({
        'Age': range(n),
        'qx': qx_table,
        'lx': lx,
        'dx': dx,
        'ex': ex
    })

life_table = build_life_table(qx_parametric)

print("LIFE TABLE FROM GRADUATED RATES:")
print(life_table[life_table['Age'].isin([0, 20, 40, 60, 80])].to_string(index=False))
print()

# 9. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Crude vs Graduated
ax = axes[0, 0]
ax.scatter(ages[::5], qx_crude[::5], s=50, alpha=0.6, label='Crude (empirical)', color='black')
ax.semilogy(ages, qx_parametric, 'r-', linewidth=2.5, label='Parametric (GM)')
ax.semilogy(ages, qx_wh, 'b--', linewidth=2, label='Whittaker-Henderson')
ax.semilogy(ages, qx_spline, 'g--', linewidth=2, label='Spline')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Mortality Rate qx (log scale)', fontsize=11)
ax.set_title('Crude vs Graduated Mortality Rates', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')
ax.set_ylim([1e-5, 1])

# Plot 2: Residuals
ax = axes[0, 1]
standardized_residuals = (deaths_per_age - exposures * qx_parametric) / np.sqrt(np.maximum(exposures * qx_parametric * (1 - qx_parametric), 1))
ax.scatter(ages, standardized_residuals, s=30, alpha=0.7, color='darkblue')
ax.axhline(0, color='r', linestyle='-', linewidth=1.5)
ax.axhline(2, color='orange', linestyle='--', alpha=0.7, linewidth=1)
ax.axhline(-2, color='orange', linestyle='--', alpha=0.7, linewidth=1)
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Standardized Residuals', fontsize=11)
ax.set_title('Goodness-of-Fit: Residuals', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim([-5, 5])

# Plot 3: Improvement factors
ax = axes[1, 0]
ax.plot(ages, improvement_factors, linewidth=2.5, color='darkblue')
ax.axhline(1.0, color='r', linestyle='--', linewidth=2)
ax.fill_between(ages, 1, improvement_factors, alpha=0.2, where=(improvement_factors >= 1))
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Improvement Factor', fontsize=11)
ax.set_title('Mortality Improvement (Historical / Current)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Life expectancy comparison
ax = axes[1, 1]
ax.plot(ages, life_table['ex'].values, linewidth=2.5, color='darkblue', label='From Graduated')
ax.fill_between(ages, life_table['ex'].values, alpha=0.2, color='blue')
ax.scatter([0, 20, 40, 60, 80], 
          life_table.loc[[0, 20, 40, 60, 80], 'ex'].values,
          s=100, color='red', zorder=5)
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Life Expectancy (years)', fontsize=11)
ax.set_title('Life Expectancy from Graduated Table', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mortality_table_construction.png', dpi=300, bbox_inches='tight')
plt.show()

