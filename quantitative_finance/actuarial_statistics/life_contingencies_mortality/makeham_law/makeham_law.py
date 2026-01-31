# Auto-extracted from markdown file
# Source: makeham_law.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2

# 1. SIMULATE REALISTIC DATA WITH TRUE MAKEHAM
np.random.seed(42)

ages = np.arange(0, 101)

# True Makeham parameters
A_true = 0.00008  # Background/accident component
B_true = 0.00005  # Senescence coefficient
C_true = 1.08     # Age escalation

# True force of mortality
mu_true = A_true + B_true * (C_true ** ages)

# Convert to annual probability
# ₚₓ = exp[-Ax - B(C^{x+1} - C^x)/(ln C)]
ln_C = np.log(C_true)
px_true = np.exp(-A_true - B_true * (C_true**(ages + 1) - C_true**ages) / ln_C)
qx_true = 1 - px_true
qx_true = np.minimum(np.maximum(qx_true, 1e-10), 0.99999)

# Simulate data: More data at young/old ages (insurance focus)
exposures = np.concatenate([
    np.random.poisson(500, 20),    # Ages 0-19: few insured children
    np.random.poisson(8000, 50),   # Ages 20-69: peak insurance population
    np.random.poisson(3000, 20),   # Ages 70-89: declining population
    np.random.poisson(500, 11)     # Ages 90-100: very few
])

deaths = np.random.binomial(exposures, qx_true)
qx_empirical = deaths / np.maximum(exposures, 1)
mu_empirical = -np.log(1 - qx_empirical)

print("MAKEHAM LAW FITTING: Insurance Portfolio Data")
print(f"True parameters: A = {A_true:.6f}, B = {B_true:.6f}, C = {C_true:.5f}")
print(f"Total exposures: {exposures.sum():.0f}, Total deaths: {deaths.sum():.0f}")
print()

# 2. FIT GOMPERTZ ALONE (for comparison)
def gompertz_mu(x, params):
    """Gompertz force of mortality"""
    A, B = params
    return A * (B ** x)

def gompertz_qx(ages_data, A, B):
    """Gompertz mortality probability"""
    ln_B = np.log(B)
    px = np.exp(-A * (B**(ages_data + 1) - B**ages_data) / ln_B)
    qx = 1 - px
    return np.minimum(np.maximum(qx, 1e-10), 0.99999)

def neg_ll_gompertz(params, ages_data, deaths_data, exposures_data):
    """Negative log-likelihood for Gompertz"""
    A, B = params
    if A <= 0 or B <= 1 or A > 0.1 or B > 1.3:
        return 1e10
    
    qx = gompertz_qx(ages_data, A, B)
    ll = np.sum(deaths_data * np.log(qx) + (exposures_data - deaths_data) * np.log(1 - qx))
    return -ll

# Fit on ages 50-95 (where Gompertz works well)
mask_gompertz = (ages >= 50) & (ages <= 95)
ages_fit_g = ages[mask_gompertz]
deaths_fit_g = deaths[mask_gompertz]
exp_fit_g = exposures[mask_gompertz]

p0_gompertz = [0.00008, 1.08]
result_g = minimize(neg_ll_gompertz, p0_gompertz,
                   args=(ages_fit_g, deaths_fit_g, exp_fit_g),
                   method='Nelder-Mead')

A_gompertz, B_gompertz = result_g.x

print("GOMPERTZ FIT (ages 50-95 only):")
print(f"A = {A_gompertz:.6f} (true: --)")
print(f"B = {B_gompertz:.5f} (true: --)")
print(f"Negative LL on subset: {result_g.fun:.2f}")
print()

# 3. FIT MAKEHAM (3 parameters)
def makeham_mu(x, params):
    """Makeham force of mortality"""
    A, B, C = params
    return A + B * (C ** x)

def makeham_qx(ages_data, A, B, C):
    """Makeham mortality probability"""
    ln_C = np.log(C)
    px = np.exp(-A - B * (C**(ages_data + 1) - C**ages_data) / ln_C)
    qx = 1 - px
    return np.minimum(np.maximum(qx, 1e-10), 0.99999)

def neg_ll_makeham(params, ages_data, deaths_data, exposures_data):
    """Negative log-likelihood for Makeham"""
    A, B, C = params
    if A <= 0 or B <= 0 or C <= 1 or A > 0.05 or B > 0.001 or C > 1.2:
        return 1e10
    
    qx = makeham_qx(ages_data, A, B, C)
    ll = np.sum(deaths_data * np.log(qx) + (exposures_data - deaths_data) * np.log(1 - qx))
    return -ll

# Fit on all ages 0-100
mask_all = ages < 100
ages_fit_m = ages[mask_all]
deaths_fit_m = deaths[mask_all]
exp_fit_m = exposures[mask_all]

# Use global optimization for 3 parameters
bounds_makeham = [(0.00001, 0.01), (0.00001, 0.0005), (1.01, 1.15)]
result_m = differential_evolution(neg_ll_makeham,
                                 bounds_makeham,
                                 args=(ages_fit_m, deaths_fit_m, exp_fit_m),
                                 seed=42, maxiter=1000)

A_makeham, B_makeham, C_makeham = result_m.x

print("MAKEHAM FIT (all ages 0-99):")
print(f"A = {A_makeham:.6f} (true: {A_true:.6f})")
print(f"B = {B_makeham:.6f} (true: {B_true:.6f})")
print(f"C = {C_makeham:.5f} (true: {C_true:.5f})")
print(f"Negative LL on full data: {result_m.fun:.2f}")
print()

# 4. COMPONENT ANALYSIS
# Decompose force of mortality into accident and senescence
mu_accident = A_makeham * np.ones_like(ages)
mu_senescence = B_makeham * (C_makeham ** ages)
mu_makeham = mu_accident + mu_senescence

# Calculate percentage contribution by age
pct_accident = 100 * mu_accident / np.maximum(mu_makeham, 1e-10)
pct_senescence = 100 * mu_senescence / np.maximum(mu_makeham, 1e-10)

print("MORTALITY COMPONENT DECOMPOSITION:")
print("Age\tAccident %\tSenescence %")
for age in [10, 20, 40, 60, 80]:
    print(f"{age}\t{pct_accident[age]:.1f}%\t\t{pct_senescence[age]:.1f}%")
print()

# 5. GOODNESS-OF-FIT COMPARISON
qx_makeham = makeham_qx(ages, A_makeham, B_makeham, C_makeham)
qx_gompertz_all = gompertz_qx(ages, A_gompertz, B_gompertz)

deaths_exp_makeham = exposures * qx_makeham
deaths_exp_gompertz = exposures * qx_gompertz_all

# Chi-square test (exclude ages 0-49 where Gompertz expected to fail)
mask_young = ages < 50
deaths_young = deaths[mask_young]
exp_young = exposures[mask_young]

chi2_makeham_young = np.sum((deaths_young - exp_young * qx_makeham[mask_young])**2 / 
                             np.maximum(exp_young * qx_makeham[mask_young], 1))
chi2_gompertz_young = np.sum((deaths_young - exp_young * qx_gompertz_all[mask_young])**2 / 
                              np.maximum(exp_young * qx_gompertz_all[mask_young], 1))

print("GOODNESS-OF-FIT: YOUNG AGES (0-49)")
print(f"Makeham χ² = {chi2_makeham_young:.1f} (df = {mask_young.sum() - 3})")
print(f"Gompertz χ² = {chi2_gompertz_young:.1f} (df = {mask_young.sum() - 2})")
print(f"Makeham better by {chi2_gompertz_young - chi2_makeham_young:.0f} points")
print()

# 6. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Force of mortality (log scale)
ax = axes[0, 0]
ax.semilogy(ages, mu_empirical, 'o', markersize=4, alpha=0.5, label='Empirical', color='black')
ax.semilogy(ages, mu_makeham, '-', linewidth=2.5, label='Makeham fit', color='red')
ax.semilogy(ages, A_makeham + B_makeham * (C_makeham ** ages), '--', 
           linewidth=2, alpha=0.7, label='Makeham (from params)', color='darkred')
ax.semilogy(ages, mu_true, ':', linewidth=2, alpha=0.6, label='True', color='gray')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Force of Mortality μx (log scale)', fontsize=11)
ax.set_title('Makeham: Force of Mortality Fit', fontsize=12, fontweight='bold')
ax.set_ylim([1e-5, 1])
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')

# Plot 2: Component decomposition
ax = axes[0, 1]
ax.fill_between(ages, 0, mu_accident, alpha=0.5, color='steelblue', label='Accident component (A)')
ax.fill_between(ages, mu_accident, mu_accident + mu_senescence, alpha=0.5, 
               color='coral', label='Senescence component (B·Cˣ)')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Force of Mortality μx', fontsize=11)
ax.set_title('Mortality Decomposition: Accident vs Senescence', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Comparison with Gompertz
ax = axes[1, 0]
qx_gompertz_young_only = gompertz_qx(ages, A_gompertz, B_gompertz)
ax.semilogy(ages, qx_empirical, 'o', markersize=4, alpha=0.5, label='Empirical', color='black')
ax.semilogy(ages, qx_makeham, '-', linewidth=2.5, label='Makeham', color='red')
ax.semilogy(ages, qx_gompertz_young_only, '--', linewidth=2, label='Gompertz', color='blue')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Annual Mortality Rate qx (log scale)', fontsize=11)
ax.set_title('Makeham vs Gompertz: Full Age Range', fontsize=12, fontweight='bold')
ax.set_ylim([1e-5, 1])
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')

# Plot 4: Percentage accident contribution
ax = axes[1, 1]
ax.plot(ages, pct_accident, linewidth=2.5, color='steelblue', label='Accident component %')
ax.fill_between(ages, 0, pct_accident, alpha=0.2, color='steelblue')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('% of Total Force', fontsize=11)
ax.set_title('Accident Mortality Dominance by Age', fontsize=12, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(alpha=0.3)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)

# Add annotations for key ages
ages_to_mark = [15, 40, 80]
for age in ages_to_mark:
    pct = pct_accident[age]
    ax.annotate(f'Age {age}: {pct:.0f}%', xy=(age, pct),
               xytext=(5, 5), textcoords='offset points', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('makeham_law_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

