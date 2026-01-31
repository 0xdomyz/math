# Auto-extracted from markdown file
# Source: life_annuity_continuous.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson

np.random.seed(42)

print("=== Continuous Life Annuity (āₓ) Analysis ===\n")

# Mortality table setup
def gompertz_force(age, A=0.0001, B=1.08, C=0.00035):
    """Force of mortality: μₓ = A + C·B^x"""
    return A + C * (B ** age)

# Build survival function
def survival_prob(x, t, max_age=120):
    """
    ₜpₓ = P(survive from age x to x+t)
    Using force of mortality: ₜpₓ = exp(-∫₀ᵗ μₓ₊ₛ ds)
    """
    if x + t > max_age:
        return 0.0
    
    # Numerical integration of force
    ages = np.linspace(x, x + t, 100)
    forces = gompertz_force(ages)
    integral_mu = np.trapz(forces, ages)
    
    return np.exp(-integral_mu)

# Discrete mortality table for comparison
def build_discrete_mortality():
    ages = np.arange(0, 121)
    mu_x = gompertz_force(ages)
    q_x = 1 - np.exp(-mu_x)
    
    l_x = np.zeros(len(ages))
    l_x[0] = 100000
    for i in range(1, len(ages)):
        l_x[i] = l_x[i-1] * (1 - q_x[i-1])
    
    return pd.DataFrame({'Age': ages, 'l_x': l_x, 'q_x': q_x})

mortality = build_discrete_mortality()

# Continuous annuity calculation
def continuous_annuity(x, i, max_age=120):
    """
    āₓ = ∫₀^∞ e^(-δt) · ₜpₓ dt
    where δ = ln(1+i) = force of interest
    """
    delta = np.log(1 + i)  # Force of interest
    
    def integrand(t):
        if x + t > max_age:
            return 0.0
        return np.exp(-delta * t) * survival_prob(x, t, max_age)
    
    # Numerical integration
    result, _ = quad(integrand, 0, max_age - x, limit=100)
    return result

# Discrete annuity functions (for comparison)
def immediate_annuity(x, i, mortality):
    v = 1 / (1 + i)
    l_current = mortality.loc[mortality['Age'] == x, 'l_x'].values[0]
    
    value = 0
    for k in range(1, 121 - x):
        l_future = mortality.loc[mortality['Age'] == x + k, 'l_x'].values
        if len(l_future) == 0 or l_future[0] <= 0:
            break
        k_p_x = l_future[0] / l_current
        value += (v ** k) * k_p_x
    
    return value

def annuity_due(x, i, mortality):
    return (1 + i) * immediate_annuity(x, i, mortality)

# Calculate values for various ages
print("=== Annuity Comparison: Continuous vs Discrete (i = 5%) ===\n")
ages_test = [30, 45, 60, 65, 75, 85]
i_rate = 0.05

results = []
for age in ages_test:
    a_continuous = continuous_annuity(age, i_rate)
    a_immediate = immediate_annuity(age, i_rate, mortality)
    a_due = annuity_due(age, i_rate, mortality)
    
    # Woolhouse approximation: āₓ ≈ (aₓ + äₓ)/2
    a_woolhouse = (a_immediate + a_due) / 2
    
    results.append({
        'Age': age,
        'Continuous (āₓ)': a_continuous,
        'Immediate (aₓ)': a_immediate,
        'Due (äₓ)': a_due,
        'Woolhouse': a_woolhouse,
        'Error (%)': abs(a_continuous - a_woolhouse) / a_continuous * 100
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format='%.4f'))

# Verify Woolhouse approximation
print("\n=== Woolhouse Approximation Verification ===\n")
age_verify = 65
a_cont = continuous_annuity(age_verify, i_rate)
a_imm = immediate_annuity(age_verify, i_rate, mortality)
a_d = annuity_due(age_verify, i_rate, mortality)

# Simple Woolhouse: āₓ ≈ aₓ + 0.5
a_wool_simple = a_imm + 0.5

# Average: āₓ ≈ (aₓ + äₓ)/2
a_wool_average = (a_imm + a_d) / 2

print(f"Age {age_verify}, i = {i_rate*100:.0f}%:")
print(f"  Continuous (āₓ): {a_cont:.4f}")
print(f"  Immediate + 0.5: {a_wool_simple:.4f} (error: {abs(a_cont - a_wool_simple):.4f})")
print(f"  (aₓ + äₓ)/2: {a_wool_average:.4f} (error: {abs(a_cont - a_wool_average):.4f})")
print(f"\nBest approximation: Average method")

# Special case: Constant force of mortality
print("\n=== Special Case: Constant Force of Mortality ===\n")

mu_constant = 0.02  # 2% annual force
delta = np.log(1 + i_rate)

# Closed form: āₓ = 1/(δ + μ)
a_closed_form = 1 / (delta + mu_constant)

# Numerical verification
def survival_prob_constant(x, t, mu):
    return np.exp(-mu * t)

def continuous_annuity_constant(mu, i):
    delta = np.log(1 + i)
    def integrand(t):
        return np.exp(-delta * t) * survival_prob_constant(0, t, mu)
    result, _ = quad(integrand, 0, 100)
    return result

a_numerical = continuous_annuity_constant(mu_constant, i_rate)

print(f"Constant force μ = {mu_constant:.2f}, i = {i_rate*100:.0f}%:")
print(f"  Closed form: āₓ = 1/(δ + μ) = {a_closed_form:.4f}")
print(f"  Numerical integral: {a_numerical:.4f}")
print(f"  Match: {abs(a_closed_form - a_numerical) < 0.01}")

# Relationship to continuous insurance
print("\n=== Relationship to Continuous Whole Life Insurance ===\n")

def continuous_insurance(x, i, max_age=120):
    """
    Āₓ = ∫₀^∞ e^(-δt) · ₜpₓ · μₓ₊ₜ dt
    """
    delta = np.log(1 + i)
    
    def integrand(t):
        age_t = x + t
        if age_t > max_age:
            return 0.0
        return np.exp(-delta * t) * survival_prob(x, t, max_age) * gompertz_force(age_t)
    
    result, _ = quad(integrand, 0, max_age - x, limit=100)
    return result

age_test = 45
delta = np.log(1 + i_rate)

a_bar = continuous_annuity(age_test, i_rate)
A_bar = continuous_insurance(age_test, i_rate)
a_from_insurance = (1 - A_bar) / delta

print(f"Age {age_test}, i = {i_rate*100:.0f}%:")
print(f"  āₓ (direct): {a_bar:.4f}")
print(f"  Āₓ (insurance): {A_bar:.4f}")
print(f"  āₓ = (1 - Āₓ)/δ: {a_from_insurance:.4f}")
print(f"  Identity holds: {abs(a_bar - a_from_insurance) < 0.01}")

# Interest rate sensitivity
print("\n=== Interest Rate Sensitivity ===\n")
interest_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
age_sens = 65

print(f"Age {age_sens} Continuous Annuity:")
print("Interest | Continuous (āₓ) | Immediate (aₓ) | Difference")
print("-" * 60)

for i_val in interest_rates:
    a_cont_val = continuous_annuity(age_sens, i_val)
    a_imm_val = immediate_annuity(age_sens, i_val, mortality)
    diff = a_cont_val - a_imm_val
    
    print(f"{i_val*100:7.0f}%  | {a_cont_val:14.4f} | {a_imm_val:14.4f} | {diff:9.4f}")

# Variance calculation
print("\n=== Variance of Continuous Annuity ===\n")

def continuous_annuity_second_moment(x, i, max_age=120):
    """
    E[Ȳ²] = ∫₀^∞ (∫₀^T e^(-δt) dt)² · μₓ₊ₜ · ₜpₓ dT
    Simplified: ∫₀^∞ e^(-2δt) · ₜpₓ · μₓ₊ₜ dt (using integration by parts result)
    """
    delta = np.log(1 + i)
    
    def integrand(t):
        age_t = x + t
        if age_t > max_age:
            return 0.0
        
        # (1 - e^(-δt))/δ)² term from integral of payment stream
        payment_integral = (1 - np.exp(-delta * t)) / delta
        
        return (payment_integral ** 2) * survival_prob(x, t, max_age) * gompertz_force(age_t)
    
    result, _ = quad(integrand, 0, max_age - x, limit=100)
    return result

age_var = 65
a_bar_mean = continuous_annuity(age_var, i_rate)
second_moment = continuous_annuity_second_moment(age_var, i_rate)
variance = second_moment - a_bar_mean**2
std_dev = np.sqrt(variance)

print(f"Age {age_var}, i = {i_rate*100:.0f}%:")
print(f"  E[Ȳ] = āₓ: {a_bar_mean:.4f}")
print(f"  E[Ȳ²]: {second_moment:.4f}")
print(f"  Var[Ȳ]: {variance:.4f}")
print(f"  Std Dev: {std_dev:.4f}")
print(f"  Coefficient of variation: {std_dev / a_bar_mean * 100:.1f}%")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: All three annuity types by age
ax1 = axes[0, 0]
ages_plot = np.arange(30, 91, 5)
continuous_vals = [continuous_annuity(age, 0.05) for age in ages_plot]
immediate_vals = [immediate_annuity(age, 0.05, mortality) for age in ages_plot]
due_vals = [annuity_due(age, 0.05, mortality) for age in ages_plot]

ax1.plot(ages_plot, continuous_vals, 'o-', linewidth=2, label='Continuous (āₓ)', markersize=6)
ax1.plot(ages_plot, immediate_vals, 's-', linewidth=2, label='Immediate (aₓ)', markersize=6)
ax1.plot(ages_plot, due_vals, '^-', linewidth=2, label='Due (äₓ)', markersize=6)
ax1.set_xlabel('Age')
ax1.set_ylabel('Annuity Value')
ax1.set_title('Continuous vs Discrete Annuities\n(i = 5%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Woolhouse approximation error
ax2 = axes[0, 1]
errors = []
for age in ages_plot:
    a_cont = continuous_annuity(age, 0.05)
    a_imm = immediate_annuity(age, 0.05, mortality)
    a_d = annuity_due(age, 0.05, mortality)
    a_wool = (a_imm + a_d) / 2
    error_pct = abs(a_cont - a_wool) / a_cont * 100
    errors.append(error_pct)

ax2.plot(ages_plot, errors, 'o-', linewidth=2, color='red', markersize=6)
ax2.fill_between(ages_plot, 0, errors, alpha=0.2, color='red')
ax2.set_xlabel('Age')
ax2.set_ylabel('Approximation Error (%)')
ax2.set_title('Woolhouse Approximation Accuracy\nāₓ ≈ (aₓ + äₓ)/2')
ax2.grid(True, alpha=0.3)

# Plot 3: Difference visualization
ax3 = axes[0, 2]
diff_immediate = np.array(continuous_vals) - np.array(immediate_vals)
diff_due = np.array(due_vals) - np.array(continuous_vals)

ax3.plot(ages_plot, diff_immediate, 'o-', linewidth=2, label='āₓ - aₓ', markersize=6)
ax3.plot(ages_plot, diff_due, 's-', linewidth=2, label='äₓ - āₓ', markersize=6)
ax3.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='0.5 (approximation)')
ax3.set_xlabel('Age')
ax3.set_ylabel('Difference')
ax3.set_title('Continuous Annuity Position\n(Between immediate and due)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Interest rate sensitivity
ax4 = axes[1, 0]
i_range = np.linspace(0.01, 0.10, 20)
cont_by_i = [continuous_annuity(65, i) for i in i_range]
imm_by_i = [immediate_annuity(65, i, mortality) for i in i_range]

ax4.plot(i_range * 100, cont_by_i, linewidth=2, label='Continuous')
ax4.plot(i_range * 100, imm_by_i, linewidth=2, label='Immediate', linestyle='--')
ax4.set_xlabel('Interest Rate (%)')
ax4.set_ylabel('Annuity Value (Age 65)')
ax4.set_title('Interest Rate Impact\n(Both decrease similarly)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Constant force closed form
ax5 = axes[1, 1]
mu_range = np.linspace(0.001, 0.05, 30)
delta = np.log(1.05)
closed_form_vals = [1 / (delta + mu) for mu in mu_range]

ax5.plot(mu_range * 100, closed_form_vals, linewidth=2)
ax5.fill_between(mu_range * 100, 0, closed_form_vals, alpha=0.2)
ax5.set_xlabel('Force of Mortality μ (%)')
ax5.set_ylabel('Continuous Annuity āₓ')
ax5.set_title('Constant Force Case\nāₓ = 1/(δ + μ)')
ax5.grid(True, alpha=0.3)

# Plot 6: Insurance-annuity relationship
ax6 = axes[1, 2]
ages_relationship = np.arange(30, 86, 5)
annuity_vals_rel = []
insurance_vals_rel = []

for age in ages_relationship:
    a_bar = continuous_annuity(age, 0.05)
    A_bar = continuous_insurance(age, 0.05)
    annuity_vals_rel.append(a_bar)
    insurance_vals_rel.append(A_bar)

ax6_2 = ax6.twinx()
ax6.plot(ages_relationship, annuity_vals_rel, 'o-', linewidth=2, color='blue', label='āₓ (annuity)', markersize=6)
ax6_2.plot(ages_relationship, insurance_vals_rel, 's-', linewidth=2, color='red', label='Āₓ (insurance)', markersize=6)

ax6.set_xlabel('Age')
ax6.set_ylabel('Annuity Value (āₓ)', color='blue')
ax6_2.set_ylabel('Insurance Value (Āₓ)', color='red')
ax6.set_title('Complementary Relationship\nāₓ = (1 - Āₓ)/δ')
ax6.tick_params(axis='y', labelcolor='blue')
ax6_2.tick_params(axis='y', labelcolor='red')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print("Continuous annuity (āₓ): Instantaneous payments, āₓ = ∫ e^(-δt)·ₜpₓ dt")
print("Woolhouse approximation: āₓ ≈ (aₓ + äₓ)/2 (error < 1%)")
print("Theoretical tool for derivations; convert to discrete for applications")

