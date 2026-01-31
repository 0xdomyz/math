import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit

np.random.seed(123)
n = 1500

# ===== Data Generating Process: Probit DGP =====
# Labor force participation: Depends on wage offer (unobserved), education, age, children
education = np.random.normal(12, 3, n)  # Years of education
age = np.random.uniform(20, 60, n)
children = np.random.poisson(1.5, n)  # Number of children
non_labor_income = np.random.lognormal(10, 0.5, n)  # Partner income, etc.

# Latent variable: Utility difference from working
# True model: Probit with normal errors
latent_utility = (-2 + 0.3*education + 0.05*age - 0.15*age**2/100 - 
                  0.4*children - 0.0001*non_labor_income + 
                  np.random.normal(0, 1, n))

# Observed binary outcome
works = (latent_utility > 0).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'works': works,
    'education': education,
    'age': age,
    'age_squared': age**2 / 100,  # Scaled for numerical stability
    'children': children,
    'non_labor_income': non_labor_income / 1000  # Rescaled
})

print("="*70)
print("PROBIT MODEL: LABOR FORCE PARTICIPATION")
print("="*70)
print(f"\nSample Size: {n}")
print(f"Participation Rate: {works.mean():.1%}")
print("\nDescriptive Statistics:")
print(df.describe().round(2))

# ===== Probit Estimation =====
X = sm.add_constant(df[['education', 'age', 'age_squared', 
                         'children', 'non_labor_income']])
y = df['works']

probit_model = Probit(y, X).fit()

print("\n" + "="*70)
print("PROBIT MODEL RESULTS")
print("="*70)
print(probit_model.summary())

# ===== Logit Estimation (for comparison) =====
logit_model = Logit(y, X).fit(disp=0)

print("\n" + "="*70)
print("COMPARISON: PROBIT vs LOGIT COEFFICIENTS")
print("="*70)

comparison_df = pd.DataFrame({
    'Probit Î²': probit_model.params,
    'Logit Î²': logit_model.params,
    'Logit/1.6': logit_model.params / 1.6  # Approximate scaling
})
print(comparison_df.round(4))
print("\nNote: Logit coefficients â‰ˆ 1.6 Ã— Probit coefficients (rule of thumb)")

# ===== Marginal Effects =====
print("\n" + "="*70)
print("MARGINAL EFFECTS")
print("="*70)

# Get marginal effects using statsmodels
me_probit = probit_model.get_margeff(at='mean')  # MEM
ame_probit = probit_model.get_margeff(at='overall')  # AME

print("\nMarginal Effects at Mean (MEM):")
print(me_probit.summary())

print("\nAverage Marginal Effects (AME):")
print(ame_probit.summary())

# Manual calculation for transparency
X_values = X.values[:, 1:]  # Exclude constant
linear_index = X @ probit_model.params
phi_vals = stats.norm.pdf(linear_index)  # Ï†(X'Î²)

# AME for each variable
ame_manual = {}
for i, var in enumerate(X.columns[1:]):  # Skip constant
    me_i = phi_vals * probit_model.params[var]
    ame_manual[var] = me_i.mean()

print("\nManual AME Calculation:")
for var, ame in ame_manual.items():
    print(f"  {var:20s}: {ame:+.6f}")

# ===== Predictions =====
# Predicted probabilities
prob_probit = probit_model.predict(X)
prob_logit = logit_model.predict(X)

# Predicted outcomes (threshold 0.5)
pred_probit = (prob_probit >= 0.5).astype(int)
pred_logit = (prob_logit >= 0.5).astype(int)

# Accuracy
accuracy_probit = (pred_probit == y).mean()
accuracy_logit = (pred_logit == y).mean()

print("\n" + "="*70)
print("PREDICTION ACCURACY")
print("="*70)
print(f"Probit: {accuracy_probit:.1%}")
print(f"Logit:  {accuracy_logit:.1%}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Probit vs Logit Link Functions
z = np.linspace(-4, 4, 200)
probit_link = stats.norm.cdf(z)
logit_link = 1 / (1 + np.exp(-z))

axes[0, 0].plot(z, probit_link, linewidth=2, label='Probit (Normal CDF)')
axes[0, 0].plot(z, logit_link, linewidth=2, linestyle='--', 
                label='Logit (Logistic CDF)')
axes[0, 0].set_xlabel('Linear Index (X\'Î²)')
axes[0, 0].set_ylabel('P(Y = 1)')
axes[0, 0].set_title('Link Functions: Probit vs Logit')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)

# Plot 2: Difference in Link Functions
axes[0, 1].plot(z, probit_link - logit_link, linewidth=2, color='purple')
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].set_xlabel('Linear Index (X\'Î²)')
axes[0, 1].set_ylabel('Probit - Logit')
axes[0, 1].set_title('Difference: Probit vs Logit')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].fill_between(z, 0, probit_link - logit_link, 
                        alpha=0.3, color='purple')

# Plot 3: PDF (Marginal Effect Weights)
phi = stats.norm.pdf(z)
logistic_pdf = np.exp(-z) / (1 + np.exp(-z))**2

axes[0, 2].plot(z, phi, linewidth=2, label='Normal Ï†(z)')
axes[0, 2].plot(z, logistic_pdf, linewidth=2, linestyle='--',
                label='Logistic Î»(z)(1-Î»(z))')
axes[0, 2].set_xlabel('Linear Index (X\'Î²)')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_title('PDF: Marginal Effect Weights')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Predicted Probability Comparison
axes[1, 0].scatter(prob_probit, prob_logit, alpha=0.3, s=10)
axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='45Â° line')
axes[1, 0].set_xlabel('Probit Predicted Probability')
axes[1, 0].set_ylabel('Logit Predicted Probability')
axes[1, 0].set_title('Predicted Probabilities: Probit vs Logit')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(0, 1)

# Plot 5: Marginal Effect of Education (by age)
ages_to_plot = [25, 35, 45, 55]
education_range = np.linspace(8, 20, 50)

for age_val in ages_to_plot:
    me_by_ed = []
    for ed in education_range:
        X_temp = np.array([1, ed, age_val, age_val**2/100, 
                          df['children'].mean(), 
                          df['non_labor_income'].mean()])
        linear_idx = X_temp @ probit_model.params.values
        phi_temp = stats.norm.pdf(linear_idx)
        me = phi_temp * probit_model.params['education']
        me_by_ed.append(me)
    
    axes[1, 1].plot(education_range, me_by_ed, linewidth=2,
                    label=f'Age {age_val}')

axes[1, 1].set_xlabel('Education (Years)')
axes[1, 1].set_ylabel('Marginal Effect')
axes[1, 1].set_title('ME of Education by Age')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot 6: Participation Probability by Age
age_range = np.linspace(20, 60, 100)
prob_by_age_low_ed = []
prob_by_age_high_ed = []

for age_val in age_range:
    # Low education (10 years)
    X_low = np.array([1, 10, age_val, age_val**2/100,
                      df['children'].mean(), df['non_labor_income'].mean()])
    prob_by_age_low_ed.append(stats.norm.cdf(X_low @ probit_model.params.values))
    
    # High education (16 years)
    X_high = np.array([1, 16, age_val, age_val**2/100,
                       df['children'].mean(), df['non_labor_income'].mean()])
    prob_by_age_high_ed.append(stats.norm.cdf(X_high @ probit_model.params.values))

axes[1, 2].plot(age_range, prob_by_age_low_ed, linewidth=2,
                label='10 Years Education', color='red')
axes[1, 2].plot(age_range, prob_by_age_high_ed, linewidth=2,
                label='16 Years Education', color='blue')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('P(Works = 1)')
axes[1, 2].set_title('Participation Probability by Age & Education')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# ===== Hypothesis Tests =====
print("\n" + "="*70)
print("HYPOTHESIS TESTS")
print("="*70)

# Test: Age effect (age + age_squared jointly = 0)
hypothesis = '(age = 0), (age_squared = 0)'
wald_test = probit_model.wald_test(hypothesis)
print(f"\nWald Test: Hâ‚€: Age has no effect")
print(f"Ï‡Â²(2) = {wald_test.statistic:.2f}, p-value = {wald_test.pvalue:.4f}")

# Test: All coefficients = 0 (overall model significance)
lr_test = probit_model.llr
lr_pval = probit_model.llr_pvalue
print(f"\nLikelihood Ratio Test: Hâ‚€: All Î² = 0")
print(f"Ï‡Â²({len(probit_model.params)-1}) = {lr_test:.2f}, p-value < {lr_pval:.4f}")

# ===== Goodness of Fit =====
print("\n" + "="*70)
print("GOODNESS OF FIT")
print("="*70)

# Pseudo R-squared
print(f"McFadden's Pseudo RÂ²: {probit_model.prsquared:.4f}")

# Count R-squared (fraction correctly predicted)
count_r2 = (pred_probit == y).mean()
print(f"Count RÂ² (% Correct): {count_r2:.1%}")

# AIC and BIC
print(f"AIC: {probit_model.aic:.2f}")
print(f"BIC: {probit_model.bic:.2f}")

# ===== Heterogeneous Treatment Effects =====
print("\n" + "="*70)
print("HETEROGENEOUS MARGINAL EFFECTS")
print("="*70)

# Marginal effect of education varies across individuals
me_education_by_person = phi_vals * probit_model.params['education']

print(f"Education Marginal Effect:")
print(f"  Minimum: {me_education_by_person.min():.6f}")
print(f"  Mean (AME): {me_education_by_person.mean():.6f}")
print(f"  Median: {np.median(me_education_by_person):.6f}")
print(f"  Maximum: {me_education_by_person.max():.6f}")
print(f"  Std Dev: {me_education_by_person.std():.6f}")

print("\nInterpretation:")
print("  Marginal effects are heterogeneous across observations")
print("  Largest effects for those with predicted probability near 0.5")
print("  Smallest effects in tails (very high/low predicted probability)")

# ===== Specification Tests =====
print("\n" + "="*70)
print("SPECIFICATION TESTS")
print("="*70)

# Link test: Regress Y on Å· and Å·Â²
# If model correctly specified, Å·Â² should be insignificant
y_hat = probit_model.predict(X)
y_hat_sq = y_hat ** 2

X_link = sm.add_constant(pd.DataFrame({
    'y_hat': y_hat,
    'y_hat_sq': y_hat_sq
}))
link_model = Probit(y, X_link).fit(disp=0)

print("Link Test (Specification):")
print(f"  Coefficient on Å·Â²: {link_model.params['y_hat_sq']:.4f}")
print(f"  p-value: {link_model.pvalues['y_hat_sq']:.4f}")
if link_model.pvalues['y_hat_sq'] < 0.05:
    print("  âœ— Reject Hâ‚€: Specification may be inadequate")
else:
    print("  âœ“ Fail to reject: Specification appears adequate")
