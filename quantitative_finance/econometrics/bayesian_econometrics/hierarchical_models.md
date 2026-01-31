# Hierarchical Models

## 1. Concept Skeleton
**Definition:** Multi-level probability models where parameters vary across groups; group-level parameters themselves have probability distributions (hyperpriors)  
**Purpose:** Partial pooling of information across groups; automatic regularization; accommodate heterogeneity; improve predictions for new groups  
**Prerequisites:** Multilevel structure, exchangeability, variance decomposition, MCMC/HMC (or Gibbs for conjugate)

## 2. Comparative Framing
| Approach | Pooling | Degrees of Freedom | Estimation | Uncertainty | Generalization |
|----------|---------|------------------|-----------|------------|-----------------|
| **No pooling** | None; separate model per group | n×k (overfitting) | Group-specific data | Large SE; group-specific | Poor (overshrinking new groups) |
| **Complete pooling** | All groups same; ignore | k (underfitting) | Aggregate data | Small SE; biased | Good (but wrong structure) |
| **Hierarchical** | Partial; informative prior from hyperprior | k+g (balanced) | Data + structure | Moderate SE; grouped | Best (borrow strength) |
| **Mixed effects** | Two-stage; random intercepts/slopes | k+g (balanced) | REML/MLE | Frequentist SE | Comparable to hierarchical |

## 3. Examples + Counterexamples

**Simple Example:**  
Schools efficacy: 30 schools, 50 students each; each school has intercept α_j. No pooling: estimate α̂_j per school (high variance, small n). Complete pooling: one α̂ (biased if schools differ). Hierarchical: α_j ~ N(μ, σ²_schools); learn μ and σ² → Partial pooling (shrinkage toward overall mean proportional to school size)

**Failure Case:**  
Misspecified hierarchy: Assume students within school exchangeable, but students nested in classrooms within schools → Model ignores classroom effects; biased variance; solution: three-level hierarchy

**Edge Case:**  
Few groups (J=3); each group large (n_j=1000) → Hierarchical not needed (group-specific estimates low SE); but with J=100, n_j=10 → Hierarchical essential (leverages across groups)

## 4. Layer Breakdown
```
Hierarchical Models Structure:
├─ Background: Why Hierarchy Useful:
│   ├─ Problem: Separate groups likely share structure
│   │   ├─ Different intercepts but similar slopes
│   │   ├─ Some groups small n → Noisy group-specific estimates
│   │   ├─ Extreme estimates likely noise, not truth
│   │   └─ Want to "borrow strength" from other groups
│   ├─ Intuition: Exchange rates as case study
│   │   ├─ 50 currency pairs; limited data on each
│   │   ├─ Pairs likely correlated (common macroeconomic factors)
│   │   ├─ Naive: Fit each pair separately → High variance
│   │   ├─ Hierarchical: Model pair-specific parameters as N(μ,σ²)
│   │   └─ Benefit: Extreme parameter estimates shrink toward center
│   ├─ Exchangeability:
│   │   ├─ Definition: Groups indexed by i; no prior reason to order them
│   │   ├─ p(y₁,...,y_J) = p(y_{π(1)},...,y_{π(J)}) [invariant to permutation π]
│   │   ├─ Justifies common hyperprior p(θ_j|φ)
│   │   └─ Key assumption: Groups "similar enough" to learn from each other
│   └─ Partial pooling formalization:
│       ├─ Point estimate: θ̂_j = λ_j·θ̂_j^{no pooling} + (1-λ_j)·θ̂_j^{pooled}
│       ├─ λ_j ∈ [0,1]: Shrinkage proportion
│       ├─ λ_j ≈ 0: Heavy shrinkage (small n_j, high within-group variance)
│       ├─ λ_j ≈ 1: Light shrinkage (large n_j, low within-group variance)
│       └─ Automatic from Bayesian update (no ad-hoc choice needed)
├─ Two-Level Hierarchy:
│   ├─ Structure (school effectiveness example):
│   │   ├─ Level 1: Individual students
│   │   │   ├─ Observation: y_ij ~ N(α_j + β_j·X_ij, σ²_within)
│   │   │   ├─ i: Student index (within school j)
│   │   │   ├─ j: School index
│   │   │   └─ Residual: σ²_within (student-level variability)
│   │   ├─ Level 2: Schools
│   │   │   ├─ Parameters: α_j ~ N(μ_α, σ²_α)
│   │   │   ├─ Slopes: β_j ~ N(μ_β, σ²_β)
│   │   │   └─ Interpretation: School-level heterogeneity
│   │   └─ Hyperpriors: μ_α ~ N(μ₀, σ²₀), σ²_α ~ Inverse-Gamma(...)
│   │       └─ Estimation: Learn from data (shrinkage learned from data)
│   ├─ Bayesian formulation:
│   │   ├─ Full posterior: p(α, β, μ, σ, σ_within | y)
│   │   │   ├─ α = {α₁,...,α_J}, β = {β₁,...,β_J}, μ = (μ_α, μ_β)
│   │   │   ├─ σ = (σ_α, σ_β), σ_within
│   │   │   └─ Integrates over all parameters & hyperparameters
│   │   ├─ Interpretation:
│   │   │   ├─ E[α_j | data] combines within-school and between-school info
│   │   │   ├─ SE[α_j] accounts for two sources of uncertainty (within & between)
│   │   │   ├─ Prediction for new school: Uses estimated μ, σ (partial pooling)
│   │   │   └─ Posterior for new school: t-distribution (accounts for hyperprior uncertainty)
│   │   └─ Comparison to fixed effects:
│   │       ├─ Fixed effects: α_j treated as unknown constants
│   │       ├─ Random effects (hierarchical): α_j random ~ hyperprior
│   │       ├─ Efficiency: Hierarchical gains leverage across schools
│   │       └─ Trade-off: Assumes schools exchangeable (random effects ok if true)
│   ├─ Degrees of freedom (effective):
│   │   ├─ No pooling: n×J parameters (J schools, n students each)
│   │   ├─ Complete pooling: 1 (single intercept)
│   │   ├─ Hierarchical: Between n×J and 1 (depends on variance decomposition)
│   │   ├─ Example: J=30 schools, n=50 per school
│   │   │   ├─ No pooling: 30 intercepts
│   │   │   ├─ Hierarchical: Effective ~10-20 (depends on σ_α)
│   │   │   └─ Information borrowed: Shrinks extreme estimates
│   │   └─ Estimated via Bayesian method
│   └─ Estimation:
│       ├─ Gibbs sampling (if conjugate):
│       │   ├─ Conditional on hyperparameters: Group effects normal (conjugate)
│       │   ├─ Conditional on group effects: Hyperparameters inverse-gamma
│       │   ├─ Cycle through: Group effects → Hyperparameters → Repeat
│       │   └─ Fast; stable; standard approach
│       ├─ MCMC (HMC/NUTS if non-conjugate):
│       │   ├─ One block: Sample all parameters + hyperparameters
│       │   ├─ Gradient-based: Efficient in high dimensions
│       │   └─ Slower but more flexible
│       └─ Empirical Bayes (practical shortcut):
│           ├─ Estimate hyperparameters from data (ignore hyperprior uncertainty)
│           ├─ Fix μ̂, σ̂; then posterior for α_j
│           ├─ Fast; slightly underestimates posterior variance
│           └─ Trade-off: Speed vs underestimation of uncertainty
├─ Three-Level Hierarchy:
│   ├─ Structure (students within classrooms within schools):
│   │   ├─ Level 1: Individual y_ijk (student k, classroom j, school i)
│   │   ├─ Level 2: Classroom effects γ_ij ~ N(α_i, σ²_classroom)
│   │   ├─ Level 3: School effects α_i ~ N(μ, σ²_school)
│   │   └─ Hyperpriors on μ, σ²_school
│   ├─ Interpretation:
│   │   ├─ Variance decomposition: Total var = within-student + classroom + school
│   │   ├─ Intraclass correlation (ICC): ρ = σ²_school / (σ²_school + σ²_classroom + σ²_within)
│   │   ├─ ICC measures: What fraction of variance from school vs classroom vs within?
│   │   └─ Example: ICC=0.15 means 15% of variance between schools
│   ├─ Computation:
│   │   ├─ More complex (3 levels → more hyperparameters)
│   │   ├─ Gibbs still tractable (conditional each level normal)
│   │   ├─ MCMC slower but feasible (K~100-1000 parameters typical)
│   │   └─ Interpretation harder (multiple variance components)
│   └─ When to use:
│       ├─ Natural hierarchy in problem (students → classrooms → schools)
│       ├─ If 2-level sufficient: Simpler is better (Occam's razor)
│       ├─ Check ICC at each level: If ICC<0.05 → Consider removing level
│       └─ Trade-off: More parameters; harder to interpret; more data needed
├─ Shrinkage & Regularization:
│   ├─ Shrinkage mechanism:
│   │   ├─ Group j with small n_j: Estimate noisy; posterior pulls toward μ
│   │   ├─ Amount of shrinkage determined by:
│   │   │   ├─ Group-level variance σ²_group (higher → less shrinkage)
│   │   │   ├─ Within-group variance σ²_within (higher → more shrinkage)
│   │   │   └─ Group sample size n_j (larger → less shrinkage)
│   │   └─ Optimal shrinkage automatic from Bayes (no tuning parameter)
│   ├─ Relationship to regularization:
│   │   ├─ Ridge regression: Penalty λ||β||² ≈ Normal prior with σ² ∝ 1/λ
│   │   ├─ Hierarchical: Prior σ² learned from data (empirical Bayes)
│   │   ├─ Advantage: Adapt regularization strength to problem
│   │   └─ No hyperparameter tuning (vs Ridge: choose λ via cross-validation)
│   ├─ James-Stein shrinkage:
│   │   ├─ Classic result: Separate estimates better than combined
│   │   ├─ But shrunk toward mean often beats both
│   │   ├─ Formula: θ̂_JS = (1 - (k-2)/(||y-ȳ1||²))y
│   │   ├─ Hierarchical Bayes: Automatic; interpreted as shrinkage estimate
│   │   └─ Philosophical: Seemingly paradoxical (shrinking helps) → Practical in high-d
│   └─ Predictions for new group:
│       ├─ Predict y_new for group not in training data
│       ├─ No group-specific data: Use hyperprior p(α|μ,σ)
│       ├─ Posterior predictive: p(y_new) = ∫ p(y_new|α)p(α|μ,σ)p(μ,σ|data) dα dμ dσ
│       ├─ Account for:
│       │   ├─ Uncertainty in α (hyperprior)
│       │   ├─ Uncertainty in μ, σ (learned from data)
│       │   └─ Residual variation σ²_within
│       └─ Practical: Draw α from posterior hyperprior; then generate y_new
├─ Random Slopes & Intercepts:
│   ├─ Varying intercept only:
│   │   ├─ y_ij = α_j + β·X_ij + ε_ij
│   │   ├─ α_j ~ N(μ_α, σ²_α)
│   │   └─ Slope β same across groups (one parameter; learned from all data)
│   ├─ Varying slope only:
│   │   ├─ y_ij = α + β_j·X_ij + ε_ij
│   │   ├─ β_j ~ N(μ_β, σ²_β)
│   │   └─ Intercept α same across groups
│   ├─ Varying both (most complex):
│   │   ├─ y_ij = α_j + β_j·X_ij + ε_ij
│   │   ├─ (α_j, β_j) jointly multivariate normal
│   │   ├─ Correlation between intercepts & slopes possible
│   │   │   ├─ Example: School with high intercept also steep slope
│   │   │   ├─ ρ quantifies relationship
│   │   │   └─ Interpretation: School-level heterogeneity structure
│   │   └─ Computation: 2-D hyperprior; more MCMC samples needed
│   ├─ Trade-offs:
│   │   ├─ More random effects → More flexibility; more parameters
│   │   ├─ Fewer groups (J small) → Shrinkage less effective
│   │   ├─ Rule of thumb: J > 10 needed for reliable variance estimates
│   │   └─ With J=3-5: Very aggressive shrinkage; priors critical
│   └─ Identifiability:
│       ├─ Random intercepts + overall intercept: Aliasing
│       ├─ Convention: Center α_j: α_j ~ N(0, σ²); overall intercept α₀
│       ├─ Or: Center predictors; α_j ~ N(α₀, σ²)
│       └─ Software handles automatically (Stan, PyMC)
├─ Applications:
│   ├─ Meta-analysis:
│   │   ├─ Combine effect sizes from multiple studies
│   │   ├─ Studies likely heterogeneous (different populations, designs)
│   │   ├─ Hierarchical: Effect_i ~ N(μ_effect, σ²_studies)
│   │   ├─ Posterior μ: Weighted average across studies
│   │   ├─ Credible interval: Larger than naive pooling (accounts for heterogeneity)
│   │   └─ Prediction for new study: Use hyperprior (partially pools)
│   ├─ Multilevel econometrics:
│   │   ├─ Firms → Sectors; individuals → Regions; agents → Markets
│   │   ├─ Each level contributes heterogeneity
│   │   ├─ Hierarchical: Learn structure; make predictions
│   │   └─ Example: Worker earnings ~ worker FE + firm FE + sector FE
│   ├─ Time series (hierarchical evolution):
│   │   ├─ Multiple time series (assets, regions, firms)
│   │   ├─ Each evolves independently; share common trend/level
│   │   ├─ Hierarchy: Individual evolution ~ shared process
│   │   └─ Benefit: Stabilize forecasts for noisy short series
│   ├─ Spatial models:
│   │   ├─ Observations at locations; nearby locations correlated
│   │   ├─ Hierarchy: Local effects ~ spatial distribution (CAR, ICAR)
│   │   ├─ Example: Disease rates by region; pooling by geography
│   │   └─ Interpretation: Smooth spatial variation
│   └─ Cross-classified data:
│       ├─ Non-nested: Students cross teachers (not tree structure)
│       ├─ Hierarchy extends: Effects for both students + teachers
│       ├─ Computation: More complex (non-nested Gibbs)
│       └─ Software: Stan, PyMC3 handle automatically
├─ Model Checking & Diagnostics:
│   ├─ Posterior predictive checks:
│   │   ├─ Simulate y_rep from posterior predictive
│   │   ├─ Compare y_rep to observed y
│   │   ├─ Statistics: Mean, variance, extremes per group
│   │   └─ Good fit: y_rep similar to y for most groups
│   ├─ Residual plots:
│   │   ├─ Standardized residuals vs fitted: Should be ~N(0,1)
│   │   ├─ By group: Check for group-specific patterns
│   │   ├─ QQ-plot: Check normality
│   │   └─ ACF: Check autocorrelation (especially time series)
│   ├─ Variance decomposition:
│   │   ├─ ICC: Intraclass correlation
│   │   │   ├─ ICC = σ²_between / (σ²_between + σ²_within)
│   │   ├─ Interpretation: Proportion variance explained by groups
│   │   ├─ Small ICC (< 0.05): Grouping not important; simpler model ok
│   │   └─ Large ICC (> 0.25): Strong clustering; hierarchy essential
│   ├─ Shrinkage diagnostics:
│   │   ├─ Compare posterior α̂_j to no-pooling estimates
│   │   ├─ Scatter plot: Points on diagonal = no shrinkage; off-diagonal = shrinkage
│   │   ├─ Extreme no-pooling estimates: Expect largest shrinkage
│   │   └─ Check SE: Hierarchical posterior SE smaller (borrowing strength)
│   └─ Cross-validation:
│       ├─ Leave-one-group-out: Fit without group j; predict group j
│       ├─ Compare to no-pooling CV
│       ├─ Hierarchical typically outperforms (uses partial pooling)
│       └─ Strong test of model quality
├─ Practical Workflow:
│   ├─ Step 1: Identify hierarchy (problem structure)
│   ├─ Step 2: Specify level-1 model (observation-level)
│   ├─ Step 3: Specify level-2 model (group effects)
│   ├─ Step 4: Choose hyperpriors (weakly informative)
│   │   ├─ Group-level SD: Exp(1) or Exp(1/α) [α prior guess of SD]
│   │   ├─ Hyperprior mean: Flat or centered at observed mean
│   │   └─ Avoid: Very tight priors (suppress heterogeneity)
│   ├─ Step 5: Fit (MCMC; Stan/PyMC default)
│   ├─ Step 6: Check convergence (R̂, ESS, traces)
│   ├─ Step 7: Summarize
│   │   ├─ Posterior means + CIs for group effects
│   │   ├─ Hyperparameters (μ, σ)
│   │   ├─ ICC (variance decomposition)
│   │   └─ Posterior predictive for new group
│   └─ Documentation:
│       ├─ Model specification (levels; random effects)
│       ├─ Prior specification (justification)
│       ├─ Computational details (MCMC diagnostics)
│       ├─ ICC and variance decomposition
│       └─ Predictions for new observations/groups
└─ Software Implementation:
    ├─ Stan: `real<lower=0> sigma_group; vector[J] alpha_j;`
    ├─ PyMC3: `pm.Normal('alpha', mu=0, sigma=sigma_group, shape=J)`
    ├─ R (frequentist): `lme4::lmer(y ~ 1 + (1|group))`
    ├─ R (Bayesian): `brms::brm(y ~ 1 + (1|group), family=gaussian())`
    └─ Python: `statsmodels.mixed_effects`; less feature-rich
```

**Key Insight:** Hierarchical models automate partial pooling across groups; borrowing strength reduces variance; predictions for new groups naturally incorporate uncertainty; variance decomposition reveals structure; essential for nested/clustered data

## 5. Mini-Project
Hierarchical model for test scores across schools:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set seed
np.random.seed(42)

# Generate hierarchical data
J = 30  # Number of schools
n_students = 50  # Students per school

# True hyperparameters
mu_true = 70  # Overall mean score
sigma_school_true = 5  # Between-school SD
sigma_within_true = 10  # Within-school SD

# Generate school effects
school_effects = np.random.normal(mu_true, sigma_school_true, J)

# Generate student scores
y = []
school_idx = []

for j in range(J):
    y_j = np.random.normal(school_effects[j], sigma_within_true, n_students)
    y.extend(y_j)
    school_idx.extend([j] * n_students)

y = np.array(y)
school_idx = np.array(school_idx)
n_total = len(y)

print("="*70)
print("Hierarchical Model: Test Scores Across Schools")
print("="*70)
print(f"Schools: {J}")
print(f"Students per school: {n_students}")
print(f"Total students: {n_total}")
print(f"True hyperparameters:")
print(f"  Overall mean: {mu_true}")
print(f"  Between-school SD: {sigma_school_true}")
print(f"  Within-school SD: {sigma_within_true}")
print("")

# Fit hierarchical model (simplified Bayesian via conjugate priors)
# Prior: μ ~ N(70, 10²), σ²_school ~ IG(2, 1), σ²_within ~ IG(2, 50)

# Estimate school effects using empirical Bayes (simpler than full MCMC)
# Posterior: α_j | μ, σ²_school, σ²_within is normal (mixture of data & prior)

# Step 1: Estimate σ²_within (within-school variance)
within_vars = []
for j in range(J):
    y_j = y[school_idx == j]
    var_j = np.var(y_j, ddof=1)
    within_vars.append(var_j)

sigma_within_est = np.sqrt(np.mean(within_vars))

# Step 2: Estimate school means
school_means = np.array([y[school_idx == j].mean() for j in range(J)])

# Step 3: Estimate between-school variance (method of moments)
grand_mean = y.mean()
between_var_est = np.var(school_means, ddof=1) - sigma_within_est**2 / n_students
sigma_school_est = np.sqrt(max(between_var_est, 0.01))  # Ensure positive

mu_est = grand_mean

print("Empirical Bayes Estimation:")
print("-"*70)
print(f"Estimated overall mean: {mu_est:.2f}")
print(f"Estimated between-school SD: {sigma_school_est:.2f}")
print(f"Estimated within-school SD: {sigma_within_est:.2f}")
print("")

# Compute posterior school effects (hierarchical shrinkage)
shrinkage_factor = (sigma_school_est**2) / (sigma_school_est**2 + sigma_within_est**2 / n_students)
alpha_posterior = mu_est + shrinkage_factor * (school_means - mu_est)
se_posterior = np.sqrt(sigma_school_est**2 * (1 - shrinkage_factor))

# Compare methods
print("Posterior Estimates (First 10 Schools):")
print("-"*70)
print(f"{'School':<8} {'No Pooling':<15} {'Pooled':<15} {'Hierarchical':<15} {'SE':<10}")
print("-"*70)

for j in range(10):
    no_pool = school_means[j]
    pool = mu_est
    hier = alpha_posterior[j]
    se = se_posterior[j]
    print(f"{j+1:<8} {no_pool:>13.2f}   {pool:>13.2f}   {hier:>13.2f}   {se:>8.2f}")

# Shrinkage analysis
print("\n" + "="*70)
print("Shrinkage Analysis:")
print("-"*70)
print(f"Shrinkage factor: {shrinkage_factor:.3f}")
print(f"  → Each school estimate pulled {shrinkage_factor*100:.1f}% toward overall mean")
print(f"Posterior SE (hierarchy): {np.mean(se_posterior):.2f}")
print(f"No-pooling SE (group-specific): ~{sigma_within_est/np.sqrt(n_students):.2f}")
print(f"  → Hierarchy reduces uncertainty by pooling")
print("")

# Variance decomposition
icc = sigma_school_est**2 / (sigma_school_est**2 + sigma_within_est**2)
print(f"Intraclass Correlation (ICC): {icc:.3f}")
print(f"  → {icc*100:.1f}% of variance is between schools")
print(f"  → {(1-icc)*100:.1f}% of variance is within schools")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: School effects comparison
ax = axes[0, 0]
schools = np.arange(J)
colors = ['red' if abs(sm - mu_est) > 2*se for sm, se in zip(school_means, se_posterior)] else 'blue' for sm, se in zip(school_means, se_posterior)]

ax.scatter(schools, school_means, alpha=0.5, s=50, color='blue', label='No-pooling (group-specific)')
ax.scatter(schools, alpha_posterior, alpha=0.7, s=50, color='red', marker='^', label='Hierarchical (shrunk)')
ax.axhline(mu_est, color='black', linestyle='--', linewidth=2, label='Estimated mean')
ax.axhline(mu_true, color='green', linestyle=':', linewidth=2, label='True mean')

# Add error bars
for j in range(J):
    ax.plot([j, j], [alpha_posterior[j] - 1.96*se_posterior[j], alpha_posterior[j] + 1.96*se_posterior[j]], 
           'r-', linewidth=1, alpha=0.3)

ax.set_xlabel('School')
ax.set_ylabel('Mean Test Score')
ax.set_title('School Effects: No-Pooling vs Hierarchical')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: Shrinkage visualization
ax = axes[0, 1]
ax.scatter(school_means, alpha_posterior, alpha=0.6, s=50, color='blue')

# Reference lines
min_val = min(school_means.min(), alpha_posterior.min())
max_val = max(school_means.max(), alpha_posterior.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='No shrinkage')
ax.axhline(mu_est, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(mu_est, color='red', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('No-Pooling Estimate')
ax.set_ylabel('Hierarchical Posterior')
ax.set_title('Shrinkage: Points off diagonal show pulling toward mean')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Distribution of school effects
ax = axes[1, 0]
ax.hist(school_means - mu_est, bins=15, alpha=0.5, label='No-pooling residuals', color='blue')
ax.hist(alpha_posterior - mu_est, bins=15, alpha=0.5, label='Hierarchical residuals', color='red')

# Overlay true distribution
x_range = np.linspace(-20, 20, 100)
true_dist = stats.norm.pdf(x_range, 0, sigma_school_true)
ax.plot(x_range, true_dist * J * 3, 'g-', linewidth=2, label='True between-school dist')

ax.set_xlabel('Deviation from Mean')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of School Effects')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Posterior SD by school (uncertainty)
ax = axes[1, 1]
ax.scatter(school_means, se_posterior, alpha=0.6, s=50, color='purple')
ax.axhline(np.mean(se_posterior), color='red', linestyle='--', linewidth=2, label='Average posterior SE')

# Color by school size effect (though all equal in this case)
ax.set_xlabel('School Mean (No-Pooling)')
ax.set_ylabel('Posterior SE (Hierarchical)')
ax.set_title('Uncertainty in School Effects')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hierarchical_model.png', dpi=300, bbox_inches='tight')
plt.show()

# Cross-validation: Predict for new school
print("\n" + "="*70)
print("Prediction for New School (Not in Data):")
print("-"*70)
new_school_mean_posterior = np.random.normal(mu_est, sigma_school_est)
new_students = np.random.normal(new_school_mean_posterior, sigma_within_est, 10)

print(f"Predicted new school effect: {new_school_mean_posterior:.2f}")
print(f"Expected new school mean: ≈ {mu_est:.2f} (with uncertainty {sigma_school_est:.2f})")
print(f"10 new student predictions: {new_students.round(1)}")
print(f"Average of 10 new students: {new_students.mean():.2f}")

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Shrinkage: Extreme school estimates pulled toward mean")
print("   → Reduces overfitting; improves out-of-sample predictions")
print("")
print("2. Partial pooling: Uses both within-school and between-school info")
print("   → Balances group-specific estimates with shared structure")
print("")
print("3. ICC indicates: Most variation within schools, less between")
print("   → But hierarchy still beneficial (regularization)")
print("")
print("4. New school prediction uses hyperprior (μ, σ²_school)")
print("   → Accounts for uncertainty about where new school falls")
```

## 6. Challenge Round
When hierarchical models fail or are misspecified:
- **Too few groups (J<10)**: Variance components underestimated (hyperparameters hard to identify); aggressive shrinkage; use strong weakly informative priors on σ²_group
- **Non-exchangeability**: Groups not similar (e.g., groups A vs B different processes) → Hierarchical assumes exchangeable; solution: stratified priors or mixture model
- **Omitted grouping levels**: Students within classrooms within schools, but model only schools → Biased variances; solution: add classroom level
- **Unbalanced data**: Different n_j per group → Shrinkage varies; smaller groups shrink more (expected); check if by design (ok) or data quality issue (problem)
- **Extreme hyperprior**: Very tight prior on σ²_group → Suppresses heterogeneity; solution: use weakly informative; check prior predictive
- **Computational cost high-d**: Many groups + many parameters → MCMC slow; use HMC/NUTS; or empirical Bayes (approximate but fast)

## 7. Key References
- [Gelman & Hill: Multilevel Modeling (2006)](https://www.cambridge.org/us/academic/subjects/statistics-probability/statistical-theory-and-methods/data-analysis-using-regression-and-multilevelhierarchical-models) - Comprehensive hierarchical models textbook
- [McElreath: Statistical Rethinking (2nd ed, 2020)](https://xcelab.net/rm/statistical-rethinking/) - Bayesian hierarchical models; intuitive explanations
- [Betancourt: Hierarchical Modeling in Stan (2017)](https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html) - Technical; implementation focus

---
**Status:** Advanced Bayesian methodology | **Complements:** Bayesian Inference, Prior Distributions, MCMC, Model Comparison
