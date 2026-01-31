# Prior Distributions

## 1. Concept Skeleton
**Definition:** Probability distributions encoding prior beliefs about parameters before observing data; form first component of Bayes' theorem posterior update  
**Purpose:** Incorporate domain knowledge, regularize estimation in high dimensions, enable sequential learning, express uncertainty about parameters  
**Prerequisites:** Probability distributions, conjugacy, hyperparameters, elicitation methods, sensitivity analysis

## 2. Comparative Framing
| Prior Type | Examples | Conjugacy | Flexibility | Interpretability | Use Case |
|-----------|----------|-----------|------------|-----------------|----------|
| **Informative** | Normal, Beta(α>1) | Often yes | Low (fixed family) | Clear beliefs | Domain expertise strong |
| **Weakly Informative** | Normal(0,10), Beta(2,2) | Sometimes | Moderate | Regularization | Mild beliefs; stability |
| **Flat/Uniform** | Uniform(a,b) | Rarely | High (supports range) | No info | Rare; problematic scale-free |
| **Jeffreys** | ∝√I(θ) for Fisher info | Rarely | Low | Theory-driven | Objective Bayes |
| **Reference** | Derived via information theory | Rare | High | Complex | Objective Bayes advanced |

## 3. Examples + Counterexamples

**Simple Example:**  
Binomial p: Beta(2,2) prior (mean 0.5, variance 0.033); observe 7 heads, 3 tails → Beta(9,5) posterior (mean 0.64, variance 0.013); posterior narrower (more certainty from data)

**Failure Case:**  
Strong incorrect prior Beta(100,2) (believes p≈0.98) with true p=0.1; observe 1 head, 9 tails → Posterior Beta(101,11) (mean≈0.90, still near prior); posterior resists small sample evidence

**Edge Case:**  
Improper prior Beta(0,0) (Jeffreys for Bernoulli) with no data → Posterior undefined; but with any data → Proper posterior exists; used for "least informative" objective Bayes

## 4. Layer Breakdown
```
Prior Distributions Structure:
├─ Informative Priors:
│   ├─ Definition: Encode specific domain beliefs
│   │   ├─ Narrow distribution: High certainty about parameter
│   │   ├─ Location reflects belief; scale reflects confidence
│   │   └─ Dominates likelihood if data weak
│   ├─ Elicitation methods:
│   │   ├─ Expert judgment:
│   │   │   ├─ Ask experts: "What's plausible range for θ?"
│   │   │   ├─ Quantile assessment: P(θ < x₁)=0.25, P(θ < x₂)=0.75
│   │   │   ├─ Match to distribution (Normal, Beta, etc.)
│   │   │   └─ Challenge: Experts' intuitions biased; overconfident
│   │   ├─ Historical data:
│   │   │   ├─ Previous studies; fit posterior as prior for new data
│   │   │   ├─ Meta-analysis: Pool evidence across studies
│   │   │   └─ Caveat: Older studies may reflect different populations
│   │   ├─ Moment matching:
│   │   │   ├─ Specify E[θ] and Var[θ]; solve for distribution parameters
│   │   │   ├─ Example: E[β]=2, Var[β]=1 → Normal(2, 1)
│   │   │   └─ Flexible but loses tail information
│   │   └─ Predictive distribution:
│   │       ├─ Specify beliefs about observable y (not θ)
│   │       ├─ Back-solve for prior on θ
│   │       └─ More intuitive for non-statisticians
│   ├─ Common informative priors:
│   │   ├─ Normal: θ ~ N(μ₀, σ²₀)
│   │   │   ├─ Mean μ₀ = location; variance σ²₀ = confidence
│   │   │   ├─ Symmetric; unimodal (standard choice)
│   │   │   └─ Example: Inflation coefficient β ~ N(0.04, 0.01²)
│   │   ├─ Beta: p ~ Beta(α, β)
│   │   │   ├─ Support [0,1]; mean α/(α+β); variance ∝ 1/(α+β)
│   │   │   ├─ α>1, β>1: Peaked (informative); α=β=1: Uniform
│   │   │   ├─ Conjugate with Binomial/Bernoulli
│   │   │   └─ Example: Market share p ~ Beta(5, 15) (mean 0.25, concentrated)
│   │   ├─ Gamma/Inverse-Gamma: σ² ~ IG(α, δ)
│   │   │   ├─ Shape α, scale δ; conjugate with Normal likelihood
│   │   │   ├─ Concentration: Higher α/δ → narrower prior on variance
│   │   │   └─ Example: Variance σ² ~ IG(2, 0.5) (mean 0.5, confident)
│   │   └─ Dirichlet: (p₁,...,p_K) ~ Dir(α₁,...,α_K)
│   │       ├─ Simplex (sum=1); conjugate with multinomial
│   │       ├─ Mean pᵢ = αᵢ/Σαⱼ; concentration = Σαⱼ
│   │       └─ Example: Market shares ~ Dir(10,20,30) (3 brands, weighted beliefs)
│   └─ Advantages & limitations:
│       ├─ Advantage: Incorporates expert knowledge; regularization
│       ├─ Limitation: Subjectivity; sensitivity to prior; small sample dominance
│       └─ Mitigation: Sensitivity analysis; robustness checks; prior predictive
├─ Weakly Informative Priors:
│   ├─ Motivation: Between informative and flat
│   │   ├─ Goal: Stabilize inference without strong dogmatic beliefs
│   │   ├─ Use: Default choice when prior uncertain
│   │   └─ Rationale: Data large enough to overwhelm; regularize computational problems
│   ├─ Common patterns:
│   │   ├─ Centered at zero (reasonable default):
│   │   │   ├─ β ~ N(0, σ²) with σ large (e.g., 10× data SD)
│   │   │   ├─ Penalizes extreme values; allows data to determine direction
│   │   │   └─ Example: Regression coeff ~ N(0, 10) in standardized units
│   │   ├─ Student-t for robustness:
│   │   │   ├─ β ~ t(ν, 0, σ) with ν low (e.g., 3-7 df)
│   │   │   ├─ Fatter tails than Normal; allows occasional large values
│   │   │   └─ Example: Intercept ~ t(3, 0, 10)
│   │   ├─ Exponential for scale parameters:
│   │   │   ├─ σ ~ Exp(λ) with λ small (e.g., 0.1)
│   │   │   ├─ Support [0,∞); geometric decline
│   │   │   └─ Example: Residual SD ~ Exp(0.1)
│   │   └─ Ordered priors for coeff patterns:
│   │       ├─ (β₁,...,β_K) ordered: β₁<β₂<...<β_K
│   │       ├─ Reflect domain knowledge (e.g., increasing dose → response)
│   │       └─ Computational: Truncated distributions or Hamiltonian MC
│   ├─ Regularization effect:
│   │   ├─ Ridge regression = Normal prior on coefficients
│   │   │   ├─ Bias-variance trade-off: Smaller SE; biased toward 0
│   │   │   ├─ Reduces overfitting (especially high-dimensional)
│   │   │   └─ Penalty λ ~ 1/σ² (strength of prior)
│   │   ├─ LASSO = Laplace prior (spike at 0, tails slow decay)
│   │   │   ├─ Induces sparsity (many coefficients exact zero)
│   │   │   └─ Variable selection built-in
│   │   └─ Elastic net = mixture of priors
│       └─ Compromise between Ridge and LASSO
│   └─ Implementation:
│       ├─ Stan/PyMC: Explicit syntax for weakly informative defaults
│       ├─ Example (Stan): `parameters { real beta; } prior: beta ~ normal(0, 10);`
│       └─ Recommendation: Always specify explicit priors (avoid implicit assumptions)
├─ Non-Informative/Objective Priors:
│   ├─ Philosophy: "Let data speak" without bias
│   │   ├─ Minimize prior influence
│   │   ├─ Paradox: Impossible (all priors biased); compromise goal
│   │   └─ Application: When domain knowledge unavailable; reproducibility
│   ├─ Uniform prior:
│   │   ├─ p(θ) = constant on support
│   │   ├─ Examples: θ ~ Uniform(0,1), β ~ Uniform(-∞,∞)
│   │   ├─ Issues:
│   │   │   ├─ Scale-dependent: Uniform on θ ≠ uniform on log θ
│   │   │   ├─ Improper on unbounded support
│   │   │   └─ Leads to inconsistent marginal inference
│   │   └─ Rarely recommended in practice
│   ├─ Jeffreys Prior:
│   │   ├─ Definition: p(θ) ∝ √(I(θ)) where I(θ) = Fisher information
│   │   │   ├─ Invariant under reparameterization (elegant theory)
│   │   │   ├─ For location: p(θ) = constant (uniform on line)
│   │   │   ├─ For scale: p(σ) ∝ 1/σ (scale-invariant)
│   │   │   └─ For probability: p(p) ∝ p^(-1/2)(1-p)^(-1/2) = Beta(1/2, 1/2)
│   │   ├─ Properties:
│   │   │   ├─ Asymptotically optimal (frequentist coverage accuracy)
│   │   │   ├─ Marginalization-consistent (marginal Jeffreys = Jeffreys for marginal)
│   │   │   └─ Often improper (but yields proper posterior with data)
│   │   ├─ Computation:
│   │   │   ├─ For standard models (Normal, Bernoulli, Exponential): Known formulas
│   │   │   ├─ For complex models: Numerical computation of Fisher information
│   │   │   └─ Software: Often automatic (Stan, PyMC)
│   │   └─ Limitation: Becomes informative in high dimensions (curse of dimensionality)
│   ├─ Reference priors:
│   │   ├─ Extension: Maximize missing information between prior and posterior
│   │   ├─ Berger-Bernardo framework (advanced theory)
│   │   ├─ Properties: Data-dependent; context-specific; optimal asymptotic coverage
│   │   └─ Computational: Complex; rarely implemented outside specialized software
│   └─ Maximal entropy priors:
│       ├─ Idea: Choose prior with highest entropy subject to constraints
│       ├─ Example: If only know E[θ]=μ → MaxEnt: Exponential family with mean μ
│       ├─ Theory: Incorporates moment constraints only; minimal additional info
│       └─ Application: Rare; primarily foundational interest
├─ Hierarchical (Multilevel) Priors:
│   ├─ Motivation: Prior parameters (hyperparameters) themselves random
│   │   ├─ Two-level: θⱼ ~ p(θ|φ), φ ~ p(φ) [hyperprior]
│   │   ├─ Enables borrowing strength across groups (partial pooling)
│   │   └─ Natural for repeated measurements, meta-analysis
│   ├─ Example - student performance:
│   │   ├─ Level 1: Test score Yᵢⱼ ~ N(θⱼ, σ²_within) [student i, school j]
│   │   ├─ Level 2: School effects θⱼ ~ N(μ, σ²_between)
│   │   ├─ Hyperprior: μ ~ N(μ₀, σ²₀), σ²_between ~ IG(α, δ)
│   │   ├─ Interpretation: Schools share common distribution; variance estimated
│   │   └─ Benefits: Regularization; predictions for new schools
│   ├─ Advantages:
│   │   ├─ Flexible: Can model group-level heterogeneity
│   │   ├─ Principled partial pooling (between complete pooling and no pooling)
│   │   ├─ Enables predictions for new groups
│   │   └─ Asymptotically (groups → ∞) recovers group-specific estimates
│   └─ Computation: MCMC required (Gibbs sampling or HMC)
├─ Empirical Bayes:
│   ├─ Idea: Estimate hyperprior from data; use as prior
│   │   ├─ Two-stage: (1) Estimate φ from marginal likelihood; (2) Posterior using p(θ|φ̂)
│   │   ├─ Quasi-Bayesian: Prior not fully Bayesian (φ fixed, not marginalized)
│   │   └─ Pragmatic: Reduce hyperparameter sensitivity; stable estimates
│   ├─ Method of moments:
│   │   ├─ Match sample moments (group means, variances) to prior expectation
│   │   ├─ Estimate hyperparameters analytically
│   │   └─ Fast: No MCMC needed
│   ├─ Maximum marginal likelihood:
│   │   ├─ φ̂_EB = argmax_φ ∫ p(y|θ)p(θ|φ) dθ
│   │   ├─ Optimization (EM algorithm often used)
│   │   ├─ Accurate but computationally intensive
│   │   └─ Used in applications: Genome-wide association studies, meta-analysis
│   ├─ Limitations:
│   │   ├─ Underestimates posterior uncertainty (ignores φ uncertainty)
│   │   ├─ Can overfit hyperparameters to data
│   │   ├─ Not fully Bayesian (φ treated as fixed)
│   │   └─ Solution: Full hierarchical Bayes with hyperprior
│   └─ When useful: Large number of groups (G→∞); hyperparameter stability more important
├─ Prior Sensitivity Analysis:
│   ├─ Why check: Small priors can strongly influence small samples
│   │   ├─ Robust conclusion: Posterior stable across reasonable priors
│   │   ├─ Fragile: Sensitive posterior; prior dominates or misspecified
│   │   └─ Practice: Always conduct sensitivity analysis
│   ├─ Methods:
│   │   ├─ Grid search: Vary prior parameters; compute posterior
│   │   ├─ Visualize: Plot posteriors across prior range
│   │   ├─ Quantify: Posterior mean, credible interval width
│   │   └─ Interpret: If robust → confidence; if sensitive → caution
│   ├─ Example - regression coefficient:
│   │   ├─ Prior 1: β ~ N(0, 1) → Posterior mean 0.85
│   │   ├─ Prior 2: β ~ N(0, 10) → Posterior mean 0.87
│   │   ├─ Prior 3: β ~ N(1, 1) → Posterior mean 0.89
│   │   ├─ Difference 4% → Robust; reasonable priors lead similar inference
│   │   └─ Decision: Use one prior; note small sensitivity
│   └─ Tools: Stan's prior_sensitivity(); manual plotting in Python/R
└─ Practical Recommendations:
    ├─ Default approach (recommended):
    │   ├─ Use weakly informative priors (Stan default; Gelman et al.)
    │   ├─ Center at 0; scale 2-10× data SD
    │   ├─ Allow data to determine; mild regularization
    │   └─ Check prior predictive distribution (simulate from prior alone)
    ├─ If domain knowledge strong:
    │   ├─ Elicit informative prior from experts
    │   ├─ Conduct prior sensitivity (test robustness)
    │   └─ Report both prior and posterior (transparency)
    ├─ For hierarchical models:
    │   ├─ Specify weakly informative on group-level variance
    │   ├─ Example: σ_group ~ Exp(1/α) where α ≈ prior SD
    │   ├─ Allows data to learn heterogeneity
    │   └─ Improves computational stability (HMC sampling)
    └─ Antipatterns to avoid:
        ├─ Very strong priors without justification (induces bias)
        ├─ Flat priors (scale-dependent; improper issues)
        ├─ Mixing different prior specifications without thought
        ├─ Ignoring prior predictive checks (prior unreasonable for domain)
        └─ Not documenting prior choices (reproducibility)
```

**Key Insight:** Prior selection balances domain knowledge with data fit; informative priors leverage expertise; weakly informative priors provide default regularization; objective priors philosophically appealing but practically limited in high dimensions

## 5. Mini-Project
Prior sensitivity analysis: regression with multiple priors:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set seed
np.random.seed(42)

# Generate synthetic data
n = 50
X = np.random.normal(0, 1, n)
X_design = np.column_stack([np.ones(n), X])
true_beta = np.array([1.0, 2.0])
sigma_true = 0.5
y = X_design @ true_beta + np.random.normal(0, sigma_true, n)

# Frequentist OLS
XtX = X_design.T @ X_design
Xty = X_design.T @ y
beta_ols = np.linalg.solve(XtX, Xty)
residuals = y - X_design @ beta_ols
sigma_ols = np.sqrt(np.sum(residuals**2) / (n - 2))

print("="*70)
print("Prior Sensitivity Analysis: Bayesian Linear Regression")
print("="*70)
print(f"OLS estimates: β₀={beta_ols[0]:.4f}, β₁={beta_ols[1]:.4f}, σ={sigma_ols:.4f}")
print("")

# Define prior specifications to test
priors = {
    'Weak (σ²=100)': {'Sigma_0': np.diag([100, 100])},
    'Moderate (σ²=10)': {'Sigma_0': np.diag([10, 10])},
    'Strong (σ²=1)': {'Sigma_0': np.diag([1, 1])},
    'Informative (μ=[1,2])': {'Sigma_0': np.diag([1, 1]), 'beta_0': np.array([1.0, 2.0])},
    'Misspecified (μ=[0,5])': {'Sigma_0': np.diag([1, 1]), 'beta_0': np.array([0.0, 5.0])},
}

# Compute posteriors for each prior
results = {}

for prior_name, prior_spec in priors.items():
    beta_0_prior = prior_spec.get('beta_0', np.array([0.0, 0.0]))
    Sigma_0_prior = prior_spec['Sigma_0']
    
    # Posterior (conjugate Normal-Normal)
    Sigma_0_inv = np.linalg.inv(Sigma_0_prior)
    Sigma_n_inv = Sigma_0_inv + XtX / (sigma_ols**2)
    Sigma_n = np.linalg.inv(Sigma_n_inv)
    
    beta_n = Sigma_n @ (Sigma_0_inv @ beta_0_prior + Xty / (sigma_ols**2))
    
    se_posterior = np.sqrt(np.diag(Sigma_n) * sigma_ols**2)
    
    results[prior_name] = {
        'beta_n': beta_n,
        'se': se_posterior,
        'Sigma_n': Sigma_n
    }
    
    print(f"{prior_name}:")
    print(f"  β₀ posterior mean: {beta_n[0]:.4f} ± {1.96*se_posterior[0]:.4f}")
    print(f"  β₁ posterior mean: {beta_n[1]:.4f} ± {1.96*se_posterior[1]:.4f}")

print("\n" + "="*70)
print("Sensitivity Assessment:")
print("="*70)

# Extract β₁ posteriors (main parameter of interest)
beta1_means = [results[name]['beta_n'][1] for name in priors.keys()]
beta1_ses = [results[name]['se'][1] for name in priors.keys()]

mean_beta1 = np.mean(beta1_means)
std_beta1 = np.std(beta1_means)
range_beta1 = np.max(beta1_means) - np.min(beta1_means)

print(f"\nβ₁ Posterior Sensitivity:")
print(f"  Mean: {mean_beta1:.4f}")
print(f"  Std Dev: {std_beta1:.4f}")
print(f"  Range: {range_beta1:.4f}")
print(f"  CV (Coefficient of Variation): {std_beta1/mean_beta1*100:.2f}%")

if range_beta1 / mean_beta1 < 0.1:
    print(f"  Interpretation: ROBUST (range < 10% of mean)")
elif range_beta1 / mean_beta1 < 0.25:
    print(f"  Interpretation: MODERATE sensitivity (range 10-25% of mean)")
else:
    print(f"  Interpretation: SENSITIVE (range > 25% of mean)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Posterior means for β₁ across priors
prior_names = list(priors.keys())
ax = axes[0, 0]
colors = ['blue' if 'Mis' not in name and 'Inform' not in name else 'red' if 'Mis' in name else 'green' 
          for name in prior_names]

bars = ax.bar(range(len(prior_names)), beta1_means, color=colors, alpha=0.6, edgecolor='black')
ax.axhline(beta_ols[1], color='black', linestyle='--', linewidth=2, label='OLS estimate')
ax.axhline(true_beta[1], color='green', linestyle='--', linewidth=2, label='True value')

# Add error bars
for i, (mean, se) in enumerate(zip(beta1_means, beta1_ses)):
    ax.plot([i, i], [mean - 1.96*se, mean + 1.96*se], 'k-', linewidth=2)

ax.set_ylabel('β₁ Posterior Mean')
ax.set_title('Prior Sensitivity: β₁ Posterior Estimates')
ax.set_xticks(range(len(prior_names)))
ax.set_xticklabels(prior_names, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Prior vs Posterior for β₁ (moderate prior)
ax = axes[0, 1]
prior_name = 'Moderate (σ²=10)'
beta_0_prior = priors[prior_name].get('beta_0', np.array([0.0, 0.0]))
Sigma_0_prior = priors[prior_name]['Sigma_0']

# Prior distribution
x_range = np.linspace(beta_ols[1] - 2, beta_ols[1] + 2, 100)
prior_dist = stats.norm.pdf(x_range, beta_0_prior[1], np.sqrt(Sigma_0_prior[1, 1]))

# Posterior distribution
posterior_dist = stats.norm.pdf(x_range, results[prior_name]['beta_n'][1], 
                                results[prior_name]['se'][1])

ax.plot(x_range, prior_dist, 'b-', linewidth=2, label='Prior')
ax.plot(x_range, posterior_dist, 'r-', linewidth=2, label='Posterior')
ax.axvline(true_beta[1], color='green', linestyle='--', linewidth=2, label='True β₁')
ax.set_xlabel('β₁')
ax.set_ylabel('Density')
ax.set_title(f'Prior vs Posterior: {prior_name}')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Posterior uncertainty (SE across priors)
ax = axes[1, 0]
ax.bar(range(len(prior_names)), beta1_ses, color='purple', alpha=0.6, edgecolor='black')
ax.set_ylabel('Posterior SE (β₁)')
ax.set_title('Posterior Uncertainty Across Priors')
ax.set_xticks(range(len(prior_names)))
ax.set_xticklabels(prior_names, rotation=45, ha='right', fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Prior vs Posterior credible intervals
ax = axes[1, 1]
y_pos = np.arange(len(prior_names))
credible_width = [1.96 * se for se in beta1_ses]

# Color code: misspecified vs correct
colors_cred = ['red' if 'Mis' in name else 'green' if 'Inform' in name else 'blue' 
               for name in prior_names]

for i, (mean, width, color) in enumerate(zip(beta1_means, credible_width, colors_cred)):
    ax.plot([mean - width/2, mean + width/2], [i, i], 'o-', linewidth=3, color=color, markersize=6)
    ax.plot(mean, i, 'o', markersize=8, color=color)

ax.axvline(beta_ols[1], color='black', linestyle='--', linewidth=2, label='OLS')
ax.set_yticks(y_pos)
ax.set_yticklabels(prior_names, fontsize=8)
ax.set_xlabel('β₁')
ax.set_title('95% Posterior Credible Intervals for β₁')
ax.legend()
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('prior_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Weak priors (large σ²): Posteriors dominated by likelihood → Similar to OLS")
print("   → Less regularization; higher posterior uncertainty")
print("")
print("2. Strong priors (small σ²): Posteriors pulled toward prior")
print("   → More regularization; lower posterior uncertainty")
print("")
print("3. Misspecified informative prior: Posterior biased toward wrong prior")
print("   → Small sample can't overcome strong wrong belief")
print("")
print("4. Correct informative prior: Posterior improved (lower SE)")
print("   → When domain knowledge reliable; use informative priors")
print("")
print("5. Sensitivity robust: Most priors → Similar posteriors (data informative)")
print("   → Recommend reporting range of reasonable priors")
```

## 6. Challenge Round
When prior specification misleads:
- **Misspecified informative**: Prior β~N(5,1) but true β~1 → Posterior ≈4.8 (biased); solution: prior sensitivity analysis; check prior predictive distribution
- **Flat/improper priors**: Uniform on unbounded support → Posterior undefined (improper) if likelihood weak; solution: use weakly informative (Normal, Student-t)
- **Scale sensitivity**: Prior on θ vs log θ → Different inference (non-invariance); solution: Jeffreys priors (invariant) or specify carefully for interpretation
- **High-dimensional curse**: Jeffreys prior becomes informative (concentrates mass); weak uniform becomes too vague; solution: weakly informative + hierarchical structure
- **Hyperparameter uncertainty**: Empirical Bayes fixes hyperpriors → Underestimates uncertainty; solution: full hierarchical Bayes (hyperprior on hyperpriors)
- **Conflicting priors**: Combine studies with incompatible priors → Posterior incoherent; solution: meta-analytical approach (model heterogeneity)

## 7. Key References
- [Gelman et al: Prior Distributions for Variance Parameters (2006)](https://projecteuclid.org/euclid.ba/1340371048) - Weakly informative priors in hierarchical models
- [Kass & Wasserman: Selection of Prior Distributions (1996)](https://www.jstor.org/stable/2291521) - Comprehensive prior elicitation
- [Berger & Pericchi: Objective Bayesian Methods (2001)](https://projecteuclid.org/euclid.ba/1340370944) - Reference priors; objective Bayes

---
**Status:** Core Bayesian methodology | **Complements:** Bayesian Inference, MCMC, Model Comparison, Hierarchical Models
