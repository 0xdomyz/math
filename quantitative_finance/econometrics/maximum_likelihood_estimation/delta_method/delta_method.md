# Delta Method

## 1. Concept Skeleton
**Definition:** Approximates distribution of function g(θ̂) using Taylor expansion; derives variance via gradient; bridges point estimators to transformed parameters  
**Purpose:** Inference for nonlinear transformations, marginal effects computation, odds ratios/IRRs standard errors, hypothesis tests for g(θ)  
**Prerequisites:** Asymptotic normality, Taylor series, chain rule, variance propagation, MLE theory

## 2. Comparative Framing
| Method | Delta Method | Bootstrap | Simulation (Monte Carlo) | Profile Likelihood | Fieller's Method |
|--------|--------------|-----------|--------------------------|-------------------|------------------|
| **Transformation** | g(θ̂) any smooth | Any function | Any function | Ratio focus | Ratio specialized |
| **Approximation** | First-order Taylor | Resampling | Repeated estimation | Exact likelihood | Exact ratio |
| **Validity** | Asymptotic | Asymptotic | Depends on reps | Exact (correct model) | Exact ratio CI |
| **Computation** | Analytical gradient | Intensive | Very intensive | Moderate | Simple ratio |
| **Coverage** | Wald-type (symmetric) | Can be asymmetric | Depends on method | Asymmetric | Asymmetric |
| **Failures** | g' ≈ 0 at θ̂ | Small n, ties | Rare events | Misspecification | Works when Delta fails |

## 3. Examples + Counterexamples

**Classic Example:**  
Logit odds ratio OR = exp(β): Delta method SE(OR) = OR·SE(β). For β̂=0.5 (SE=0.1), OR=1.65 with 95% CI [1.35, 2.01]. Wald test equivalent for β or OR.

**Failure Case:**  
Ratio Y/X near X≈0: Delta method breaks down (gradient unbounded). Fieller's method provides valid CI. Example: Treatment effect / Cost with small cost → SE(ratio) → ∞.

**Edge Case:**  
Log transformation g(θ)=log(θ) with θ̂ near zero: Delta method SE explodes. Bootstrap or bias-corrected transformation (log(θ+c)) more stable.

## 4. Layer Breakdown
```
Delta Method Framework:
├─ Univariate Case (θ scalar):
│   ├─ Setup:
│   │   ├─ θ̂ ~ AN(θ₀, σ²/n) (asymptotically normal)
│   │   ├─ g(·): ℝ → ℝ differentiable
│   │   └─ Want: Distribution of g(θ̂)
│   ├─ First-Order Taylor Expansion:
│   │   ├─ g(θ̂) ≈ g(θ₀) + g'(θ₀)·(θ̂ - θ₀)
│   │   ├─ Linear approximation around true θ₀
│   │   └─ Higher-order terms O((θ̂-θ₀)²) negligible for large n
│   ├─ Asymptotic Distribution:
│   │   ├─ √n[g(θ̂) - g(θ₀)] → N(0, [g'(θ₀)]²·σ²)
│   │   ├─ Variance: Var[g(θ̂)] ≈ [g'(θ₀)]²·Var(θ̂)
│   │   └─ Standard Error: SE[g(θ̂)] = |g'(θ̂)|·SE(θ̂)
│   ├─ Plug-In Estimator:
│   │   ├─ Replace θ₀ with θ̂ (consistency)
│   │   └─ ŜE[g(θ̂)] = |g'(θ̂)|·SE(θ̂)
│   ├─ Confidence Interval:
│   │   ├─ g(θ̂) ± z_α/2·SE[g(θ̂)]
│   │   └─ Symmetric (Wald-type)
│   └─ Examples:
│       ├─ g(θ) = θ²: SE(θ̂²) = 2|θ̂|·SE(θ̂)
│       ├─ g(θ) = exp(θ): SE(exp(θ̂)) = exp(θ̂)·SE(θ̂)
│       ├─ g(θ) = log(θ): SE(log(θ̂)) = SE(θ̂)/|θ̂|
│       └─ g(θ) = 1/θ: SE(1/θ̂) = SE(θ̂)/θ̂²
├─ Multivariate Case (θ vector):
│   ├─ Setup:
│   │   ├─ θ̂ ~ AN(θ₀, Σ/n) where θ ∈ ℝᵏ
│   │   ├─ g(·): ℝᵏ → ℝ (scalar function of vector)
│   │   └─ Gradient: ∇g(θ) = [∂g/∂θ₁, ..., ∂g/∂θₖ]'
│   ├─ Taylor Expansion:
│   │   ├─ g(θ̂) ≈ g(θ₀) + ∇g(θ₀)'·(θ̂ - θ₀)
│   │   └─ First-order approximation (linear)
│   ├─ Asymptotic Distribution:
│   │   ├─ √n[g(θ̂) - g(θ₀)] → N(0, ∇g(θ₀)'·Σ·∇g(θ₀))
│   │   ├─ Variance: Var[g(θ̂)] ≈ ∇g(θ₀)'·Var(θ̂)·∇g(θ₀)
│   │   └─ Quadratic form (sandwich)
│   ├─ Standard Error:
│   │   ├─ SE[g(θ̂)] = √[∇g(θ̂)'·Σ̂·∇g(θ̂)]
│   │   ├─ Gradient evaluated at θ̂
│   │   └─ Σ̂: Estimated covariance matrix of θ̂
│   ├─ Wald Test:
│   │   ├─ H₀: g(θ₀) = c
│   │   ├─ W = [g(θ̂) - c]² / Var[g(θ̂)]
│   │   └─ Under H₀: W ~ χ²(1)
│   └─ Examples:
│       ├─ g(β) = β₁/β₂ (ratio)
│       ├─ g(β) = β₁·β₂ (product)
│       ├─ g(β) = β₁ + β₂ + ... + βₖ (sum)
│       └─ g(β, σ²) = β/σ (standardized effect)
├─ Vector-Valued Functions (g: ℝᵏ → ℝᵐ):
│   ├─ g(θ̂) = [g₁(θ̂), ..., gₘ(θ̂)]'
│   ├─ Jacobian Matrix:
│   │   ├─ J(θ) = ∂g/∂θ' (m × k matrix)
│   │   └─ J_ij = ∂gᵢ/∂θⱼ
│   ├─ Covariance Matrix:
│   │   ├─ Var[g(θ̂)] ≈ J(θ̂)·Var(θ̂)·J(θ̂)'
│   │   └─ m × m matrix
│   └─ Wald Test (Vector):
│       ├─ H₀: g(θ₀) = c (m constraints)
│       ├─ W = [g(θ̂) - c]'·Var[g(θ̂)]⁻¹·[g(θ̂) - c]
│       └─ W ~ χ²(m)
├─ Common Transformations:
│   ├─ Odds Ratio (Logit):
│   │   ├─ OR = exp(β)
│   │   ├─ log(OR) = β → Var[log(OR)] = Var(β)
│   │   ├─ SE(OR) = OR·SE(β)
│   │   └─ CI: exp(β ± z·SE(β)) (symmetric on log scale)
│   ├─ Incidence Rate Ratio (Poisson):
│   │   ├─ IRR = exp(β)
│   │   ├─ SE(IRR) = IRR·SE(β)
│   │   └─ Same structure as odds ratio
│   ├─ Elasticity:
│   │   ├─ η = β·(X̄/Ȳ)
│   │   ├─ SE(η) ≈ (X̄/Ȳ)·SE(β) (treating X̄, Ȳ as fixed)
│   │   └─ More complex if X̄, Ȳ also estimated
│   ├─ Marginal Effects (Logit/Probit):
│   │   ├─ ME = β·φ(Xβ) where φ = pdf
│   │   ├─ Gradient: ∂ME/∂β = φ + β·φ'·X
│   │   └─ SE via multivariate Delta method
│   ├─ Ratio of Coefficients:
│   │   ├─ R = β₁/β₂
│   │   ├─ ∇R = [1/β₂, -β₁/β₂²]'
│   │   └─ SE(R) = √[(1/β₂²)·Var(β₁) + (β₁²/β₄₂)·Var(β₂) - 2(β₁/β₃₂)·Cov(β₁,β₂)]
│   ├─ Correlation from Covariance:
│   │   ├─ ρ = σ₁₂/(σ₁·σ₂)
│   │   ├─ Complex gradient (3 parameters)
│   │   └─ Fisher Z-transform often better
│   └─ Survival Probability:
│       ├─ S(t) = exp(-Λ(t)) where Λ = cumulative hazard
│       └─ SE via Delta method with gradient of S
├─ Computational Methods:
│   ├─ Analytical Gradient:
│   │   ├─ Derive ∂g/∂θ symbolically
│   │   ├─ Most accurate
│   │   └─ Not always feasible for complex g
│   ├─ Numerical Gradient:
│   │   ├─ Finite differences: [g(θ+h) - g(θ-h)]/(2h)
│   │   ├─ Central difference more accurate than forward
│   │   └─ h ≈ 10⁻⁸√θ (balance truncation & rounding)
│   ├─ Automatic Differentiation:
│   │   ├─ Exact gradient via chain rule
│   │   └─ Tools: autograd, JAX, PyTorch
│   └─ Software Implementation:
│       ├─ R: deltamethod() in msm package
│       ├─ Stata: nlcom command
│       └─ Python: statsmodels delta_method or manual
├─ Hypothesis Testing:
│   ├─ Wald Test for g(θ):
│   │   ├─ Test H₀: g(θ) = c
│   │   ├─ Z = [g(θ̂) - c] / SE[g(θ̂)]
│   │   └─ Under H₀: Z ~ N(0,1)
│   ├─ Equivalence to Testing θ:
│   │   ├─ Monotonic g: Test g(θ)=c ⟺ Test θ=g⁻¹(c)
│   │   ├─ P-value identical
│   │   └─ Example: Test OR=1 ⟺ Test β=0
│   ├─ Joint Tests (Multiple Constraints):
│   │   ├─ H₀: g(θ) = c where g: ℝᵏ → ℝᵐ
│   │   ├─ W = [g(θ̂)-c]'·V⁻¹·[g(θ̂)-c] ~ χ²(m)
│   │   └─ V = Var[g(θ̂)] via multivariate Delta
│   └─ Confidence Regions:
│       └─ Invert Wald test: {θ: W(θ) ≤ χ²_α(m)}
├─ Limitations & Alternatives:
│   ├─ Delta Method Fails When:
│   │   ├─ g'(θ₀) = 0: Requires second-order Delta method
│   │   ├─ θ̂ near boundary: Gradient discontinuous
│   │   ├─ g highly nonlinear: First-order poor approximation
│   │   └─ Small sample: Asymptotic approximation inadequate
│   ├─ Second-Order Delta Method:
│   │   ├─ Include Hessian term: g(θ̂) ≈ g + g'·Δ + ½Δ'·H·Δ
│   │   ├─ Var[g(θ̂)] ≈ [g']²·Var(θ̂) + ½·tr(H·Var(θ̂))
│   │   └─ Bias correction: E[g(θ̂)] ≈ g + ½·tr(H·Var(θ̂))
│   ├─ Bootstrap Alternative:
│   │   ├─ Resample data, re-estimate θ̂*, compute g(θ̂*)
│   │   ├─ SE[g(θ̂)] = SD of g(θ̂*) across bootstrap samples
│   │   └─ Works when Delta method fails
│   ├─ Fieller's Method (Ratio):
│   │   ├─ For R = β₁/β₂, construct CI without Delta
│   │   ├─ Exact under normality
│   │   └─ Can be unbounded if β₂ near zero
│   ├─ Profile Likelihood:
│   │   ├─ For each value of g, maximize likelihood
│   │   ├─ CI: {g: ℓ_profile(g) > ℓ_max - χ²_α/2}
│   │   └─ More accurate for nonlinear g
│   └─ Bayesian Posterior:
│       ├─ p(g(θ)|data) via transformation theorem
│       └─ Credible intervals from posterior samples
├─ Software & Practical Implementation:
│   ├─ R Examples:
│   │   ├─ library(msm)
│   │   ├─ deltamethod(~ exp(x1), coef(model), vcov(model))
│   │   └─ Symbolic differentiation
│   ├─ Stata:
│   │   ├─ nlcom (exp(_b[x1]))
│   │   └─ lincom for linear combinations
│   ├─ Python (Manual):
│   │   ├─ grad = numerical_gradient(g, theta_hat)
│   │   ├─ var_g = grad.T @ cov_theta @ grad
│   │   └─ se_g = np.sqrt(var_g)
│   └─ Python (statsmodels):
│       ├─ from statsmodels.stats.moment_helpers import cov_nobs_func
│       └─ Specialized for common transformations
├─ Econometric Applications:
│   ├─ Binary Choice Models:
│   │   ├─ Marginal effects: ∂P/∂X = β·φ(Xβ)
│   │   ├─ Average marginal effect: mean(βφ(Xᵢβ))
│   │   └─ SE via Delta method with gradient matrix
│   ├─ Count Models:
│   │   ├─ IRR = exp(β): SE(IRR) = IRR·SE(β)
│   │   ├─ Predicted counts: E[Y|X] = exp(Xβ)
│   │   └─ Delta method for nonlinear predictions
│   ├─ Duration Models:
│   │   ├─ Hazard ratios: HR = exp(β)
│   │   ├─ Median survival time: t_med = -log(0.5)/λ
│   │   └─ SE via Delta method
│   ├─ Treatment Effects:
│   │   ├─ Relative risk: RR = P(Y=1|D=1)/P(Y=1|D=0)
│   │   ├─ Risk difference: RD = P(Y=1|D=1) - P(Y=1|D=0)
│   │   └─ Delta method for derived quantities
│   └─ Elasticities:
│       ├─ Semi-elasticity: % change Y per unit X
│       ├─ Elasticity: % change Y per % change X
│       └─ Computed at means or individual levels
└─ Extensions:
    ├─ Delta Method for GMM:
    │   ├─ GMM estimates minimize Q(θ) = ḡ'Wḡ
    │   ├─ Var(θ̂) = (G'WG)⁻¹G'WSWG(G'WG)⁻¹
    │   └─ Delta method applies with GMM covariance
    ├─ Delta Method for M-estimators:
    │   ├─ General framework: θ̂ solves Σψ(Wᵢ, θ) = 0
    │   └─ Asymptotic covariance via sandwich formula
    ├─ Simulation-Based Delta Method:
    │   ├─ Draw θ* ~ N(θ̂, Σ̂) many times
    │   ├─ Compute g(θ*) for each draw
    │   └─ SE[g(θ̂)] = SD(g(θ*))
    └─ Machine Learning Integration:
        ├─ Post-selection inference with Delta method
        └─ Debiased LASSO + Delta method for elasticities
```

**Interaction:** Estimate θ̂ with SE → Define transformation g(·) → Compute gradient ∇g(θ̂) → Calculate SE[g(θ̂)] = √[∇g'·Σ̂·∇g] → Wald test or CI

## 5. Mini-Project
Implement Delta method for various transformations with bootstrap comparison:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from scipy.special import expit  # Logistic function
import seaborn as sns

np.random.seed(654)

# ===== Simulate Binary Choice Data =====
print("="*80)
print("DELTA METHOD FOR NONLINEAR TRANSFORMATIONS")
print("="*80)

n = 1000
X1 = np.random.randn(n)
X2 = np.random.randn(n)
X = np.column_stack([np.ones(n), X1, X2])

# True parameters
beta_true = np.array([0.5, 1.2, -0.8])

# Logit model
p_true = expit(X @ beta_true)
Y = (np.random.rand(n) < p_true).astype(int)

print(f"Simulation Setup:")
print(f"  Model: Logit regression")
print(f"  Sample size: {n}")
print(f"  True β: {beta_true}")
print(f"  Outcome prevalence: {Y.mean():.1%}")

# ===== Maximum Likelihood Estimation =====
print("\n" + "="*80)
print("MLE ESTIMATION (LOGIT)")
print("="*80)

def logit_loglik(beta, Y, X):
    """Logit log-likelihood"""
    Xb = X @ beta
    p = expit(Xb)
    # Numerical stability
    ll = Y * np.log(p + 1e-10) + (1 - Y) * np.log(1 - p + 1e-10)
    return -np.sum(ll)  # Negative for minimization

def logit_score(beta, Y, X):
    """Score (gradient) of log-likelihood"""
    Xb = X @ beta
    p = expit(Xb)
    score = X.T @ (Y - p)
    return -score  # Negative for minimization

def logit_hessian(beta, Y, X):
    """Hessian of log-likelihood"""
    Xb = X @ beta
    p = expit(Xb)
    W = p * (1 - p)
    H = X.T @ (X * W[:, np.newaxis])
    return H  # Already positive for minimization

# Optimize
result = minimize(
    logit_loglik,
    np.zeros(3),
    args=(Y, X),
    method='BFGS',
    jac=logit_score
)

beta_hat = result.x
print(f"MLE converged: {result.success}")

# Fisher Information and Covariance
I_hat = logit_hessian(beta_hat, Y, X)
Sigma_hat = np.linalg.inv(I_hat)
se_beta = np.sqrt(np.diag(Sigma_hat))

print(f"\nParameter Estimates:")
for i, (b, b_true, se) in enumerate(zip(beta_hat, beta_true, se_beta)):
    z_stat = (b - b_true) / se
    print(f"  β{i}: {b:7.4f} (SE={se:.4f}, true={b_true:.2f}) [z={z_stat:+.2f}]")

# ===== Transformation 1: Odds Ratio =====
print("\n" + "="*80)
print("TRANSFORMATION 1: ODDS RATIO (Univariate)")
print("="*80)

# OR = exp(β₁)
OR_hat = np.exp(beta_hat[1])
OR_true = np.exp(beta_true[1])

print(f"Odds Ratio for X₁:")
print(f"  OR = exp(β₁)")
print(f"  True OR: {OR_true:.4f}")
print(f"  Estimated OR: {OR_hat:.4f}")

# Delta Method
# g(β₁) = exp(β₁) → g'(β₁) = exp(β₁)
grad_OR = np.exp(beta_hat[1])
var_OR_delta = grad_OR**2 * Sigma_hat[1, 1]
se_OR_delta = np.sqrt(var_OR_delta)

print(f"\nDelta Method:")
print(f"  Gradient g'(β₁) = exp(β₁) = {grad_OR:.4f}")
print(f"  Var(OR) = [g']²·Var(β₁) = {var_OR_delta:.6f}")
print(f"  SE(OR) = OR·SE(β₁) = {se_OR_delta:.4f}")

# Confidence Interval
ci_OR_lower = OR_hat - 1.96 * se_OR_delta
ci_OR_upper = OR_hat + 1.96 * se_OR_delta

print(f"  95% CI (Delta): [{ci_OR_lower:.4f}, {ci_OR_upper:.4f}]")

# Alternative: Transform CI on log scale (exact for monotonic transformation)
ci_log_lower = beta_hat[1] - 1.96 * se_beta[1]
ci_log_upper = beta_hat[1] + 1.96 * se_beta[1]
ci_OR_exact_lower = np.exp(ci_log_lower)
ci_OR_exact_upper = np.exp(ci_log_upper)

print(f"  95% CI (Log-transform): [{ci_OR_exact_lower:.4f}, {ci_OR_exact_upper:.4f}]")
print(f"  ✓ Log-transform CI better for ratio parameters")

# Wald Test: H₀: OR = 1 ⟺ β₁ = 0
z_OR = (OR_hat - 1) / se_OR_delta
p_val_OR = 2 * (1 - stats.norm.cdf(abs(z_OR)))

z_beta = beta_hat[1] / se_beta[1]
p_val_beta = 2 * (1 - stats.norm.cdf(abs(z_beta)))

print(f"\nWald Test H₀: OR=1:")
print(f"  Z-stat (on OR): {z_OR:.4f} (p={p_val_OR:.4f})")
print(f"  Z-stat (on β₁): {z_beta:.4f} (p={p_val_beta:.4f})")
print(f"  ⚠ Tests not identical (Delta method approximate)")

# ===== Transformation 2: Marginal Effect =====
print("\n" + "="*80)
print("TRANSFORMATION 2: MARGINAL EFFECT (Multivariate)")
print("="*80)

# ME = ∂P/∂X₁ = β₁·φ(Xβ) evaluated at mean(X)
X_mean = X.mean(axis=0)
Xb_mean = X_mean @ beta_hat
p_mean = expit(Xb_mean)
phi_mean = p_mean * (1 - p_mean)  # Logistic pdf

ME_hat = beta_hat[1] * phi_mean

print(f"Average Marginal Effect (AME) at X̄:")
print(f"  ME = β₁·φ(X̄β)")
print(f"  X̄β = {Xb_mean:.4f}")
print(f"  P(Y=1|X̄) = {p_mean:.4f}")
print(f"  φ(X̄β) = P·(1-P) = {phi_mean:.4f}")
print(f"  ME = {ME_hat:.6f}")

# Delta Method (Multivariate)
# g(β) = β₁·expit(Xβ)·(1 - expit(Xβ))
# Gradient: ∂g/∂β = [∂g/∂β₀, ∂g/∂β₁, ∂g/∂β₂]

# ∂ME/∂β₁ = φ(Xβ) + β₁·φ'(Xβ)·X₁
# ∂ME/∂βⱼ = β₁·φ'(Xβ)·Xⱼ for j≠1
# φ'(z) = φ(z)·(1 - 2·expit(z))

phi_prime_mean = phi_mean * (1 - 2 * p_mean)

grad_ME = np.zeros(3)
grad_ME[1] = phi_mean + beta_hat[1] * phi_prime_mean * X_mean[1]
grad_ME[0] = beta_hat[1] * phi_prime_mean * X_mean[0]
grad_ME[2] = beta_hat[1] * phi_prime_mean * X_mean[2]

# Variance: ∇g'·Σ·∇g
var_ME_delta = grad_ME @ Sigma_hat @ grad_ME
se_ME_delta = np.sqrt(var_ME_delta)

print(f"\nDelta Method:")
print(f"  Gradient ∇g(β) = {grad_ME}")
print(f"  Var(ME) = ∇g'·Σ·∇g = {var_ME_delta:.8f}")
print(f"  SE(ME) = {se_ME_delta:.6f}")

ci_ME_lower = ME_hat - 1.96 * se_ME_delta
ci_ME_upper = ME_hat + 1.96 * se_ME_delta

print(f"  95% CI: [{ci_ME_lower:.6f}, {ci_ME_upper:.6f}]")

# Wald Test: H₀: ME = 0
z_ME = ME_hat / se_ME_delta
p_val_ME = 2 * (1 - stats.norm.cdf(abs(z_ME)))

print(f"  Wald test: Z={z_ME:.4f}, p={p_val_ME:.4f}")

# ===== Transformation 3: Ratio of Coefficients =====
print("\n" + "="*80)
print("TRANSFORMATION 3: RATIO OF COEFFICIENTS")
print("="*80)

# R = β₁ / β₂
R_hat = beta_hat[1] / beta_hat[2]
R_true = beta_true[1] / beta_true[2]

print(f"Coefficient Ratio β₁/β₂:")
print(f"  True ratio: {R_true:.4f}")
print(f"  Estimated ratio: {R_hat:.4f}")

# Delta Method
# g(β₁, β₂) = β₁/β₂
# ∇g = [0, 1/β₂, -β₁/β₂²]'

grad_R = np.array([
    0,
    1 / beta_hat[2],
    -beta_hat[1] / beta_hat[2]**2
])

var_R_delta = grad_R @ Sigma_hat @ grad_R
se_R_delta = np.sqrt(var_R_delta)

print(f"\nDelta Method:")
print(f"  Gradient ∇g = {grad_R}")
print(f"  Var(R) = {var_R_delta:.6f}")
print(f"  SE(R) = {se_R_delta:.4f}")

# Expanded formula
var_R_expanded = (
    (1 / beta_hat[2]**2) * Sigma_hat[1, 1] +
    (beta_hat[1]**2 / beta_hat[2]**4) * Sigma_hat[2, 2] -
    2 * (beta_hat[1] / beta_hat[2]**3) * Sigma_hat[1, 2]
)
print(f"  Var(R) expanded: {var_R_expanded:.6f} (same)")

ci_R_lower = R_hat - 1.96 * se_R_delta
ci_R_upper = R_hat + 1.96 * se_R_delta

print(f"  95% CI: [{ci_R_lower:.4f}, {ci_R_upper:.4f}]")

# ===== Bootstrap Comparison =====
print("\n" + "="*80)
print("BOOTSTRAP COMPARISON")
print("="*80)

n_boot = 1000
print(f"Running {n_boot} bootstrap samples...")

OR_boot = np.zeros(n_boot)
ME_boot = np.zeros(n_boot)
R_boot = np.zeros(n_boot)

for b in range(n_boot):
    # Resample
    idx = np.random.choice(n, size=n, replace=True)
    Y_b = Y[idx]
    X_b = X[idx]
    
    # Re-estimate
    try:
        result_b = minimize(
            logit_loglik,
            beta_hat,  # Start from full sample estimate
            args=(Y_b, X_b),
            method='BFGS',
            options={'disp': False}
        )
        
        if result_b.success:
            beta_b = result_b.x
            
            # Compute transformations
            OR_boot[b] = np.exp(beta_b[1])
            
            Xb_mean_b = X_mean @ beta_b
            p_mean_b = expit(Xb_mean_b)
            phi_mean_b = p_mean_b * (1 - p_mean_b)
            ME_boot[b] = beta_b[1] * phi_mean_b
            
            R_boot[b] = beta_b[1] / beta_b[2]
        else:
            OR_boot[b] = np.nan
            ME_boot[b] = np.nan
            R_boot[b] = np.nan
    except:
        OR_boot[b] = np.nan
        ME_boot[b] = np.nan
        R_boot[b] = np.nan

# Remove failed bootstraps
OR_boot = OR_boot[~np.isnan(OR_boot)]
ME_boot = ME_boot[~np.isnan(ME_boot)]
R_boot = R_boot[~np.isnan(R_boot)]

# Bootstrap standard errors
se_OR_boot = np.std(OR_boot, ddof=1)
se_ME_boot = np.std(ME_boot, ddof=1)
se_R_boot = np.std(R_boot, ddof=1)

# Bootstrap percentile CIs
ci_OR_boot = np.percentile(OR_boot, [2.5, 97.5])
ci_ME_boot = np.percentile(ME_boot, [2.5, 97.5])
ci_R_boot = np.percentile(R_boot, [2.5, 97.5])

print(f"Bootstrap Results ({len(OR_boot)} successful):")

print(f"\nOdds Ratio:")
print(f"  Delta Method SE: {se_OR_delta:.4f}")
print(f"  Bootstrap SE: {se_OR_boot:.4f}")
print(f"  Difference: {abs(se_OR_delta - se_OR_boot):.4f}")
print(f"  Bootstrap CI: [{ci_OR_boot[0]:.4f}, {ci_OR_boot[1]:.4f}]")
print(f"  Delta CI: [{ci_OR_lower:.4f}, {ci_OR_upper:.4f}]")

print(f"\nMarginal Effect:")
print(f"  Delta Method SE: {se_ME_delta:.6f}")
print(f"  Bootstrap SE: {se_ME_boot:.6f}")
print(f"  Difference: {abs(se_ME_delta - se_ME_boot):.6f}")
print(f"  Bootstrap CI: [{ci_ME_boot[0]:.6f}, {ci_ME_boot[1]:.6f}]")
print(f"  Delta CI: [{ci_ME_lower:.6f}, {ci_ME_upper:.6f}]")

print(f"\nRatio:")
print(f"  Delta Method SE: {se_R_delta:.4f}")
print(f"  Bootstrap SE: {se_R_boot:.4f}")
print(f"  Difference: {abs(se_R_delta - se_R_boot):.4f}")
print(f"  Bootstrap CI: [{ci_R_boot[0]:.4f}, {ci_R_boot[1]:.4f}]")
print(f"  Delta CI: [{ci_R_lower:.4f}, {ci_R_upper:.4f}]")

# ===== Numerical vs Analytical Gradient =====
print("\n" + "="*80)
print("NUMERICAL GRADIENT COMPARISON")
print("="*80)

def numerical_gradient(func, theta, h=1e-8):
    """Compute numerical gradient via central differences"""
    k = len(theta)
    grad = np.zeros(k)
    
    for i in range(k):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += h
        theta_minus[i] -= h
        
        grad[i] = (func(theta_plus) - func(theta_minus)) / (2 * h)
    
    return grad

# Marginal Effect function
def ME_func(beta):
    Xb = X_mean @ beta
    p = expit(Xb)
    phi = p * (1 - p)
    return beta[1] * phi

grad_ME_numerical = numerical_gradient(ME_func, beta_hat)

print(f"Marginal Effect Gradient:")
print(f"  Analytical: {grad_ME}")
print(f"  Numerical:  {grad_ME_numerical}")
print(f"  Max difference: {np.max(np.abs(grad_ME - grad_ME_numerical)):.2e}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Odds Ratio Distribution
axes[0, 0].hist(OR_boot, bins=40, density=True, alpha=0.6, 
                label='Bootstrap', color='blue')
x_grid = np.linspace(OR_boot.min(), OR_boot.max(), 200)
pdf_delta = stats.norm.pdf(x_grid, OR_hat, se_OR_delta)
axes[0, 0].plot(x_grid, pdf_delta, 'r-', linewidth=2, 
                label='Delta Method')
axes[0, 0].axvline(OR_true, color='green', linestyle='--', 
                  linewidth=2, label='True')
axes[0, 0].axvline(OR_hat, color='black', linestyle='-', 
                  linewidth=2, label='Estimate')
axes[0, 0].set_xlabel('Odds Ratio')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Odds Ratio: Delta vs Bootstrap')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Marginal Effect Distribution
axes[0, 1].hist(ME_boot, bins=40, density=True, alpha=0.6, 
                label='Bootstrap', color='blue')
x_grid_ME = np.linspace(ME_boot.min(), ME_boot.max(), 200)
pdf_delta_ME = stats.norm.pdf(x_grid_ME, ME_hat, se_ME_delta)
axes[0, 1].plot(x_grid_ME, pdf_delta_ME, 'r-', linewidth=2, 
                label='Delta Method')
axes[0, 1].axvline(ME_hat, color='black', linestyle='-', 
                  linewidth=2, label='Estimate')
axes[0, 1].set_xlabel('Marginal Effect')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Marginal Effect: Delta vs Bootstrap')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Ratio Distribution
axes[0, 2].hist(R_boot, bins=40, density=True, alpha=0.6, 
                label='Bootstrap', color='blue')
x_grid_R = np.linspace(R_boot.min(), R_boot.max(), 200)
pdf_delta_R = stats.norm.pdf(x_grid_R, R_hat, se_R_delta)
axes[0, 2].plot(x_grid_R, pdf_delta_R, 'r-', linewidth=2, 
                label='Delta Method')
axes[0, 2].axvline(R_true, color='green', linestyle='--', 
                  linewidth=2, label='True')
axes[0, 2].axvline(R_hat, color='black', linestyle='-', 
                  linewidth=2, label='Estimate')
axes[0, 2].set_xlabel('β₁/β₂')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_title('Coefficient Ratio: Delta vs Bootstrap')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3)

# Plot 4: SE Comparison
methods = ['Delta\nMethod', 'Bootstrap']
OR_ses = [se_OR_delta, se_OR_boot]
ME_ses = [se_ME_delta, se_ME_boot]
R_ses = [se_R_delta, se_R_boot]

x_pos = np.arange(2)
width = 0.25

axes[1, 0].bar(x_pos - width, OR_ses, width, label='Odds Ratio', alpha=0.7)
axes[1, 0].bar(x_pos, [me * 10 for me in ME_ses], width, 
               label='Marg. Eff. (×10)', alpha=0.7)
axes[1, 0].bar(x_pos + width, R_ses, width, label='Ratio', alpha=0.7)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(methods)
axes[1, 0].set_ylabel('Standard Error')
axes[1, 0].set_title('SE Comparison Across Methods')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 5: Confidence Interval Comparison
transformations = ['Odds\nRatio', 'Marginal\nEffect', 'Ratio']
delta_widths = [
    ci_OR_upper - ci_OR_lower,
    ci_ME_upper - ci_ME_lower,
    ci_R_upper - ci_R_lower
]
boot_widths = [
    ci_OR_boot[1] - ci_OR_boot[0],
    ci_ME_boot[1] - ci_ME_boot[0],
    ci_R_boot[1] - ci_R_boot[0]
]

x_pos = np.arange(len(transformations))
axes[1, 1].bar(x_pos - 0.2, delta_widths, 0.4, label='Delta Method', alpha=0.7)
axes[1, 1].bar(x_pos + 0.2, boot_widths, 0.4, label='Bootstrap', alpha=0.7)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(transformations)
axes[1, 1].set_ylabel('CI Width')
axes[1, 1].set_title('95% Confidence Interval Width')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Q-Q Plot (Bootstrap vs Delta for OR)
# Standardize bootstrap samples
OR_boot_std = (OR_boot - OR_hat) / se_OR_boot
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(OR_boot_std)))
empirical_quantiles = np.sort(OR_boot_std)

axes[1, 2].scatter(theoretical_quantiles, empirical_quantiles, 
                  alpha=0.5, s=20)
axes[1, 2].plot([-3, 3], [-3, 3], 'r--', linewidth=2, 
               label='Perfect Normal')
axes[1, 2].set_xlabel('Theoretical Quantiles')
axes[1, 2].set_ylabel('Bootstrap Quantiles')
axes[1, 2].set_title('Q-Q Plot: Bootstrap vs Normal\n(Odds Ratio)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('delta_method_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Delta Method Accuracy:")
for name, se_delta, se_boot in [
    ('Odds Ratio', se_OR_delta, se_OR_boot),
    ('Marginal Effect', se_ME_delta, se_ME_boot),
    ('Ratio', se_R_delta, se_R_boot)
]:
    diff_pct = abs(se_delta - se_boot) / se_boot * 100
    print(f"   {name}: {diff_pct:.1f}% difference")

print("\n2. CI Coverage:")
print(f"   Delta vs Bootstrap CI widths comparable")
print(f"   Bootstrap CI can be asymmetric (better for ratios)")

print("\n3. Computational Cost:")
print(f"   Delta Method: Instant (analytical gradient)")
print(f"   Bootstrap: {n_boot} optimizations ({n_boot} × slower)")

print("\n4. Practical Guidelines:")
print("   • Use Delta Method for standard transformations (OR, IRR)")
print("   • Transform CI for ratios (e.g., exp(β ± z·SE))")
print("   • Bootstrap when gradient complex or near boundary")
print("   • Numerical gradient accurate (h ≈ 1e-8)")
print("   • Multivariate case requires full covariance matrix")

print("\n5. Common Pitfalls:")
print("   ⚠ Forgetting covariance terms in multivariate case")
print("   ⚠ Testing on transformed scale without adjustment")
print("   ⚠ Using Delta method near g'(θ) = 0")
print("   ⚠ Symmetric CI for highly skewed transformations")

print("\n6. When Delta Method Fails:")
print("   • g'(θ₀) = 0: Need second-order Delta method")
print("   • Small samples: Bootstrap more reliable")
print("   • Extreme nonlinearity: Profile likelihood better")
print("   • Ratio with denominator near zero: Fieller's method")
```

## 6. Challenge Round
When does Delta method fail or mislead?
- **Zero gradient**: g'(θ₀)=0 (e.g., minimum of function) → First-order Delta breaks; second-order required with E[g(θ̂)]≈g(θ)+½tr(H·Var(θ̂))
- **Near-boundary estimation**: θ̂ at constraint boundary (e.g., variance=0) → Gradient discontinuous; bootstrap or likelihood-based inference
- **Ratio with small denominator**: R=β₁/β₂ with β₂≈0 → SE(R)→∞; Fieller's method constructs valid CI even when unbounded
- **Highly nonlinear g**: Exponential growth, extreme transformation → First-order Taylor poor approximation; simulation or profile likelihood
- **Small samples**: n<30 → Asymptotic approximation inadequate; bootstrap or exact methods
- **Misspecified covariance**: Wrong Σ̂ (ignoring correlation) → Incorrect SE; always use full covariance matrix

## 7. Key References
- [Oehlert (1992) - A Note on the Delta Method](https://www.jstor.org/stable/2684406)
- [Ver Hoef (2012) - Who Invented the Delta Method?](https://doi.org/10.1198/000313007X247942)
- [Dorfman (1938) - A Note on the δ-Method for Finding Variance Formulae](https://www.jstor.org/stable/2235515)

---
**Status:** Standard for transformed parameter inference | **Complements:** MLE, GMM, Bootstrap, Profile Likelihood
