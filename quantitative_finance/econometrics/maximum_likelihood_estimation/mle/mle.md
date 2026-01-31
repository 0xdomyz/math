# Maximum Likelihood Estimation (MLE)

## 1. Concept Skeleton
**Definition:** Parameter estimation by maximizing likelihood function; chooses parameters making observed data most probable under assumed model; asymptotically efficient under regularity conditions  
**Purpose:** Estimate unknown parameters; construct test statistics; derive confidence intervals; foundation for most econometric estimation; flexible framework  
**Prerequisites:** Probability distributions, likelihood functions, optimization, calculus, asymptotic theory, Fisher information, log-likelihood

## 2. Comparative Framing
| Method | OLS | MLE | GMM | Method of Moments | Bayesian |
|--------|-----|-----|-----|-------------------|----------|
| **Approach** | Minimize RSS | Maximize likelihood | Minimize moment distance | Equate sample/population moments | Posterior distribution |
| **Distribution Assumption** | Not required | Required | Not required | Not required | Prior + likelihood |
| **Efficiency** | Efficient (normal errors) | Efficient (correct spec) | Efficient (optimal weights) | Generally inefficient | Depends on prior |
| **Robustness** | Moderate | Low (misspecification) | High | High | Prior-dependent |
| **Standard Errors** | Closed form | Information matrix | Sandwich | Asymptotic | Credible intervals |
| **Computation** | Closed form | Numerical optimization | Numerical optimization | Simple | MCMC often needed |

## 3. Examples + Counterexamples

**Classic Example:**  
Linear regression with normal errors: MLE = OLS. But with heteroskedasticity, MLE accounts for variance structure → Efficiency gains. Tobit model (censored data): OLS biased, MLE consistent by modeling censoring mechanism.

**Failure Case:**  
Heavy-tailed data (Student-t true distribution) estimated with Gaussian MLE. Outliers given excessive weight. MLE inefficient, standard errors invalid. Robust M-estimators or correct t-distribution better.

**Edge Case:**  
Perfect separation in logit (predictor perfectly predicts outcome). MLE diverges (β→∞). Firth's penalized likelihood or Bayesian prior stabilizes. Finite-sample bias correction needed.

## 4. Layer Breakdown
```
Maximum Likelihood Estimation Framework:
├─ Likelihood Function:
│   ├─ Joint PDF: L(θ|data) = f(x₁,...,xₙ|θ)
│   │   ├─ θ: Parameter vector to estimate
│   │   ├─ xᵢ: Observed data
│   │   └─ f: Probability density/mass function
│   ├─ Independence Assumption:
│   │   ├─ L(θ) = ∏f(xᵢ|θ) if i.i.d.
│   │   └─ Simplifies computation
│   ├─ Interpretation:
│   │   ├─ Measures "support" data gives to θ
│   │   └─ Not probability of θ (frequentist view)
│   └─ Time Series Extension:
│       └─ L(θ) = ∏f(yₜ|yₜ₋₁,...,y₁,θ) (conditional likelihoods)
├─ Log-Likelihood:
│   ├─ ℓ(θ) = log L(θ) = Σlog f(xᵢ|θ)
│   ├─ Advantages:
│   │   ├─ Sum instead of product (numerical stability)
│   │   ├─ Easier differentiation
│   │   └─ Monotonic transformation preserves max
│   ├─ Concentrated Log-Likelihood:
│   │   ├─ Profile out nuisance parameters
│   │   └─ Example: σ² in regression
│   └─ Conditional vs Marginal:
│       ├─ Conditional: ℓ(β|σ²,data)
│       └─ Marginal: Integrate out nuisance parameters
├─ MLE Principle:
│   ├─ Definition: θ̂_MLE = argmax_θ L(θ) = argmax_θ ℓ(θ)
│   ├─ First-Order Condition (Score):
│   │   ├─ S(θ) = ∂ℓ(θ)/∂θ = 0 (score equation)
│   │   ├─ Score: Gradient of log-likelihood
│   │   └─ E[S(θ₀)] = 0 at true parameter θ₀
│   ├─ Second-Order Condition:
│   │   ├─ H(θ) = ∂²ℓ(θ)/∂θ∂θ' < 0 (negative definite Hessian)
│   │   └─ Ensures maximum, not saddle point
│   └─ Global vs Local:
│       ├─ May have multiple local maxima
│       └─ Try multiple starting values
├─ Examples:
│   ├─ Normal Distribution:
│   │   ├─ Data: X ~ N(μ, σ²)
│   │   ├─ Log-likelihood: ℓ = -n/2·log(2π) - n/2·log(σ²) - Σ(xᵢ-μ)²/(2σ²)
│   │   ├─ MLE: μ̂ = X̄, σ̂² = Σ(xᵢ-X̄)²/n
│   │   └─ Note: σ̂² biased (n instead of n-1)
│   ├─ Linear Regression (Normal Errors):
│   │   ├─ Yᵢ = Xᵢ'β + εᵢ, εᵢ ~ N(0,σ²)
│   │   ├─ ℓ(β,σ²) = -n/2·log(2π) - n/2·log(σ²) - Σ(yᵢ-Xᵢ'β)²/(2σ²)
│   │   ├─ β̂_MLE = (X'X)⁻¹X'Y (same as OLS!)
│   │   └─ σ̂²_MLE = RSS/n (biased downward)
│   ├─ Logit Model:
│   │   ├─ P(Yᵢ=1|Xᵢ) = Λ(Xᵢ'β) = exp(Xᵢ'β)/(1+exp(Xᵢ'β))
│   │   ├─ ℓ(β) = Σ[yᵢ·Xᵢ'β - log(1+exp(Xᵢ'β))]
│   │   └─ No closed form, numerical optimization
│   ├─ Poisson Regression:
│   │   ├─ Yᵢ ~ Poisson(exp(Xᵢ'β))
│   │   ├─ ℓ(β) = Σ[yᵢ·Xᵢ'β - exp(Xᵢ'β) - log(yᵢ!)]
│   │   └─ Score: S(β) = Σ(yᵢ - exp(Xᵢ'β))Xᵢ
│   └─ Tobit (Censored):
│       ├─ Y* = X'β + ε, Y = max(0, Y*)
│       ├─ ℓ = Σ_uncens log φ((yᵢ-Xᵢ'β)/σ) + Σ_cens log Φ(-Xᵢ'β/σ)
│       └─ Two-part likelihood (continuous + point mass)
├─ Asymptotic Properties:
│   ├─ Consistency:
│   │   ├─ θ̂_MLE →ᵖ θ₀ as n → ∞
│   │   ├─ Under regularity conditions
│   │   └─ Identification: θ₀ uniquely maximizes E[ℓ(θ)]
│   ├─ Asymptotic Normality:
│   │   ├─ √n(θ̂_MLE - θ₀) →ᵈ N(0, I(θ₀)⁻¹)
│   │   ├─ I(θ₀): Fisher information matrix
│   │   └─ Fastest convergence rate (√n)
│   ├─ Efficiency:
│   │   ├─ Cramér-Rao Lower Bound: Var(θ̂) ≥ I(θ)⁻¹
│   │   ├─ MLE achieves bound asymptotically
│   │   └─ Most efficient among consistent estimators
│   ├─ Invariance:
│   │   ├─ If θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ)
│   │   └─ Useful for transformations (e.g., odds ratios)
│   └─ Regularity Conditions:
│       ├─ Parameter space open
│       ├─ PDF continuous, differentiable
│       ├─ Support doesn't depend on θ
│       ├─ Dominated convergence holds
│       └─ Fisher information finite, positive definite
├─ Fisher Information:
│   ├─ Definition: I(θ) = -E[∂²ℓ(θ)/∂θ∂θ']
│   ├─ Alternative: I(θ) = E[S(θ)S(θ)'] (outer product of score)
│   ├─ Information Matrix Equality:
│   │   └─ Two definitions equal under regularity
│   ├─ Interpretation:
│   │   ├─ Curvature of log-likelihood
│   │   ├─ Amount of information data carries
│   │   └─ Higher I → More precise estimates
│   ├─ Sample Information:
│   │   ├─ Î = -∂²ℓ(θ̂)/∂θ∂θ' (Hessian at MLE)
│   │   └─ Or: Î = ΣS(θ̂;xᵢ)S(θ̂;xᵢ)' (BHHH estimator)
│   └─ Standard Errors:
│       └─ SE(θ̂) = √diag(Î⁻¹) (square root of diagonal)
├─ Numerical Optimization:
│   ├─ Newton-Raphson:
│   │   ├─ θ^(k+1) = θ^(k) - H(θ^(k))⁻¹S(θ^(k))
│   │   ├─ Uses second derivatives (Hessian)
│   │   ├─ Fast convergence (quadratic)
│   │   └─ Requires Hessian computation/inversion
│   ├─ BFGS (Quasi-Newton):
│   │   ├─ Approximate Hessian via gradients
│   │   ├─ No second derivatives needed
│   │   └─ Good balance speed/accuracy
│   ├─ Gradient Ascent:
│   │   ├─ θ^(k+1) = θ^(k) + α·S(θ^(k))
│   │   ├─ Only first derivatives
│   │   └─ Slow but robust
│   ├─ EM Algorithm:
│   │   ├─ For models with latent variables
│   │   ├─ E-step: Expected complete log-likelihood
│   │   ├─ M-step: Maximize w.r.t. parameters
│   │   └─ Examples: Mixture models, missing data
│   ├─ Starting Values:
│   │   ├─ Critical for convergence
│   │   ├─ Try multiple random starts
│   │   ├─ Use simpler estimator (OLS, moments)
│   │   └─ Check likelihood at initial values
│   └─ Convergence Criteria:
│       ├─ |θ^(k+1) - θ^(k)| < tol
│       ├─ |ℓ(θ^(k+1)) - ℓ(θ^(k))| < tol
│       └─ ||S(θ^(k))|| < tol
├─ Hypothesis Testing:
│   ├─ Likelihood Ratio Test (LRT):
│   │   ├─ LR = 2[ℓ(θ̂_unrestricted) - ℓ(θ̂_restricted)]
│   │   ├─ Under H₀: LR ~ χ²(q) where q = # restrictions
│   │   ├─ Most powerful test asymptotically
│   │   └─ Requires nested models
│   ├─ Wald Test:
│   │   ├─ W = (Rθ̂ - r)'[R·Î⁻¹·R']⁻¹(Rθ̂ - r)
│   │   ├─ Under H₀: W ~ χ²(q)
│   │   ├─ Only requires unrestricted model
│   │   └─ Not invariant to reparameterization
│   ├─ Score/LM Test:
│   │   ├─ LM = S(θ̂_restricted)'I(θ̂_restricted)⁻¹S(θ̂_restricted)
│   │   ├─ Under H₀: LM ~ χ²(q)
│   │   ├─ Only requires restricted model
│   │   └─ Useful when unrestricted hard to estimate
│   └─ Relationships:
│       ├─ LR, Wald, LM asymptotically equivalent
│       ├─ Finite sample: Often LR ≤ LM ≤ Wald
│       └─ LR generally best finite-sample properties
├─ Model Selection:
│   ├─ AIC (Akaike Information Criterion):
│   │   ├─ AIC = -2ℓ(θ̂) + 2k
│   │   ├─ k: Number of parameters
│   │   └─ Lower is better (trade-off fit vs complexity)
│   ├─ BIC (Bayesian Information Criterion):
│   │   ├─ BIC = -2ℓ(θ̂) + k·log(n)
│   │   ├─ Stronger penalty than AIC
│   │   └─ Consistent for model selection
│   ├─ Vuong Test:
│   │   ├─ Compare non-nested models
│   │   └─ Based on log-likelihood differences
│   └─ Cross-Validation:
│       └─ Out-of-sample predictive performance
├─ Robust Standard Errors:
│   ├─ Sandwich/Huber-White:
│   │   ├─ V_robust = Î⁻¹·B̂·Î⁻¹ where B̂ = ΣS(θ̂;xᵢ)S(θ̂;xᵢ)'
│   │   ├─ Robust to misspecification
│   │   └─ Always use in practice
│   ├─ Clustered:
│   │   └─ Account for within-cluster correlation
│   └─ HAC (Newey-West):
│       └─ Robust to heteroskedasticity and autocorrelation
├─ Quasi-Maximum Likelihood (QML):
│   ├─ Maximize likelihood even if distribution wrong
│   ├─ Consistency: Under correct mean specification
│   ├─ Efficiency: Lost, but still consistent
│   └─ Standard Errors: Use sandwich estimator
├─ Limitations:
│   ├─ Distributional Assumption:
│   │   ├─ MLE requires specifying full distribution
│   │   └─ Misspecification → Inconsistency
│   ├─ Local Maxima:
│   │   └─ Non-concave likelihoods may have multiple peaks
│   ├─ Finite Sample Bias:
│   │   ├─ MLE can be biased (e.g., σ̂² in normal)
│   │   └─ Bias correction or bootstrap
│   ├─ Outliers:
│   │   └─ Not robust to heavy tails, outliers
│   └─ Computational:
│       └─ Complex models may be difficult to optimize
└─ Applications:
    ├─ Discrete Choice: Logit, probit, multinomial logit
    ├─ Duration Analysis: Weibull, Cox proportional hazards
    ├─ Count Data: Poisson, negative binomial
    ├─ Limited Dependent: Tobit, Heckman selection
    ├─ Time Series: ARMA, GARCH (already seen)
    ├─ Panel Data: Random effects, dynamic panels
    └─ Structural Models: Demand estimation, auctions
```

**Interaction:** Specify model/distribution → Write likelihood → Differentiate (score) → Optimize numerically → Compute information matrix → Standard errors/tests

## 5. Mini-Project
Implement MLE for logit, Poisson, and Tobit models with diagnostics:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from scipy.special import ndtr as norm_cdf
import seaborn as sns

np.random.seed(654)

# ===== Helper Functions =====
def sigmoid(z):
    """Logistic function"""
    return 1 / (1 + np.exp(-z))

def logit_loglik(beta, y, X):
    """Log-likelihood for logit model"""
    Xb = X @ beta
    ll = np.sum(y * Xb - np.log(1 + np.exp(Xb)))
    return ll

def logit_score(beta, y, X):
    """Score (gradient) for logit"""
    p = sigmoid(X @ beta)
    score = X.T @ (y - p)
    return score

def logit_hessian(beta, y, X):
    """Hessian for logit"""
    p = sigmoid(X @ beta)
    W = np.diag(p * (1 - p))
    H = -X.T @ W @ X
    return H

def poisson_loglik(beta, y, X):
    """Log-likelihood for Poisson regression"""
    mu = np.exp(X @ beta)
    ll = np.sum(y * np.log(mu) - mu - np.log(stats.gamma(y + 1)))
    return ll

def tobit_loglik(params, y, X):
    """Log-likelihood for Tobit (censored at 0)"""
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    
    Xb = X @ beta
    
    # Censored observations (y=0)
    censored = (y == 0)
    ll_censored = np.sum(np.log(norm_cdf(-Xb[censored] / sigma)))
    
    # Uncensored observations (y>0)
    uncensored = (y > 0)
    ll_uncensored = np.sum(
        -np.log(sigma) - 0.5 * np.log(2*np.pi) 
        - 0.5 * ((y[uncensored] - Xb[uncensored]) / sigma)**2
    )
    
    return ll_censored + ll_uncensored

print("="*80)
print("MAXIMUM LIKELIHOOD ESTIMATION")
print("="*80)

# ===== LOGIT MODEL =====
print("\n" + "="*80)
print("BINARY LOGIT MODEL")
print("="*80)

# Simulate data
n = 1000
X_logit = np.column_stack([
    np.ones(n),
    np.random.randn(n),
    np.random.randn(n),
    np.random.randn(n)
])

beta_true_logit = np.array([0.5, 1.2, -0.8, 0.6])
prob_true = sigmoid(X_logit @ beta_true_logit)
y_logit = np.random.binomial(1, prob_true)

print(f"Sample size: {n}")
print(f"Outcome: Binary (0/1)")
print(f"Covariates: {X_logit.shape[1]} (including intercept)")
print(f"True β: {beta_true_logit}")
print(f"\nOutcome distribution:")
print(f"  Y=0: {np.sum(y_logit==0)} ({np.mean(y_logit==0)*100:.1f}%)")
print(f"  Y=1: {np.sum(y_logit==1)} ({np.mean(y_logit==1)*100:.1f}%)")

# MLE estimation
def neg_logit_loglik(beta, y, X):
    """Negative log-likelihood for minimization"""
    return -logit_loglik(beta, y, X)

# Starting values (zeros)
beta_init = np.zeros(X_logit.shape[1])

# Optimize
result_logit = minimize(
    neg_logit_loglik,
    beta_init,
    args=(y_logit, X_logit),
    method='BFGS',
    jac=lambda beta, y, X: -logit_score(beta, y, X)
)

beta_hat_logit = result_logit.x
loglik_logit = -result_logit.fun

print(f"\n✓ Optimization converged: {result_logit.success}")
print(f"  Iterations: {result_logit.nit}")
print(f"  Log-likelihood: {loglik_logit:.4f}")

# Standard errors via information matrix
H = logit_hessian(beta_hat_logit, y_logit, X_logit)
I_inv = -np.linalg.inv(H)
se_logit = np.sqrt(np.diag(I_inv))

# Results
print(f"\nMLE Estimates:")
for i, (b_true, b_hat, se) in enumerate(zip(beta_true_logit, beta_hat_logit, se_logit)):
    z_stat = b_hat / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"  β{i}: {b_hat:7.4f} (SE={se:.4f}, true={b_true:.2f}) "
          f"[z={z_stat:.2f}, p={p_val:.4f}]")

# Marginal effects (at mean)
X_mean = X_logit.mean(axis=0)
prob_mean = sigmoid(X_mean @ beta_hat_logit)
mfx = beta_hat_logit * prob_mean * (1 - prob_mean)

print(f"\nAverage Marginal Effects:")
for i, m in enumerate(mfx[1:], 1):  # Skip intercept
    print(f"  ∂P/∂x{i}: {m:.4f}")

# Pseudo R²
loglik_null = np.sum(np.log([1-y_logit.mean()] * np.sum(y_logit==0)) + 
                     np.log([y_logit.mean()] * np.sum(y_logit==1)))
pseudo_r2 = 1 - loglik_logit / loglik_null

print(f"\nMcFadden's Pseudo R²: {pseudo_r2:.4f}")

# ===== POISSON REGRESSION =====
print("\n" + "="*80)
print("POISSON REGRESSION")
print("="*80)

# Simulate count data
X_poisson = np.column_stack([
    np.ones(n),
    np.random.randn(n),
    np.random.randn(n)
])

beta_true_poisson = np.array([1.0, 0.5, -0.3])
lambda_true = np.exp(X_poisson @ beta_true_poisson)
y_poisson = np.random.poisson(lambda_true)

print(f"Sample size: {n}")
print(f"Outcome: Count (Poisson)")
print(f"True β: {beta_true_poisson}")
print(f"\nCount distribution:")
print(f"  Mean: {y_poisson.mean():.2f}")
print(f"  Variance: {y_poisson.var():.2f}")
print(f"  Max: {y_poisson.max()}")

# Check overdispersion
print(f"  Variance/Mean ratio: {y_poisson.var()/y_poisson.mean():.2f} "
      f"(should be ≈1 for Poisson)")

# MLE estimation
def neg_poisson_loglik(beta, y, X):
    return -poisson_loglik(beta, y, X)

beta_init_poisson = np.zeros(X_poisson.shape[1])

result_poisson = minimize(
    neg_poisson_loglik,
    beta_init_poisson,
    args=(y_poisson, X_poisson),
    method='BFGS'
)

beta_hat_poisson = result_poisson.x
loglik_poisson = -result_poisson.fun

print(f"\n✓ Optimization converged: {result_poisson.success}")
print(f"  Log-likelihood: {loglik_poisson:.4f}")

# Numerical Hessian for standard errors
from scipy.optimize import approx_fprime

def score_poisson(beta):
    mu = np.exp(X_poisson @ beta)
    return X_poisson.T @ (y_poisson - mu)

H_poisson = np.zeros((len(beta_hat_poisson), len(beta_hat_poisson)))
eps = 1e-5
for i in range(len(beta_hat_poisson)):
    def score_i(beta):
        return score_poisson(beta)[i]
    H_poisson[i, :] = approx_fprime(beta_hat_poisson, score_i, eps)

I_inv_poisson = -np.linalg.inv(H_poisson)
se_poisson = np.sqrt(np.diag(I_inv_poisson))

print(f"\nMLE Estimates:")
for i, (b_true, b_hat, se) in enumerate(zip(beta_true_poisson, beta_hat_poisson, se_poisson)):
    z_stat = b_hat / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"  β{i}: {b_hat:7.4f} (SE={se:.4f}, true={b_true:.2f}) "
          f"[z={z_stat:.2f}, p={p_val:.4f}]")

# Incidence rate ratios (IRR)
irr = np.exp(beta_hat_poisson)
print(f"\nIncidence Rate Ratios (exp(β)):")
for i, (b_hat, irr_val) in enumerate(zip(beta_hat_poisson[1:], irr[1:]), 1):
    pct_change = (irr_val - 1) * 100
    print(f"  x{i}: IRR={irr_val:.4f} ({pct_change:+.1f}% change in rate)")

# ===== TOBIT MODEL (CENSORED) =====
print("\n" + "="*80)
print("TOBIT MODEL (Censored Regression)")
print("="*80)

# Simulate censored data
X_tobit = np.column_stack([
    np.ones(n),
    np.random.randn(n),
    np.random.randn(n)
])

beta_true_tobit = np.array([2.0, 1.5, -1.0])
sigma_true = 2.0

# Latent variable
y_star = X_tobit @ beta_true_tobit + np.random.normal(0, sigma_true, n)

# Observed (censored at 0)
y_tobit = np.maximum(0, y_star)

n_censored = np.sum(y_tobit == 0)
pct_censored = n_censored / n * 100

print(f"Sample size: {n}")
print(f"Outcome: Continuous, censored at 0")
print(f"True β: {beta_true_tobit}")
print(f"True σ: {sigma_true:.2f}")
print(f"\nCensoring:")
print(f"  Censored (y=0): {n_censored} ({pct_censored:.1f}%)")
print(f"  Uncensored (y>0): {n - n_censored} ({100-pct_censored:.1f}%)")
print(f"  Mean (all): {y_tobit.mean():.2f}")
print(f"  Mean (uncensored): {y_tobit[y_tobit>0].mean():.2f}")

# MLE estimation
def neg_tobit_loglik(params, y, X):
    return -tobit_loglik(params, y, X)

params_init = np.append(np.zeros(X_tobit.shape[1]), 0)  # Last is log(sigma)

result_tobit = minimize(
    neg_tobit_loglik,
    params_init,
    args=(y_tobit, X_tobit),
    method='BFGS'
)

params_hat_tobit = result_tobit.x
beta_hat_tobit = params_hat_tobit[:-1]
sigma_hat_tobit = np.exp(params_hat_tobit[-1])
loglik_tobit = -result_tobit.fun

print(f"\n✓ Optimization converged: {result_tobit.success}")
print(f"  Log-likelihood: {loglik_tobit:.4f}")

# Standard errors (numerical Hessian)
H_tobit = result_tobit.hess_inv
se_tobit = np.sqrt(np.diag(H_tobit))
se_tobit[-1] = se_tobit[-1] * sigma_hat_tobit  # Delta method for sigma

print(f"\nMLE Estimates:")
for i, (b_true, b_hat, se) in enumerate(zip(beta_true_tobit, beta_hat_tobit, se_tobit[:-1])):
    z_stat = b_hat / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"  β{i}: {b_hat:7.4f} (SE={se:.4f}, true={b_true:.2f}) "
          f"[z={z_stat:.2f}, p={p_val:.4f}]")

print(f"  σ: {sigma_hat_tobit:7.4f} (SE={se_tobit[-1]:.4f}, true={sigma_true:.2f})")

# Compare to naive OLS (biased)
from numpy.linalg import lstsq
beta_ols = lstsq(X_tobit, y_tobit, rcond=None)[0]

print(f"\nComparison: Tobit MLE vs Naive OLS:")
for i, (b_tobit, b_ols, b_true) in enumerate(zip(beta_hat_tobit, beta_ols, beta_true_tobit)):
    print(f"  β{i}: Tobit={b_tobit:.4f}, OLS={b_ols:.4f}, True={b_true:.2f}")
    if abs(b_tobit - b_true) < abs(b_ols - b_true):
        print(f"       ✓ Tobit closer to truth")

# ===== LIKELIHOOD RATIO TEST =====
print("\n" + "="*80)
print("LIKELIHOOD RATIO TEST")
print("="*80)

# Test H0: β2 = β3 = 0 in logit model
# Restricted model
X_logit_restricted = X_logit[:, :2]

result_restricted = minimize(
    neg_logit_loglik,
    np.zeros(2),
    args=(y_logit, X_logit_restricted),
    method='BFGS',
    jac=lambda beta, y, X: -logit_score(beta, y, X)
)

loglik_restricted = -result_restricted.fun

# LR statistic
LR = 2 * (loglik_logit - loglik_restricted)
df = X_logit.shape[1] - X_logit_restricted.shape[1]
p_value_lr = 1 - stats.chi2.cdf(LR, df)

print(f"Testing H₀: β₂ = β₃ = 0 (Logit model)")
print(f"  Unrestricted log-likelihood: {loglik_logit:.4f}")
print(f"  Restricted log-likelihood: {loglik_restricted:.4f}")
print(f"  LR statistic: {LR:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  P-value: {p_value_lr:.4f}")

if p_value_lr < 0.05:
    print(f"  ✓ Reject H₀ at 5% level (variables jointly significant)")
else:
    print(f"  ✗ Cannot reject H₀")

# ===== WALD TEST =====
print("\n" + "="*80)
print("WALD TEST")
print("="*80)

# Test same hypothesis: β2 = β3 = 0
R = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1]])
r = np.zeros(2)

Rbeta = R @ beta_hat_logit - r
RVR = R @ I_inv @ R.T

W = Rbeta @ np.linalg.inv(RVR) @ Rbeta
p_value_wald = 1 - stats.chi2.cdf(W, 2)

print(f"Testing H₀: β₂ = β₃ = 0 (Wald test)")
print(f"  Wald statistic: {W:.4f}")
print(f"  P-value: {p_value_wald:.4f}")

if p_value_wald < 0.05:
    print(f"  ✓ Reject H₀")

print(f"\nLR vs Wald:")
print(f"  Both reject H₀, asymptotically equivalent")
print(f"  Finite sample: LR={LR:.4f}, Wald={W:.4f}")

# ===== VISUALIZATIONS =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Logit - Predicted Probabilities
prob_hat = sigmoid(X_logit @ beta_hat_logit)
axes[0, 0].hist(prob_hat[y_logit==0], bins=30, alpha=0.6, label='Y=0', density=True)
axes[0, 0].hist(prob_hat[y_logit==1], bins=30, alpha=0.6, label='Y=1', density=True)
axes[0, 0].set_xlabel('Predicted Probability')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Logit: Predicted Probabilities')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Logit - ROC-style
sorted_idx = np.argsort(prob_hat)
axes[0, 1].plot(np.arange(n), y_logit[sorted_idx], 'o', markersize=2, alpha=0.5, label='True')
axes[0, 1].plot(np.arange(n), prob_hat[sorted_idx], linewidth=2, label='Predicted')
axes[0, 1].set_xlabel('Observation (sorted by predicted prob)')
axes[0, 1].set_ylabel('Outcome / Probability')
axes[0, 1].set_title('Logit: Predicted vs Actual (sorted)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Poisson - Predicted vs Actual Counts
lambda_hat = np.exp(X_poisson @ beta_hat_poisson)
count_bins = np.arange(0, max(y_poisson.max(), lambda_hat.max()) + 1)

axes[0, 2].hist(y_poisson, bins=count_bins, alpha=0.6, label='Observed', density=True)
axes[0, 2].hist(lambda_hat, bins=count_bins, alpha=0.6, label='Predicted', density=True)
axes[0, 2].set_xlabel('Count')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_title('Poisson: Observed vs Predicted')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Poisson - Scatter
axes[1, 0].scatter(lambda_hat, y_poisson, alpha=0.3, s=10)
axes[1, 0].plot([lambda_hat.min(), lambda_hat.max()],
               [lambda_hat.min(), lambda_hat.max()],
               'r--', linewidth=2)
axes[1, 0].set_xlabel('Predicted λ')
axes[1, 0].set_ylabel('Observed Count')
axes[1, 0].set_title('Poisson: Predicted vs Observed')
axes[1, 0].grid(alpha=0.3)

# Plot 5: Tobit - Censoring Effect
y_hat_tobit = X_tobit @ beta_hat_tobit

axes[1, 1].scatter(y_hat_tobit[y_tobit==0], y_tobit[y_tobit==0],
                  alpha=0.5, s=20, label='Censored', color='red')
axes[1, 1].scatter(y_hat_tobit[y_tobit>0], y_tobit[y_tobit>0],
                  alpha=0.3, s=10, label='Uncensored', color='blue')
axes[1, 1].plot([y_hat_tobit.min(), y_hat_tobit.max()],
               [y_hat_tobit.min(), y_hat_tobit.max()],
               'k--', linewidth=2)
axes[1, 1].axhline(0, color='gray', linestyle=':', linewidth=1)
axes[1, 1].set_xlabel('Predicted Y*')
axes[1, 1].set_ylabel('Observed Y')
axes[1, 1].set_title('Tobit: Censoring at 0')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Coefficient Comparison (Tobit vs OLS)
coef_names = ['β0', 'β1', 'β2']
x_pos = np.arange(len(coef_names))
width = 0.25

axes[1, 2].bar(x_pos - width, beta_true_tobit, width, label='True', alpha=0.8)
axes[1, 2].bar(x_pos, beta_hat_tobit, width, label='Tobit MLE', alpha=0.8)
axes[1, 2].bar(x_pos + width, beta_ols, width, label='OLS (biased)', alpha=0.8)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(coef_names)
axes[1, 2].set_ylabel('Coefficient Value')
axes[1, 2].set_title('Tobit: MLE vs OLS')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mle_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== SUMMARY =====
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n1. Logit Model:")
print(f"   Pseudo R²: {pseudo_r2:.4f}")
print(f"   All coefficients recovered successfully")
print(f"   Marginal effects interpretable")

print("\n2. Poisson Regression:")
print(f"   Mean count: {y_poisson.mean():.2f}")
print(f"   IRRs show % change in rate")
if y_poisson.var() / y_poisson.mean() > 1.5:
    print(f"   ⚠ Overdispersion detected: Consider negative binomial")

print("\n3. Tobit Model:")
print(f"   Censoring: {pct_censored:.1f}%")
print(f"   MLE accounts for censoring mechanism")
print(f"   OLS biased (ignores censoring)")

print("\n4. Hypothesis Testing:")
print(f"   LR test: {LR:.2f}, p={p_value_lr:.4f}")
print(f"   Wald test: {W:.2f}, p={p_value_wald:.4f}")
print(f"   Both tests agree (asymptotic equivalence)")

print("\n5. Key Takeaways:")
print("   • MLE provides efficient estimates under correct specification")
print("   • Fisher information → Standard errors automatically")
print("   • LR, Wald, Score tests all available")
print("   • Numerical optimization required for most models")
print("   • Check convergence, try multiple starting values")
print("   • Robust SEs recommended in practice")
```

## 6. Challenge Round
When does MLE fail or mislead?
- **Misspecification**: Wrong distribution assumed → Inconsistent estimates; QML with sandwich SEs mitigates; robust methods better
- **Perfect separation**: Logit/probit with perfect predictor → MLE diverges (β→∞); Firth's penalized likelihood or drop variable
- **Weak identification**: Multiple parameter combinations yield similar likelihood → Flat likelihood surface; prior information or constraints needed
- **Finite sample bias**: Small n → MLE biased (e.g., σ² in normal); bootstrap or bias correction
- **Local maxima**: Non-convex likelihood → Algorithm finds local, not global max; multiple starting values critical
- **Outliers**: Heavy tails misspecified → MLE inefficient, SEs invalid; robust M-estimators or correct distribution (Student-t)

## 7. Key References
- [Wooldridge - Econometric Analysis of Cross Section and Panel Data (Ch 12-13)](https://mitpress.mit.edu/9780262232586/econometric-analysis-of-cross-section-and-panel-data/)
- [Greene - Econometric Analysis (Ch 14-17)](https://www.pearson.com/us/higher-education/program/Greene-Econometric-Analysis-8th-Edition/PGM334862.html)
- [Cameron & Trivedi - Microeconometrics: Methods and Applications (Ch 5)](http://cameron.econ.ucdavis.edu/mmabook/mma.html)

---
**Status:** Foundation of parametric econometrics | **Complements:** GMM, Quasi-ML, Bayesian Methods, M-Estimation
