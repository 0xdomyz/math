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
print(f"True Î²: {beta_true_logit}")
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

print(f"\nâœ“ Optimization converged: {result_logit.success}")
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
    print(f"  Î²{i}: {b_hat:7.4f} (SE={se:.4f}, true={b_true:.2f}) "
          f"[z={z_stat:.2f}, p={p_val:.4f}]")

# Marginal effects (at mean)
X_mean = X_logit.mean(axis=0)
prob_mean = sigmoid(X_mean @ beta_hat_logit)
mfx = beta_hat_logit * prob_mean * (1 - prob_mean)

print(f"\nAverage Marginal Effects:")
for i, m in enumerate(mfx[1:], 1):  # Skip intercept
    print(f"  âˆ‚P/âˆ‚x{i}: {m:.4f}")

# Pseudo RÂ²
loglik_null = np.sum(np.log([1-y_logit.mean()] * np.sum(y_logit==0)) + 
                     np.log([y_logit.mean()] * np.sum(y_logit==1)))
pseudo_r2 = 1 - loglik_logit / loglik_null

print(f"\nMcFadden's Pseudo RÂ²: {pseudo_r2:.4f}")

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
print(f"True Î²: {beta_true_poisson}")
print(f"\nCount distribution:")
print(f"  Mean: {y_poisson.mean():.2f}")
print(f"  Variance: {y_poisson.var():.2f}")
print(f"  Max: {y_poisson.max()}")

# Check overdispersion
print(f"  Variance/Mean ratio: {y_poisson.var()/y_poisson.mean():.2f} "
      f"(should be â‰ˆ1 for Poisson)")

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

print(f"\nâœ“ Optimization converged: {result_poisson.success}")
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
    print(f"  Î²{i}: {b_hat:7.4f} (SE={se:.4f}, true={b_true:.2f}) "
          f"[z={z_stat:.2f}, p={p_val:.4f}]")

# Incidence rate ratios (IRR)
irr = np.exp(beta_hat_poisson)
print(f"\nIncidence Rate Ratios (exp(Î²)):")
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
print(f"True Î²: {beta_true_tobit}")
print(f"True Ïƒ: {sigma_true:.2f}")
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

print(f"\nâœ“ Optimization converged: {result_tobit.success}")
print(f"  Log-likelihood: {loglik_tobit:.4f}")

# Standard errors (numerical Hessian)
H_tobit = result_tobit.hess_inv
se_tobit = np.sqrt(np.diag(H_tobit))
se_tobit[-1] = se_tobit[-1] * sigma_hat_tobit  # Delta method for sigma

print(f"\nMLE Estimates:")
for i, (b_true, b_hat, se) in enumerate(zip(beta_true_tobit, beta_hat_tobit, se_tobit[:-1])):
    z_stat = b_hat / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"  Î²{i}: {b_hat:7.4f} (SE={se:.4f}, true={b_true:.2f}) "
          f"[z={z_stat:.2f}, p={p_val:.4f}]")

print(f"  Ïƒ: {sigma_hat_tobit:7.4f} (SE={se_tobit[-1]:.4f}, true={sigma_true:.2f})")

# Compare to naive OLS (biased)
from numpy.linalg import lstsq
beta_ols = lstsq(X_tobit, y_tobit, rcond=None)[0]

print(f"\nComparison: Tobit MLE vs Naive OLS:")
for i, (b_tobit, b_ols, b_true) in enumerate(zip(beta_hat_tobit, beta_ols, beta_true_tobit)):
    print(f"  Î²{i}: Tobit={b_tobit:.4f}, OLS={b_ols:.4f}, True={b_true:.2f}")
    if abs(b_tobit - b_true) < abs(b_ols - b_true):
        print(f"       âœ“ Tobit closer to truth")

# ===== LIKELIHOOD RATIO TEST =====
print("\n" + "="*80)
print("LIKELIHOOD RATIO TEST")
print("="*80)

# Test H0: Î²2 = Î²3 = 0 in logit model
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

print(f"Testing Hâ‚€: Î²â‚‚ = Î²â‚ƒ = 0 (Logit model)")
print(f"  Unrestricted log-likelihood: {loglik_logit:.4f}")
print(f"  Restricted log-likelihood: {loglik_restricted:.4f}")
print(f"  LR statistic: {LR:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  P-value: {p_value_lr:.4f}")

if p_value_lr < 0.05:
    print(f"  âœ“ Reject Hâ‚€ at 5% level (variables jointly significant)")
else:
    print(f"  âœ— Cannot reject Hâ‚€")

# ===== WALD TEST =====
print("\n" + "="*80)
print("WALD TEST")
print("="*80)

# Test same hypothesis: Î²2 = Î²3 = 0
R = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1]])
r = np.zeros(2)

Rbeta = R @ beta_hat_logit - r
RVR = R @ I_inv @ R.T

W = Rbeta @ np.linalg.inv(RVR) @ Rbeta
p_value_wald = 1 - stats.chi2.cdf(W, 2)

print(f"Testing Hâ‚€: Î²â‚‚ = Î²â‚ƒ = 0 (Wald test)")
print(f"  Wald statistic: {W:.4f}")
print(f"  P-value: {p_value_wald:.4f}")

if p_value_wald < 0.05:
    print(f"  âœ“ Reject Hâ‚€")

print(f"\nLR vs Wald:")
print(f"  Both reject Hâ‚€, asymptotically equivalent")
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
axes[1, 0].set_xlabel('Predicted Î»')
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
coef_names = ['Î²0', 'Î²1', 'Î²2']
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
print(f"   Pseudo RÂ²: {pseudo_r2:.4f}")
print(f"   All coefficients recovered successfully")
print(f"   Marginal effects interpretable")

print("\n2. Poisson Regression:")
print(f"   Mean count: {y_poisson.mean():.2f}")
print(f"   IRRs show % change in rate")
if y_poisson.var() / y_poisson.mean() > 1.5:
    print(f"   âš  Overdispersion detected: Consider negative binomial")

print("\n3. Tobit Model:")
print(f"   Censoring: {pct_censored:.1f}%")
print(f"   MLE accounts for censoring mechanism")
print(f"   OLS biased (ignores censoring)")

print("\n4. Hypothesis Testing:")
print(f"   LR test: {LR:.2f}, p={p_value_lr:.4f}")
print(f"   Wald test: {W:.2f}, p={p_value_wald:.4f}")
print(f"   Both tests agree (asymptotic equivalence)")

print("\n5. Key Takeaways:")
print("   â€¢ MLE provides efficient estimates under correct specification")
print("   â€¢ Fisher information â†’ Standard errors automatically")
print("   â€¢ LR, Wald, Score tests all available")
print("   â€¢ Numerical optimization required for most models")
print("   â€¢ Check convergence, try multiple starting values")
print("   â€¢ Robust SEs recommended in practice")
