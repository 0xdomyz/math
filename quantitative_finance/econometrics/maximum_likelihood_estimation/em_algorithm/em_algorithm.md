# Expectation-Maximization (EM) Algorithm

## 1. Concept Skeleton
**Definition:** Iterative method for maximum likelihood with latent variables; alternates E-step (compute expected log-likelihood) and M-step (maximize); monotonically increases likelihood  
**Purpose:** Handle missing data, mixture models, hidden Markov models; simplify complex likelihood; guaranteed convergence to local maximum  
**Prerequisites:** Maximum likelihood, latent variables, Jensen's inequality, conditional expectation, complete-data likelihood

## 2. Comparative Framing
| Method | EM Algorithm | Direct MLE | Gibbs Sampling | Variational Bayes | Gradient Ascent | Data Augmentation |
|--------|--------------|------------|----------------|-------------------|-----------------|-------------------|
| **Latent Variables** | Yes (E-step integrates) | No (marginalized) | Yes (sampled) | Yes (approximated) | No | Yes (sampled) |
| **Convergence** | Monotonic (guaranteed) | Depends on optimizer | Asymptotic | Lower bound | Depends on step size | Asymptotic |
| **Speed** | Moderate | Fast (if tractable) | Slow | Fast | Fast (with good gradient) | Moderate |
| **Global Optimum** | No (local) | No | Explores modes | No | No | Explores modes |
| **Closed Form** | Often M-step | Rarely | N/A | Sometimes | N/A | Conditional |
| **Missing Data** | Natural framework | Requires integration | Imputes | Imputes | Difficult | Imputes |

## 3. Examples + Counterexamples

**Classic Example:**  
Gaussian mixture model (K=2 components): EM separates clusters with soft assignments. E-step computes responsibilities γᵢₖ=P(cluster k|xᵢ). M-step updates μₖ, Σₖ, πₖ. Converges in 20-50 iterations with log-likelihood increase ~0.1% per iteration.

**Failure Case:**  
Identifiability issue: Mixture labels arbitrary (label switching). Multiple local maxima depending on initialization. Solution: Run multiple random starts, select highest likelihood.

**Edge Case:**  
Censored regression (Tobit): Latent Y*ᵢ unobserved when Yᵢ=0. E-step computes E[Y*ᵢ|Yᵢ=0, θ]. M-step OLS on completed data. Equivalent to direct MLE but conceptually clearer.

## 4. Layer Breakdown
```
EM Algorithm Framework:
├─ Problem Setup:
│   ├─ Observed Data: Y = {y₁, ..., yₙ}
│   ├─ Latent Variables: Z = {z₁, ..., zₙ} (unobserved)
│   ├─ Complete Data: X = (Y, Z)
│   ├─ Complete-Data Likelihood: L_c(θ|X) = p(Y, Z|θ)
│   ├─ Observed-Data Likelihood: L(θ|Y) = ∫ p(Y, Z|θ) dZ
│   └─ Goal: Maximize L(θ|Y) when direct optimization intractable
├─ E-Step (Expectation):
│   ├─ Compute Q-function:
│   │   ├─ Q(θ|θ⁽ᵗ⁾) = E_Z|Y,θ⁽ᵗ⁾[log L_c(θ|Y,Z)]
│   │   ├─ Expectation over posterior p(Z|Y, θ⁽ᵗ⁾)
│   │   └─ Use current parameter θ⁽ᵗ⁾
│   ├─ Sufficient Statistics:
│   │   ├─ Compute E[T(Z)|Y, θ⁽ᵗ⁾] where T(Z) are sufficient statistics
│   │   └─ Often simpler than full conditional distribution
│   ├─ Responsibilities (Mixture Models):
│   │   ├─ γᵢₖ = P(Zᵢ=k|Yᵢ, θ⁽ᵗ⁾)
│   │   ├─ Bayes rule: γᵢₖ = πₖ·f(yᵢ|θₖ) / Σⱼ πⱼ·f(yᵢ|θⱼ)
│   │   └─ Soft cluster assignments
│   ├─ Conditional Expectations:
│   │   ├─ Censored data: E[Yᵢ*|Yᵢ censored, θ⁽ᵗ⁾]
│   │   ├─ Missing data: E[Yᵢ_missing|Yᵢ_observed, θ⁽ᵗ⁾]
│   │   └─ Latent factors: E[fᵢ|Yᵢ, θ⁽ᵗ⁾]
│   └─ Computational Methods:
│       ├─ Analytical (Gaussian models)
│       ├─ Monte Carlo EM (intractable E-step)
│       └─ Stochastic EM (sample Z instead of integrate)
├─ M-Step (Maximization):
│   ├─ Definition:
│   │   ├─ θ⁽ᵗ⁺¹⁾ = argmax_θ Q(θ|θ⁽ᵗ⁾)
│   │   ├─ Maximize expected complete-data log-likelihood
│   │   └─ Often has closed-form solution
│   ├─ Parameter Updates:
│   │   ├─ Use sufficient statistics from E-step
│   │   ├─ Often weighted MLE (weights = responsibilities)
│   │   └─ Example (mixture): μₖ = Σᵢ γᵢₖ·yᵢ / Σᵢ γᵢₖ
│   ├─ Constraints:
│   │   ├─ Mixing proportions: Σₖ πₖ = 1
│   │   ├─ Covariance positive definite
│   │   └─ Lagrange multipliers or constrained optimization
│   └─ Generalized M-step (GEM):
│       ├─ Only require Q(θ⁽ᵗ⁺¹⁾) ≥ Q(θ⁽ᵗ⁾)
│       └─ Useful when maximization intractable
├─ Convergence Properties:
│   ├─ Monotonic Increase:
│   │   ├─ L(θ⁽ᵗ⁺¹⁾|Y) ≥ L(θ⁽ᵗ⁾|Y) (guaranteed)
│   │   ├─ Proof via Jensen's inequality
│   │   └─ Likelihood never decreases
│   ├─ Convergence to Local Maximum:
│   │   ├─ lim_{t→∞} θ⁽ᵗ⁾ = θ* where ∇L(θ*) = 0
│   │   ├─ Not necessarily global maximum
│   │   └─ Depends on initialization
│   ├─ Convergence Rate:
│   │   ├─ Linear convergence: ||θ⁽ᵗ⁺¹⁾ - θ*|| ≈ λ·||θ⁽ᵗ⁾ - θ*||
│   │   ├─ λ = fraction of missing information
│   │   └─ Slower with more missing data
│   ├─ Stopping Criteria:
│   │   ├─ Parameter change: ||θ⁽ᵗ⁺¹⁾ - θ⁽ᵗ⁾|| < ε
│   │   ├─ Likelihood change: |L⁽ᵗ⁺¹⁾ - L⁽ᵗ⁾| < ε
│   │   ├─ Relative change: |L⁽ᵗ⁺¹⁾ - L⁽ᵗ⁾|/|L⁽ᵗ⁾| < ε
│   │   └─ Maximum iterations: t > t_max
│   └─ Aitken Acceleration:
│       ├─ Estimate θ* from sequence {θ⁽ᵗ⁾}
│       └─ θ̃ = θ⁽ᵗ⁾ + Δθ⁽ᵗ⁾/(1 - λ̂)
├─ Theoretical Justification:
│   ├─ Jensen's Inequality:
│   │   ├─ log L(θ|Y) = log ∫ p(Y,Z|θ) dZ
│   │   ├─ = log ∫ [p(Y,Z|θ)/q(Z)]·q(Z) dZ
│   │   ├─ ≥ ∫ q(Z)·log[p(Y,Z|θ)/q(Z)] dZ (Jensen)
│   │   └─ = ELBO (evidence lower bound)
│   ├─ Variational Lower Bound:
│   │   ├─ log L(θ) ≥ E_q[log p(Y,Z|θ)] + H(q)
│   │   ├─ E-step: Set q(Z) = p(Z|Y,θ⁽ᵗ⁾) (tighten bound)
│   │   └─ M-step: Maximize bound w.r.t. θ
│   ├─ KL Divergence Decomposition:
│   │   ├─ log L(θ) = ELBO + KL(q||p(Z|Y,θ))
│   │   ├─ E-step sets KL = 0
│   │   └─ M-step increases ELBO
│   └─ Why Monotonic:
│       ├─ L(θ⁽ᵗ⁺¹⁾) ≥ ELBO(θ⁽ᵗ⁺¹⁾, q⁽ᵗ⁾) (definition)
│       ├─ ≥ ELBO(θ⁽ᵗ⁾, q⁽ᵗ⁾) (M-step increases)
│       └─ = L(θ⁽ᵗ⁾) (E-step tightens)
├─ Gaussian Mixture Model (GMM):
│   ├─ Model:
│   │   ├─ p(yᵢ) = Σₖ πₖ·𝒩(yᵢ|μₖ, Σₖ)
│   │   ├─ πₖ: Mixing proportions (Σₖ πₖ = 1)
│   │   ├─ K components
│   │   └─ Latent: Zᵢ ∈ {1,...,K} cluster membership
│   ├─ E-Step:
│   │   ├─ Responsibilities: γᵢₖ = πₖ·𝒩(yᵢ|μₖ,Σₖ) / Σⱼ πⱼ·𝒩(yᵢ|μⱼ,Σⱼ)
│   │   ├─ P(Zᵢ=k|yᵢ, θ⁽ᵗ⁾)
│   │   └─ Soft assignments (sum to 1)
│   ├─ M-Step:
│   │   ├─ nₖ = Σᵢ γᵢₖ (effective sample size)
│   │   ├─ πₖ = nₖ / n
│   │   ├─ μₖ = Σᵢ γᵢₖ·yᵢ / nₖ (weighted mean)
│   │   └─ Σₖ = Σᵢ γᵢₖ·(yᵢ-μₖ)(yᵢ-μₖ)' / nₖ (weighted cov)
│   ├─ Initialization:
│   │   ├─ K-means clustering
│   │   ├─ Random assignment
│   │   └─ Multiple random starts
│   └─ Identifiability:
│       ├─ Label switching (permutation invariance)
│       └─ Post-hoc label alignment
├─ Missing Data:
│   ├─ Missing at Random (MAR):
│   │   ├─ P(missing|Y_obs, Y_miss) = P(missing|Y_obs)
│   │   ├─ EM valid under MAR
│   │   └─ Ignorable missingness mechanism
│   ├─ Missing Completely at Random (MCAR):
│   │   ├─ P(missing) constant
│   │   └─ Stronger assumption
│   ├─ Not Missing at Random (NMAR):
│   │   ├─ P(missing|Y_obs, Y_miss) depends on Y_miss
│   │   ├─ EM biased
│   │   └─ Need selection model
│   ├─ E-Step:
│   │   ├─ Impute missing values: E[Y_miss|Y_obs, θ⁽ᵗ⁾]
│   │   ├─ Predict from observed data
│   │   └─ Account for uncertainty
│   └─ M-Step:
│       ├─ MLE using observed + imputed data
│       └─ Standard complete-data estimators
├─ Censored/Truncated Data:
│   ├─ Tobit Model (Censoring):
│   │   ├─ Latent: Yᵢ* = Xᵢβ + εᵢ
│   │   ├─ Observed: Yᵢ = max(0, Yᵢ*)
│   │   ├─ E-step: E[Yᵢ*|Yᵢ=0, θ⁽ᵗ⁾] = Xᵢβ - σ·λ where λ=φ/Φ (IMR)
│   │   └─ M-step: OLS on completed data
│   ├─ Truncation:
│   │   ├─ Observations only if Y > c
│   │   └─ Conditional distribution p(Y|Y>c, θ)
│   └─ Interval Censoring:
│       ├─ Y ∈ [L, U]
│       └─ E-step: E[Y|L<Y<U, θ⁽ᵗ⁾]
├─ Hidden Markov Models (HMM):
│   ├─ Model:
│   │   ├─ States: Sₜ ∈ {1,...,K} (latent Markov chain)
│   │   ├─ Observations: Yₜ|Sₜ ~ f(·|θ_Sₜ)
│   │   ├─ Transition: P(Sₜ=j|Sₜ₋₁=i) = Aᵢⱼ
│   │   └─ Emission: P(Yₜ|Sₜ=k) = fₖ(yₜ)
│   ├─ Forward-Backward Algorithm (E-Step):
│   │   ├─ Forward: αₜ(k) = P(Y₁:ₜ, Sₜ=k)
│   │   ├─ Backward: βₜ(k) = P(Yₜ₊₁:ₜ|Sₜ=k)
│   │   ├─ Smoothing: γₜ(k) = P(Sₜ=k|Y₁:ₜ) ∝ αₜ(k)·βₜ(k)
│   │   └─ Pairwise: ξₜ(i,j) = P(Sₜ=i, Sₜ₊₁=j|Y₁:ₜ)
│   ├─ M-Step:
│   │   ├─ Initial: π₀(k) = γ₁(k)
│   │   ├─ Transition: Aᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)
│   │   └─ Emission: Update θₖ using {yₜ} weighted by γₜ(k)
│   └─ Applications:
│       ├─ Speech recognition
│       ├─ Regime-switching models (finance)
│       └─ Biological sequences
├─ Factor Analysis:
│   ├─ Model:
│   │   ├─ Yᵢ = Λ·fᵢ + εᵢ
│   │   ├─ fᵢ ~ 𝒩(0, I) latent factors
│   │   ├─ εᵢ ~ 𝒩(0, Ψ) unique variances (diagonal)
│   │   └─ Yᵢ ~ 𝒩(0, ΛΛ' + Ψ)
│   ├─ E-Step:
│   │   ├─ E[fᵢ|Yᵢ, θ⁽ᵗ⁾] = (Λ'Ψ⁻¹Λ + I)⁻¹Λ'Ψ⁻¹Yᵢ
│   │   └─ E[fᵢfᵢ'|Yᵢ, θ⁽ᵗ⁾] = Var(fᵢ|Yᵢ) + E[fᵢ|Yᵢ]E[fᵢ|Yᵢ]'
│   ├─ M-Step:
│   │   ├─ Λ = [Σᵢ Yᵢ E[fᵢ]'][Σᵢ E[fᵢfᵢ']]⁻¹
│   │   └─ Ψ = diag{(1/n)Σᵢ YᵢYᵢ' - Λ E[fᵢYᵢ']}
│   └─ Rotation Indeterminacy:
│       └─ Post-hoc rotation (varimax, etc.)
├─ Variants & Extensions:
│   ├─ Stochastic EM (SEM):
│   │   ├─ E-step: Sample Z ~ p(Z|Y, θ⁽ᵗ⁾) instead of integrate
│   │   ├─ M-step: Maximize using sampled Z
│   │   └─ Better exploration of parameter space
│   ├─ Monte Carlo EM (MCEM):
│   │   ├─ E-step intractable: Use MC integration
│   │   ├─ Q̂(θ) = (1/M)Σₘ log p(Y, Z⁽ᵐ⁾|θ)
│   │   └─ Increase M as iterations progress
│   ├─ Expectation-Conditional Maximization (ECM):
│   │   ├─ M-step in blocks (easier optimization)
│   │   ├─ CM-step 1: Maximize θ₁ given θ₂⁽ᵗ⁾
│   │   ├─ CM-step 2: Maximize θ₂ given θ₁⁽ᵗ⁺¹⁾
│   │   └─ Still monotonic
│   ├─ Expectation-Conditional Maximization Either (ECME):
│   │   ├─ Some CM-steps maximize observed-data likelihood
│   │   └─ Faster convergence
│   ├─ Generalized EM (GEM):
│   │   ├─ M-step only improves: Q(θ⁽ᵗ⁺¹⁾) ≥ Q(θ⁽ᵗ⁾)
│   │   └─ Useful for constrained optimization
│   ├─ Incremental EM:
│   │   ├─ Online learning (streaming data)
│   │   └─ Update after each observation
│   └─ Variational EM:
│       ├─ Approximate posterior q(Z) (not exact)
│       └─ Variational Bayes inference
├─ Standard Errors & Inference:
│   ├─ Observed Information:
│   │   ├─ I_obs(θ̂) = -∂²log L(θ|Y)/∂θ∂θ'|_{θ̂}
│   │   ├─ Numerical Hessian at convergence
│   │   └─ SE(θ̂) = √diag(I_obs⁻¹)
│   ├─ Louis's Formula:
│   │   ├─ I_obs = I_complete - Var[S_complete|Y]
│   │   ├─ Use E-step calculations
│   │   └─ Computationally efficient
│   ├─ Bootstrap:
│   │   ├─ Resample data, re-run EM
│   │   └─ SE from bootstrap distribution
│   ├─ Parametric Bootstrap:
│   │   ├─ Simulate data from p(·|θ̂)
│   │   └─ Account for missing data structure
│   └─ Supplemented EM (SEM):
│       ├─ Simultaneously estimate θ and I_obs
│       └─ One-step Newton-Raphson after EM
├─ Model Selection:
│   ├─ Number of Components (K):
│   │   ├─ BIC: log L - (k/2)log(n) (prefer lower)
│   │   ├─ AIC: log L - k
│   │   └─ Integrated classification likelihood (ICL)
│   ├─ Cross-Validation:
│   │   ├─ K-fold CV on held-out data
│   │   └─ Avoid overfitting
│   ├─ Silhouette Score:
│   │   └─ Cluster quality measure
│   └─ Likelihood Ratio Test:
│       ├─ LR = 2[log L(K) - log L(K-1)]
│       └─ Not standard χ² (boundary issue)
├─ Computational Considerations:
│   ├─ Initialization Sensitivity:
│   │   ├─ Multiple random starts (10-100)
│   │   ├─ K-means for mixture models
│   │   └─ Select highest likelihood
│   ├─ Convergence Diagnostics:
│   │   ├─ Plot log-likelihood vs iteration
│   │   ├─ Check parameter stability
│   │   └─ Monitor Q-function increase
│   ├─ Numerical Stability:
│   │   ├─ Log-sum-exp trick for probabilities
│   │   ├─ Regularization (add small ε to covariance)
│   │   └─ Avoid underflow in responsibilities
│   └─ Computational Complexity:
│       ├─ GMM: O(nKd²) per iteration
│       ├─ HMM: O(TK²) (forward-backward)
│       └─ Typically 10-100 iterations
└─ Applications:
    ├─ Mixture Models: Clustering, density estimation
    ├─ Missing Data: Multiple imputation, survey data
    ├─ Hidden Markov Models: Time series, speech, finance
    ├─ Factor Analysis: Psychometrics, dimension reduction
    ├─ Item Response Theory: Educational testing
    ├─ Survival Analysis: Interval censoring
    ├─ Image Segmentation: Computer vision
    ├─ Bioinformatics: Gene expression, motif finding
    └─ Econometrics: Regime switching, censored models
```

**Interaction:** Initialize θ⁽⁰⁾ → E-step: Compute Q(θ|θ⁽ᵗ⁾) → M-step: θ⁽ᵗ⁺¹⁾ = argmax Q → Check convergence → Repeat until converged

## 5. Mini-Project
Implement EM for Gaussian mixture model with missing data:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns

np.random.seed(321)

# ===== Simulate Gaussian Mixture Data =====
print("="*80)
print("EXPECTATION-MAXIMIZATION (EM) ALGORITHM")
print("="*80)

n = 500
K = 3  # Number of clusters

# True parameters
pi_true = np.array([0.3, 0.5, 0.2])
mu_true = np.array([[-2, -2], [0, 3], [3, 0]])
Sigma_true = np.array([
    [[1.0, 0.3], [0.3, 1.0]],
    [[0.8, -0.2], [-0.2, 0.8]],
    [[1.2, 0.5], [0.5, 1.2]]
])

print(f"Simulation Setup:")
print(f"  Sample size: {n}")
print(f"  Number of components: {K}")
print(f"  True mixing proportions: {pi_true}")

# Generate data
cluster_labels = np.random.choice(K, size=n, p=pi_true)
Y_complete = np.zeros((n, 2))

for k in range(K):
    mask = cluster_labels == k
    n_k = np.sum(mask)
    Y_complete[mask] = np.random.multivariate_normal(
        mu_true[k], Sigma_true[k], size=n_k
    )

print(f"  True cluster sizes: {np.bincount(cluster_labels)}")

# Introduce missing data (MCAR)
missing_prob = 0.20
missing_mask = np.random.rand(n, 2) < missing_prob
Y_observed = Y_complete.copy()
Y_observed[missing_mask] = np.nan

n_missing = np.sum(missing_mask)
missing_pct = n_missing / (n * 2) * 100

print(f"\nMissing Data:")
print(f"  Total missing values: {n_missing}/{n*2} ({missing_pct:.1f}%)")
print(f"  Rows with any missing: {np.sum(np.any(missing_mask, axis=1))}/{n}")

# ===== EM Algorithm for GMM with Missing Data =====
print("\n" + "="*80)
print("EM ALGORITHM IMPLEMENTATION")
print("="*80)

def initialize_params(Y, K, method='kmeans'):
    """Initialize parameters"""
    n, d = Y.shape
    
    # Use complete cases for initialization
    complete_cases = ~np.any(np.isnan(Y), axis=1)
    Y_complete_init = Y[complete_cases]
    
    if method == 'kmeans':
        # Simple K-means on complete cases
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Y_complete_init)
        
        pi = np.bincount(labels, minlength=K) / len(labels)
        mu = kmeans.cluster_centers_
        
        # Initialize covariances
        Sigma = np.zeros((K, d, d))
        for k in range(K):
            if np.sum(labels == k) > 1:
                Y_k = Y_complete_init[labels == k]
                Sigma[k] = np.cov(Y_k.T) + np.eye(d) * 0.1
            else:
                Sigma[k] = np.eye(d)
    
    elif method == 'random':
        # Random initialization
        pi = np.ones(K) / K
        idx = np.random.choice(len(Y_complete_init), K, replace=False)
        mu = Y_complete_init[idx]
        Sigma = np.array([np.eye(d) for _ in range(K)])
    
    return pi, mu, Sigma

def mvn_logpdf(Y, mu, Sigma):
    """Multivariate normal log-pdf handling missing data"""
    n, d = Y.shape
    logprob = np.zeros(n)
    
    for i in range(n):
        obs_idx = ~np.isnan(Y[i])
        
        if np.sum(obs_idx) == 0:
            logprob[i] = 0  # No data, uniform
            continue
        
        y_obs = Y[i, obs_idx]
        mu_obs = mu[obs_idx]
        Sigma_obs = Sigma[np.ix_(obs_idx, obs_idx)]
        
        # Log-pdf
        try:
            logprob[i] = stats.multivariate_normal.logpdf(
                y_obs, mu_obs, Sigma_obs
            )
        except:
            logprob[i] = -1e10  # Numerical issue
    
    return logprob

def e_step(Y, pi, mu, Sigma):
    """E-step: Compute responsibilities"""
    n = len(Y)
    K = len(pi)
    
    # Log-responsibilities (n x K)
    log_resp = np.zeros((n, K))
    
    for k in range(K):
        log_resp[:, k] = np.log(pi[k] + 1e-10) + mvn_logpdf(Y, mu[k], Sigma[k])
    
    # Normalize (log-sum-exp trick)
    log_sum = logsumexp(log_resp, axis=1, keepdims=True)
    log_resp -= log_sum
    resp = np.exp(log_resp)
    
    # Log-likelihood
    loglik = np.sum(log_sum)
    
    return resp, loglik

def impute_missing(Y, resp, mu, Sigma):
    """Impute missing values using current parameters"""
    Y_imputed = Y.copy()
    n, d = Y.shape
    K = len(mu)
    
    for i in range(n):
        if np.any(np.isnan(Y[i])):
            obs_idx = ~np.isnan(Y[i])
            miss_idx = np.isnan(Y[i])
            
            if np.sum(obs_idx) == 0:
                # No observed values: Use weighted mean
                Y_imputed[i] = np.sum(resp[i][:, None] * mu, axis=0)
            else:
                # Conditional expectation E[Y_miss|Y_obs, k]
                imputed_values = np.zeros(d)
                
                for k in range(K):
                    y_obs = Y[i, obs_idx]
                    mu_obs = mu[k, obs_idx]
                    mu_miss = mu[k, miss_idx]
                    
                    Sigma_obs_obs = Sigma[k][np.ix_(obs_idx, obs_idx)]
                    Sigma_miss_obs = Sigma[k][np.ix_(miss_idx, obs_idx)]
                    
                    try:
                        Sigma_obs_inv = np.linalg.inv(Sigma_obs_obs)
                        conditional_mean = mu_miss + Sigma_miss_obs @ Sigma_obs_inv @ (y_obs - mu_obs)
                        imputed_values[miss_idx] += resp[i, k] * conditional_mean
                    except:
                        imputed_values[miss_idx] += resp[i, k] * mu_miss
                
                Y_imputed[i, miss_idx] = imputed_values[miss_idx]
    
    return Y_imputed

def m_step(Y, resp):
    """M-step: Update parameters"""
    n, d = Y.shape
    K = resp.shape[1]
    
    # Effective sample sizes
    n_k = np.sum(resp, axis=0)
    
    # Mixing proportions
    pi = n_k / n
    
    # Means (weighted)
    mu = np.zeros((K, d))
    for k in range(K):
        mu[k] = np.sum(resp[:, k][:, None] * Y, axis=0) / n_k[k]
    
    # Covariances (weighted)
    Sigma = np.zeros((K, d, d))
    for k in range(K):
        Y_centered = Y - mu[k]
        Sigma[k] = (resp[:, k][:, None, None] * Y_centered[:, :, None] * Y_centered[:, None, :]).sum(axis=0) / n_k[k]
        
        # Regularization
        Sigma[k] += np.eye(d) * 1e-6
    
    return pi, mu, Sigma

# Initialize
pi, mu, Sigma = initialize_params(Y_observed, K, method='kmeans')

print(f"Initialization:")
print(f"  π: {pi}")
print(f"  μ:\n{mu}")

# EM Iterations
max_iter = 100
tol = 1e-6
loglik_history = []

print(f"\nRunning EM Algorithm:")
print(f"  Max iterations: {max_iter}")
print(f"  Tolerance: {tol}")

for t in range(max_iter):
    # E-step
    resp, loglik = e_step(Y_observed, pi, mu, Sigma)
    loglik_history.append(loglik)
    
    # Impute missing data
    Y_imputed = impute_missing(Y_observed, resp, mu, Sigma)
    
    # M-step
    pi_new, mu_new, Sigma_new = m_step(Y_imputed, resp)
    
    # Check convergence
    if t > 0:
        loglik_change = loglik - loglik_history[-2]
        rel_change = abs(loglik_change) / abs(loglik_history[-2])
        
        if t % 10 == 0:
            print(f"  Iter {t:3d}: log-lik = {loglik:.2f}, "
                  f"change = {loglik_change:+.4f}")
        
        if rel_change < tol:
            print(f"\n  ✓ Converged at iteration {t}")
            print(f"    Final log-likelihood: {loglik:.4f}")
            break
    
    pi, mu, Sigma = pi_new, mu_new, Sigma_new
else:
    print(f"\n  ⚠ Maximum iterations reached")

# Final responsibilities
resp_final, loglik_final = e_step(Y_observed, pi, mu, Sigma)
cluster_pred = np.argmax(resp_final, axis=1)

print(f"\nFinal Parameters:")
print(f"  π: {pi}")
print(f"  μ:\n{mu}")

print(f"\nPredicted Cluster Sizes: {np.bincount(cluster_pred, minlength=K)}")

# ===== Model Comparison: Complete Data vs EM with Missing =====
print("\n" + "="*80)
print("COMPARISON: COMPLETE DATA vs MISSING DATA EM")
print("="*80)

# Run EM on complete data
pi_complete, mu_complete, Sigma_complete = initialize_params(Y_complete, K)

for t in range(max_iter):
    resp_complete, _ = e_step(Y_complete, pi_complete, mu_complete, Sigma_complete)
    pi_complete, mu_complete, Sigma_complete = m_step(Y_complete, resp_complete)
    
    if t > 0 and abs(loglik_history[-1] - loglik_history[-2]) < tol:
        break

print(f"Complete Data Estimates:")
print(f"  π: {pi_complete}")
print(f"  μ:\n{mu_complete}")

print(f"\nMissing Data EM Estimates:")
print(f"  π: {pi}")
print(f"  μ:\n{mu}")

print(f"\nTrue Parameters:")
print(f"  π: {pi_true}")
print(f"  μ:\n{mu_true}")

# ===== Multiple Random Starts =====
print("\n" + "="*80)
print("MULTIPLE RANDOM STARTS")
print("="*80)

n_starts = 10
best_loglik = -np.inf
best_params = None

print(f"Running {n_starts} random initializations...")

for start in range(n_starts):
    np.random.seed(start)
    
    pi_init, mu_init, Sigma_init = initialize_params(Y_observed, K, method='random')
    
    pi_temp, mu_temp, Sigma_temp = pi_init, mu_init, Sigma_init
    
    for t in range(max_iter):
        resp_temp, loglik_temp = e_step(Y_observed, pi_temp, mu_temp, Sigma_temp)
        Y_imputed_temp = impute_missing(Y_observed, resp_temp, mu_temp, Sigma_temp)
        pi_temp, mu_temp, Sigma_temp = m_step(Y_imputed_temp, resp_temp)
        
        if t > 0 and abs(loglik_temp - loglik_history[-1]) < tol:
            break
    
    resp_temp, loglik_temp = e_step(Y_observed, pi_temp, mu_temp, Sigma_temp)
    
    print(f"  Start {start+1}: log-lik = {loglik_temp:.2f}")
    
    if loglik_temp > best_loglik:
        best_loglik = loglik_temp
        best_params = (pi_temp, mu_temp, Sigma_temp)

print(f"\nBest log-likelihood: {best_loglik:.4f}")
print(f"Original initialization: {loglik_final:.4f}")
print(f"Improvement: {best_loglik - loglik_final:.4f}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: True Data with True Clusters
axes[0, 0].scatter(Y_complete[:, 0], Y_complete[:, 1], 
                  c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
axes[0, 0].scatter(mu_true[:, 0], mu_true[:, 1], 
                  c='red', marker='X', s=200, edgecolors='black', 
                  linewidths=2, label='True Centers')
axes[0, 0].set_xlabel('X₁')
axes[0, 0].set_ylabel('X₂')
axes[0, 0].set_title('True Data (Complete)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Observed Data (with missing)
Y_plot = Y_observed.copy()
complete_mask = ~np.any(np.isnan(Y_observed), axis=1)

axes[0, 1].scatter(Y_plot[complete_mask, 0], Y_plot[complete_mask, 1],
                  c=cluster_pred[complete_mask], cmap='viridis', 
                  alpha=0.6, s=30, label='Complete obs')
axes[0, 1].scatter(mu[:, 0], mu[:, 1], 
                  c='red', marker='X', s=200, edgecolors='black', 
                  linewidths=2, label='EM Centers')
axes[0, 1].set_xlabel('X₁')
axes[0, 1].set_ylabel('X₂')
axes[0, 1].set_title(f'EM Clustering ({missing_pct:.0f}% Missing)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Log-Likelihood Convergence
axes[0, 2].plot(loglik_history, linewidth=2)
axes[0, 2].set_xlabel('Iteration')
axes[0, 2].set_ylabel('Log-Likelihood')
axes[0, 2].set_title('EM Convergence')
axes[0, 2].grid(alpha=0.3)

# Plot 4: Responsibilities Heatmap
axes[1, 0].imshow(resp_final[:50].T, aspect='auto', cmap='YlOrRd', 
                 interpolation='nearest')
axes[1, 0].set_xlabel('Observation')
axes[1, 0].set_ylabel('Component')
axes[1, 0].set_title('Responsibilities γᵢₖ (first 50 obs)')
axes[1, 0].set_yticks([0, 1, 2])
axes[1, 0].colorbar = plt.colorbar(
    axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046
)

# Plot 5: Hard vs Soft Clustering
uncertainty = 1 + np.sum(resp_final * np.log(resp_final + 1e-10), axis=1) / np.log(K)
axes[1, 1].scatter(Y_complete[:, 0], Y_complete[:, 1], 
                  c=uncertainty, cmap='coolwarm', alpha=0.6, s=30)
axes[1, 1].scatter(mu[:, 0], mu[:, 1], 
                  c='black', marker='X', s=200, edgecolors='white', 
                  linewidths=2)
axes[1, 1].set_xlabel('X₁')
axes[1, 1].set_ylabel('X₂')
axes[1, 1].set_title('Clustering Uncertainty (Entropy)')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Certainty')
axes[1, 1].grid(alpha=0.3)

# Plot 6: Parameter Comparison
param_names = [f'π₁', f'π₂', f'π₃', 
               f'μ₁₁', f'μ₁₂', f'μ₂₁', f'μ₂₂', f'μ₃₁', f'μ₃₂']
true_vals = np.concatenate([pi_true, mu_true.flatten()])
em_vals = np.concatenate([pi, mu.flatten()])
complete_vals = np.concatenate([pi_complete, mu_complete.flatten()])

x_pos = np.arange(len(param_names))
width = 0.25

axes[1, 2].bar(x_pos - width, true_vals, width, label='True', alpha=0.7)
axes[1, 2].bar(x_pos, em_vals, width, label='EM (Missing)', alpha=0.7)
axes[1, 2].bar(x_pos + width, complete_vals, width, 
              label='EM (Complete)', alpha=0.7)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(param_names, rotation=45)
axes[1, 2].set_ylabel('Parameter Value')
axes[1, 2].set_title('Parameter Estimates Comparison')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('em_algorithm_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Convergence Properties:")
print(f"   Iterations to converge: {len(loglik_history)}")
print(f"   Final log-likelihood: {loglik_final:.4f}")
print(f"   Monotonic increase: ✓ (guaranteed by EM)")

print("\n2. Missing Data Handling:")
print(f"   {missing_pct:.1f}% data missing (MCAR)")
print(f"   EM imputes via E[Y_miss|Y_obs, θ]")
print(f"   Accounts for uncertainty in imputation")

print("\n3. Parameter Recovery:")
mse_pi = np.mean((pi - pi_true)**2)
mse_mu = np.mean((mu.flatten() - mu_true.flatten())**2)
print(f"   MSE(π): {mse_pi:.6f}")
print(f"   MSE(μ): {mse_mu:.6f}")

print("\n4. Multiple Initializations:")
print(f"   {n_starts} random starts explored")
print(f"   Local maxima issue mitigated")
print(f"   Best found {best_loglik - loglik_final:+.4f} better")

print("\n5. Practical Recommendations:")
print("   • Use K-means or hierarchical clustering for initialization")
print("   • Run multiple random starts (10-100)")
print("   • Monitor log-likelihood for convergence")
print("   • Check responsibilities for cluster uncertainty")
print("   • Regularize covariances (add small ε to diagonal)")
print("   • Use BIC/AIC for selecting K")

print("\n6. EM Advantages:")
print("   • Handles missing data naturally (MAR assumption)")
print("   • Guaranteed monotonic likelihood increase")
print("   • Often closed-form M-step")
print("   • Interpretable latent structure")
print("   • Flexible framework (HMM, factor analysis, etc.)")

print("\n7. Limitations:")
print("   ⚠ Converges to local maximum (initialization critical)")
print("   ⚠ Slow convergence with high missing data")
print("   ⚠ Identifiability issues (label switching)")
print("   ⚠ Requires correct model specification")
print("   ⚠ Standard errors require Louis's formula or bootstrap")
```

## 6. Challenge Round
When does EM fail or mislead?
- **Local maxima**: Non-convex likelihood → Solution depends on initialization; multiple random starts required; GMM with K>2 highly multimodal
- **Identifiability**: Mixture labels arbitrary (permutation invariance) → Post-hoc matching; label switching across runs
- **Slow convergence**: High fraction missing information → Linear rate λ=(fraction missing); accelerated EM or quasi-Newton methods
- **Model misspecification**: Wrong number of components K → BIC/AIC for selection; overfitting if K too large
- **Singular covariances**: Cluster collapses to single point → Regularize Σ̂ₖ + εI; constrain minimum eigenvalue
- **Not Missing at Random (NMAR)**: Missingness depends on unobserved values → EM biased; need selection model or sensitivity analysis

## 7. Key References
- [Dempster, Laird & Rubin (1977) - Maximum Likelihood from Incomplete Data via the EM Algorithm](https://www.jstor.org/stable/2984875)
- [McLachlan & Krishnan (2008) - The EM Algorithm and Extensions](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470191613)
- [Bishop (2006) - Pattern Recognition and Machine Learning, Chapter 9](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

---
**Status:** Foundational for latent variable models | **Complements:** MLE, Missing Data Imputation, Mixture Models, HMM
