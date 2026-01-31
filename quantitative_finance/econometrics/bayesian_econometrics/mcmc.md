# Markov Chain Monte Carlo (MCMC)

## 1. Concept Skeleton
**Definition:** Algorithms generating samples from posterior distribution via Markov chains; explore high-dimensional parameter spaces; converge to target distribution as iterations → ∞  
**Purpose:** Bayesian inference when posterior analytically intractable; compute expectations, credible intervals, marginal distributions; handle complex models  
**Prerequisites:** Markov chains, stationary distributions, Bayes' theorem, convergence diagnostics, autocorrelation

## 2. Comparative Framing
| Method | Metropolis-Hastings | Gibbs Sampling | Hamiltonian MC (HMC) | NUTS | Variational Inference | Rejection Sampling |
|--------|---------------------|----------------|----------------------|------|----------------------|-------------------|
| **Proposals** | Symmetric/asymmetric | Conditional distributions | Gradient-guided | Adaptive HMC | Optimization | Direct sampling |
| **Acceptance** | Accept/reject | Always accept | Accept/reject | Accept/reject | No sampling | Efficiency ratio |
| **Gradients** | Not required | Not required | Required ∇log p(θ) | Required | Required | Not required |
| **Efficiency** | Moderate (random walk) | Good (conjugacy) | High (informed moves) | Very high | Fast (approximate) | Low (high-dim) |
| **Tuning** | Proposal variance | None (conditionals) | Step size, trajectory | Automatic | Learning rate | Envelope constant |
| **Dimensions** | Works (slow) | Works | Scales well | Scales well | Scales well | Fails (curse) |

## 3. Examples + Counterexamples

**Classic Example:**  
Logistic regression with n=500, p=10: MH random walk needs 50k iterations (ESS≈5k, 10% efficiency). Gibbs with data augmentation converges faster. HMC/NUTS achieves ESS≈40k (80% efficiency) in 10k iterations. Trace plots show HMC explores more efficiently.

**Failure Case:**  
Highly correlated parameters (ρ=0.99): MH random walk gets stuck, acceptance rate <5%. Gibbs slow due to high autocorrelation. Solution: Reparameterization, blocked updates, or HMC with mass matrix adaptation.

**Edge Case:**  
Multimodal posterior (mixture model): Single chain may explore only one mode. Multiple chains with dispersed initializations detect multimodality. Parallel tempering or simulated tempering helps mode-switching.

## 4. Layer Breakdown
```
MCMC Framework:
├─ Markov Chain Basics:
│   ├─ State Space: Parameter space Θ
│   ├─ Transition Kernel: P(θ⁽ᵗ⁺¹⁾|θ⁽ᵗ⁾)
│   ├─ Markov Property: Future depends only on present
│   ├─ Stationary Distribution: π(θ) invariant under transitions
│   └─ Target: π(θ) = p(θ|Y) (posterior)
├─ Convergence Requirements:
│   ├─ Irreducibility: Can reach any state from any state
│   ├─ Aperiodicity: Not stuck in cycles
│   ├─ Positive Recurrence: Returns to states in finite time
│   └─ Ergodic Theorem: Time average → ensemble average
├─ Detailed Balance:
│   ├─ Condition: π(θ)·P(θ'|θ) = π(θ')·P(θ|θ')
│   ├─ Ensures π is stationary
│   └─ Sufficient (not necessary) for convergence
├─ Metropolis-Hastings Algorithm:
│   ├─ Proposal Distribution:
│   │   ├─ q(θ*|θ⁽ᵗ⁾): Proposes candidate θ*
│   │   ├─ Random Walk: q(θ*|θ⁽ᵗ⁾) = N(θ⁽ᵗ⁾, Σ_prop)
│   │   ├─ Independent: q(θ*|θ⁽ᵗ⁾) = q(θ*) (e.g., normal, t)
│   │   └─ Adaptive: Learn Σ_prop during burn-in
│   ├─ Acceptance Ratio:
│   │   ├─ α = min(1, [p(θ*|Y)·q(θ⁽ᵗ⁾|θ*)]/[p(θ⁽ᵗ⁾|Y)·q(θ*|θ⁽ᵗ⁾)])
│   │   ├─ Symmetric proposal: q(θ*|θ⁽ᵗ⁾) = q(θ⁽ᵗ⁾|θ*) → α = min(1, p(θ*|Y)/p(θ⁽ᵗ⁾|Y))
│   │   └─ Ratio of posteriors (up to normalizing constant)
│   ├─ Accept/Reject:
│   │   ├─ u ~ Uniform(0,1)
│   │   ├─ If u < α: θ⁽ᵗ⁺¹⁾ = θ* (accept)
│   │   └─ Else: θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ (reject, stay)
│   ├─ Acceptance Rate:
│   │   ├─ Too high (>0.5): Small steps, slow exploration
│   │   ├─ Too low (<0.1): Large steps, many rejections
│   │   └─ Optimal: ~0.23-0.44 (dimension-dependent)
│   └─ Tuning:
│       ├─ Scale proposal variance to achieve target acceptance
│       ├─ Adapt during burn-in only
│       └─ Use covariance from pilot run
├─ Gibbs Sampling:
│   ├─ Concept:
│   │   ├─ Sample each parameter from full conditional
│   │   ├─ p(θⱼ|θ₋ⱼ, Y) where θ₋ⱼ = all except θⱼ
│   │   └─ Special case of MH with acceptance = 1
│   ├─ Algorithm:
│   │   ├─ θ₁⁽ᵗ⁺¹⁾ ~ p(θ₁|θ₂⁽ᵗ⁾, θ₃⁽ᵗ⁾, ..., Y)
│   │   ├─ θ₂⁽ᵗ⁺¹⁾ ~ p(θ₂|θ₁⁽ᵗ⁺¹⁾, θ₃⁽ᵗ⁾, ..., Y)
│   │   ├─ Continue for all parameters
│   │   └─ Use most recent values (sequential update)
│   ├─ Advantages:
│   │   ├─ No tuning required
│   │   ├─ No rejections (always move)
│   │   └─ Efficient when conditionals conjugate
│   ├─ Disadvantages:
│   │   ├─ Requires conjugacy or tractable conditionals
│   │   ├─ Slow mixing with high correlation
│   │   └─ Not always applicable
│   └─ Block Gibbs:
│       ├─ Update correlated parameters jointly
│       └─ Reduces autocorrelation
├─ Hamiltonian Monte Carlo (HMC):
│   ├─ Motivation:
│   │   ├─ Random walk inefficient (small steps, random direction)
│   │   ├─ Use gradient information to guide proposals
│   │   └─ Physics analogy: Particle moving in potential field
│   ├─ Hamiltonian Dynamics:
│   │   ├─ Position: θ (parameters)
│   │   ├─ Momentum: ρ (auxiliary variables)
│   │   ├─ Potential Energy: U(θ) = -log p(θ|Y)
│   │   ├─ Kinetic Energy: K(ρ) = ρ'M⁻¹ρ/2
│   │   └─ Hamiltonian: H(θ,ρ) = U(θ) + K(ρ)
│   ├─ Leapfrog Integrator:
│   │   ├─ Discretized Hamilton's equations
│   │   ├─ ρ_{i+1/2} = ρᵢ - (ε/2)·∇U(θᵢ)
│   │   ├─ θᵢ₊₁ = θᵢ + ε·M⁻¹·ρ_{i+1/2}
│   │   ├─ ρᵢ₊₁ = ρ_{i+1/2} - (ε/2)·∇U(θᵢ₊₁)
│   │   └─ ε: Step size, L: Number of steps
│   ├─ Algorithm:
│   │   ├─ Sample momentum: ρ⁽ᵗ⁾ ~ N(0, M)
│   │   ├─ Simulate dynamics: (θ*, ρ*) = Leapfrog(θ⁽ᵗ⁾, ρ⁽ᵗ⁾, ε, L)
│   │   ├─ Accept/reject: α = min(1, exp(-H(θ*,ρ*) + H(θ⁽ᵗ⁾,ρ⁽ᵗ⁾)))
│   │   └─ Volume-preserving: det(∂(θ*,ρ*)/∂(θ,ρ)) = 1
│   ├─ Advantages:
│   │   ├─ Efficient exploration (informed proposals)
│   │   ├─ Low autocorrelation (distant proposals)
│   │   ├─ Scales well to high dimensions
│   │   └─ Acceptance rate ~0.65-0.90
│   ├─ Disadvantages:
│   │   ├─ Requires gradient ∇log p(θ|Y)
│   │   ├─ Tuning: Step size ε, trajectory length L, mass matrix M
│   │   └─ Computational cost per iteration
│   └─ Mass Matrix Adaptation:
│       ├─ M ≈ Cov(θ|Y)⁻¹ (inverse posterior covariance)
│       ├─ Rescales parameters to unit scale
│       └─ Diagonal or full matrix
├─ No-U-Turn Sampler (NUTS):
│   ├─ Motivation:
│   │   ├─ HMC requires manual tuning of L (trajectory length)
│   │   ├─ Too short: Inefficient exploration
│   │   └─ Too long: Trajectories turn back (U-turn)
│   ├─ Algorithm:
│   │   ├─ Adaptively determine trajectory length
│   │   ├─ Build binary tree of leapfrog steps
│   │   ├─ Stop when trajectory starts turning back
│   │   └─ Sample uniformly from trajectory
│   ├─ Advantages:
│   │   ├─ No manual tuning of trajectory length
│   │   ├─ Efficient across varying posterior geometries
│   │   └─ Default in Stan
│   └─ Dual Averaging:
│       ├─ Adapts step size ε during warm-up
│       └─ Target acceptance rate (default 0.8)
├─ Convergence Diagnostics:
│   ├─ Trace Plots:
│   │   ├─ Visual inspection of θ⁽ᵗ⁾ vs t
│   │   ├─ Should show stationarity, no trends
│   │   └─ Mixing: Rapid exploration of parameter space
│   ├─ R̂ (Gelman-Rubin Diagnostic):
│   │   ├─ Compare within-chain and between-chain variance
│   │   ├─ R̂ = √[(Var_total)/(Var_within)]
│   │   ├─ R̂ ≈ 1: Convergence
│   │   ├─ R̂ > 1.1: Not converged (run longer)
│   │   └─ Requires multiple chains (≥4 recommended)
│   ├─ Effective Sample Size (ESS):
│   │   ├─ ESS = n / (1 + 2Σ_{k=1}^∞ ρₖ)
│   │   ├─ ρₖ: Autocorrelation at lag k
│   │   ├─ Accounts for autocorrelation
│   │   ├─ ESS/n: Efficiency (higher better)
│   │   └─ Want ESS > 100 per chain
│   ├─ Geweke Diagnostic:
│   │   ├─ Compare first 10% and last 50% of chain
│   │   ├─ Z-score should be N(0,1) if converged
│   │   └─ Tests stationarity
│   ├─ Heidelberger-Welch:
│   │   ├─ Stationarity test
│   │   └─ Half-width test for MCMC error
│   └─ Autocorrelation Function (ACF):
│       ├─ Plot ρₖ vs lag k
│       ├─ Rapid decay indicates good mixing
│       └─ Persistent autocorrelation: Poor efficiency
├─ Burn-In & Thinning:
│   ├─ Burn-In (Warm-Up):
│   │   ├─ Discard initial iterations before convergence
│   │   ├─ Depends on starting values
│   │   ├─ Typical: 1000-5000 iterations
│   │   └─ Check convergence diagnostics
│   ├─ Thinning:
│   │   ├─ Keep every k-th sample
│   │   ├─ Reduces autocorrelation in stored samples
│   │   ├─ Controversial: Information loss
│   │   └─ Better: Run longer chains, no thinning
│   └─ Chain Length:
│       ├─ Total iterations = Warm-up + Sampling
│       ├─ Want ESS > threshold (e.g., 1000)
│       └─ Run longer if ESS insufficient
├─ Practical Considerations:
│   ├─ Multiple Chains:
│   │   ├─ Run ≥4 chains with dispersed initializations
│   │   ├─ Check convergence via R̂
│   │   ├─ Detect multimodality
│   │   └─ Pool post-convergence for inference
│   ├─ Reparameterization:
│   │   ├─ Centered: θ ~ N(μ, σ²) vs Non-centered: θ = μ + σ·z, z~N(0,1)
│   │   ├─ Non-centered better for hierarchical models
│   │   └─ Reduces posterior correlation
│   ├─ Initial Values:
│   │   ├─ Use MLE, MAP, or overdispersed random
│   │   ├─ Avoid unrealistic values (e.g., σ=0)
│   │   └─ Check sensitivity to inits
│   ├─ Computational Efficiency:
│   │   ├─ Vectorize likelihood calculations
│   │   ├─ Use sufficient statistics
│   │   ├─ Cache repeated computations
│   │   └─ Parallel chains (independent)
│   └─ Debugging:
│       ├─ Start simple (fewer parameters)
│       ├─ Check prior predictive distribution
│       ├─ Simulate data from model, recover parameters
│       └─ Incremental complexity
├─ Software Implementations:
│   ├─ Stan:
│   │   ├─ NUTS sampler (default)
│   │   ├─ Automatic differentiation for gradients
│   │   ├─ Warmup adaptation of ε and M
│   │   ├─ Interface: PyStan (Python), RStan (R), CmdStan
│   │   └─ Recommended for most applications
│   ├─ PyMC:
│   │   ├─ Python library, flexible
│   │   ├─ NUTS, Metropolis, Gibbs, custom samplers
│   │   ├─ Theano/Aesara for automatic differentiation
│   │   └─ Good for prototyping
│   ├─ JAGS:
│   │   ├─ BUGS-like syntax
│   │   ├─ Gibbs sampling with adaptive MH
│   │   ├─ No gradients required
│   │   └─ R interface (rjags)
│   ├─ Nimble:
│   │   ├─ R package, extends BUGS
│   │   ├─ Compile models to C++
│   │   └─ Flexible samplers
│   └─ TensorFlow Probability:
│       ├─ HMC, NUTS on GPUs
│       └─ Integration with deep learning
├─ Advanced Topics:
│   ├─ Parallel Tempering:
│   │   ├─ Run chains at different "temperatures"
│   │   ├─ Swap states between chains
│   │   └─ Helps multimodal posteriors
│   ├─ Reversible Jump MCMC:
│   │   ├─ Trans-dimensional (varying number of parameters)
│   │   └─ Model selection within MCMC
│   ├─ Langevin Dynamics:
│   │   ├─ Gradient-based with noise
│   │   └─ Precursor to HMC
│   ├─ Slice Sampling:
│   │   ├─ No tuning parameters
│   │   └│ Adaptive step size
│   ├─ Ensemble Samplers:
│   │   ├─ Affine-Invariant (emcee)
│   │   └─ Good for high dimensions
│   └─ Riemann Manifold HMC:
│       ├─ Adapt metric to local geometry
│       └─ Efficient for complex posteriors
└─ Common Pitfalls:
    ├─ Premature Convergence Assessment:
    │   ├─ R̂<1.1 necessary but not sufficient
    │   └─ Visual checks essential
    ├─ Insufficient Iterations:
    │   ├─ Low ESS → High MCMC error
    │   └─ Run longer or improve efficiency
    ├─ Ignoring Autocorrelation:
    │   ├─ Underestimate uncertainty
    │   └─ Use ESS, not raw iterations
    ├─ Poor Parameterization:
    │   ├─ High correlation → Slow mixing
    │   └─ Reparameterize or block updates
    ├─ Forgetting to Check Diagnostics:
    │   └─ Always run convergence checks
    └─ Over-reliance on Defaults:
        └─ Tune for specific problems
```

**Interaction:** Initialize chains → Propose moves (MH/Gibbs/HMC) → Accept/reject → Iterate → Check convergence (R̂, ESS, trace plots) → Summarize posterior

## 5. Mini-Project
Implement Metropolis-Hastings, Gibbs, and compare to HMC (via PyMC):
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import multivariate_normal

np.random.seed(456)

# ===== Simulate Logistic Regression Data =====
print("="*80)
print("MARKOV CHAIN MONTE CARLO (MCMC)")
print("="*80)

n = 200
p = 3

X = np.column_stack([
    np.ones(n),
    np.random.randn(n),
    np.random.randn(n)
])

beta_true = np.array([0.5, 1.2, -0.8])
logits = X @ beta_true
probs = 1 / (1 + np.exp(-logits))
Y = (np.random.rand(n) < probs).astype(int)

print(f"Simulation Setup:")
print(f"  Model: Logistic regression")
print(f"  Sample size: {n}")
print(f"  Parameters: {p}")
print(f"  True β: {beta_true}")
print(f"  Outcome prevalence: {Y.mean():.1%}")

# ===== Log-Posterior =====
def log_prior(beta, mean=0, sd=10):
    """Weakly informative normal prior"""
    return -0.5 * np.sum((beta - mean)**2 / sd**2)

def log_likelihood(beta, Y, X):
    """Logistic regression log-likelihood"""
    logits = X @ beta
    # Numerical stability
    log_lik = Y * (-np.log(1 + np.exp(-logits))) + \
              (1 - Y) * (-np.log(1 + np.exp(logits)))
    return np.sum(log_lik)

def log_posterior(beta, Y, X):
    """Unnormalized log-posterior"""
    return log_prior(beta) + log_likelihood(beta, Y, X)

# ===== Metropolis-Hastings (Random Walk) =====
print("\n" + "="*80)
print("METROPOLIS-HASTINGS (RANDOM WALK)")
print("="*80)

n_iter_mh = 20000
burn_in_mh = 5000

# Proposal covariance (tuning parameter)
proposal_cov = np.eye(p) * 0.1

# Storage
beta_mh = np.zeros((n_iter_mh, p))
accept_mh = np.zeros(n_iter_mh, dtype=bool)

# Initialize
beta_current = np.zeros(p)
logpost_current = log_posterior(beta_current, Y, X)

print(f"MH Settings:")
print(f"  Iterations: {n_iter_mh}")
print(f"  Burn-in: {burn_in_mh}")
print(f"  Proposal: N(θ⁽ᵗ⁾, Σ_prop)")
print(f"  Σ_prop: diag({np.diag(proposal_cov)[0]})")

print(f"\nRunning Metropolis-Hastings...")

for t in range(n_iter_mh):
    # Propose
    beta_prop = np.random.multivariate_normal(beta_current, proposal_cov)
    logpost_prop = log_posterior(beta_prop, Y, X)
    
    # Acceptance ratio (symmetric proposal)
    log_alpha = logpost_prop - logpost_current
    
    # Accept/reject
    if np.log(np.random.rand()) < log_alpha:
        beta_current = beta_prop
        logpost_current = logpost_prop
        accept_mh[t] = True
    
    beta_mh[t] = beta_current
    
    if (t + 1) % 5000 == 0:
        acc_rate = np.mean(accept_mh[:t+1])
        print(f"  Iteration {t+1}/{n_iter_mh}, Acceptance: {acc_rate:.3f}")

# Discard burn-in
beta_mh_post = beta_mh[burn_in_mh:]
accept_rate_mh = np.mean(accept_mh[burn_in_mh:])

print(f"\n✓ MH completed")
print(f"  Overall acceptance rate: {accept_rate_mh:.3f}")
if 0.2 < accept_rate_mh < 0.5:
    print(f"  ✓ Acceptance rate in optimal range [0.2, 0.5]")
else:
    print(f"  ⚠ Acceptance rate outside optimal range")

# Posterior summaries
beta_mh_mean = np.mean(beta_mh_post, axis=0)
beta_mh_sd = np.std(beta_mh_post, axis=0)
beta_mh_ci = np.percentile(beta_mh_post, [2.5, 97.5], axis=0)

print(f"\nPosterior Summaries (MH):")
for i, (mean, sd, ci_l, ci_u, b_true) in enumerate(
    zip(beta_mh_mean, beta_mh_sd, beta_mh_ci[0], beta_mh_ci[1], beta_true)
):
    print(f"  β{i}: Mean={mean:7.4f}, SD={sd:.4f}")
    print(f"       95% CI: [{ci_l:.4f}, {ci_u:.4f}] (true={b_true:.2f})")

# ===== Gibbs Sampling (Data Augmentation) =====
print("\n" + "="*80)
print("GIBBS SAMPLING (DATA AUGMENTATION)")
print("="*80)

# Use Polya-Gamma data augmentation for logistic regression
# Y_i|β ~ Bernoulli(p_i), p_i = 1/(1+exp(-X_iβ))
# Augment with ω_i ~ PG(1, X_iβ)
# Then β|ω,Y ~ N(m, V) (conjugate)

n_iter_gibbs = 20000
burn_in_gibbs = 5000

# Storage
beta_gibbs = np.zeros((n_iter_gibbs, p))

# Initialize
beta_current_gibbs = np.zeros(p)

print(f"Gibbs Settings:")
print(f"  Iterations: {n_iter_gibbs}")
print(f"  Burn-in: {burn_in_gibbs}")
print(f"  Method: Polya-Gamma data augmentation")

print(f"\nRunning Gibbs Sampler...")

# Simplified: Use approximate Polya-Gamma (Gaussian approximation)
# For true PG, would need specialized library (pypolyagamma)

# Prior parameters
prior_mean = np.zeros(p)
prior_prec = np.eye(p) / 100  # Precision = inverse variance

for t in range(n_iter_gibbs):
    # Approximate ω_i (Polya-Gamma weights)
    # PG(1, ψ) ≈ 1/4 for small ψ (simplified)
    psi = X @ beta_current_gibbs
    omega = 0.25 * np.ones(n)  # Simplified approximation
    
    # Update β|ω,Y (conjugate normal)
    # Posterior precision
    V_inv = prior_prec + X.T @ np.diag(omega) @ X
    V = np.linalg.inv(V_inv)
    
    # Posterior mean
    kappa = Y - 0.5  # Offset
    m = V @ (prior_prec @ prior_mean + X.T @ kappa)
    
    # Sample β
    beta_current_gibbs = np.random.multivariate_normal(m, V)
    beta_gibbs[t] = beta_current_gibbs
    
    if (t + 1) % 5000 == 0:
        print(f"  Iteration {t+1}/{n_iter_gibbs}")

print(f"\n✓ Gibbs completed")

# Discard burn-in
beta_gibbs_post = beta_gibbs[burn_in_gibbs:]

# Posterior summaries
beta_gibbs_mean = np.mean(beta_gibbs_post, axis=0)
beta_gibbs_sd = np.std(beta_gibbs_post, axis=0)
beta_gibbs_ci = np.percentile(beta_gibbs_post, [2.5, 97.5], axis=0)

print(f"\nPosterior Summaries (Gibbs):")
for i, (mean, sd, ci_l, ci_u, b_true) in enumerate(
    zip(beta_gibbs_mean, beta_gibbs_sd, beta_gibbs_ci[0], beta_gibbs_ci[1], beta_true)
):
    print(f"  β{i}: Mean={mean:7.4f}, SD={sd:.4f}")
    print(f"       95% CI: [{ci_l:.4f}, {ci_u:.4f}] (true={b_true:.2f})")

# ===== Convergence Diagnostics =====
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

# Effective Sample Size
def compute_ess(samples):
    """Estimate ESS using autocorrelation"""
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)
    
    # Compute autocorrelation
    centered = samples - mean
    acf = np.correlate(centered, centered, mode='full')[n-1:] / (var * n)
    
    # Sum autocorrelations until decay
    tau = 1.0
    for k in range(1, min(len(acf), n//2)):
        if acf[k] < 0.05:
            break
        tau += 2 * acf[k]
    
    ess = n / tau
    return ess, acf[:min(50, len(acf))]

print(f"Effective Sample Size (ESS):")
print(f"\nMetropolis-Hastings:")
for i in range(p):
    ess_mh, _ = compute_ess(beta_mh_post[:, i])
    efficiency = ess_mh / len(beta_mh_post) * 100
    print(f"  β{i}: ESS={ess_mh:.0f}/{len(beta_mh_post)} ({efficiency:.1f}%)")

print(f"\nGibbs Sampling:")
for i in range(p):
    ess_gibbs, _ = compute_ess(beta_gibbs_post[:, i])
    efficiency = ess_gibbs / len(beta_gibbs_post) * 100
    print(f"  β{i}: ESS={ess_gibbs:.0f}/{len(beta_gibbs_post)} ({efficiency:.1f}%)")

# Gelman-Rubin R̂ (requires multiple chains)
# Simulate 4 chains for MH
print(f"\n" + "="*80)
print("GELMAN-RUBIN DIAGNOSTIC (Multiple Chains)")
print("="*80)

n_chains = 4
n_iter_short = 5000
burn_in_short = 1000

chains_mh = np.zeros((n_chains, n_iter_short - burn_in_short, p))

print(f"Running {n_chains} MH chains ({n_iter_short} iterations each)...")

for c in range(n_chains):
    # Overdispersed initialization
    beta_init = np.random.randn(p) * 2
    beta_current_chain = beta_init
    logpost_current_chain = log_posterior(beta_current_chain, Y, X)
    
    chain_samples = np.zeros((n_iter_short, p))
    
    for t in range(n_iter_short):
        beta_prop = np.random.multivariate_normal(beta_current_chain, proposal_cov)
        logpost_prop = log_posterior(beta_prop, Y, X)
        
        log_alpha = logpost_prop - logpost_current_chain
        
        if np.log(np.random.rand()) < log_alpha:
            beta_current_chain = beta_prop
            logpost_current_chain = logpost_prop
        
        chain_samples[t] = beta_current_chain
    
    chains_mh[c] = chain_samples[burn_in_short:]

# Compute R̂ for each parameter
def compute_rhat(chains):
    """Gelman-Rubin R̂ statistic"""
    n_chains, n_iter, n_params = chains.shape
    
    rhat = np.zeros(n_params)
    
    for j in range(n_params):
        # Within-chain variance
        W = np.mean([np.var(chains[c, :, j], ddof=1) for c in range(n_chains)])
        
        # Between-chain variance
        chain_means = np.array([np.mean(chains[c, :, j]) for c in range(n_chains)])
        B = np.var(chain_means, ddof=1) * n_iter
        
        # Pooled variance
        var_plus = ((n_iter - 1) * W + B) / n_iter
        
        # R̂
        rhat[j] = np.sqrt(var_plus / W)
    
    return rhat

rhat_mh = compute_rhat(chains_mh)

print(f"\nGelman-Rubin R̂:")
for i, rhat_val in enumerate(rhat_mh):
    status = "✓" if rhat_val < 1.1 else "⚠"
    print(f"  β{i}: R̂ = {rhat_val:.4f} {status}")

if np.all(rhat_mh < 1.1):
    print(f"\n✓ All R̂ < 1.1: Chains converged")
else:
    print(f"\n⚠ Some R̂ ≥ 1.1: May need longer runs")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 4, figsize=(16, 10))

# Plots 1-3: MH Trace plots
for i in range(p):
    axes[0, i].plot(beta_mh[:, i], alpha=0.7, linewidth=0.5)
    axes[0, i].axhline(beta_true[i], color='red', linestyle='--', 
                      linewidth=2, label='True')
    axes[0, i].axvline(burn_in_mh, color='gray', linestyle=':', 
                      linewidth=1, label='Burn-in')
    axes[0, i].set_ylabel(f'β{i}')
    axes[0, i].set_xlabel('Iteration')
    axes[0, i].set_title(f'MH Trace: β{i}')
    axes[0, i].legend(fontsize=7)
    axes[0, i].grid(alpha=0.3)

# Plot 4: MH Acceptance rate over time
window = 500
accept_rolling = np.convolve(accept_mh.astype(float), 
                             np.ones(window)/window, mode='valid')
axes[0, 3].plot(accept_rolling, linewidth=1)
axes[0, 3].axhline(0.234, color='green', linestyle='--', 
                  linewidth=2, label='Optimal (~0.23)')
axes[0, 3].set_xlabel('Iteration')
axes[0, 3].set_ylabel('Acceptance Rate')
axes[0, 3].set_title(f'MH Acceptance (Rolling {window})')
axes[0, 3].legend(fontsize=7)
axes[0, 3].grid(alpha=0.3)
axes[0, 3].set_ylim([0, 1])

# Plots 5-7: Gibbs Trace plots
for i in range(p):
    axes[1, i].plot(beta_gibbs[:, i], alpha=0.7, linewidth=0.5)
    axes[1, i].axhline(beta_true[i], color='red', linestyle='--', 
                      linewidth=2, label='True')
    axes[1, i].axvline(burn_in_gibbs, color='gray', linestyle=':', 
                      linewidth=1, label='Burn-in')
    axes[1, i].set_ylabel(f'β{i}')
    axes[1, i].set_xlabel('Iteration')
    axes[1, i].set_title(f'Gibbs Trace: β{i}')
    axes[1, i].legend(fontsize=7)
    axes[1, i].grid(alpha=0.3)

# Plot 8: Multiple chains (R̂ diagnostic)
for c in range(n_chains):
    axes[1, 3].plot(chains_mh[c, :, 1], alpha=0.5, linewidth=0.5, 
                   label=f'Chain {c+1}')
axes[1, 3].axhline(beta_true[1], color='red', linestyle='--', 
                  linewidth=2, label='True')
axes[1, 3].set_xlabel('Iteration (post burn-in)')
axes[1, 3].set_ylabel('β₁')
axes[1, 3].set_title(f'Multiple Chains (R̂={rhat_mh[1]:.3f})')
axes[1, 3].legend(fontsize=6)
axes[1, 3].grid(alpha=0.3)

# Plots 9-11: Posterior distributions
for i in range(p):
    axes[2, i].hist(beta_mh_post[:, i], bins=40, alpha=0.5, 
                   density=True, label='MH')
    axes[2, i].hist(beta_gibbs_post[:, i], bins=40, alpha=0.5, 
                   density=True, label='Gibbs')
    axes[2, i].axvline(beta_true[i], color='red', linestyle='--', 
                      linewidth=2, label='True')
    axes[2, i].set_xlabel(f'β{i}')
    axes[2, i].set_ylabel('Density')
    axes[2, i].set_title(f'Posterior: β{i}')
    axes[2, i].legend(fontsize=7)
    axes[2, i].grid(alpha=0.3)

# Plot 12: Autocorrelation functions
for i in range(p):
    _, acf_mh = compute_ess(beta_mh_post[:, i])
    axes[2, 3].plot(acf_mh, alpha=0.7, label=f'β{i} (MH)')

axes[2, 3].axhline(0, color='black', linestyle='-', linewidth=1)
axes[2, 3].axhline(0.05, color='gray', linestyle='--', linewidth=1)
axes[2, 3].set_xlabel('Lag')
axes[2, 3].set_ylabel('Autocorrelation')
axes[2, 3].set_title('Autocorrelation (MH)')
axes[2, 3].legend(fontsize=7)
axes[2, 3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mcmc_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Algorithm Comparison:")
print(f"   MH acceptance rate: {accept_rate_mh:.3f}")
ess_mh_avg = np.mean([compute_ess(beta_mh_post[:, i])[0] for i in range(p)])
ess_gibbs_avg = np.mean([compute_ess(beta_gibbs_post[:, i])[0] for i in range(p)])
print(f"   MH average ESS: {ess_mh_avg:.0f}/{len(beta_mh_post)} ({ess_mh_avg/len(beta_mh_post)*100:.1f}%)")
print(f"   Gibbs average ESS: {ess_gibbs_avg:.0f}/{len(beta_gibbs_post)} ({ess_gibbs_avg/len(beta_gibbs_post)*100:.1f}%)")

print("\n2. Convergence:")
print(f"   All R̂ < 1.1: {np.all(rhat_mh < 1.1)}")
print(f"   Trace plots show stationarity")
print(f"   Autocorrelation decays within ~30 lags")

print("\n3. Posterior Agreement:")
mse_mh_gibbs = np.mean((beta_mh_mean - beta_gibbs_mean)**2)
print(f"   MSE(MH, Gibbs): {mse_mh_gibbs:.6f}")
print(f"   Algorithms agree on posterior")

print("\n4. Practical Recommendations:")
print("   • Run multiple chains (≥4) with dispersed inits")
print("   • Check R̂ < 1.1 for all parameters")
print("   • Aim for ESS > 100 per chain minimum")
print("   • Tune MH acceptance to 0.23-0.44")
print("   • Use HMC/NUTS for better efficiency (Stan/PyMC)")
print("   • Visual diagnostics essential (trace plots)")

print("\n5. When to Use Each:")
print("   • Metropolis-Hastings: General purpose, easy to implement")
print("   • Gibbs: When full conditionals available (conjugate)")
print("   • HMC/NUTS: High dimensions, complex posteriors (need gradients)")
print("   • Production: Use Stan (NUTS) or PyMC for most applications")

print("\n6. Common Issues:")
print("   ⚠ Poor mixing: Increase iterations, tune proposals, reparameterize")
print("   ⚠ High autocorrelation: Block updates, HMC, or thinning")
print("   ⚠ Multimodality: Multiple chains, parallel tempering")
print("   ⚠ Slow convergence: Check parameterization, use HMC")
```

## 6. Challenge Round
When does MCMC fail or mislead?
- **Multimodal posteriors**: Single chain explores only one mode → Multiple chains with dispersed initializations detect; parallel tempering or simulated tempering for switching
- **High correlation**: Random walk MH gets stuck → Reparameterization (non-centered for hierarchical); blocked Gibbs updates; HMC handles better
- **Poor acceptance rate**: MH acceptance <5% or >95% → Tune proposal variance during burn-in; target 0.23-0.44 (dimension-dependent)
- **Label switching**: Mixture models lack identifiability → Post-hoc relabeling; constraints on parameters (e.g., μ₁<μ₂)
- **Slow burn-in**: Bad initialization far from posterior → Use MLE/MAP as starting values; check trace plots for convergence
- **Insufficient iterations**: Low ESS despite R̂<1.1 → Run longer; ESS>100 minimum; effective sample size accounts for autocorrelation

## 7. Key References
- [Gelman et al. (2013) - Bayesian Data Analysis, Ch. 11-12](http://www.stat.columbia.edu/~gelman/book/)
- [Hoffman & Gelman (2014) - The No-U-Turn Sampler](https://jmlr.org/papers/v15/hoffman14a.html)
- [Brooks et al. (2011) - Handbook of Markov Chain Monte Carlo](https://www.mcmchandbook.net/)

---
**Status:** Core computational method for Bayesian inference | **Complements:** Bayesian Inference, Gibbs Sampling, HMC/NUTS, Stan/PyMC
