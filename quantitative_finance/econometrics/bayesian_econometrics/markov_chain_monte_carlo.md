# Markov Chain Monte Carlo (MCMC)

## 1. Concept Skeleton
**Definition:** Computational method generating samples from posterior distribution by constructing Markov chains with stationary distribution equal to target posterior  
**Purpose:** Approximate intractable posteriors; enable Bayesian inference in high-dimensional problems without closed-form conjugate updates  
**Prerequisites:** Markov chains, probability distributions, acceptance-rejection methods, convergence diagnostics, parallel sampling

## 2. Comparative Framing
| Algorithm | Sampling Method | Efficiency | Tuning | Dimensionality | Implementation |
|-----------|-----------------|-----------|--------|-----------------|----------------|
| **Gibbs Sampling** | Conditional distributions | Moderate | Minimal | Low-moderate | Direct sampling |
| **Metropolis-Hastings (MH)** | Proposal + acceptance | Variable | Proposal variance | Moderate | Rejection loop |
| **Hamiltonian MC (HMC)** | Gradient + physics | High | Step size, L | High | Complex (autodiff) |
| **No-U-Turn Sampler (NUTS)** | HMC adaptive | Very high | Adaptive | High | Stan default |
| **Slice Sampling** | Auxiliary variable | Moderate | Width parameter | Moderate | Specialized |

## 3. Examples + Counterexamples

**Simple Example:**  
Logistic regression posterior non-conjugate → Metropolis-Hastings: propose β* from random walk; accept w/ probability min(1, posterior(β*)/posterior(β_t)) → Chain converges to posterior; burn-in first 1000 iterations; then use samples for inference

**Failure Case:**  
Poor proposal covariance (underestimated) → Chain explores slowly (high autocorr); inefficient (~50% waste); acceptance rate ~23% (too low); run 100K iterations for effective N=5K; use adaptive algorithms (NUTS) to auto-tune

**Edge Case:**  
Multimodal posterior (two isolated modes) → HMC can get stuck in one mode; low probability of jumping between; solution: tempering (sample at different temperatures), parallel chains, or reparameterization

## 4. Layer Breakdown
```
MCMC Sampling Structure:
├─ Background: Why MCMC Needed:
│   ├─ Posterior p(θ|y) often non-conjugate (no closed form)
│   │   ├─ Likelihood complex (generalized linear models, latent variables)
│   │   ├─ Prior not conjugate to likelihood
│   │   └─ Integration ∫p(y|θ)p(θ)dθ intractable (high dimension)
│   ├─ Goal: Sample θ ~ p(θ|y) without computing normalization constant
│   │   ├─ Sufficient: Compute posterior up to constant (unnormalized)
│   │   ├─ Allows inference: E[θ|y] ≈ (1/M)Σθ_m, Var[θ|y] ≈ s²(θ_m)
│   │   └─ Benefits: Arbitrary functions of θ; credible intervals; marginals
│   └─ Markov chain property: X_t depends only on X_{t-1} (memoryless)
│       ├─ Transition kernel P(θ_t|θ_{t-1})
│       ├─ Stationary distribution π(θ): Invariant if ~ π always
│       └─ Key: Design kernel such that π(θ) = p(θ|y)
├─ Gibbs Sampling:
│   ├─ Principle: Sample conditional distributions
│   │   ├─ Posterior: p(θ₁,...,θ_K|y)
│   │   ├─ Conditional: p(θ_j|θ_{-j}, y) where θ_{-j} = all except θ_j
│   │   ├─ Gibbs step: θ_j^{(t+1)} ~ p(θ_j|θ₁^{(t)}, ..., θ_{j-1}^{(t)}, θ_{j+1}^{(t-1)}, ..., θ_K^{(t-1)}, y)
│   │   └─ Cycle through all j; repeat T times
│   ├─ Procedure:
│   │   ├─ Initialize θ^{(0)} (arbitrary or crude estimate)
│   │   ├─ For t=1 to T:
│   │   │   ├─ Sample θ₁^{(t)} ~ p(θ₁|θ₂^{(t-1)}, ..., θ_K^{(t-1)}, y)
│   │   │   ├─ Sample θ₂^{(t)} ~ p(θ₂|θ₁^{(t)}, θ₃^{(t-1)}, ..., θ_K^{(t-1)}, y)
│   │   │   ├─ ... (cycle through all K parameters)
│   │   │   └─ θ_K^{(t)} ~ p(θ_K|θ₁^{(t)}, ..., θ_{K-1}^{(t)}, y)
│   │   └─ Discard first T_burn iterations; use remainder for inference
│   ├─ Advantages:
│   │   ├─ Direct sampling (no rejection); always accept
│   │   ├─ No tuning required (if conditionals tractable)
│   │   ├─ Scalable to moderate dimensions (K~10-100)
│   │   └─ Conditional updates easy to implement
│   ├─ Example - mixture model:
│   │   ├─ Data y ~ mixture of K normals; unknown z_i (component assignment)
│   │   ├─ Conditional p(μ_k|z,y): Normal (conjugate with data)
│   │   ├─ Conditional p(z_i|μ,σ,y): Categorical (proportional to mixture components)
│   │   ├─ Gibbs: Alternate updating cluster means μ and assignments z
│   │   └─ Converges to posterior over (μ,z)
│   └─ Limitations:
│       ├─ Requires tractable conditionals (not always available)
│       ├─ Poor mixing if correlations strong (random walk within Gibbs)
│       ├─ High autocorrelation: Effective sample size << T (wasteful)
│       └─ Solution: Blocking (joint sample subsets) or reparameterization
├─ Metropolis-Hastings (MH):
│   ├─ Framework: Construct chain with acceptance rule
│   │   ├─ Proposal: θ* ~ q(θ*|θ_t) [arbitrary proposal density]
│   │   ├─ Acceptance probability: α = min(1, [p(θ*|y)/p(θ_t|y)] × [q(θ_t|θ*)/q(θ*|θ_t)])
│   │   ├─ Accept θ_{t+1} = θ* with prob α; else θ_{t+1} = θ_t (repeat current)
│   │   └─ Key: Metropolis-Hastings ratio corrects for proposal asymmetry
│   ├─ Special case - Random walk MH:
│   │   ├─ Proposal: θ* = θ_t + ε where ε ~ N(0, Σ_prop)
│   │   ├─ Symmetric: q(θ*|θ_t) = q(θ_t|θ*) → Ratio cancels
│   │   ├─ Acceptance: α = min(1, p(θ*|y)/p(θ_t|y))
│   │   ├─ Intuition: Accept uphill always; downhill probabilistically
│   │   └─ Example: Proposal σ = 0.5 → Acceptance rate ~50% (optimal)
│   ├─ Tuning proposal:
│   │   ├─ Too small: Acceptance high but chain explores slowly
│   │   ├─ Too large: Acceptance low; many rejections
│   │   ├─ Goldilocks: Acceptance rate 20-40% (higher in 1-D; lower in high-D)
│   │   ├─ Adaptive: Learn Σ_prop from earlier samples (adapting during burn-in)
│   │   └─ Automatic tuning: Stan, PyMC implement this
│   ├─ Advantages:
│   │   ├─ General: Works with any posterior (only need density evaluation)
│   │   ├─ No tuning required (basic version; adaptive versions help)
│   │   ├─ Flexible proposal (random walk most common)
│   │   └─ Diagnostics available
│   ├─ Example - logistic regression:
│   │   ├─ Posterior p(β|y) non-conjugate
│   │   ├─ MH: Propose β* ~ N(β_t, Σ_prop)
│   │   ├─ Compute log-posterior (costly: evaluate likelihood n times)
│   │   ├─ Accept/reject; repeat
│   │   └─ After convergence: Use β samples for credible intervals
│   └─ Limitations:
│       ├─ Inefficient in high dimensions (acceptance rate → 0)
│       ├─ Random walk explores O(K) steps to move distance O(1) (curse of dimension)
│       ├─ High autocorrelation: Need many samples
│       └─ Breaks down for K > 20 without tuning
├─ Hamiltonian Monte Carlo (HMC):
│   ├─ Motivation: Exploit gradient information to improve efficiency
│   │   ├─ MH: Blind random walk (no direction)
│   │   ├─ HMC: Use ∇log p(θ|y) to propose good direction
│   │   ├─ Physics analogy: Particle on energy surface with momentum
│   │   └─ Result: Fewer rejections; longer effective moves; better mixing
│   ├─ Algorithm:
│   │   ├─ Augmented space: (θ, ρ) where ρ = momentum ~ N(0, M) [mass matrix]
│   │   ├─ Hamiltonian: H(θ,ρ) = -log p(θ|y) + (1/2)ρ'M⁻¹ρ [energy]
│   │   ├─ Dynamics: dθ/dt = M⁻¹ρ, dρ/dt = ∇log p(θ|y)
│   │   ├─ Procedure (one step):
│   │   │   ├─ Sample momentum: ρ ~ N(0, M)
│   │   │   ├─ Simulate dynamics: Leapfrog integration for L steps
│   │   │   ├─ Propose: θ* = θ(T), ρ* = -ρ(T) [final state]
│   │   │   └─ Accept/reject: α = min(1, exp(-H(θ*,ρ*)+H(θ_t,ρ_t)))
│   │   └─ Energy conservation: Leapfrog preserves H → Acceptance near 100%
│   ├─ Leapfrog integrator:
│   │   ├─ Half-step momentum: ρ = ρ + (ε/2)∇log p(θ)
│   │   ├─ Full-step position: θ = θ + ε·M⁻¹ρ
│   │   ├─ Half-step momentum: ρ = ρ + (ε/2)∇log p(θ)
│   │   ├─ Parameters: Step size ε (small → accurate but slow), steps L
│   │   └─ Total trajectory length L·ε controls move distance
│   ├─ Properties:
│   │   ├─ Reversible: Running time T vs -T gives same distribution
│   │   ├─ Volume-preserving: Jacobian det = 1 (simplifies acceptance)
│   │   ├─ Near-deterministic: Mostly accept; few rejections
│   │   └─ Efficient: Fewer samples needed for convergence
│   ├─ No-U-Turn Sampler (NUTS):
│   │   ├─ Adaptation: Automatically choose L (avoid overshooting)
│   │   ├─ Criterion: Stop simulation when trajectory doubles back (U-turn)
│   │   ├─ Benefit: Optimal trajectory length without tuning
│   │   ├─ Implementation: Stan default; PyMC3 option
│   │   └─ Result: Near-automatic sampling (minimal tuning)
│   ├─ Advantages:
│   │   ├─ Efficient in high dimensions (K > 100 feasible)
│   │   ├─ High acceptance rate (~65-90%)
│   │   ├─ Low autocorrelation (effective sample size >> iterations)
│   │   ├─ Fewer tuning parameters (step size often auto-adjusted)
│   │   └─ NUTS removes manual tuning
│   └─ Example - hierarchical model:
│       ├─ Group effects μ_j ~ N(μ, σ²); likelihoods complex
│       ├─ Gradient ∇log p available (autodiff in Stan)
│       ├─ HMC proposes jumps toward high-probability region
│       ├─ Efficiency: 1000 HMC iterations ≈ 10000 MH iterations
│       └─ Practical: Stan default for complex models
├─ Diagnostics & Convergence Assessment:
│   ├─ Visual inspection:
│   │   ├─ Trace plots: Plot θ_t vs t
│   │   │   ├─ Good: Fuzzy cloud (rapid mixing)
│   │   │   ├─ Bad: Trending, stuck regions (non-convergence)
│   │   │   └─ Remedy: Longer burn-in or reparameterization
│   │   ├─ Autocorrelation: ρ(θ_t, θ_{t-k}) vs lag k
│   │   │   ├─ Fast decay → Low correlation; efficient
│   │   │   ├─ Slow decay → High correlation; wasteful
│   │   │   └─ Effective sample size: ESS = T/(1 + 2Σρ_k)
│   │   └─ Density plots: Marginal posterior shape
│   │       ├─ Bimodal → Chain may miss one mode
│   │       └─ Heavy tails → Higher variance estimate
│   ├─ Quantitative tests:
│   │   ├─ Gelman-Rubin R̂ (potential scale reduction):
│   │   │   ├─ Run M chains from different starting points
│   │   │   ├─ Compute within-chain and between-chain variance
│   │   │   ├─ R̂ = √[(n-1)/n + M·B/(M·W)] where n=chain length, B/W = variance ratio
│   │   │   ├─ R̂ < 1.01 → Converged; R̂ > 1.05 → Not converged
│   │   │   └─ Intuition: Chains indistinguishable from single long run
│   │   ├─ Effective sample size (ESS):
│   │   │   ├─ ESS = T/(1 + 2·Σ_lag ρ_lag)
│   │   │   ├─ T = total samples; ρ = autocorr
│   │   │   ├─ ESS/T = efficiency (1 = perfect; 0.1 = 10% waste)
│   │   │   └─ High-dim: ESS often 10-50% of T
│   │   └─ Geweke test:
│   │       ├─ Compare early vs late iterations (t-test)
│   │       ├─ p > 0.05 → No difference (converged)
│   │       ├─ p < 0.05 → Significant difference (not converged)
│   │       └─ Less sensitive than R̂; useful second check
│   ├─ Burn-in determination:
│   │   ├─ Discarding early samples (before convergence)
│   │   ├─ Amount: 10-50% of total samples (depends on chain mixing)
│   │   ├─ Conservative: Double burn-in; check if inference changes
│   │   └─ Modern: Automatic in Stan/PyMC (half of total by default)
│   └─ Thinning:
│       ├─ Keep every k-th sample (reduce autocorrelation)
│       ├─ Example: k=5 → Keep 20% of samples
│       ├─ Trade-off: Reduces storage/memory vs ESS overhead
│       ├─ Modern: Unnecessary (store full chain; compute ESS later)
│       └─ Benefit: Less relevant with efficient samplers (HMC, NUTS)
├─ Practical Implementation:
│   ├─ Software:
│   │   ├─ Stan: Probabilistic programming; HMC/NUTS default
│   │   │   ├─ Syntax: Specify model (data, parameters, model block)
│   │   │   ├─ Autodiff: Automatic gradient computation
│   │   │   └─ Output: Samples, diagnostics, inference summaries
│   │   ├─ PyMC3: Python; multiple samplers (NUTS, Metropolis, Slice)
│   │   ├─ JAGS: Declarative; Gibbs sampling primarily
│   │   └─ Greta: R interface; HMC via TensorFlow
│   ├─ Workflow:
│   │   ├─ Specify model (priors + likelihood)
│   │   ├─ Choose sampler (usually HMC/NUTS; defaults good)
│   │   ├─ Run chains: 2-4 chains × 2000-10000 iterations
│   │   ├─ Check convergence: R̂, trace plots, ESS
│   │   ├─ Discard burn-in; use post-convergence samples
│   │   └─ Summarize posterior (means, SD, credible intervals)
│   └─ Computational cost:
│       ├─ Depends on n (data), K (parameters), model complexity
│       ├─ Simple (logistic): Seconds-minutes (K~10)
│       ├─ Moderate (hierarchical): Minutes-hours (K~100)
│       ├─ Complex (GP, structured): Hours-days (K~1000)
│       └─ Parallel chains: Speed up via multiple cores (minimal overhead)
└─ Advanced Topics:
    ├─ Tempering (parallel tempering):
    │   ├─ Problem: Multimodal posteriors; chain can get stuck
    │   ├─ Solution: Run chains at different temperatures
    │   │   ├─ Low temperature (cold): β=1 (target posterior)
    │   │   ├─ High temperature (hot): β<1 (flatter; easier mixing)
    │   │   └─ Exchanges: Periodically swap between chains
    │   ├─ Benefit: Cold chain explores both modes
    │   ├─ Implementation: Complex; research area
    │   └─ Alternative: Reparameterization (often simpler)
    ├─ Adaptive MCMC:
    │   ├─ Learn proposal covariance during burn-in
    │   ├─ Metropolis-Hastings acceptance rate → Adjust step size
    │   ├─ Example: Too many rejections → increase step size
    │   └─ Benefit: Automatic tuning; less manual intervention
    ├─ Variational Inference (approximate alternative):
    │   ├─ Find simple distribution Q(θ) ≈ p(θ|y)
    │   ├─ Minimize KL divergence via optimization
    │   ├─ Fast: Seconds vs MCMC minutes-hours
    │   ├─ Trade-off: Approximation (less detail) vs speed
    │   └─ Hybrid: Variational initialization; refine with MCMC
    └─ Gibbs with auxiliary variables:
        ├─ Augment parameter space with latent variables
        ├─ Conditional on augmented θ: Easy sampling
        ├─ Marginalizes latent variables: Posterior over original θ
        ├─ Example: Probit regression with latent utilities
        └─ Benefit: Escape high-dimension challenges
```

**Key Insight:** MCMC converts intractable posteriors into samples; Gibbs simple but limited; MH general but inefficient high-d; HMC/NUTS efficient and scales; convergence assessment critical (R̂, ESS, traces)

## 5. Mini-Project
Compare Gibbs, Metropolis-Hastings, and HMC on mixture model:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set seed
np.random.seed(42)

# Generate data from mixture of 2 normals
n = 200
true_components = np.random.binomial(1, 0.6, n)
y = np.where(true_components==0, 
             np.random.normal(-2, 0.5, n), 
             np.random.normal(2, 0.5, n))

print("="*70)
print("MCMC Comparison: Mixture Model")
print("="*70)
print(f"Data: {n} observations from mixture")
print(f"True mixture: 60% N(-2, 0.5²), 40% N(2, 0.5²)")
print("")

# Model: y_i ~ p·N(μ₁,σ²) + (1-p)·N(μ₂,σ²)
# Unknowns: μ₁, μ₂, p

# Log-likelihood (unnormalized posterior with flat priors)
def log_posterior(mu1, mu2, p):
    if not (0 < p < 1):
        return -np.inf
    
    ll = np.sum(np.log(
        p * stats.norm.pdf(y, mu1, 1) + 
        (1-p) * stats.norm.pdf(y, mu2, 1)
    ))
    
    # Priors: μ ~ N(0,10), p ~ Beta(1,1)
    lp = stats.norm.logpdf(mu1, 0, 10) + stats.norm.logpdf(mu2, 0, 10) + stats.beta.logpdf(p, 1, 1)
    return ll + lp

# MCMC Sampling: Metropolis-Hastings
def metropolis_hastings(iterations=2000, proposal_sd=0.5):
    samples = np.zeros((iterations, 3))
    samples[0] = [-1, 1, 0.5]  # Initial values
    
    accepted = 0
    
    for t in range(1, iterations):
        # Propose
        proposal = samples[t-1] + np.random.normal(0, proposal_sd, 3)
        
        # Accept/reject
        log_alpha = log_posterior(proposal[0], proposal[1], proposal[2]) - \
                    log_posterior(samples[t-1,0], samples[t-1,1], samples[t-1,2])
        
        if np.log(np.random.uniform()) < log_alpha:
            samples[t] = proposal
            accepted += 1
        else:
            samples[t] = samples[t-1]
    
    acceptance_rate = accepted / iterations
    return samples, acceptance_rate

# Run MH
print("Metropolis-Hastings Sampling...")
mh_samples, mh_accept = metropolis_hastings(iterations=5000, proposal_sd=0.3)
print(f"  Acceptance rate: {mh_accept*100:.1f}%")

# Estimate ESS
def effective_sample_size(samples):
    """Compute effective sample size for each parameter"""
    T = len(samples)
    ess = []
    
    for i in range(samples.shape[1]):
        # Autocorrelation
        x = samples[:, i]
        x_centered = x - x.mean()
        c0 = np.dot(x_centered, x_centered) / T
        
        # Sum autocorr up to lag where <0.05
        tau = 1
        for lag in range(1, T//2):
            c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / T
            if abs(c_lag) < 0.05 * abs(c0):
                break
            tau += 2 * c_lag / c0
        
        ess_param = T / (2 * tau)
        ess.append(ess_param)
    
    return np.array(ess)

mh_ess = effective_sample_size(mh_samples[1000:])

print("\nPost-convergence summary (burn-in=1000):")
print("-"*70)
print(f"{'Parameter':<15} {'Mean':<12} {'SD':<12} {'ESS':<12} {'ESS/n':<10}")
param_names = ['μ₁', 'μ₂', 'p']
for i, name in enumerate(param_names):
    mean_val = mh_samples[1000:, i].mean()
    sd_val = mh_samples[1000:, i].std()
    ess_val = mh_ess[i]
    efficiency = ess_val / len(mh_samples[1000:])
    print(f"{name:<15} {mean_val:>10.3f}   {sd_val:>10.3f}   {ess_val:>10.0f}   {efficiency:>8.1%}")

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

# Trace plots (left column)
for i in range(3):
    axes[i, 0].plot(mh_samples[:, i], linewidth=0.5, alpha=0.7)
    axes[i, 0].axvline(1000, color='red', linestyle='--', linewidth=1)
    axes[i, 0].set_ylabel(param_names[i])
    axes[i, 0].set_title(f'Trace: {param_names[i]}' if i==0 else '')
    axes[i, 0].grid(alpha=0.3)

# Posterior density (middle column)
for i in range(3):
    axes[i, 1].hist(mh_samples[1000:, i], bins=50, density=True, alpha=0.6, color='blue')
    axes[i, 1].set_ylabel('Density')
    axes[i, 1].set_title(f'Posterior: {param_names[i]}' if i==0 else '')
    axes[i, 1].grid(alpha=0.3)

# Autocorrelation (right column)
for i in range(3):
    x = mh_samples[1000:, i]
    acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    acf = acf[acf.size//2:] / acf[acf.size//2]  # Normalize
    axes[i, 2].bar(range(min(50, len(acf))), acf[:50], alpha=0.6, color='green')
    axes[i, 2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[i, 2].set_ylabel('Autocorrelation')
    axes[i, 2].set_title(f'ACF: {param_names[i]}' if i==0 else '')
    axes[i, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mcmc_sampling.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Observations:")
print("="*70)
print("1. MH trace plots show mixture model (bimodality) in posterior")
print("   → Two modes: (μ₁, μ₂) ≈ (-2, 2) or (2, -2)")
print("")
print("2. Acceptance rate ~30% (reasonable for 3-D)")
print("   → Too low (<10%): increase proposal SD")
print("   → Too high (>50%): decrease proposal SD")
print("")
print("3. Autocorrelation slowly decays")
print("   → Effective sample size lower than nominal N")
print("   → ESS/n efficiency ~10-30%")
print("")
print("4. Solution: Run longer chains or use HMC/NUTS (higher efficiency)")
```

## 6. Challenge Round
When MCMC fails or is inefficient:
- **Poor mixing (high autocorr)**: Random walk explores slowly; try: smaller proposal, reparameterization (centered parameterization), adaptive MCMC
- **Non-convergence (R̂>1.1)**: Chain stuck in local mode; longer burn-in, multiple chains, tempering, or model reparameterization
- **Multimodal posterior**: Chain misses one mode; parallel tempering, initialization from different modes, or improve model specification
- **High rejection (MH accept <5%)**: Proposal too aggressive; reduce step size or use adaptive methods
- **Computational cost (hours)**: Large data n; use approximate methods (variational inference, stochastic gradients) or simplify model
- **Dimension explosion (K>1000)**: Random walk unfeasible; gradient-based (HMC) needed; consider dimension reduction (factor model, PCA)

## 7. Key References
- [Gelman et al: MCMC Diagnostics (2013)](https://arxiv.org/pdf/1312.0906.pdf) - Potential scale reduction, ESS computation
- [Betancourt: Generalized HMC via NUTS (2017)](https://arxiv.org/pdf/1701.02434.pdf) - HMC theoretical foundations; NUTS algorithm
- [Roberts & Rosenthal: MCMC Efficiency (2009)](https://arxiv.org/pdf/0805.4591.pdf) - Optimal proposal scaling; acceptance rates

---
**Status:** Core computational Bayes | **Complements:** Bayesian Inference, Prior Distributions, Model Comparison, Hierarchical Models
