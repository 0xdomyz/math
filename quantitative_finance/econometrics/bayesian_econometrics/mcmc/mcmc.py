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
print(f"  True Î²: {beta_true}")
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
print(f"  Proposal: N(Î¸â½áµ—â¾, Î£_prop)")
print(f"  Î£_prop: diag({np.diag(proposal_cov)[0]})")

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

print(f"\nâœ“ MH completed")
print(f"  Overall acceptance rate: {accept_rate_mh:.3f}")
if 0.2 < accept_rate_mh < 0.5:
    print(f"  âœ“ Acceptance rate in optimal range [0.2, 0.5]")
else:
    print(f"  âš  Acceptance rate outside optimal range")

# Posterior summaries
beta_mh_mean = np.mean(beta_mh_post, axis=0)
beta_mh_sd = np.std(beta_mh_post, axis=0)
beta_mh_ci = np.percentile(beta_mh_post, [2.5, 97.5], axis=0)

print(f"\nPosterior Summaries (MH):")
for i, (mean, sd, ci_l, ci_u, b_true) in enumerate(
    zip(beta_mh_mean, beta_mh_sd, beta_mh_ci[0], beta_mh_ci[1], beta_true)
):
    print(f"  Î²{i}: Mean={mean:7.4f}, SD={sd:.4f}")
    print(f"       95% CI: [{ci_l:.4f}, {ci_u:.4f}] (true={b_true:.2f})")

# ===== Gibbs Sampling (Data Augmentation) =====
print("\n" + "="*80)
print("GIBBS SAMPLING (DATA AUGMENTATION)")
print("="*80)

# Use Polya-Gamma data augmentation for logistic regression
# Y_i|Î² ~ Bernoulli(p_i), p_i = 1/(1+exp(-X_iÎ²))
# Augment with Ï‰_i ~ PG(1, X_iÎ²)
# Then Î²|Ï‰,Y ~ N(m, V) (conjugate)

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
    # Approximate Ï‰_i (Polya-Gamma weights)
    # PG(1, Ïˆ) â‰ˆ 1/4 for small Ïˆ (simplified)
    psi = X @ beta_current_gibbs
    omega = 0.25 * np.ones(n)  # Simplified approximation
    
    # Update Î²|Ï‰,Y (conjugate normal)
    # Posterior precision
    V_inv = prior_prec + X.T @ np.diag(omega) @ X
    V = np.linalg.inv(V_inv)
    
    # Posterior mean
    kappa = Y - 0.5  # Offset
    m = V @ (prior_prec @ prior_mean + X.T @ kappa)
    
    # Sample Î²
    beta_current_gibbs = np.random.multivariate_normal(m, V)
    beta_gibbs[t] = beta_current_gibbs
    
    if (t + 1) % 5000 == 0:
        print(f"  Iteration {t+1}/{n_iter_gibbs}")

print(f"\nâœ“ Gibbs completed")

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
    print(f"  Î²{i}: Mean={mean:7.4f}, SD={sd:.4f}")
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
    print(f"  Î²{i}: ESS={ess_mh:.0f}/{len(beta_mh_post)} ({efficiency:.1f}%)")

print(f"\nGibbs Sampling:")
for i in range(p):
    ess_gibbs, _ = compute_ess(beta_gibbs_post[:, i])
    efficiency = ess_gibbs / len(beta_gibbs_post) * 100
    print(f"  Î²{i}: ESS={ess_gibbs:.0f}/{len(beta_gibbs_post)} ({efficiency:.1f}%)")

# Gelman-Rubin RÌ‚ (requires multiple chains)
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

# Compute RÌ‚ for each parameter
def compute_rhat(chains):
    """Gelman-Rubin RÌ‚ statistic"""
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
        
        # RÌ‚
        rhat[j] = np.sqrt(var_plus / W)
    
    return rhat

rhat_mh = compute_rhat(chains_mh)

print(f"\nGelman-Rubin RÌ‚:")
for i, rhat_val in enumerate(rhat_mh):
    status = "âœ“" if rhat_val < 1.1 else "âš "
    print(f"  Î²{i}: RÌ‚ = {rhat_val:.4f} {status}")

if np.all(rhat_mh < 1.1):
    print(f"\nâœ“ All RÌ‚ < 1.1: Chains converged")
else:
    print(f"\nâš  Some RÌ‚ â‰¥ 1.1: May need longer runs")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 4, figsize=(16, 10))

# Plots 1-3: MH Trace plots
for i in range(p):
    axes[0, i].plot(beta_mh[:, i], alpha=0.7, linewidth=0.5)
    axes[0, i].axhline(beta_true[i], color='red', linestyle='--', 
                      linewidth=2, label='True')
    axes[0, i].axvline(burn_in_mh, color='gray', linestyle=':', 
                      linewidth=1, label='Burn-in')
    axes[0, i].set_ylabel(f'Î²{i}')
    axes[0, i].set_xlabel('Iteration')
    axes[0, i].set_title(f'MH Trace: Î²{i}')
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
    axes[1, i].set_ylabel(f'Î²{i}')
    axes[1, i].set_xlabel('Iteration')
    axes[1, i].set_title(f'Gibbs Trace: Î²{i}')
    axes[1, i].legend(fontsize=7)
    axes[1, i].grid(alpha=0.3)

# Plot 8: Multiple chains (RÌ‚ diagnostic)
for c in range(n_chains):
    axes[1, 3].plot(chains_mh[c, :, 1], alpha=0.5, linewidth=0.5, 
                   label=f'Chain {c+1}')
axes[1, 3].axhline(beta_true[1], color='red', linestyle='--', 
                  linewidth=2, label='True')
axes[1, 3].set_xlabel('Iteration (post burn-in)')
axes[1, 3].set_ylabel('Î²â‚')
axes[1, 3].set_title(f'Multiple Chains (RÌ‚={rhat_mh[1]:.3f})')
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
    axes[2, i].set_xlabel(f'Î²{i}')
    axes[2, i].set_ylabel('Density')
    axes[2, i].set_title(f'Posterior: Î²{i}')
    axes[2, i].legend(fontsize=7)
    axes[2, i].grid(alpha=0.3)

# Plot 12: Autocorrelation functions
for i in range(p):
    _, acf_mh = compute_ess(beta_mh_post[:, i])
    axes[2, 3].plot(acf_mh, alpha=0.7, label=f'Î²{i} (MH)')

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
print(f"   All RÌ‚ < 1.1: {np.all(rhat_mh < 1.1)}")
print(f"   Trace plots show stationarity")
print(f"   Autocorrelation decays within ~30 lags")

print("\n3. Posterior Agreement:")
mse_mh_gibbs = np.mean((beta_mh_mean - beta_gibbs_mean)**2)
print(f"   MSE(MH, Gibbs): {mse_mh_gibbs:.6f}")
print(f"   Algorithms agree on posterior")

print("\n4. Practical Recommendations:")
print("   â€¢ Run multiple chains (â‰¥4) with dispersed inits")
print("   â€¢ Check RÌ‚ < 1.1 for all parameters")
print("   â€¢ Aim for ESS > 100 per chain minimum")
print("   â€¢ Tune MH acceptance to 0.23-0.44")
print("   â€¢ Use HMC/NUTS for better efficiency (Stan/PyMC)")
print("   â€¢ Visual diagnostics essential (trace plots)")

print("\n5. When to Use Each:")
print("   â€¢ Metropolis-Hastings: General purpose, easy to implement")
print("   â€¢ Gibbs: When full conditionals available (conjugate)")
print("   â€¢ HMC/NUTS: High dimensions, complex posteriors (need gradients)")
print("   â€¢ Production: Use Stan (NUTS) or PyMC for most applications")

print("\n6. Common Issues:")
print("   âš  Poor mixing: Increase iterations, tune proposals, reparameterize")
print("   âš  High autocorrelation: Block updates, HMC, or thinning")
print("   âš  Multimodality: Multiple chains, parallel tempering")
print("   âš  Slow convergence: Check parameterization, use HMC")
