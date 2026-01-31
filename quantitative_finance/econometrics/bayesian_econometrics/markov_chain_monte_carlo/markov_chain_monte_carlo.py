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
print(f"True mixture: 60% N(-2, 0.5Â²), 40% N(2, 0.5Â²)")
print("")

# Model: y_i ~ pÂ·N(Î¼â‚,ÏƒÂ²) + (1-p)Â·N(Î¼â‚‚,ÏƒÂ²)
# Unknowns: Î¼â‚, Î¼â‚‚, p

# Log-likelihood (unnormalized posterior with flat priors)
def log_posterior(mu1, mu2, p):
    if not (0 < p < 1):
        return -np.inf
    
    ll = np.sum(np.log(
        p * stats.norm.pdf(y, mu1, 1) + 
        (1-p) * stats.norm.pdf(y, mu2, 1)
    ))
    
    # Priors: Î¼ ~ N(0,10), p ~ Beta(1,1)
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
param_names = ['Î¼â‚', 'Î¼â‚‚', 'p']
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
print("   â†’ Two modes: (Î¼â‚, Î¼â‚‚) â‰ˆ (-2, 2) or (2, -2)")
print("")
print("2. Acceptance rate ~30% (reasonable for 3-D)")
print("   â†’ Too low (<10%): increase proposal SD")
print("   â†’ Too high (>50%): decrease proposal SD")
print("")
print("3. Autocorrelation slowly decays")
print("   â†’ Effective sample size lower than nominal N")
print("   â†’ ESS/n efficiency ~10-30%")
print("")
print("4. Solution: Run longer chains or use HMC/NUTS (higher efficiency)")
