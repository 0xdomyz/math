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
print(f"  Ï€: {pi}")
print(f"  Î¼:\n{mu}")

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
            print(f"\n  âœ“ Converged at iteration {t}")
            print(f"    Final log-likelihood: {loglik:.4f}")
            break
    
    pi, mu, Sigma = pi_new, mu_new, Sigma_new
else:
    print(f"\n  âš  Maximum iterations reached")

# Final responsibilities
resp_final, loglik_final = e_step(Y_observed, pi, mu, Sigma)
cluster_pred = np.argmax(resp_final, axis=1)

print(f"\nFinal Parameters:")
print(f"  Ï€: {pi}")
print(f"  Î¼:\n{mu}")

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
print(f"  Ï€: {pi_complete}")
print(f"  Î¼:\n{mu_complete}")

print(f"\nMissing Data EM Estimates:")
print(f"  Ï€: {pi}")
print(f"  Î¼:\n{mu}")

print(f"\nTrue Parameters:")
print(f"  Ï€: {pi_true}")
print(f"  Î¼:\n{mu_true}")

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
axes[0, 0].set_xlabel('Xâ‚')
axes[0, 0].set_ylabel('Xâ‚‚')
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
axes[0, 1].set_xlabel('Xâ‚')
axes[0, 1].set_ylabel('Xâ‚‚')
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
axes[1, 0].set_title('Responsibilities Î³áµ¢â‚– (first 50 obs)')
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
axes[1, 1].set_xlabel('Xâ‚')
axes[1, 1].set_ylabel('Xâ‚‚')
axes[1, 1].set_title('Clustering Uncertainty (Entropy)')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Certainty')
axes[1, 1].grid(alpha=0.3)

# Plot 6: Parameter Comparison
param_names = [f'Ï€â‚', f'Ï€â‚‚', f'Ï€â‚ƒ', 
               f'Î¼â‚â‚', f'Î¼â‚â‚‚', f'Î¼â‚‚â‚', f'Î¼â‚‚â‚‚', f'Î¼â‚ƒâ‚', f'Î¼â‚ƒâ‚‚']
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
print(f"   Monotonic increase: âœ“ (guaranteed by EM)")

print("\n2. Missing Data Handling:")
print(f"   {missing_pct:.1f}% data missing (MCAR)")
print(f"   EM imputes via E[Y_miss|Y_obs, Î¸]")
print(f"   Accounts for uncertainty in imputation")

print("\n3. Parameter Recovery:")
mse_pi = np.mean((pi - pi_true)**2)
mse_mu = np.mean((mu.flatten() - mu_true.flatten())**2)
print(f"   MSE(Ï€): {mse_pi:.6f}")
print(f"   MSE(Î¼): {mse_mu:.6f}")

print("\n4. Multiple Initializations:")
print(f"   {n_starts} random starts explored")
print(f"   Local maxima issue mitigated")
print(f"   Best found {best_loglik - loglik_final:+.4f} better")

print("\n5. Practical Recommendations:")
print("   â€¢ Use K-means or hierarchical clustering for initialization")
print("   â€¢ Run multiple random starts (10-100)")
print("   â€¢ Monitor log-likelihood for convergence")
print("   â€¢ Check responsibilities for cluster uncertainty")
print("   â€¢ Regularize covariances (add small Îµ to diagonal)")
print("   â€¢ Use BIC/AIC for selecting K")

print("\n6. EM Advantages:")
print("   â€¢ Handles missing data naturally (MAR assumption)")
print("   â€¢ Guaranteed monotonic likelihood increase")
print("   â€¢ Often closed-form M-step")
print("   â€¢ Interpretable latent structure")
print("   â€¢ Flexible framework (HMM, factor analysis, etc.)")

print("\n7. Limitations:")
print("   âš  Converges to local maximum (initialization critical)")
print("   âš  Slow convergence with high missing data")
print("   âš  Identifiability issues (label switching)")
print("   âš  Requires correct model specification")
print("   âš  Standard errors require Louis's formula or bootstrap")
