
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

def latin_hypercube_sample(N, d, seed=None):
    """Generate N samples from d-dimensional Latin hypercube"""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize matrix: N × d
    X = np.zeros((N, d))
    
    for j in range(d):
        # Stratum indices
        strata = np.arange(N)
        # Random permutation
        perm = np.random.permutation(N)
        strata_perm = strata[perm]
        
        # Uniform sample in assigned stratum
        u = np.random.uniform(0, 1, N)
        X[:, j] = (strata_perm + u) / N
    
    return X

def standard_mc_sample(N, d, seed=None):
    """Generate N standard MC samples from [0,1]^d"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, (N, d))

# Test 1: Visual coverage comparison (2D)
print("=== 2D COVERAGE COMPARISON ===")
N = 500
d = 2

np.random.seed(42)
X_lhs = latin_hypercube_sample(N, d)
np.random.seed(42)
X_mc = standard_mc_sample(N, d)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Latin Hypercube
axes[0].scatter(X_lhs[:, 0], X_lhs[:, 1], alpha=0.5, s=10)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_aspect('equal')
axes[0].set_title('Latin Hypercube Sampling (500 samples)')
axes[0].set_xlabel('Dimension 1')
axes[0].set_ylabel('Dimension 2')
axes[0].grid(alpha=0.3)

# Standard MC
axes[1].scatter(X_mc[:, 0], X_mc[:, 1], alpha=0.5, s=10, color='orange')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_aspect('equal')
axes[1].set_title('Standard MC (500 samples)')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Test 2: Margin uniformity check
print("\nMargin Uniformity (D1 distribution):")
print(f"LHS D1 - Mean: {np.mean(X_lhs[:, 0]):.4f} (theory: 0.5)")
print(f"LHS D1 - Std: {np.std(X_lhs[:, 0]):.4f} (theory: {1/np.sqrt(12):.4f})")
print(f"MC D1 - Mean: {np.mean(X_mc[:, 0]):.4f}")
print(f"MC D1 - Std: {np.std(X_mc[:, 0]):.4f}")

# Test 3: Multidimensional test function
def test_function_nd(X):
    """Test function: product of sines"""
    return np.prod(np.sin(np.pi * X), axis=1)

def test_function_nd_smooth(X):
    """Smooth test function: exponential of sum"""
    return np.exp(-np.sum(X**2, axis=1))

# Multidimensional integration
print("\n=== MULTIDIMENSIONAL INTEGRATION ===")
for d in [2, 5, 10]:
    N = 10000
    
    np.random.seed(42)
    X_lhs = latin_hypercube_sample(N, d)
    X_lhs_normal = norm.ppf(X_lhs)  # Transform to normal
    values_lhs = test_function_nd_smooth(X_lhs_normal)
    estimate_lhs = np.mean(values_lhs)
    se_lhs = np.std(values_lhs) / np.sqrt(N)
    
    np.random.seed(42)
    X_mc = standard_mc_sample(N, d)
    X_mc_normal = norm.ppf(X_mc)
    values_mc = test_function_nd_smooth(X_mc_normal)
    estimate_mc = np.mean(values_mc)
    se_mc = np.std(values_mc) / np.sqrt(N)
    
    var_reduction = (1 - (se_lhs/se_mc)**2) * 100
    print(f"Dimension {d}:")
    print(f"  LHS SE: {se_lhs:.6f}")
    print(f"  MC SE: {se_mc:.6f}")
    print(f"  Variance Reduction: {var_reduction:.1f}%")

# Test 4: Basket option pricing with correlation
print("\n=== BASKET OPTION PRICING (2 Assets) ===")
S0 = np.array([100, 100])
K = 100
T = 1
r = 0.05
sigma = np.array([0.2, 0.2])
rho = 0.7  # Correlation
N_paths = 5000

# Correlation matrix
Sigma = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],
                  [rho*sigma[0]*sigma[1], sigma[1]**2]])
L = np.linalg.cholesky(Sigma)

# Black-Scholes approximation for reference
d1 = (np.log(S0[0]/K) + (r + 0.5*sigma[0]**2)*T) / (sigma[0]*np.sqrt(T))
d2 = d1 - sigma[0]*np.sqrt(T)
bs_price = S0[0]*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Method 1: Standard MC
print("\nStandard MC:")
np.random.seed(42)
Z_mc = np.random.normal(0, 1, (N_paths, 2))
dW = L @ Z_mc.T  # Correlate
S_final = S0[:, np.newaxis] * np.exp((r - 0.5*sigma[:, np.newaxis]**2)*T + 
                                      dW * np.sqrt(T))
basket_price_mc = np.mean(S_final[0] + S_final[1]) / 2
basket_payoff_mc = np.maximum(basket_price_mc - K, 0)
price_mc = np.exp(-r*T) * np.mean(np.maximum((S_final[0] + S_final[1])/2 - K, 0))
se_mc = np.exp(-r*T) * np.std(np.maximum((S_final[0] + S_final[1])/2 - K, 0)) / np.sqrt(N_paths)

print(f"  Price: ${price_mc:.4f}")
print(f"  SE: ${se_mc:.4f}")

# Method 2: Latin Hypercube
print("Latin Hypercube:")
np.random.seed(42)
Z_lhs = norm.ppf(latin_hypercube_sample(N_paths, 2))
dW = L @ Z_lhs.T
S_final = S0[:, np.newaxis] * np.exp((r - 0.5*sigma[:, np.newaxis]**2)*T + 
                                      dW * np.sqrt(T))
price_lhs = np.exp(-r*T) * np.mean(np.maximum((S_final[0] + S_final[1])/2 - K, 0))
se_lhs = np.exp(-r*T) * np.std(np.maximum((S_final[0] + S_final[1])/2 - K, 0)) / np.sqrt(N_paths)

print(f"  Price: ${price_lhs:.4f}")
print(f"  SE: ${se_lhs:.4f}")

var_reduction = (1 - (se_lhs/se_mc)**2) * 100
print(f"Variance Reduction: {var_reduction:.1f}%")

# Test 5: High-dimensional test (10D)
print("\n=== HIGH-DIMENSIONAL TEST (10D) ===")
d = 10
N_trials = 50
N_paths = 1000

ses_lhs = []
ses_mc = []

for trial in range(N_trials):
    X_lhs = latin_hypercube_sample(N_paths, d, seed=trial)
    X_lhs_normal = norm.ppf(X_lhs)
    values_lhs = test_function_nd_smooth(X_lhs_normal)
    se_lhs_trial = np.std(values_lhs) / np.sqrt(N_paths)
    ses_lhs.append(se_lhs_trial)
    
    X_mc = standard_mc_sample(N_paths, d, seed=trial)
    X_mc_normal = norm.ppf(X_mc)
    values_mc = test_function_nd_smooth(X_mc_normal)
    se_mc_trial = np.std(values_mc) / np.sqrt(N_paths)
    ses_mc.append(se_mc_trial)

print(f"LHS Mean SE: {np.mean(ses_lhs):.6f}")
print(f"MC Mean SE: {np.mean(ses_mc):.6f}")
print(f"Variance Reduction: {(1 - (np.mean(ses_lhs)/np.mean(ses_mc))**2)*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Margin distributions
axes[0, 0].hist(X_lhs[:, 0], bins=50, alpha=0.7, label='LHS D1', color='C0', density=True)
axes[0, 0].hist(X_mc[:, 0], bins=50, alpha=0.7, label='MC D1', color='C1', density=True)
axes[0, 0].axvline(0.5, color='r', linestyle='--', linewidth=2, label='Expected mean')
axes[0, 0].set_xlabel('Dimension 1 Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Margin Distribution Comparison (2D)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: SE across trials (10D)
axes[0, 1].plot(ses_lhs, 'o-', label='LHS', linewidth=1, markersize=4, alpha=0.7)
axes[0, 1].plot(ses_mc, 's-', label='MC', linewidth=1, markersize=4, alpha=0.7)
axes[0, 1].set_xlabel('Trial')
axes[0, 1].set_ylabel('Standard Error')
axes[0, 1].set_title('SE Comparison Across 50 Trials (10D)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: SE comparison bar chart
methods = ['LHS', 'Standard MC']
se_values = [np.mean(ses_lhs), np.mean(ses_mc)]
colors = ['C0', 'C1']

axes[1, 0].bar(methods, se_values, color=colors, alpha=0.7, width=0.5)
axes[1, 0].set_ylabel('Mean Standard Error')
axes[1, 0].set_title('SE Comparison (10D, avg over 50 trials)')
axes[1, 0].grid(alpha=0.3, axis='y')

var_red = (1 - (np.mean(ses_lhs)/np.mean(ses_mc))**2)*100
for i, se in enumerate(se_values):
    axes[1, 0].text(i, se + 0.002, f'{se:.5f}', ha='center', fontweight='bold')

axes[1, 0].text(0.5, min(se_values)*0.5, f'Reduction: {var_red:.1f}%', 
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 4: Scaled version (log-log) to show dimension scaling
dimensions = [2, 3, 5, 10]
lhs_means = []
mc_means = []

for d in dimensions:
    ses_lhs_d = []
    ses_mc_d = []
    for trial in range(10):
        X_lhs = latin_hypercube_sample(1000, d, seed=trial)
        X_lhs_normal = norm.ppf(X_lhs)
        values = test_function_nd_smooth(X_lhs_normal)
        ses_lhs_d.append(np.std(values) / np.sqrt(1000))
        
        X_mc = standard_mc_sample(1000, d, seed=trial)
        X_mc_normal = norm.ppf(X_mc)
        values = test_function_nd_smooth(X_mc_normal)
        ses_mc_d.append(np.std(values) / np.sqrt(1000))
    
    lhs_means.append(np.mean(ses_lhs_d))
    mc_means.append(np.mean(ses_mc_d))

axes[1, 1].loglog(dimensions, lhs_means, 'o-', linewidth=2, markersize=8, label='LHS')
axes[1, 1].loglog(dimensions, mc_means, 's-', linewidth=2, markersize=8, label='Standard MC')
axes[1, 1].set_xlabel('Dimension')
axes[1, 1].set_ylabel('Mean Standard Error')
axes[1, 1].set_title('SE Scaling with Dimension')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()