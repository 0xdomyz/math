
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# Black-Scholes
def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Parameters
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
N_paths = 5000
dt = T / 252
n_steps = int(T / dt)

# Theoretical moments of standard normal (shocks)
theo_mean = 0
theo_var = 1
theo_skew = 0
theo_kurtosis = 3  # Excess kurtosis = 0, so kurtosis = 3

# Method 1: Standard Monte Carlo
print("=== STANDARD MONTE CARLO ===")
np.random.seed(42)
prices_standard = []
for trial in range(50):
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    asian_payoff = np.maximum(np.mean(S, axis=1) - K, 0)
    price = np.exp(-r*T) * np.mean(asian_payoff)
    prices_standard.append(price)

# Method 2: Moment Matching (first 2 moments only)
print("=== MOMENT MATCHING (Mean & Variance) ===")
np.random.seed(42)
prices_mm2 = []
for trial in range(50):
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    
    # Adjust each timestep's shocks
    Z_adjusted = np.zeros_like(Z)
    for t in range(n_steps):
        Z_t = Z[:, t]
        # Center
        Z_centered = Z_t - np.mean(Z_t)
        # Scale
        Z_adjusted[:, t] = Z_centered * (np.sqrt(theo_var) / np.std(Z_centered)) + theo_mean
    
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_adjusted
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    asian_payoff = np.maximum(np.mean(S, axis=1) - K, 0)
    price = np.exp(-r*T) * np.mean(asian_payoff)
    prices_mm2.append(price)

# Method 3: Moment Matching (first 4 moments - Cornish-Fisher)
print("=== MOMENT MATCHING (Mean, Variance, Skew, Kurtosis) ===")
np.random.seed(42)
prices_mm4 = []
for trial in range(50):
    Z = np.random.normal(0, 1, (N_paths, n_steps))
    
    Z_adjusted = np.zeros_like(Z)
    for t in range(n_steps):
        Z_t = Z[:, t]
        
        # Standardize to (0,1)
        Z_std = (Z_t - np.mean(Z_t)) / np.std(Z_t)
        
        # Cornish-Fisher adjustment for skewness and kurtosis
        G3 = skew(Z_std)
        G4 = kurtosis(Z_std, fisher=True)  # Excess kurtosis
        
        # Cornish-Fisher transform
        w = Z_std + (G3/6)*Z_std**2 + (G4/24)*(Z_std**3 - 3*Z_std)
        w = w - (G3**2/36)*(2*Z_std**3 - 5*Z_std)
        
        # Scale and center to match theory
        Z_adjusted[:, t] = (w - np.mean(w)) / np.std(w)
    
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_adjusted
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    asian_payoff = np.maximum(np.mean(S, axis=1) - K, 0)
    price = np.exp(-r*T) * np.mean(asian_payoff)
    prices_mm4.append(price)

# Comparison
print(f"\nResults (50 trials, {N_paths} paths each):")
print(f"Standard MC:")
print(f"  Mean: ${np.mean(prices_standard):.6f}")
print(f"  Std Dev: ${np.std(prices_standard):.6f}")

print(f"Moment Matching (2 moments):")
print(f"  Mean: ${np.mean(prices_mm2):.6f}")
print(f"  Std Dev: ${np.std(prices_mm2):.6f}")
print(f"  Variance Reduction: {(1 - (np.std(prices_mm2)/np.std(prices_standard))**2)*100:.1f}%")

print(f"Moment Matching (4 moments):")
print(f"  Mean: ${np.mean(prices_mm4):.6f}")
print(f"  Std Dev: ${np.std(prices_mm4):.6f}")
print(f"  Variance Reduction: {(1 - (np.std(prices_mm4)/np.std(prices_standard))**2)*100:.1f}%")

# Distribution analysis: single trial
print("\n=== DISTRIBUTION ANALYSIS (Single Trial) ===")
np.random.seed(42)
Z = np.random.normal(0, 1, (N_paths, n_steps))
Z_mm2 = np.zeros_like(Z)
Z_mm4 = np.zeros_like(Z)

for t in range(n_steps):
    # MM2
    Z_mm2[:, t] = (Z[:, t] - np.mean(Z[:, t])) / np.std(Z[:, t])
    
    # MM4
    Z_std = (Z[:, t] - np.mean(Z[:, t])) / np.std(Z[:, t])
    G3 = skew(Z_std)
    G4 = kurtosis(Z_std, fisher=True)
    w = Z_std + (G3/6)*Z_std**2 + (G4/24)*(Z_std**3 - 3*Z_std)
    w = w - (G3**2/36)*(2*Z_std**3 - 5*Z_std)
    Z_mm4[:, t] = (w - np.mean(w)) / np.std(w)

print(f"Sample Statistics (first timestep):")
print(f"Standard MC:")
print(f"  Mean: {np.mean(Z[:, 0]):.6f} (theory: 0)")
print(f"  Variance: {np.var(Z[:, 0]):.6f} (theory: 1)")
print(f"  Skewness: {skew(Z[:, 0]):.6f} (theory: 0)")
print(f"  Kurtosis: {kurtosis(Z[:, 0], fisher=True):.6f} (theory: 0)")

print(f"MM2:")
print(f"  Mean: {np.mean(Z_mm2[:, 0]):.6f}")
print(f"  Variance: {np.var(Z_mm2[:, 0]):.6f}")
print(f"  Skewness: {skew(Z_mm2[:, 0]):.6f}")

print(f"MM4:")
print(f"  Mean: {np.mean(Z_mm4[:, 0]):.6f}")
print(f"  Variance: {np.var(Z_mm4[:, 0]):.6f}")
print(f"  Skewness: {skew(Z_mm4[:, 0]):.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Price estimates distribution
axes[0, 0].hist(prices_standard, bins=15, alpha=0.6, label='Standard', color='C0', density=True)
axes[0, 0].hist(prices_mm2, bins=15, alpha=0.6, label='MM2', color='C1', density=True)
axes[0, 0].hist(prices_mm4, bins=15, alpha=0.6, label='MM4', color='C2', density=True)
axes[0, 0].set_xlabel('Asian Call Price ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Price Estimates (50 trials)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Convergence
methods = ['Standard', 'MM2', 'MM4']
std_devs = [np.std(prices_standard), np.std(prices_mm2), np.std(prices_mm4)]
colors = ['C0', 'C1', 'C2']

axes[0, 1].bar(methods, std_devs, color=colors, alpha=0.7, width=0.5)
axes[0, 1].set_ylabel('Standard Deviation ($)')
axes[0, 1].set_title('Standard Deviation of Estimates')
axes[0, 1].grid(alpha=0.3, axis='y')

for i, sd in enumerate(std_devs):
    axes[0, 1].text(i, sd + 0.0005, f'${sd:.5f}', ha='center', fontweight='bold')

# Plot 3: Normal QQ plot (standard MC)
from scipy import stats
stats.probplot(Z[:, 0], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Standard MC - Normal Q-Q Plot (first timestep)')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Normal QQ plot (MM4)
stats.probplot(Z_mm4[:, 0], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('MM4 - Normal Q-Q Plot (first timestep)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()