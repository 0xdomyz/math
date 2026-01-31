# Kullback-Leibler Divergence

## 1.1 Concept Skeleton
**Definition:** Non-symmetric measure of difference between two probability distributions P and Q  
**Purpose:** Quantify information loss, model selection, measure distribution similarity  
**Prerequisites:** Probability distributions, entropy, logarithms, expectation

## 1.2 Comparative Framing
| Measure | KL Divergence | Entropy | Mutual Information |
|---------|--------------|---------|-------------------|
| **Formula** | D_KL(P\|\|Q) = E_P[log(P/Q)] | H(P) = -E_P[log P] | I(X;Y) = D_KL(P_{X,Y}\|\|P_X⊗P_Y) |
| **Symmetry** | Not symmetric | N/A | Symmetric |
| **Meaning** | Info loss using Q instead of P | Average surprise | Shared information |

## 1.3 Examples + Counterexamples

**Simple Example:**  
P=N(0,1), Q=N(0,2): D_KL(P||Q) = 0.193 (how much info lost approximating P with Q)

**Failure Case:**  
Q(x)=0 where P(x)>0: D_KL(P||Q) = ∞ (Q cannot represent P's support)

**Edge Case:**  
P=Q: D_KL(P||Q) = 0 (no information loss, distributions identical)

## 1.4 Layer Breakdown
```
KL Divergence Framework:
├─ Definition:
│   ├─ Discrete: D_KL(P||Q) = Σ P(x)·log(P(x)/Q(x))
│   ├─ Continuous: D_KL(P||Q) = ∫ p(x)·log(p(x)/q(x)) dx
│   └─ Alternative: E_P[log P] - E_P[log Q]
├─ Properties:
│   ├─ Non-negative: D_KL(P||Q) ≥ 0
│   ├─ Zero iff P=Q (Gibbs' inequality)
│   ├─ Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
│   ├─ Not a metric: Fails triangle inequality
│   └─ Additive: D_KL(P₁×P₂||Q₁×Q₂) = D_KL(P₁||Q₁) + D_KL(P₂||Q₂)
├─ Interpretations:
│   ├─ Information gain: Using P instead of Q
│   ├─ Relative entropy: Extra bits needed
│   ├─ Expected log-likelihood ratio: E[log LR]
│   └─ Discrimination information: Ability to distinguish P from Q
├─ Special Cases:
│   ├─ Normal distributions:
│   │   D_KL(N(μ₁,σ₁²)||N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁²+(μ₁-μ₂)²)/(2σ₂²) - 1/2
│   ├─ Bernoulli:
│   │   D_KL(Ber(p)||Ber(q)) = p·log(p/q) + (1-p)·log((1-p)/(1-q))
│   └─ Categorical: Sum over all categories
├─ Applications:
│   ├─ Model Selection: AIC, BIC approximate KL divergence
│   ├─ Variational Inference: Minimize D_KL(Q||P) for approximation
│   ├─ Information Theory: Channel capacity, coding
│   ├─ Machine Learning: Loss functions, generative models
│   └─ Hypothesis Testing: Power analysis
└─ Related Concepts:
    ├─ Cross-entropy: H(P,Q) = H(P) + D_KL(P||Q)
    ├─ Jensen-Shannon divergence: Symmetric version
    └─ Bhattacharyya distance: Related divergence measure
```

## 1.5 Mini-Project
Compute and visualize KL divergence:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import kl_div

np.random.seed(42)

# Function to compute KL divergence for continuous distributions
def kl_divergence_continuous(p_dist, q_dist, x_range):
    """Numerical integration for KL divergence"""
    p_vals = p_dist.pdf(x_range)
    q_vals = q_dist.pdf(x_range)
    
    # Avoid log(0) and division by zero
    mask = (p_vals > 1e-10) & (q_vals > 1e-10)
    kl = np.sum(p_vals[mask] * np.log(p_vals[mask] / q_vals[mask])) * (x_range[1] - x_range[0])
    return kl

# Function for analytical KL between normal distributions
def kl_normal(mu1, sigma1, mu2, sigma2):
    """Analytical KL divergence: N(mu1,sigma1^2) || N(mu2,sigma2^2)"""
    return (np.log(sigma2/sigma1) + 
            (sigma1**2 + (mu1-mu2)**2) / (2*sigma2**2) - 0.5)

# Example 1: Normal distributions
mu_p, sigma_p = 0, 1
mu_q, sigma_q = 1, 1.5

P = stats.norm(mu_p, sigma_p)
Q = stats.norm(mu_q, sigma_q)

x = np.linspace(-5, 6, 1000)
kl_pq_numerical = kl_divergence_continuous(P, Q, x)
kl_pq_analytical = kl_normal(mu_p, sigma_p, mu_q, sigma_q)
kl_qp_analytical = kl_normal(mu_q, sigma_q, mu_p, sigma_p)

print("Normal Distribution Example:")
print(f"  P = N({mu_p}, {sigma_p}²)")
print(f"  Q = N({mu_q}, {sigma_q}²)")
print(f"  D_KL(P||Q) numerical: {kl_pq_numerical:.4f}")
print(f"  D_KL(P||Q) analytical: {kl_pq_analytical:.4f}")
print(f"  D_KL(Q||P) analytical: {kl_qp_analytical:.4f}")
print(f"  Asymmetry: D_KL(P||Q) ≠ D_KL(Q||P)")

# Example 2: Bernoulli distributions
def kl_bernoulli(p, q):
    """KL divergence for Bernoulli distributions"""
    if p == 0:
        return -np.log(1-q)
    if p == 1:
        return -np.log(q)
    return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

p_true = 0.3
q_approx = 0.5
print(f"\nBernoulli Example:")
print(f"  P = Ber({p_true}), Q = Ber({q_approx})")
print(f"  D_KL(P||Q): {kl_bernoulli(p_true, q_approx):.4f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Distributions P and Q
axes[0, 0].plot(x, P.pdf(x), 'b-', linewidth=2, label=f'P: N({mu_p},{sigma_p})')
axes[0, 0].plot(x, Q.pdf(x), 'r-', linewidth=2, label=f'Q: N({mu_q},{sigma_q})')
axes[0, 0].fill_between(x, 0, P.pdf(x), alpha=0.2, color='blue')
axes[0, 0].fill_between(x, 0, Q.pdf(x), alpha=0.2, color='red')
axes[0, 0].legend()
axes[0, 0].set_title('Distributions P and Q')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')
axes[0, 0].grid(True, alpha=0.3)

# 2. Log ratio log(P/Q)
log_ratio = np.log(P.pdf(x) / Q.pdf(x))
axes[0, 1].plot(x, log_ratio, 'g-', linewidth=2)
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Log Ratio: log(P(x)/Q(x))')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('log(P/Q)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Integrand: P(x) * log(P(x)/Q(x))
integrand = P.pdf(x) * log_ratio
axes[0, 2].plot(x, integrand, 'purple', linewidth=2)
axes[0, 2].fill_between(x, 0, integrand, alpha=0.3, color='purple')
axes[0, 2].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[0, 2].set_title(f'KL Integrand (Area = {kl_pq_analytical:.3f})')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('P(x)·log(P(x)/Q(x))')
axes[0, 2].grid(True, alpha=0.3)

# 4. KL divergence as function of Q's parameters
mu_q_range = np.linspace(-2, 2, 50)
sigma_q_range = np.linspace(0.5, 3, 50)
MU_Q, SIGMA_Q = np.meshgrid(mu_q_range, sigma_q_range)
KL_grid = np.zeros_like(MU_Q)

for i in range(len(mu_q_range)):
    for j in range(len(sigma_q_range)):
        KL_grid[j, i] = kl_normal(mu_p, sigma_p, MU_Q[j, i], SIGMA_Q[j, i])

contour = axes[1, 0].contourf(MU_Q, SIGMA_Q, KL_grid, levels=20, cmap='viridis')
axes[1, 0].contour(MU_Q, SIGMA_Q, KL_grid, levels=10, colors='white', alpha=0.3)
axes[1, 0].plot(mu_p, sigma_p, 'r*', markersize=15, label='True P')
axes[1, 0].set_xlabel('μ_Q')
axes[1, 0].set_ylabel('σ_Q')
axes[1, 0].set_title('D_KL(P||Q) as function of Q parameters')
axes[1, 0].legend()
plt.colorbar(contour, ax=axes[1, 0])

# 5. Asymmetry visualization
mu_range = np.linspace(-2, 2, 100)
kl_pq = [kl_normal(mu_p, sigma_p, mu, sigma_q) for mu in mu_range]
kl_qp = [kl_normal(mu, sigma_q, mu_p, sigma_p) for mu in mu_range]

axes[1, 1].plot(mu_range, kl_pq, 'b-', linewidth=2, label='D_KL(P||Q)')
axes[1, 1].plot(mu_range, kl_qp, 'r-', linewidth=2, label='D_KL(Q||P)')
axes[1, 1].axvline(mu_p, color='g', linestyle='--', alpha=0.5, label='μ_P')
axes[1, 1].legend()
axes[1, 1].set_xlabel('μ_Q')
axes[1, 1].set_ylabel('KL Divergence')
axes[1, 1].set_title('Asymmetry of KL Divergence')
axes[1, 1].grid(True, alpha=0.3)

# 6. Bernoulli KL divergence heatmap
p_range = np.linspace(0.01, 0.99, 100)
q_range = np.linspace(0.01, 0.99, 100)
P_grid, Q_grid = np.meshgrid(p_range, q_range)
KL_bernoulli = np.zeros_like(P_grid)

for i in range(len(p_range)):
    for j in range(len(q_range)):
        KL_bernoulli[j, i] = kl_bernoulli(P_grid[j, i], Q_grid[j, i])

# Cap extremely large values for visualization
KL_bernoulli_capped = np.minimum(KL_bernoulli, 5)

contour2 = axes[1, 2].contourf(P_grid, Q_grid, KL_bernoulli_capped, 
                                levels=20, cmap='plasma')
axes[1, 2].plot([0, 1], [0, 1], 'w--', linewidth=2, label='P=Q (KL=0)')
axes[1, 2].set_xlabel('p (True)')
axes[1, 2].set_ylabel('q (Approximation)')
axes[1, 2].set_title('D_KL(Ber(p)||Ber(q))')
axes[1, 2].legend()
plt.colorbar(contour2, ax=axes[1, 2])

plt.tight_layout()
plt.show()

# Application: Model Selection
print("\nModel Selection Application:")
print("  Given data from P, which Q approximates better?")
data = np.random.normal(mu_p, sigma_p, 1000)

Q1 = stats.norm(0.5, 1)
Q2 = stats.norm(0, 2)

# Approximate KL by average log-likelihood difference
kl_approx_q1 = np.mean(P.logpdf(data) - Q1.logpdf(data))
kl_approx_q2 = np.mean(P.logpdf(data) - Q2.logpdf(data))

print(f"  Q1 = N(0.5, 1): KL ≈ {kl_approx_q1:.4f}")
print(f"  Q2 = N(0, 2): KL ≈ {kl_approx_q2:.4f}")
print(f"  Better model: {'Q1' if kl_approx_q1 < kl_approx_q2 else 'Q2'} (lower KL)")
```

## 1.6 Challenge Round
When is KL divergence the wrong choice?
- **Need symmetry**: Use Jensen-Shannon divergence or symmetric KL (mean of both directions)
- **Need metric properties**: Use total variation distance or Wasserstein distance
- **Q has narrower support than P**: KL divergence infinite; use reverse KL or other divergences
- **Robust comparison**: KL sensitive to tail behavior; consider χ² divergence or Hellinger distance
- **Multimodal distributions**: Single KL value may not capture mode-matching issues

## 1.7 Key References
- [Wikipedia - Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- [Cover & Thomas - Elements of Information Theory](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959)
- [KL Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
- Thinking: Measures "surprise" when using Q instead of true P; Not symmetric (forward vs reverse KL); Zero only when distributions identical; Core to information theory and ML

---
**Status:** Fundamental divergence measure | **Complements:** Entropy, Information Theory, Model Selection, Variational Inference
