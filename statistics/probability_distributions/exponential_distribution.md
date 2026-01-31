# Exponential Distribution

## 1. Concept Skeleton
**Definition:** Continuous probability distribution modeling time until next event in Poisson process; f(x) = λ × e^(-λx) for x≥0  
**Purpose:** Model waiting times, lifetimes, inter-arrival intervals; foundation for reliability engineering; inverse relationship with Poisson event counts  
**Prerequisites:** Poisson process, continuous distributions, probability density functions

## 2. Comparative Framing
| Property | Exponential | Poisson | Normal | Gamma |
|----------|-------------|---------|--------|-------|
| **Type** | Continuous | Discrete | Continuous | Continuous |
| **Parameter** | λ (rate) | λ (rate) | μ, σ | α (shape), β (rate) |
| **Support** | [0, ∞) | 0,1,2,... | (-∞,+∞) | [0, ∞) |
| **Relation** | Inter-arrival time | Event count | General | Exponential is Gamma(1,λ) |
| **Mean** | 1/λ | λ | μ | α/β |
| **Key Property** | Memoryless | Equidispersion | Symmetric | Right-skewed |
| **Use** | Failure times | Rare events | Natural phenomena | Generalized lifetimes |

## 3. Examples + Counterexamples

**Simple Example:**  
Website uptime: Average time to failure λ=1/200 hours (mean=200 hours). What's P(system lasts > 150 hours)? P(X > 150) = e^(-150/200) ≈ 0.472 or 47.2%.

**Perfect Fit:**  
Call center waiting time: Arrivals Poisson, service times independent. Inter-arrival times ~ Exponential. Waiting until next call ~ Exponential.

**Memoryless Property:**  
Light bulb: Already burned 40 hours. P(burn 30+ more) = P(burn 30+ from start). Past history irrelevant. Hazard rate constant (unlike human lifespans).

**Poor Fit:**  
Human lifespan: Not exponential (hazard rate increases with age). Bathtub curve (high early infant mortality, constant middle, increasing elderly) → Weibull better.

**Edge Case:**  
λ→0: Very long average waiting time; distribution flattens. λ→∞: Very short average; distribution concentrates near 0.

## 4. Layer Breakdown
```
Exponential Distribution:

├─ Probability Density Function (PDF):
│  ├─ f(x) = λ × e^(-λx) for x ≥ 0
│  ├─ λ > 0 is the rate parameter
│  ├─ f(0) = λ (maximum)
│  ├─ f(∞) → 0 (asymptotic decay)
│  └─ ∫₀^∞ f(x)dx = 1 (total probability)
├─ Cumulative Distribution Function (CDF):
│  ├─ F(x) = P(X ≤ x) = 1 - e^(-λx)
│  ├─ Closed form (unlike many distributions)
│  ├─ F(0) = 0 (no probability at origin)
│  ├─ F(∞) = 1 (all probability eventually)
│  └─ Survival Function: S(x) = 1 - F(x) = e^(-λx)
├─ Parameters and Properties:
│  ├─ Single parameter: λ (rate, inverse of mean)
│  ├─ Alternative form: θ = 1/λ (mean)
│  ├─ Mean: E[X] = 1/λ
│  ├─ Median: ln(2)/λ ≈ 0.693/λ
│  ├─ Mode: 0 (always at origin)
│  ├─ Variance: Var(X) = 1/λ²
│  ├─ SD: σ = 1/λ (equals mean)
│  ├─ Coefficient of Variation: CV = 1 (high variability)
│  ├─ Skewness: 2 (right-skewed)
│  ├─ Kurtosis: 6 (heavy right tail)
│  └─ MGF: M(t) = λ/(λ-t) for t < λ
├─ Memoryless Property:
│  ├─ P(X > t+s | X > t) = P(X > s) for all t, s ≥ 0
│  ├─ Intuition: Waiting additional time independent of elapsed
│  ├─ Mathematically: P(X > t+s, X > t) / P(X > t)
│  │   = P(X > t+s) / P(X > t)
│  │   = e^(-λ(t+s)) / e^(-λt)
│  │   = e^(-λs) = P(X > s)
│  ├─ Only continuous distribution with this property
│  └─ Implies constant hazard rate
├─ Hazard Rate (Failure Rate):
│  ├─ h(x) = f(x) / [1 - F(x)] = λ (constant)
│  ├─ Probability of failure in next infinitesimal interval
│  ├─ Constant across all ages → no aging/wear-out
│  ├─ Contrast with Weibull (increasing/decreasing hazard)
│  └─ Used in reliability engineering
├─ Relationship to Other Distributions:
│  ├─ Inverse of Poisson: If events ~ Poisson(λ), inter-arrival ~ Exp(λ)
│  ├─ Special case of Gamma: Gamma(1, λ)
│  ├─ Sum of n exponentials: Gamma(n, λ)
│  ├─ Minimum of n exponentials: Exp(nλ)
│  └─ Standardization: Let Z = λX, then Z ~ Exp(1)
├─ Inference on λ:
│  ├─ Point Estimate: λ̂ = n / Σxᵢ (inverse of sample mean)
│  ├─ MLE property: Unbiased asymptotically
│  ├─ Confidence Interval: Based on chi-square distribution
│  ├─ Hypothesis Test: H₀: λ=λ₀ vs H₁: λ≠λ₀
│  └─ Test Statistic: 2λ₀ Σxᵢ ~ χ²(2n)
└─ Simulation and Modeling:
   ├─ Generate: X = -ln(U) / λ where U ~ Uniform(0,1)
   ├─ Service times in queuing models
   ├─ Equipment failure times
   ├─ Radioactive decay waiting times
   └─ Survival analysis with no censoring
```

**Interaction:** Rate λ inversely related to mean waiting time. Higher λ (more frequent events) → shorter average wait.

## 5. Mini-Project
Exponential distribution in reliability and Poisson processes:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, chi2, gamma, poisson
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EXPONENTIAL DISTRIBUTION EXPLORATION")
print("="*60)

# Scenario 1: Effect of λ on distribution shape
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

lambdas = [0.5, 1, 2, 5, 10, 20]

for idx, lam in enumerate(lambdas):
    ax = axes[idx // 3, idx % 3]
    
    # PDF and CDF
    x_range = np.linspace(0, 5/lam, 200)  # Plot to roughly 5 mean values
    pdf = expon.pdf(x_range, scale=1/lam)  # scale = 1/λ
    
    # Statistics
    mean = 1 / lam
    variance = 1 / (lam**2)
    std = 1 / lam
    
    # Plot
    ax.plot(x_range, pdf, 'b-', linewidth=2.5, label=f'PDF λ={lam}')
    ax.fill_between(x_range, pdf, alpha=0.3)
    ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean=1/λ={mean:.2f}')
    ax.axvline(mean + std, color='g', linestyle=':', alpha=0.7, label=f'Mean+σ')
    
    ax.set_title(f'Exponential(λ={lam})\nMean={mean:.3f}, Var={variance:.3f}', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)

plt.tight_layout()
plt.show()

# Scenario 2: Reliability/Lifetime modeling
print("\n" + "="*60)
print("RELIABILITY: Light Bulb Lifetime")
print("="*60)

# Average lifetime: 1000 hours
lambda_bulb = 1/1000  # rate parameter

times_of_interest = [500, 1000, 2000]
print(f"\nLamp average lifetime: {1/lambda_bulb:.0f} hours")
print(f"Rate parameter λ: {lambda_bulb:.6f} per hour")

for t in times_of_interest:
    survival_prob = expon.sf(t, scale=1/lambda_bulb)  # sf = survival function
    failure_prob = expon.cdf(t, scale=1/lambda_bulb)  # cdf = failure probability
    print(f"\nAt t={t} hours:")
    print(f"  P(Bulb survives > {t}h) = {survival_prob:.4f} ({survival_prob*100:.2f}%)")
    print(f"  P(Bulb fails by {t}h) = {failure_prob:.4f} ({failure_prob*100:.2f}%)")

# Memoryless property
print(f"\nMemoryless Property:")
t1, t2 = 500, 1000
prob1 = expon.sf(t1 + t2, scale=1/lambda_bulb)  # Wait 1500 from start
prob2 = expon.sf(t2, scale=1/lambda_bulb)  # Wait 1000 from after 500
print(f"  P(bulb > 1500h from start) = {prob1:.4f}")
print(f"  P(bulb > 1000h, given already 500h) = {prob2:.4f}")
print(f"  Equal? {np.isclose(prob1, prob2)} (difference: {abs(prob1-prob2):.10f})")

# Scenario 3: Queue waiting times
print("\n" + "="*60)
print("QUEUE: Customer Service Times")
print("="*60)

# Average service time: 3 minutes
lambda_service = 1/3  # per minute

# Simulate service times
n_customers = 100
service_times = np.random.exponential(scale=1/lambda_service, size=n_customers)

print(f"\nSimulated {n_customers} customers")
print(f"Average service time: {service_times.mean():.2f} minutes")
print(f"Theoretical: {1/lambda_service:.2f} minutes")
print(f"Median service time: {np.median(service_times):.2f} minutes (theoretical: {np.log(2)/lambda_service:.2f})")
print(f"90th percentile: {np.percentile(service_times, 90):.2f} minutes")

# Scenario 4: Poisson-Exponential duality
print("\n" + "="*60)
print("POISSON-EXPONENTIAL DUALITY")
print("="*60)

# Generate Poisson events with λ=2 per hour
lambda_arrivals = 2
hours_simulated = 50
arrivals_per_hour = np.random.poisson(lambda_arrivals, hours_simulated)

# Calculate inter-arrival times
total_arrivals = arrivals_per_hour.sum()
inter_arrivals = np.random.exponential(scale=1/lambda_arrivals, size=total_arrivals)

print(f"\nPoissonProcess with λ={lambda_arrivals} arrivals/hour:")
print(f"Simulated {hours_simulated} hours")
print(f"Total arrivals: {total_arrivals}")
print(f"Average arrivals/hour: {total_arrivals/hours_simulated:.2f}")
print(f"\nInter-arrival times:")
print(f"  Mean: {inter_arrivals.mean():.4f} hours")
print(f"  Theoretical (1/λ): {1/lambda_arrivals:.4f} hours")

# Scenario 5: Hazard rate constancy
print("\n" + "="*60)
print("HAZARD RATE (Constant vs Increasing)")
print("="*60)

# Exponential: constant hazard
x_values = np.linspace(0, 5, 100)
hazard_exp = lambda_bulb * np.ones_like(x_values)

# Weibull (k=2, scale): increasing hazard (aging)
k_shape = 2
scale_weibull = 1/lambda_bulb
hazard_weibull = (k_shape / scale_weibull) * (x_values / scale_weibull)**(k_shape - 1)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: PDFs for different λ
ax = axes[0, 0]
for lam in [0.5, 1, 2, 5]:
    x = np.linspace(0, 3, 200)
    ax.plot(x, expon.pdf(x, scale=1/lam), linewidth=2, label=f'λ={lam}')
ax.set_title('PDFs for Different λ')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: CDFs
ax = axes[0, 1]
for lam in [0.5, 1, 2, 5]:
    x = np.linspace(0, 3, 200)
    ax.plot(x, expon.cdf(x, scale=1/lam), linewidth=2, label=f'λ={lam}')
ax.set_title('CDFs for Different λ')
ax.set_xlabel('x')
ax.set_ylabel('Cumulative Probability')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Survival curves (reliability)
ax = axes[0, 2]
for lam in [0.5, 1, 2, 5]:
    x = np.linspace(0, 3, 200)
    ax.plot(x, expon.sf(x, scale=1/lam), linewidth=2, label=f'λ={lam}')
ax.set_title('Survival Curves')
ax.set_xlabel('Time')
ax.set_ylabel('P(X > t)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Histogram of service times
ax = axes[1, 0]
ax.hist(service_times, bins=30, alpha=0.6, density=True, 
        color='blue', edgecolor='black', label='Simulated')
x_theo = np.linspace(0, service_times.max(), 200)
ax.plot(x_theo, expon.pdf(x_theo, scale=1/lambda_service), 'r-', 
        linewidth=2.5, label='Theoretical')
ax.set_title('Service Times Distribution')
ax.set_xlabel('Service Time (minutes)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Hazard rates
ax = axes[1, 1]
x_hazard = np.linspace(0, 3000, 200)
hazard_const = np.ones_like(x_hazard) * lambda_bulb
hazard_weibull_vals = (2 / 1000) * (x_hazard / 1000)**1
ax.plot(x_hazard, hazard_const * 1000, 'b-', linewidth=2.5, label='Exponential (no aging)')
ax.plot(x_hazard, hazard_weibull_vals * 1000, 'r-', linewidth=2.5, label='Weibull (aging)')
ax.set_title('Hazard Rates: Constant vs Increasing')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Hazard Rate (per 1000 hours)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Poisson arrivals vs Exponential inter-arrivals
ax = axes[1, 2]
# Simulate cumulative arrivals
n_hours = 100
arrivals_cumsum = np.cumsum(np.random.poisson(lambda_arrivals, n_hours))
ax.plot(np.arange(n_hours), arrivals_cumsum, 'b-', linewidth=2, label='Cumulative arrivals')
ax.plot(np.arange(n_hours), lambda_arrivals * np.arange(n_hours), 'r--', 
        linewidth=2, label=f'Linear (λ={lambda_arrivals})')
ax.set_title('Poisson Arrivals Over Time')
ax.set_xlabel('Hour')
ax.set_ylabel('Cumulative Arrivals')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Confidence interval for λ (rate)
print("\n" + "="*60)
print("INFERENCE: Confidence Interval for λ")
print("="*60)

# Sample data: 10 observations
sample_data = np.array([120, 145, 98, 156, 132, 167, 89, 143, 125, 151])
n = len(sample_data)
sum_x = sample_data.sum()
lambda_mle = n / sum_x  # MLE

print(f"\nSample of {n} lifetimes (hours): {sample_data}")
print(f"Sample mean: {sample_data.mean():.2f} hours")
print(f"MLE of λ: {lambda_mle:.6f} per hour")
print(f"Estimated mean lifetime: {1/lambda_mle:.2f} hours")

# Chi-square based CI
chi2_lower = chi2.ppf(0.025, 2*n) / (2*sum_x)
chi2_upper = chi2.ppf(0.975, 2*n) / (2*sum_x)

print(f"\n95% Confidence Interval for λ (chi-square method):")
print(f"  [{chi2_lower:.6f}, {chi2_upper:.6f}] per hour")
print(f"  [{1/chi2_upper:.2f}, {1/chi2_lower:.2f}] hours (for mean lifetime)")
```

## 6. Challenge Round
1. **Bathtub Curve:** Model system with three phases: high early failure (Weibull k<1), constant middle (Exponential), increasing end (Weibull k>1). Fit piecewise.

2. **Memoryless Paradox:** Renewal theory—if buses arrive Poisson, expected time until next bus for random arrival ≠ average headway. Explore why (residual life problem).

3. **Minimum of Exponentials:** If n systems independent with mean lifetime 1/λ each, time until first failure ~ Exp(nλ). Verify simulation.

4. **Cox Proportional Hazards:** Survival analysis with covariates; baseline hazard exponential. Fit model with hazard ratio scaling.

5. **Competing Risks:** Person exposed to two independent exponential failure modes (disease A, disease B). Model joint failure time distribution.

## 7. Key References
- [Exponential Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Exponential_distribution)
- [Memoryless Property Proof](https://en.wikipedia.org/wiki/Memorylessness)
- [Reliability Engineering (Ebeling)](https://www.routledge.com/An-Introduction-to-Reliability-and-Maintainability-Engineering-Second-Edition/Ebeling/p/book/9781498735184)
- [Cox Proportional Hazards Model](https://en.wikipedia.org/wiki/Proportional_hazards_model)

---
**Status:** Foundation continuous distribution | **Complements:** Poisson Process, Gamma Distribution, Weibull Distribution, Survival Analysis
