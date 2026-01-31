# Poisson Distribution

## 1. Concept Skeleton
**Definition:** Discrete probability distribution modeling count of events occurring in fixed interval of time/space; P(X=k) = (e^(-λ) × λ^k) / k!  
**Purpose:** Model rare events, counts over time/area, arrival processes; foundation for event modeling; approximates binomial when n large and p small  
**Prerequisites:** Exponential distribution, independence, discrete distributions, event counting

## 2. Comparative Framing
| Property | Poisson | Binomial | Exponential | Normal |
|----------|---------|----------|-------------|--------|
| **Type** | Discrete | Discrete | Continuous | Continuous |
| **Parameters** | λ (rate) | n, p | λ (rate) | μ, σ |
| **Support** | 0,1,2,... | 0,1,...,n | [0, ∞) | (-∞,+∞) |
| **Event Model** | Count in interval | Successes in n trials | Time until event | Natural phenomena |
| **Mean** | λ | np | 1/λ | μ |
| **Variance** | λ | np(1-p) | 1/λ² | σ² |
| **Property** | Mean = Variance | Mean ≠ Variance | Memoryless | N/A |
| **Use** | Rare events | Fixed trials | Waiting time | General purpose |

## 3. Examples + Counterexamples

**Simple Example:**  
Website receives 3 spam emails per hour on average. What's P(exactly 5 spam in next hour)? λ=3, P(X=5) = (e^(-3) × 3^5) / 5! ≈ 0.101 or 10.1%.

**Perfect Fit:**  
Customer arrivals at service desk: λ=5 customers/hour (average). Count arrivals in random hour. Independent, rare relative to population, constant rate → Poisson.

**Approximation to Binomial:**  
Quality defects: n=1000 items, p=0.001 defect rate (rare). Use Poisson(λ=np=1) instead of Binomial(1000, 0.001). Same probabilities, simpler calculation.

**Poor Fit:**  
Traffic accidents per week in city: High count, not rare; Poisson underdispersed. Overdispersed data better fit by negative binomial.

**Edge Case:**  
λ=0: P(X=0)=1 (no events occur with certainty). λ→∞: Poisson → Normal (large count limit).

## 4. Layer Breakdown
```
Poisson Distribution:

├─ Probability Mass Function (PMF):
│  ├─ P(X = k) = (e^(-λ) × λ^k) / k!
│  ├─ λ = expected number of events in interval
│  ├─ e ≈ 2.71828 (Euler's number)
│  ├─ k! = k × (k-1) × ... × 1 (factorial)
│  ├─ Valid for k = 0, 1, 2, ...
│  └─ Sum: Σₖ₌₀^∞ P(X=k) = 1
├─ Key Assumptions (Poisson Process):
│  ├─ Independence: Non-overlapping intervals independent
│  ├─ Stationary Rate: λ constant across intervals
│  ├─ Rarity: Events occur individually, no simultaneous events
│  ├─ Individuality: Events don't cluster
│  └─ Small Probability: p small in any sub-interval
├─ Parameters and Properties:
│  ├─ Single parameter: λ (rate, intensity)
│  ├─ Mean: E[X] = λ
│  ├─ Variance: Var(X) = λ
│  ├─ Standard Deviation: σ = √λ
│  ├─ Skewness: 1/√λ (decreases as λ increases)
│  ├─ Kurtosis: 1/λ (heavy tails for small λ)
│  ├─ Mode: ⌊λ⌋ (floor of λ)
│  └─ Median: ≈ λ + 1/3 (approximately)
├─ Poisson Process Modeling:
│  ├─ Number of events in time interval t: Poisson(λt)
│  ├─ Time between events: Exponential(λ)
│  ├─ Relationship: If events ~ Poisson(λ), 
│  │   then inter-arrival time ~ Exponential(1/λ)
│  ├─ Memoryless property: P(wait > t+s | wait > t) = P(wait > s)
│  └─ Superposition: Sum of independent Poisson → Poisson
├─ Cumulative Distribution Function (CDF):
│  ├─ F(k) = P(X ≤ k) = Σⱼ₌₀ᵏ P(X=j)
│  ├─ No closed form; requires summation or tables
│  ├─ Related to incomplete gamma function
│  └─ Software: Use built-in functions
├─ Goodness-of-fit:
│  ├─ Test assumption of Poisson
│  ├─ Check mean ≈ variance (equidispersion)
│  ├─ Chi-square goodness-of-fit test
│  ├─ Overdispersion: Var > Mean (use negative binomial)
│  └─ Underdispersion: Var < Mean (rare; check assumptions)
├─ Inference on λ:
│  ├─ Point Estimate: λ̂ = mean count
│  ├─ Confidence Interval: Based on chi-square or exact method
│  ├─ Hypothesis Test: H₀: λ=λ₀ vs H₁: λ≠λ₀
│  └─ Variance typically estimated as sample variance ≈ mean
└─ Applications:
   ├─ Call center arrivals
   ├─ Network traffic events
   ├─ Radioactive decay counts
   ├─ Traffic accident modeling
   ├─ Disease outbreak counts
   └─ Wildlife population encounters
```

**Interaction:** λ controls both location and spread (mean=variance property). Larger λ → distribution spreads and approaches normal.

## 5. Mini-Project
Poisson distribution in event counting and process modeling:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chi2, expon
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("POISSON DISTRIBUTION EXPLORATION")
print("="*60)

# Scenario 1: Effect of λ on distribution shape
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

lambdas = [0.5, 1, 2, 5, 10, 20]

for idx, lam in enumerate(lambdas):
    ax = axes[idx // 3, idx % 3]
    
    # Generate PMF
    x_max = int(lam + 5 * np.sqrt(lam))  # Support extends to roughly λ + 5σ
    x = np.arange(0, x_max + 1)
    pmf = poisson.pmf(x, lam)
    
    # Statistics
    mean = lam
    variance = lam
    std = np.sqrt(lam)
    
    # Plot
    ax.bar(x, pmf, alpha=0.6, color='steelblue', edgecolor='black', width=0.8)
    ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'λ={lam}')
    
    # Overlay normal approximation for large λ
    if lam >= 5:
        x_cont = np.linspace(0, x_max, 200)
        from scipy.stats import norm
        normal_approx = norm.pdf(x_cont, mean, std)
        ax.plot(x_cont, normal_approx, 'g-', linewidth=2, alpha=0.7, label='Normal Approx')
    
    ax.set_title(f'Poisson(λ={lam})\nMean=Var={lam}', fontweight='bold')
    ax.set_xlabel('Number of Events (k)')
    ax.set_ylabel('Probability P(X=k)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Scenario 2: Real-world event modeling
print("\n" + "="*60)
print("EVENT MODELING: Website Errors Per Day")
print("="*60)

# Historical data: average 2.5 errors per day
lambda_errors = 2.5
days_observed = 365

# Generate historical data (simulation)
errors_per_day = np.random.poisson(lambda_errors, days_observed)

print(f"\nHistorical data from {days_observed} days:")
print(f"Sample mean: {errors_per_day.mean():.3f}")
print(f"Sample variance: {errors_per_day.var():.3f}")
print(f"Theoretical mean: {lambda_errors}")
print(f"Theoretical variance: {lambda_errors}")
print(f"Ratio (variance/mean): {errors_per_day.var() / errors_per_day.mean():.3f} (≈1 for Poisson)")

# Calculate probabilities
print(f"\nUsing Poisson(λ={lambda_errors}) to model:")
probs = {
    0: poisson.pmf(0, lambda_errors),
    1: poisson.pmf(1, lambda_errors),
    2: poisson.pmf(2, lambda_errors),
    3: poisson.pmf(3, lambda_errors),
}

for k, prob in probs.items():
    print(f"  P(X={k}) = {prob:.4f} ({prob*100:.2f}%)")

print(f"  P(X≤2) = {poisson.cdf(2, lambda_errors):.4f}")
print(f"  P(X>2) = {1 - poisson.cdf(2, lambda_errors):.4f}")
print(f"  P(1≤X≤3) = {poisson.cdf(3, lambda_errors) - poisson.cdf(0, lambda_errors):.4f}")

# Find percentiles
print(f"\nPercentiles:")
print(f"  75th percentile: {poisson.ppf(0.75, lambda_errors):.0f} errors")
print(f"  95th percentile: {poisson.ppf(0.95, lambda_errors):.0f} errors")
print(f"  99th percentile: {poisson.ppf(0.99, lambda_errors):.0f} errors")

# Scenario 3: Poisson Process - Inter-arrival times
print("\n" + "="*60)
print("POISSON PROCESS: Inter-arrival Times")
print("="*60)

# If events arrive as Poisson(λ=3/hour), then inter-arrival time ~ Exponential(1/3)
lambda_arrivals = 3  # per hour
events = 1000
inter_arrivals = np.random.exponential(scale=1/lambda_arrivals, size=events)

print(f"\nCustomer arrivals: λ={lambda_arrivals} per hour")
print(f"Simulated {events} inter-arrival times")
print(f"Mean inter-arrival time: {inter_arrivals.mean():.4f} hours = {inter_arrivals.mean()*60:.2f} minutes")
print(f"Theoretical: {1/lambda_arrivals:.4f} hours = {60/lambda_arrivals:.2f} minutes")

# Validate by counting events in hour-long intervals
hours_simulated = 100
cumulative_time = np.cumsum(inter_arrivals)
events_per_hour = []

for hour in range(hours_simulated):
    count = np.sum((cumulative_time > hour) & (cumulative_time <= hour + 1))
    events_per_hour.append(count)

events_per_hour = np.array(events_per_hour)
print(f"\nEvents per hour (from {hours_simulated} hours):")
print(f"Mean: {events_per_hour.mean():.3f} (theoretical: {lambda_arrivals})")
print(f"Variance: {events_per_hour.var():.3f} (theoretical: {lambda_arrivals})")

# Scenario 4: Goodness-of-fit test
print("\n" + "="*60)
print("GOODNESS-OF-FIT TEST")
print("="*60)

# Test dataset: count of calls in 24-hour intervals
observed_calls = np.array([3, 1, 2, 5, 2, 4, 3, 2, 1, 4, 2, 3, 2, 1, 5,
                          2, 3, 1, 4, 2, 3, 2, 1, 2, 3])
lambda_est = observed_calls.mean()

print(f"\nObserved call counts over 25 intervals:")
print(f"Data: {observed_calls}")
print(f"Estimated λ: {lambda_est:.3f}")

# Chi-square goodness-of-fit
observed_freq = np.bincount(observed_calls, minlength=6)
expected_freq = np.array([poisson.pmf(k, lambda_est) for k in range(6)]) * len(observed_calls)

# Combine small expected frequencies
min_expected = 5
observed_combined = []
expected_combined = []
obs_temp = 0
exp_temp = 0

for obs, exp in zip(observed_freq, expected_freq):
    if exp < min_expected:
        obs_temp += obs
        exp_temp += exp
    else:
        if obs_temp > 0 or exp_temp > 0:
            observed_combined.append(obs_temp)
            expected_combined.append(exp_temp)
            obs_temp = 0
            exp_temp = 0
        observed_combined.append(obs)
        expected_combined.append(exp)

if obs_temp > 0 or exp_temp > 0:
    observed_combined.append(obs_temp)
    expected_combined.append(exp_temp)

chi2_stat = np.sum((np.array(observed_combined) - np.array(expected_combined))**2 / np.array(expected_combined))
df = len(observed_combined) - 1 - 1  # -1 for parameter estimation
p_value = 1 - chi2.cdf(chi2_stat, df)

print(f"\nChi-square goodness-of-fit test:")
print(f"  χ² = {chi2_stat:.3f}")
print(f"  df = {df}")
print(f"  p-value = {p_value:.4f}")
print(f"  Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} Poisson model (α=0.05)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram of errors per day vs Poisson PMF
ax = axes[0, 0]
bins = np.arange(0, max(errors_per_day) + 2) - 0.5
ax.hist(errors_per_day, bins=bins, density=True, alpha=0.6, 
        color='blue', edgecolor='black', label='Simulated')
x_plot = np.arange(0, max(errors_per_day) + 1)
ax.plot(x_plot, poisson.pmf(x_plot, lambda_errors), 'ro-', linewidth=2, 
        markersize=8, label='Theoretical Poisson')
ax.set_title('Errors Per Day: Simulated vs Theoretical')
ax.set_xlabel('Number of Errors')
ax.set_ylabel('Probability')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Mean vs Variance (equidispersion property)
ax = axes[0, 1]
lambdas_test = np.array([0.5, 1, 2, 5, 10, 20, 50])
sample_sizes_range = [100, 500, 1000]
for n in sample_sizes_range:
    variances = []
    for lam in lambdas_test:
        samples = np.random.poisson(lam, n)
        variances.append(samples.var())
    ax.plot(lambdas_test, variances, 'o-', linewidth=2, label=f'n={n}', markersize=6)

ax.plot(lambdas_test, lambdas_test, 'k--', linewidth=2, label='y=x (theoretical)')
ax.set_xlabel('λ (Mean)')
ax.set_ylabel('Variance')
ax.set_title('Mean=Variance Property of Poisson')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Inter-arrival times (exponential)
ax = axes[1, 0]
ax.hist(inter_arrivals[:1000] * 60, bins=50, alpha=0.6, density=True, 
        color='green', edgecolor='black', label='Simulated')
x_exp = np.linspace(0, inter_arrivals[:1000].max() * 60, 200)
ax.plot(x_exp, expon.pdf(x_exp, scale=60/lambda_arrivals), 'r-', 
        linewidth=2, label='Theoretical Exponential')
ax.set_title('Inter-arrival Times (minutes)')
ax.set_xlabel('Time between arrivals (minutes)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Cumulative arrivals over time
ax = axes[1, 1]
cumsum = np.cumsum(np.random.poisson(lambda_arrivals, 100))
ax.plot(np.arange(100), cumsum, 'b-', linewidth=2, label='Simulated arrivals')
ax.plot(np.arange(100), lambda_arrivals * np.arange(100), 'r--', 
        linewidth=2, label=f'Linear trend (λ={lambda_arrivals})')
ax.set_title('Cumulative Arrivals Over Time')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Cumulative Events')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Overdispersion Detection:** Collect count data. Test variance ≈ mean. If variance >> mean, Poisson inadequate. Fit negative binomial; compare likelihood.

2. **Inhomogeneous Rates:** Emails vary by time of day (λ=2/hour during day, λ=0.5/hour at night). Model as time-varying Poisson; simulate.

3. **Rare Event Simulation:** Model rare disease outbreak (λ=0.1 cases/week). What's P(no cases for 30 weeks)? Compare Poisson exact to normal approximation failure.

4. **Poisson Regression:** Events depend on covariates (e.g., calls depend on marketing spend). Fit generalized linear model with Poisson link.

5. **Compound Poisson:** Each arriving customer orders random quantity. Model total quantity (not just count). Use law of total expectation: E[Total] = E[N] × E[Order].

## 7. Key References
- [Poisson Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Poisson Process (Durrett, Probability Theory)](https://www.springer.com/gp/book/9780521765398)
- [Cox-Stuart Trend Test](https://en.wikipedia.org/wiki/Cox%E2%80%93Stuart_test)
- [Poisson Regression (GLM)](https://en.wikipedia.org/wiki/Poisson_regression)

---
**Status:** Core event-count distribution | **Complements:** Exponential Distribution, Poisson Regression, Poisson Processes
