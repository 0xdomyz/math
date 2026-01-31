# Statistics: Core Concepts Learning Guide

Based on structured learning methodology - each concept follows the pattern:
1. Concept Skeleton | 2. Comparative Framing | 3. Examples + Counterexamples | 4. Layer Breakdown | 5. Mini-Project | 6. Challenge Round | 7. References

---

## 1. Hypothesis Testing

### 1.1 Concept Skeleton
**Definition:** Testing whether observed data provides sufficient evidence to reject a null hypothesis (H₀)
**Purpose:** Make probabilistic decisions about population parameters from sample data
**Prerequisites:** Normal distributions, sampling, statistical inference

### 1.2 Comparative Framing
| Concept | vs. Confidence Intervals | vs. Bayesian | vs. Effect Size |
|---------|------------------------|-------------|-----------------|
| **Hypothesis Test** | Tests specific claim (binary decision) | Prior beliefs included | Ignores practical significance |
| **Confidence Interval** | Estimates range of parameter values | No prior | Shows magnitude + uncertainty |
| **Bayesian** | Frequentist approach | Incorporates prior beliefs | Can estimate any parameter |
| **Effect Size** | Tests presence of effect | - | Measures practical importance |

### 1.3 Examples + Counterexamples
**Simple Example:**  
Drug company tests if new medicine improves recovery: H₀: μ = 50%, H₁: μ ≠ 50%

**Failure Case:**  
Using hypothesis test on ALL data collected (data dredging). Multiple tests inflate false positive rate → need correction (Bonferroni)

**Edge Case:**  
Very large sample: Even tiny, negligible effects become statistically significant (p < 0.05). Paradox: statistical significance ≠ practical importance

### 1.4 Layer Breakdown
```
Hypothesis Test Structure:
├─ Null Hypothesis (H₀): Status quo, no effect
├─ Alternative Hypothesis (H₁): Effect exists
├─ Test Statistic: t, z, χ², F (depending on data type)
├─ p-value: Probability of observing data IF H₀ true
├─ Significance Level (α): Decision threshold (usually 0.05)
└─ Decision: Reject H₀ if p < α
```

### 1.5 Mini-Project
Implement: t-test comparing two group means
```python
from scipy import stats
group1 = [2, 4, 6, 8, 10]
group2 = [3, 5, 7, 9, 11]
t_stat, p_value = stats.ttest_ind(group1, group2)
# Decision: reject H₀ if p_value < 0.05
```

### 1.6 Challenge Round
When is hypothesis testing the WRONG choice?
- Exploratory analysis (use confidence intervals instead)
- Multiple comparisons without correction
- When you really care about effect size, not just existence
- Small samples with multiple testing (high false positive rate)

### 1.7 Key References
- [Hypothesis Testing Intuition](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
- [p-values explained](https://www.statisticsdonewrong.com/)
- [Type I/II errors and power](https://www.khanacademy.org/math/statistics-probability/test-of-significance)

---

## 2. Probability Distributions

### 2.1 Concept Skeleton
**Definition:** Mathematical function describing likelihood of outcomes in a random experiment
**Purpose:** Model real-world phenomena, calculate probabilities, estimate parameters
**Prerequisites:** Probability basics, functions, calculus

### 2.2 Comparative Framing
| Distribution | Use Case | Shape | Parameters | Domain |
|--------------|----------|-------|-----------|--------|
| **Normal** | Natural measurements, errors | Bell curve (symmetric) | μ, σ | (-∞, ∞) |
| **Binomial** | Repeated binary trials | Discrete, peaked | n, p | {0,1,...,n} |
| **Exponential** | Time until next event | Right-skewed decay | λ | (0, ∞) |
| **Poisson** | Count of events in time | Discrete, peaked | λ | {0,1,2,...} |
| **Uniform** | Equal probability outcomes | Flat | a, b | [a, b] |
| **Chi-square** | Variance of normal data | Right-skewed | k | (0, ∞) |

### 2.3 Examples + Counterexamples
**Normal Distribution Example:**  
Heights in population: μ=170cm, σ=10cm → most people 160-180cm

**Failure Case:**  
Assuming normality when data is right-skewed (income, wait times). Model predictions fail at extremes.

**Edge Case:**  
Central Limit Theorem: Sample means are approximately normal EVEN if population isn't, if n is large enough

### 2.4 Layer Breakdown
```
Distribution Components:
├─ PDF/PMF: Probability density (continuous) or mass (discrete)
├─ CDF: Cumulative probability up to x
├─ Mean (μ): Center of distribution
├─ Variance (σ²): Spread around mean
├─ Skewness: Asymmetry (left/right)
└─ Kurtosis: Tail heaviness
```

### 2.5 Mini-Project
Visualize and compare distributions:
```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-4, 4, 1000)
plt.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), label='Normal')
plt.plot(x, 1/(1 + x**2) / np.pi, label='Cauchy')
# Note: Cauchy has undefined mean/variance!
```

### 2.6 Challenge Round
When does distribution choice matter least?
- Large samples (CLT makes everything approximately normal)
- When only testing for shift in location (rank tests robust)
- When using methods robust to outliers (median, IQR)

When does it matter MOST?
- Small samples (distribution shape critical)
- Extreme value prediction (tail behavior matters)
- Hypothesis testing about variance/shape

### 2.7 Key References
- [Probability Distributions Guide](https://en.wikipedia.org/wiki/Probability_distribution)
- [Interactive Distribution Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html)
- [When to use which distribution](https://www.khanacademy.org/math/statistics-probability)

---

## 3. Correlation vs Causation

### 3.1 Concept Skeleton
**Definition:** Correlation = statistical association; Causation = one variable directly changes another
**Purpose:** Avoid false conclusions from observational data
**Prerequisites:** Regression, confounding variables, experimental design

### 3.2 Comparative Framing
| Aspect | Correlation | Causation | Confounding |
|--------|-------------|-----------|-------------|
| **Measured by** | r ∈ [-1, 1] | Regression coefficient | Lurking variable |
| **Strength** | Perfect: r=±1 | Requires mechanism | Hides true relationship |
| **Evidence needed** | Co-variation only | + temporal order + mechanism | Needs adjustment |
| **Graph shape** | Scatter plot | DAG (directed acyclic graph) | Triangle: Z→X, Z→Y |

### 3.3 Examples + Counterexamples
**False Causation (Classic):**  
"Ice cream sales correlate with drowning deaths"  
Truth: Both caused by warm weather (Z-variable)

**Real Causation Example:**  
Randomized drug trial: Random assignment breaks confounding, enables causal inference

**Edge Case:**  
Reverse causation: Does depression cause poor sleep, or poor sleep cause depression? Need longitudinal data to establish temporal order

### 3.4 Layer Breakdown
```
Path to Causal Inference:
├─ Observational data: correlation only (Z confounds X→Y)
├─ Temporal sequencing: X before Y (rules out reverse causation)
├─ Mechanism: Theory explaining X→Y (plausible?)
├─ Adjustment: Control for confounders (stratification, regression)
├─ Randomization: Ideal but often impossible
└─ Causal diagram: DAG showing all relationships
```

### 3.5 Mini-Project
Identify confounding in real data:
```python
# Question: Does coffee cause heart disease?
# Correlation: coffee ↔ heart disease
# Confounders: age, smoking (both correlate with coffee AND disease)
# Solution: Stratify by age/smoking, run regression with covariates
# Result: Confounding explained most correlation
```

### 3.6 Challenge Round
When is observational correlation ENOUGH?
- Consistent pattern across populations (reproducible correlation)
- Dose-response relationship (more X → more Y)
- No plausible alternative explanations
- Strong theoretical mechanism

When must you have randomization?
- Studying harmful interventions (can't randomize people to smoking)
- Policy decisions requiring high confidence
- When confounders are unmeasured

### 3.7 Key References
- [Bradford Hill Criteria for Causation](https://en.wikipedia.org/wiki/Bradford_Hill_criteria)
- [Simpson's Paradox (confounding visual)](https://en.wikipedia.org/wiki/Simpson%27s_paradox)
- [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

---

## 4. Statistical Significance vs Practical Significance

### 4.1 Concept Skeleton
**Definition:**  
- Statistical: Result unlikely if null hypothesis true (p < α)
- Practical: Effect size large enough to matter in real world
**Purpose:** Avoid misleading conclusions from large samples
**Prerequisites:** Effect size, hypothesis testing, sample size

### 4.2 Comparative Framing
| Factor | Statistical Sig. | Practical Sig. | Confidence Interval |
|--------|-----------------|---------------|--------------------|
| **Depends on** | p-value, sample size | Effect size magnitude | Both: effect + uncertainty |
| **"A bigger jump"** | Small sample needed | Large effect needed | Shows plausible range |
| **Paradox** | Large n: trivial effect sig. | Small n: large effect not sig. | Always informative |

### 4.3 Examples + Counterexamples
**Large Sample, No Practical Significance:**  
Study of 1,000,000 people: Weight loss program gives 0.1 kg average loss (p=0.01). Statistically significant but clinically irrelevant.

**Small Sample, Practical Difference, Not Sig.:**  
10 patients: New treatment 50% vs old 30% success (p=0.20). Practical difference but low power.

**Edge Case:**  
Equivalence testing: Proving treatments are approximately equal (inverse of typical hypothesis test)

### 4.4 Layer Breakdown
```
Comprehensive Analysis:
├─ Point Estimate: Best guess for effect (mean difference)
├─ Confidence Interval: Plausible range (95% CI)
├─ Effect Size: Standardized measure (Cohen's d, r, odds ratio)
├─ Hypothesis Test: p-value
├─ Sample Size: Power to detect meaningful effect
└─ Clinical/Practical Threshold: "What size difference matters?"
```

### 4.5 Mini-Project
Compare statistical vs practical significance:
```python
from scipy import stats
import numpy as np

# Scenario 1: Large effect, small sample
effect1 = [10, 12, 11, 13, 12]
baseline1 = [8, 9, 7, 8, 9]
t1, p1 = stats.ttest_ind(effect1, baseline1)
# Large effect (2 units), maybe p > 0.05 (not sig)

# Scenario 2: Tiny effect, huge sample
effect2 = np.random.normal(100.1, 1, 10000)
baseline2 = np.random.normal(100, 1, 10000)
t2, p2 = stats.ttest_ind(effect2, baseline2)
# Tiny effect (0.1 units), likely p < 0.05 (sig)
```

### 4.6 Challenge Round
When is statistical significance sufficient?
- Early exploratory phase (screening for candidates)
- Consistent with multiple studies
- Clear theoretical mechanism

When do you NEED practical significance?
- Clinical/health decisions (patients care about real improvement)
- Policy/business decisions (costs must justify benefits)
- Any applied research impacting real world

### 4.7 Key References
- [Effect Size Interpretation Guide](https://en.wikipedia.org/wiki/Effect_size)
- [Why p < 0.05 is problematic](https://www.nature.com/articles/d41586-019-00857-7)
- [Moving to practical significance](https://www.statisticsdonewrong.com/)

---

## 5. Sampling Bias & Representative Sampling

### 5.1 Concept Skeleton
**Definition:** Bias when sample differs systematically from population; Representative when sample reflects population
**Purpose:** Ensure inferences from sample generalize to population
**Prerequisites:** Population concepts, sampling methods, probability

### 5.2 Comparative Framing
| Sampling Method | Representativeness | Bias Risk | Use Case |
|-----------------|-------------------|-----------|----------|
| **Random** | High if n large | None (theor.) | Gold standard |
| **Stratified** | High if strata defined well | Minimal | Ensure groups included |
| **Convenience** | Low (volunteers) | Selection bias | Pilot/exploratory |
| **Cluster** | Depends on cluster homogeneity | Geographic bias | Large populations |
| **Systematic** | If population random order | Periodic pattern bias | Structured lists |

### 5.3 Examples + Counterexamples
**Classic Bias Example:**  
1936 election poll: Surveyed car owners + phone owners (wealthy) → Wrong prediction. Poor people couldn't afford these.

**Good Sampling:**  
Exit polls (random sample of voters leaving polls) → usually accurate

**Edge Case:**  
Nonresponse bias: People who respond to survey differ from non-responders (depression prevalence underestimated if depressed less likely to respond)

### 5.4 Layer Breakdown
```
Sources of Bias:
├─ Selection Bias: Who gets sampled (excluded groups)
├─ Nonresponse Bias: Who responds (self-selection)
├─ Measurement Bias: How we measure (question wording)
├─ Survivorship Bias: Only observe survivors (survivorship)
└─ Healthy User Bias: People making effort differ systematically
```

### 5.5 Mini-Project
Detect selection bias in data:
```python
# Survey: "How much do you value exercise?"
# Bias: Surveyed at gym (selection) vs online (self-selection)
# Result: Mean rating at gym > online (biased upward at gym)
# Solution: Weight responses inversely to inclusion probability
```

### 5.6 Challenge Round
When is perfect representativeness impossible?
- Studying homeless populations (no sampling frame)
- Rare diseases (can't find enough cases)
- Studying past events (survivors only exist)

When can you proceed despite bias?
- Bias direction known (can adjust estimates)
- Internal comparisons (comparing subgroups within sample)
- Machine learning (predictive accuracy matters more than rep.)

### 5.7 Key References
- [Sampling Bias Examples](https://en.wikipedia.org/wiki/Sampling_bias)
- [Literary Digest Fiasco (1936)](https://en.wikipedia.org/wiki/1936_Literary_Digest_prediction_error)
- [Selection Bias in Clinical Trials](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3879437/)

---

## 6. Central Limit Theorem (CLT)

### 6.1 Concept Skeleton
**Definition:** Distribution of sample means approaches normal distribution as n increases, regardless of population distribution shape
**Purpose:** Justifies using normal distribution for inference about means
**Prerequisites:** Distributions, sampling, normal distribution

### 6.2 Comparative Framing
| Aspect | CLT | Law of Large Numbers | Normal Distribution |
|--------|-----|-------------------|-------------------|
| **Concerns** | Distribution of means | Sample mean convergence | Population shape |
| **Implication** | Means ~ Normal | Mean → μ as n→∞ | Predictions about individuals |
| **Sample size** | n ≥ 30 usually works | Needs very large n | Applies to single observations |

### 6.3 Examples + Counterexamples
**CLT Success:**  
Population: Exponential (right-skewed). Draw 100 samples of n=50, plot sample means → Bell curve! (Despite population skewed)

**Where CLT Fails:**  
Heavy-tailed distribution (Cauchy): Sample means don't converge. Undefined mean/variance break CLT assumptions.

**Edge Case:**  
Skewed populations: n ≥ 30 rule of thumb breaks down. With heavy skew, need n > 100 for approximate normality

### 6.4 Layer Breakdown
```
CLT Components:
├─ Original Population: Any distribution, mean μ, std σ
├─ Sampling Process: Draw samples of size n, compute mean x̄
├─ Distribution of x̄: 
│   ├─ Mean: μ (unbiased)
│   ├─ Std Dev: σ/√n (decreases with n!)
│   └─ Shape: Approaches Normal as n→∞
└─ Convergence Rate: Faster for symmetric populations
```

### 6.5 Mini-Project
Visualize CLT:
```python
import numpy as np
import matplotlib.pyplot as plt

# Exponential population (skewed!)
population = np.random.exponential(scale=2, size=100000)

# Draw 1000 samples, compute means
sample_means = []
for i in range(1000):
    sample = np.random.choice(population, size=50)
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=50, density=True)
# Result: Bell curve despite exponential population!
```

### 6.6 Challenge Round
When does CLT break down?
- Extremely heavy-tailed distributions (Cauchy, some financial data)
- Dependent observations (time series, spatial data)
- Finite population, sampling without replacement significantly

When is CLT not needed?
- Non-parametric tests (don't assume normality)
- Direct probability calculation (don't need normal approx)
- Modern computing (bootstrap instead)

### 6.7 Key References
- [CLT Interactive Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html)
- [Formal Statement & Proof](https://en.wikipedia.org/wiki/Central_limit_theorem)
- [Convergence Rate Discussion](https://stats.stackexchange.com/questions/146920/)

---

## 7. Confidence Intervals

### 7.1 Concept Skeleton
**Definition:** Range of values containing true parameter with specified probability (e.g., 95% CI)
**Purpose:** Quantify uncertainty about parameter estimates from sample data
**Prerequisites:** Sampling distributions, standard error, normal distribution

### 7.2 Comparative Framing
| Concept | Confidence Interval | Prediction Interval | Credible Interval |
|---------|------------------|------------------|------------------|
| **What it bounds** | Population parameter | Future individual value | Parameter (Bayesian) |
| **Wider** | Narrower | Much wider (includes variation) | Depends on prior |
| **Interpretation** | Procedure covers true param 95% of samples | Future obs falls in range 95% of time | 95% probability param in range |
| **Frequentist** | Yes | Yes | No (Bayesian) |

### 7.3 Examples + Counterexamples
**Simple Example:**  
Sample of 100 people: mean height 170cm, SE=2cm → 95% CI = [166, 174]cm  
Interpretation: If we repeated sampling, 95% of such intervals would contain true mean

**Misinterpretation:**  
NOT "95% probability true mean in [166, 174]" (frequentist: param is fixed, not random)

**Edge Case:**  
Confidence level vs coverage: Nominal 95% CI may not have exactly 95% coverage (depends on assumptions)

### 7.4 Layer Breakdown
```
CI Construction:
├─ Point Estimate: Sample statistic (mean, proportion)
├─ Standard Error: Std dev of sampling distribution
├─ Critical Value: z or t value for desired confidence
├─ Margin of Error: CV × SE
└─ Interval: Point ± Margin of Error
```

### 7.5 Mini-Project
Calculate confidence intervals:
```python
import scipy.stats as stats

# Sample data
data = [5, 7, 8, 6, 9, 7, 8, 7]
n = len(data)
mean = np.mean(data)
se = np.std(data, ddof=1) / np.sqrt(n)

# 95% CI using t-distribution
t_crit = stats.t.ppf(0.975, df=n-1)
ci = (mean - t_crit*se, mean + t_crit*se)
print(f"95% CI: {ci}")
```

### 7.6 Challenge Round
When is confidence interval better than hypothesis test?
- Estimation focus (how big is the effect?)
- Multiple parameters (shows all simultaneously)
- Effect size matters (CI shows magnitude)

When might CI fail?
- Non-coverage: Nominal 95% CI has < 95% coverage (small n, non-normal)
- Misinterpretation: Leads to Bayesian conclusions
- Multiple comparisons: Need adjustment

### 7.7 Key References
- [CI Interpretation Guide](https://en.wikipedia.org/wiki/Confidence_interval)
- [Why not use CI for everything](https://www.nature.com/articles/d41586-019-00857-7)
- [Bootstrap Confidence Intervals](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

---

## 8. Type I & Type II Errors, Power

### 8.1 Concept Skeleton
**Definition:**  
- Type I (α): Reject H₀ when true (false positive)
- Type II (β): Fail to reject H₀ when false (false negative)
- Power: 1-β (probability of detecting true effect)
**Purpose:** Balance risk of false conclusions
**Prerequisites:** Hypothesis testing, decision rules

### 8.2 Comparative Framing
| Error Type | Cost if Occurs | Typical α | Typical β | Example |
|-----------|----------------|----------|----------|---------|
| **Type I** | False alarm, wasted resources | 0.05 | - | Claiming cure doesn't work (stops good treatment) |
| **Type II** | Miss real effect, miss opportunity | - | 0.20 | Missing actual cure (patients suffer) |
| **Power** | - | - | 0.80 typical | Probability of finding cure if it exists |

### 8.3 Examples + Counterexamples
**Type I Error (α=0.05):**  
Testing 100 true null hypotheses → expect 5 false positives (random noise)

**Type II Error (β):**  
Small sample size → low power → likely miss real effect even if present

**Edge Case:**  
Very low α (0.001) reduces Type I but increases Type II. Tradeoff exists!

### 8.4 Layer Breakdown
```
Decision Matrix:
            Reality: H₀ True  | Reality: H₀ False
Decide: H₀  ├─ Correct (1-α)  | Type II Error (β)
Decide: H₁  ├─ Type I Error (α)| Correct (Power=1-β)

Power Determinants:
├─ Effect Size: Larger effect → higher power
├─ Sample Size: Larger n → higher power
├─ α level: Higher α → higher power (but more Type I errors)
├─ Variability: Lower σ → higher power
└─ Test Type: One-tailed > two-tailed power
```

### 8.5 Mini-Project
Calculate statistical power:
```python
from scipy.stats import norm

# Parameters
effect_size = 0.5  # Cohen's d
alpha = 0.05
sample_size = 64

# Power calculation (simplified for 2-sample t-test)
# Power ≈ Φ(√(n/2) * effect_size - z_α)
z_alpha = norm.ppf(1 - alpha/2)
z_power = np.sqrt(sample_size/2) * effect_size - z_alpha
power = norm.cdf(z_power)
print(f"Power: {power:.3f}")  # Usually ~0.80 is target
```

### 8.6 Challenge Round
When do you accept high Type II error?
- Expensive interventions (low α acceptable to avoid false positives)
- Regulatory approval (side effects feared)

When do you prioritize detecting effects?
- Cheap screening (false positives caught later)
- Medical diagnosis (missing disease = big cost)
- Exploratory research (low bar to investigate further)

### 8.7 Key References
- [Type I/II Error Visualization](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
- [Power Analysis Primer](https://www.khanacademy.org/math/statistics-probability/test-of-significance)
- [Sample Size/Power Calculators](https://www.biostathandbook.com/power.html)

---

## 9. Regression Analysis

### 9.1 Concept Skeleton
**Definition:** Statistical technique predicting outcome variable (y) from predictor(s) (x) via linear combination
**Purpose:** Quantify relationships, predict future values, control for confounders
**Prerequisites:** Linear algebra, correlation, hypothesis testing

### 9.2 Comparative Framing
| Method | Linear Regression | Logistic Regression | Multiple Regression |
|--------|------------------|-------------------|-------------------|
| **Outcome** | Continuous (unbounded) | Binary (0/1) | Multiple y or multiple x |
| **Assumptions** | Normality of residuals | Logistic link function | Extended linearity |
| **Interpretation** | Unit change in x → β change in y | Unit change in x → odds multiply by e^β | Holds other x constant |

### 9.3 Examples + Counterexamples
**Simple Example:**  
House price ~ square footage: For each +100 sq ft, price increases $50,000

**Failure Case:**  
Assuming linear relationship when true is quadratic (x²). Residuals show pattern → wrong model

**Edge Case:**  
Multicollinearity: Two predictors highly correlated → coefficients unstable, hard to interpret

### 9.4 Layer Breakdown
```
Regression Components:
├─ Model: y = β₀ + β₁x₁ + ... + βₖxₖ + ε
├─ β₀: Intercept (y when all x=0)
├─ βⱼ: Slope of xⱼ (effect holding others constant)
├─ ε: Error (residual variation)
├─ Assumptions:
│   ├─ Linearity: True relationship linear
│   ├─ Independence: Observations independent
│   ├─ Homoscedasticity: Constant variance of errors
│   ├─ Normality: Residuals normally distributed
│   └─ No multicollinearity: Predictors not too correlated
└─ Fit: R² (variance explained), F-test (overall significance)
```

### 9.5 Mini-Project
Fit and evaluate regression:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit
model = LinearRegression().fit(x, y)
r2 = model.score(x, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"R²: {r2:.2f}")
```

### 9.6 Challenge Round
When is regression the wrong tool?
- Severely non-linear relationships (use splines, GAM)
- Clustered data (use mixed models, GEE)
- Classification outcome (use logistic)
- Time-series (use ARIMA, not naive regression)

### 9.7 Key References
- [Regression Assumptions Visual](https://en.wikipedia.org/wiki/Linear_regression)
- [Multicollinearity Diagnosis](https://stats.stackexchange.com/questions/tagged/multicollinearity)
- [Regression Diagnostics](https://www.r-bloggers.com/linear-regression-assumptions/)

---

## 10. Bayesian vs Frequentist Statistics

### 10.1 Concept Skeleton
**Definition:**  
- Frequentist: Probability = long-run frequency; inference without priors
- Bayesian: Probability = degree of belief; incorporates prior knowledge via Bayes rule
**Purpose:** Different frameworks for statistical inference with different interpretations
**Prerequisites:** Conditional probability, prior distributions, Bayes theorem

### 10.2 Comparative Framing
| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Probability** | Long-run frequency | Degree of belief |
| **Parameters** | Fixed but unknown | Random variables (distributions) |
| **Prior** | Not used | Essential input |
| **Inference** | p-values, CI, hypothesis tests | Posterior distributions |
| **Interpretation** | "If repeated infinitely, 95% CI contain true param" | "Given data, 95% probability param in range" |
| **Advantage** | Objective (no prior needed) | Incorporates domain knowledge |
| **Disadvantage** | Counterintuitive interpretation | Prior choice subjective |

### 10.3 Examples + Counterexamples
**Frequentist Approach:**  
Test: "Is coin fair?" H₀: p=0.5, observe 12 heads in 20 flips, p=0.058 > 0.05. Fail to reject (not enough evidence)

**Bayesian Approach:**  
Prior: Coin probably fair (p~Beta(10,10)). Data: 12 heads in 20. Posterior: p probably ~0.55-0.60 given data

**Edge Case:**  
Optional stopping: Frequentist p-value depends on stopping rule (when you decide to stop). Bayesian posterior unaffected!

### 10.4 Layer Breakdown
```
Bayes Theorem: P(θ|Data) = P(Data|θ) × P(θ) / P(Data)

Components:
├─ Prior P(θ): Belief before seeing data
├─ Likelihood P(Data|θ): Data probability given parameter
├─ Posterior P(θ|Data): Updated belief after data
└─ Marginal P(Data): Normalizing constant

Workflow:
├─ Specify prior distribution P(θ)
├─ Observe data
├─ Update to posterior (computation hard in general)
└─ Make decisions from posterior (credible intervals, MAP)
```

### 10.5 Mini-Project
Simple Bayesian inference:
```python
import numpy as np
from scipy.special import beta as beta_function

# Coin flips: 7 heads in 10 flips
data_heads, data_tails = 7, 3

# Prior: Beta(1,1) = uniform
prior_a, prior_b = 1, 1

# Posterior: Beta(1+7, 1+3) = Beta(8,4)
posterior_a = prior_a + data_heads
posterior_b = prior_b + data_tails

# Posterior mean
posterior_mean = posterior_a / (posterior_a + posterior_b)
print(f"Posterior mean: {posterior_mean:.2f}")  # 0.67 (updated toward data)
```

### 10.6 Challenge Round
When choose Frequentist?
- Large data (differences between approaches vanish)
- Standardized, objective protocol needed
- Prior too controversial or expensive to elicit

When choose Bayesian?
- Small sample, strong prior knowledge available
- Sequential decision-making (updating)
- Need probability statements about parameters
- Incorporating expert opinion required

### 10.7 Key References
- [Bayes Theorem Intuition](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Frequentist vs Bayesian Comparison](https://www.nature.com/articles/d41586-020-00609-0)
- [Bayesian Inference Tutorial](https://www.probabilistic-programming.org)

---

## Quick Reference: When to Use What

| Goal | Use This | Key Tool |
|------|----------|----------|
| **Test if effect exists** | Hypothesis test | p-value |
| **Estimate size + uncertainty** | Confidence interval | Point ± Margin |
| **Predict individual value** | Regression + prediction interval | Wider CI |
| **Compare groups** | ANOVA or t-test | F or t statistic |
| **Categorical outcome** | Logistic regression | Odds ratios |
| **Check association** | Correlation + scatter plot | r or Spearman ρ |
| **Relationship causal?** | Randomized experiment | Experimental design |
| **Incorporate prior knowledge** | Bayesian methods | Posterior distribution |
| **No normality assumption** | Non-parametric (rank tests) | Wilcoxon, Kruskal-Wallis |
| **Time-series data** | ARIMA, state-space models | Autocorrelation |

---

**Last Updated:** 2026-01-31  
**Status:** Core concepts framework - add domain-specific applications as needed
