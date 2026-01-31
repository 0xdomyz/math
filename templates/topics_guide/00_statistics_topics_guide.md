# Statistics Topics Guide

**Complete reference of foundational and advanced statistics concepts with categories, brief descriptions, and sources.**

---

## I. Descriptive Statistics & Data Exploration

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Mean, Median, Mode** | N/A | Central tendency measures; mean affected by outliers | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) |
| **Variance & Standard Deviation** | N/A | Measures of spread/dispersion around mean | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) |
| **Interquartile Range (IQR)** | N/A | Middle 50% spread; robust to outliers | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) |
| **Outliers & Identification** | N/A | Data points beyond 1.5×IQR from quartiles | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) |
| **Percentiles & Z-scores** | N/A | Position in distribution; standardized units | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data) |
| **Data Visualization** | N/A | Histograms, scatter plots, box plots, bar charts, dot plots | [Wiki - Statistical Graphics](https://en.wikipedia.org/wiki/Statistical_graphics) |

---

## II. Probability Theory Foundations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Basic Probability** | N/A | P(A) = favorable outcomes / total outcomes | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library) |
| **Conditional Probability** | N/A | P(A\|B) = joint probability modified by event B | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library) |
| **Independence vs Dependence** | N/A | Events unrelated (independent) or related (dependent) | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library) |
| **Law of Large Numbers** | N/A | Sample mean → population mean as n→∞ | [Wiki - LLN](https://en.wikipedia.org/wiki/Law_of_large_numbers) |
| **Counting & Combinatorics** | N/A | Permutations (order matters), Combinations (order irrelevant) | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/counting-permutations-and-combinations) |

---

## III. Probability Distributions

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Probability Distributions** | ✓ probability_distributions.md | Normal, Binomial, Poisson, Exponential, Uniform, Chi-square | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data) |
| **Normal Distribution** | ✓ probability_distributions.md | Bell curve; mean μ, std σ; symmetric, continuous | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/normal-distributions-library) |
| **Binomial Distribution** | ✓ probability_distributions.md | n repeated trials, probability p; discrete outcomes | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/binomial-random-variables) |
| **Poisson Distribution** | ✓ probability_distributions.md | Count of events in time interval; rare events | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) |
| **Exponential Distribution** | ✓ probability_distributions.md | Time until next event; right-skewed decay | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) |
| **Chi-square Distribution** | N/A | Variance of normal data; right-skewed | [Wiki - Chi-square](https://en.wikipedia.org/wiki/Chi-squared_distribution) |
| **Density Curves & CDF** | N/A | PDF/PMF (probability), CDF (cumulative) | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/density-curve) |
| **Empirical Rule** | N/A | ~68% (±1σ), ~95% (±2σ), ~99.7% (±3σ) in normal | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data) |

---

## IV. Sampling & Experimental Design

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Sampling Theory** | N/A | Random, Stratified, Quota, Cluster sampling methods | [Wiki - Sampling](https://en.wikipedia.org/wiki/Sampling_(statistics)) |
| **Sampling Bias** | ✓ sampling_bias.md | Non-random selection distorts estimates; survivorship bias, spectrum bias | [Wiki - Biased Sample](https://en.wikipedia.org/wiki/Biased_sample) |
| **Design of Experiments** | N/A | Factorial, Randomized block, Cross-over, Repeated measures | [Wiki - DOE](https://en.wikipedia.org/wiki/Design_of_experiments) |
| **Randomization** | N/A | Eliminates confounding; essential for causal inference | [Wiki - Randomization](https://en.wikipedia.org/wiki/Randomization) |
| **Control Groups** | N/A | Comparison baseline in experiments; isolates treatment effect | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/designing-studies) |

---

## V. Statistical Inference & Hypothesis Testing

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Hypothesis Testing** | ✓ hypothesis_testing.md | Test H₀ vs H₁; p-value < α rejects null | [Wiki - Hypothesis Testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing) |
| **Null & Alternative Hypothesis** | ✓ hypothesis_testing.md | H₀: No effect; H₁: Effect exists | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample) |
| **P-values** | ✓ hypothesis_testing.md | P(data \| H₀ true); probability of observing evidence if null true | [Wiki - P-value](https://en.wikipedia.org/wiki/P-value) |
| **Significance Level (α)** | ✓ hypothesis_testing.md | Decision threshold; typically 0.05; controls false positive rate | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample) |
| **Type I Error (False Positive)** | ✓ type_errors_power.md | Reject H₀ when it's true; probability = α | [Wiki - Type I/II Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors) |
| **Type II Error (False Negative)** | ✓ type_errors_power.md | Fail to reject H₀ when it's false; probability = β | [Wiki - Type I/II Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors) |
| **Statistical Power** | ✓ type_errors_power.md | 1 - β; probability of detecting true effect | [Wiki - Power](https://en.wikipedia.org/wiki/Statistical_power) |
| **Test Statistics (t, z, χ², F)** | ✓ hypothesis_testing.md | Standardized measure for decision; distribution determines p-value | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample) |
| **Confidence Intervals** | ✓ confidence_intervals.md | Parameter estimate range with known coverage (e.g., 95%) | [Wiki - CI](https://en.wikipedia.org/wiki/Confidence_interval) |
| **Margin of Error** | ✓ confidence_intervals.md | Range width; decreases with larger n | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample) |
| **Effect Size** | ✓ statistical_vs_practical_significance.md | Magnitude of effect (Cohen's d); independent of sample size | [Wiki - Effect Size](https://en.wikipedia.org/wiki/Effect_size) |

---

## VI. Sampling Distributions & Central Limit Theorem

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Central Limit Theorem** | ✓ central_limit_theorem.md | Sample means → normal as n increases, regardless of population shape | [Wiki - CLT](https://en.wikipedia.org/wiki/Central_limit_theorem) |
| **Sampling Distribution** | N/A | Distribution of sample statistics (means, proportions) | [Wiki - Sampling Distribution](https://en.wikipedia.org/wiki/Sampling_distribution) |
| **Standard Error** | N/A | SD of sample mean = σ/√n; decreases with n | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library) |
| **Sample Proportion Distribution** | N/A | p̂ ~ Normal(p, √(p(1-p)/n)) for large n | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library/sample-proportions) |
| **Unbiased Estimators** | N/A | Sample statistic mean equals population parameter | [Wiki - Estimator](https://en.wikipedia.org/wiki/Estimator) |

---

## VII. Inference Methods & Estimators

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Maximum Likelihood Estimation (MLE)** | N/A | Find parameters that maximize likelihood of observed data | [Wiki - MLE](https://en.wikipedia.org/wiki/Maximum_likelihood) |
| **Bayesian Inference** | ✓ bayesian_vs_frequentist.md | Prior × Likelihood = Posterior; incorporates domain knowledge | [Wiki - Bayesian](https://en.wikipedia.org/wiki/Bayesian_inference) |
| **Prior & Posterior Distributions** | ✓ bayesian_vs_frequentist.md | Prior: initial belief; Posterior: updated after data | [Wiki - Bayes](https://en.wikipedia.org/wiki/Bayes%27_theorem) |
| **Frequentist Inference** | ✓ bayesian_vs_frequentist.md | Long-run frequency interpretation; no priors; hypothesis testing focus | [Wiki - Frequentist](https://en.wikipedia.org/wiki/Frequentist_inference) |
| **Decision Theory** | N/A | Choose action minimizing expected loss; Type I/II errors, minimax | [Wiki - Decision Theory](https://en.wikipedia.org/wiki/Decision_theory) |

---

## VIII. Correlation & Regression

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Correlation** | ✓ correlation_vs_causation.md | Association strength -1 to +1; correlation ≠ causation | [Wiki - Correlation](https://en.wikipedia.org/wiki/Correlation_and_dependence) |
| **Causation** | ✓ correlation_vs_causation.md | Mechanism-based; requires experimental control or domain theory | [Wiki - Causation](https://en.wikipedia.org/wiki/Causality) |
| **Regression Analysis** | ✓ regression_analysis.md | Linear: y = a + bx; predicts Y given X | [Wiki - Regression](https://en.wikipedia.org/wiki/Regression_analysis) |
| **Least-Squares Method** | ✓ regression_analysis.md | Minimizes sum of squared residuals; ordinary least squares (OLS) | [Wiki - Least Squares](https://en.wikipedia.org/wiki/Least_squares) |
| **Residuals** | ✓ regression_analysis.md | Actual - Predicted; should be random, normally distributed | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/regression-library) |
| **Generalized Linear Models (GLM)** | N/A | Extends regression to non-normal outcomes; logistic, Poisson | [Wiki - GLM](https://en.wikipedia.org/wiki/Generalized_linear_model) |
| **Ridge & Lasso Regression** | N/A | Regularization methods; prevent overfitting via penalties | [Wiki - Ridge](https://en.wikipedia.org/wiki/Ridge_regression) |

---

## IX. Multivariate & Advanced Analysis

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Analysis of Variance (ANOVA)** | N/A | Compare means across 3+ groups; F-statistic | [Wiki - ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance) |
| **Chi-Square Tests** | N/A | Goodness-of-fit (categorical data); independence in contingency tables | [Wiki - Chi-square](https://en.wikipedia.org/wiki/Chi-squared_test) |
| **Principal Component Analysis (PCA)** | N/A | Dimensionality reduction; compress high-dimensional data | [Wiki - PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) |
| **Factor Analysis** | N/A | Identify latent structures in multivariate data | [Wiki - Factor Analysis](https://en.wikipedia.org/wiki/Factor_analysis) |
| **Cluster Analysis** | N/A | Group similar observations; k-means, hierarchical clustering | [Wiki - Clustering](https://en.wikipedia.org/wiki/Cluster_analysis) |
| **Survival Analysis** | N/A | Time-to-event data; Kaplan-Meier curves, Cox proportional hazards | [Wiki - Survival](https://en.wikipedia.org/wiki/Survival_analysis) |
| **Time Series Analysis** | N/A | Sequential temporal data; ARIMA, Box-Jenkins, seasonality | [Wiki - Time Series](https://en.wikipedia.org/wiki/Time_series) |

---

## X. Computational Statistics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Markov Chain Monte Carlo (MCMC)** | N/A | Sample from posterior; Metropolis-Hastings algorithm | [Wiki - MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) |
| **Bootstrapping** | N/A | Resample with replacement; estimate uncertainty without assumptions | [Wiki - Bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) |
| **Jackknife Resampling** | N/A | Leave-one-out validation; estimate bias and SE | [Wiki - Jackknife](https://en.wikipedia.org/wiki/Jackknife_resampling) |
| **Kernel Density Estimation** | N/A | Non-parametric PDF estimation; smooth histogram | [Wiki - KDE](https://en.wikipedia.org/wiki/Kernel_density_estimation) |
| **Cross-Validation** | N/A | Train-test split; k-fold, leave-one-out for model evaluation | [Wiki - Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) |

---

## XI. Non-Parametric Methods

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Non-Parametric Statistics** | N/A | No distribution assumptions; rank-based tests | [Wiki - Non-parametric](https://en.wikipedia.org/wiki/Nonparametric_statistics) |
| **Rank-Based Tests** | N/A | Mann-Whitney U, Wilcoxon; robust to outliers | [Wiki - Rank Tests](https://en.wikipedia.org/wiki/Nonparametric_statistics) |
| **Nonparametric Regression** | N/A | LOWESS, splines; flexible function fitting | [Wiki - Nonparametric Regression](https://en.wikipedia.org/wiki/Nonparametric_regression) |

---

## XII. Significance & Effect Size Concepts

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Statistical Significance** | ✓ statistical_vs_practical_significance.md | p < 0.05 indicates statistical evidence; not magnitude | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample) |
| **Practical Significance** | ✓ statistical_vs_practical_significance.md | Real-world importance; effect size matters; large n can show trivial effects | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) |
| **Multiple Comparisons Problem** | N/A | Inflate false positive rate; Bonferroni, FDR correction | [Wiki - Multiple Comparisons](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) |

---

## XIII. Meta-Topics & Foundations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Statistics Core Concepts** | ✓ statistics_core_concepts.md | Integrated overview: hypothesis testing, distributions, CLT, etc. | Internal |
| **Likelihood Function** | N/A | P(data \| θ); maximum likelihood estimation foundation | [Wiki - Likelihood](https://en.wikipedia.org/wiki/Likelihood_function) |
| **Sufficient Statistics** | N/A | Summary stat capturing all data information about parameter | [Wiki - Sufficient Statistic](https://en.wikipedia.org/wiki/Sufficient_statistic) |
| **Fisher Information** | N/A | Curvature of log-likelihood; measures parameter precision | [Wiki - Fisher Info](https://en.wikipedia.org/wiki/Fisher_information) |
| **Kullback-Leibler Divergence** | N/A | Distance between two probability distributions | [Wiki - KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%20%93Leibler_divergence) |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **Wikipedia Outline of Statistics** | https://en.wikipedia.org/wiki/Outline_of_statistics | Comprehensive taxonomy; 100+ topics organized by domain |
| **Khan Academy Statistics & Probability** | https://www.khanacademy.org/math/statistics-probability | 16 units, 157 skills; beginner to intermediate focus |
| **Statistics Done Wrong** | https://www.statisticsdonewrong.com | P-value misuse, multiple comparisons, common errors |

---

## Quick Stats

- **Total Topics Documented**: 80+
- **Workspace Files Created**: 11
- **Categories**: 13
- **External Concepts Added**: ~70
- **Coverage**: Descriptive → Inference → Advanced Computation
