# Maximum Likelihood Estimation (MLE)

## 1. Concept Skeleton
**Definition:** Parameter estimation method that finds θ̂ maximizing L(θ|x) = P(data|θ), producing estimators with consistency, efficiency, and asymptotic normality properties
**Purpose:** Obtain statistically optimal parameter estimates; foundational for hypothesis testing, confidence intervals, and model comparison
**Prerequisites:** Probability distributions, calculus (partial derivatives), conditional probability, logarithmic functions

## 2. Comparative Framing
| Aspect | Maximum Likelihood | Method of Moments | Bayesian MAP | Least Squares |
|---|---|---|---|---|
| **Optimizes** | P(data\|θ) | Sample moments = theory moments | P(θ\|data) with prior | Residual sum of squares |
| **Requires prior** | No | No | Yes | No |
| **Computational method** | Optimization (gradient=0) | Algebraic equations | Integration/MCMC | Direct solution |
| **Asymptotic properties** | Efficient, unbiased | Less efficient | Incorporates knowledge | Depends on model |
| **Robustness** | Sensitive to outliers | More robust | Prior-dependent | Robust to some outliers |

## 3. Examples + Counterexamples

**Simple Example: Poisson Distribution**
Given data [3, 5, 4, 6, 2], estimate λ for Poisson(λ).
- Likelihood: $L(\lambda) = \prod_{i=1}^{5} \frac{e^{-\lambda}\lambda^{x_i}}{x_i!}$
- Log-likelihood: $\ell(\lambda) = \sum (x_i \log \lambda - \lambda - \log(x_i!))$
- Score: $\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - 5 = 0$
- Solution: $\hat{\lambda} = \text{mean}(x) = 4$

**Failure Case: Uniform(0, θ) Distribution**
Sample: [2.1, 4.3, 1.8, 3.9]
- Standard approach: Take derivative → no interior critical point
- Boundary maximum: $\hat{\theta} = \max(\text{sample}) = 4.3$
- Problem: This is a biased estimator; true maximum is always ≥ sample maximum
- Fix: Use $\hat{\theta} = 1.25 \times \max(\text{sample})$ for unbiased adjustment

**Edge Case: Normal Distribution with Outliers**
Data: 98 values from N(10, 2²) + 1 outlier = 100
- MLE gives $\hat{\mu} = 10.5, \hat{\sigma} = 2.8$ (pulled by outlier)
- Robust alternative (median, MAD): $\tilde{\mu} = 10.0, \tilde{\sigma} = 2.0$
- Trade-off: MLE is efficient in standard conditions; breaks under contamination

## 4. Layer Breakdown

**Layer 1 (Intuition):** 
Find parameter values that make your observed data most probable. If you computed a likelihood for each possible μ, the MLE picks the tallest peak.

**Layer 2 (Mechanics):**
1. Write likelihood: $L(\theta|x) = \prod_{i=1}^{n} f(x_i|\theta)$
2. Convert to log-likelihood: $\ell(\theta) = \sum_{i=1}^{n} \log f(x_i|\theta)$ (easier to optimize)
3. Take derivative: $\frac{d\ell}{d\theta} = 0$ (score function = 0)
4. Solve for θ̂ analytically or numerically
5. Verify: $\frac{d^2\ell}{d\theta^2} < 0$ (confirms maximum, not minimum/saddle)

**Layer 3 (Properties):**
- **Consistency:** As n → ∞, θ̂ → θ₀ (true parameter)
- **Asymptotic Normality:** $\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, I^{-1}(\theta_0))$ where I is Fisher Information
- **Efficiency:** Among unbiased estimators, MLE has minimum variance (achieves Cramér-Rao lower bound asymptotically)
- **Invariance:** If η = g(θ), then $\hat{\eta} = g(\hat{\theta})$ (transformation property holds)

## 5. Mini-Project

**Goal:** Fit normal distribution N(μ, σ²) to data using both analytical and numerical MLE

**Data/Inputs:**
```python
np.random.seed(42)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 100)
```

**Procedure:**
1. Compute analytical MLE: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$
2. Numerical verification using scipy.optimize.minimize on negative log-likelihood
3. Compute standard errors via Fisher Information: $SE(\hat{\mu}) = \frac{\hat{\sigma}}{\sqrt{n}}$
4. Plot log-likelihood surface (2D heatmap showing peak at MLE)
5. Construct 95% confidence intervals: $\hat{\mu} \pm 1.96 \cdot SE(\hat{\mu})$

**Output:** 
- Parameter estimates with confidence intervals
- Log-likelihood visualization highlighting MLE location
- Comparison: analytical vs numerical results (should match within tolerance)

## 6. Challenge Round

- **When does MLE fail?** Uniform distribution: MLE = max(sample), which is biased. Small samples: estimator is biased for σ².
- **How does MLE differ from Bayesian?** MLE ignores prior information; Bayesian integrates prior × likelihood.
- **Why log-likelihood instead of likelihood?** Log converts products to sums, enabling calculus; prevents numerical underflow with small probabilities.
- **What if no closed-form exists?** Use numerical optimization: Newton-Raphson, gradient descent, EM algorithm.
- **Is MLE always most efficient?** Only asymptotically; in small samples, biased estimators (e.g., ridge regression) can have lower MSE.

## 7. Key References
- [Maximum Likelihood Estimation (Wikipedia)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) - Theory, properties, worked examples
- [Likelihood Inference (StatQuest with Josh Starmer)](https://www.youtube.com/watch?v=XepXtl9YKwc) - Visual intuition, animation
- [MLE Asymptotic Theory](https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood) - Convergence proofs, efficiency bounds

---
**Status:** Core | **Complements:** Bayesian Inference, Hypothesis Testing, Fisher Information, Confidence Intervals
