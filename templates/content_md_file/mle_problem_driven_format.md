# Maximum Likelihood Estimation (MLE)

## 1. Learning Goal
**You will be able to:**
- Construct likelihood functions and derive MLE estimators analytically or numerically
- Interpret asymptotic properties (consistency, efficiency, normality) and apply them to inference
- Diagnose when MLE fails and select alternative estimation methods

**Why this matters:**
MLE is the gold standard for parameter estimation across statistics, machine learning, and applied fields. Understanding MLE enables you to build statistical models, construct confidence intervals, and perform hypothesis tests reliably.

## 2. Anchor Problem
**Problem statement:**  
You collect n = 100 measurements of product lifetime (hours). Data appears roughly normally distributed. Your goal: estimate mean μ and std dev σ, along with 95% confidence intervals for μ.

**Desired output:**
- Point estimates: μ̂, σ̂
- Confidence interval: [μ̂ - 1.96·SE, μ̂ + 1.96·SE]
- Visualization showing log-likelihood surface and estimated parameters

## 3. Minimum Theory
- **Likelihood function:** L(θ|x) = ∏ᵢ f(xᵢ|θ) measures how probable your data is for each θ
- **Log-likelihood:** ℓ(θ) = log L(θ) = Σᵢ log f(xᵢ|θ) is equivalent but easier to optimize
- **Score function:** S(θ) = ∂ℓ/∂θ; set = 0 to find critical points
- **Fisher Information:** I(θ) = E[-∂²ℓ/∂θ²]; inverse gives asymptotic variance of MLE
- **Asymptotic distribution:** √n(θ̂ - θ₀) ~ᵈ N(0, I⁻¹(θ₀)) as n → ∞

## 4. Solution Walkthrough

**Step 1: Set up likelihood for normal distribution**
$$L(\mu, \sigma^2 | x) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**Step 2: Convert to log-likelihood**
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Step 3: Take partial derivatives**
$$\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0 \quad \Rightarrow \quad \hat{\mu} = \bar{x}$$

$$\frac{\partial\ell}{\partial\sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \bar{x})^2 = 0 \quad \Rightarrow \quad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Step 4: Verify maximum (second derivative test)**
$$\frac{\partial^2\ell}{\partial\mu^2} = -\frac{n}{\sigma^2} < 0 \quad \checkmark$$

**Step 5: Compute standard errors via Fisher Information**
$$SE(\hat{\mu}) = \frac{\hat{\sigma}}{\sqrt{n}}, \quad SE(\hat{\sigma}^2) = \sqrt{\frac{2\sigma^4}{n}}$$

## 5. Validation

**Sanity check:**  
- MLE for normal: μ̂ = sample mean ✓ (intuitive)
- σ̂² = biased version of sample variance (uses n, not n-1) ✓
- As n increases, SE → 0 ✓

**Edge check:**
- Small n (n=5): MLE estimates will be unstable; confidence intervals wide (as expected)
- Large outlier: Both μ̂ and σ̂ shift; MLE is not robust
- Truly normal data: CI coverage ≈ 95% in repeated sampling ✓

## 6. Variations

**Easier:**
- Estimate λ from Poisson data: μ̂ = sample mean (same formula as normal)
- Estimate p from Bernoulli: p̂ = proportion of 1s

**Harder:**
- Mixture of two normals: N(μ₁, σ₁²) + (1-π)N(μ₂, σ₂²); use EM algorithm
- Censored data (survival analysis): likelihood includes both observed and censored observations
- Constrained MLE (e.g., monotonicity): use constrained optimization

**Real data:**
- Fit normal to log-transformed stock returns; compare with other distributions (Student's t, skew-normal)
- Check normality assumption via Q-Q plot before using normal-based confidence intervals

## 7. Retrieval Practice

- **Q1:** What is the log-likelihood for data x₁, ..., xₙ ~ Exp(λ)?  
  **A:** ℓ(λ) = n log λ - λ Σxᵢ; MLE is λ̂ = 1/x̄

- **Q2:** Why is it wrong to say "likelihood is the probability of θ given data"?  
  **A:** Likelihood L(θ|x) is P(data|θ), not P(θ|data). P(θ|data) is posterior (Bayesian). Likelihood treats θ as variable, data as fixed.

- **Q3:** Under what conditions might Method of Moments outperform MLE?  
  **A:** Small samples with model misspecification; MLE fragile to wrong assumptions, MM more robust

## 8. Key Reference
- [MLE Comprehensive Review (Statlect)](https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood) - Theory, multivariate examples, efficiency proofs

---
**Status:** Core | **Complements:** Bayesian Inference, Hypothesis Testing, Confidence Intervals, EM Algorithm
