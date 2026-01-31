# Maximum Likelihood Estimation (MLE)

## 1. Concept Snapshot
**Definition:** Find θ̂ that maximizes L(θ|x) = ∏f(xᵢ|θ); most widely used parameter estimation method
**Why it matters:** Produces asymptotically unbiased, efficient estimators; foundation for hypothesis testing, confidence intervals, model selection
**Prerequisites:** Probability distributions, derivatives, logarithms, basic calculus

## 2. Core Idea in 3 Steps
1. **Write likelihood:** L(θ) = ∏ᵢ f(xᵢ|θ) = probability of observing your data given θ
2. **Optimize:** Take derivative ∂ℓ/∂θ = 0 where ℓ = log L (easier to work with)
3. **Solve for θ̂:** Analytical solution (set derivative = 0) or numerical optimization

## 3. One Worked Example
**Problem:** Data [3, 5, 4, 6, 2] from Poisson(λ). Estimate λ.

**Solution:** 
- Log-likelihood: ℓ(λ) = Σ(xᵢ log λ - λ)
- Derivative: ∂ℓ/∂λ = Σxᵢ/λ - n = 0
- Result: λ̂ = mean(x) = (3+5+4+6+2)/5 = **4**

**Takeaway:** For many common distributions, MLE yields familiar statistics (sample mean for Poisson/exponential, sample proportion for Bernoulli)

## 4. Common Pitfalls
- **Confusing likelihood with probability:** L(θ|x) is NOT P(θ|x); it's a function of θ given fixed data
- **Ignoring boundary solutions:** Derivative = 0 fails for Uniform(0, θ); check boundaries too
- **Bias in small samples:** MLE of σ² uses n, not n-1; biased for small n
- **Numerical instability:** Optimize log-likelihood, not likelihood (prevents underflow)
- **Assuming MLE is always efficient:** Only true asymptotically; can be outperformed by biased estimators in finite samples

## 5. Quick Check
- **Q: What is the likelihood function?**  
  A: Probability of observed data as function of unknown parameter: L(θ) = P(data|θ)
- **Q: Why use log-likelihood instead of likelihood?**  
  A: Converts products to sums (easier calculus); prevents numerical underflow
- **Q: Is MLE always unbiased?**  
  A: No; asymptotically unbiased, but biased in small samples (e.g., σ² estimator)

## 6. Key Reference
- [Maximum Likelihood (MrExcel Statistics)](https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood) - Concise theory with worked examples for multiple distributions

---
**Status:** Core | **Complements:** Bayesian Inference, Confidence Intervals, Hypothesis Testing
