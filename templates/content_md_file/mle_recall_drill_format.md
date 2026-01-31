# Maximum Likelihood Estimation (MLE)

## 1. One-Page Overview
**Definition:** Parameter estimation method selecting θ̂ to maximize L(θ|data) = ∏ f(xᵢ|θ)
**Key formulas:**
- Likelihood: $L(\theta|x) = \prod_{i=1}^{n} f(x_i|\theta)$
- Log-likelihood: $\ell(\theta) = \sum_{i=1}^{n} \log f(x_i|\theta)$
- Score: $S(\theta) = \frac{d\ell}{d\theta}$; set = 0 for MLE
- Asymptotic distribution: $\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, I^{-1}(\theta_0))$

**Assumptions:**
- Correct distributional form f(·|θ) specified
- Sample iid from f(·|θ₀)
- Parameter space Θ compact
- Regularity conditions for asymptotic normality

## 2. Concept Map (Text)
- **Maximum Likelihood Estimation**
  - **Core principle:** Find θ maximizing P(data|θ)
    - Likelihood vs probability: L(θ|x) ≠ P(θ|x)
    - Log-likelihood for numerical stability
      - Converts products → sums (easier calculus)
      - Prevents floating-point underflow
  - **Solving for MLE**
    - Analytical: ∂ℓ/∂θ = 0; solve algebraically
    - Numerical: Optimization algorithms (Newton-Raphson, gradient descent)
    - Boundary: Check edges if interior optimum fails
  - **Properties (asymptotic)**
    - Consistency: θ̂ →ᵖ θ₀
    - Efficiency: Achieves Cramér-Rao lower bound
    - Normality: √n(θ̂ - θ₀) ~ᵈ N(0, I⁻¹(θ₀))
    - Invariance: g(θ̂) = MLE of g(θ)
  - **When MLE fails**
    - Biased in small samples (e.g., σ̂² uses n not n-1)
    - Not robust to outliers
    - Boundary issues (e.g., Uniform(0,θ))
    - Model misspecification fragility

## 3. Micro-Examples (3)

**Example 1: Bernoulli(p) with data [1, 0, 1, 1, 0]**
- Likelihood: L(p) = p³(1-p)²
- Log-likelihood: ℓ(p) = 3 log p + 2 log(1-p)
- Score: S(p) = 3/p - 2/(1-p) = 0 → **p̂ = 3/5 = 0.6**

**Example 2: Exponential(λ) with data [2.1, 3.4, 1.8]**
- Log-likelihood: ℓ(λ) = 3 log λ - λ(2.1 + 3.4 + 1.8) = 3 log λ - 7.3λ
- Score: S(λ) = 3/λ - 7.3 = 0 → **λ̂ = 3/7.3 ≈ 0.41**

**Example 3: Normal(μ, σ²) — μ unknown, σ² known**
- Log-likelihood: ℓ(μ) = -n/(2σ²) Σ(xᵢ - μ)²
- Score: S(μ) = (1/σ²) Σ(xᵢ - μ) = 0 → **μ̂ = x̄** (sample mean)

## 4. Error Triggers
- **Confusing likelihood with posterior:** L(θ|x) is NOT P(θ|x). Likelihood fixed x, varies θ. Posterior is Bayesian.
- **Forgetting to check second derivative:** Critical point may be minimum or saddle; verify ∂²ℓ/∂θ² < 0
- **Using likelihood instead of log-likelihood:** Products underflow numerically; always work in log space
- **Believing MLE always unbiased:** True asymptotically; false in finite samples (σ̂² biased downward)
- **Ignoring boundary solutions:** Derivative approach misses boundaries; Uniform(0, θ) MLE = max(x)

## 5. Flashcards

| Q | A |
|---|---|
| **Q: What is the likelihood function?** | A: Joint probability density of data treated as function of θ: L(θ\|x) = P(x\|θ) |
| **Q: Why maximize log-likelihood instead of likelihood?** | A: Numerical stability; log converts ∏ to Σ (easier calculus, prevents underflow) |
| **Q: Is MLE always unbiased?** | A: No; asymptotically unbiased. Finite-sample bias exists (e.g., σ̂² = n⁻¹Σ(xᵢ-x̄)² biased) |
| **Q: What is Fisher Information?** | A: I(θ) = E[−∂²ℓ/∂θ²]; inverse is asymptotic variance of MLE |
| **Q: When does MLE fail?** | A: Model misspecification, boundary constraints, outliers/non-robustness, small samples |
| **Q: What is the score function?** | A: S(θ) = ∂ℓ/∂θ; set = 0 to find critical points |
| **Q: Can MLE handle censored data?** | A: Yes; likelihood includes both observed (f) and censored (1-F) components |
| **Q: What is asymptotic efficiency of MLE?** | A: Minimum-variance unbiased estimator among all estimators (Cramér-Rao lower bound) |

## 6. Self-Test

- **T/F:** The likelihood function is always a probability distribution over θ.  
  **Answer:** FALSE. Likelihood is not normalized (∫L(θ)dθ ≠ 1); it's density of data for fixed θ.

- **T/F:** MLE of σ using √[n⁻¹Σ(xᵢ-x̄)²] is unbiased for normal data.  
  **Answer:** FALSE. This is MLE for σ but biased. Unbiased uses √[n⁻¹ * c₄(n)] where c₄ ≈ 0.9 for n=10.

- **Short answer:** If θ̂ is MLE of θ, what is the MLE of θ²?  
  **Answer:** (θ̂)² by invariance property of MLE.

## 7. Key Reference
- [MLE Fundamentals (Statlect)](https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood) - Definitions, properties, examples across distributions

---
**Status:** Core | **Complements:** Fisher Information, Bayesian Inference, Hypothesis Testing, Confidence Intervals
