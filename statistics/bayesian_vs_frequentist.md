# Bayesian vs Frequentist Statistics

## 10.1 Concept Skeleton
**Definition:**  
- Frequentist: Probability = long-run frequency; inference without priors
- Bayesian: Probability = degree of belief; incorporates prior knowledge via Bayes rule

**Purpose:** Different frameworks for statistical inference with different interpretations  
**Prerequisites:** Conditional probability, prior distributions, Bayes theorem

## 10.2 Comparative Framing
| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Probability** | Long-run frequency | Degree of belief |
| **Parameters** | Fixed but unknown | Random variables (distributions) |
| **Prior** | Not used | Essential input |
| **Inference** | p-values, CI, hypothesis tests | Posterior distributions |
| **Interpretation** | "If repeated infinitely, 95% CI contain true param" | "Given data, 95% probability param in range" |
| **Advantage** | Objective (no prior needed) | Incorporates domain knowledge |
| **Disadvantage** | Counterintuitive interpretation | Prior choice subjective |

## 10.3 Examples + Counterexamples

**Frequentist Approach:**  
Test: "Is coin fair?" H₀: p=0.5, observe 12 heads in 20 flips, p=0.058 > 0.05. Fail to reject (not enough evidence)

**Bayesian Approach:**  
Prior: Coin probably fair (p~Beta(10,10)). Data: 12 heads in 20. Posterior: p probably ~0.55-0.60 given data

**Edge Case:**  
Optional stopping: Frequentist p-value depends on stopping rule (when you decide to stop). Bayesian posterior unaffected!

## 10.4 Layer Breakdown
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

## 10.5 Mini-Project
Simple Bayesian inference:
```python
import numpy as np
from scipy.special import beta as beta_function
import matplotlib.pyplot as plt

# Coin flips: 7 heads in 10 flips
data_heads, data_tails = 7, 3

# Prior: Beta(1,1) = uniform (no preference)
prior_a, prior_b = 1, 1

# Posterior: Beta(prior_a + heads, prior_b + tails)
posterior_a = prior_a + data_heads
posterior_b = prior_b + data_tails

# Posterior mean (updated estimate)
posterior_mean = posterior_a / (posterior_a + posterior_b)

print(f"Prior mean (p): {prior_a / (prior_a + prior_b):.2f}")
print(f"Posterior mean (p): {posterior_mean:.2f}")
print(f"Data suggested: {data_heads / (data_heads + data_tails):.2f}")

# Visualize
x = np.linspace(0, 1, 1000)
prior_pdf = (x**(prior_a-1) * (1-x)**(prior_b-1)) / beta_function(prior_a, prior_b)
posterior_pdf = (x**(posterior_a-1) * (1-x)**(posterior_b-1)) / beta_function(posterior_a, posterior_b)

plt.plot(x, prior_pdf, label='Prior (Beta(1,1))', alpha=0.5)
plt.plot(x, posterior_pdf, label='Posterior (Beta(8,4))', linewidth=2)
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Density')
plt.title('Bayesian Update: Prior + Data → Posterior')
plt.legend()
plt.show()
```

Sequential Bayesian Update (more data):
```python
# Update with more evidence
for n_flips in [10, 20, 50, 100]:
    heads = int(0.7 * n_flips)  # Assume true p=0.7
    posterior_a = 1 + heads
    posterior_b = 1 + (n_flips - heads)
    posterior_mean = posterior_a / (posterior_a + posterior_b)
    print(f"After {n_flips} flips: posterior mean = {posterior_mean:.3f}")
```

## 10.6 Challenge Round
When choose Frequentist?
- Large data (differences between approaches vanish)
- Standardized, objective protocol needed
- Prior too controversial or expensive to elicit

When choose Bayesian?
- Small sample, strong prior knowledge available
- Sequential decision-making (updating)
- Need probability statements about parameters
- Incorporating expert opinion required
- Optional stopping not a problem

## 10.7 Key References
- [Bayes Theorem Intuition](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Frequentist vs Bayesian Comparison](https://www.nature.com/articles/d41586-020-00609-0)
- [Bayesian Inference Tutorial](https://www.probabilistic-programming.org)
- [Practical Bayesian Examples](https://mc-stan.org/docs/tutorials)

---
**Status:** Competing frameworks | **Complements:** All inference methods, Study Design
