# Decision Theory

## 2.1 Concept Skeleton
**Definition:** Framework for choosing optimal actions under uncertainty by minimizing expected loss or maximizing expected utility  
**Purpose:** Formalize decision-making with incomplete information, balance Type I/II errors, quantify trade-offs  
**Prerequisites:** Probability theory, expected value, loss functions, Bayes' theorem

## 2.2 Comparative Framing
| Framework | Decision Theory | Hypothesis Testing | Cost-Benefit Analysis |
|-----------|----------------|-------------------|---------------------|
| **Focus** | Optimal action under uncertainty | Accept/reject null hypothesis | Economic value comparison |
| **Loss Function** | Explicit (0-1, quadratic, etc.) | Implicit (α, β errors) | Monetary costs/benefits |
| **Prior Info** | Can incorporate (Bayesian) | Typically ignored | Market/historical data |
| **Output** | Action with min expected loss | Binary decision (p-value) | Net present value |

## 2.3 Examples + Counterexamples

**Simple Example:**  
Medical test: Treat if P(disease|test+) > 0.1. False positive cost=$1000, false negative cost=$100,000. Decision threshold balances risks

**Failure Case:**  
Equal loss for false positive/negative when consequences differ drastically (e.g., spam filter vs cancer diagnosis). Ignoring asymmetric costs

**Edge Case:**  
Minimax strategy: Choose action minimizing worst-case loss. Conservative but ignores probability distribution of outcomes (too pessimistic)

## 2.4 Layer Breakdown
```
Decision Theory Components:
├─ States of Nature (θ): Unknown truth (disease present/absent)
├─ Actions (a): Decisions available (treat, don't treat)
├─ Loss Function L(θ, a): Cost of action a when truth is θ
│   ├─ 0-1 Loss: L = 1 if wrong, 0 if correct
│   ├─ Quadratic Loss: L = (θ - a)²
│   ├─ Absolute Loss: L = |θ - a|
│   └─ Asymmetric Loss: Different penalties for false positive/negative
├─ Risk Function: R(θ, δ) = E[L(θ, δ(X)) | θ]
│       └─ Expected loss for decision rule δ
├─ Bayesian Approach:
│   ├─ Prior: π(θ)
│   ├─ Posterior: π(θ|x) ∝ P(x|θ)π(θ)
│   └─ Bayes Risk: r(δ) = ∫ R(θ, δ)π(θ)dθ
│       └─ Choose δ* minimizing Bayes risk
├─ Frequentist Approach:
│   ├─ Minimax: min_δ max_θ R(θ, δ)
│   └─ Admissibility: No other rule uniformly better
└─ Type I/II Errors as Special Case:
    ├─ Type I (α): P(reject H₀ | H₀ true)
    ├─ Type II (β): P(accept H₀ | H₁ true)
    └─ Power = 1 - β: P(reject H₀ | H₁ true)
```

## 2.5 Mini-Project
Medical testing decision with asymmetric costs:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Scenario: Disease test with false positive/negative costs
# True disease prevalence
prevalence = 0.05  # 5% have disease

# Test characteristics
sensitivity = 0.95  # P(test+ | disease)
specificity = 0.90  # P(test- | no disease)

# Loss function (costs in $)
cost_false_positive = 1000  # Unnecessary treatment
cost_false_negative = 50000  # Missed disease
cost_true_positive = 5000   # Correct treatment
cost_true_negative = 0      # No action needed

# Bayes' theorem: P(disease | test+)
p_test_pos_given_disease = sensitivity
p_test_pos_given_no_disease = 1 - specificity
p_test_pos = (p_test_pos_given_disease * prevalence + 
              p_test_pos_given_no_disease * (1 - prevalence))
p_disease_given_test_pos = (p_test_pos_given_disease * prevalence / 
                            p_test_pos)

print(f"P(disease | test+) = {p_disease_given_test_pos:.3f}")

# Decision rule: Treat if P(disease|test+) > threshold
# Expected loss for "treat" action after test+:
loss_treat = (p_disease_given_test_pos * cost_true_positive + 
              (1 - p_disease_given_test_pos) * cost_false_positive)

# Expected loss for "don't treat" action after test+:
loss_no_treat = (p_disease_given_test_pos * cost_false_negative + 
                 (1 - p_disease_given_test_pos) * cost_true_negative)

print(f"\nExpected loss if treat: ${loss_treat:.0f}")
print(f"Expected loss if don't treat: ${loss_no_treat:.0f}")

optimal_action = "TREAT" if loss_treat < loss_no_treat else "DON'T TREAT"
print(f"Optimal decision: {optimal_action}")

# Decision threshold analysis
thresholds = np.linspace(0, 1, 100)
expected_loss_values = []

for threshold in thresholds:
    # Expected loss across all possible test outcomes
    # If P(disease|test+) > threshold, treat; else don't treat
    if p_disease_given_test_pos > threshold:
        loss = loss_treat
    else:
        loss = loss_no_treat
    expected_loss_values.append(loss)

plt.figure(figsize=(12, 5))

# Plot 1: Expected loss vs threshold
plt.subplot(1, 2, 1)
optimal_threshold = cost_false_positive / (cost_false_positive + cost_false_negative)
plt.axvline(optimal_threshold, color='r', linestyle='--', 
            label=f'Optimal threshold={optimal_threshold:.3f}')
plt.axvline(p_disease_given_test_pos, color='g', linestyle='--', 
            label=f'P(disease|test+)={p_disease_given_test_pos:.3f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Expected Loss ($)')
plt.title('Expected Loss vs Decision Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: ROC-style analysis
sensitivities = np.linspace(0.5, 1, 20)
specificities = np.linspace(0.5, 1, 20)
SEN, SPE = np.meshgrid(sensitivities, specificities)
expected_loss_grid = np.zeros_like(SEN)

for i in range(len(sensitivities)):
    for j in range(len(specificities)):
        sens, spec = SEN[j, i], SPE[j, i]
        # Recalculate posterior
        p_tp = (sens * prevalence / 
                (sens * prevalence + (1-spec) * (1-prevalence)))
        # Expected loss with optimal decision
        loss_t = p_tp * cost_true_positive + (1-p_tp) * cost_false_positive
        loss_nt = p_tp * cost_false_negative + (1-p_tp) * cost_true_negative
        expected_loss_grid[j, i] = min(loss_t, loss_nt)

plt.subplot(1, 2, 2)
plt.contourf(SEN, SPE, expected_loss_grid, levels=15, cmap='RdYlGn_r')
plt.colorbar(label='Min Expected Loss ($)')
plt.plot(sensitivity, specificity, 'r*', markersize=15, label='Current test')
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.title('Expected Loss vs Test Performance')
plt.legend()
plt.tight_layout()
plt.show()

# Minimax vs Bayes comparison
print("\n--- Decision Rule Comparison ---")
# Bayes rule: minimize expected loss with prior
bayes_decision = "Treat if test+" if loss_treat < loss_no_treat else "Don't treat"
print(f"Bayes Rule (incorporates prior): {bayes_decision}")

# Minimax: minimize worst-case loss
worst_case_treat = max(cost_true_positive, cost_false_positive)
worst_case_no_treat = max(cost_false_negative, cost_true_negative)
minimax_decision = "Treat" if worst_case_treat < worst_case_no_treat else "Don't treat"
print(f"Minimax Rule (worst-case): {minimax_decision}")
print(f"Worst-case treat: ${worst_case_treat}, don't treat: ${worst_case_no_treat}")
```

## 2.6 Challenge Round
When is decision theory the wrong approach?
- Incomplete loss specification: Can't quantify all costs (e.g., emotional suffering, reputation)
- Adversarial settings: Game theory better when opponent adapts to your strategy
- Exploratory analysis: Want to understand data, not make immediate decision
- Computational limits: Complex state/action spaces intractable (use heuristics, RL)
- Ethical constraints: Some actions forbidden regardless of expected loss (deontological ethics)

## 2.7 Key References
- [Decision Theory Overview](https://en.wikipedia.org/wiki/Decision_theory) - Bayesian vs frequentist, loss functions, minimax
- [Type I/II Errors in Decision Framework](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/type-i-and-type-ii-errors-decision-theory/) - Connection to hypothesis testing
- [Medical Decision Making](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2727357/) - Real-world application with asymmetric costs

---
**Status:** Unifying framework for inference | **Complements:** Hypothesis Testing, Bayesian Inference, Cost-Benefit Analysis
