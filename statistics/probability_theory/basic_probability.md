# Basic Probability

## 1.1 Concept Skeleton
**Definition:** Quantifies likelihood of events; P(A) = favorable outcomes / total outcomes in sample space  
**Purpose:** Formalize uncertainty, predict long-run frequencies, foundation for statistical inference  
**Prerequisites:** Set theory, fractions, sample spaces, counting

## 1.2 Comparative Framing
| Framework | Classical Probability | Frequentist Probability | Subjective Probability | Axiomatic Probability |
|-----------|---------------------|----------------------|---------------------|---------------------|
| **Definition** | Favorable/total equally likely | Long-run relative frequency | Personal belief degree | Kolmogorov axioms |
| **Assumption** | Outcomes equally likely | Infinite repetitions possible | Prior knowledge available | Mathematical consistency |
| **Example** | Fair coin P(H)=0.5 | Flip 10,000 times → 0.498 | Expert estimates 0.6 | P: Ω→[0,1] satisfying axioms |
| **Limitation** | Requires symmetry | Can't repeat all experiments | Subjective | Abstract framework |

## 1.3 Examples + Counterexamples

**Simple Example:**  
Fair die: P(rolling 4) = 1/6. Sample space Ω={1,2,3,4,5,6}, favorable outcomes={4}

**Failure Case:**  
Assuming all outcomes equally likely without justification. Loaded die: P(6)≠1/6; need empirical data or physics model

**Edge Case:**  
Zero probability events still possible. Dart hitting exact point on continuous target: P(exact point)=0 but can occur

## 1.4 Layer Breakdown
```
Probability Foundations:
├─ Sample Space (Ω): Set of all possible outcomes
│   ├─ Discrete: Finite or countable (coin flips, die rolls)
│   └─ Continuous: Uncountable (heights, times, temperatures)
├─ Event (A): Subset of sample space
│   ├─ Simple event: Single outcome
│   ├─ Compound event: Multiple outcomes
│   └─ Empty event (∅): Impossible
├─ Probability Function P(·):
│   ├─ Axiom 1: P(A) ≥ 0 for all events A
│   ├─ Axiom 2: P(Ω) = 1 (certainty)
│   └─ Axiom 3: P(A∪B) = P(A) + P(B) if A∩B = ∅ (disjoint)
├─ Basic Rules:
│   ├─ Complement: P(A') = 1 - P(A)
│   ├─ Addition (general): P(A∪B) = P(A) + P(B) - P(A∩B)
│   ├─ Impossibility: P(∅) = 0
│   └─ Bounds: 0 ≤ P(A) ≤ 1
└─ Classical Formula:
    └─ P(A) = |A| / |Ω| (when outcomes equally likely)
```

## 1.5 Mini-Project
Simulate basic probability experiments:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)

# Experiment 1: Fair coin flips
print("=== Fair Coin Experiment ===")
n_flips = 10000
flips = np.random.choice(['H', 'T'], size=n_flips)
p_heads = np.mean(flips == 'H')
print(f"Theoretical P(H) = 0.500")
print(f"Empirical P(H) = {p_heads:.3f} ({n_flips} flips)")

# Experiment 2: Fair die rolls
print("\n=== Fair Die Experiment ===")
n_rolls = 10000
rolls = np.random.randint(1, 7, size=n_rolls)
counts = Counter(rolls)
print("Theoretical probabilities: 1/6 ≈ 0.167 for each outcome")
for outcome in range(1, 7):
    p_empirical = counts[outcome] / n_rolls
    print(f"P({outcome}) = {p_empirical:.3f}")

# Experiment 3: Multiple events
print("\n=== Compound Events ===")
# P(even number) = P(2 ∪ 4 ∪ 6) = 3/6 = 0.5
p_even_theoretical = 3/6
p_even_empirical = np.mean(rolls % 2 == 0)
print(f"P(even) theoretical: {p_even_theoretical:.3f}")
print(f"P(even) empirical: {p_even_empirical:.3f}")

# P(>4) = P(5 ∪ 6) = 2/6 ≈ 0.333
p_gt4_theoretical = 2/6
p_gt4_empirical = np.mean(rolls > 4)
print(f"P(>4) theoretical: {p_gt4_theoretical:.3f}")
print(f"P(>4) empirical: {p_gt4_empirical:.3f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Convergence to true probability (coin)
n_range = np.logspace(1, 4, 100, dtype=int)
running_prob = []
cumsum_heads = 0

for n in n_range:
    if len(running_prob) == 0:
        new_flips = np.random.choice([0, 1], size=n)
    else:
        prev_n = n_range[len(running_prob)-1] if len(running_prob) > 0 else 0
        new_flips = np.random.choice([0, 1], size=n-prev_n)
    
    cumsum_heads += np.sum(new_flips)
    running_prob.append(cumsum_heads / n)

axes[0, 0].plot(n_range, running_prob, linewidth=2)
axes[0, 0].axhline(0.5, color='r', linestyle='--', label='True P(H)=0.5')
axes[0, 0].set_xlabel('Number of Flips')
axes[0, 0].set_ylabel('Proportion of Heads')
axes[0, 0].set_title('Convergence to True Probability')
axes[0, 0].set_xscale('log')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Die roll distribution
axes[0, 1].bar(counts.keys(), [counts[i]/n_rolls for i in counts.keys()], 
               edgecolor='black', alpha=0.7)
axes[0, 1].axhline(1/6, color='r', linestyle='--', label='Theoretical 1/6')
axes[0, 1].set_xlabel('Die Outcome')
axes[0, 1].set_ylabel('Probability')
axes[0, 1].set_title(f'Empirical Die Distribution (n={n_rolls})')
axes[0, 1].set_xticks(range(1, 7))
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Venn diagram simulation (Addition rule)
print("\n=== Addition Rule Verification ===")
# Event A: roll ≤ 3, Event B: roll is even
n_sim = 10000
rolls_sim = np.random.randint(1, 7, size=n_sim)
A = rolls_sim <= 3  # {1, 2, 3}
B = rolls_sim % 2 == 0  # {2, 4, 6}
A_and_B = A & B  # {2}
A_or_B = A | B  # {1, 2, 3, 4, 6}

p_A = np.mean(A)
p_B = np.mean(B)
p_A_and_B = np.mean(A_and_B)
p_A_or_B = np.mean(A_or_B)

print(f"P(A) = P(≤3) = {p_A:.3f} (theoretical: {3/6:.3f})")
print(f"P(B) = P(even) = {p_B:.3f} (theoretical: {3/6:.3f})")
print(f"P(A∩B) = {p_A_and_B:.3f} (theoretical: {1/6:.3f})")
print(f"P(A∪B) = {p_A_or_B:.3f} (theoretical: {5/6:.3f})")
print(f"Addition rule check: P(A)+P(B)-P(A∩B) = {p_A+p_B-p_A_and_B:.3f}")

# Venn diagram style
venn_data = {'A only': p_A - p_A_and_B,
             'B only': p_B - p_A_and_B,
             'A and B': p_A_and_B,
             'Neither': 1 - p_A_or_B}
axes[0, 2].bar(venn_data.keys(), venn_data.values(), edgecolor='black', alpha=0.7)
axes[0, 2].set_ylabel('Probability')
axes[0, 2].set_title('Event Decomposition\nA: ≤3, B: even')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Plot 4: Complement rule
axes[1, 0].bar(['P(A)', 'P(A\')'], [p_A, 1-p_A], color=['blue', 'orange'], 
               edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Probability')
axes[1, 0].set_title(f'Complement Rule\nP(A) + P(A\') = {p_A + (1-p_A):.1f}')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 5: Sample size effect on accuracy
sample_sizes = [10, 50, 100, 500, 1000, 5000]
errors = []

for n in sample_sizes:
    trials = []
    for _ in range(100):
        sample_rolls = np.random.randint(1, 7, size=n)
        p_est = np.mean(sample_rolls == 4)
        trials.append(abs(p_est - 1/6))
    errors.append(np.mean(trials))

axes[1, 1].plot(sample_sizes, errors, 'o-', linewidth=2)
axes[1, 1].set_xlabel('Sample Size')
axes[1, 1].set_ylabel('Mean Absolute Error')
axes[1, 1].set_title('Estimation Accuracy vs Sample Size')
axes[1, 1].set_xscale('log')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Probability distributions comparison
outcomes = np.arange(1, 7)
uniform = np.ones(6) / 6
loaded = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3])

x = np.arange(len(outcomes))
width = 0.35
axes[1, 2].bar(x - width/2, uniform, width, label='Fair die', 
               edgecolor='black', alpha=0.7)
axes[1, 2].bar(x + width/2, loaded, width, label='Loaded die', 
               edgecolor='black', alpha=0.7)
axes[1, 2].set_xlabel('Outcome')
axes[1, 2].set_ylabel('Probability')
axes[1, 2].set_title('Fair vs Loaded Die')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(outcomes)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Experiment 4: Non-equally likely outcomes
print("\n=== Non-Uniform Distribution Example ===")
# Biased coin: P(H) = 0.7
n_biased = 10000
biased_flips = np.random.choice(['H', 'T'], size=n_biased, p=[0.7, 0.3])
p_heads_biased = np.mean(biased_flips == 'H')
print(f"Biased coin: Theoretical P(H) = 0.700")
print(f"Empirical P(H) = {p_heads_biased:.3f}")
print("Note: Classical formula P(A)=|A|/|Ω| requires equally likely outcomes")
```

## 1.6 Challenge Round
When is basic probability inadequate?
- **Dependent events**: Card drawing without replacement → need conditional probability
- **Continuous outcomes**: Dart position on board → probability density functions, not P(A)=|A|/|Ω|
- **Subjective uncertainty**: Election outcomes → Bayesian/subjective probability more appropriate
- **Complex systems**: Weather prediction → need stochastic processes, time series models
- **Rare events**: Extreme outcomes (pandemics, financial crashes) → fat-tailed distributions, not simple counting

## 1.7 Key References
- [Khan Academy Probability Basics](https://www.khanacademy.org/math/statistics-probability/probability-library) - Interactive lessons on sample spaces, events, basic rules
- [Probability Axioms Explained](https://en.wikipedia.org/wiki/Probability_axioms) - Kolmogorov axioms, formal foundation of probability theory
- [Probability Interpretation Types](https://plato.stanford.edu/entries/probability-interpret/) - Classical, frequentist, subjective, and logical probability philosophies

---
**Status:** Foundation for all probability theory | **Complements:** Conditional Probability, Random Variables, Statistical Inference
