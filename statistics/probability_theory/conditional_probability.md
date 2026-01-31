# Conditional Probability

## 2.1 Concept Skeleton
**Definition:** Probability of event A given event B occurred; P(A|B) = P(A∩B) / P(B) where P(B)>0  
**Purpose:** Update probabilities with new information; foundation for Bayes' theorem and inference  
**Prerequisites:** Basic probability, set operations, multiplication, fractions

## 2.2 Comparative Framing
| Concept | Unconditional P(A) | Conditional P(A|B) | Joint P(A∩B) | Bayes' Theorem |
|---------|-------------------|-------------------|--------------|----------------|
| **Meaning** | Probability of A | Probability of A given B | Both A and B occur | P(B|A) from P(A|B) |
| **Formula** | |A|/|Ω| | P(A∩B)/P(B) | P(A)·P(B|A) | P(A|B)·P(B)/P(A) |
| **Use** | No prior info | B is known/given | Neither event known | Reverse conditioning |
| **Example** | P(rain)=0.3 | P(rain|clouds)=0.7 | P(rain AND clouds) | P(clouds|rain) |

## 2.3 Examples + Counterexamples

**Simple Example:**  
Two-card draw: P(2nd card is Ace | 1st is Ace) = 3/51 ≈ 0.059 (reduced deck after conditioning)

**Failure Case:**  
Confusing P(A|B) with P(B|A). P(positive test | disease) ≠ P(disease | positive test). Prosecutor's fallacy

**Edge Case:**  
P(B)=0 makes P(A|B) undefined. Cannot condition on impossible events; need alternative formulations

## 2.4 Layer Breakdown
```
Conditional Probability Framework:
├─ Definition: P(A|B) = P(A∩B) / P(B)
│   └─ Restricts sample space from Ω to B
├─ Multiplication Rule: P(A∩B) = P(A|B)·P(B) = P(B|A)·P(A)
├─ Chain Rule: P(A₁∩A₂∩...∩Aₙ) = P(A₁)·P(A₂|A₁)·P(A₃|A₁∩A₂)·...
├─ Law of Total Probability:
│   └─ P(A) = Σᵢ P(A|Bᵢ)·P(Bᵢ) where {Bᵢ} partitions Ω
├─ Bayes' Theorem:
│   ├─ P(A|B) = P(B|A)·P(A) / P(B)
│   ├─ P(A|B) = P(B|A)·P(A) / [P(B|A)·P(A) + P(B|A')·P(A')]
│   └─ Components:
│       ├─ P(A): Prior probability
│       ├─ P(B|A): Likelihood
│       ├─ P(A|B): Posterior probability
│       └─ P(B): Marginal likelihood (normalizing constant)
└─ Applications:
    ├─ Medical diagnosis (test accuracy)
    ├─ Spam filtering (word frequencies)
    ├─ Weather forecasting (condition on observations)
    └─ Legal reasoning (evidence evaluation)
```

## 2.5 Mini-Project
Implement conditional probability and Bayes' theorem:
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns

np.random.seed(42)

# Example 1: Card drawing without replacement
print("=== Card Drawing Example ===")
# P(2nd Ace | 1st Ace)
p_first_ace = 4/52
p_second_ace_given_first_ace = 3/51
p_both_aces = p_first_ace * p_second_ace_given_first_ace

print(f"P(1st card is Ace) = {p_first_ace:.4f}")
print(f"P(2nd is Ace | 1st is Ace) = {p_second_ace_given_first_ace:.4f}")
print(f"P(both Aces) = {p_both_aces:.6f}")

# Simulation verification
n_trials = 100000
both_aces = 0
second_ace_given_first = 0
first_ace_count = 0

for _ in range(n_trials):
    deck = list(range(52))  # 0-3 are aces
    np.random.shuffle(deck)
    first = deck[0]
    second = deck[1]
    
    if first < 4:  # First is ace
        first_ace_count += 1
        if second < 4:  # Second is also ace
            second_ace_given_first += 1
            both_aces += 1

p_conditional_empirical = second_ace_given_first / first_ace_count if first_ace_count > 0 else 0
print(f"Empirical P(2nd Ace | 1st Ace) = {p_conditional_empirical:.4f}")

# Example 2: Medical test (Bayes' Theorem)
print("\n=== Medical Diagnosis (Bayes' Theorem) ===")
# Disease prevalence
p_disease = 0.01  # 1% have disease

# Test characteristics
sensitivity = 0.95  # P(positive | disease)
specificity = 0.90  # P(negative | no disease)

# Derived probabilities
p_positive_given_disease = sensitivity
p_positive_given_no_disease = 1 - specificity  # False positive rate
p_no_disease = 1 - p_disease

# Law of total probability: P(positive)
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * p_no_disease)

# Bayes' theorem: P(disease | positive)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"Disease prevalence: {p_disease:.1%}")
print(f"Test sensitivity: {sensitivity:.1%}")
print(f"Test specificity: {specificity:.1%}")
print(f"P(positive test) = {p_positive:.4f}")
print(f"P(disease | positive test) = {p_disease_given_positive:.4f} ({p_disease_given_positive:.1%})")
print("Note: Even with positive test, only ~9% actually have disease!")

# Visualizations
fig = plt.figure(figsize=(15, 10))

# Plot 1: Conditional probability tree
ax1 = plt.subplot(2, 3, 1)
ax1.text(0.5, 0.95, 'Medical Test Tree', ha='center', fontsize=14, weight='bold')

# Draw tree structure
ax1.plot([0.2, 0.4], [0.7, 0.8], 'k-', linewidth=2)
ax1.plot([0.2, 0.4], [0.7, 0.6], 'k-', linewidth=2)
ax1.text(0.1, 0.7, f'Start', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue'))
ax1.text(0.42, 0.8, f'Disease\n{p_disease:.2f}', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral'))
ax1.text(0.42, 0.6, f'No Disease\n{p_no_disease:.2f}', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen'))

# Second level
ax1.plot([0.6, 0.8], [0.8, 0.85], 'k-', linewidth=1)
ax1.plot([0.6, 0.8], [0.8, 0.75], 'k-', linewidth=1)
ax1.plot([0.6, 0.8], [0.6, 0.65], 'k-', linewidth=1)
ax1.plot([0.6, 0.8], [0.6, 0.55], 'k-', linewidth=1)

ax1.text(0.82, 0.85, f'+\n{sensitivity:.2f}', fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow'))
ax1.text(0.82, 0.75, f'-\n{1-sensitivity:.2f}', fontsize=8, bbox=dict(boxstyle='round', facecolor='white'))
ax1.text(0.82, 0.65, f'+\n{1-specificity:.2f}', fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow'))
ax1.text(0.82, 0.55, f'-\n{specificity:.2f}', fontsize=8, bbox=dict(boxstyle='round', facecolor='white'))

ax1.set_xlim(0, 1)
ax1.set_ylim(0.4, 1)
ax1.axis('off')

# Plot 2: Bayes' theorem visualization
ax2 = plt.subplot(2, 3, 2)
categories = ['P(D)', 'P(+|D)', 'P(D|+)']
values = [p_disease, sensitivity, p_disease_given_positive]
colors = ['red', 'orange', 'green']
bars = ax2.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
ax2.set_ylabel('Probability')
ax2.set_title('Bayes\' Theorem in Action\nD=Disease, +=Positive Test')
ax2.set_ylim(0, 1)
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}', ha='center', va='bottom')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Sensitivity to prevalence
ax3 = plt.subplot(2, 3, 3)
prevalences = np.linspace(0.001, 0.20, 100)
posteriors = []

for prev in prevalences:
    p_pos = sensitivity * prev + (1-specificity) * (1-prev)
    p_d_given_pos = (sensitivity * prev) / p_pos
    posteriors.append(p_d_given_pos)

ax3.plot(prevalences, posteriors, linewidth=2)
ax3.axvline(p_disease, color='r', linestyle='--', label=f'Current prevalence={p_disease}')
ax3.axhline(p_disease_given_positive, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Disease Prevalence')
ax3.set_ylabel('P(Disease | Positive Test)')
ax3.set_title('Posterior Depends on Prior')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion matrix
ax4 = plt.subplot(2, 3, 4)
n_population = 10000
n_disease = int(n_population * p_disease)
n_no_disease = n_population - n_disease

true_positive = int(n_disease * sensitivity)
false_negative = n_disease - true_positive
false_positive = int(n_no_disease * (1-specificity))
true_negative = n_no_disease - false_positive

confusion = np.array([[true_positive, false_negative],
                     [false_positive, true_negative]])

sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax4,
           xticklabels=['Disease', 'No Disease'],
           yticklabels=['Test +', 'Test -'])
ax4.set_title(f'Confusion Matrix (n={n_population})\nPopulation: {p_disease:.1%} prevalence')

# Plot 5: Law of Total Probability
ax5 = plt.subplot(2, 3, 5)
print("\n=== Law of Total Probability Example ===")
# P(rain) = P(rain|clouds)P(clouds) + P(rain|no clouds)P(no clouds)
p_clouds = 0.40
p_rain_given_clouds = 0.70
p_rain_given_no_clouds = 0.10

p_rain_total = (p_rain_given_clouds * p_clouds + 
                p_rain_given_no_clouds * (1-p_clouds))

print(f"P(rain | clouds) = {p_rain_given_clouds}")
print(f"P(rain | no clouds) = {p_rain_given_no_clouds}")
print(f"P(clouds) = {p_clouds}")
print(f"P(rain) = {p_rain_total:.2f}")

components = ['P(R|C)·P(C)', 'P(R|C\')·P(C\')', 'Total P(R)']
values_total = [p_rain_given_clouds * p_clouds, 
                p_rain_given_no_clouds * (1-p_clouds),
                p_rain_total]
colors_total = ['skyblue', 'lightblue', 'blue']
bars = ax5.bar(components, values_total, color=colors_total, edgecolor='black', alpha=0.7)
ax5.set_ylabel('Probability')
ax5.set_title('Law of Total Probability\nR=Rain, C=Clouds')
ax5.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, values_total):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}', ha='center', va='bottom')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Prosecutor's fallacy
ax6 = plt.subplot(2, 3, 6)
# P(match | innocent) = 1/1,000,000 does NOT mean P(innocent | match) = 1/1,000,000
p_match_given_innocent = 1/1000000
p_innocent = 0.99  # Prior: 99% of suspects are innocent
p_match_given_guilty = 1.0

p_match = (p_match_given_guilty * (1-p_innocent) + 
           p_match_given_innocent * p_innocent)
p_innocent_given_match = (p_match_given_innocent * p_innocent) / p_match

print("\n=== Prosecutor's Fallacy ===")
print(f"P(DNA match | innocent) = {p_match_given_innocent:.6f}")
print(f"P(innocent | DNA match) = {p_innocent_given_match:.6f}")
print(f"These are NOT the same! Fallacy to equate them.")

labels = ['P(match|innocent)\n(Likelihood)', 'P(innocent|match)\n(Posterior)']
fallacy_vals = [p_match_given_innocent, p_innocent_given_match]
ax6.bar(labels, fallacy_vals, color=['red', 'orange'], edgecolor='black', alpha=0.7)
ax6.set_ylabel('Probability')
ax6.set_title('Prosecutor\'s Fallacy\n(Confusing P(A|B) with P(B|A))')
ax6.set_yscale('log')
ax6.tick_params(axis='x', rotation=0)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Simulation: Monty Hall problem
print("\n=== Monty Hall Problem ===")
n_games = 10000
wins_stay = 0
wins_switch = 0

for _ in range(n_games):
    # Car behind door 1, 2, or 3
    car_door = np.random.randint(1, 4)
    # Contestant picks door 1
    chosen_door = 1
    
    # Monty opens a door (not car, not chosen)
    available_doors = [d for d in [1, 2, 3] if d != chosen_door and d != car_door]
    monty_opens = np.random.choice(available_doors)
    
    # Strategy 1: Stay with original choice
    if chosen_door == car_door:
        wins_stay += 1
    
    # Strategy 2: Switch to remaining door
    remaining_door = [d for d in [1, 2, 3] if d != chosen_door and d != monty_opens][0]
    if remaining_door == car_door:
        wins_switch += 1

p_win_stay = wins_stay / n_games
p_win_switch = wins_switch / n_games

print(f"P(win | stay) = {p_win_stay:.3f} (theoretical: {1/3:.3f})")
print(f"P(win | switch) = {p_win_switch:.3f} (theoretical: {2/3:.3f})")
print("Explanation: Conditioning on Monty's action changes probabilities!")
```

## 2.6 Challenge Round
When is conditional probability misleading?
- **Independence**: If A,B independent then P(A|B)=P(A); conditioning provides no information
- **Simpson's paradox**: Aggregate trend reverses when conditioning on subgroups; pooling hides confounders
- **Base rate neglect**: Ignoring prior P(A) leads to overweighting P(B|A); common in rare diseases
- **Conditioning on colliders**: Creates spurious association between independent variables (selection bias)
- **Zero probability events**: P(A|B) undefined when P(B)=0; continuous distributions need densities

## 2.7 Key References
- [Bayes' Theorem Interactive](https://seeing-theory.brown.edu/bayesian-inference/index.html) - Visual explanation with sliders for prior, likelihood, posterior
- [Conditional Probability Explained](https://www.khanacademy.org/math/statistics-probability/probability-library/conditional-probability-independence) - Khan Academy lessons with exercises
- [Prosecutor's Fallacy Cases](https://en.wikipedia.org/wiki/Prosecutor%27s_fallacy) - Real legal cases where P(A|B) confused with P(B|A)

---
**Status:** Core tool for updating beliefs | **Complements:** Basic Probability, Independence, Bayesian Inference
