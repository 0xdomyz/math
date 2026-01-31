# Independence vs Dependence

## 3.1 Concept Skeleton
**Definition:** Events independent if P(A∩B) = P(A)·P(B); equivalently P(A|B) = P(A); otherwise dependent  
**Purpose:** Simplify probability calculations, test for associations, identify when information matters  
**Prerequisites:** Basic probability, conditional probability, multiplication rule

## 3.2 Comparative Framing
| Relationship | Independent Events | Dependent Events | Mutually Exclusive | Conditional Independence |
|--------------|-------------------|------------------|-------------------|------------------------|
| **Definition** | P(A∩B) = P(A)·P(B) | P(A∩B) ≠ P(A)·P(B) | A∩B = ∅ | P(A∩B\|C) = P(A\|C)·P(B\|C) |
| **Condition Effect** | P(A\|B) = P(A) | P(A\|B) ≠ P(A) | P(A\|B) = 0 | Depends on C |
| **Example** | Coin flips | Card draws no replacement | Rain & no rain | Symptoms independent given disease |
| **Implication** | Info about B useless for A | B informs about A | Can't both occur | Conditioning breaks dependence |

## 3.3 Examples + Counterexamples

**Simple Example:**  
Fair coin flips: P(H₁ ∩ H₂) = 0.25 = P(H₁)·P(H₂) = 0.5·0.5. Knowing first flip doesn't change second

**Failure Case:**  
Assuming independence without verification. Stock prices of competing companies often dependent (correlated)

**Edge Case:**  
Mutually exclusive events (A∩B=∅) are dependent if both have positive probability. P(A|B)=0≠P(A)

## 3.4 Layer Breakdown
```
Independence Framework:
├─ Mathematical Tests:
│   ├─ P(A∩B) = P(A)·P(B) [Definition]
│   ├─ P(A|B) = P(A) [Conditioning doesn't change probability]
│   ├─ P(B|A) = P(B) [Symmetric property]
│   └─ All three equivalent for P(A),P(B) > 0
├─ Multiple Events:
│   ├─ Pairwise independence: P(Aᵢ∩Aⱼ) = P(Aᵢ)·P(Aⱼ) for all i≠j
│   ├─ Mutual independence: P(A₁∩...∩Aₙ) = P(A₁)·...·P(Aₙ)
│   └─ Note: Pairwise ≠ mutual (can have one without other)
├─ Multiplication Rules:
│   ├─ Independent: P(A∩B) = P(A)·P(B)
│   └─ Dependent: P(A∩B) = P(A)·P(B|A) = P(B)·P(A|B)
├─ Dependence Measures:
│   ├─ Covariance: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
│   ├─ Correlation: ρ = Cov(X,Y)/(σₓσᵧ), range [-1,1]
│   ├─ χ² test: Statistical test for independence in contingency tables
│   └─ Mutual information: I(X;Y) bits of information shared
└─ Conditional Independence:
    ├─ Definition: P(A∩B|C) = P(A|C)·P(B|C)
    ├─ Example: Symptoms independent given disease diagnosis
    └─ Application: Naive Bayes classifier, graphical models
```

## 3.5 Mini-Project
Test and visualize independence vs dependence:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns

np.random.seed(42)

# Example 1: Independent events (coin flips)
print("=== Independent Events: Coin Flips ===")
n_trials = 10000
flip1 = np.random.choice([0, 1], size=n_trials)
flip2 = np.random.choice([0, 1], size=n_trials)

p_A = np.mean(flip1 == 1)
p_B = np.mean(flip2 == 1)
p_A_and_B = np.mean((flip1 == 1) & (flip2 == 1))
p_A_given_B = np.mean(flip1[flip2 == 1] == 1)

print(f"P(1st heads) = {p_A:.3f}")
print(f"P(2nd heads) = {p_B:.3f}")
print(f"P(both heads) = {p_A_and_B:.3f}")
print(f"P(A)·P(B) = {p_A * p_B:.3f}")
print(f"P(1st heads | 2nd heads) = {p_A_given_B:.3f}")
print(f"Independence test: |P(A∩B) - P(A)·P(B)| = {abs(p_A_and_B - p_A*p_B):.4f}")

# Example 2: Dependent events (card draws without replacement)
print("\n=== Dependent Events: Card Draws ===")
n_trials = 10000
both_aces = 0
first_ace_count = 0
second_ace_given_first = 0

for _ in range(n_trials):
    deck = list(range(52))  # 0-3 are aces
    np.random.shuffle(deck)
    first = deck[0]
    second = deck[1]
    
    if first < 4:
        first_ace_count += 1
        if second < 4:
            second_ace_given_first += 1
            both_aces += 1

p_first_ace = first_ace_count / n_trials
p_both_aces = both_aces / n_trials
p_second_given_first = second_ace_given_first / first_ace_count if first_ace_count > 0 else 0

print(f"P(1st ace) = {p_first_ace:.4f} (theoretical: {4/52:.4f})")
print(f"P(both aces) = {p_both_aces:.6f} (theoretical: {4/52 * 3/51:.6f})")
print(f"P(2nd ace | 1st ace) = {p_second_given_first:.4f} (theoretical: {3/51:.4f})")
print(f"P(2nd ace | 1st not ace) ≈ {4/51:.4f}")
print(f"Dependence: P(2nd|1st ace) ≠ P(2nd|1st not ace)")

# Example 3: Mutually exclusive events (dependent!)
print("\n=== Mutually Exclusive Events (Always Dependent) ===")
die_rolls = np.random.randint(1, 7, size=10000)
p_even = np.mean(die_rolls % 2 == 0)
p_odd = np.mean(die_rolls % 2 == 1)
p_even_and_odd = np.mean((die_rolls % 2 == 0) & (die_rolls % 2 == 1))

print(f"P(even) = {p_even:.3f}")
print(f"P(odd) = {p_odd:.3f}")
print(f"P(even ∩ odd) = {p_even_and_odd:.3f}")
print(f"P(even)·P(odd) = {p_even * p_odd:.3f}")
print("Note: Mutually exclusive → P(A∩B)=0 ≠ P(A)·P(B) → Dependent!")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Joint probability heatmap (independent)
ax1 = axes[0, 0]
outcomes1 = np.arange(6) + 1
outcomes2 = np.arange(6) + 1
joint_independent = np.outer(np.ones(6)/6, np.ones(6)/6)

im1 = ax1.imshow(joint_independent, cmap='Blues', aspect='auto')
ax1.set_xlabel('Die 1')
ax1.set_ylabel('Die 2')
ax1.set_title('Independent Events\nP(A∩B) = P(A)·P(B)')
ax1.set_xticks(range(6))
ax1.set_yticks(range(6))
ax1.set_xticklabels(outcomes1)
ax1.set_yticklabels(outcomes2)
plt.colorbar(im1, ax=ax1)

# Plot 2: Joint probability heatmap (dependent)
ax2 = axes[0, 1]
# Card example: first card affects second
joint_dependent = np.zeros((4, 4))  # 4 suits
for i in range(4):
    for j in range(4):
        if i == j:
            joint_dependent[i, j] = (13/52) * (12/51)
        else:
            joint_dependent[i, j] = (13/52) * (13/51)

im2 = ax2.imshow(joint_dependent, cmap='Reds', aspect='auto')
ax2.set_xlabel('Card 1 Suit')
ax2.set_ylabel('Card 2 Suit')
ax2.set_title('Dependent Events\nP(same suit|1st) ≠ P(same suit)')
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(['♠', '♥', '♦', '♣'])
ax2.set_yticklabels(['♠', '♥', '♦', '♣'])
plt.colorbar(im2, ax=ax2)

# Plot 3: Conditional probabilities comparison
ax3 = axes[0, 2]
# Independent case
p_B_unconditional = 0.5
p_B_given_A = 0.5
p_B_given_not_A = 0.5

# Dependent case
p_B_unconditional_dep = 4/52
p_B_given_A_dep = 3/51
p_B_given_not_A_dep = 4/51

x = np.arange(3)
width = 0.35
independent_vals = [p_B_unconditional, p_B_given_A, p_B_given_not_A]
dependent_vals = [p_B_unconditional_dep, p_B_given_A_dep, p_B_given_not_A_dep]

ax3.bar(x - width/2, independent_vals, width, label='Independent', alpha=0.7, edgecolor='black')
ax3.bar(x + width/2, dependent_vals, width, label='Dependent', alpha=0.7, edgecolor='black')
ax3.set_ylabel('Probability')
ax3.set_title('Independence: Conditioning Doesn\'t Matter')
ax3.set_xticks(x)
ax3.set_xticklabels(['P(B)', 'P(B|A)', 'P(B|A\')'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Pairwise vs mutual independence
print("\n=== Pairwise vs Mutual Independence ===")
# Example: Three events where pairwise independent but not mutually
n_sim = 100000
# Two fair coins: A = first heads, B = second heads, C = exactly one head
coin1 = np.random.choice([0, 1], size=n_sim)
coin2 = np.random.choice([0, 1], size=n_sim)
A = coin1 == 1
B = coin2 == 1
C = (coin1 + coin2) == 1  # XOR

p_A = np.mean(A)
p_B = np.mean(B)
p_C = np.mean(C)
p_AB = np.mean(A & B)
p_AC = np.mean(A & C)
p_BC = np.mean(B & C)
p_ABC = np.mean(A & B & C)

print(f"P(A) = {p_A:.3f}, P(B) = {p_B:.3f}, P(C) = {p_C:.3f}")
print(f"P(A∩B) = {p_AB:.3f}, P(A)·P(B) = {p_A*p_B:.3f} ✓")
print(f"P(A∩C) = {p_AC:.3f}, P(A)·P(C) = {p_A*p_C:.3f} ✓")
print(f"P(B∩C) = {p_BC:.3f}, P(B)·P(C) = {p_B*p_C:.3f} ✓")
print(f"P(A∩B∩C) = {p_ABC:.3f}, P(A)·P(B)·P(C) = {p_A*p_B*p_C:.3f} ✗")
print("Pairwise independent but NOT mutually independent!")

ax4 = axes[1, 0]
tests = ['P(A∩B)\nvs\nP(A)P(B)', 'P(A∩C)\nvs\nP(A)P(C)', 'P(B∩C)\nvs\nP(B)P(C)', 'P(A∩B∩C)\nvs\nP(A)P(B)P(C)']
observed = [p_AB, p_AC, p_BC, p_ABC]
expected = [p_A*p_B, p_A*p_C, p_B*p_C, p_A*p_B*p_C]

x_pos = np.arange(len(tests))
width = 0.35
ax4.bar(x_pos - width/2, observed, width, label='Observed', alpha=0.7, edgecolor='black')
ax4.bar(x_pos + width/2, expected, width, label='If independent', alpha=0.7, edgecolor='black')
ax4.set_ylabel('Probability')
ax4.set_title('Pairwise ≠ Mutual Independence')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(tests, fontsize=8)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Chi-square test for independence
ax5 = axes[1, 1]
# Contingency table: Treatment vs Outcome
# Independent case
observed_indep = np.array([[40, 60], [40, 60]])
chi2_indep, p_val_indep, dof_indep, expected_indep = chi2_contingency(observed_indep)

# Dependent case
observed_dep = np.array([[60, 40], [20, 80]])
chi2_dep, p_val_dep, dof_dep, expected_dep = chi2_contingency(observed_dep)

print(f"\n=== Chi-Square Independence Test ===")
print("Independent table:")
print(observed_indep)
print(f"χ² = {chi2_indep:.3f}, p-value = {p_val_indep:.3f}")

print("\nDependent table:")
print(observed_dep)
print(f"χ² = {chi2_dep:.3f}, p-value = {p_val_dep:.4f}")

tests_chi = ['Independent\nTable', 'Dependent\nTable']
chi_values = [chi2_indep, chi2_dep]
colors = ['green' if p_val_indep > 0.05 else 'red',
          'green' if p_val_dep > 0.05 else 'red']

bars = ax5.bar(tests_chi, chi_values, color=colors, alpha=0.7, edgecolor='black')
ax5.set_ylabel('χ² Statistic')
ax5.set_title('Chi-Square Independence Test\n(Green=independent, Red=dependent)')
ax5.axhline(3.841, color='gray', linestyle='--', label='Critical value (α=0.05)')
for bar, chi_val, test in zip(bars, chi_values, tests_chi):
    height = bar.get_height()
    p_val = p_val_indep if 'Independent' in test else p_val_dep
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'p={p_val:.3f}', ha='center', va='bottom', fontsize=9)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Correlation vs independence
ax6 = axes[1, 2]
n = 1000
# Independent: uncorrelated
x_indep = np.random.normal(0, 1, n)
y_indep = np.random.normal(0, 1, n)

# Dependent: correlated
x_dep = np.random.normal(0, 1, n)
y_dep = 0.8*x_dep + 0.6*np.random.normal(0, 1, n)

# Dependent: uncorrelated but dependent (X² relationship)
x_nonlinear = np.random.normal(0, 1, n)
y_nonlinear = x_nonlinear**2 + np.random.normal(0, 0.5, n)

correlation_indep = np.corrcoef(x_indep, y_indep)[0, 1]
correlation_dep = np.corrcoef(x_dep, y_dep)[0, 1]
correlation_nonlinear = np.corrcoef(x_nonlinear, y_nonlinear)[0, 1]

ax6_1 = plt.subplot(2, 3, 6)
ax6_1.scatter(x_indep[:200], y_indep[:200], alpha=0.5, s=20, label=f'Indep: r={correlation_indep:.2f}')
ax6_1.scatter(x_dep[:200], y_dep[:200], alpha=0.5, s=20, label=f'Linear dep: r={correlation_dep:.2f}')
ax6_1.scatter(x_nonlinear[:200], y_nonlinear[:200], alpha=0.5, s=20, label=f'Nonlinear: r={correlation_nonlinear:.2f}')
ax6_1.set_xlabel('X')
ax6_1.set_ylabel('Y')
ax6_1.set_title('Correlation ≠ Independence\n(Nonlinear dependence, r≈0)')
ax6_1.legend(fontsize=8)
ax6_1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Example 4: Conditional independence
print("\n=== Conditional Independence ===")
# Disease causes two symptoms independently
n_patients = 10000
has_disease = np.random.choice([0, 1], size=n_patients, p=[0.9, 0.1])

# Symptoms more likely with disease
symptom1 = np.where(has_disease == 1,
                    np.random.choice([0, 1], size=n_patients, p=[0.2, 0.8]),
                    np.random.choice([0, 1], size=n_patients, p=[0.95, 0.05]))

symptom2 = np.where(has_disease == 1,
                    np.random.choice([0, 1], size=n_patients, p=[0.3, 0.7]),
                    np.random.choice([0, 1], size=n_patients, p=[0.9, 0.1]))

# Unconditional: symptoms are dependent
p_s1 = np.mean(symptom1)
p_s2 = np.mean(symptom2)
p_s1_and_s2 = np.mean((symptom1 == 1) & (symptom2 == 1))

print(f"Unconditional:")
print(f"P(S1) = {p_s1:.3f}, P(S2) = {p_s2:.3f}")
print(f"P(S1∩S2) = {p_s1_and_s2:.3f}")
print(f"P(S1)·P(S2) = {p_s1*p_s2:.3f}")
print(f"Dependent: {abs(p_s1_and_s2 - p_s1*p_s2) > 0.01}")

# Conditional on disease: symptoms independent
disease_patients = has_disease == 1
p_s1_given_d = np.mean(symptom1[disease_patients])
p_s2_given_d = np.mean(symptom2[disease_patients])
p_s1_and_s2_given_d = np.mean((symptom1[disease_patients] == 1) & (symptom2[disease_patients] == 1))

print(f"\nConditional on disease:")
print(f"P(S1|D) = {p_s1_given_d:.3f}, P(S2|D) = {p_s2_given_d:.3f}")
print(f"P(S1∩S2|D) = {p_s1_and_s2_given_d:.3f}")
print(f"P(S1|D)·P(S2|D) = {p_s1_given_d*p_s2_given_d:.3f}")
print(f"Independent: {abs(p_s1_and_s2_given_d - p_s1_given_d*p_s2_given_d) < 0.01}")
```

## 3.6 Challenge Round
When does independence assumption fail?
- **Time series data**: Sequential observations typically dependent (autocorrelation)
- **Spatial data**: Nearby locations correlated; assuming independence underestimates variance
- **Clustered sampling**: Individuals within groups more similar; violates independence
- **Common causes**: Unobserved confounders create dependence between seemingly unrelated variables
- **Selection bias**: Conditioning on collider creates spurious dependence between independent causes

## 3.7 Key References
- [Independence Explained](https://en.wikipedia.org/wiki/Independence_(probability_theory)) - Formal definition, pairwise vs mutual, conditional independence
- [Chi-Square Test Tutorial](https://www.statisticshowto.com/probability-and-statistics/chi-square/) - Testing independence in contingency tables
- [Correlation vs Independence](https://stats.stackexchange.com/questions/12842/what-is-the-difference-between-zero-correlation-and-independence) - Zero correlation doesn't imply independence; nonlinear dependence examples

---
**Status:** Fundamental relationship between events | **Complements:** Conditional Probability, Correlation, Hypothesis Testing
