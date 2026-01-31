# Counting & Combinatorics

## 5.1 Concept Skeleton
**Definition:** Mathematical techniques for counting arrangements; permutations (order matters) and combinations (order irrelevant)  
**Purpose:** Calculate probabilities by enumerating outcomes; solve complex counting problems systematically  
**Prerequisites:** Multiplication principle, factorials, set theory, basic probability

## 5.2 Comparative Framing
| Method | Permutations P(n,r) | Combinations C(n,r) | Arrangements with Repetition | Partition/Multinomial |
|--------|-------------------|-------------------|----------------------------|---------------------|
| **Formula** | n!/(n-r)! | n!/[r!(n-r)!] | n^r | n!/[k₁!k₂!...kₘ!] |
| **Order** | Matters | Doesn't matter | Matters | Groups matter |
| **Example** | Medal winners 1st/2nd/3rd | Committee of 3 from 10 | PIN code (4 digits) | Anagrams of "MISSISSIPPI" |
| **Repetition** | No repeats | No repeats | Repeats allowed | Identical items grouped |

## 5.3 Examples + Counterexamples

**Simple Example:**  
Choose 2 cards from 52: C(52,2) = 52!/(2!·50!) = 1,326 combinations. Order doesn't matter for poker hand

**Failure Case:**  
Using permutation when order doesn't matter. Choosing 3 students from 20: C(20,3)=1,140, not P(20,3)=6,840

**Edge Case:**  
Overcounting due to symmetry. Seating n people at round table: (n-1)!/2 (rotation and reflection equivalence)

## 5.4 Layer Breakdown
```
Counting Principles:
├─ Fundamental Rules:
│   ├─ Addition Principle: |A∪B| = |A| + |B| if A∩B = ∅
│   ├─ Multiplication Principle: |A×B| = |A| × |B|
│   └─ Inclusion-Exclusion: |A∪B| = |A| + |B| - |A∩B|
├─ Permutations (Order Matters):
│   ├─ All items: P(n,n) = n! arrangements of n objects
│   ├─ r from n: P(n,r) = n!/(n-r)! = n(n-1)...(n-r+1)
│   ├─ With repetition: n^r (each position has n choices)
│   └─ Circular: (n-1)! (fix one position, arrange rest)
├─ Combinations (Order Irrelevant):
│   ├─ r from n: C(n,r) = n!/[r!(n-r)!] = "n choose r"
│   ├─ Properties:
│   │   ├─ C(n,r) = C(n,n-r) (symmetry)
│   │   ├─ C(n,0) = C(n,n) = 1
│   │   └─ Pascal's identity: C(n,r) = C(n-1,r-1) + C(n-1,r)
│   └─ With repetition: C(n+r-1, r) = "stars and bars"
├─ Multinomial Coefficients:
│   ├─ Formula: n!/[k₁!k₂!...kₘ!]
│   ├─ Use: Partition n items into groups of size k₁,k₂,...,kₘ
│   └─ Example: Anagrams with repeated letters
└─ Advanced Techniques:
    ├─ Pigeonhole Principle: n+1 items in n boxes → ≥2 in one box
    ├─ Derangements: Permutations with no fixed points
    ├─ Catalan Numbers: Binary trees, parentheses matching
    └─ Stirling Numbers: Set partitions, cycle decompositions
```

## 5.5 Mini-Project
Implement and verify counting formulas:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, perm, factorial
from itertools import combinations, permutations, product
import math

np.random.seed(42)

# Implementation of counting functions
def permutation(n, r):
    """Calculate P(n,r) = n!/(n-r)!"""
    return math.factorial(n) // math.factorial(n - r)

def combination(n, r):
    """Calculate C(n,r) = n!/(r!(n-r)!)"""
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def multinomial(n, *groups):
    """Calculate n!/(k1! * k2! * ... * km!)"""
    result = math.factorial(n)
    for k in groups:
        result //= math.factorial(k)
    return result

# Example 1: Permutations vs Combinations
print("=== Permutations vs Combinations ===")
n, r = 5, 3
p_nr = permutation(n, r)
c_nr = combination(n, r)

print(f"Arrange {r} from {n} items:")
print(f"Permutations P({n},{r}) = {p_nr} (order matters)")
print(f"Combinations C({n},{r}) = {c_nr} (order doesn't matter)")
print(f"Ratio: P/C = {p_nr/c_nr:.1f} = {r}! (overcounting factor)")

# Verify with explicit enumeration
items = ['A', 'B', 'C', 'D', 'E']
perms = list(permutations(items, r))
combs = list(combinations(items, r))
print(f"Explicit count: {len(perms)} permutations, {len(combs)} combinations")

# Example 2: Card problems
print("\n=== Card Counting Examples ===")
# Poker hand: 5 cards from 52
poker_hands = combination(52, 5)
print(f"Total 5-card poker hands: {poker_hands:,}")

# Specific hands
# Royal flush: 10,J,Q,K,A same suit (4 possibilities)
royal_flushes = 4
p_royal = royal_flushes / poker_hands
print(f"Royal flushes: {royal_flushes}, probability: {p_royal:.10f}")

# Four of a kind: 4 cards same rank, 1 other
# Choose rank for quad (13), choose other card (48)
four_of_kind = 13 * 48
p_four = four_of_kind / poker_hands
print(f"Four of a kind: {four_of_kind:,}, probability: {p_four:.6f}")

# Full house: 3 of one rank, 2 of another
# Choose rank for triple (13), choose 3 suits (C(4,3)), 
# choose rank for pair (12), choose 2 suits (C(4,2))
full_houses = 13 * combination(4, 3) * 12 * combination(4, 2)
p_full = full_houses / poker_hands
print(f"Full houses: {full_houses:,}, probability: {p_full:.6f}")

# Example 3: Birthday problem
print("\n=== Birthday Problem ===")
# Probability at least 2 people share birthday in group of n
def birthday_collision_prob(n, days=365):
    """P(at least 2 share birthday) = 1 - P(all different)"""
    if n > days:
        return 1.0
    # P(all different) = 365/365 * 364/365 * ... * (365-n+1)/365
    p_all_different = 1.0
    for i in range(n):
        p_all_different *= (days - i) / days
    return 1 - p_all_different

for n_people in [10, 20, 23, 30, 50, 70]:
    prob = birthday_collision_prob(n_people)
    print(f"n={n_people:2d} people: P(collision) = {prob:.4f} ({prob*100:.1f}%)")

# Simulation verification
n_people = 23
n_trials = 10000
collisions = 0

for _ in range(n_trials):
    birthdays = np.random.randint(0, 365, n_people)
    if len(birthdays) != len(set(birthdays)):  # Duplicate exists
        collisions += 1

prob_empirical = collisions / n_trials
prob_theoretical = birthday_collision_prob(23)
print(f"\nn=23 simulation: {prob_empirical:.4f} (theoretical: {prob_theoretical:.4f})")

# Example 4: Multinomial coefficients
print("\n=== Anagram Counting (Multinomial) ===")
# MISSISSIPPI: 11 letters, M:1, I:4, S:4, P:2
word = "MISSISSIPPI"
n_total = len(word)
from collections import Counter
counts = Counter(word)
anagrams = multinomial(n_total, *counts.values())
print(f"Word: {word}")
print(f"Letter counts: {dict(counts)}")
print(f"Number of anagrams: {anagrams:,}")

# Verify formula
print(f"Manual: 11!/(1!×4!×4!×2!) = {math.factorial(11)//(1*24*24*2):,}")

# Example 5: Stars and bars (combinations with repetition)
print("\n=== Stars and Bars (Repetition Allowed) ===")
# Distribute r identical items into n bins
# Formula: C(n+r-1, r)
n_bins = 3
r_items = 5
ways = combination(n_bins + r_items - 1, r_items)
print(f"Distribute {r_items} identical items into {n_bins} bins: {ways} ways")
print(f"Formula: C({n_bins}+{r_items}-1, {r_items}) = C({n_bins+r_items-1}, {r_items})")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Pascal's triangle
ax1 = axes[0, 0]
n_rows = 10
pascal = []
for n in range(n_rows):
    row = [combination(n, r) for r in range(n+1)]
    pascal.append(row)

# Plot triangle
for i, row in enumerate(pascal):
    for j, val in enumerate(row):
        x = j - i/2
        y = -i
        ax1.text(x, y, str(val), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

ax1.set_xlim(-n_rows/2-1, n_rows/2+1)
ax1.set_ylim(-n_rows, 1)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title("Pascal's Triangle\nC(n,r) = C(n-1,r-1) + C(n-1,r)")

# Plot 2: Growth of C(n,r) for fixed r
ax2 = axes[0, 1]
r_values = [1, 2, 3, 4, 5]
n_range = np.arange(5, 31)

for r in r_values:
    c_values = [combination(n, r) for n in n_range]
    ax2.plot(n_range, c_values, marker='o', label=f'r={r}', markersize=4)

ax2.set_xlabel('n')
ax2.set_ylabel('C(n,r)')
ax2.set_title('Combinations C(n,r) vs n\n(Polynomial growth in n)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Symmetry C(n,r) = C(n,n-r)
ax3 = axes[0, 2]
n = 20
r_range = np.arange(0, n+1)
c_values = [combination(n, r) for r in r_range]

ax3.bar(r_range, c_values, edgecolor='black', alpha=0.7)
ax3.axvline(n/2, color='r', linestyle='--', linewidth=2, label='Middle')
ax3.set_xlabel('r')
ax3.set_ylabel(f'C({n},r)')
ax3.set_title(f'Symmetry: C({n},r) = C({n},{n}-r)\n(Peaked at middle)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Birthday problem curve
ax4 = axes[1, 0]
n_people_range = np.arange(1, 101)
probs = [birthday_collision_prob(n) for n in n_people_range]

ax4.plot(n_people_range, probs, linewidth=2)
ax4.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='50%')
ax4.axvline(23, color='g', linestyle='--', alpha=0.5, label='n=23 (50.7%)')
ax4.set_xlabel('Number of People')
ax4.set_ylabel('P(at least 2 share birthday)')
ax4.set_title('Birthday Problem\n(Surprising collision probability)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Permutations vs Combinations ratio
ax5 = axes[1, 1]
n = 10
r_range = np.arange(1, n+1)
p_values = [permutation(n, r) for r in r_range]
c_values = [combination(n, r) for r in r_range]
ratios = [p/c for p, c in zip(p_values, c_values)]

ax5.plot(r_range, ratios, 'o-', linewidth=2, markersize=8)
ax5.plot(r_range, [math.factorial(r) for r in r_range], 'r--', 
         linewidth=2, label='r! (overcounting factor)')
ax5.set_xlabel('r')
ax5.set_ylabel('P(n,r) / C(n,r)')
ax5.set_title(f'Permutation Overcounting (n={n})\nP(n,r) = r! × C(n,r)')
ax5.set_yscale('log')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Poker hand probabilities
ax6 = axes[1, 2]
hands = ['Royal\nFlush', 'Straight\nFlush', 'Four of\nKind', 
         'Full\nHouse', 'Flush', 'Straight']
counts = [4, 36, 624, 3744, 5108, 10200]
probs_poker = [c / poker_hands for c in counts]

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(hands)))
bars = ax6.bar(hands, probs_poker, color=colors, edgecolor='black', alpha=0.7)
ax6.set_ylabel('Probability')
ax6.set_title('Poker Hand Probabilities\n(5-card hands from 52)')
ax6.set_yscale('log')
ax6.tick_params(axis='x', rotation=0)
for bar, prob, count in zip(bars, probs_poker, counts):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height*1.5,
             f'{count}', ha='center', va='bottom', fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Example 6: Derangements
print("\n=== Derangements (No Fixed Points) ===")
def derangements(n):
    """Count permutations where no element in original position"""
    if n == 0:
        return 1
    if n == 1:
        return 0
    # D(n) = (n-1)[D(n-1) + D(n-2)]
    # Or: D(n) = n! * Σ(-1)^k/k! for k=0 to n
    d = [0] * (n + 1)
    d[0] = 1
    d[1] = 0
    for i in range(2, n + 1):
        d[i] = (i - 1) * (d[i-1] + d[i-2])
    return d[n]

print("Derangements D(n) / Total permutations n!:")
for n in range(1, 11):
    d_n = derangements(n)
    total = math.factorial(n)
    prob = d_n / total
    print(f"n={n:2d}: D({n}) = {d_n:7d}, n! = {total:7d}, ratio = {prob:.6f}")

print(f"\nAs n→∞, D(n)/n! → 1/e ≈ {1/np.e:.6f}")
```

## 5.6 Challenge Round
When are counting techniques insufficient?
- **Large numbers**: Factorials grow extremely fast; C(100,50) overflows; need logarithms or approximations
- **Constrained arrangements**: Complex restrictions (e.g., no adjacent pairs) require inclusion-exclusion or recursion
- **Continuous domains**: Counting discrete outcomes; probability densities needed for continuous variables
- **Dependent events**: Combinatorics assumes equal probability; weighted/conditional probabilities need careful treatment
- **Symmetry breaking**: Naive counting may overcount due to rotations, reflections, or other equivalences

## 5.7 Key References
- [Combinatorics Overview](https://en.wikipedia.org/wiki/Combinatorics) - Comprehensive coverage of counting principles, generating functions
- [Art of Problem Solving: Counting](https://artofproblemsolving.com/wiki/index.php/Combinatorics) - Competition-level problems with detailed solutions
- [Birthday Problem Explained](https://betterexplained.com/articles/understanding-the-birthday-paradox/) - Intuitive explanation of surprising collision probabilities

---
**Status:** Foundation for probability calculations | **Complements:** Basic Probability, Sample Spaces, Probability Distributions
