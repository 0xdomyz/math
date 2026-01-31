# Transition Matrices (Rating Migration)

## 1. Concept Skeleton
**Definition:** Matrix showing probability that borrower migrates from one credit rating to another within specified period (typically one year)  
**Purpose:** Forecast credit quality changes, estimate multi-year default probability, model rating-based PD trends  
**Prerequisites:** Markov chains, rating systems (S&P, Moody's, internal), historical default data, stochastic modeling

## 2. Comparative Framing
| Framework | Time Orientation | Data Source | Update Frequency | Use |
|-----------|-----------------|------------|------------------|-----|
| **Transition Matrix** | Historical | Rating agencies | Annual | Portfolio dynamics |
| **PIT-PD** | Current conditions | Scorecards, models | Real-time | Immediate underwriting |
| **TTC-PD** | Through-cycle | Smoothed history | Rare updates | Capital budgeting |
| **Forward-Looking** | Projected | Analyst forecasts | Quarterly | Stress testing |

## 3. Examples + Counterexamples

**Simple Example:**  
BBB rating: 90% stay BBB, 5% upgrade to A, 3% downgrade to BB, 2% default. Track 100 BBB bonds over 1 year

**Failure Case:**  
Using pre-2008 transition matrix during crisis; downgrade rates spike 10x. Historical assumptions break in regime change

**Edge Case:**  
New rating category (e.g., CCC+ split into subcategories); limited history for transition probabilities; use peer comparison

## 4. Layer Breakdown
```
Transition Matrix Framework:
├─ Matrix Structure:
│   ├─ Rows: Current rating (AAA, AA, A, BBB, BB, B, CCC, D)
│   ├─ Columns: Ending rating (same set + default)
│   ├─ Entry P[i,j]: Probability rating i → rating j in 1 year
│   ├─ Row sum = 1 (all outcomes exhaustive)
│   └─ Example: P[BBB→D] = 0.02 (2% default rate)
├─ Mathematical Properties:
│   ├─ Markov chain: Future state depends only on current state
│   ├─ Stationarity assumption: Matrix stays constant over time
│   ├─ Multi-period: T-year transition = M^T (matrix power)
│   ├─ Eigenvalues determine long-run behavior
│   └─ Absorbing state: D (default) is absorbing (P[D,D]=1)
├─ Construction Methods:
│   ├─ Cohort: Track fixed group over time (accurate but slow)
│   ├─ Duration: Treats each rating duration separately
│   ├─ Hazard rate: Continuous-time intensity approach
│   └─ Adjusted: Point-in-time vs through-the-cycle
├─ Key Patterns:
│   ├─ Rating drift: Migration from investment → speculative
│   ├─ Default clustering: In downturns, all ratings see more defaults
│   ├─ Upgrades rarer: Downgrades > Upgrades in cycle
│   └─ Non-homogeneity: Bank ratings differ from corporate
├─ Multi-Year Predictions:
│   ├─ 2-year default: Use M² to find cumulative PD
│   ├─ Rating path: Most likely path through intermediate states
│   └─ Distribution: Calculate probability of any ending rating
└─ Practical Applications:
    ├─ Portfolio analysis: Forecast concentration by rating
    ├─ Provision calculation: Estimate rating-based EL
    ├─ Capital modeling: Stress rating migration
    └─ Loss distribution: Combine with LGD for loss scenarios
```

## 5. Mini-Project
Build and analyze transition matrices:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power

np.random.seed(42)

# Standard rating categories
ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Default']
n_ratings = len(ratings)

print("=== Transition Matrix Analysis ===")

# Empirical transition matrix (Moody's-like, 2018-2022 average)
# Rows: From rating, Columns: To rating
transition_matrix = np.array([
    [0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # AAA
    [0.01, 0.90, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00],  # AA
    [0.00, 0.02, 0.88, 0.08, 0.02, 0.00, 0.00, 0.00],  # A
    [0.00, 0.00, 0.05, 0.80, 0.10, 0.04, 0.01, 0.00],  # BBB
    [0.00, 0.00, 0.00, 0.05, 0.75, 0.15, 0.04, 0.01],  # BB
    [0.00, 0.00, 0.00, 0.00, 0.08, 0.70, 0.18, 0.04],  # B
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.60, 0.30],  # CCC
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # Default
])

# Display matrix
df_trans = pd.DataFrame(transition_matrix, index=ratings, columns=ratings)
print("\n1-Year Transition Matrix:")
print(df_trans.round(3))

# Extract default rates by rating
default_rates = transition_matrix[:, -1]
print("\n1-Year Default Rates by Rating:")
for rating, pd_val in zip(ratings[:-1], default_rates[:-1]):
    print(f"{rating}: {pd_val:.2%}")

# Multi-year transitions
print("\n=== Multi-Year Transitions ===")
print("Starting from BBB (100 units):")

for years in [1, 3, 5, 10]:
    trans_t = matrix_power(transition_matrix, years)
    bbb_row = trans_t[3, :]  # BBB is index 3
    
    print(f"\n{years}-Year Outcomes (from BBB):")
    for rating, prob in zip(ratings, bbb_row):
        print(f"  {rating:8s}: {prob*100:6.2f}%")

# Calculate cumulative default probability
cumulative_default = []
for years in range(1, 11):
    trans_t = matrix_power(transition_matrix, years)
    bbb_row = trans_t[3, :]
    cum_default = bbb_row[-1]
    cumulative_default.append(cum_default)

# Cumulative PD formula check: 1 - survival
bbb_idx = 3
survival_probs = []
for years in range(1, 11):
    # Prob of surviving = 1 - ending in default state
    trans_t = matrix_power(transition_matrix, years)
    survival = trans_t[bbb_idx, :-1].sum()  # Sum of non-default probabilities
    survival_probs.append(survival)

print("\n=== Cumulative Default Probability (from BBB) ===")
print("Years | Cum Default | Survival Prob")
for year, cum_pd, surv in zip(range(1, 11), cumulative_default, survival_probs):
    print(f"{year:5d} | {cum_pd*100:10.2f}% | {surv*100:12.2f}%")

# Rating migration concentration
print("\n=== Rating Drift Analysis ===")
print("Probability of ending at each rating after N years (start: BBB):")

drift_data = []
for years in [1, 3, 5]:
    trans_t = matrix_power(transition_matrix, years)
    bbb_outcomes = trans_t[3, :]
    drift_data.append(bbb_outcomes)

# Create heatmap data
heatmap_years = [1, 3, 5]
heatmap_data = np.array(drift_data)

# Transition matrix by cohort (simulate 1000 BBB-rated firms)
print("\n=== Portfolio Simulation ===")
n_firms = 1000
current_ratings = np.full(n_firms, 3)  # All start as BBB

# Track migration over 5 years
rating_counts = np.zeros((5, len(ratings)))

for year in range(5):
    for firm in range(n_firms):
        current_rating = current_ratings[firm]
        # Transition: sample new rating based on probabilities
        transition_probs = transition_matrix[current_rating, :]
        new_rating = np.random.choice(len(ratings), p=transition_probs)
        current_ratings[firm] = new_rating
    
    # Count ratings
    unique, counts = np.unique(current_ratings, return_counts=True)
    for rating_idx, count in zip(unique, counts):
        rating_counts[year, rating_idx] = count

print("\nPortfolio Evolution (1000 BBB-rated firms, 5 years):")
print("Year | AAA  | AA   | A    | BBB  | BB   | B    | CCC  | Default")
print("-" * 65)
for year in range(5):
    print(f"{year+1:4d} |", end=" ")
    for rating_idx in range(len(ratings)):
        count = int(rating_counts[year, rating_idx])
        print(f"{count:4d} |" if rating_idx < len(ratings)-1 else f"{count:7d}", end=" ")
    print()

# Stress scenario: Recession transition matrix
print("\n=== Stress Scenario: Recession Transition Matrix ===")
# Increase downgrade probability, decrease upgrade probability
stress_factor = 2.0  # Downgrades 2x higher in recession
recession_matrix = transition_matrix.copy()

for i in range(n_ratings - 1):
    for j in range(n_ratings):
        if j < i:  # Downgrade
            recession_matrix[i, j] *= stress_factor
        elif j > i and j < n_ratings - 1:  # Upgrade
            recession_matrix[i, j] *= 0.5
        elif j == n_ratings - 1:  # Default
            recession_matrix[i, j] *= stress_factor

# Normalize rows to sum to 1
recession_matrix = recession_matrix / recession_matrix.sum(axis=1, keepdims=True)

print("\n1-Year Default Rates: Normal vs Recession")
print("Rating | Normal | Recession | Increase")
print("-" * 45)
for i, rating in enumerate(ratings[:-1]):
    normal_pd = transition_matrix[i, -1]
    stress_pd = recession_matrix[i, -1]
    increase = (stress_pd / normal_pd - 1) * 100 if normal_pd > 0 else 0
    print(f"{rating:6s} | {normal_pd:6.2%} | {stress_pd:9.2%} | {increase:7.0f}%")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Heatmap of transition matrix
ax1 = axes[0, 0]
im1 = ax1.imshow(transition_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(len(ratings)))
ax1.set_yticks(range(len(ratings)))
ax1.set_xticklabels(ratings, rotation=45)
ax1.set_yticklabels(ratings)
ax1.set_xlabel('To Rating')
ax1.set_ylabel('From Rating')
ax1.set_title('1-Year Transition Matrix\n(Normal economic conditions)')
for i in range(n_ratings):
    for j in range(n_ratings):
        text = ax1.text(j, i, f'{transition_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)
plt.colorbar(im1, ax=ax1)

# Plot 2: Default rates by rating
ax2 = axes[0, 1]
default_rates_plot = transition_matrix[:-1, -1] * 100
bars = ax2.bar(ratings[:-1], default_rates_plot, color='red', alpha=0.7, edgecolor='black')
ax2.set_ylabel('1-Year Default Rate (%)')
ax2.set_title('Default Rates by Rating')
ax2.set_yscale('log')
for bar, rate in zip(bars, default_rates_plot):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.2f}%', ha='center', va='bottom', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Multi-year cumulative default
ax3 = axes[0, 2]
years_range = np.arange(1, 11)
ax3.plot(years_range, np.array(cumulative_default)*100, 'o-', linewidth=2, markersize=6)
ax3.fill_between(years_range, 0, np.array(cumulative_default)*100, alpha=0.2)
ax3.set_xlabel('Years')
ax3.set_ylabel('Cumulative Default Probability (%)')
ax3.set_title('Multi-Year Default Probability\n(Starting from BBB)')
ax3.grid(True, alpha=0.3)

# Plot 4: Rating migration paths
ax4 = axes[1, 0]
years_axis = [1, 3, 5]
for i, year in enumerate(years_axis):
    trans_t = matrix_power(transition_matrix, year)
    bbb_outcomes = trans_t[3, :] * 100
    ax4.bar(np.arange(len(ratings)) + i*0.25, bbb_outcomes, width=0.25, 
           label=f'{year}yr', alpha=0.7, edgecolor='black')
ax4.set_xticks(np.arange(len(ratings)) + 0.25)
ax4.set_xticklabels(ratings, rotation=45)
ax4.set_ylabel('Probability (%)')
ax4.set_title('Rating Distribution Over Time\n(Starting from BBB)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Portfolio evolution
ax5 = axes[1, 1]
years_axis = np.arange(rating_counts.shape[0])
for rating_idx, rating in enumerate(ratings[:-1]):  # Exclude default for clarity
    ax5.plot(years_axis + 1, rating_counts[:, rating_idx], 'o-', label=rating, linewidth=2)
ax5.set_xlabel('Years')
ax5.set_ylabel('Number of Firms')
ax5.set_title('Portfolio Evolution\n(1000 BBB-rated firms)')
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)

# Plot 6: Normal vs Recession scenario
ax6 = axes[1, 2]
x_pos = np.arange(len(ratings[:-1]))
width = 0.35
normal_defaults = transition_matrix[:-1, -1] * 100
stress_defaults = recession_matrix[:-1, -1] * 100

ax6.bar(x_pos - width/2, normal_defaults, width, label='Normal', alpha=0.7, edgecolor='black')
ax6.bar(x_pos + width/2, stress_defaults, width, label='Recession', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Rating')
ax6.set_ylabel('1-Year Default Rate (%)')
ax6.set_title('Stress Test: Normal vs Recession')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(ratings[:-1])
ax6.set_yscale('log')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n=== Transition Matrix Summary ===")
print(f"Framework: Markov chain with {len(ratings)} states")
print(f"Property: Forward simulation captures rating migration")
print(f"Limitation: Assumes stationarity; breaks in crisis")
```

## 6. Challenge Round
When are transition matrices problematic?
- **Stationarity violation**: Matrix changes dramatically with economic cycle; 2008 vs 2019 matrices incomparable
- **Limited data**: Some transitions rare (e.g., AAA → Default); estimates unreliable
- **Rating action lag**: Agencies slow to downgrade; matrix reflects late recognition
- **Cohort effects**: Different cohorts may have different migration (e.g., bonds vs loans)
- **Default definition**: Varies by source (payment vs restructuring vs rating trigger); incomparable matrices

## 7. Key References
- [Markov Chain Rating Dynamics](https://en.wikipedia.org/wiki/Markov_chain) - Stochastic process framework
- [Moody's Rating Transitions](https://www.moodysanalytics.com/research/insight/2022/rating-transitions) - Historical transition data
- [Basel III Multi-year PD](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory applications

---
**Status:** Historical and forward-looking rating dynamics tool | **Complements:** Ratings, PD forecasting, stress testing
