"""
Transition Matrices
Extracted from transition_matrices.md

Credit rating transition analysis using Markov chains.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import matrix_power

np.random.seed(42)

ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]
n_ratings = len(ratings)

print("=== Transition Matrix Analysis ===")

# Empirical transition matrix (Moody's-like)
transition_matrix = np.array(
    [
        [0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # AAA
        [0.01, 0.90, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00],  # AA
        [0.00, 0.02, 0.88, 0.08, 0.02, 0.00, 0.00, 0.00],  # A
        [0.00, 0.00, 0.05, 0.80, 0.10, 0.04, 0.01, 0.00],  # BBB
        [0.00, 0.00, 0.00, 0.05, 0.75, 0.15, 0.04, 0.01],  # BB
        [0.00, 0.00, 0.00, 0.00, 0.08, 0.70, 0.18, 0.04],  # B
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.60, 0.30],  # CCC
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # Default
    ]
)

df_trans = pd.DataFrame(transition_matrix, index=ratings, columns=ratings)
print("\n1-Year Transition Matrix:")
print(df_trans.round(3))

# Extract default rates
default_rates = transition_matrix[:, -1]
print("\n1-Year Default Rates by Rating:")
for rating, pd_val in zip(ratings[:-1], default_rates[:-1]):
    print(f"{rating}: {pd_val:.2%}")

# Multi-year transitions
print("\n=== Multi-Year Transitions ===")
print("Starting from BBB (100 units):")

for years in [1, 3, 5, 10]:
    trans_t = matrix_power(transition_matrix, years)
    bbb_row = trans_t[3, :]

    print(f"\n{years}-Year Outcomes (from BBB):")
    for rating, prob in zip(ratings, bbb_row):
        print(f"  {rating:8s}: {prob*100:6.2f}%")

# Cumulative default probability
cumulative_default = []
for years in range(1, 11):
    trans_t = matrix_power(transition_matrix, years)
    bbb_row = trans_t[3, :]
    cum_default = bbb_row[-1]
    cumulative_default.append(cum_default)

bbb_idx = 3
survival_probs = []
for years in range(1, 11):
    trans_t = matrix_power(transition_matrix, years)
    survival = trans_t[bbb_idx, :-1].sum()
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

heatmap_years = [1, 3, 5]
heatmap_data = np.array(drift_data)

# Portfolio simulation
print("\n=== Portfolio Simulation ===")
n_firms = 1000
current_ratings = np.full(n_firms, 3)  # All start as BBB

rating_counts = np.zeros((5, len(ratings)))

for year in range(5):
    for firm in range(n_firms):
        current_rating = current_ratings[firm]
        transition_probs = transition_matrix[current_rating, :]
        new_rating = np.random.choice(len(ratings), p=transition_probs)
        current_ratings[firm] = new_rating

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
        print(
            f"{count:4d} |" if rating_idx < len(ratings) - 1 else f"{count:7d}", end=" "
        )
    print()

# Stress scenario: Recession transition matrix
print("\n=== Stress Scenario: Recession Transition Matrix ===")

stress_factor = 2.0
recession_matrix = transition_matrix.copy()

for i in range(n_ratings - 1):
    for j in range(n_ratings):
        if j < i:  # Downgrade
            recession_matrix[i, j] *= stress_factor
        elif j > i and j < n_ratings - 1:  # Upgrade
            recession_matrix[i, j] *= 0.5
        elif j == n_ratings - 1:  # Default
            recession_matrix[i, j] *= stress_factor

recession_matrix = recession_matrix / recession_matrix.sum(axis=1, keepdims=True)

print("\n1-Year Default Rates: Normal vs Recession")
print("Rating | Normal | Recession | Increase")
print("-" * 45)
for i, rating in enumerate(ratings[:-1]):
    normal_pd = transition_matrix[i, -1]
    stress_pd = recession_matrix[i, -1]
    increase = (stress_pd / normal_pd - 1) * 100 if normal_pd > 0 else 0
    print(f"{rating:6s} | {normal_pd:6.2%} | {stress_pd:9.2%} | {increase:7.0f}%")

print("\n=== Transition Matrix Summary ===")
print(f"Framework: Markov chain with {len(ratings)} states")
print(f"Property: Forward simulation captures rating migration")
print(f"Limitation: Assumes stationarity; breaks in crisis")

if __name__ == "__main__":
    print("\nTransition matrices execution complete.")
