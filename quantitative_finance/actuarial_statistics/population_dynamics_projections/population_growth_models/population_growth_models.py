# Auto-extracted from markdown file
# Source: population_growth_models.md

# --- Code Block 1 ---
import numpy as np

P = 1000
r = 0.05
K = 10000
years = 50
trajectory = [P]

for _ in range(years):
    P = P + r * P * (1 - P / K)
    trajectory.append(P)

print(trajectory[-1])

