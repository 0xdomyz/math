# Auto-extracted from markdown file
# Source: longevity_risk_modeling.md

# --- Code Block 1 ---
import numpy as np

# Simplified: estimate k_t trend
k = np.array([0, -0.5, -1.0, -1.5, -2.0])
drift = -0.5
sigma = 0.3

# project next 5 years
k_proj = [k[-1]]
for _ in range(5):
    k_proj.append(k_proj[-1] + drift + sigma * np.random.normal())

print("Projected k:", k_proj)

