# Auto-extracted from markdown file
# Source: multiple_decrement_probabilities.md

# --- Code Block 1 ---
import numpy as np

cause_rates = np.array([0.02, 0.01, 0.005])
total = 0.04
scaled = total * cause_rates / cause_rates.sum()
print(scaled, scaled.sum())

