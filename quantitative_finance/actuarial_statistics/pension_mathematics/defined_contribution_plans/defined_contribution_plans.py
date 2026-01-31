# Auto-extracted from markdown file
# Source: defined_contribution_plans.md

# --- Code Block 1 ---
import numpy as np

balance = 10000
annual_contribution = 5000
return_rate = 0.06
years = 20

for _ in range(years):
    balance = balance * (1 + return_rate) + annual_contribution

print("Final balance:", balance)

