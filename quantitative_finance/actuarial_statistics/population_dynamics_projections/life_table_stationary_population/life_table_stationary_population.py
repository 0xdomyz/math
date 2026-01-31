# Auto-extracted from markdown file
# Source: life_table_stationary_population.md

# --- Code Block 1 ---
import numpy as np

Lx = np.array([100000, 95000, 90000, 80000, 50000])
T_x = np.cumsum(Lx[::-1])[::-1]
Cx = Lx / T_x[0]
print("Proportion in each age group:", Cx)

