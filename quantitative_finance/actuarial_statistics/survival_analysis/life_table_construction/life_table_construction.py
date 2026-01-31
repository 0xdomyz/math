# Auto-extracted from markdown file
# Source: life_table_construction.md

# --- Code Block 1 ---
import numpy as np

qx = np.array([0.01, 0.012, 0.015])
lx = [100000]
for q in qx:
    lx.append(lx[-1] * (1 - q))
print(lx)

