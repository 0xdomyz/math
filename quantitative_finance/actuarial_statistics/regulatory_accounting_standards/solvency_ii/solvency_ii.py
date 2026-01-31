# Auto-extracted from markdown file
# Source: solvency_ii.md

# --- Code Block 1 ---
import numpy as np

modules = np.array([120, 90, 60])
cor = np.array([
    [1.0, 0.25, 0.1],
    [0.25, 1.0, 0.2],
    [0.1, 0.2, 1.0]
])
SCR = np.sqrt(modules @ cor @ modules)
print("SCR:", SCR)

