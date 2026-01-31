# Auto-extracted from markdown file
# Source: projection_methodologies.md

# --- Code Block 1 ---
import numpy as np

pop = np.array([10000, 8000, 5000, 2000])
asfr = np.array([0.08, 0.15, 0.10, 0.02])
lx = np.array([0.95, 0.92, 0.85, 0.50])

births = (pop[1:-1] * asfr[1:-1]).sum() * 0.5  # half female
next_pop = np.zeros(4)
next_pop[0] = births
next_pop[1:] = pop[:-1] * lx[:-1]
print(next_pop)

