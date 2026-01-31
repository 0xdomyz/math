# Auto-extracted from markdown file
# Source: graduation_of_mortality_data.md

# --- Code Block 1 ---
import numpy as np

qx = np.array([0.01, 0.012, 0.02, 0.018, 0.022])
window = 3
smooth = np.convolve(qx, np.ones(window)/window, mode='valid')
print(smooth)

