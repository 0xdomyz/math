# Auto-extracted from markdown file
# Source: longevity_risk.md

# --- Code Block 1 ---
import numpy as np

# simple longevity stress: reduce mortality by 10%
qx = np.array([0.010, 0.012, 0.014, 0.016])
qx_stress = 0.9 * qx
px = 1 - qx
px_stress = 1 - qx_stress
print(px, px_stress)

