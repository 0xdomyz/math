# Auto-extracted from markdown file
# Source: lapse_risk.md

# --- Code Block 1 ---
import numpy as np

inforce = np.array([1000, 980, 960])
base_lapse = np.array([0.02, 0.02, 0.02])
shock = 0.03
lapse = base_lapse + shock
survivors = inforce * (1 - lapse)
print(survivors)

