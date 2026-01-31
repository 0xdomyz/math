# Auto-extracted from markdown file
# Source: associated_single_decrement.md

# --- Code Block 1 ---
import numpy as np

qx_total = 0.05
qx_cause = 0.02
qx_single = qx_cause / (1 - (qx_total - qx_cause))
print(qx_single)

