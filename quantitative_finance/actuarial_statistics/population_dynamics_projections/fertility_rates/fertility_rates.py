# Auto-extracted from markdown file
# Source: fertility_rates.md

# --- Code Block 1 ---
import numpy as np

asfr = np.array([0.05, 0.12, 0.15, 0.08, 0.02])
tfr = asfr.sum() * 5  # 5-year age groups
print("TFR:", tfr)

