# Auto-extracted from markdown file
# Source: ifrs_17.md

# --- Code Block 1 ---
import numpy as np

csm = 120.0
years = 4
release = np.full(years, csm / years)
print(release)

