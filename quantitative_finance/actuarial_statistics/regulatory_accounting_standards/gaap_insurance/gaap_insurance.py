# Auto-extracted from markdown file
# Source: gaap_insurance.md

# --- Code Block 1 ---
import numpy as np

D = 100.0
years = 5
amort = D / years
schedule = np.full(years, amort)
print(schedule)

