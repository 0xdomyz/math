# Auto-extracted from markdown file
# Source: age_structure.md

# --- Code Block 1 ---
import numpy as np

pop_0_14 = 30000
pop_15_64 = 100000
pop_65_plus = 20000

old_age_dr = pop_65_plus / pop_15_64
youth_dr = pop_0_14 / pop_15_64
total_dr = (pop_0_14 + pop_65_plus) / pop_15_64
print("Old-age:", old_age_dr, "Youth:", youth_dr, "Total:", total_dr)

