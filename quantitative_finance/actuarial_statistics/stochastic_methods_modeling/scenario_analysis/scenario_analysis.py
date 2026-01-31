# Auto-extracted from markdown file
# Source: scenario_analysis.md

# --- Code Block 1 ---
import numpy as np

reserve_base = 1000000
duration = 7.0
rate_shock = 0.01  # 100bps increase

reserve_impact = -reserve_base * duration * rate_shock
stressed_reserve = reserve_base + reserve_impact
print("Stressed reserve:", stressed_reserve)

