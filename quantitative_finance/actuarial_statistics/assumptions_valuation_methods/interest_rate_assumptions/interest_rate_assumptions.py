# Auto-extracted from markdown file
# Source: interest_rate_assumptions.md

# --- Code Block 1 ---
import numpy as np

cash_flows = np.array([1000, 1000, 1000, 1000])
discount_rate = 0.03
years = np.arange(1, 5)
pv = (cash_flows / (1 + discount_rate) ** years).sum()
print("PV:", pv)

