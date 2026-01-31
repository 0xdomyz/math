# Auto-extracted from markdown file
# Source: statutory_accounting.md

# --- Code Block 1 ---
import numpy as np

assets = np.array([100, 50, 20])
admitted = np.array([1, 1, 0])  # 0 = non-admitted
stat_assets = (assets * admitted).sum()
print("Statutory admitted assets:", stat_assets)

