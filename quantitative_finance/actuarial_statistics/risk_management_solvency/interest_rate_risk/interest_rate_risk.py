# Auto-extracted from markdown file
# Source: interest_rate_risk.md

# --- Code Block 1 ---
import numpy as np

asset_dur = np.array([3.0, 7.0])
asset_weights = np.array([0.6, 0.4])
liability_dur = 5.5

port_dur = (asset_dur * asset_weights).sum()
print("Duration gap:", port_dur - liability_dur)

