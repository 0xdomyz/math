# Auto-extracted from markdown file
# Source: mortality_risk.md

# --- Code Block 1 ---
import numpy as np

actual = np.array([12, 15, 10, 14])
expected = np.array([10, 12, 11, 13])
ratio = actual.sum() / expected.sum()
print("A/E ratio:", ratio)

