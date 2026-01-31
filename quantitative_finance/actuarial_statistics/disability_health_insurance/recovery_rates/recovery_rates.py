# Auto-extracted from markdown file
# Source: recovery_rates.md

# --- Code Block 1 ---
import numpy as np

recovered = 200
still_disabled = 100
recovery_rate = recovered / (recovered + still_disabled)
print("Recovery rate:", recovery_rate)

