# Auto-extracted from markdown file
# Source: nelson_aalen_estimator.md

# --- Code Block 1 ---
import numpy as np

times = np.array([2, 3, 5, 5, 8])
status = np.array([1, 1, 1, 0, 0])

unique_times = np.unique(times[status == 1])
H = 0.0
for t in unique_times:
    at_risk = np.sum(times >= t)
    events = np.sum((times == t) & (status == 1))
    H += events / at_risk
    print(t, H)

