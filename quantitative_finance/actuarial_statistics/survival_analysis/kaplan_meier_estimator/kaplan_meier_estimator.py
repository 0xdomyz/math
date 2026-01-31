# Auto-extracted from markdown file
# Source: kaplan_meier_estimator.md

# --- Code Block 1 ---
import numpy as np

times = np.array([2, 3, 3, 5, 8])
status = np.array([1, 1, 0, 1, 0])  # 1=event, 0=censored

# naive KM calculation for illustration
unique_times = np.unique(times[status == 1])
surv = 1.0
for t in unique_times:
    at_risk = np.sum(times >= t)
    events = np.sum((times == t) & (status == 1))
    surv *= (1 - events / at_risk)
    print(t, surv)

