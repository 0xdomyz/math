"""
Extracted from: position_information.md
"""

import numpy as np

positions = ["Early", "Middle", "Late"]
base_ev = np.array([-0.02, 0.00, 0.02])
info_bonus = np.array([0.00, 0.01, 0.03])

ev = base_ev + info_bonus

for p, v in zip(positions, ev):
    print(f"{p} EV per hand: {v:.3f}")
