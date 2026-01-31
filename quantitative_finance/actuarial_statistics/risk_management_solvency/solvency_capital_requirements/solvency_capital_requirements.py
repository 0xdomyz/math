# Auto-extracted from markdown file
# Source: solvency_capital_requirements.md

# --- Code Block 1 ---
import numpy as np

risks = np.array([100, 80, 60])
cor = np.array([
    [1.0, 0.25, 0.1],
    [0.25, 1.0, 0.2],
    [0.1, 0.2, 1.0]
])

capital = np.sqrt(risks @ cor @ risks)
print("Diversified capital:", capital)

