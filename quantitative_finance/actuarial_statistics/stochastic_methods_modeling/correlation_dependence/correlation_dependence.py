# Auto-extracted from markdown file
# Source: correlation_dependence.md

# --- Code Block 1 ---
import numpy as np

n = 1000
rho = 0.5

# correlated normals
Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
U = np.apply_along_axis(lambda x: np.exp(x), 0, Z)  # transform to lognormal

print("Correlation:", np.corrcoef(U.T))

