# Auto-extracted from markdown file
# Source: stochastic_modeling.md

# --- Code Block 1 ---
import numpy as np

np.random.seed(0)
loss = np.random.lognormal(mean=0.0, sigma=0.6, size=10000)
var_99 = np.quantile(loss, 0.99)
print("VaR 99%:", var_99)

