# Auto-extracted from markdown file
# Source: stochastic_interest_rates.md

# --- Code Block 1 ---
import numpy as np

r0 = 0.03
kappa = 0.5
theta = 0.04
sigma = 0.01
T = 10
n = 252

dt = T / n
r = np.zeros(n)
r[0] = r0

for t in range(1, n):
    dW = np.random.normal(0, np.sqrt(dt))
    r[t] = r[t-1] + kappa * (theta - r[t-1]) * dt + sigma * dW

print("Final rate:", r[-1])

