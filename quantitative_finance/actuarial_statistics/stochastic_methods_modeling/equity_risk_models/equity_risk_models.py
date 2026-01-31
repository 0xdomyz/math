# Auto-extracted from markdown file
# Source: equity_risk_models.md

# --- Code Block 1 ---
import numpy as np

S0 = 100
mu = 0.08
sigma = 0.20
T = 1
n = 252

dt = T / n
S = np.zeros(n)
S[0] = S0

for t in range(1, n):
    dW = np.random.normal(0, np.sqrt(dt))
    S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

print("Final price:", S[-1])

