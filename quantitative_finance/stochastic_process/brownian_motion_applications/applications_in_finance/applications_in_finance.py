# Auto-generated from markdown code blocks

# Block 1
import numpy as np

S0, mu, sigma = 100, 0.05, 0.2
T, n = 1.0, 252
 dt = T / n

Z = np.random.normal(size=n)
S = np.zeros(n)
S[0] = S0

for t in range(1, n):
    S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])

print(S[:5])

