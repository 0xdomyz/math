# Auto-generated from markdown code blocks

# Block 1
import numpy as np

T, n = 1.0, 1000
 dt = T / n

mu, sigma = 0.1, 0.3
X = np.zeros(n)

for t in range(1, n):
    X[t] = X[t-1] + mu*X[t-1]*dt + sigma*X[t-1]*np.sqrt(dt)*np.random.normal()

print(X[:5])

