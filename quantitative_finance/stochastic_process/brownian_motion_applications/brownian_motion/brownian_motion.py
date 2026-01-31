# Auto-generated from markdown code blocks

# Block 1
import numpy as np

T, n = 1.0, 1000
 dt = T / n
increments = np.sqrt(dt) * np.random.normal(size=n)
W = np.cumsum(increments)
print(W[:5])

