# Auto-generated from markdown code blocks

# Block 1
import numpy as np

n = 10000
steps = np.random.choice([-1, 1], size=n)
M = np.cumsum(steps)
print(M[:5])

