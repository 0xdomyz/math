# Auto-generated from markdown code blocks

# Block 1
import numpy as np

n = 100
steps = np.random.choice([-1, 1], size=n)
walk = np.cumsum(steps)
print(walk[:10])

