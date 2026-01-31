# Auto-generated from markdown code blocks

# Block 1
import numpy as np

P = np.array([
    [0.9, 0.1, 0.0],
    [0.2, 0.6, 0.2],
    [0.1, 0.2, 0.7]
])

state = 0
counts = np.zeros(3, dtype=int)

for _ in range(20000):
    counts[state] += 1
    state = np.random.choice([0, 1, 2], p=P[state])

print(counts / counts.sum())

