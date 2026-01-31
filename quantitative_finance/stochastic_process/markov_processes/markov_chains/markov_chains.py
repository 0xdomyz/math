# Auto-generated from markdown code blocks

# Block 1
import numpy as np

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

n_steps = 10000
state = 0
counts = np.zeros(3, dtype=int)

for _ in range(n_steps):
    counts[state] += 1
    state = np.random.choice([0, 1, 2], p=P[state])

print(counts / n_steps)

