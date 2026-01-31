# Auto-generated from markdown code blocks

# Block 1
import numpy as np

lam = 3.0
T = 10.0

# simulate inter-arrivals
times = []
current = 0.0
while current < T:
    current += np.random.exponential(1/lam)
    if current <= T:
        times.append(current)

print("Number of events:", len(times))

