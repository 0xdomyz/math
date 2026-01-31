# Auto-generated from markdown code blocks

# Block 1
import numpy as np

shape, scale = 1.5, 2.0
T = 20.0

current = 0.0
count = 0
while current < T:
    current += np.random.weibull(shape) * scale
    if current <= T:
        count += 1

print("Renewals:", count)

