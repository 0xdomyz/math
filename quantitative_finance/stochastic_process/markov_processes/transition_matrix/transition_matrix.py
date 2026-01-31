# Auto-generated from markdown code blocks

# Block 1
import numpy as np

P = np.array([
    [0.8, 0.2],
    [0.3, 0.7]
])

P5 = np.linalg.matrix_power(P, 5)
print("P^5=\n", P5)

# steady-state solve (π = πP)
A = np.vstack([P.T - np.eye(2), np.ones(2)])
b = np.array([0, 0, 1])
pi = np.linalg.lstsq(A, b, rcond=None)[0]
print("pi=", pi)

