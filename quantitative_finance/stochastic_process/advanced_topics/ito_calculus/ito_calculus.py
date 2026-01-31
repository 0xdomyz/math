# Auto-generated from markdown code blocks

# Block 1
import numpy as np

T, n = 1.0, 10000
 dt = T / n

W = np.cumsum(np.sqrt(dt) * np.random.normal(size=n))
W2 = W**2
approx = np.diff(W2) - 2*W[:-1]*np.diff(W)
print(approx.mean(), "~", dt)

