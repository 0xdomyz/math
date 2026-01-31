# Auto-generated from markdown code blocks

# Block 1
import numpy as np

n = 1000
lam = 2.0
inter_arrivals = np.random.exponential(1/lam, size=n)
arrival_times = np.cumsum(inter_arrivals)
print(arrival_times[:5])

