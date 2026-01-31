# Auto-extracted from markdown file
# Source: monte_carlo_simulation.md

# --- Code Block 1 ---
import numpy as np

S0 = 100
K = 110
r = 0.03
sigma = 0.20
T = 1
n_sim = 10000

Z = np.random.normal(size=n_sim)
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoff = np.maximum(ST - K, 0)
price = np.exp(-r*T) * payoff.mean()
print("Call price:", price)

