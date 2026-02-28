# %% [markdown]
# # monte carlo simulation
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for monte carlo simulation.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: monte carlo simulation")

# %% [markdown]
# Section 2 - Data Generation
# Prepare or generate data used by the model/analysis.

# %%
print("Data preparation step is included in the implementation section below.")

# %% [markdown]
# Section 3 - Model Implementation
# Core actuarial/statistical model logic.

# %%
# Original implementation starts here.
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



# %% [markdown]
# Section 4 - Training & Evaluation
# Run calculations and report quantitative performance metrics.

# %%
print("Training/evaluation is executed in the implementation block above.")

# %% [markdown]
# Section 5 - Visualization & Interpretation
# Produce charts/tables and interpret outputs.

# %%
print("Visualization outputs, if any, are produced in the implementation block above.")

# %% [markdown]
# Section 6 - Summary & Deployment
# Summarize findings and note deployment-readiness considerations.

# %%
print("Summary complete for topic: monte carlo simulation")


